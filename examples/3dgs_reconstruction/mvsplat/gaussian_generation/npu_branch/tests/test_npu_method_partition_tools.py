"""Keep default module tools usable when the catalog gains an opt-in method."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

from gaussian_generation.npu_branch import benchmark, catalog, depth_coverage


class ProfileModel(torch.nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Identity())
        self.depth_predictor = torch.nn.Sequential()
        self.depth_predictor.corr_refine_net = torch.nn.Identity()
        self.depth_predictor.refine_unet = torch.nn.Identity()
        self.gaussian_adapter = torch.nn.Identity()
        self.fail = fail

    def forward(self, context, **kwargs):
        value = self.backbone(context['image'])
        value = self.depth_predictor(value)
        if self.fail:
            raise RuntimeError('deliberate model error')
        value = self.gaussian_adapter(value)
        return types.SimpleNamespace(**{name: value for name in
            ('means', 'covariances', 'harmonics', 'opacities')})


class MethodPartitionToolsTests(unittest.TestCase):
    def test_default_cpu_profile_skips_method_and_keeps_component_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            context = out/'context.npz'
            np.savez(context, image=np.ones((2, 3, 4, 4), np.float32))
            model = ProfileModel().eval()
            runtime = types.SimpleNamespace(model=model, torch=torch, threads=1)
            names = dict(backbone_cnn='backbone.0',
                         corr_depth_pipeline='depth_predictor._corr_to_depth_logits')
            with mock.patch.dict(catalog.PARTITIONS, names, clear=True), contextlib.redirect_stdout(io.StringIO()):
                depth_coverage.profile_cpu(runtime, context, out, 2)
            record = json.loads((out/'cpu_profile.json').read_text(encoding='utf-8'))
            self.assertEqual(record['excluded_method_partitions'], ['corr_depth_pipeline'])
            self.assertFalse(record['npu_executed'])
            self.assertEqual(len(record['rounds']), 2)
            for result in record['rounds']:
                self.assertTrue(result['finite'])
                self.assertNotIn('corr_depth_pipeline', result['stages_seconds'])
                self.assertEqual(set(result['stages_seconds']), {
                    'backbone_cnn', 'backbone', 'corr_refine_net', 'refine_unet',
                    'depth', 'gaussian_adapter'})
            self.assertFalse(hasattr(model.depth_predictor, '_corr_to_depth_logits'))
            self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                                for module in model.modules()))

    def test_cpu_profile_removes_hooks_on_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            context = out/'context.npz'
            np.savez(context, image=np.ones((2, 3, 4, 4), np.float32))
            model = ProfileModel(fail=True).eval()
            runtime = types.SimpleNamespace(model=model, torch=torch, threads=1)
            with mock.patch.dict(catalog.PARTITIONS,
                    {'corr_depth_pipeline': 'depth_predictor._corr_to_depth_logits'}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'deliberate model error'):
                    depth_coverage.profile_cpu(runtime, context, out, 1)
            self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                                for module in model.modules()))

    def arguments(self, out, partitions):
        return ['--backend', 'onnx_reference', '--graphs', 'missing_graphs',
                '--bundle', 'missing_bundle', '--weights', 'missing_weights',
                '--vendor', 'missing_vendor', '--out', str(out), '--partitions', *partitions]

    def test_method_benchmark_fails_before_files_model_or_worker(self):
        for names in (['corr_depth_pipeline'], ['backbone_cnn', 'corr_depth_pipeline']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)/'must_not_exist'
                stderr = io.StringIO()
                with mock.patch.object(benchmark, 'new_directory') as mkdir, \
                        mock.patch.object(benchmark, 'ModelRuntime') as model, \
                        mock.patch.object(benchmark, 'PartitionRuntime') as worker, \
                        contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        benchmark.main(self.arguments(out, names))
                self.assertEqual(error.exception.code, 2)
                self.assertIn('explicitly enabled model seam', stderr.getvalue())
                self.assertIn('paired_inference', stderr.getvalue())
                mkdir.assert_not_called()
                model.assert_not_called()
                worker.assert_not_called()
                self.assertFalse(out.exists())

    def test_original_module_benchmark_still_uses_default_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)/'run'
            with mock.patch.object(benchmark, 'ModelRuntime',
                    side_effect=RuntimeError('stop before loading a real model')) as model:
                with self.assertRaisesRegex(RuntimeError, 'stop before loading'):
                    benchmark.main(self.arguments(out, ['backbone_cnn']))
            model.assert_called_once_with(Path('missing_weights'), Path('missing_vendor'),
                                          2, 16, out/'runtime')
            record = json.loads((out/'benchmark.json').read_text(encoding='utf-8'))
            self.assertFalse(record['complete'])
            self.assertFalse(record['npu_executed'])


if __name__ == '__main__':
    unittest.main()
