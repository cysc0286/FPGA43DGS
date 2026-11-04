"""Exercise tool entry wiring without starting an SDK worker or renderer."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common import FULL_WEIGHT_SHA256
from gaussian_generation import runtime as generation_runtime
from gaussian_generation.npu_branch import benchmark, catalog, depth_coverage, qualify, verify
from gaussian_generation.npu_branch.tests.test_npu_method_partition_tools import ProfileModel


CASES = [(['backbone_cnn', 'depth_head', 'to_gaussians', 'to_disparity'] +
          [f'transformer_ffn_{i}' for i in range(6)], {}),
         (['corr_depth_pipeline'], {'corr_depth_pipeline': True}),
         (['refine_heads_pipeline'], {'refine_heads_pipeline': True}),
         (['corr_depth_pipeline', 'refine_heads_pipeline'],
          {'corr_depth_pipeline': True, 'refine_heads_pipeline': True})]


class RefineHeadsToolTests(unittest.TestCase):
    def test_verify_forwards_only_selected_methods_before_worker(self):
        for names, expected in CASES:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)/'verify'
                args = ['--backend', 'onnx_reference', '--graphs', 'graphs', '--bundle', 'bundle',
                        '--input', 'input', '--weights', 'weights', '--vendor', 'vendor',
                        '--out', str(out), '--partitions', *names]
                with patch.object(verify, 'ModelRuntime', side_effect=RuntimeError('stop before model')) as model, \
                        patch.object(verify, 'PartitionRuntime') as worker:
                    with self.assertRaisesRegex(RuntimeError, 'stop before model'):
                        verify.main(args)
                    model.assert_called_once_with(Path('weights'), Path('vendor'), 2, 16,
                                                  out/'runtime', **expected)
                    worker.assert_not_called()
                record = json.loads((out/'verification.json').read_text())
                self.assertFalse(record['complete'])
                self.assertFalse(record['npu_executed'])

    def test_qualify_forwards_flags_but_mock_never_issues_acceptance(self):
        for names, expected in CASES:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = root/'input'
                inputs.mkdir()
                (inputs/'input.json').write_text(json.dumps(dict(context_sha256='context',
                    context_indices=[0, 1], target_indices=[2, 3, 4],
                    views=[dict(name=str(i), role='target') for i in range(3)])))
                out = root/'qualify'
                args = ['--backend', 'npu', '--partition-bundle', 'bundle', '--npu-library', 'bridge',
                        '--input', str(inputs), '--weights', 'weights', '--vendor', 'vendor',
                        '--renderer', 'renderer', '--live-renderer', 'live', '--boot-image', 'boot',
                        '--out', str(out), '--partitions', *names]
                def digest(path):
                    return FULL_WEIGHT_SHA256 if Path(path).name == 'weights' else (
                        'boot' if Path(path).name == 'boot' else 'context')
                with patch.object(qualify.platform, 'machine', return_value='aarch64'), \
                        patch.object(qualify, 'sha', side_effect=digest), \
                        patch('rendering.runtime.load_mainline_profile', return_value={
                            'required_hardware': {'accepted_boot_sha256': 'boot'}}), \
                        patch.object(generation_runtime, 'ModelRuntime',
                                     side_effect=RuntimeError('stop before model')) as model:
                    with self.assertRaisesRegex(RuntimeError, 'stop before model'):
                        qualify.main(args)
                    model.assert_called_once_with(Path('weights'), Path('vendor'), 4, 16,
                                                  out/'initialize', **expected)
                record = json.loads((out/'acceptance.json').read_text())
                self.assertFalse(record['complete'])
                self.assertFalse(record['npu_executed'])
                self.assertFalse(record['fpga_executed'])

    def test_both_method_benchmarks_fail_before_io_model_or_worker(self):
        for names in (['refine_heads_pipeline'], ['corr_depth_pipeline', 'refine_heads_pipeline'],
                      ['backbone_cnn', 'refine_heads_pipeline']):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                args = ['--backend', 'onnx_reference', '--graphs', 'g', '--bundle', 'b',
                        '--weights', 'w', '--vendor', 'v', '--out', str(Path(directory)/'out'),
                        '--partitions', *names]
                stderr = io.StringIO()
                with patch.object(benchmark, 'new_directory') as mkdir, \
                        patch.object(benchmark, 'ModelRuntime') as model, \
                        patch.object(benchmark, 'PartitionRuntime') as worker, \
                        contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as error:
                        benchmark.main(args)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn('refine_heads_pipeline', stderr.getvalue())
                    self.assertIn('paired_inference', stderr.getvalue())
                    self.assertIn('explicitly enabled model seam', stderr.getvalue())
                    mkdir.assert_not_called()
                    model.assert_not_called()
                    worker.assert_not_called()

    def test_cpu_profiler_excludes_both_absent_methods_and_keeps_real_components(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context = root/'context.npz'
                np.savez(context, image=np.ones((2, 3, 4, 4), np.float32))
                model = ProfileModel(fail=fail).eval()
                runtime = SimpleNamespace(model=model, torch=torch, threads=1)
                names = dict(backbone_cnn='backbone.0',
                    corr_depth_pipeline='depth_predictor._corr_to_depth_logits',
                    refine_heads_pipeline='depth_predictor._refine_to_heads')
                with patch.dict(catalog.PARTITIONS, names, clear=True), contextlib.redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'deliberate model error'):
                            depth_coverage.profile_cpu(runtime, context, root, 1)
                    else:
                        depth_coverage.profile_cpu(runtime, context, root, 2)
                        record = json.loads((root/'cpu_profile.json').read_text())
                        self.assertEqual(record['excluded_method_partitions'],
                                         ['corr_depth_pipeline', 'refine_heads_pipeline'])
                        for item in record['rounds']:
                            self.assertNotIn('refine_heads_pipeline', item['stages_seconds'])
                            self.assertIn('refine_unet', item['stages_seconds'])
                            self.assertTrue(item['finite'])
                self.assertFalse(hasattr(model.depth_predictor, '_refine_to_heads'))
                self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks
                                    for module in model.modules()))


if __name__ == '__main__':
    unittest.main()
