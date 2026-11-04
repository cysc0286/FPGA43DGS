"""Refinement candidate entry prechecks; no worker/board inference is executed."""
import argparse
import builtins
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gaussian_generation import runtime as generation_runtime
from gaussian_generation.npu_branch.config import BackendConfig, GenerationSession, add_arguments
from gaussian_generation.npu_branch.tests.test_npu_refine_heads_source import vendor_fixture


class RefineHeadsConfigTests(unittest.TestCase):
    def test_only_explicit_selection_forwards_seam_keywords_and_preflights(self):
        cases = [(['depth_head'], {}),
            (['backbone_cnn', 'depth_head', 'to_gaussians', 'to_disparity'] +
             [f'transformer_ffn_{i}' for i in range(6)], {}),
            (['refine_heads_pipeline'], {'refine_heads_pipeline': True}),
            (['corr_depth_pipeline'], {'corr_depth_pipeline': True}),
            (['corr_depth_pipeline', 'refine_heads_pipeline'],
             {'corr_depth_pipeline': True, 'refine_heads_pipeline': True})]
        for names, expected in cases:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory, \
                    patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker, \
                    patch.object(generation_runtime, 'ModelRuntime') as model, \
                    patch.object(generation_runtime, 'validate_refine_heads_config') as preflight:
                args = SimpleNamespace(backend='onnx_reference', partition_bundle='bundle',
                                       partitions=names, weights='w', vendor='v')
                order = []
                preflight.side_effect = lambda vendor: order.append('config')
                worker.side_effect = lambda *a, **k: order.append('worker') or worker.return_value
                model.side_effect = lambda *a, **k: order.append('model') or model.return_value
                with GenerationSession(args, directory, threads=2, depth_chunk=16):
                    self.assertEqual(model.call_args.kwargs, expected)
                    model.return_value.attach_partitions.assert_called_once_with(
                        worker.return_value.__enter__.return_value)
                self.assertEqual(order, (['config'] if 'refine_heads_pipeline' in names else []) +
                                 ['worker', 'model'])
                if 'refine_heads_pipeline' in names:
                    preflight.assert_called_once_with('v')
                else:
                    preflight.assert_not_called()
                worker.return_value.__exit__.assert_called_once()

    def test_invalid_chunk_rejected_before_config_model_or_worker(self):
        args = SimpleNamespace(backend='onnx_reference', partition_bundle='bundle',
                               partitions=['refine_heads_pipeline'], weights='w', vendor='v')
        for chunk in (True, False, 0, -1, 129, 1.5, None, '16'):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as directory, \
                    patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker, \
                    patch.object(generation_runtime, 'ModelRuntime') as model, \
                    patch.object(generation_runtime, 'validate_refine_heads_config') as preflight:
                with self.assertRaisesRegex(ValueError, 'before worker startup'):
                    GenerationSession(args, directory, threads=2, depth_chunk=chunk)
                worker.assert_not_called()
                model.assert_not_called()
                preflight.assert_not_called()
                self.assertFalse(json.loads((Path(directory)/'generation_failure.json').read_text())['public_ready'])

    def test_real_ablation_yaml_preflight_fails_before_worker_and_checkpoint(self):
        for base, override in ((True, None), (False, True)):
            with self.subTest(base=base, override=override), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vendor = vendor_fixture(root, base=base, override=override)
                (root/'job').mkdir()
                args = SimpleNamespace(backend='onnx_reference', partition_bundle='bundle',
                    partitions=['refine_heads_pipeline'], weights='missing', vendor=vendor)
                with patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker, \
                        patch.object(generation_runtime, 'ModelRuntime') as model:
                    with self.assertRaisesRegex(ValueError, 'wo_depth_refine=False before worker'):
                        GenerationSession(args, root/'job', threads=2, depth_chunk=16)
                    worker.assert_not_called()
                    model.assert_not_called()
                self.assertFalse(json.loads((root/'job/generation_failure.json').read_text())['public_ready'])

    def test_static_preflight_uses_experiment_override_without_loading_torch_or_weights(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ('torch', 'numpy'):
                raise AssertionError('static config preflight imported numerical runtime')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            vendor = vendor_fixture(Path(directory), base=True, override=False)
            with patch('builtins.__import__', side_effect=guarded_import), \
                    patch.object(generation_runtime, 'sha', side_effect=AssertionError('checkpoint access')):
                cfg = generation_runtime.validate_refine_heads_config(vendor)
            self.assertIs(cfg.wo_depth_refine, False)

    def test_parser_and_cpu_default_remain_explicit(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(BackendConfig.from_args(args).cli(), ['--backend', 'cpu'])
        self.assertFalse(hasattr(args, 'refine_heads_pipeline'))
        args = parser.parse_args(['--backend', 'onnx_reference', '--partition-bundle', 'bundle',
                                  '--partitions', 'refine_heads_pipeline', 'transformer_ffn_0'])
        self.assertEqual(BackendConfig.from_args(args).cli()[-3:],
                         ['--partitions', 'refine_heads_pipeline', 'transformer_ffn_0'])


if __name__ == '__main__':
    unittest.main()
