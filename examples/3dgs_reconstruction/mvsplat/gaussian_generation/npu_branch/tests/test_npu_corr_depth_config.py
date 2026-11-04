"""Continuous-graph entry contracts; these tests never attest board performance."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gaussian_generation.npu_branch.artifacts import validate_bundle
from gaussian_generation.npu_branch.catalog import (
    CORR_DEPTH_COMPONENTS, semantic_coverage, validate_selection)
from gaussian_generation.npu_branch.config import BackendConfig, GenerationSession, add_arguments
from gaussian_generation.npu_branch.acceptance import validate_worker
from gaussian_generation.npu_branch.tests.test_npu_candidate_contracts import reference_bundle
from gaussian_generation.npu_branch.tests.test_npu_integration import receipt


def continuous_meta():
    return dict(complete=True, host_onnx_passed=True, full_module=True, returns_list=False,
                dtype='float32', logical_layout='NCHW',
                input_shape=[2, 256, 32, 32], output_shape=[2, 128, 32, 32],
                onnx_output_layout='NHWC', onnx_output_shape=[2, 32, 32, 128],
                composed_modules=['corr_refine_net', 'regressor_residual', 'depth_head_lowres'],
                source_provenance={name: {'source': name} for name in CORR_DEPTH_COMPONENTS},
                oracle_provenance='source_onnx_derived_chain_on_original_real_ARM_inputs',
                independent_arm_depth_reference=False)


class ContinuousConfigTests(unittest.TestCase):
    def test_composite_conflicts_rejected_before_any_bundle_access(self):
        for component in CORR_DEPTH_COMPONENTS:
            names = ['corr_depth_pipeline', component]
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                validate_bundle('no-file-access', names, 'npu')
            with self.subTest(component=component), self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                BackendConfig(backend='npu', partition_bundle=Path('b'),
                              npu_library=Path('bridge'), partitions=tuple(names))

    def test_known_unique_nonempty_names_required(self):
        for names in ([], ['corr_depth_pipeline'] * 2, ['unknown'], 'corr_depth_pipeline', [[1]]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_selection(names)

    def test_semantic_coverage_tracks_fusion_without_duplicate_compute(self):
        baseline = ['backbone_cnn', *CORR_DEPTH_COMPONENTS, 'to_gaussians', 'to_disparity']
        fused = ['backbone_cnn', 'corr_depth_pipeline', 'fused_heads']
        self.assertEqual(semantic_coverage(baseline), semantic_coverage(fused))
        self.assertEqual(validate_selection(fused), fused)

    def test_cli_forwards_explicit_composite_but_does_not_enable_by_default(self):
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        self.assertEqual(BackendConfig.from_args(parser.parse_args([])).cli(), ['--backend', 'cpu'])
        args = parser.parse_args(['--backend', 'onnx_reference', '--partition-bundle', 'bundle',
                                  '--partitions', 'corr_depth_pipeline', 'transformer_ffn_0'])
        config = BackendConfig.from_args(args)
        self.assertEqual(config.cli()[-3:], ['--partitions', 'corr_depth_pipeline', 'transformer_ffn_0'])

    def test_chunk_guard_precedes_worker_and_model(self):
        args = SimpleNamespace(backend='onnx_reference', partition_bundle='bundle',
                               partitions=['corr_depth_pipeline'], weights='w', vendor='v')
        for chunk in (0, -1, 129, True, 1.5, '16', None):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as directory, \
                 patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker, \
                 patch('gaussian_generation.runtime.ModelRuntime') as model:
                with self.assertRaisesRegex(ValueError, 'before worker startup'):
                    GenerationSession(args, directory, threads=2, depth_chunk=chunk)
                worker.assert_not_called()
                model.assert_not_called()
                self.assertFalse(json.loads((Path(directory)/'generation_failure.json').read_text())['public_ready'])

    def test_only_explicit_composite_enables_generated_source_seam(self):
        for names, expected in ((['depth_head'], {}),
                                (['corr_depth_pipeline'], {'corr_depth_pipeline': True})):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory, \
                 patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker, \
                 patch('gaussian_generation.runtime.ModelRuntime') as model:
                args = SimpleNamespace(backend='onnx_reference', partition_bundle='bundle',
                                       partitions=names, weights='w', vendor='v')
                with GenerationSession(args, directory, threads=2, depth_chunk=16):
                    self.assertEqual(model.call_args.kwargs, expected)
                    model.return_value.attach_partitions.assert_called_once_with(
                        worker.return_value.__enter__.return_value)
                worker.return_value.__exit__.assert_called_once()

    def test_composite_artifact_preserves_reference_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory, {'corr_depth_pipeline': continuous_meta()})
            _, parts = validate_bundle(directory, ['corr_depth_pipeline'], 'onnx_reference')
            self.assertFalse(parts['corr_depth_pipeline']['meta']['independent_arm_depth_reference'])
            self.assertEqual(parts['corr_depth_pipeline']['output']['shape'], [2, 32, 32, 128])

    def test_unreviewed_composition_or_shape_is_rejected(self):
        for key, value in (('composed_modules', ['corr_refine_net']),
                           ('input_shape', [2, 128, 32, 32]), ('source_provenance', {}),
                           ('oracle_provenance', ''), ('full_module', False)):
            meta = copy.deepcopy(continuous_meta())
            meta[key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                reference_bundle(directory, {'corr_depth_pipeline': meta})
                with self.assertRaisesRegex(ValueError, 'composition/ABI/provenance'):
                    validate_bundle(directory, ['corr_depth_pipeline'], 'onnx_reference')

    def test_existing_head_receipt_cannot_qualify_new_composite(self):
        record = receipt()
        worker = SimpleNamespace(parts={'corr_depth_pipeline': {}},
                                 readiness={'runtime_identity': record['runtime_identity']},
                                 provenance=record['provenance'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                validate_worker(path, worker, record['oracle_checks'])


if __name__ == '__main__':
    unittest.main()
