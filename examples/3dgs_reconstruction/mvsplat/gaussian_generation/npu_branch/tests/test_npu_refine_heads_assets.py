"""Composition ABI, catalog and provenance tests; graph fixtures are not run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.artifacts import from_wire, to_wire, validate_bundle
from gaussian_generation.npu_branch.catalog import (
    PARTITIONS, REFINE_HEADS_COMPONENTS, semantic_coverage, validate_selection,
)
from gaussian_generation.npu_branch.config import BackendConfig
from gaussian_generation.npu_branch.tests.test_npu_candidate_contracts import reference_bundle


NAME = 'refine_heads_pipeline'


def refinement_meta():
    return dict(complete=True, host_onnx_passed=True, full_module=True, returns_list=False,
        dtype='float32', logical_layout='NCHW', input_shape=[2, 133, 128, 128],
        output_shape=[2, 86, 128, 128], onnx_output_layout='NHWC',
        onnx_output_shape=[2, 128, 128, 86],
        composed_modules=list(REFINE_HEADS_COMPONENTS),
        source_provenance={name: dict(manifest_sha256='1'*64, graph_sha256='2'*64,
            oracle_sha256='3'*64, context_sha256='4'*64, source_oracle_check=dict(passed=True))
            for name in REFINE_HEADS_COMPONENTS},
        oracle_provenance='source_onnx_derived_chain_on_explicit_packed_diagnostic_inputs',
        independent_arm_pipeline_reference=False, upsampler_included=False,
        upsampled_feature_reused=True, full_view_batch_preserved=True, spatial_tiling_used=False,
        packed_input=[dict(name=name, channels=channels) for name, channels in
            [('upsampled_features', 128), ('rgb', 3), ('coarse_disparity', 1), ('pdf_max', 1)]],
        packed_output=[dict(name='raw_gaussians', channels=84),
                       dict(name='delta_disparity_raw_density', channels=2)])


def compiled_bundle(root, meta):
    root = Path(root)
    folder = root/NAME
    folder.mkdir()
    graph, raw = folder/'network.json', folder/'network.raw'
    graph.write_bytes(b'not-executed compiled graph fixture')
    raw.write_bytes(b'not-executed compiled weights fixture')
    audit = dict(input=dict(layout='NHWC', shape=[2, 128, 128, 133]),
                 output=dict(layout='NHWC', shape=[2, 128, 128, 86]),
                 graph_sha256=sha(graph), raw_sha256=sha(raw))
    manifest = dict(schema='mvsplat_compiled_v1', weight_sha256=FULL_WEIGHT_SHA256,
        partitions={NAME: dict(complete=True, graph=NAME+'/network.json', raw=NAME+'/network.raw',
                               export=meta, audit=audit)})
    (root/'manifest.json').write_text(json.dumps(manifest))
    return audit


class RefineHeadsAssetTests(unittest.TestCase):
    def test_catalog_exact_seam_and_semantic_coverage(self):
        self.assertEqual(PARTITIONS[NAME], 'depth_predictor._refine_to_heads')
        self.assertEqual(REFINE_HEADS_COMPONENTS,
                         ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity'))
        self.assertEqual(semantic_coverage([NAME]), set(REFINE_HEADS_COMPONENTS))
        self.assertEqual(semantic_coverage(['corr_depth_pipeline', NAME]),
            {'corr_refine_net', 'regressor_residual', 'depth_head', *REFINE_HEADS_COMPONENTS})
        selected = ['backbone_cnn', 'corr_depth_pipeline', NAME, 'transformer_ffn_5']
        self.assertEqual(validate_selection(selected), selected)

    def test_composite_and_four_components_or_fused_reject_before_assets(self):
        for name in (*REFINE_HEADS_COMPONENTS, 'fused_heads'):
            for names in ([NAME, name], [name, NAME]):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    validate_bundle('must-not-access', names, 'npu')
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    BackendConfig(backend='npu', partition_bundle=Path('b'), npu_library=Path('bridge'),
                                  partitions=tuple(names))
        for names in ([], [NAME, NAME], ['unknown'], NAME):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_selection(names)

    def test_reference_and_compiled_static_abi_preserve_diagnostic_provenance(self):
        for backend in ('onnx_reference', 'npu'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                meta = refinement_meta()
                if backend == 'npu':
                    audit = compiled_bundle(directory, meta)
                    with patch('gaussian_generation.npu_branch.compile_graphs.audit', return_value=audit):
                        _, parts = validate_bundle(directory, [NAME], backend)
                else:
                    reference_bundle(directory, {NAME: meta})
                    _, parts = validate_bundle(directory, [NAME], backend)
                self.assertEqual(parts[NAME]['meta']['packed_input'], meta['packed_input'])
                self.assertEqual(parts[NAME]['output'], dict(layout='NHWC', shape=[2, 128, 128, 86]))
                self.assertFalse(parts[NAME]['meta']['independent_arm_pipeline_reference'])

    def test_contract_shape_layout_packing_components_and_provenance_tampering_rejected(self):
        mutations = [('full_module', False), ('returns_list', True), ('dtype', 'float16'),
            ('logical_layout', 'NHWC'), ('input_shape', [1, 133, 128, 128]),
            ('input_shape', [2, 134, 128, 128]), ('output_shape', [2, 84, 128, 128]),
            ('composed_modules', ['refine_unet']), ('source_provenance', {}),
            ('upsampler_included', True), ('upsampled_feature_reused', False),
            ('full_view_batch_preserved', False), ('spatial_tiling_used', True),
            ('packed_input', list(reversed(refinement_meta()['packed_input']))),
            ('packed_output', list(reversed(refinement_meta()['packed_output']))),
            ('oracle_provenance', '')]
        for backend in ('onnx_reference', 'npu'):
            for key, value in mutations:
                with self.subTest(backend=backend, key=key), tempfile.TemporaryDirectory() as directory:
                    meta = refinement_meta()
                    meta[key] = value
                    if backend == 'npu':
                        audit = compiled_bundle(directory, meta)
                        with patch('gaussian_generation.npu_branch.compile_graphs.audit', return_value=audit), \
                                self.assertRaises(ValueError):
                            validate_bundle(directory, [NAME], backend)
                    else:
                        reference_bundle(directory, {NAME: meta})
                        with self.assertRaises(ValueError):
                            validate_bundle(directory, [NAME], backend)

    def test_reference_wire_layout_changes_and_changed_graph_rejected(self):
        for key, value in [('onnx_output_layout', '***C'), ('onnx_output_shape', [2, 128, 86, 128])]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                meta = refinement_meta()
                meta[key] = value
                reference_bundle(directory, {NAME: meta})
                with self.assertRaises(ValueError):
                    validate_bundle(directory, [NAME], 'onnx_reference')
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory, {NAME: refinement_meta()})
            (Path(directory)/NAME/'model.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Unvalidated ONNX'):
                validate_bundle(directory, [NAME], 'onnx_reference')

    def test_source_provenance_requires_real_hash_fields_and_passed_source_oracle(self):
        """A dict with the right component names is not proof of audited sources."""
        mutations = []
        for value in (None, [], {}, 'not provenance'):
            mutations.append(('source record type', lambda m, v=value:
                              m['source_provenance'].__setitem__('proj_feature', v)))
        for field in ('manifest_sha256', 'graph_sha256', 'oracle_sha256', 'context_sha256'):
            mutations.append((field+' missing', lambda m, f=field:
                              m['source_provenance']['proj_feature'].pop(f)))
            for value in ('', 'a'*63, 'g'*64, 'A'*64, 123, None):
                mutations.append((field+' invalid '+str(value), lambda m, f=field, v=value:
                                  m['source_provenance']['proj_feature'].__setitem__(f, v)))
        for value in (None, {}, [], {'passed': False}, {'passed': 1}, {'passed': 'true'}):
            mutations.append(('source oracle result '+str(value), lambda m, v=value:
                m['source_provenance']['refine_unet'].__setitem__('source_oracle_check', v)))
        for value in (' ', ['fake'], 'original_arm_hook', True, None):
            mutations.append(('oracle origin '+str(value), lambda m, v=value:
                              m.__setitem__('oracle_provenance', v)))
        for backend in ('onnx_reference', 'npu'):
            for description, mutate in mutations:
                with self.subTest(backend=backend, mutation=description), \
                        tempfile.TemporaryDirectory() as directory:
                    meta = refinement_meta()
                    mutate(meta)
                    if backend == 'npu':
                        audit = compiled_bundle(directory, meta)
                        with patch('gaussian_generation.npu_branch.compile_graphs.audit', return_value=audit), \
                                self.assertRaises(ValueError):
                            validate_bundle(directory, [NAME], backend)
                    else:
                        reference_bundle(directory, {NAME: meta})
                        with self.assertRaises(ValueError):
                            validate_bundle(directory, [NAME], backend)

    def test_small_exact_wire_roundtrip_keeps_batch_and_raw_channel_boundary(self):
        value = np.arange(2*86*3*5, dtype=np.float32).reshape(2, 86, 3, 5)
        spec = dict(layout='NHWC', shape=[2, 3, 5, 86])
        wire = to_wire(value, spec)
        restored = from_wire(wire, spec)
        np.testing.assert_array_equal(restored[:, :84], value[:, :84])
        np.testing.assert_array_equal(restored[:, 84:], value[:, 84:])
        wire.fill(-1)
        np.testing.assert_array_equal(restored, value)


if __name__ == '__main__':
    unittest.main()
