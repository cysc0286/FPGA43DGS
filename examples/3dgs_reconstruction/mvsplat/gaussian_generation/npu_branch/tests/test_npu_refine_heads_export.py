"""Small real-ORT composition/provenance tests, without NPU or full MVSplat."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.corr_depth_export import _run, _session
from gaussian_generation.npu_branch.diagnostic_runner import array_sha
from gaussian_generation.npu_branch.refine_heads_export import (
    FIXTURE_SCHEMA, _fixtures, _source, _source_reference, compose,
)
from gaussian_generation.npu_branch.tests.test_npu_corr_depth_export import conv_model


class RefineHeadsExportTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(701)
        self.up_shape = [2, 4, 7, 11]
        self.projection = conv_model(self.up_shape, 2, 3, self.rng)
        self.refinement = conv_model([2, 7, 7, 11], 3, 3, self.rng, layout='NHWC')
        self.gaussian = conv_model([2, 10, 7, 11], 5, 3, self.rng)
        self.disparity = conv_model([2, 3, 7, 11], 2, 3, self.rng)
        self.models = [self.projection, self.refinement, self.gaussian, self.disparity]
        self.sources = [(None, dict(onnx_output_layout=layout), model, _session(model))
                        for model, layout in zip(self.models, ('NCHW', 'NHWC', 'NCHW', 'NCHW'))]

    def test_real_non_square_two_view_abba_and_both_branch_slices(self):
        model, meta = compose(*self.models)
        session = _session(model)
        values = {label: self.rng.normal(size=[2, 9, 7, 11]).astype(np.float32) for label in ('A', 'B')}
        repeats = {}
        for label in ('A', 'B', 'B', 'A'):
            expected = _source_reference(self.sources, values[label], up_channels=4)
            actual = _run(session, values[label], 'NHWC')
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(actual[:, :5], expected[:, :5], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(actual[:, 5:], expected[:, 5:], rtol=1e-5, atol=1e-6)
            if label in repeats:
                np.testing.assert_array_equal(actual, repeats[label])
            repeats[label] = actual
        self.assertFalse(np.array_equal(repeats['A'], repeats['B']))
        self.assertEqual(meta['input_shape'], [2, 9, 7, 11])
        self.assertEqual(meta['output_shape'], [2, 7, 7, 11])
        self.assertEqual(meta['onnx_output_shape'], [2, 7, 11, 7])
        self.assertTrue(meta['upsampled_feature_reused'])
        self.assertFalse(meta['upsampler_included'])

    def test_upsampled_feature_fans_out_unchanged_and_parameters_preserved(self):
        before = [model.SerializeToString() for model in self.models]
        model, meta = compose(*self.models)
        initializers = {value.name: value for value in model.graph.initializer}
        for prefix, original in zip(('projection/', 'refine/', 'gaussians/', 'disparity/'), self.models):
            for value in original.graph.initializer:
                copied = copy.deepcopy(initializers[prefix+value.name])
                copied.name = value.name
                self.assertEqual(copied.SerializeToString(), value.SerializeToString())
        self.assertEqual(before, [model.SerializeToString() for model in self.models])
        projection = next(node for node in model.graph.node if node.name == 'projection/conv')
        gaussian_input = next(node for node in model.graph.node if node.name == 'original_gaussian_input_with_reused_up')
        self.assertEqual(projection.input[0], 'up')
        self.assertEqual(gaussian_input.input[-1], 'up')
        self.assertFalse(meta['trained_parameters_changed'])

    def test_identity_convolution_before_real_tail_is_retained(self):
        refinement = copy.deepcopy(self.refinement)
        transpose = refinement.graph.node[-1]
        original_value = transpose.input[0]
        transpose.input[0] = 'identity_value'
        identity = helper.make_node('Conv', [original_value, 'identity_weight', 'identity_bias'],
            ['identity_value'], name='output_control_identity', kernel_shape=[1, 1])
        refinement.graph.node.insert(len(refinement.graph.node)-1, identity)
        refinement.graph.initializer.extend([
            numpy_helper.from_array(np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1), 'identity_weight'),
            numpy_helper.from_array(np.zeros(3, np.float32), 'identity_bias')])
        model, _ = compose(self.projection, refinement, self.gaussian, self.disparity)
        self.assertIn('refine/output_control_identity', [node.name for node in model.graph.node])
        self.assertIn('refine/identity_weight', [value.name for value in model.graph.initializer])
        value = self.rng.normal(size=[2, 9, 7, 11]).astype(np.float32)
        original, _ = compose(*self.models)
        np.testing.assert_allclose(_run(_session(model), value, 'NHWC'),
                                   _run(_session(original), value, 'NHWC'), rtol=1e-5, atol=1e-6)

    def test_refine_nchw_is_valid_but_false_nhwc_tail_rejected(self):
        refinement = conv_model([2, 7, 7, 11], 3, 3, self.rng)
        model, _ = compose(self.projection, refinement, self.gaussian, self.disparity,
                           refine_output_layout='NCHW')
        onnx.checker.check_model(model)
        self.assertEqual(sum(node.op_type == 'Transpose' for node in model.graph.node), 1)
        with self.assertRaisesRegex(ValueError, 'exact terminal'):
            compose(self.projection, refinement, self.gaussian, self.disparity)
        bad = copy.deepcopy(self.refinement)
        bad.graph.node[-1].attribute[0].ints[:] = [0, 3, 1, 2]
        with self.assertRaisesRegex(ValueError, 'exact terminal'):
            compose(self.projection, bad, self.gaussian, self.disparity)

    def test_mismatched_batch_spatial_channels_opset_and_dtype_rejected(self):
        cases = [
            (1, conv_model([1, 7, 7, 11], 3, 3, self.rng, layout='NHWC')),
            (1, conv_model([2, 7, 11, 7], 3, 3, self.rng, layout='NHWC')),
            (2, conv_model([2, 9, 7, 11], 5, 3, self.rng)),
            (3, conv_model([2, 4, 7, 11], 2, 3, self.rng)),
        ]
        for index, bad in cases:
            models = list(self.models)
            models[index] = bad
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'boundary mismatch'):
                compose(*models)
        bad = copy.deepcopy(self.gaussian)
        bad.opset_import[0].version = 16
        with self.assertRaisesRegex(ValueError, 'versions differ'):
            compose(self.projection, self.refinement, bad, self.disparity)
        bad = copy.deepcopy(self.disparity)
        bad.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
        with self.assertRaisesRegex(ValueError, 'FP32'):
            compose(self.projection, self.refinement, self.gaussian, bad)

    def test_real_128_boundary_accounting_includes_padded_head_tiles(self):
        projection = conv_model([2, 128, 128, 128], 32, 1, self.rng)
        refine = conv_model([2, 37, 128, 128], 32, 1, self.rng, layout='NHWC')
        gaussian = conv_model([2, 163, 128, 128], 84, 1, self.rng)
        disparity = conv_model([2, 32, 128, 128], 2, 1, self.rng)
        _, meta = compose(projection, refine, gaussian, disparity)
        self.assertEqual(meta['input_shape'], [2, 133, 128, 128])
        self.assertEqual(meta['output_shape'], [2, 86, 128, 128])
        self.assertEqual(meta['logical_io_bytes']['current_refine_and_tiled_heads']/2**20, 53.080078125)
        self.assertEqual(meta['logical_io_bytes']['continuous']/2**20, 27.375)
        self.assertEqual(meta['logical_io_bytes']['refine_and_full_heads']/2**20, 43.75)
        self.assertEqual(meta['logical_io_bytes']['projection_refine_and_full_heads']/2**20, 63.75)
        self.assertEqual(meta['logical_requests'], dict(current_refine_and_tiled_heads=65, continuous=1))

    def test_source_entry_revalidates_hash_oracle_layout_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'refine_unet'
            folder.mkdir()
            graph = folder/'model.onnx'
            onnx.save(self.refinement, graph)
            value = self.rng.normal(size=[2, 7, 7, 11]).astype(np.float32)
            expected = _run(_session(self.refinement), value, 'NHWC')
            np.savez(folder/'oracle.npz', input=value, expected=expected)
            meta = dict(complete=True, host_onnx_passed=True, graph_sha256=sha(graph),
                oracle_sha256=sha(folder/'oracle.npz'), input_shape=list(value.shape),
                output_shape=list(expected.shape), dtype='float32', logical_layout='NCHW',
                onnx_output_layout='NHWC', returns_list=False)
            manifest = dict(schema='mvsplat_partitions_v1', npu_executed=False,
                            context_sha256='0'*64, partitions={'refine_unet': meta})
            def write():
                (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            write()
            self.assertEqual(_source(root, 'refine_unet')[1], meta)
            meta['output_shape'] = [2, 3, 11, 7]
            write()
            with self.assertRaisesRegex(ValueError, 'logical metadata'):
                _source(root, 'refine_unet')
            meta['output_shape'] = list(expected.shape)
            meta['graph_sha256'] = '0'*64
            write()
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                _source(root, 'refine_unet')


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(63)
        self.shape = [2, 9, 7, 11]

    def fixtures(self, root, *, expected=False):
        manifest = dict(schema=FIXTURE_SCHEMA, complete=True, npu_executed=False,
            weight_sha256=FULL_WEIGHT_SHA256,
            input_provenance='synthetic_coarse_signal_intervention_on_same_legacy_up_and_RGB',
            context_binding_scope='packed input fingerprints; not original video context identities', contexts={})
        for index, label in enumerate(('A', 'B')):
            value = self.rng.normal(size=self.shape).astype(np.float32)
            values = dict(input=value)
            if expected:
                values['expected'] = self.rng.normal(size=[2, 7, 7, 11]).astype(np.float32)
                manifest['reference_provenance'] = 'source_onnx_derived'
            path = root/(label+'.npz')
            np.savez(path, **values)
            item = dict(context_sha256=str(index+1)*64, context_views=2, fixture=path.name,
                        fixture_sha256=sha(path), input_sha256=array_sha(value))
            if expected:
                item['expected_sha256'] = array_sha(values['expected'])
            manifest['contexts'][label] = item
        self.write(root, manifest)
        return manifest

    def write(self, root, manifest):
        (root/'capture_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    def test_input_only_preserves_stated_synthetic_provenance_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.fixtures(root)
            manifest, _, samples, supplied, provenance = _fixtures(root, self.shape, FULL_WEIGHT_SHA256)
            self.assertEqual(manifest['input_provenance'], expected['input_provenance'])
            self.assertEqual(manifest['context_binding_scope'], expected['context_binding_scope'])
            self.assertEqual(supplied, {})
            self.assertEqual(set(samples), {'A', 'B'})
            self.assertNotEqual(provenance['A']['input_sha256'], provenance['B']['input_sha256'])

    def test_optional_expected_requires_source_derived_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixtures(root, expected=True)
            self.assertEqual(set(_fixtures(root, self.shape, FULL_WEIGHT_SHA256)[3]), {'A', 'B'})
            manifest['reference_provenance'] = 'board ARM CPU'
            self.write(root, manifest)
            with self.assertRaisesRegex(ValueError, 'source-ONNX-derived'):
                _fixtures(root, self.shape, FULL_WEIGHT_SHA256)

    def test_rejects_path_escape_hash_change_equal_inputs_and_false_arm_claim(self):
        for change, message in (
            (lambda m: m['contexts']['A'].update(fixture='../A.npz'), 'identity or path'),
            (lambda m: m['contexts']['A'].update(fixture_sha256='0'*64), 'fixture file'),
            (lambda m: m['contexts']['B'].update(context_sha256=m['contexts']['A']['context_sha256']), 'must differ'),
            (lambda m: m.update(reference_device='board ARM CPU'), 'not an ARM'),
            (lambda m: m.update(context_binding_scope=''), 'provenance'),
            (lambda m: m.update(weight_sha256='0'*64), 'provenance'),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixtures(root)
                change(manifest)
                self.write(root, manifest)
                with self.assertRaisesRegex(ValueError, message):
                    _fixtures(root, self.shape, FULL_WEIGHT_SHA256)


if __name__ == '__main__':
    unittest.main()
