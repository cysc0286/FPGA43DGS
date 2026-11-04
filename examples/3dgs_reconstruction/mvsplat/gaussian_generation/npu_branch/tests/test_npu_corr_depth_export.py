"""Real ONNX composition/ABI checks; these do not emulate an NPU device."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from common import sha
from gaussian_generation.npu_branch.corr_depth_export import compose, _run, _session, _source


def conv_model(incoming, outgoing, kernel, rng, *, layout='NCHW', opset=17):
    n, channels, h, w = incoming
    weight = rng.normal(0, .1, (outgoing, channels, kernel, kernel)).astype(np.float32)
    bias = rng.normal(0, .1, (outgoing,)).astype(np.float32)
    values = [numpy_helper.from_array(weight, 'weight'), numpy_helper.from_array(bias, 'bias')]
    nodes = [helper.make_node('Conv', ['features', 'weight', 'bias'], ['values'], name='conv',
        kernel_shape=[kernel, kernel], pads=[kernel//2]*4, strides=[1, 1])]
    shape, name = [n, outgoing, h, w], 'values'
    if layout == 'NHWC':
        name, shape = 'nhwc', [n, h, w, outgoing]
        nodes.append(helper.make_node('Transpose', ['values'], ['nhwc'], name='transpose',
                                     perm=[0, 2, 3, 1]))
    graph = helper.make_graph(nodes, 'test_branch',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, incoming)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)], values)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = 9
    return model


class CorrDepthExportTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(421)
        self.shape = [2, 5, 7, 11]
        self.corr = conv_model(self.shape, 3, 3, self.rng, layout='NHWC')
        self.res = conv_model(self.shape, 3, 1, self.rng)
        self.head = conv_model([2, 3, 7, 11], 4, 3, self.rng)

    def test_actual_onnx_non_square_two_view_abba_equivalence(self):
        model, meta = compose(self.corr, self.res, self.head)
        sessions = [_session(m) for m in (self.corr, self.res, self.head)]
        combined = _session(model)
        values = {label: self.rng.normal(size=self.shape).astype(np.float32) for label in ('A', 'B')}
        repeats = {}
        for label in ('A', 'B', 'B', 'A'):
            x = values[label]
            intermediate = _run(sessions[0], x, 'NHWC')+_run(sessions[1], x, 'NCHW')
            expected = _run(sessions[2], intermediate, 'NCHW')
            actual = _run(combined, x, 'NHWC')
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            if label in repeats:
                np.testing.assert_array_equal(actual, repeats[label])
            repeats[label] = actual
        self.assertFalse(np.array_equal(repeats['A'], repeats['B']))
        self.assertEqual(meta['output_shape'], [2, 4, 7, 11])
        self.assertEqual(meta['onnx_output_shape'], [2, 7, 11, 4])
        self.assertEqual(sum(n.op_type == 'Transpose' for n in model.graph.node), 1)

    def test_every_original_parameter_value_preserved_and_inputs_unmodified(self):
        originals = [self.corr, self.res, self.head]
        before = [m.SerializeToString() for m in originals]
        model, meta = compose(*originals)
        actual = {t.name: t for t in model.graph.initializer}
        for prefix, original in zip(('corr/', 'residual/', 'head/'), originals):
            for value in original.graph.initializer:
                copied = copy.deepcopy(actual[prefix+value.name])
                copied.name = value.name
                self.assertEqual(copied.SerializeToString(), value.SerializeToString())
        self.assertEqual(before, [m.SerializeToString() for m in originals])
        self.assertFalse(meta['trained_parameters_changed'])

    def test_io_accounting_matches_two_distinct_comparisons(self):
        _, meta = compose(self.corr, self.res, self.head)
        x, y, z = (2*5*7*11*4, 2*3*7*11*4, 2*4*7*11*4)
        self.assertEqual(meta['logical_io_bytes'], dict(
            three_npu_boundaries=2*x+3*y+z, corr_head_npu_residual_cpu=x+2*y+z,
            continuous=x+z, saved_vs_three=x+3*y, saved_vs_corr_head=2*y))

    def test_nchw_source_is_supported_without_layout_relabel(self):
        corr = conv_model(self.shape, 3, 3, self.rng)
        model, _ = compose(corr, self.res, self.head, corr_output_layout='NCHW')
        self.assertEqual(sum(n.op_type == 'Transpose' for n in model.graph.node), 1)
        onnx.checker.check_model(model)

    def test_nhwc_source_requires_real_transpose(self):
        bad = copy.deepcopy(self.corr)
        bad.graph.node[-1].attribute[0].ints[:] = [0, 3, 1, 2]
        with self.assertRaisesRegex(ValueError, 'exact terminal'):
            compose(bad, self.res, self.head)
        with self.assertRaisesRegex(ValueError, 'output layout'):
            compose(self.corr, self.res, self.head, corr_output_layout='NCHWc16')

    def test_shape_or_view_mismatch_rejected(self):
        for incoming, outgoing in (([1, 5, 7, 11], 3), (self.shape, 4), ([2, 5, 11, 7], 3)):
            bad = conv_model(incoming, outgoing, 1, self.rng)
            with self.assertRaisesRegex(ValueError, 'boundary mismatch'):
                compose(self.corr, bad, self.head)

    def test_opset_and_dtype_mismatch_rejected(self):
        bad = copy.deepcopy(self.head)
        bad.opset_import[0].version = 16
        with self.assertRaisesRegex(ValueError, 'versions differ'):
            compose(self.corr, self.res, bad)
        bad = copy.deepcopy(self.res)
        bad.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
        with self.assertRaisesRegex(ValueError, 'FP32'):
            compose(self.corr, bad, self.head)

    def test_source_rejects_relabelled_layout_and_manifest_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'regressor_residual'
            folder.mkdir()
            onnx.save(self.res, folder/'model.onnx')
            x = self.rng.normal(size=self.shape).astype(np.float32)
            y = _run(_session(self.res), x, 'NCHW')
            np.savez(folder/'oracle.npz', input=x, expected=y)
            meta = dict(complete=True, host_onnx_passed=True, graph_sha256=sha(folder/'model.onnx'),
                oracle_sha256=sha(folder/'oracle.npz'), input_shape=self.shape,
                output_shape=list(y.shape), dtype='float32', logical_layout='NCHW', returns_list=False)
            manifest = dict(schema='mvsplat_partitions_v1', npu_executed=False,
                context_sha256='0'*64, partitions={'regressor_residual':meta})
            def write():
                (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            write()
            self.assertEqual(_source(root, 'regressor_residual')[1], meta)
            meta['onnx_output_layout'] = 'NHWC'
            write()
            with self.assertRaisesRegex(ValueError, 'require NCHW'):
                _source(root, 'regressor_residual')
            meta['onnx_output_layout'] = 'NCHW'
            meta['output_shape'] = [2, 3, 11, 7]
            write()
            with self.assertRaisesRegex(ValueError, 'logical metadata'):
                _source(root, 'regressor_residual')


if __name__ == '__main__':
    unittest.main()
