"""Synthetic topology tests for the U-Net ONNX prefix diagnostic extractor."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian_generation.npu_branch.onnx_prefix_probe import extract_prefix


class OnnxPrefixProbeTests(unittest.TestCase):
    def test_prefix_prunes_only_suffix_and_preserves_initializer_values(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        weights = np.arange(6, dtype=np.float32).reshape(3, 2, 1, 1)
        initializer = numpy_helper.from_array(weights, 'weight')
        nodes = [
            helper.make_node('Conv', ['input', 'weight'], ['conv_out'], name='/inner/inner.0/Conv',
                             pads=[0, 0, 0, 0]),
            helper.make_node('Relu', ['conv_out'], ['relu_out'], name='/inner/inner.2/Mul_1'),
            helper.make_node('Relu', ['relu_out'], ['tail_out'], name='suffix'),
        ]
        graph = helper.make_graph(nodes, 'source',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 2, 3])],
            [helper.make_tensor_value_info('tail_out', TensorProto.FLOAT, [2, 3, 2, 3])],
            initializer=[initializer])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prefix.onnx'
            record = extract_prefix(model, path, op_type='Conv', node_name='/inner/inner.0/Conv')
            prefix = onnx.load(str(path))
            onnx.checker.check_model(prefix)
            self.assertEqual(record['input_shape'], [2, 2, 2, 3])
            self.assertEqual(record['logical_output_shape'], [2, 3, 2, 3])
            self.assertEqual([n.name for n in prefix.graph.node],
                             ['/inner/inner.0/Conv', 'diagnostic_explicit_nchw_to_nhwc'])
            self.assertEqual([x.name for x in prefix.graph.initializer], ['weight'])
            np.testing.assert_array_equal(numpy_helper.to_array(prefix.graph.initializer[0]), weights)
            self.assertEqual(prefix.graph.initializer[0].SerializeToString(), initializer.SerializeToString())
            import onnxruntime as ort
            prefix.ir_version = min(prefix.ir_version, 10)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(prefix.SerializeToString(), options, providers=['CPUExecutionProvider'])
            value = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
            expected = np.einsum('nchw,oc->nohw', value, weights[:, :, 0, 0])
            raw = session.run(None, {'input': value})[0]
            self.assertEqual(raw.shape, (2, 2, 3, 3))
            np.testing.assert_array_equal(raw.transpose(0, 3, 1, 2), expected)

    def test_unknown_endpoint_fails_closed(self):
        import onnx
        from onnx import TensorProto, helper

        node = helper.make_node('Identity', ['input'], ['output'], name='known')
        graph = helper.make_graph([node], 'source',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])])
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                extract_prefix(model, Path(folder) / 'prefix.onnx',
                               op_type='Identity', node_name='missing')

    def test_skip_dependency_is_retained_and_suffix_weights_removed(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        nodes = [helper.make_node('Relu', ['input'], ['positive'], name='branch'),
                 helper.make_node('Add', ['input', 'positive'], ['joined'], name='join'),
                 helper.make_node('Add', ['joined', 'unused_weight'], ['tail'], name='suffix')]
        graph = helper.make_graph(nodes, 'residual',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 1, 2, 3])],
            [helper.make_tensor_value_info('tail', TensorProto.FLOAT, [2, 1, 2, 3])],
            initializer=[numpy_helper.from_array(np.array(2, dtype=np.float32), 'unused_weight')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prefix.onnx'
            extract_prefix(model, path, op_type='Add', node_name='join')
            prefix = onnx.load(str(path))
            self.assertEqual([n.name for n in prefix.graph.node[:-1]], ['branch', 'join'])
            self.assertEqual(len(prefix.graph.initializer), 0)
            self.assertEqual(list(prefix.graph.node[1].input), ['input', 'positive'])

    def test_cannot_relabel_an_inferred_shape_or_drop_view_batch(self):
        from onnx import TensorProto, helper
        graph = helper.make_graph([helper.make_node('Identity', ['input'], ['output'], name='known')], 'source',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 1, 2, 3])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 1, 2, 3])])
        model = helper.make_model(graph)
        with tempfile.TemporaryDirectory() as folder:
            for shape in ([1, 1, 2, 3], [2, 3, 2, 1], [2, 1, 6]):
                with self.assertRaisesRegex(ValueError, 'shape'):
                    extract_prefix(model, Path(folder) / 'prefix.onnx', op_type='Identity',
                                   node_name='known', output_shape=shape)

    def test_rank_three_endpoint_cannot_be_disguised_with_matching_first_dimensions(self):
        from onnx import TensorProto, helper, numpy_helper
        shape = numpy_helper.from_array(np.array([2, 1, 6], dtype=np.int64), 'shape')
        graph = helper.make_graph(
            [helper.make_node('Reshape', ['input', 'shape'], ['rank_three'], name='flatten')], 'source',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 1, 2, 3])],
            [helper.make_tensor_value_info('rank_three', TensorProto.FLOAT, [2, 1, 6])], initializer=[shape])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                extract_prefix(model, Path(folder) / 'prefix.onnx', op_type='Reshape',
                               node_name='flatten', output_shape=[2, 1, 6, 1])


if __name__ == '__main__':
    unittest.main()
