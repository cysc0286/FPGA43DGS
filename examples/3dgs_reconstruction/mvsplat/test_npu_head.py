"""Real graph transformations must preserve both views and padded channels."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from gaussian_generation.npu_branch.head_probe import aligned_model, compare_output, make_cases


class HeadProbeChecks(unittest.TestCase):
    def test_sink_comparison_rejects_stale_nonfinite_and_missing_mode(self):
        import copy
        from gaussian_generation.npu_branch.head_probe import output_consistency
        values = [dict(input_label=label, output_sink=mode,
                       output_sha256=label, nonfinite_count=0)
                  for label in ('A','B') for mode in ('string','direct')]
        self.assertTrue(output_consistency(values, True)['output_sink_byte_exact'])
        stale = copy.deepcopy(values)
        stale[2]['output_sha256'] = 'A'
        self.assertFalse(output_consistency(stale, True)['freshness_passed'])
        nonfinite = copy.deepcopy(values)
        nonfinite[0]['nonfinite_count'] = 1
        self.assertFalse(output_consistency(nonfinite, True)['output_sink_byte_exact'])
        self.assertFalse(output_consistency(values[1:], True)['output_sink_byte_exact'])

    def test_old_bridge_cannot_silently_ignore_direct_output_option(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from gaussian_generation.npu_branch.protocol import check_direct_sfb
        with self.assertRaisesRegex(ValueError, 'lacks direct'):
            check_direct_sfb(SimpleNamespace())
        with self.assertRaisesRegex(ValueError, 'rejected'):
            check_direct_sfb(SimpleNamespace(mgs_direct_sfb_supported=Mock(return_value=0)))
        check_direct_sfb(SimpleNamespace(mgs_direct_sfb_supported=Mock(return_value=1)))

    def test_nonfinite_output_never_passes(self):
        for value in (np.nan, np.inf, -np.inf):
            result = compare_output(np.array([value], np.float32), np.zeros(1, np.float32))
            self.assertFalse(result['passed'])
            self.assertEqual(result['nonfinite_count'], 1)

    def test_shape_mismatch_never_broadcasts(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            compare_output(np.zeros((1, 3), np.float32), np.zeros(3, np.float32))

    def test_direct_sink_has_distinct_qualification_identity(self):
        from gaussian_generation.npu_branch.worker import runtime_identity
        with patch.object(Path, 'read_text', return_value=''), patch.dict('os.environ', {}, clear=True):
            original = runtime_identity()
            self.assertNotIn('direct_sfb_output', original)
            with patch.dict('os.environ', {'MGS_NPU_DIRECT_SFB': '1'}):
                candidate = runtime_identity()
        self.assertTrue(candidate['direct_sfb_output'])
        self.assertNotEqual(original, candidate)

    def test_deployment_contains_native_output_header(self):
        from initialize.board_sync import NPU_SOURCE_FILES
        self.assertIn('gaussian_generation/npu_branch/output_sink.h', NPU_SOURCE_FILES)

    def test_fp32_diagnostic_patches_only_requested_output_mode(self):
        from gaussian_generation.npu_branch.fp32_blocked_probe import diagnostic_bridge
        raw, sfb = diagnostic_bridge('raw'), diagnostic_bridge('sfb')
        self.assertNotIn('copy_sfb(values[0],out,c.out_bytes);', raw)
        self.assertIn('values[0].read(reinterpret_cast<char*>(out),0,c.out_bytes);', raw)
        self.assertIn('copy_sfb(values[0],out,c.out_bytes);', sfb)

    def test_diagnostic_graph_manifest_rejected_by_production(self):
        import json
        import tempfile
        from common import FULL_WEIGHT_SHA256
        from gaussian_generation.npu_branch.artifacts import validate_bundle
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'manifest.json').write_text(json.dumps(dict(
                schema='mvsplat_head_probe_v1', weight_sha256=FULL_WEIGHT_SHA256)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'schema'):
                validate_bundle(directory, ['to_gaussians'], 'onnx_reference')


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                     'ONNX and ORT required for graph rewrite validation')
class HeadGraphChecks(unittest.TestCase):
    def graph(self):
        import onnx
        from onnx import helper as h, numpy_helper as nh
        rng = np.random.default_rng(142)
        initializers = [nh.from_array(rng.normal(0, .1, s).astype(np.float32), n)
                        for n, s in [('w0', (7, 5, 3, 3)), ('b0', (7,)),
                                     ('w1', (3, 7, 3, 3)), ('b1', (3,))]]
        # The production head is Conv / pointwise GELU / Conv. Relu tests the
        # topology transform separately from the real graph's GELU oracle.
        nodes = [h.make_node('Conv', ['features', 'w0', 'b0'], ['hidden'], pads=[1]*4),
                 h.make_node('Relu', ['hidden'], ['activated']),
                 h.make_node('Conv', ['activated', 'w1', 'b1'], ['result'], pads=[1]*4)]
        graph = h.make_graph(nodes, 'head',
            [h.make_tensor_value_info('features', onnx.TensorProto.FLOAT, [2, 5, 7, 9])],
            [h.make_tensor_value_info('result', onnx.TensorProto.FLOAT, [2, 3, 7, 9])],
            initializer=initializers)
        return h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=9)

    @staticmethod
    def run_graph(model, x):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        return ort.InferenceSession(model.SerializeToString(), sess_options=options,
            providers=['CPUExecutionProvider']).run(None, {'features': x})[0]

    def test_aligned_head_preserves_both_views_and_zero_tail(self):
        model = self.graph()
        x = np.random.default_rng(38).normal(size=(2, 5, 7, 9)).astype(np.float32)
        expected = self.run_graph(model, x)
        transformed = aligned_model(model, 8)
        actual = self.run_graph(transformed, np.pad(x, ((0,0),(0,3),(0,0),(0,0))))
        np.testing.assert_allclose(actual[:, :3], expected, rtol=2e-3, atol=2e-4)
        np.testing.assert_array_equal(actual[:, 3:], 0)
        self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value, 5)

    def test_prefix_suffix_and_single_view_match_full_head(self):
        model = self.graph()
        cases = make_cases(model)
        x = np.random.default_rng(19).normal(size=(2, 5, 7, 9)).astype(np.float32)
        expected = self.run_graph(model, x)
        hidden = self.run_graph(cases['conv_gelu_b2'], x)
        np.testing.assert_allclose(self.run_graph(cases['conv_out_b2'], hidden), expected,
                                   rtol=2e-3, atol=2e-4)
        for index in range(2):
            actual = self.run_graph(cases['full_b1'], x[index:index+1])
            np.testing.assert_allclose(actual, expected[index:index+1], rtol=2e-3, atol=2e-4)

    def test_unknown_topology_fails_before_rewrite(self):
        model = self.graph()
        model.graph.node[1].op_type = 'Softmax'
        with self.assertRaisesRegex(ValueError, 'pointwise'):
            aligned_model(model)


if __name__ == '__main__':
    unittest.main()
