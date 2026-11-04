"""Local synthetic contracts; these tests never load a device/bridge library."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from common import sha
from gaussian_generation.npu_branch.artifacts import from_wire, to_wire
from gaussian_generation.npu_branch import onnx_prefix_diagnostic as diagnostic
from gaussian_generation.npu_branch.onnx_prefix_probe import extract_prefix
from gaussian_generation.npu_branch.onnx_output_control import change_output_tail


def samples_and_spec():
    shape = [2, 2, 2, 3]
    a = np.arange(24, dtype=np.float32).reshape(shape) + 1
    b = a * 3 + 7
    samples = {key: (value, value * 2 + 3) for key, value in (('A', a), ('B', b))}
    spec = dict(layout='NHWC', shape=[2, 2, 3, 2], logical_shape=shape)
    return samples, spec


class FakeSession:
    """Explicit SDK calls, with controllable stale/nonfinite/failing behavior."""
    def __init__(self, samples, spec, mode='correct'):
        self.samples, self.spec, self.mode = samples, spec, mode
        self.calls, self.previous = [], 'A'

    def forward_into(self, value, output):
        logical = from_wire(value, self.spec)
        label = next(key for key, sample in self.samples.items() if np.array_equal(logical, sample[0]))
        self.calls.append(label)
        if self.mode == 'raises_second' and len(self.calls) == 2:
            raise RuntimeError('injected SDK failure')
        if self.mode == 'unwritten':
            return dict(input_write=.001, sdk_total=.005)
        selected = self.previous if self.mode == 'one_call_late' else ('A' if self.mode == 'constant' else label)
        actual = self.samples[selected][1].copy()
        self.previous = label
        if self.mode == 'varying':
            actual.flat[0] += np.float32(len(self.calls) * .00001)
        np.copyto(output, to_wire(actual, self.spec))
        if self.mode == 'bad_timing':
            return dict(sdk_total=float('nan'))
        return dict(input_write=.001, forward_submit=.001, wait=.002,
                    output_convert=.001, sdk_total=.005)


class PrefixSequenceTests(unittest.TestCase):
    def run_fake(self, mode='correct', **kwargs):
        samples, spec = samples_and_spec()
        session = FakeSession(samples, spec, mode)
        return diagnostic.run_sequence(session, samples, spec, spec, **kwargs), session

    def test_fixed_sequence_and_nhwc_are_preserved(self):
        result, session = self.run_fake()
        self.assertEqual(session.calls, list('AABBAA'))
        summary = result['summary']
        self.assertTrue(summary['diagnostic_pass'])
        self.assertEqual(summary['sdk_forward_attempted'], 6)
        self.assertEqual(summary['sdk_forward_completed'], 6)
        self.assertTrue(summary['repeat_exact'])
        self.assertTrue(summary['actual_ab_distinguishable'])
        self.assertEqual(summary['oracle_provenance'], 'source_onnx_derived')
        self.assertFalse(summary['acceptance'])
        self.assertFalse(summary['production_qualified'])
        self.assertAlmostEqual(summary['bridge_component_sums_seconds']['sdk_total'], .030)

    def test_abba_is_a_fixed_original_sequence(self):
        result, session = self.run_fake(sequence='abba')
        self.assertEqual(session.calls, list('ABBA'))
        self.assertTrue(result['summary']['diagnostic_pass'])

    def test_stale_first_pass_fails_without_hidden_settling(self):
        result, session = self.run_fake('one_call_late')
        summary = result['summary']
        self.assertEqual(len(session.calls), 6)
        self.assertTrue(summary['stale_output_detected'])
        self.assertEqual(summary['stale_final_pass_count'], 2)
        self.assertFalse(summary['final_outputs_strict_pass'])
        self.assertFalse(summary['diagnostic_pass'])

    def test_explicit_second_pass_charges_every_stale_call(self):
        result, session = self.run_fake('one_call_late', sdk_passes=2)
        summary = result['summary']
        self.assertEqual(session.calls, list('AAAABBBBAAAA'))
        self.assertEqual(summary['sdk_forward_attempted'], 12)
        self.assertEqual(summary['sdk_forward_completed'], 12)
        self.assertEqual(summary['stale_pass_count'], 2)
        self.assertEqual(summary['stale_final_pass_count'], 0)
        self.assertFalse(summary['all_sdk_passes_strict_pass'])
        self.assertTrue(summary['final_outputs_strict_pass'])
        self.assertTrue(summary['diagnostic_pass'])
        self.assertAlmostEqual(summary['bridge_component_sums_seconds']['sdk_total'], .060)
        self.assertEqual(len(result['sdk_calls']), 12)
        self.assertFalse(summary['acceptance'])

    def test_constant_output_is_not_input_responsive(self):
        result, _ = self.run_fake('constant')
        self.assertFalse(result['summary']['actual_ab_distinguishable'])
        self.assertFalse(result['summary']['diagnostic_pass'])

    def test_repeat_changes_cannot_hide_inside_numerical_tolerance(self):
        result, _ = self.run_fake('varying')
        summary = result['summary']
        self.assertTrue(summary['final_outputs_strict_pass'])
        self.assertFalse(summary['repeat_exact'])
        self.assertFalse(summary['all_sdk_same_input_same_pass_repeats_exact'])
        self.assertFalse(summary['diagnostic_pass'])

    def test_unwritten_nan_buffer_is_detected_and_charged(self):
        result, session = self.run_fake('unwritten')
        self.assertEqual(len(session.calls), 1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['summary']['sdk_forward_completed'], 1)
        self.assertEqual(result['summary']['nonfinite_or_unwritten_count'], 1)
        self.assertIn('nonfinite or unwritten', result['error'])

    def test_sdk_failure_records_attempt_and_completed_separately(self):
        result, _ = self.run_fake('raises_second')
        self.assertFalse(result['complete'])
        self.assertEqual(result['summary']['sdk_forward_attempted'], 2)
        self.assertEqual(result['summary']['sdk_forward_completed'], 1)
        self.assertEqual(len(result['requests']), 1)
        self.assertIn('injected SDK failure', result['error'])

    def test_invalid_timing_does_not_escape_or_emit_non_json_nan(self):
        result, _ = self.run_fake('bad_timing')
        self.assertIn('Invalid bridge timing', result['error'])
        self.assertEqual(result['summary']['sdk_forward_completed'], 1)
        self.assertEqual(result['summary']['nonfinite_or_unwritten_count'], 0)
        json.dumps(result, allow_nan=False)

    def test_unfixed_sequence_or_extra_passes_are_rejected(self):
        for kwargs in ({'sequence': 'AB'}, {'sdk_passes': 0}, {'sdk_passes': 3}, {'sdk_passes': True}):
            with self.assertRaises(ValueError):
                self.run_fake(**kwargs)


class PrefixLineageTests(unittest.TestCase):
    def make_models(self, directory, *, identity=False):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        source = Path(directory) / 'full.onnx'
        candidate = Path(directory) / 'prefix.onnx'
        model = helper.make_model(helper.make_graph([
            helper.make_node('Conv', ['features', 'weight'], ['conv'], name='/inner/inner.0/Conv'),
            helper.make_node('Relu', ['conv'], ['output'], name='suffix')], 'source',
            [helper.make_tensor_value_info('features', TensorProto.FLOAT, [2, 2, 2, 3])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 2, 3])],
            initializer=[numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2, 1, 1), 'weight')]),
            opset_imports=[helper.make_opsetid('', 17)])
        onnx.save(model, str(source))
        info = extract_prefix(model, candidate, op_type='Conv', node_name='/inner/inner.0/Conv')
        meta = dict(info, prefix_label='first_conv', output_shape=[2, 3, 2, 3],
                    onnx_output_shape=[2, 2, 3, 3])
        if identity:
            changed, tail = change_output_tail(onnx.load(str(candidate)), [2, 3, 2, 3],
                                               identity_conv=True, output_layout='NHWC')
            onnx.save(changed, str(candidate))
            meta.update(output_tail_control=tail, injected_identity_parameters=12)
        return source, candidate, meta

    def test_exact_original_prefix_and_real_identity_are_audited(self):
        with tempfile.TemporaryDirectory() as folder:
            for identity in (False, True):
                source, candidate, meta = self.make_models(folder, identity=identity)
                result = diagnostic._check_lineage(source, candidate, meta)
                self.assertTrue(result['original_initializer_bytes_exact'])
                self.assertEqual(result['identity_conv_tail'], identity)
                self.assertEqual(result['injected_identity_parameters'], 12 if identity else 0)
                self.assertEqual(result['injected_identity_multiplications'], 108 if identity else 0)

    def test_weight_tampering_or_layout_relabel_is_rejected(self):
        import onnx
        from onnx import numpy_helper
        with tempfile.TemporaryDirectory() as folder:
            source, candidate, meta = self.make_models(folder)
            model = onnx.load(str(candidate))
            model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((3, 2, 1, 1), dtype=np.float32), 'weight'))
            onnx.save(model, str(candidate))
            with self.assertRaisesRegex(ValueError, 'initializer bytes'):
                diagnostic._check_lineage(source, candidate, meta)
            source, candidate, meta = self.make_models(folder)
            model = onnx.load(str(candidate))
            model.graph.node[-1].attribute[0].ints[:] = [0, 1, 2, 3]
            onnx.save(model, str(candidate))
            with self.assertRaisesRegex(ValueError, 'transpose/axis'):
                diagnostic._check_lineage(source, candidate, meta)

    def test_identity_tail_must_actually_be_identity(self):
        import onnx
        from onnx import numpy_helper
        with tempfile.TemporaryDirectory() as folder:
            source, candidate, meta = self.make_models(folder, identity=True)
            model = onnx.load(str(candidate))
            for index, value in enumerate(model.graph.initializer):
                if value.name == '__mgs_output_control_weight':
                    model.graph.initializer[index].CopyFrom(numpy_helper.from_array(
                        np.zeros((3, 3, 1, 1), dtype=np.float32), value.name))
            onnx.save(model, str(candidate))
            with self.assertRaisesRegex(ValueError, 'not exactly the declared identity'):
                diagnostic._check_lineage(source, candidate, meta)

    def test_manifest_pins_and_registered_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text('{}', encoding='utf-8')
            (root / 'capture_manifest.json').write_text('{}', encoding='utf-8')
            pins = {key: sha(root / ('capture_manifest.json' if key == 'arm_fixtures' else 'manifest.json'))
                    for key in diagnostic.MANIFEST_KEYS}
            bad_pins = dict(pins, export='0' * 64)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                diagnostic.validate_candidate(root, root, 'corr_p_conv', root, root, bad_pins)
            with self.assertRaisesRegex(ValueError, 'registered production'):
                diagnostic.validate_candidate(root, root, 'backbone_cnn', root, root, pins)

    def test_partition_path_traversal_is_rejected_before_any_file_read(self):
        for name in ('', '.', '..', '../corr_p_conv', 'c/corr_p_conv',
                     'c\\corr_p_conv', 'C:corr_p_conv', '/absolute', 'bad\x00name'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'path component'):
                diagnostic.validate_candidate('absent', 'absent', name, 'absent', 'absent', {})

    def test_root_shape_metadata_cannot_silently_change(self):
        expected = dict(input_shape=[1, 2, 3, 128, 128])
        self.assertEqual(diagnostic._check_root_input_shapes(expected, expected, expected), expected['input_shape'])
        for shape in (None, [2, 3, 128, 128], [1, 1, 3, 128, 128],
                      [1, 2, 3, 0, 128], [1, 2, 3, 256, 128]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'root input shapes'):
                diagnostic._check_root_input_shapes(dict(input_shape=shape), expected, expected)
            with self.subTest(compiled_shape=shape), self.assertRaisesRegex(ValueError, 'root input shapes'):
                diagnostic._check_root_input_shapes(expected, dict(input_shape=shape), expected)

    def test_capture_metadata_must_match_full_stage_shapes(self):
        stage = dict(input_shape=[2, 4, 8, 8], output_shape=[2, 2, 8, 8])
        diagnostic._check_capture_shapes(stage, stage, stage['input_shape'])
        for changed in (dict(stage, input_shape=[1, 4, 8, 8]),
                        dict(stage, output_shape=[2, 4, 8, 8]),
                        dict(output_shape=stage['output_shape'])):
            with self.assertRaisesRegex(ValueError, 'Capture input/output shape'):
                diagnostic._check_capture_shapes(changed, stage, stage['input_shape'])

    def test_preflight_never_opens_bridge_or_generates_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(out=Path(folder) / 'preflight', partition='corr_p_conv',
                graphs='unused', bundle='unused', full_source='unused', arm_fixtures='unused',
                preflight_only=True, library=None, library_sha256=None,
                **{key + '_manifest_sha256': 'pinned' for key in diagnostic.MANIFEST_KEYS})
            spec = dict(identities={}, provenance={}, lineage={}, audit={})
            with mock.patch.object(diagnostic, 'validate_candidate', return_value=spec), \
                    mock.patch.object(diagnostic, 'BridgeSession') as bridge:
                result = diagnostic.execute(args)
            bridge.assert_not_called()
            stored = json.loads((args.out / 'prefix_diagnostic_result.json').read_text())
            self.assertTrue(result['complete'])
            self.assertFalse(stored['npu_executed'])
            self.assertFalse(stored['execution_performed'])
            self.assertFalse(stored['formal_receipt'])
            self.assertFalse(stored['acceptance'])
            self.assertEqual(stored['schema'], diagnostic.SCHEMA)
            self.assertFalse(stored['independent_arm_intermediate'])


if __name__ == '__main__':
    unittest.main()
