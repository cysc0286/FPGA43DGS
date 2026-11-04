"""Host-only contracts for the isolated U-Net diagnostic runner."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import save, sha
from gaussian_generation.npu_branch import diagnostic_runner as runner


class FakeBridge:
    def __init__(self, *, constant=False, partial=False, fail_at=None):
        self.calls = 0
        self.constant = constant
        self.partial = partial
        self.fail_at = fail_at

    def forward_into(self, input_wire, output_wire):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('fake SDK failure')
        if self.partial:
            output_wire.flat[0] = 7.0
        elif self.constant:
            output_wire[:] = 1.0
        else:
            output_wire[:] = input_wire*2.0+1.0
        return {'sdk_total': 0.001}


class OneCallLateBridge(FakeBridge):
    def __init__(self):
        super().__init__()
        self.previous = None

    def forward_into(self, input_wire, output_wire):
        current = input_wire*2.0+1.0
        output_wire[:] = current if self.previous is None else self.previous
        self.previous = current.copy()
        self.calls += 1
        return {'sdk_total': 0.001}


class FakeSessionContext:
    def __init__(self, bridge, closed):
        self.bridge, self.closed = bridge, closed
        self.bindings, self.memory_layout = 1, None

    def __enter__(self):
        return self.bridge

    def __exit__(self, *_):
        self.closed.append(self.bridge)


class DiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        self.input_spec = dict(layout='NHWC', shape=[1, 2, 2, 1])
        self.output_spec = dict(layout='NHWC', shape=[1, 2, 2, 1])
        self.a = np.array([[[[0, 1], [2, 3]]]], dtype=np.float32)
        self.b = np.array([[[[4, 5], [6, 7]]]], dtype=np.float32)
        self.samples = {'A': (self.a, self.a*2+1), 'B': (self.b, self.b*2+1)}

    def test_a3_b3_a3_uses_one_session_and_audits_nhwc(self):
        fake = FakeBridge()
        result = runner.run_sequence(fake, self.samples, self.input_spec, self.output_spec)
        self.assertEqual(fake.calls, 9)
        self.assertEqual(result['summary']['sdk_forward_calls'], 9)
        self.assertTrue(result['summary']['diagnostic_pass'])
        self.assertTrue(all(row['cpu_reference']['passed'] for row in result['calls']))

    def test_fresh_sdk_context_at_input_boundaries_isolates_one_call_lag(self):
        persistent = runner.run_sequence(OneCallLateBridge(), self.samples,
                                         self.input_spec, self.output_spec)
        self.assertTrue(persistent['summary']['a_initial_repeat_exact'])
        self.assertFalse(persistent['summary']['strict_reference_pass'])
        self.assertFalse(persistent['summary']['diagnostic_pass'])

        closed, created = [], []
        def factory():
            context = FakeSessionContext(OneCallLateBridge(), closed)
            created.append(context)
            return context
        reset = runner.run_sequence(None, self.samples, self.input_spec, self.output_spec,
                                    session_factory=factory)
        self.assertEqual(len(created), 3)
        self.assertEqual(len(closed), 3)
        self.assertEqual(reset['summary']['sdk_context_count'], 3)
        self.assertEqual(reset['summary']['session_policy'], 'one_context_per_input_segment')
        self.assertTrue(reset['summary']['diagnostic_pass'])

    def test_constant_output_is_rejected_as_input_insensitive(self):
        fake = FakeBridge(constant=True)
        result = runner.run_sequence(fake, self.samples, self.input_spec, self.output_spec)
        self.assertFalse(result['summary']['actual_a_b_distinguishable'])
        self.assertFalse(result['summary']['diagnostic_pass'])

    def test_partial_bridge_write_is_caught_by_nan_sentinel(self):
        result = runner.run_sequence(FakeBridge(partial=True), self.samples,
                                     self.input_spec, self.output_spec)
        self.assertFalse(result['complete'])
        self.assertEqual(result['summary']['attempted_calls'], 1)
        self.assertEqual(result['calls'], [])
        self.assertIn('unwritten output', result['error'])

    def test_early_bridge_error_retains_completed_calls(self):
        result = runner.run_sequence(FakeBridge(fail_at=4), self.samples,
                                     self.input_spec, self.output_spec)
        self.assertFalse(result['complete'])
        self.assertEqual(len(result['calls']), 3)
        self.assertEqual(result['summary']['attempted_calls'], 4)
        self.assertIn('fake SDK failure', result['error'])

    def test_manifest_candidate_is_checked_without_catalog_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graphs, bundle, capture = root/'graphs', root/'bundle', root/'capture'
            name = 'refine_unet__stem_00__block'
            for directory in (graphs/name, bundle/name, capture/'refine_unet'/'stem_00'):
                directory.mkdir(parents=True)
            graph_path = graphs/name/'model.onnx'
            oracle_a, oracle_b = graphs/name/'oracle.npz', graphs/name/'oracle_b.npz'
            graph_path.write_bytes(b'onnx-test-graph')
            np.savez(oracle_a, input=self.a, expected=self.samples['A'][1])
            np.savez(oracle_b, input=self.b, expected=self.samples['B'][1])
            source_a, source_b = capture/'refine_unet'/'source_a.npz', capture/'refine_unet'/'source_b.npz'
            source_a.parent.mkdir(parents=True, exist_ok=True)
            np.savez(source_a, input=self.a, expected=self.samples['A'][1])
            np.savez(source_b, input=self.b, expected=self.samples['B'][1])
            cap_a, cap_b = capture/'refine_unet'/'stem_00'/'oracle_a.npz', capture/'refine_unet'/'stem_00'/'oracle_b.npz'
            np.savez(cap_a, input=self.a, expected=self.samples['A'][1])
            np.savez(cap_b, input=self.b, expected=self.samples['B'][1])
            capture_manifest = dict(schema='mvsplat_unet_boundaries_v1',
                modules={'refine_unet': {'captures': {
                    'A': dict(complete=True, source_origin='real-context-A',
                        source_fixture='source_a.npz', source_fixture_sha256=sha(source_a),
                        boundaries={'stem_00': dict(path='stem_00', oracle='stem_00/oracle_a.npz', oracle_sha256=sha(cap_a))}),
                    'B': dict(complete=True, source_origin='real-context-B',
                        source_fixture='source_b.npz', source_fixture_sha256=sha(source_b),
                        boundaries={'stem_00': dict(path='stem_00', oracle='stem_00/oracle_b.npz', oracle_sha256=sha(cap_b))})}}})
            save(capture/'capture_manifest.json', capture_manifest)
            input_shape, output_shape = list(self.a.shape), list(self.samples['A'][1].shape)
            abi = dict(passed=True,
                input=dict(layout='NHWC', shape=[1, 2, 2, 1], logical_shape=input_shape),
                output=dict(layout='NHWC', shape=[1, 2, 2, 1], logical_shape=output_shape),
                npu_executed=False, host_compute_ops=0)
            export = dict(complete=True, diagnostic_only=True, production_qualified=False,
                npu_executed=False, host_onnx_passed=True, input_shape=input_shape,
                output_shape=output_shape, dtype='float32', logical_layout='NCHW',
                oracle_sha256=sha(oracle_a), oracle_b_sha256=sha(oracle_b),
                module='refine_unet', kind='block', boundary=dict(name='stem_00'),
                capture_manifest_sha256=sha(capture/'capture_manifest.json'))
            source = dict(schema='mvsplat_partitions_v1', weight_sha256='weights',
                partitions={name: {**export, 'complete':True}})
            save(graphs/'manifest.json', source)
            compiled_graph = bundle/name/'model.json'
            compiled_graph.write_text('{}', encoding='utf-8')
            raw = bundle/name/'model.raw'
            raw.write_bytes(b'raw')
            compiled = dict(schema='mvsplat_compiled_v1',
                source_manifest_sha256=sha(graphs/'manifest.json'), weight_sha256='weights',
                partitions={name: dict(complete=True, status='compiled_not_executed',
                    graph=f'{name}/model.json', raw=f'{name}/model.raw', export=export, audit=abi)})
            save(bundle/'manifest.json', compiled)
            with patch.object(runner, 'audit', return_value=abi):
                spec = runner.validate_candidate(graphs, bundle, name, capture)
            self.assertEqual(spec['input_spec']['layout'], 'NHWC')
            self.assertEqual(spec['output_spec']['layout'], 'NHWC')
            self.assertEqual(spec['capture_identity']['source_b'], 'real-context-B')

    def test_synthetic_b_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            # The source identity guard is exercised in the full manifest test;
            # this directly checks the explicit synthetic-source policy.
            root = Path(temp)
            (root/'capture_manifest.json').write_text('{}', encoding='utf-8')
            candidate = dict(capture_manifest_sha256=sha(root/'capture_manifest.json'),
                module='m', boundary={'name':'stem_00'}, kind='block')
            data = dict(modules={'m': {'captures': {
                'A': dict(complete=True, source_origin='real-A', boundaries={'stem_00': {}}),
                'B': dict(complete=True, source_origin='synthetic zero B', boundaries={'stem_00': {}})}}})
            save(root/'capture_manifest.json', data)
            candidate['capture_manifest_sha256'] = sha(root/'capture_manifest.json')
            with self.assertRaisesRegex(ValueError, 'B is synthetic'):
                runner._capture_binding(root, candidate)


if __name__ == '__main__':
    unittest.main()
