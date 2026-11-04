"""Diagnostic call-accounting tests; no SDK, checkpoint or board execution."""
import unittest
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import io
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gaussian_generation.npu_branch import paired_inference as paired


class PairedInferenceTests(unittest.TestCase):
    def test_bad_cli_selection_is_rejected_before_model_device_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [flag for name in ('input', 'weights', 'vendor', 'bundle', 'graphs', 'library', 'out')
                      for flag in ('--'+name, str(root/name))]
            for tail in (['--partitions', 'backbone_cnn', '--baseline-partitions', 'refine_unet'],
                         ['--partitions', 'backbone_cnn', 'backbone_cnn'],
                         ['--partitions', 'backbone_cnn', '--worker-timeout', 'nan'],
                         ['--partitions', 'backbone_cnn', '--worker-timeout', '0']):
                with self.subTest(tail=tail), redirect_stderr(io.StringIO()), \
                     patch('gaussian_generation.runtime.ModelRuntime') as model, \
                     patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker:
                    with self.assertRaises(SystemExit): paired.main(common+tail)
                    model.assert_not_called()
                    worker.assert_not_called()
                    self.assertFalse((root/'out').exists())

    def test_main_retains_one_model_and_closes_workers_between_abba_cases(self):
        events, workers, models = [], [], []
        class Tensor:
            def __init__(self, value): self.value = np.asarray(value)
            def unsqueeze(self, axis): return Tensor(np.expand_dims(self.value, axis))
            def detach(self): return self
            def numpy(self): return self.value
        class Model:
            def __init__(self, *args):
                self.np = np
                self.torch = SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
                self.model = self
                self.partition_runtime = None
                models.append(self)
            def attach_partitions(self, worker):
                self.partition_runtime = worker
                worker.attach(self)
            def detach_partitions(self):
                self.partition_runtime.close()
                self.partition_runtime = None
            def __call__(self, *args, **kwargs):
                worker = self.partition_runtime
                events.append(('infer', tuple(worker.parts) if worker else ()))
                if worker:
                    for name in paired.expected_calls(worker.parts):
                        call = PairedInferenceTests.call(name, worker.sequence, 2)
                        call.update(sdk_seconds=[0.] * 5, total_seconds=0.)
                        worker.calls.append(call)
                        worker.sequence += 1
                value = len(worker.parts) if worker else 0
                return SimpleNamespace(**{k: Tensor(np.full((1, 2), value, dtype=np.float32))
                    for k in ('means', 'covariances', 'harmonics', 'opacities')})
        class Worker:
            def __init__(self, backend, bundle, names, work, **options):
                if any(not w.closed for w in workers):
                    raise AssertionError('Two NPU workers were alive together')
                self.parts = {name: {'meta': {}} for name in names}
                self.sequence, self.calls, self.closed = 0, [], False
                self.readiness = {'runtime_identity': {'backbone_settle_passes': 2}}
                self.provenance = {'test_only': True}
                self.timeout = options['timeout']
                workers.append(self)
                events.append(('create', tuple(names)))
            def attach(self, model): events.append(('attach', tuple(self.parts)))
            def report(self, first): return {'calls': [c for c in self.calls if c['sequence'] >= first]}
            def close(self):
                self.closed = True
                events.append(('close', tuple(self.parts)))
        def probe(worker, graphs, out):
            events.append(('probe', tuple(worker.parts)))
            worker.sequence += 10
            return dict(complete=True, strict_passed=False,
                input_freshness_checks=[{'partition': n, 'input_label': 'B'} for n in worker.parts],
                checks=[dict(partition=n, round=1, output_sha256=n) for n in worker.parts])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input'
            source.mkdir()
            np.savez(source/'context.npz', image=np.zeros((2, 3, 4, 4), np.float32))
            args = [flag for name in ('input', 'weights', 'vendor', 'bundle', 'graphs', 'library', 'out')
                    for flag in ('--'+name, str(source if name == 'input' else root/name))]
            args += ['--partitions', 'backbone_cnn', 'corr_refine_net', '--baseline-partitions',
                     'backbone_cnn', '--worker-timeout', '240']
            with patch.object(paired.platform, 'machine', return_value='aarch64'), \
                 patch('gaussian_generation.runtime.ModelRuntime', Model), \
                 patch('gaussian_generation.npu_branch.runtime.PartitionRuntime', Worker), \
                 patch('gaussian_generation.npu_branch.readiness.probe_oracles', side_effect=probe), \
                 patch('initialize.session.worker_environment', return_value={}):
                with redirect_stdout(io.StringIO()): paired.main(args)
            report = json.loads((root/'out/paired.json').read_text())
            self.assertTrue(report['complete'])
            self.assertTrue(report['repeat_exact'])
            self.assertEqual([c['label'] for c in report['cases']], ['baseline', 'expanded', 'expanded', 'baseline'])
            self.assertEqual(len(models), 1)
            self.assertEqual(len(workers), 4)
            self.assertTrue(all(w.closed and w.timeout == 240 for w in workers))
            self.assertIsNone(models[0].partition_runtime)
            self.assertEqual([e[0] for e in events], ['create', 'probe', 'attach', 'infer', 'probe', 'close'] * 4)
            for case in report['cases']:
                self.assertFalse(case['strict_tensor_passed'])
                self.assertTrue(case['oracle_output_stable'])
                self.assertIn('input_freshness_checks', case['oracle_before'])
                self.assertIn('input_freshness_checks', case['oracle_after'])
                self.assertIn('worker_load_seconds', case)
                self.assertIn('oracle_before_seconds', case)
                self.assertIn('oracle_after_seconds', case)
                self.assertIn('close_seconds', case)
            # A changed post-inference oracle fails closed, retaining strict
            # diagnostics and restoring the model before any second worker.
            count = 0
            def changed_probe(worker, graphs, out):
                nonlocal count
                value = probe(worker, graphs, out)
                count += 1
                if count == 2: value['checks'][0]['output_sha256'] = 'changed'
                return value
            failed_args = list(args)
            failed_args[failed_args.index('--out')+1] = str(root/'failed')
            with patch.object(paired.platform, 'machine', return_value='aarch64'), \
                 patch('gaussian_generation.runtime.ModelRuntime', Model), \
                 patch('gaussian_generation.npu_branch.runtime.PartitionRuntime', Worker), \
                 patch('gaussian_generation.npu_branch.readiness.probe_oracles', side_effect=changed_probe), \
                 patch('initialize.session.worker_environment', return_value={}):
                with self.assertRaisesRegex(ValueError, 'oracle changed'): paired.main(failed_args)
            failed = json.loads((root/'failed/paired.json').read_text())
            self.assertFalse(failed['complete'])
            self.assertFalse(failed['cases'][0]['oracle_output_stable'])
            self.assertEqual(len(workers), 5)
            self.assertTrue(workers[-1].closed)
            self.assertIsNone(models[-1].partition_runtime)
            legacy_args = list(args)
            start = legacy_args.index('--baseline-partitions')
            del legacy_args[start:start+2]
            start = legacy_args.index('--worker-timeout')
            del legacy_args[start:start+2]
            legacy_args[legacy_args.index('--out')+1] = str(root/'legacy')
            with patch.object(paired.platform, 'machine', return_value='aarch64'), \
                 patch('gaussian_generation.runtime.ModelRuntime', Model), \
                 patch('gaussian_generation.npu_branch.runtime.PartitionRuntime', Worker), \
                 patch('gaussian_generation.npu_branch.readiness.probe_oracles', side_effect=probe), \
                 patch('initialize.session.worker_environment', return_value={}):
                with redirect_stdout(io.StringIO()): paired.main(legacy_args)
            legacy = json.loads((root/'legacy/paired.json').read_text())
            self.assertTrue(legacy['complete'])
            self.assertEqual([c['label'] for c in legacy['cases']],
                             ['cpu', 'backbone_only', 'expanded', 'expanded', 'backbone_only', 'cpu'])
            self.assertEqual(len(workers), 9)  # CPU cases never create a worker.
            self.assertTrue(all(w.closed and w.timeout == 90 for w in workers[-4:]))

    def test_explicit_baseline_schedule_is_abba_and_rejects_invalid_subsets(self):
        expanded = ['backbone_cnn', 'depth_head', 'corr_refine_net']
        baseline = ['backbone_cnn', 'depth_head']
        self.assertEqual(paired.comparison_cases(expanded, baseline, 2), [
            (0, 'baseline', baseline), (0, 'expanded', expanded),
            (1, 'expanded', expanded), (1, 'baseline', baseline)])
        for bad in ([], ['missing'], ['refine_unet'], ['backbone_cnn', 'backbone_cnn']):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired.comparison_cases(expanded, bad, 2)
        with self.assertRaises(ValueError):
            paired.comparison_cases(['backbone_cnn', 'fused_heads', 'to_gaussians'], None, 2)
        self.assertEqual([c[1] for c in paired.comparison_cases(expanded, None, 2)],
                         ['cpu', 'backbone_only', 'expanded', 'expanded', 'backbone_only', 'cpu'])

    @staticmethod
    def call(name='corr_refine_net', sequence=10, passes=2):
        return dict(name=name, sequence=sequence, event='COMPLETE', backend='npu', npu_executed=True,
            sdk_execution_passes=passes, sdk_stage_names=['input_write', 'forward_submit', 'wait',
                'output_convert', 'sdk_total'], sdk_seconds=[.01, .02, .03, .04, .1], total_seconds=.12)

    def test_sdk_passes_are_charged_with_order_and_sequence_checked(self):
        parts = {'corr_refine_net': {'meta': {}}, 'refine_unet': {'meta': {}}}
        calls = [self.call(), self.call('refine_unet', 11)]
        report = paired.check_calls(calls, parts, 10, {}, .4)
        self.assertEqual(report['logical_calls'], 2)
        self.assertEqual(report['sdk_forward_calls'], 4)
        self.assertAlmostEqual(report['sdk_seconds'], .2)
        for bad in (calls[:-1], list(reversed(calls)), calls + calls[:1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired.check_calls(bad, parts, 10, {}, .4)
        changes = [('sdk_execution_passes', 1), ('sequence', 4), ('backend', 'cpu'),
                   ('npu_executed', False), ('total_seconds', .01),
                   ('sdk_seconds', [.01, .02, .03, .04, float('nan')])]
        for key, value in changes:
            bad = copy.deepcopy(calls)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                paired.check_calls(bad, parts, 10, {}, .4)
        with self.assertRaises(ValueError):
            paired.check_calls(calls, parts, 10, {}, .1)

    def test_cpu_and_legacy_three_bucket_accounting(self):
        self.assertEqual(paired.check_calls([], {}, 0, {}, .2)['sdk_forward_calls'], 0)
        call = self.call('backbone_cnn', 0, 1)
        call.update(sdk_stage_names=['input_write', 'execute_wait', 'output_convert'],
                    sdk_seconds=[.01, .02, .03])
        self.assertAlmostEqual(paired.check_calls([call], {'backbone_cnn': {'meta': {}}},
                                               0, {}, .2)['sdk_seconds'], .06)
        with self.assertRaises(ValueError):
            paired.check_calls([call], {'backbone_cnn': {'meta': {}}}, 0,
                               {'backbone_settle_passes': 2}, .2)

    def test_full_ffn_and_spatial_heads_have_72_logical_calls(self):
        parts = {'backbone_cnn': {'meta': {}}, 'depth_head': {'meta': {}}}
        for name, channels, output in (('to_gaussians', 163, 84), ('to_disparity', 32, 2)):
            parts[name] = {'meta': dict(input_shape=[1, channels, 36, 36],
                output_shape=[1, output, 36, 36], spatial_tiling=dict(
                    schema='mvsplat_local_head_tiles_v1', halo=2, core_size=32,
                    input_shape=[2, channels, 128, 128], output_shape=[2, output, 128, 128]))}
        for index in range(6):
            name = 'transformer_ffn_' + str(index)
            parts[name] = {'meta': dict(input_shape=[2, 256, 32, 32], output_shape=[2, 128, 32, 32],
                ffn_adapter=dict(schema='mvsplat_token_ffn_1x1_v1', partition=name,
                    path=f'backbone.transformer.layers.{index}.cross_attn_ffn.mlp',
                    full_input_shape=[2, 256, 32, 32], full_output_shape=[2, 128, 32, 32],
                    token_layout='NLC', image_layout='NCHW', halo=0))}
        actual = paired.expected_calls(parts)
        self.assertEqual(len(actual), 72)
        self.assertEqual(actual[:8], ['backbone_cnn', *[f'transformer_ffn_{i}' for i in range(6)], 'depth_head'])
        self.assertEqual(actual[8:40], ['to_gaussians'] * 32)
        self.assertEqual(actual[40:], ['to_disparity'] * 32)
        parts.update(corr_refine_net={'meta': {}}, refine_unet={'meta': {}})
        expanded = paired.expected_calls(parts)
        self.assertEqual(len(expanded), 74)
        self.assertEqual(expanded[7:10], ['corr_refine_net', 'depth_head', 'refine_unet'])
        calls = []
        for sequence, name in enumerate(expanded):
            value = self.call(name, sequence, 1 if name == 'depth_head' else 2)
            calls.append(value)
        report = paired.check_calls(calls, parts, 0, {'backbone_settle_passes': 2}, 20)
        self.assertEqual(report['sdk_forward_calls'], 147)

    def test_ffn_odd_tail_and_batch_tiles_count_every_token(self):
        name = 'transformer_ffn_0'
        meta = dict(input_shape=[1, 6, 3, 4], output_shape=[1, 3, 3, 4],
            ffn_adapter=dict(schema='mvsplat_token_ffn_1x1_v1', partition=name,
                path='backbone.transformer.layers.0.cross_attn_ffn.mlp', halo=0,
                full_input_shape=[2, 6, 5, 7], full_output_shape=[2, 3, 5, 7],
                token_layout='NLC', image_layout='NCHW'))
        self.assertEqual(paired.expected_calls({name: {'meta': meta}}), [name] * 8)


if __name__ == '__main__':
    unittest.main()
