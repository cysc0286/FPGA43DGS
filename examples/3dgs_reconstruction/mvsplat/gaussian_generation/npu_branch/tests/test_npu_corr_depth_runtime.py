"""Continuous-depth attachment/accounting contracts; no board or speed claim."""
from collections import deque
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from gaussian_generation.npu_branch.catalog import CORR_DEPTH_COMPONENTS, validate_selection
from gaussian_generation.npu_branch.corr_depth import METHOD, OWNER, patch_source
from gaussian_generation.npu_branch.ffn import PointwiseFFN, adapter_spec
from gaussian_generation.npu_branch.fused_heads import torch_fused_heads
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from gaussian_generation.npu_branch import paired_inference as paired
from gaussian_generation.npu_branch.worker import _forward_sdk, runtime_identity, sdk_passes
from gaussian_generation.npu_branch.tests.test_npu_corr_depth import TOY_SOURCE, load


def predictor_type(*, seam=True):
    extra = '''        self.refine_unet = torch.nn.Conv2d(4, 32, 1)
        self.project = torch.nn.Conv2d(4, 128, 1)
        self.to_gaussians = torch.nn.Sequential(torch.nn.Conv2d(163, 8, 3, padding=1),
            torch.nn.GELU(), torch.nn.Conv2d(8, 84, 3, padding=1))
        self.to_disparity = torch.nn.Sequential(torch.nn.Conv2d(32, 8, 3, padding=1),
            torch.nn.GELU(), torch.nn.Conv2d(8, 2, 3, padding=1))

'''
    source = TOY_SOURCE.replace('    def forward(self, raw_correlation_in):',
        extra + '    def forward(self, raw_correlation_in):')
    source = source.replace('        return pdf', '''        refine = self.refine_unet(raw_correlation_in)
        packed = torch.cat((refine, raw_correlation_in[:, :3], self.project(raw_correlation_in)), dim=1)
        return pdf, self.to_gaussians(packed), self.to_disparity(refine)''')
    return load(patch_source(source)[0] if seam else source)


class SmallEncoder(nn.Module):
    def __init__(self, *, seam=True):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.backbone = nn.Conv2d(4, 4, 1)
        self.backbone.transformer = nn.Module()
        block = nn.Module()
        block.cross_attn_ffn = nn.Module()
        block.cross_attn_ffn.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        self.backbone.transformer.layers = nn.ModuleList([block])
        self.depth_predictor = predictor_type(seam=seam)()

    def forward(self, value):
        value = self.backbone.backbone(value)
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        tokens = self.backbone.transformer.layers[0].cross_attn_ffn.mlp(tokens)
        return self.depth_predictor(tokens.transpose(1, 2).reshape(batch, channels, height, width))


def fake_runtime(model):
    reference = copy.deepcopy(model)
    worker = PartitionRuntime.__new__(PartitionRuntime)
    worker.closed = worker.failed = False
    worker.process = worker.reader = worker.arena = worker.log = None
    worker.model_adapters, worker.model_replacements = [], []
    worker.attached_model = None
    worker.backend, worker.calls, worker.sequence, worker.provenance = 'npu', deque(), 0, {}
    worker.parts = {name: dict(meta=dict(returns_list=False)) for name in
        ('backbone_cnn', 'corr_depth_pipeline', 'refine_unet', 'fused_heads')}
    worker.parts['transformer_ffn_0'] = dict(meta=dict(returns_list=False,
        input_shape=[2, 4, 5, 7], output_shape=[2, 4, 5, 7],
        ffn_adapter=adapter_spec('transformer_ffn_0', [2, 4, 5, 7], [2, 4, 5, 7],
                                [2, 4, 5, 7], [2, 4, 5, 7])))
    routes = dict(backbone_cnn=reference.backbone.backbone,
        transformer_ffn_0=PointwiseFFN(reference.backbone.transformer.layers[0].cross_attn_ffn.mlp),
        corr_depth_pipeline=reference.depth_predictor._corr_to_depth_logits if hasattr(reference.depth_predictor, METHOD) else None,
        refine_unet=reference.depth_predictor.refine_unet,
        fused_heads=torch_fused_heads(reference.depth_predictor.to_gaussians, reference.depth_predictor.to_disparity))
    worker.test_calls = []
    def execute(name, value):
        worker.test_calls.append(name)
        return routes[name](torch.from_numpy(value)).detach().numpy()
    worker.execute = execute
    return worker


def full_parts(*, continuous=False):
    parts = {name: {'meta': {}} for name in ('backbone_cnn', 'corr_refine_net', 'depth_head', 'refine_unet')}
    for name, channels, output in (('to_gaussians', 163, 84), ('to_disparity', 32, 2)):
        parts[name] = {'meta': dict(input_shape=[1, channels, 36, 36], output_shape=[1, output, 36, 36],
            spatial_tiling=dict(schema='mvsplat_local_head_tiles_v1', halo=2, core_size=32,
                input_shape=[2, channels, 128, 128], output_shape=[2, output, 128, 128]))}
    for index in range(6):
        name = 'transformer_ffn_' + str(index)
        parts[name] = {'meta': dict(input_shape=[2, 256, 32, 32], output_shape=[2, 128, 32, 32],
            ffn_adapter=adapter_spec(name, [2, 256, 32, 32], [2, 128, 32, 32],
                                    [2, 256, 32, 32], [2, 128, 32, 32]))}
    if continuous:
        del parts['corr_refine_net'], parts['depth_head']
        parts['corr_depth_pipeline'] = {'meta': {}}
    return parts


class CorrDepthRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.model = SmallEncoder().eval()
        self.value = torch.randn(2, 4, 5, 7)

    def test_real_method_adapter_coexists_with_ffn_and_fused_heads_then_restores(self):
        worker = fake_runtime(self.model)
        predictor = self.model.depth_predictor
        originals = {name: getattr(predictor, name) for name in
                     ('corr_refine_net', 'regressor_residual', 'depth_head_lowres', 'refine_unet', 'to_gaussians', 'to_disparity')}
        original_method = getattr(predictor, METHOD).__func__
        original_ffn = self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        counts = {name: 0 for name in ('corr_refine_net', 'regressor_residual', 'depth_head_lowres')}
        handles = [getattr(predictor, name).register_forward_hook(
            lambda _module, _args, _out, name=name: counts.__setitem__(name, counts[name] + 1)) for name in counts]
        with torch.inference_mode():
            expected = self.model(self.value)
            for name in counts: counts[name] = 0
            worker.attach(self.model)
            self.assertNotIsInstance(getattr(predictor, METHOD), nn.Module)
            for name in ('corr_refine_net', 'regressor_residual', 'depth_head_lowres'):
                self.assertIs(getattr(predictor, name), originals[name])
            actual = self.model(self.value)
            for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            self.assertEqual(worker.test_calls, ['backbone_cnn', 'transformer_ffn_0', 'corr_depth_pipeline', 'refine_unet', 'fused_heads'])
            self.assertEqual(counts, {name: 0 for name in counts})
            worker.close(); worker.close()
            self.assertNotIn(METHOD, predictor.__dict__)
            self.assertFalse(hasattr(predictor, OWNER))
            self.assertIs(getattr(predictor, METHOD).__func__, original_method)
            self.assertIs(self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp, original_ffn)
            for name, original in originals.items(): self.assertIs(getattr(predictor, name), original)
            worker.execute = lambda *_: self.fail('Restored CPU model called the closed worker')
            actual = self.model(self.value)
            for a, b in zip(actual, expected): torch.testing.assert_close(a, b, rtol=0, atol=0)
        for handle in handles: handle.remove()
        for key, value in state.items(): self.assertTrue(torch.equal(value, self.model.state_dict()[key]))

    def test_missing_method_seam_fails_before_any_other_replacement(self):
        model = SmallEncoder(seam=False).eval()
        worker = fake_runtime(model)
        original_ffn = model.backbone.transformer.layers[0].cross_attn_ffn.mlp
        with patch('gaussian_generation.npu_branch.runtime.replace') as replacement:
            with self.assertRaisesRegex(ValueError, 'lacks the generated'):
                worker.attach(model)
        replacement.assert_not_called()
        self.assertIs(model.backbone.transformer.layers[0].cross_attn_ffn.mlp, original_ffn)
        self.assertEqual(worker.model_adapters, [])
        self.assertEqual(worker.model_replacements, [])
        worker.close()

    def test_late_attach_failure_restores_method_and_ffn(self):
        worker = fake_runtime(self.model)
        original = getattr(self.model.depth_predictor, METHOD)
        original_ffn = self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp
        with patch('gaussian_generation.npu_branch.runtime.replace', side_effect=RuntimeError('replacement failed')):
            with self.assertRaisesRegex(RuntimeError, 'replacement failed'):
                worker.attach(self.model)
        self.assertEqual(getattr(self.model.depth_predictor, METHOD), original)
        self.assertFalse(hasattr(self.model.depth_predictor, OWNER))
        self.assertIs(self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp, original_ffn)
        self.assertEqual(worker.model_adapters, [])
        worker.attach(self.model)
        with torch.inference_mode(): self.model(self.value)
        worker.close()

    def test_preexisting_instance_method_binding_is_restored_exactly(self):
        predictor = self.model.depth_predictor
        original = getattr(predictor, METHOD)
        setattr(predictor, METHOD, original)
        worker = fake_runtime(self.model)
        worker.attach(self.model)
        worker.close()
        self.assertIs(predictor.__dict__[METHOD], original)

    def test_composite_conflict_is_rejected_before_bundle_worker_or_allocation(self):
        for name in CORR_DEPTH_COMPONENTS:
            with self.subTest(name=name), \
                    patch('gaussian_generation.npu_branch.runtime.validate_bundle') as bundle, \
                    patch('gaussian_generation.npu_branch.runtime.MappedBuffers') as buffers, \
                    patch('gaussian_generation.npu_branch.runtime.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    PartitionRuntime('npu', 'unused', ['corr_depth_pipeline', name], 'unused')
                bundle.assert_not_called(); buffers.assert_not_called(); process.assert_not_called()

    def test_attach_rejects_conflict_before_mutating_method(self):
        worker = fake_runtime(self.model)
        worker.parts['depth_head'] = {'meta': {}}
        original = getattr(self.model.depth_predictor, METHOD)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            worker.attach(self.model)
        self.assertEqual(getattr(self.model.depth_predictor, METHOD), original)
        worker.close()

    def test_continuous_graph_requires_fresh_input_before_buffer_allocation(self):
        part = {'corr_depth_pipeline': {'meta': {}}}
        with patch('gaussian_generation.npu_branch.runtime.validate_bundle', return_value=({}, part)), \
                patch('gaussian_generation.npu_branch.runtime.buffer_plan') as plan:
            with self.assertRaisesRegex(ValueError, 'fresh SDK input'):
                PartitionRuntime('npu', 'unused', list(part), 'unused', environment={'MGS_NPU_FRESH_INPUT': '0'})
        plan.assert_not_called()

    def test_report_omits_all_fused_operations_from_cpu_list(self):
        worker = fake_runtime(self.model)
        report = worker.report()
        for name in (*CORR_DEPTH_COMPONENTS, 'to_gaussians', 'to_disparity', 'corr_depth_pipeline', 'fused_heads'):
            self.assertNotIn(name, report['cpu_partitions'])
        self.assertIn('upsampler', report['cpu_partitions'])
        del worker.parts['corr_depth_pipeline']
        for name in CORR_DEPTH_COMPONENTS: self.assertIn(name, worker.report()['cpu_partitions'])
        worker.close()


class CorrDepthWorkerAccountingTests(unittest.TestCase):
    def test_every_continuous_sdk_forward_is_executed_and_charged(self):
        value = np.ones((2, 4, 3, 5), dtype=np.float32)
        output = np.empty((2, 2, 3, 5), dtype=np.float32)
        calls = []
        def forward(context, ip, ib, op, ob, measured):
            calls.append((context, ib, ob))
            for index, seconds in enumerate((.01, .02, .03, .04, .10)): measured[index] = seconds
            return 0
        stages, count = _forward_sdk('corr_depth_pipeline', {}, 7, value, output, forward, lambda: b'failure', 5)
        self.assertEqual(count, 2)
        self.assertEqual(calls, [(7, value.nbytes, output.nbytes)] * 2)
        np.testing.assert_allclose(stages, [.02, .04, .06, .08, .20])
        self.assertEqual(sdk_passes('depth_head', {}), 1)

    def test_second_sdk_failure_cannot_return_a_successful_cached_output(self):
        value = np.ones((2, 4, 3, 5), dtype=np.float32)
        output = np.empty_like(value)
        calls = []
        def forward(*args):
            calls.append(1)
            return 1 if len(calls) == 2 else 0
        with self.assertRaisesRegex(RuntimeError, 'second forward failed'):
            _forward_sdk('corr_depth_pipeline', {}, 1, value, output, forward, lambda: b'second forward failed', 3)
        self.assertEqual(len(calls), 2)

    def test_sdk_identity_names_continuous_graph_separately(self):
        with patch.dict(os.environ, {'MGS_NPU_DDR_GUARD_MIB': '64', 'MGS_NPU_FRESH_INPUT': '1'}, clear=True), \
                patch.object(Path, 'read_text', return_value=''):
            identity = runtime_identity([], [], ['refine_unet'], ['corr_depth_pipeline'])
        self.assertEqual(identity['corr_depth_partitions'], ['corr_depth_pipeline'])
        self.assertEqual(identity['corr_depth_settle_passes'], 2)
        self.assertEqual(identity['refinement_partitions'], ['refine_unet'])
        self.assertTrue(identity['fresh_input_tensor'])

    def test_74_147_to_73_146_contract_and_order(self):
        for continuous, logical, sdk in ((False, 74, 147), (True, 73, 146)):
            parts = full_parts(continuous=continuous)
            order = paired.expected_calls(parts)
            self.assertEqual(len(order), logical)
            if continuous:
                self.assertEqual(order[7:9], ['corr_depth_pipeline', 'refine_unet'])
                for name in CORR_DEPTH_COMPONENTS: self.assertNotIn(name, order)
            calls = [dict(event='COMPLETE', name=name, sequence=index, backend='npu', npu_executed=True,
                sdk_execution_passes=1 if name == 'depth_head' else 2,
                sdk_stage_names=['input_write', 'execute_wait', 'output_convert'],
                sdk_seconds=[.01, .02, .03], total_seconds=.10) for index, name in enumerate(order)]
            report = paired.check_calls(calls, parts, 0, {'backbone_settle_passes': 2}, 20)
            self.assertEqual(report['sdk_forward_calls'], sdk)
            if continuous:
                calls[7]['sdk_execution_passes'] = 1
                with self.assertRaisesRegex(ValueError, 'charged SDK pass'):
                    paired.check_calls(calls, parts, 0, {'backbone_settle_passes': 2}, 20)

    def test_paired_fusion_comparison_preserves_semantic_coverage(self):
        for baseline in (['backbone_cnn', 'corr_refine_net', 'depth_head'],
                         ['backbone_cnn', *CORR_DEPTH_COMPONENTS]):
            expanded = ['backbone_cnn', 'corr_depth_pipeline']
            cases = paired.comparison_cases(expanded, baseline, 2)
            self.assertEqual([x[1] for x in cases], ['baseline', 'expanded', 'expanded', 'baseline'])
        for baseline in ([], ['corr_depth_pipeline', 'corr_depth_pipeline'],
                         ['upsampler'], ['corr_depth_pipeline', 'backbone_cnn']):
            with self.assertRaises(ValueError):
                paired.comparison_cases(['backbone_cnn', 'corr_depth_pipeline'], baseline, 2)

    def test_comparison_tool_only_opts_into_model_seam_for_new_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            flags = [x for name in ('input', 'weights', 'vendor', 'bundle', 'graphs', 'library', 'out')
                     for x in ('--' + name, str(root / name))]
            for continuous in (False, True):
                chosen = 'corr_depth_pipeline' if continuous else 'corr_refine_net'
                with patch('gaussian_generation.npu_branch.paired_inference.platform.machine', return_value='aarch64'), \
                        patch('gaussian_generation.runtime.ModelRuntime', side_effect=RuntimeError('stop before model load')) as model:
                    args = flags.copy()
                    args[args.index('--out') + 1] = str(root / ('continuous' if continuous else 'ordinary'))
                    with self.assertRaisesRegex(RuntimeError, 'stop before model load'):
                        paired.main(args + ['--partitions', 'backbone_cnn', chosen,
                                            '--baseline-partitions', 'backbone_cnn'])
                self.assertEqual(model.call_args.kwargs, {'corr_depth_pipeline': True} if continuous else {})


if __name__ == '__main__':
    unittest.main()
