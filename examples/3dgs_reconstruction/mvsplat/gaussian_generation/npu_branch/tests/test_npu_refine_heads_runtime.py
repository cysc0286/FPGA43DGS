"""Opt-in tail/correlation lifecycle and SDK accounting; no physical NPU test."""
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

from gaussian_generation.npu_branch import corr_depth, paired_inference as paired, refine_heads
from gaussian_generation.npu_branch.catalog import (
    CORR_DEPTH_COMPONENTS, REFINE_HEADS_COMPONENTS, semantic_coverage,
)
from gaussian_generation.npu_branch.ffn import PointwiseFFN, adapter_spec
from gaussian_generation.npu_branch.fused_heads import attach_fused_heads
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from gaussian_generation.npu_branch.worker import _forward_sdk, runtime_identity, sdk_passes
from gaussian_generation.npu_branch.tests.test_npu_refine_heads import TOY_SOURCE, load


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
        source, _ = corr_depth.patch_source(TOY_SOURCE)
        if seam:
            source, _ = refine_heads.patch_source(source)
        self.depth_predictor = load(source)()

    def forward(self, value, images, near, far):
        value = self.backbone.backbone(value)
        batch, channels, height, width = value.shape
        tokens = value.flatten(2).transpose(1, 2)
        tokens = self.backbone.transformer.layers[0].cross_attn_ffn.mlp(tokens)
        return self.depth_predictor(tokens.transpose(1, 2).reshape(batch, channels, height, width),
                                    images, near, far)


def fake_runtime(model):
    """Independent CPU test executor; never used by a production backend."""
    reference = copy.deepcopy(model)
    worker = PartitionRuntime.__new__(PartitionRuntime)
    worker.closed = worker.failed = False
    worker.process = worker.reader = worker.arena = worker.log = None
    worker.model_adapters, worker.model_replacements = [], []
    worker.attached_model = None
    worker.backend, worker.calls, worker.sequence, worker.provenance = 'npu', deque(), 0, {}
    worker.parts = {name: dict(meta=dict(returns_list=False)) for name in
                    ('backbone_cnn', 'corr_depth_pipeline', 'refine_heads_pipeline')}
    worker.parts['transformer_ffn_0'] = dict(meta=dict(returns_list=False,
        input_shape=[2, 4, 5, 7], output_shape=[2, 4, 5, 7],
        ffn_adapter=adapter_spec('transformer_ffn_0', [2, 4, 5, 7], [2, 4, 5, 7],
                                [2, 4, 5, 7], [2, 4, 5, 7])))
    routes = dict(backbone_cnn=reference.backbone.backbone,
        transformer_ffn_0=PointwiseFFN(reference.backbone.transformer.layers[0].cross_attn_ffn.mlp),
        corr_depth_pipeline=reference.depth_predictor._corr_to_depth_logits)
    worker.test_calls = []
    def execute(name, value):
        worker.test_calls.append(name)
        tensor = torch.from_numpy(value)
        if name == 'refine_heads_pipeline':
            raw = reference.depth_predictor._refine_to_heads(
                *(tensor[:, begin:end] for begin, end in ((0, 128), (128, 131), (131, 132), (132, 133))))
            return np.concatenate([item.detach().numpy() for item in raw], axis=1)
        return routes[name](tensor).detach().numpy()
    worker.execute = execute
    return worker


def full_parts():
    parts = {name: {'meta': {}} for name in
             ('backbone_cnn', 'corr_depth_pipeline', 'refine_heads_pipeline')}
    for index in range(6):
        name = 'transformer_ffn_' + str(index)
        parts[name] = {'meta': dict(input_shape=[2, 256, 32, 32], output_shape=[2, 128, 32, 32],
            ffn_adapter=adapter_spec(name, [2, 256, 32, 32], [2, 128, 32, 32],
                                    [2, 256, 32, 32], [2, 128, 32, 32]))}
    return parts


class RefineHeadsRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(79)
        self.model = SmallEncoder().eval()
        self.inputs = (torch.randn(2, 4, 5, 7), torch.randn(2, 3, 5, 7),
                       torch.full((1, 2), .1), torch.full((1, 2), 10.))

    def test_two_method_adapters_with_ffn_abba_and_full_restore(self):
        worker = fake_runtime(self.model)
        predictor = self.model.depth_predictor
        original_modules = {name: getattr(predictor, name) for name in
                            (*CORR_DEPTH_COMPONENTS[:2], 'depth_head_lowres', *REFINE_HEADS_COMPONENTS, 'upsampler')}
        original_ffn = self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp
        originals = {name: getattr(predictor, name).__func__ for name in
                     (corr_depth.METHOD, refine_heads.METHOD)}
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        counts = dict.fromkeys(REFINE_HEADS_COMPONENTS, 0)
        handles = [getattr(predictor, name).register_forward_hook(
            lambda *args, name=name: counts.__setitem__(name, counts[name] + 1)) for name in counts]
        up_calls = []
        handles.append(predictor.upsampler.register_forward_hook(lambda *args: up_calls.append(1)))
        values = [self.inputs, (self.inputs[0]+.5, self.inputs[1]-.25, *self.inputs[2:])]
        try:
            with torch.inference_mode():
                expected = [self.model(*args) for args in values]
                for name in counts: counts[name] = 0
                up_calls.clear()
                worker.attach(self.model)
                for name, module in original_modules.items(): self.assertIs(getattr(predictor, name), module)
                repeats = {}
                for index in (0, 1, 1, 0):
                    actual = self.model(*values[index])
                    # Existing FFN proxy changes Linear to equivalent Conv1x1;
                    # its floating point accumulation is not bit identical.
                    for a, b in zip(actual, expected[index]): torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
                    if index in repeats:
                        for a, b in zip(actual, repeats[index]): torch.testing.assert_close(a, b, rtol=0, atol=0)
                    repeats[index] = actual
                self.assertEqual(counts, dict.fromkeys(counts, 0))
                self.assertEqual(len(up_calls), 4)
                self.assertEqual(worker.test_calls, ['backbone_cnn', 'transformer_ffn_0',
                    'corr_depth_pipeline', 'refine_heads_pipeline'] * 4)
                worker.close(); worker.close()
                self.assertEqual(worker.model_adapters, [])
                for method, function in originals.items():
                    self.assertNotIn(method, predictor.__dict__)
                    self.assertIs(getattr(predictor, method).__func__, function)
                self.assertFalse(hasattr(predictor, corr_depth.OWNER))
                self.assertFalse(hasattr(predictor, refine_heads.OWNER))
                self.assertIs(self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp, original_ffn)
                worker.execute = lambda *_: self.fail('Restored CPU model used the closed worker')
                actual = self.model(*self.inputs)
                for a, b in zip(actual, expected[0]): torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(state.keys(), self.model.state_dict().keys())
        for name, value in state.items(): self.assertTrue(torch.equal(value, self.model.state_dict()[name]))

    def test_missing_tail_seam_restores_corr_before_ordinary_replacements(self):
        model = SmallEncoder(seam=False).eval()
        worker = fake_runtime(model)
        original_corr = model.depth_predictor._corr_to_depth_logits
        original_ffn = model.backbone.transformer.layers[0].cross_attn_ffn.mlp
        with patch('gaussian_generation.npu_branch.runtime.replace') as replace:
            with self.assertRaisesRegex(ValueError, 'generated refinement-head seam'):
                worker.attach(model)
        replace.assert_not_called()
        self.assertEqual(model.depth_predictor._corr_to_depth_logits, original_corr)
        self.assertFalse(hasattr(model.depth_predictor, corr_depth.OWNER))
        self.assertFalse(hasattr(model.depth_predictor, refine_heads.OWNER))
        self.assertIs(model.backbone.transformer.layers[0].cross_attn_ffn.mlp, original_ffn)
        self.assertEqual(worker.model_adapters, [])
        worker.close()

    def test_later_replacement_failure_restores_both_methods_and_ffn(self):
        worker = fake_runtime(self.model)
        predictor = self.model.depth_predictor
        original = (predictor._corr_to_depth_logits, predictor._refine_to_heads)
        ffn = self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp
        with patch('gaussian_generation.npu_branch.runtime.replace', side_effect=RuntimeError('ordinary failed')):
            with self.assertRaisesRegex(RuntimeError, 'ordinary failed'): worker.attach(self.model)
        self.assertEqual((predictor._corr_to_depth_logits, predictor._refine_to_heads), original)
        self.assertIs(self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp, ffn)
        self.assertFalse(hasattr(predictor, refine_heads.OWNER))
        worker.attach(self.model)
        with torch.inference_mode(): self.model(*self.inputs)
        worker.close()

    def test_fused_heads_cannot_attach_after_tail_owner_or_mutate_it(self):
        worker = fake_runtime(self.model)
        worker.attach(self.model)
        predictor = self.model.depth_predictor
        method = predictor._refine_to_heads
        modules = (predictor.to_gaussians, predictor.to_disparity)
        with self.assertRaisesRegex(ValueError, 'conflict'):
            attach_fused_heads(predictor, lambda *_: self.fail('Must not execute'))
        self.assertIs(predictor._refine_to_heads, method)
        self.assertEqual((predictor.to_gaussians, predictor.to_disparity), modules)
        self.assertFalse(hasattr(predictor, '_mgs_fused_heads_adapter'))
        worker.close()

    def test_conflicting_selection_fails_before_bundle_or_allocation(self):
        for name in (*REFINE_HEADS_COMPONENTS, 'fused_heads'):
            with self.subTest(name=name), \
                    patch('gaussian_generation.npu_branch.runtime.validate_bundle') as bundle, \
                    patch('gaussian_generation.npu_branch.runtime.MappedBuffers') as buffers:
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    PartitionRuntime('npu', 'unused', ['refine_heads_pipeline', name], 'unused')
                bundle.assert_not_called(); buffers.assert_not_called()

    def test_fresh_input_required_before_shared_buffer_planning(self):
        parts = {'refine_heads_pipeline': {'meta': {}}}
        with patch('gaussian_generation.npu_branch.runtime.validate_bundle', return_value=({}, parts)), \
                patch('gaussian_generation.npu_branch.runtime.buffer_plan') as plan:
            with self.assertRaisesRegex(ValueError, 'fresh SDK input'):
                PartitionRuntime('npu', 'unused', list(parts), 'unused', environment={'MGS_NPU_FRESH_INPUT': '0'})
        plan.assert_not_called()

    def test_report_marks_every_composite_component_accelerated_and_cpu_upsampler(self):
        worker = fake_runtime(self.model)
        report = worker.report()
        for name in (*CORR_DEPTH_COMPONENTS, *REFINE_HEADS_COMPONENTS,
                     'corr_depth_pipeline', 'refine_heads_pipeline', 'fused_heads'):
            self.assertNotIn(name, report['cpu_partitions'])
        self.assertIn('upsampler', report['cpu_partitions'])
        del worker.parts['refine_heads_pipeline']
        for name in REFINE_HEADS_COMPONENTS: self.assertIn(name, worker.report()['cpu_partitions'])
        worker.close()


class RefineHeadsAccountingTests(unittest.TestCase):
    def test_double_sdk_execution_and_second_failure_are_both_real(self):
        value = np.zeros((2, 133, 3, 5), np.float32)
        output = np.empty((2, 86, 3, 5), np.float32)
        calls = []
        def forward(context, ip, ib, op, ob, measured):
            calls.append((context, ib, ob))
            for index, seconds in enumerate((.01, .02, .03, .04, .1)): measured[index] = seconds
            return 0
        stages, count = _forward_sdk('refine_heads_pipeline', {}, 9, value, output, forward, lambda: b'bad', 5)
        self.assertEqual(count, 2)
        self.assertEqual(calls, [(9, value.nbytes, output.nbytes)]*2)
        np.testing.assert_allclose(stages, [.02, .04, .06, .08, .2])
        self.assertEqual(sdk_passes('refine_heads_pipeline', {}), 2)
        calls.clear()
        def second_fails(*args):
            calls.append(1)
            return int(len(calls) == 2)
        with self.assertRaisesRegex(RuntimeError, 'second failed'):
            _forward_sdk('refine_heads_pipeline', {}, 9, value, output, second_fails,
                         lambda: b'second failed', 3)
        self.assertEqual(len(calls), 2)

    def test_runtime_identity_records_both_continuous_graph_policies(self):
        with patch.dict(os.environ, {'MGS_NPU_FRESH_INPUT': '1'}, clear=True), \
                patch.object(Path, 'read_text', return_value=''):
            identity = runtime_identity([], [], [], ['corr_depth_pipeline'], ['refine_heads_pipeline'])
        self.assertEqual(identity['corr_depth_settle_passes'], 2)
        self.assertEqual(identity['refine_heads_settle_passes'], 2)
        self.assertEqual(identity['refine_heads_partitions'], ['refine_heads_pipeline'])
        self.assertTrue(identity['fresh_input_tensor'])

    def test_nine_logical_eighteen_sdk_and_no_hidden_components(self):
        parts = full_parts()
        order = paired.expected_calls(parts)
        self.assertEqual(order, ['backbone_cnn', *['transformer_ffn_'+str(i) for i in range(6)],
                                 'corr_depth_pipeline', 'refine_heads_pipeline'])
        calls = [dict(event='COMPLETE', name=name, sequence=10+index, backend='npu', npu_executed=True,
            sdk_execution_passes=2, sdk_stage_names=['input_write', 'execute_wait', 'output_convert'],
            sdk_seconds=[.01, .02, .03], total_seconds=.1) for index, name in enumerate(order)]
        report = paired.check_calls(calls, parts, 10, {'backbone_settle_passes': 2}, 2.)
        self.assertEqual(report['logical_calls'], 9)
        self.assertEqual(report['sdk_forward_calls'], 18)
        for offset in (7, 8):
            invalid = copy.deepcopy(calls)
            invalid[offset]['sdk_execution_passes'] = 1
            with self.assertRaisesRegex(ValueError, 'charged SDK pass'):
                paired.check_calls(invalid, parts, 10, {'backbone_settle_passes': 2}, 2.)
        with self.assertRaisesRegex(ValueError, 'order/count'):
            paired.check_calls(calls[:-1], parts, 10, {'backbone_settle_passes': 2}, 2.)
        self.assertEqual(semantic_coverage(['corr_depth_pipeline', 'refine_heads_pipeline']),
                         set(CORR_DEPTH_COMPONENTS) | set(REFINE_HEADS_COMPONENTS))
        cases = paired.comparison_cases(list(parts), ['backbone_cnn', 'corr_refine_net',
            'depth_head', 'refine_unet', 'to_gaussians', 'to_disparity'], 2)
        self.assertEqual([value[1] for value in cases], ['baseline', 'expanded', 'expanded', 'baseline'])

    def test_paired_sets_only_explicit_required_source_flags(self):
        for corr, tail in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(corr=corr, tail=tail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = [value for name in ('input', 'weights', 'vendor', 'bundle', 'graphs', 'library', 'out')
                        for value in ('--'+name, str(root/name))]
                names = ['backbone_cnn']
                if corr: names.append('corr_depth_pipeline')
                if tail: names.append('refine_heads_pipeline')
                else: names.append('refine_unet')
                with patch.object(paired.platform, 'machine', return_value='aarch64'), \
                        patch('gaussian_generation.runtime.ModelRuntime', side_effect=RuntimeError('before load')) as model:
                    with self.assertRaisesRegex(RuntimeError, 'before load'):
                        paired.main(args + ['--partitions', *names, '--baseline-partitions', 'backbone_cnn'])
                expected = {}
                if corr: expected['corr_depth_pipeline'] = True
                if tail: expected['refine_heads_pipeline'] = True
                self.assertEqual(model.call_args.kwargs, expected)


if __name__ == '__main__':
    unittest.main()
