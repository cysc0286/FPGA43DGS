"""Small real-Torch source/lifecycle tests; no board or NPU speed claim."""
import copy
import hashlib
import types
import unittest

import numpy as np
import torch
from einops import rearrange, repeat

from gaussian_generation.npu_branch import corr_depth
from gaussian_generation.npu_branch.refine_heads import (
    MARKER, METHOD, ORIGINAL_BLOCK, ORIGINAL_DISPARITY, OWNER, SCHEMA,
    attach_refine_heads, patch_source,
)


TOY_SOURCE = '''class DepthPredictorMultiView(torch.nn.Module):
    def __init__(self, wo_depth_refine=False, wo_cost_volume_refine=False):
        super().__init__()
        self.wo_depth_refine = wo_depth_refine
        self.wo_cost_volume_refine = wo_cost_volume_refine
        self.corr_refine_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 2, 3, padding=1),
            torch.nn.GroupNorm(1, 2), torch.nn.GELU())
        self.regressor_residual = torch.nn.Conv2d(4, 2, 1)
        self.corr_project = torch.nn.Conv2d(4, 2, 3, padding=1)
        self.depth_head_lowres = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(4, 2, 3, padding=1))
        self.upsampler = torch.nn.Conv2d(4, 128, 1)
        self.proj_feature = torch.nn.Conv2d(128, 32, 3, padding=1)
        if wo_depth_refine:
            self.refine_unet = torch.nn.Conv2d(37, 32, 3, padding=1)
        else:
            self.refine_unet = torch.nn.Sequential(
                torch.nn.Conv2d(37, 32, 3, padding=1),
                torch.nn.GroupNorm(4, 32), torch.nn.GELU(),
                torch.nn.Conv2d(32, 32, 3, padding=1))
            self.to_disparity = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.GELU(),
                torch.nn.Conv2d(64, 2, 3, padding=1))
        self.to_gaussians = torch.nn.Sequential(
            torch.nn.Conv2d(163, 168, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(168, 84, 3, padding=1))

    def forward(self, value, images, near, far):
        raw_correlation_in = value
        feat01, cnn_features = value[:, :2], value[:, 2:]
        extra_info = {"images": images}
        v, b, gaussians_per_pixel = 2, 1, 1
        disp_candi_curr = value.new_tensor([0.2, 0.8]).reshape(1, 2, 1, 1)
''' + corr_depth.ORIGINAL_BLOCK + '''        coarse_disps = (disp_candi_curr * pdf).sum(dim=1, keepdim=True)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]
        fullres_disps = coarse_disps
        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
''' + ORIGINAL_BLOCK + '''        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1
            )
            depths = 1.0 / fullres_disps
            depths = repeat(
                depths, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1
            )
        else:
            # delta fine depth and density
''' + ORIGINAL_DISPARITY + '''            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            )
            densities = repeat(
                F.sigmoid(raw_densities),
                "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1
            )
            fine_disps = (fullres_disps + delta_disps).clamp(
                1.0 / rearrange(far, "b v -> (v b) () () ()"),
                1.0 / rearrange(near, "b v -> (v b) () () ()")
            )
            depths = 1.0 / fine_disps
            depths = repeat(
                depths, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1
            )
        return depths, densities, raw_gaussians
'''


def load(source):
    namespace = dict(torch=torch, F=torch.nn.functional, rearrange=rearrange, repeat=repeat)
    exec(compile(source, '<refine-heads-toy>', 'exec'), namespace)
    return namespace['DepthPredictorMultiView']


def forward_inputs():
    return (torch.randn(2, 4, 5, 7), torch.randn(2, 3, 5, 7),
            torch.full((1, 2), 0.1), torch.full((1, 2), 10.0))


class SourcePatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_identity_and_no_implicit_runtime_or_parameter_change(self):
        patched, metadata = patch_source(TOY_SOURCE)
        self.assertEqual(metadata['schema'], SCHEMA)
        self.assertEqual(metadata['matched_blocks'], 2)
        self.assertEqual(metadata['original_sha256'], hashlib.sha256(TOY_SOURCE.encode()).hexdigest())
        self.assertEqual(metadata['modified_sha256'], hashlib.sha256(patched.encode()).hexdigest())
        self.assertEqual(metadata['packed_input_channels'], 133)
        self.assertEqual(metadata['packed_output_channels'], 86)
        self.assertTrue(metadata['cpu_upsampler_preserved'])
        self.assertTrue(metadata['ablation_cpu_path_preserved'])
        for key in ('source_written', 'parameter_names_changed', 'downstream_operations_changed',
                    'default_runtime_changed', 'npu_executed'):
            self.assertFalse(metadata[key])
        self.assertNotIn(METHOD, TOY_SOURCE)
        self.assertEqual(patched.count('def _refine_to_heads('), 1)

    def test_full_cpu_paths_exact_including_ablation_and_all_state(self):
        patched, _ = patch_source(TOY_SOURCE)
        arguments = forward_inputs()
        for ablation in (False, True):
            with self.subTest(ablation=ablation), torch.inference_mode():
                original = load(TOY_SOURCE)(ablation).eval()
                changed = load(patched)(ablation).eval()
                changed.load_state_dict(original.state_dict(), strict=True)
                self.assertEqual(original.state_dict().keys(), changed.state_dict().keys())
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                    original(*arguments), changed(*arguments))))
                self.assertTrue(all(torch.equal(value, changed.state_dict()[name])
                    for name, value in original.state_dict().items()))
                if ablation:
                    self.assertFalse(hasattr(changed, 'to_disparity'))

    def test_corr_depth_patch_commutes_and_both_methods_execute_exactly(self):
        first, _ = patch_source(TOY_SOURCE)
        first, _ = corr_depth.patch_source(first)
        second, _ = corr_depth.patch_source(TOY_SOURCE)
        second, _ = patch_source(second)
        original = load(TOY_SOURCE)().eval()
        inputs = forward_inputs()
        for source in (first, second):
            changed = load(source)().eval()
            changed.load_state_dict(original.state_dict(), strict=True)
            self.assertTrue(hasattr(changed, METHOD))
            self.assertTrue(hasattr(changed, corr_depth.METHOD))
            with torch.inference_mode():
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                    original(*inputs), changed(*inputs))))

    def test_changed_operations_duplicate_and_invalid_source_fail_closed(self):
        patched, _ = patch_source(TOY_SOURCE)
        variants = [patched, TOY_SOURCE + '\n' + TOY_SOURCE,
            TOY_SOURCE.replace('proj_feature, fullres_disps, pdf_max', 'proj_feature, pdf_max, fullres_disps'),
            TOY_SOURCE.replace(ORIGINAL_DISPARITY, ORIGINAL_DISPARITY.replace('refine_out)', 'refine_out + 1)')),
            TOY_SOURCE.replace('\n', '\r\n'),
            TOY_SOURCE.replace('    def forward(', '    @staticmethod\n    def forward('),
            TOY_SOURCE.replace('DepthPredictorMultiView', 'Other'), 'invalid Python!']
        for source in variants:
            with self.subTest(source=source[:60]), self.assertRaises(ValueError):
                patch_source(source)
        with self.assertRaises(TypeError):
            patch_source(None)

    def test_escaped_temporaries_nested_or_text_only_blocks_fail_closed(self):
        variants = [
            TOY_SOURCE.replace('        return depths, densities, raw_gaussians',
                               '        return refine_out, densities, raw_gaussians'),
            TOY_SOURCE.replace('        raw_gaussians = rearrange(',
                               '        extra = delta_disps_density\n        raw_gaussians = rearrange('),
            TOY_SOURCE.replace(ORIGINAL_BLOCK, '        """\n' + ORIGINAL_BLOCK + '        """\n'),
            TOY_SOURCE.replace(ORIGINAL_BLOCK, '        if True:\n' + ORIGINAL_BLOCK),
            TOY_SOURCE.replace('        else:\n            # delta fine depth and density',
                               '        elif True:\n            # delta fine depth and density'),
        ]
        for source in variants:
            with self.subTest(source=source[-80:]), self.assertRaises(ValueError):
                patch_source(source)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        patched, _ = patch_source(TOY_SOURCE)
        self.model = load(patched)().eval()
        self.reference = copy.deepcopy(self.model).eval()
        self.inputs = [tuple(torch.randn(2, channels, 5, 7) + shift for channels in (128, 3, 1, 1))
                       for shift in (0, 1)]
        self.calls = []

        def execute(packed):
            self.assertEqual(packed.dtype, np.float32)
            self.assertEqual(packed.shape, (2, 133, 5, 7))
            self.assertTrue(packed.flags.c_contiguous)
            self.calls.append(packed.copy())
            raw = self.reference._refine_to_heads(
                *(torch.from_numpy(packed[:, begin:end])
                  for begin, end in ((0, 128), (128, 131), (131, 132), (132, 133))))
            return np.concatenate([value.detach().numpy() for value in raw], axis=1)
        self.execute = execute

    def test_ab_switch_packing_and_single_executor_no_original_module_calls(self):
        counts = {name: 0 for name in ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity')}
        handles = []
        for name in counts:
            def hook(module, args, output, name=name):
                counts[name] += 1
            handles.append(getattr(self.model, name).register_forward_hook(hook))
        state = {key: value.clone() for key, value in self.model.state_dict().items()}
        try:
            with torch.inference_mode():
                expected = [self.reference._refine_to_heads(*values) for values in self.inputs]
                self.assertFalse(torch.equal(expected[0][0], expected[1][0]))
                with attach_refine_heads(self.model, self.execute):
                    for index in (0, 0, 1, 1, 0, 0):
                        actual = self.model._refine_to_heads(*self.inputs[index])
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected[index])))
                        self.assertTrue(np.array_equal(self.calls[-1],
                            np.concatenate([x.numpy() for x in self.inputs[index]], axis=1)))
                self.assertEqual(len(self.calls), 6)
                self.assertEqual(counts, dict.fromkeys(counts, 0))
                self.model._refine_to_heads(*self.inputs[0])
                self.assertEqual(counts, dict.fromkeys(counts, 1))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(state.keys(), self.model.state_dict().keys())
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in state.items()))

    def test_entire_forward_keeps_cpu_upsampler_and_downstream_outputs_exact(self):
        calls = []
        handle = self.model.upsampler.register_forward_hook(lambda *args: calls.append(1))
        inputs = forward_inputs()
        try:
            with torch.inference_mode():
                expected = self.reference(*inputs)
                with attach_refine_heads(self.model, self.execute):
                    actual = self.model(*inputs)
                self.assertEqual(calls, [1])
                self.assertEqual(len(self.calls), 1)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        finally:
            handle.remove()

    def test_corr_and_refine_adapters_coexist_without_cross_ownership(self):
        source, _ = corr_depth.patch_source(TOY_SOURCE)
        source, _ = patch_source(source)
        model = load(source)().eval()
        model.load_state_dict(self.model.state_dict(), strict=True)
        reference = copy.deepcopy(model)
        corr_calls = []
        def execute_corr(value):
            corr_calls.append(1)
            return reference._corr_to_depth_logits(torch.from_numpy(value)).numpy()
        inputs = forward_inputs()
        with torch.inference_mode():
            expected = reference(*inputs)
            with corr_depth.attach_corr_depth(model, execute_corr):
                corr_method = model._corr_to_depth_logits
                with attach_refine_heads(model, self.execute):
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(model(*inputs), expected)))
                self.assertIs(model._corr_to_depth_logits, corr_method)
            self.assertEqual(corr_calls, [1])
            self.assertNotIn(OWNER, model.__dict__)
            self.assertNotIn(corr_depth.OWNER, model.__dict__)

    def test_restore_class_instance_methods_and_idempotent_close(self):
        function = self.model._refine_to_heads.__func__
        adapter = attach_refine_heads(self.model, self.execute)
        adapter.close()
        adapter.close()
        self.assertNotIn(METHOD, self.model.__dict__)
        self.assertNotIn(OWNER, self.model.__dict__)
        self.assertIs(self.model._refine_to_heads.__func__, function)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            adapter.__enter__()
        previous = types.MethodType(function, self.model)
        setattr(self.model, METHOD, previous)
        with attach_refine_heads(self.model, self.execute):
            self.assertIsNot(self.model._refine_to_heads, previous)
        self.assertIs(self.model._refine_to_heads, previous)

    def test_invalid_seam_ablation_training_and_duplicate_before_mutation(self):
        variants = [(load(TOY_SOURCE)().eval(), 'generated'),
                    (load(patch_source(TOY_SOURCE)[0])(True).eval(), 'ablation'),
                    (copy.deepcopy(self.model).train(), 'eval-only')]
        for model, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                attach_refine_heads(model, self.execute)
            self.assertNotIn(OWNER, model.__dict__)
        with attach_refine_heads(self.model, self.execute):
            with self.assertRaisesRegex(ValueError, 'already attached'):
                attach_refine_heads(self.model, self.execute)

    def test_fused_and_independent_components_rejected_before_mutation(self):
        from gaussian_generation.npu_branch.fused_heads import attach_fused_heads
        with attach_fused_heads(self.model, lambda x: None):
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                attach_refine_heads(self.model, self.execute)
            self.assertNotIn(OWNER, self.model.__dict__)
        for name in ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity'):
            model = copy.deepcopy(self.model)
            setattr(model, name, torch.nn.Identity())
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'component is replaced'):
                attach_refine_heads(model, self.execute)
            self.assertNotIn(OWNER, model.__dict__)
        model = copy.deepcopy(self.model).double()
        with self.assertRaisesRegex(ValueError, 'CPU FP32'):
            attach_refine_heads(model, self.execute)

    def test_invalid_inputs_are_rejected_without_executor_or_state_poisoning(self):
        invalid = [(self.inputs[0][0].double(), *self.inputs[0][1:]),
                   (self.inputs[0][0][:1], *self.inputs[0][1:]),
                   (self.inputs[0][0][:, :127], *self.inputs[0][1:]),
                   (self.inputs[0][0].clone().requires_grad_(), *self.inputs[0][1:]),
                   (self.inputs[0][0], self.inputs[0][1][:, :, :4], *self.inputs[0][2:]),
                   (self.inputs[0][0], self.inputs[0][1], torch.ones(2, 1, 5), self.inputs[0][3]),
                   (torch.full_like(self.inputs[0][0], float('nan')), *self.inputs[0][1:])]
        with attach_refine_heads(self.model, self.execute) as adapter:
            for values in invalid:
                with self.subTest(shape=values[0].shape), self.assertRaises(ValueError):
                    self.model._refine_to_heads(*values)
            self.assertFalse(adapter.failed)
        self.assertEqual(self.calls, [])

    def test_contract_mutations_checked_before_execution(self):
        with attach_refine_heads(self.model, self.execute):
            self.model.train()
            with self.assertRaisesRegex(ValueError, 'eval-only'):
                self.model._refine_to_heads(*self.inputs[0])
            self.model.eval()
            self.model.wo_depth_refine = True
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.model._refine_to_heads(*self.inputs[0])
            self.model.wo_depth_refine = False
            original = self.model.proj_feature
            self.model.proj_feature = copy.deepcopy(original)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.model._refine_to_heads(*self.inputs[0])
            self.model.proj_feature = original
        self.assertEqual(self.calls, [])

    def test_wrong_outputs_nonfinite_and_execution_errors_fail_closed(self):
        bad = [np.zeros((2, 86, 5, 7), np.float64), np.zeros((2, 5, 7, 86), np.float32),
               np.zeros((1, 86, 5, 7), np.float32), np.zeros((2, 85, 5, 7), np.float32),
               np.full((2, 86, 5, 7), np.inf, np.float32), torch.zeros(2, 86, 5, 7)]
        for result in bad:
            with self.subTest(shape=result.shape), torch.inference_mode():
                with attach_refine_heads(self.model, lambda x: result) as adapter:
                    with self.assertRaisesRegex(ValueError, 'Invalid refinement-head'):
                        self.model._refine_to_heads(*self.inputs[0])
                    self.assertTrue(adapter.failed)
                    with self.assertRaisesRegex(RuntimeError, 'failed'):
                        self.model._refine_to_heads(*self.inputs[0])
        def fail(value):
            raise RuntimeError('deliberate executor failure')
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with attach_refine_heads(self.model, fail) as adapter:
                self.model._refine_to_heads(*self.inputs[0])
        self.assertTrue(adapter.failed)
        self.assertTrue(adapter.closed)
        self.assertNotIn(OWNER, self.model.__dict__)

    def test_noncontiguous_inputs_and_both_outputs_own_storage(self):
        buffer = np.zeros((2, 86, 5, 7), np.float32)
        def execute(value):
            buffer[:] = self.execute(value)
            return buffer
        values = tuple(value.transpose(-2, -1).contiguous().transpose(-2, -1)
                       for value in self.inputs[0])
        self.assertFalse(values[0].is_contiguous())
        with attach_refine_heads(self.model, execute), torch.inference_mode():
            first = self.model._refine_to_heads(*values)
            saved = tuple(value.clone() for value in first)
            self.model._refine_to_heads(*self.inputs[1])
            buffer[:] = np.nan
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, saved)))
            self.assertFalse(np.shares_memory(first[0].numpy(), first[1].numpy()))
            self.assertTrue(all(value.is_contiguous() for value in first))
            self.assertTrue(all(not np.shares_memory(value.numpy(), buffer) for value in first))

    def test_install_failure_rolls_back_and_active_detach_or_reentry_rejected(self):
        original_setattr = type(self.model).__setattr__
        def reject(instance, name, value):
            if name == METHOD and name not in instance.__dict__:
                raise RuntimeError('deliberate install failure')
            return original_setattr(instance, name, value)
        type(self.model).__setattr__ = reject
        try:
            with self.assertRaisesRegex(RuntimeError, 'install failure'):
                attach_refine_heads(self.model, self.execute)
            self.assertNotIn(OWNER, self.model.__dict__)
            self.assertNotIn(METHOD, self.model.__dict__)
        finally:
            type(self.model).__setattr__ = original_setattr
        adapter = None
        def reenter(value):
            with self.assertRaisesRegex(RuntimeError, 'active'):
                adapter.close()
            self.model._refine_to_heads(*self.inputs[0])
        with attach_refine_heads(self.model, reenter) as adapter:
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                self.model._refine_to_heads(*self.inputs[0])
        self.assertTrue(adapter.failed)
        self.assertTrue(adapter.closed)


if __name__ == '__main__':
    unittest.main()
