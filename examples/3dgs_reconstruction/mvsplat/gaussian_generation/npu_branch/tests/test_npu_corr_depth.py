"""Real small-Torch seam/lifetime checks; no board or full-model performance claim."""
import copy
import hashlib
import types
import unittest

import numpy as np
import torch

from gaussian_generation.npu_branch.corr_depth import (
    MARKER, METHOD, ORIGINAL_BLOCK, OWNER, SCHEMA, attach_corr_depth, patch_source,
)


TOY_SOURCE = '''class DepthPredictorMultiView(torch.nn.Module):
    def __init__(self, wo_cost_volume_refine=False):
        super().__init__()
        self.wo_cost_volume_refine = wo_cost_volume_refine
        self.corr_refine_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 2, 3, padding=1),
            torch.nn.GroupNorm(1, 2), torch.nn.GELU())
        self.regressor_residual = torch.nn.Conv2d(4, 2, 1)
        self.corr_project = torch.nn.Conv2d(4, 2, 3, padding=1)
        self.depth_head_lowres = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(4, 2, 3, padding=1))

    def forward(self, raw_correlation_in):
''' + ORIGINAL_BLOCK + '''        return pdf
'''


def load(source):
    namespace = dict(torch=torch, F=torch.nn.functional)
    exec(compile(source, '<corr-depth-toy>', 'exec'), namespace)
    return namespace['DepthPredictorMultiView']


class SourcePatchTests(unittest.TestCase):
    def test_strict_patch_metadata_and_original_text_untouched(self):
        patched, meta = patch_source(TOY_SOURCE)
        self.assertEqual(meta['schema'], SCHEMA)
        self.assertEqual(meta['matched_blocks'], 1)
        self.assertEqual(meta['original_sha256'], hashlib.sha256(TOY_SOURCE.encode()).hexdigest())
        self.assertEqual(meta['modified_sha256'], hashlib.sha256(patched.encode()).hexdigest())
        self.assertNotIn(METHOD, TOY_SOURCE)
        self.assertTrue(meta['ablation_cpu_path_preserved'])
        self.assertFalse(meta['default_runtime_changed'])
        self.assertFalse(meta['npu_executed'])
        self.assertEqual(patched.count('depth_logits = self._corr_to_depth_logits('), 1)
        self.assertEqual(patched.count('def _corr_to_depth_logits('), 1)

    def test_no_silent_second_patch_or_changed_original_operation(self):
        patched, _ = patch_source(TOY_SOURCE)
        for source in (patched, TOY_SOURCE.replace('raw_correlation + self.regressor_residual',
                                                   'raw_correlation - self.regressor_residual'),
                       TOY_SOURCE.replace('dim=1\n', 'dim=0\n'),
                       TOY_SOURCE + '\n' + TOY_SOURCE):
            with self.subTest(source=source[-60:]), self.assertRaises(ValueError):
                patch_source(source)

    def test_block_must_be_inside_the_exact_target_forward(self):
        outside = 'class DepthPredictorMultiView:\n    def forward(self, x):\n        return x\n'
        outside += 'class Other:\n    def forward(self, raw_correlation_in):\n' + ORIGINAL_BLOCK
        with self.assertRaisesRegex(ValueError, 'outside'):
            patch_source(outside)
        for source in (TOY_SOURCE.replace('DepthPredictorMultiView', 'Other'),
                       TOY_SOURCE.replace('    def forward(', '    @staticmethod\n    def forward('),
                       TOY_SOURCE.replace('\n', '\r\n'), 'not valid Python!'):
            with self.subTest(source=source[:60]), self.assertRaises(ValueError):
                patch_source(source)

    def test_both_cpu_paths_and_parameter_names_unchanged(self):
        patched, _ = patch_source(TOY_SOURCE)
        original_type, patched_type = load(TOY_SOURCE), load(patched)
        torch.manual_seed(123)
        value = torch.randn(2, 4, 7, 9)
        for ablation in (False, True):
            original, changed = original_type(ablation).eval(), patched_type(ablation).eval()
            changed.load_state_dict(original.state_dict(), strict=True)
            self.assertEqual(list(original.state_dict()), list(changed.state_dict()))
            with torch.inference_mode():
                self.assertTrue(torch.equal(original(value), changed(value)))
            for name, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, changed.state_dict()[name]))

    def test_rejects_escaped_temporaries_and_text_only_match(self):
        for source in (
            TOY_SOURCE.replace('        return pdf', '        return pdf, raw_correlation'),
            TOY_SOURCE.replace('        return pdf', '        depth_logits = pdf\n        return depth_logits'),
            TOY_SOURCE.replace(ORIGINAL_BLOCK, '        """\n' + ORIGINAL_BLOCK + '        """\n'),
        ):
            with self.subTest(source=source[-80:]), self.assertRaises(ValueError):
                patch_source(source)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        patched, _ = patch_source(TOY_SOURCE)
        self.model = load(patched)().eval()
        self.reference = copy.deepcopy(self.model).eval()
        self.inputs = [torch.randn(2, 4, 7, 9), torch.randn(2, 4, 7, 9) + 1]
        self.calls = []

        def execute(array):
            self.calls.append(array.copy())
            return self.reference._corr_to_depth_logits(torch.from_numpy(array)).detach().numpy()
        self.execute = execute

    def test_real_ab_return_and_one_executor_without_cpu_module_calls(self):
        counts = dict(corr_refine_net=0, regressor_residual=0, depth_head_lowres=0)
        handles = []
        for name in counts:
            def hook(module, args, result, name=name):
                counts[name] += 1
            handles.append(getattr(self.model, name).register_forward_hook(hook))
        state = {name: value.clone() for name, value in self.model.state_dict().items()}
        try:
            with torch.inference_mode():
                expected = [self.reference(value) for value in self.inputs]
                self.assertFalse(torch.equal(*expected))
                with attach_corr_depth(self.model, self.execute):
                    for index in (0, 0, 1, 1, 0):
                        actual = self.model(self.inputs[index])
                        self.assertTrue(torch.equal(actual, expected[index]))
                self.assertEqual(len(self.calls), 5)
                self.assertEqual(counts, {name: 0 for name in counts})
                self.assertTrue(torch.equal(self.model(self.inputs[0]), expected[0]))
                self.assertEqual(counts, {name: 1 for name in counts})
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(list(state), list(self.model.state_dict()))
        self.assertTrue(all(torch.equal(tensor, self.model.state_dict()[name])
                            for name, tensor in state.items()))

    def test_restores_class_method_and_idempotent_close(self):
        function = self.model._corr_to_depth_logits.__func__
        state_keys = list(self.model.state_dict())
        adapter = attach_corr_depth(self.model, self.execute)
        self.assertIn(METHOD, self.model.__dict__)
        self.assertEqual(state_keys, list(self.model.state_dict()))
        adapter.close()
        adapter.close()
        self.assertNotIn(METHOD, self.model.__dict__)
        self.assertNotIn(OWNER, self.model.__dict__)
        self.assertIs(self.model._corr_to_depth_logits.__func__, function)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            adapter.__enter__()

    def test_restores_exact_existing_instance_method(self):
        previous = types.MethodType(type(self.model)._corr_to_depth_logits, self.model)
        setattr(self.model, METHOD, previous)
        with attach_corr_depth(self.model, self.execute):
            self.assertIsNot(getattr(self.model, METHOD), previous)
        self.assertIs(getattr(self.model, METHOD), previous)

    def test_executor_error_restores_on_context_exit_and_fails_closed(self):
        def fail(value):
            raise RuntimeError('deliberate executor failure')
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                with attach_corr_depth(self.model, fail) as adapter:
                    self.model(self.inputs[0])
            self.assertTrue(adapter.failed)
            self.assertTrue(adapter.closed)
            self.assertNotIn(OWNER, self.model.__dict__)
            self.assertNotIn(METHOD, self.model.__dict__)
            self.assertTrue(torch.equal(self.model(self.inputs[0]), self.reference(self.inputs[0])))

    def test_transactional_restore_if_installation_raises(self):
        model = self.model
        original_setattr = type(model).__setattr__
        def reject(instance, name, value):
            if name == METHOD and name not in instance.__dict__:
                raise RuntimeError('deliberate install failure')
            return original_setattr(instance, name, value)
        type(model).__setattr__ = reject
        try:
            with self.assertRaisesRegex(RuntimeError, 'install failure'):
                attach_corr_depth(model, self.execute)
            self.assertNotIn(OWNER, model.__dict__)
            self.assertNotIn(METHOD, model.__dict__)
        finally:
            type(model).__setattr__ = original_setattr

    def test_rejects_missing_seam_ablation_training_and_duplicate_before_mutation(self):
        original = load(TOY_SOURCE)().eval()
        original_state = list(original.state_dict())
        with self.assertRaisesRegex(ValueError, 'generated'):
            attach_corr_depth(original, self.execute)
        self.assertEqual(original_state, list(original.state_dict()))
        self.assertNotIn(OWNER, original.__dict__)
        self.model.wo_cost_volume_refine = True
        with self.assertRaisesRegex(ValueError, 'ablation'):
            attach_corr_depth(self.model, self.execute)
        self.model.wo_cost_volume_refine = False
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval-only'):
            attach_corr_depth(self.model, self.execute)
        self.assertNotIn(OWNER, self.model.__dict__)
        self.model.eval()
        with attach_corr_depth(self.model, self.execute):
            with self.assertRaisesRegex(ValueError, 'already attached'):
                attach_corr_depth(self.model, self.execute)

    def test_rejects_invalid_input_before_executor(self):
        bad = [self.inputs[0].double(), self.inputs[0][:1], self.inputs[0][:, :3],
               self.inputs[0].requires_grad_(), torch.ones(2, 4, 7),
               torch.full((2, 4, 7, 9), float('nan'))]
        with attach_corr_depth(self.model, self.execute):
            for value in bad:
                with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                    self.model._corr_to_depth_logits(value)
        self.assertEqual(self.calls, [])

    def test_rechecks_training_ablation_and_module_identity(self):
        with attach_corr_depth(self.model, self.execute):
            self.model.train()
            with self.assertRaisesRegex(ValueError, 'eval-only'):
                self.model._corr_to_depth_logits(self.inputs[0])
            self.model.eval()
            self.model.wo_cost_volume_refine = True
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.model._corr_to_depth_logits(self.inputs[0])
            self.model.wo_cost_volume_refine = False
            previous = self.model.regressor_residual
            self.model.regressor_residual = copy.deepcopy(previous)
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                self.model._corr_to_depth_logits(self.inputs[0])
            self.model.regressor_residual = previous
        self.assertEqual(self.calls, [])

    def test_rejects_wrong_output_dtype_shape_layout_and_nonfinite(self):
        for result in (np.zeros((2, 2, 7, 9), np.float64),
                       np.zeros((2, 7, 9, 2), np.float32),
                       np.zeros((1, 2, 7, 9), np.float32),
                       np.full((2, 2, 7, 9), np.inf, np.float32), torch.zeros(2, 2, 7, 9)):
            with self.subTest(kind=type(result), shape=result.shape):
                with attach_corr_depth(self.model, lambda value: result) as adapter:
                    with self.assertRaisesRegex(ValueError, 'Invalid correlation-depth'):
                        self.model._corr_to_depth_logits(self.inputs[0])
                    self.assertTrue(adapter.failed)
                    with self.assertRaisesRegex(RuntimeError, 'failed'):
                        self.model._corr_to_depth_logits(self.inputs[0])

    def test_noncontiguous_input_and_owned_output_with_reused_worker_storage(self):
        buffer = np.zeros((2, 2, 7, 9), np.float32)
        def execute(value):
            self.assertTrue(value.flags.c_contiguous)
            buffer[:] = self.execute(value)
            return buffer
        value = self.inputs[0].transpose(-2, -1).contiguous().transpose(-2, -1)
        self.assertFalse(value.is_contiguous())
        with attach_corr_depth(self.model, execute), torch.inference_mode():
            first = self.model._corr_to_depth_logits(value)
            saved = first.clone()
            self.model._corr_to_depth_logits(self.inputs[1])
            buffer[:] = np.nan
            self.assertTrue(torch.equal(first, saved))
            self.assertTrue(torch.isfinite(first).all())

    def test_reentrant_execution_and_active_detach_rejected(self):
        adapter = None
        def execute(value):
            with self.assertRaisesRegex(RuntimeError, 'active'):
                adapter.close()
            self.model._corr_to_depth_logits(self.inputs[0])
        with attach_corr_depth(self.model, execute) as adapter:
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                self.model._corr_to_depth_logits(self.inputs[0])
        self.assertTrue(adapter.failed)
        self.assertTrue(adapter.closed)

    def test_invalid_parameter_or_head_contract_fails_before_attach(self):
        self.model.depth_head_lowres = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, 'channel contract'):
            attach_corr_depth(self.model, self.execute)
        self.assertNotIn(OWNER, self.model.__dict__)
        patched, _ = patch_source(TOY_SOURCE)
        model = load(patched)().eval().double()
        with self.assertRaisesRegex(ValueError, 'CPU FP32'):
            attach_corr_depth(model, self.execute)
        self.assertNotIn(OWNER, model.__dict__)


if __name__ == '__main__':
    unittest.main()
