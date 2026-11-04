"""Small real Torch forwards exercise hooks; no board or full MVSplat runs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from gaussian_generation.npu_branch.refine_heads_capture import capture_refine_heads, SCHEMA


CHANNELS = (4, 3, 2, 5, 2)


class TinyPredictor(nn.Module):
    def __init__(self, mode='normal'):
        super().__init__()
        self.mode = mode
        self.wo_depth_refine = False
        self.proj_feature = nn.Conv2d(4, 2, 1)
        self.refine_unet = nn.Conv2d(7, 2, 1)
        self.to_gaussians = nn.Conv2d(9, 5, 1)
        self.to_disparity = nn.Conv2d(2, 2, 1)

    def forward(self, up, images, disp, pdf):
        if self.mode == 'exception':
            raise RuntimeError('original forward failed')
        projected = self.proj_feature(up)
        if self.mode == 'duplicate':
            projected = self.proj_feature(up)
        if self.mode == 'projected_changed':
            projected = projected + 1
        refined = self.refine_unet(torch.cat((images, projected, disp, pdf), dim=1))
        if self.mode == 'skip':
            return refined
        head_up = up + 1 if self.mode == 'up_changed' else up
        head_images = images + 1 if self.mode == 'images_changed' else images
        gaussian = self.to_gaussians(torch.cat((refined, head_images, head_up), dim=1))
        if self.mode == 'refined_changed':
            refined.add_(1)
        disparity = self.to_disparity(refined)
        if self.mode == 'post_exception':
            raise RuntimeError('failed after heads')
        return gaussian, disparity


class RefineHeadsCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1403)
        torch.set_num_threads(1)
        self.model = TinyPredictor().eval()
        self.values = tuple(torch.randn(2, c, 3, 5) for c in (4, 3, 1, 1))
        self.identities = dict(context_sha256='1' * 64, weight_sha256='2' * 64,
                               source_hashes={'original.py': '3' * 64}, channels=CHANNELS)

    def capture(self, model=None, **changes):
        return capture_refine_heads(model or self.model, **(self.identities | changes))

    def no_hooks(self, model):
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_one_real_forward_exact_arrays_identity_and_bounded_copies(self):
        before = {k:v.detach().clone() for k, v in self.model.state_dict().items()}
        with self.capture() as capture, torch.no_grad():
            gauss, disp = self.model(*self.values)
            with self.assertRaisesRegex(RuntimeError, 'finish successfully'):
                _ = capture.record
        self.no_hooks(self.model)
        arrays = capture.arrays
        np.testing.assert_array_equal(arrays['input'], torch.cat(self.values, 1).numpy())
        np.testing.assert_array_equal(arrays['expected'], torch.cat((gauss, disp), 1).numpy())
        self.assertFalse(arrays['input'].flags.writeable)
        self.assertFalse(arrays['expected'].flags.writeable)
        record = capture.record
        self.assertEqual(record['schema'], SCHEMA)
        self.assertEqual(record['context_sha256'], '1' * 64)
        self.assertEqual(record['input_sha256'], hashlib.sha256(arrays['input'].tobytes()).hexdigest())
        self.assertEqual(record['predictor_calls'], 1)
        self.assertFalse(record['npu_executed'])
        self.assertFalse(record['production_qualified'])
        # input9 + output7 + refinement2 channels; never retain all original inputs.
        self.assertEqual(record['peak_logical_owned_bytes'], 2 * (9 + 7 + 2) * 3 * 5 * 4)
        self.assertEqual(record['retained_logical_bytes'], 2 * (9 + 7) * 3 * 5 * 4)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        for value in self.values:
            value.zero_()
        self.assertTrue(np.any(arrays['input'] != 0))
        record['source_hashes'].clear()
        self.assertEqual(capture.record['source_hashes'], {'original.py': '3' * 64})

    def test_fixture_save_hash_and_refuses_overwrite(self):
        with self.capture() as capture, torch.no_grad():
            self.model(*self.values)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'capture'
            record = capture.save(target)
            self.assertEqual(record['fixture_sha256'], hashlib.sha256((target/'oracle.npz').read_bytes()).hexdigest())
            self.assertEqual(json.loads((target/'capture.json').read_text('utf-8')), record)
            with np.load(target/'oracle.npz', allow_pickle=False) as fixture:
                np.testing.assert_array_equal(fixture['input'], capture.arrays['input'])
            with self.assertRaises(FileExistsError):
                capture.save(target)

    def test_changed_shared_values_fail_and_hooks_cleaned(self):
        for mode in ('up_changed', 'images_changed', 'projected_changed', 'refined_changed'):
            with self.subTest(mode=mode):
                model = TinyPredictor(mode).eval()
                capture = self.capture(model)
                with self.assertRaisesRegex(ValueError, 'tensor mismatch'):
                    with capture, torch.no_grad():
                        model(*self.values)
                self.no_hooks(model)
                self.assertIsNone(capture._input)
                with self.assertRaises(RuntimeError):
                    _ = capture.arrays

    def test_duplicate_skip_outside_root_and_two_roots_rejected(self):
        for mode in ('duplicate', 'skip'):
            model = TinyPredictor(mode).eval()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                with self.capture(model), torch.no_grad():
                    model(*self.values)
            self.no_hooks(model)
        with self.assertRaisesRegex(ValueError, 'invocation boundary'):
            with self.capture(), torch.no_grad():
                self.model.proj_feature(self.values[0])
        self.no_hooks(self.model)
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            with self.capture(), torch.no_grad():
                self.model(*self.values)
                self.model(*self.values)
        self.no_hooks(self.model)

    def test_original_exception_before_or_after_stages_propagates(self):
        for mode in ('exception', 'post_exception'):
            model = TinyPredictor(mode).eval()
            capture = self.capture(model)
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'failed'):
                with capture, torch.no_grad():
                    model(*self.values)
            self.no_hooks(model)
            self.assertIsNone(capture._expected)

    def test_no_forward_reentry_training_and_grad_enabled_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete same-forward'):
            with self.capture():
                pass
        capture = self.capture()
        with self.assertRaisesRegex(ValueError, 'no-grad'):
            with capture:
                self.model(*self.values)
        self.no_hooks(self.model)
        with self.assertRaises(RuntimeError):
            capture.__enter__()
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            self.capture()

    def test_finite_view_dtype_and_spatial_contracts(self):
        variants = [self.values[0][:1], self.values[0].double(),
                    self.values[0].clone().fill_(float('nan')),
                    self.values[0].clone().requires_grad_()]
        for value in variants:
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                with self.capture(), torch.no_grad():
                    self.model(value, *self.values[1:])
            self.no_hooks(self.model)

    def test_parameter_module_identity_and_sha_fail_closed(self):
        for changes in (dict(context_sha256='wrong'), dict(weight_sha256='A' * 64),
                        dict(source_hashes={}), dict(source_hashes={'x': '0'})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.capture(**changes)
        capture = self.capture()
        self.model.proj_feature = nn.Conv2d(4, 2, 1).eval()
        with self.assertRaisesRegex(ValueError, 'replaced'):
            with capture, torch.no_grad():
                self.model(*self.values)
        self.no_hooks(self.model)

    def test_owned_memory_guard_and_adapter_rejection(self):
        capture = self.capture(max_owned_bytes=100)
        with self.assertRaisesRegex(MemoryError, 'byte budget'):
            with capture, torch.no_grad():
                self.model(*self.values)
        self.no_hooks(self.model)
        self.assertIsNone(capture._input)
        self.model._mgs_refine_heads_adapter = object()
        with self.assertRaisesRegex(ValueError, 'adapter'):
            self.capture()

    def test_default_real_contract_rejects_tiny_or_proxy_structure(self):
        with self.assertRaisesRegex(ValueError, 'original full'):
            self.capture(channels=(128, 3, 32, 84, 2))

    def test_corr_adapter_owner_rejected_at_creation_and_every_step(self):
        self.model._mgs_corr_depth_adapter = object()
        with self.assertRaisesRegex(ValueError, 'adapter'):
            self.capture()
        del self.model._mgs_corr_depth_adapter
        capture = self.capture()
        with self.assertRaisesRegex(ValueError, 'adapter'):
            with capture, torch.no_grad():
                self.model._mgs_corr_depth_adapter = object()
                self.model(*self.values)
        self.no_hooks(self.model)
        del self.model._mgs_corr_depth_adapter
        # Change the model between component pre/post hooks, after root entry.
        external = self.model.proj_feature.register_forward_hook(
            lambda module, args, result: setattr(self.model, '_mgs_corr_depth_adapter', object()))
        capture = self.capture()
        try:
            with self.assertRaisesRegex(ValueError, 'adapter'):
                with capture, torch.no_grad():
                    self.model(*self.values)
            self.assertEqual(capture.root_calls, 1)
            self.assertGreater(capture.phase, 0)
        finally:
            external.remove()
        self.no_hooks(self.model)

    def test_upstream_partition_proxy_detected_without_importing_runtime(self):
        partition = type('Partition', (nn.Module,),
                         {'__module__': 'gaussian_generation.npu_branch.runtime'})
        self.model.upstream = partition().eval()
        with self.assertRaisesRegex(ValueError, 'Partition or proxy'):
            self.capture()
        del self.model.upstream
        capture = self.capture()
        with self.assertRaisesRegex(ValueError, 'Partition or proxy'):
            with capture, torch.no_grad():
                self.model.upstream = partition().eval()
                self.model(*self.values)
        self.no_hooks(self.model)
        del self.model.upstream
        for source in ('gaussian_generation.npu_branch.ffn',
                       'gaussian_generation.npu_branch.fused_heads'):
            proxy = type('Proxy', (nn.Module,), {'__module__': source})
            self.model.upstream = proxy().eval()
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'proxy'):
                self.capture()
            del self.model.upstream

    def test_all_predictor_parameters_checked_beyond_four_targets(self):
        self.model.upstream = nn.Linear(1, 1).double().eval()
        with self.assertRaisesRegex(ValueError, 'All predictor parameters'):
            self.capture()
        del self.model.upstream
        capture = self.capture()
        with self.assertRaisesRegex(ValueError, 'All predictor parameters'):
            with capture, torch.no_grad():
                self.model.upstream = nn.Linear(1, 1).double().eval()
                self.model(*self.values)
        self.no_hooks(self.model)


if __name__ == '__main__':
    unittest.main()
