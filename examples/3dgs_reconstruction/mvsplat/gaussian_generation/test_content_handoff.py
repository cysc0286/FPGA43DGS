"""Array identity and memory-only handoff; no Torch model, weights or device."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from common import gaussian_content_sha256, sha
from gaussian_generation.runtime import ModelRuntime


def gaussians(count=4):
    return dict(means=np.arange(count*3, dtype=np.float32).reshape(count, 3)/16,
        covariances=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        harmonics=np.arange(count*75, dtype=np.float32).reshape(count, 3, 25)/256,
        opacities=np.full(count, .5, np.float32))


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))

    def __getitem__(self, key):
        return FakeTensor(self.array[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeModule:
    def __init__(self):
        self.handles = []

    def register_forward_pre_hook(self, callback):
        return self.register_forward_hook(callback)

    def register_forward_hook(self, callback):
        handle = SimpleNamespace(removed=False)
        handle.remove = lambda:setattr(handle, 'removed', True)
        self.handles.append(handle)
        return handle


class FakeModel:
    def __init__(self, arrays):
        self.arrays = arrays
        self.backbone, self.depth_predictor, self.gaussian_adapter = (FakeModule() for _ in range(3))
        self.calls = 0

    def __call__(self, context, **kwargs):
        self.calls += 1
        return SimpleNamespace(**{key:FakeTensor(value.copy()[None])
                                  for key, value in self.arrays.items()})


def fake_runtime(arrays):
    from omegaconf import OmegaConf
    runtime = ModelRuntime.__new__(ModelRuntime)
    runtime.model = FakeModel(arrays)
    runtime.np = SimpleNamespace(load=np.load, array_equal=np.array_equal,
        savez=Mock(side_effect=AssertionError('Memory-only inference must not serialize NPZ')))
    runtime.torch = SimpleNamespace(from_numpy=FakeTensor, inference_mode=nullcontext)
    runtime._pending_archive = runtime._pending_gaussians = None
    runtime.partition_runtime = None
    runtime.weight_hash = 'a'*64
    runtime.threads, runtime.depth_chunk = 1, 16
    runtime.source_hashes, runtime.metadata = {}, {'test_model':True}
    runtime.config = OmegaConf.create({'test_model':True})
    runtime.prepare_gaussian_archive = Mock(side_effect=AssertionError('No archive payload is allowed'))
    return runtime


class ContentIdentityTests(unittest.TestCase):
    def test_field_order_strides_and_byte_order_are_equivalent(self):
        original = gaussians()
        expected = gaussian_content_sha256(original)
        self.assertEqual(expected, gaussian_content_sha256(dict(reversed(list(original.items())))))
        strided = {}
        for key, value in original.items():
            storage = np.zeros((value.shape[0]*2, *value.shape[1:]), value.dtype)
            storage[::2] = value
            strided[key] = storage[::2]
            self.assertFalse(strided[key].flags.c_contiguous)
        self.assertEqual(expected, gaussian_content_sha256(strided))
        self.assertEqual(expected, gaussian_content_sha256(
            {key:value.astype('>f4') for key, value in original.items()}))
        self.assertEqual(expected, gaussian_content_sha256(
            {key:np.asfortranarray(value) for key, value in original.items()}))

    def test_value_count_and_precision_are_bound_to_identity(self):
        original = gaussians()
        digest = gaussian_content_sha256(original)
        for key in original:
            changed = {name:value.copy() for name, value in original.items()}
            changed[key].flat[0] += .01
            self.assertNotEqual(digest, gaussian_content_sha256(changed), key)
        self.assertNotEqual(digest, gaussian_content_sha256(gaussians(5)))
        self.assertNotEqual(digest, gaussian_content_sha256(
            {key:value.astype(np.float64) for key, value in original.items()}))
        changed = dict(original, means=original['means'].astype(np.float64))
        self.assertNotEqual(digest, gaussian_content_sha256(changed))

    def test_nonfinite_opacity_shape_and_integer_inputs_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            invalid = gaussians()
            invalid['harmonics'][0, 0, 0] = bad
            with self.assertRaises(ValueError):
                gaussian_content_sha256(invalid)
        for bad in (-.01, 1.01):
            invalid = gaussians()
            invalid['opacities'][0] = bad
            with self.assertRaises(ValueError):
                gaussian_content_sha256(invalid)
        invalid = gaussians()
        invalid['means'] = invalid['means'].reshape(2, 6)
        with self.assertRaises(ValueError):
            gaussian_content_sha256(invalid)
        invalid = gaussians()
        invalid['means'] = invalid['means'].astype(np.int32)
        with self.assertRaisesRegex(ValueError, 'floating arrays'):
            gaussian_content_sha256(invalid)


class MemoryHandoffTests(unittest.TestCase):
    def input(self, root):
        root.mkdir()
        np.savez(root/'context.npz', image=np.zeros((2, 3, 2, 2), np.float32))
        return sha(root/'context.npz')

    def test_memory_only_infer_uses_no_npz_or_pending_archive_and_consumes_once(self):
        arrays = gaussians()
        runtime = fake_runtime(arrays)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            digest = self.input(root/'input')
            with patch('gaussian_generation.runtime.load_gaussians',
                       side_effect=AssertionError('No NPZ readback')):
                for n in range(2):
                    output = root/str(n)
                    record = runtime.infer(root/'input', output, digest, memory_only=True)
                    self.assertTrue(record['complete'])
                    self.assertFalse(record['archive_verified'])
                    self.assertEqual(record['gaussian_identity_kind'], 'array_content_v1')
                    self.assertEqual(record['gaussian_archive']['policy'], 'memory_only')
                    self.assertEqual(record['gaussian_archive']['resident_payload_bytes'], 0)
                    self.assertEqual(record['gaussian_sha256'], gaussian_content_sha256(arrays))
                    self.assertIsNone(runtime._pending_archive)
                    self.assertFalse(list(output.glob('*.npz')))
                    actual = runtime.take_gaussians(record['gaussian_sha256'])
                    self.assertIsNone(runtime._pending_gaussians)
                    for key in arrays:
                        np.testing.assert_array_equal(actual[key], arrays[key])
                    with self.assertRaisesRegex(ValueError, 'No validated'):
                        runtime.take_gaussians(record['gaussian_sha256'])
                    persisted = json.loads((output/'inference.json').read_text())
                    self.assertFalse(persisted['archive_verified'])
            self.assertEqual(runtime.model.calls, 2)
            runtime.np.savez.assert_not_called()
            runtime.prepare_gaussian_archive.assert_not_called()
            for module in (runtime.model.backbone, runtime.model.depth_predictor,
                           runtime.model.gaussian_adapter):
                self.assertTrue(all(handle.removed for handle in module.handles))

    def test_memory_only_and_deferred_modes_reject_before_any_access_or_state_mutation(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        pending = object()
        runtime._pending_gaussians = pending
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            runtime.infer('nonexistent_input', 'nonexistent_output', 'invalid',
                          defer_archive=True, memory_only=True)
        self.assertIs(runtime._pending_gaussians, pending)

    def test_pending_archive_still_blocks_memory_only_mode(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        pending = object()
        runtime._pending_archive = pending
        with self.assertRaisesRegex(ValueError, 'archive audit is pending'):
            runtime.infer('nonexistent_input', 'nonexistent_output', 'invalid', memory_only=True)
        self.assertIs(runtime._pending_archive, pending)


if __name__ == '__main__':
    unittest.main()
