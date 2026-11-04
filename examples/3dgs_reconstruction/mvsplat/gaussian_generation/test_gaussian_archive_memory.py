"""The deferred Gaussian archive must not require a second scene-sized buffer."""
import hashlib
import io
from pathlib import Path
import shutil
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from gaussian_generation.runtime import ModelRuntime, persistent_archive_directory
from gaussian_generation import runtime as generation_runtime


def scene(count=16):
    return dict(means=np.zeros((count, 3), np.float32),
                covariances=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
                harmonics=np.zeros((count, 3, 25), np.float32),
                opacities=np.full(count, 0.5, np.float32))


def runtime():
    value = ModelRuntime.__new__(ModelRuntime)
    value.np = np
    value._pending_archive = None
    return value


class DeferredArchiveMemoryTests(unittest.TestCase):
    def test_disk_policy_preserves_exact_npz_and_has_no_resident_payload(self):
        gaussian = scene()
        stream = io.BytesIO()
        np.savez(stream, **gaussian)
        expected = stream.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = runtime()
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'disk'}):
                info = model.prepare_gaussian_archive(root, gaussian)
            self.assertEqual(info['sha256'], hashlib.sha256(expected).hexdigest())
            self.assertEqual(info['resident_payload_bytes'], 0)
            self.assertEqual(info['archive_bytes'], len(expected))
            pending = model._pending_archive[1]
            self.assertIsInstance(pending, Path)
            self.assertEqual(pending.read_bytes(), expected)
            self.assertFalse((root/'gaussians.npz').exists())
            record = dict(gaussian_sha256=info['sha256'], archive_verified=False)
            model.persist_gaussians(record, gaussian)
            self.assertTrue(record['archive_verified'])
            self.assertIsNone(model._pending_archive)
            self.assertFalse(pending.exists())
            self.assertEqual((root/'gaussians.npz').read_bytes(), expected)

    def test_disk_roundtrip_failure_keeps_pending_for_validated_retry(self):
        gaussian = scene()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = runtime()
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'disk'}):
                info = model.prepare_gaussian_archive(root, gaussian)
            record = dict(gaussian_sha256=info['sha256'], archive_verified=False)
            pending = model._pending_archive[1]
            gaussian['means'][0, 0] = 1
            with self.assertRaisesRegex(ValueError, 'differs from rendered'):
                model.persist_gaussians(record, gaussian)
            self.assertFalse(record['archive_verified'])
            self.assertTrue(pending.exists())
            self.assertFalse((root/'gaussians.npz').exists())
            gaussian['means'][0, 0] = 0
            model.persist_gaussians(record, gaussian)
            self.assertTrue(record['archive_verified'])

    def test_disk_serialization_failure_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = runtime()
            def fail(stream, **values):
                stream.write(b'partial')
                raise OSError('disk full')
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'disk'}), \
                    patch.object(np, 'savez', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    model.prepare_gaussian_archive(root, scene())
            self.assertEqual(list(root.iterdir()), [])
            self.assertIsNone(model._pending_archive)

    def test_memory_fallback_keeps_existing_archive_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            model = runtime()
            root = Path(directory)
            gaussian = scene()
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'memory'}):
                info = model.prepare_gaussian_archive(root, gaussian)
            self.assertIsInstance(model._pending_archive[1], bytes)
            self.assertEqual(info['resident_payload_bytes'], info['archive_bytes'])
            record = dict(gaussian_sha256=info['sha256'], archive_verified=False)
            model.persist_gaussians(record, gaussian)
            self.assertTrue(record['archive_verified'])

    def test_invalid_policy_and_pending_archive_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = runtime()
            root = Path(directory)
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'unknown'}):
                with self.assertRaisesRegex(ValueError, 'archive policy'):
                    model.prepare_gaussian_archive(root, scene())
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'memory'}):
                model.prepare_gaussian_archive(root, scene())
                with self.assertRaisesRegex(ValueError, 'pending'):
                    model.prepare_gaussian_archive(root, scene())

    def test_existing_pending_file_is_not_deleted_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = root/'.gaussians.pending.npz'
            pending.write_bytes(b'previous artifact')
            with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':'disk'}):
                with self.assertRaises(FileExistsError):
                    runtime().prepare_gaussian_archive(root, scene())
            self.assertEqual(pending.read_bytes(), b'previous artifact')

    def test_linux_disk_mode_rejects_ram_mount_even_under_persistent_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root.parent.as_posix().replace(' ', r'\040')
            mount = root.as_posix().replace(' ', r'\040')
            info = '1 0 8:1 / '+parent+' rw - ext4 /dev/root rw\n'
            info += '2 1 0:2 / '+mount+' rw - tmpfs tmpfs rw\n'
            with patch('sys.platform', 'linux'), patch.object(Path, 'read_text', return_value=info):
                with self.assertRaisesRegex(ValueError, 'RAM filesystem'):
                    persistent_archive_directory(root)
            with patch('sys.platform', 'linux'), patch.object(Path, 'read_text', return_value=info.splitlines()[0]):
                persistent_archive_directory(root)

    def test_native_scene_disk_mode_removes_archive_sized_live_python_allocation(self):
        # Allocate the real 32768-point shapes before starting the tracker;
        # compare only extra *live* allocations after serialization returns.
        # BytesIO.getvalue() may share its bytes, so never assume two payloads.
        gaussian = scene(32768)
        live = {}
        for policy in ('memory', 'disk'):
            with tempfile.TemporaryDirectory() as directory:
                model = runtime()
                tracemalloc.start()
                try:
                    with patch.dict('os.environ', {'MGS_GAUSSIAN_ARCHIVE':policy}):
                        info = model.prepare_gaussian_archive(Path(directory), gaussian)
                    live[policy] = tracemalloc.get_traced_memory()[0]
                finally:
                    tracemalloc.stop()
                model._pending_archive = None
        self.assertGreater(live['memory']-live['disk'], info['archive_bytes']*.9)


class ExecutionSourceIdentityTests(unittest.TestCase):
    def test_identity_binds_executed_vendor_copy_without_changing_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/'vendor'
            source = original/'src/model/depth.py'
            source.parent.mkdir(parents=True)
            source.write_text('def forward(): return 1\n')
            active = root/'generated_source'
            shutil.copytree(original, active)
            base = generation_runtime.execution_source_hashes(original)
            same = generation_runtime.execution_source_hashes(active)
            self.assertEqual(base, same)  # Relocation is not a code change.
            (active/'src/model/depth.py').write_text('def forward(): return 2\n')
            changed = generation_runtime.execution_source_hashes(active)
            self.assertNotEqual(base, changed)
            self.assertNotEqual(base['src/model/depth.py'], changed['src/model/depth.py'])
            self.assertEqual(source.read_text(), 'def forward(): return 1\n')
            self.assertEqual(generation_runtime.execution_source_hashes(original), base)

    def test_identity_binds_both_runtime_and_memory_schedule_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root/'vendor'
            (vendor/'src').mkdir(parents=True)
            (vendor/'src/encoder.py').write_text('original encoder\n')
            driver = root/'mvsplat/gaussian_generation/runtime.py'
            driver.parent.mkdir(parents=True)
            helper = root/'mvsplat/gaussian_generation/memory.py'
            shared = root/'mvsplat/common.py'
            driver.write_text('runtime version one\n')
            helper.write_text('schedule version one\n')
            shared.write_text('content identity version one\n')
            with patch.object(generation_runtime, '__file__', str(driver)):
                initial = generation_runtime.execution_source_hashes(vendor)
                helper.write_text('schedule version two\n')
                after_helper = generation_runtime.execution_source_hashes(vendor)
                self.assertNotEqual(initial['runtime/gaussian_generation/memory.py'], after_helper['runtime/gaussian_generation/memory.py'])
                self.assertEqual(initial['src/encoder.py'], after_helper['src/encoder.py'])
                driver.write_text('runtime version two\n')
                after_driver = generation_runtime.execution_source_hashes(vendor)
                self.assertNotEqual(after_helper['runtime/gaussian_generation/runtime.py'],
                                    after_driver['runtime/gaussian_generation/runtime.py'])
                self.assertEqual(set(initial), {'src/encoder.py', 'runtime/gaussian_generation/runtime.py',
                                                'runtime/gaussian_generation/memory.py', 'runtime/common.py'})
                shared.write_text('content identity version two\n')
                after_shared = generation_runtime.execution_source_hashes(vendor)
                self.assertNotEqual(after_driver['runtime/common.py'], after_shared['runtime/common.py'])


if __name__ == '__main__':
    unittest.main()
