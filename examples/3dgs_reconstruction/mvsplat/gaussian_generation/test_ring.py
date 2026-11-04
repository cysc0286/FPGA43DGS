"""Angular ownership and bounded multi-pair orchestration, without device claims."""
import tempfile
import unittest
from pathlib import Path
import weakref
from unittest.mock import patch

import numpy as np

from common import gaussian_content_sha256, sha
from gaussian_generation.ring import angular_owned_indices, generate_ring, persist_ring_archive
from video_input.ring import select_ring_pairs


def axis(degrees):
    r = np.radians(degrees)
    return np.array([np.sin(r), 0., np.cos(r)])


def points(directions):
    n = len(directions)
    return dict(means=np.array(directions, np.float32)*10,
                covariances=np.repeat(np.eye(3)[None], n, axis=0).astype(np.float32),
                harmonics=np.zeros((n, 3, 25), np.float32), opacities=np.ones(n, np.float32)*.9)


class RingTest(unittest.TestCase):
    def test_overlap_owned_once_and_angular_not_depth(self):
        directions = np.array([axis(0), axis(20), axis(180), axis(200)])
        g = points([axis(0), axis(19), axis(20), axis(1)])
        self.assertEqual(angular_owned_indices(g, directions[:2], directions, [0, 1], 8).tolist(), [0, 2])
        g['means'] *= np.array([1., 2., 3., 4.])[:, None]
        self.assertEqual(angular_owned_indices(g, directions[:2], directions, [0, 1], 8).tolist(), [0, 2])

    def test_point_budget_and_bad_frame(self):
        directions = np.array([axis(0), axis(180)])
        g = points([axis(0)]*20+[axis(180)]*20)
        self.assertEqual(len(angular_owned_indices(g, directions, directions, [0, 1], 3)), 6)
        with self.assertRaises(ValueError):
            angular_owned_indices(g, directions[::-1], directions, [0, 1], 3)

    def test_actual_rotation_drives_pair_selection(self):
        poses = {}
        for i, angle in enumerate(np.linspace(0, 360, 25)):
            r = np.radians(angle)
            p = np.eye(4)
            p[:3, :3] = [[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]]
            poses[100+i*2] = p
        pairs, coverage = select_ring_pairs(poses, 4)
        self.assertEqual(len(pairs), 4)
        self.assertAlmostEqual(coverage['observed_yaw_span_degrees'], 360)
        self.assertFalse(coverage['full_360_geometry'])
        self.assertEqual(len({i for p in pairs for i in p['context']}), 8)
        for p in pairs:
            self.assertNotIn(p['target'], p['context'])
        self.assertFalse({i for p in pairs for i in p['context']} & {p['target'] for p in pairs})
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            select_ring_pairs(dict(list(poses.items())[:5]), 2)

    def test_one_model_reused_and_aggregate_budget(self):
        class Model:
            calls = 0
            def infer(self, folder, output, digest, *, memory_only=False):
                if memory_only is not True:
                    raise AssertionError('Each pair must use the retained memory-only handoff')
                if getattr(self, 'pending', None) is not None:
                    raise AssertionError('Previous pair was not consumed')
                self.calls += 1
                d = directions[2*(self.calls-1):2*self.calls]
                self.pending = points([d[0]]*20+[d[1]]*20)
                return dict(complete=True, gaussian_sha256=gaussian_content_sha256(self.pending),
                    gaussian_identity_kind='array_content_v1', archive_verified=False,
                    gaussian_archive=dict(policy='memory_only'),
                    inference_seconds=1., backend='npu', npu_executed=True)
            def take_gaussians(self, digest):
                p, self.pending = self.pending, None
                return p
        directions = np.array([axis(0), axis(20), axis(180), axis(200)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = []
            for i in range(2):
                f = root/str(i)
                f.mkdir()
                (f/'context.npz').write_bytes(bytes([i]))
                pairs.append(dict(directory=str(f), context=[i*2, i*2+1], context_sha256=sha(f/'context.npz')))
            model = Model()
            meta = dict(pairs=pairs, depth_observable=False, context_sha256='b'*64,
                        coverage=dict(directions=directions.tolist()))
            r, g = generate_ring(model, meta, root/'out', max_gaussians=1024)
            self.assertEqual(model.calls, 2)
            self.assertEqual(len(g['means']), 80)
            self.assertTrue(r['npu_executed'])
            self.assertEqual(r['inference_seconds'], 2.)
            self.assertFalse(r['full_3d_reconstruction'])
            self.assertIsNone(model.pending)
            self.assertTrue(r['archive_verified'])
            self.assertEqual(r['gaussian_identity_kind'], 'npz_file_v1')
            self.assertEqual(r['gaussian_sha256'], sha(root/'out/gaussians.npz'))

    def fake_inputs(self, root):
        directions = np.array([axis(0), axis(20), axis(180), axis(200)])
        arrays = []
        pairs = []
        for pair in range(2):
            d = directions[2*pair:2*pair+2]
            g = points([d[0]]*4+[d[1]]*4)
            # Nontrivial fields catch accidental SH/covariance/opacity rewrites.
            g['means'] *= np.arange(1, 9, dtype=np.float32)[:, None]
            g['covariances'] *= np.arange(1, 9, dtype=np.float32)[:, None, None]
            g['harmonics'][:] = np.arange(8*3*25, dtype=np.float32).reshape(8, 3, 25)/256
            g['opacities'][:] = np.linspace(.05, .95, 8, dtype=np.float32)
            arrays.append(g)
            folder = root/str(pair)
            folder.mkdir()
            (folder/'context.npz').write_bytes(bytes([pair]))
            pairs.append(dict(directory=str(folder), context=[pair*2, pair*2+1],
                              context_sha256=sha(folder/'context.npz')))
        prepared = dict(pairs=pairs, depth_observable=False, context_sha256='b'*64,
                        coverage=dict(directions=directions.tolist()))
        return arrays, prepared

    def model(self, arrays, *, retain_sources=True):
        class Model:
            def __init__(self):
                self.calls, self.pending, self.references = 0, None, []
            def infer(self, folder, output, digest, *, memory_only=False):
                if memory_only is not True or self.pending is not None:
                    raise AssertionError('Invalid memory-only handoff')
                if not retain_sources and any(ref() is not None for ref in self.references):
                    raise AssertionError('A complete previous pair remains alive')
                self.pending = {key:value.copy() for key, value in arrays[self.calls].items()}
                self.references = [weakref.ref(value) for value in self.pending.values()]
                self.calls += 1
                return dict(complete=True, gaussian_sha256=gaussian_content_sha256(self.pending),
                    gaussian_identity_kind='array_content_v1', archive_verified=False,
                    gaussian_archive=dict(policy='memory_only'), inference_seconds=.25,
                    backend='npu', npu_executed=True)
            def take_gaussians(self, digest):
                if digest != gaussian_content_sha256(self.pending):
                    raise ValueError('Memory Gaussian identity mismatch')
                result, self.pending = self.pending, None
                return result
        return Model()

    def test_deferred_path_does_not_serialize_any_gaussians_before_first_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays, prepared = self.fake_inputs(root)
            model = self.model(arrays, retain_sources=False)
            with patch('gaussian_generation.ring.np.savez',
                       side_effect=AssertionError('NPZ must not block first frame')):
                record, result = generate_ring(model, prepared, root/'out',
                                               max_gaussians=1024, defer_archive=True)
            self.assertEqual(model.calls, 2)
            self.assertIsNone(model.pending)
            self.assertTrue(all(ref() is None for ref in model.references))
            self.assertFalse((root/'out/gaussians.npz').exists())
            self.assertFalse(record['archive_verified'])
            self.assertEqual(record['gaussian_identity_kind'], 'array_content_v1')
            self.assertEqual(record['gaussian_sha256'], gaussian_content_sha256(result))
            for key in result:
                np.testing.assert_array_equal(result[key], np.concatenate([g[key] for g in arrays]))
            expected_bytes = sum(value.nbytes for g in arrays for value in g.values())
            self.assertEqual(record['storage_io_estimate']['omitted_total_pair_storage_payload_bytes'],
                             3*expected_bytes)
            self.assertIn('not_ddr', record['storage_io_estimate']['kind'])

    def test_deferred_archive_preserves_arrays_and_identity_and_matches_old_npz(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays, prepared = self.fake_inputs(root)
            originals = [{key:value.copy() for key, value in g.items()} for g in arrays]
            record, result = generate_ring(self.model(arrays), prepared, root/'out',
                                           max_gaussians=1024, defer_archive=True)
            identity = record['gaussian_sha256']
            persist_ring_archive(record, result, root/'out')
            self.assertEqual(record['gaussian_sha256'], identity)
            self.assertEqual(record['gaussian_identity_kind'], 'array_content_v1')
            self.assertEqual(record['gaussian_archive_sha256'], sha(root/'out/gaussians.npz'))
            self.assertTrue(record['archive_verified'])
            expected = {key:np.concatenate([g[key] for g in originals]) for key in result}
            np.savez(root/'old_reference.npz', **expected)
            self.assertEqual(sha(root/'old_reference.npz'), record['gaussian_archive_sha256'])
            with np.load(root/'out/gaussians.npz', allow_pickle=False) as restored:
                for key in result:
                    np.testing.assert_array_equal(restored[key], expected[key])
                    np.testing.assert_array_equal(result[key], expected[key])
            for original, actual in zip(originals, arrays):
                for key in original:
                    np.testing.assert_array_equal(original[key], actual[key])
            # Idempotent retry checks the existing digest without writing again.
            with patch('gaussian_generation.ring.np.savez', side_effect=AssertionError('rewrite')):
                persist_ring_archive(record, result, root/'out')

    def test_archive_rejects_mutation_and_never_overwrites_existing_unknown_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays, prepared = self.fake_inputs(root)
            record, result = generate_ring(self.model(arrays), prepared, root/'out',
                                           max_gaussians=1024, defer_archive=True)
            result['harmonics'][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                persist_ring_archive(record, result, root/'out')
            self.assertFalse((root/'out/gaussians.npz').exists())
            result['harmonics'][0, 0, 0] -= 1
            archive = root/'out/gaussians.npz'
            archive.write_bytes(b'unknown archive must remain')
            with self.assertRaises(FileExistsError):
                persist_ring_archive(record, result, root/'out')
            self.assertEqual(archive.read_bytes(), b'unknown archive must remain')

    def test_failed_archive_validation_cleans_only_its_own_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            arrays, prepared = self.fake_inputs(root)
            record, result = generate_ring(self.model(arrays), prepared, root/'out',
                                           max_gaussians=1024, defer_archive=True)
            with patch('gaussian_generation.ring.load_gaussians', side_effect=ValueError('readback')):
                with self.assertRaisesRegex(ValueError, 'readback'):
                    persist_ring_archive(record, result, root/'out')
            self.assertFalse(record['archive_verified'])
            self.assertFalse((root/'out/.gaussians.pending.npz').exists())
            self.assertFalse((root/'out/gaussians.npz').exists())


if __name__ == '__main__':
    unittest.main()
