"""Numerical interface tests, independent of checkpoints and board availability."""
import struct
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from export import camera_bytes, factor_covariances, recover_covariances
from prepare import centered_affine
from reference import real_sh_basis, render


class BridgeTests(unittest.TestCase):
    def test_world_covariance_with_nontrivial_camera_rotation(self):
        rng = np.random.default_rng(19)
        local_rotation = Rotation.random(100, random_state=rng).as_matrix()
        camera_rotation = Rotation.from_euler("xyz", [31, -29, 82], degrees=True).as_matrix()
        world_rotation = camera_rotation @ local_rotation
        scales = np.exp(rng.uniform(-9, 2, (100, 3)))
        world = (world_rotation * scales[:, None, :]**2) @ world_rotation.swapaxes(-1, -2)
        logs, q, clamped = factor_covariances(world)
        restored = recover_covariances(logs, q)
        error = np.linalg.norm(restored-world, axis=(1, 2))/np.linalg.norm(world, axis=(1, 2))
        self.assertLess(error.max(), 1e-5)
        self.assertEqual(clamped, 0)

    def test_indefinite_covariance_rejected(self):
        with self.assertRaises(ValueError):
            factor_covariances(np.array([np.diag([1., 1., -.1])]))

    def test_pixel_center_and_camera_roundtrip(self):
        w, h, cx, cy = 480, 360, 234.2, 181.3
        affine = centered_affine(w, h, cx, cy, 256)
        np.testing.assert_allclose(affine @ [cx, cy, 1], [127.5, 127.5])
        view = dict(c2w=np.eye(4).tolist(), width=256, height=256,
                    intrinsics_normalized=[[1.2, 0, .5], [0, 1.1, .5], [0, 0, 1]])
        raw = camera_bytes(view)
        self.assertEqual(len(raw), 136)
        vals = struct.unpack("<4I14d", raw[8:])
        point = np.array([.21, -.34, 3.])
        mvs_normalized = np.array([1.2, 1.1])*point[:2]/point[2]+.5
        backend_pixel = np.array(vals[-2:])*point[:2]/point[2]+127.5
        np.testing.assert_allclose(backend_pixel, mvs_normalized*256-.5)

    def test_sh_axes_match_renderer_sign_and_order(self):
        d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]], float)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        b = real_sh_basis(d)
        np.testing.assert_allclose(b[:, 0], .28209479177387814, rtol=1e-6)
        expected = np.stack([-d[:, 1], d[:, 2], -d[:, 0]], -1)*.4886025119029199
        np.testing.assert_allclose(b[:, 1:4], expected, atol=1e-7)
        np.testing.assert_allclose(b[:, 4], 1.0925484305920792*d[:, 0]*d[:, 1], atol=1e-7)
        np.testing.assert_allclose(b[:, 9], -.5900435899266435*d[:, 1]*(3*d[:, 0]**2-d[:, 1]**2), atol=1e-7)

    def test_early_stop_rejects_threshold_crossing_contribution(self):
        n = 8
        g = dict(means=np.stack([np.zeros(n), np.zeros(n), np.linspace(3, 4, n)], -1).astype(np.float32),
                 covariances=np.tile(np.eye(3, dtype=np.float32)*.1, (n, 1, 1)),
                 harmonics=np.zeros((n, 3, 25), np.float32), opacities=np.full(n, .8, np.float32))
        g["harmonics"][:, :, 0] = .5/.28209479177387814
        view = dict(width=17, height=17, c2w=np.eye(4).tolist(),
                    intrinsics_normalized=[[1, 0, .5], [0, 1, .5], [0, 0, 1]])
        im3, im4, _ = render(g, view)
        np.testing.assert_allclose(im3[8, 8], 1-.2**5, atol=2e-7)
        np.testing.assert_array_equal(im3, im4)


if __name__ == "__main__":
    unittest.main()
