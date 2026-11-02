"""Actual two-view geometry gates against known synthetic three-dimensional points."""
from unittest.mock import patch
import unittest

import cv2
import numpy as np

from pose_estimation.geometry import estimate_pair, estimate_target, select_views


class CameraGeometryTests(unittest.TestCase):
    def synthetic_features(self, camera_x):
        rng = np.random.default_rng(20261029)
        world = np.column_stack((rng.uniform(-1, 1, 120), rng.uniform(-.6, .6, 120), rng.uniform(3, 8, 120)))
        k = np.array([[400, 0, 239.5], [0, 400, 134.5], [0, 0, 1.]])
        points, _ = cv2.projectPoints(world, np.zeros(3), np.array([-camera_x, 0., 0.]), k, None)
        keys = [cv2.KeyPoint(float(x), float(y), 8) for x, y in points.reshape(-1, 2)]
        descriptors = rng.uniform(0, 1, (120, 128)).astype(np.float32)
        return k, (keys, descriptors)

    def test_pose_and_pnp_recover_shared_scale_without_target_leakage(self):
        image = np.zeros((270, 480, 3), np.uint8)
        k, first = self.synthetic_features(0)
        _, second = self.synthetic_features(.4)
        _, target = self.synthetic_features(.2)
        pair = estimate_pair(image, image, k, 30, features=(first, second))
        camera, info = estimate_target(image, pair[2], pair[3], pair[1], k, 30, features=target)
        self.assertGreater(pair[4]["median_parallax_deg"], 1)
        np.testing.assert_allclose(camera[:3, 3], pair[0][:3, 3]/2, atol=2e-4)
        self.assertLess(info["median_reprojection_px"], .001)
        self.assertGreaterEqual(info["pnp_inliers"], 100)

    def test_selection_rejects_textureless_video(self):
        image = np.zeros((270, 480, 3), np.uint8)
        k, _ = self.synthetic_features(0)
        with self.assertRaisesRegex(ValueError, "No valid context pair"):
            select_views({0: image, 1: image, 2: image}, k)

    def test_selects_context_and_held_out_target_with_real_geometric_solver(self):
        image = np.random.default_rng(1).integers(0, 256, (270, 480, 3), dtype=np.uint8)
        k, first = self.synthetic_features(0)
        _, middle = self.synthetic_features(.2)
        _, last = self.synthetic_features(.4)
        with patch("pose_estimation.geometry.image_features", side_effect=[first, middle, last]):
            context, targets, pair, results, diagnostics = select_views({0: image, 5: image, 9: image}, k)
        self.assertEqual((context, targets), ([0, 9], [5]))
        self.assertEqual(diagnostics["candidates"], [0, 5, 9])
        self.assertLess(results[5][1]["median_reprojection_px"], .001)

    def test_pure_rotation_gate_does_not_accept_arbitrary_pixel_motion(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (270, 480, 3), dtype=np.uint8)
        k, features = self.synthetic_features(0)
        pair = (np.eye(4), {}, *features, dict(positive_points=100, median_parallax_deg=.05, positive_fraction=1.))
        with patch("pose_estimation.geometry.estimate_pair", return_value=pair), \
                self.assertRaisesRegex(ValueError, "parallax"):
            select_views({0: image, 1: image, 2: image}, k)


if __name__ == "__main__":
    unittest.main()
