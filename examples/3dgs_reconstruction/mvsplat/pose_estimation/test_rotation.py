"""Known image rotations, rejected motion and bounded pose-chain lifecycle."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pose_estimation.rotation import estimate_rotation_sequence


class RotationSequenceTests(unittest.TestCase):
    def synthetic(self, angles):
        rng = np.random.default_rng(20261103)
        k = np.array([[220., 0, 159.5], [0, 220., 119.5], [0, 0, 1.]])
        pixels = np.column_stack((rng.uniform(20, 300, 120), rng.uniform(20, 220, 120)))
        rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
        descriptors = rng.uniform(0, 1, (len(pixels), 128)).astype(np.float32)
        frames, features, expected = {}, {}, {}
        for index, angle in enumerate(angles):
            rotation = cv2.Rodrigues(np.array([0., np.radians(angle), 0.]))[0]
            projected = rays @ rotation @ k.T
            points = projected[:, :2]/projected[:, 2:]
            keys = [cv2.KeyPoint(float(x), float(y), 8) for x, y in points]
            frame = np.full((240, 320, 3), index, np.uint8)
            frames[index] = frame
            features[index] = (keys, descriptors.copy())
            expected[index] = rotation
        def detector(image):
            return features[int(image[0, 0, 0])]
        return frames, k, expected, features, detector

    def test_known_rotations_recover_c2w_and_exact_zero_translation(self):
        frames, k, expected, _, detector = self.synthetic([0, 6, 12, 18])
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        for index in frames:
            np.testing.assert_allclose(poses[index][:3, :3], expected[index], atol=2e-6)
            np.testing.assert_array_equal(poses[index][:3, 3], np.zeros(3))
            np.testing.assert_allclose(np.linalg.det(poses[index][:3, :3]), 1., atol=1e-12)
        self.assertAlmostEqual(info['yaw_degrees_by_index']['3'], 18., places=4)
        self.assertLess(max(e['median_reprojection_px'] for e in info['per_edge']), 1e-3)
        self.assertEqual(info['translation'], 'zero_unobserved')
        self.assertFalse(info['depth_observable'])
        self.assertIsNone(info['closure_error_degrees'])
        self.assertLessEqual(info['feature_cache_peak_frames'], 4)
        self.assertEqual(info['anchor_index'], 0)
        self.assertEqual(info['component_sizes'], [4])

    def test_explicit_weak_texture_detector_keeps_rotation_validation(self):
        frames, k, expected, features, _ = self.synthetic([0, 6, 12, 18])
        class Detector:
            def detectAndCompute(self, gray, mask):
                return features[int(gray[0, 0])]
        with patch('pose_estimation.rotation.cv2.SIFT_create', return_value=Detector()) as factory:
            poses, info = estimate_rotation_sequence(frames, k, feature_contrast=.01)
        factory.assert_called_once_with(nfeatures=2048, contrastThreshold=.01)
        self.assertEqual(info['feature_contrast'], .01)
        self.assertEqual(info['minimum_rotation_inliers'], 20)
        self.assertEqual(info['maximum_reprojection_px'], 3.)
        for index in frames:
            np.testing.assert_allclose(poses[index][:3, :3], expected[index], atol=2e-6)
            np.testing.assert_array_equal(poses[index][:3, 3], np.zeros(3))

    def test_invalid_feature_contrast_is_rejected(self):
        frames, k, _, _, _ = self.synthetic([0, 6])
        for contrast in (0, -.01, .05, float('nan')):
            with self.subTest(contrast=contrast), self.assertRaises(ValueError):
                estimate_rotation_sequence(frames, k, feature_contrast=contrast)

    def test_bad_texture_at_start_reanchors_only_valid_later_component(self):
        frames, k, expected, features, detector = self.synthetic([0, 5, 10, 15, 20])
        features[0] = ([], None)
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        self.assertEqual(info['anchor_index'], 1)
        self.assertEqual(info['registered_frame_indices'], [1, 2, 3, 4])
        self.assertEqual(info['excluded_frame_indices'], [0])
        self.assertEqual(info['component_sizes'], [4, 1])
        np.testing.assert_array_equal(poses[1], np.eye(4))
        for index in poses:
            np.testing.assert_allclose(poses[index][:3, :3], expected[1].T @ expected[index], atol=2e-6)
            np.testing.assert_array_equal(poses[index][:3, 3], np.zeros(3))
        self.assertLessEqual(info['feature_cache_peak_frames'], 4)

    def test_inconsistent_initial_motion_does_not_discard_valid_later_sweep(self):
        frames, k, _, features, detector = self.synthetic([0, 5, 10, 15, 20])
        keys, descriptors = features[0]
        features[0] = ([cv2.KeyPoint(key.pt[0]+70, key.pt[1], 8) for key in keys], descriptors)
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        self.assertEqual(sorted(poses), [1, 2, 3, 4])
        self.assertEqual(info['anchor_index'], 1)
        self.assertTrue(any(edge['source'] == 0 for edge in info['rejected_edges']))
        self.assertLess(info['observed_yaw_span_degrees'], 16.)

    def test_disconnected_components_are_not_merged_and_largest_wins(self):
        frames, k, expected, features, detector = self.synthetic(list(range(0, 40, 5)))
        rng = np.random.default_rng(701)
        second_descriptors = rng.uniform(20, 21, features[0][1].shape).astype(np.float32)
        for index in range(3, 8):
            features[index] = (features[index][0], second_descriptors.copy())
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        self.assertEqual(info['component_sizes'], [5, 3])
        self.assertEqual(info['anchor_index'], 3)
        self.assertEqual(sorted(poses), [3, 4, 5, 6, 7])
        self.assertEqual(info['missing_frame_indices'], [0, 1, 2])
        self.assertFalse(any(edge['source'] < 3 for edge in info['per_edge']))
        for index in poses:
            np.testing.assert_allclose(poses[index][:3, :3], expected[3].T @ expected[index], atol=2e-6)
            np.testing.assert_array_equal(poses[index][:3, 3], np.zeros(3))

    def test_equal_components_choose_earliest_anchor_deterministically(self):
        frames, k, _, features, detector = self.synthetic(list(range(0, 30, 5)))
        rng = np.random.default_rng(703)
        second_descriptors = rng.uniform(20, 21, features[0][1].shape).astype(np.float32)
        for index in range(3, 6):
            features[index] = (features[index][0], second_descriptors.copy())
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        self.assertEqual(info['component_sizes'], [3, 3])
        self.assertEqual(info['anchor_index'], 0)
        self.assertEqual(sorted(poses), [0, 1, 2])

    def test_missing_frame_is_explicit_and_two_frame_fallback_has_real_rotation(self):
        frames, k, expected, features, detector = self.synthetic([0, 5, 10, 15])
        features[1] = ([], None)
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        self.assertNotIn(1, poses)
        self.assertEqual(info['missing_frame_indices'], [1])
        np.testing.assert_allclose(poses[2][:3, :3], expected[2], atol=2e-6)
        self.assertEqual(info['per_edge'][0]['source'], 0)
        self.assertEqual(info['per_edge'][0]['target'], 2)

    def test_projective_translation_is_not_accepted_as_rotation(self):
        frames, k, _, features, detector = self.synthetic([0, 0])
        keys, desc = features[1]
        features[1] = ([cv2.KeyPoint(key.pt[0]+70, key.pt[1], 8) for key in keys], desc)
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            with self.assertRaisesRegex(ValueError, 'No valid rotation edge'):
                estimate_rotation_sequence(frames, k)

    def test_textureless_frames_fail_instead_of_returning_uniform_fake_poses(self):
        frames = {i: np.zeros((96, 128, 3), np.uint8) for i in range(4)}
        k = np.array([[90., 0, 63.5], [0, 90., 47.5], [0, 0, 1.]])
        with self.assertRaisesRegex(ValueError, 'No valid rotation edge'):
            estimate_rotation_sequence(frames, k)

    def test_actual_sift_on_warped_texture_recovers_known_camera_rotation(self):
        rng = np.random.default_rng(8)
        image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), .7)
        k = np.array([[250., 0, 159.5], [0, 250., 119.5], [0, 0, 1.]])
        expected = cv2.Rodrigues(np.array([0., np.radians(5.), 0.]))[0]
        second = cv2.warpPerspective(image, k @ expected.T @ np.linalg.inv(k), (320, 240))
        poses, info = estimate_rotation_sequence({0: image, 1: second}, k)
        error = cv2.Rodrigues(poses[1][:3, :3] @ expected.T)[0]
        self.assertLess(float(np.degrees(np.linalg.norm(error))), .1)
        self.assertGreaterEqual(info['per_edge'][0]['inliers'], 20)

    def test_local_homography_noise_does_not_discard_valid_bearing_consensus(self):
        from pose_estimation.rotation import _nearest_rotation, _relative
        rng = np.random.default_rng(13)
        k = np.array([[220., 0., 159.5], [0., 220., 119.5], [0., 0., 1.]])
        p0 = np.column_stack((rng.uniform(98, 122, 120), rng.uniform(93, 117, 120), np.ones(120)))
        rotation = cv2.Rodrigues(np.array([0., np.radians(-6.), 0.]))[0]
        truth = k @ rotation @ np.linalg.inv(k)
        projected = p0 @ truth.T
        p1 = projected[:, :2]/projected[:, 2:]
        centre = np.array([[1., 0., 110.], [0., 1., 105.], [0., 0., 1.]])
        local_noise = centre @ np.array([[1., 0., 0.], [0., 1., 0.], [.0025, 0., 1.]]) @ np.linalg.inv(centre)
        noisy_h = truth @ local_noise
        noisy_pixels = p0 @ noisy_h.T
        noisy_pixels = noisy_pixels[:, :2]/noisy_pixels[:, 2:]
        self.assertLess(float(np.max(np.linalg.norm(noisy_pixels-p1, axis=1))), .5)
        # H remains an accurate consensus model locally, while its matrix-SVD
        # rotation is badly biased. The old initialization loses all120 points.
        calibrated = np.linalg.inv(k) @ noisy_h @ k
        calibrated /= np.cbrt(np.linalg.det(calibrated))
        old_r = _nearest_rotation(calibrated)
        old_pixels = p0 @ np.linalg.inv(k).T @ old_r.T @ k.T
        old_pixels = old_pixels[:, :2]/old_pixels[:, 2:]
        self.assertEqual(int(np.sum(np.linalg.norm(old_pixels-p1, axis=1) <= 3.)), 0)
        descriptors = rng.uniform(0, 1, (120, 128)).astype(np.float32)
        first = ([cv2.KeyPoint(float(x), float(y), 8) for x, y in p0[:, :2]], descriptors)
        second = ([cv2.KeyPoint(float(x), float(y), 8) for x, y in p1], descriptors.copy())
        with patch('pose_estimation.rotation.cv2.findHomography',
                   return_value=(noisy_h, np.ones((120, 1), np.uint8))):
            actual, details = _relative(first, second, k, 20, 3.)
        np.testing.assert_allclose(actual, rotation, atol=2e-6)
        self.assertEqual(details['inliers'], 120)
        self.assertLess(details['max_reprojection_px'], 1e-3)
        self.assertEqual(details['initialization'], 'homography_inlier_unit_bearings')

    def test_full_turn_uses_observed_rotations_and_reports_unadjusted_closure(self):
        rng = np.random.default_rng(9)
        azimuth = rng.uniform(-np.pi, np.pi, 1800)
        elevation = rng.uniform(-.45, .45, len(azimuth))
        world = np.column_stack((np.sin(azimuth)*np.cos(elevation), np.sin(elevation),
                                 np.cos(azimuth)*np.cos(elevation)))
        desc = rng.uniform(0, 1, (len(world), 128)).astype(np.float32)
        k = np.array([[160., 0, 159.5], [0, 160., 119.5], [0, 0, 1.]])
        frames, features = {}, {}
        for index, angle in enumerate(range(0, 361, 20)):
            rotation = cv2.Rodrigues(np.array([0., np.radians(angle), 0.]))[0]
            rays = world @ rotation
            projected = rays @ k.T
            pixels = projected[:, :2]/projected[:, 2:]
            visible = (rays[:, 2] > 0) & (pixels[:, 0] > 5) & (pixels[:, 0] < 315)
            visible &= (pixels[:, 1] > 5) & (pixels[:, 1] < 235)
            keys = [cv2.KeyPoint(float(x), float(y), 8) for x, y in pixels[visible]]
            features[index] = (keys, desc[visible].copy())
            frames[index] = np.full((240, 320, 3), index, np.uint8)
        def detector(image):
            return features[int(image[0, 0, 0])]
        with patch('pose_estimation.rotation.image_features', side_effect=detector):
            poses, info = estimate_rotation_sequence(frames, k)
        self.assertEqual(len(poses), len(frames))
        self.assertAlmostEqual(info['observed_yaw_span_degrees'], 360., places=3)
        self.assertIsNotNone(info['closure_error_degrees'])
        self.assertLess(info['closure_error_degrees'], .001)
        self.assertFalse(info['closure_adjusted'])
        self.assertLessEqual(info['feature_cache_peak_frames'], 4)

    def test_invalid_inputs_are_rejected(self):
        frames, k, _, _, _ = self.synthetic([0, 5])
        for invalid in (np.full((3, 3), np.nan), np.zeros((3, 3)), np.eye(4)):
            with self.subTest(intrinsic=invalid.shape), self.assertRaises(ValueError):
                estimate_rotation_sequence(frames, invalid)
        for count in (1, 33):
            with self.subTest(count=count), self.assertRaises(ValueError):
                estimate_rotation_sequence({i: frames[0] for i in range(count)}, k)
        with self.assertRaises(ValueError):
            estimate_rotation_sequence({0: frames[0], 1: frames[0].astype(np.float32)}, k)
        with self.assertRaises(ValueError):
            estimate_rotation_sequence(frames, k, max_reprojection=np.inf)


if __name__ == '__main__':
    unittest.main()
