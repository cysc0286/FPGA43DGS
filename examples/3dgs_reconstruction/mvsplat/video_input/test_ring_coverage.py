"""Actual-ray panorama coverage and disjoint selection; no model or board needed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from video_input.ring import horizontal_frustum_coverage, prepare_ring, select_ring_pairs


def camera(yaw=0., pitch=0., roll=0.):
    y, p, r = np.radians([yaw, pitch, roll])
    ry = np.array([[np.cos(y), 0., np.sin(y)], [0., 1., 0.],
                   [-np.sin(y), 0., np.cos(y)]])
    rx = np.array([[1., 0., 0.], [0., np.cos(p), -np.sin(p)],
                   [0., np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0.], [np.sin(r), np.cos(r), 0.],
                   [0., 0., 1.]])
    result = np.eye(4)
    result[:3, :3] = ry @ rx @ rz
    return result


def intrinsics(fx=1., fy=None, cx=.5, cy=.5):
    return np.array([[fx, 0., cx], [0., fx if fy is None else fy, cy], [0., 0., 1.]])


class PairCoverageTest(unittest.TestCase):
    def poses(self, start, end, count=24):
        return {100+3*i:camera(angle) for i, angle in enumerate(np.linspace(start, end, count))}

    def check_unique(self, pairs):
        all_indices = [i for pair in pairs for i in (*pair['context'], pair['target'])]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_full_circle_uses_circular_anchors_and_real_nearest_views(self):
        pairs, meta = select_ring_pairs(self.poses(0, 360), 2)
        self.check_unique(pairs)
        self.assertEqual(meta['anchor_mode'], 'circular_yaw')
        self.assertAlmostEqual(np.diff(meta['anchor_yaw_degrees'])[0], 180.)
        centres = meta['pair_center_yaw_degrees']
        self.assertLess(abs((centres[1]-centres[0])-180.), 360/23)
        # Angles remain image-estimated: no 360/observed-span rescaling.
        for centre in centres:
            self.assertAlmostEqual((centre/(360/23)) % 1, 0., places=8)
        self.assertFalse(meta['full_360_geometry'])

    def test_full_circle_crossing_zero_and_reverse_keep_target_exclusion(self):
        for start, end in [(330, 690), (20, -340)]:
            pairs, meta = select_ring_pairs(self.poses(start, end, 25), 4)
            self.check_unique(pairs)
            self.assertEqual(meta['anchor_mode'], 'circular_yaw')
            np.testing.assert_allclose(np.abs(np.diff(meta['anchor_yaw_degrees'])), 90.)
            self.assertAlmostEqual(meta['observed_yaw_span_degrees'], 360.)

    def test_near_full_circle_is_not_stretched(self):
        _, meta = select_ring_pairs(self.poses(0, 335), 2)
        self.assertEqual(meta['anchor_mode'], 'circular_yaw')
        self.assertAlmostEqual(meta['observed_yaw_span_degrees'], 335.)
        self.assertAlmostEqual(meta['pair_center_yaw_degrees'][0], 335/23)
        self.assertFalse(meta['full_360_geometry'])

    def test_partial_sweep_spreads_over_actual_feasible_interval(self):
        poses = self.poses(0, 120)
        pairs, meta = select_ring_pairs(poses, 2)
        self.check_unique(pairs)
        self.assertEqual(meta['anchor_mode'], 'observed_interval')
        self.assertEqual(pairs[0]['context'], [100, 106])
        self.assertEqual(pairs[-1]['context'], [163, 169])
        np.testing.assert_allclose(meta['pair_center_yaw_degrees'], [120/23, 120-120/23])
        self.assertFalse(meta['full_360_geometry'])
        contexts = [poses[index] for pair in pairs for index in pair['context']]
        coverage = horizontal_frustum_coverage(contexts, intrinsics())
        self.assertLess(coverage['sample_covered_fraction'], .5)
        self.assertGreater(coverage['max_continuous_gap_degrees'], 180)

    def test_partial_crossing_zero_stays_partial(self):
        pairs, meta = select_ring_pairs(self.poses(330, 410), 2)
        self.check_unique(pairs)
        self.assertEqual(meta['anchor_mode'], 'observed_interval')
        self.assertAlmostEqual(meta['observed_yaw_span_degrees'], 80.)
        self.assertLess(abs(np.diff(meta['anchor_yaw_degrees'])[0]), 80.)

    def test_one_group_uses_midpoint_of_partial_sweep(self):
        pairs, meta = select_ring_pairs(self.poses(0, 100, 25), 1)
        self.assertEqual(pairs[0]['target'], 136)
        self.assertAlmostEqual(meta['anchor_yaw_degrees'][0], 50.)

    def test_bad_or_insufficient_rotations_do_not_become_coverage(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_ring_pairs(self.poses(0, 90, 5), 2)
        with self.assertRaisesRegex(ValueError, 'Too little'):
            select_ring_pairs(self.poses(0, 10), 2)
        with self.assertRaises(ValueError):
            select_ring_pairs(self.poses(0, 90), True)
        with self.assertRaisesRegex(ValueError, 'vertical'):
            select_ring_pairs({i:camera(i*15, 90) for i in range(24)}, 2)

    def test_auto_coverage_accepts_64_candidates_with_bounded_eight_pair_output(self):
        poses = {i:camera(angle) for i, angle in enumerate(np.linspace(0., 360., 64, endpoint=False))}
        pairs, meta = select_ring_pairs(poses, 'auto', intrinsic_normalized=intrinsics(.913),
                                       size=128, max_pairs=8, coverage_target=.95)
        self.assertLessEqual(len(pairs), 8)
        self.assertEqual(meta['eligible_triples'], 62)
        self.assertGreaterEqual(meta['sample_covered_fraction'], .95)
        self.check_unique(pairs)

    def test_auto_minimum_pair_budget_continues_after_frustum_target(self):
        poses = {i:camera(angle) for i, angle in enumerate(np.linspace(0., 360., 64, endpoint=False))}
        pairs, meta = select_ring_pairs(poses, 'auto', intrinsic_normalized=intrinsics(.913),
                                       size=128, max_pairs=8, min_pairs=8, coverage_target=.95)
        self.assertEqual(len(pairs), 8)
        self.assertTrue(meta['minimum_pair_count_met'])
        self.assertTrue(meta['target_met'])
        self.assertGreaterEqual(meta['sample_covered_fraction'], .95)
        self.assertEqual(meta['min_pairs'], 8)
        self.assertEqual(meta['selection_stop_reason'], 'pair_budget_after_minimum')
        self.check_unique(pairs)

    def test_auto_rejects_too_few_disjoint_pairs_even_when_angular_target_is_met(self):
        poses = self.poses(0., 360., 20)
        # Six usable triples cover the horizontal rays, but cannot satisfy the
        # requested eight independent pair groups used for the demo scene.
        pairs, meta = select_ring_pairs(poses, 'auto', intrinsic_normalized=intrinsics(.5),
                                       max_pairs=8, min_pairs=1, coverage_target=1.)
        self.assertTrue(meta['target_met'])
        self.assertLess(len(pairs), 8)
        with self.assertRaisesRegex(ValueError, 'Panorama.*minimum.*8'):
            select_ring_pairs(poses, 'auto', intrinsic_normalized=intrinsics(.5),
                              max_pairs=8, min_pairs=8, coverage_target=1.)


class FrustumCoverageTest(unittest.TestCase):
    def coverage(self, cameras, k=None):
        return horizontal_frustum_coverage(cameras, intrinsics() if k is None else k)

    def test_heading_zero_wraps_and_counts_one_circular_gap(self):
        meta = self.coverage([camera()])
        self.assertEqual(meta['sample_covered_degrees'], 53)
        self.assertIn(0, meta['covered_yaw_sample_degrees'])
        self.assertIn(359, meta['covered_yaw_sample_degrees'])
        self.assertEqual(meta['max_continuous_gap_degrees'], 307)
        self.assertEqual(meta['sample_count'], 360)
        self.assertEqual(meta['sample_step_degrees'], 1.)
        self.assertFalse(meta['spherical_coverage_measured'])
        self.assertIn('sampling', meta['coverage_estimate'])

    def test_missing_gap_across_zero_is_not_split_in_two(self):
        meta = self.coverage([camera(180)])
        self.assertEqual(meta['sample_covered_degrees'], 53)
        self.assertEqual(meta['max_continuous_gap_degrees'], 307)
        self.assertNotIn(0, meta['covered_yaw_sample_degrees'])

    def test_pitch_checks_vertical_frustum_not_just_yaw(self):
        level = self.coverage([camera()])
        pitched = self.coverage([camera(pitch=80)])
        self.assertGreater(level['sample_covered_degrees'], 0)
        self.assertEqual(pitched['sample_covered_degrees'], 0)
        self.assertEqual(pitched['max_continuous_gap_degrees'], 360)

    def test_roll_with_non_square_intrinsics_changes_horizontal_coverage(self):
        k = intrinsics(1.5, .5)
        level = self.coverage([camera()], k)
        rolled = self.coverage([camera(roll=90)], k)
        self.assertGreater(rolled['sample_covered_degrees'], level['sample_covered_degrees'])
        self.assertLess(rolled['max_continuous_gap_degrees'], level['max_continuous_gap_degrees'])

    def test_more_real_context_directions_shrink_measured_gaps(self):
        sparse = self.coverage([camera(0), camera(180)])
        denser = self.coverage([camera(angle) for angle in range(0, 360, 90)])
        dense = self.coverage([camera(angle) for angle in range(0, 360, 45)])
        self.assertLess(sparse['sample_covered_degrees'], denser['sample_covered_degrees'])
        self.assertGreater(sparse['max_continuous_gap_degrees'], denser['max_continuous_gap_degrees'])
        self.assertEqual(dense['sample_covered_degrees'], 360)
        self.assertEqual(dense['max_continuous_gap_degrees'], 0)
        self.assertTrue(dense['horizontal_samples_complete'])
        self.assertFalse(dense['spherical_coverage_measured'])

    def test_each_context_uses_its_own_normalized_intrinsic(self):
        narrow, wide = intrinsics(2.), intrinsics(.5)
        separate = self.coverage([camera(0), camera(180)], np.stack((narrow, wide)))
        counts = [self.coverage([camera()], narrow)['sample_covered_degrees'],
                  self.coverage([camera(180)], wide)['sample_covered_degrees']]
        self.assertEqual(separate['per_context_sample_covered_degrees'], counts)
        self.assertEqual(separate['sample_covered_degrees'], sum(counts))

    def test_actual_projection_honours_vertical_principal_point(self):
        centred = self.coverage([camera()])
        shifted = self.coverage([camera()], intrinsics(cy=1.1))
        self.assertGreater(centred['sample_covered_degrees'], 0)
        self.assertEqual(shifted['sample_covered_degrees'], 0)

    def test_bad_inputs_fail_instead_of_inventing_coverage(self):
        invalid = [camera(), camera(), camera(), camera()]
        invalid[0][0, 3] = .001
        invalid[1][0, 0] = 2.
        invalid[2][1, 2] = np.nan
        invalid[3][3, 3] = 0.
        for pose in invalid:
            with self.assertRaises(ValueError):
                self.coverage([pose])
        for k in [intrinsics(-1), np.eye(2), np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                self.coverage([camera()], k)
        with self.assertRaises(ValueError):
            self.coverage([])

    def test_calibrated_focal_none_reaches_prepared_manifest(self):
        poses = {i:camera(angle) for i, angle in enumerate(np.linspace(0, 360, 25))}
        frames = {i:np.zeros((128, 128, 3), np.uint8) for i in poses}
        calibration = SimpleNamespace(intrinsic=np.array([[128., 0., 63.5], [0., 128., 63.5], [0., 0., 1.]]),
                                      metadata=lambda:dict(kind='test_calibration'))
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp)/'input.mp4'
            video.write_bytes(b'fixture-only')
            with patch('video_input.calibration.Calibration', return_value=calibration), \
                    patch('video_input.ring.decode_ring', return_value=(frames, 25, 30.)), \
                    patch('pose_estimation.rotation.estimate_rotation_sequence', return_value=(poses, {})):
                result = prepare_ring(video, Path(tmp)/'prepared', calibration='fixture.json', focal_ratio=None)
            self.assertTrue((Path(tmp)/'prepared/input.json').exists())
            self.assertEqual(result['calibration']['kind'], 'test_calibration')
            self.assertEqual(result['coverage']['image_size'], [128, 128])
            self.assertEqual(len(result['coverage']['per_context_sample_covered_degrees']), 4)
            self.assertFalse(result['coverage']['spherical_coverage_measured'])
            self.assertFalse(result['full_3d_reconstruction'])
        with self.assertRaisesRegex(ValueError, 'valid intrinsics'):
            prepare_ring('unused', 'unused', focal_ratio=None)


class AutomaticCoverageSelectionTest(unittest.TestCase):
    def poses(self, end=360., count=32):
        return {100+3*i:camera(angle) for i,angle in enumerate(np.linspace(0.,end,count))}

    def test_multiple_real_pairs_improve_two_pair_coverage_and_reach_declared_target(self):
        poses=self.poses();k=intrinsics(1.6)
        two,_=select_ring_pairs(poses,2,intrinsic_normalized=k)
        original=horizontal_frustum_coverage([poses[i] for pair in two for i in pair['context']],k)
        pairs,meta=select_ring_pairs(poses,'auto',intrinsic_normalized=k)
        actual=horizontal_frustum_coverage([poses[i] for pair in pairs for i in pair['context']],k)
        self.assertGreater(len(pairs),2);self.assertLessEqual(len(pairs),8)
        self.assertGreater(meta['achieved_fraction'],original['sample_covered_fraction'])
        self.assertEqual(meta['achieved_fraction'],actual['sample_covered_fraction'])
        self.assertTrue(meta['target_met']);self.assertGreaterEqual(meta['achieved_fraction'],.95)
        self.assertEqual(meta['selection_stop_reason'],'coverage_target')
        self.assertEqual(meta['pair_count'],len(pairs))
        self.assertEqual(sum(meta['coverage_gain_degrees_per_pair']),actual['sample_covered_degrees'])
        self.assertTrue(all(v>0 for v in meta['coverage_gain_degrees_per_pair']))
        self.assertEqual(set(meta['uncovered_yaw_sample_degrees']),
                         set(range(360))-set(actual['covered_yaw_sample_degrees']))
        self.assertFalse(meta['full_360_geometry']);self.assertFalse(meta['spherical_coverage_measured'])

    def test_budget_exhaustion_is_reported_as_partial_instead_of_fake_success(self):
        pairs,meta=select_ring_pairs(self.poses(),'auto',intrinsic_normalized=intrinsics(1.6),
                                     max_pairs=2,coverage_target=.99)
        self.assertEqual(len(pairs),2);self.assertEqual(meta['max_pairs'],2)
        self.assertEqual(meta['selection_stop_reason'],'pair_budget');self.assertFalse(meta['target_met'])
        self.assertLess(meta['achieved_fraction'],.99)
        self.assertEqual(len(meta['uncovered_yaw_sample_degrees']),meta['sample_uncovered_degrees'])
        self.assertGreater(meta['max_continuous_gap_degrees'],0)

    def test_no_cropped_overlap_is_rejected_even_if_legacy_full_image_pair_exists(self):
        poses=self.poses()
        self.assertEqual(len(select_ring_pairs(poses,2)[0]),2)
        with self.assertRaisesRegex(ValueError,'cropped-frustum overlap'):
            select_ring_pairs(poses,'auto',intrinsic_normalized=intrinsics(10.))
        with self.assertRaisesRegex(ValueError,'overlapping panorama pair'):
            select_ring_pairs(poses,2,intrinsic_normalized=intrinsics(10.))

    def test_every_target_and_context_is_globally_disjoint_and_comes_from_one_triple(self):
        poses=self.poses();indices=sorted(poses)
        pairs,meta=select_ring_pairs(poses,'auto',intrinsic_normalized=intrinsics(1.))
        context=[i for pair in pairs for i in pair['context']];targets=[pair['target'] for pair in pairs]
        self.assertEqual(len(context+targets),len(set(context+targets)))
        self.assertFalse(set(context)&set(targets))
        for pair,separation,limit in zip(pairs,meta['selected_context_axis_separation_degrees'],
                                        meta['selected_context_axis_limit_degrees']):
            start=indices.index(pair['context'][0])
            self.assertEqual([pair['context'][0],pair['target'],pair['context'][1]],indices[start:start+3])
            self.assertLessEqual(separation,limit+1e-10)

    def test_partial_capture_does_not_turn_into_a_full_circle(self):
        _,meta=select_ring_pairs(self.poses(120),'auto',intrinsic_normalized=intrinsics(1.),max_pairs=8)
        self.assertFalse(meta['target_met']);self.assertLess(meta['achieved_fraction'],.5)
        self.assertAlmostEqual(meta['observed_yaw_span_degrees'],120.)
        self.assertGreater(meta['max_continuous_gap_degrees'],180)

    def test_auto_requires_known_cropped_intrinsics_and_a_bounded_finite_budget(self):
        with self.assertRaisesRegex(ValueError,'final cropped intrinsics'):
            select_ring_pairs(self.poses(),'auto')
        for kwargs in (dict(max_pairs=0),dict(max_pairs=9),dict(max_pairs=True),
                       dict(min_pairs=0),dict(min_pairs=9),dict(min_pairs=True),
                       dict(coverage_target=0),dict(coverage_target=1.01),
                       dict(coverage_target=float('nan')),dict(coverage_target=True)):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'budget or target'):
                select_ring_pairs(self.poses(),'auto',intrinsic_normalized=intrinsics(),**kwargs)
        pairs,meta=select_ring_pairs(self.poses(),'auto',intrinsic_normalized=intrinsics(),
                                     coverage_target=1e-20)
        self.assertEqual(len(pairs),1);self.assertTrue(meta['target_met'])

    def test_prepare_auto_records_actual_pair_count_and_the_final_crop_coverage(self):
        poses={i:camera(a) for i,a in enumerate(np.linspace(0.,360.,32))}
        frames={i:np.zeros((128,128,3),np.uint8) for i in poses}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'input.mp4';video.write_bytes(b'fixture-only')
            with patch('video_input.ring.decode_ring',return_value=(frames,32,30.)),patch(
                    'pose_estimation.rotation.estimate_rotation_sequence',return_value=(poses,{})):
                result=prepare_ring(video,root/'prepared',groups='auto',candidates=32,
                                    focal_ratio=1.6,max_pairs=8,coverage_target=.95)
            meta=result['coverage']
            self.assertEqual(meta['pair_count'],len(result['pairs']))
            self.assertGreater(meta['pair_count'],2);self.assertTrue(meta['target_met'])
            self.assertEqual(len(result['context_indices']),2*meta['pair_count'])
            self.assertFalse(set(result['target_indices'])&set(result['context_indices']))
            self.assertEqual(len(meta['per_context_sample_covered_degrees']),2*meta['pair_count'])
            self.assertFalse(result['full_3d_reconstruction'])

    def test_prepare_uses_cropped_not_full_frame_fov_to_check_context_overlap(self):
        poses={i:camera(a) for i,a in enumerate(np.linspace(0.,360.,32))}
        # Full image horizontal FOV is 36.9 degrees; the square 128 crop is
        # only 18.9 degrees. Context endpoints 23.2 degrees apart do not overlap.
        frames={i:np.zeros((128,256,3),np.uint8) for i in poses}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'input.mp4';video.write_bytes(b'fixture-only')
            with patch('video_input.ring.decode_ring',return_value=(frames,32,30.)),patch(
                    'pose_estimation.rotation.estimate_rotation_sequence',return_value=(poses,{})):
                with self.assertRaisesRegex(ValueError,'cropped-frustum overlap'):
                    prepare_ring(video,root/'rejected',groups='auto',candidates=32,focal_ratio=1.5)
            self.assertFalse((root/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
