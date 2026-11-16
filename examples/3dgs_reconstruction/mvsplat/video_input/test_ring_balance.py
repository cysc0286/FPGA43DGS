"""Direction allocation tests; synthetic cameras are not physical validation."""
import unittest

import numpy as np

from video_input.ring import select_ring_pairs
from video_input.ring_balance import circular_gap, interior_support
from video_input.test_ring_coverage import camera, intrinsics


class BalancedRingTest(unittest.TestCase):
    def select(self, angles, policy='balanced', k=None, **kwargs):
        poses = {i: camera(angle) for i, angle in enumerate(angles)}
        k = intrinsics(.7255555555555555) if k is None else k
        pairs, meta = select_ring_pairs(poses, 'auto', intrinsic_normalized=k,
                                       coverage_target=1., selection_policy=policy, **kwargs)
        contexts = [i for pair in pairs for i in pair['context']]
        targets = [pair['target'] for pair in pairs]
        self.assertEqual(len(contexts+targets), len(set(contexts+targets)))
        for pair in pairs:
            self.assertLess(pair['context'][0], pair['target'])
            self.assertLess(pair['target'], pair['context'][1])
        return pairs, meta

    def test_full_turn_can_run_in_both_directions_across_zero(self):
        for start, end in ((0, 360), (355, 715), (10, -350)):
            with self.subTest(start=start, end=end):
                pairs, meta = self.select(np.linspace(start, end, 32))
                self.assertTrue(meta['target_met'])
                self.assertEqual(meta['max_continuous_gap_degrees'], 0)
                self.assertLessEqual(len(pairs), 8)
                self.assertEqual(len(meta['sector_support_mean']), 12)
                self.assertFalse(meta['spherical_coverage_measured'])

    def test_nonuniform_motion_uses_angular_spacing_and_retains_coverage(self):
        angles = 360.*np.linspace(0., 1., 64)**2.5
        legacy, old = self.select(angles, 'legacy')
        pairs, new = self.select(angles)
        self.assertGreaterEqual(new['sample_covered_degrees'], old['sample_covered_degrees'])
        self.assertTrue(any(p['context'][1]-p['context'][0] > 2 for p in pairs))
        self.assertGreater(new['eligible_pair_candidates'], new['eligible_triples'])
        self.assertTrue(new['target_met'])

    def test_pauses_do_not_force_duplicate_direction_groups(self):
        angles = np.r_[np.zeros(20), np.linspace(0., 360., 44)]
        pairs, meta = self.select(angles)
        self.assertTrue(meta['target_met'])
        self.assertTrue(all(g > 0 for g in meta['coverage_gain_degrees_per_pair']))
        self.assertLessEqual(len(pairs), 8)

    def test_partial_capture_stays_partial_and_reports_missing_angles(self):
        _, meta = self.select(np.linspace(0., 120., 32))
        self.assertFalse(meta['target_met'])
        self.assertGreater(len(meta['uncovered_yaw_sample_degrees']), 0)
        self.assertGreater(meta['max_continuous_gap_degrees'], 100)
        self.assertAlmostEqual(meta['observed_yaw_span_degrees'], 120.)

    def test_interior_proxy_has_zero_weight_behind_and_near_edges(self):
        support = interior_support(np.array([camera()]), np.array([intrinsics(1.)]))[0]
        self.assertAlmostEqual(support[0], 1.)
        self.assertEqual(support[180], 0.)
        self.assertGreater(support[10], support[25])

    def test_reducing_calls_preserves_parent_interior_and_weakest_sector_support(self):
        rng = np.random.default_rng(31)
        angles = np.r_[0., np.cumsum(rng.uniform(.5, 1.5, 17))]
        angles *= 360./angles[-1]
        old_pairs, old = self.select(angles, 'legacy')
        new_pairs, new = self.select(angles)
        self.assertTrue(old['target_met'])
        k = intrinsics(.7255555555555555)
        def support(pairs):
            cameras = np.array([camera(angles[i]) for pair in pairs for i in pair['context']])
            return interior_support(cameras, np.repeat(k[None], len(cameras), 0)).max(axis=0)
        original, candidate = support(old_pairs), support(new_pairs)
        self.assertGreaterEqual(np.count_nonzero(candidate >= .2), np.count_nonzero(original >= .2))
        self.assertGreaterEqual(candidate.reshape(12, 30).mean(axis=1).min()+1e-12,
                                original.reshape(12, 30).mean(axis=1).min())
        self.assertLessEqual(circular_gap(candidate >= .2), circular_gap(original >= .2))

    def test_explicit_legacy_path_keeps_original_triples(self):
        pairs, meta = self.select(np.linspace(0., 360., 32), 'legacy')
        self.assertEqual(meta['selection_policy'], 'legacy')
        self.assertTrue(all(p['context'][1]-p['context'][0] == 2 for p in pairs))
        self.assertNotIn('central_sample_covered_fraction', meta)

    def test_unknown_policy_fails_before_any_pair_selection(self):
        with self.assertRaisesRegex(ValueError, 'selection policy'):
            self.select(np.linspace(0., 360., 32), 'unsupported')


if __name__ == '__main__':
    unittest.main()
