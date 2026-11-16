"""Spatial quality regressions at the actual multi-pair retention boundary."""
import unittest

import numpy as np

from gaussian_generation.ring import angular_owned_indices
from gaussian_generation.sampling import balanced_sample_indices,spatial_sample_indices


class RingSamplingTest(unittest.TestCase):
    def setUp(self):
        self.side = 128
        yy, xx = np.mgrid[:self.side, :self.side]
        front = np.stack([(xx.ravel()-63.5)/128, (yy.ravel()-63.5)/128,
                          np.ones(xx.size)], axis=-1).astype(np.float32)
        self.count = len(front)
        means = np.concatenate([front, -front])
        self.g = dict(means=means,
                      covariances=np.repeat(np.eye(3, dtype=np.float32)[None], len(means), axis=0),
                      harmonics=np.zeros((len(means), 3, 25), np.float32),
                      opacities=np.full(len(means), .9, np.float32))
        self.axes = np.array([[0., 0., 1.], [0., 0., -1.]])

    def selected(self, budget):
        return angular_owned_indices(self.g, self.axes, self.axes, [0, 1], budget,sampling='spatial')

    def test_half_budget_never_erases_a_source_column(self):
        for budget in (4096, 5461, 8192):
            selected = self.selected(budget)
            front = selected[selected < self.count]
            for coordinate in (front % self.side, front//self.side):
                counts = np.bincount(coordinate, minlength=self.side)
                self.assertGreater(int(counts.min()), 0,
                                   'Thinning must not create empty image rows or columns')

    def test_half_budget_spreads_each_block_across_both_axes(self):
        selected = self.selected(self.count//2)
        mask = np.zeros(self.count, dtype=np.uint8)
        mask[selected[selected < self.count]] = 1
        blocks = mask.reshape(64, 2, 64, 2)
        np.testing.assert_array_equal(blocks.sum(axis=(1, 3)), np.full((64, 64), 2))
        np.testing.assert_array_equal(blocks.sum(axis=1), np.ones((64, 64, 2)))
        np.testing.assert_array_equal(blocks.sum(axis=3), np.ones((64, 2, 64)))

    def test_all_scene_budgets_are_exact_unique_deterministic_and_leave_inputs_untouched(self):
        before = {k:v.copy() for k, v in self.g.items()}
        for budget in (1, 4096, 5461, 8192, self.count):
            selected = self.selected(budget)
            np.testing.assert_array_equal(selected, self.selected(budget))
            self.assertEqual(len(selected), 2*budget)
            self.assertEqual(len(np.unique(selected)), len(selected))
            self.assertTrue(np.all(selected[1:] > selected[:-1]))
        for key in self.g:
            np.testing.assert_array_equal(self.g[key], before[key])

    def test_irregular_ownership_on_odd_rectangular_source_keeps_only_existing_pixels(self):
        y, x = np.mgrid[:7,:11]
        eligible = np.flatnonzero((x < 8) & ((x+y) % 5 != 0))
        for budget in (1, 9, len(eligible)-1, len(eligible), len(eligible)+5):
            selected = spatial_sample_indices(eligible, budget, (7,11))
            self.assertEqual(len(selected), min(budget,len(eligible)))
            self.assertTrue(np.all(np.isin(selected,eligible)))
            self.assertTrue(np.all(selected[1:] > selected[:-1]))
            np.testing.assert_array_equal(selected, spatial_sample_indices(eligible,budget,(7,11)))

    def test_legacy_ablation_still_reproduces_empty_columns(self):
        selected = angular_owned_indices(self.g,self.axes,self.axes,[0,1],8192,
                                         sampling='legacy_linear')
        counts = np.bincount(selected[selected < self.count] % 128,minlength=128)
        self.assertEqual(int(np.count_nonzero(counts==0)),63)

    def test_invalid_source_mapping_is_rejected(self):
        for ids,shape in ((np.array([1,1]),(2,2)),(np.array([-1]),(2,2)),
                          (np.array([4]),(2,2)),(np.array([.5]),(2,2))):
            with self.assertRaises(ValueError):
                spatial_sample_indices(ids,1,shape)
        with self.assertRaises(ValueError):
            angular_owned_indices(self.g,self.axes,self.axes,[0,1],8192,source_shape=(128,127))

    def test_balanced_policy_changes_only_aliased_line_patterns(self):
        indices = np.arange(128*128)
        for budget in (4096,8192):
            selected = balanced_sample_indices(indices,budget,(128,128))
            self.assertTrue(np.all(np.bincount(selected % 128,minlength=128)>0))
        # One-third decimation is already spread across rows/columns. Keep it.
        selected = balanced_sample_indices(indices,5461,(128,128))
        np.testing.assert_array_equal(selected,indices[np.linspace(0,len(indices)-1,5461,dtype=int)])


if __name__ == '__main__':
    unittest.main()
