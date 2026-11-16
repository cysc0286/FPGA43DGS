"""Projected-moment preservation and safe bounded neighbour reduction."""
import unittest

import numpy as np

from gaussian_generation.sampling import merge_retained_neighbours
from gaussian_generation import test_ring_footprint
from rendering.reference import project


class MomentReductionTest(unittest.TestCase):
    def fixture(self):
        means=np.column_stack([np.linspace(-.06,.06,6),np.linspace(-.02,.02,6),np.full(6,2.)])
        means=np.concatenate([means,-means]).astype(np.float32)
        g=dict(means=means,covariances=np.repeat(np.diag([1e-5,2e-5,3e-5])[None],12,axis=0).astype(np.float32),
               harmonics=np.random.default_rng(7).normal(0,.1,(12,3,25)).astype(np.float32),
               opacities=np.full(12,.05,np.float32))
        camera=dict(width=128,height=128,c2w=np.eye(4).tolist(),
                    intrinsics_normalized=[[1.,0.,.5],[0.,1.,.5],[0.,0.,1.]])
        views=[camera,dict(camera,c2w=np.diag([-1.,1.,-1.,1.]).tolist())]
        return g,np.array([0,3,6,9]),np.arange(12),views

    @staticmethod
    def screen_moments(g,view):
        projected=project(g,view)
        covariance=test_ring_footprint.FootprintTest.covariance2d(projected)
        mass=projected['opacity']*np.sqrt(np.linalg.det(covariance))
        center=projected['center']
        return (mass.sum(),np.einsum('n,ni->i',mass,center),
                np.einsum('n,nij->ij',mass,covariance+center[:,:,None]*center[:,None,:]))

    def test_reduction_preserves_screen_mass_mean_second_moment_and_colour_mass(self):
        g,selected,owned,views=self.fixture()
        original={k:v.copy() for k,v in g.items()}
        merged,records=merge_retained_neighbours(g,selected,owned,views)
        self.assertEqual(len(merged['means']),len(selected))
        self.assertEqual([r['opacity_clipped_clusters'] for r in records],[0,0])
        for view in views:
            for before,after in zip(self.screen_moments(g,view),self.screen_moments(merged,view)):
                np.testing.assert_allclose(after,before,rtol=3e-6,atol=2e-6)
        for key in g:
            np.testing.assert_array_equal(g[key],original[key])
            self.assertEqual(merged[key].dtype,np.float32)
        self.assertTrue(np.all(np.linalg.eigvalsh(merged['covariances'])>0))
        # All source SH coefficients are the same: any correct weighted merge
        # must preserve them, regardless of the spatial cluster membership.
        g['harmonics'][:]=g['harmonics'][0]
        merged,_=merge_retained_neighbours(g,selected,owned,views)
        np.testing.assert_allclose(merged['harmonics'],g['harmonics'][selected],rtol=1e-6)

    def test_no_reduction_is_bitwise_identity_and_needs_no_camera(self):
        g,_,owned,_=self.fixture()
        merged,_=merge_retained_neighbours(g,owned,owned,[None,None])
        for key in g:
            np.testing.assert_array_equal(merged[key],g[key])

    def test_zero_opacity_clusters_preserve_finite_original_fields(self):
        g,selected,owned,views=self.fixture()
        g['opacities'][:]=0
        with np.errstate(all='raise'):
            merged,_=merge_retained_neighbours(g,selected,owned,views)
        for key in g:
            np.testing.assert_array_equal(merged[key],g[key][selected])

    def test_coincident_representatives_do_not_produce_nan_or_drop_points(self):
        g,selected,owned,views=self.fixture()
        g['means'][:6]=g['means'][0]
        merged,_=merge_retained_neighbours(g,selected,owned,views)
        self.assertEqual(len(merged['means']),4)
        self.assertTrue(all(np.isfinite(v).all() for v in merged.values()))

    def test_invalid_membership_camera_and_empty_representative_are_rejected(self):
        g,selected,owned,views=self.fixture()
        for keep,available in ((selected,owned[1:]),(selected[::-1],owned),(selected[:2],owned)):
            with self.assertRaises(ValueError):
                merge_retained_neighbours(g,keep,available,views)
        with self.assertRaises(ValueError):
            merge_retained_neighbours(g,selected,owned,[None,None])
        shifted=np.eye(4)
        shifted[0,3]=.1
        with self.assertRaises(ValueError):
            merge_retained_neighbours(g,selected,owned,[dict(views[0],c2w=shifted.tolist()),views[1]])


if __name__=='__main__':
    unittest.main()
