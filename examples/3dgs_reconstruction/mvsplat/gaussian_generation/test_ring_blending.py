"""Continuous observed-source transitions, peak transmittance and zero support."""
import unittest

import numpy as np

from gaussian_generation.blending import AngularFeathering,attenuate_opacity


def camera(degrees):
    a=np.radians(degrees)
    pose=np.eye(4)
    pose[:3,:3]=[[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]
    return dict(width=128,height=128,c2w=pose.tolist(),
                intrinsics_normalized=[[1.,0,.5],[0,1.,.5],[0,0,1.]])


def rays(angles):
    a=np.radians(angles)
    return np.stack([np.sin(a),np.zeros(len(a)),np.cos(a)],axis=1).astype(np.float32)


class SourceBlendingTest(unittest.TestCase):
    def fixture(self):
        views=[camera(-10),camera(10)]
        axes=[np.asarray(v['c2w'])[:3,2] for v in views]
        return views,axes

    def test_old_nearest_axis_has_a_step_but_feathering_is_continuous_at_the_same_boundary(self):
        views,axes=self.fixture()
        points=rays([-.001,0,.001])
        owner=(points@np.asarray(axes).T).argmax(axis=1)
        self.assertNotEqual(owner[0],owner[-1])
        weights=AngularFeathering(axes,views).weights(points)
        np.testing.assert_allclose(weights.sum(axis=1),1,rtol=0,atol=1e-12)
        np.testing.assert_allclose(weights[1],[.5,.5],rtol=0,atol=1e-7)
        self.assertLess(float(np.max(abs(weights[0]-weights[-1]))),.001)

    def test_unobserved_directions_stay_empty_and_single_observed_source_has_full_weight(self):
        views,axes=self.fixture()
        points=np.concatenate([rays([90,-90,180,30]),[[0.,1.,0.],[0,0,0]]]).astype(np.float32)
        weights=AngularFeathering(axes,views).weights(points)
        np.testing.assert_array_equal(weights[[0,1,2,4,5]],0)
        np.testing.assert_array_equal(weights[3],[0,1])

    def test_peak_optical_depth_split_preserves_transmittance_and_does_not_modify_inputs(self):
        original=np.array([0.,.2,.7,.99,1.],np.float32)
        before=original.copy()
        a=attenuate_opacity(original,np.full(5,.3))
        b=attenuate_opacity(original,np.full(5,.7))
        np.testing.assert_allclose((1-a)*(1-b),1-np.minimum(original,.99),atol=8e-8,rtol=1e-6)
        np.testing.assert_array_equal(original,before)
        np.testing.assert_array_equal(attenuate_opacity(original,np.zeros(5)),0)

    def test_duplicate_source_axes_share_weight_without_nan_or_arbitrary_winner(self):
        view=camera(0)
        weights=AngularFeathering([[0,0,1],[0,0,1]],[view,view]).weights(rays([0,5]))
        np.testing.assert_allclose(weights,.5,rtol=0,atol=1e-12)

    def test_wrong_camera_axes_and_invalid_controls_are_rejected(self):
        views,axes=self.fixture()
        for band in (True,-1,0,13,float('nan')):
            with self.assertRaises(ValueError):
                AngularFeathering(axes,views,band)
        with self.assertRaises(ValueError):
            AngularFeathering(axes,views[::-1])
        with self.assertRaises(ValueError):
            AngularFeathering(axes,views,8,.3)
        with self.assertRaises(ValueError):
            attenuate_opacity(np.array([.5]),np.array([1.1]))


if __name__=='__main__':
    unittest.main()
