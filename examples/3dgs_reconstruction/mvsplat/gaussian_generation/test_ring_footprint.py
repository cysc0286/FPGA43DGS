"""Retained footprint coverage, source-camera validation and field preservation."""
import unittest

import numpy as np

from gaussian_generation.ring import angular_owned_indices
from gaussian_generation.sampling import compensate_covariances
from gaussian_generation import test_ring_sampling
from rendering.reference import project


class FootprintTest(unittest.TestCase):
    def fixture(self):
        base = test_ring_sampling.RingSamplingTest()
        base.setUp()
        base.g['covariances'] *= np.float32(1e-7)
        front = dict(width=128,height=128,c2w=np.eye(4).tolist(),
                     intrinsics_normalized=[[1.,0.,.5],[0.,1.,.5],[0.,0.,1.]])
        back = dict(front,c2w=np.diag([-1.,1.,-1.,1.]).tolist())
        selected,counts = angular_owned_indices(base.g,base.axes,base.axes,[0,1],8192,
                                                return_counts=True)
        return base.g,selected,counts,[front,back]

    @staticmethod
    def covariance2d(attributes):
        conic = attributes['conic']
        matrix = np.empty((len(conic),2,2),np.float32)
        matrix[:,0,0],matrix[:,0,1] = conic[:,0],conic[:,1]
        matrix[:,1,0],matrix[:,1,1] = conic[:,1],conic[:,2]
        return np.linalg.inv(matrix)

    def test_compensation_restores_projected_area_including_projector_filter(self):
        g,index,counts,views = self.fixture()
        original = {k:v.copy() for k,v in g.items()}
        cov,records = compensate_covariances(g,index,views,counts)
        reduced = {k:v[index].copy() for k,v in g.items()}
        for view in views:
            before = self.covariance2d(project(reduced,view))
            corrected = dict(reduced,covariances=cov)
            after = self.covariance2d(project(corrected,view))
            np.testing.assert_allclose(after,2*before,rtol=2e-6,atol=2e-7)
        self.assertEqual([r['footprint_area_factor'] for r in records],[2.,2.])
        self.assertTrue(np.all(np.linalg.eigvalsh(cov)>0))
        self.assertEqual(cov.dtype,np.float32)
        for key in g:
            np.testing.assert_array_equal(g[key],original[key])

    def test_unthinned_output_is_bitwise_unchanged(self):
        g,_,counts,views = self.fixture()
        index = np.arange(len(g['means']))
        cov,records = compensate_covariances(g,index,views,counts)
        np.testing.assert_array_equal(cov,g['covariances'])
        self.assertEqual([r['footprint_area_factor'] for r in records],[1.,1.])

    def test_compensation_rejects_missing_cameras_and_false_rotation_only_metadata(self):
        g,index,counts,views = self.fixture()
        for cameras in ([None,None],[dict(views[0],c2w=np.diag([1.,1.,1.,2.]).tolist()),views[1]]):
            with self.assertRaises(ValueError):
                compensate_covariances(g,index,cameras,counts)
        with self.assertRaises(ValueError):
            compensate_covariances(g,index,views,[1,1])


if __name__=='__main__':
    unittest.main()
