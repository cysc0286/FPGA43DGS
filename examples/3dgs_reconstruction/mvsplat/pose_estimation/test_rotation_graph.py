"""Measured loop constraints must correct drift without inventing a full turn."""
import unittest
import cv2
import numpy as np

from pose_estimation.rotation_graph import refine_observed_rotations


class ObservedRotationGraphTests(unittest.TestCase):
    @staticmethod
    def yaw(degrees):
        return cv2.Rodrigues(np.array([0., np.radians(degrees), 0.]))[0]

    def fixture(self):
        truth={i:self.yaw(i*12.) for i in range(30)}
        poses={i:np.eye(4) for i in truth}
        for i in poses:poses[i][:3,:3]=self.yaw(i*11.)
        local=[(i,i+1,self.yaw(-11.),30.) for i in range(29)]
        loops=[(0,i,truth[i].T,100.) for i in (28,29)]
        return truth,poses,local,loops

    def test_actual_partial_endpoint_is_preserved_instead_of_forcing_360(self):
        truth,poses,local,loops=self.fixture()
        result,info=refine_observed_rotations(poses,local,loops)
        self.assertTrue(info['applied'])
        for index in (28,29):
            error=cv2.Rodrigues(result[index][:3,:3]@truth[index].T)[0]
            self.assertLess(np.degrees(np.linalg.norm(error)),1.)
        # The last observation is 348 degrees, not an artificial 360-degree pose.
        self.assertGreater(np.linalg.norm(result[29][:3,:3]-np.eye(3)),.1)
        for pose in result.values():
            np.testing.assert_array_equal(pose[:3,3],np.zeros(3))
            np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-10)
        np.testing.assert_array_equal(result[0],np.eye(4))
        self.assertLess(info['max_loop_after_degrees'],3.)

    def test_conflicting_loop_edges_do_not_replace_the_original_poses(self):
        _,poses,local,loops=self.fixture()
        loops[1]=(0,29,self.yaw(-250.),100.)
        result,info=refine_observed_rotations(poses,local,loops)
        self.assertFalse(info['applied'])
        for index in poses:np.testing.assert_array_equal(result[index],poses[index])

    def test_one_loop_is_not_sufficient_evidence(self):
        _,poses,local,loops=self.fixture()
        result,info=refine_observed_rotations(poses,local,loops[:1])
        self.assertFalse(info['applied'])

    def test_already_consistent_sequence_is_unchanged(self):
        truth,poses,local,loops=self.fixture()
        for i in poses:poses[i][:3,:3]=truth[i]
        local=[(i,i+1,self.yaw(-12.),30.) for i in range(29)]
        result,info=refine_observed_rotations(poses,local,loops)
        self.assertFalse(info['applied'])
        for index in poses:np.testing.assert_array_equal(result[index],poses[index])


if __name__=='__main__':unittest.main()
