import importlib.util
import math
import unittest
import torch
import numpy as np


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('icgs.geometry.transforms'), 'Missing isolated geometry')
        from icgs.geometry import transforms as geometry
        self.g = geometry

    def test_pose_and_rigid_point_transform(self):
        pose = np.array([1., 2., 3., 0., 0., 0., 1.])
        transform = self.g.pose_to_transform(pose)
        np.testing.assert_allclose(self.g.transform_to_pose(transform), pose)
        np.testing.assert_allclose(self.g.transform_pcd(np.array([[0., 0., 1.]]), transform), [[1., 2., 4.]])

    def test_rotation_action_roundtrip_and_svd_alignment(self):
        values = torch.tensor([[0.01, -0.02, 0.03, 0., 0., math.pi / 6]])
        transforms = self.g.actions_to_transforms(values)
        torch.testing.assert_close(self.g.transforms_to_actions(transforms), values, atol=2e-6, rtol=2e-5)
        points = torch.tensor([[[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]])
        target = points + torch.tensor([1.,2.,3.])
        rigid = self.g.get_rigid_transforms(points, target)
        torch.testing.assert_close(rigid[0,:3,3], torch.tensor([1.,2.,3.]))
        torch.testing.assert_close(rigid[0,:3,:3], torch.eye(3), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
