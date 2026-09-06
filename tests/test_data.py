"""CPU data-contract tests using real torch/PyG serialization."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch import nn


def prepared_sample():
    first = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
    second = first + 10
    demos = [
        dict(obs=[first[:3], first[3:]], grips=[0.0, 1.0],
             T_w_es=[np.eye(4), np.eye(4)]),
        dict(obs=[second[:3], second[3:]], grips=[1.0, 0.0],
             T_w_es=[np.eye(4), np.eye(4)]),
    ]
    target = np.eye(4, dtype=np.float32)
    target[0, 3] = 0.25
    live = dict(
        obs=[first[:3], second[:3]],
        grips=[0.0, 1.0],
        actions=[np.stack([target, np.eye(4)]), np.stack([np.eye(4), target])],
        actions_grip=[[1.0, 0.0], [0.0, 1.0]],
        T_w_es=[np.eye(4), target],
    )
    return dict(demos=demos, live=live)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.embd_dim = 2

    def forward(self, features, positions, batch):
        indices = torch.stack([torch.where(batch == value)[0][0]
                               for value in batch.unique(sorted=True)])
        embeddings = self.weight * torch.ones(len(indices), self.embd_dim,
                                               device=positions.device)
        return embeddings, positions[indices], batch[indices]


class DataTests(unittest.TestCase):
    def test_save_sample_roundtrips_real_pyg_files_and_schema(self):
        from icgs.data.preprocessing.native import save_sample

        with tempfile.TemporaryDirectory(prefix="ip-data-test-") as directory:
            save_sample(prepared_sample(), directory, offset=7)
            paths = sorted(Path(directory).glob("*.pt"))
            self.assertEqual([path.name for path in paths], ["data_7.pt", "data_8.pt"])
            data = torch.load(paths[0], weights_only=False)

        self.assertEqual(data.pos_demos.shape, (12, 3))
        self.assertEqual(data.batch_demos.tolist(), [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
        self.assertEqual(data.graps_demos.shape, (1, 2, 2, 1))
        torch.testing.assert_close(data.graps_demos.flatten(), torch.tensor([-1.0, 1.0, 1.0, -1.0]))
        self.assertEqual(data.actions.shape, (1, 2, 4, 4))
        self.assertEqual(data.actions_grip.tolist(), [[1.0, -1.0]])
        self.assertFalse(hasattr(data, "demo_scene_node_embds"))
        self.assertFalse(hasattr(data, "live_scene_node_embds"))

    def test_cached_fields_exist_only_when_encoder_is_supplied(self):
        from icgs.data.preprocessing.native import save_sample

        data = save_sample(prepared_sample(), scene_encoder=TinyEncoder())

        self.assertIsNone(data.get("pos_demos"))
        self.assertIsNone(data.get("batch_demos"))
        self.assertEqual(data.demo_scene_node_embds.shape, (1, 2, 2, 1, 2))
        self.assertEqual(data.demo_scene_node_pos.shape, (1, 2, 2, 1, 3))
        self.assertEqual(data.live_scene_node_embds.shape, (1, 1, 2))
        self.assertEqual(data.live_scene_node_pos.shape, (1, 1, 3))
        self.assertIsNotNone(data.pos_obs)

    def test_live_targets_share_anchor_and_pad_with_identity_and_last_grip(self):
        from icgs.data.preprocessing import native as preprocessing

        poses = [np.eye(4) for _ in range(3)]
        poses[0] = poses[0].copy()
        poses[0][0, 3] = 1.0
        poses[1] = poses[1].copy()
        poses[1][0, 3] = 1.5
        poses[2] = poses[2].copy()
        poses[2][1, 3] = 2.0
        raw = dict(T_w_es=poses, grips=[0.0, 1.0, 0.0],
                   pcds=[np.arange(9).reshape(3, 3).astype(float) + i for i in range(3)])

        with mock.patch.object(preprocessing, "remove_statistical_outliers",
                               side_effect=lambda points, **kwargs: (points, np.arange(len(points)))):
            live = preprocessing.sample_to_live(raw, pred_horizon=3, num_points=3,
                                                subsample=False)

        np.testing.assert_allclose(live["actions"][0][0], np.linalg.inv(poses[0]) @ poses[1])
        np.testing.assert_allclose(live["actions"][0][1], np.linalg.inv(poses[0]) @ poses[2])
        np.testing.assert_allclose(live["actions"][0][2], np.eye(4))
        self.assertEqual(live["actions_grip"][0], [1.0, 0.0, 0.0])
        actual = live["obs"][0][np.argsort(live["obs"][0][:, 0])]
        expected = raw["pcds"][0] - np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(actual, expected)

    def test_subsample_thresholds_state_changes_and_rotation_units(self):
        from icgs.data.preprocessing.native import subsample_traj

        poses = [np.eye(4) for _ in range(3)]
        poses[1] = poses[1].copy()
        poses[1][0, 3] = 0.005
        poses[2] = poses[2].copy()
        poses[2][:3, :3] = Rot.from_euler("z", 2.0, degrees=True).as_matrix()
        sampled, grips = subsample_traj(poses, [0.0, 1.0, 1.0],
                                        trans_space=0.01, rot_space=3)

        self.assertEqual(len(sampled), 3)
        self.assertEqual(grips, [0.0, 1.0, 1.0])
        np.testing.assert_allclose(sampled[1], poses[1])
        np.testing.assert_allclose(sampled[-1], poses[-1])

    def test_public_preprocessing_exports_reach_canonical_implementation(self):
        from icgs.data import preprocessing
        from icgs.data.preprocessing import native
        self.assertIs(preprocessing.save_sample,native.save_sample)
        self.assertIs(preprocessing.sample_to_live,native.sample_to_live)
        self.assertIs(preprocessing.downsample_pcd,native.downsample_pcd)

    def test_running_dataset_retries_and_preserves_original_augmentations(self):
        from torch_geometric.data import Data
        from icgs.data.datasets import native as dataset

        stored = Data(actions=torch.eye(4).repeat(2, 1, 1),
                      actions_grip=torch.ones(2),
                      current_grip=torch.tensor([1.0]))
        source = dataset.RunningDataset("unused", 2, rand_g_prob=1.0)
        with mock.patch.object(dataset.torch, "load",
                               side_effect=[ValueError("bad sample"), stored]), \
             mock.patch.object(dataset.np.random, "randint", return_value=1), \
             mock.patch.object(dataset.np.random, "uniform", return_value=0.0):
            result = source[0]
        self.assertIs(result, stored)
        torch.testing.assert_close(result.current_grip, torch.tensor([-1.0]))

        reconstruction = Data(pos=torch.tensor([[1.0, 0.0, 0.0]]),
                              queries=torch.tensor([[0.0, 1.0, 0.0]]),
                              batch_queries=torch.zeros(1, dtype=torch.long),
                              batch_pos=torch.zeros(1, dtype=torch.long),
                              occupancy=torch.ones(1))
        quarter_turn = Rot.from_euler("z", 90, degrees=True)
        source = dataset.RunningDataset("unused", 1, rec=True, random_rotation=True)
        with mock.patch.object(dataset.torch, "load", return_value=reconstruction), \
             mock.patch.object(dataset, "Rot") as rotation:
            rotation.random.return_value=quarter_turn
            result = source[0]
        torch.testing.assert_close(result.pos, torch.tensor([[0.0, 1.0, 0.0]]),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(result.queries, torch.tensor([[-1.0, 0.0, 0.0]]),
                                   atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("open3d"),
                         "SKIPPED: Open3D is not installed")
    def test_open3d_outlier_and_voxel_paths(self):
        from icgs.data.preprocessing.native import downsample_pcd, remove_statistical_outliers

        points = np.vstack([np.zeros((25, 3)), np.array([[100.0, 100.0, 100.0]])])
        filtered, indices = remove_statistical_outliers(points)
        self.assertEqual(filtered.shape[1], 3)
        self.assertEqual(len(filtered), len(indices))
        self.assertEqual(downsample_pcd(points, voxel_size=0.01).shape[1], 3)


if __name__ == "__main__":
    unittest.main()
