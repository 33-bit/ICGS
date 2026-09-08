import math
from types import SimpleNamespace
import unittest

import torch


class PhysicalGeometryTests(unittest.TestCase):
    def test_se3_exp_log_round_trip_uses_metric_left_jacobian(self):
        from icgs.geometry.se3 import se3_exp, se3_log

        xi = torch.tensor([[0.02, -0.01, 0.03, 1e-7, 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(se3_log(se3_exp(xi)), xi, atol=1e-9, rtol=1e-7)

    def test_se3_round_trip_handles_noncommuting_translation_near_pi(self):
        from icgs.geometry.se3 import se3_exp, se3_log

        axis = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
        axis = axis / axis.norm()
        xi = torch.cat((torch.tensor([[0.7, -0.2, 0.4]], dtype=torch.float64),
                        (axis * (math.pi - 1e-5)).view(1, 3)), dim=-1)
        recovered = se3_log(se3_exp(xi))
        torch.testing.assert_close(recovered, xi, atol=2e-5, rtol=2e-5)

    def test_prepare_physical_cloud_is_masked_and_deterministic(self):
        from icgs.data.preprocessing.physical import prepare_physical_cloud

        points = torch.tensor([
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.03, 0.00],
            [0.00, 0.00, 0.04],
            [999., 999., 999.],
            [-999., -999., -999.],
        ], dtype=torch.float32)
        valid = torch.tensor([True, True, True, True, False, False])
        prepared = prepare_physical_cloud(points, valid)

        self.assertEqual(tuple(prepared.points.shape), (1, 2048, 3))
        self.assertEqual(tuple(prepared.anchors.shape), (1, 128, 3))
        self.assertEqual(tuple(prepared.neighbor_indices.shape), (1, 128, 32))
        self.assertEqual(int(prepared.point_valid.sum().item()), 4)
        self.assertEqual(int(prepared.anchor_valid.sum().item()), 4)
        self.assertTrue(bool(prepared.neighbor_valid[:, :4].any(dim=-1).all().item()))

        changed_invalid = points.clone()
        changed_invalid[4:] = torch.tensor([[1e6, -1e6, 3e6], [-4e6, 2e6, -5e6]])
        repeated = prepare_physical_cloud(changed_invalid, valid)
        torch.testing.assert_close(prepared.points, repeated.points)
        torch.testing.assert_close(prepared.anchors, repeated.anchors)
        self.assertTrue(torch.equal(prepared.point_valid, repeated.point_valid))
        self.assertTrue(torch.equal(prepared.anchor_valid, repeated.anchor_valid))

    def test_same_voxel_points_are_averaged_and_identity_overrides_are_rejected(self):
        from icgs.data.preprocessing.physical import PhysicalPreprocessConfig, prepare_physical_cloud

        points = torch.tensor([[0.001, 0.0, 0.0], [0.004, 0.0, 0.0], [0.020, 0.0, 0.0]])
        prepared = prepare_physical_cloud(points, torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(prepared.points[0, 0], torch.tensor([0.0025, 0.0, 0.0]))

        for override in (
            {"voxel_size_m": 0.01},
            {"num_points": 16},
            {"num_anchors": 16},
            {"num_neighbors": 8},
            {"ell0": 2.0},
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "P03|identity"):
                    PhysicalPreprocessConfig(**override)

    def test_physical_preprocessing_rejects_non_xyz_and_nonempty_invalid_inputs(self):
        from icgs.data.preprocessing.physical import prepare_physical_cloud

        with self.assertRaisesRegex(ValueError, "XYZ|shape"):
            prepare_physical_cloud(torch.zeros(2, 4), torch.ones(2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "valid"):
            prepare_physical_cloud(torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool))

    def test_yaw_augmentation_moves_cloud_pose_and_commands_together(self):
        from icgs.data.preprocessing.physical import augment_yaw

        points = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        pose = torch.eye(4)
        pose[0, 3] = 2.
        commands = torch.stack((torch.eye(4), pose))
        rotated_points, rotated_pose, rotated_commands = augment_yaw(
            points, pose, commands, math.pi / 2
        )

        torch.testing.assert_close(rotated_points, torch.tensor([[0., 1., 0.], [-1., 0., 0.]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(rotated_pose[:3, 3], torch.tensor([0., 2., 0.]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(rotated_commands[1], rotated_pose, atol=1e-6, rtol=1e-6)

    def test_masked_mean_and_encoder_zero_invalid_anchors(self):
        from icgs.models.layers.method import masked_mean

        x = torch.tensor([[[2.], [6.], [999.]]])
        m = torch.tensor([[True, True, False]])
        torch.testing.assert_close(masked_mean(x, m, dim=1), torch.tensor([[4.]]))
        with self.assertRaises(ValueError):
            masked_mean(x, torch.zeros_like(m), dim=1)

        from icgs.models.encoders.physical import PhysicalEncoder

        points = torch.zeros(1, 2048, 3)
        valid = torch.zeros(1, 2048, dtype=torch.bool)
        valid[:, 0] = True
        encoded = PhysicalEncoder()(points, valid)
        self.assertTrue(torch.isfinite(encoded.X).all().item())
        self.assertTrue((encoded.X[~encoded.anchor_valid] == 0).all().item())

    def test_encoder_ignores_invalid_padding_and_decoder_returns_masked_patches(self):
        from icgs.models.decoders.physical import PhysicalDecoder, decode_cloud
        from icgs.models.encoders.physical import PhysicalEncoder, encode_cloud

        points = torch.zeros(1, 2048, 3)
        points[:, :8, 0] = torch.arange(8, dtype=torch.float32) * 0.02
        valid = torch.zeros(1, 2048, dtype=torch.bool)
        valid[:, :8] = True
        changed = points.clone()
        changed[:, 8:] = 1000.

        torch.manual_seed(19)
        encoder = PhysicalEncoder().eval()
        encoded = encode_cloud(points, valid, encoder=encoder)
        changed_encoded = encode_cloud(changed, valid, encoder=encoder)
        torch.testing.assert_close(encoded.X, changed_encoded.X)
        torch.testing.assert_close(encoded.x, changed_encoded.x)
        self.assertTrue(torch.equal(encoded.anchor_valid, changed_encoded.anchor_valid))

        decoder = PhysicalDecoder().eval()
        decoded = decode_cloud(encoded, decoder=decoder)
        self.assertEqual(tuple(decoded.points_w.shape), (1, 128 * 16, 3))
        self.assertEqual(int(decoded.point_valid.sum().item()), int(encoded.anchor_valid.sum().item()) * 16)
        self.assertTrue(torch.isfinite(decoded.points_w).all().item())

        loss = decoded.points_w[decoded.point_valid].square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in decoder.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        encoder_gradients = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
        attention_gradients = [
            parameter.grad for name, parameter in encoder.named_parameters()
            if "geometry_blocks" in name and parameter.grad is not None
        ]
        self.assertTrue(encoder_gradients)
        self.assertTrue(attention_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in encoder_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in attention_gradients))

    def test_empty_physical_sample_is_rejected(self):
        from icgs.models.encoders.physical import PhysicalEncoder

        with self.assertRaises(ValueError):
            PhysicalEncoder()(torch.zeros(1, 2048, 3), torch.zeros(1, 2048, dtype=torch.bool))

    def test_bridge_source_accepts_only_measured_or_cached_world_cloud(self):
        from icgs.execution.observation_bridge import bridge_source

        self.assertEqual(bridge_source("real"), "measured")
        self.assertEqual(bridge_source("imagined"), "cached-decoded-world")
        with self.assertRaises(ValueError):
            bridge_source("oracle")

    def test_imagined_observation_preserves_cached_world_frame_and_achieved_pose(self):
        from icgs.execution.observation_bridge import imagined_observation

        pose = torch.eye(4).unsqueeze(0)
        pose[0, 0, 3] = 2.0
        cached_world_cloud = torch.tensor([[[1.0, 0.0, 0.0]]])
        state = SimpleNamespace(
            origin="imagined",
            cached_world_cloud=cached_world_cloud,
            cached_world_cloud_valid=torch.tensor([[True]]),
            T_w_e=pose,
            grip=torch.tensor([[1.0]]),
        )
        observation = imagined_observation(state)

        torch.testing.assert_close(observation.points, cached_world_cloud[0])
        torch.testing.assert_close(observation.T_w_e, pose[0])
        self.assertEqual(observation.grip, 1.0)
        with self.assertRaises(ValueError):
            imagined_observation(SimpleNamespace(**{**vars(state), "origin": "real"}))

    def test_imagined_bridge_rejects_scaled_rotation(self):
        from icgs.execution.observation_bridge import imagined_observation

        pose = torch.eye(4).unsqueeze(0)
        pose[0, 0, 0] = 2.0
        state = SimpleNamespace(
            origin="imagined",
            cached_world_cloud=torch.tensor([[[1.0, 0.0, 0.0]]]),
            cached_world_cloud_valid=torch.tensor([[True]]),
            T_w_e=pose,
            grip=torch.tensor([[1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "SE\\(3\\)|pose"):
            imagined_observation(state)

    def test_imagined_bridge_rejects_malformed_homogeneous_bottom_row(self):
        from icgs.execution.observation_bridge import imagined_observation

        pose = torch.eye(4).unsqueeze(0)
        pose[0, 3, 3] = 0.0
        state = SimpleNamespace(
            origin="imagined",
            cached_world_cloud=torch.tensor([[[1.0, 0.0, 0.0]]]),
            cached_world_cloud_valid=torch.tensor([[True]]),
            T_w_e=pose,
            grip=torch.tensor([[1.0]]),
        )
        with self.assertRaisesRegex(ValueError, "SE\\(3\\)|pose"):
            imagined_observation(state)

    def test_imagined_bridge_rejects_nonbinary_grip(self):
        from icgs.execution.observation_bridge import imagined_observation

        state = SimpleNamespace(
            origin="imagined",
            cached_world_cloud=torch.tensor([[[1.0, 0.0, 0.0]]]),
            cached_world_cloud_valid=torch.tensor([[True]]),
            T_w_e=torch.eye(4).unsqueeze(0),
            grip=torch.tensor([[2.0]]),
        )
        with self.assertRaisesRegex(ValueError, "grip|0|1"):
            imagined_observation(state)

    def test_imagined_bridge_removes_sentinel_padding_and_requires_batch_one(self):
        from icgs.execution.observation_bridge import imagined_observation

        pose = torch.eye(4).unsqueeze(0)
        state = SimpleNamespace(
            origin="imagined",
            cached_world_cloud=torch.tensor([[[1.0, 2.0, 3.0], [999.0, 999.0, 999.0], [4.0, 5.0, 6.0]]]),
            cached_world_cloud_valid=torch.tensor([[True, False, True]]),
            T_w_e=pose,
            grip=torch.tensor([[0.0]]),
        )
        observation = imagined_observation(state)

        self.assertEqual(tuple(observation.points.shape), (2, 3))
        torch.testing.assert_close(
            observation.points,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        with self.assertRaisesRegex(ValueError, "batch|B=1"):
            imagined_observation(SimpleNamespace(**{**vars(state), "cached_world_cloud": state.cached_world_cloud.expand(2, -1, -1),
                                                     "cached_world_cloud_valid": state.cached_world_cloud_valid.expand(2, -1)}))
        with self.assertRaisesRegex(ValueError, "mask|shape"):
            imagined_observation(SimpleNamespace(**{**vars(state), "cached_world_cloud_valid": torch.ones(1, 2, dtype=torch.bool)}))
        with self.assertRaisesRegex(ValueError, "empty|valid"):
            imagined_observation(SimpleNamespace(**{**vars(state), "cached_world_cloud_valid": torch.zeros(1, 3, dtype=torch.bool)}))
        missing_mask = vars(state).copy()
        del missing_mask["cached_world_cloud_valid"]
        with self.assertRaisesRegex(ValueError, "cloud|mask"):
            imagined_observation(SimpleNamespace(**missing_mask))
        with self.assertRaisesRegex(ValueError, "finite|empty"):
            imagined_observation(SimpleNamespace(**{**vars(state), "cached_world_cloud": torch.tensor([[[float("nan"), 0.0, 0.0],
                                                                                                           [999.0, 999.0, 999.0],
                                                                                                           [4.0, 5.0, 6.0]]])}))

    def test_decoder_rejects_empty_or_nonfinite_direct_encoded_cloud(self):
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.encoders.physical import EncodedCloud

        decoder = PhysicalDecoder()
        X = torch.zeros(1, 128, 256)
        x = torch.zeros(1, 128, 3)
        with self.assertRaisesRegex(ValueError, "valid|empty"):
            decoder(EncodedCloud(X, x, torch.zeros(1, 128, dtype=torch.bool)))

        valid = torch.zeros(1, 128, dtype=torch.bool)
        valid[:, 0] = True
        invalid_X = X.clone()
        invalid_X[:, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            decoder(EncodedCloud(invalid_X, x, valid))
        invalid_x = x.clone()
        invalid_x[:, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            decoder(EncodedCloud(X, invalid_x, valid))


if __name__ == "__main__":
    unittest.main()
