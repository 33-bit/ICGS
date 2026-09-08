import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch


class PhysicalMemoryTests(unittest.TestCase):
    @staticmethod
    def _state(*, boundary=0, origin="real", encoder_lineage="encoder-v1", memory_lineage="memory-v1", memory=None):
        from icgs.state.physical import PhysicalState

        if memory is None:
            memory = torch.zeros(1, 2, 256)
        valid = torch.zeros(1, 128, dtype=torch.bool)
        valid[:, :3] = True
        return PhysicalState(
            X=torch.ones(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=valid,
            p=torch.zeros(1, 13),
            memory=memory,
            T_w_e=torch.eye(4)[None],
            grip=torch.zeros(1, 1),
            cached_world_cloud=torch.zeros(1, 3, 3),
            cached_world_cloud_valid=torch.tensor([[True, True, False]]),
            boundary=boundary,
            encoder_lineage=encoder_lineage,
            memory_lineage=memory_lineage,
            origin=origin,
        )

    def test_proprioception_has_translation_rot6_grip_and_gravity(self):
        from icgs.models.memories.physical import proprioception

        pose = torch.eye(4, dtype=torch.float32)[None]
        p = proprioception(
            pose,
            torch.zeros(1, 1),
            torch.tensor([[0.0, 0.0, -1.0]]),
        )

        self.assertEqual(tuple(p.shape), (1, 13))
        torch.testing.assert_close(
            p[0, 3:9],
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        torch.testing.assert_close(p[0, :3], torch.zeros(3))
        torch.testing.assert_close(p[0, 10:], torch.tensor([0.0, 0.0, -1.0]))

    def test_proprioception_rot6_uses_first_two_world_pose_columns(self):
        from icgs.models.memories.physical import proprioception

        yaw = math.pi / 2.0
        pose = torch.eye(4, dtype=torch.float64)[None]
        pose[0, :3, :3] = torch.tensor(
            [[math.cos(yaw), -math.sin(yaw), 0.0],
             [math.sin(yaw), math.cos(yaw), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

        p = proprioception(
            pose,
            torch.ones(1, 1, dtype=torch.float64),
            torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64),
        )

        torch.testing.assert_close(
            p[0, 3:9],
            torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_action_descriptor_uses_before_pose_and_planned_duration(self):
        from icgs.contracts.method import TimedCommand
        from icgs.models.memories.physical import action_descriptor

        target = np.eye(4)
        target[0, 3] = 0.10
        command = TimedCommand(target, 0, 0.1)
        before = torch.eye(4, dtype=torch.float64)[None]
        successor = before.clone()
        successor[0, 0, 3] = 0.06

        u_before = action_descriptor(before, command)
        u_successor = action_descriptor(successor, command)

        self.assertEqual(tuple(u_before.shape), (1, 8))
        self.assertAlmostEqual(u_before[0, 0].item(), 0.10, places=6)
        self.assertAlmostEqual(u_before[0, 6].item(), 0.0, places=6)
        self.assertAlmostEqual(u_before[0, 7].item(), 0.0, places=6)
        self.assertNotAlmostEqual(u_before[0, 0].item(), u_successor[0, 0].item())

    def test_action_descriptor_changes_with_before_pose_for_same_absolute_target(self):
        from icgs.contracts.method import TimedCommand
        from icgs.models.memories.physical import action_descriptor

        target = np.eye(4)
        target[0, 3] = 0.10
        command = TimedCommand(target, 1, 0.2)
        identity = torch.eye(4, dtype=torch.float64)[None]
        rotated = identity.clone()
        rotated[0, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )

        first = action_descriptor(identity, command)
        second = action_descriptor(rotated, command)

        self.assertFalse(torch.allclose(first, second))
        self.assertAlmostEqual(first[0, 6].item(), 1.0, places=6)
        self.assertAlmostEqual(first[0, 7].item(), math.log(2.0), places=6)
        np.testing.assert_allclose(command.target_w, target)

    def test_physical_feature_boundaries_reject_invalid_pose_gravity_and_grip(self):
        from icgs.contracts.method import TimedCommand
        from icgs.models.memories.physical import action_descriptor, proprioception

        identity = torch.eye(4, dtype=torch.float32)[None]
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        command = TimedCommand(np.eye(4), 0, 0.1)

        bad_pose = identity.clone()
        bad_pose[0, 0, 0] = 2.0
        for feature in (
            lambda: proprioception(torch.zeros(1, 3, 4), torch.zeros(1, 1), gravity),
            lambda: proprioception(bad_pose, torch.zeros(1, 1), gravity),
            lambda: proprioception(identity, torch.zeros(1, 2), gravity),
            lambda: proprioception(identity, torch.full((1, 1), 0.5), gravity),
            lambda: proprioception(identity, torch.zeros(1, 1), torch.zeros(1, 2)),
            lambda: proprioception(identity, torch.zeros(1, 1), torch.tensor([[float("nan"), 0.0, -1.0]])),
            lambda: action_descriptor(torch.zeros(1, 3, 4), command),
            lambda: action_descriptor(bad_pose, command),
        ):
            with self.subTest(feature=feature):
                with self.assertRaises((TypeError, ValueError)):
                    feature()

    def test_physical_memory_returns_two_layers_and_has_temporal_gradients(self):
        from icgs.models.memories.physical import PhysicalMemory

        torch.manual_seed(7)
        net = PhysicalMemory()
        memory = net(
            torch.ones(1, 128, 256),
            torch.ones(1, 128, dtype=torch.bool),
            torch.zeros(1, 13),
            torch.zeros(1, 8),
            torch.zeros(1, 2, 256),
        )

        self.assertEqual(tuple(memory.shape), (1, 2, 256))
        memory.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in net.parameters()))
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0 for parameter in net.gru1.parameters()))
        self.assertTrue(any(parameter.grad.abs().sum().item() > 0 for parameter in net.gru2.parameters()))

    def test_physical_memory_excludes_invalid_geometry_from_pooling(self):
        from icgs.models.memories.physical import PhysicalMemory

        torch.manual_seed(11)
        net = PhysicalMemory().eval()
        values = torch.ones(1, 128, 256)
        valid = torch.zeros(1, 128, dtype=torch.bool)
        valid[:, :2] = True
        changed = values.clone()
        changed[:, 2:] = 1e6

        first = net(values, valid, torch.zeros(1, 13), torch.zeros(1, 8), torch.zeros(1, 2, 256))
        second = net(changed, valid, torch.zeros(1, 13), torch.zeros(1, 8), torch.zeros(1, 2, 256))

        torch.testing.assert_close(first, second)

    def test_physical_memory_rejects_an_empty_geometry_reduction(self):
        from icgs.models.memories.physical import PhysicalMemory

        with self.assertRaisesRegex(ValueError, "valid|empty"):
            PhysicalMemory()(
                torch.zeros(1, 128, 256),
                torch.zeros(1, 128, dtype=torch.bool),
                torch.zeros(1, 13),
                torch.zeros(1, 8),
                torch.zeros(1, 2, 256),
            )

    def test_physical_tokens_keep_invalid_geometry_masked_and_add_three_valid_rows(self):
        from icgs.models.memories.physical import physical_tokens

        valid = torch.zeros(1, 128, dtype=torch.bool)
        valid[:, :3] = True
        state = SimpleNamespace(
            X=torch.ones(1, 128, 256),
            valid=valid,
            p=torch.zeros(1, 13),
            memory=torch.zeros(1, 2, 256),
        )

        tokens, token_valid = physical_tokens(state)

        self.assertEqual(tuple(tokens.shape), (1, 131, 256))
        self.assertEqual(tuple(token_valid.shape), (1, 131))
        self.assertTrue(torch.equal(token_valid[:, :128], valid))
        self.assertTrue(bool(token_valid[:, 128:].all().item()))
        self.assertTrue(bool((tokens[:, 3:128] == 0).all().item()))

    def test_next_boundary_requires_exact_successor(self):
        from icgs.state.physical import validate_next_boundary

        validate_next_boundary(3, 4)
        for boundary in (3, 5):
            with self.assertRaisesRegex(ValueError, "boundary"):
                validate_next_boundary(3, boundary)

    def test_branch_copy_marks_imagination_without_mutating_root_or_sibling(self):
        root = self._state()
        first = root.branch_copy()
        second = root.branch_copy()

        first.X[0, 0, 0] = 11.0
        first.memory[0, 0, 0] = 12.0
        first.cached_world_cloud[0, 0, 0] = 13.0

        self.assertEqual(root.origin, "real")
        self.assertEqual(first.origin, "imagined")
        self.assertEqual(second.origin, "imagined")
        self.assertEqual(root.X[0, 0, 0].item(), 1.0)
        self.assertEqual(second.X[0, 0, 0].item(), 1.0)
        self.assertEqual(root.memory[0, 0, 0].item(), 0.0)
        self.assertEqual(second.memory[0, 0, 0].item(), 0.0)
        self.assertEqual(root.cached_world_cloud[0, 0, 0].item(), 0.0)
        self.assertEqual(second.cached_world_cloud[0, 0, 0].item(), 0.0)

    def test_real_update_requires_causal_boundary_origin_and_matching_lineage(self):
        from icgs.state.physical import validate_real_update

        root = self._state()
        successor = self._state(boundary=1)
        validate_real_update(root, successor)

        for candidate in (
            self._state(boundary=0),
            self._state(boundary=2),
            self._state(boundary=1, encoder_lineage="encoder-v2"),
            self._state(boundary=1, memory_lineage="memory-v2"),
        ):
            with self.assertRaisesRegex(ValueError, "boundary|lineage"):
                validate_real_update(root, candidate)

        with self.assertRaisesRegex(ValueError, "origin|imagined"):
            validate_real_update(root.branch_copy(), successor)
        with self.assertRaisesRegex(ValueError, "origin|imagined"):
            validate_real_update(root, successor.branch_copy())

    def test_full_history_replay_matches_sequential_physical_memory_updates(self):
        from icgs.models.memories.physical import PhysicalMemory
        from icgs.state.physical import validate_real_update

        torch.manual_seed(23)
        net = PhysicalMemory().eval()
        X0 = torch.zeros(1, 128, 256)
        X1 = torch.ones(1, 128, 256)
        valid = torch.ones(1, 128, dtype=torch.bool)
        p0 = torch.zeros(1, 13)
        p1 = torch.ones(1, 13)
        u0 = torch.zeros(1, 8)
        u1 = torch.ones(1, 8)
        reset = torch.zeros(1, 2, 256)
        first_memory = net(X0, valid, p0, u0, reset)
        second_memory = net(X1, valid, p1, u1, first_memory)

        root = self._state(memory=reset)
        first = self._state(boundary=1, memory=first_memory)
        second = self._state(boundary=2, memory=second_memory)
        validate_real_update(root, first)
        validate_real_update(first, second)

        replay_first = net(X0, valid, p0, u0, reset)
        replay_second = net(X1, valid, p1, u1, replay_first)
        torch.testing.assert_close(second.memory, replay_second)


if __name__ == "__main__":
    unittest.main()
