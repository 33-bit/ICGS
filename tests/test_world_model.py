import unittest
import math

import numpy as np
import torch


class WorldModelTests(unittest.TestCase):
    @staticmethod
    def _state():
        from icgs.models.memories.physical import proprioception
        from icgs.state.physical import PhysicalState

        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid = torch.zeros(1, 128, dtype=torch.bool)
        valid[:, :3] = True
        return PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=valid,
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 3, 3),
            cached_world_cloud_valid=torch.tensor([[True, False, False]]),
            boundary=0,
            encoder_lineage="encoder-v1",
            memory_lineage="memory-v1",
        )

    def test_validate_head_id_accepts_only_three_integer_heads(self):
        from icgs.models.dynamics.physical import validate_head_id

        for head in range(3):
            validate_head_id(head)
        for head in (-1, 3, True):
            with self.assertRaises(ValueError):
                validate_head_id(head)

    def test_physical_dynamics_returns_geometry_and_pose_grip_residuals(self):
        from icgs.configuration.method import MethodConfig
        from icgs.models.dynamics.physical import PhysicalDynamics

        torch.manual_seed(101)
        config = MethodConfig()
        model = PhysicalDynamics(config).eval()
        tokens = torch.randn(2, 131, 256)
        valid = torch.ones(2, 131, dtype=torch.bool)
        valid[1, 120:128] = False
        command = torch.randn(2, 8)

        geometry_delta, pose_grip = model(tokens, valid, command, head_id=1)

        self.assertEqual(tuple(geometry_delta.shape), (2, 128, 259))
        self.assertEqual(tuple(pose_grip.shape), (2, 7))
        self.assertTrue(torch.isfinite(geometry_delta).all().item())
        self.assertTrue(torch.isfinite(pose_grip).all().item())

    def test_invalid_physical_rows_cannot_change_dynamics_prediction(self):
        from icgs.configuration.method import MethodConfig
        from icgs.models.dynamics.physical import PhysicalDynamics

        torch.manual_seed(102)
        model = PhysicalDynamics(MethodConfig()).eval()
        tokens = torch.randn(1, 131, 256)
        valid = torch.ones(1, 131, dtype=torch.bool)
        valid[:, 4:128] = False
        changed = tokens.clone()
        changed[:, 4:128] = 1e6
        command = torch.randn(1, 8)

        first = model(tokens, valid, command, head_id=0)
        second = model(changed, valid, command, head_id=0)

        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_heads_have_independent_parameters_and_configured_depth(self):
        from icgs.configuration.method import MethodConfig
        from icgs.models.dynamics.physical import PhysicalDynamics

        config = MethodConfig.from_dict({
            "dynamics": {"transformer_layers": 1, "residual_init_std": 0.002},
        })
        model = PhysicalDynamics(config=config)

        self.assertEqual(len(model.trunk), 1)
        self.assertTrue(
            set(map(id, model.geometry_heads[0].parameters())).isdisjoint(
                map(id, model.geometry_heads[1].parameters())
            )
        )
        self.assertTrue(
            set(map(id, model.pose_heads[0].parameters())).isdisjoint(
                map(id, model.pose_heads[2].parameters())
            )
        )
        self.assertAlmostEqual(model.residual_init_std, 0.002)

    def test_forward_is_physical_only_and_has_no_task_context_parameter(self):
        import inspect

        from icgs.models.dynamics.physical import PhysicalDynamics

        self.assertEqual(
            tuple(inspect.signature(PhysicalDynamics.forward).parameters),
            ("self", "S", "valid", "u", "head_id"),
        )

    def test_bound_predict_step_has_literal_public_signature(self):
        import inspect

        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.rollout.physical import bind_predict_step

        config = MethodConfig()
        step = bind_predict_step(
            _FixedDynamics(),
            _IdentityEncoder(),
            _IdentityDecoder(),
            _MemorySpy(),
            config=config,
        )

        self.assertEqual(
            tuple(inspect.signature(step).parameters),
            ("state", "command", "head_id"),
        )

    def test_predict_step_decodes_and_reencodes_once_and_keeps_head(self):
        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.rollout.physical import bind_predict_step
        from icgs.contracts.method import TimedCommand

        config = MethodConfig()
        decoder = _Counting(_IdentityDecoder())
        encoder = _Counting(_IdentityEncoder())
        memory = _MemorySpy()
        step = bind_predict_step(_FixedDynamics(), encoder, decoder, memory, config=config)
        target = np.eye(4)
        target[0, 3] = 0.10

        prediction = step(self._state(), TimedCommand(target, 0, 0.1), head_id=2)

        self.assertEqual(decoder.calls, 1)
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(prediction.head_id, 2)
        self.assertEqual(prediction.next_state.origin, "imagined")
        self.assertEqual(prediction.next_state.boundary, 1)

    def test_predict_step_describes_command_from_before_pose(self):
        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.rollout.physical import bind_predict_step
        from icgs.contracts.method import TimedCommand

        config = MethodConfig()
        memory = _MemorySpy()
        step = bind_predict_step(
            _FixedDynamics(achieved_translation=0.06),
            _IdentityEncoder(),
            _IdentityDecoder(),
            memory,
            config=config,
        )
        target = np.eye(4)
        target[0, 3] = 0.10

        prediction = step(self._state(), TimedCommand(target, 0, 0.1), head_id=0)

        self.assertAlmostEqual(memory.previous_u[0, 0].item(), 0.10, places=6)
        self.assertAlmostEqual(prediction.next_state.T_w_e[0, 0, 3].item(), 0.06, places=6)

    def test_closed_rollout_recomputes_descriptor_from_each_achieved_pose(self):
        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.contracts.method import TimedCommand

        config = MethodConfig()
        memory = _MemorySpy()
        rollout = PhysicalRollout(
            _FixedDynamics(achieved_translation=0.06),
            _IdentityEncoder(),
            _IdentityDecoder(),
            memory,
            config=config,
        )
        target = np.eye(4)
        target[0, 3] = 0.10
        command = TimedCommand(target, 0, 0.1)

        predictions = rollout.rollout(self._state(), (command, command), head_id=1)

        self.assertEqual(tuple(prediction.head_id for prediction in predictions), (1, 1))
        self.assertEqual(tuple(prediction.next_state.boundary for prediction in predictions), (1, 2))
        self.assertEqual(len(memory.previous_us), 2)
        self.assertAlmostEqual(memory.previous_us[0][0, 0].item(), 0.10, places=6)
        self.assertAlmostEqual(memory.previous_us[1][0, 0].item(), 0.04, places=6)

    def test_predict_step_rejects_nonfinite_dynamics_outputs(self):
        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.rollout.physical import bind_predict_step
        from icgs.contracts.method import TimedCommand

        config = MethodConfig()
        step = bind_predict_step(
            _NonfiniteDynamics(),
            _IdentityEncoder(),
            _IdentityDecoder(),
            _MemorySpy(),
            config=config,
        )

        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            step(self._state(), TimedCommand(np.eye(4), 0, 0.1), head_id=0)

    def test_real_rollout_grip_bce_preserves_pose_grip_gradient(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import TimedCommand
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory

        config = MethodConfig()
        dynamics = PhysicalDynamics(config).eval()
        encoder = PhysicalEncoder(method_config=config).eval()
        decoder = PhysicalDecoder(method_config=config).eval()
        memory = PhysicalMemory(config=config).eval()
        rollout = PhysicalRollout(dynamics, encoder, decoder, memory, config=config)
        prediction = rollout.predict_step(
            self._state(), TimedCommand(np.eye(4), 0, 0.1), head_id=0
        )

        self.assertIsInstance(prediction.grip_logits, torch.Tensor)
        target_grip = 1.0 - float(prediction.next_state.grip[0, 0].item())
        cached_valid = prediction.next_state.cached_world_cloud_valid[0]
        target_points = prediction.next_state.cached_world_cloud[0][cached_valid].detach().cpu().numpy()
        target_pose = prediction.next_state.T_w_e[0].detach().cpu().numpy()
        target = _executed_transition(target_points, target_pose, target_grip, boundary=1)

        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)
        self.assertGreater(float(loss.item()), 0.0)
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.grip_logits,
            torch.tensor([[target_grip]], dtype=prediction.grip_logits.dtype),
        )
        expected_rotation = _rotation_clip_floor(config, dtype=loss.dtype)
        torch.testing.assert_close(
            loss,
            expected_bce + torch.tensor(expected_rotation, dtype=loss.dtype),
            rtol=1e-5,
            atol=1e-7,
        )
        loss.backward()

        row_gradient = dynamics.pose_heads[0].output.weight.grad[6]
        self.assertTrue(torch.isfinite(row_gradient).all().item())
        self.assertGreater(float(row_gradient.abs().sum().item()), 0.0)

    def test_real_two_step_rollout_loss_reaches_earlier_prediction_through_frozen_bridge(self):
        from icgs.algorithms.objectives.physical import rollout_loss
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import TimedCommand
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory

        config = MethodConfig.from_dict({"losses": {"grip_weight": 0.0}})
        dynamics = PhysicalDynamics(config).eval()
        encoder = PhysicalEncoder(method_config=config).eval()
        decoder = PhysicalDecoder(method_config=config).eval()
        memory = PhysicalMemory(config=config).eval()
        for module in (encoder, decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        rollout = PhysicalRollout(dynamics, encoder, decoder, memory, config=config)
        command = TimedCommand(np.eye(4), 0, 0.1)
        predictions = rollout.rollout(self._state(), (command, command), head_id=0)
        earlier_features = predictions[0].next_state.X
        earlier_features.retain_grad()

        targets = []
        for step_index, prediction in enumerate(predictions, start=1):
            valid = prediction.next_state.cached_world_cloud_valid[0]
            points = prediction.next_state.cached_world_cloud[0][valid].detach().cpu().numpy()
            if step_index == 2:
                points = points.copy()
                points[0, 0] += 0.01
            pose = prediction.next_state.T_w_e[0].detach().cpu().numpy()
            grip = float(prediction.next_state.grip[0, 0].item())
            targets.append(_executed_transition(points, pose, grip, boundary=step_index))

        loss = rollout_loss(
            [[list(predictions), [], []]],
            [targets],
            torch.tensor([[True, False, False]]),
            valid=torch.tensor([[True, True]]),
            encoder_decoder_trainable=False,
            config=config,
        )
        loss.backward()

        self.assertIsNotNone(earlier_features.grad)
        self.assertTrue(torch.isfinite(earlier_features.grad).all().item())
        self.assertGreater(float(earlier_features.grad.abs().sum().item()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))

    def test_bootstrap_mask_is_deterministic_and_covers_each_episode(self):
        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.objectives.physical import bootstrap_mask

        config = MethodConfig()
        mask = bootstrap_mask(["episode-a", "episode-b"], seed=9, config=config)

        self.assertEqual(tuple(mask.shape), (2, 3))
        self.assertTrue(mask.any(dim=1).all().item())
        self.assertTrue((mask == bootstrap_mask(["episode-a", "episode-b"], seed=9, config=config)).all().item())

    def test_physical_loss_is_finite_and_transmits_prediction_gradients(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig()
        logits = torch.zeros(1, 1, requires_grad=True)
        prediction = PhysicalPrediction(self._state(), logits, 0)
        target = _executed_transition(
            np.array([[0.0, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )

        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_training_rotation_uses_positive_clip_floor_below_threshold(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig.from_dict({"losses": {"grip_weight": 0.0}})
        expected = _rotation_clip_floor(config)
        for angle in (0.0, 0.0005, 0.001):
            state = self._state()
            state.T_w_e.requires_grad_(True)
            prediction = PhysicalPrediction(state, torch.zeros(1, 1), 0)
            target = _executed_transition(
                np.array([[0.0, 0.0, 0.0]]), _z_rotation_pose(angle), 0, boundary=1
            )

            loss = physical_loss(prediction, target, torch.tensor([True]), config=config)
            torch.testing.assert_close(
                loss,
                torch.tensor(expected, dtype=loss.dtype),
                rtol=1e-5,
                atol=1e-7,
            )
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertIsNotNone(state.T_w_e.grad)
            self.assertTrue(torch.isfinite(state.T_w_e.grad).all().item())

    def test_training_rotation_has_finite_nonzero_gradient_outside_clip_threshold(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig.from_dict({"losses": {"grip_weight": 0.0}})
        state = self._state()
        state.T_w_e.requires_grad_(True)
        prediction = PhysicalPrediction(state, torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.0, 0.0, 0.0]]), _z_rotation_pose(0.005), 0, boundary=1
        )

        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        self.assertIsNotNone(state.T_w_e.grad)
        self.assertTrue(torch.isfinite(state.T_w_e.grad).all().item())
        self.assertGreater(float(state.T_w_e.grad.abs().sum().item()), 0.0)

    def test_training_rotation_is_finite_near_pi(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig.from_dict({"losses": {"grip_weight": 0.0}})
        state = self._state()
        state.T_w_e.requires_grad_(True)
        prediction = PhysicalPrediction(state, torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.0, 0.0, 0.0]]), _z_rotation_pose(math.pi - 0.0001), 0, boundary=1
        )

        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        self.assertIsNotNone(state.T_w_e.grad)
        self.assertTrue(torch.isfinite(state.T_w_e.grad).all().item())

    def test_physical_loss_accepts_executed_transition_target(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig()
        logits = torch.zeros(1, 1, requires_grad=True)
        prediction = PhysicalPrediction(self._state(), logits, 0)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 1, boundary=1
        )

        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_rollout_loss_detaches_executed_teacher_targets(self):
        from icgs.algorithms.objectives.physical import rollout_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig()
        logits = torch.zeros(1, 1, requires_grad=True)
        prediction = PhysicalPrediction(self._state(), logits, 0)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )

        loss = rollout_loss(
            [[[prediction], [], []]],
            [[target]],
            torch.tensor([[True, False, False]]),
            valid=torch.tensor([[True]]),
            encoder_decoder_trainable=False,
            config=config,
        )
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_rollout_loss_is_invariant_to_masked_padding(self):
        from icgs.algorithms.objectives.physical import rollout_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig.from_dict({"losses": {"grip_weight": 0.0}})
        prediction = PhysicalPrediction(self._state(), torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )
        one_step = rollout_loss(
            [[[prediction], [], []]],
            [[target]],
            torch.tensor([[True, False, False]]),
            valid=torch.tensor([[True]]),
            encoder_decoder_trainable=False,
            config=config,
        )
        padded_step = rollout_loss(
            [[[prediction, prediction], [], []]],
            [[target, target]],
            torch.tensor([[True, False, False]]),
            valid=torch.tensor([[True, False]]),
            encoder_decoder_trainable=False,
            config=config,
        )

        expected_rotation = _rotation_clip_floor(config, dtype=one_step.dtype)
        torch.testing.assert_close(
            one_step,
            torch.tensor(2.0 + expected_rotation, dtype=one_step.dtype),
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(one_step, padded_step)

    def test_rollout_loss_rejects_invalid_supervision_mask_shapes_and_empty_supervision(self):
        from icgs.algorithms.objectives.physical import rollout_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig()
        prediction = PhysicalPrediction(self._state(), torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            rollout_loss(
                [[[prediction], [], []]], [[target]], torch.tensor([[True, False, False]]),
                valid=torch.tensor([True]),
                encoder_decoder_trainable=False, config=config,
            )
        with self.assertRaisesRegex(ValueError, "supervised"):
            rollout_loss(
                [[[prediction], [], []]], [[target]], torch.tensor([[True, False, False]]),
                valid=torch.tensor([[False]]),
                encoder_decoder_trainable=False, config=config,
            )

    def test_rollout_loss_rejects_selected_prediction_head_mismatch(self):
        from icgs.algorithms.objectives.physical import rollout_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig()
        prediction = PhysicalPrediction(self._state(), torch.zeros(1, 1), 1)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )
        with self.assertRaisesRegex(ValueError, "head"):
            rollout_loss(
                [[[prediction], [], []]], [[target]], torch.tensor([[True, False, False]]),
                valid=torch.tensor([[True]]),
                encoder_decoder_trainable=False, config=config,
            )

    def test_physical_loss_uses_sum_of_directed_cd_terms(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction

        config = MethodConfig.from_dict({"losses": {"grip_weight": 0.0}})
        prediction = PhysicalPrediction(self._state(), torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )

        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)

        expected_rotation = _rotation_clip_floor(config, dtype=loss.dtype)
        self.assertAlmostEqual(float(loss.item()) - expected_rotation, 2.0, places=6)

    def test_physical_loss_rejects_valid_samples_without_geometry(self):
        from icgs.algorithms.objectives.physical import physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction
        from icgs.contracts.records import Observation

        config = MethodConfig()
        invalid_state = self._state()
        invalid_state.cached_world_cloud_valid.zero_()
        prediction = PhysicalPrediction(invalid_state, torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.0, 0.0, 0.0]]), np.eye(4), 0, boundary=1
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            physical_loss(prediction, target, torch.tensor([True]), config=config)

        empty_target = Observation(np.empty((0, 3)), np.eye(4), 0.0)
        valid_prediction = PhysicalPrediction(self._state(), torch.zeros(1, 1), 0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            physical_loss(valid_prediction, empty_target, torch.tensor([True]), config=config)

    def test_physical_dynamics_rejects_mixed_batch_without_geometry_anchor(self):
        from icgs.configuration.method import MethodConfig
        from icgs.models.dynamics.physical import PhysicalDynamics

        config = MethodConfig()
        dimensions = config.derived_dimensions()
        model = PhysicalDynamics(config).eval()
        tokens = torch.zeros(2, dimensions["physical_tokens"], config.geometry.width)
        valid = torch.ones(2, dimensions["physical_tokens"], dtype=torch.bool)
        valid[1, :config.geometry.num_anchors] = False
        command = torch.zeros(2, config.memory.descriptor_dim)

        with self.assertRaisesRegex(ValueError, "geometry"):
            model(tokens, valid, command, head_id=0)

    def test_p07_boundaries_require_explicit_method_config(self):
        from icgs.algorithms.objectives.physical import bootstrap_mask, physical_loss, rollout_loss
        from icgs.algorithms.rollout.physical import PhysicalRollout, bind_predict_step
        from icgs.models.dynamics.physical import PhysicalDynamics

        with self.assertRaises(TypeError):
            PhysicalDynamics()
        with self.assertRaises(TypeError):
            PhysicalRollout(_FixedDynamics(), _IdentityEncoder(), _IdentityDecoder(), _MemorySpy())
        with self.assertRaises(TypeError):
            bind_predict_step(_FixedDynamics(), _IdentityEncoder(), _IdentityDecoder(), _MemorySpy())
        with self.assertRaises(TypeError):
            bootstrap_mask(["episode-a"], seed=9)
        with self.assertRaises(TypeError):
            physical_loss(None, None, torch.tensor([True]))
        with self.assertRaises(TypeError):
            rollout_loss([], [], torch.zeros(0, 3, dtype=torch.bool))

    def test_nondefault_p07_config_reaches_bootstrap_depth_loss_and_weights(self):
        from icgs.algorithms.objectives.physical import bootstrap_mask, physical_loss
        from icgs.configuration.method import MethodConfig
        from icgs.contracts.method import PhysicalPrediction
        from icgs.models.dynamics.physical import PhysicalDynamics

        config = MethodConfig.from_dict({
            "dynamics": {
                "transformer_layers": 1,
                "residual_init_std": 0.002,
                "bootstrap_probability": 1.0,
            },
            "losses": {"cloud_scale_m": 0.02, "grip_weight": 2.0},
        })
        model = PhysicalDynamics(config)
        mask = bootstrap_mask(["episode-a"], seed=9, config=config)
        prediction = PhysicalPrediction(self._state(), torch.zeros(1, 1), 0)
        target = _executed_transition(
            np.array([[0.01, 0.0, 0.0]]), np.eye(4), 1, boundary=1
        )
        loss = physical_loss(prediction, target, torch.tensor([True]), config=config)

        self.assertEqual(len(model.trunk), 1)
        self.assertAlmostEqual(model.residual_init_std, 0.002)
        self.assertTrue(mask.all().item())
        expected_rotation = _rotation_clip_floor(config, dtype=loss.dtype)
        self.assertAlmostEqual(
            float(loss.item()), 0.5 + 2.0 * math.log(2.0) + expected_rotation, places=6
        )

    def test_frozen_physical_encoder_decoder_keep_input_gradient_path(self):
        from icgs.configuration.method import MethodConfig
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.contracts.method import TimedCommand
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory

        config = MethodConfig()
        dynamics = PhysicalDynamics(config).eval()
        encoder = PhysicalEncoder(method_config=config).eval()
        decoder = PhysicalDecoder(method_config=config).eval()
        memory = PhysicalMemory(config=config).eval()
        for module in (encoder, decoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        state = self._state()
        state.X.requires_grad_(True)
        prediction = PhysicalRollout(
            dynamics, encoder, decoder, memory, config=config
        ).predict_step(
            state,
            TimedCommand(np.eye(4), 0, 0.1),
            head_id=0,
        )
        (prediction.next_state.X.sum() + prediction.next_state.memory.sum()).backward()

        self.assertIsNotNone(state.X.grad)
        self.assertTrue(torch.isfinite(state.X.grad).all().item())
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in decoder.parameters()))


def _executed_transition(points, pose, grip, *, boundary):
    from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
    from icgs.contracts.records import Observation

    points = np.array(points, dtype=np.float64, copy=True)
    pose = np.array(pose, dtype=np.float64, copy=True)
    before = TimedObservation(
        Observation(points.copy(), pose.copy(), float(grip)),
        boundary - 1,
        float(boundary - 1) * 0.1,
        float(boundary - 1) * 0.1,
        "sensor",
    )
    after = TimedObservation(
        Observation(points.copy(), pose.copy(), float(grip)),
        boundary,
        float(boundary) * 0.1,
        float(boundary) * 0.1,
        "sensor",
    )
    command = TimedCommand(pose, int(grip), 0.1)
    return ExecutedTransition(before, after, command, 0.1, 1, "ok")


def _z_rotation_pose(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    return pose


def _rotation_clip_floor(config, *, dtype=torch.float32):
    clipped_cosine = torch.tensor(
        1.0 - config.numerics.rotation_training_clip_margin, dtype=dtype
    )
    scale = math.radians(config.losses.rotation_scale_deg)
    return float((torch.acos(clipped_cosine).square() / (scale * scale)).item())


class _FixedDynamics:
    def __init__(self, achieved_translation=0.0):
        self.achieved_translation = achieved_translation

    def __call__(self, S, valid, u, head_id):
        del valid, u, head_id
        geometry_delta = torch.zeros(S.shape[0], 128, 259, dtype=S.dtype, device=S.device)
        pose_grip = torch.zeros(S.shape[0], 7, dtype=S.dtype, device=S.device)
        pose_grip[:, 0] = self.achieved_translation
        return geometry_delta, pose_grip


class _IdentityDecoder:
    def __call__(self, encoded):
        from icgs.models.decoders.physical import DecodedCloud

        return DecodedCloud(encoded.x, encoded.anchor_valid)


class _IdentityEncoder:
    def __call__(self, points_w, point_valid):
        from icgs.models.encoders.physical import EncodedCloud

        return EncodedCloud(
            X=torch.zeros(
                points_w.shape[0], 128, 256, dtype=points_w.dtype, device=points_w.device
            ),
            x=points_w,
            anchor_valid=point_valid,
        )


class _Counting:
    def __init__(self, collaborator):
        self.collaborator = collaborator
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.collaborator(*args, **kwargs)


class _MemorySpy:
    def __init__(self):
        from icgs.models.memories.physical import PhysicalMemory

        self.inner = PhysicalMemory().eval()
        self.previous_u = None
        self.previous_us = []

    def physical_tokens(self, state):
        return self.inner.physical_tokens(state)

    def __call__(self, X, valid, p, previous_u, memory):
        self.previous_u = previous_u.detach().clone()
        self.previous_us.append(self.previous_u)
        return self.inner(X, valid, p, previous_u, memory)


class _NonfiniteDynamics:
    def __call__(self, S, valid, u, head_id):
        del valid, u, head_id
        geometry_delta = torch.zeros(S.shape[0], 128, 259, dtype=S.dtype, device=S.device)
        geometry_delta[:, 0, 0] = float("nan")
        return geometry_delta, torch.zeros(S.shape[0], 7, dtype=S.dtype, device=S.device)


if __name__ == "__main__":
    unittest.main()
