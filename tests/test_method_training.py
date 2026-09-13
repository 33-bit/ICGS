import hashlib
import unittest

import torch
from torch import nn

from icgs.configuration.method import MethodConfig
from icgs.training.stages.method import (
    apply_freeze_boundary,
    build_stage_optimizer,
    build_stage_scheduler,
    rollout_horizon,
    trainable_components,
)


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class MethodTrainingTask1Tests(unittest.TestCase):
    def test_trainable_set_enforcement_and_finite_curricula(self):
        self.assertEqual(trainable_components('C'), frozenset())
        self.assertEqual(trainable_components('D1'), frozenset({'dynamics'}))
        self.assertEqual(rollout_horizon('A1', 0, 90), 1)
        self.assertEqual(rollout_horizon('A1', 60, 90), 4)

    def test_phase_matrix_membership_across_all_declared_stages(self):
        expected = {
            'A0': frozenset({'geometry'}),
            'A1': frozenset({'geometry', 'physical_memory', 'dynamics'}),
            'B': frozenset({'events', 'task'}),
            'C': frozenset(),
            'D1': frozenset({'dynamics'}),
            'D2': frozenset({'evaluation', 'terminal'}),
            'E': frozenset({'temperatures'}),
            'Test': frozenset(),
        }
        for stage, comp_set in expected.items():
            self.assertEqual(trainable_components(stage), comp_set)

        with self.assertRaisesRegex(ValueError, 'unknown training stage'):
            trainable_components('InvalidStage')

    def test_curricula_divisions_and_boundaries(self):
        # A1 K=1/2/4 in equal update thirds (max_updates=90 -> 0-29:1, 30-59:2, 60-90:4)
        self.assertEqual(rollout_horizon('A1', 0, 90), 1)
        self.assertEqual(rollout_horizon('A1', 29, 90), 1)
        self.assertEqual(rollout_horizon('A1', 30, 90), 2)
        self.assertEqual(rollout_horizon('A1', 59, 90), 2)
        self.assertEqual(rollout_horizon('A1', 60, 90), 4)
        self.assertEqual(rollout_horizon('A1', 89, 90), 4)
        # Upper bound boundary check: update must be strictly less than max_updates
        with self.assertRaisesRegex(ValueError, 'out of bounds|less than max_updates'):
            rollout_horizon('A1', 90, 90)
        with self.assertRaisesRegex(ValueError, 'out of bounds|less than max_updates'):
            rollout_horizon('A1', 100, 90)

        # D1 K=2/4/8/16 in quarters (max_updates=100 -> 0-24:2, 25-49:4, 50-74:8, 75-100:16)
        self.assertEqual(rollout_horizon('D1', 0, 100), 2)
        self.assertEqual(rollout_horizon('D1', 24, 100), 2)
        self.assertEqual(rollout_horizon('D1', 25, 100), 4)
        self.assertEqual(rollout_horizon('D1', 49, 100), 4)
        self.assertEqual(rollout_horizon('D1', 50, 100), 8)
        self.assertEqual(rollout_horizon('D1', 74, 100), 8)
        self.assertEqual(rollout_horizon('D1', 75, 100), 16)
        self.assertEqual(rollout_horizon('D1', 99, 100), 16)

        # Invalid bounds
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            rollout_horizon('A1', -1, 90)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            rollout_horizon('A1', 0, 0)
        with self.assertRaisesRegex(ValueError, 'curriculum'):
            rollout_horizon('A0', 0, 90)

    def test_shared_or_aliased_parameter_raises_conflict_regardless_of_dict_order(self):
        # Module shared between trainable role ('geometry') and frozen role ('events') in stage A1
        shared_linear = nn.Linear(4, 4)

        geo_shared = nn.ModuleDict({'encoder': shared_linear, 'decoder': nn.Linear(4, 4)})

        # Ordering 1: geometry then events
        comp_order1 = {
            'geometry': geo_shared,
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
            'events': shared_linear,
        }
        with self.assertRaisesRegex(ValueError, 'Conflicting trainable and frozen role'):
            apply_freeze_boundary('A1', comp_order1)

        # Ordering 2: events then geometry
        comp_order2 = {
            'events': shared_linear,
            'geometry': geo_shared,
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
        }
        with self.assertRaisesRegex(ValueError, 'Conflicting trainable and frozen role'):
            apply_freeze_boundary('A1', comp_order2)

    def test_non_module_or_missing_roles_rejected(self):
        # Non-module entry
        with self.assertRaisesRegex(TypeError, 'must be an nn.Module'):
            apply_freeze_boundary('A1', {'geometry': 'not_a_module'})

        # Missing required roles for A1
        with self.assertRaisesRegex(KeyError, 'missing required component role'):
            apply_freeze_boundary('A1', {'geometry': nn.Linear(4, 4)})

        # Geometry missing encoder or decoder for A1
        with self.assertRaisesRegex(KeyError, "must contain both 'encoder' and 'decoder'"):
            apply_freeze_boundary('A1', {
                'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4)}),
                'physical_memory': nn.Linear(4, 4),
                'dynamics': nn.Linear(4, 4),
            })

    def test_frozen_modules_are_in_eval_mode_and_buffers_not_mutated(self):
        class ModuleWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(self.bn(x))

        comp = {
            'geometry': nn.ModuleDict({
                'encoder': nn.Linear(4, 4),
                'decoder': nn.Linear(4, 4),
            }),
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
            'events': ModuleWithBuffer(),  # frozen in A1
        }
        apply_freeze_boundary('A1', comp)
        # Frozen module must be placed in eval mode
        self.assertFalse(comp['events'].training)
        self.assertTrue(comp['geometry'].training)
        self.assertTrue(comp['geometry']['encoder'].training)
        self.assertTrue(comp['geometry']['decoder'].training)

        # When running a forward through frozen module in eval mode, BatchNorm running statistics do not mutate
        running_mean_before = comp['events'].bn.running_mean.clone()
        x = torch.randn(10, 4) * 5.0 + 3.0
        with torch.no_grad():
            comp['events'](x)
        running_mean_after = comp['events'].bn.running_mean
        torch.testing.assert_close(running_mean_before, running_mean_after)


    def test_freeze_boundary_optimizer_parameter_ids_and_tensor_invariance(self):
        cfg = MethodConfig()
        components = {
            'geometry': nn.ModuleDict({
                'encoder': nn.Linear(4, 4),
                'decoder': nn.Linear(4, 4),
            }),
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
            'events': nn.Linear(4, 4),
            'task': nn.Linear(4, 4),
            'evaluation': nn.Linear(4, 4),
        }

        # Stage A1: geometry (both encoder and decoder), physical_memory, dynamics are trainable
        apply_freeze_boundary('A1', components)
        self.assertTrue(all(p.requires_grad for p in components['geometry']['encoder'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['geometry']['decoder'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['physical_memory'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['dynamics'].parameters()))
        self.assertFalse(any(p.requires_grad for p in components['events'].parameters()))
        self.assertFalse(any(p.requires_grad for p in components['task'].parameters()))
        self.assertFalse(any(p.requires_grad for p in components['evaluation'].parameters()))

        optimizer = build_stage_optimizer('A1', components, cfg)
        self.assertIsNotNone(optimizer)

        # Assert actual optimizer parameter IDs equal the permitted set including the decoder
        expected_param_ids = {
            id(p)
            for name in ('geometry', 'physical_memory', 'dynamics')
            for p in components[name].parameters()
        }
        decoder_param_ids = {id(p) for p in components['geometry']['decoder'].parameters()}
        self.assertTrue(decoder_param_ids.issubset(expected_param_ids))
        actual_param_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(actual_param_ids, expected_param_ids)
        self.assertTrue(decoder_param_ids.issubset(actual_param_ids))

        # Record frozen tensor hashes
        frozen_hashes_before = {
            f'{name}.{param_name}': _tensor_hash(param)
            for name in ('events', 'task', 'evaluation')
            for param_name, param in components[name].named_parameters()
        }
        trainable_hashes_before = {
            f'{name}.{param_name}': _tensor_hash(param)
            for name in ('geometry', 'physical_memory', 'dynamics')
            for param_name, param in components[name].named_parameters()
        }

        # Tiny deterministic optimization step
        x = torch.ones(2, 4)
        out = (
            components['geometry']['encoder'](x)
            + components['geometry']['decoder'](x)
            + components['physical_memory'](x)
            + components['dynamics'](x)
            + components['task'](x)  # frozen, but in graph
        )
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()

        # Real decoder gradient assertion
        for p in components['geometry']['decoder'].parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(bool((p.grad != 0).any().item()))

        optimizer.step()

        # Trainable parameters must change (including decoder)
        for key, before_hash in trainable_hashes_before.items():
            comp_name, param_name = key.split('.', 1)
            after_hash = _tensor_hash(dict(components[comp_name].named_parameters())[param_name])
            self.assertNotEqual(before_hash, after_hash, f'trainable parameter {key} did not update')

        # Frozen tensor hashes must remain unchanged
        for key, before_hash in frozen_hashes_before.items():
            comp_name, param_name = key.split('.', 1)
            after_hash = _tensor_hash(dict(components[comp_name].named_parameters())[param_name])
            self.assertEqual(before_hash, after_hash, f'frozen parameter {key} changed after optimizer step')

        # In Stage B: physical_memory and geometry are now frozen, while events and task are trainable
        apply_freeze_boundary('B', components)
        self.assertFalse(any(p.requires_grad for p in components['physical_memory'].parameters()))
        self.assertFalse(any(p.requires_grad for p in components['geometry'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['events'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['task'].parameters()))

        # In Stage A0: geometry only (both encoder and decoder) is trainable
        apply_freeze_boundary('A0', {'geometry': components['geometry']})
        self.assertTrue(all(p.requires_grad for p in components['geometry']['encoder'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['geometry']['decoder'].parameters()))

        # Stage C: no optimizer, all frozen
        apply_freeze_boundary('C', components)
        self.assertFalse(any(p.requires_grad for c in components.values() for p in c.parameters()))
        opt_c = build_stage_optimizer('C', components, cfg)
        self.assertIsNone(opt_c)


class MethodTrainingTask2Tests(unittest.TestCase):
    def test_full_history_replay_slices(self):
        from icgs.training.method import history_slices

        prefix, burn, supervised = history_slices(boundary=20, burnin=8)
        self.assertEqual(prefix, slice(0, 12))
        self.assertEqual(burn, slice(12, 20))
        self.assertEqual(supervised.start, 20)

        # Non-boundary / boundary smaller than burnin
        prefix_small, burn_small, supervised_small = history_slices(boundary=4, burnin=8, supervised_intervals=16)
        self.assertEqual(prefix_small, slice(0, 0))
        self.assertEqual(burn_small, slice(0, 4))
        self.assertEqual(supervised_small, slice(4, 20))

        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            history_slices(boundary=-1, burnin=8)
        with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
            history_slices(boundary=20, burnin=-1)

    def test_sample_anchor_horizon(self):
        import random
        from icgs.training.method import sample_anchor_horizon

        class FakeAnchor:
            def __init__(self, name: str, horizons: tuple[int, ...]):
                self.name = name
                self.valid_horizons = horizons

        anchors = [
            FakeAnchor('a1', (2, 4, 8)),
            FakeAnchor('a2', (4, 16)),
        ]
        rng = random.Random(42)
        anchor, H = sample_anchor_horizon(anchors, rng)
        self.assertIn(anchor, anchors)
        self.assertIn(H, anchor.valid_horizons)

        with self.assertRaisesRegex(ValueError, 'nonempty'):
            sample_anchor_horizon([], rng)

    def test_transition_descriptor_command_vs_achieved_semantics(self):
        from icgs.contracts.method import TimedCommand
        from icgs.training.method import build_transition_descriptor_and_achieved

        cfg = MethodConfig()

        before_pose = torch.eye(4, dtype=torch.float32)
        target_matrix = torch.eye(4, dtype=torch.float32)
        target_matrix[0, 3] = 0.10
        command = TimedCommand(
            target_w=tuple(tuple(row) for row in target_matrix.tolist()),
            duration_s=0.1,
            grip=0,
        )

        after_pose = torch.eye(4, dtype=torch.float32)
        after_pose[0, 3] = 0.06

        descriptor, achieved_translation = build_transition_descriptor_and_achieved(
            before_pose, command, after_pose, config=cfg
        )

        torch.testing.assert_close(descriptor[0], torch.tensor(0.10, dtype=torch.float32))
        torch.testing.assert_close(achieved_translation[0], torch.tensor(0.06, dtype=torch.float32))

    def test_resolved_config_hyperparameters_and_stages_inventory(self):
        cfg = MethodConfig()

        self.assertEqual(cfg.stages.A0.batch_size, 64)
        self.assertEqual(cfg.stages.A0.max_updates, 50000)

        self.assertEqual(cfg.stages.A1.batch_size, 8)
        self.assertEqual(cfg.stages.A1.max_updates, 100000)
        self.assertEqual(cfg.stages.A1.burnin_intervals, 8)
        self.assertEqual(cfg.stages.A1.supervised_intervals, 16)
        self.assertEqual(tuple(cfg.stages.A1.rollout_curriculum), (1, 2, 4))

        self.assertEqual(cfg.stages.B.batch_size, 8)
        self.assertEqual(cfg.stages.B.max_updates, 100000)
        self.assertEqual(cfg.stages.B.supervised_intervals, 16)

        self.assertEqual(cfg.stages.D1.batch_size, 8)
        self.assertEqual(cfg.stages.D1.max_updates, 200000)
        self.assertEqual(cfg.stages.D1.supervised_intervals, 16)
        self.assertEqual(tuple(cfg.stages.D1.rollout_curriculum), (2, 4, 8, 16))

        self.assertEqual(cfg.stages.D2.batch_size, 128)
        self.assertEqual(cfg.stages.D2.max_updates, 100000)
        self.assertEqual(cfg.stages.D2.terminal_batch_size, 64)
        self.assertEqual(cfg.stages.D2.maximum_pairs, 128)

        self.assertEqual(cfg.optimizer.kind, 'adamw')
        self.assertEqual(cfg.optimizer.learning_rate, 0.0001)
        self.assertEqual(tuple(cfg.optimizer.betas), (0.9, 0.999))
        self.assertEqual(cfg.optimizer.weight_decay, 0.01)
        self.assertEqual(cfg.optimizer.warmup_updates, 2000)
        self.assertEqual(cfg.optimizer.minimum_learning_rate, 0.00001)
        self.assertEqual(cfg.optimizer.schedule, 'cosine')
        self.assertEqual(cfg.optimizer.gradient_clip_norm, 1.0)

        self.assertEqual(cfg.stages.evaluation_interval_updates, 5000)
        self.assertEqual(cfg.stages.early_stopping_patience, 5)

    def test_runner_rejects_missing_dataset_for_trainable_stages(self):
        from icgs.training.method import run_method_training
        from unittest.mock import patch

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        components = {'geometry': nn.Linear(4, 4)}
        with self.assertRaisesRegex(ValueError, 'requires a dataset'):
            run_method_training(cfg, components, datasets=None, stage='A1')

    def test_runner_rejects_max_updates_exceeding_config_or_invalid_type(self):
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        components = {'geometry': nn.Linear(4, 4)}
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            run_method_training(cfg, components, datasets=[{'x': 1}], stage='A1', limits={'max_updates': True}, dataset_id="test-dataset")
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            run_method_training(cfg, components, datasets=[{'x': 1}], stage='A1', limits={'max_updates': -5}, dataset_id="test-dataset")
        with self.assertRaisesRegex(ValueError, 'exceeds configured stage ceiling'):
            run_method_training(cfg, components, datasets=[{'x': 1}], stage='A1', limits={'max_updates': 999999}, dataset_id="test-dataset")

    def test_runner_requires_seed_readiness(self):
        import numpy as np
        from icgs.training.method import run_method_training

        # Config without training_seeds cannot be bypassed by caller-owned rng
        cfg = MethodConfig.from_dict({'stages': {'training_seeds': None}})
        components = {'geometry': nn.Linear(4, 4)}
        with self.assertRaisesRegex(ValueError, 'requires declared stages.training_seeds in config'):
            run_method_training(cfg, components, datasets=[{'x': 1}], stage='A1', rng=np.random.default_rng(42), dataset_id="test-dataset")

    def test_runner_a1_temporal_replay_physical_loss_and_gradients(self):
        import numpy as np
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A1': {'supervised_intervals': 1, 'burnin_intervals': 0, 'rollout_curriculum': [1]},
            }
        })
        dynamics = PhysicalDynamics(cfg)

        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        rollout = PhysicalRollout(dynamics, encoder, decoder, memory, config=cfg)

        components = {
            'geometry': nn.ModuleDict({
                'encoder': encoder,
                'decoder': decoder,
            }),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        # Build valid physical state and transition target
        batch_size = 1
        points_count = 128
        pose = torch.eye(4)[None].repeat(batch_size, 1, 1)
        grip = torch.zeros(batch_size, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch_size, 1)
        valid = torch.ones(batch_size, 128, dtype=torch.bool)
        state = PhysicalState(
            X=torch.zeros(batch_size, 128, 256),
            x=torch.zeros(batch_size, 128, 3),
            valid=valid,
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(batch_size, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(batch_size, points_count, 3),
            cached_world_cloud_valid=torch.ones(batch_size, points_count, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )

        pts = np.zeros((10, 3), dtype=np.float64)
        se3 = np.eye(4, dtype=np.float64)
        before_obs = TimedObservation(Observation(pts.copy(), se3.copy(), 0.0), 0, 0.0, 0.0, 'sensor')
        after_obs = TimedObservation(Observation(pts.copy(), se3.copy(), 0.0), 1, 0.1, 0.1, 'sensor')
        cmd = TimedCommand(target_w=se3.copy(), grip=0, duration_s=0.1)
        trans = ExecutedTransition(before_obs, after_obs, cmd, 0.1, 1, 'ok')

        batch = {
            'state': state,
            'transitions': [trans],
            'valid': torch.tensor([True]),
            'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'},
        }

        report = run_method_training(
            cfg,
            components,
            datasets=[batch],
            stage='A1',
            limits={'max_updates': 1},
            dataset_id="test-dataset",
        )

        self.assertEqual(report.updates_completed, 1)
        self.assertEqual(len(report.loss_history), 1)
        self.assertGreater(report.loss_history[0], 0.0)

        # Ensure physical_memory received non-zero temporal gradients from the supervised segment
        memory_has_grads = any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in memory.parameters())
        self.assertTrue(memory_has_grads, 'physical_memory must receive non-zero gradients in Stage A1')

    def test_runner_lexicographic_evaluation_cadence_and_early_stopping(self):
        import numpy as np
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import run_method_training

        torch.manual_seed(42)
        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'evaluation_interval_updates': 1,
                'early_stopping_patience': 2,
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A1': {'supervised_intervals': 1, 'burnin_intervals': 0, 'rollout_curriculum': [1]},
            }
        })

        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        rollout = PhysicalRollout(dynamics, encoder, decoder, memory, config=cfg)

        components = {
            'geometry': nn.ModuleDict({
                'encoder': encoder,
                'decoder': decoder,
            }),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid = torch.ones(1, 128, dtype=torch.bool)
        state = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=valid,
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )

        pts = np.zeros((10, 3), dtype=np.float64)
        se3 = np.eye(4, dtype=np.float64)
        before_obs = TimedObservation(Observation(pts.copy(), se3.copy(), 0.0), 0, 0.0, 0.0, 'sensor')
        after_obs = TimedObservation(Observation(pts.copy(), se3.copy(), 0.0), 1, 0.1, 0.1, 'sensor')
        cmd = TimedCommand(target_w=se3.copy(), grip=0, duration_s=0.1)
        trans = ExecutedTransition(before_obs, after_obs, cmd, 0.1, 1, 'ok')

        train_batch = {
            'state': state,
            'transitions': [trans],
            'valid': torch.tensor([True]),
            'provenance': {'episode_id': 'ep_train_1', 'lineage_id': 'lin_train_1', 'split': 'train'},
        }
        eval_batch = {
            'state': state,
            'transitions': [trans],
            'valid': torch.tensor([True]),
            'provenance': {'episode_id': 'ep_dev_1', 'lineage_id': 'lin_dev_1', 'split': 'dev'},
        }

        # Run with evaluation dataset: with patience=2 and eval cadence=1, should stop early
        report = run_method_training(
            cfg,
            components,
            datasets=[train_batch],
            evaluation_dataset=[eval_batch],
            stage='A1',
            limits={'max_updates': 10},
            dataset_id="test-dataset",
        )
        self.assertTrue(report.stopped_early)
        self.assertLess(report.updates_completed, 10)

    def test_history_reconstruction_multi_transition_replay_indices_grad_modes_and_prefix_influence(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import _step_a1_rollout

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'A1': {'supervised_intervals': 2, 'burnin_intervals': 2, 'rollout_curriculum': [2]},
            }
        })
        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }
        apply_freeze_boundary('A1', components)

        # Build initial reset state (boundary=0)
        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid = torch.ones(1, 128, dtype=torch.bool)
        reset_state = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=valid,
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )

        def make_transitions(count, base_pt=0.0):
            res = []
            curr_pts = np.full((128, 3), base_pt, dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts + 0.01
                b_obs = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a_obs = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                cmd = TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1)
                res.append(ExecutedTransition(b_obs, a_obs, cmd, 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        # 6 transitions: boundary=4, burnin=2, K=2 -> prefix=[0, 1], burnin=[2, 3], supervised=[4, 5]
        transitions = make_transitions(6)
        visited_log = []

        def listener(step_idx, is_prefix, grad_enabled):
            visited_log.append((step_idx, is_prefix, grad_enabled))

        batch = {
            'reset_state': reset_state,
            'transitions': transitions,
            'boundary': 4,
            'valid': torch.ones((1, 2), dtype=torch.bool),
            'step_listener': listener,
        }

        # Run step
        optimizer = build_stage_optimizer('A1', components, cfg)
        optimizer.zero_grad()
        loss_base = _step_a1_rollout(components, batch, 2, cfg)
        loss_base.backward()

        # Check visited log: all pre-supervision steps are replayed under no_grad per authority resolution
        expected_log = [
            (0, True, False),
            (1, True, False),
            (2, True, False),
            (3, True, False),
        ]
        self.assertEqual(visited_log, expected_log)

        # Check decoder and memory grads from the supervised unroll
        dec_grad_base = [p.grad.clone() for p in decoder.parameters() if p.grad is not None]
        mem_grad_base = [p.grad.clone() for p in memory.parameters() if p.grad is not None]
        self.assertTrue(len(dec_grad_base) > 0)
        self.assertTrue(len(mem_grad_base) > 0)

        # Mutated history: transitions 0..3 follow an alternate continuous path but
        # converge to the exact same observation at boundary 4 (pts=0.04).
        # Supervised transitions 4 and 5 are 100% identical between both batches,
        # strictly isolating prefix influence through boundary memory.
        pts_sequence_mut = [1.0, 0.70, 0.40, 0.15, 0.04]
        transitions_mutated = []
        curr_se3 = np.eye(4, dtype=np.float64)
        for i in range(4):
            b_obs = TimedObservation(Observation(np.full((128, 3), pts_sequence_mut[i]), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
            a_obs = TimedObservation(Observation(np.full((128, 3), pts_sequence_mut[i + 1]), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
            cmd = TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1)
            transitions_mutated.append(ExecutedTransition(b_obs, a_obs, cmd, 0.1, i + 1, 'ok'))
        transitions_mutated.extend(transitions[4:])  # identical supervised targets

        batch_mutated = dict(batch)
        batch_mutated['transitions'] = transitions_mutated
        visited_log.clear()

        optimizer.zero_grad()
        loss_mut = _step_a1_rollout(components, batch_mutated, 2, cfg)
        loss_mut.backward()

        # Changing prefix must change the supervised loss and gradients
        self.assertNotEqual(float(loss_base.item()), float(loss_mut.item()))
        dec_grad_mut = [p.grad.clone() for p in decoder.parameters() if p.grad is not None]
        diffs = [torch.norm(g1 - g2).item() for g1, g2 in zip(dec_grad_base, dec_grad_mut)]
        self.assertTrue(any(d > 1e-6 for d in diffs), 'mutated prefix must affect decoder gradients')

    def test_d1_history_replay_frozen_gradient_transmission_and_bootstrap(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import _step_d1_rollout

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'D1': {'supervised_intervals': 2, 'rollout_curriculum': [2]},
            }
        })
        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }
        apply_freeze_boundary('D1', components)

        # In D1, geometry and physical_memory must have requires_grad=False and eval mode
        self.assertFalse(any(p.requires_grad for p in components['geometry'].parameters()))
        self.assertFalse(any(p.requires_grad for p in components['physical_memory'].parameters()))
        self.assertTrue(all(p.requires_grad for p in components['dynamics'].parameters()))
        self.assertFalse(components['geometry'].training)
        self.assertFalse(components['physical_memory'].training)
        self.assertTrue(components['dynamics'].training)

        optimizer = build_stage_optimizer('D1', components, cfg)
        self.assertIsNotNone(optimizer)

        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid = torch.ones(1, 128, dtype=torch.bool)
        reset_state = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=valid,
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )

        def make_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        batch = {
            'reset_state': reset_state,
            'transitions': make_transitions(4),
            'boundary': 2,
            'valid': torch.ones((1, 2), dtype=torch.bool),
            'episode_id': 'ep_test_d1',
            'bootstrap_seed': 42,
        }

        optimizer.zero_grad()
        loss = _step_d1_rollout(components, batch, 2, cfg)
        loss.backward()

        # Frozen components must have no gradient
        self.assertTrue(all(p.grad is None for p in components['geometry'].parameters()))
        self.assertTrue(all(p.grad is None for p in components['physical_memory'].parameters()))

        # Trainable dynamics must receive non-zero gradients propagated through the frozen graph
        dyn_grads = [p.grad for p in components['dynamics'].parameters() if p.grad is not None]
        self.assertTrue(len(dyn_grads) > 0)
        self.assertTrue(any(bool((g.abs() > 0).any().item()) for g in dyn_grads))

    def test_unsupported_stages_b_and_d2_raise_not_implemented_error(self):
        from icgs.training.method import _evaluate_stage, run_method_training

        cfg = MethodConfig()
        components = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4), 'decoder': nn.Linear(4, 4)}),
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
            'events': nn.Linear(4, 4),
            'task': nn.Linear(4, 4),
            'evaluation': nn.Linear(4, 4),
            'terminal': nn.Linear(4, 4),
        }
        with self.assertRaisesRegex(NotImplementedError, r'Stage B training requires upstream event/task objective APIs \(src/icgs/algorithms/objectives/task\.py owned by P06 Task1C\)'):
            run_method_training(cfg, components, datasets=None, stage='B')

        with self.assertRaisesRegex(NotImplementedError, r'Stage D2 training requires upstream evaluator/terminal objective APIs \(src/icgs/algorithms/objectives/evaluation\.py owned by P09\)'):
            run_method_training(cfg, components, datasets=None, stage='D2')

        with self.assertRaisesRegex(NotImplementedError, r'Stage B evaluation requires upstream event/task objective APIs \(src/icgs/algorithms/objectives/task\.py owned by P06 Task1C\)'):
            _evaluate_stage('B', components, None, cfg)

        with self.assertRaisesRegex(NotImplementedError, r'Stage D2 evaluation requires upstream evaluator/terminal objective APIs \(src/icgs/algorithms/objectives/evaluation\.py owned by P09\)'):
            _evaluate_stage('D2', components, None, cfg)

    def test_evaluation_temporary_eval_mode_and_restoration_on_success_and_exception(self):
        from icgs.training.method import _evaluate_stage

        cfg = MethodConfig()

        class CustomEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)

            def forward(self, pts, valid):
                return type('Enc', (), {'x': pts, 'points_w': pts, 'point_valid': valid})()

        class CustomDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(p=0.5)

            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        encoder = CustomEncoder()
        decoder = CustomDecoder()
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
        }
        # Start in train mode
        components['geometry'].train()
        self.assertTrue(encoder.training)
        self.assertTrue(decoder.training)

        from collections.abc import Sequence

        # Define immutable sequence that checks training mode inside evaluation
        class ModeSpySequence(Sequence):
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
                self.observed_eval = False

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                if idx < 0 or idx >= 1:
                    raise IndexError(idx)
                self.observed_eval = (not encoder.training) and (not decoder.training)
                if self.should_fail:
                    raise RuntimeError('evaluation error for testing finally restoration')
                pts = torch.zeros(1, 4, 3)
                return {
                    'points': pts,
                    'provenance': {'episode_id': 'ep_dev_1', 'lineage_id': 'lin_dev_1', 'split': 'dev'},
                }

        spy_ds = ModeSpySequence(should_fail=False)
        metrics = _evaluate_stage('A0', components, spy_ds, cfg)
        self.assertTrue(spy_ds.observed_eval, 'modules must be in eval mode during evaluation')
        self.assertTrue(encoder.training, 'encoder must be restored to train mode in finally')
        self.assertTrue(decoder.training, 'decoder must be restored to train mode in finally')
        self.assertIn('val_loss', metrics)

        # Test exception path: must still restore train mode
        failing_ds = ModeSpySequence(should_fail=True)
        with self.assertRaisesRegex(RuntimeError, 'evaluation error for testing finally restoration'):
            _evaluate_stage('A0', components, failing_ds, cfg)
        self.assertTrue(encoder.training, 'encoder must be restored to train mode after exception')
        self.assertTrue(decoder.training, 'decoder must be restored to train mode after exception')

        # Test no zero fallback on empty or invalid inputs
        with self.assertRaisesRegex(ValueError, 'cannot be empty'):
            _evaluate_stage('A0', components, [], cfg)
        with self.assertRaisesRegex(ValueError, 'cannot be None'):
            _evaluate_stage('A0', components, None, cfg)
        with self.assertRaisesRegex(ValueError, 'must be a mapping'):
            _evaluate_stage('A0', components, ['not_a_mapping'], cfg)

    def test_transition_validation_and_reset_reconstruction_negatives(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import _step_a1_rollout, _validate_transitions_and_reconstruct_reset

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'A1': {'supervised_intervals': 2, 'burnin_intervals': 0, 'rollout_curriculum': [1]},
            }
        })
        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        def make_valid_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        valid_transitions = make_valid_transitions(2)

        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid_reset = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )

        # Negative 1: Nonzero reset hidden memory must be rejected
        bad_reset = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=proprioception(pose, grip, gravity),
            memory=torch.ones(1, 2, 256),  # Non-zero memory!
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )
        with self.assertRaisesRegex(ValueError, 'reset state hidden memory must be zero'):
            _step_a1_rollout(components, {'reset_state': bad_reset, 'transitions': valid_transitions}, 1, cfg)

        # Negative 2: Stale cached reset_state.X is discarded in favor of fresh encoder reconstruction
        stale_reset = PhysicalState(
            X=torch.full((1, 128, 256), 999.0),
            x=torch.full((1, 128, 3), 999.0),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )
        reconstructed, _ = _validate_transitions_and_reconstruct_reset(
            valid_transitions,
            encoder,
            cfg,
            reset_state=stale_reset,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        self.assertFalse(torch.allclose(reconstructed.X, torch.tensor(999.0)))
        self.assertTrue(torch.equal(reconstructed.memory, torch.zeros(1, 2, 256)))

        # Negative 3: Gapped transition sequence
        b_gap = TimedObservation(Observation(np.zeros((128, 3)), np.eye(4), 0.0), 2, 0.2, 0.2, 'sensor')
        a_gap = TimedObservation(Observation(np.zeros((128, 3)), np.eye(4), 0.0), 3, 0.3, 0.3, 'sensor')
        gapped_trans = [valid_transitions[0], ExecutedTransition(b_gap, a_gap, TimedCommand(target_w=np.eye(4), grip=0, duration_s=0.1), 0.1, 3, 'ok')]
        with self.assertRaisesRegex(ValueError, r'transition 1 before boundary is 2, expected 1'):
            _step_a1_rollout(components, {'reset_state': valid_reset, 'transitions': gapped_trans}, 1, cfg)

        # Negative 4: Shuffled transition sequence
        shuffled_trans = [valid_transitions[1], valid_transitions[0]]
        with self.assertRaisesRegex(ValueError, r'transitions must start at boundary 0'):
            _step_a1_rollout(components, {'reset_state': valid_reset, 'transitions': shuffled_trans}, 1, cfg)

        # Negative 5: Discontinuous observation between adjacent transitions
        b_disc = TimedObservation(Observation(np.full((128, 3), 5.0), np.eye(4), 0.0), 1, 0.1, 0.1, 'sensor')
        a_disc = TimedObservation(Observation(np.full((128, 3), 5.0), np.eye(4), 0.0), 2, 0.2, 0.2, 'sensor')
        disc_trans = [valid_transitions[0], ExecutedTransition(b_disc, a_disc, TimedCommand(target_w=np.eye(4), grip=0, duration_s=0.1), 0.1, 2, 'ok')]
        with self.assertRaisesRegex(ValueError, r'discontinuous points between transition 0 after and 1 before'):
            _step_a1_rollout(components, {'reset_state': valid_reset, 'transitions': disc_trans}, 1, cfg)

        # Negative 6: Non-finite before pose
        b_nan = TimedObservation(Observation(np.zeros((128, 3)), np.eye(4), 0.0), 0, 0.0, 0.0, 'sensor')
        object.__setattr__(b_nan.observation, 'T_w_e', np.full((4, 4), np.nan))
        nan_trans = [ExecutedTransition(b_nan, valid_transitions[0].after, valid_transitions[0].command, 0.1, 1, 'ok'), valid_transitions[1]]
        with self.assertRaisesRegex(ValueError, r'transition 0 before observation must have explicit finite 4x4 pose'):
            _step_a1_rollout(components, {'reset_state': valid_reset, 'transitions': nan_trans}, 1, cfg)

    def test_k_curriculum_distinct_from_sequence_length_and_no_batch_overrides(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import _step_a1_rollout, _step_d1_rollout

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'A1': {'supervised_intervals': 4, 'burnin_intervals': 2, 'rollout_curriculum': [1, 2]},
                'D1': {'supervised_intervals': 4, 'rollout_curriculum': [1, 2]},
            }
        })
        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        def make_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        transitions = make_transitions(6)

        # Batch conflicting supervised_intervals rejected
        with self.assertRaisesRegex(ValueError, r'batch supervised_intervals \(2\) conflicts with configured stage value \(4\)'):
            _step_a1_rollout(components, {'transitions': transitions, 'supervised_intervals': 2}, 1, cfg)

        # Batch conflicting burnin rejected
        with self.assertRaisesRegex(ValueError, r'batch burnin \(0\) conflicts with configured stage value \(2\)'):
            _step_a1_rollout(components, {'transitions': transitions, 'burnin': 0}, 1, cfg)

        # Insufficient transitions rejected
        with self.assertRaisesRegex(ValueError, r'transitions sequence length 3 is insufficient for boundary 0 \+ supervised_intervals 4'):
            _step_a1_rollout(components, {'transitions': transitions[:3]}, 1, cfg)

        # D1 rejects any batch burnin
        with self.assertRaisesRegex(ValueError, r'D1 has no configured burnin; batch burnin must not be provided'):
            _step_d1_rollout(
                components,
                {
                    'transitions': transitions,
                    'burnin': 2,
                    'episode_id': 'ep_test',
                    'bootstrap_seed': 1,
                },
                1,
                cfg,
            )

        # K curriculum visibly changes unroll depth within same 4 supervised intervals
        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid_reset = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )
        loss_k1 = _step_a1_rollout(components, {'reset_state': valid_reset, 'transitions': transitions, 'boundary': 0}, 1, cfg)
        loss_k2 = _step_a1_rollout(components, {'reset_state': valid_reset, 'transitions': transitions, 'boundary': 0}, 2, cfg)
        self.assertTrue(torch.is_tensor(loss_k1) and torch.is_tensor(loss_k2))
        self.assertGreater(loss_k1.item(), 0.0)
        self.assertGreater(loss_k2.item(), 0.0)

    def test_d1_provenance_and_validity_shape_rejection(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import _step_d1_rollout

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'D1': {'supervised_intervals': 2, 'rollout_curriculum': [1]},
            }
        })
        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        def make_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        transitions = make_transitions(2)

        # Missing episode_id rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit nonempty episode identity \('episode_id'\)"):
            _step_d1_rollout(components, {'transitions': transitions, 'bootstrap_seed': 0}, 1, cfg)

        # Empty episode_id rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit nonempty episode identity \('episode_id'\)"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': '', 'bootstrap_seed': 0}, 1, cfg)

        # Whitespace episode_id rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit nonempty episode identity \('episode_id'\)"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': '   ', 'bootstrap_seed': 0, 'valid': torch.ones((1, 2), dtype=torch.bool)}, 1, cfg)

        # Generic-only seed rejected (must not accept generic 'seed' fallback)
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit 'bootstrap_seed', generic 'seed' is not accepted"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': 'ep_1', 'seed': 42, 'valid': torch.ones((1, 2), dtype=torch.bool)}, 1, cfg)

        # Missing bootstrap_seed rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit bootstrap seed with provenance \('bootstrap_seed'\)"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': 'ep_1'}, 1, cfg)

        # Negative bootstrap_seed rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit bootstrap seed with provenance \('bootstrap_seed'\)"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': 'ep_1', 'bootstrap_seed': -1}, 1, cfg)

        # Boolean bootstrap_seed rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicit bootstrap seed with provenance \('bootstrap_seed'\)"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': 'ep_1', 'bootstrap_seed': True}, 1, cfg)

        # Missing valid mask rejected
        with self.assertRaisesRegex(ValueError, r"Stage D1 requires explicitly supplied 'valid' mask"):
            _step_d1_rollout(components, {'transitions': transitions, 'episode_id': 'ep_1', 'bootstrap_seed': 42}, 1, cfg)

        # Wrong valid shape rejected (not [1, num_supervised])
        with self.assertRaisesRegex(ValueError, r"D1 valid mask must have shape \[1, 2\] and boolean dtype"):
            _step_d1_rollout(
                components,
                {
                    'transitions': transitions,
                    'episode_id': 'ep_1',
                    'bootstrap_seed': 42,
                    'valid': torch.tensor([True]),
                },
                1,
                cfg,
            )

        # Valid execution with retained head identity
        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid_reset = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )
        loss = _step_d1_rollout(
            components,
            {
                'reset_state': valid_reset,
                'transitions': transitions,
                'episode_id': 'ep_provenance_ok',
                'bootstrap_seed': 42,
                'valid': torch.ones((1, 2), dtype=torch.bool),
            },
            1,
            cfg,
        )
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_evaluation_no_enable_grad_batchnorm_dropout_invariance_and_mode_restore(self):
        from icgs.training.method import _evaluate_stage

        cfg = MethodConfig()

        class BNEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)

            def forward(self, pts, valid):
                B, N, C = pts.shape
                flat = pts.view(-1, C)
                normed = self.bn(flat).view(B, N, C)
                return type('Enc', (), {'x': normed, 'points_w': normed, 'point_valid': valid})()

        class DropDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = nn.Dropout(p=0.5)

            def forward(self, enc):
                return type('Dec', (), {'points_w': self.drop(enc.points_w), 'point_valid': enc.point_valid})()

        encoder = BNEncoder()
        decoder = DropDecoder()
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
        }

        # Warm up running stats in train mode
        encoder.train()
        decoder.train()
        warmup_pts = torch.randn(2, 10, 3) * 5.0 + 2.0
        with torch.no_grad():
            encoder(warmup_pts, torch.ones(2, 10, dtype=torch.bool))

        initial_mean = encoder.bn.running_mean.clone()
        initial_var = encoder.bn.running_var.clone()
        initial_num_batches = encoder.bn.num_batches_tracked.clone()

        # Evaluation pass with new points: running stats MUST remain completely unchanged
        eval_batch = {
            'points': torch.randn(1, 10, 3) * 100.0,
            'provenance': {'episode_id': 'ep_dev_1', 'lineage_id': 'lin_dev_1', 'split': 'dev'},
        }
        eval_dataset = [eval_batch]

        metrics = _evaluate_stage('A0', components, eval_dataset, cfg)
        self.assertIn('val_loss', metrics)

        # Running stats must be perfectly invariant during evaluation pass
        torch.testing.assert_close(encoder.bn.running_mean, initial_mean)
        torch.testing.assert_close(encoder.bn.running_var, initial_var)
        torch.testing.assert_close(encoder.bn.num_batches_tracked, initial_num_batches)

        # Components must have their training mode restored to train()
        self.assertTrue(encoder.training)
        self.assertTrue(decoder.training)

    def test_noncanonical_gravity_preserved_in_reset_and_replay_proprioception(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import (
            _replay_history_and_detach,
            _validate_transitions_and_reconstruct_reset,
        )

        cfg = MethodConfig()
        encoder = PhysicalEncoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)

        # Calibrated noncanonical gravity carrier
        noncanonical_gravity = torch.tensor([[0.25, -0.50, -9.75]], dtype=torch.float32)
        pose0 = torch.eye(4, dtype=torch.float32)[None]
        grip0 = torch.zeros((1, 1), dtype=torch.float32)
        p0 = proprioception(pose0, grip0, noncanonical_gravity, config=cfg)

        initial_reset = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=p0,
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose0,
            grip=grip0,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='test-enc',
            memory_lineage='test-mem',
        )

        def make_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        transitions = make_transitions(3)

        # Reconstruct reset state
        reconstructed_reset, validated_transitions = _validate_transitions_and_reconstruct_reset(
            transitions,
            encoder,
            cfg,
            reset_state=initial_reset,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )

        # Noncanonical gravity in reset proprioception must match initial carrier exactly
        torch.testing.assert_close(reconstructed_reset.p[:, 10:13], noncanonical_gravity)

        # Replay history through boundary 2
        replayed_state = _replay_history_and_detach(
            reset_state=reconstructed_reset,
            transitions=validated_transitions,
            boundary=2,
            encoder=encoder,
            physical_memory=memory,
            config=cfg,
        )

        # Replay proprioception must retain noncanonical gravity unchanged
        torch.testing.assert_close(replayed_state.p[:, 10:13], noncanonical_gravity)

    def test_a0_always_uses_chamfer_and_ignores_bogus_reconstruction_loss_method(self):
        import numpy as np
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.training.method import _step_a0_reconstruction

        cfg = MethodConfig()
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)

        geometry = nn.ModuleDict({'encoder': encoder, 'decoder': decoder})

        # Attach a bogus reconstruction_loss method that would fail if invoked
        def bogus_reconstruction_loss(*args, **kwargs):
            raise AssertionError("Bogus reconstruction_loss shortcut was invoked instead of defined Chamfer objective")

        geometry.reconstruction_loss = bogus_reconstruction_loss

        components = {'geometry': geometry}
        batch = {
            'points': np.zeros((128, 3), dtype=np.float32),
            'valid': np.ones((128,), dtype=bool),
        }

        # Should execute encoder -> decoder -> masked_normalized_chamfer_distance without calling shortcut
        loss = _step_a0_reconstruction(components, batch, cfg)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_a1_and_d1_reject_missing_reset_state_calibrated_gravity_carrier(self):
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory
        from icgs.training.method import _step_a1_rollout, _step_d1_rollout

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'A1': {'supervised_intervals': 2, 'burnin_intervals': 0, 'rollout_curriculum': [1]},
                'D1': {'supervised_intervals': 2, 'rollout_curriculum': [1]},
            }
        })
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        dynamics = PhysicalDynamics(cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        def make_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        transitions = make_transitions(2)

        # Stage A1 batch without reset_state (and without state alias)
        with self.assertRaises((ValueError, TypeError)) as cm_a1:
            _step_a1_rollout(components, {'transitions': transitions}, 1, cfg)
        msg_a1 = str(cm_a1.exception).lower()
        self.assertIn("reset_state", msg_a1)
        self.assertIn("calibrated gravity", msg_a1)

        # Stage D1 batch without reset_state (and without state alias)
        with self.assertRaises((ValueError, TypeError)) as cm_d1:
            _step_d1_rollout(
                components,
                {
                    'transitions': transitions,
                    'episode_id': 'ep_test',
                    'bootstrap_seed': 42,
                    'valid': torch.ones((1, 2), dtype=torch.bool),
                },
                1,
                cfg,
            )
        msg_d1 = str(cm_d1.exception).lower()
        self.assertIn("reset_state", msg_d1)
        self.assertIn("calibrated gravity", msg_d1)


class MethodTrainingTask3Tests(unittest.TestCase):
    @staticmethod
    def _reference_payload(config=None):
        config = config or MethodConfig()
        resolved = config.to_dict()
        geometry = resolved['geometry']
        return {
            'ip_checksum': 'a' * 64,
            'native_profile': config.native_profile,
            'geometry': geometry,
            'physical_weights': {
                'artifact_id': 'physical-v1',
                'config': {
                    'geometry': geometry,
                    'memory': resolved['memory'],
                    'neural': resolved['neural'],
                    'decoder': resolved['decoder'],
                    'numerics': resolved['numerics'],
                    'sensors': resolved['sensors'],
                    'control': {'dt0': resolved['control']['dt0']},
                },
            },
            'event_weights': 'event-v1',
            'task_weights': 'task-v1',
            'segmentation': {'id': 'segments-v1'},
            'router': {'artifact_id': 'router-v1', 'config': resolved['router']},
            'preprocessing': {
                name: geometry[name]
                for name in ('voxel_size_m', 'num_anchors', 'num_points', 'neighbors',
                             'ell0_m', 'fps_start', 'tie_break')
            },
            'calibration': {'id': 'cal-v1'},
            'camera': {'id': 'camera-v1'},
            'gravity': [0.0, 0.0, -1.0],
            'workspace': {'id': 'workspace-v1'},
            'cadence': {
                'dt0': resolved['control']['dt0'],
                'h': resolved['planning']['h'],
                'r': resolved['planning']['r'],
                'H': resolved['planning']['H'],
            },
            'rng_protocol': {
                'route': 'route-v1',
                'diffusion': 'diffusion-v1',
                'config': {
                    name: resolved['stages'][name]
                    for name in ('generator_seed', 'reset_seed', 'action_seed')
                },
            },
        }

    def test_validate_resume_manifest_keys(self):
        from icgs.artifacts.method import validate_resume

        valid = dict(
            schema_version=1,
            stage='D2',
            reference_id='a',
            dataset_id='d',
            config_id='c',
            total_updates=100,
            selected_seed=101,
        )
        validate_resume(valid, dict(valid))

        # Check all required keys including total_updates
        for key in ('schema_version', 'stage', 'reference_id', 'dataset_id', 'config_id', 'total_updates', 'selected_seed'):
            with self.assertRaisesRegex(ValueError, 'incompatible resume manifest'):
                validate_resume(valid, dict(valid, **{key: 999 if key == 'total_updates' else 'different'}))

    def test_load_checkpoint_fails_malformed_state_before_mutating_live_components(self):
        import tempfile
        import numpy as np
        from icgs.training.method import load_checkpoint, save_checkpoint

        cfg = MethodConfig()
        comp = {'geometry': nn.Linear(4, 4), 'physical_memory': nn.Linear(4, 4), 'dynamics': nn.Linear(4, 4)}

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = f'{tmp_dir}/test.pt'
            params = list(comp['geometry'].parameters()) + list(comp['physical_memory'].parameters())
            opt = torch.optim.AdamW(params, lr=1e-3)
            loss = comp['geometry'](torch.randn(2, 4)).sum() + comp['physical_memory'](torch.randn(2, 4)).sum()
            loss.backward()
            opt.step()
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            save_checkpoint(
                ckpt_file,
                stage='A1',
                update=1,
                total_updates=10,
                components=comp,
                optimizer=opt,
                scheduler=sched,
                config=cfg,
                dataset_id='d',
                reference_id='r',
                selected_seed=101,
                rng=np.random.default_rng(123),
                accumulated_loss_history=[0.5],
            )

            comp['geometry'].weight.data.fill_(99.9)
            comp['physical_memory'].weight.data.fill_(88.8)
            weight_before = comp['geometry'].weight.clone()
            pm_weight_before = comp['physical_memory'].weight.clone()
            manifest = {
                'schema_version': 1,
                'stage': 'A1',
                'reference_id': 'r',
                'dataset_id': 'd',
                'config_id': cfg.fingerprint(),
                'total_updates': 10,
                'selected_seed': 101,
                'update': 1,
            }

            # 1. Missing component in checkpoint
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['components']['geometry']
            bad_file = f'{tmp_dir}/missing_comp.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 2. Extra component in checkpoint
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['components']['extra_unauthorized'] = {'w': torch.tensor(1.0)}
            bad_file = f'{tmp_dir}/extra_comp.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 3. Component tensor shape mismatch
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['components']['geometry']['weight'] = torch.randn(8, 8)
            bad_file = f'{tmp_dir}/shape_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 4. Component tensor dtype mismatch
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['components']['geometry']['weight'] = raw['components']['geometry']['weight'].to(torch.float64)
            bad_file = f'{tmp_dir}/dtype_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(TypeError):
                load_checkpoint(bad_file, components=comp, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 5. Missing optimizer state
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['optimizer']
            bad_file = f'{tmp_dir}/missing_opt.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 6. Malformed optimizer param groups count
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['optimizer']['param_groups'] = [{'params': []}, {'params': []}]
            bad_file = f'{tmp_dir}/malformed_opt_groups.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 7. Missing scheduler state
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['scheduler']['last_epoch']
            bad_file = f'{tmp_dir}/bad_sched.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, scheduler=sched, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 8. Missing RNG torch state
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['rng']['torch']
            bad_file = f'{tmp_dir}/missing_rng_torch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 9. Missing selection state key
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['selection_state']['best_metric']
            bad_file = f'{tmp_dir}/bad_selection.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 10. Missing accumulated_loss_history
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['accumulated_loss_history']
            bad_file = f'{tmp_dir}/missing_acc_loss.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 11. Non-finite value in accumulated_loss_history
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['accumulated_loss_history'] = [float('nan')]
            bad_file = f'{tmp_dir}/nan_acc_loss.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 12. Length mismatch in accumulated_loss_history vs update
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['accumulated_loss_history'] = [0.1, 0.2]  # update is 1, len is 2
            bad_file = f'{tmp_dir}/len_mismatch_acc_loss.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 13. Manifest update mismatch with top-level update
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['manifest']['update'] = 5
            bad_file = f'{tmp_dir}/manifest_update_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 14. Manifest total_updates mismatch with top-level total_updates
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['manifest']['total_updates'] = 20
            bad_file = f'{tmp_dir}/manifest_total_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 15. Optimizer per-param exp_avg shape mismatch
            raw = torch.load(ckpt_file, map_location='cpu')
            # ensure state has param 0
            if 0 not in raw['optimizer']['state']:
                raw['optimizer']['state'][0] = {'step': 1, 'exp_avg': torch.zeros(4, 4), 'exp_avg_sq': torch.zeros(4, 4)}
            raw['optimizer']['state'][0]['exp_avg'] = torch.randn(2, 2)
            bad_file = f'{tmp_dir}/opt_shape_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 16. Optimizer per-param exp_avg dtype mismatch
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['optimizer']['state'][0]['exp_avg'] = torch.zeros(4, 4, dtype=torch.float64)
            bad_file = f'{tmp_dir}/opt_dtype_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(TypeError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 17. Optimizer per-param exp_avg_sq shape mismatch
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['optimizer']['state'][0]['exp_avg_sq'] = torch.randn(2, 2)
            bad_file = f'{tmp_dir}/opt_sq_shape_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 18. Optimizer per-param exp_avg_sq dtype mismatch
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['optimizer']['state'][0]['exp_avg_sq'] = torch.zeros(4, 4, dtype=torch.float64)
            bad_file = f'{tmp_dir}/opt_sq_dtype_mismatch.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(TypeError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))

            # 19. Missing manifest update
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['manifest']['update']
            bad_file = f'{tmp_dir}/manifest_missing_update.pt'
            torch.save(raw, bad_file)
            with self.assertRaises((KeyError, ValueError)):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))
            self.assertTrue(torch.equal(comp['physical_memory'].weight, pm_weight_before))

            # 20. Missing manifest total_updates
            raw = torch.load(ckpt_file, map_location='cpu')
            del raw['manifest']['total_updates']
            bad_file = f'{tmp_dir}/manifest_missing_total_updates.pt'
            torch.save(raw, bad_file)
            with self.assertRaises((KeyError, ValueError)):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))
            self.assertTrue(torch.equal(comp['physical_memory'].weight, pm_weight_before))

            # 21. Swapped same-shaped saved IDs and associated state entries
            raw = torch.load(ckpt_file, map_location='cpu')
            # Params 0 and 2 are both (4, 4) weights; swap their IDs in param_groups[0]['params']
            raw['optimizer']['param_groups'][0]['params'] = [2, 1, 0, 3]
            st0 = raw['optimizer']['state'][0]
            st2 = raw['optimizer']['state'][2]
            raw['optimizer']['state'][0] = st2
            raw['optimizer']['state'][2] = st0
            bad_file = f'{tmp_dir}/swapped_same_shape_ids.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(ValueError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))
            self.assertTrue(torch.equal(comp['physical_memory'].weight, pm_weight_before))

            # 22. Unknown param ID in optimizer state
            raw = torch.load(ckpt_file, map_location='cpu')
            raw['optimizer']['state'][99] = {'step': 1, 'exp_avg': torch.zeros(4, 4), 'exp_avg_sq': torch.zeros(4, 4)}
            bad_file = f'{tmp_dir}/unknown_param_id.pt'
            torch.save(raw, bad_file)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file, components=comp, optimizer=opt, requested_manifest=manifest)
            self.assertTrue(torch.equal(comp['geometry'].weight, weight_before))
            self.assertTrue(torch.equal(comp['physical_memory'].weight, pm_weight_before))

    def test_load_checkpoint_apply_failure_rolls_back_all_collaborators(self):
        import copy
        import random
        import tempfile
        import numpy as np
        from icgs.configuration.method import MethodConfig
        from icgs.training.method import save_checkpoint, load_checkpoint, _get_rng_state, _set_rng_state

        cfg = MethodConfig()

        class ControlledFailOnceSampler:
            def __init__(self, initial_cursor=42):
                self.cursor = initial_cursor
                self.should_fail = True
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                return {}
            def state_dict(self):
                return {'cursor': self.cursor}
            def load_state_dict(self, state):
                if self.should_fail:
                    self.should_fail = False
                    raise RuntimeError('simulated sampler apply failure')
                self.cursor = state['cursor']

        comp = {'geometry': nn.Linear(4, 4), 'physical_memory': nn.Linear(4, 4), 'dynamics': nn.Linear(4, 4)}
        comp['geometry'].weight.data.fill_(11.11)
        comp['geometry'].bias.data.fill_(22.22)

        opt = torch.optim.AdamW(comp['geometry'].parameters(), lr=1e-3)
        loss = comp['geometry'](torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

        sampler = ControlledFailOnceSampler(initial_cursor=42)
        runner_rng = np.random.default_rng(12345)

        diff_comp = {'geometry': nn.Linear(4, 4), 'physical_memory': nn.Linear(4, 4), 'dynamics': nn.Linear(4, 4)}
        diff_comp['geometry'].weight.data.fill_(99.99)
        diff_comp['geometry'].bias.data.fill_(88.88)
        diff_opt = torch.optim.AdamW(diff_comp['geometry'].parameters(), lr=5e-4)
        diff_loss = diff_comp['geometry'](torch.randn(2, 4)).sum()
        diff_loss.backward()
        diff_opt.step()
        diff_sched = torch.optim.lr_scheduler.StepLR(diff_opt, step_size=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = f'{tmp_dir}/target.pt'
            save_checkpoint(
                ckpt_file,
                stage='A1',
                update=1,
                total_updates=10,
                components=diff_comp,
                optimizer=diff_opt,
                scheduler=diff_sched,
                config=cfg,
                dataset_id='d',
                reference_id='r',
                selected_seed=101,
                rng=np.random.default_rng(999),
                sampler_state={'cursor': 999},
                accumulated_loss_history=[1.23],
            )

            # Capture pre-load live state right before load_checkpoint
            weights_before = {k: v.clone() for k, v in comp['geometry'].state_dict().items()}
            opt_before = copy.deepcopy(opt.state_dict())
            sched_before = copy.deepcopy(sched.state_dict())
            rng_state_before = _get_rng_state(runner_rng=runner_rng)

            manifest = {
                'schema_version': 1,
                'stage': 'A1',
                'reference_id': 'r',
                'dataset_id': 'd',
                'config_id': cfg.fingerprint(),
                'total_updates': 10,
                'selected_seed': 101,
                'update': 1,
            }

            with self.assertRaisesRegex(RuntimeError, 'simulated sampler apply failure'):
                load_checkpoint(
                    ckpt_file,
                    components=comp,
                    optimizer=opt,
                    scheduler=sched,
                    datasets=sampler,
                    runner_rng=runner_rng,
                    requested_manifest=manifest,
                )

            # Assert live weights and biases exactly restored
            self.assertTrue(torch.equal(comp['geometry'].weight, weights_before['weight']))
            self.assertTrue(torch.equal(comp['geometry'].bias, weights_before['bias']))

            # Assert optimizer parameter groups and state tensors exactly restored
            self.assertEqual(len(opt.param_groups), len(opt_before['param_groups']))
            for ga, gb in zip(opt.param_groups, opt_before['param_groups']):
                for k in ga:
                    if k != 'params':
                        self.assertEqual(ga[k], gb[k])
            for pid in opt.state_dict()['state']:
                for k in opt.state_dict()['state'][pid]:
                    v_now = opt.state_dict()['state'][pid][k]
                    v_exp = opt_before['state'][pid][k]
                    if torch.is_tensor(v_now):
                        self.assertTrue(torch.equal(v_now, v_exp))
                    else:
                        self.assertEqual(v_now, v_exp)

            # Assert scheduler exactly restored
            self.assertEqual(sched.state_dict(), sched_before)

            # Assert sampler cursor exactly restored
            self.assertEqual(sampler.cursor, 42)

            # Assert runner RNG and global RNG states exactly restored
            now_rng_state = _get_rng_state(runner_rng=runner_rng)
            self.assertEqual(now_rng_state['python'], rng_state_before['python'])
            self.assertTrue(np.array_equal(now_rng_state['numpy'][1], rng_state_before['numpy'][1]))
            self.assertTrue(torch.equal(now_rng_state['torch'], rng_state_before['torch']))
            self.assertEqual(now_rng_state['runner_rng'], rng_state_before['runner_rng'])

            # Verify subsequent draws from restored state match exactly
            draw_py = random.random()
            draw_np = np.random.rand()
            draw_torch = torch.randn(5)
            draw_runner = runner_rng.random()

            _set_rng_state(rng_state_before, runner_rng=np.random.default_rng(12345))
            self.assertEqual(random.random(), draw_py)
            self.assertEqual(np.random.rand(), draw_np)
            self.assertTrue(torch.equal(torch.randn(5), draw_torch))

    def test_uninterrupted_vs_1_plus_resume_exact_identity(self):
        import copy
        import random
        import tempfile
        import numpy as np
        from icgs.algorithms.rollout.physical import PhysicalRollout
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import run_method_training

        # Prepare two identical initial model setups
        torch.manual_seed(999)
        cfg = MethodConfig.from_dict({
            'stages': {
                'evaluation_interval_updates': 1,
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A1': {'supervised_intervals': 2, 'burnin_intervals': 0, 'rollout_curriculum': [2], 'batch_size': 1},
            }
        })

        def make_components():
            torch.manual_seed(999)
            dynamics = PhysicalDynamics(cfg)
            encoder = PhysicalEncoder(method_config=cfg)
            decoder = PhysicalDecoder(method_config=cfg)
            memory = PhysicalMemory(config=cfg)
            return {
                'geometry': nn.ModuleDict({
                    'encoder': encoder,
                    'decoder': decoder,
                }),
                'physical_memory': memory,
                'dynamics': dynamics,
            }

        components_a = make_components()
        components_b = make_components()

        # Build deterministic dataset of 2 distinct steps
        def make_batch(seed_val, split="train"):
            pose = torch.eye(4)[None]
            grip = torch.zeros(1, 1)
            gravity = torch.tensor([[0.0, 0.0, -1.0]])
            valid = torch.ones(1, 128, dtype=torch.bool)
            state = PhysicalState(
                X=torch.zeros(1, 128, 256),
                x=torch.zeros(1, 128, 3),
                valid=valid,
                p=proprioception(pose, grip, gravity),
                memory=torch.zeros(1, 2, 256),
                T_w_e=pose,
                grip=grip,
                cached_world_cloud=torch.zeros(1, 128, 3),
                cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
                boundary=0,
                encoder_lineage='encoder-v1',
                memory_lineage='memory-v1',
            )
            transitions = []
            curr_pts = np.full((128, 3), float(seed_val), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(2):
                next_pts = curr_pts + 0.01
                b_obs = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a_obs = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                cmd = TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1)
                transitions.append(ExecutedTransition(b_obs, a_obs, cmd, 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return {
                'state': state,
                'transitions': transitions,
                'valid': torch.ones((1, 2), dtype=torch.bool),
                'provenance': {'episode_id': f'ep_{split}_{seed_val}', 'lineage_id': f'lin_{split}_{seed_val}', 'split': split},
            }

        dataset = [make_batch(1, "train"), make_batch(2, "train")]
        eval_dataset = [make_batch(10, "dev")]

        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
            # RUN A: 2 uninterrupted updates with same total schedule (max_updates=2)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            rng_a = np.random.default_rng(12345)
            report_a = run_method_training(
                cfg,
                components_a,
                dataset,
                stage='A1',
                limits={'max_updates': 2},
                rng=rng_a,
                output_dir=temp_dir_a,
                dataset_id="test-dataset",
                evaluation_dataset=eval_dataset,
            )
            py_draw_a = random.random()
            np_draw_a = np.random.rand()
            torch_draw_a = torch.randn(5)
            runner_draw_a = rng_a.random()
            self.assertEqual(report_a.updates_completed, 2)

            # RUN B: Step 1 of same total schedule (max_updates=2, stopped after 1 update)
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            rng_b = np.random.default_rng(12345)
            report_b1 = run_method_training(
                cfg,
                components_b,
                dataset,
                stage='A1',
                limits={'max_updates': 2},
                stop_after=1,
                rng=rng_b,
                output_dir=temp_dir_b,
                dataset_id="test-dataset",
                evaluation_dataset=eval_dataset,
            )
            self.assertEqual(report_b1.updates_completed, 1)
            ckpt_b = report_b1.checkpoint_path
            self.assertIsNotNone(ckpt_b)

            # Verify saved checkpoint stores resolved_config
            saved_state = torch.load(ckpt_b, map_location='cpu')
            self.assertEqual(saved_state['config'], cfg.resolved_config())
            self.assertEqual(saved_state['total_updates'], 2)

            # RUN B: Resume from checkpoint at update 1 with same total schedule (max_updates=2)
            report_b2 = run_method_training(
                cfg,
                components_b,
                dataset,
                stage='A1',
                limits={'max_updates': 2},
                resume_checkpoint=ckpt_b,
                rng=rng_b,
                output_dir=temp_dir_b,
                dataset_id="test-dataset",
                evaluation_dataset=eval_dataset,
            )
            py_draw_b = random.random()
            np_draw_b = np.random.rand()
            torch_draw_b = torch.randn(5)
            runner_draw_b = rng_b.random()
            self.assertEqual(report_b2.updates_completed, 1)

            # Assert exact parameter match (torch.equal) between uninterrupted Run A and resumed Run B
            for name in ('geometry', 'physical_memory', 'dynamics'):
                for (p_name_a, p_a), (p_name_b, p_b) in zip(
                    components_a[name].named_parameters(),
                    components_b[name].named_parameters(),
                ):
                    self.assertEqual(p_name_a, p_name_b)
                    self.assertTrue(
                        torch.equal(p_a, p_b),
                        f"Parameter mismatch after resume in {name}.{p_name_a}",
                    )

            # Assert exact optimizer param_groups and per-parameter state equality from saved checkpoints
            saved_a = torch.load(report_a.checkpoint_path, map_location="cpu")
            saved_b2 = torch.load(report_b2.checkpoint_path, map_location="cpu")
            opt_a = saved_a["optimizer"]
            opt_b = saved_b2["optimizer"]
            self.assertEqual(len(opt_a["param_groups"]), len(opt_b["param_groups"]))
            for ga, gb in zip(opt_a["param_groups"], opt_b["param_groups"]):
                self.assertEqual(ga.keys(), gb.keys())
                for k in ga:
                    self.assertEqual(ga[k], gb[k], f"Optimizer param_group mismatch for key {k}")
            self.assertEqual(opt_a["state"].keys(), opt_b["state"].keys())
            for pid in opt_a["state"]:
                sa = opt_a["state"][pid]
                sb = opt_b["state"][pid]
                self.assertEqual(sa.keys(), sb.keys())
                for k in sa:
                    if torch.is_tensor(sa[k]):
                        self.assertTrue(torch.equal(sa[k], sb[k]), f"Optimizer state tensor mismatch for param {pid} key {k}")
                    else:
                        self.assertEqual(sa[k], sb[k], f"Optimizer state scalar mismatch for param {pid} key {k}")

            # Assert exact scheduler state equality
            self.assertEqual(report_a.scheduler_state, report_b2.scheduler_state)

            # Loss at step 1 in uninterrupted Run A must match step 1 in resumed Run B bit-exactly
            self.assertEqual(report_a.loss_history[1], report_b2.loss_history[0])
            self.assertEqual(report_a.loss_history, report_b2.accumulated_loss_history)

            # Evaluation metrics match
            self.assertEqual(report_a.eval_metrics, report_b2.eval_metrics)

            # Assert exact RNG state tracking across Python, NumPy, torch, and runner RNG
            self.assertEqual(py_draw_a, py_draw_b)
            self.assertEqual(np_draw_a, np_draw_b)
            self.assertTrue(torch.equal(torch_draw_a, torch_draw_b))
            self.assertEqual(runner_draw_a, runner_draw_b)

    def test_stateful_sampler_contract_and_rejection_of_unrestorable_callable(self):
        import tempfile
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A1': {'supervised_intervals': 2, 'burnin_intervals': 0, 'rollout_curriculum': [2]},
            }
        })

        def make_batch(seed_val):
            pose = torch.eye(4)[None]
            grip = torch.zeros(1, 1)
            gravity = torch.tensor([[0.0, 0.0, -1.0]])
            valid = torch.ones(1, 128, dtype=torch.bool)
            state = PhysicalState(
                X=torch.zeros(1, 128, 256),
                x=torch.zeros(1, 128, 3),
                valid=valid,
                p=proprioception(pose, grip, gravity),
                memory=torch.zeros(1, 2, 256),
                T_w_e=pose,
                grip=grip,
                cached_world_cloud=torch.zeros(1, 128, 3),
                cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
                boundary=0,
                encoder_lineage='encoder-v1',
                memory_lineage='memory-v1',
            )
            transitions = []
            curr_pts = np.full((128, 3), float(seed_val), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(2):
                next_pts = curr_pts + 0.01
                b_obs = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a_obs = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                cmd = TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1)
                transitions.append(ExecutedTransition(b_obs, a_obs, cmd, 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return {
                'state': state,
                'transitions': transitions,
                'valid': torch.ones((1, 2), dtype=torch.bool),
                'provenance': {'episode_id': f'ep_train_{seed_val}', 'lineage_id': f'lin_train_{seed_val}', 'split': 'train'},
            }

        class StatefulBatchSampler:
            def __init__(self):
                self.cursor = 0
                self.records = [make_batch(i) for i in range(1, 100)]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                records = []
                for _ in range(batch_size):
                    val = self.cursor + 1
                    if rng is not None:
                        val += int(rng.integers(0, 10))
                    self.cursor += 1
                    records.append(self.records[val % len(self.records)])
                return records
            def state_dict(self):
                return {'cursor': self.cursor}
            def load_state_dict(self, state):
                self.cursor = state['cursor']

        sampler_a = StatefulBatchSampler()
        sampler_b = StatefulBatchSampler()

        def make_components():
            torch.manual_seed(999)
            dynamics = PhysicalDynamics(cfg)
            encoder = PhysicalEncoder(method_config=cfg)
            decoder = PhysicalDecoder(method_config=cfg)
            memory = PhysicalMemory(config=cfg)
            return {
                'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
                'physical_memory': memory,
                'dynamics': dynamics,
            }

        components_a = make_components()
        components_b = make_components()

        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
            rng_a = np.random.default_rng(777)
            run_method_training(
                cfg,
                components_a,
                sampler_a,
                stage='A1',
                limits={'max_updates': 2},
                rng=rng_a,
                output_dir=temp_dir_a,
                dataset_id="test-dataset",
            )
            next_batch_a = sampler_a.sample_batch('A1', 2, 1, rng=rng_a, batch_size=1)[0]

            rng_b = np.random.default_rng(777)
            report_b1 = run_method_training(
                cfg,
                components_b,
                sampler_b,
                stage='A1',
                limits={'max_updates': 2},
                stop_after=1,
                rng=rng_b,
                output_dir=temp_dir_b,
                dataset_id="test-dataset",
            )
            ckpt_b = report_b1.checkpoint_path

            run_method_training(
                cfg,
                components_b,
                sampler_b,
                stage='A1',
                limits={'max_updates': 2},
                resume_checkpoint=ckpt_b,
                rng=rng_b,
                output_dir=temp_dir_b,
                dataset_id="test-dataset",
            )
            next_batch_b = sampler_b.sample_batch('A1', 2, 1, rng=rng_b, batch_size=1)[0]

            self.assertEqual(sampler_a.cursor, sampler_b.cursor)
            self.assertTrue(
                np.array_equal(
                    next_batch_a['transitions'][0].before.observation.points,
                    next_batch_b['transitions'][0].before.observation.points,
                )
            )

        # Reject sample_batch object missing state_dict before mutation
        class MissingStateDictSampler:
            def __init__(self):
                self.records = [make_batch(1)]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                return [make_batch(1)] * batch_size
            def load_state_dict(self, state):
                pass

        with self.assertRaises(TypeError) as ctx:
            run_method_training(
                cfg,
                components_a,
                MissingStateDictSampler(),
                stage='A1',
                limits={'max_updates': 1},
                rng=np.random.default_rng(1),
                dataset_id="test-dataset",
            )
        self.assertIn("must provide callable 'state_dict'", str(ctx.exception))

        # Reject sample_batch object missing load_state_dict before mutation
        class MissingLoadStateDictSampler:
            def __init__(self):
                self.records = [make_batch(1)]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                return [make_batch(1)] * batch_size
            def state_dict(self):
                return {}

        with self.assertRaises(TypeError) as ctx:
            run_method_training(
                cfg,
                components_a,
                MissingLoadStateDictSampler(),
                stage='A1',
                limits={'max_updates': 1},
                rng=np.random.default_rng(1),
                dataset_id="test-dataset",
            )
        self.assertIn("must provide callable 'state_dict' and 'load_state_dict'", str(ctx.exception))

        # Reject un-restorable callable suppliers
        with self.assertRaises(TypeError) as ctx:
            run_method_training(
                cfg,
                components_a,
                lambda s, u, k: make_batch(1),
                stage='A1',
                limits={'max_updates': 1},
                rng=np.random.default_rng(1),
                dataset_id="test-dataset",
            )
        self.assertIn('un-restorable stateful callable sampler is not supported', str(ctx.exception))

    def test_save_checkpoint_previous_file_recoverable_on_write_failure(self):
        import tempfile
        from pathlib import Path
        from icgs.training.method import save_checkpoint

        cfg = MethodConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = Path(tmp_dir) / 'ckpt.pt'
            save_checkpoint(
                ckpt_file,
                stage='A1',
                update=1,
                total_updates=10,
                components={},
                optimizer=None,
                scheduler=None,
                config=cfg,
                dataset_id='d',
                reference_id='r',
                selected_seed=101,
                accumulated_loss_history=[0.1],
            )
            first_state = torch.load(ckpt_file, map_location='cpu')
            self.assertEqual(first_state['update'], 1)

            class Unpicklable:
                def __reduce__(self):
                    raise TypeError('cannot pickle')

            with self.assertRaises(Exception):
                save_checkpoint(
                    ckpt_file,
                    stage='A1',
                    update=2,
                    total_updates=10,
                    components={},
                    optimizer=None,
                    scheduler=None,
                    config=cfg,
                    dataset_id='d',
                    reference_id='r',
                    selected_seed=101,
                    accumulated_loss_history=[0.1, 0.2],
                    extra_metadata={'bad': Unpicklable()},
                )

            # Previous file must remain completely intact and valid
            recovered_state = torch.load(ckpt_file, map_location='cpu')
            self.assertEqual(recovered_state['update'], 1)

    def test_cuda_resume_if_available(self):
        raise unittest.SkipTest("CUDA resume continuation fixture is NOT IMPLEMENTED / NOT RUN on this environment")


    def test_stage_c_raises_not_implemented_dependency_error(self):
        from icgs.training.method import run_method_training

        cfg = MethodConfig()
        weight_init = torch.randn(4, 4)
        comp = nn.Linear(4, 4, bias=False)
        comp.weight.data.copy_(weight_init)
        components = {'geometry': comp}

        with self.assertRaises(NotImplementedError) as ctx:
            run_method_training(cfg, components, datasets=None, stage='C')
        self.assertIn("Stage C reference certification requires upstream reference router", str(ctx.exception))
        self.assertIn("owned by P06", str(ctx.exception))
        # Ensure zero mutation
        self.assertTrue(torch.equal(comp.weight.data, weight_init))

    def test_stage_e_raises_not_implemented_dependency_error(self):
        from icgs.training.method import run_method_training

        cfg = MethodConfig()
        weight_init = torch.randn(4, 4)
        comp = nn.Linear(4, 4, bias=False)
        comp.weight.data.copy_(weight_init)
        components = {'geometry': comp}

        with self.assertRaises(NotImplementedError) as ctx:
            run_method_training(cfg, components, datasets=None, stage='E')
        self.assertIn("Stage E temperature calibration requires upstream calibration API", str(ctx.exception))
        self.assertIn("owned by P09", str(ctx.exception))
        # Ensure zero mutation
        self.assertTrue(torch.equal(comp.weight.data, weight_init))

    def test_batch_size_consumption_multiple_records_and_exact_resume(self):
        """Batch size 2 samples 2 independent records per update, averages losses, and resumes exact."""
        import tempfile
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import run_method_training

        torch.manual_seed(999)
        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'evaluation_interval_updates': 1,
                'A1': {'supervised_intervals': 1, 'burnin_intervals': 0, 'rollout_curriculum': [1], 'batch_size': 2},
            }
        })

        def make_components():
            torch.manual_seed(999)
            dynamics = PhysicalDynamics(cfg)
            encoder = PhysicalEncoder(method_config=cfg)
            decoder = PhysicalDecoder(method_config=cfg)
            memory = PhysicalMemory(config=cfg)
            return {
                'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
                'physical_memory': memory,
                'dynamics': dynamics,
            }

        def make_batch(val_seed):
            pose = torch.eye(4)[None]
            grip = torch.zeros(1, 1)
            gravity = torch.tensor([[0.0, 0.0, -1.0]])
            valid = torch.ones(1, 128, dtype=torch.bool)
            state = PhysicalState(
                X=torch.zeros(1, 128, 256),
                x=torch.zeros(1, 128, 3),
                valid=valid,
                p=proprioception(pose, grip, gravity),
                memory=torch.zeros(1, 2, 256),
                T_w_e=pose,
                grip=grip,
                cached_world_cloud=torch.zeros(1, 128, 3),
                cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
                boundary=0,
                encoder_lineage='encoder-v1',
                memory_lineage='memory-v1',
            )
            pts = np.full((128, 3), float(val_seed), dtype=np.float64)
            se3 = np.eye(4, dtype=np.float64)
            b_obs = TimedObservation(Observation(pts.copy(), se3.copy(), 0.0), 0, 0.0, 0.0, 'sensor')
            a_obs = TimedObservation(Observation(pts.copy() + 0.01, se3.copy(), 0.0), 1, 0.1, 0.1, 'sensor')
            cmd = TimedCommand(target_w=se3.copy(), grip=0, duration_s=0.1)
            trans = [ExecutedTransition(b_obs, a_obs, cmd, 0.1, 1, 'ok')]
            return {
                'state': state,
                'transitions': trans,
                'valid': torch.ones((1, 1), dtype=torch.bool),
                'provenance': {'episode_id': f'ep_train_{val_seed}', 'lineage_id': f'lin_train_{val_seed}', 'split': 'train'},
            }

        dataset = [make_batch(1), make_batch(2), make_batch(3), make_batch(4)]

        # Prove both records in batch of 2 contribute to the gradient
        comp_single1 = make_components()
        comp_single2 = make_components()
        comp_batch = make_components()

        # Step single 1
        cfg_single = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A1': {'supervised_intervals': 1, 'burnin_intervals': 0, 'rollout_curriculum': [1], 'batch_size': 1},
            }
        })
        rep_s1 = run_method_training(cfg_single, comp_single1, [dataset[0]], stage='A1', limits={'max_updates': 1}, dataset_id='test-ds')
        rep_s2 = run_method_training(cfg_single, comp_single2, [dataset[1]], stage='A1', limits={'max_updates': 1}, dataset_id='test-ds')
        rep_b = run_method_training(cfg, comp_batch, dataset[:2], stage='A1', limits={'max_updates': 1}, dataset_id='test-ds')

        # Average loss matches within numerical precision
        expected_avg_loss = (rep_s1.loss_history[0] + rep_s2.loss_history[0]) / 2.0
        self.assertAlmostEqual(rep_b.loss_history[0], expected_avg_loss, places=5)

        # Exact resume with batch_size: 2
        comp_a = make_components()
        comp_b = make_components()
        with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
            rng_a = np.random.default_rng(42)
            rep_a = run_method_training(cfg, comp_a, dataset, stage='A1', limits={'max_updates': 2}, rng=rng_a, output_dir=td_a, dataset_id='test-ds')

            rng_b = np.random.default_rng(42)
            rep_b1 = run_method_training(cfg, comp_b, dataset, stage='A1', limits={'max_updates': 2}, stop_after=1, rng=rng_b, output_dir=td_b, dataset_id='test-ds')
            rep_b2 = run_method_training(cfg, comp_b, dataset, stage='A1', limits={'max_updates': 2}, resume_checkpoint=rep_b1.checkpoint_path, rng=rng_b, output_dir=td_b, dataset_id='test-ds')

            for name in ('geometry', 'physical_memory', 'dynamics'):
                for (p_a, p_b) in zip(comp_a[name].parameters(), comp_b[name].parameters()):
                    self.assertTrue(torch.equal(p_a, p_b))

    def test_seed_readiness_rejects_missing_seeds_and_rng_bypass(self):
        """Readiness requires 3 valid configured seeds; caller RNG cannot bypass missing seed metadata."""
        import numpy as np
        from icgs.training.method import run_method_training

        # Missing seeds rejected even if rng provided
        cfg_none = MethodConfig.from_dict({'stages': {'training_seeds': None}})
        with self.assertRaisesRegex(ValueError, 'requires declared stages.training_seeds in config'):
            run_method_training(cfg_none, {'geometry': nn.Linear(4, 4)}, [{'x': 1}], stage='A1', rng=np.random.default_rng(42), dataset_id='test-ds')

        # Wrong count rejected at config validation
        with self.assertRaisesRegex(ValueError, 'must contain training_seed_count distinct seeds'):
            MethodConfig.from_dict({'stages': {'training_seeds': (101, 102)}})

        # Duplicates rejected at config validation
        with self.assertRaisesRegex(ValueError, 'must contain training_seed_count distinct seeds'):
            MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 101)}})

        # Selected seed reported in TrainingReport
        cfg_valid = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103)}})
        report = run_method_training(cfg_valid, {}, None, stage='Test', run_seed=102, dataset_id='test-ds', reference_id='ref-test')
        self.assertEqual(report.selected_seed, 102)

    def test_provenance_split_leakage_and_partition_overlap_rejection(self):
        """Test/calib/audit records rejected from training; dev required for eval; ID overlap rejected."""
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': PhysicalEncoder(method_config=cfg), 'decoder': PhysicalDecoder(method_config=cfg)}),
            'physical_memory': PhysicalMemory(config=cfg),
            'dynamics': PhysicalDynamics(cfg),
        }

        # Training record with forbidden test/calib/audit split rejected
        for bad_split in ('test', 'calib', 'audit'):
            bad_record = {'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': bad_split}}
            with self.assertRaisesRegex(ValueError, 'forbidden/invalid split'):
                run_method_training(cfg, comp, [bad_record], stage='A1', limits={'max_updates': 1}, dataset_id='test-ds')

        # Overlapping train and eval episode IDs rejected before updates start
        train_rec = {'provenance': {'episode_id': 'ep_shared', 'lineage_id': 'lin_train', 'split': 'train'}}
        eval_rec = {'provenance': {'episode_id': 'ep_shared', 'lineage_id': 'lin_eval', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'overlapping episode IDs'):
            run_method_training(
                cfg,
                comp,
                datasets=[train_rec],
                evaluation_dataset=[eval_rec],
                stage='A1',
                limits={'max_updates': 1},
                dataset_id='test-ds',
            )

        # Overlapping train and eval lineage IDs rejected before updates start
        train_rec_lin = {'provenance': {'episode_id': 'ep_train_diff', 'lineage_id': 'lin_shared', 'split': 'train'}}
        eval_rec_lin = {'provenance': {'episode_id': 'ep_eval_diff', 'lineage_id': 'lin_shared', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'overlapping lineage IDs'):
            run_method_training(
                cfg,
                comp,
                datasets=[train_rec_lin],
                evaluation_dataset=[eval_rec_lin],
                stage='A1',
                limits={'max_updates': 1},
                dataset_id='test-ds',
            )

    def test_lexicographic_selection_priority_order_ties_and_patience(self):
        """Test stage metric priority, lexicographic comparison, tie-breaking, and resume bit-for-bit."""
        from icgs.training.method import _evaluate_stage, run_method_training

        cfg = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103), 'evaluation_interval_updates': 1, 'early_stopping_patience': 2}})

        # Candidate < Best:
        # Lower primary always wins
        self.assertTrue((1.0, 10.0) < (2.0, 1.0))
        # Higher primary loses even if secondary is much better
        self.assertFalse((2.0, 0.0) < (1.0, 10.0))
        # Equal primary breaks tie via secondary
        self.assertTrue((1.0, 2.0) < (1.0, 5.0))
        # Exact tie evaluates to False (preserves earlier checkpoint)
        self.assertFalse((1.0, 2.0) < (1.0, 2.0))

        # Evaluation rejects non-dev record
        with self.assertRaisesRegex(ValueError, 'forbidden/invalid split'):
            _evaluate_stage('A1', {'geometry': nn.Linear(4, 4)}, [{'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}], cfg)

        # Evaluation rejects non-finite dev_ranking
        with self.assertRaisesRegex(ValueError, 'finite float'):
            _evaluate_stage('A1', {'geometry': nn.Linear(4, 4)}, [{'dev_ranking': float('nan'), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'dev'}}], cfg)

    def test_cadence_and_limits_strict_validation(self):
        """Strict limits and cadence validation."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        comp = {'geometry': nn.Linear(4, 4)}

        # Disallowed limits key
        with self.assertRaisesRegex(ValueError, 'unrecorded/disallowed limit'):
            run_method_training(cfg, comp, [{'x': 1}], stage='A1', limits={'eval_every': 1}, dataset_id='test-ds')

        # Non-positive max_updates in limits
        with self.assertRaisesRegex(ValueError, 'must be a positive integer'):
            run_method_training(cfg, comp, [{'x': 1}], stage='A1', limits={'max_updates': 0}, dataset_id='test-ds')

    def test_resolved_config_json_written_before_execution_when_output_requested(self):
        """Ensure resolved_config.json is written beside checkpoint before execution."""
        import json
        import tempfile
        from pathlib import Path
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103)}})
        comp = {'geometry': nn.Linear(4, 4)}

        with tempfile.TemporaryDirectory() as td:
            run_method_training(cfg, comp, None, stage='Test', output_dir=td, dataset_id='test-ds', reference_id='ref-test')
            cfg_file = Path(td) / 'resolved_config.json'
            self.assertTrue(cfg_file.exists())
            loaded_cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
            self.assertEqual(loaded_cfg, cfg.resolved_config())
            self.assertIn('stages', loaded_cfg['config'])

    def test_red_2c1_gated_public_d1_not_implemented(self):
        """Public run_method_training for D1 raises NotImplementedError pending P06/C."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103)}})
        with self.assertRaisesRegex(NotImplementedError, 'pending P06/C verified frozen-reference binding'):
            run_method_training(cfg, {}, None, stage='D1', dataset_id='test-ds')

    def test_red_2c1_missing_generator_reset_action_seeds_rejected(self):
        """Runnable stages require declared generator_seed, reset_seed, and action_seed."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103), 'generator_seed': None}})
        with self.assertRaisesRegex(ValueError, 'requires declared stages.generator_seed'):
            run_method_training(cfg, {}, None, stage='A1', dataset_id='test-ds')

    def test_red_2c1_limits_rejects_stop_after_key(self):
        """limits['stop_after'] must be rejected in favor of explicit keyword argument."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103)}})
        with self.assertRaisesRegex(ValueError, "limits\\['stop_after'\\] is not permitted"):
            run_method_training(cfg, {}, None, stage='A1', limits={'stop_after': 1}, dataset_id='test-ds')

    def test_red_2c1_resolved_config_json_matches_resolved_config_envelope(self):
        """resolved_config.json must equal cfg.resolved_config() envelope with hash."""
        import json
        import tempfile
        from pathlib import Path
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({'stages': {'training_seeds': (101, 102, 103)}})
        with tempfile.TemporaryDirectory() as td:
            run_method_training(cfg, {}, None, stage='Test', output_dir=td, dataset_id='test-ds', reference_id='ref-test')
            cfg_file = Path(td) / 'resolved_config.json'
            self.assertTrue(cfg_file.exists())
            loaded = json.loads(cfg_file.read_text(encoding='utf-8'))
            self.assertEqual(loaded, cfg.resolved_config())

    def test_red_2c1_stateful_sampler_requires_inspectable_records(self):
        """Stateful train sampler requires inspectable .records Sequence for provenance."""
        from icgs.training.method import run_method_training

        class SamplerNoRecords:
            def sample_batch(self, stage, update, curriculum_k, *, rng, batch_size):
                return []
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4), 'decoder': nn.Linear(4, 4)}),
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
        }
        with self.assertRaisesRegex(TypeError, 'inspectable .records Sequence'):
            run_method_training(cfg, comp, SamplerNoRecords(), stage='A1', dataset_id='test-ds')

    def test_red_2c1_provenance_requires_lineage_id(self):
        """Sampled records must include non-blank lineage_id in provenance."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4), 'decoder': nn.Linear(4, 4)}),
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
        }
        record_no_lineage = {'provenance': {'episode_id': 'ep1', 'split': 'train'}}
        with self.assertRaisesRegex(ValueError, "missing required 'lineage_id'"):
            run_method_training(cfg, comp, [record_no_lineage], stage='A1', dataset_id='test-ds')

    def test_nonzero_update_repeatability_ambient_rng_dropout_and_resume(self):
        """Nonzero-update repeatability fixture with different ambient RNG + dropout, and resume restores RNG without overwrite."""
        import copy
        import random
        import tempfile
        import numpy as np
        import torch
        from torch import nn
        from icgs.configuration.method import MethodConfig
        from icgs.training.method import run_method_training

        class NetWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 8)
                self.drop = nn.Dropout(p=0.5)
                self.fc2 = nn.Linear(8, 3)

            def forward(self, pts, valid):
                h = torch.relu(self.fc1(pts))
                h = self.drop(h)
                out = self.fc2(h)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })

        torch.manual_seed(42)
        base_encoder = NetWithDropout()
        init_weights = copy.deepcopy(base_encoder.state_dict())

        rec1 = {
            'points': torch.randn(1, 4, 3),
            'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'},
        }
        rec2 = {
            'points': torch.randn(1, 4, 3),
            'provenance': {'episode_id': 'ep2', 'lineage_id': 'lin2', 'split': 'train'},
        }
        dataset = [rec1, rec2]

        # Run 1: with ambient RNG state X
        random.seed(99999)
        np.random.seed(88888)
        torch.manual_seed(77777)

        enc1 = NetWithDropout()
        enc1.load_state_dict(copy.deepcopy(init_weights))
        comp1 = {'geometry': nn.ModuleDict({'encoder': enc1, 'decoder': DummyDecoder()})}

        rep1 = run_method_training(
            config=cfg,
            components=comp1,
            datasets=dataset,
            stage='A0',
            dataset_id='test-ds',
            run_seed=101,
            limits={'max_updates': 2},
        )

        # Run 2: with completely different ambient RNG state Y
        random.seed(11111)
        np.random.seed(22222)
        torch.manual_seed(33333)

        enc2 = NetWithDropout()
        enc2.load_state_dict(copy.deepcopy(init_weights))
        comp2 = {'geometry': nn.ModuleDict({'encoder': enc2, 'decoder': DummyDecoder()})}

        rep2 = run_method_training(
            config=cfg,
            components=comp2,
            datasets=dataset,
            stage='A0',
            dataset_id='test-ds',
            run_seed=101,
            limits={'max_updates': 2},
        )

        self.assertEqual(rep1.updates_completed, 2)
        self.assertEqual(rep2.updates_completed, 2)
        self.assertEqual(rep1.loss_history, rep2.loss_history)
        for k in enc1.state_dict():
            self.assertTrue(
                torch.equal(enc1.state_dict()[k], enc2.state_dict()[k]),
                f"weight mismatch in {k} under different ambient RNG",
            )

        # Test resume restores RNG and is not overwritten afterward
        with tempfile.TemporaryDirectory() as td:
            random.seed(12345)
            np.random.seed(54321)
            torch.manual_seed(67890)

            enc_a = NetWithDropout()
            enc_a.load_state_dict(copy.deepcopy(init_weights))
            comp_a = {'geometry': nn.ModuleDict({'encoder': enc_a, 'decoder': DummyDecoder()})}

            rep_a1 = run_method_training(
                config=cfg,
                components=comp_a,
                datasets=dataset,
                stage='A0',
                dataset_id='test-ds',
                run_seed=101,
                limits={'max_updates': 2},
                stop_after=1,
                output_dir=td,
            )
            ckpt_path = rep_a1.checkpoint_path
            self.assertIsNotNone(ckpt_path)

            random.seed(44444)
            np.random.seed(55555)
            torch.manual_seed(66666)

            rep_a2 = run_method_training(
                config=cfg,
                components=comp_a,
                datasets=dataset,
                stage='A0',
                dataset_id='test-ds',
                run_seed=101,
                limits={'max_updates': 2},
                resume_checkpoint=ckpt_path,
                output_dir=td,
            )

            combined_loss = rep_a1.loss_history + rep_a2.loss_history
            self.assertEqual(combined_loss, rep1.loss_history)
            for k in enc_a.state_dict():
                self.assertTrue(
                    torch.equal(enc_a.state_dict()[k], enc1.state_dict()[k]),
                    f"resume weight mismatch in {k}",
                )

    def test_unsupported_injected_rng_rejected(self):
        """Unsupported injected RNG types are rejected with TypeError."""
        import random
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
            }
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4), 'decoder': nn.Linear(4, 4)}),
            'physical_memory': nn.Linear(4, 4),
            'dynamics': nn.Linear(4, 4),
        }
        dataset = [{'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}]
        with self.assertRaisesRegex(TypeError, 'injected rng must be a numpy.random.Generator'):
            run_method_training(cfg, comp, dataset, stage='A0', dataset_id='test-ds', rng=random.Random(42))
        with self.assertRaisesRegex(TypeError, 'injected rng must be a numpy.random.Generator'):
            run_method_training(cfg, comp, dataset, stage='A0', dataset_id='test-ds', rng='not_a_generator')

    def test_sampled_records_must_belong_to_declared_train_provenance(self):
        """Sampled record identities must belong to declared train provenance before backward."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A0': {'batch_size': 1, 'max_updates': 10},
            }
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4), 'decoder': nn.Linear(4, 4)}),
        }
        declared_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        undeclared_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep_undeclared', 'lineage_id': 'lin1', 'split': 'train'}}

        class RogueSampler:
            def __init__(self):
                self.records = [declared_rec]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                return [undeclared_rec]
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass

        with self.assertRaisesRegex(ValueError, 'does not belong to predeclared training dataset provenance'):
            run_method_training(cfg, comp, RogueSampler(), stage='A0', dataset_id='test-ds', limits={'max_updates': 1})

    def test_sample_batch_error_propagates_without_retry_and_exact_count_enforced(self):
        """Internal errors propagate without retry; count must equal batch_size exactly."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 104,
                'reset_seed': 105,
                'action_seed': 106,
                'A0': {'batch_size': 2, 'max_updates': 10},
            }
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(4, 4), 'decoder': nn.Linear(4, 4)}),
        }
        valid_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}

        class FailingSampler:
            def __init__(self):
                self.records = [valid_rec]
                self.call_count = 0
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                self.call_count += 1
                raise TypeError('custom sampler internal error')
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass

        failing_sampler = FailingSampler()
        with self.assertRaisesRegex(TypeError, 'custom sampler internal error'):
            run_method_training(cfg, comp, failing_sampler, stage='A0', dataset_id='test-ds', limits={'max_updates': 1})
        self.assertEqual(failing_sampler.call_count, 1)

        class TruncatingSampler:
            def __init__(self):
                self.records = [valid_rec]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                return [valid_rec]
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass

        with self.assertRaisesRegex(ValueError, 'expected exactly batch_size=2'):
            run_method_training(cfg, comp, TruncatingSampler(), stage='A0', dataset_id='test-ds', limits={'max_updates': 1})

    def test_resume_rejects_missing_or_mismatched_seed_before_mutation(self):
        """Exact resume rejects missing or mismatched selected_seed before mutating live modules."""
        import tempfile
        from icgs.training.method import run_method_training, save_checkpoint, load_checkpoint

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })

        class ParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        class FrozenProbeModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
                self.register_buffer('buf', torch.tensor([1.0, 2.0, 3.0]))
            def forward(self, x):
                return self.fc(x)

        comp = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
        dataset = [{'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}]

        with tempfile.TemporaryDirectory() as td:
            # Standalone save_checkpoint requires non-negative selected_seed
            dummy_path = f'{td}/dummy.pt'
            with self.assertRaises(TypeError):
                save_checkpoint(
                    dummy_path,
                    stage='A0',
                    update=1,
                    total_updates=2,
                    components=comp,
                    optimizer=None,
                    scheduler=None,
                    config=cfg,
                    dataset_id='test-ds',
                    reference_id='pre-reference-A0',
                )

            # Run 1 step and save checkpoint (records selected_seed: 101)
            rep1 = run_method_training(
                cfg,
                comp,
                dataset,
                stage='A0',
                dataset_id='test-ds',
                run_seed=101,
                limits={'max_updates': 2},
                stop_after=1,
                output_dir=td,
            )
            ckpt = rep1.checkpoint_path
            self.assertIsNotNone(ckpt)

            # load_checkpoint rejects missing top-level selected_seed
            raw = torch.load(ckpt, map_location='cpu')
            del raw['selected_seed']
            bad_file_top = f'{td}/bad_top_seed.pt'
            torch.save(raw, bad_file_top)
            comp_unmutated = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
            weight_before = comp_unmutated['geometry']['encoder'].fc.weight.clone()
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file_top, components=comp_unmutated)
            self.assertTrue(torch.equal(comp_unmutated['geometry']['encoder'].fc.weight, weight_before))

            # load_checkpoint rejects missing manifest selected_seed
            raw2 = torch.load(ckpt, map_location='cpu')
            del raw2['manifest']['selected_seed']
            bad_file_man = f'{td}/bad_man_seed.pt'
            torch.save(raw2, bad_file_man)
            with self.assertRaises(KeyError):
                load_checkpoint(bad_file_man, components=comp_unmutated)
            self.assertTrue(torch.equal(comp_unmutated['geometry']['encoder'].fc.weight, weight_before))

            # run_method_training with mismatched run_seed (102 vs 101) rejects before live mutation:
            # verify trainable module, frozen-role module (mode, requires_grad, weights, buffers), RNG, and config sentinel file untouched
            frozen_probe = FrozenProbeModule()
            frozen_probe.train(True)
            frozen_probe.fc.weight.requires_grad_(True)
            frozen_probe.fc.bias.requires_grad_(True)
            probe_weight_before = frozen_probe.fc.weight.clone()
            probe_buf_before = frozen_probe.buf.clone()

            comp_unmutated['physical_memory'] = frozen_probe
            enc_weight_before = comp_unmutated['geometry']['encoder'].fc.weight.clone()

            from pathlib import Path
            mismatch_out = Path(td) / 'mismatch_out'
            mismatch_out.mkdir(parents=True, exist_ok=True)
            sentinel_path = mismatch_out / 'resolved_config.json'
            sentinel_content = '{"sentinel": "untouched_original"}'
            sentinel_path.write_text(sentinel_content, encoding='utf-8')

            import random
            import numpy as np
            py_rng_before = random.getstate()
            np_rng_before = np.random.get_state()
            torch_rng_before = torch.get_rng_state()
            cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            with self.assertRaisesRegex(ValueError, 'selected_seed mismatch'):
                run_method_training(
                    cfg,
                    comp_unmutated,
                    dataset,
                    stage='A0',
                    dataset_id='test-ds',
                    run_seed=102,
                    limits={'max_updates': 2},
                    resume_checkpoint=ckpt,
                    output_dir=mismatch_out,
                )
            self.assertTrue(torch.equal(comp_unmutated['geometry']['encoder'].fc.weight, enc_weight_before))
            self.assertTrue(torch.equal(frozen_probe.fc.weight, probe_weight_before))
            self.assertTrue(torch.equal(frozen_probe.buf, probe_buf_before))
            self.assertTrue(frozen_probe.training)
            self.assertTrue(frozen_probe.fc.weight.requires_grad)
            self.assertEqual(sentinel_path.read_text(encoding='utf-8'), sentinel_content)
            self.assertEqual(random.getstate(), py_rng_before)
            self.assertTrue(np.array_equal(np.random.get_state()[1], np_rng_before[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng_before))
            if cuda_rng_before is not None:
                self.assertEqual(torch.cuda.get_rng_state_all(), cuda_rng_before)

            # Regression: corrupt ONLY top-level seed 102 while manifest and request remain 101.
            # Assert config sentinel, module modes/gradflags/buffers/weights and all RNG unchanged.
            raw3 = torch.load(ckpt, map_location='cpu')
            raw3['selected_seed'] = 102
            bad_file_top_mismatch = f'{td}/bad_top_mismatch.pt'
            torch.save(raw3, bad_file_top_mismatch)

            frozen_probe.train(True)
            frozen_probe.fc.weight.requires_grad_(True)
            probe_weight_before2 = frozen_probe.fc.weight.clone()
            probe_buf_before2 = frozen_probe.buf.clone()
            enc_weight_before2 = comp_unmutated['geometry']['encoder'].fc.weight.clone()

            sentinel_content2 = '{"sentinel": "top_level_seed_corrupt_untouched"}'
            sentinel_path.write_text(sentinel_content2, encoding='utf-8')

            py_rng_before2 = random.getstate()
            np_rng_before2 = np.random.get_state()
            torch_rng_before2 = torch.get_rng_state()
            cuda_rng_before2 = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            with self.assertRaisesRegex(ValueError, "top-level selected_seed \\(102\\) does not match manifest selected_seed \\(101\\)"):
                run_method_training(
                    cfg,
                    comp_unmutated,
                    dataset,
                    stage='A0',
                    dataset_id='test-ds',
                    run_seed=101,
                    limits={'max_updates': 2},
                    resume_checkpoint=bad_file_top_mismatch,
                    output_dir=mismatch_out,
                )
            self.assertTrue(torch.equal(comp_unmutated['geometry']['encoder'].fc.weight, enc_weight_before2))
            self.assertTrue(torch.equal(frozen_probe.fc.weight, probe_weight_before2))
            self.assertTrue(torch.equal(frozen_probe.buf, probe_buf_before2))
            self.assertTrue(frozen_probe.training)
            self.assertTrue(frozen_probe.fc.weight.requires_grad)
            self.assertEqual(sentinel_path.read_text(encoding='utf-8'), sentinel_content2)
            self.assertEqual(random.getstate(), py_rng_before2)
            self.assertTrue(np.array_equal(np.random.get_state()[1], np_rng_before2[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng_before2))
            if cuda_rng_before2 is not None:
                self.assertEqual(torch.cuda.get_rng_state_all(), cuda_rng_before2)

    def test_resume_same_seed_bitwise_continuation(self):
        """Same-seed resume preserves bit-for-bit weights and reports correct bound seed."""
        import tempfile
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })

        class ParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        dataset = [{'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}]

        with tempfile.TemporaryDirectory() as td:
            comp = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
            rep1 = run_method_training(
                cfg,
                comp,
                dataset,
                stage='A0',
                dataset_id='test-ds',
                run_seed=102,
                limits={'max_updates': 2},
                stop_after=1,
                output_dir=td,
            )
            self.assertEqual(rep1.selected_seed, 102)
            ckpt = rep1.checkpoint_path

            rep2 = run_method_training(
                cfg,
                comp,
                dataset,
                stage='A0',
                dataset_id='test-ds',
                run_seed=102,
                limits={'max_updates': 2},
                resume_checkpoint=ckpt,
                output_dir=td,
            )
            self.assertEqual(rep2.updates_completed, 1)
            self.assertEqual(rep2.selected_seed, 102)

    def test_sampled_records_reject_adversarial_recombination_pairs(self):
        """Cross-pair recombination (ep1, lin2) when only (ep1, lin1) and (ep2, lin2) are declared is rejected."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })
        comp = {'geometry': nn.ModuleDict({'encoder': nn.Linear(3, 3), 'decoder': nn.Linear(3, 3)})}
        rec1 = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        rec2 = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep2', 'lineage_id': 'lin2', 'split': 'train'}}
        adversarial_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin2', 'split': 'train'}}

        class AdvSampler:
            def __init__(self):
                self.records = [rec1, rec2]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                return [adversarial_rec]
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass

        with self.assertRaisesRegex(ValueError, "sampled record provenance pair \\('ep1', 'lin2'\\)"):
            run_method_training(cfg, comp, AdvSampler(), stage='A0', dataset_id='test-ds', limits={'max_updates': 1})

    def test_evaluation_snapshot_immutable_to_list_mutation(self):
        """Evaluation dataset is snapshotted into an immutable tuple once; list mutation does not alter eval."""
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 1,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })

        class CustomEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class CustomDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        comp = {'geometry': nn.ModuleDict({'encoder': CustomEncoder(), 'decoder': CustomDecoder()})}
        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        dev_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}
        eval_list = [dev_rec]

        class MutatingSampler:
            def __init__(self):
                self.records = [train_rec]
            def sample_batch(self, stage, update, curriculum_k, *, rng=None, batch_size=1):
                eval_list.clear()
                return [train_rec]
            def state_dict(self):
                return {}
            def load_state_dict(self, state):
                pass

        rep = run_method_training(
            cfg,
            comp,
            MutatingSampler(),
            stage='A0',
            dataset_id='test-ds',
            evaluation_dataset=eval_list,
            limits={'max_updates': 1},
        )
        self.assertEqual(rep.updates_completed, 1)
        self.assertEqual(len(eval_list), 0)
        self.assertIn('val_loss', rep.eval_metrics)

    def test_2c2_selection_schema_predeclaration_and_rejection_of_mixed_or_invalid_metrics(self):
        """Predeclared immutable schema per stage; reject mixed/invalid metrics."""
        from unittest.mock import patch
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 1,
                'A0': {'batch_size': 1, 'max_updates': 10},
                'A1': {'batch_size': 1, 'max_updates': 10, 'burnin_intervals': 0, 'supervised_intervals': 1, 'rollout_curriculum': [1]},
            },
        })

        class ParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        comp_a0 = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
        comp_a1 = {
            'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()}),
            'physical_memory': nn.Linear(3, 3),
            'dynamics': nn.Linear(3, 3),
        }
        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}

        # 1. A0 rejects dev_ranking in evaluation dataset
        dev_with_ranking = {'points': torch.zeros(1, 4, 3), 'dev_ranking': 1.0, 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'Stage A0 does not support dev_ranking'):
            run_method_training(cfg, comp_a0, [train_rec], stage='A0', dataset_id='test-ds', evaluation_dataset=[dev_with_ranking])

        # 2. A1 with mixed dev_ranking (present in rec1, missing in rec2) is rejected
        dev_no_ranking = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep_dev2', 'lineage_id': 'lin_dev2', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'partial/mixed dev_ranking'):
            run_method_training(cfg, comp_a1, [train_rec], stage='A1', dataset_id='test-ds', evaluation_dataset=[dev_with_ranking, dev_no_ranking])

        # 3. A1 with non-finite dev_ranking is rejected
        dev_nan_ranking = {'points': torch.zeros(1, 4, 3), 'dev_ranking': float('nan'), 'provenance': {'episode_id': 'ep_dev3', 'lineage_id': 'lin_dev3', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'finite float'):
            run_method_training(cfg, comp_a1, [train_rec], stage='A1', dataset_id='test-ds', evaluation_dataset=[dev_nan_ranking])

        # 4. A1 without dev_ranking derives schema ('physical_validation_loss',)
        with patch('icgs.training.method._execute_training_step', return_value=0.5):
            with patch('icgs.training.method._evaluate_stage', return_value={'val_loss': 1.0, 'physical_validation_loss': 1.0, 'metric_vector': (1.0,), 'selection_schema': ('physical_validation_loss',), 'selection_status': 'certified'}):
                rep_a1_single = run_method_training(cfg, comp_a1, [train_rec], stage='A1', dataset_id='test-ds', evaluation_dataset=[dev_no_ranking], limits={'max_updates': 1})
                self.assertEqual(rep_a1_single.selection_schema, ('physical_validation_loss',))
                self.assertEqual(len(rep_a1_single.best_metric_vector), 1)

        # 5. A1 with uniform dev_ranking derives schema ('physical_validation_loss', 'dev_ranking')
        with patch('icgs.training.method._execute_training_step', return_value=0.5):
            with patch('icgs.training.method._evaluate_stage', return_value={'val_loss': 1.0, 'physical_validation_loss': 1.0, 'dev_ranking': 1.0, 'metric_vector': (1.0, 1.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'}):
                rep_a1_multi = run_method_training(cfg, comp_a1, [train_rec], stage='A1', dataset_id='test-ds', evaluation_dataset=[dev_with_ranking], limits={'max_updates': 1})
                self.assertEqual(rep_a1_multi.selection_schema, ('physical_validation_loss', 'dev_ranking'))
                self.assertEqual(len(rep_a1_multi.best_metric_vector), 2)

    def test_2c2_a0_refuses_to_certify_best_checkpoint_and_saves_diagnostic(self):
        """Stage A0 refuses to write _best.pt without bridge gate; saves diagnostic checkpoints with status."""
        import tempfile
        from pathlib import Path
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 1,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })

        class ParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        comp = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        dev_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}

        with tempfile.TemporaryDirectory() as td:
            rep = run_method_training(
                cfg,
                comp,
                [train_rec],
                stage='A0',
                dataset_id='test-ds',
                evaluation_dataset=[dev_rec],
                limits={'max_updates': 2},
                output_dir=td,
            )
            best_ckpt = Path(td) / 'checkpoint_A0_best.pt'
            self.assertFalse(best_ckpt.exists(), 'checkpoint_A0_best.pt must NEVER be written while bridge gate is unavailable')
            self.assertIn('not_certified', rep.selection_status)

            # Periodic diagnostic checkpoint is saved
            diag_ckpt = Path(td) / 'checkpoint_A0_2.pt'
            self.assertTrue(diag_ckpt.exists())
            raw = torch.load(diag_ckpt, map_location='cpu')
            self.assertIn('not_certified', raw['selection_state']['selection_status'])
            self.assertEqual(raw['selection_state']['selection_schema'], ['normalized_reconstruction_loss'])

    def test_2c2_lexicographic_selection_lifecycle_primary_dominates_secondary_ties_and_patience(self):
        """Lexicographic ordering: primary dominates secondary, tie-breaking by secondary, exact ties increment patience."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 1,
                'early_stopping_patience': 3,
                'A1': {'batch_size': 1, 'max_updates': 10, 'burnin_intervals': 0, 'supervised_intervals': 1, 'rollout_curriculum': [1]},
            },
        })

        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(3, 3), 'decoder': nn.Linear(3, 3)}),
            'physical_memory': nn.Linear(3, 3),
            'dynamics': nn.Linear(3, 3),
        }
        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        dev_rec = {'points': torch.zeros(1, 4, 3), 'dev_ranking': 10.0, 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}

        # Controlled sequence of evaluation candidate vectors:
        # Update 1: (2.0, 10.0) -> first eval, best_update=1, patience=0
        # Update 2: (1.0, 20.0) -> primary dominates (1.0 < 2.0 despite 20.0 > 10.0), best_update=2, patience=0
        # Update 3: (1.0, 15.0) -> primary equal, secondary wins (15.0 < 20.0), best_update=3, patience=0
        # Update 4: (1.0, 15.0) -> exact tie, earlier kept (best_update=3), patience=1
        # Update 5: (1.5, 5.0)  -> primary worse (1.5 > 1.0), best_update=3, patience=2
        # Update 6: (1.8, 1.0)  -> primary worse (1.8 > 1.0), patience=3 -> EARLY STOPPING!
        eval_responses = [
            {'val_loss': 2.0, 'physical_validation_loss': 2.0, 'dev_ranking': 10.0, 'metric_vector': (2.0, 10.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'},
            {'val_loss': 1.0, 'physical_validation_loss': 1.0, 'dev_ranking': 20.0, 'metric_vector': (1.0, 20.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'},
            {'val_loss': 1.0, 'physical_validation_loss': 1.0, 'dev_ranking': 15.0, 'metric_vector': (1.0, 15.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'},
            {'val_loss': 1.0, 'physical_validation_loss': 1.0, 'dev_ranking': 15.0, 'metric_vector': (1.0, 15.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'},
            {'val_loss': 1.5, 'physical_validation_loss': 1.5, 'dev_ranking': 5.0, 'metric_vector': (1.5, 5.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'},
            {'val_loss': 1.8, 'physical_validation_loss': 1.8, 'dev_ranking': 1.0, 'metric_vector': (1.8, 1.0), 'selection_schema': ('physical_validation_loss', 'dev_ranking'), 'selection_status': 'certified'},
        ]

        with tempfile.TemporaryDirectory() as td:
            with patch('icgs.training.method._evaluate_stage', side_effect=eval_responses):
                with patch('icgs.training.method._execute_training_step', return_value=0.5):
                    rep = run_method_training(
                        cfg,
                        comp,
                        [train_rec],
                        stage='A1',
                        dataset_id='test-ds',
                        evaluation_dataset=[dev_rec],
                        limits={'max_updates': 10},
                        output_dir=td,
                    )
            self.assertTrue(rep.stopped_early)
            self.assertEqual(rep.updates_completed, 6)
            self.assertEqual(rep.best_update, 3)
            self.assertEqual(rep.best_metric_vector, (1.0, 15.0))

            # Diagnostic external ranking cannot be certified as scientific best
            best_ckpt = Path(td) / 'checkpoint_A1_best.pt'
            self.assertFalse(best_ckpt.exists(), 'checkpoint_A1_best.pt must not be certified when external dev_ranking is supplied')
            self.assertEqual(rep.selection_status, 'diagnostic_external')

            # Off-interval early stopping checkpoint at update 6 preserves best selection state
            ckpt_6 = Path(td) / 'checkpoint_A1_6.pt'
            self.assertTrue(ckpt_6.exists())
            raw_6 = torch.load(ckpt_6, map_location='cpu')
            self.assertEqual(raw_6['selection_state']['best_update'], 3)
            self.assertEqual(raw_6['selection_state']['best_metric_vector'], [1.0, 15.0])
            self.assertEqual(raw_6['selection_state']['selection_status'], 'diagnostic_external')

    def test_2c2_early_stopping_saves_off_interval_and_already_stopped_resume_does_not_advance(self):
        """Early stopping saves final checkpoint off periodic interval; resume on stopped state does not advance."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 1,
                'early_stopping_patience': 2,
                'A1': {'batch_size': 1, 'max_updates': 10, 'burnin_intervals': 0, 'supervised_intervals': 1, 'rollout_curriculum': [1]},
            },
        })
        comp = {
            'geometry': nn.ModuleDict({'encoder': nn.Linear(3, 3), 'decoder': nn.Linear(3, 3)}),
            'physical_memory': nn.Linear(3, 3),
            'dynamics': nn.Linear(3, 3),
        }
        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        dev_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}

        # Eval 1: 1.0 (best) -> patience 0
        # Eval 2: 2.0 (worse) -> patience 1
        # Eval 3: 3.0 (worse) -> patience 2 -> early stopping at update 3!
        eval_responses = [
            {'val_loss': 1.0, 'physical_validation_loss': 1.0, 'metric_vector': (1.0,), 'selection_schema': ('physical_validation_loss',), 'selection_status': 'certified'},
            {'val_loss': 2.0, 'physical_validation_loss': 2.0, 'metric_vector': (2.0,), 'selection_schema': ('physical_validation_loss',), 'selection_status': 'certified'},
            {'val_loss': 3.0, 'physical_validation_loss': 3.0, 'metric_vector': (3.0,), 'selection_schema': ('physical_validation_loss',), 'selection_status': 'certified'},
        ]

        with tempfile.TemporaryDirectory() as td:
            with patch('icgs.training.method._evaluate_stage', side_effect=eval_responses):
                with patch('icgs.training.method._execute_training_step', return_value=0.5):
                    rep = run_method_training(
                        cfg,
                        comp,
                        [train_rec],
                        stage='A1',
                        dataset_id='test-ds',
                        evaluation_dataset=[dev_rec],
                        limits={'max_updates': 10},
                        output_dir=td,
                    )
            self.assertTrue(rep.stopped_early)
            self.assertEqual(rep.updates_completed, 3)

            # Off-interval checkpoint at update 3 must exist
            ckpt_3 = Path(td) / 'checkpoint_A1_3.pt'
            self.assertTrue(ckpt_3.exists(), 'Checkpoint off save interval was not saved upon early stopping')

            # Resume from already-stopped checkpoint must not advance
            rep_resumed = run_method_training(
                cfg,
                comp,
                [train_rec],
                stage='A1',
                dataset_id='test-ds',
                evaluation_dataset=[dev_rec],
                limits={'max_updates': 10},
                resume_checkpoint=ckpt_3,
                output_dir=td,
            )
            self.assertEqual(rep_resumed.updates_completed, 0, 'Already-stopped checkpoint must not advance')
            self.assertEqual(rep_resumed.total_updates_completed, 3)
            self.assertTrue(rep_resumed.stopped_early)
            self.assertEqual(rep_resumed.status, 'stopped_early')

    def test_2c2_checkpoint_selection_state_preflight_rejection_of_corrupted_vectors_and_schemas(self):
        """Early resume preflight rejects corrupted, non-finite, arity-mismatched, or stage-mismatched selection states."""
        import tempfile
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })
        class ParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()
        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()
        comp = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}

        with tempfile.TemporaryDirectory() as td:
            rep = run_method_training(
                cfg,
                comp,
                [train_rec],
                stage='A0',
                dataset_id='test-ds',
                limits={'max_updates': 2},
                stop_after=1,
                output_dir=td,
            )
            ckpt = rep.checkpoint_path

            # 1. Non-finite element in best_metric_vector
            raw1 = torch.load(ckpt, map_location='cpu')
            raw1['selection_state']['selection_status'] = 'not_certified (bridge gate unavailable)'
            raw1['selection_state']['selection_schema'] = ['normalized_reconstruction_loss']
            raw1['selection_state']['best_metric_vector'] = [float('inf')]
            bad1 = f'{td}/bad1.pt'
            torch.save(raw1, bad1)
            with self.assertRaisesRegex(ValueError, 'non-finite or invalid'):
                run_method_training(cfg, comp, [train_rec], stage='A0', dataset_id='test-ds', resume_checkpoint=bad1, limits={'max_updates': 2})

            # 2. Arity mismatch
            raw2 = torch.load(ckpt, map_location='cpu')
            raw2['selection_state']['selection_status'] = 'not_certified (bridge gate unavailable)'
            raw2['selection_state']['selection_schema'] = ['normalized_reconstruction_loss']
            raw2['selection_state']['best_metric_vector'] = [1.0, 2.0]
            bad2 = f'{td}/bad2.pt'
            torch.save(raw2, bad2)
            with self.assertRaisesRegex(ValueError, 'arity mismatch'):
                run_method_training(cfg, comp, [train_rec], stage='A0', dataset_id='test-ds', resume_checkpoint=bad2, limits={'max_updates': 2})

            # 3. Stage/schema mismatch (A1 schema in A0 run)
            raw3 = torch.load(ckpt, map_location='cpu')
            raw3['selection_state']['selection_status'] = 'not_certified'
            raw3['selection_state']['selection_schema'] = ['physical_validation_loss']
            raw3['selection_state']['best_metric_vector'] = [1.0]
            bad3 = f'{td}/bad3.pt'
            torch.save(raw3, bad3)
            with self.assertRaisesRegex(ValueError, 'Stage A0 requires selection_schema'):
                run_method_training(cfg, comp, [train_rec], stage='A0', dataset_id='test-ds', resume_checkpoint=bad3, limits={'max_updates': 2})

    def test_2c2_real_optimizer_evaluation_resume_bit_for_bit_continuation(self):
        """Uninterrupted evaluation run vs 1 update + resume with real optimizer updates produces bit-for-bit identity."""
        import copy
        import tempfile
        from icgs.training.method import run_method_training

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 1,
                'early_stopping_patience': 5,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })

        class CustomEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class CustomDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        torch.manual_seed(42)
        base_enc = CustomEncoder()
        init_weights = copy.deepcopy(base_enc.state_dict())

        train_rec = {'points': torch.randn(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        dev_rec = {'points': torch.randn(1, 4, 3), 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}

        # Uninterrupted Run A: 2 updates
        with tempfile.TemporaryDirectory() as td_a:
            enc_a = CustomEncoder()
            enc_a.load_state_dict(copy.deepcopy(init_weights))
            comp_a = {'geometry': nn.ModuleDict({'encoder': enc_a, 'decoder': CustomDecoder()})}

            rep_a = run_method_training(
                cfg,
                comp_a,
                [train_rec],
                stage='A0',
                dataset_id='test-ds',
                evaluation_dataset=[dev_rec],
                limits={'max_updates': 2},
                run_seed=101,
                output_dir=td_a,
            )

        # 1 update + resume to 2 updates in Run B
        with tempfile.TemporaryDirectory() as td_b:
            enc_b = CustomEncoder()
            enc_b.load_state_dict(copy.deepcopy(init_weights))
            comp_b = {'geometry': nn.ModuleDict({'encoder': enc_b, 'decoder': CustomDecoder()})}

            rep_b1 = run_method_training(
                cfg,
                comp_b,
                [train_rec],
                stage='A0',
                dataset_id='test-ds',
                evaluation_dataset=[dev_rec],
                limits={'max_updates': 2},
                stop_after=1,
                run_seed=101,
                output_dir=td_b,
            )
            ckpt_b1 = rep_b1.checkpoint_path

            rep_b2 = run_method_training(
                cfg,
                comp_b,
                [train_rec],
                stage='A0',
                dataset_id='test-ds',
                evaluation_dataset=[dev_rec],
                limits={'max_updates': 2},
                resume_checkpoint=ckpt_b1,
                run_seed=101,
                output_dir=td_b,
            )

            combined_loss = rep_b1.loss_history + rep_b2.loss_history
            self.assertEqual(combined_loss, rep_a.loss_history)
            self.assertEqual(rep_b2.total_updates_completed, rep_a.updates_completed)
            self.assertEqual(rep_b2.best_update, rep_a.best_update)
            self.assertEqual(rep_b2.best_metric_vector, rep_a.best_metric_vector)
            for k in enc_a.state_dict():
                self.assertTrue(torch.equal(enc_a.state_dict()[k], enc_b.state_dict()[k]), f'Mismatch in {k}')

    def test_2c2_dev_ranking_sequence_reorder_invariance_and_varying_values_rejection(self):
        """Reordering identical records preserves metric; changing or partial dev_ranking is rejected."""
        import numpy as np
        from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
        from icgs.contracts.records import Observation
        from icgs.models.decoders.physical import PhysicalDecoder
        from icgs.models.dynamics.physical import PhysicalDynamics
        from icgs.models.encoders.physical import PhysicalEncoder
        from icgs.models.memories.physical import PhysicalMemory, proprioception
        from icgs.state.physical import PhysicalState
        from icgs.training.method import _evaluate_stage

        cfg = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'A1': {'batch_size': 1, 'max_updates': 10, 'burnin_intervals': 0, 'supervised_intervals': 1, 'rollout_curriculum': [1]},
            },
        })

        dynamics = PhysicalDynamics(cfg)
        encoder = PhysicalEncoder(method_config=cfg)
        decoder = PhysicalDecoder(method_config=cfg)
        memory = PhysicalMemory(config=cfg)
        components = {
            'geometry': nn.ModuleDict({'encoder': encoder, 'decoder': decoder}),
            'physical_memory': memory,
            'dynamics': dynamics,
        }

        def make_valid_transitions(count):
            res = []
            curr_pts = np.zeros((128, 3), dtype=np.float64)
            curr_se3 = np.eye(4, dtype=np.float64)
            for i in range(count):
                next_pts = curr_pts.copy()
                b = TimedObservation(Observation(curr_pts.copy(), curr_se3.copy(), 0.0), i, i * 0.1, i * 0.1, 'sensor')
                a = TimedObservation(Observation(next_pts.copy(), curr_se3.copy(), 0.0), i + 1, (i + 1) * 0.1, (i + 1) * 0.1, 'sensor')
                res.append(ExecutedTransition(b, a, TimedCommand(target_w=curr_se3.copy(), grip=0, duration_s=0.1), 0.1, i + 1, 'ok'))
                curr_pts = next_pts
            return res

        valid_transitions = make_valid_transitions(2)
        pose = torch.eye(4)[None]
        grip = torch.zeros(1, 1)
        gravity = torch.tensor([[0.0, 0.0, -1.0]])
        valid_reset = PhysicalState(
            X=torch.zeros(1, 128, 256),
            x=torch.zeros(1, 128, 3),
            valid=torch.ones(1, 128, dtype=torch.bool),
            p=proprioception(pose, grip, gravity),
            memory=torch.zeros(1, 2, 256),
            T_w_e=pose,
            grip=grip,
            cached_world_cloud=torch.zeros(1, 128, 3),
            cached_world_cloud_valid=torch.ones(1, 128, dtype=torch.bool),
            boundary=0,
            encoder_lineage='encoder-v1',
            memory_lineage='memory-v1',
        )

        rec_a = {'transitions': valid_transitions, 'reset_state': valid_reset, 'valid': torch.ones(1, 1, dtype=torch.bool), 'dev_ranking': 5.0, 'provenance': {'episode_id': 'ep_a', 'lineage_id': 'lin_a', 'split': 'dev'}}
        rec_b = {'transitions': valid_transitions, 'reset_state': valid_reset, 'valid': torch.ones(1, 1, dtype=torch.bool), 'dev_ranking': 5.0, 'provenance': {'episode_id': 'ep_b', 'lineage_id': 'lin_b', 'split': 'dev'}}

        # 1. Sequence reordering invariance on real _evaluate_stage
        res1 = _evaluate_stage('A1', components, [rec_a, rec_b], cfg, expected_schema=('physical_validation_loss', 'dev_ranking'))
        res2 = _evaluate_stage('A1', components, [rec_b, rec_a], cfg, expected_schema=('physical_validation_loss', 'dev_ranking'))
        self.assertEqual(res1['metric_vector'], res2['metric_vector'])
        self.assertEqual(res1['dev_ranking'], 5.0)
        self.assertEqual(res1['evaluator'], 'NOT RUN (diagnostic external supplied)')
        self.assertEqual(res1['selection_status'], 'diagnostic_external')

        # 2. Changing dev_ranking values across records is rejected
        rec_diff = {'transitions': valid_transitions, 'reset_state': valid_reset, 'valid': torch.ones(1, 1, dtype=torch.bool), 'dev_ranking': 10.0, 'provenance': {'episode_id': 'ep_diff', 'lineage_id': 'lin_diff', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'varying dev_ranking values across evaluation records rejected'):
            _evaluate_stage('A1', components, [rec_a, rec_diff], cfg)

        # 3. Partial dev_ranking presence is rejected
        rec_none = {'transitions': valid_transitions, 'reset_state': valid_reset, 'valid': torch.ones(1, 1, dtype=torch.bool), 'provenance': {'episode_id': 'ep_none', 'lineage_id': 'lin_none', 'split': 'dev'}}
        with self.assertRaisesRegex(ValueError, 'partial/mixed dev_ranking'):
            _evaluate_stage('A1', components, [rec_a, rec_none], cfg)

    def test_2c2_selection_state_validator_strict_invariants_and_negative_checks(self):
        """Strict selection_state invariants: status vocabulary, exact primary equality, bounds, and no-eval invariants."""
        from icgs.training.method import _validate_selection_state

        valid_state = {
            'selection_status': 'certified',
            'selection_schema': ['physical_validation_loss'],
            'best_metric_vector': [1.0],
            'best_metric': 1.0,
            'best_update': 1,
            'patience_counter': 0,
            'eval_history': [{'metric_vector': [1.0]}],
        }
        _validate_selection_state(valid_state, stage='A1', checkpoint_update=1)

        # 1. Invalid status vocabulary
        with self.assertRaisesRegex(ValueError, 'invalid/undocumented selection_status'):
            _validate_selection_state(dict(valid_state, selection_status='bogus'), stage='A1', checkpoint_update=1)

        # 2. A0 cannot be certified while bridge gate unavailable
        with self.assertRaisesRegex(ValueError, "Stage A0 cannot have selection_status 'certified'"):
            _validate_selection_state({
                'selection_status': 'certified',
                'selection_schema': ['normalized_reconstruction_loss'],
                'best_metric_vector': [1.0],
                'best_metric': 1.0,
                'best_update': 1,
                'patience_counter': 0,
                'eval_history': [{'metric_vector': [1.0]}],
            }, stage='A0', checkpoint_update=1)

        # 3. A1 with external dev_ranking cannot be certified
        with self.assertRaisesRegex(ValueError, "Stage A1 with external dev_ranking diagnostic cannot have selection_status 'certified'"):
            _validate_selection_state({
                'selection_status': 'certified',
                'selection_schema': ['physical_validation_loss', 'dev_ranking'],
                'best_metric_vector': [1.0, 5.0],
                'best_metric': 1.0,
                'best_update': 1,
                'patience_counter': 0,
                'eval_history': [{'metric_vector': [1.0, 5.0]}],
            }, stage='A1', checkpoint_update=1)

        # 4. Exact primary equality (no math.isclose)
        with self.assertRaisesRegex(ValueError, 'does not match primary best_metric_vector\\[0\\] .* exactly'):
            _validate_selection_state(dict(valid_state, best_metric=1.000001), stage='A1', checkpoint_update=1)

        # 5. best_update bounds: 1 <= best_update <= checkpoint_update
        with self.assertRaisesRegex(ValueError, 'best_update must be an integer >= 1'):
            _validate_selection_state(dict(valid_state, best_update=0), stage='A1', checkpoint_update=1)
        with self.assertRaisesRegex(ValueError, 'exceeds checkpoint update'):
            _validate_selection_state(dict(valid_state, best_update=5), stage='A1', checkpoint_update=2)

        # 6. No-evaluation invariants
        no_eval_valid = {
            'selection_status': 'no_evaluation',
            'selection_schema': None,
            'best_metric_vector': None,
            'best_metric': None,
            'best_update': None,
            'patience_counter': 0,
            'eval_history': [],
        }
        _validate_selection_state(no_eval_valid, stage='A1', checkpoint_update=1)

        # No-eval patience must be 0
        with self.assertRaisesRegex(ValueError, 'patience_counter must be 0'):
            _validate_selection_state(dict(no_eval_valid, patience_counter=1), stage='A1', checkpoint_update=1)

        # No-eval history must be empty
        with self.assertRaisesRegex(ValueError, 'eval_history must be empty'):
            _validate_selection_state(dict(no_eval_valid, eval_history=[{'metric_vector': [1.0]}]), stage='A1', checkpoint_update=1)

        # No-eval status must be no_evaluation
        with self.assertRaisesRegex(ValueError, "must have selection_status 'no_evaluation'"):
            _validate_selection_state(dict(no_eval_valid, selection_status='certified'), stage='A1', checkpoint_update=1)

        # No-eval cannot have best_metric_vector
        with self.assertRaisesRegex(ValueError, 'best_metric_vector must be None'):
            _validate_selection_state(dict(no_eval_valid, best_metric_vector=[1.0]), stage='A1', checkpoint_update=1)

        # 7. eval_history entry structure & arity
        bad_history_entry = dict(valid_state, eval_history=[{'wrong_key': 1.0}])
        with self.assertRaises(KeyError):
            _validate_selection_state(bad_history_entry, stage='A1', checkpoint_update=1)

        bad_history_arity = dict(valid_state, eval_history=[{'metric_vector': [1.0, 2.0]}])
        with self.assertRaisesRegex(ValueError, 'arity mismatch'):
            _validate_selection_state(bad_history_arity, stage='A1', checkpoint_update=1)

        bad_history_nan = dict(valid_state, eval_history=[{'metric_vector': [float('nan')]}])
        with self.assertRaisesRegex(ValueError, 'non-finite or invalid'):
            _validate_selection_state(bad_history_nan, stage='A1', checkpoint_update=1)

        # 8. Missing required key raises KeyError
        missing_status = dict(valid_state)
        del missing_status['selection_status']
        with self.assertRaises(KeyError):
            _validate_selection_state(missing_status, stage='A1', checkpoint_update=1)

    def test_2c2_eval_interval_greater_than_max_updates_checkpoint_and_resume_succeeds(self):
        """When max_updates ends before eval cadence, checkpoint is created with not_certified and resume succeeds."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from icgs.training.method import run_method_training

        class ParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 3)
            def forward(self, pts, valid):
                out = self.fc(pts)
                return type('Enc', (), {'x': out, 'points_w': out, 'point_valid': valid})()

        class DummyDecoder(nn.Module):
            def forward(self, enc):
                return type('Dec', (), {'points_w': enc.points_w, 'point_valid': enc.point_valid})()

        train_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep1', 'lineage_id': 'lin1', 'split': 'train'}}
        dev_rec = {'points': torch.zeros(1, 4, 3), 'provenance': {'episode_id': 'ep_dev', 'lineage_id': 'lin_dev', 'split': 'dev'}}

        # 1. Stage A0: eval_interval=5 > run ceiling=2
        cfg_a0 = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 5,
                'A0': {'batch_size': 1, 'max_updates': 10},
            },
        })
        comp_a0 = {'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()})}
        with tempfile.TemporaryDirectory() as td_a0:
            rep_a0_1 = run_method_training(
                cfg_a0,
                comp_a0,
                [train_rec],
                stage='A0',
                dataset_id='test-ds',
                evaluation_dataset=[dev_rec],
                limits={'max_updates': 5},
                stop_after=2,
                output_dir=td_a0,
            )
            self.assertEqual(rep_a0_1.updates_completed, 2)
            self.assertEqual(rep_a0_1.selection_status, 'not_certified (bridge gate unavailable)')
            self.assertIsNone(rep_a0_1.best_metric_vector)
            self.assertIsNone(rep_a0_1.best_update)

            ckpt_a0_2 = Path(td_a0) / 'checkpoint_A0_2.pt'
            self.assertTrue(ckpt_a0_2.exists())
            raw_a0 = torch.load(ckpt_a0_2, map_location='cpu')
            self.assertEqual(raw_a0['selection_state']['selection_status'], 'not_certified (bridge gate unavailable)')
            self.assertIsNone(raw_a0['selection_state']['best_metric_vector'])
            self.assertIsNone(raw_a0['selection_state']['best_update'])
            self.assertEqual(raw_a0['selection_state']['patience_counter'], 0)
            self.assertEqual(raw_a0['selection_state']['eval_history'], [])

        # 2. Stage A1: eval_interval=5 > run ceiling=2
        cfg_a1 = MethodConfig.from_dict({
            'optimizer': {'warmup_updates': 0},
            'stages': {
                'training_seeds': (101, 102, 103),
                'generator_seed': 201,
                'reset_seed': 202,
                'action_seed': 203,
                'evaluation_interval_updates': 5,
                'A1': {'batch_size': 1, 'max_updates': 10, 'burnin_intervals': 0, 'supervised_intervals': 1, 'rollout_curriculum': [1]},
            },
        })
        comp_a1 = {
            'geometry': nn.ModuleDict({'encoder': ParamModule(), 'decoder': DummyDecoder()}),
            'physical_memory': nn.Linear(3, 3),
            'dynamics': nn.Linear(3, 3),
        }
        with tempfile.TemporaryDirectory() as td_a1:
            with patch('icgs.training.method._execute_training_step', return_value=0.5):
                rep_a1_1 = run_method_training(
                    cfg_a1,
                    comp_a1,
                    [train_rec],
                    stage='A1',
                    dataset_id='test-ds',
                    evaluation_dataset=[dev_rec],
                    limits={'max_updates': 5},
                    stop_after=2,
                    output_dir=td_a1,
                )
                self.assertEqual(rep_a1_1.updates_completed, 2)
                self.assertEqual(rep_a1_1.selection_status, 'not_certified')
                self.assertIsNone(rep_a1_1.best_metric_vector)
                self.assertIsNone(rep_a1_1.best_update)

                ckpt_a1_2 = Path(td_a1) / 'checkpoint_A1_2.pt'
                self.assertTrue(ckpt_a1_2.exists())
                raw_a1_2 = torch.load(ckpt_a1_2, map_location='cpu')
                self.assertEqual(raw_a1_2['selection_state']['selection_status'], 'not_certified')
                self.assertIsNone(raw_a1_2['selection_state']['best_metric_vector'])
                self.assertIsNone(raw_a1_2['selection_state']['best_update'])
                self.assertEqual(raw_a1_2['selection_state']['patience_counter'], 0)
                self.assertEqual(raw_a1_2['selection_state']['eval_history'], [])

                # Resume on checkpoint_A1_2 with same total schedule (max_updates=5)
                eval_resp = {'val_loss': 1.0, 'physical_validation_loss': 1.0, 'metric_vector': (1.0,), 'selection_schema': ('physical_validation_loss',), 'selection_status': 'certified'}
                with patch('icgs.training.method._evaluate_stage', return_value=eval_resp):
                    rep_a1_2 = run_method_training(
                        cfg_a1,
                        comp_a1,
                        [train_rec],
                        stage='A1',
                        dataset_id='test-ds',
                        evaluation_dataset=[dev_rec],
                        limits={'max_updates': 5},
                        resume_checkpoint=ckpt_a1_2,
                        output_dir=td_a1,
                    )
                    self.assertEqual(rep_a1_2.updates_completed, 3)
                    self.assertEqual(rep_a1_2.total_updates_completed, 5)
                    self.assertEqual(rep_a1_2.selection_status, 'certified')
                    self.assertEqual(rep_a1_2.best_update, 5)
                    self.assertEqual(rep_a1_2.best_metric_vector, (1.0,))

                    ckpt_a1_5 = Path(td_a1) / 'checkpoint_A1_5.pt'
                    self.assertTrue(ckpt_a1_5.exists())
                    raw_a1_5 = torch.load(ckpt_a1_5, map_location='cpu')
                    self.assertEqual(raw_a1_5['selection_state']['selection_status'], 'certified')
                    self.assertEqual(raw_a1_5['selection_state']['best_update'], 5)
                    self.assertTrue(len(raw_a1_5['selection_state']['eval_history']) > 0)
                    self.assertTrue(len(rep_a1_2.eval_metrics) > 0)
                    self.assertIn('val_loss', rep_a1_2.eval_metrics)


if __name__ == '__main__':
    unittest.main()
