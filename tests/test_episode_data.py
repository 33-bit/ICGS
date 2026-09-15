import unittest

import numpy as np

from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation


class EpisodeDataTests(unittest.TestCase):
    def test_online_fields_reject_privileged_data(self):
        from icgs.data.schemas.episodes import validate_online_fields

        validate_online_fields({'points': [], 'T_w_e': [], 'grip': 0})
        with self.assertRaisesRegex(ValueError, 'privileged'):
            validate_online_fields({'points': [], 'program_id': 'T01'})

    @staticmethod
    def _transition(
        before_boundary=0,
        after_boundary=1,
        *,
        before_simulator_timestamp=None,
        before_wall_timestamp=None,
        before_sensor_profile_id='sensor-v1',
    ):
        observation = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
        if before_simulator_timestamp is None:
            before_simulator_timestamp = 1.0 + 0.1 * before_boundary
        if before_wall_timestamp is None:
            before_wall_timestamp = 2.0 + 0.1 * before_boundary
        before = TimedObservation(
            observation, before_boundary, before_simulator_timestamp,
            before_wall_timestamp, before_sensor_profile_id,
        )
        after = TimedObservation(
            observation, after_boundary, 1.0 + 0.1 * after_boundary,
            2.0 + 0.1 * after_boundary, 'sensor-v1',
        )
        return ExecutedTransition(before, after, TimedCommand(np.eye(4), 0, 0.1), 0.1, 20, 'ok')

    @classmethod
    def _episode(cls):
        online = {
            'points': np.zeros((1, 3), dtype=np.float32),
            'T_w_e': np.eye(4, dtype=np.float64),
            'grip': 0,
            'point_valid': np.array([True]),
        }
        return {
            'schema_version': 'icgs_episode_v1',
            'provenance': {
                'episode_id': 'episode-1',
                'program_id': 'T01',
                'source_lineage_id': 'seed-1',
                'asset_family_id': 'asset-a',
                'split': 'train',
                'calibration_id': 'calibration-v1',
                'observation_origin': 'measured',
                'raw_commands_id': 'raw-commands-1',
                'materialized_commands_id': 'materialized-commands-1',
            },
            'online_observations': [online, {key: value.copy() if isinstance(value, np.ndarray) else value
                                              for key, value in online.items()}],
            'transitions': [cls._transition()],
        }

    def test_episode_requires_causal_online_values_and_metadata_outside_them(self):
        from icgs.data.schemas.episodes import validate_episode

        record = self._episode()
        validate_episode(record)
        record['online_observations'][0]['program_id'] = 'T01'
        with self.assertRaisesRegex(ValueError, 'privileged'):
            validate_episode(record)

    def test_episode_rejects_invalid_numeric_storage_and_boundary_gaps(self):
        from icgs.data.schemas.episodes import validate_episode

        record = self._episode()
        record['online_observations'][0]['points'] = np.array([[np.nan, 0., 0.]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_episode(record)

        record = self._episode()
        record['online_observations'][0]['point_valid'] = np.array([1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            validate_episode(record)

        record = self._episode()
        record['online_observations'][0]['points'] = np.array([[0., 0., 0.]], dtype=object)
        with self.assertRaisesRegex(ValueError, 'object'):
            validate_episode(record)

        record = self._episode()
        record['transitions'] = [self._transition(0, 2)]
        with self.assertRaisesRegex(ValueError, 'adjacent'):
            validate_episode(record)

    def test_episode_rejects_online_transition_observation_mismatches(self):
        from icgs.data.schemas.episodes import validate_episode

        record = self._episode()
        record['online_observations'][0]['points'][0, 0] = 0.01
        with self.assertRaisesRegex(ValueError, 'transition observation|points'):
            validate_episode(record)

        record = self._episode()
        record['online_observations'][1]['T_w_e'][0, 3] = 0.01
        with self.assertRaisesRegex(ValueError, 'transition observation|T_w_e'):
            validate_episode(record)

        record = self._episode()
        record['online_observations'][1]['grip'] = 1
        with self.assertRaisesRegex(ValueError, 'transition observation|grip'):
            validate_episode(record)

        record = self._episode()
        record['online_observations'][1]['points'] = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
        record['online_observations'][1]['point_valid'] = np.array([False, True])
        with self.assertRaisesRegex(ValueError, 'transition observation|points'):
            validate_episode(record)

    def test_episode_rejects_inconsistent_shared_boundary_metadata(self):
        from icgs.data.schemas.episodes import validate_episode

        for overrides, pattern in (
            ({'before_simulator_timestamp': 1.100001}, 'simulator_timestamp'),
            ({'before_wall_timestamp': 2.100001}, 'measured_wall_timestamp'),
            ({'before_sensor_profile_id': 'sensor-v2'}, 'sensor_profile_id'),
        ):
            with self.subTest(overrides=overrides):
                record = self._episode()
                record['transitions'] = [
                    self._transition(0, 1),
                    self._transition(1, 2, **overrides),
                ]
                online = record['online_observations'][0]
                record['online_observations'].append(
                    {key: value.copy() if isinstance(value, np.ndarray) else value
                     for key, value in online.items()})
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_episode(record)

    def test_split_lineage_rejects_cross_split_literal_lineage(self):
        from icgs.data.datasets.episodes import validate_split_lineage

        with self.assertRaisesRegex(ValueError, 'lineage'):
            validate_split_lineage([
                {'lineage_id': 'seed-a', 'split': 'train'},
                {'lineage_id': 'seed-a', 'split': 'test'}])

    def test_split_lineage_and_causal_prefix_validate_edges(self):
        from icgs.data.datasets.episodes import causal_prefix, validate_split_lineage

        validate_split_lineage([
            {'lineage_id': 'seed-a', 'split': 'train'},
            {'lineage_id': 'seed-a', 'split': 'train'},
            {'lineage_id': 'seed-b', 'split': 'dev'},
        ])
        with self.assertRaisesRegex(ValueError, 'lineage_id'):
            validate_split_lineage([{'split': 'train'}])

        episode = self._episode()
        episode['transitions'] = [self._transition(0, 1), self._transition(1, 2)]
        online = episode['online_observations'][0]
        episode['online_observations'].append(
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in online.items()})
        prefix = causal_prefix(episode, 1)
        self.assertEqual(len(prefix), 1)
        self.assertEqual(prefix[-1].after.boundary, 1)
        self.assertEqual(causal_prefix(episode, 0), ())
        with self.assertRaisesRegex(ValueError, 'boundary'):
            causal_prefix(episode, 3)

    def test_attempt_counts_retains_failures_and_rejects_malformed_statuses(self):
        from icgs.data.collection.attempts import attempt_counts

        counts = attempt_counts([
            {'status': 'success'},
            {'status': 'timeout'},
            {'status': 'invalid-input'},
        ])
        self.assertEqual(counts, {'attempts': 3, 'successes': 1, 'invalid': 1})
        with self.assertRaisesRegex(ValueError, 'status'):
            attempt_counts([{}])
        with self.assertRaisesRegex(ValueError, 'status'):
            attempt_counts([{'status': ''}])
        with self.assertRaisesRegex(ValueError, 'canonical'):
            attempt_counts([{'status': ' success '}])

    def test_program_catalog_is_explicit_immutable_and_uses_locked_ids(self):
        from icgs.data.collection.programs import get_program, program_catalog

        catalog = program_catalog()
        self.assertEqual(get_program('T01').steps, ('grasp A', 'lift A'))
        self.assertEqual(get_program('V01').split, 'development')
        self.assertEqual(get_program('P1').steps, ('open', 'place A', 'place B', 'close'))
        self.assertEqual(get_program('G1').family, 'grasp-transport-fit')
        self.assertEqual(get_program('R1').family, 'park-retrieve-restore')
        self.assertEqual(tuple(catalog), tuple(program_catalog()))
        self.assertEqual(sum(spec.split == 'train' for spec in catalog.values()), 20)
        self.assertEqual(sum(spec.split == 'development' for spec in catalog.values()), 4)
        self.assertEqual(
            tuple(spec.program_id for spec in catalog.values() if spec.split == 'test'),
            ('P1', 'P2', 'P3', 'P4', 'G1', 'G2', 'G3', 'G4', 'R1', 'R2', 'R3', 'R4'),
        )
        with self.assertRaises(TypeError):
            catalog['T01'] = get_program('T01')
        with self.assertRaisesRegex(ValueError, 'unknown program_id'):
            get_program('P01')

    def test_event_annotation_uses_event_coverage_ties_and_semantic_masks(self):
        from icgs.data.collection.annotations import Interval, PrimitiveInterval, map_event_annotation

        event = Interval(10, 20)
        annotation = map_event_annotation(event, (
            PrimitiveInterval('earlier', 5, 15),
            PrimitiveInterval('later', 15, 25),
        ))
        self.assertEqual(annotation.primitive_id, 'earlier')
        self.assertTrue(annotation.mapping_valid)
        self.assertTrue(annotation.semantic_valid)

        event_coverage = map_event_annotation(
            Interval(10, 20), (PrimitiveInterval('long', 15, 115),)
        )
        self.assertEqual(event_coverage.primitive_id, 'long')
        self.assertTrue(event_coverage.mapping_valid)

        insufficient = map_event_annotation(Interval(0, 10), (PrimitiveInterval('short', 0, 4),))
        self.assertIsNone(insufficient.primitive_id)
        self.assertFalse(insufficient.mapping_valid)
        self.assertFalse(insufficient.semantic_valid)

        free_space = map_event_annotation(
            Interval(0, 10), (PrimitiveInterval('move', 0, 10, semantic_valid=False),)
        )
        self.assertEqual(free_space.primitive_id, 'move')
        self.assertTrue(free_space.mapping_valid)
        self.assertFalse(free_space.semantic_valid)

        ambiguous = map_event_annotation(Interval(0, 10), (
            PrimitiveInterval('first', 0, 10),
            PrimitiveInterval('second', 0, 10),
        ))
        self.assertIsNone(ambiguous.primitive_id)
        self.assertFalse(ambiguous.mapping_valid)
        self.assertFalse(ambiguous.semantic_valid)

    def test_task_labels_keep_history_current_relations_eligibility_and_masks_separate(self):
        from icgs.data.collection.annotations import task_labels

        initial = task_labels(occurred=False, current_relation=False, eligible=False)
        self.assertEqual((initial.rho, initial.nu, initial.eligibility), (False, False, False))
        self.assertEqual((initial.rho_valid, initial.nu_valid, initial.eligibility_valid), (True, True, True))

        complete = task_labels(occurred=True, current_relation=True, eligible=True)
        self.assertEqual((complete.rho, complete.nu, complete.eligibility), (True, True, True))

        displaced = task_labels(occurred=True, current_relation=False, eligible=False)
        self.assertEqual((displaced.rho, displaced.nu, displaced.eligibility), (True, False, False))

        free_space = task_labels(
            occurred=True, current_relation=True, eligible=True, postcondition_identifiable=False,
        )
        self.assertTrue(free_space.rho)
        self.assertTrue(free_space.rho_valid)
        self.assertFalse(free_space.nu_valid)
        self.assertTrue(free_space.eligibility)
        self.assertTrue(free_space.eligibility_valid)

        unknown = task_labels(occurred=True, current_relation=None, eligible=None)
        self.assertFalse(unknown.nu_valid)
        self.assertFalse(unknown.eligibility_valid)

    def test_first_terminal_resolution_preserves_precedence(self):
        from icgs.data.collection.annotations import first_terminal

        self.assertEqual(first_terminal(success=True, failure=False, absorbed_success=False), 'success')
        self.assertEqual(first_terminal(success=False, failure=True, absorbed_success=False), 'failure')
        self.assertEqual(first_terminal(success=False, failure=False, absorbed_success=False), 'continue')
        self.assertEqual(first_terminal(success=True, failure=True, absorbed_success=False), 'failure')
        self.assertEqual(first_terminal(success=False, failure=True, absorbed_success=True), 'success')

    def test_injected_task_monitor_keeps_predicates_outside_online_observation(self):
        from icgs.evaluation.dependencies import TaskMonitor
        from icgs.data.schemas.episodes import validate_online_fields

        class Predicates:
            def __init__(self):
                self.seen_config = None

            def annotate(self, transition, *, config):
                self.seen_config = config
                return {'predicate_label': 'inside', 'simulator_only': {'joint': 3}}

        predicates = Predicates()
        config = object()
        annotation = TaskMonitor(predicates, config).annotate(self._transition())
        self.assertEqual(annotation['predicate_label'], 'inside')
        self.assertIs(predicates.seen_config, config)
        with self.assertRaises(TypeError):
            annotation['predicate_label'] = 'changed'
        with self.assertRaises(ValueError):
            TaskMonitor(Predicates(), None)

        class BadPredicates:
            def annotate(self, transition, *, config):
                return ['not', 'a', 'mapping']

        with self.assertRaisesRegex(ValueError, 'mapping'):
            TaskMonitor(BadPredicates(), config).annotate(self._transition())

        for key, value in (
            ('program_id', 'T01'),
            ('role_id', 'A'),
            ('predicate_label', 'inside'),
            ('simulator_only', {'joint': 3}),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'privileged'):
                    validate_online_fields({key: value})

    def test_collect_attempt_lifecycle_cap_and_order(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import (
            AttemptResult,
            attempt_counts,
            collect_attempt,
        )

        class FakeEnv:
            def __init__(self):
                self.reset_seed = None
                self.closed = False
                self.current = 0

            def reset(self, seed=None):
                self.reset_seed = seed
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                before = TimedObservation(obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1')
                after = TimedObservation(obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1')
                return ExecutedTransition(before, after, command, command.duration_s, 20, 'ok')

            def close(self):
                self.closed = True

        class FakeMonitor:
            def __init__(self):
                self.calls = []

            def annotate(self, transition):
                ann = {'boundary': transition.before.boundary, 'predicate_label': 'inside'}
                self.calls.append(ann)
                return ann

        consumed_count = 0
        def command_generator():
            nonlocal consumed_count
            for _ in range(10):
                consumed_count += 1
                yield TimedCommand(np.eye(4), 0, 0.1)

        env = FakeEnv()
        monitor = FakeMonitor()
        config = MethodConfig(collection={'max_episode_intervals': 3})

        result = collect_attempt(
            env,
            command_generator(),
            config=config,
            monitor=monitor,
            seed=42,
        )

        self.assertIsInstance(result, AttemptResult)
        self.assertEqual(env.reset_seed, 42)
        self.assertTrue(env.closed)
        self.assertEqual(consumed_count, 3)
        self.assertEqual(len(result.transitions), 3)
        self.assertEqual(len(result.annotations), 3)
        self.assertEqual(len(monitor.calls), 3)
        self.assertEqual(result.status, 'timeout')
        counts = attempt_counts([result])
        self.assertEqual(counts, {'attempts': 1, 'successes': 0, 'invalid': 0})

    def test_collect_attempt_zero_and_short_inputs(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import collect_attempt

        class FakeEnv:
            def __init__(self):
                self.closed = False
                self.current = 0

            def reset(self, seed=None):
                self.grip = 0.0
                obs = Observation(np.zeros((1, 3)), np.eye(4), self.grip)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                before_obs = Observation(np.zeros((1, 3)), np.eye(4), self.grip)
                self.grip = float(command.grip)
                after_obs = Observation(np.zeros((1, 3)), np.eye(4), self.grip)
                return ExecutedTransition(
                    TimedObservation(before_obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1'),
                    TimedObservation(after_obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )

            def close(self):
                self.closed = True

        cfg = MethodConfig()
        env = FakeEnv()
        res_empty = collect_attempt(env, [], config=cfg)
        self.assertTrue(env.closed)
        self.assertEqual(len(res_empty.transitions), 0)
        self.assertEqual(res_empty.status, 'completed')

        env2 = FakeEnv()
        commands = [TimedCommand(np.eye(4), 0, 0.1), TimedCommand(np.eye(4), 1, 0.1)]
        res_short = collect_attempt(env2, iter(commands), config=cfg)
        self.assertTrue(env2.closed)
        self.assertEqual(len(res_short.transitions), 2)
        self.assertEqual(res_short.status, 'completed')

    def test_collect_attempt_terminal_looking_metadata_does_not_classify_or_truncate(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import attempt_counts, collect_attempt

        class FakeEnv:
            def __init__(self):
                self.closed = False
                self.current = 0

            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                return ExecutedTransition(
                    TimedObservation(obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1'),
                    TimedObservation(obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )

            def close(self):
                self.closed = True

        class TerminalLookingMonitor:
            def annotate(self, transition):
                return {
                    'success': 'false',
                    'failure': True,
                    'status': 'success',
                    'terminal': True,
                    'absorbed_success': True,
                }

        consumed = 0
        def commands():
            nonlocal consumed
            for _ in range(3):
                consumed += 1
                yield TimedCommand(np.eye(4), 0, 0.1)

        env = FakeEnv()
        cfg = MethodConfig(collection={'max_episode_intervals': 10})
        result = collect_attempt(env, commands(), config=cfg, monitor=TerminalLookingMonitor())

        self.assertTrue(env.closed)
        # All 3 commands executed; terminal-looking metadata did not truncate execution
        self.assertEqual(consumed, 3)
        self.assertEqual(len(result.transitions), 3)
        # Did not classify as "success" or "failure"
        self.assertEqual(result.status, 'completed')
        self.assertEqual(attempt_counts([result])['successes'], 0)

    def test_collect_attempt_wall_limit(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import collect_attempt

        class FakeEnv:
            def __init__(self):
                self.closed = False
                self.current = 0

            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                return ExecutedTransition(
                    TimedObservation(obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1'),
                    TimedObservation(obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )

            def close(self):
                self.closed = True

        time_points = [0.0, 1.0, 6.0]
        def fake_clock():
            return time_points.pop(0) if time_points else 10.0

        env = FakeEnv()
        config = MethodConfig(collection={'wall_limit_s': 5.0, 'max_episode_intervals': 10})
        commands = [TimedCommand(np.eye(4), 0, 0.1) for _ in range(5)]
        res = collect_attempt(env, commands, config=config, clock=fake_clock)
        self.assertTrue(env.closed)
        self.assertEqual(len(res.transitions), 1)
        self.assertEqual(res.status, 'timeout')

    def test_collect_attempt_evidence_retention_on_operational_failure(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        class FailingEnv:
            def __init__(self):
                self.closed = False
                self.current = 0

            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                if b == 2:
                    raise RuntimeError('physics exploded at step 2')
                self.current += 1
                obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                return ExecutedTransition(
                    TimedObservation(obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1'),
                    TimedObservation(obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )

            def close(self):
                self.closed = True

        env = FailingEnv()
        cfg = MethodConfig()
        commands = [TimedCommand(np.eye(4), 0, 0.1) for _ in range(5)]
        with self.assertRaises(AttemptExecutionError) as cm:
            collect_attempt(env, commands, config=cfg)

        self.assertTrue(env.closed)
        self.assertEqual(len(cm.exception.transitions), 2)
        self.assertIn('physics exploded', str(cm.exception.cause))

    def test_collect_attempt_annotation_failure_does_not_drop_transition(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        class Env:
            def __init__(self):
                self.closed = False
                self.current = 0

            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                return ExecutedTransition(
                    TimedObservation(obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1'),
                    TimedObservation(obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )

            def close(self):
                self.closed = True

        class FailingMonitor:
            def annotate(self, transition):
                if transition.before.boundary == 1:
                    raise RuntimeError('annotation compute failed')
                return {'predicate': 'ok'}

        env = Env()
        cfg = MethodConfig()
        commands = [TimedCommand(np.eye(4), 0, 0.1) for _ in range(5)]
        with self.assertRaises(AttemptExecutionError) as cm:
            collect_attempt(env, commands, config=cfg, monitor=FailingMonitor())

        self.assertTrue(env.closed)
        # Advance for step 1 succeeded BEFORE monitor failure, so transition 1 is NOT dropped!
        self.assertEqual(len(cm.exception.transitions), 2)
        self.assertEqual(len(cm.exception.annotations), 1)
        self.assertIn('annotation compute failed', str(cm.exception.cause))

    def test_collect_attempt_cleanup_failure_preserves_primary_cause(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        class BadEnv:
            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                raise RuntimeError('primary advance error')

            def close(self):
                raise IOError('secondary close error')

        env = BadEnv()
        cfg = MethodConfig()
        commands = [TimedCommand(np.eye(4), 0, 0.1)]
        with self.assertRaises(AttemptExecutionError) as cm:
            collect_attempt(env, commands, config=cfg)

        self.assertIn('primary advance error', str(cm.exception.cause))
        self.assertIn('secondary close error', str(cm.exception.cleanup_error))

    def test_collect_attempt_requires_explicit_config_and_validates_bounds(self):
        import dataclasses
        from icgs.configuration.method import CollectionConfig, MethodConfig
        from icgs.data.collection.attempts import collect_attempt

        class Env:
            def __init__(self):
                self.closed = False
            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')
            def close(self):
                self.closed = True

        env = Env()
        with self.assertRaisesRegex(TypeError, 'MethodConfig or CollectionConfig'):
            collect_attempt(env, [], config=None)
        self.assertFalse(env.closed)

        with self.assertRaisesRegex(TypeError, 'MethodConfig or CollectionConfig'):
            collect_attempt(env, [], config={'max_episode_intervals': 10})

        with self.assertRaisesRegex(ValueError, 'max_episode_intervals'):
            collect_attempt(env, [], config=CollectionConfig(max_episode_intervals=0))

        # CollectionConfig accepted directly
        c_cfg = CollectionConfig(max_episode_intervals=2)
        res = collect_attempt(env, [], config=c_cfg)
        self.assertTrue(env.closed)
        self.assertEqual(res.status, 'completed')

        # Limit propagated via dataclasses.replace
        env2 = Env()
        m_cfg = MethodConfig()
        modified_c = dataclasses.replace(m_cfg.collection, max_episode_intervals=1)
        res2 = collect_attempt(env2, [], config=modified_c)
        self.assertEqual(res2.status, 'completed')

    def test_collect_attempt_initial_clock_and_iter_failures_execute_cleanup(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        class Env:
            def __init__(self):
                self.closed_count = 0
            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')
            def close(self):
                self.closed_count += 1

        cfg = MethodConfig()

        # Failing clock on initial call
        env1 = Env()
        def bad_clock():
            raise RuntimeError('clock failure')
        with self.assertRaises(AttemptExecutionError) as cm:
            collect_attempt(env1, [], config=cfg, clock=bad_clock)
        self.assertEqual(env1.closed_count, 1)
        self.assertIn('clock failure', str(cm.exception.cause))
        self.assertEqual(len(cm.exception.transitions), 0)
        self.assertIsNone(cm.exception.initial_observation)
        self.assertIsNone(cm.exception.rejected_transition)

        # Failing iterator construction
        env2 = Env()
        class BadCommands:
            def __iter__(self):
                raise RuntimeError('commands iter failure')
        with self.assertRaises(AttemptExecutionError) as cm2:
            collect_attempt(env2, BadCommands(), config=cfg)
        self.assertEqual(env2.closed_count, 1)
        self.assertIn('commands iter failure', str(cm2.exception.cause))
        self.assertEqual(len(cm2.exception.transitions), 0)
        self.assertIsNone(cm2.exception.initial_observation)
        self.assertIsNone(cm2.exception.rejected_transition)

        # Failing reset
        class ResetFailingEnv:
            def __init__(self):
                self.closed_count = 0
            def reset(self, seed=None):
                raise RuntimeError('reset failure')
            def close(self):
                self.closed_count += 1

        env_reset = ResetFailingEnv()
        with self.assertRaises(AttemptExecutionError) as cm_reset:
            collect_attempt(env_reset, [], config=cfg)
        self.assertEqual(env_reset.closed_count, 1)
        self.assertIn('reset failure', str(cm_reset.exception.cause))
        self.assertEqual(len(cm_reset.exception.transitions), 0)
        self.assertIsNone(cm_reset.exception.initial_observation)
        self.assertIsNone(cm_reset.exception.rejected_transition)

        # Failing command stream next() after step 0
        class Step0Env:
            def __init__(self):
                self.closed_count = 0
            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')
            def advance(self, command):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return ExecutedTransition(
                    TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1'),
                    TimedObservation(obs, 1, 1.1, 2.1, 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )
            def close(self):
                self.closed_count += 1

        def failing_stream():
            yield TimedCommand(np.eye(4), 0, 0.1)
            raise RuntimeError('stream next failure')

        env_next = Step0Env()
        with self.assertRaises(AttemptExecutionError) as cm_next:
            collect_attempt(env_next, failing_stream(), config=cfg)
        self.assertEqual(env_next.closed_count, 1)
        self.assertIn('stream next failure', str(cm_next.exception.cause))
        self.assertEqual(len(cm_next.exception.transitions), 1)
        self.assertIsNone(cm_next.exception.rejected_transition)

        # Missing callable close method
        class NoCloseEnv:
            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')
        with self.assertRaises(AttemptExecutionError) as cm3:
            collect_attempt(NoCloseEnv(), [], config=cfg)
        self.assertIn('close', str(cm3.exception.cleanup_error))

    def test_collect_attempt_shared_boundary_and_command_mismatch_retains_rejected_evidence(self):
        import dataclasses
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        cfg = MethodConfig()

        class ParameterizedEnv:
            def __init__(self, mutate_fn):
                self.mutate_fn = mutate_fn
                self.closed_count = 0
                self.current = 0

            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                before_obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                after_obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                before = TimedObservation(before_obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1')
                after = TimedObservation(after_obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1')
                tx = ExecutedTransition(before, after, command, command.duration_s, 20, 'ok')
                return self.mutate_fn(tx)

            def close(self):
                self.closed_count += 1

        rot_90_z = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        cases = [
            (
                'discontinuous_boundary',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(tx.before, boundary=5),
                    after=dataclasses.replace(tx.after, boundary=6),
                ),
                'transition before metadata mismatch',
            ),
            (
                'broken_boundary_step',
                lambda tx: dataclasses.replace(
                    tx,
                    after=dataclasses.replace(tx.after, boundary=10),
                ),
                'boundary continuity broken',
            ),
            (
                'simulator_timestamp_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(tx.before, simulator_timestamp=99.0),
                ),
                'transition before metadata mismatch',
            ),
            (
                'measured_wall_timestamp_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(tx.before, measured_wall_timestamp=99.0),
                ),
                'transition before metadata mismatch',
            ),
            (
                'sensor_profile_id_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(tx.before, sensor_profile_id='sensor-v2'),
                ),
                'transition before metadata mismatch',
            ),
            (
                'observation_grip_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(
                        tx.before,
                        observation=Observation(np.zeros((1, 3)), np.eye(4), 1.0),
                    ),
                ),
                'transition before observation',
            ),
            (
                'observation_points_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(
                        tx.before,
                        observation=Observation(np.ones((2, 3)), np.eye(4), 0.0),
                    ),
                ),
                'transition before observation',
            ),
            (
                'observation_pose_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    before=dataclasses.replace(
                        tx.before,
                        observation=Observation(np.zeros((1, 3)), rot_90_z, 0.0),
                    ),
                ),
                'transition before observation',
            ),
            (
                'command_grip_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    command=TimedCommand(tx.command.target_w, 1, tx.command.duration_s),
                ),
                'transition command does not match',
            ),
            (
                'command_duration_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    command=TimedCommand(tx.command.target_w, tx.command.grip, tx.command.duration_s + 0.5),
                ),
                'transition command does not match',
            ),
            (
                'command_target_w_mismatch',
                lambda tx: dataclasses.replace(
                    tx,
                    command=TimedCommand(rot_90_z, tx.command.grip, tx.command.duration_s),
                ),
                'transition command does not match',
            ),
        ]

        cmd = TimedCommand(np.eye(4), 0, 0.1)
        for name, mutate_fn, expected_msg in cases:
            with self.subTest(case=name):
                env = ParameterizedEnv(mutate_fn)
                with self.assertRaises(AttemptExecutionError) as cm:
                    collect_attempt(env, [cmd], config=cfg)
                self.assertEqual(env.closed_count, 1)
                self.assertIsNotNone(cm.exception.rejected_transition)
                self.assertEqual(len(cm.exception.transitions), 0)
                self.assertIn(expected_msg, str(cm.exception.cause))

    def test_collect_attempt_annotation_prefix_semantics(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import AttemptExecutionError, collect_attempt

        class Env:
            def __init__(self):
                self.closed_count = 0
                self.current = 0

            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')

            def advance(self, command):
                b = self.current
                self.current += 1
                before_obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                after_obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return ExecutedTransition(
                    TimedObservation(before_obs, b, 1.0 + 0.1 * b, 2.0 + 0.1 * b, 'sensor-v1'),
                    TimedObservation(after_obs, b + 1, 1.0 + 0.1 * (b + 1), 2.0 + 0.1 * (b + 1), 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )

            def close(self):
                self.closed_count += 1

        cfg = MethodConfig()
        commands = [TimedCommand(np.eye(4), 0, 0.1), TimedCommand(np.eye(4), 0, 0.1)]

        # No monitor supplied: annotations is empty tuple
        env_no_mon = Env()
        res = collect_attempt(env_no_mon, commands, config=cfg, monitor=None)
        self.assertEqual(env_no_mon.closed_count, 1)
        self.assertEqual(len(res.transitions), 2)
        self.assertEqual(res.annotations, ())

        # Monitor succeeds on step 0, fails on step 1: annotations has only step 0 prefix
        class FailingOnStep1Monitor:
            def annotate(self, transition):
                if transition.before.boundary == 1:
                    raise RuntimeError('step 1 annotation failed')
                return {'step': transition.before.boundary}

        env_fail_mon = Env()
        with self.assertRaises(AttemptExecutionError) as cm:
            collect_attempt(env_fail_mon, commands, config=cfg, monitor=FailingOnStep1Monitor())
        self.assertEqual(env_fail_mon.closed_count, 1)
        self.assertEqual(len(cm.exception.transitions), 2)  # both transitions executed
        self.assertEqual(len(cm.exception.annotations), 1)  # shorter prefix
        self.assertEqual(cm.exception.annotations[0], {'step': 0})
        self.assertIn('step 1 annotation failed', str(cm.exception.cause))

    def test_collect_attempt_annotation_snapshot_mutation_isolation(self):
        from icgs.configuration.method import MethodConfig
        from icgs.data.collection.attempts import collect_attempt

        class Env:
            def __init__(self):
                self.closed = False
            def reset(self, seed=None):
                obs = Observation(np.zeros((1, 3)), np.eye(4), 0.0)
                return TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1')
            def advance(self, command):
                obs = Observation(np.zeros((1, 3)), np.eye(4), float(command.grip))
                return ExecutedTransition(
                    TimedObservation(obs, 0, 1.0, 2.0, 'sensor-v1'),
                    TimedObservation(obs, 1, 1.1, 2.1, 'sensor-v1'),
                    command, command.duration_s, 20, 'ok'
                )
            def close(self):
                self.closed = True

        payload = {'meta': {'nested': 'original'}}
        class MutableMonitor:
            def annotate(self, transition):
                return payload

        env = Env()
        cfg = MethodConfig()
        cmd = TimedCommand(np.eye(4), 0, 0.1)
        res = collect_attempt(env, [cmd], config=cfg, monitor=MutableMonitor())
        self.assertTrue(env.closed)

        # Mutate the source dictionary post-annotation
        payload['meta']['nested'] = 'tampered'
        # Snapshot in result must remain untouched
        self.assertEqual(res.annotations[0]['meta']['nested'], 'original')




if __name__ == '__main__':
    unittest.main()
