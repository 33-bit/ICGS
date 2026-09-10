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


if __name__ == '__main__':
    unittest.main()
