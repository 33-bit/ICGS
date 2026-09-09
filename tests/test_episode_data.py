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


if __name__ == '__main__':
    unittest.main()
