import math
import unittest
from dataclasses import replace

import numpy as np

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation


class EventSegmentationTests(unittest.TestCase):
    @staticmethod
    def _pose(x=0.0, rotation_deg=0.0):
        angle = math.radians(rotation_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        pose = np.eye(4)
        pose[:3, :3] = np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        pose[0, 3] = x
        return pose

    @classmethod
    def _transitions(cls, xs, *, rotations=None, grips=None, commands=None, boundaries=None):
        count = len(xs)
        rotations = [0.0] * count if rotations is None else rotations
        grips = [0] * count if grips is None else grips
        boundaries = list(range(count)) if boundaries is None else boundaries
        if not (len(rotations) == len(grips) == len(boundaries) == count):
            raise AssertionError("fixture fields must have equal lengths")
        observations = tuple(
            TimedObservation(
                Observation(np.zeros((1, 3)), cls._pose(x, angle), grip),
                boundary,
                float(index),
                float(index),
                "sensor-v1",
            )
            for index, (x, angle, grip, boundary) in enumerate(
                zip(xs, rotations, grips, boundaries)
            )
        )
        if commands is None:
            commands = tuple(TimedCommand(np.eye(4), 0, 0.1) for _ in range(count - 1))
        if len(commands) != count - 1:
            raise AssertionError("fixture requires one command per transition")
        return tuple(
            ExecutedTransition(
                observations[index], observations[index + 1], commands[index], 0.1, 1, "ok"
            )
            for index in range(count - 1)
        )

    @staticmethod
    def _positions(final_boundary, jump_boundaries):
        position = 0.0
        result = []
        for boundary in range(final_boundary + 1):
            if boundary in jump_boundaries:
                position += 0.13
            result.append(position)
        return result

    @staticmethod
    def _structure(segments):
        return tuple(
            (segment.a, segment.b, segment.kind, segment.valid_action_window)
            for segment in segments
        )

    @staticmethod
    def _interactions(segments):
        return tuple((segment.a, segment.b) for segment in segments if segment.kind == "interaction")

    @staticmethod
    def _config(**changes):
        return replace(MethodConfig().event, **changes)

    def test_timed_demo_input_validates_without_transforming(self):
        from icgs.data.preprocessing.events import TimedDemoInput

        transitions = self._transitions([0.0, 0.0])
        demo = TimedDemoInput(transitions, " hash-with-owner-spacing ")
        self.assertIs(demo.transitions, transitions)
        self.assertEqual(demo.demo_content_hash, " hash-with-owner-spacing ")

        with self.assertRaisesRegex(TypeError, "tuple"):
            TimedDemoInput(list(transitions), "hash")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            TimedDemoInput((), "hash")
        with self.assertRaisesRegex(TypeError, "ExecutedTransition"):
            TimedDemoInput((object(),), "hash")
        with self.assertRaisesRegex(TypeError, "string"):
            TimedDemoInput(transitions, 1)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            TimedDemoInput(transitions, "  ")

        nonadjacent = self._transitions([0.0, 0.0], boundaries=[0, 2])
        with self.assertRaisesRegex(ValueError, "adjacent"):
            TimedDemoInput(nonadjacent, "hash")

        first = self._transitions([0.0, 0.0], boundaries=[0, 1])[0]
        second = self._transitions([0.0, 0.0], boundaries=[2, 3])[0]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            TimedDemoInput((first, second), "hash")

    def test_debounced_grip_boundaries_confirm_without_backdating(self):
        from icgs.data.preprocessing.events import debounced_grip_boundaries

        config = self._config()
        self.assertEqual(debounced_grip_boundaries([0, 1, 0, 1, 1], config), (4,))
        self.assertEqual(debounced_grip_boundaries([1, 1, 0, 0], config), (3,))
        self.assertEqual(debounced_grip_boundaries([0], config), ())
        self.assertEqual(debounced_grip_boundaries([0, 0, 0], config), ())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            debounced_grip_boundaries([], config)
        with self.assertRaisesRegex(ValueError, "grip"):
            debounced_grip_boundaries([0, 2], config)
        with self.assertRaisesRegex(TypeError, "config"):
            debounced_grip_boundaries([0], None)

    def test_segment_demo_uses_measured_state_not_commands_and_preserves_hash(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        class GuardedCommand(TimedCommand):
            _guarded = False

            def __getattribute__(self, name):
                if name in {"target_w", "grip"} and object.__getattribute__(self, "_guarded"):
                    raise AssertionError("segment_demo read a command field")
                return super().__getattribute__(name)

        guarded = GuardedCommand(self._pose(9.0, 170.0), 1, 0.1)
        object.__setattr__(guarded, "_guarded", True)
        ordinary = TimedCommand(self._pose(-9.0, -170.0), 0, 0.1)
        transitions_a = self._transitions([0.0, 0.0], commands=(guarded,))
        transitions_b = self._transitions([0.0, 0.0], commands=(ordinary,))

        segments_a = segment_demo(TimedDemoInput(transitions_a, "hash-a"), self._config())
        segments_b = segment_demo(TimedDemoInput(transitions_b, "hash-b"), self._config())
        self.assertEqual(self._structure(segments_a), self._structure(segments_b))
        self.assertEqual(
            self._structure(segments_a),
            ((0, 0, "start", False), (0, 1, "interaction", True), (1, 1, "end", False)),
        )
        self.assertTrue(all(segment.demo_content_hash == "hash-a" for segment in segments_a))
        self.assertTrue(all(segment.demo_content_hash == "hash-b" for segment in segments_b))
        with self.assertRaisesRegex(TypeError, "config"):
            segment_demo(TimedDemoInput(transitions_b, "hash-b"), None)

    def test_cumulative_translation_and_rotation_use_strict_motion_thresholds(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=100)
        exact_translation = TimedDemoInput(self._transitions([0.0, 0.06, 0.12, 0.12]), "exact-t")
        over_translation = TimedDemoInput(self._transitions([0.0, 0.061, 0.122, 0.122]), "over-t")
        self.assertEqual(self._interactions(segment_demo(exact_translation, config)), ((0, 3),))
        self.assertEqual(self._interactions(segment_demo(over_translation, config)), ((0, 2), (2, 3)))

        exact_rotation = TimedDemoInput(
            self._transitions([0.0] * 4, rotations=[0.0, 15.0, 30.0, 30.0]), "exact-r"
        )
        over_rotation = TimedDemoInput(
            self._transitions([0.0] * 4, rotations=[0.0, 15.1, 30.2, 30.2]), "over-r"
        )
        self.assertEqual(self._interactions(segment_demo(exact_rotation, config)), ((0, 3),))
        self.assertEqual(self._interactions(segment_demo(over_rotation, config)), ((0, 2), (2, 3)))

    def test_time_boundary_counts_transitions_not_observations(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=20)
        demo = TimedDemoInput(self._transitions([0.0] * 22), "time")
        self.assertEqual(self._interactions(segment_demo(demo, config)), ((0, 20), (20, 21)))

    def test_short_segments_merge_with_shorter_duration_neighbor_then_earlier(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=3, max_intervals=100)
        cases = (
            (16, {5, 7}, ((0, 7), (7, 16))),
            (16, {9, 11}, ((0, 9), (9, 16))),
            (12, {5, 7}, ((0, 7), (7, 12))),
        )
        for final_boundary, jumps, expected in cases:
            with self.subTest(final_boundary=final_boundary, jumps=jumps):
                demo = TimedDemoInput(
                    self._transitions(self._positions(final_boundary, jumps)), "short"
                )
                self.assertEqual(self._interactions(segment_demo(demo, config)), expected)

    def test_short_merge_never_erases_a_protected_grip_boundary(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=3, max_intervals=100)
        grips = [0] * 4 + [1] * 12
        demo = TimedDemoInput(
            self._transitions(self._positions(15, {7}), grips=grips), "protected-short"
        )
        self.assertEqual(self._interactions(segment_demo(demo, config)), ((0, 5), (5, 15)))

    def test_capacity_merge_uses_minimum_combined_duration_then_earlier_pair(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=100, max_interactions=2)
        demo = TimedDemoInput(self._transitions(self._positions(20, {4, 10, 13})), "cap")
        self.assertEqual(self._interactions(segment_demo(demo, config)), ((0, 13), (13, 20)))

        tied = TimedDemoInput(self._transitions(self._positions(12, {4, 8})), "cap-tie")
        self.assertEqual(self._interactions(segment_demo(tied, config)), ((0, 8), (8, 12)))

    def test_capacity_overflow_is_explicit_when_grip_boundaries_are_protected(self):
        from icgs.data.preprocessing.events import EventOverflowError, TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=100, max_interactions=2)
        grips = [0, 1, 1, 0, 0, 1, 1]
        demo = TimedDemoInput(self._transitions([0.0] * len(grips), grips=grips), "overflow")
        with self.assertRaisesRegex(EventOverflowError, "protected|overflow"):
            segment_demo(demo, config)


if __name__ == "__main__":
    unittest.main()
