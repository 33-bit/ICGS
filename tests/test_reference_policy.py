import ast
from contextlib import contextmanager, ExitStack
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import random
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from icgs.contracts.method import (
    ExecutedTransition,
    TimedCommand,
    TimedObservation,
)
from icgs.contracts.records import Observation
from icgs.data.preprocessing.events import TimedDemoInput


class ContextPreparationRngTests(unittest.TestCase):
    _TASK_3B0_FORBIDDEN_DEPENDENCIES = frozenset(
        {
            "split_route_seed",
            "scoped_seed",
            "PreparedContext",
            "InstantPolicy",
            "MethodContext",
            "Candidate",
            "sample_to_cond_demo",
            "subsample_pcd",
            "composition",
        }
    )

    @staticmethod
    def _assert_numpy_state_equal(left, right):
        if left[0] != right[0] or left[2:] != right[2:]:
            raise AssertionError("NumPy global RNG metadata changed")
        np.testing.assert_array_equal(left[1], right[1])

    @staticmethod
    def _record(**changes):
        from icgs.policies.reference import (
            CONTEXT_RNG_PROTOCOL,
            ContextPreparationRecord,
        )

        values = {
            "context_seed": 17,
            "full_demo_seeds": (3302413169, 2035845825),
            "window_slot_seeds": (1193624495, 3388839295, 637109157),
            "rng_protocol": CONTEXT_RNG_PROTOCOL,
            "full_context_id": "full-context",
            "window_context_ids": ("window-0", None, "window-2"),
        }
        values.update(changes)
        return ContextPreparationRecord(**values)

    def _assert_task_3b0_dependency_boundary(self, *surfaces):
        for surface in surfaces:
            source = textwrap.dedent(inspect.getsource(surface))
            tree = ast.parse(source)
            referenced = {
                node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            }
            referenced.update(
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
            )
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(alias.name for alias in node.names)
            found = sorted(
                dependency
                for dependency in self._TASK_3B0_FORBIDDEN_DEPENDENCIES
                if dependency in referenced
                or any(dependency in item for item in imported)
                or dependency in source
            )
            if found:
                self.fail(
                    f"Task 3B.0 surface {surface.__name__} must not reference "
                    f"later reference-policy dependencies: {found}; keep that "
                    "dependency in the owning Task 3B.1-3C surface"
                )

    def test_split_context_seeds_matches_exact_seed_sequence_order(self):
        from icgs.policies.reference import split_context_seeds

        full_demo_seeds, window_slot_seeds = split_context_seeds(
            17,
            demo_count=2,
            event_count=3,
        )
        self.assertEqual(full_demo_seeds, (3302413169, 2035845825))
        self.assertEqual(
            window_slot_seeds,
            (1193624495, 3388839295, 637109157),
        )
        self.assertEqual(
            (full_demo_seeds, window_slot_seeds),
            split_context_seeds(17, demo_count=2, event_count=3),
        )
        self.assertTrue(
            all(
                type(seed) is int
                for seed in full_demo_seeds + window_slot_seeds
            )
        )

    def test_split_reserves_every_window_slot_and_accepts_numpy_integers(self):
        from icgs.policies.reference import split_context_seeds

        full_demo_seeds, window_slot_seeds = split_context_seeds(
            np.int64(23),
            demo_count=np.int64(1),
            event_count=np.int64(3),
        )
        self.assertEqual(full_demo_seeds, (2987300412,))
        self.assertEqual(
            window_slot_seeds,
            (115931989, 2583041193, 4149776935),
        )
        # Slot seeds are allocated before materialization success is known. Ignoring
        # one result cannot shift the seeds assigned to later EventMemory slots.
        self.assertEqual(window_slot_seeds[2], 4149776935)

        full_only, no_windows = split_context_seeds(
            np.uint32(23), demo_count=np.uint8(1), event_count=np.uint8(0)
        )
        self.assertEqual(full_only, (2987300412,))
        self.assertEqual(no_windows, ())

    def test_split_context_seed_contract_rejects_malformed_values(self):
        from icgs.policies.reference import split_context_seeds

        invalid_seeds = (True, np.bool_(False), -1, np.int64(-1), 1.5, None)
        for value in invalid_seeds:
            with self.subTest(field="context_seed", value=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "context_seed.*nonnegative integer"
                ):
                    split_context_seeds(value, demo_count=1, event_count=0)

        invalid_demo_counts = (
            0,
            np.int64(0),
            -1,
            True,
            np.bool_(True),
            1.5,
            None,
        )
        for value in invalid_demo_counts:
            with self.subTest(field="demo_count", value=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "demo_count.*positive integer"
                ):
                    split_context_seeds(1, demo_count=value, event_count=0)

        invalid_event_counts = (
            -1,
            np.int64(-1),
            True,
            np.bool_(False),
            1.5,
            None,
        )
        for value in invalid_event_counts:
            with self.subTest(field="event_count", value=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "event_count.*nonnegative integer"
                ):
                    split_context_seeds(1, demo_count=1, event_count=value)

    def test_context_preparation_record_is_frozen_exact_and_non_normalizing(self):
        from icgs.policies.reference import (
            CONTEXT_RNG_PROTOCOL,
            ContextPreparationRecord,
        )

        record = ContextPreparationRecord(
            context_seed=np.int64(17),
            full_demo_seeds=(np.uint32(1), 2),
            window_slot_seeds=(np.int64(3), 4),
            rng_protocol=CONTEXT_RNG_PROTOCOL,
            full_context_id=" full-context ",
            window_context_ids=(" window-0 ", None),
        )
        self.assertEqual(
            tuple(ContextPreparationRecord.__dataclass_fields__),
            (
                "context_seed",
                "full_demo_seeds",
                "window_slot_seeds",
                "rng_protocol",
                "full_context_id",
                "window_context_ids",
            ),
        )
        self.assertEqual(record.full_context_id, " full-context ")
        self.assertEqual(record.window_context_ids, (" window-0 ", None))
        self.assertIsInstance(record.context_seed, np.integer)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.context_seed = 19

    def test_context_preparation_record_requires_exact_tuples_and_alignment(self):
        tuple_fields = (
            ("full_demo_seeds", [1, 2]),
            ("window_slot_seeds", [3, 4, 5]),
            ("window_context_ids", ["window-0", None, "window-2"]),
        )
        for field, value in tuple_fields:
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, f"{field}.*tuple"):
                    self._record(**{field: value})

        with self.assertRaisesRegex(ValueError, "full_demo_seeds.*nonempty"):
            self._record(full_demo_seeds=())
        with self.assertRaisesRegex(ValueError, "window.*align"):
            self._record(window_context_ids=("window-0", None))

    def test_context_preparation_record_validates_every_seed(self):
        invalid_values = (True, np.bool_(False), -1, np.int64(-1), 1.5, None)
        for value in invalid_values:
            cases = (
                ("context_seed", value),
                ("full_demo_seeds", (1, value)),
                ("window_slot_seeds", (3, value, 5)),
            )
            for field, replacement in cases:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(
                        (TypeError, ValueError), "nonnegative integer"
                    ):
                        self._record(**{field: replacement})

    def test_context_preparation_record_validates_protocol_and_identifiers(self):
        from icgs.policies.reference import CONTEXT_RNG_PROTOCOL

        self.assertEqual(CONTEXT_RNG_PROTOCOL, "seedsequence-native-choice-v1")
        for protocol in ("", "seedsequence-native-choice-v2", None):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "rng_protocol.*exactly"
                ):
                    self._record(rng_protocol=protocol)

        for identifier in ("", "   ", None, 7):
            with self.subTest(field="full_context_id", value=identifier):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "full_context_id.*nonempty string"
                ):
                    self._record(full_context_id=identifier)
        for identifier in ("", "   ", 7):
            with self.subTest(field="window_context_ids", value=identifier):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "window_context_ids.*nonempty string"
                ):
                    self._record(
                        window_context_ids=("window-0", identifier, "window-2")
                    )

        self.assertIsNone(self._record().window_context_ids[1])

    def test_success_and_malformed_input_preserve_all_global_rng_states(self):
        from icgs.policies.reference import split_context_seeds

        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        split_context_seeds(17, demo_count=2, event_count=3)
        self._record()
        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

        with self.assertRaisesRegex(ValueError, "demo_count.*positive integer"):
            split_context_seeds(17, demo_count=0, event_count=3)
        with self.assertRaisesRegex(ValueError, "rng_protocol.*exactly"):
            self._record(rng_protocol="wrong")
        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_public_surface_and_static_dependency_boundary(self):
        from icgs.policies.reference import (
            ContextPreparationRecord,
            split_context_seeds,
        )

        parameters = inspect.signature(split_context_seeds).parameters
        self.assertEqual(
            tuple(parameters), ("context_seed", "demo_count", "event_count")
        )
        self.assertEqual(
            parameters["demo_count"].kind, inspect.Parameter.KEYWORD_ONLY
        )
        self.assertEqual(
            parameters["event_count"].kind, inspect.Parameter.KEYWORD_ONLY
        )

        self._assert_task_3b0_dependency_boundary(
            split_context_seeds,
            ContextPreparationRecord,
        )

        def violating_surface():
            return scoped_seed

        with self.assertRaisesRegex(
            AssertionError,
            "violating_surface.*scoped_seed.*Task 3B.1-3C",
        ):
            self._assert_task_3b0_dependency_boundary(violating_surface)


class NativeDemoMaterializationTests(unittest.TestCase):
    _TASK_3B1B_FORBIDDEN_DEPENDENCIES = frozenset(
        {
            "ReferenceSessions",
            "InstantPolicy",
            "PreparedContext",
            "MethodContext",
            "materialize_indexed_native_demo",
            "build_method_context",
            "prepare_context",
            "composition",
        }
    )

    @staticmethod
    def _pose(boundary):
        pose = np.eye(4)
        pose[0, 3] = 0.25 * boundary
        return pose

    @classmethod
    def _demo(cls, *, unique_points=False, guarded_commands=False, grips=None):
        class GuardedTransition(ExecutedTransition):
            _guard_command = False

            def __getattribute__(self, name):
                if name == "command" and object.__getattribute__(
                    self, "_guard_command"
                ):
                    raise AssertionError("materializer read transition.command")
                return super().__getattribute__(name)

        grips = tuple(boundary % 2 for boundary in range(5)) if grips is None else grips
        if len(grips) != 5:
            raise AssertionError("fixture requires five measured grips")
        observations = []
        for boundary in range(5):
            pose = cls._pose(boundary)
            if unique_points:
                local_points = np.stack(
                    (
                        np.arange(24, dtype=np.float64),
                        np.full(24, 2.0),
                        np.full(24, 3.0),
                    ),
                    axis=-1,
                )
            else:
                local_points = np.tile(
                    np.array([[1.0, 2.0, 3.0]]),
                    (6, 1),
                )
            world_points = local_points + pose[:3, 3]
            observations.append(
                TimedObservation(
                    Observation(world_points, pose, float(grips[boundary])),
                    boundary,
                    float(boundary),
                    float(boundary),
                    "sensor-v1",
                )
            )

        commands = []
        for boundary in range(4):
            commands.append(
                TimedCommand(cls._pose(99 + boundary), boundary % 2, 0.1)
            )
        transitions = []
        for index in range(4):
            transition_type = (
                GuardedTransition if guarded_commands else ExecutedTransition
            )
            transition = transition_type(
                observations[index],
                observations[index + 1],
                commands[index],
                0.1,
                1,
                "ok",
            )
            if guarded_commands:
                object.__setattr__(transition, "_guard_command", True)
            transitions.append(transition)
        return TimedDemoInput(tuple(transitions), "demo-content")

    @staticmethod
    def _identity_outlier_filter(points, **_kwargs):
        return np.asarray(points), np.arange(len(points))

    @staticmethod
    def _assert_numpy_state_equal(left, right):
        if left[0] != right[0] or left[2:] != right[2:]:
            raise AssertionError("NumPy global RNG metadata changed")
        np.testing.assert_array_equal(left[1], right[1])

    @staticmethod
    def _measured_observations(demo):
        return (
            demo.transitions[0].before.observation,
            *(transition.after.observation for transition in demo.transitions),
        )

    def _assert_task_3b1b_dependency_boundary(self, surface):
        source = textwrap.dedent(inspect.getsource(surface))
        tree = ast.parse(source)
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        referenced.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(alias.name for alias in node.names)
        found = sorted(
            dependency
            for dependency in self._TASK_3B1B_FORBIDDEN_DEPENDENCIES
            if dependency in referenced
            or any(dependency in item for item in imported)
            or dependency in source
        )
        if found:
            self.fail(
                f"Task 3B.1b surface {surface.__name__} must not reference "
                f"session/context/later-policy dependencies: {found}; keep "
                "that dependency in its owning Task 3B.2-3C surface"
            )

    def test_exact_indices_bypass_native_waypoint_selection(self):
        from icgs.data.preprocessing.native import extract_waypoints

        demo = self._demo(grips=(0, 0, 0, 0, 0))
        measured = self._measured_observations(demo)
        native_indices = tuple(
            extract_waypoints(
                np.array([observation.T_w_e for observation in measured]),
                np.array([observation.grip for observation in measured]),
                num_waypoints=2,
            )
        )
        exact_indices = (1, 3)
        self.assertEqual(native_indices, (0, 4))
        self.assertTrue(set(exact_indices).isdisjoint(native_indices))

        from icgs.policies.reference import materialize_indexed_native_demo

        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            result = materialize_indexed_native_demo(
                demo,
                exact_indices,
                point_seed=7,
                native_waypoint_count=2,
                native_point_count=3,
            )
        self.assertEqual(result["grips"], (0.0, 0.0))
        for output_pose, index in zip(result["T_w_es"], exact_indices):
            np.testing.assert_array_equal(output_pose, measured[index].T_w_e)

    def test_exact_indices_preserve_measured_pose_grip_and_ignore_commands(self):
        from icgs.policies.reference import materialize_indexed_native_demo

        demo = self._demo(guarded_commands=True)
        boundary_indices = (np.int64(1), np.uint8(2))
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            result = materialize_indexed_native_demo(
                demo,
                boundary_indices,
                point_seed=7,
                native_waypoint_count=np.int64(2),
                native_point_count=np.uint16(3),
            )

        self.assertEqual(set(result), {"obs", "grips", "T_w_es"})
        self.assertTrue(all(type(values) is tuple for values in result.values()))
        self.assertEqual(result["grips"], (1.0, 0.0))
        expected_observations = self._measured_observations(demo)
        for output_pose, index in zip(result["T_w_es"], boundary_indices):
            np.testing.assert_array_equal(
                output_pose,
                expected_observations[int(index)].T_w_e,
            )
        for points in result["obs"]:
            np.testing.assert_array_equal(
                points,
                np.tile(np.array([[1.0, 2.0, 3.0]]), (3, 1)),
            )

        source = inspect.getsource(materialize_indexed_native_demo)
        self.assertNotIn("extract_waypoints", source)

    def test_materializer_rejects_malformed_indices_before_preprocessing(self):
        from icgs.policies.reference import materialize_indexed_native_demo

        demo = self._demo()
        invalid_cases = (
            (object(), (0,), 1, 3, "TimedDemoInput"),
            (demo, [0], 1, 3, "tuple"),
            (demo, (), 1, 3, "nonempty"),
            (demo, (2, 1), 2, 3, "sorted"),
            (demo, (1, 1), 2, 3, "unique"),
            (demo, (True,), 1, 3, "bool|integer"),
            (demo, (1.5,), 1, 3, "integer"),
            (demo, (-1,), 1, 3, "nonnegative"),
            (demo, (5,), 1, 3, "range"),
            (demo, (1, 2), 3, 3, "native_waypoint_count|length"),
        )
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=AssertionError("preprocessing ran before validation"),
        ):
            for value, indices, waypoint_count, point_count, message in invalid_cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(
                        (TypeError, ValueError), message
                    ):
                        materialize_indexed_native_demo(
                            value,
                            indices,
                            point_seed=7,
                            native_waypoint_count=waypoint_count,
                            native_point_count=point_count,
                        )

    def test_materializer_rejects_invalid_explicit_native_counts(self):
        from icgs.policies.reference import materialize_indexed_native_demo

        demo = self._demo()
        invalid_values = (0, -1, True, np.bool_(False), 1.5, None)
        for value in invalid_values:
            with self.subTest(field="native_waypoint_count", value=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    "native_waypoint_count.*positive integer",
                ):
                    materialize_indexed_native_demo(
                        demo,
                        (1,),
                        point_seed=7,
                        native_waypoint_count=value,
                        native_point_count=3,
                    )
            with self.subTest(field="native_point_count", value=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    "native_point_count.*positive integer",
                ):
                    materialize_indexed_native_demo(
                        demo,
                        (1,),
                        point_seed=7,
                        native_waypoint_count=1,
                        native_point_count=value,
                    )

    def test_point_seed_changes_only_sampled_points(self):
        from icgs.policies.reference import materialize_indexed_native_demo

        demo = self._demo(unique_points=True)
        kwargs = {
            "native_waypoint_count": 2,
            "native_point_count": 4,
        }
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            first = materialize_indexed_native_demo(
                demo, (1, 3), point_seed=7, **kwargs
            )
            repeated = materialize_indexed_native_demo(
                demo, (1, 3), point_seed=7, **kwargs
            )
            changed = materialize_indexed_native_demo(
                demo, (1, 3), point_seed=8, **kwargs
            )

        for key in ("obs", "grips", "T_w_es"):
            for left, right in zip(first[key], repeated[key]):
                np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first["obs"][0][:, 0], (1.0, 5.0, 11.0, 13.0))
        np.testing.assert_array_equal(
            first["obs"][1][:, 0], (13.0, 15.0, 22.0, 19.0)
        )
        self.assertEqual(first["grips"], changed["grips"])
        for left, right in zip(first["T_w_es"], changed["T_w_es"]):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(
            any(
                not np.array_equal(left, right)
                for left, right in zip(first["obs"], changed["obs"])
            )
        )

    def test_materializer_does_not_mutate_inputs(self):
        from icgs.policies.reference import materialize_indexed_native_demo

        demo = self._demo(unique_points=True)
        transitions = demo.transitions
        indices = (1, 3)
        observations = self._measured_observations(demo)
        points_before = tuple(np.array(item.points, copy=True) for item in observations)
        poses_before = tuple(np.array(item.T_w_e, copy=True) for item in observations)
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            materialize_indexed_native_demo(
                demo,
                indices,
                point_seed=7,
                native_waypoint_count=2,
                native_point_count=4,
            )
        self.assertEqual(indices, (1, 3))
        self.assertIs(demo.transitions, transitions)
        for observation, points, pose in zip(
            observations, points_before, poses_before
        ):
            np.testing.assert_array_equal(observation.points, points)
            np.testing.assert_array_equal(observation.T_w_e, pose)

    def test_materializer_restores_global_rng_on_success_and_failure(self):
        from icgs.policies.reference import materialize_indexed_native_demo

        demo = self._demo(unique_points=True)
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            materialize_indexed_native_demo(
                demo,
                (1, 3),
                point_seed=7,
                native_waypoint_count=2,
                native_point_count=4,
            )
        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=RuntimeError("injected preprocessing failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected preprocessing"):
                materialize_indexed_native_demo(
                    demo,
                    (1, 3),
                    point_seed=11,
                    native_waypoint_count=2,
                    native_point_count=4,
                )
        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_full_context_keeps_real_native_waypoint_selection(self):
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(grips=(0, 0, 0, 0, 0))
        measured = self._measured_observations(demo)
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            result = materialize_full_native_demo(
                demo,
                point_seed=7,
                native_waypoint_count=2,
                native_point_count=3,
            )

        self.assertEqual(result["grips"], (0.0, 0.0))
        np.testing.assert_array_equal(result["T_w_es"][0], measured[0].T_w_e)
        np.testing.assert_array_equal(result["T_w_es"][1], measured[4].T_w_e)

    def test_full_context_calls_native_collaborator_once_with_explicit_counts(self):
        from icgs.policies import reference
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(
            guarded_commands=True,
            grips=(0, 0, 0, 0, 0),
        )
        measured = self._measured_observations(demo)
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ), mock.patch(
            "icgs.policies.reference.sample_to_cond_demo",
            wraps=reference.sample_to_cond_demo,
        ) as collaborator:
            result = materialize_full_native_demo(
                demo,
                point_seed=np.int64(7),
                native_waypoint_count=np.uint8(2),
                native_point_count=np.uint16(3),
            )

        collaborator.assert_called_once()
        args, kwargs = collaborator.call_args
        self.assertEqual(len(args), 2)
        raw_demo, waypoint_count = args
        self.assertEqual(set(raw_demo), {"pcds", "grips", "T_w_es"})
        self.assertTrue(all(type(values) is tuple for values in raw_demo.values()))
        self.assertEqual(waypoint_count, 2)
        self.assertEqual(kwargs, {"num_points": 3})
        for raw_points, observation in zip(raw_demo["pcds"], measured):
            np.testing.assert_array_equal(raw_points, observation.points)
        self.assertEqual(raw_demo["grips"], tuple(item.grip for item in measured))
        for raw_pose, observation in zip(raw_demo["T_w_es"], measured):
            np.testing.assert_array_equal(raw_pose, observation.T_w_e)
        self.assertEqual(set(result), {"obs", "grips", "T_w_es"})
        self.assertTrue(all(type(values) is tuple for values in result.values()))

    def test_full_context_rejects_bad_inputs_before_native_collaborator(self):
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(grips=(0, 0, 0, 0, 0))
        cases = (
            (object(), 7, 2, 3, "demo.*TimedDemoInput"),
            (demo, True, 2, 3, "point_seed.*nonnegative integer"),
            (demo, -1, 2, 3, "point_seed.*nonnegative integer"),
            (demo, 1.5, 2, 3, "point_seed.*nonnegative integer"),
            (demo, 7, True, 3, "native_waypoint_count.*positive integer"),
            (demo, 7, 0, 3, "native_waypoint_count.*positive integer"),
            (demo, 7, 2, np.bool_(False), "native_point_count.*positive integer"),
            (demo, 7, 2, 0, "native_point_count.*positive integer"),
        )
        with mock.patch(
            "icgs.policies.reference.sample_to_cond_demo",
            side_effect=AssertionError("native collaborator ran before validation"),
        ):
            for value, seed, waypoint_count, point_count, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        materialize_full_native_demo(
                            value,
                            point_seed=seed,
                            native_waypoint_count=waypoint_count,
                            native_point_count=point_count,
                        )

    def test_full_context_rejects_malformed_native_results_without_repair(self):
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(grips=(0, 0, 0, 0, 0))
        pose = self._pose(0)
        points = np.zeros((3, 3))
        malformed = (
            ([], "mapping"),
            ({"obs": (), "grips": ()}, "fields"),
            (
                {"obs": (), "grips": (), "T_w_es": (), "extra": ()},
                "fields",
            ),
            ({"obs": 1, "grips": (), "T_w_es": ()}, "sequence"),
            (
                {"obs": (points,), "grips": (), "T_w_es": (pose,)},
                "cardinalities",
            ),
            (
                {"obs": (points,), "grips": (0.0,), "T_w_es": (pose,)},
                "native_waypoint_count",
            ),
        )
        for result, message in malformed:
            with self.subTest(message=message):
                with mock.patch(
                    "icgs.policies.reference.sample_to_cond_demo",
                    return_value=result,
                ) as collaborator:
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        materialize_full_native_demo(
                            demo,
                            point_seed=7,
                            native_waypoint_count=2,
                            native_point_count=3,
                        )
                    collaborator.assert_called_once()

    def test_full_context_seed_changes_only_native_sampled_points(self):
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(
            unique_points=True,
            grips=(0, 0, 0, 0, 0),
        )
        kwargs = {
            "native_waypoint_count": 2,
            "native_point_count": 4,
        }
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            first = materialize_full_native_demo(
                demo, point_seed=7, **kwargs
            )
            repeated = materialize_full_native_demo(
                demo, point_seed=7, **kwargs
            )
            changed = materialize_full_native_demo(
                demo, point_seed=8, **kwargs
            )

        for key in ("obs", "grips", "T_w_es"):
            for left, right in zip(first[key], repeated[key]):
                np.testing.assert_array_equal(left, right)
        self.assertEqual(first["grips"], changed["grips"])
        for left, right in zip(first["T_w_es"], changed["T_w_es"]):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(
            any(
                not np.array_equal(left, right)
                for left, right in zip(first["obs"], changed["obs"])
            )
        )

    def test_full_context_does_not_mutate_demo(self):
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(
            unique_points=True,
            grips=(0, 0, 0, 0, 0),
        )
        transitions = demo.transitions
        observations = self._measured_observations(demo)
        points_before = tuple(np.array(item.points, copy=True) for item in observations)
        poses_before = tuple(np.array(item.T_w_e, copy=True) for item in observations)
        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            materialize_full_native_demo(
                demo,
                point_seed=7,
                native_waypoint_count=2,
                native_point_count=4,
            )

        self.assertIs(demo.transitions, transitions)
        for observation, points, pose in zip(
            observations, points_before, poses_before
        ):
            np.testing.assert_array_equal(observation.points, points)
            np.testing.assert_array_equal(observation.T_w_e, pose)

    def test_full_context_restores_global_rng_on_success_and_failure(self):
        from icgs.policies.reference import materialize_full_native_demo

        demo = self._demo(
            unique_points=True,
            grips=(0, 0, 0, 0, 0),
        )
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        with mock.patch(
            "icgs.data.preprocessing.native.remove_statistical_outliers",
            side_effect=self._identity_outlier_filter,
        ):
            materialize_full_native_demo(
                demo,
                point_seed=7,
                native_waypoint_count=2,
                native_point_count=4,
            )
        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

        with mock.patch(
            "icgs.policies.reference.sample_to_cond_demo",
            side_effect=RuntimeError("injected native conversion failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected native conversion"):
                materialize_full_native_demo(
                    demo,
                    point_seed=11,
                    native_waypoint_count=2,
                    native_point_count=4,
                )
        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))

    def test_full_context_function_scoped_dependency_boundary(self):
        from icgs.policies.reference import materialize_full_native_demo

        self._assert_task_3b1b_dependency_boundary(materialize_full_native_demo)
        source = inspect.getsource(materialize_full_native_demo)
        self.assertNotIn("2048", source)

        def violating_surface():
            return materialize_indexed_native_demo

        with self.assertRaisesRegex(
            AssertionError,
            "violating_surface.*materialize_indexed_native_demo.*Task 3B.2-3C",
        ):
            self._assert_task_3b1b_dependency_boundary(violating_surface)


class ReferenceSessionsValidationTests(unittest.TestCase):
    _FORBIDDEN_SESSION_DEPENDENCIES = frozenset(
        {
            "PreparedContext",
            "MethodContext",
            "build_method_context",
            "prepare_context",
            "predict",
            "load_policy",
            "load_published_policy",
            "resolve_native_profile",
            "artifacts",
            "composition",
            "open",
        }
    )
    _CHECKPOINT = "a" * 64
    _REFERENCE_ID = "b" * 64
    _NATIVE_PROFILE = "instant_policy_published_vv19_119fa871"

    class _Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("scratch_buffer", torch.zeros(2))
            self.graph = SimpleNamespace(graph=object())
            self.codec = object()

    @staticmethod
    def _session_id(reference_id, role):
        payload = {
            "domain": "icgs.reference-session",
            "schema": 1,
            "reference_id": reference_id,
            "role": role,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _configs():
        from icgs.configuration.defaults import instant_policy_original

        base = instant_policy_original()
        d1 = dataclasses.replace(
            base,
            graph=dataclasses.replace(base.graph, num_demos=1),
            runtime=dataclasses.replace(base.runtime, device="cpu"),
        )
        d2 = dataclasses.replace(
            d1,
            graph=dataclasses.replace(d1.graph, num_demos=2),
        )
        return d1, d2

    @classmethod
    def _policy(cls, config, *, checksum=None):
        from icgs.policies.instant_policy import InstantPolicy

        scheduler = object()
        network = cls._Network()
        sampler = SimpleNamespace(
            config=config.sampling,
            diffusion=config.diffusion,
            codec=network.codec,
            noise_scheduler=scheduler,
        )
        objective = SimpleNamespace(
            config=config.diffusion,
            codec=network.codec,
            noise_scheduler=scheduler,
        )
        policy = InstantPolicy(
            network,
            sampler,
            objective,
            config.graph,
            config.runtime,
        )
        policy.artifact_sha256 = checksum or cls._CHECKPOINT
        return policy

    @classmethod
    def _values(cls):
        d1_config, d2_config = cls._configs()
        return {
            "d1": cls._policy(d1_config),
            "d2": cls._policy(d2_config),
            "d1_config": d1_config,
            "d2_config": d2_config,
            "reference_id": cls._REFERENCE_ID,
            "native_profile": cls._NATIVE_PROFILE,
            "checkpoint_sha256": cls._CHECKPOINT,
            "native_point_count": 2048,
            "d1_session_id": cls._session_id(cls._REFERENCE_ID, "d1"),
            "d2_session_id": cls._session_id(cls._REFERENCE_ID, "d2"),
        }

    def _assert_session_dependency_boundary(self, surface):
        source = textwrap.dedent(inspect.getsource(surface))
        tree = ast.parse(source)
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        referenced.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        found = sorted(
            dependency
            for dependency in self._FORBIDDEN_SESSION_DEPENDENCIES
            if dependency in referenced or dependency in source
        )
        if found:
            self.fail(
                f"Task 3B.2a surface {surface.__name__} must remain "
                f"validation-only and not reference IO/context/inference "
                f"dependencies: {found}; move them to Task 3B.2b/3B.3/3C"
            )

    def test_reference_sessions_is_frozen_canonical_and_nonmutating(self):
        from icgs.policies.reference import ReferenceSessions

        values = self._values()
        ownership = {}
        parameters = {}
        buffers = {}
        for role in ("d1", "d2"):
            policy = values[role]
            ownership[role] = (
                policy.context_owner,
                policy.network,
                policy.network.graph,
                policy.network.graph.graph,
                policy.network.codec,
                policy.sampler,
                policy.sampler.codec,
                policy.sampler.noise_scheduler,
                policy.objective,
                policy.objective.codec,
                policy.objective.noise_scheduler,
                policy.graph_config,
                policy.runtime,
            )
            parameters[role] = tuple(
                (name, id(parameter), parameter.detach().clone())
                for name, parameter in policy.network.named_parameters()
            )
            buffers[role] = tuple(
                (name, id(buffer), buffer.detach().clone())
                for name, buffer in policy.network.named_buffers()
            )
        sessions = ReferenceSessions(**values)

        self.assertEqual(
            tuple(ReferenceSessions.__dataclass_fields__),
            (
                "d1",
                "d2",
                "d1_config",
                "d2_config",
                "reference_id",
                "native_profile",
                "checkpoint_sha256",
                "native_point_count",
                "d1_session_id",
                "d2_session_id",
            ),
        )
        self.assertIs(sessions.d1, values["d1"])
        self.assertIs(sessions.d2, values["d2"])
        self.assertIs(sessions.d1_config, values["d1_config"])
        self.assertIs(sessions.d2_config, values["d2_config"])
        for role in ("d1", "d2"):
            policy = getattr(sessions, role)
            self.assertEqual(
                tuple(
                    id(owner)
                    for owner in (
                        policy.context_owner,
                        policy.network,
                        policy.network.graph,
                        policy.network.graph.graph,
                        policy.network.codec,
                        policy.sampler,
                        policy.sampler.codec,
                        policy.sampler.noise_scheduler,
                        policy.objective,
                        policy.objective.codec,
                        policy.objective.noise_scheduler,
                        policy.graph_config,
                        policy.runtime,
                    )
                ),
                tuple(id(owner) for owner in ownership[role]),
            )
            self.assertIs(
                policy.sampler.noise_scheduler,
                policy.objective.noise_scheduler,
            )
            actual_parameters = tuple(policy.network.named_parameters())
            self.assertEqual(
                tuple(name for name, _, _ in parameters[role]),
                tuple(name for name, _ in actual_parameters),
            )
            for (name, identity, expected), (actual_name, actual) in zip(
                parameters[role], actual_parameters
            ):
                self.assertEqual(actual_name, name)
                self.assertEqual(id(actual), identity)
                torch.testing.assert_close(actual, expected)
            actual_buffers = tuple(policy.network.named_buffers())
            self.assertEqual(
                tuple(name for name, _, _ in buffers[role]),
                tuple(name for name, _ in actual_buffers),
            )
            for (name, identity, expected), (actual_name, actual) in zip(
                buffers[role], actual_buffers
            ):
                self.assertEqual(actual_name, name)
                self.assertEqual(id(actual), identity)
                torch.testing.assert_close(actual, expected)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            sessions.native_point_count = 1024

    def test_reference_sessions_requires_canonical_configs_and_policy_correspondence(self):
        from icgs.policies.reference import ReferenceSessions

        cases = []

        values = self._values()
        wrong_d1 = dataclasses.replace(
            values["d1_config"],
            graph=dataclasses.replace(values["d1_config"].graph, num_demos=2),
        )
        values["d1_config"] = wrong_d1
        values["d1"] = self._policy(wrong_d1)
        cases.append((values, "D1|num_demos"))

        values = self._values()
        wrong_d2 = dataclasses.replace(
            values["d2_config"],
            sampling=dataclasses.replace(values["d2_config"].sampling, steps=3),
        )
        values["d2_config"] = wrong_d2
        values["d2"] = self._policy(wrong_d2)
        cases.append((values, "differ|config|num_demos"))

        for role in ("d1", "d2"):
            config_name = f"{role}_config"

            values = self._values()
            config = values[config_name]
            values[role].graph_config = dataclasses.replace(
                config.graph,
                traj_horizon=config.graph.traj_horizon + 1,
            )
            cases.append((values, "graph.*config"))

            values = self._values()
            config = values[config_name]
            values[role].runtime = dataclasses.replace(
                config.runtime,
                cache_context=not config.runtime.cache_context,
            )
            cases.append((values, "runtime.*config"))

            values = self._values()
            config = values[config_name]
            values[role].sampler.config = dataclasses.replace(
                config.sampling,
                steps=config.sampling.steps + 1,
            )
            cases.append((values, "sampler.*config"))

            values = self._values()
            config = values[config_name]
            values[role].sampler.diffusion = dataclasses.replace(
                config.diffusion,
                train_steps=config.diffusion.train_steps + 1,
            )
            cases.append((values, "sampler.*diffusion"))

            values = self._values()
            config = values[config_name]
            values[role].objective.config = dataclasses.replace(
                config.diffusion,
                train_steps=config.diffusion.train_steps + 1,
            )
            cases.append((values, "objective.*config"))

            other_role = "d2" if role == "d1" else "d1"
            for collaborator_name in ("sampler", "objective"):
                values = self._values()
                collaborator = getattr(values[role], collaborator_name)
                collaborator.codec = values[other_role].network.codec
                cases.append((values, f"{collaborator_name}.*codec|codec"))

        values = self._values()
        values["d2"].artifact_sha256 = "c" * 64
        cases.append((values, "artifact|checkpoint"))

        for values, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    ReferenceSessions(**values)

    def test_reference_sessions_validates_canonical_lineage_and_point_count(self):
        from icgs.policies.reference import ReferenceSessions

        cases = (
            ("reference_id", "", "reference_id"),
            ("reference_id", "   ", "reference_id"),
            ("native_profile", "", "native_profile"),
            ("native_profile", None, "native_profile"),
            ("checkpoint_sha256", "A" * 64, "checkpoint.*canonical|SHA"),
            ("checkpoint_sha256", "abc", "checkpoint.*SHA"),
            ("native_point_count", 0, "native_point_count.*positive"),
            ("native_point_count", -1, "native_point_count.*positive"),
            ("native_point_count", 1.5, "native_point_count.*positive"),
            ("native_point_count", None, "native_point_count.*positive"),
            ("native_point_count", True, "native_point_count.*positive"),
            ("native_point_count", np.bool_(True), "native_point_count.*positive"),
            ("d1_session_id", "c" * 64, "d1_session_id|canonical"),
            ("d2_session_id", "c" * 64, "d2_session_id|canonical"),
        )
        for field, value, message in cases:
            values = self._values()
            values[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    ReferenceSessions(**values)

    def test_reference_sessions_rejects_cross_policy_mutable_owner_aliases(self):
        from icgs.policies.reference import ReferenceSessions

        def share_d1_scheduler(values):
            scheduler = values["d1"].sampler.noise_scheduler
            values["d2"].sampler.noise_scheduler = scheduler
            values["d2"].objective.noise_scheduler = scheduler

        def share_d1_sampler(values):
            values["d2"].sampler = values["d1"].sampler
            values["d2"].objective.noise_scheduler = (
                values["d1"].sampler.noise_scheduler
            )

        def share_d1_objective(values):
            values["d2"].objective = values["d1"].objective
            values["d2"].sampler.noise_scheduler = (
                values["d1"].objective.noise_scheduler
            )

        def share_d1_network(values):
            values["d2"].network = values["d1"].network
            values["d2"].sampler.codec = values["d1"].network.codec
            values["d2"].objective.codec = values["d1"].network.codec

        def share_d1_network_codec(values):
            codec = values["d1"].network.codec
            values["d2"].network.codec = codec
            values["d2"].sampler.codec = codec
            values["d2"].objective.codec = codec

        mutations = (
            (
                "policy",
                lambda values: values.update(d2=values["d1"]),
                "alias|distinct",
            ),
            (
                "context_owner",
                lambda values: setattr(
                    values["d2"], "context_owner", values["d1"].context_owner
                ),
                "alias|distinct",
            ),
            (
                "network",
                share_d1_network,
                "alias|distinct",
            ),
            (
                "network.graph",
                lambda values: setattr(
                    values["d2"].network,
                    "graph",
                    values["d1"].network.graph,
                ),
                "alias|distinct",
            ),
            (
                "network.graph.graph",
                lambda values: setattr(
                    values["d2"].network.graph,
                    "graph",
                    values["d1"].network.graph.graph,
                ),
                "alias|distinct",
            ),
            (
                "network.codec",
                share_d1_network_codec,
                "alias|distinct",
            ),
            (
                "sampler",
                share_d1_sampler,
                "sampler.*codec|codec",
            ),
            (
                "scheduler",
                share_d1_scheduler,
                "alias|distinct",
            ),
            (
                "objective",
                share_d1_objective,
                "objective.*codec|codec",
            ),
        )
        for name, mutate, message in mutations:
            values = self._values()
            mutate(values)
            with self.subTest(owner=name):
                with self.assertRaisesRegex(ValueError, message):
                    ReferenceSessions(**values)

        for role in ("d1", "d2"):
            values = self._values()
            values[role].objective.noise_scheduler = object()
            with self.subTest(owner=f"{role}.internal_scheduler"):
                with self.assertRaisesRegex(ValueError, "scheduler|share|same"):
                    ReferenceSessions(**values)

    def test_reference_sessions_rejects_cross_policy_parameter_and_buffer_storage(self):
        from icgs.policies.reference import ReferenceSessions

        values = self._values()
        values["d2"].network.weight = torch.nn.Parameter(
            values["d1"].network.weight
        )
        with self.assertRaisesRegex(ValueError, "parameter|storage|alias"):
            ReferenceSessions(**values)

        values = self._values()
        values["d2"].network.scratch_buffer = values["d1"].network.scratch_buffer
        with self.assertRaisesRegex(ValueError, "buffer|storage|alias"):
            ReferenceSessions(**values)

        disjoint_backing = np.arange(4, dtype=np.float32)
        values = self._values()
        values["d1"].network.weight = torch.nn.Parameter(
            torch.from_numpy(disjoint_backing[:2])
        )
        values["d2"].network.weight = torch.nn.Parameter(
            torch.from_numpy(disjoint_backing[2:])
        )
        ReferenceSessions(**values)

        overlapping_backing = np.arange(6, dtype=np.float32)
        values = self._values()
        values["d1"].network.weight = torch.nn.Parameter(
            torch.from_numpy(overlapping_backing[:4])
        )
        values["d2"].network.weight = torch.nn.Parameter(
            torch.from_numpy(overlapping_backing[2:])
        )
        with self.assertRaisesRegex(ValueError, "parameter|storage|overlap|alias"):
            ReferenceSessions(**values)

    def test_reference_sessions_has_validation_only_scoped_dependency_guard(self):
        from icgs.policies.reference import ReferenceSessions

        self._assert_session_dependency_boundary(ReferenceSessions)

        def violating_surface():
            return load_published_policy

        with self.assertRaisesRegex(
            AssertionError,
            "violating_surface.*load_published_policy.*Task 3B.2b",
        ):
            self._assert_session_dependency_boundary(violating_surface)


class ReferenceSessionFactoryTests(unittest.TestCase):
    _PROFILE_ID = "instant_policy_published_vv19_119fa871"
    _ORIGINAL_PROFILE_ID = "instant-policy-original-65dc94e"
    _CHECKPOINT = (
        "119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5"
    )
    _REFERENCE_ID = "reference-fixture-v1"

    @staticmethod
    def _profile_path():
        return (
            Path(__file__).resolve().parent.parent
            / "src/icgs/artifacts/profiles/vv19-119fa871.json"
        )

    @staticmethod
    def _resolved_profile(d1_config, d2_config):
        def config_for(*, num_demos, device):
            if device != "cpu":
                raise AssertionError("fixture expected the injected CPU device")
            if num_demos == 1:
                return d1_config
            if num_demos == 2:
                return d2_config
            raise AssertionError("fixture expected only D1/D2 config requests")

        return SimpleNamespace(
            profile_id=ReferenceSessionFactoryTests._PROFILE_ID,
            artifact_sha256=ReferenceSessionFactoryTests._CHECKPOINT,
            native_point_count=2048,
            config_for=mock.Mock(side_effect=config_for),
        )

    @staticmethod
    def _factory_configs():
        d1_config, d2_config = ReferenceSessionsValidationTests._configs()
        return d1_config, d2_config

    @staticmethod
    def _policy(config):
        return ReferenceSessionsValidationTests._policy(
            config,
            checksum=ReferenceSessionFactoryTests._CHECKPOINT,
        )

    def _assert_loader_profile_boundary(self, surface):
        source = textwrap.dedent(inspect.getsource(surface))
        tree = ast.parse(source)
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        referenced.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        forbidden = {
            "files",
            "read_text",
            "json",
            "from_legacy",
            "replace",
            "published_config",
        }
        found = sorted(forbidden & referenced)
        if found or "resolve_native_profile" not in referenced:
            self.fail(
                f"{surface.__name__} must delegate profile/config authority to "
                "resolve_native_profile and must not read or reconstruct it; "
                f"forbidden references: {found}"
            )

    def _assert_factory_dependency_boundary(self, surface):
        source = textwrap.dedent(inspect.getsource(surface))
        tree = ast.parse(source)
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        referenced.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        required = {
            "resolve_native_profile",
            "load_published_policy",
            "ReferenceSessions",
        }
        forbidden = {
            "PreparedContext",
            "MethodContext",
            "prepare_context",
            "predict",
            "materialize_indexed_native_demo",
            "materialize_full_native_demo",
        }
        missing = sorted(required - referenced)
        found = sorted(forbidden & referenced)
        if missing or found:
            self.fail(
                f"{surface.__name__} must remain a Task 3B.2b outer session "
                f"factory; missing owners: {missing}; forbidden 3B.3/3C "
                f"references: {found}"
            )

    @contextmanager
    def _patched_factory_collaborators(self, *, profile, loader_side_effect):
        """Patch either supported import style without making syntax contractual."""
        from icgs import composition
        from icgs.artifacts import published

        resolver = mock.Mock(return_value=profile)
        loader = mock.Mock(side_effect=loader_side_effect)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    published,
                    "resolve_native_profile",
                    resolver,
                    create=True,
                )
            )
            stack.enter_context(
                mock.patch.object(published, "load_published_policy", loader)
            )
            if hasattr(composition, "resolve_native_profile"):
                stack.enter_context(
                    mock.patch.object(
                        composition,
                        "resolve_native_profile",
                        resolver,
                    )
                )
            if hasattr(composition, "load_published_policy"):
                stack.enter_context(
                    mock.patch.object(
                        composition,
                        "load_published_policy",
                        loader,
                    )
                )
            yield resolver, loader

    @contextmanager
    def _patched_sessions_constructor(self, *, side_effect):
        """Intercept construction for local-import and module-alias factories."""
        from icgs import composition
        from icgs.policies import reference

        constructor = mock.Mock(side_effect=side_effect)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(reference, "ReferenceSessions", constructor)
            )
            if hasattr(composition, "ReferenceSessions"):
                stack.enter_context(
                    mock.patch.object(
                        composition,
                        "ReferenceSessions",
                        constructor,
                    )
                )
            yield constructor

    def test_published_profile_json_has_authoritative_identity_and_point_count(self):
        payload = json.loads(self._profile_path().read_text())

        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("profile_id", payload, "published profile needs profile_id")
        self.assertEqual(payload["profile_id"], self._PROFILE_ID)
        self.assertEqual(payload["artifact_sha256"], self._CHECKPOINT)
        self.assertIn(
            "preprocessing",
            payload,
            "published profile needs authoritative preprocessing metadata",
        )
        preprocessing = payload["preprocessing"]
        self.assertIn("native_point_count", preprocessing)
        self.assertIs(type(preprocessing["native_point_count"]), int)
        self.assertGreater(preprocessing["native_point_count"], 0)
        self.assertEqual(preprocessing["native_point_count"], 2048)

    def test_resolver_rejects_unknown_profiles_and_derives_exact_configs(self):
        from icgs.artifacts.published import resolve_native_profile

        profile = resolve_native_profile(self._PROFILE_ID)
        self.assertEqual(profile.profile_id, self._PROFILE_ID)
        self.assertEqual(profile.artifact_sha256, self._CHECKPOINT)
        self.assertEqual(profile.native_point_count, 2048)

        d1_config = profile.config_for(num_demos=1, device="cpu")
        d2_config = profile.config_for(num_demos=2, device="cpu")
        self.assertEqual(d1_config.graph.num_demos, 1)
        self.assertEqual(
            d2_config,
            dataclasses.replace(
                d1_config,
                graph=dataclasses.replace(d1_config.graph, num_demos=2),
            ),
        )
        for unknown in ("unknown-native-profile", self._ORIGINAL_PROFILE_ID):
            with self.subTest(native_profile=unknown):
                with self.assertRaisesRegex(ValueError, "unknown.*profile"):
                    resolve_native_profile(unknown)

    def test_published_loader_delegates_profile_and_config_authority(self):
        from icgs.artifacts import published

        self._assert_loader_profile_boundary(published.load_published_policy)

        def violating_surface():
            return published_config

        with self.assertRaisesRegex(
            AssertionError,
            "violating_surface.*resolve_native_profile.*published_config",
        ):
            self._assert_loader_profile_boundary(violating_surface)

        profile = published.resolve_native_profile(self._PROFILE_ID)
        config = profile.config_for(num_demos=1, device="cpu")
        policy = self._policy(config)
        with mock.patch.object(
            published,
            "resolve_native_profile",
            wraps=published.resolve_native_profile,
        ) as resolver_spy, mock.patch.object(
            published,
            "sha256_file",
            return_value=self._CHECKPOINT,
        ), mock.patch(
            "torch.load",
            return_value={"state_dict": {}},
        ), mock.patch(
            "icgs.composition.build_policy",
            return_value=policy,
        ) as build_spy, mock.patch(
            "icgs.artifacts.checkpoints.load_state_dict_compatible",
            return_value="strict-report",
        ):
            loaded = published.load_published_policy(
                "/missing/model.pt",
                native_profile=self._PROFILE_ID,
                config=config,
            )

        resolver_spy.assert_called_once_with(self._PROFILE_ID)
        build_spy.assert_called_once_with(config)
        self.assertIs(loaded, policy)
        self.assertEqual(loaded.artifact_sha256, self._CHECKPOINT)

    def test_loader_rejects_unknown_profile_before_checkpoint_or_model_io(self):
        from icgs.artifacts import published

        for profile_id in ("unknown-native-profile", self._ORIGINAL_PROFILE_ID):
            resolver = mock.Mock(
                side_effect=ValueError(f"unknown native profile: {profile_id!r}")
            )
            with self.subTest(native_profile=profile_id), mock.patch.object(
                published,
                "resolve_native_profile",
                resolver,
                create=True,
            ), mock.patch.object(
                published,
                "sha256_file",
                side_effect=AssertionError("checkpoint IO occurred"),
            ), mock.patch(
                "torch.load",
                side_effect=AssertionError("model IO occurred"),
            ):
                with self.assertRaisesRegex(ValueError, "unknown.*profile"):
                    published.load_published_policy(
                        "/missing/model.pt",
                        native_profile=profile_id,
                        device="cpu",
                    )
                resolver.assert_called_once_with(profile_id)

    def test_builder_loads_exact_d1_then_d2_and_record_reuse_does_not_reload(self):
        from icgs.composition import build_reference_sessions
        from icgs.policies.reference import ReferenceSessions

        d1_config, d2_config = self._factory_configs()
        profile = self._resolved_profile(d1_config, d2_config)
        policies = (self._policy(d1_config), self._policy(d2_config))
        order = []

        def load_policy(checkpoint, *, native_profile, config):
            order.append(f"load:d{config.graph.num_demos}")
            return policies[config.graph.num_demos - 1]

        def construct_sessions(*args, **kwargs):
            order.append("construct")
            return ReferenceSessions(*args, **kwargs)

        policy_state = tuple(
            (
                policy.graph_config,
                policy.runtime,
                policy.context_owner,
                policy.network,
                policy.network.graph.graph,
            )
            for policy in policies
        )

        with self._patched_factory_collaborators(
            profile=profile,
            loader_side_effect=load_policy,
        ) as (resolver, loader), self._patched_sessions_constructor(
            side_effect=construct_sessions,
        ) as constructor:
            sessions = build_reference_sessions(
                "/checkpoint/model.pt",
                reference_id=self._REFERENCE_ID,
                native_profile=self._PROFILE_ID,
                device="cpu",
            )
            self.assertIsNot(sessions.d1, sessions.d2)
            self.assertIs(sessions.d1, policies[0])
            self.assertIs(sessions.d2, policies[1])

            for policy, expected in zip(policies, policy_state):
                self.assertEqual(policy.graph_config, expected[0])
                self.assertEqual(policy.runtime, expected[1])
                self.assertIs(policy.context_owner, expected[2])
                self.assertIs(policy.network, expected[3])
                self.assertIs(policy.network.graph.graph, expected[4])
            self.assertIs(sessions.d1_config, d1_config)
            self.assertIs(sessions.d2_config, d2_config)
            self.assertEqual(sessions.reference_id, self._REFERENCE_ID)
            self.assertEqual(sessions.native_profile, profile.profile_id)
            self.assertEqual(
                sessions.checkpoint_sha256,
                profile.artifact_sha256,
            )
            self.assertEqual(
                sessions.native_point_count,
                profile.native_point_count,
            )
            self.assertEqual(
                sessions.d1_session_id,
                ReferenceSessionsValidationTests._session_id(
                    self._REFERENCE_ID,
                    "d1",
                ),
            )
            self.assertEqual(
                sessions.d2_session_id,
                ReferenceSessionsValidationTests._session_id(
                    self._REFERENCE_ID,
                    "d2",
                ),
            )

            self.assertIs(sessions.d1, policies[0])
            self.assertIs(sessions.d2, policies[1])

        self.assertEqual(order, ["load:d1", "load:d2", "construct"])
        constructor.assert_called_once()
        resolver.assert_called_once_with(self._PROFILE_ID)
        self.assertEqual(
            profile.config_for.call_args_list,
            [
                mock.call(num_demos=1, device="cpu"),
                mock.call(num_demos=2, device="cpu"),
            ],
        )
        self.assertEqual(
            loader.call_args_list,
            [
                mock.call(
                    "/checkpoint/model.pt",
                    native_profile=self._PROFILE_ID,
                    config=d1_config,
                ),
                mock.call(
                    "/checkpoint/model.pt",
                    native_profile=self._PROFILE_ID,
                    config=d2_config,
                ),
            ],
        )

    def test_builder_second_load_failure_does_not_return_partial_record(self):
        from icgs.composition import build_reference_sessions
        from icgs.policies.reference import ReferenceSessions

        d1_config, d2_config = self._factory_configs()
        profile = self._resolved_profile(d1_config, d2_config)
        d1_policy = self._policy(d1_config)

        def load_policy(checkpoint, *, native_profile, config):
            if config.graph.num_demos == 1:
                return d1_policy
            raise RuntimeError("injected D2 load failure")

        with self._patched_factory_collaborators(
            profile=profile,
            loader_side_effect=load_policy,
        ) as (_, loader), self._patched_sessions_constructor(
            side_effect=ReferenceSessions,
        ) as constructor:
            with self.assertRaisesRegex(RuntimeError, "injected D2 load failure"):
                build_reference_sessions(
                    "/checkpoint/model.pt",
                    reference_id=self._REFERENCE_ID,
                    native_profile=self._PROFILE_ID,
                    device="cpu",
                )
        self.assertEqual(loader.call_count, 2)
        constructor.assert_not_called()
        self.assertEqual(d1_policy.graph_config, d1_config.graph)
        self.assertEqual(d1_policy.runtime, d1_config.runtime)

    def test_builder_signature_and_dependency_boundary_remain_outer_only(self):
        from icgs.composition import build_reference_sessions

        parameters = inspect.signature(build_reference_sessions).parameters
        self.assertNotIn("d1_session_id", parameters)
        self.assertNotIn("d2_session_id", parameters)
        self._assert_factory_dependency_boundary(build_reference_sessions)

        def violating_surface():
            return PreparedContext

        with self.assertRaisesRegex(
            AssertionError,
            "violating_surface.*PreparedContext",
        ):
            self._assert_factory_dependency_boundary(violating_surface)


class MethodContextBuilderTests(unittest.TestCase):
    _PUBLISHED_PROFILE = "instant_policy_published_vv19_119fa871"

    @classmethod
    def _demo(cls, content_hash, *, grips=(0, 0, 1, 1, 1)):
        source = NativeDemoMaterializationTests._demo(grips=grips)
        return TimedDemoInput(source.transitions, content_hash)

    @classmethod
    def _events(cls, demos, *, mixed=False, gap=False, batch=1):
        from icgs.contracts.method import SegmentRef
        from icgs.models.encoders.event import EventEncoding
        from icgs.state.method_context import build_event_memory

        refs = []
        for demo in demos:
            content_hash = demo.demo_content_hash
            refs.append(SegmentRef(content_hash, 0, 0, "start", False))
            if mixed:
                refs.extend(
                    (
                        SegmentRef(content_hash, 0, 1, "interaction", True),
                        SegmentRef(content_hash, 1, 3, "interaction", True),
                        SegmentRef(content_hash, 3, 4, "interaction", False),
                    )
                )
            else:
                second_start = 3 if gap else 2
                refs.extend(
                    (
                        SegmentRef(content_hash, 0, 2, "interaction", True),
                        SegmentRef(
                            content_hash,
                            second_start,
                            4,
                            "interaction",
                            True,
                        ),
                    )
                )
            refs.append(SegmentRef(content_hash, 4, 4, "end", False))
        row_refs = tuple((*refs, None))
        row_valid = tuple((*([True] * len(refs)), False))
        valid = torch.tensor([row_valid] * batch, dtype=torch.bool)
        encoding = EventEncoding(
            torch.zeros(batch, len(row_refs), 256),
            valid,
        )
        hashes = tuple(demo.demo_content_hash for demo in demos)
        return build_event_memory(
            encoding,
            tuple(row_refs for _ in range(batch)),
            tuple(hashes for _ in range(batch)),
            "encoder-fixture",
            "segmentation-fixture",
        )

    @classmethod
    def _sessions(cls):
        from icgs.policies.reference import ReferenceSessions

        values = ReferenceSessionsValidationTests._values()
        d1_config = dataclasses.replace(
            values["d1_config"],
            graph=dataclasses.replace(
                values["d1_config"].graph,
                traj_horizon=2,
            ),
        )
        d2_config = dataclasses.replace(
            d1_config,
            graph=dataclasses.replace(d1_config.graph, num_demos=2),
        )
        values.update(
            d1=ReferenceSessionsValidationTests._policy(d1_config),
            d2=ReferenceSessionsValidationTests._policy(d2_config),
            d1_config=d1_config,
            d2_config=d2_config,
            native_point_count=3,
        )
        return ReferenceSessions(**values)

    @classmethod
    def _config(cls):
        from icgs.configuration.method import MethodConfig

        return dataclasses.replace(
            MethodConfig(),
            native_profile=cls._PUBLISHED_PROFILE,
        )

    @staticmethod
    def _prepared(owner, source_id):
        from icgs.state.context_cache import PreparedContext

        return PreparedContext([], owner, source_id=source_id)

    @staticmethod
    def _native_demo(name):
        return {"native-demo": name}

    def _assert_builder_dependency_boundary(self, surface):
        source = textwrap.dedent(inspect.getsource(surface))
        tree = ast.parse(source)
        referenced = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        referenced.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        forbidden = {
            "segment_demo",
            "predict",
            "predict_batch",
            "draw_route",
            "router_probabilities",
            "Candidate",
            "ReferenceProposal",
            "resolve_native_profile",
            "load_published_policy",
            "build_reference_sessions",
            "sha256_file",
        }
        artifact_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                artifact_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name == "icgs.artifacts"
                    or alias.name.startswith("icgs.artifacts.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (
                    node.module == "icgs.artifacts"
                    or node.module.startswith("icgs.artifacts.")
                )
            ):
                artifact_imports.append(node.module)
        artifact_imports.sort()
        found = sorted(forbidden & referenced)
        if found or artifact_imports or "torch.load" in source:
            self.fail(
                f"Task 3B.3 surface {surface.__name__} must assemble existing "
                "contexts without re-segmentation, routing, inference, artifact "
                f"IO or session construction; forbidden identifiers: {found}; "
                f"forbidden artifact imports: {artifact_imports}"
            )

    def test_builder_rejects_preflight_mismatch_before_planning_or_materialization(self):
        from icgs.configuration.method import MethodConfig
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        demo_a = self._demo("demo-a")
        demo_b = self._demo("demo-b")
        events_a = self._events((demo_a,))
        events_ab = self._events((demo_a, demo_b))
        events_ba = self._events((demo_b, demo_a))
        events_batch_two = self._events((demo_a,), batch=2)
        config = self._config()
        cases = (
            ([demo_a], events_a, sessions, config, sessions.reference_id, "tuple"),
            ((), events_a, sessions, config, sessions.reference_id, "one or two|D"),
            (
                (demo_a, demo_b, demo_a),
                events_a,
                sessions,
                config,
                sessions.reference_id,
                "one or two|D",
            ),
            (
                (object(),),
                events_a,
                sessions,
                config,
                sessions.reference_id,
                "TimedDemoInput",
            ),
            ((demo_a,), object(), sessions, config, sessions.reference_id, "EventMemory"),
            ((demo_a,), events_batch_two, sessions, config, sessions.reference_id, "B=1|batch"),
            ((demo_a,), events_a, object(), config, sessions.reference_id, "ReferenceSessions"),
            ((demo_a,), events_a, sessions, object(), sessions.reference_id, "MethodConfig"),
            ((demo_a,), events_a, sessions, config, "other-reference", "reference_id"),
            (
                (demo_a,),
                events_a,
                sessions,
                MethodConfig(),
                sessions.reference_id,
                "native_profile|profile",
            ),
            (
                (demo_a, demo_b),
                events_ba,
                sessions,
                config,
                sessions.reference_id,
                "hash|order",
            ),
        )
        collaborators = (
            "split_context_seeds",
            "debounced_grip_boundaries",
            "select_window_indices",
            "materialize_full_native_demo",
            "materialize_indexed_native_demo",
        )
        with ExitStack() as stack:
            spies = tuple(
                stack.enter_context(
                    mock.patch.object(
                        reference,
                        name,
                        side_effect=AssertionError(
                            f"{name} ran before builder preflight validation"
                        ),
                    )
                )
                for name in collaborators
            )
            for raw_demos, events, owner, resolved, reference_id, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        build_method_context(
                            raw_demos,
                            events,
                            owner,
                            context_seed=17,
                            reference_id=reference_id,
                            config=resolved,
                        )
        self.assertTrue(all(spy.call_count == 0 for spy in spies))
        self.assertEqual(events_ab.raw_hashes[0], ("demo-a", "demo-b"))

        length = events_a.tokens.shape[1]
        seed_cardinality_cases = (
            (((), tuple(range(length))), "full_demo_seeds|demo"),
            (((101,), tuple(range(length - 1))), "window_slot_seeds|event|L"),
        )
        for seed_plan, message in seed_cardinality_cases:
            with self.subTest(seed_cardinality=message), mock.patch.object(
                reference,
                "split_context_seeds",
                return_value=seed_plan,
            ), mock.patch.object(
                reference,
                "debounced_grip_boundaries",
                side_effect=AssertionError("debounce ran before seed cardinality validation"),
            ), mock.patch.object(
                reference,
                "select_window_indices",
                side_effect=AssertionError("selector ran before seed cardinality validation"),
            ), mock.patch.object(
                reference,
                "materialize_full_native_demo",
                side_effect=AssertionError(
                    "full materialization ran before seed cardinality validation"
                ),
            ) as materialize, mock.patch.object(
                sessions.d1,
                "prepare_context",
                side_effect=AssertionError(
                    "prepare ran before seed cardinality validation"
                ),
            ) as prepare:
                with self.assertRaisesRegex(ValueError, message):
                    build_method_context(
                        (demo_a,),
                        events_a,
                        sessions,
                        context_seed=17,
                        reference_id=sessions.reference_id,
                        config=config,
                    )
                materialize.assert_not_called()
                prepare.assert_not_called()

    def test_malformed_event_partition_fails_before_first_materialization(self):
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        demo = self._demo("demo-a")
        base = self._events((demo,))
        start, first, second, end, padding = base.refs[0]
        cases = (
            ("internal gap", first, dataclasses.replace(second, a=3), "contiguous"),
            ("leading gap", dataclasses.replace(first, a=1), second, "complete measured demo"),
            ("trailing gap", first, dataclasses.replace(second, b=3), "complete measured demo"),
            ("empty partition", None, None, "nonempty"),
        )
        for name, first_ref, second_ref, message in cases:
            refs = (start, first_ref, second_ref, end, padding)
            events = dataclasses.replace(
                base,
                refs=(refs,),
                valid=torch.tensor([[ref is not None for ref in refs]], dtype=torch.bool),
            )
            with self.subTest(partition=name), mock.patch.object(
                reference,
                "materialize_full_native_demo",
                side_effect=AssertionError("full materialization ran before planning"),
            ) as full_materializer, mock.patch.object(
                reference,
                "materialize_indexed_native_demo",
                side_effect=AssertionError("window materialization ran before planning"),
            ) as window_materializer, mock.patch.object(
                sessions.d1,
                "prepare_context",
                side_effect=AssertionError("D1 prepare ran before planning"),
            ) as d1_prepare, mock.patch.object(
                sessions.d2,
                "prepare_context",
                side_effect=AssertionError("D2 prepare ran before planning"),
            ) as d2_prepare:
                with self.assertRaisesRegex(ValueError, message):
                    build_method_context(
                        (demo,),
                        events,
                        sessions,
                        context_seed=17,
                        reference_id=sessions.reference_id,
                        config=self._config(),
                    )
                for collaborator in (
                    full_materializer,
                    window_materializer,
                    d1_prepare,
                    d2_prepare,
                ):
                    collaborator.assert_not_called()

    def test_one_demo_full_context_uses_d1_and_exact_provenance(self):
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        demo = self._demo("demo-a")
        raw_demos = (demo,)
        events = self._events(raw_demos)
        config = self._config()
        full_native = self._native_demo("full-a")
        full = self._prepared(sessions.d1.context_owner, "full-context")
        full_seed = 101
        window_seeds = tuple(range(201, 201 + events.tokens.shape[1]))

        with mock.patch.object(
            reference,
            "split_context_seeds",
            return_value=((full_seed,), window_seeds),
        ) as split, mock.patch.object(
            reference,
            "select_window_indices",
            return_value=None,
        ), mock.patch.object(
            reference,
            "materialize_full_native_demo",
            return_value=full_native,
        ) as materialize, mock.patch.object(
            sessions.d1,
            "prepare_context",
            return_value=full,
        ) as prepare, mock.patch.object(
            sessions.d1,
            "validate_context",
        ) as validate, mock.patch.object(
            sessions.d2,
            "prepare_context",
            side_effect=AssertionError("D2 must not prepare a one-demo full context"),
        ):
            context, record = build_method_context(
                raw_demos,
                events,
                sessions,
                context_seed=17,
                reference_id=sessions.reference_id,
                config=config,
            )

        split.assert_called_once_with(
            17,
            demo_count=1,
            event_count=events.tokens.shape[1],
        )
        materialize.assert_called_once_with(
            demo,
            point_seed=full_seed,
            native_waypoint_count=sessions.d1.graph_config.traj_horizon,
            native_point_count=sessions.native_point_count,
        )
        prepared_demos = prepare.call_args.args[0]
        self.assertEqual(tuple(prepared_demos), (full_native,))
        self.assertEqual(prepare.call_args.kwargs, {"prepared": True})
        validate.assert_called_once_with(full)
        self.assertIs(context.raw_demos, raw_demos)
        self.assertIs(context.events, events)
        self.assertIs(context.native_full, full)
        self.assertTrue(all(window is None for window in context.native_windows))
        self.assertEqual(record.context_seed, 17)
        self.assertEqual(record.full_demo_seeds, (full_seed,))
        self.assertEqual(record.window_slot_seeds, window_seeds)
        self.assertEqual(record.full_context_id, full.source_id)
        self.assertEqual(record.window_context_ids, (None,) * len(window_seeds))

    def test_two_demo_full_context_uses_d2_with_d1_canonical_waypoint_count(self):
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        raw_demos = (
            self._demo("demo-a"),
            self._demo("demo-b", grips=(1, 0, 0, 0, 0)),
        )
        events = self._events(raw_demos)
        full_seeds = (101, 102)
        window_seeds = tuple(range(301, 301 + events.tokens.shape[1]))
        native_demos = (
            self._native_demo("full-a"),
            self._native_demo("full-b"),
        )
        full = self._prepared(sessions.d2.context_owner, "full-d2")
        order = []
        native_demo_iterator = iter(native_demos)
        real_debounce = reference.debounced_grip_boundaries

        def debounce(grips, resolved):
            order.append("debounce")
            self.assertEqual(resolved, self._config().event)
            return real_debounce(grips, resolved)

        def select(target, interactions, *args, **kwargs):
            order.append(f"select:{target.demo_content_hash}:{target.a}")
            expected = tuple(
                reference_ref
                for reference_ref in events.refs[0]
                if reference_ref is not None
                and reference_ref.kind == "interaction"
                and reference_ref.demo_content_hash == target.demo_content_hash
            )
            self.assertEqual(interactions, expected)
            expected_grip_indices = (3,) if target.demo_content_hash == "demo-a" else (2,)
            self.assertEqual(args[0], expected_grip_indices)
            return None

        def materialize_full(*args, **kwargs):
            order.append("materialize-full")
            return next(native_demo_iterator)

        with mock.patch.object(
            reference,
            "split_context_seeds",
            return_value=(full_seeds, window_seeds),
        ), mock.patch.object(
            reference,
            "debounced_grip_boundaries",
            side_effect=debounce,
        ) as debounce_spy, mock.patch.object(
            reference,
            "select_window_indices",
            side_effect=select,
        ) as select_spy, mock.patch.object(
            reference,
            "materialize_full_native_demo",
            side_effect=materialize_full,
        ) as materialize, mock.patch.object(
            sessions.d2,
            "prepare_context",
            return_value=full,
        ) as prepare, mock.patch.object(
            sessions.d2,
            "validate_context",
        ) as validate, mock.patch.object(
            sessions.d1,
            "prepare_context",
            side_effect=AssertionError("D1 must not prepare a D2 full context"),
        ):
            context, record = build_method_context(
                raw_demos,
                events,
                sessions,
                context_seed=19,
                reference_id=sessions.reference_id,
                config=self._config(),
            )

        first_materialization = order.index("materialize-full")
        self.assertEqual(first_materialization, 6)
        self.assertEqual(order[:2], ["debounce", "debounce"])
        self.assertEqual(
            [tuple(call.args[0]) for call in debounce_spy.call_args_list],
            [(0, 0, 1, 1, 1), (1, 0, 0, 0, 0)],
        )
        self.assertTrue(
            all(item.startswith("select:") for item in order[2:first_materialization])
        )
        self.assertEqual(select_spy.call_count, 4)
        self.assertEqual(
            materialize.call_args_list,
            [
                mock.call(
                    demo,
                    point_seed=seed,
                    native_waypoint_count=sessions.d1.graph_config.traj_horizon,
                    native_point_count=sessions.native_point_count,
                )
                for demo, seed in zip(raw_demos, full_seeds)
            ],
        )
        self.assertEqual(tuple(prepare.call_args.args[0]), native_demos)
        self.assertEqual(prepare.call_args.kwargs, {"prepared": True})
        validate.assert_called_once_with(full)
        self.assertIs(context.native_full, full)
        self.assertIs(context.raw_demos, raw_demos)
        self.assertEqual(record.full_demo_seeds, full_seeds)
        self.assertEqual(record.full_context_id, full.source_id)

    def test_mixed_windows_are_preplanned_slot_aligned_and_owned_by_d1(self):
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        demo = self._demo("demo-a")
        raw_demos = (demo,)
        events = self._events(raw_demos, mixed=True)
        config = self._config()
        length = events.tokens.shape[1]
        full_seed = 101
        window_seeds = tuple(range(401, 401 + length))
        full_native = self._native_demo("full")
        window_native = self._native_demo("window")
        full = self._prepared(sessions.d1.context_owner, "full-context")
        window = self._prepared(sessions.d1.context_owner, "window-context")
        order = []

        def debounce(grips, resolved):
            order.append("debounce")
            self.assertEqual(tuple(grips), (0.0, 0.0, 1.0, 1.0, 1.0))
            self.assertIn(resolved, (config, config.event))
            return (2,)

        def select(target, interactions, grip_indices, resolved, **kwargs):
            order.append(f"select:{target.a}-{target.b}")
            self.assertEqual(
                interactions,
                tuple(
                    reference_ref
                    for reference_ref in events.refs[0]
                    if reference_ref is not None
                    and reference_ref.kind == "interaction"
                ),
            )
            self.assertEqual(grip_indices, (2,))
            self.assertIn(resolved, (config, config.router))
            self.assertEqual(
                kwargs,
                {
                    "native_waypoint_count": sessions.d1.graph_config.traj_horizon
                },
            )
            return (0, 4) if target.a == 0 else None

        def materialize_full(*args, **kwargs):
            order.append("materialize-full")
            return full_native

        def materialize_window(*args, **kwargs):
            order.append("materialize-window")
            return window_native

        prepared_contexts = iter((full, window))

        def prepare_context(demos, *, prepared):
            context = next(prepared_contexts)
            order.append(f"prepare:{context.source_id}")
            return context

        def validate_context(context):
            order.append(f"validate:{context.source_id}")

        with mock.patch.object(
            reference,
            "split_context_seeds",
            return_value=((full_seed,), window_seeds),
        ), mock.patch.object(
            reference,
            "debounced_grip_boundaries",
            side_effect=debounce,
        ) as debounce_spy, mock.patch.object(
            reference,
            "select_window_indices",
            side_effect=select,
        ) as select_spy, mock.patch.object(
            reference,
            "materialize_full_native_demo",
            side_effect=materialize_full,
        ), mock.patch.object(
            reference,
            "materialize_indexed_native_demo",
            side_effect=materialize_window,
        ) as window_materializer, mock.patch.object(
            sessions.d1,
            "prepare_context",
            side_effect=prepare_context,
        ) as prepare, mock.patch.object(
            sessions.d1,
            "validate_context",
            side_effect=validate_context,
        ) as validate:
            points_before = tuple(
                observation.points.copy()
                for observation in NativeDemoMaterializationTests._measured_observations(
                    demo
                )
            )
            context, record = build_method_context(
                raw_demos,
                events,
                sessions,
                context_seed=23,
                reference_id=sessions.reference_id,
                config=config,
            )

        self.assertEqual(order[:3], ["debounce", "select:0-1", "select:1-3"])
        self.assertEqual(order[3], "materialize-full")
        self.assertEqual(
            order[4:],
            [
                "prepare:full-context", "validate:full-context",
                "materialize-window",
                "prepare:window-context", "validate:window-context",
            ],
        )
        debounce_spy.assert_called_once()
        self.assertEqual(select_spy.call_count, 2)
        window_materializer.assert_called_once_with(
            demo,
            (0, 4),
            point_seed=window_seeds[1],
            native_waypoint_count=sessions.d1.graph_config.traj_horizon,
            native_point_count=sessions.native_point_count,
        )
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(tuple(prepare.call_args_list[0].args[0]), (full_native,))
        self.assertEqual(tuple(prepare.call_args_list[1].args[0]), (window_native,))
        self.assertEqual(
            [item.kwargs for item in prepare.call_args_list],
            [{"prepared": True}, {"prepared": True}],
        )
        self.assertEqual(validate.call_args_list, [mock.call(full), mock.call(window)])
        self.assertIs(context.raw_demos, raw_demos)
        self.assertIs(context.events, events)
        self.assertIs(context.native_full, full)
        expected_windows = (None, window, None, None, None, None)
        self.assertEqual(context.native_windows, expected_windows)
        self.assertEqual(
            record.window_context_ids,
            (None, window.source_id, None, None, None, None),
        )
        self.assertEqual(record.window_slot_seeds, window_seeds)
        torch.testing.assert_close(
            context.native_window_valid,
            torch.tensor([[False, True, False, False, False, False]]),
        )
        self.assertIs(context.native_windows[1], window)
        for observation, before in zip(
            NativeDemoMaterializationTests._measured_observations(demo),
            points_before,
        ):
            np.testing.assert_array_equal(observation.points, before)

    def test_collaborator_failures_and_wrong_owners_propagate(self):
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        demo = self._demo("demo-a")
        raw_demos = (demo,)
        events = self._events(raw_demos)
        config = self._config()
        full_native = self._native_demo("full")
        length = events.tokens.shape[1]
        seeds = ((101,), tuple(range(501, 501 + length)))

        with mock.patch.object(
            reference,
            "split_context_seeds",
            return_value=seeds,
        ), mock.patch.object(
            reference,
            "select_window_indices",
            return_value=None,
        ), mock.patch.object(
            reference,
            "materialize_full_native_demo",
            side_effect=RuntimeError("injected full materializer failure"),
        ), mock.patch.object(
            sessions.d1,
            "prepare_context",
        ) as prepare:
            with self.assertRaisesRegex(RuntimeError, "full materializer"):
                build_method_context(
                    raw_demos,
                    events,
                    sessions,
                    context_seed=29,
                    reference_id=sessions.reference_id,
                    config=config,
                )
            prepare.assert_not_called()

        with mock.patch.object(
            reference,
            "split_context_seeds",
            return_value=seeds,
        ), mock.patch.object(
            reference,
            "select_window_indices",
            return_value=None,
        ), mock.patch.object(
            reference,
            "materialize_full_native_demo",
            return_value=full_native,
        ), mock.patch.object(
            sessions.d1,
            "prepare_context",
            side_effect=RuntimeError("injected full prepare failure"),
        ), mock.patch.object(sessions.d1, "validate_context") as validate:
            with self.assertRaisesRegex(RuntimeError, "full prepare"):
                build_method_context(
                    raw_demos,
                    events,
                    sessions,
                    context_seed=30,
                    reference_id=sessions.reference_id,
                    config=config,
                )
            validate.assert_not_called()

        wrong_owner = self._prepared(object(), "wrong-owner")
        with mock.patch.object(
            reference,
            "split_context_seeds",
            return_value=seeds,
        ), mock.patch.object(
            reference,
            "select_window_indices",
            return_value=None,
        ), mock.patch.object(
            reference,
            "materialize_full_native_demo",
            return_value=full_native,
        ), mock.patch.object(
            sessions.d1,
            "prepare_context",
            return_value=wrong_owner,
        ), mock.patch.object(
            sessions.d1,
            "validate_context",
            side_effect=ValueError("context belongs to another policy"),
        ) as validate:
            with self.assertRaisesRegex(ValueError, "belongs to another policy"):
                build_method_context(
                    raw_demos,
                    events,
                    sessions,
                    context_seed=31,
                    reference_id=sessions.reference_id,
                    config=config,
                )
            validate.assert_called_once_with(wrong_owner)

    def test_window_materialize_prepare_and_validate_failures_propagate(self):
        from icgs.policies import reference
        from icgs.policies.reference import build_method_context

        sessions = self._sessions()
        demo = self._demo("demo-a")
        raw_demos = (demo,)
        events = self._events(raw_demos)
        config = self._config()
        length = events.tokens.shape[1]
        seeds = ((101,), tuple(range(601, 601 + length)))
        full_native = self._native_demo("full")
        window_native = self._native_demo("window")

        for stage in ("materialize", "prepare", "validate"):
            with self.subTest(stage=stage):
                full = self._prepared(
                    sessions.d1.context_owner,
                    f"full-{stage}",
                )
                window_owner = (
                    object()
                    if stage == "validate"
                    else sessions.d1.context_owner
                )
                window = self._prepared(window_owner, f"window-{stage}")
                prepare_count = 0

                def materialize_window(*args, **kwargs):
                    if stage == "materialize":
                        raise RuntimeError("injected window materialize failure")
                    return window_native

                def prepare_context(demos, *, prepared):
                    nonlocal prepare_count
                    prepare_count += 1
                    if prepare_count == 1:
                        return full
                    if stage == "prepare":
                        raise RuntimeError("injected window prepare failure")
                    return window

                def validate_context(context):
                    if context is window and stage == "validate":
                        raise ValueError("context belongs to another policy")

                expected = {
                    "materialize": "window materialize",
                    "prepare": "window prepare",
                    "validate": "belongs to another policy",
                }[stage]
                with mock.patch.object(
                    reference,
                    "split_context_seeds",
                    return_value=seeds,
                ), mock.patch.object(
                    reference,
                    "select_window_indices",
                    side_effect=((0, 4), None),
                ), mock.patch.object(
                    reference,
                    "materialize_full_native_demo",
                    return_value=full_native,
                ), mock.patch.object(
                    reference,
                    "materialize_indexed_native_demo",
                    side_effect=materialize_window,
                ), mock.patch.object(
                    sessions.d1,
                    "prepare_context",
                    side_effect=prepare_context,
                ), mock.patch.object(
                    sessions.d1,
                    "validate_context",
                    side_effect=validate_context,
                ):
                    with self.assertRaisesRegex(
                        (RuntimeError, ValueError),
                        expected,
                    ):
                        build_method_context(
                            raw_demos,
                            events,
                            sessions,
                            context_seed=37,
                            reference_id=sessions.reference_id,
                            config=config,
                        )

    def test_builder_function_scoped_dependency_guard(self):
        from icgs.policies.reference import build_method_context

        self._assert_builder_dependency_boundary(build_method_context)

        def violating_surface():
            import icgs.artifacts.checkpoints as checkpoints
            from icgs.artifacts.published import resolve_native_profile

            return segment_demo, torch.load, resolve_native_profile, checkpoints

        with self.assertRaisesRegex(
            AssertionError,
            "violating_surface.*resolve_native_profile.*segment_demo",
        ):
            self._assert_builder_dependency_boundary(violating_surface)

        # Isolate each import form so another forbidden identifier cannot hide
        # a broken artifact-import check.
        def plain_artifact_import():
            import icgs.artifacts.checkpoints as checkpoints

            return checkpoints

        def from_artifact_import():
            from icgs.artifacts import checkpoints

            return checkpoints

        for surface in (plain_artifact_import, from_artifact_import):
            with self.subTest(surface=surface.__name__), self.assertRaisesRegex(
                AssertionError, "forbidden artifact imports:.*icgs.artifacts"
            ):
                self._assert_builder_dependency_boundary(surface)


class ReferenceProposalTests(unittest.TestCase):
    @staticmethod
    def _values():
        from icgs.algorithms.planning.candidates import Candidate
        from icgs.contracts.records import ActionTrajectory

        candidate = Candidate(
            index=0,
            trajectory=ActionTrajectory(
                np.tile(np.eye(4), (1, 2, 1, 1)), np.ones((1, 2, 1))
            ),
            root_pose=np.eye(4),
            context_id="context-fixture",
            source_id="artifact-fixture",
            seed=29,
            seconds=0.01,
            batch_size=1,
            horizon=2,
        )
        return dict(
            candidate=candidate,
            reference_id="reference-fixture",
            route_index=0,
            event_index=None,
            route_seed=17,
            diffusion_seed=29,
            native_context_id=candidate.context_id,
            native_session_id="session-fixture",
        )

    def test_reference_proposal_accepts_full_and_window_routes(self):
        from icgs.policies.reference import ReferenceProposal

        for route, event in ((0, None), (3, 2), (np.int64(3), np.int64(2))):
            with self.subTest(route=route, event=event):
                values = self._values()
                values.update(route_index=route, event_index=event)
                if isinstance(route, np.integer):
                    values.update(route_seed=np.int64(17), diffusion_seed=np.int64(29))
                    values["candidate"] = dataclasses.replace(
                        values["candidate"], seed=values["diffusion_seed"]
                    )
                proposal = ReferenceProposal(**values)
                self.assertIs(proposal.candidate, values["candidate"])
                for name, value in values.items():
                    if name != "candidate":
                        self.assertEqual(getattr(proposal, name), value, name)
                        self.assertIs(type(getattr(proposal, name)), type(value), name)

    def test_reference_proposal_rejects_invalid_route_event_mapping(self):
        from icgs.policies.reference import ReferenceProposal

        cases = [(0, 0), (1, None), (2, 0), (-1, None), (1, -1)]
        for invalid in (True, False, np.bool_(True), np.bool_(False), 1.0, "1", None):
            cases.append((invalid, None))
            if invalid is not None:
                cases.append((1, invalid))
        for route, event in cases:
            with self.subTest(route=route, event=event):
                values = self._values()
                values.update(route_index=route, event_index=event)
                with self.assertRaisesRegex((TypeError, ValueError), "route|event"):
                    ReferenceProposal(**values)

    def test_reference_proposal_rejects_invalid_seeds(self):
        from icgs.policies.reference import ReferenceProposal

        class ComparisonForbidden:
            def __eq__(self, other):
                raise AssertionError("malformed candidate.seed compared before validation")

        invalid_seeds = (-1, True, False, np.bool_(True), 29.0, "29", None)
        for field in ("route_seed", "diffusion_seed", "candidate.seed"):
            for invalid in (*invalid_seeds, ComparisonForbidden()):
                with self.subTest(field=field, invalid=repr(invalid)):
                    values = self._values()
                    if field == "candidate.seed":
                        values["candidate"] = dataclasses.replace(values["candidate"], seed=invalid)
                        # Equality alone would accept bools/floats matching an int.
                        if isinstance(invalid, (bool, np.bool_, float)):
                            values["diffusion_seed"] = int(invalid)
                    else:
                        values[field] = invalid
                    with self.assertRaisesRegex((TypeError, ValueError), "seed"):
                        ReferenceProposal(**values)

    def test_reference_proposal_validates_ids_without_normalizing(self):
        from icgs.policies.reference import ReferenceProposal

        for field in ("reference_id", "native_context_id", "native_session_id"):
            for invalid in ("", "   ", None, 7, b"id"):
                with self.subTest(field=field, invalid=invalid):
                    values = self._values()
                    values[field] = invalid
                    if field == "native_context_id":
                        values["candidate"] = dataclasses.replace(
                            values["candidate"], context_id=invalid
                        )
                    with self.assertRaises((TypeError, ValueError)):
                        ReferenceProposal(**values)
        values = self._values()
        values.update(
            reference_id="  ref-id  ", native_context_id="  context-id  ",
            native_session_id="  session-id  ",
        )
        values["candidate"] = dataclasses.replace(
            values["candidate"], context_id=values["native_context_id"]
        )
        proposal = ReferenceProposal(**values)
        for field in ("reference_id", "native_context_id", "native_session_id"):
            self.assertEqual(getattr(proposal, field), values[field])

    def test_reference_proposal_rejects_candidate_provenance_mismatch(self):
        from icgs.policies.reference import ReferenceProposal

        for changed, message in (({"context_id": "other-context"}, "context"), ({"seed": 30}, "seed")):
            with self.subTest(changed=changed):
                values = self._values()
                values["candidate"] = dataclasses.replace(values["candidate"], **changed)
                with self.assertRaisesRegex(ValueError, message):
                    ReferenceProposal(**values)
        for invalid in (None, object(), SimpleNamespace(**vars(self._values()["candidate"]))):
            with self.subTest(candidate=type(invalid).__name__):
                values = self._values()
                values["candidate"] = invalid
                with self.assertRaisesRegex((TypeError, ValueError), "Candidate|candidate"):
                    ReferenceProposal(**values)

    def test_reference_proposal_is_frozen_but_retains_candidate_identity(self):
        from icgs.policies.reference import ReferenceProposal

        self.assertEqual(
            tuple(ReferenceProposal.__dataclass_fields__),
            (
                "candidate",
                "reference_id",
                "route_index",
                "event_index",
                "route_seed",
                "diffusion_seed",
                "native_context_id",
                "native_session_id",
            ),
        )
        values = self._values()
        proposal = ReferenceProposal(**values)
        self.assertIs(proposal.candidate, values["candidate"])
        for field in dataclasses.fields(proposal):
            with self.subTest(field=field.name), self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(proposal, field.name, getattr(proposal, field.name))


if __name__ == "__main__":
    unittest.main()
