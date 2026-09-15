import ast
import dataclasses
import inspect
import random
import textwrap
import unittest
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


if __name__ == "__main__":
    unittest.main()
