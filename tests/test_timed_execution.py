import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


class _LifecycleController:
    """Deterministic low-level collaborator for lifecycle boundary tests."""

    def __init__(
        self,
        *,
        reset_failures=0,
        step_failures=0,
        observe_failure_on_call=None,
        status_failure=False,
        safe_hold_failures=0,
        close_failures=0,
    ):
        self.calls = []
        self.reset_failures = reset_failures
        self.step_failures = step_failures
        self.observe_failure_on_call = observe_failure_on_call
        self.status_failure = status_failure
        self.safe_hold_failures = safe_hold_failures
        self.close_failures = close_failures
        self.observe_calls = 0
        self.sim_time = 0.0
        self.pose = np.eye(4)
        self.grip = 0
        self.target = np.eye(4)
        self.target_grip = 0

    def reset(self, seed=None):
        self.calls.append("reset")
        if self.reset_failures:
            self.reset_failures -= 1
            raise RuntimeError("reset failure")
        self.sim_time = 0.0
        self.pose = np.eye(4)
        self.grip = 0

    def set_target(self, target_w, grip):
        self.calls.append("set_target")
        self.target = np.array(target_w, copy=True)
        self.target_grip = grip

    def step_physics(self):
        self.calls.append("step_physics")
        if self.step_failures:
            self.step_failures -= 1
            raise RuntimeError("step failure")
        self.sim_time += 0.005
        self.pose = self.target.copy()
        self.grip = self.target_grip

    def observe(self):
        self.calls.append("observe")
        self.observe_calls += 1
        if self.observe_calls == self.observe_failure_on_call:
            raise RuntimeError("observe failure")
        from icgs.contracts.records import Observation

        return Observation(np.zeros((1, 3)), self.pose, self.grip)

    def simulator_time(self):
        return self.sim_time

    def status(self):
        self.calls.append("status")
        if self.status_failure:
            raise RuntimeError("status failure")
        return "holding"

    def safe_hold(self):
        self.calls.append("safe_hold")
        if self.safe_hold_failures:
            self.safe_hold_failures -= 1
            raise RuntimeError("safe_hold failure")

    def close(self):
        self.calls.append("close")
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("close failure")


class TimedExecutionTests(unittest.TestCase):
    def test_materialize_grips_holds_first_grip_per_group(self):
        from icgs.execution.timed import materialize_grips

        self.assertEqual(materialize_grips([-1, 1, 0, 1], 2), (0, 0, 0, 0))
        self.assertEqual(materialize_grips([1, -1, -1, 1], 2), (1, 1, 0, 0))
        with self.assertRaises(ValueError):
            materialize_grips([1], 0)

    def test_materialize_prefix_anchors_each_target_at_one_root(self):
        from icgs.execution.timed import materialize_prefix

        root = np.eye(4)
        root[:3, :3] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        root[:3, 3] = (1.0, 2.0, 3.0)
        actions = np.tile(np.eye(4), (1, 4, 1, 1))
        actions[0, 0, 0, 3] = 0.1
        actions[0, 1, :3, :3] = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        actions[0, 1, :3, 3] = (0.0, 0.2, 0.0)
        actions[0, 2, 1, 3] = 0.3
        actions[0, 3, 2, 3] = -0.1
        grips = np.array([[[1.0], [-1.0], [-1.0], [1.0]]])
        candidate = SimpleNamespace(
            trajectory=SimpleNamespace(transforms=actions, grips=grips),
            root_pose=root,
            raw_candidate_id="candidate-1",
        )

        prefix = materialize_prefix(candidate, h=4, r=2, duration_s=0.1)

        self.assertEqual(prefix.raw_candidate_id, "candidate-1")
        for command, action in zip(prefix.commands, actions[0]):
            np.testing.assert_allclose(command.target_w, root @ action)
            self.assertEqual(command.duration_s, 0.1)
        self.assertEqual([command.grip for command in prefix.commands], [1, 1, 0, 0])
        self.assertFalse(np.allclose(prefix.commands[1].target_w, actions[0, 1] @ root))
        self.assertFalse(np.allclose(prefix.commands[2].target_w,
                                     prefix.commands[1].target_w @ actions[0, 2]))

    def test_materialize_prefix_rejects_non_single_batch_and_bad_bounds(self):
        from icgs.execution.timed import materialize_prefix

        root = np.eye(4)
        actions = np.tile(np.eye(4), (1, 3, 1, 1))
        grips = np.zeros((1, 3, 1))
        candidate = SimpleNamespace(
            trajectory=SimpleNamespace(transforms=actions, grips=grips),
            root_pose=root,
            index=0,
        )

        for h, r in ((0, 1), (3, 0), (1, 2), (4, 1)):
            with self.subTest(h=h, r=r):
                with self.assertRaises(ValueError):
                    materialize_prefix(candidate, h=h, r=r, duration_s=0.1)
        with self.assertRaises(ValueError):
            materialize_prefix(candidate, h=1, r=1, duration_s=float("nan"))

        batched = SimpleNamespace(
            trajectory=SimpleNamespace(transforms=np.tile(actions, (2, 1, 1, 1)), grips=np.tile(grips, (2, 1, 1))),
            root_pose=root,
            index=0,
        )
        with self.assertRaisesRegex(ValueError, "batch"):
            materialize_prefix(batched, h=1, r=1, duration_s=0.1)

    def test_materialize_prefix_returns_owned_read_only_command_poses(self):
        from icgs.execution.timed import materialize_prefix

        root = np.eye(4)
        actions = np.tile(np.eye(4), (1, 1, 1, 1))
        grips = np.ones((1, 1, 1))
        candidate = SimpleNamespace(
            trajectory=SimpleNamespace(transforms=actions, grips=grips),
            root_pose=root,
            index=4,
        )

        prefix = materialize_prefix(candidate, h=1, r=1, duration_s=0.1)
        root[0, 3] = 9.0
        actions[0, 0, 1, 3] = 9.0

        self.assertEqual(prefix.raw_candidate_id, "4")
        self.assertEqual(prefix.proposal_root[0, 3], 0.0)
        self.assertEqual(prefix.commands[0].target_w[1, 3], 0.0)
        self.assertFalse(prefix.proposal_root.flags.writeable)
        self.assertFalse(prefix.commands[0].target_w.flags.writeable)

    def test_materialize_prefix_consumes_resolved_method_config_scalars(self):
        from icgs.configuration.method import MethodConfig
        from icgs.execution.timed import materialize_prefix

        root = np.eye(4)
        root[:3, :3] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        root[:3, 3] = (1.0, 2.0, 3.0)
        actions = np.tile(np.eye(4), (1, 4, 1, 1))
        actions[0, 0, 0, 3] = 0.1
        actions[0, 1, :3, :3] = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        actions[0, 1, :3, 3] = (0.0, 0.2, 0.0)
        actions[0, 2, 1, 3] = 0.3
        actions[0, 3, 2, 3] = -0.1
        grips = np.array([[[1.0], [-1.0], [-1.0], [1.0]]])
        candidate = SimpleNamespace(
            trajectory=SimpleNamespace(transforms=actions, grips=grips),
            root_pose=root,
            raw_candidate_id="configured-candidate",
        )

        # Composition resolves the immutable config once, outside any execution
        # loop, and passes only the owned timed scalars into the existing seam.
        def materialize_from_config(config):
            h = config.planning.h
            r = config.planning.r
            duration_s = config.control.dt0
            return materialize_prefix(candidate, h=h, r=r, duration_s=duration_s)

        cases = (
            (MethodConfig(), 2, 2, 0.1, (1, 1)),
            (
                MethodConfig.from_dict({
                    "planning": {"h": 4, "r": 1},
                    "control": {"dt0": 0.2},
                }),
                4,
                1,
                0.2,
                (1, 0, 0, 1),
            ),
            (
                MethodConfig.from_dict({
                    "planning": {"h": 4, "r": 2},
                    "control": {"dt0": 0.15},
                }),
                4,
                2,
                0.15,
                (1, 1, 0, 0),
            ),
        )
        for config, h, r, duration_s, expected_grips in cases:
            with self.subTest(h=h, r=r, duration_s=duration_s):
                prefix = materialize_from_config(config)
                self.assertEqual(len(prefix.commands), h)
                self.assertEqual([command.grip for command in prefix.commands], list(expected_grips))
                self.assertEqual(
                    [command.duration_s for command in prefix.commands],
                    [duration_s] * h,
                )
                for command, action in zip(prefix.commands, actions[0, :h]):
                    np.testing.assert_allclose(command.target_w, root @ action)
                self.assertFalse(
                    np.allclose(prefix.commands[1].target_w, actions[0, 1] @ root)
                )

    def test_timed_method_config_rejects_invalid_seam_values(self):
        from icgs.configuration.method import MethodConfig

        invalid_payloads = (
            {"planning": {"h": True, "r": 1}},
            {"planning": {"h": 4, "r": 0}},
            {"planning": {"h": 9, "r": 1}},
            {"planning": {"h": 4, "r": 1}, "control": {"dt0": True}},
            {"planning": {"h": 4, "r": 1}, "control": {"dt0": 0.0}},
            {"planning": {"h": 4, "r": 1, "unknown": 2}},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    MethodConfig.from_dict(payload)

    def test_interval_substeps_requires_integral_physics_steps(self):
        from icgs.environments.rlbench.timed import interval_substeps

        self.assertEqual(interval_substeps(0.1, 0.005), 20)
        with self.assertRaisesRegex(ValueError, "interval"):
            interval_substeps(0.1, 0.03)

    def test_timed_adapter_holds_target_and_tracks_separate_clocks(self):
        from icgs.contracts.method import TimedCommand
        from icgs.contracts.records import Observation
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        class DeterministicController:
            def __init__(self):
                self.calls = []
                self.sim_time = 0.0
                self.pose = np.eye(4)
                self.grip = 0
                self.target = None
                self.target_grip = None
                self.close_calls = 0

            def reset(self, seed=None):
                self.calls.append(("reset", seed))
                self.sim_time = 0.0
                self.pose = np.eye(4)
                self.grip = 0

            def set_target(self, target_w, grip):
                self.calls.append(("set_target", np.array(target_w, copy=True), grip))
                self.target = np.array(target_w, copy=True)
                self.target_grip = grip

            def step_physics(self):
                self.calls.append("step_physics")
                self.sim_time += 0.005
                self.pose = self.target.copy()
                self.grip = self.target_grip

            def observe(self):
                self.calls.append("observe")
                return Observation(np.zeros((1, 3)), self.pose, self.grip)

            def simulator_time(self):
                return self.sim_time

            def status(self):
                return "holding"

            def task_step(self, _command):
                raise AssertionError("task.step must not implement timed stepping")

            def safe_hold(self):
                self.calls.append("safe_hold")

            def close(self):
                self.close_calls += 1
                self.calls.append("close")

        controller = DeterministicController()
        wall_times = iter((100.0, 100.25))
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
            wall_clock=lambda: next(wall_times),
        )
        initial = environment.reset(seed=17)
        command_pose = np.eye(4)
        command_pose[0, 3] = 0.25
        transition = environment.advance(TimedCommand(command_pose, 1, 0.1))

        self.assertEqual(initial.boundary, 0)
        self.assertEqual(initial.simulator_timestamp, 0.0)
        self.assertEqual(initial.measured_wall_timestamp, 100.0)
        self.assertEqual(transition.after.boundary, 1)
        self.assertAlmostEqual(transition.after.simulator_timestamp, 0.1)
        self.assertEqual(transition.after.measured_wall_timestamp, 100.25)
        self.assertAlmostEqual(transition.achieved_duration_s, 0.1)
        self.assertEqual(transition.physics_substeps, 20)
        self.assertEqual(transition.controller_status, "holding")
        self.assertEqual(sum(call == "step_physics" for call in controller.calls), 20)
        target_index = next(i for i, call in enumerate(controller.calls)
                            if isinstance(call, tuple) and call[0] == "set_target")
        first_step_index = controller.calls.index("step_physics")
        self.assertLess(target_index, first_step_index)
        self.assertEqual(controller.calls[target_index][2], 1)
        np.testing.assert_allclose(controller.calls[target_index][1], command_pose)
        self.assertEqual(controller.close_calls, 0)

        environment.close()
        environment.close()
        self.assertEqual(controller.close_calls, 1)

    def test_timed_adapter_bounds_reset_attempts_and_closes(self):
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        class ResettingController:
            def __init__(self):
                self.reset_calls = 0
                self.close_calls = 0

            def reset(self, seed=None):
                self.reset_calls += 1
                raise RuntimeError("reset unavailable")

            def set_target(self, target_w, grip):
                pass

            def step_physics(self):
                pass

            def observe(self):
                raise AssertionError("observe is unreachable")

            def simulator_time(self):
                return 0.0

            def status(self):
                return "failed"

            def safe_hold(self):
                pass

            def close(self):
                self.close_calls += 1

        controller = ResettingController()
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
            max_reset_attempts=2,
        )
        with self.assertRaisesRegex(RuntimeError, "reset"):
            environment.reset(seed=3)
        self.assertEqual(controller.reset_calls, 2)
        self.assertEqual(controller.close_calls, 1)

    def test_timed_adapter_records_measured_transition_and_cleanup_chain(self):
        from icgs.contracts.method import TimedCommand
        from icgs.environments.rlbench.timed import TimedLifecycleError, TimedRLBenchAdapter
        from icgs.observability.config import from_mapping
        from icgs.observability.recorder import RunRecorder

        controller = _LifecycleController()
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder.start(
                from_mapping({
                    "output_dir": directory,
                    "flush_interval_s": 0.01,
                    "shutdown_timeout_s": 1.0,
                }),
                command="timed-observability",
            )
            environment = TimedRLBenchAdapter(
                controller,
                physics_dt=0.005,
                sensor_profile_id="fixture-sensor",
                recorder=recorder,
            )
            environment.reset(seed=5)
            environment.advance(TimedCommand(np.eye(4), 1, 0.1))
            controller.step_failures = 1
            controller.safe_hold_failures = 1
            controller.close_failures = 1
            with self.assertRaises(TimedLifecycleError) as raised:
                environment.advance(TimedCommand(np.eye(4), 0, 0.1))
            environment.close()
            recorder.close(status="failed", error=raised.exception)

            records = []
            for path in sorted((Path(recorder.path) / "events").glob("*.jsonl")):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            transitions = [item for item in records if item["event"] == "execution.transition"]
            self.assertEqual(len(transitions), 1)
            self.assertEqual(transitions[0]["fields"]["physics_substeps"], 20)
            self.assertAlmostEqual(transitions[0]["fields"]["achieved_duration_s"], 0.1)
            fields = transitions[0]["fields"]
            self.assertEqual(fields["before_boundary"], 0)
            self.assertEqual(fields["after_boundary"], 1)
            self.assertEqual(fields["commanded_grip"], 1)
            self.assertEqual(fields["before_grip"], 0.0)
            self.assertEqual(fields["after_grip"], 1.0)
            self.assertEqual(fields["sensor_profile_id"], "fixture-sensor")
            np.testing.assert_allclose(fields["commanded_target_w"], np.eye(4).tolist())
            np.testing.assert_allclose(fields["before_pose_w"], np.eye(4).tolist())
            np.testing.assert_allclose(fields["after_pose_w"], np.eye(4).tolist())
            cleanup = [item for item in records if item["event"] == "execution.lifecycle.error"]
            self.assertTrue(cleanup)
            self.assertEqual(cleanup[-1]["fields"]["operation_error"], "RuntimeError")
            self.assertEqual(cleanup[-1]["fields"]["safe_hold_error"], "RuntimeError")
            self.assertEqual(cleanup[-1]["fields"]["close_error"], "RuntimeError")

    def test_timed_adapter_contains_recorder_span_entry_failure(self):
        from icgs.contracts.method import TimedCommand
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        class BrokenRecorder:
            def span(self, *args, **kwargs):
                raise OSError("span entry failed")

            def event(self, *args, **kwargs):
                raise OSError("event failed")

        controller = _LifecycleController()
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
            recorder=BrokenRecorder(),
        )
        environment.reset(seed=7)
        transition = environment.advance(TimedCommand(np.eye(4), 1, 0.1))
        self.assertEqual(transition.after.boundary, 1)
        self.assertIn("set_target", controller.calls)

    def test_timed_cleanup_does_not_format_or_replace_unprintable_operation_error(self):
        from icgs.contracts.method import TimedCommand
        from icgs.environments.rlbench.timed import TimedLifecycleError, TimedRLBenchAdapter

        class UnprintableError(RuntimeError):
            def __str__(self):
                raise OSError("formatting failed")

        class UnprintableController(_LifecycleController):
            def __init__(self):
                super().__init__(safe_hold_failures=1, close_failures=1)
                self.failure = UnprintableError()

            def step_physics(self):
                self.calls.append("step_physics")
                raise self.failure

        controller = UnprintableController()
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
        )
        environment.reset(seed=8)
        with self.assertRaises(TimedLifecycleError) as raised:
            environment.advance(TimedCommand(np.eye(4), 1, 0.1))
        self.assertIs(raised.exception.operation_error, controller.failure)

    def test_timed_adapter_rejects_missing_low_level_physics_step(self):
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        class TaskOnlyController:
            def reset(self, seed=None):
                pass

            def set_target(self, target_w, grip):
                pass

            def observe(self):
                pass

            def simulator_time(self):
                return 0.0

            def status(self):
                return "ok"

            def close(self):
                pass

            def step(self, command):
                raise AssertionError("task.step cannot stand in for physics stepping")

        with self.assertRaisesRegex(RuntimeError, "step_physics"):
            TimedRLBenchAdapter(
                TaskOnlyController(),
                physics_dt=0.005,
                sensor_profile_id="fixture-sensor",
            )

    def test_replay_report_rejects_non_report_and_missing_fields(self):
        from icgs.environments.rlbench.replay import validate_replay_report

        with self.assertRaisesRegex(ValueError, "report"):
            validate_replay_report(True)
        with self.assertRaisesRegex(ValueError, "mode|fields"):
            validate_replay_report({"mode": "exact-snapshot"})

    def test_replay_report_requires_explicit_state_coverage(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        self.assertTrue({
            "object_poses",
            "grip",
            "controller_state",
            "task_history",
        }.issubset(REQUIRED_REPLAY_FIELDS))
        stored = ("joints", "velocities", "articulations")
        report = {
            "mode": "approximate-reset",
            "stored_fields": stored,
            "unavailable_fields": tuple(
                field for field in REQUIRED_REPLAY_FIELDS if field not in stored
            ),
            "restored_fields": stored,
            "discrepancies": {"pose": 0.0, "cloud": 0.0, "outcome": 0.0},
            "protocol_id": "fixture-replay-v1",
            "repetitions": 2,
            "uncertainty": {"pose": 0.0},
        }
        validated = validate_replay_report(report)
        self.assertEqual(validated["mode"], "approximate-reset")
        self.assertEqual(tuple(validated["stored_fields"]), stored)
        self.assertEqual(tuple(validated["unavailable_fields"]), report["unavailable_fields"])
        self.assertEqual(validated["repetitions"], 2)

    def test_replay_report_cannot_call_partial_snapshot_exact(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        report = {
            "mode": "exact-snapshot",
            "stored_fields": tuple(REQUIRED_REPLAY_FIELDS),
            "unavailable_fields": (),
            "restored_fields": tuple(REQUIRED_REPLAY_FIELDS[:-1]),
            "discrepancies": {"pose": 0.0, "cloud": 0.0, "outcome": 0.0},
            "protocol_id": "fixture-replay-v1",
        }
        with self.assertRaisesRegex(ValueError, "exact|restored"):
            validate_replay_report(report)

    def test_replay_report_rejects_unknown_coverage_and_nonfinite_measurements(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        report = {
            "mode": "deterministic-replay",
            "stored_fields": tuple(REQUIRED_REPLAY_FIELDS[:-1]) + ("unobserved",),
            "unavailable_fields": (),
            "restored_fields": (),
            "discrepancies": {"pose": float("nan"), "cloud": 0.0, "outcome": 0.0},
            "protocol_id": "fixture-replay-v1",
        }
        with self.assertRaisesRegex(ValueError, "field|finite"):
            validate_replay_report(report)

    def test_replay_exact_allows_extra_known_restored_fields(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        restored = tuple(REQUIRED_REPLAY_FIELDS) + ("command_history",)
        report = {
            "mode": "exact-snapshot",
            "stored_fields": restored,
            "unavailable_fields": (),
            "restored_fields": restored,
            "discrepancies": {"pose": 0.0, "cloud": 0.0, "outcome": 0.0},
            "protocol_id": "fixture-replay-v1",
        }
        validated = validate_replay_report(report)
        self.assertEqual(tuple(validated["restored_fields"]), restored)

    def test_replay_report_rejects_overlapping_stored_and_unavailable_fields(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        report = {
            "mode": "deterministic-replay",
            "stored_fields": tuple(REQUIRED_REPLAY_FIELDS),
            "unavailable_fields": ("joints",),
            "restored_fields": (),
            "discrepancies": {"pose": 0.0, "cloud": 0.0, "outcome": 0.0},
            "protocol_id": "fixture-replay-v1",
        }
        with self.assertRaisesRegex(ValueError, "disjoint|overlap"):
            validate_replay_report(report)

    def test_replay_report_requires_measured_discrepancy_entries(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        base = {
            "mode": "deterministic-replay",
            "stored_fields": tuple(REQUIRED_REPLAY_FIELDS),
            "unavailable_fields": (),
            "restored_fields": (),
            "protocol_id": "fixture-replay-v1",
        }
        for discrepancies in (
            {},
            {"pose": 0.0, "cloud": 0.0},
            {"pose": [0.0], "cloud": 0.0, "outcome": 0.0},
        ):
            with self.subTest(discrepancies=discrepancies):
                with self.assertRaisesRegex(ValueError, "discrepanc|finite|pose|cloud|outcome"):
                    validate_replay_report({**base, "discrepancies": discrepancies})

    def test_replay_approximate_reset_requires_repeated_finite_uncertainty(self):
        from icgs.environments.rlbench.replay import (
            REQUIRED_REPLAY_FIELDS,
            validate_replay_report,
        )

        base = {
            "mode": "approximate-reset",
            "stored_fields": tuple(REQUIRED_REPLAY_FIELDS),
            "unavailable_fields": (),
            "restored_fields": (),
            "discrepancies": {"pose": 0.0, "cloud": 0.0, "outcome": 0.0},
            "protocol_id": "fixture-replay-v1",
        }
        with self.assertRaisesRegex(ValueError, "repetitions"):
            validate_replay_report({**base, "repetitions": True, "uncertainty": {"pose": 0.1}})
        with self.assertRaisesRegex(ValueError, "repetitions"):
            validate_replay_report({**base, "repetitions": 1, "uncertainty": {"pose": 0.1}})
        with self.assertRaisesRegex(ValueError, "uncertainty"):
            validate_replay_report({**base, "repetitions": 2, "uncertainty": {}})
        with self.assertRaisesRegex(ValueError, "uncertainty|finite"):
            validate_replay_report({
                **base,
                "repetitions": 2,
                "uncertainty": {"pose": float("nan")},
            })

    def test_timed_adapter_requires_controller_safe_hold_capability(self):
        from icgs.environments.rlbench.timed import (
            TimedRLBenchAdapter,
            UnsupportedTimedController,
        )

        class NoSafeHoldController:
            def reset(self, seed=None):
                pass

            def set_target(self, target_w, grip):
                pass

            def step_physics(self):
                pass

            def observe(self):
                pass

            def simulator_time(self):
                return 0.0

            def status(self):
                return "ok"

            def close(self):
                pass

        with self.assertRaisesRegex(UnsupportedTimedController, "safe_hold"):
            TimedRLBenchAdapter(
                NoSafeHoldController(),
                physics_dt=0.005,
                sensor_profile_id="fixture-sensor",
            )

    def test_timed_adapter_cleans_up_safe_hold_before_close_and_retries_failed_close(self):
        from icgs.contracts.method import TimedCommand
        from icgs.environments.rlbench.timed import (
            TimedLifecycleError,
            TimedRLBenchAdapter,
        )

        controller = _LifecycleController(
            step_failures=1,
            safe_hold_failures=1,
            close_failures=1,
        )
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
        )
        environment.reset(seed=5)
        with self.assertRaises(TimedLifecycleError) as raised:
            environment.advance(TimedCommand(np.eye(4), 1, 0.1))

        error = raised.exception
        self.assertIn("step failure", str(error))
        self.assertIsInstance(error.operation_error, RuntimeError)
        self.assertEqual(str(error.operation_error), "step failure")
        self.assertEqual(str(error.safe_hold_error), "safe_hold failure")
        self.assertEqual(str(error.close_error), "close failure")
        self.assertEqual(
            controller.calls[-4:],
            ["set_target", "step_physics", "safe_hold", "close"],
        )
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            environment.advance(TimedCommand(np.eye(4), 1, 0.1))
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            environment.reset(seed=6)

        environment.close()
        self.assertEqual(controller.calls[-2:], ["safe_hold", "close"])
        environment.close()
        self.assertEqual(controller.calls[-2:], ["safe_hold", "close"])

    def test_timed_adapter_invalidates_after_observe_failure(self):
        from icgs.contracts.method import TimedCommand
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        controller = _LifecycleController(observe_failure_on_call=2)
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
        )
        environment.reset(seed=1)
        with self.assertRaisesRegex(RuntimeError, "observe failure"):
            environment.advance(TimedCommand(np.eye(4), 0, 0.1))
        self.assertEqual(controller.calls[-2:], ["safe_hold", "close"])
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            environment.reset(seed=2)

    def test_timed_adapter_invalidates_after_status_failure(self):
        from icgs.contracts.method import TimedCommand
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        controller = _LifecycleController(status_failure=True)
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
        )
        environment.reset(seed=1)
        with self.assertRaisesRegex(RuntimeError, "status failure"):
            environment.advance(TimedCommand(np.eye(4), 0, 0.1))
        self.assertEqual(controller.calls[-2:], ["safe_hold", "close"])
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            environment.advance(TimedCommand(np.eye(4), 0, 0.1))

    def test_timed_adapter_reset_failure_preserves_cleanup_diagnostics_and_invalidates(self):
        from icgs.environments.rlbench.timed import (
            TimedLifecycleError,
            TimedRLBenchAdapter,
        )

        controller = _LifecycleController(
            reset_failures=2,
            safe_hold_failures=1,
            close_failures=1,
        )
        environment = TimedRLBenchAdapter(
            controller,
            physics_dt=0.005,
            sensor_profile_id="fixture-sensor",
            max_reset_attempts=2,
        )
        with self.assertRaises(TimedLifecycleError) as raised:
            environment.reset(seed=3)

        error = raised.exception
        self.assertIn("reset failed", str(error))
        self.assertIn("reset failure", str(error.operation_error))
        self.assertEqual(str(error.safe_hold_error), "safe_hold failure")
        self.assertEqual(str(error.close_error), "close failure")
        self.assertEqual(controller.calls[:2], ["reset", "reset"])
        self.assertEqual(controller.calls[-2:], ["safe_hold", "close"])
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            environment.reset(seed=4)
        environment.close()
        self.assertEqual(controller.calls[-2:], ["safe_hold", "close"])


if __name__ == "__main__":
    unittest.main()
