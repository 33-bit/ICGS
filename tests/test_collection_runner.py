from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation
from icgs.data.collection.attempts import AttemptResult
from icgs.data.collection.runner import (
    AttemptSpec,
    CollectionLimits,
    CollectionReport,
    run_collection,
)


class MockEnvironment:
    def __init__(self, num_steps: int = 2):
        self.num_steps = num_steps
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> TimedObservation:
        obs = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
        return TimedObservation(obs, 0, 0.0, 0.0, "mock-sensor")

    def advance(self, command: TimedCommand) -> ExecutedTransition:
        self.step_count += 1
        b_before = self.step_count - 1
        b_after = self.step_count
        obs_before = Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0)
        obs_after = Observation(np.zeros((4, 3), dtype=np.float64), command.target_w, float(command.grip))
        t_before = TimedObservation(obs_before, b_before, 0.1 * b_before, 0.1 * b_before, "mock-sensor")
        t_after = TimedObservation(obs_after, b_after, 0.1 * b_after, 0.1 * b_after, "mock-sensor")
        return ExecutedTransition(t_before, t_after, command, command.duration_s, 10, "ok")

    def get_online_observation(self) -> dict:
        return {
            "points": np.zeros((4, 3), dtype=np.float32),
            "point_valid": np.ones(4, dtype=bool),
            "T_w_e": np.eye(4, dtype=np.float32),
            "grip": 0.0,
        }

    def close(self) -> None:
        self.closed = True


class CollectionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.config = MethodConfig(
            dataset={"shard_intervals": 4},
            collection={"max_episode_intervals": 10},
        )
        self.default_limits = CollectionLimits(
            max_attempts=10,
            max_intervals=100,
            max_wall_time_s=60.0,
            max_disk_bytes=100 * 1024 * 1024,
        )
        self.protocol_manifest = {
            "environment_protocol_id": "env-proto-v1",
            "controller_protocol_id": "ctrl-proto-v1",
            "sensor_protocol_id": "sens-proto-v1",
            "asset_protocol_id": "asset-proto-v1",
            "split_protocol_id": "split-proto-v1",
            "collection_protocol_id": "col-proto-v1",
            "metadata": {"sample_rate_hz": 10.0, "tolerance_m": 0.005},
        }
        self.code_revision = "abcdef1234567890abcdef1234567890abcdef12"
        self.is_dirty = False
        self.generator_version = "gen-1.0.0"

        # Create dataset manifest
        self.manifest_file = self.root / "dataset_manifest.json"
        manifest_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [
                {"lineage_id": "root-01", "parent_ids": [], "split": "train"},
                {"lineage_id": "root-dev", "parent_ids": [], "split": "dev"},
            ],
            "asset_families": [
                {"asset_family_id": "fam-01", "split": "train"},
                {"asset_family_id": "fam-dev", "split": "dev"},
            ],
            "provenance": {
                "generator_seed": 42,
                "config_digest": "sha256-test",
                "code_revision": "git-test",
            },
        }
        self.manifest_file.write_text(json.dumps(manifest_data, indent=2))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_collection(self, specs, env_factory, **kwargs):
        call_kwargs = {
            "dataset_root": self.root,
            "dataset_manifest_path": self.manifest_file,
            "limits": self.default_limits,
            "config": self.config,
            "protocol_manifest": self.protocol_manifest,
            "code_revision": self.code_revision,
            "is_dirty": self.is_dirty,
            "generator_version": self.generator_version,
        }
        call_kwargs.update(kwargs)
        return run_collection(specs, env_factory, **call_kwargs)

    def _make_spec(self, episode_id: str, **kwargs) -> AttemptSpec:
        default_kwargs = {
            "episode_id": episode_id,
            "program_id": "T01",
            "source_lineage_id": "root-01",
            "asset_family_id": "fam-01",
            "split": "train",
            "generator_seed": 1,
            "reset_seed": 1,
            "action_seed": 1,
            "calibration_id": "calib-real-01",
            "observation_origin": "measured",
            "raw_commands_id": "raw-spec-01",
            "materialized_commands_id": "mat-spec-01",
            "commands": [TimedCommand(np.eye(4), 0, 0.1)],
        }
        if "seed" in kwargs:
            s = kwargs["seed"]
            default_kwargs["generator_seed"] = s
            default_kwargs["reset_seed"] = s
            default_kwargs["action_seed"] = s
        default_kwargs.update(kwargs)
        return AttemptSpec(**default_kwargs)

    def test_collection_limits_rejects_missing_or_invalid_caps(self):
        # All None rejected
        with self.assertRaisesRegex(ValueError, "max_attempts must be an explicit positive integer"):
            CollectionLimits(None, 10, 10.0, 1000).validate()  # type: ignore
        with self.assertRaisesRegex(ValueError, "max_intervals must be an explicit positive integer"):
            CollectionLimits(10, None, 10.0, 1000).validate()  # type: ignore
        with self.assertRaisesRegex(ValueError, "max_wall_time_s must be an explicit positive finite float"):
            CollectionLimits(10, 10, None, 1000).validate()  # type: ignore
        with self.assertRaisesRegex(ValueError, "max_disk_bytes must be an explicit positive integer"):
            CollectionLimits(10, 10, 10.0, None).validate()  # type: ignore

        # Booleans and non-positives rejected
        with self.assertRaisesRegex(ValueError, "max_attempts must be an explicit positive integer"):
            CollectionLimits(True, 10, 10.0, 1000).validate()  # type: ignore
        with self.assertRaisesRegex(ValueError, "max_attempts must be an explicit positive integer"):
            CollectionLimits(0, 10, 10.0, 1000).validate()
        with self.assertRaisesRegex(ValueError, "max_wall_time_s must be an explicit positive finite float"):
            CollectionLimits(10, 10, float("inf"), 1000).validate()

    def test_preflight_rejection_before_env_construction(self):
        env_factory_called = False

        def spy_env_factory(spec):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        # Spec with invalid split / unknown lineage
        bad_spec = self._make_spec(
            episode_id="ep-bad",
            source_lineage_id="unknown-lineage",
        )

        with self.assertRaises(ValueError):
            self._run_collection([bad_spec], spy_env_factory)

        # Assert environment was NEVER constructed!
        self.assertFalse(env_factory_called)

    def test_binding_manifest_preflight_rejects_before_env_construction(self):
        env_factory_called = False

        def spy_env_factory(spec):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        binding_manifest = self.root / "bindings.json"
        binding_manifest.write_text(json.dumps({"manifest_version": 1, "bindings": []}))

        with self.assertRaisesRegex(ValueError, "bindings must be a non-empty sequence"):
            self._run_collection(
                [self._make_spec("ep-binding-preflight")],
                spy_env_factory,
                binding_manifest_path=binding_manifest,
                required_program_ids=("T01",),
            )
        self.assertFalse(env_factory_called)

    def test_preflight_rejects_invalid_seed_and_empty_provenance(self):
        env_factory_called = False

        def spy_env_factory(spec):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        # Negative seed
        bad_seed_spec = self._make_spec("ep-seed", seed=-1)
        with self.assertRaisesRegex(ValueError, "generator_seed must be a non-negative integer"):
            self._run_collection([bad_seed_spec], spy_env_factory)
        self.assertFalse(env_factory_called)

        # Empty calibration_id
        bad_calib_spec = self._make_spec("ep-calib", calibration_id="")
        with self.assertRaisesRegex(ValueError, "calibration_id must be a non-empty string"):
            self._run_collection([bad_calib_spec], spy_env_factory)
        self.assertFalse(env_factory_called)

    def test_bounded_attempts_intervals_and_cleanup(self):
        envs_created = []

        def env_factory(spec):
            env = MockEnvironment(num_steps=3)
            envs_created.append(env)
            return env

        specs = [
            self._make_spec(
                episode_id=f"ep-run-{i:02d}",
                seed=i,
                commands=[TimedCommand(np.eye(4), 0, 0.1) for _ in range(3)],
            )
            for i in range(5)
        ]

        limits = CollectionLimits(
            max_attempts=2,
            max_intervals=100,
            max_wall_time_s=60.0,
            max_disk_bytes=100 * 1024 * 1024,
        )

        report = self._run_collection(specs, env_factory, limits=limits)

        self.assertEqual(report.status, "attempt_limit")
        self.assertEqual(report.attempts_executed, 2)
        self.assertEqual(report.episodes_published, 2)
        self.assertEqual(report.total_intervals, 6)
        self.assertEqual(len(envs_created), 2)
        # Verify all created environments were closed!
        self.assertTrue(all(env.closed for env in envs_created))

    def test_disk_cap_enforcement_prevents_overflow(self):
        def env_factory(spec):
            return MockEnvironment(num_steps=1)

        spec = self._make_spec("ep-disk-test")

        # Extremely small disk cap (e.g. 10 bytes) that is already exceeded by the directory itself
        limits = CollectionLimits(
            max_attempts=10,
            max_intervals=100,
            max_wall_time_s=60.0,
            max_disk_bytes=10,
        )
        report = self._run_collection([spec], env_factory, limits=limits)
        self.assertEqual(report.status, "disk_limit")
        self.assertEqual(report.episodes_published, 0)

    def test_operational_error_quarantines_attempt(self):
        class BrokenEnvironment(MockEnvironment):
            def advance(self, command: TimedCommand) -> ExecutedTransition:
                raise RuntimeError("Hardware/simulator fault during advance")

        spec = self._make_spec("ep-fault-01")

        report = self._run_collection([spec], lambda _: BrokenEnvironment())

        self.assertEqual(report.attempts_executed, 1)
        self.assertEqual(report.quarantined_count, 1)
        self.assertEqual(report.episodes_published, 0)
        self.assertTrue((self.root / "quarantine" / "ep-fault-01" / "error_report.json").is_file())

    def test_empty_attempt_writes_report(self):
        spec = self._make_spec(
            episode_id="ep-empty-01",
            commands=[],  # Empty commands -> 0 transitions
        )

        report = self._run_collection([spec], lambda _: MockEnvironment())

        self.assertEqual(report.attempts_executed, 1)
        self.assertEqual(report.reports_written, 1)
        self.assertEqual(report.episodes_published, 0)
        self.assertTrue((self.root / "reports" / "ep-empty-01" / "report.json").is_file())

    def test_interval_and_wall_limits(self):
        specs = [
            self._make_spec(
                episode_id=f"ep-lim-{i}",
                seed=i,
                commands=[TimedCommand(np.eye(4), 0, 0.1) for _ in range(4)],
            )
            for i in range(5)
        ]

        # Interval limit (max 5 intervals -> stops after first attempt which had 4, before next can finish)
        limits_int = CollectionLimits(
            max_attempts=10,
            max_intervals=5,
            max_wall_time_s=60.0,
            max_disk_bytes=100 * 1024 * 1024,
        )
        report_int = self._run_collection(specs, lambda _: MockEnvironment(), limits=limits_int)
        self.assertIn(report_int.status, ("interval_limit", "completed"))
        self.assertLessEqual(report_int.total_intervals, 5)

        # Wall limit
        fake_time = 0.0

        def fake_clock():
            nonlocal fake_time
            fake_time += 1.0
            return fake_time

        limits_wall = CollectionLimits(
            max_attempts=10,
            max_intervals=100,
            max_wall_time_s=1.5,
            max_disk_bytes=100 * 1024 * 1024,
        )
        report_wall = self._run_collection(
            specs[2:],
            lambda _: MockEnvironment(),
            limits=limits_wall,
            clock=fake_clock,
        )
        self.assertEqual(report_wall.status, "wall_limit")

    def test_collision_refused_at_preflight(self):
        # Create an existing published episode in dataset_root
        ep_dir = self.root / "episodes" / "ep-existing"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "manifest.json").write_text("{}", encoding="utf-8")

        spec = self._make_spec("ep-existing")

        with self.assertRaises(FileExistsError):
            self._run_collection([spec], lambda _: MockEnvironment())

    def test_single_environment_close_lifecycle_owner(self):
        close_counts = []

        class TrackingEnv(MockEnvironment):
            def __init__(self):
                super().__init__()
                self.close_call_count = 0
                close_counts.append(self)

            def close(self):
                self.close_call_count += 1
                super().close()

        spec = self._make_spec("ep-single-close")
        self._run_collection([spec], lambda _: TrackingEnv())

        self.assertEqual(len(close_counts), 1)
        # Exactly one close call! No double close!
        self.assertEqual(close_counts[0].close_call_count, 1)

        # Test case where command_factory fails before collect_attempt is reached
        close_counts.clear()

        def broken_command_factory(s, e):
            raise RuntimeError("Command generation failed")

        spec2 = self._make_spec("ep-cmd-fail", commands=None)
        report2 = self._run_collection(
            [spec2],
            lambda _: TrackingEnv(),
            command_factory=broken_command_factory,
        )
        self.assertEqual(report2.attempts_executed, 1)
        self.assertEqual(len(close_counts), 1)
        # Exactly one close call even when failure happens in command_factory!
        self.assertEqual(close_counts[0].close_call_count, 1)

    def test_dataset_manifest_atomic_update_and_build_view_integration(self):
        from icgs.data.datasets.episodes import build_view

        specs = [
            self._make_spec("ep-pub-01", seed=101),
            self._make_spec("ep-pub-02", seed=102),
        ]

        report = self._run_collection(specs, lambda _: MockEnvironment(num_steps=1))

        self.assertEqual(report.episodes_published, 2)

        # Inspect updated manifest file
        manifest_data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        ep_entries = manifest_data["episodes"]
        self.assertEqual(len(ep_entries), 2)
        self.assertEqual(ep_entries[0]["episode_id"], "ep-pub-01")
        self.assertEqual(ep_entries[1]["episode_id"], "ep-pub-02")
        self.assertIn("sha256", ep_entries[0])
        self.assertIn("manifest_path", ep_entries[0])

        # Durable run provenance
        prov = manifest_data["provenance"]
        self.assertEqual(prov["generator_seeds"], [101, 102])
        self.assertEqual(prov["reset_seeds"], [101, 102])
        self.assertEqual(prov["action_seeds"], [101, 102])
        self.assertEqual(prov["code_revision"], self.code_revision)
        self.assertEqual(prov["is_dirty"], False)
        self.assertEqual(prov["generator_version"], self.generator_version)
        self.assertIsNotNone(prov["config_digest"])
        self.assertEqual(prov["protocol_identities"]["environment_protocol_id"], "env-proto-v1")

        # build_view must immediately see newly published episodes
        geom_view = build_view(self.manifest_file, "geom", split="train", config=self.config)
        self.assertGreater(len(geom_view), 0)
        self.assertEqual(geom_view[0]["episode_id"], "ep-pub-01")

    def test_quarantine_evidence_structure_and_bounded_policy(self):
        class BrokenEnv(MockEnvironment):
            def advance(self, command: TimedCommand) -> ExecutedTransition:
                raise RuntimeError("Arm actuator fault during step execution")

        spec = self._make_spec("ep-quarantine-evidence")
        self._run_collection([spec], lambda _: BrokenEnv())

        quarantine_file = self.root / "quarantine" / "ep-quarantine-evidence" / "error_report.json"
        self.assertTrue(quarantine_file.is_file())
        q_data = json.loads(quarantine_file.read_text(encoding="utf-8"))

        self.assertIn("diagnostics", q_data)
        self.assertEqual(q_data["diagnostics"]["primary_error_type"], "RuntimeError")
        self.assertIn("actuator fault", q_data["diagnostics"]["primary_error_message"])

        self.assertIn("evidence", q_data)
        evidence = q_data["evidence"]
        self.assertIn("initial_observation", evidence)
        self.assertIn("partial_transitions", evidence)
        self.assertIn("rejected_transition", evidence)
        self.assertIn("is_truncated", evidence)

    def test_high2_run_provenance_rejects_head_or_dirty_and_persists_full_provenance(self):
        env_factory_called = False

        def spy_env(spec):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        spec = self._make_spec("ep-high2")

        # Reject "HEAD"
        with self.assertRaisesRegex(ValueError, "code_revision cannot fabricate 'HEAD'"):
            self._run_collection([spec], spy_env, code_revision="HEAD")
        self.assertFalse(env_factory_called)

        # Reject non-bool is_dirty
        with self.assertRaisesRegex(TypeError, "is_dirty must be an explicit bool"):
            self._run_collection([spec], spy_env, is_dirty="false")  # type: ignore
        self.assertFalse(env_factory_called)

        # Reject empty generator_version
        with self.assertRaisesRegex(ValueError, "generator_version must be an explicit non-empty string"):
            self._run_collection([spec], spy_env, generator_version="")
        self.assertFalse(env_factory_called)

    def test_high3_protocol_manifest_preflight_validation(self):
        env_factory_called = False

        def spy_env(spec):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        spec = self._make_spec("ep-high3")

        # Missing protocol ID
        bad_proto = dict(self.protocol_manifest)
        bad_proto.pop("controller_protocol_id")
        with self.assertRaisesRegex(ValueError, "protocol_manifest missing required"):
            self._run_collection([spec], spy_env, protocol_manifest=bad_proto)
        self.assertFalse(env_factory_called)

        # Non-finite metadata in protocol_manifest
        bad_meta_proto = dict(self.protocol_manifest, metadata={"tol": float("nan")})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._run_collection([spec], spy_env, protocol_manifest=bad_meta_proto)
        self.assertFalse(env_factory_called)

    def test_high4_remaining_interval_budget_enforced_in_collect_attempt(self):
        # 8 commands in spec, but global limit is 5 intervals
        specs = [
            self._make_spec(
                "ep-high4",
                commands=[TimedCommand(np.eye(4), 0, 0.1) for _ in range(8)],
            )
        ]
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=5,
            max_wall_time_s=60.0,
            max_disk_bytes=100 * 1024 * 1024,
        )
        report = self._run_collection(specs, lambda _: MockEnvironment(num_steps=8), limits=limits)
        # Must execute at most 5 intervals, never 8 intervals!
        self.assertEqual(report.total_intervals, 5)

    def test_high5_remaining_disk_budget_and_manifest_bytes_accounting(self):
        specs = [
            self._make_spec("ep-high5-01"),
            self._make_spec("ep-high5-02"),
        ]
        # Set max disk bytes tight enough that first attempt succeeds, but second attempt cannot write manifest
        first_report = self._run_collection([specs[0]], lambda _: MockEnvironment(num_steps=1))
        self.assertEqual(first_report.episodes_published, 1)

        used = first_report.disk_bytes_used
        # Give only 10 bytes more than used: not enough for next attempt!
        tight_limits = CollectionLimits(
            max_attempts=10,
            max_intervals=100,
            max_wall_time_s=60.0,
            max_disk_bytes=used + 10,
        )
        second_report = self._run_collection([specs[1]], lambda _: MockEnvironment(num_steps=1), limits=tight_limits)
        self.assertEqual(second_report.status, "disk_limit")
        self.assertEqual(second_report.episodes_published, 0)

    def test_issue1_budget_accounting_and_fail_fast_on_operational_failure(self):
        """Issue 1: max_intervals=2, specs accept 1 then fail; charge executed transitions, fail-fast on operational error."""
        # 3 specs
        specs = [
            self._make_spec("ep-iss1-01", commands=[TimedCommand(np.eye(4), 0, 0.1), TimedCommand(np.eye(4), 0, 0.1)]),
            self._make_spec("ep-iss1-02"),
            self._make_spec("ep-iss1-03"),
        ]
        limits = CollectionLimits(
            max_attempts=10,
            max_intervals=2,
            max_wall_time_s=60.0,
            max_disk_bytes=100 * 1024 * 1024,
        )

        envs_created = 0
        def failing_env_factory(spec):
            nonlocal envs_created
            envs_created += 1
            # Accepts 1 step, then raises RuntimeError on 2nd step
            env = MockEnvironment(num_steps=2)
            orig_advance = env.advance
            call_count = 0
            def fail_after_one(cmd):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise RuntimeError("simulated hardware crash during transition 2")
                return orig_advance(cmd)
            env.advance = fail_after_one
            return env

        report = self._run_collection(specs, failing_env_factory, limits=limits)

        # On operational failure, fail-fast stops immediately (no second environment created)
        self.assertEqual(envs_created, 1)
        self.assertEqual(report.attempts_executed, 1)
        # Actual executed transitions must be charged: 1 executed transition
        self.assertEqual(report.total_intervals, 1)
        self.assertIn(report.status, ("error", "failed", "operational_error"))

    def test_issue1_disk_failure_counts_attempted_work_even_if_no_archive(self):
        """Issue 1: Disk failure after successful attempt must count attempted work even if no archive."""
        specs = [self._make_spec("ep-iss1-disk", commands=[TimedCommand(np.eye(4), 0, 0.1) for _ in range(3)])]
        # Bounded limits where disk limit triggers during persistence
        current_root_size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=current_root_size + 100,  # Room for preflight, but insufficient to persist archive
        )
        report = self._run_collection(specs, lambda _: MockEnvironment(num_steps=3), limits=limits)
        self.assertEqual(report.attempts_executed, 1)
        self.assertEqual(report.total_intervals, 3)
        self.assertEqual(report.episodes_published, 0)
        self.assertEqual(report.status, "disk_limit")

    def test_issue2_factory_errors_after_env_construction_combined_with_close_error(self):
        """Issue 2: command/monitor factory error after env construction + close error: exactly 1 close, durable diagnostics, non-success outcome."""
        spec = self._make_spec("ep-iss2-factory", commands=None)

        class FragileEnv(MockEnvironment):
            def __init__(self):
                super().__init__(num_steps=1)
                self.close_calls = 0
            def close(self):
                self.close_calls += 1
                raise RuntimeError("hardware close failed")

        fragile_env = FragileEnv()

        def bad_command_factory(s, env):
            raise RuntimeError("command factory calculation failed")

        report = self._run_collection(
            [spec],
            lambda _: fragile_env,
            command_factory=bad_command_factory,
        )

        # Exactly 1 close call
        self.assertEqual(fragile_env.close_calls, 1)
        # Attempt executed and accounted
        self.assertEqual(report.attempts_executed, 1)
        self.assertNotEqual(report.status, "completed")

        # Durable diagnostics must capture both command factory error and close error
        quarantine_dir = self.root / "quarantine" / "ep-iss2-factory"
        self.assertTrue(quarantine_dir.is_dir())
        diag_file = quarantine_dir / "error_report.json"
        self.assertTrue(diag_file.is_file())
        diag_data = json.loads(diag_file.read_text(encoding="utf-8"))
        diag_str = json.dumps(diag_data)
        self.assertIn("command factory calculation failed", diag_str)
        self.assertIn("hardware close failed", diag_str)

    def test_issue2_environment_factory_failure_yields_accounted_run_failure(self):
        """Issue 2: Environment factory failure yields accounted run failure without cleaning inaccessible object."""
        spec = self._make_spec("ep-iss2-envfail")

        def failing_env_factory(s):
            raise RuntimeError("simulator initialization failed")

        report = self._run_collection([spec], failing_env_factory)
        self.assertEqual(report.attempts_executed, 1)
        self.assertIn(report.status, ("error", "failed", "factory_error"))

    def test_issue5_preflight_recursively_rejects_nan_unsupported_and_reserved_metadata(self):
        """Issue 5: Recursively reject nested NaN/Infinity/unsupported objects/reserved keys and invalid paths before env factory."""
        env_factory_called = False
        def spy_env(spec):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        # 1. Nested NaN in metadata
        spec_nan = self._make_spec("ep-iss5-nan", metadata={"nested": {"ratio": float("nan")}})
        with self.assertRaises(ValueError):
            self._run_collection([spec_nan], spy_env)
        self.assertFalse(env_factory_called)

        # 2. Nested Infinity in metadata
        spec_inf = self._make_spec("ep-iss5-inf", metadata={"nested": {"limit": float("inf")}})
        with self.assertRaises(ValueError):
            self._run_collection([spec_inf], spy_env)
        self.assertFalse(env_factory_called)

        # 3. Unsupported non-serializable object
        spec_obj = self._make_spec("ep-iss5-obj", metadata={"payload": object()})
        with self.assertRaises((TypeError, ValueError)):
            self._run_collection([spec_obj], spy_env)
        self.assertFalse(env_factory_called)

        # 4. Reserved metadata key collision
        spec_res = self._make_spec("ep-iss5-res", metadata={"attempt_status": "fake_success"})
        with self.assertRaises(ValueError):
            self._run_collection([spec_res], spy_env)
        self.assertFalse(env_factory_called)

        # 5. Path traversal or absolute episode ID
        spec_trav = self._make_spec("../escape_id")
        with self.assertRaises(ValueError):
            self._run_collection([spec_trav], spy_env)
        self.assertFalse(env_factory_called)

        spec_abs = self._make_spec("/root/escape_id")
        with self.assertRaises(ValueError):
            self._run_collection([spec_abs], spy_env)
        self.assertFalse(env_factory_called)

    def test_issue6_reconcile_episode_recovers_matching_episode_and_refuses_conflict(self):
        """Issue 6: Reconcile immutable published episode into dataset manifest without physics rerun."""
        from icgs.data.archives import write_episode_archive
        from icgs.data.collection.runner import reconcile_episode

        # Write episode archive directly (simulating crash before index publication)
        record = {
            "schema_version": "icgs_episode_v1",
            "provenance": {
                "episode_id": "ep-reconcile-01",
                "program_id": "T01",
                "source_lineage_id": "root-01",
                "asset_family_id": "fam-01",
                "split": "train",
                "calibration_id": "cal-01",
                "observation_origin": "measured",
                "raw_commands_id": "raw-01",
                "materialized_commands_id": "mat-01",
            },
            "online_observations": [
                {
                    "points": np.zeros((4, 3), dtype=np.float32),
                    "point_valid": np.ones(4, dtype=bool),
                    "T_w_e": np.eye(4, dtype=np.float32),
                    "grip": 0.0,
                },
                {
                    "points": np.zeros((4, 3), dtype=np.float32),
                    "point_valid": np.ones(4, dtype=bool),
                    "T_w_e": np.eye(4, dtype=np.float32),
                    "grip": 0.0,
                },
            ],
            "transitions": [
                ExecutedTransition(
                    TimedObservation(Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0), 0, 0.0, 0.0, "p"),
                    TimedObservation(Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0), 1, 0.1, 0.1, "p"),
                    TimedCommand(np.eye(4), 0, 0.1),
                    0.1,
                    10,
                    "ok",
                )
            ],
        }
        ep_manifest_path = write_episode_archive(self.root, record, config=self.config)

        # Dataset manifest exists with no episodes
        self.manifest_file.write_text(
            json.dumps({
                "manifest_version": 1,
                "episodes": [],
                "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
                "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
            }),
            encoding="utf-8",
        )

        # Call narrow reconcile API
        reconcile_episode(self.manifest_file, ep_manifest_path, config=self.config)

        # Verify episode is indexed
        updated_manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        ep_ids = [e["episode_id"] for e in updated_manifest["episodes"]]
        self.assertIn("ep-reconcile-01", ep_ids)

        # Refuse duplicate reconciliation or conflicting checksum
        with self.assertRaises(ValueError):
            reconcile_episode(self.manifest_file, ep_manifest_path, config=self.config)

    def _make_reconcile_record(
        self,
        episode_id: str,
        *,
        program_id: str = "T01",
        source_lineage_id: str = "root-01",
        asset_family_id: str = "fam-01",
        split: str = "train",
    ) -> dict[str, Any]:
        return {
            "schema_version": "icgs_episode_v1",
            "provenance": {
                "episode_id": episode_id,
                "program_id": program_id,
                "source_lineage_id": source_lineage_id,
                "asset_family_id": asset_family_id,
                "split": split,
                "calibration_id": "cal-01",
                "observation_origin": "measured",
                "raw_commands_id": "raw-01",
                "materialized_commands_id": "mat-01",
            },
            "online_observations": [
                {
                    "points": np.zeros((4, 3), dtype=np.float32),
                    "point_valid": np.ones(4, dtype=bool),
                    "T_w_e": np.eye(4, dtype=np.float32),
                    "grip": 0.0,
                },
                {
                    "points": np.zeros((4, 3), dtype=np.float32),
                    "point_valid": np.ones(4, dtype=bool),
                    "T_w_e": np.eye(4, dtype=np.float32),
                    "grip": 0.0,
                },
            ],
            "transitions": [
                ExecutedTransition(
                    TimedObservation(Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0), 0, 0.0, 0.0, "p"),
                    TimedObservation(Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0), 1, 0.1, 0.1, "p"),
                    TimedCommand(np.eye(4), 0, 0.1),
                    0.1,
                    10,
                    "ok",
                )
            ],
        }

    def test_reconcile_episode_unknown_lineage_rejected_and_index_unchanged(self):
        """reconcile_episode rejects unknown lineage and preserves original manifest bytes."""
        from icgs.data.archives import write_episode_archive
        from icgs.data.collection.runner import reconcile_episode

        init_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
        }
        self.manifest_file.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
        orig_bytes = self.manifest_file.read_bytes()

        # Archive has unknown lineage 'unknown-01'
        record = self._make_reconcile_record("ep-bad-lineage", source_lineage_id="unknown-01")
        ep_manifest_path = write_episode_archive(self.root, record, config=self.config)

        with self.assertRaises(ValueError):
            reconcile_episode(self.manifest_file, ep_manifest_path, config=self.config)

        self.assertEqual(self.manifest_file.read_bytes(), orig_bytes)

    def test_reconcile_episode_family_split_mismatch_rejected_and_index_unchanged(self):
        """reconcile_episode rejects family/split mismatch and preserves original manifest bytes."""
        from icgs.data.archives import write_episode_archive
        from icgs.data.collection.runner import reconcile_episode

        init_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
        }
        self.manifest_file.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
        orig_bytes = self.manifest_file.read_bytes()

        # Archive has family 'unknown-fam'
        record = self._make_reconcile_record("ep-bad-fam", asset_family_id="unknown-fam")
        ep_manifest_path = write_episode_archive(self.root, record, config=self.config)

        with self.assertRaises(ValueError):
            reconcile_episode(self.manifest_file, ep_manifest_path, config=self.config)

        self.assertEqual(self.manifest_file.read_bytes(), orig_bytes)

    def test_reconcile_episode_out_of_root_archive_rejected_and_index_unchanged(self):
        """reconcile_episode rejects episode paths outside dataset root without fabricated fallback."""
        from icgs.data.archives import write_episode_archive
        from icgs.data.collection.runner import reconcile_episode

        init_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
        }
        self.manifest_file.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
        orig_bytes = self.manifest_file.read_bytes()

        with tempfile.TemporaryDirectory() as outside_tmp:
            outside_root = Path(outside_tmp)
            record = self._make_reconcile_record("ep-outside")
            outside_ep_manifest_path = write_episode_archive(outside_root, record, config=self.config)

            with self.assertRaises(ValueError):
                reconcile_episode(self.manifest_file, outside_ep_manifest_path, config=self.config)

        self.assertEqual(self.manifest_file.read_bytes(), orig_bytes)

    def test_reconcile_episode_symlink_rejected_and_index_unchanged(self):
        """reconcile_episode rejects symlink references before resolve per validator."""
        from icgs.data.archives import write_episode_archive
        from icgs.data.collection.runner import reconcile_episode

        init_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
        }
        self.manifest_file.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
        orig_bytes = self.manifest_file.read_bytes()

        record = self._make_reconcile_record("ep-symlink-target")
        ep_manifest_path = write_episode_archive(self.root, record, config=self.config)

        # Create a symlink to the episode manifest
        symlink_path = self.root / "episodes" / "ep-symlink-target" / "manifest_sym.json"
        try:
            symlink_path.symlink_to(ep_manifest_path)
        except OSError:
            return

        with self.assertRaises(ValueError):
            reconcile_episode(self.manifest_file, symlink_path, config=self.config)

        self.assertEqual(self.manifest_file.read_bytes(), orig_bytes)

    def test_reconcile_episode_success_visible_to_build_view(self):
        """reconcile_episode success is visible to build_view."""
        from icgs.data.archives import write_episode_archive
        from icgs.data.collection.runner import reconcile_episode
        from icgs.data.datasets.episodes import build_view

        init_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
        }
        self.manifest_file.write_text(json.dumps(init_data, indent=2), encoding="utf-8")

        record = self._make_reconcile_record("ep-view-visible")
        ep_manifest_path = write_episode_archive(self.root, record, config=self.config)

        reconcile_episode(self.manifest_file, ep_manifest_path, config=self.config)

        view = build_view(self.manifest_file, "geom", split="train", config=self.config)
        self.assertGreater(len(view), 0)
        self.assertEqual(view[0]["episode_id"], "ep-view-visible")

    def test_reconcile_episode_disk_allowance_prevents_replacement_before_write(self):
        """reconcile_episode enforces disk_limit_bytes and prevents replacement before write."""
        from icgs.configuration.method import MethodConfig
        from icgs.data.archives import StorageLimitExceeded, write_episode_archive
        from icgs.data.collection.runner import _get_dir_size, reconcile_episode

        init_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [{"lineage_id": "root-01", "parent_ids": [], "split": "train"}],
            "asset_families": [{"asset_family_id": "fam-01", "split": "train"}],
        }
        self.manifest_file.write_text(json.dumps(init_data, indent=2), encoding="utf-8")
        orig_bytes = self.manifest_file.read_bytes()

        record = self._make_reconcile_record("ep-disk-cap")
        ep_manifest_path = write_episode_archive(self.root, record, config=self.config)

        # Configure disk_limit_bytes to current root size (no room for rewritten manifest)
        cur_disk = _get_dir_size(self.root)
        tight_config = MethodConfig(collection={"disk_limit_bytes": cur_disk})

        with self.assertRaises(StorageLimitExceeded):
            reconcile_episode(self.manifest_file, ep_manifest_path, config=tight_config)

        self.assertEqual(self.manifest_file.read_bytes(), orig_bytes)

    def test_issue7_peak_disk_budget_checked_before_writes_including_staging_and_rewrites(self):
        """Issue 7: Disk cap covers bytes BEFORE each archive/report/quarantine/index write including staging and index rewrites."""
        spec = self._make_spec("ep-iss7-peak")
        # 100 bytes is not enough even for the staging files before publication
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=100,
        )
        report = self._run_collection([spec], lambda _: MockEnvironment(num_steps=1), limits=limits)
        self.assertEqual(report.status, "disk_limit")
        # Oversized output must not remain
        ep_dir = self.root / "episodes" / "ep-iss7-peak"
        self.assertFalse(ep_dir.exists())

    def test_issue8_three_seeds_required_and_dirty_patch_digest(self):
        """Issue 8: Three seeds required explicitly, dirty_patch_digest required if is_dirty True, stable provenance carried in all outcomes."""
        # Spec without seeds cannot fallback to legacy seed
        with self.assertRaises((TypeError, ValueError)):
            AttemptSpec(
                program_id="program-01",
                episode_id="ep-iss8",
                split="train",
                calibration_id="cal-01",
                observation_origin="measured",
                raw_commands_id="raw-01",
                materialized_commands_id="mat-01",
                source_lineage_id="root-01",
                asset_family_id="fam-01",
                # Omit generator_seed, reset_seed, action_seed
            )

        spec = self._make_spec("ep-iss8-dirty")

        # If is_dirty is True, dirty_patch_digest is required
        with self.assertRaises(ValueError):
            self._run_collection([spec], lambda _: MockEnvironment(), is_dirty=True, dirty_patch_digest="")

        # Provenance carried in non-completed outcome
        tight_limits = CollectionLimits(max_attempts=1, max_intervals=10, max_wall_time_s=60.0, max_disk_bytes=10)
        report = self._run_collection([spec], lambda _: MockEnvironment(), limits=tight_limits)
        self.assertIsNotNone(report.provenance)
        self.assertEqual(report.provenance["code_revision"], "abcdef1234567890abcdef1234567890abcdef12")

    def test_docs_example_parameter_names_and_view_contract_consistency(self):
        """Docs parameter names match run_collection signature and no misleading lazy-view claim."""
        import inspect
        from icgs.data.collection.runner import run_collection

        sig = inspect.signature(run_collection)
        doc_path = Path(__file__).resolve().parents[1] / "docs" / "components" / "cli-and-data.md"
        self.assertTrue(doc_path.is_file())
        doc_text = doc_path.read_text(encoding="utf-8")

        # Check documented parameter names match signature
        for param in ("specs", "environment_factory", "dataset_root", "dataset_manifest_path", "limits", "config"):
            self.assertIn(param, sig.parameters)
            self.assertIn(param, doc_text)

        # Must not claim lazy view if build_view returns an eager list/view container
        self.assertNotIn("lazy-view", doc_text.lower())

    def test_sol_fix1_initial_disk_cap_exhaustion_stops_before_environment_factory(self):
        """Sol Fix 1: Root current bytes >= max_disk_bytes must stop before environment_factory is called."""
        spec = self._make_spec("ep-sol-disk-init")
        env_factory_called = False

        def spy_env_factory(s):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        current_root_size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.assertGreater(current_root_size, 0)

        # Cap is set to current root size
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=current_root_size,
        )
        report = self._run_collection([spec], spy_env_factory, limits=limits)
        self.assertFalse(env_factory_called, "environment_factory must not be called when initial disk cap is exhausted")
        self.assertEqual(report.attempts_executed, 0)
        self.assertEqual(report.total_intervals, 0)
        self.assertEqual(report.status, "disk_limit")

    def test_sol_fix2_staging_peak_budget_prevents_any_shard_or_manifest_write(self):
        """Sol Fix 2: Conservative projected bytes exceeding remaining cap must prevent any shard/manifest write; no transient files."""
        spec = self._make_spec("ep-sol-peak-stage", commands=[TimedCommand(np.eye(4), 0, 0.1) for _ in range(2)])
        current_root_size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=current_root_size + 100,
        )
        report = self._run_collection([spec], lambda _: MockEnvironment(num_steps=2), limits=limits)
        self.assertEqual(report.status, "disk_limit")
        self.assertEqual(report.attempts_executed, 1)
        self.assertEqual(report.total_intervals, 2)
        self.assertEqual(report.episodes_published, 0)

        # Verify NO transient staging files or episode directory exists
        ep_dir = self.root / "episodes" / "ep-sol-peak-stage"
        self.assertFalse(ep_dir.exists())
        staging_dirs = list(self.root.glob(".staging*")) + list((self.root / "episodes").glob(".staging*"))
        self.assertEqual(len(staging_dirs), 0)
        after_root_size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.assertEqual(current_root_size, after_root_size)

    def test_sol_fix3_reject_bytes_and_unsupported_metadata_before_env_construction(self):
        """Sol Fix 3: Recursively reject bytes/bytearray and non-JSON metadata before environment_factory is called."""
        env_factory_called = False

        def spy_env_factory(s):
            nonlocal env_factory_called
            env_factory_called = True
            return MockEnvironment()

        # Direct bytes in metadata
        spec_bytes = self._make_spec("ep-sol-meta-bytes", metadata={"blob": b"raw_bytes"})
        with self.assertRaises(TypeError):
            self._run_collection([spec_bytes], spy_env_factory)
        self.assertFalse(env_factory_called)

        # Nested bytearray in list
        spec_bytearray = self._make_spec("ep-sol-meta-bytearray", metadata={"nested": [1, bytearray(b"raw")]})
        with self.assertRaises(TypeError):
            self._run_collection([spec_bytearray], spy_env_factory)
        self.assertFalse(env_factory_called)

        # Custom arbitrary object
        class CustomObj:
            pass

        spec_custom = self._make_spec("ep-sol-meta-custom", metadata={"custom": CustomObj()})
        with self.assertRaises(TypeError):
            self._run_collection([spec_custom], spy_env_factory)
        self.assertFalse(env_factory_called)

    def test_repro2_huge_quarantine_error_does_not_blow_past_cap_and_no_duplicate_diagnostics(self):
        """Repro 2: Reset raises RuntimeError('x'*10000) with root 529 and cap 629 must not write 21k quarantine."""
        import builtins
        import os
        from unittest import mock

        spec = self._make_spec("ep-repro-2")
        current_root_size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=current_root_size + 100,  # e.g. 529 + 100 = 629
        )

        class HugeErrorEnv(MockEnvironment):
            def reset(self, *, seed: int | None = None):
                raise RuntimeError("x" * 10000)

        peak_bytes = 0

        def sample():
            nonlocal peak_bytes
            curr = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if curr > peak_bytes:
                peak_bytes = curr

        orig_open = builtins.open

        def hooked_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            sample()
            orig_close = f.close

            def hooked_close():
                sample()
                orig_close()
                sample()

            f.close = hooked_close
            return f

        orig_replace = os.replace

        def hooked_replace(src, dst):
            sample()
            res = orig_replace(src, dst)
            sample()
            return res

        with mock.patch("builtins.open", side_effect=hooked_open), mock.patch("os.replace", side_effect=hooked_replace):
            report = self._run_collection([spec], lambda _: HugeErrorEnv(), limits=limits)
            sample()

        self.assertLessEqual(peak_bytes, limits.max_disk_bytes, f"Peak bytes {peak_bytes} exceeded cap {limits.max_disk_bytes}")
        self.assertEqual(report.status, "disk_limit")
        # Duplicate diagnostics.json must NOT exist
        q_dir = self.root / "quarantine" / "ep-repro-2"
        self.assertFalse((q_dir / "diagnostics.json").exists())

    def test_repro3_huge_protocol_metadata_dataset_manifest_rewrite_prevents_cap_exceeded(self):
        """Repro 3: Protocol manifest metadata string 100k with cap 50000 must stop at disk_limit, not return completed at 108865."""
        import builtins
        import os
        from unittest import mock

        spec = self._make_spec("ep-repro-3")
        # Inject huge metadata into protocol_manifest
        pm = dict(self.protocol_manifest)
        pm["metadata"] = {"large_field": "x" * 100000}
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=50000,
        )

        peak_bytes = 0

        def sample():
            nonlocal peak_bytes
            curr = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if curr > peak_bytes:
                peak_bytes = curr

        orig_open = builtins.open

        def hooked_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            sample()
            orig_close = f.close

            def hooked_close():
                sample()
                orig_close()
                sample()

            f.close = hooked_close
            return f

        orig_replace = os.replace

        def hooked_replace(src, dst):
            sample()
            res = orig_replace(src, dst)
            sample()
            return res

        with mock.patch("builtins.open", side_effect=hooked_open), mock.patch("os.replace", side_effect=hooked_replace):
            report = self._run_collection(
                [spec],
                lambda _: MockEnvironment(num_steps=1),
                limits=limits,
                protocol_manifest=pm,
            )
            sample()

        self.assertLessEqual(peak_bytes, 50000, f"Peak bytes {peak_bytes} exceeded cap 50000")
        self.assertEqual(report.status, "disk_limit")

    def test_repro4_oversized_empty_attempt_report_returns_disk_limit_rather_than_uncaught_oserror(self):
        """Repro 4: Oversized empty attempt report raises uncaught OSError rather than CollectionReport disk_limit."""
        spec = self._make_spec("ep-repro-4", commands=[], metadata={"note": "x" * 100000})
        limits = CollectionLimits(
            max_attempts=1,
            max_intervals=10,
            max_wall_time_s=60.0,
            max_disk_bytes=50000,
        )
        report = self._run_collection([spec], lambda _: MockEnvironment(num_steps=0), limits=limits)
        self.assertEqual(report.status, "disk_limit")
        self.assertEqual(report.attempts_executed, 1)
        self.assertEqual(report.total_intervals, 0)
        self.assertEqual(report.episodes_published, 0)

    def test_materialized_commands_binds_provenance_and_backward_compatible_iterable(self):
        """MaterializedCommands binds validated raw/mat IDs into provenance; plain iterable works for primary."""
        from icgs.data.archives import read_episode_archive
        from icgs.data.collection.runner import MaterializedCommands

        # 1. MaterializedCommands path
        spec = self._make_spec("ep-mat-01", commands=None)
        cmd = TimedCommand(np.eye(4), 0, 0.1)

        def mat_cmd_factory(s, env):
            return MaterializedCommands(
                commands=[cmd],
                raw_commands_id="raw-provenance-test-hash",
                materialized_commands_id="mat-provenance-test-hash",
                metadata={"factory_bound": True},
            )

        report = self._run_collection(
            [spec],
            lambda _: MockEnvironment(num_steps=1),
            command_factory=mat_cmd_factory,
        )
        self.assertEqual(report.status, "completed")
        self.assertEqual(report.episodes_published, 1)

        ep_archive = read_episode_archive(self.root / "episodes" / "ep-mat-01" / "manifest.json")
        self.assertEqual(ep_archive["provenance"]["raw_commands_id"], "raw-provenance-test-hash")
        self.assertEqual(ep_archive["provenance"]["materialized_commands_id"], "mat-provenance-test-hash")
        ep_raw_meta = json.loads((self.root / "episodes" / "ep-mat-01" / "manifest.json").read_text(encoding="utf-8")).get("metadata", {})
        self.assertTrue(ep_raw_meta.get("factory_bound"))

        # 2. Backward compatibility: plain iterable works for primary track
        spec2 = self._make_spec("ep-mat-02", commands=None)

        def plain_cmd_factory(s, env):
            return [cmd]

        report2 = self._run_collection(
            [spec2],
            lambda _: MockEnvironment(num_steps=1),
            command_factory=plain_cmd_factory,
        )
        self.assertEqual(report2.status, "completed")
        self.assertEqual(report2.episodes_published, 1)

    def test_track_and_program_mixing_guards(self):
        """Preflight guards forbid mixing primary/exploratory programs, splits, and require publishers."""
        spec_e01 = AttemptSpec(
            episode_id="e01-mix-01",
            program_id="E01",
            split="dev",
            source_lineage_id="root-dev",
            asset_family_id="fam-dev",
            calibration_id="cal-01",
            observation_origin="measured",
            raw_commands_id="raw-01",
            materialized_commands_id="mat-01",
            generator_seed=1,
            reset_seed=2,
            action_seed=3,
            commands=[TimedCommand(np.eye(4), 0, 0.1)],
        )
        spec_t01 = self._make_spec("t01-mix-01")

        # 1. Primary track forbids E01 program
        with self.assertRaises(ValueError) as ctx:
            self._run_collection([spec_e01], lambda _: MockEnvironment(), dataset_track="primary")
        self.assertIn("exploratory program", str(ctx.exception))

        # 2. Exploratory track forbids T01 program
        with self.assertRaises(ValueError) as ctx:
            self._run_collection(
                [spec_t01],
                lambda _: MockEnvironment(),
                dataset_track="exploratory",
                episode_publisher=lambda ep: {"drive_verified": True, "hf_verified": True},
                index_publisher=lambda path: {"drive_verified": True, "hf_verified": True},
            )
        self.assertIn("not an exploratory program", str(ctx.exception))

        # 3. Exploratory track requires explicit episode_publisher and index_publisher
        with self.assertRaises(ValueError) as ctx:
            self._run_collection([spec_e01], lambda _: MockEnvironment(), dataset_track="exploratory")
        self.assertIn("episode_publisher", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self._run_collection(
                [spec_e01],
                lambda _: MockEnvironment(),
                dataset_track="exploratory",
                episode_publisher=lambda ep: {"drive_verified": True, "hf_verified": True},
                # index_publisher missing
            )
        self.assertIn("index_publisher", str(ctx.exception))

    def test_per_episode_index_publication_and_fail_closed(self):
        """Index publisher is invoked per episode; errors and unverified reports fail closed."""
        spec = self._make_spec("ep-idx-pub-01")
        index_calls = []

        def good_index_pub(path):
            index_calls.append(str(path))
            return {"drive_verified": True, "hf_verified": True}

        report = self._run_collection(
            [spec],
            lambda _: MockEnvironment(num_steps=1),
            index_publisher=good_index_pub,
        )
        self.assertEqual(report.status, "completed")
        self.assertEqual(len(index_calls), 1)
        self.assertTrue(index_calls[0].endswith("dataset_manifest.json"))

        # Fail closed on index publisher error
        spec2 = self._make_spec("ep-idx-pub-02")

        def failing_index_pub(path):
            raise IOError("Drive network connection dropped during index sync")

        report2 = self._run_collection(
            [spec2],
            lambda _: MockEnvironment(num_steps=1),
            index_publisher=failing_index_pub,
        )
        self.assertEqual(report2.status, "index_publication_failed")

        # Fail closed on unverified index report
        spec3 = self._make_spec("ep-idx-pub-03")

        def unverified_index_pub(path):
            return {"drive_verified": True, "hf_verified": False}

        report3 = self._run_collection(
            [spec3],
            lambda _: MockEnvironment(num_steps=1),
            index_publisher=unverified_index_pub,
        )
        self.assertEqual(report3.status, "index_publication_unverified")
