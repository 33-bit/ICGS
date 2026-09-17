"""Focused tests for exploratory RLBench physical geom/dyn track.

Verifies:
1. Program E01 split 'dev' isolation (never mapped to T06..T14, rejected in train/test).
2. Controller conversion (RGB-D/points/pose/grip -> unquantized Observation).
3. Seed determinism and protocol metadata.
4. Drive-first uploader idempotency, verified checksums, and credential filtering.
5. Phase-0 receipt gating and bounded G1 probe lifecycle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from huggingface_hub.utils import EntryNotFoundError
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script_module(module_name: str, script_filename: str):
    script_path = SCRIPTS_DIR / script_filename
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing required script: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

from icgs.contracts.records import Observation
from icgs.data.collection.programs import get_program, program_catalog
from icgs.data.collection.uploader import (
    ExploratoryUploader,
    compute_file_sha256,
)
from icgs.data.datasets.episodes import validate_dataset_manifest
from icgs.environments.rlbench.controller import (
    RLBenchTimedController,
    convert_rlbench_observation,
    filter_and_downsample_points,
    pose_to_matrix,
    quaternion_to_rotation_matrix,
)
from icgs.environments.rlbench.teleop_oracle import (
    discretize_waypoints_to_commands,
    interpolate_poses,
)


class MockTaskEnv:
    """Mock RLBench task environment for testing controller behavior."""

    def __init__(self):
        self.reset_called = False
        self.step_count = 0
        self.pyrep = SimpleNamespace(t=0.0)
        self.pyrep.get_simulation_timestep = lambda: 0.05
        self.scene = SimpleNamespace(_pyrep=self.pyrep, step=self._step)
        self.arm = SimpleNamespace(
            get_tip=lambda: SimpleNamespace(get_pose=lambda: [0.1, -0.2, 0.85, 0, 0, 0, 1]),
            solve_ik_via_jacobian=lambda position, quaternion=None: np.zeros(7),
            set_joint_target_positions=lambda values: None,
            get_joint_positions=lambda: np.zeros(7),
        )
        self.gripper = SimpleNamespace(
            amount=np.array([1.0]),
            get_open_amount=lambda: self.gripper.amount.copy(),
            actuate=self._actuate,
        )
        self._scene = self.scene
        self._robot = SimpleNamespace(arm=self.arm, gripper=self.gripper)

    def _step(self):
        self.step_count += 1
        self.pyrep.t += self.pyrep.get_simulation_timestep()

    def _actuate(self, amount, velocity):
        self.gripper.amount[:] = amount

    def reset(self, seed: int | None = None):
        self.reset_called = True
        self.step_count = 0

    def get_observation(self):
        # 100 points around [0, 0, 1.0]
        pts = np.zeros((100, 3), dtype=np.float64)
        pts[:, 2] = 1.0
        return {
            "wrist_point_cloud": pts,
            "gripper_pose": [0.1, -0.2, 0.85, 0.0, 0.0, 0.0, 1.0],
            "gripper_open": 1.0,
        }



class MockOuterEnv:
    def __init__(self, task):
        self.task = task
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class MockSim:
    def __init__(self, task):
        self.task = task

    def simGetSimulationTime(self):
        return self.task.pyrep.t


class MockHFClient:
    def __init__(self):
        self.objects = {}
        self.commits = 0
        self.head = None
        self.permission = "write"

    def get_token_permission(self, token: str | None = None) -> str:
        del token
        return self.permission

    def whoami(self, token: str | None = None) -> dict[str, str]:
        del token
        return {"name": "mock-test-user"}

    def get_head_commit(self, revision: str = "main") -> str | None:
        return self.head

    def repo_info(self, *, repo_id: str, repo_type: str = "dataset"):
        del repo_type
        from types import SimpleNamespace
        return SimpleNamespace(id=repo_id, sha=self.head)

    def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit=None, **kwargs):
        del repo_id, repo_type, commit_message, parent_commit, kwargs
        self.commits += 1
        for op in operations:
            self.objects[op["path_in_repo"]] = Path(op["path_or_fileobj"]).read_bytes()
        self.head = f"commit-{self.commits}"
        return {"commit_hash": self.head}

    def readback(self, path, revision):
        del revision
        if path not in self.objects:
            raise EntryNotFoundError(f"File {path} not found")
        return self.objects[path]


class RLBenchExploratoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_file = self.root / "dataset_manifest.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_e01_program_spec_isolated_in_dev_split_and_never_mapped_to_t_series(self):
        """E01 must be strictly in dev split and never confused with T06..T14 tasks."""
        prog = get_program("E01")
        self.assertEqual(prog.program_id, "E01")
        self.assertEqual(prog.split, "dev")
        self.assertEqual(prog.family, "rlbench-exploratory")

        # Must never be present in train or test splits
        catalog = program_catalog()
        train_ids = {p.program_id for p in catalog.values() if p.split == "train"}
        test_ids = {p.program_id for p in catalog.values() if p.split == "test"}
        self.assertNotIn("E01", train_ids)
        self.assertNotIn("E01", test_ids)

        # Must be completely separate from approved custom tasks T06..T14
        absent_approved_tasks = {"T06", "T08", "T09", "T11", "T13", "T14"}
        for tid in absent_approved_tasks:
            t_spec = get_program(tid)
            self.assertEqual(t_spec.split, "train")
            self.assertNotEqual(t_spec.program_id, prog.program_id)
            self.assertNotEqual(t_spec.family, prog.family)

    def test_e01_catalog_rejects_primary_train_or_test_manifest_entries(self):
        """E01 cannot be relabelled as a primary train/test program while Phase 0 is blocked."""
        catalog = program_catalog()
        cat_prog = catalog["E01"]
        expected_split = "dev" if cat_prog.split in ("dev", "development") else cat_prog.split
        self.assertEqual(expected_split, "dev")

        ep_dir = self.root / "episodes" / "ep-e01-train"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_manifest = ep_dir / "manifest.json"
        ep_manifest.write_text(
            json.dumps({
                "episode_id": "ep-e01-train",
                "provenance": {
                    "split": "train",
                    "source_lineage_id": "lin-tr",
                    "asset_family_id": "fam-tr",
                    "program_id": "E01",
                },
            }),
            encoding="utf-8",
        )
        ep_sha = compute_file_sha256(ep_manifest)

        invalid_train_manifest = {
            "manifest_version": 1,
            "lineage": [{"lineage_id": "lin-tr", "split": "train", "parent_ids": []}],
            "asset_families": [{"asset_family_id": "fam-tr", "split": "train"}],
            "episodes": [
                {
                    "episode_id": "ep-e01-train",
                    "manifest_path": "episodes/ep-e01-train/manifest.json",
                    "sha256": ep_sha,
                    "split": "train",
                    "program_id": "E01",
                    "source_lineage_id": "lin-tr",
                    "asset_family_id": "fam-tr",
                }
            ],
        }
        self.manifest_file.write_text(json.dumps(invalid_train_manifest, indent=2))
        with self.assertRaisesRegex(ValueError, "catalog split dev != episode split train"):
            validate_dataset_manifest(self.manifest_file, dataset_root=self.root)

    def test_controller_observation_conversion_shapes_types_and_bounds(self):
        """Controller converts point cloud, pose, and gripper state into float64 Observation."""
        raw_obs = {
            "wrist_point_cloud": np.array([
                [0.0, 0.0, 1.0],        # In bounds
                [10.0, 10.0, 10.0],    # Out of bounds
                [np.nan, 0.0, 1.0],     # Non-finite
                [0.1, -0.1, 0.9],       # In bounds
            ], dtype=np.float32),
            "gripper_pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_open": 1.0,
        }
        obs = convert_rlbench_observation(raw_obs)

        # Points must be unquantized float64 array of shape [2, 3] (only in-bounds finite points retained)
        self.assertEqual(obs.points.dtype, np.float64)
        self.assertEqual(obs.points.shape, (2, 3))
        np.testing.assert_allclose(obs.points[0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(obs.points[1], [0.1, -0.1, 0.9])

        # Pose must be 4x4 float64 transformation matrix
        self.assertEqual(obs.T_w_e.dtype, np.float64)
        self.assertEqual(obs.T_w_e.shape, (4, 4))
        np.testing.assert_allclose(obs.T_w_e[:3, 3], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(obs.T_w_e[:3, :3], np.eye(3))

        # Gripper must be float 1.0
        self.assertEqual(obs.grip, 1.0)

    def test_controller_physics_stepping_and_lifecycle(self):
        """RLBenchTimedController satisfies single-physics-step and lifecycle methods."""
        mock_env = MockTaskEnv()
        outer = MockOuterEnv(mock_env)
        controller = RLBenchTimedController(mock_env, outer_env=outer, sim_api=MockSim(mock_env))

        controller.reset(seed=42)
        self.assertTrue(mock_env.reset_called)
        self.assertEqual(controller.status(), "ok")
        self.assertEqual(controller.simulator_time(), 0.0)
        controller.set_target(np.eye(4), 1)

        # Step two queried 0.05 s physical steps (0.1 s equivalent)
        for _ in range(2):
            controller.step_physics()

        self.assertEqual(mock_env.step_count, 2)
        self.assertAlmostEqual(controller.simulator_time(), 0.1, places=5)

        # Observation retrieval
        obs = controller.observe()
        self.assertEqual(obs.points.dtype, np.float64)
        self.assertEqual(obs.points.shape, (100, 3))

        controller.safe_hold()
        controller.close()
        self.assertEqual(outer.shutdown_calls, 1)

    def test_oracle_waypoint_interpolation_generates_bounded_timed_commands(self):
        """Waypoint discretization produces strictly bounded 0.1s TimedCommands."""
        start_T = np.eye(4, dtype=np.float64)
        start_T[:3, 3] = [0.0, 0.0, 1.0]

        end_T = np.eye(4, dtype=np.float64)
        end_T[:3, 3] = [0.0, 0.0, 1.01]

        waypoints = [
            (start_T, 1),
            (end_T, 0),
        ]
        commands = discretize_waypoints_to_commands(
            waypoints,
            interval_s=0.1,
            max_intervals=32,
            nominal_speed=0.05,
        )

        self.assertGreater(len(commands), 0)
        self.assertLess(len(commands), 32)
        for cmd in commands:
            self.assertEqual(cmd.duration_s, 0.1)
            self.assertEqual(cmd.target_w.shape, (4, 4))
            self.assertIn(cmd.grip, (0, 1))

    def test_uploader_drive_first_then_hf_with_verified_checksums(self):
        """Uploader verifies SHA-256 before writing and mirrors to Drive first."""
        from tests.test_rlbench_phase34 import _write_valid_test_episode

        drive_dir = self.root / "drive_mirror"
        local_dir = self.root / "local_data"
        _write_valid_test_episode(local_dir, "ep-test-01")

        manifest_file = local_dir / "dataset_manifest.json"
        manifest_file.write_text('{"dataset_track": "exploratory", "episodes": [{"episode_id": "ep-test-01"}]}', encoding="utf-8")

        uploader = ExploratoryUploader(local_dir, drive_root=drive_dir, hf_repo_id="org/repo", hf_client=MockHFClient())
        report = uploader.sync_all()

        # Check that file exists on Drive with identical SHA-256
        drive_manifest = drive_dir / "episodes" / "ep-test-01" / "manifest.json"
        self.assertTrue(drive_manifest.is_file())
        self.assertEqual(
            compute_file_sha256(local_dir / "episodes" / "ep-test-01" / "manifest.json"),
            compute_file_sha256(drive_manifest),
        )
        self.assertTrue(report["manifest"]["drive_synced"])

    def test_uploader_idempotency_and_resume_skips_matching_files(self):
        """Uploader skips files already present with identical SHA-256."""
        from tests.test_rlbench_phase34 import _write_valid_test_episode

        drive_dir = self.root / "drive_mirror"
        local_dir = self.root / "local_data"
        _write_valid_test_episode(local_dir, "ep-test-02")

        uploader = ExploratoryUploader(local_dir, drive_root=drive_dir, hf_repo_id="org/repo", hf_client=MockHFClient())
        # First sync
        uploader.sync_episode("ep-test-02")
        drive_file = drive_dir / "episodes" / "ep-test-02" / "manifest.json"
        self.assertTrue(drive_file.is_file())
        orig_mtime = drive_file.stat().st_mtime

        # Second sync: should skip writing to drive_file (idempotent resume)
        uploader.sync_episode("ep-test-02")
        self.assertEqual(drive_file.stat().st_mtime, orig_mtime)

    def test_uploader_rejects_credentials_and_source_snapshots(self):
        """Uploader rejects secret tokens, .env, and Python source files."""
        local_dir = self.root / "local_data"
        drive_dir = self.root / "drive_mirror"
        ep_dir = local_dir / "episodes" / "ep-sec"
        ep_dir.mkdir(parents=True)

        # Writing a forbidden .env file
        env_file = ep_dir / ".env"
        env_file.write_text("SECRET=12345", encoding="utf-8")

        uploader = ExploratoryUploader(local_dir, drive_root=drive_dir, hf_repo_id="org/repo", hf_client=MockHFClient())
        with self.assertRaisesRegex(ValueError, "Refusing to mirror forbidden file"):
            uploader.sync_episode("ep-sec")

class G1ReceiptGuardInvariantTests(unittest.TestCase):
    """Authority: ADR0012 + ADR0007.
    Scope: scripts/run_rlbench_exploratory.py E01 entrypoint.
    Rule: A valid PASS G1 receipt explicitly declaring PASS for the compatible pinned
    protocol is required before launching RLBench or writing any data.
    Allowed case: A valid PASS G1 receipt matching the pinned protocol.
    Forbidden case: Any real collection launch without that receipt; no exceptions.
    Diagnostic: Must name the rule owner ('ADR0012 and ADR0007') and say to run the
    bounded G1 probe first ('scripts/colab_g1_api_probe.py').
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_dir = self.root / "exploratory_output"
        self.mod = _load_script_module("run_rlbench_exploratory", "run_rlbench_exploratory.py")
        self.mod.G1_DRIVE_ROOT = self.root

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_valid_receipt(self, path: Path) -> Path:
        data = {
            "kind": "g1_api_probe_receipt",
            "receipt_version": 2,
            "probe_implementation_id": "g1-api-probe-v2",
            "status": "PASS",
            "authority": "ADR0012 and ADR0007",
            "protocol_id": "rlbench-g1-coppeliasim-4.1-franka",
            "task_name": "PickAndLift",
            "resolved_task_name": "pick_and_lift",
            "clock": {
                "dt_s": 0.05,
                "sim_time_before_s": 0.0,
                "sim_time_after_s": 0.1,
                "clock_advanced": True,
                "physics_steps": 2,
                "scene_steps": 2,
                "elapsed_s": 0.1,
            },
            "environment": {
                "outer_env_class": "Environment",
                "task_env_class": "PickAndLift",
            },
            "apis": {
                "outer_env_shutdown": True,
                "task_reset": True,
                "task_get_demos": True,
                "task_scene_step": True,
                "robot_arm_get_tip": True,
                "robot_arm_joint_positions": True,
                "robot_arm_ik": True,
                "robot_arm_set_joint_targets": True,
                "robot_gripper": True,
                "robot_gripper_actuate": True,
            },
            "target_hold": {
                "target_from_live_tip": True,
                "ik_solution_applied": True,
                "scene_steps": 2,
                "gripper_actuation_calls": 2,
                "target_hold_verified": True,
                "joint_target_error_linf_rad": 0.0,
                "joint_target_atol_rad": 0.01,
                "gripper_hold_target": 1.0,
                "gripper_hold_error_linf": 0.0,
                "gripper_hold_atol": 0.001,
            },
            "raw_cloud": {
                "points_found": True,
                "points_finite": True,
                "points_shape": [128, 128, 3],
                "points_dtype": "float32",
                "point_count": 16384,
            },
            "demonstrations": {
                "get_demos_supported": True,
                "live_demo_executed": False,
                "demo_count": 0,
            },
            "wall_time": {
                "max_s": 120.0,
                "elapsed_s": 1.0,
            },
            "upstream": {
                "rlbench": {
                    "revision": "02720bba4c73fe02eb75df946b8791b806028a9d",
                    "source_tree_clean": True,
                    "source_root": "/content/icgs-simulator/RLBench",
                },
                "pyrep": {
                    "revision": "8f420be8064b1970aae18a9cfbc978dfb15747ef",
                    "source_tree_clean": True,
                    "source_root": "/content/icgs-simulator/PyRep",
                },
                "coppeliasim": {
                    "version": "4.1.0",
                    "archive_sha256": "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8",
                    "simulator_root": "/content/icgs-simulator/CoppeliaSim",
                },
            },
            "cleanup": {
                "outer_env_shutdown": True,
            },
            "started_at_utc": "2026-09-16T00:00:00+00:00",
            "completed_at_utc": "2026-09-16T00:00:01+00:00",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def test_forbidden_without_receipt_names_owner_and_probe_and_writes_no_data(self):
        """Forbidden case: calling run_exploratory_collection without a receipt fails loudly before writing data."""
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=None,
                use_mock_for_testing=False,
            )
        err = str(ctx.exception)
        self.assertIn("ADR0012 and ADR0007", err)
        self.assertIn("scripts/colab_g1_api_probe.py", err)
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())
        self.assertFalse(self.output_dir.exists())

    def test_forbidden_with_nonexistent_receipt_file(self):
        """Forbidden case: nonexistent receipt path fails before writing data."""
        nonexistent = self.root / "missing_receipt.json"
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=nonexistent,
                use_mock_for_testing=False,
            )
        err = str(ctx.exception)
        self.assertIn("ADR0012 and ADR0007", err)
        self.assertIn("scripts/colab_g1_api_probe.py", err)
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())

    def test_forbidden_when_receipt_is_not_under_mounted_drive(self):
        """A syntactically valid receipt outside Drive cannot satisfy the G1 gate."""
        with tempfile.TemporaryDirectory() as outside_temp:
            receipt_path = Path(outside_temp) / "valid_receipt.json"
            self._create_valid_receipt(receipt_path)
            with self.assertRaises(RuntimeError) as ctx:
                self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("mounted google drive", str(ctx.exception).lower())

    def test_forbidden_with_failed_receipt_status(self):
        """Forbidden case: receipt with status != PASS fails loudly."""
        receipt_path = self.root / "fail_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["status"] = "FAIL"
        data["error"] = "Clock did not advance"
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        err = str(ctx.exception)
        self.assertIn("ADR0012 and ADR0007", err)
        self.assertIn("status='FAIL'", err)
        self.assertIn("scripts/colab_g1_api_probe.py", err)

    def test_forbidden_with_incompatible_protocol_id(self):
        """Forbidden case: receipt with wrong protocol fails loudly."""
        receipt_path = self.root / "wrong_proto_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["protocol_id"] = "unsupported-protocol-v99"
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        err = str(ctx.exception)
        self.assertIn("ADR0012 and ADR0007", err)
        self.assertIn("protocol mismatch", err)
        self.assertIn("scripts/colab_g1_api_probe.py", err)

    def test_forbidden_with_unadvancing_clock_receipt(self):
        """Forbidden case: receipt without valid advancing live clock fails loudly."""
        receipt_path = self.root / "stalled_clock_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["clock"]["clock_advanced"] = False
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        err = str(ctx.exception)
        self.assertIn("ADR0012 and ADR0007", err)
        self.assertIn("advancing live clock", err)
        self.assertIn("scripts/colab_g1_api_probe.py", err)

    def test_forbidden_with_minimally_forged_pass_receipt(self):
        """A status/protocol/boolean-only JSON must not unlock E01 collection."""
        receipt_path = self.root / "forged_receipt.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol_id": "rlbench-g1-coppeliasim-4.1-franka",
                    "clock": {"clock_advanced": True},
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("receipt kind", str(ctx.exception).lower())

    def test_forbidden_with_wrong_task_identity(self):
        """A G1 receipt for one upstream task must not unlock another task."""
        receipt_path = self.root / "wrong_task_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["task_name"] = "OpenDrawer"
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path, expected_task="PickAndLift")
        self.assertIn("task identity", str(ctx.exception).lower())

    def test_forbidden_without_target_hold_evidence(self):
        """A receipt must prove commanded Scene-step target hold, not a passive clock."""
        receipt_path = self.root / "no_target_hold_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        del data["target_hold"]
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("target hold", str(ctx.exception).lower())

    def test_forbidden_without_clean_shutdown_evidence(self):
        """A PASS receipt without a clean outer-environment shutdown is invalid."""
        receipt_path = self.root / "unclean_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["cleanup"]["outer_env_shutdown"] = False
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("clean shutdown", str(ctx.exception).lower())

    def test_forbidden_without_bounded_wall_time_evidence(self):
        """A PASS receipt must show that the bounded probe did not exceed its cap."""
        receipt_path = self.root / "unbounded_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        del data["wall_time"]
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("wall-time", str(ctx.exception).lower())

    def test_forbidden_with_inconsistent_two_step_clock_evidence(self):
        """Two actual scene steps and their measured elapsed time are mandatory."""
        receipt_path = self.root / "bad_step_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["clock"]["physics_steps"] = 1
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("two physical scene steps", str(ctx.exception).lower())

    def test_forbidden_with_wrong_pinned_upstream_revision(self):
        """A receipt from a different RLBench revision cannot unlock the pinned path."""
        receipt_path = self.root / "wrong_revision_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        data["upstream"]["rlbench"]["revision"] = "deadbeef"
        receipt_path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.verify_g1_receipt(receipt_path)
        self.assertIn("rlbench revision", str(ctx.exception).lower())

    def test_allowed_fixture_with_valid_pass_receipt(self):
        """Allowed case: valid PASS G1 receipt matching pinned protocol passes verification."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        data = self.mod.verify_g1_receipt(receipt_path)
        self.assertEqual(data["status"], "PASS")
        self.assertEqual(data["protocol_id"], "rlbench-g1-coppeliasim-4.1-franka")

    def test_phase_zero_refuses_mock_smoke_with_valid_receipt(self):
        """Mock quarantine/failure output is not an E01 collection success."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                num_attempts=1,
                max_intervals=4,
                use_mock_for_testing=True,
            )
        self.assertIn("mock collection", str(ctx.exception).lower())
        self.assertFalse(self.output_dir.exists())

    def test_exploratory_collection_requires_dual_publication_targets(self):
        """Phase 3-4 publication requires both mounted Drive root and HF dataset repo."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                use_mock_for_testing=False,
            )
        err = str(ctx.exception)
        self.assertIn("Phase 3-4 publication requires both", err)
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())

    def test_unavailable_drive_target_calls_env_factory_zero_times_and_writes_no_data(self):
        """Preflight rejects unavailable Drive root before any environment construction or data writes."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        nonexistent_drive = self.root / "does_not_exist_drive"
        hf_client = MockHFClient()

        env_factory_calls = []
        def _factory(spec):
            env_factory_calls.append(spec)
            return unittest.mock.MagicMock()

        with self.assertRaises(FileNotFoundError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=nonexistent_drive,
                hf_repo_id="test-org/test-dataset",
                hf_client=hf_client,
                environment_factory=_factory,
            )
        self.assertIn("Drive root does not exist", str(ctx.exception))
        self.assertEqual(len(env_factory_calls), 0)
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())
        self.assertEqual(list(self.output_dir.glob("episodes/*")), [])

    def test_unwritable_drive_target_calls_env_factory_zero_times_and_writes_no_data(self):
        """Preflight fails when Drive target write probe fails before constructing simulator."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "readonly_drive"
        drive_dir.mkdir(parents=True, exist_ok=True)
        # Make drive_dir non-writable
        drive_dir.chmod(0o444)
        hf_client = MockHFClient()

        env_factory_calls = []
        def _factory(spec):
            env_factory_calls.append(spec)
            return unittest.mock.MagicMock()

        try:
            with self.assertRaises((IOError, PermissionError)) as ctx:
                self.mod.run_exploratory_collection(
                    self.output_dir,
                    g1_receipt_path=receipt_path,
                    drive_dir=drive_dir,
                    hf_repo_id="test-org/test-dataset",
                    hf_client=hf_client,
                    environment_factory=_factory,
                )
            self.assertIn("Drive target write/read probe failed", str(ctx.exception))
            self.assertEqual(len(env_factory_calls), 0)
            self.assertFalse((self.output_dir / "dataset_manifest.json").exists())
            self.assertEqual(list(self.output_dir.glob("episodes/*")), [])
        finally:
            drive_dir.chmod(0o755)

    def test_unavailable_hf_target_calls_env_factory_zero_times_and_writes_no_data(self):
        """Preflight fails when HF target is unauthorized before constructing simulator."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        class FailingHFClient:
            def repo_info(self, *, repo_id, repo_type="dataset"):
                raise RuntimeError(f"401 Unauthorized for {repo_id}")

        env_factory_calls = []
        def _factory(spec):
            env_factory_calls.append(spec)
            return unittest.mock.MagicMock()

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/unauthorized-dataset",
                hf_client=FailingHFClient(),
                environment_factory=_factory,
            )
        self.assertIn("401 Unauthorized", str(ctx.exception))
        self.assertEqual(len(env_factory_calls), 0)
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())
        self.assertEqual(list(self.output_dir.glob("episodes/*")), [])

    def test_read_only_hf_token_calls_env_factory_zero_times_and_writes_no_data(self):
        """Preflight fails when HF token has read-only permission before constructing simulator."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        class ReadOnlyHFClient:
            def get_token_permission(self, token=None):
                return "read"
            def repo_info(self, *, repo_id, repo_type="dataset"):
                return unittest.mock.MagicMock()

        env_factory_calls = []
        def _factory(spec):
            env_factory_calls.append(spec)
            return unittest.mock.MagicMock()

        with self.assertRaises(PermissionError) as ctx:
            self.mod.run_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/read-only-dataset",
                hf_client=ReadOnlyHFClient(),
                environment_factory=_factory,
            )
        self.assertIn("read' permission only", str(ctx.exception))
        self.assertEqual(len(env_factory_calls), 0)
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())
        self.assertEqual(list(self.output_dir.glob("episodes/*")), [])

    def test_preflight_never_logs_secret_token(self):
        """Preflight never logs or leaks secret tokens into output report or log messages."""
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)
        secret_token = "hf_secret_super_private_token_xyz123"

        from icgs.data.collection.uploader import ExploratoryUploader
        uploader = ExploratoryUploader(
            self.output_dir,
            drive_root=drive_dir,
            hf_repo_id="test-org/test-dataset",
            hf_token=secret_token,
            hf_client=MockHFClient(),
        )

        with self.assertLogs("icgs.data.collection.uploader", level="DEBUG") as log_cm:
            report = uploader.verify_publication_preflight()

        self.assertEqual(report["drive_target"], "PASS")
        self.assertEqual(report["hf_target"], "PASS")
        # Ensure secret token is nowhere in report values
        report_str = json.dumps(report)
        self.assertNotIn(secret_token, report_str)
        for log_msg in log_cm.output:
            self.assertNotIn(secret_token, log_msg)

    def test_exploratory_collection_concrete_composition(self):
        """Entry point concretely composes specs, live expert command factory, limits, and publishers."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        hf_client = MockHFClient()

        class FakeObservation:
            def __init__(self, pos, grip):
                self.gripper_pose = [*pos, 0.0, 0.0, 0.0, 1.0]
                self.gripper_open = float(grip)

        class FakeDemo:
            def __init__(self):
                self._observations = [
                    FakeObservation([0.1, -0.2, 0.85], 1.0),
                    FakeObservation([0.1, -0.2, 0.85], 0.0),
                ]

        demos_queried = []
        def _get_demos(amount, live_demos):
            demos_queried.append((amount, live_demos))
            return [FakeDemo()]

        from icgs.environments.rlbench.timed import TimedRLBenchAdapter

        task = MockTaskEnv()
        task.get_demos = _get_demos
        task.reset_to_demo = lambda demo: None
        outer = MockOuterEnv(task)
        sim = MockSim(task)
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
        adapter = TimedRLBenchAdapter(controller, physics_dt=0.05, sensor_profile_id="rlbench-wrist-depth-raw-v1")

        published_count = self.mod.run_exploratory_collection(
            self.output_dir,
            g1_receipt_path=receipt_path,
            drive_dir=drive_dir,
            hf_repo_id="test-org/test-dataset",
            hf_client=hf_client,
            num_attempts=1,
            max_intervals=8,
            wall_time_s=60.0,
            environment_factory=lambda spec: adapter,
            # Notice: specs, command_factory, config, limits, protocol_manifest are all None
            # to verify full concrete composition!
        )
        self.assertEqual(published_count, 1)
        self.assertEqual(len(demos_queried), 1)
        self.assertTrue((self.output_dir / "dataset_manifest.json").exists())
        self.assertTrue((drive_dir / "dataset_manifest.json").exists())
        self.assertGreater(hf_client.commits, 0)
        self.assertIsNotNone(controller._last_demo)
        self.assertIsNotNone(controller._last_oracle)
        self.assertIsNotNone(controller._last_demo_hash)
        self.assertIsNotNone(controller._last_materialization_id)

    def test_make_concrete_environment_shuts_down_outer_env_on_launch_error(self):
        """Failure during launch must clean up outer environment exactly once with exception preserved."""
        mock_env = unittest.mock.MagicMock()
        mock_env.launch.side_effect = RuntimeError("launch-failed")

        with unittest.mock.patch.dict(
            "sys.modules",
            {
                "rlbench": unittest.mock.MagicMock(),
                "rlbench.action_modes": unittest.mock.MagicMock(),
                "rlbench.action_modes.action_mode": unittest.mock.MagicMock(),
                "rlbench.action_modes.arm_action_modes": unittest.mock.MagicMock(),
                "rlbench.action_modes.gripper_action_modes": unittest.mock.MagicMock(),
                "rlbench.environment": unittest.mock.MagicMock(Environment=unittest.mock.MagicMock(return_value=mock_env)),
                "rlbench.observation_config": unittest.mock.MagicMock(),
                "rlbench.tasks": unittest.mock.MagicMock(ReachTarget=unittest.mock.MagicMock()),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "launch-failed"):
                self.mod.make_concrete_rlbench_environment(task_name="ReachTarget")
            mock_env.shutdown.assert_called_once()

    def test_make_concrete_environment_shuts_down_outer_env_on_get_task_error(self):
        """Failure during get_task must clean up outer environment exactly once with exception preserved."""
        mock_env = unittest.mock.MagicMock()
        mock_env.get_task.side_effect = RuntimeError("get-task-failed")

        with unittest.mock.patch.dict(
            "sys.modules",
            {
                "rlbench": unittest.mock.MagicMock(),
                "rlbench.action_modes": unittest.mock.MagicMock(),
                "rlbench.action_modes.action_mode": unittest.mock.MagicMock(),
                "rlbench.action_modes.arm_action_modes": unittest.mock.MagicMock(),
                "rlbench.action_modes.gripper_action_modes": unittest.mock.MagicMock(),
                "rlbench.environment": unittest.mock.MagicMock(Environment=unittest.mock.MagicMock(return_value=mock_env)),
                "rlbench.observation_config": unittest.mock.MagicMock(),
                "rlbench.tasks": unittest.mock.MagicMock(ReachTarget=unittest.mock.MagicMock()),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "get-task-failed"):
                self.mod.make_concrete_rlbench_environment(task_name="ReachTarget")
            mock_env.shutdown.assert_called_once()

    def test_make_concrete_environment_shuts_down_outer_env_on_controller_error(self):
        """Failure during controller init must clean up outer environment exactly once with exception preserved."""
        mock_env = unittest.mock.MagicMock()
        mock_env.get_task.return_value = unittest.mock.MagicMock()

        with unittest.mock.patch.dict(
            "sys.modules",
            {
                "rlbench": unittest.mock.MagicMock(),
                "rlbench.action_modes": unittest.mock.MagicMock(),
                "rlbench.action_modes.action_mode": unittest.mock.MagicMock(),
                "rlbench.action_modes.arm_action_modes": unittest.mock.MagicMock(),
                "rlbench.action_modes.gripper_action_modes": unittest.mock.MagicMock(),
                "rlbench.environment": unittest.mock.MagicMock(Environment=unittest.mock.MagicMock(return_value=mock_env)),
                "rlbench.observation_config": unittest.mock.MagicMock(),
                "rlbench.tasks": unittest.mock.MagicMock(ReachTarget=unittest.mock.MagicMock()),
            },
        ), unittest.mock.patch(
            "icgs.environments.rlbench.controller.RLBenchTimedController",
            side_effect=RuntimeError("controller-init-failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "controller-init-failed"):
                self.mod.make_concrete_rlbench_environment(task_name="ReachTarget")
            mock_env.shutdown.assert_called_once()

    def test_parameter_validation_rejects_num_attempts_not_one(self):
        """E01 parameter validation fails closed BEFORE preflight/worker if num_attempts != 1."""
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(ValueError, "num_attempts == 1"):
            self.mod._validate_e01_parameters(
                num_attempts=2,
                max_intervals=8,
                wall_time_s=10.0,
            )
        with self.assertRaisesRegex(ValueError, "num_attempts == 1"):
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                num_attempts=2,
                max_intervals=8,
                wall_time_s=10.0,
            )
        with self.assertRaisesRegex(ValueError, "num_attempts == 1"):
            self.mod.run_exploratory_collection(
                self.output_dir,
                num_attempts=0,
                max_intervals=8,
                wall_time_s=10.0,
            )

    def test_parameter_validation_rejects_max_intervals_out_of_range(self):
        """E01 parameter validation fails closed if max_intervals is not an integer in 1..1024."""
        for bad_val in (0, 1025, 2000, -1, 4.5, "16"):
            with self.assertRaises(ValueError):
                self.mod._validate_e01_parameters(
                    num_attempts=1,
                    max_intervals=bad_val,  # type: ignore
                    wall_time_s=10.0,
                )

    def test_parameter_validation_rejects_non_finite_or_non_positive_caps(self):
        """E01 parameter validation fails closed on non-positive or non-finite wall/disk/supervisor limits."""
        with self.assertRaisesRegex(ValueError, "wall_time_s"):
            self.mod._validate_e01_parameters(num_attempts=1, max_intervals=8, wall_time_s=-1.0)
        with self.assertRaisesRegex(ValueError, "wall_time_s"):
            self.mod._validate_e01_parameters(num_attempts=1, max_intervals=8, wall_time_s=float("nan"))
        with self.assertRaisesRegex(ValueError, "supervisor_timeout_s"):
            self.mod._validate_e01_parameters(num_attempts=1, max_intervals=8, wall_time_s=10.0, supervisor_timeout_s=-0.5)
        with self.assertRaisesRegex(ValueError, "disk_limit_bytes"):
            self.mod._validate_e01_parameters(num_attempts=1, max_intervals=8, wall_time_s=10.0, disk_limit_bytes=0)

    def test_cli_worker_flag_enforces_parameter_validation(self):
        """CLI --worker flag cannot bypass strict parameter validation."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        script_path = Path(self.mod.__file__).resolve()
        res = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--worker",
                "--output-dir",
                str(self.output_dir),
                "--g1-receipt",
                str(receipt_path),
                "--num-attempts",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("num_attempts == 1", res.stderr)

    def test_supervisor_normal_success_with_verified_worker_result(self):
        """Supervisor succeeds when worker exits 0 and writes valid, nonce-matched worker result."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        worker_code = (
            "import sys, json, argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--result-file')\n"
            "parser.add_argument('--nonce')\n"
            "args, _ = parser.parse_known_args()\n"
            "if args.result_file and args.nonce:\n"
            "    data = {\n"
            "        'schema_version': 'icgs_worker_result_v1',\n"
            "        'kind': 'worker_result',\n"
            "        'nonce': args.nonce,\n"
            "        'terminal_status': 'completed',\n"
            "        'episodes_published': 1,\n"
            "        'manifest_sha256': 'a' * 64,\n"
            "        'index_publication': {'drive_verified': True, 'hf_verified': True, 'index_commit': 'c1', 'manifest_sha256': 'a' * 64},\n"
            "        'attempt_records': [{'episode_id': 'ep0', 'status': 'published', 'transitions': 8}],\n"
            "        'cleanup': 'COMPLETED',\n"
            "    }\n"
            "    with open(args.result_file, 'w') as f:\n"
            "        json.dump(data, f)\n"
            "sys.exit(0)\n"
        )
        worker_script = [sys.executable, "-c", worker_code]

        outcome = self.mod.run_supervised_exploratory_collection(
            self.output_dir,
            g1_receipt_path=receipt_path,
            drive_dir=drive_dir,
            hf_repo_id="test-org/test-dataset",
            wall_time_s=10.0,
            worker_cmd=worker_script,
        )
        self.assertEqual(outcome["status"], "SUCCESS")
        self.assertFalse(outcome["forced_timeout"])
        self.assertEqual(outcome["exit_code"], 0)
        self.assertEqual(outcome["cleanup"], "COMPLETED")
        self.assertEqual(outcome["outer_env_shutdown"], "COMPLETED")
        self.assertEqual(outcome["episodes_published"], 1)
        self.assertTrue(outcome["index_verified"])

        local_outcome = self.output_dir / "supervisor-outcome.json"
        drive_outcome = drive_dir / "supervisor-outcome.json"
        self.assertTrue(local_outcome.exists())
        self.assertTrue(drive_outcome.exists())
        saved = json.loads(local_outcome.read_text())
        self.assertEqual(saved["status"], "SUCCESS")

    def test_supervisor_fails_closed_if_worker_exits_zero_without_result(self):
        """Worker exiting 0 without writing structured worker-result JSON fails closed."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        # Worker exits 0 but does not write result file
        worker_script = [sys.executable, "-c", "import sys; sys.exit(0)"]

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/test-dataset",
                wall_time_s=10.0,
                worker_cmd=worker_script,
            )
        self.assertIn("result file is missing or invalid", str(ctx.exception))
        local_outcome = self.output_dir / "supervisor-outcome.json"
        saved = json.loads(local_outcome.read_text())
        self.assertEqual(saved["status"], "WORKER_RESULT_INVALID")
        self.assertEqual(saved["cleanup"], "UNKNOWN")

    def test_supervisor_fails_closed_if_worker_result_nonce_mismatched(self):
        """Worker result with stale/forged nonce fails closed."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        worker_code = (
            "import sys, json, argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--result-file')\n"
            "args, _ = parser.parse_known_args()\n"
            "if args.result_file:\n"
            "    data = {\n"
            "        'schema_version': 'icgs_worker_result_v1',\n"
            "        'kind': 'worker_result',\n"
            "        'nonce': 'forged_nonce_123',\n"
            "        'terminal_status': 'completed',\n"
            "        'episodes_published': 1,\n"
            "        'manifest_sha256': 'a' * 64,\n"
            "        'index_publication': {'drive_verified': True, 'hf_verified': True, 'index_commit': 'c1', 'manifest_sha256': 'a' * 64},\n"
            "        'cleanup': 'COMPLETED',\n"
            "    }\n"
            "    with open(args.result_file, 'w') as f:\n"
            "        json.dump(data, f)\n"
            "sys.exit(0)\n"
        )
        worker_script = [sys.executable, "-c", worker_code]

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/test-dataset",
                wall_time_s=10.0,
                worker_cmd=worker_script,
            )
        self.assertIn("nonce mismatch", str(ctx.exception))

    def test_supervisor_fails_closed_if_worker_result_unverified_index_or_zero_episodes(self):
        """Worker result claiming 0 episodes or unverified index fails closed."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        worker_code = (
            "import sys, json, argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--result-file')\n"
            "parser.add_argument('--nonce')\n"
            "args, _ = parser.parse_known_args()\n"
            "if args.result_file and args.nonce:\n"
            "    data = {\n"
            "        'schema_version': 'icgs_worker_result_v1',\n"
            "        'kind': 'worker_result',\n"
            "        'nonce': args.nonce,\n"
            "        'terminal_status': 'completed',\n"
            "        'episodes_published': 0,\n"
            "        'manifest_sha256': 'a' * 64,\n"
            "        'index_publication': {'drive_verified': False, 'hf_verified': False},\n"
            "        'cleanup': 'COMPLETED',\n"
            "    }\n"
            "    with open(args.result_file, 'w') as f:\n"
            "        json.dump(data, f)\n"
            "sys.exit(0)\n"
        )
        worker_script = [sys.executable, "-c", worker_code]

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/test-dataset",
                wall_time_s=10.0,
                worker_cmd=worker_script,
            )
        self.assertIn("did not publish exactly one episode", str(ctx.exception))

    def test_supervisor_pipe_deadlock_prevention_large_stdout_succeeds(self):
        """Child writing > pipe capacity (128KB+) to stdout does not deadlock and supervisor succeeds."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        worker_code = (
            "import sys, json, argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--result-file')\n"
            "parser.add_argument('--nonce')\n"
            "args, _ = parser.parse_known_args()\n"
            "# Write 256KB to stdout (exceeds default 64KB OS pipe buffer)\n"
            "sys.stdout.write('x' * (256 * 1024))\n"
            "sys.stdout.flush()\n"
            "if args.result_file and args.nonce:\n"
            "    data = {\n"
            "        'schema_version': 'icgs_worker_result_v1',\n"
            "        'kind': 'worker_result',\n"
            "        'nonce': args.nonce,\n"
            "        'terminal_status': 'completed',\n"
            "        'episodes_published': 1,\n"
            "        'manifest_sha256': 'a' * 64,\n"
            "        'index_publication': {'drive_verified': True, 'hf_verified': True, 'index_commit': 'c1', 'manifest_sha256': 'a' * 64},\n"
            "        'attempt_records': [{'episode_id': 'ep0', 'status': 'published', 'transitions': 8}],\n"
            "        'cleanup': 'COMPLETED',\n"
            "    }\n"
            "    with open(args.result_file, 'w') as f:\n"
            "        json.dump(data, f)\n"
            "sys.exit(0)\n"
        )
        worker_script = [sys.executable, "-c", worker_code]

        outcome = self.mod.run_supervised_exploratory_collection(
            self.output_dir,
            g1_receipt_path=receipt_path,
            drive_dir=drive_dir,
            hf_repo_id="test-org/test-dataset",
            wall_time_s=10.0,
            worker_cmd=worker_script,
        )
        self.assertEqual(outcome["status"], "SUCCESS")
        # Tail memory is bounded
        self.assertLessEqual(len(outcome["stdout_tail"]), 4096)

    def test_supervisor_hanging_noisy_child_terminates_group_and_captures_bounded_tail(self):
        """Hanging noisy child writing continuous output is forcefully killed without pipe deadlock."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        worker_code = (
            "import subprocess, sys, time, signal\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(100)'])\n"
            "for _ in range(1000):\n"
            "    sys.stdout.write('noisy continuous stream chunk\\n')\n"
            "    sys.stdout.flush()\n"
            "time.sleep(100)\n"
        )
        worker_script = [sys.executable, "-c", worker_code]

        with self.assertRaises(TimeoutError) as ctx:
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/test-dataset",
                wall_time_s=0.2,
                supervisor_timeout_s=0.2,
                worker_cmd=worker_script,
                kill_grace_s=0.2,
            )
        self.assertIn("Supervisor hard wall timeout", str(ctx.exception))
        local_outcome = self.output_dir / "supervisor-outcome.json"
        saved = json.loads(local_outcome.read_text())
        self.assertEqual(saved["status"], "FORCED_TIMEOUT")
        self.assertEqual(saved["cleanup"], "UNKNOWN")
        self.assertEqual(saved["outer_env_shutdown"], "UNKNOWN")
        self.assertFalse(saved["surviving_children"])
        self.assertLessEqual(len(saved["stdout_tail"]), 4096)

    def test_supervisor_nonzero_worker(self):
        """Supervisor handles nonzero worker exit, writes WORKER_FAILED outcome with cleanup UNKNOWN."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        worker_script = [
            sys.executable,
            "-c",
            "import sys; sys.exit(42)",
        ]

        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                g1_receipt_path=receipt_path,
                drive_dir=drive_dir,
                hf_repo_id="test-org/test-dataset",
                wall_time_s=10.0,
                worker_cmd=worker_script,
            )
        self.assertIn("42", str(ctx.exception))

        local_outcome = self.output_dir / "supervisor-outcome.json"
        drive_outcome = drive_dir / "supervisor-outcome.json"
        self.assertTrue(local_outcome.exists())
        self.assertTrue(drive_outcome.exists())
        saved = json.loads(local_outcome.read_text())
        self.assertEqual(saved["status"], "WORKER_FAILED")
        self.assertEqual(saved["exit_code"], 42)
        self.assertFalse(saved["forced_timeout"])
        self.assertEqual(saved["cleanup"], "UNKNOWN")
        self.assertEqual(saved["outer_env_shutdown"], "UNKNOWN")

    def test_supervisor_avoids_shell_injection(self):
        """Worker command arguments are passed without shell=True, preventing injection."""
        receipt_path = self.root / "valid_receipt.json"
        self._create_valid_receipt(receipt_path)
        drive_dir = self.root / "drive"
        drive_dir.mkdir(parents=True, exist_ok=True)

        canary_file = self.root / "injected_canary.txt"
        malicious_arg = f"; touch {canary_file}"
        worker_code = (
            "import sys, json, argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--result-file')\n"
            "parser.add_argument('--nonce')\n"
            "args, _ = parser.parse_known_args()\n"
            "if args.result_file and args.nonce:\n"
            "    data = {\n"
            "        'schema_version': 'icgs_worker_result_v1',\n"
            "        'kind': 'worker_result',\n"
            "        'nonce': args.nonce,\n"
            "        'terminal_status': 'completed',\n"
            "        'episodes_published': 1,\n"
            "        'manifest_sha256': 'a' * 64,\n"
            "        'index_publication': {'drive_verified': True, 'hf_verified': True, 'index_commit': 'c1', 'manifest_sha256': 'a' * 64},\n"
            "        'attempt_records': [{'episode_id': 'ep0', 'status': 'published', 'transitions': 8}],\n"
            "        'cleanup': 'COMPLETED',\n"
            "    }\n"
            "    with open(args.result_file, 'w') as f:\n"
            "        json.dump(data, f)\n"
            "sys.exit(0)\n"
        )
        worker_script = [
            sys.executable,
            "-c",
            worker_code,
            malicious_arg,
        ]

        outcome = self.mod.run_supervised_exploratory_collection(
            self.output_dir,
            g1_receipt_path=receipt_path,
            drive_dir=drive_dir,
            hf_repo_id="test-org/test-dataset",
            wall_time_s=10.0,
            worker_cmd=worker_script,
        )
        self.assertEqual(outcome["status"], "SUCCESS")
        self.assertFalse(canary_file.exists(), "Shell injection vulnerability detected!")

    def test_supervisor_refuses_without_valid_g1_receipt(self):
        """Supervisor refuses to launch subprocess without a valid strict G1 receipt."""
        invalid_receipt = self.root / "nonexistent_receipt.json"
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.run_supervised_exploratory_collection(
                self.output_dir,
                g1_receipt_path=invalid_receipt,
                wall_time_s=10.0,
            )
        self.assertIn("Receipt file does not exist", str(ctx.exception))

    def test_secure_hf_token_file_validation(self):
        """HF token file must be a regular non-symlink non-empty file and token is never logged."""
        # Non-existent file
        with self.assertRaises(ValueError):
            self.mod._read_secure_hf_token_file(self.root / "nonexistent_token")

        # Empty file
        empty_file = self.root / "empty_token"
        empty_file.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.mod._read_secure_hf_token_file(empty_file)

        # Symlink file
        real_file = self.root / "real_token"
        real_file.write_text("hf_valid_token_value_123", encoding="utf-8")
        symlink_file = self.root / "symlink_token"
        try:
            symlink_file.symlink_to(real_file)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.mod._read_secure_hf_token_file(symlink_file)
        except OSError:
            pass

        # Valid file
        token = self.mod._read_secure_hf_token_file(real_file)
        self.assertEqual(token, "hf_valid_token_value_123")
        del token

    def test_supervisor_constrains_drive_dir(self):
        """drive_dir cannot be a symlink or root."""
        with self.assertRaises(ValueError):
            self.mod._validate_drive_dir_constraint(Path("/"))
        symlink_dir = self.root / "symlink_drive"
        target_dir = self.root / "real_drive"
        target_dir.mkdir()
        try:
            symlink_dir.symlink_to(target_dir)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.mod._validate_drive_dir_constraint(symlink_dir)
        except OSError:
            pass

    def test_durable_outcome_atomic_fsync_and_failure_reporting(self):
        """_write_durable_outcome performs atomic write with fsync and reports failures loudly."""
        local_path = self.output_dir / "outcome.json"
        drive_path = self.root / "drive" / "outcome.json"
        payload = {"status": "SUCCESS", "cleanup": "COMPLETED"}
        self.mod._write_durable_outcome(payload, local_path, drive_path)
        self.assertTrue(local_path.is_file())
        self.assertTrue(drive_path.is_file())
        self.assertEqual(json.loads(local_path.read_text())["status"], "SUCCESS")

        # Failure reporting if unwritable
        bad_local = Path("/nonexistent_forbidden_dir/sub/outcome.json")
        with self.assertRaises(IOError):
            self.mod._write_durable_outcome(payload, bad_local, None)


class ColabG1ApiProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.drive_root = self.root / "mounted_drive"
        self.drive_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.drive_root / "drive_output"
        self.probe_mod = _load_script_module("colab_g1_api_probe", "colab_g1_api_probe.py")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _pinned_upstream_evidence(self):
        return {
            "rlbench": {
                "revision": "02720bba4c73fe02eb75df946b8791b806028a9d",
                "source_tree_clean": True,
                "source_root": "/content/icgs-simulator/RLBench",
            },
            "pyrep": {
                "revision": "8f420be8064b1970aae18a9cfbc978dfb15747ef",
                "source_tree_clean": True,
                "source_root": "/content/icgs-simulator/PyRep",
            },
            "coppeliasim": {
                "version": "4.1.0",
                "archive_sha256": "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8",
                "simulator_root": "/content/icgs-simulator/CoppeliaSim",
            },
        }

    def _probe(self, *args, **kwargs):
        with patch.object(self.probe_mod, "DRIVE_ROOT", self.drive_root, create=True), patch.object(
            self.probe_mod,
            "_collect_pinned_upstream_evidence",
            return_value=self._pinned_upstream_evidence(),
            create=True,
        ):
            return self.probe_mod.probe_g1_environment(*args, **kwargs)

    def _make_git_source(self, path: Path) -> str:
        path.mkdir(parents=True)
        (path / "tracked.txt").write_text("pinned source\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=ICGS Test",
                "-c",
                "user.email=icgs-test@example.invalid",
                "commit",
                "-q",
                "-m",
                "pin source",
            ],
            check=True,
        )
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    def _make_mock_environment(
        self,
        *,
        dt=0.05,
        advance_time=True,
        missing_api=None,
        shutdown_error: Exception | None = None,
        get_task_error: Exception | None = None,
    ):
        outer_env = MagicMock()
        outer_env.__class__.__name__ = "Environment"
        task_env = MagicMock()
        task_env.__class__.__name__ = "PickAndLift"
        outer_env.get_task.return_value = task_env
        task_env.get_name.return_value = "pick_and_lift"
        if get_task_error is not None:
            outer_env.get_task.side_effect = get_task_error
        if shutdown_error is not None:
            outer_env.shutdown.side_effect = shutdown_error

        sim_api = MagicMock()
        sim_times = [0.0]

        def get_sim_time():
            return sim_times[-1]

        sim_api.simGetSimulationTime.side_effect = get_sim_time

        pyrep = MagicMock()
        pyrep.get_simulation_timestep.return_value = dt

        def step():
            if advance_time:
                sim_times.append(sim_times[-1] + dt)

        pyrep.step.side_effect = step
        outer_env._pyrep = pyrep

        pts = np.ones((128, 128, 3), dtype=np.float32)
        task_env.reset.return_value = (
            ["pinched task description"],
            SimpleNamespace(wrist_point_cloud=pts, gripper_open=1.0),
        )

        scene = MagicMock(spec=["step"])
        scene.step.side_effect = step
        task_env._scene = scene

        tip = MagicMock(spec=["get_pose"])
        tip.get_pose.return_value = np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
        arm = MagicMock(
            spec=[
                "get_joint_positions",
                "get_tip",
                "solve_ik_via_jacobian",
                "set_joint_target_positions",
            ]
        )
        arm.get_joint_positions.return_value = [0.0] * 7
        arm.get_tip.return_value = tip
        arm.solve_ik_via_jacobian.return_value = [0.0] * 7
        gripper = MagicMock(spec=["get_open_amount", "actuate"])
        gripper.get_open_amount.return_value = [1.0, 1.0]
        gripper.actuate.return_value = True

        robot = MagicMock()
        robot.arm = arm
        robot.gripper = gripper
        task_env._robot = robot
        outer_env._robot = robot

        if missing_api == "shutdown":
            del outer_env.shutdown
        elif missing_api == "solve_ik":
            del arm.solve_ik_via_jacobian
        elif missing_api == "get_demos":
            del task_env.get_demos

        return outer_env, task_env, pyrep, sim_api, scene, arm, gripper

    def test_probe_requires_drive_output_directory(self):
        """Probe fails loudly if output directory is missing or None."""
        with self.assertRaises(ValueError):
            self.probe_mod.verify_drive_directory(None)

    def test_probe_rejects_local_and_prefix_sibling_output_directories(self):
        """Only a real descendant of the mounted Drive root may receive a receipt."""
        sibling = self.root / "mounted_drive_evil" / "g1"
        local = self.root / "not_drive" / "g1"
        with patch.object(self.probe_mod, "DRIVE_ROOT", self.drive_root, create=True):
            for forbidden in (sibling, local):
                with self.subTest(forbidden=forbidden), self.assertRaises(RuntimeError):
                    self.probe_mod.verify_drive_directory(forbidden)
        self.assertFalse(sibling.exists())
        self.assertFalse(local.exists())

    def test_probe_makes_no_episode_archive_or_index(self):
        """Probe writes only receipt and does NOT make episode archives or indexes."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment()
        receipt = self._probe(
            self.output_dir,
            environment_instance=outer_env,
            sim_api=sim_api,
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue((self.output_dir / "g1-receipt.json").is_file())

        # Absolutely no episode artifacts or dataset manifests
        self.assertFalse((self.output_dir / "episodes").exists())
        self.assertFalse((self.output_dir / "dataset_manifest.json").exists())
        self.assertFalse((self.output_dir / "manifest.json").exists())
        self.assertEqual(list(self.output_dir.glob("*.npz")), [])

    def test_probe_never_reads_env_or_tokens(self):
        """Script must never import dotenv/huggingface_hub or reference secret tokens."""
        script_path = SCRIPTS_DIR / "colab_g1_api_probe.py"
        content = script_path.read_text(encoding="utf-8")
        self.assertNotIn("huggingface_hub", content)
        self.assertNotIn("dotenv", content)
        self.assertNotIn("HF_ACCESS_TOKEN", content)
        self.assertNotIn("HF_TOKEN", content)
        self.assertNotIn("WANDB_API_KEY", content)

    def test_probe_fails_loudly_on_missing_expected_api(self):
        """Probe fails loudly when an expected API is missing on the environment/arm."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment(missing_api="solve_ik")
        with self.assertRaises(RuntimeError) as ctx:
            self._probe(
                self.output_dir,
                environment_instance=outer_env,
                sim_api=sim_api,
            )
        self.assertIn("Missing expected", str(ctx.exception))

    def test_probe_fails_loudly_on_unchanging_clock(self):
        """Probe fails loudly when live clock does not advance."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment(advance_time=False)
        with self.assertRaises(RuntimeError) as ctx:
            self._probe(
                self.output_dir,
                environment_instance=outer_env,
                sim_api=sim_api,
            )
        self.assertIn("clock failed to advance", str(ctx.exception))

    def test_probe_rejects_nonfloating_raw_cloud(self):
        """A PASS candidate must use the same raw floating-cloud contract as its verifier."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment()
        task_env.reset.return_value = (
            ["description"],
            SimpleNamespace(wrist_point_cloud=np.ones((8, 8, 3), dtype=np.int32), gripper_open=1.0),
        )
        with self.assertRaisesRegex(RuntimeError, "floating dtype"):
            self._probe(self.output_dir, environment_instance=outer_env, sim_api=sim_api)

    def test_probe_rejects_ambient_module_outside_verified_source_tree(self):
        """Pinned source metadata cannot certify a different imported package."""
        source_root = self.root / "pinned_pyrep"
        source_root.mkdir()
        ambient = SimpleNamespace(__file__=str(self.root / "ambient" / "pyrep" / "__init__.py"))
        with self.assertRaisesRegex(RuntimeError, "does not resolve beneath"):
            self.probe_mod._require_module_origin("pyrep", ambient, source_root)

    def test_probe_binds_cached_ambient_imports_to_verified_source_trees(self):
        """Cached ambient packages and submodules are purged before pinned imports resolve."""
        provision_root = self.root / "provision"
        pyrep_package = provision_root / "PyRep" / "pyrep"
        rlbench_package = provision_root / "RLBench" / "rlbench"
        pyrep_package.mkdir(parents=True)
        rlbench_package.mkdir(parents=True)
        (pyrep_package / "__init__.py").write_text("PINNED = 'pyrep'\n", encoding="utf-8")
        (rlbench_package / "__init__.py").write_text("PINNED = 'rlbench'\n", encoding="utf-8")
        ambient = SimpleNamespace(__file__=str(self.root / "ambient" / "pyrep" / "__init__.py"))
        ambient_submodule = SimpleNamespace(__file__=str(self.root / "ambient" / "pyrep" / "backend.py"))
        original_sys_path = sys.path[:]
        try:
            with patch.object(self.probe_mod, "PROVISION_ROOT", provision_root), patch.dict(
                sys.modules,
                {"pyrep": ambient, "pyrep.backend": ambient_submodule, "rlbench": ambient},
            ):
                source_roots = self.probe_mod._bind_pinned_imports()
                self.assertEqual(Path(sys.modules["pyrep"].__file__).resolve().parent, pyrep_package.resolve())
                self.assertEqual(Path(sys.modules["rlbench"].__file__).resolve().parent, rlbench_package.resolve())
                self.assertEqual(source_roots["pyrep"], (provision_root / "PyRep").resolve())
                self.assertNotIn("pyrep.backend", sys.modules)
        finally:
            sys.path[:] = original_sys_path

    def test_probe_collects_archive_and_git_evidence_from_real_temporary_sources(self):
        """Pinned evidence rejects provenance fiction by reading actual archive and Git state."""
        provision_root = self.root / "provision"
        pyrep_root = provision_root / "PyRep"
        rlbench_root = provision_root / "RLBench"
        pyrep_revision = self._make_git_source(pyrep_root)
        rlbench_revision = self._make_git_source(rlbench_root)
        simulator_root = provision_root / "CoppeliaSim"
        simulator_root.mkdir()
        archive = provision_root / self.probe_mod.PINNED_COPPELIASIM_ARCHIVE_NAME
        archive.write_bytes(b"tiny pinned archive fixture")
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        with patch.object(self.probe_mod, "PROVISION_ROOT", provision_root), patch.object(
            self.probe_mod, "PINNED_PYREP_REVISION", pyrep_revision
        ), patch.object(self.probe_mod, "PINNED_RLBENCH_REVISION", rlbench_revision), patch.object(
            self.probe_mod, "PINNED_COPPELIASIM_ARCHIVE_SHA256", archive_digest
        ), patch.dict(os.environ, {"COPPELIASIM_ROOT": str(simulator_root)}, clear=False):
            evidence = self.probe_mod._collect_pinned_upstream_evidence()

        self.assertEqual(evidence["pyrep"]["revision"], pyrep_revision)
        self.assertEqual(evidence["rlbench"]["revision"], rlbench_revision)
        self.assertEqual(evidence["coppeliasim"]["archive_sha256"], archive_digest)

    def test_probe_rejects_dirty_pinned_source_tree(self):
        """An editable-build source tree with unexpected changes cannot issue PASS evidence."""
        provision_root = self.root / "provision"
        pyrep_root = provision_root / "PyRep"
        rlbench_root = provision_root / "RLBench"
        pyrep_revision = self._make_git_source(pyrep_root)
        rlbench_revision = self._make_git_source(rlbench_root)
        (pyrep_root / "unexpected.patch").write_text("not pinned\n", encoding="utf-8")
        simulator_root = provision_root / "CoppeliaSim"
        simulator_root.mkdir()
        archive = provision_root / self.probe_mod.PINNED_COPPELIASIM_ARCHIVE_NAME
        archive.write_bytes(b"tiny pinned archive fixture")
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        with patch.object(self.probe_mod, "PROVISION_ROOT", provision_root), patch.object(
            self.probe_mod, "PINNED_PYREP_REVISION", pyrep_revision
        ), patch.object(self.probe_mod, "PINNED_RLBENCH_REVISION", rlbench_revision), patch.object(
            self.probe_mod, "PINNED_COPPELIASIM_ARCHIVE_SHA256", archive_digest
        ), patch.dict(os.environ, {"COPPELIASIM_ROOT": str(simulator_root)}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "source tree is dirty"):
                self.probe_mod._collect_pinned_upstream_evidence()

    def test_probe_rejects_nonpositive_wall_time_before_creating_a_receipt(self):
        """The bounded probe rejects an invalid deadline before any Drive-side artifact exists."""
        with self.assertRaisesRegex(ValueError, "max_wall_time_s"):
            self._probe(self.output_dir, max_wall_time_s=0.0)
        self.assertFalse(self.output_dir.exists())

    def test_colab_provisioning_pins_and_editably_installs_the_executed_source_trees(self):
        """Future G1 setup must bind imports to the exact source commits the probe verifies."""
        environment_mod = _load_script_module("colab_data_environment", "colab_data_environment.py")
        self.assertEqual(
            environment_mod.UPSTREAM_REVISIONS["RLBench"],
            "02720bba4c73fe02eb75df946b8791b806028a9d",
        )
        self.assertEqual(
            environment_mod.UPSTREAM_REVISIONS["PyRep"],
            "8f420be8064b1970aae18a9cfbc978dfb15747ef",
        )
        self.assertEqual(
            environment_mod.PINNED_COPPELIASIM_ARCHIVE_SHA256,
            "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8",
        )
        commands = environment_mod.pinned_source_commands("RLBench", self.root / "RLBench")
        self.assertIn(
            [
                "git",
                "-C",
                str(self.root / "RLBench"),
                "fetch",
                "--depth",
                "1",
                "origin",
                "02720bba4c73fe02eb75df946b8791b806028a9d",
            ],
            commands,
        )
        self.assertEqual(
            environment_mod.editable_install_command("/content/icgs-data-env/bin/python", self.root / "RLBench"),
            [
                "uv",
                "pip",
                "install",
                "--python",
                "/content/icgs-data-env/bin/python",
                "--editable",
                str(self.root / "RLBench"),
            ],
        )

    def test_probe_cleanly_shuts_down_outer_environment_on_failure(self):
        """Outer Environment shutdown() must be called even on failure."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment(advance_time=False)
        with self.assertRaises(RuntimeError):
            self._probe(
                self.output_dir,
                environment_instance=outer_env,
                sim_api=sim_api,
            )
        outer_env.shutdown.assert_called_once()

    def test_probe_handles_the_pinned_reset_tuple_and_executes_target_hold(self):
        """Pinned reset output drives IK, joint targets, gripper hold, and Scene.step()."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment()
        receipt = self._probe(
            self.output_dir,
            environment_instance=outer_env,
            sim_api=sim_api,
        )

        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["clock"]["physics_steps"], 2)
        self.assertTrue(receipt["target_hold"]["target_hold_verified"])
        self.assertEqual(receipt["target_hold"]["gripper_hold_error_linf"], 0.0)
        arm.get_tip.assert_called_once_with()
        arm.solve_ik_via_jacobian.assert_called_once_with(
            [0.0, 0.0, 0.9], quaternion=[0.0, 0.0, 0.0, 1.0]
        )
        arm.set_joint_target_positions.assert_called_once_with([0.0] * 7)
        self.assertEqual(gripper.actuate.call_count, 2)
        self.assertEqual(scene.step.call_count, 2)
        self.assertEqual(pyrep.step.call_count, 0)

    def test_raw_wrist_probe_configuration_enables_depth_capture(self):
        """Pinned Scene only creates a point cloud when RGB or depth capture is enabled."""
        config = SimpleNamespace(
            wrist_camera=SimpleNamespace(point_cloud=False, depth=False),
            gripper_pose=False,
            gripper_open=False,
            set_all_high_dim=MagicMock(),
            set_all_low_dim=MagicMock(),
        )

        self.probe_mod._configure_raw_wrist_observation(config)

        config.set_all_high_dim.assert_called_once_with(False)
        config.set_all_low_dim.assert_called_once_with(False)
        self.assertTrue(config.wrist_camera.point_cloud)
        self.assertTrue(config.wrist_camera.depth)
        self.assertTrue(config.gripper_pose)
        self.assertTrue(config.gripper_open)

    def test_probe_replaces_a_prior_pass_with_fail_when_shutdown_fails(self):
        """A shutdown error leaves a durable FAIL receipt, never a stale PASS."""
        self.output_dir.mkdir()
        receipt_path = self.output_dir / "g1-receipt.json"
        receipt_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment(
            shutdown_error=RuntimeError("shutdown failed")
        )

        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            self._probe(
                self.output_dir,
                environment_instance=outer_env,
                sim_api=sim_api,
            )

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "FAIL")
        self.assertFalse(receipt["cleanup"]["outer_env_shutdown"])

    def test_probe_cleans_up_and_writes_fail_when_task_resolution_fails(self):
        """Task resolution is inside the protected lifecycle and cannot leak CoppeliaSim."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment(
            get_task_error=RuntimeError("task resolution failed")
        )
        with self.assertRaisesRegex(RuntimeError, "task resolution failed"):
            self._probe(
                self.output_dir,
                environment_instance=outer_env,
                sim_api=sim_api,
            )
        outer_env.shutdown.assert_called_once_with()
        receipt = json.loads((self.output_dir / "g1-receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "FAIL")

    def test_probe_live_demo_opt_in_only(self):
        """Live demo execution is opt-in, not default."""
        outer_env, task_env, pyrep, sim_api, scene, arm, gripper = self._make_mock_environment()
        # Default run_live_demo=False
        receipt = self._probe(
            self.output_dir,
            environment_instance=outer_env,
            sim_api=sim_api,
            run_live_demo=False,
        )
        task_env.get_demos.assert_not_called()
        self.assertFalse(receipt["demonstrations"]["live_demo_executed"])

        # Opt-in run_live_demo=True
        outer_env2, task_env2, pyrep2, sim_api2, scene2, arm2, gripper2 = self._make_mock_environment()
        task_env2.get_demos.return_value = [MagicMock()]
        receipt2 = self._probe(
            self.output_dir,
            environment_instance=outer_env2,
            sim_api=sim_api2,
            run_live_demo=True,
        )
        task_env2.get_demos.assert_called_once_with(1, live_demos=True)
        self.assertTrue(receipt2["demonstrations"]["live_demo_executed"])

    def test_probe_cli_rejects_the_removed_local_directory_bypass(self):
        """The production CLI has no switch that redirects a G1 receipt off Drive."""
        with patch.object(sys, "argv", ["colab_g1_api_probe.py", "--allow-local-dir"]), patch.object(
            self.probe_mod,
            "probe_g1_environment",
            side_effect=AssertionError("probe must not run when argument parsing rejects the bypass"),
        ):
            with self.assertRaises(SystemExit) as ctx:
                self.probe_mod.main()
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
