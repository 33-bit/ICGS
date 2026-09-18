import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests
from huggingface_hub.utils import EntryNotFoundError, RevisionNotFoundError, HfHubHTTPError

from icgs.configuration.method import MethodConfig
from icgs.data.archives import read_episode_archive
from icgs.data.collection.runner import AttemptSpec, CollectionLimits, run_collection
from icgs.data.collection.uploader import ExploratoryUploader, compute_file_sha256
from icgs.environments.rlbench.controller import (
    RLBenchTimedController,
    convert_rlbench_raw_observation,
    make_online_observation,
)
from icgs.environments.rlbench.teleop_oracle import (
    RLBenchCommandOracle,
    create_live_expert_command_factory,
    extract_rlbench_demo_waypoints,
    materialize_expert_demo,
)
from icgs.environments.rlbench.timed import TimedRLBenchAdapter


class _FakeTip:
    def get_pose(self):
        return [0.1, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0]


class _FakeArm:
    def __init__(self):
        self.tip = _FakeTip()
        self.joints = np.zeros(7, dtype=float)
        self.ik_calls = 0
        self.target_calls = 0

    def get_tip(self):
        return self.tip

    def solve_ik_via_jacobian(self, position, quaternion=None):
        self.ik_calls += 1
        return np.ones(7, dtype=float) * 0.1

    def set_joint_target_positions(self, values):
        self.target_calls += 1
        self.joints = np.asarray(values, dtype=float)

    def get_joint_positions(self):
        return self.joints.copy()


class _FakeGripper:
    def __init__(self):
        self.amount = np.array([1.0], dtype=float)
        self.actuate_calls = 0

    def get_open_amount(self):
        return self.amount.copy()

    def actuate(self, amount, velocity):
        self.actuate_calls += 1
        self.amount[:] = amount


class _FakePyRep:
    def __init__(self):
        self.t = 0.0

    def get_simulation_timestep(self):
        return 0.05


class _FakeScene:
    def __init__(self, pyrep):
        self._pyrep = pyrep
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        self._pyrep.t += self._pyrep.get_simulation_timestep()


class _StalledScene(_FakeScene):
    def step(self):
        self.step_calls += 1


class _FakeOuter:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class _FakeSim:
    def __init__(self, pyrep):
        self.pyrep = pyrep

    def simGetSimulationTime(self):
        return self.pyrep.t


class _FakeTask:
    def __init__(self):
        pyrep = _FakePyRep()
        self._scene = _FakeScene(pyrep)
        self._robot = SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper())
        self._pyrep = pyrep

    def reset(self):
        return ([], {"wrist_point_cloud": np.ones((2, 3), dtype=np.float64)})

    def get_observation(self):
        return {
            "wrist_point_cloud": np.ones((2, 3), dtype=np.float64),
            "gripper_pose": [0.1, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
            "gripper_open": 1.0,
        }


class _FakeHF:
    def __init__(self):
        self.commits = []
        self.objects = {}
        self.head = None

    def get_head_commit(self, revision: str = "main") -> str | None:
        return self.head

    def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit=None, **kwargs):
        self.commits.append((repo_id, repo_type, operations, commit_message, parent_commit))
        for operation in operations:
            path = operation["path_in_repo"]
            self.objects[path] = Path(operation["path_or_fileobj"]).read_bytes()
        rev = f"commit-{len(self.commits)}"
        self.head = rev
        return {"commit_hash": rev}

    def readback(self, path, revision):
        del revision
        if path not in self.objects:
            raise EntryNotFoundError(f"File {path} not found")
        return self.objects[path]


class SnapshotHFClient:
    """Injected fake HF client that honors immutable revisions and tracks snapshots."""

    def __init__(self, *, auth_fail: bool = False, timeout_fail: bool = False, permission: str = "write"):
        self.auth_fail = auth_fail
        self.timeout_fail = timeout_fail
        self.permission = permission
        self.commits: dict[str, dict[str, bytes]] = {"main": {}}
        self.revisions: list[str] = []
        self.commit_calls: list[dict] = []
        self.head: str | None = None

    def get_token_permission(self, token: str | None = None) -> str:
        del token
        if self.auth_fail:
            resp = requests.Response()
            resp.status_code = 401
            raise HfHubHTTPError("401 Unauthorized: Invalid token", response=resp)
        return self.permission

    def whoami(self, token: str | None = None) -> dict[str, str]:
        del token
        if self.auth_fail:
            resp = requests.Response()
            resp.status_code = 401
            raise HfHubHTTPError("401 Unauthorized: Invalid token", response=resp)
        return {"name": "test-authenticated-user"}

    def get_head_commit(self, revision: str = "main") -> str | None:
        return self.head

    def repo_info(self, *, repo_id: str, repo_type: str = "dataset"):
        del repo_type
        if self.auth_fail:
            resp = requests.Response()
            resp.status_code = 401
            raise HfHubHTTPError("401 Unauthorized: Invalid token", response=resp)
        if self.timeout_fail:
            raise requests.exceptions.Timeout("Connection timed out to huggingface.co")
        from types import SimpleNamespace
        return SimpleNamespace(id=repo_id, sha=self.head)

    def create_commit(
        self,
        *,
        repo_id: str,
        repo_type: str,
        operations: list[dict],
        commit_message: str,
        parent_commit: str | None = None,
        **kwargs,
    ) -> dict[str, str]:
        if self.auth_fail:
            resp = requests.Response()
            resp.status_code = 401
            raise HfHubHTTPError("401 Unauthorized: Invalid token", response=resp)
        if self.timeout_fail:
            raise requests.exceptions.Timeout("Connection timed out to huggingface.co")

        if self.head is not None and parent_commit is not None and parent_commit != self.head:
            resp = requests.Response()
            resp.status_code = 412
            raise HfHubHTTPError(
                f"412 Precondition Failed: parent_commit {parent_commit} does not match current head {self.head}",
                response=resp,
            )

        rev = hashlib.sha1(f"rev-{len(self.revisions) + 1}".encode()).hexdigest()
        new_files = dict(self.commits.get("main", {}))
        for op in operations:
            path_in_repo = op["path_in_repo"]
            src_file = Path(op["path_or_fileobj"])
            new_files[path_in_repo] = src_file.read_bytes()
        self.commits[rev] = new_files
        self.commits["main"] = new_files
        self.revisions.append(rev)
        self.head = rev
        self.commit_calls.append({
            "repo_id": repo_id,
            "revision": rev,
            "parent_commit": parent_commit,
            "operations": operations,
            "message": commit_message,
        })
        return {"commit_hash": rev}

    def readback(self, path_in_repo: str, revision: str = "main") -> bytes:
        if self.auth_fail:
            resp = requests.Response()
            resp.status_code = 401
            raise HfHubHTTPError("401 Unauthorized: Invalid token", response=resp)
        if self.timeout_fail:
            raise requests.exceptions.Timeout("Connection timed out to huggingface.co")

        if revision not in self.commits:
            raise RevisionNotFoundError(f"Revision {revision} not found")
        rev_files = self.commits[revision]
        if path_in_repo not in rev_files:
            raise EntryNotFoundError(f"File {path_in_repo} not found in revision {revision}")
        return rev_files[path_in_repo]


def _write_valid_test_episode(root: Path, episode_id: str) -> Path:
    from icgs.data.archives import write_episode_archive
    from icgs.contracts.method import TimedObservation, TimedCommand, ExecutedTransition
    from icgs.contracts.records import Observation

    t_obs1 = TimedObservation(
        Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0),
        0, 0.0, 0.0, "p",
    )
    t_obs2 = TimedObservation(
        Observation(np.zeros((4, 3), dtype=np.float64), np.eye(4, dtype=np.float64), 0.0),
        1, 0.1, 0.1, "p",
    )
    rec = {
        "schema_version": "icgs_episode_v1",
        "provenance": {
            "episode_id": episode_id,
            "program_id": "E01",
            "source_lineage_id": "exploratory_rlbench_dev",
            "asset_family_id": "rlbench-exploratory",
            "split": "dev",
            "calibration_id": "cal-01",
            "observation_origin": "measured",
            "raw_commands_id": "raw-01",
            "materialized_commands_id": "mat-01",
        },
        "online_observations": [
            {"points": np.zeros((4, 3), dtype=np.float32), "point_valid": np.ones(4, dtype=bool), "T_w_e": np.eye(4, dtype=np.float32), "grip": 0.0},
            {"points": np.zeros((4, 3), dtype=np.float32), "point_valid": np.ones(4, dtype=bool), "T_w_e": np.eye(4, dtype=np.float32), "grip": 0.0},
        ],
        "transitions": [
            ExecutedTransition(t_obs1, t_obs2, TimedCommand(np.eye(4), 0, 0.1), 0.1, 10, "ok")
        ],
    }
    return write_episode_archive(root, rec)


class RLBenchPhase34Tests(unittest.TestCase):
    def test_controller_uses_private_scene_and_live_backend_clock(self):
        task = _FakeTask()
        outer = _FakeOuter()
        sim = _FakeSim(task._pyrep)
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
        self.assertAlmostEqual(controller.physics_timestep, 0.05)
        controller.set_target(np.eye(4), 1)
        controller.step_physics()
        self.assertAlmostEqual(controller.simulator_time(), 0.05)
        self.assertEqual(task._scene.step_calls, 1)
        self.assertEqual(task._robot.arm.ik_calls, 1)
        self.assertGreaterEqual(task._robot.gripper.actuate_calls, 2)
        controller.close()
        controller.close()
        self.assertEqual(outer.shutdown_calls, 1)

    def test_controller_rejects_missing_real_environment(self):
        with self.assertRaisesRegex(RuntimeError, "live RLBench"):
            RLBenchTimedController(None)

    def test_controller_rejects_stalled_live_clock(self):
        task = _FakeTask()
        task._scene = _StalledScene(task._pyrep)
        controller = RLBenchTimedController(task, outer_env=_FakeOuter(), sim_api=_FakeSim(task._pyrep))
        controller.set_target(np.eye(4), 1)
        with self.assertRaisesRegex(RuntimeError, "clock"):
            controller.step_physics()

    def test_raw_observation_preserves_full_sensor_cloud(self):
        points = np.array([[100.0, 100.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        obs = convert_rlbench_raw_observation({
            "wrist_point_cloud": points,
            "gripper_pose": [0, 0, 1, 0, 0, 0, 1],
            "gripper_open": 1.0,
        })
        self.assertEqual(obs.points.shape, (2, 3))
        np.testing.assert_array_equal(obs.points, points)

    def test_oracle_requires_demo_and_never_pads_commands(self):
        with self.assertRaisesRegex(ValueError, "expert demonstration"):
            RLBenchCommandOracle()
        start = np.eye(4)
        end = np.eye(4)
        end[0, 3] = 0.01
        oracle = RLBenchCommandOracle(
            [(start, 1), (end, 0)], demo_content_hash="demo-sha", materialization_id="mat-v1", max_intervals=32
        )
        commands = oracle.generate_commands()
        self.assertGreater(len(commands), 0)
        self.assertLess(len(commands), 32)

    def test_demo_waypoints_reject_missing_pose_instead_of_identity_fallback(self):
        with self.assertRaisesRegex(ValueError, "gripper_pose"):
            extract_rlbench_demo_waypoints([SimpleNamespace(gripper_open=1.0)])

    def test_materializer_records_demo_hash_and_protocol_identity(self):
        demo = [
            SimpleNamespace(gripper_pose=[0, 0, 1, 0, 0, 0, 1], gripper_open=1.0),
            SimpleNamespace(gripper_pose=[0, 0, 1.1, 0, 0, 0, 1], gripper_open=0.0),
        ]
        oracle = materialize_expert_demo(demo, source_protocol_id="rlbench-demo-v1")
        self.assertTrue(oracle.demo_content_hash)
        self.assertTrue(oracle.materialization_id.startswith("rlbench-demo-v1:"))

    def test_uploader_requires_drive_and_hf_and_publishes_one_verified_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            ep = root / "episodes" / "ep-1"
            _write_valid_test_episode(root, "ep-1")
            hf = _FakeHF()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf)
            report = uploader.sync_episode("ep-1")
            self.assertTrue(report["drive_verified"])
            self.assertTrue(report["hf_verified"])
            self.assertEqual(len(hf.commits), 1)
            for path in ep.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(root)
                    self.assertEqual(compute_file_sha256(path), compute_file_sha256(drive / relative))

    def test_uploader_rejects_immutable_episode_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-1")
            hf = _FakeHF()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf)
            uploader.sync_episode("ep-1")
            (drive / "episodes" / "ep-1" / "manifest.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "immutable"):
                uploader.sync_episode("ep-1")

    def test_uploader_requires_both_publication_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            _write_valid_test_episode(root, "ep-1")
            with self.assertRaisesRegex(RuntimeError, "HF-second"):
                ExploratoryUploader(root, drive_root=Path(td) / "drive").sync_episode("ep-1")

    def test_uploader_rejects_hf_readback_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-1")

            class BadHF(_FakeHF):
                def readback(self, path, revision):
                    if revision == "main":
                        return None
                    return b"wrong"

            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=BadHF())
            with self.assertRaisesRegex(IOError, "HF readback"):
                uploader.sync_episode("ep-1")

    def test_controller_resolves_scene_pyrep_attribute_from_pinned_source(self):
        pyrep = _FakePyRep()
        # Pinned source rlbench.backend.scene.Scene defines self.pyrep (not self._pyrep)
        scene = SimpleNamespace(pyrep=pyrep, step=lambda: None)
        task = SimpleNamespace(
            _scene=scene,
            _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
            get_observation=lambda: {
                "wrist_point_cloud": np.ones((10, 3), dtype=np.float64),
                "gripper_pose": [0.1, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": 1.0,
            },
        )
        sim = _FakeSim(pyrep)
        outer = _FakeOuter()
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
        self.assertAlmostEqual(controller.physics_timestep, 0.05)
        self.assertEqual(controller._pyrep, pyrep)

    def test_controller_observe_wires_raw_unmodified_cloud_without_crop(self):
        pyrep = _FakePyRep()
        scene = SimpleNamespace(pyrep=pyrep, step=lambda: None)
        # Create a point cloud with points far outside DEFAULT_WORKSPACE_BOUNDS and >2048 points
        raw_points = np.zeros((2500, 3), dtype=np.float64)
        raw_points[:1000] = [100.0, 100.0, 100.0]  # Out of bounds
        raw_points[1000:] = [0.1, 0.2, 0.3]        # In bounds
        task = SimpleNamespace(
            _scene=scene,
            _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
            get_observation=lambda: {
                "wrist_point_cloud": raw_points,
                "gripper_pose": [0.1, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": 1.0,
            },
        )
        sim = _FakeSim(pyrep)
        outer = _FakeOuter()
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
        obs = controller.observe()
        self.assertEqual(obs.points.shape, (2500, 3))
        np.testing.assert_array_equal(obs.points, raw_points)

    def test_controller_reset_does_not_silently_lose_seed_on_type_error(self):
        pyrep = _FakePyRep()
        scene = SimpleNamespace(pyrep=pyrep, step=lambda: None)
        reset_called = False

        def _task_reset(*args, **kwargs):
            nonlocal reset_called
            if "seed" in kwargs:
                # Pinned TaskEnvironment.reset(self, demo=None) does not accept seed kwarg
                raise TypeError("reset() got an unexpected keyword argument 'seed'")
            reset_called = True
            return ([], {})

        task = SimpleNamespace(
            _scene=scene,
            _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
            reset=_task_reset,
        )
        sim = _FakeSim(pyrep)
        outer = _FakeOuter()
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=sim)

        np.random.seed(99999)
        expected_val = np.random.randint(0, 1000000)

        # Calling controller.reset with seed=99999 must seed the RNG deterministically
        controller.reset(seed=99999)
        self.assertTrue(reset_called)
        actual_val = np.random.randint(0, 1000000)
        self.assertEqual(actual_val, expected_val)

        # Non-integer seed must be rejected fail-closed
        with self.assertRaises(TypeError):
            controller.reset(seed="invalid_seed")  # type: ignore

    def test_controller_reset_to_demo_and_stage_demo(self):
        pyrep = _FakePyRep()
        scene = SimpleNamespace(pyrep=pyrep, step=lambda: None)
        demo_reset_called = False
        received_demo = None

        def _reset_to_demo(demo):
            nonlocal demo_reset_called, received_demo
            demo_reset_called = True
            received_demo = demo
            return ([], {})

        task = SimpleNamespace(
            _scene=scene,
            _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
            reset=lambda: ([], {}),
            reset_to_demo=_reset_to_demo,
        )
        sim = _FakeSim(pyrep)
        outer = _FakeOuter()
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=sim)

        fake_demo = SimpleNamespace(num_reset_attempts=1)
        controller.stage_demo(fake_demo)
        controller.reset()
        self.assertTrue(demo_reset_called)
        self.assertEqual(received_demo, fake_demo)

        # Staged demo should be consumed; next reset should call normal reset
        demo_reset_called = False
        controller.reset()
        self.assertFalse(demo_reset_called)

    def test_orchestrate_live_expert_demo_and_command_factory(self):
        from icgs.environments.rlbench.teleop_oracle import (
            orchestrate_live_expert_demo,
            create_live_expert_command_factory,
        )

        class FakeObservation:
            def __init__(self, pos, grip):
                self.gripper_pose = [*pos, 0.0, 0.0, 0.0, 1.0]
                self.gripper_open = float(grip)

        class FakeDemo:
            def __init__(self):
                self._observations = [
                    FakeObservation([0.1, 0.2, 0.8], 1.0),
                    FakeObservation([0.1, 0.2, 0.85], 1.0),
                    FakeObservation([0.1, 0.2, 0.85], 0.0),
                ]

        demos_called_with = []
        def _get_demos(amount, live_demos):
            demos_called_with.append((amount, live_demos))
            return [FakeDemo()]

        pyrep = _FakePyRep()
        scene = SimpleNamespace(pyrep=pyrep, step=lambda: None)
        task = SimpleNamespace(
            _scene=scene,
            _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
            get_demos=_get_demos,
            reset=lambda: ([], {}),
            reset_to_demo=lambda demo: ([], {}),
        )
        outer = _FakeOuter()
        controller = RLBenchTimedController(task, outer_env=outer, sim_api=_FakeSim(pyrep))

        # Test direct orchestration
        demo, oracle, commands = orchestrate_live_expert_demo(
            task,
            controller=controller,
            source_protocol_id="rlbench-live-v1",
            max_intervals=32,
            seed=42,
        )
        self.assertEqual(len(demos_called_with), 1)
        self.assertEqual(demos_called_with[0], (1, True))
        self.assertTrue(oracle.materialization_id.startswith("rlbench-live-v1:"))
        self.assertGreater(len(commands), 0)
        self.assertLessEqual(len(commands), 32)
        # Verify demo was staged on controller
        self.assertIs(controller._staged_demo, demo)

        # Test command factory wrapping and retention
        factory = create_live_expert_command_factory(
            source_protocol_id="rlbench-live-v1",
            max_intervals=32,
        )
        fake_adapter = SimpleNamespace(controller=controller)
        fake_spec = SimpleNamespace(episode_id="ep-test-01", action_seed=43, reset_seed=43)
        mat_cmd = factory(fake_spec, fake_adapter)
        self.assertGreater(len(mat_cmd.commands), 0)
        self.assertEqual(len(demos_called_with), 2)
        # Verify demo and oracle were not discarded
        self.assertIsNotNone(controller._last_demo)
        self.assertIsNotNone(controller._last_oracle)
        self.assertEqual(controller._last_demo_hash, factory.last_oracle.demo_content_hash)
        self.assertEqual(controller._last_materialization_id, factory.last_oracle.materialization_id)
        self.assertIs(factory.last_demo, controller._last_demo)
        self.assertEqual(mat_cmd.raw_commands_id, f"rlbench-live-v1:demo:{factory.last_oracle.demo_content_hash}")
        self.assertEqual(mat_cmd.materialized_commands_id, factory.last_oracle.materialization_id)

        # Test create_bound_attempt_spec produces fully-bound AttemptSpec
        from icgs.environments.rlbench.teleop_oracle import create_bound_attempt_spec
        bound_spec, bound_demo, bound_oracle = create_bound_attempt_spec(
            task,
            episode_id="ep-bound-01",
            source_protocol_id="rlbench-live-v1",
            controller=controller,
        )
        self.assertEqual(bound_spec.episode_id, "ep-bound-01")
        self.assertEqual(bound_spec.raw_commands_id, f"rlbench-live-v1:demo:{bound_oracle.demo_content_hash}")
        self.assertEqual(bound_spec.materialized_commands_id, bound_oracle.materialization_id)
        self.assertIsNotNone(bound_spec.commands)
        self.assertEqual(len(bound_spec.commands), len(bound_oracle.generate_commands()))

    def test_make_concrete_rlbench_environment_action_modes_and_obs_config(self):
        """Verify concrete environment composition uses proven submodules and gripper observations."""
        import sys
        from unittest import mock
        from unittest.mock import MagicMock
        from scripts.run_rlbench_exploratory import make_concrete_rlbench_environment

        mock_move_arm = MagicMock()
        mock_ee_pose = MagicMock()
        mock_discrete = MagicMock()
        mock_env_cls = MagicMock()
        mock_obs_cfg_cls = MagicMock()
        mock_tasks = MagicMock()

        obs_cfg_instance = MagicMock()
        mock_obs_cfg_cls.return_value = obs_cfg_instance
        env_instance = MagicMock()
        mock_env_cls.return_value = env_instance
        mock_task_env = MagicMock()
        mock_task_env._scene = SimpleNamespace(pyrep=_FakePyRep(), step=lambda: None)
        mock_task_env._robot = SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper())
        env_instance.get_task.return_value = mock_task_env
        mock_tasks.ReachTarget = MagicMock()

        modules_to_patch = {
            "pyrep": MagicMock(),
            "pyrep.backend": MagicMock(sim=_FakeSim(_FakePyRep())),
            "rlbench": MagicMock(),
            "rlbench.action_modes": MagicMock(),
            "rlbench.action_modes.action_mode": MagicMock(MoveArmThenGripper=mock_move_arm),
            "rlbench.action_modes.arm_action_modes": MagicMock(EndEffectorPoseViaIK=mock_ee_pose),
            "rlbench.action_modes.gripper_action_modes": MagicMock(Discrete=mock_discrete),
            "rlbench.environment": MagicMock(Environment=mock_env_cls),
            "rlbench.observation_config": MagicMock(ObservationConfig=mock_obs_cfg_cls),
            "rlbench.tasks": mock_tasks,
        }
        with mock.patch.dict(sys.modules, modules_to_patch):
            adapter = make_concrete_rlbench_environment("ReachTarget", headless=True)

            # Check obs_config set_all_high_dim / low_dim called
            obs_cfg_instance.set_all_high_dim.assert_called_once_with(False)
            obs_cfg_instance.set_all_low_dim.assert_called_once_with(False)
            # Check gripper pose and open enabled
            self.assertTrue(obs_cfg_instance.gripper_pose)
            self.assertTrue(obs_cfg_instance.gripper_open)
            self.assertTrue(obs_cfg_instance.wrist_camera.point_cloud)
            self.assertTrue(obs_cfg_instance.wrist_camera.depth)

            # Check action mode composition uses proven submodules
            mock_ee_pose.assert_called_once()
            mock_discrete.assert_called_once()
            mock_move_arm.assert_called_once()

    def test_git_info_fail_closed_and_repo_root_hashing(self):
        """Verify _get_git_info fails closed on error and hashes tracked + untracked at repo root."""
        import subprocess
        from unittest import mock
        from scripts.run_rlbench_exploratory import _get_git_info

        # Normal execution on the repo returns valid revision and clean/dirty status
        rev, is_dirty, patch_digest = _get_git_info()
        self.assertEqual(len(rev), 40)
        self.assertIsInstance(is_dirty, bool)
        if is_dirty:
            self.assertIsInstance(patch_digest, str)
            self.assertEqual(len(patch_digest), 64)
        else:
            self.assertIsNone(patch_digest)

        # Fails closed on git rev-parse failure
        with mock.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(RuntimeError):
                _get_git_info()

    def test_uploader_archive_prevalidation_and_deterministic_order_and_hf_conflict(self):
        """Uploader pre-validates archive via read_episode_archive, orders files deterministically, and detects HF conflicts."""
        import tempfile
        from icgs.data.collection.uploader import ExploratoryUploader

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            local_root = tmp / "local"
            drive_root = tmp / "drive"
            local_root.mkdir()
            drive_root.mkdir()

            ep_dir = local_root / "episodes" / "ep-corrupt"
            ep_dir.mkdir(parents=True)
            # 1. Missing or corrupted manifest.json raises error before upload
            (ep_dir / "manifest.json").write_text("not json", encoding="utf-8")
            (ep_dir / "transitions.npz").write_bytes(b"some npz bytes")

            fake_hf = _FakeHF()
            uploader = ExploratoryUploader(local_root, drive_root=drive_root, hf_repo_id="org/repo", hf_client=fake_hf)
            with self.assertRaises(Exception):
                uploader.sync_episode("ep-corrupt")

            # 2. Test HF conflict check
            ep_good = local_root / "episodes" / "ep-good"
            _write_valid_test_episode(local_root, "ep-good")

            # Pre-seed fake_hf with conflicting bytes for one of the episode files
            conflicting_path = "exploratory/episodes/ep-good/manifest.json"
            fake_hf.objects[conflicting_path] = b'{"conflicting": "remote bytes"}'

            with self.assertRaises(ValueError) as ctx:
                uploader.sync_episode("ep-good")
            self.assertIn("immutable episode artifact differs on HF", str(ctx.exception))

            # When remote bytes match local bytes exactly, idempotent sync succeeds
            fake_hf.objects[conflicting_path] = (ep_good / "manifest.json").read_bytes()
            res = uploader.sync_episode("ep-good")
            self.assertTrue(res["hf_verified"])
            self.assertTrue(res["drive_verified"])

    def test_real_run_collection_to_archive_loader_integration(self):
        """End-to-end integration test: run_collection -> read_episode_archive proving IDs match content hashes, reset_to_demo order, and full measured cloud."""
        import tempfile
        from icgs.data.archives import read_episode_archive
        from icgs.data.collection.runner import AttemptSpec, CollectionLimits, run_collection
        from icgs.configuration.method import MethodConfig
        from icgs.environments.rlbench.teleop_oracle import create_live_expert_command_factory
        from icgs.environments.rlbench.controller import RLBenchTimedController, make_online_observation
        from icgs.environments.rlbench.timed import TimedRLBenchAdapter
        from icgs.data.collection.uploader import ExploratoryUploader

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            out_dir = tmp / "exploratory"
            drive_dir = tmp / "drive"
            out_dir.mkdir()
            drive_dir.mkdir()

            manifest_file = out_dir / "dataset_manifest.json"
            manifest_file.write_text(json.dumps({
                "manifest_version": 1,
                "dataset_track": "exploratory",
                "episodes": [],
                "lineage": [{"lineage_id": "exploratory_rlbench_dev", "split": "dev", "parent_ids": []}],
                "asset_families": [{"asset_family_id": "rlbench-exploratory", "split": "dev"}],
            }, indent=2), encoding="utf-8")

            raw_measured_points = np.zeros((2500, 3), dtype=np.float64)
            raw_measured_points[:1000] = [50.0, 50.0, 50.0]
            raw_measured_points[1000:] = [0.1, 0.2, 0.3]

            class _Obs:
                def __init__(self):
                    self.wrist_point_cloud = raw_measured_points.copy()
                    self.gripper_pose = [0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
                    self.gripper_open = 1.0

            demo_obs_list = [
                SimpleNamespace(gripper_pose=[0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=0.0),
            ]
            fake_demo = SimpleNamespace(_observations=demo_obs_list, num_reset_attempts=1)

            demo_reset_calls = []
            steps_called = 0

            pyrep = _FakePyRep()
            def fake_step():
                nonlocal steps_called
                steps_called += 1
                pyrep.t += pyrep.get_simulation_timestep()

            scene = SimpleNamespace(pyrep=pyrep, step=fake_step)

            def _reset_to_demo(d):
                demo_reset_calls.append((steps_called, d))
                return ([], {})

            task = SimpleNamespace(
                _scene=scene,
                _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
                reset=lambda: ([], {}),
                reset_to_demo=_reset_to_demo,
                get_demos=lambda amount=1, live_demos=True: [fake_demo],
                get_observation=lambda: _Obs(),
            )
            sim = _FakeSim(pyrep)
            outer = _FakeOuter()

            def env_factory(spec):
                ctrl = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
                return TimedRLBenchAdapter(
                    ctrl, physics_dt=0.05, sensor_profile_id="rlbench-wrist-depth-raw-v1"
                )

            # Bound <= 16 intervals
            command_factory = create_live_expert_command_factory(
                source_protocol_id="rlbench-live-expert-v1",
                max_intervals=8,
                interval_s=0.1,
            )

            fake_hf = _FakeHF()
            uploader = ExploratoryUploader(
                out_dir,
                drive_root=drive_dir,
                hf_repo_id="org/repo",
                hf_client=fake_hf,
            )

            index_pub_calls = []
            def spy_index_pub(path):
                res = uploader.sync_manifest(path)
                index_pub_calls.append((str(path), res))
                return res

            spec = AttemptSpec(
                episode_id="e01-ep-0001",
                program_id="E01",
                source_lineage_id="exploratory_rlbench_dev",
                asset_family_id="rlbench-exploratory",
                split="dev",
                generator_seed=1001,
                reset_seed=2001,
                action_seed=3001,
                calibration_id="rlbench-wrist-depth-raw-v1",
                observation_origin="measured",
                raw_commands_id="rlbench-live-expert-v1",
                materialized_commands_id="rlbench-live-mat-placeholder",
            )

            limits = CollectionLimits(
                max_attempts=1,
                max_intervals=16,  # Bounded <= 16 intervals
                max_wall_time_s=60.0,
                max_disk_bytes=100 * 1024 * 1024,
            )
            config = MethodConfig(collection={"max_episode_intervals": 16})
            protocol_manifest = {
                "environment_protocol_id": "rlbench-live-clock-v1",
                "controller_protocol_id": "rlbench-timed-ik-v1",
                "sensor_protocol_id": "rlbench-wrist-depth-raw-v1",
                "asset_protocol_id": "rlbench-assets-pinned-v1",
                "split_protocol_id": "exploratory-split-dev-v1",
                "collection_protocol_id": "icgs-e01-exploratory-v1",
                "metadata": {"program_id": "E01", "track": "exploratory"},
            }

            report = run_collection(
                [spec],
                env_factory,
                dataset_root=out_dir,
                dataset_manifest_path=manifest_file,
                limits=limits,
                config=config,
                protocol_manifest=protocol_manifest,
                code_revision="abcdef1234567890abcdef1234567890abcdef12",
                is_dirty=False,
                generator_version="0.1.0-e01",
                command_factory=command_factory,
                online_provider=make_online_observation,
                episode_publisher=uploader.sync_episode,
                index_publisher=spy_index_pub,
                dataset_track="exploratory",
            )
            self.assertEqual(report.status, "completed")
            self.assertEqual(report.episodes_published, 1)

            # 1. Verify per-episode index publication occurred
            self.assertEqual(len(index_pub_calls), 1)
            self.assertTrue(index_pub_calls[0][1]["hf_verified"])
            self.assertTrue(index_pub_calls[0][1]["drive_verified"])

            # 2. Read back episode archive via real archive loader
            ep_manifest_path = out_dir / "episodes" / "e01-ep-0001" / "manifest.json"
            ep_archive = read_episode_archive(ep_manifest_path)

            # 3. Prove persisted IDs equal actual demo/materialization content hashes
            expected_oracle = command_factory.last_oracle
            self.assertEqual(
                ep_archive["provenance"]["raw_commands_id"],
                f"rlbench-live-expert-v1:demo:{expected_oracle.demo_content_hash}",
            )
            self.assertEqual(
                ep_archive["provenance"]["materialized_commands_id"],
                expected_oracle.materialization_id,
            )

            # 4. Prove seeded demo/reset order: reset_to_demo called before stepping
            self.assertGreater(len(demo_reset_calls), 0)
            self.assertEqual(demo_reset_calls[0][0], 0)  # steps_called was 0 at reset_to_demo!

            # 5. Prove full measured cloud: uncropped points preserved in transitions
            trans = ep_archive["transitions"]
            self.assertGreater(len(trans), 0)
            meas_pts = trans[0].before.observation.points
            self.assertEqual(meas_pts.shape, (2500, 3))
            np.testing.assert_array_equal(meas_pts, raw_measured_points)

    def test_uploader_colab_drive_root_verification(self):
        """Uploader verifies mounted Drive root under /content/drive/MyDrive in Colab."""
        from unittest import mock
        from icgs.data.collection.uploader import ExploratoryUploader

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "local"
            root.mkdir()

            # Case 1: /content exists but /content/drive/MyDrive is not mounted
            with mock.patch.object(Path, "is_dir", autospec=True) as mock_is_dir:
                def fake_is_dir(p):
                    path_str = str(p)
                    if path_str == "/content/drive/MyDrive":
                        return False
                    if path_str == "/content":
                        return True
                    return True
                mock_is_dir.side_effect = fake_is_dir

                uploader = ExploratoryUploader(
                    root,
                    drive_root=Path("/custom/drive"),
                    hf_repo_id="org/repo",
                    hf_client=_FakeHF(),
                )
                with self.assertRaisesRegex(RuntimeError, "Google Drive is not mounted under /content/drive/MyDrive"):
                    uploader.sync_episode("ep-test")

            # Case 2: /content/drive/MyDrive is mounted but drive_root is outside
            with mock.patch.object(Path, "is_dir", autospec=True) as mock_is_dir:
                def fake_is_dir2(p):
                    path_str = str(p)
                    if path_str == "/content/drive/MyDrive":
                        return True
                    if path_str == "/content":
                        return True
                    return True
                mock_is_dir.side_effect = fake_is_dir2

                uploader = ExploratoryUploader(
                    root,
                    drive_root=Path("/content/other_dir"),
                    hf_repo_id="org/repo",
                    hf_client=_FakeHF(),
                )
                with self.assertRaisesRegex(RuntimeError, "Drive root must be a descendant of /content/drive/MyDrive"):
                    uploader.sync_episode("ep-test")

    def test_telemetry_and_video_companion_archival_and_upload(self):
        """Companion telemetry.npz and video_front.mp4 are saved and synced seamlessly."""
        from icgs.data.archives import write_episode_archive, read_episode_archive
        from icgs.data.schemas.episodes import validate_episode
        from icgs.data.collection.uploader import ExploratoryUploader

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            local_root = tmp / "local"
            drive_root = tmp / "drive"
            local_root.mkdir()
            drive_root.mkdir()

            fake_hf = _FakeHF()
            uploader = ExploratoryUploader(
                local_root,
                drive_root=drive_root,
                hf_repo_id="org/repo",
                hf_client=fake_hf,
                hf_subfolder="exploratory",
            )

            # Construct valid minimal episode record
            from icgs.contracts.method import TimedObservation, TimedCommand, ExecutedTransition
            from icgs.contracts.records import Observation
            from icgs.configuration.method import MethodConfig

            pts = np.zeros((10, 3), dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            obs0 = Observation(points=pts, T_w_e=T, grip=1)
            t_obs0 = TimedObservation(observation=obs0, boundary=0, simulator_timestamp=0.0, measured_wall_timestamp=0.0, sensor_profile_id="p1")
            obs1 = Observation(points=pts, T_w_e=T, grip=1)
            t_obs1 = TimedObservation(observation=obs1, boundary=1, simulator_timestamp=0.1, measured_wall_timestamp=0.1, sensor_profile_id="p1")
            cmd = TimedCommand(target_w=T, grip=1, duration_s=0.1)
            trans = ExecutedTransition(before=t_obs0, after=t_obs1, command=cmd, achieved_duration_s=0.1, physics_substeps=5, controller_status="ok")

            online_obs = [
                {"points": pts, "point_valid": np.ones(10, dtype=bool), "T_w_e": T, "grip": 1},
                {"points": pts, "point_valid": np.ones(10, dtype=bool), "T_w_e": T, "grip": 1},
            ]
            record = {
                "schema_version": "icgs_episode_v1",
                "provenance": {
                    "episode_id": "ep-aux-001",
                    "source_lineage_id": "lineage-1",
                    "asset_family_id": "family-1",
                    "program_id": "program-1",
                    "calibration_id": "calibration-1",
                    "raw_commands_id": "cmds-1",
                    "materialized_commands_id": "mat-1",
                    "split": "dev",
                    "observation_origin": "measured",
                },
                "online_observations": online_obs,
                "transitions": [trans],
            }

            auxiliary = {
                "joint_positions": np.ones((2, 7), dtype=np.float64),
                "joint_velocities": np.zeros((2, 7), dtype=np.float64),
                "front_rgb_frames": np.zeros((2, 32, 32, 3), dtype=np.uint8),
            }

            manifest_p = write_episode_archive(local_root, record, auxiliary=auxiliary)
            ep_dir = local_root / "episodes" / "ep-aux-001"

            # Check files on disk
            self.assertTrue((ep_dir / "telemetry.npz").is_file())
            self.assertTrue((ep_dir / "video_front.mp4").is_file())

            # Read telemetry arrays
            with np.load(ep_dir / "telemetry.npz") as telem:
                self.assertEqual(telem["joint_positions"].shape, (2, 7))
                self.assertEqual(telem["joint_velocities"].shape, (2, 7))

            # Validate episode archive roundtrip
            restored = read_episode_archive(manifest_p)
            validate_episode(restored)

            # Sync episode to Drive and HF
            sync_res = uploader.sync_episode("ep-aux-001")
            self.assertTrue(sync_res["drive_verified"])
            self.assertTrue(sync_res["hf_verified"])
            self.assertIn("episodes/ep-aux-001/telemetry.npz", sync_res["synced_files"])
            self.assertIn("episodes/ep-aux-001/video_front.mp4", sync_res["synced_files"])

            # Verify Drive copy
            self.assertTrue((drive_root / "episodes/ep-aux-001/telemetry.npz").is_file())
            self.assertTrue((drive_root / "episodes/ep-aux-001/video_front.mp4").is_file())

            # Verify HF copy
            self.assertIn("exploratory/episodes/ep-aux-001/telemetry.npz", fake_hf.objects)
            self.assertIn("exploratory/episodes/ep-aux-001/video_front.mp4", fake_hf.objects)

    def test_two_episode_end_to_end_publication_and_indexing(self):
        """Integration test with TWO real archive fixtures through actual uploader + runner.

        Verifies:
        - Episode 1 completes, publishes episode to Drive and HF (rev1), and index to Drive and HF (rev2).
        - Episode 2 environment steps ONLY AFTER Episode 1 and index are completely verified.
        - Episode 2 completes, publishes episode to Drive and HF (rev3), and index to Drive and HF (rev4).
        - Dataset manifest on disk, Drive, and HF are identical.
        - Both episodes load cleanly with read_episode_archive.
        """
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "dataset"
            drive_dir = Path(td) / "drive"
            out_dir.mkdir(parents=True, exist_ok=True)
            drive_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = out_dir / "dataset_manifest.json"
            manifest_file.write_text(json.dumps({
                "manifest_version": 1,
                "dataset_track": "exploratory",
                "episodes": [],
                "lineage": [{"lineage_id": "exploratory_rlbench_dev", "split": "dev", "parent_ids": []}],
                "asset_families": [{"asset_family_id": "rlbench-exploratory", "split": "dev"}],
            }, indent=2), encoding="utf-8")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(
                out_dir,
                drive_root=drive_dir,
                hf_repo_id="org/exploratory-repo",
                hf_client=hf_client,
            )

            execution_timeline = []

            def make_task(step_idx: int, should_crash: bool = False):
                pyrep = _FakePyRep()
                raw_measured_points = np.zeros((2500, 3), dtype=np.float64)
                raw_measured_points[:1000] = [50.0, 50.0, 50.0]
                raw_measured_points[1000:] = [0.1, 0.2, 0.3]

                class _Obs:
                    def __init__(self):
                        self.wrist_point_cloud = raw_measured_points.copy()
                        self.gripper_pose = [0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
                        self.gripper_open = 1.0

                demo_obs_list = [
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=0.0),
                ]
                fake_demo = SimpleNamespace(_observations=demo_obs_list, num_reset_attempts=1)

                def fake_step():
                    execution_timeline.append(f"env_{step_idx}_step")
                    pyrep.t += pyrep.get_simulation_timestep()

                scene = SimpleNamespace(pyrep=pyrep, step=fake_step)

                def _get_demos(amount=1, live_demos=True):
                    if should_crash:
                        raise RuntimeError("Simulated crash during attempt")
                    return [fake_demo]

                task = SimpleNamespace(
                    _scene=scene,
                    _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
                    reset=lambda: ([], {}),
                    reset_to_demo=lambda d: ([], {}),
                    get_demos=_get_demos,
                    get_observation=lambda: _Obs(),
                )
                sim = _FakeSim(pyrep)
                outer = _FakeOuter()
                ctrl = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
                return TimedRLBenchAdapter(ctrl, physics_dt=0.05, sensor_profile_id="rlbench-wrist-depth-raw-v1")

            created_envs = []
            def env_factory(spec):
                step_idx = len(created_envs)
                execution_timeline.append(f"env_{step_idx}_created")
                adapter = make_task(step_idx)
                created_envs.append(adapter)
                return adapter

            command_factory = create_live_expert_command_factory(
                source_protocol_id="rlbench-live-expert-v1",
                max_intervals=8,
                interval_s=0.1,
            )

            index_pub_calls = []
            def spy_index_pub(path):
                execution_timeline.append(f"index_pub_start_{len(index_pub_calls)}")
                res = uploader.sync_manifest(path)
                index_pub_calls.append((str(path), res))
                execution_timeline.append(f"index_pub_done_{len(index_pub_calls)}")
                return res

            specs = [
                AttemptSpec(
                    episode_id="e01-ep-0001",
                    program_id="E01",
                    source_lineage_id="exploratory_rlbench_dev",
                    asset_family_id="rlbench-exploratory",
                    split="dev",
                    generator_seed=1001,
                    reset_seed=2001,
                    action_seed=3001,
                    calibration_id="rlbench-wrist-depth-raw-v1",
                    observation_origin="measured",
                    raw_commands_id="rlbench-live-expert-v1",
                    materialized_commands_id="rlbench-live-mat-placeholder",
                ),
                AttemptSpec(
                    episode_id="e01-ep-0002",
                    program_id="E01",
                    source_lineage_id="exploratory_rlbench_dev",
                    asset_family_id="rlbench-exploratory",
                    split="dev",
                    generator_seed=1002,
                    reset_seed=2002,
                    action_seed=3002,
                    calibration_id="rlbench-wrist-depth-raw-v1",
                    observation_origin="measured",
                    raw_commands_id="rlbench-live-expert-v1",
                    materialized_commands_id="rlbench-live-mat-placeholder",
                ),
            ]

            limits = CollectionLimits(
                max_attempts=2,
                max_intervals=32,
                max_wall_time_s=60.0,
                max_disk_bytes=100 * 1024 * 1024,
            )
            config = MethodConfig(collection={"max_episode_intervals": 16})
            protocol_manifest = {
                "environment_protocol_id": "rlbench-live-clock-v1",
                "controller_protocol_id": "rlbench-timed-ik-v1",
                "sensor_protocol_id": "rlbench-wrist-depth-raw-v1",
                "asset_protocol_id": "rlbench-assets-pinned-v1",
                "split_protocol_id": "exploratory-split-dev-v1",
                "collection_protocol_id": "icgs-e01-exploratory-v1",
                "metadata": {"program_id": "E01", "track": "exploratory"},
            }

            report = run_collection(
                specs,
                env_factory,
                dataset_root=out_dir,
                dataset_manifest_path=manifest_file,
                limits=limits,
                config=config,
                protocol_manifest=protocol_manifest,
                code_revision="abcdef1234567890abcdef1234567890abcdef12",
                is_dirty=False,
                generator_version="0.1.0-e01",
                command_factory=command_factory,
                online_provider=make_online_observation,
                episode_publisher=uploader.sync_episode,
                index_publisher=spy_index_pub,
                dataset_track="exploratory",
            )

            self.assertEqual(report.status, "completed")
            self.assertEqual(report.episodes_published, 2)
            self.assertEqual(len(index_pub_calls), 2)

            # Strict sequencing: env_1 must step ONLY AFTER index_pub_done_1
            self.assertIn("env_0_step", execution_timeline)
            self.assertIn("index_pub_done_1", execution_timeline)
            self.assertIn("env_1_step", execution_timeline)
            self.assertIn("index_pub_done_2", execution_timeline)

            idx_env0_step = execution_timeline.index("env_0_step")
            idx_index1_done = execution_timeline.index("index_pub_done_1")
            idx_env1_step = execution_timeline.index("env_1_step")
            idx_index2_done = execution_timeline.index("index_pub_done_2")

            self.assertLess(idx_env0_step, idx_index1_done)
            self.assertLess(idx_index1_done, idx_env1_step)
            self.assertLess(idx_env1_step, idx_index2_done)

            # Total commits on HF: 4 (ep1 -> index1 -> ep2 -> index2)
            self.assertEqual(len(hf_client.revisions), 4)

            # Both episodes verified locally and on Drive
            for ep_id in ("e01-ep-0001", "e01-ep-0002"):
                local_ep_manifest = out_dir / "episodes" / ep_id / "manifest.json"
                drive_ep_manifest = drive_dir / "episodes" / ep_id / "manifest.json"
                self.assertTrue(local_ep_manifest.is_file())
                self.assertTrue(drive_ep_manifest.is_file())
                self.assertEqual(compute_file_sha256(local_ep_manifest), compute_file_sha256(drive_ep_manifest))

                archive_local = read_episode_archive(local_ep_manifest)
                archive_drive = read_episode_archive(drive_ep_manifest)
                self.assertEqual(archive_local["provenance"]["episode_id"], ep_id)
                self.assertEqual(archive_drive["provenance"]["episode_id"], ep_id)
                self.assertEqual(len(archive_local["transitions"]), len(archive_drive["transitions"]))

            # Final manifest on disk, Drive, and HF are identical
            final_local_manifest = manifest_file.read_bytes()
            final_drive_manifest = (drive_dir / "dataset_manifest.json").read_bytes()
            final_hf_manifest = hf_client.readback("exploratory/dataset_manifest.json", "main")
            self.assertEqual(final_local_manifest, final_drive_manifest)
            self.assertEqual(final_local_manifest, final_hf_manifest)

            manifest_json = json.loads(final_local_manifest.decode("utf-8"))
            self.assertEqual(len(manifest_json["episodes"]), 2)
            self.assertEqual(manifest_json["provenance"]["generator_seeds"], [1001, 1002])

    def test_episodes_published_not_incremented_on_index_publication_failure(self):
        """Runner must not increment episodes_published or retain dirty manifest on index failure."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "dataset"
            drive_dir = Path(td) / "drive"
            out_dir.mkdir(parents=True, exist_ok=True)
            drive_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = out_dir / "dataset_manifest.json"
            manifest_file.write_text(json.dumps({
                "manifest_version": 1,
                "dataset_track": "exploratory",
                "episodes": [],
                "lineage": [{"lineage_id": "exploratory_rlbench_dev", "split": "dev", "parent_ids": []}],
                "asset_families": [{"asset_family_id": "rlbench-exploratory", "split": "dev"}],
            }, indent=2), encoding="utf-8")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(
                out_dir,
                drive_root=drive_dir,
                hf_repo_id="org/exploratory-repo",
                hf_client=hf_client,
            )

            def make_task():
                pyrep = _FakePyRep()
                raw_measured_points = np.zeros((2500, 3), dtype=np.float64)
                raw_measured_points[:1000] = [50.0, 50.0, 50.0]
                raw_measured_points[1000:] = [0.1, 0.2, 0.3]

                class _Obs:
                    def __init__(self):
                        self.wrist_point_cloud = raw_measured_points.copy()
                        self.gripper_pose = [0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
                        self.gripper_open = 1.0

                demo_obs_list = [
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=0.0),
                ]
                fake_demo = SimpleNamespace(_observations=demo_obs_list, num_reset_attempts=1)

                def fake_step():
                    pyrep.t += pyrep.get_simulation_timestep()

                task = SimpleNamespace(
                    _scene=SimpleNamespace(pyrep=pyrep, step=fake_step),
                    _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
                    reset=lambda: ([], {}),
                    reset_to_demo=lambda d: ([], {}),
                    get_demos=lambda amount=1, live_demos=True: [fake_demo],
                    get_observation=lambda: _Obs(),
                )
                sim = _FakeSim(pyrep)
                outer = _FakeOuter()
                ctrl = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
                return TimedRLBenchAdapter(ctrl, physics_dt=0.05, sensor_profile_id="rlbench-wrist-depth-raw-v1")

            command_factory = create_live_expert_command_factory(
                source_protocol_id="rlbench-live-expert-v1",
                max_intervals=8,
                interval_s=0.1,
            )

            def failing_index_pub(path):
                raise RuntimeError("simulated index publication network drop")

            spec = AttemptSpec(
                episode_id="e01-ep-fail-pub",
                program_id="E01",
                source_lineage_id="exploratory_rlbench_dev",
                asset_family_id="rlbench-exploratory",
                split="dev",
                generator_seed=1001,
                reset_seed=2001,
                action_seed=3001,
                calibration_id="rlbench-wrist-depth-raw-v1",
                observation_origin="measured",
                raw_commands_id="rlbench-live-expert-v1",
                materialized_commands_id="rlbench-live-mat-placeholder",
            )

            limits = CollectionLimits(max_attempts=1, max_intervals=16, max_wall_time_s=60.0, max_disk_bytes=100 * 1024 * 1024)
            config = MethodConfig(collection={"max_episode_intervals": 16})
            protocol_manifest = {
                "environment_protocol_id": "rlbench-live-clock-v1",
                "controller_protocol_id": "rlbench-timed-ik-v1",
                "sensor_protocol_id": "rlbench-wrist-depth-raw-v1",
                "asset_protocol_id": "rlbench-assets-pinned-v1",
                "split_protocol_id": "exploratory-split-dev-v1",
                "collection_protocol_id": "icgs-e01-exploratory-v1",
                "metadata": {"program_id": "E01", "track": "exploratory"},
            }

            report = run_collection(
                [spec],
                lambda spec: make_task(),
                dataset_root=out_dir,
                dataset_manifest_path=manifest_file,
                limits=limits,
                config=config,
                protocol_manifest=protocol_manifest,
                code_revision="abcdef1234567890abcdef1234567890abcdef12",
                is_dirty=False,
                generator_version="0.1.0-e01",
                command_factory=command_factory,
                online_provider=make_online_observation,
                episode_publisher=uploader.sync_episode,
                index_publisher=failing_index_pub,
                dataset_track="exploratory",
            )

            self.assertEqual(report.status, "index_publication_failed", msg=f"report.attempt_records: {report.attempt_records}")
            self.assertEqual(report.episodes_published, 0)
            # Disk manifest must not retain the unverified episode (restored to empty state)
            manifest_restored = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest_restored["episodes"]), 0)

    def test_final_manifest_does_not_diverge_after_later_failed_attempt(self):
        """Final manifest rewrite must not include seeds or entries of failed attempts."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "dataset"
            drive_dir = Path(td) / "drive"
            out_dir.mkdir(parents=True, exist_ok=True)
            drive_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = out_dir / "dataset_manifest.json"
            manifest_file.write_text(json.dumps({
                "manifest_version": 1,
                "dataset_track": "exploratory",
                "episodes": [],
                "lineage": [{"lineage_id": "exploratory_rlbench_dev", "split": "dev", "parent_ids": []}],
                "asset_families": [{"asset_family_id": "rlbench-exploratory", "split": "dev"}],
            }, indent=2), encoding="utf-8")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(
                out_dir,
                drive_root=drive_dir,
                hf_repo_id="org/exploratory-repo",
                hf_client=hf_client,
            )

            def make_task(should_crash=False):
                pyrep = _FakePyRep()
                raw_measured_points = np.zeros((2500, 3), dtype=np.float64)
                raw_measured_points[:1000] = [50.0, 50.0, 50.0]
                raw_measured_points[1000:] = [0.1, 0.2, 0.3]

                class _Obs:
                    def __init__(self):
                        self.wrist_point_cloud = raw_measured_points.copy()
                        self.gripper_pose = [0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0]
                        self.gripper_open = 1.0

                demo_obs_list = [
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=1.0),
                    SimpleNamespace(gripper_pose=[0.1, 0.2, 0.81, 0.0, 0.0, 0.0, 1.0], gripper_open=0.0),
                ]
                fake_demo = SimpleNamespace(_observations=demo_obs_list, num_reset_attempts=1)

                def _get_demos(amount=1, live_demos=True):
                    if should_crash:
                        raise RuntimeError("Simulated crash during attempt 2")
                    return [fake_demo]

                def fake_step():
                    pyrep.t += pyrep.get_simulation_timestep()

                task = SimpleNamespace(
                    _scene=SimpleNamespace(pyrep=pyrep, step=fake_step),
                    _robot=SimpleNamespace(arm=_FakeArm(), gripper=_FakeGripper()),
                    reset=lambda: ([], {}),
                    reset_to_demo=lambda d: ([], {}),
                    get_demos=_get_demos,
                    get_observation=lambda: _Obs(),
                )
                sim = _FakeSim(pyrep)
                outer = _FakeOuter()
                ctrl = RLBenchTimedController(task, outer_env=outer, sim_api=sim)
                return TimedRLBenchAdapter(ctrl, physics_dt=0.05, sensor_profile_id="rlbench-wrist-depth-raw-v1")

            created_envs = []
            def env_factory(spec):
                should_crash = len(created_envs) == 1
                adapter = make_task(should_crash=should_crash)
                created_envs.append(adapter)
                return adapter

            command_factory = create_live_expert_command_factory(
                source_protocol_id="rlbench-live-expert-v1",
                max_intervals=8,
                interval_s=0.1,
            )

            specs = [
                AttemptSpec(
                    episode_id="e01-ep-0001",
                    program_id="E01",
                    source_lineage_id="exploratory_rlbench_dev",
                    asset_family_id="rlbench-exploratory",
                    split="dev",
                    generator_seed=1001,
                    reset_seed=2001,
                    action_seed=3001,
                    calibration_id="rlbench-wrist-depth-raw-v1",
                    observation_origin="measured",
                    raw_commands_id="rlbench-live-expert-v1",
                    materialized_commands_id="rlbench-live-mat-placeholder",
                ),
                AttemptSpec(
                    episode_id="e01-ep-0002",
                    program_id="E01",
                    source_lineage_id="exploratory_rlbench_dev",
                    asset_family_id="rlbench-exploratory",
                    split="dev",
                    generator_seed=1002,
                    reset_seed=2002,
                    action_seed=3002,
                    calibration_id="rlbench-wrist-depth-raw-v1",
                    observation_origin="measured",
                    raw_commands_id="rlbench-live-expert-v1",
                    materialized_commands_id="rlbench-live-mat-placeholder",
                ),
            ]

            limits = CollectionLimits(max_attempts=2, max_intervals=32, max_wall_time_s=60.0, max_disk_bytes=100 * 1024 * 1024)
            config = MethodConfig(collection={"max_episode_intervals": 16})
            protocol_manifest = {
                "environment_protocol_id": "rlbench-live-clock-v1",
                "controller_protocol_id": "rlbench-timed-ik-v1",
                "sensor_protocol_id": "rlbench-wrist-depth-raw-v1",
                "asset_protocol_id": "rlbench-assets-pinned-v1",
                "split_protocol_id": "exploratory-split-dev-v1",
                "collection_protocol_id": "icgs-e01-exploratory-v1",
                "metadata": {"program_id": "E01", "track": "exploratory"},
            }

            report = run_collection(
                specs,
                env_factory,
                dataset_root=out_dir,
                dataset_manifest_path=manifest_file,
                limits=limits,
                config=config,
                protocol_manifest=protocol_manifest,
                code_revision="abcdef1234567890abcdef1234567890abcdef12",
                is_dirty=False,
                generator_version="0.1.0-e01",
                command_factory=command_factory,
                online_provider=make_online_observation,
                episode_publisher=uploader.sync_episode,
                index_publisher=uploader.sync_manifest,
                dataset_track="exploratory",
            )

            self.assertEqual(report.status, "failed")
            self.assertEqual(report.quarantined_count, 1)
            self.assertEqual(report.episodes_published, 1)

            # Ensure local manifest matches remote verified Drive and HF bytes exactly
            local_bytes = manifest_file.read_bytes()
            drive_bytes = (drive_dir / "dataset_manifest.json").read_bytes()
            hf_bytes = hf_client.readback("exploratory/dataset_manifest.json", "main")
            self.assertEqual(local_bytes, drive_bytes)
            self.assertEqual(local_bytes, hf_bytes)

            # Provenance must ONLY contain attempt 1 seed
            manifest_json = json.loads(local_bytes.decode("utf-8"))
            self.assertEqual(len(manifest_json["episodes"]), 1)
            self.assertEqual(manifest_json["provenance"]["generator_seeds"], [1001])

    def test_uploader_fail_closed_on_auth_and_timeout(self):
        """Uploader must propagate 401 Unauthorized and timeout errors without swallowing."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-1")

            # Auth failure test
            auth_hf = SnapshotHFClient(auth_fail=True)
            auth_uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=auth_hf)
            with self.assertRaises(HfHubHTTPError):
                auth_uploader.sync_episode("ep-1")

            # Timeout failure test
            timeout_hf = SnapshotHFClient(timeout_fail=True)
            timeout_uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=timeout_hf)
            with self.assertRaises(requests.exceptions.Timeout):
                timeout_uploader.sync_episode("ep-1")

    def test_uploader_sync_manifest_requires_referenced_episodes_complete_first(self):
        """Index publication must verify all referenced episodes are complete on Drive and HF first."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-1")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            # Case 1: manifest references non-existent episode
            manifest_file = root / "dataset_manifest.json"
            bad_data = {
                "dataset_track": "exploratory",
                "episodes": [{"episode_id": "ep-missing"}],
            }
            manifest_file.write_text(json.dumps(bad_data))
            with self.assertRaises(FileNotFoundError):
                uploader.sync_manifest(manifest_file)

            # Case 2: referenced episode exists locally, but missing on Drive
            good_data = {
                "dataset_track": "exploratory",
                "episodes": [{"episode_id": "ep-1", "sha256": "abc"}],
            }
            manifest_file.write_text(json.dumps(good_data))
            with self.assertRaisesRegex(ValueError, "missing on Drive"):
                uploader.sync_manifest(manifest_file)

            # Case 3: referenced episode exists on Drive, but missing on HF
            drive_ep = drive / "episodes" / "ep-1"
            drive_ep.mkdir(parents=True)
            local_ep_manifest = root / "episodes" / "ep-1" / "manifest.json"
            (drive_ep / "manifest.json").write_bytes(local_ep_manifest.read_bytes())

            with self.assertRaises(EntryNotFoundError):
                uploader.sync_manifest(manifest_file)

    def test_uploader_sync_manifest_detects_remote_index_conflicts(self):
        """Index publication must reject updates that drop or alter previously published episodes."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-1")
            _write_valid_test_episode(root, "ep-2")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            # Publish ep-1 fully
            uploader.sync_episode("ep-1")
            manifest_1 = {
                "dataset_track": "exploratory",
                "episodes": [{"episode_id": "ep-1", "sha256": "sha-ep1"}],
            }
            manifest_file = root / "dataset_manifest.json"
            manifest_file.write_text(json.dumps(manifest_1))
            uploader.sync_manifest(manifest_file)

            # Publish ep-2 episode files
            uploader.sync_episode("ep-2")

            # Now try to update index with only ep-2 (dropping ep-1)
            manifest_2_conflict = {
                "dataset_track": "exploratory",
                "episodes": [{"episode_id": "ep-2", "sha256": "sha-ep2"}],
            }
            manifest_file.write_text(json.dumps(manifest_2_conflict))

            with self.assertRaisesRegex(ValueError, "drops previously published episode ep-1"):
                uploader.sync_manifest(manifest_file)

    def test_uploader_restart_and_idempotent_resume_and_immutable_guard(self):
        """Uploader resumes matching episodes idempotently, but rejects any modified byte."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-1")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            # First run: publishes ep-1
            rep1 = uploader.sync_episode("ep-1")
            self.assertTrue(rep1["drive_verified"])
            self.assertTrue(rep1["hf_verified"])
            initial_commits = len(hf_client.revisions)

            manifest_file = root / "dataset_manifest.json"
            manifest_data = {
                "dataset_track": "exploratory",
                "episodes": [{"episode_id": "ep-1", "sha256": "sha-ep1"}],
            }
            manifest_file.write_text(json.dumps(manifest_data))
            uploader.sync_manifest(manifest_file)

            # Resume: calling sync_episode("ep-1") again skips Drive copies and re-commits identical files
            rep2 = uploader.sync_episode("ep-1")
            self.assertTrue(rep2["drive_verified"])
            self.assertTrue(rep2["hf_verified"])

            # Mutate an immutable artifact on Drive
            drive_ep_manifest = drive / "episodes" / "ep-1" / "manifest.json"
            drive_ep_manifest.write_bytes(b'{"corrupted": true}')

            with self.assertRaisesRegex(ValueError, "immutable episode artifact differs on Drive"):
                uploader.sync_episode("ep-1")

    def test_uploader_sync_episode_rejects_cas_race_when_remote_head_moves(self):
        """CAS guard rejects episode commit when remote head moves concurrently."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-cas-1")
            hf_client = SnapshotHFClient()
            # Remote has an initial commit
            hf_client.create_commit(
                repo_id="org/repo",
                repo_type="dataset",
                operations=[],
                commit_message="Initial head",
            )
            initial_head = hf_client.head
            self.assertIsNotNone(initial_head)

            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            # Hook _get_head_commit to simulate a concurrent push right after resolving head
            orig_get_head = uploader._get_head_commit
            def moving_head_commit(*args, **kwargs):
                resolved = orig_get_head(*args, **kwargs)
                hf_client.head = "concurrent-commit-sha"
                return resolved
            uploader._get_head_commit = moving_head_commit

            with self.assertRaises(HfHubHTTPError) as cm:
                uploader.sync_episode("ep-cas-1")
            self.assertIn("412 Precondition Failed", str(cm.exception))

    def test_uploader_sync_manifest_rejects_cas_race_when_remote_head_moves(self):
        """CAS guard rejects index commit when remote head moves concurrently."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-cas-2")
            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            uploader.sync_episode("ep-cas-2")

            manifest_file = root / "dataset_manifest.json"
            manifest_data = {
                "dataset_track": "exploratory",
                "episodes": [{"episode_id": "ep-cas-2", "sha256": "sha-cas2"}],
            }
            manifest_file.write_text(json.dumps(manifest_data))

            # Hook _get_head_commit to advance head concurrently during manifest sync
            orig_get_head = uploader._get_head_commit
            def moving_head_commit(*args, **kwargs):
                resolved = orig_get_head(*args, **kwargs)
                hf_client.head = "concurrent-index-head-sha"
                return resolved
            uploader._get_head_commit = moving_head_commit

            with self.assertRaises(HfHubHTTPError) as cm:
                uploader.sync_manifest(manifest_file)
            self.assertIn("412 Precondition Failed", str(cm.exception))

    def test_uploader_rejects_nested_git_allowed_extension_leak(self):
        """Uploader strictly rejects nested directories like .git/leak.json BEFORE any Drive or HF write."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-leak-1")

            # Add allowed-extension file inside nested .git directory
            git_dir = root / "episodes" / "ep-leak-1" / ".git"
            git_dir.mkdir(parents=True)
            (git_dir / "leak.json").write_text('{"leaked": true}', encoding="utf-8")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            with self.assertRaisesRegex(ValueError, "nested directories are forbidden"):
                uploader.sync_episode("ep-leak-1")

            # Verify no Drive files were written
            if drive.exists():
                drive_files = list(drive.rglob("*"))
                self.assertEqual([f for f in drive_files if f.is_file()], [])
            # Verify no HF commits
            self.assertEqual(len(hf_client.commit_calls), 0)

    def test_uploader_rejects_symlink_escape_before_write(self):
        """Uploader strictly rejects symlinks BEFORE any Drive or HF write."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-leak-2")

            # Add a symlink file inside episode directory
            target_outside = Path(td) / "secret.txt"
            target_outside.write_text("secret-data", encoding="utf-8")
            symlink_file = root / "episodes" / "ep-leak-2" / "leak.npz"
            symlink_file.symlink_to(target_outside)

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            with self.assertRaisesRegex(ValueError, "symlink"):
                uploader.sync_episode("ep-leak-2")

            # Verify no Drive files were written
            if drive.exists():
                drive_files = list(drive.rglob("*"))
                self.assertEqual([f for f in drive_files if f.is_file()], [])
            # Verify no HF commits
            self.assertEqual(len(hf_client.commit_calls), 0)

    def test_uploader_rejects_unreferenced_extra_file_before_write(self):
        """Uploader rejects files not explicitly listed in manifest shards BEFORE any Drive or HF write."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "local"
            drive = Path(td) / "drive"
            _write_valid_test_episode(root, "ep-extra")

            # Add an extra unreferenced file with allowed extension
            extra_file = root / "episodes" / "ep-extra" / "extra.npz"
            extra_file.write_bytes(b"extra")

            hf_client = SnapshotHFClient()
            uploader = ExploratoryUploader(root, drive_root=drive, hf_repo_id="org/repo", hf_client=hf_client)

            with self.assertRaisesRegex(ValueError, "unauthorized file in episode archive: extra.npz"):
                uploader.sync_episode("ep-extra")

            if drive.exists():
                drive_files = list(drive.rglob("*"))
                self.assertEqual([f for f in drive_files if f.is_file()], [])
            self.assertEqual(len(hf_client.commit_calls), 0)

    def test_oracle_fails_closed_when_cap_equals_remaining_waypoints(self):
        """Equal-cap with remaining waypoints must fail closed instead of silently truncating."""
        from icgs.environments.rlbench.teleop_oracle import discretize_waypoints_to_commands

        pose0 = np.eye(4, dtype=np.float64)
        pose1 = np.eye(4, dtype=np.float64)
        pose1[0, 3] = 0.05  # dist = 0.05m -> at nominal_speed 0.5, est_duration = 0.1s -> exactly 1 step
        pose2 = np.eye(4, dtype=np.float64)
        pose2[0, 3] = 0.10  # dist = 0.05m -> 1 step

        waypoints = [
            (pose0, 1),
            (pose1, 1),
            (pose2, 1),
        ]
        # max_intervals=1: W0->W1 consumes 1 command, so len(commands)==1 reaches cap before W2.
        # Silently truncating would return [command_1] without error.
        # Fail-closed MUST raise ValueError.
        with self.assertRaisesRegex(ValueError, "expert demonstration exceeds max_intervals; truncation/padding is forbidden"):
            discretize_waypoints_to_commands(
                waypoints,
                nominal_speed=0.5,
                interval_s=0.1,
                max_intervals=1,
            )

    def test_convert_rlbench_raw_observation_rejects_extra_cameras(self):
        """Strict sensor protocol enforces wrist-only and rejects extra camera streams."""
        from icgs.environments.rlbench.controller import convert_rlbench_raw_observation

        valid_raw = {
            "wrist_point_cloud": np.ones((10, 3), dtype=np.float64),
            "gripper_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "gripper_open": 1.0,
        }
        obs = convert_rlbench_raw_observation(valid_raw)
        self.assertEqual(obs.points.shape, (10, 3))

        # Extraneous front camera enabled
        invalid_raw = dict(valid_raw)
        invalid_raw["front_point_cloud"] = np.ones((5, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "unsupported cameras.*front_point_cloud"):
            convert_rlbench_raw_observation(invalid_raw)

        # Extraneous unsupported profile
        with self.assertRaisesRegex(ValueError, "unsupported sensor profile"):
            convert_rlbench_raw_observation(valid_raw, sensor_profile_id="rlbench-multi-view-v1")

    def test_orchestrate_live_expert_demo_enforces_wall_timeout(self):
        """Live expert generation enforces bounded wall timeout across setup, demo query, and staging."""
        import time
        from icgs.environments.rlbench.teleop_oracle import orchestrate_live_expert_demo

        class SlowTaskEnv:
            def get_demos(self, amount=1, live_demos=True):
                time.sleep(0.05)
                return ["mock-demo"]

        with self.assertRaisesRegex(TimeoutError, "expert demonstration generation exceeded wall timeout"):
            orchestrate_live_expert_demo(
                SlowTaskEnv(),
                source_protocol_id="rlbench-live-expert-v1",
                timeout_s=0.01,
            )


if __name__ == "__main__":
    unittest.main()
