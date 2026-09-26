from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    GenerationJob,
    GenerationRuntimeConfig,
    WorkerResult,
)
from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation.episode_archive import (
    EpisodeArchiveReader,
    EpisodeArchiveWriter,
    validate_archive_manifest,
)
from icgs.data.collection.generation.distributed_validation import validate_closed_result
from scripts import generation_worker


def _runtime_config(tmp_path: Path, *, worker_timeout_s: int = 47):
    repo_root = tmp_path / "repo"
    simulator_root = tmp_path / "simulator"
    rlbench_root = tmp_path / "rlbench"
    run_root = tmp_path / "run"
    python_executable = tmp_path / "venv" / "bin" / "python"
    for directory in (repo_root, simulator_root, rlbench_root, run_root, python_executable.parent):
        directory.mkdir(parents=True, exist_ok=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    return GenerationRuntimeConfig.from_dict({
        "machine": {
            "repo_root": str(repo_root),
            "python_executable": str(python_executable),
            "simulator_root": str(simulator_root),
            "rlbench_root": str(rlbench_root),
            "display_base": 37,
            "display_width": 1280,
            "display_height": 720,
            "simulator_slots": 2,
            "worker_timeout_s": worker_timeout_s,
        },
        "run": {
            "run_id": "worker-test",
            "run_root": str(run_root),
            "worker_count": 2,
            "publish_interval_s": 91,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "validation/worker-test",
            "publication_enabled": False,
            "validation_mode": True,
        },
    })


def test_simulator_slot_pool_uses_runtime_config(tmp_path, monkeypatch):
    config = _runtime_config(tmp_path)
    monkeypatch.setenv("ICGS_SIMULATOR_SLOTS", "99")
    try:
        pool = generation_worker.SimulatorSlotPool(tmp_path, config)
    except TypeError as exc:
        pytest.fail(f"SimulatorSlotPool must accept runtime config: {exc}")
    assert pool.slot_count == 2
    with pool:
        assert pool.acquired_slot is not None
        assert pool.acquired_slot.is_file()


def test_shared_filesystem_slot_pool_is_host_local(tmp_path: Path):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["machine"].update({"host_id": "host-a", "worker_ids": ["000"], "simulator_slots": 1})
    payload["run"]["distribution_mode"] = "shared_filesystem"
    host_config = GenerationRuntimeConfig.from_dict(payload)

    pool = generation_worker.SimulatorSlotPool(tmp_path, host_config)

    assert pool.root == tmp_path / "control" / "simulator-slots" / "host-a"


def test_worker_source_has_no_hf_token_or_api_access():
    text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
    assert ".icgs_hf_token" not in text
    assert "HfApi" not in text
    assert "huggingface_hub" not in text


def test_worker_display_number_uses_configured_base_and_worker_limit(tmp_path: Path):
    config = _runtime_config(tmp_path)
    try:
        assert generation_worker.display_number("001", config) == 38
    except TypeError as exc:
        pytest.fail(f"display_number must accept runtime config: {exc}")
    with pytest.raises(ValueError, match="worker_id"):
        generation_worker.display_number("002", config)


def test_worker_rejects_id_outside_host_scope_before_claim(tmp_path: Path):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["machine"].update({"host_id": "host-a", "worker_ids": ["001"], "simulator_slots": 1})
    payload["run"]["distribution_mode"] = "shared_filesystem"
    host_config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(host_config.run.run_root) / "queue")

    with pytest.raises(ValueError, match="host worker scope"):
        generation_worker.run_worker(
            "000", queue, config=host_config,
            approved_manifest=str(tmp_path / "approved.json"), once=True,
        )


def test_worker_cli_requires_runtime_config(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "generation_worker.py",
        "--worker-id", "000",
        "--approved-manifest", "approved.json",
        "--once",
    ])

    with pytest.raises(SystemExit) as exc:
        generation_worker.main()

    assert exc.value.code == 2
    assert "--runtime-config" in capsys.readouterr().err


def test_worker_subprocess_uses_configured_paths_timeout_and_slots(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, worker_timeout_s=33)
    monkeypatch.setenv("ICGS_HF_TOKEN_PATH", "/secret/token")
    monkeypatch.setenv("HF_TOKEN", "secret")
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-1", run_id=config.run.run_id, attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    try:
        generation_worker.run_worker(
            "000",
            queue,
            config=config,
            approved_manifest=str(approved_manifest),
            once=True,
            runner=runner,
        )
    except TypeError as exc:
        pytest.fail(f"run_worker must accept runtime config and injected runner: {exc}")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        config.machine.python_executable,
        "-B",
        str(Path(config.machine.repo_root) / "scripts" / "generation_episode_worker.py"),
        "T01",
    ]
    assert kwargs["timeout"] == 33
    assert kwargs["env"]["ICGS_SIMULATOR_SLOTS"] == "2"
    assert "ICGS_HF_TOKEN_PATH" not in kwargs["env"]
    assert "HF_TOKEN" not in kwargs["env"]
    assert queue.counts().ready == 1


def test_worker_stages_recovered_retry_in_a_disjoint_result_directory(tmp_path: Path):
    config = _runtime_config(tmp_path)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-retry", run_id=config.run.run_id, attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"), retry_generation=1,
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")

    generation_worker.run_worker(
        "000", queue, config=config, approved_manifest=str(approved_manifest), once=True,
        runner=lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="failed", stderr=""),
    )

    result = json.loads((queue.root / "ready" / "job-retry" / "result.json").read_text())
    assert "/worker-results/job-retry/retry-1/T01" in result["result_dir"]


def test_archive_result_detection_requires_canonical_manifest_without_legacy_fallback(tmp_path: Path):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["archive_profile"] = ArchiveProfileConfig(chunk_boundaries=2).as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "episode.manifest.json").write_text(json.dumps({
        "archive_format_id": config.archive_profile.archive_format_id,
        "episode_schema_version": config.archive_profile.episode_schema_version,
        "dataset_identity": config.archive_profile.dataset_identity,
        "archive_kind": "episode",
        "outcome": "valid_failure",
        "timeline": {"observations": 2, "transitions": 1},
    }), encoding="utf-8")

    detected = generation_worker._archive_result_payload(candidate, config)
    assert detected["archive_kind"] == "episode"
    assert detected["outcome"] == "valid_failure"

    (candidate / "episode.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        generation_worker._archive_result_payload(candidate, config)


def _worker_episode_materialization_row(*, outcome: str, depth_frames: list[np.ndarray | None]) -> dict:
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-worker-depth-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={
            "scene_signature": "worker-depth-signature",
            "asset_instance_id": "worker-depth-asset",
            "asset_family_id": "worker-depth-family",
            "camera_profile_id": "rlbench-wrist-depth-v1",
        },
        intervention=None,
    )
    observations = []
    for index, depth in enumerate(depth_frames):
        observation = {
            "points": np.asarray([[index + 0.1, 0.2, 0.3]], dtype=np.float32),
            "point_valid": np.asarray([True], dtype=np.bool_),
            "T_w_e": np.eye(4, dtype=np.float64),
            "grip": index % 2,
            "joint_positions": np.asarray([index], dtype=np.float64),
            "joint_velocities": np.asarray([0.0], dtype=np.float64),
        }
        if depth is not None:
            observation["wrist_depth"] = depth
        observations.append(observation)
    return {
        "program_id": "T01",
        "_plan": plan.as_dict(),
        "_binding": {
            "program_id": "T01",
            "split": "train",
            "asset_family_id": "worker-depth-family",
            "source_lineage_id": "worker-depth-lineage",
            "events": [],
        },
        "success": outcome == "success",
        "result_class": None if outcome in {"success", "valid_failure"} else outcome,
        "_timed_obs": observations,
        "_actions": [np.asarray([0.1, 0.2], dtype=np.float32)],
        "_robot_states": [
            {
                "T_w_e": item["T_w_e"],
                "grip": item["grip"],
                "joint_positions": item["joint_positions"],
                "joint_velocities": item["joint_velocities"],
            }
            for item in observations
        ],
        "_object_states": [{}, {}],
        "_task_labels": {},
    }


def _load_generation_episode_worker_without_simulator(monkeypatch):
    import importlib.util

    package_names = (
        "pyrep",
        "pyrep.objects",
        "rlbench",
        "rlbench.action_modes",
    )
    for name in package_names:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    external_modules = {
        "pyrep.objects.object": {"Object": type("Object", (), {})},
        "pyrep.objects.shape": {"Shape": type("Shape", (), {})},
        "rlbench.action_modes.action_mode": {"MoveArmThenGripper": type("MoveArmThenGripper", (), {})},
        "rlbench.action_modes.arm_action_modes": {"EndEffectorPoseViaIK": type("EndEffectorPoseViaIK", (), {})},
        "rlbench.action_modes.gripper_action_modes": {"Discrete": type("Discrete", (), {})},
        "rlbench.environment": {"Environment": type("Environment", (), {})},
        "rlbench.observation_config": {"ObservationConfig": type("ObservationConfig", (), {})},
    }
    for name, attributes in external_modules.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    script_path = Path("scripts/generation_episode_worker.py").resolve()
    spec = importlib.util.spec_from_file_location("generation_episode_worker_materializer_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_episode_worker_main_with_stub_task(
    monkeypatch,
    tmp_path: Path,
    *,
    failure: str,
    archive_profile: ArchiveProfileConfig | None,
):
    generation_episode_worker = _load_generation_episode_worker_without_simulator(monkeypatch)
    program_id = "T01"
    program = SimpleNamespace(
        program_id=program_id,
        family="basic-manipulation",
        module="stub_task",
        class_name="StubTask",
        routine=[],
        objects={},
        conditions=[],
        events=[],
    )
    generation_episode_worker.compile_generation_catalog = lambda: {program_id: program}
    generation_episode_worker.load_task_class = lambda _spec: type("StubTaskClass", (), {})
    generation_episode_worker.find_shape = lambda _name: None

    class StubObservationConfig:
        def __init__(self):
            self.wrist_camera = SimpleNamespace(point_cloud=False, depth=False)
            self.gripper_pose = False
            self.gripper_open = False
            self.joint_positions = False

        def set_all_high_dim(self, _value):
            pass

        def set_all_low_dim(self, _value):
            pass

    class StubTip:
        def get_position(self):
            return np.asarray([0.0, 0.0, 0.8], dtype=np.float64)

        def get_quaternion(self):
            if failure == "before_initial_capture":
                raise RuntimeError("tip state unavailable")
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    class StubArm:
        def get_tip(self):
            return tip

    def observation(index: int, *, invalid_pose: bool = False):
        pose = (
            np.asarray([index, 0.0], dtype=np.float64)
            if invalid_pose
            else np.asarray([index * 0.01, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )
        return SimpleNamespace(
            wrist_point_cloud=np.asarray(
                [[index * 0.1, 0.0, 0.8], [index * 0.1, 0.1, 0.8]], dtype=np.float32
            ),
            gripper_pose=pose,
            gripper_open=1.0,
            joint_positions=np.full((7,), index, dtype=np.float64),
            joint_velocities=np.full((7,), index / 10.0, dtype=np.float64),
            wrist_depth=np.asarray([[index + 0.125, index + 0.25]], dtype=np.float32),
        )

    class StubTask:
        def __init__(self):
            self.step_count = 0

        def reset(self):
            return ["stub task"], observation(0)

        def step(self, _action):
            self.step_count += 1
            if failure == "step_raises_after_prefix" and self.step_count == 2:
                raise RuntimeError("simulator failed after one captured action")
            if failure == "snapshot_raises_after_action":
                return observation(self.step_count, invalid_pose=True)
            return observation(self.step_count)

        def success(self):
            return False, False

    tip = StubTip()
    task = StubTask()
    environment = SimpleNamespace(
        _scene=SimpleNamespace(robot=SimpleNamespace(arm=StubArm()), task=task),
        get_task=lambda _task_class: task,
        launch=lambda: None,
        shutdown=lambda: None,
    )
    generation_episode_worker.ObservationConfig = StubObservationConfig
    generation_episode_worker.MoveArmThenGripper = lambda **_kwargs: object()
    generation_episode_worker.EndEffectorPoseViaIK = lambda **_kwargs: object()
    generation_episode_worker.Discrete = lambda: object()
    generation_episode_worker.Environment = lambda *_args, **_kwargs: environment

    plan = AttemptPlan(
        program_id=program_id,
        split="train",
        episode_index=1,
        episode_id=f"episode-t01-{failure}",
        episode_kind="nominal",
        scene_seed=7,
        collection_seed=20260920,
        randomization={
            "scene_signature": "stub-signature",
            "asset_instance_id": "stub-asset",
            "object_translation_m": {"x": 0.0, "y": 0.0, "z": 0.0},
            "camera_profile_id": "stub-camera",
        },
        intervention=None,
    )
    from icgs.data.collection.generation import batch

    monkeypatch.setattr(
        batch,
        "plan_program_attempts",
        lambda *_args, **_kwargs: [plan],
    )
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps({
        "program_id": program_id,
        "asset_family_id": "stub-family",
        "source_lineage_id": "stub-lineage",
        "split": "train",
        "randomization": {
            "translation_m": {
                "x": [-0.012, 0.012],
                "y": [-0.012, 0.012],
                "z": [0.0, 0.0],
            },
            "yaw_deg": [-30.0, 30.0],
            "scale": [0.8, 1.2],
            "camera_profile_id": "stub-camera",
        },
    }), encoding="utf-8")
    write_root = tmp_path / "episodes"
    monkeypatch.setattr(sys, "argv", ["generation_episode_worker.py", program_id])
    monkeypatch.setenv("ICGS_GENERATION_WRITE_EPISODE", str(write_root))
    monkeypatch.delenv("ICGS_GENERATION_ATTEMPT_JSON", raising=False)
    monkeypatch.setenv("ICGS_GENERATION_BINDING_JSON", str(binding_path))
    monkeypatch.setenv("ICGS_GENERATION_RUN_ID", "stub-run")
    monkeypatch.setenv("ICGS_GENERATION_CODE_REVISION", "c" * 40)
    monkeypatch.setenv("ICGS_GENERATION_APPROVED_MANIFEST", str(tmp_path / "missing-approved.json"))
    monkeypatch.delenv("ICGS_GENERATION_JOB_IDENTITY", raising=False)
    if archive_profile is None:
        monkeypatch.delenv("ICGS_GENERATION_ARCHIVE_PROFILE", raising=False)
    else:
        monkeypatch.setenv("ICGS_GENERATION_ARCHIVE_PROFILE", json.dumps(archive_profile.as_dict()))

    assert generation_episode_worker.main() == 0
    return write_root / program_id


@pytest.mark.parametrize(
    ("failure", "observations", "actions", "valid_until", "depth_boundaries"),
    [
        ("step_raises_after_prefix", 2, 1, 1, [0, 1]),
        ("snapshot_raises_after_action", 1, 1, 0, [0]),
        ("before_initial_capture", 0, 0, None, []),
    ],
)
def test_archive_episode_worker_main_writes_only_the_captured_crash_prefix(
    tmp_path: Path,
    monkeypatch,
    failure: str,
    observations: int,
    actions: int,
    valid_until: int | None,
    depth_boundaries: list[int],
):
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    result_dir = _run_episode_worker_main_with_stub_task(
        monkeypatch, tmp_path, failure=failure, archive_profile=profile
    )

    manifest_path = result_dir / "attempt.manifest.json"
    assert manifest_path.is_file()
    assert validate_archive_manifest(manifest_path)["valid"] is True
    archive = EpisodeArchiveReader(manifest_path)
    assert archive.manifest.payload["outcome"] == "simulator_crash"
    assert archive.manifest.payload["timeline"] == {
        "observations": observations,
        "transitions": actions,
        "valid_observation_until": valid_until,
    }
    attempt = archive.to_episode_record()
    assert attempt["outcome"] == "simulator_crash"
    assert attempt["episode_id"] is None
    assert attempt["terminal_t"] == (actions or None)
    assert attempt["valid_observation_until"] == valid_until
    arrays = archive.raw_arrays
    assert arrays["actions"].shape[0] == actions
    if actions:
        np.testing.assert_array_equal(
            arrays["actions"],
            np.asarray([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        )
    if observations:
        assert arrays["T_w_e"].shape[0] == observations
        assert arrays["point_offsets"].shape == (observations + 1,)
        np.testing.assert_array_equal(arrays["wrist_depth_frame_boundaries"], depth_boundaries)
        np.testing.assert_array_equal(
            arrays["wrist_depth_frames"][:, 0, :],
            np.asarray([[index + 0.125, index + 0.25] for index in depth_boundaries], dtype=np.float32),
        )
    else:
        assert "points" not in arrays
        assert "wrist_depth_frames" not in arrays

    debug_text = (result_dir / "debug.json").read_text(encoding="utf-8")
    debug = json.loads(debug_text)
    assert "points" not in debug_text
    assert "wrist_depth" not in debug_text
    assert "_timed_obs" not in debug
    assert "_actions" not in debug


def test_legacy_episode_worker_exception_keeps_v2_empty_prefix_output(
    tmp_path: Path,
    monkeypatch,
):
    result_dir = _run_episode_worker_main_with_stub_task(
        monkeypatch, tmp_path, failure="step_raises_after_prefix", archive_profile=None
    )

    assert (result_dir / "attempt.json").is_file()
    with np.load(result_dir / "valid_prefix.npz", allow_pickle=False) as prefix:
        assert prefix["actions"].shape == (0,)
        assert prefix["points"].shape == (0, 3)
        np.testing.assert_array_equal(prefix["point_offsets"], np.asarray([0], dtype=np.int64))
        assert prefix["T_w_e"].shape == (0,)


@pytest.mark.parametrize("outcome", ["success", "valid_failure"])
def test_archive_profile_episode_materializer_preserves_captured_wrist_depth(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
):
    generation_episode_worker = _load_generation_episode_worker_without_simulator(monkeypatch)

    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    monkeypatch.setenv("ICGS_GENERATION_ARCHIVE_PROFILE", json.dumps(profile.as_dict()))
    monkeypatch.delenv("ICGS_GENERATION_BINDING_JSON", raising=False)
    monkeypatch.delenv("ICGS_GENERATION_JOB_IDENTITY", raising=False)
    depth = np.asarray([[0.125, 0.25], [0.5, 0.75]], dtype=np.float32)
    output = tmp_path / f"episode-{outcome}"

    generation_episode_worker._write_episode(
        output,
        _worker_episode_materialization_row(outcome=outcome, depth_frames=[None, depth]),
    )

    reader = EpisodeArchiveReader(output / "episode.manifest.json")
    np.testing.assert_array_equal(reader.raw_arrays["wrist_depth_frames"], depth[np.newaxis, ...])
    np.testing.assert_array_equal(
        reader.raw_arrays["wrist_depth_frame_boundaries"], np.asarray([1], dtype=np.int64)
    )


def test_archive_profile_episode_materializer_omits_unavailable_wrist_depth(
    tmp_path: Path,
    monkeypatch,
):
    generation_episode_worker = _load_generation_episode_worker_without_simulator(monkeypatch)

    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    monkeypatch.setenv("ICGS_GENERATION_ARCHIVE_PROFILE", json.dumps(profile.as_dict()))
    monkeypatch.delenv("ICGS_GENERATION_BINDING_JSON", raising=False)
    monkeypatch.delenv("ICGS_GENERATION_JOB_IDENTITY", raising=False)
    output = tmp_path / "episode-no-depth"

    generation_episode_worker._write_episode(
        output,
        _worker_episode_materialization_row(outcome="success", depth_frames=[None, None]),
    )

    reader = EpisodeArchiveReader(output / "episode.manifest.json")
    assert "wrist_depth_frames" not in reader.raw_arrays
    assert "wrist_depth_frame_boundaries" not in reader.raw_arrays


def test_archive_profile_attempt_materializer_preserves_captured_wrist_depth_prefix(
    tmp_path: Path,
    monkeypatch,
):
    generation_episode_worker = _load_generation_episode_worker_without_simulator(monkeypatch)

    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    monkeypatch.setenv("ICGS_GENERATION_ARCHIVE_PROFILE", json.dumps(profile.as_dict()))
    monkeypatch.delenv("ICGS_GENERATION_BINDING_JSON", raising=False)
    monkeypatch.delenv("ICGS_GENERATION_JOB_IDENTITY", raising=False)
    depth = np.asarray([[0.125, 0.25], [0.5, 0.75]], dtype=np.float32)
    output = tmp_path / "attempt-depth"

    generation_episode_worker._write_episode(
        output,
        _worker_episode_materialization_row(outcome="simulator_crash", depth_frames=[None, depth]),
    )

    reader = EpisodeArchiveReader(output / "attempt.manifest.json")
    np.testing.assert_array_equal(reader.raw_arrays["wrist_depth_frames"], depth[np.newaxis, ...])
    np.testing.assert_array_equal(
        reader.raw_arrays["wrist_depth_frame_boundaries"], np.asarray([1], dtype=np.int64)
    )


def test_archive_profile_worker_propagates_identity_and_writes_canonical_failure(tmp_path: Path):
    base = _runtime_config(tmp_path)
    payload = base.as_dict()
    payload["archive_profile"] = ArchiveProfileConfig(chunk_boundaries=2).as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-archive-000001", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920, randomization={"scene_signature": "signature-1", "asset_instance_id": "asset-1"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-archive", run_id=config.run.run_id, attempt_id=f"att-{plan.episode_id}",
        episode_id=plan.episode_id, program_id="T01", plan=plan,
        code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=1, stdout="worker stdout", stderr="worker stderr")

    generation_worker.run_worker(
        "000", queue, config=config, approved_manifest=str(approved_manifest), once=True, runner=runner
    )

    assert calls[0]["env"]["ICGS_GENERATION_RUN_ID"] == job.run_id
    assert calls[0]["env"]["ICGS_GENERATION_CODE_REVISION"] == job.code_revision
    assert calls[0]["env"]["ICGS_GENERATION_ARCHIVE_PROFILE"]
    archive_job_identity = json.loads(calls[0]["env"]["ICGS_GENERATION_JOB_IDENTITY"])
    assert archive_job_identity["manifest_sha256"] == job.manifest_sha256
    assert archive_job_identity["plan"] == job.plan.as_dict()
    result = json.loads((queue.root / "ready" / job.job_id / "result.json").read_text())
    assert Path(result["result_dir"], "attempt.manifest.json").is_file()
    assert not Path(result["result_dir"], "attempt.json").exists()
    attempt_archive = EpisodeArchiveReader(Path(result["result_dir"]) / "attempt.manifest.json")
    assert attempt_archive.debug_metadata["job_identity"]["job_id"] == job.job_id
    validated = validate_closed_result(
        job,
        WorkerResult.from_dict(result),
        archive_profile=config.archive_profile,
    )
    assert validated.attempt_entry is not None
    assert validated.attempt_entry["outcome"] == "simulator_crash"


@pytest.mark.parametrize("outcome", ["simulator_crash", "invalid_observation"])
def test_archive_worker_preserves_closed_attempt_when_runner_raises(
    tmp_path: Path, outcome: str
):
    base = _runtime_config(tmp_path)
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    payload = base.as_dict()
    payload["archive_profile"] = profile.as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-closed-attempt", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-closed", "asset_instance_id": "asset-closed"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-closed-attempt", run_id=config.run.run_id,
        attempt_id=f"att-{plan.episode_id}", episode_id=plan.episode_id, program_id="T01",
        plan=plan, code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    measured_actions = np.asarray([[0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    measured_points = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    def runner(command, **kwargs):
        candidate = Path(kwargs["env"]["ICGS_GENERATION_WRITE_EPISODE"]) / job.program_id
        EpisodeArchiveWriter(profile).write_attempt(
            {
                "attempt_id": job.attempt_id,
                "episode_id": None,
                "program_id": job.program_id,
                "outcome": outcome,
                "valid_observation_until": 1,
            },
            prefix_arrays={
                "actions": measured_actions,
                "points": measured_points,
                "point_offsets": np.asarray([0, 1, 2], dtype=np.int64),
                "T_w_e": np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
            },
            debug_metadata={
                "source_run_id": job.run_id,
                "code_revision": job.code_revision,
                "preprocessing_identity": "runner_closed_attempt_v1",
                "stdout": "attempt archive closed before the runner exception",
            },
            output_dir=candidate,
        )
        raise subprocess.TimeoutExpired(command, 33, output="timed out after close")

    generation_worker.run_worker(
        "000", queue, config=config, approved_manifest=str(approved_manifest),
        once=True, runner=runner,
    )

    result = json.loads((queue.root / "ready" / job.job_id / "result.json").read_text())
    result_dir = Path(result["result_dir"])
    manifest_path = result_dir / "attempt.manifest.json"
    assert result["outcome"] == outcome
    assert manifest_path.is_file()
    assert validate_archive_manifest(manifest_path)["valid"] is True
    archive = EpisodeArchiveReader(manifest_path)
    np.testing.assert_array_equal(archive.raw_arrays["actions"], measured_actions)
    np.testing.assert_array_equal(archive.raw_arrays["points"], measured_points)
    assert archive.debug_metadata["stdout"] == "attempt archive closed before the runner exception"


@pytest.mark.parametrize("outcome", ["simulator_crash", "invalid_observation"])
def test_archive_worker_keeps_closed_attempt_when_publication_fails(
    tmp_path: Path, monkeypatch, outcome: str
):
    base = _runtime_config(tmp_path)
    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="keep")
    payload = base.as_dict()
    payload["archive_profile"] = profile.as_dict()
    config = GenerationRuntimeConfig.from_dict(payload)
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-publish-failure", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "signature-publish", "asset_instance_id": "asset-publish"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-publish-failure", run_id=config.run.run_id,
        attempt_id=f"att-{plan.episode_id}", episode_id=plan.episode_id, program_id="T01",
        plan=plan, code_revision="a" * 40, manifest_sha256="b" * 64,
        output_root=str(Path(config.run.run_root) / "staging"),
    )
    queue.enqueue(job)
    approved_manifest = Path(config.run.run_root) / "approved.json"
    approved_manifest.write_text(json.dumps({"catalog": [{"program_id": "T01"}]}), encoding="utf-8")
    measured_actions = np.asarray([[0.25, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    measured_points = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

    def runner(command, **kwargs):
        candidate = Path(kwargs["env"]["ICGS_GENERATION_WRITE_EPISODE"]) / job.program_id
        EpisodeArchiveWriter(profile).write_attempt(
            {
                "attempt_id": job.attempt_id,
                "episode_id": None,
                "program_id": job.program_id,
                "outcome": outcome,
                "valid_observation_until": 0,
            },
            prefix_arrays={
                "actions": measured_actions,
                "points": measured_points,
                "point_offsets": np.asarray([0, 1], dtype=np.int64),
                "T_w_e": np.eye(4, dtype=np.float64)[None],
            },
            debug_metadata={
                "source_run_id": job.run_id,
                "code_revision": job.code_revision,
                "preprocessing_identity": "runner_publish_failure_v1",
                "traceback": "original closed-attempt traceback",
            },
            output_dir=candidate,
        )
        raise subprocess.TimeoutExpired(command, 33, output="timed out after close")

    def fail_publish(*args, **kwargs):
        raise RuntimeError("publish unavailable")

    monkeypatch.setattr(queue, "publish_ready", fail_publish)

    with pytest.raises(RuntimeError, match="publish unavailable"):
        generation_worker.run_worker(
            "000", queue, config=config, approved_manifest=str(approved_manifest),
            once=True, runner=runner,
        )

    candidate = Path(job.output_root) / "worker-results" / job.job_id / "retry-0" / job.program_id
    manifest_path = candidate / "attempt.manifest.json"
    assert manifest_path.is_file()
    assert validate_archive_manifest(manifest_path)["valid"] is True
    archive = EpisodeArchiveReader(manifest_path)
    assert archive.manifest.payload["outcome"] == outcome
    np.testing.assert_array_equal(archive.raw_arrays["actions"], measured_actions)
    assert archive.debug_metadata["traceback"] == "original closed-attempt traceback"


def test_generation_subprocess_refreshes_heartbeat_until_exit(monkeypatch):
    events = []

    class Process:
        def __init__(self):
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def communicate(self):
            return "stdout", "stderr"

        def kill(self):
            self.poll_count = 2

    process = Process()
    monkeypatch.setattr(generation_worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(generation_worker.time, "sleep", lambda _: None)

    result = generation_worker._run_generation_process(
        ["python", "episode.py"], env={}, timeout_s=10,
        heartbeat=lambda: events.append("heartbeat"), heartbeat_interval_s=1,
    )

    assert result.returncode == 0
    assert len(events) >= 2


def test_worker_invalid_result_hashes_the_candidate_directory():
    text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
    assert "file_sha256=_file_hashes(candidate)" in text
    assert "file_sha256=_file_hashes(result_dir)" in text


def test_worker_exception_path_does_not_publish_empty_inventory():
    text = Path("scripts/generation_worker.py").read_text(encoding="utf-8")
    assert "file_sha256=_file_hashes(result_dir)" in text
    assert "stale candidate" in text
