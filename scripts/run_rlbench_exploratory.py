#!/usr/bin/env python3
"""Fail-closed E01 collection entrypoint.

Authority: ADR0012 and ADR0007. Phase 0 may verify a G1 receipt but cannot
collect data: the controller, expert materializer, archive split, and continuous
publication path remain unapproved. This script therefore creates no output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

logger = logging.getLogger(__name__)

import numpy as np

G1_PINNED_PROTOCOL_ID = "rlbench-g1-coppeliasim-4.1-franka"
G1_RECEIPT_KIND = "g1_api_probe_receipt"
G1_RECEIPT_VERSION = 2
G1_PROBE_IMPLEMENTATION_ID = "g1-api-probe-v2"
G1_PINNED_RLBENCH_REVISION = "02720bba4c73fe02eb75df946b8791b806028a9d"
G1_PINNED_PYREP_REVISION = "8f420be8064b1970aae18a9cfbc978dfb15747ef"
G1_PINNED_COPPELIASIM_VERSION = "4.1.0"
G1_PINNED_COPPELIASIM_ARCHIVE_SHA256 = "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8"
G1_EXPECTED_TIMESTEP_S = 0.05
G1_PHYSICAL_STEPS = 2
G1_JOINT_TARGET_ATOL_RAD = 0.01
G1_GRIPPER_HOLD_ATOL = 0.001
G1_MAX_WALL_TIME_S = 180.0
G1_DEFAULT_TASK = "PickAndLift"
G1_DRIVE_ROOT = Path("/content/drive/MyDrive")


def _validate_e01_parameters(
    *,
    num_attempts: int,
    max_intervals: int,
    wall_time_s: float,
    disk_limit_bytes: int | None = None,
    supervisor_timeout_s: float | None = None,
) -> None:
    """Validate E01 parameters fail-closed before any preflight, worker, or collection."""
    if isinstance(num_attempts, bool) or not isinstance(num_attempts, int) or num_attempts != 1:
        raise ValueError(
            f"E01 exploratory smoke requires num_attempts == 1, got {num_attempts!r}"
        )
    if (
        isinstance(max_intervals, bool)
        or not isinstance(max_intervals, int)
        or not (1 <= max_intervals <= 1024)
    ):
        raise ValueError(
            f"E01 exploratory smoke requires max_intervals to be an integer in 1..1024, got {max_intervals!r}"
        )
    if (
        isinstance(wall_time_s, bool)
        or not isinstance(wall_time_s, (int, float))
        or not math.isfinite(wall_time_s)
        or wall_time_s <= 0
    ):
        raise ValueError(f"wall_time_s must be a positive finite number, got {wall_time_s!r}")
    if supervisor_timeout_s is not None:
        if (
            isinstance(supervisor_timeout_s, bool)
            or not isinstance(supervisor_timeout_s, (int, float))
            or not math.isfinite(supervisor_timeout_s)
            or supervisor_timeout_s <= 0
        ):
            raise ValueError(
                f"supervisor_timeout_s must be a positive finite number, got {supervisor_timeout_s!r}"
            )
    if disk_limit_bytes is not None:
        if (
            isinstance(disk_limit_bytes, bool)
            or not isinstance(disk_limit_bytes, int)
            or disk_limit_bytes <= 0
        ):
            raise ValueError(
                f"disk_limit_bytes must be a positive integer, got {disk_limit_bytes!r}"
            )


def _validate_drive_dir_constraint(drive_dir: Path | str | None) -> Path | None:
    """Syntactically constrain drive_dir under mounted root before any mirrored outcome."""
    if drive_dir is None:
        return None
    raw_path = Path(drive_dir)
    if raw_path.is_symlink():
        raise ValueError(f"drive_dir cannot be a symlink: {drive_dir}")
    path = raw_path.resolve()
    if path.is_symlink():
        raise ValueError(f"drive_dir cannot be a symlink: {drive_dir}")
    if path == Path("/"):
        raise ValueError("drive_dir cannot be the filesystem root '/'")
    if not raw_path.is_absolute():
        raise ValueError(f"drive_dir must be an absolute path, got: {drive_dir}")

    content_dir = Path("/content")
    if content_dir.is_dir():
        drive_root = G1_DRIVE_ROOT.resolve()
        if not drive_root.is_dir():
            raise RuntimeError(f"Mounted Google Drive is unavailable at '{drive_root}'")
        try:
            rel = path.relative_to(drive_root)
            if not rel.parts:
                raise ValueError(f"drive_dir cannot be the mounted Drive root itself: '{path}'")
        except ValueError as exc:
            raise ValueError(
                f"drive_dir '{path}' must be strictly under mounted Google Drive root '{drive_root}'"
            ) from exc
    return path


def _read_secure_hf_token_file(token_path: Path | str | None) -> str | None:
    """Read Hugging Face authentication token securely from a validated local file."""
    if token_path is None:
        return None
    raw_path = Path(token_path)
    if raw_path.is_symlink():
        raise ValueError(f"HF token file must not be a symlink: '{raw_path}'")
    path = raw_path.resolve()
    if not path.is_file() or raw_path.is_symlink() or path.is_symlink():
        raise ValueError(f"HF token file must be a regular non-symlink file: '{path}'")
    try:
        st = raw_path.stat()
        if st.st_mode & 0o004:
            logger.warning("HF token file %s is world-readable; permissions should be restricted", raw_path)
    except OSError:
        pass
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"HF token file at '{path}' is empty")
    return token


def _diagnostic() -> str:
    return (
        "[ADR0012 and ADR0007] G1 receipt is required before launching RLBench or writing exploratory data. "
        "Run the bounded G1 probe first: 'python3 scripts/colab_g1_api_probe.py --output-dir <drive_dir>'."
    )


def _reject(detail: str) -> None:
    raise RuntimeError(f"{_diagnostic()} {detail}")


def _normalise_task_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(f"Receipt {label} must be finite numeric evidence.")
    number = float(value)
    if not math.isfinite(number):
        _reject(f"Receipt {label} must be finite numeric evidence.")
    return number


def _require_utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        _reject(f"Receipt {label} must be a non-empty UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _reject(f"Receipt {label} is not an ISO-8601 timestamp.")
    if parsed.tzinfo is None:
        _reject(f"Receipt {label} must include a UTC offset.")
    return parsed


def _require_true_fields(payload: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    if any(payload.get(field) is not True for field in fields):
        _reject(f"Receipt is missing required {label} evidence.")


def verify_g1_receipt(
    receipt_path: Path | str | None,
    *,
    expected_protocol: str = G1_PINNED_PROTOCOL_ID,
    expected_task: str = G1_DEFAULT_TASK,
) -> dict[str, Any]:
    """Validate the full versioned G1 receipt before any collection side effect.

    This checks the receipt's internally recorded evidence; it is deliberately
    not an authentication mechanism. Phase 0 still refuses collection after
    validation, so a receipt can never authorize the unrepaired collector.
    """
    if receipt_path is None:
        _reject("Receipt path was not provided.")
    path = Path(receipt_path).resolve()
    if not path.is_file():
        _reject(f"Receipt file does not exist at '{path}'.")
    drive_root = G1_DRIVE_ROOT.resolve()
    if not drive_root.is_dir():
        _reject(f"Mounted Google Drive is unavailable at '{drive_root}'.")
    try:
        path.relative_to(drive_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{_diagnostic()} Receipt path '{path}' is not under mounted Google Drive '{drive_root}'."
        ) from exc
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{_diagnostic()} Failed to parse receipt JSON from '{path}': {exc}") from exc
    if not isinstance(data, dict):
        _reject(f"Receipt at '{path}' must be a JSON object.")

    if data.get("kind") != G1_RECEIPT_KIND:
        _reject(f"Receipt kind must be {G1_RECEIPT_KIND!r}.")
    if data.get("receipt_version") != G1_RECEIPT_VERSION:
        _reject(f"Receipt version must be {G1_RECEIPT_VERSION!r}.")
    if data.get("probe_implementation_id") != G1_PROBE_IMPLEMENTATION_ID:
        _reject(f"Receipt probe implementation must be {G1_PROBE_IMPLEMENTATION_ID!r}.")
    if data.get("authority") != "ADR0012 and ADR0007":
        _reject("Receipt authority does not match ADR0012 and ADR0007.")
    if data.get("status") != "PASS":
        _reject(f"Receipt declares status={data.get('status')!r} (expected 'PASS'). Error: {data.get('error')}")
    if data.get("protocol_id") != expected_protocol:
        _reject(
            f"Pinned protocol mismatch: found {data.get('protocol_id')!r}, expected {expected_protocol!r}."
        )

    started_at = _require_utc_timestamp(data.get("started_at_utc"), "started_at_utc")
    completed_at = _require_utc_timestamp(data.get("completed_at_utc"), "completed_at_utc")
    if completed_at < started_at:
        _reject("Receipt completion timestamp predates its start timestamp.")

    task_name = data.get("task_name")
    resolved_task_name = data.get("resolved_task_name")
    if not isinstance(task_name, str) or not task_name or not isinstance(resolved_task_name, str) or not resolved_task_name:
        _reject("Receipt must contain requested and resolved task identity strings.")
    if _normalise_task_name(task_name) != _normalise_task_name(resolved_task_name):
        _reject(f"Receipt task identity mismatch: requested {task_name!r}, resolved {resolved_task_name!r}.")
    if _normalise_task_name(task_name) != _normalise_task_name(expected_task):
        _reject(f"Receipt task identity mismatch: found {task_name!r}, expected {expected_task!r}.")

    environment = data.get("environment")
    if not isinstance(environment, dict) or not all(
        isinstance(environment.get(field), str) and environment[field]
        for field in ("outer_env_class", "task_env_class")
    ):
        _reject("Receipt is missing outer/task environment identity evidence.")

    clock = data.get("clock")
    if not isinstance(clock, dict) or clock.get("clock_advanced") is not True:
        _reject("Receipt does not declare an advancing live clock.")
    dt_s = _finite_number(clock.get("dt_s"), "clock.dt_s")
    before_s = _finite_number(clock.get("sim_time_before_s"), "clock.sim_time_before_s")
    after_s = _finite_number(clock.get("sim_time_after_s"), "clock.sim_time_after_s")
    elapsed_s = _finite_number(clock.get("elapsed_s"), "clock.elapsed_s")
    if dt_s <= 0.0 or before_s < 0.0 or not after_s > before_s:
        _reject("Receipt clock has an invalid positive-time ordering.")
    if not math.isclose(dt_s, G1_EXPECTED_TIMESTEP_S, rel_tol=1e-4, abs_tol=1e-6):
        _reject(f"Receipt timestep {dt_s} does not match pinned {G1_EXPECTED_TIMESTEP_S} seconds.")
    if clock.get("physics_steps") != G1_PHYSICAL_STEPS or clock.get("scene_steps") != G1_PHYSICAL_STEPS:
        _reject("Receipt must prove exactly two physical scene steps.")
    if not math.isclose(elapsed_s, after_s - before_s, rel_tol=1e-4, abs_tol=1e-6):
        _reject("Receipt elapsed clock value disagrees with its before/after timestamps.")
    if not math.isclose(elapsed_s, dt_s * G1_PHYSICAL_STEPS, rel_tol=1e-4, abs_tol=1e-6):
        _reject("Receipt elapsed clock value disagrees with two queried physical timesteps.")

    apis = data.get("apis")
    if not isinstance(apis, dict):
        _reject("Receipt is missing pinned control API evidence.")
    _require_true_fields(
        apis,
        (
            "outer_env_shutdown",
            "task_reset",
            "task_get_demos",
            "task_scene_step",
            "robot_arm_get_tip",
            "robot_arm_joint_positions",
            "robot_arm_ik",
            "robot_arm_set_joint_targets",
            "robot_gripper",
            "robot_gripper_actuate",
        ),
        "pinned control API",
    )

    target_hold = data.get("target_hold")
    if not isinstance(target_hold, dict):
        _reject("Receipt is missing required target hold evidence.")
    _require_true_fields(
        target_hold,
        ("target_from_live_tip", "ik_solution_applied", "target_hold_verified"),
        "target hold",
    )
    if target_hold.get("scene_steps") != G1_PHYSICAL_STEPS or target_hold.get("gripper_actuation_calls") != G1_PHYSICAL_STEPS:
        _reject("Receipt target hold does not prove two commanded scene/gripper steps.")
    joint_error = _finite_number(target_hold.get("joint_target_error_linf_rad"), "target_hold.joint_target_error_linf_rad")
    joint_tolerance = _finite_number(target_hold.get("joint_target_atol_rad"), "target_hold.joint_target_atol_rad")
    gripper_target = _finite_number(target_hold.get("gripper_hold_target"), "target_hold.gripper_hold_target")
    gripper_error = _finite_number(target_hold.get("gripper_hold_error_linf"), "target_hold.gripper_hold_error_linf")
    gripper_tolerance = _finite_number(target_hold.get("gripper_hold_atol"), "target_hold.gripper_hold_atol")
    if not math.isclose(joint_tolerance, G1_JOINT_TARGET_ATOL_RAD, rel_tol=0.0, abs_tol=1e-12):
        _reject("Receipt target hold joint tolerance does not match the pinned G1 protocol.")
    if joint_error < 0.0 or joint_error > joint_tolerance:
        _reject("Receipt target hold joint error is not within its positive tolerance.")
    if not 0.0 <= gripper_target <= 1.0:
        _reject("Receipt target hold gripper target is outside [0, 1].")
    if not math.isclose(gripper_tolerance, G1_GRIPPER_HOLD_ATOL, rel_tol=0.0, abs_tol=1e-12):
        _reject("Receipt target hold gripper tolerance does not match the pinned G1 protocol.")
    if gripper_error < 0.0 or gripper_error > gripper_tolerance:
        _reject("Receipt target hold gripper error is not within its positive tolerance.")

    raw_cloud = data.get("raw_cloud")
    if not isinstance(raw_cloud, dict):
        _reject("Receipt is missing raw-cloud evidence.")
    _require_true_fields(raw_cloud, ("points_found", "points_finite"), "finite raw-cloud")
    point_shape = raw_cloud.get("points_shape")
    point_count = raw_cloud.get("point_count")
    point_dtype = raw_cloud.get("points_dtype")
    if (
        not isinstance(point_shape, list)
        or len(point_shape) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in point_shape)
        or point_shape[-1] != 3
        or isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or point_count != math.prod(point_shape[:-1])
        or point_dtype not in ("float32", "float64")
    ):
        _reject("Receipt raw-cloud shape, count, or floating dtype evidence is invalid.")

    demonstrations = data.get("demonstrations")
    if (
        not isinstance(demonstrations, dict)
        or demonstrations.get("get_demos_supported") is not True
        or not isinstance(demonstrations.get("live_demo_executed"), bool)
        or isinstance(demonstrations.get("demo_count"), bool)
        or not isinstance(demonstrations.get("demo_count"), int)
        or demonstrations["demo_count"] < 0
    ):
        _reject("Receipt demonstration-availability evidence is invalid.")

    wall_time = data.get("wall_time")
    if not isinstance(wall_time, dict):
        _reject("Receipt is missing bounded wall-time evidence.")
    wall_limit_s = _finite_number(wall_time.get("max_s"), "wall_time.max_s")
    wall_elapsed_s = _finite_number(wall_time.get("elapsed_s"), "wall_time.elapsed_s")
    if not 0.0 < wall_limit_s <= G1_MAX_WALL_TIME_S or not 0.0 <= wall_elapsed_s <= wall_limit_s:
        _reject("Receipt wall-time evidence is outside the bounded G1 limit.")

    cleanup = data.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("outer_env_shutdown") is not True:
        _reject("Receipt does not prove clean shutdown of the outer Environment.")

    upstream = data.get("upstream")
    if not isinstance(upstream, dict):
        _reject("Receipt is missing pinned upstream evidence.")
    expected_upstream = {
        "rlbench": ("revision", G1_PINNED_RLBENCH_REVISION),
        "pyrep": ("revision", G1_PINNED_PYREP_REVISION),
        "coppeliasim": ("version", G1_PINNED_COPPELIASIM_VERSION),
    }
    for component, (field, expected) in expected_upstream.items():
        evidence = upstream.get(component)
        if not isinstance(evidence, dict) or evidence.get(field) != expected:
            _reject(f"Receipt {component} {field} does not match the pinned value {expected!r}.")
    for component in ("rlbench", "pyrep"):
        evidence = upstream[component]
        if evidence.get("source_tree_clean") is not True or not isinstance(evidence.get("source_root"), str) or not evidence["source_root"]:
            _reject(f"Receipt {component} source-tree provenance is incomplete.")
    coppeliasim = upstream["coppeliasim"]
    if (
        coppeliasim.get("archive_sha256") != G1_PINNED_COPPELIASIM_ARCHIVE_SHA256
        or not isinstance(coppeliasim.get("simulator_root"), str)
        or not coppeliasim["simulator_root"]
    ):
        _reject("Receipt CoppeliaSim archive/root provenance does not match the pinned distribution.")

    return data


def _get_git_info() -> tuple[str, bool, str | None]:
    """Extract code revision and dirty status digest from git.

    Fails closed: raises RuntimeError on any git failure.
    Hashes both tracked diff (git diff HEAD) and untracked scoped source bytes
    relative to the actual repository root, not the caller's working directory.
    """
    import subprocess
    import hashlib

    repo_root = Path(__file__).resolve().parent.parent
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        ).decode("utf-8").strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve git HEAD revision in {repo_root}: {exc}") from exc

    try:
        tracked_diff = subprocess.check_output(
            ["git", "diff", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        )
        untracked_output = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            stderr=subprocess.PIPE,
        ).decode("utf-8")
        untracked_files = [line.strip() for line in untracked_output.splitlines() if line.strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect git working tree status in {repo_root}: {exc}") from exc

    is_dirty = bool(tracked_diff or untracked_files)
    if not is_dirty:
        return rev, False, None

    hasher = hashlib.sha256()
    hasher.update(tracked_diff)
    for rel_path in sorted(untracked_files):
        file_path = repo_root / rel_path
        if file_path.is_file():
            hasher.update(rel_path.encode("utf-8") + b"\0")
            hasher.update(file_path.read_bytes())

    patch_digest = hasher.hexdigest()
    return rev, True, patch_digest


def make_concrete_rlbench_environment(task_name: str = G1_DEFAULT_TASK, headless: bool = True) -> Any:
    """Instantiate a live RLBench environment wrapped in TimedController and adapter."""
    try:
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        import rlbench.tasks as rlbench_tasks
    except ImportError as err:
        raise RuntimeError(
            f"Pinned RLBench simulator environment is not importable: {err}. "
            "Real environment execution requires pinned RLBench/PyRep in runtime."
        ) from err
    from icgs.environments.rlbench.controller import RLBenchTimedController
    from icgs.environments.rlbench.timed import TimedRLBenchAdapter

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(False)
    obs_config.wrist_camera.point_cloud = True
    obs_config.wrist_camera.depth = True
    obs_config.gripper_pose = True
    obs_config.gripper_open = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete(),
    )
    outer_env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
    )
    cleaned = False
    try:
        outer_env.launch()
        task_class = getattr(rlbench_tasks, task_name, None)
        if task_class is None:
            raise RuntimeError(f"Unknown RLBench task: {task_name}")
        task_env = outer_env.get_task(task_class)
        controller = RLBenchTimedController(task_env, outer_env=outer_env)
        return TimedRLBenchAdapter(controller, physics_dt=0.05, sensor_profile_id="rlbench-wrist-depth-raw-v1")
    except BaseException:
        if not cleaned:
            cleaned = True
            try:
                outer_env.shutdown()
            except Exception as shutdown_err:
                logger.warning("Error during outer_env.shutdown() after construction failure: %s", shutdown_err)
        raise


def run_exploratory_collection(
    output_dir: Path,
    *,
    g1_receipt_path: Path | str | None = None,
    drive_dir: Path | None = None,
    hf_repo_id: str | None = None,
    hf_client: Any = None,
    hf_token_file: Path | str | None = None,
    num_attempts: int = 1,
    max_intervals: int = 16,
    wall_time_s: float = 60.0,
    disk_limit_bytes: int = 1024 * 1024 * 1024,
    task_name: str = G1_DEFAULT_TASK,
    use_mock_for_testing: bool = False,
    enable_phase34: bool = True,
    specs: Any = None,
    environment_factory: Any = None,
    config: Any = None,
    protocol_manifest: Any = None,
    code_revision: str | None = None,
    generator_version: str | None = None,
    is_dirty: bool | None = None,
    dirty_patch_digest: str | None = None,
    limits: Any = None,
    command_factory: Any = None,
    episode_publisher: Any = None,
    index_publisher: Any = None,
    online_provider: Any = None,
    result_file: Path | str | None = None,
    nonce: str | None = None,
) -> int:
    """Run concrete bounded E01 exploratory collection with live expert demonstration.

    Requires a valid strict PASS G1 receipt. Concretely composes AttemptSpecs,
    live RLBench environment adapter, live expert demonstration oracle with
    reset_to_demo staging, uncropped raw measured data, and dual Drive/HF publication.
    """
    _validate_e01_parameters(
        num_attempts=num_attempts,
        max_intervals=max_intervals,
        wall_time_s=wall_time_s,
        disk_limit_bytes=disk_limit_bytes,
    )
    verify_g1_receipt(g1_receipt_path, expected_task=task_name)

    if use_mock_for_testing:
        raise RuntimeError(
            "[ADR0012 and ADR0007] Phase 0 invariant: mock collection is refused because a quarantined/"
            "failed mock report cannot certify E01 physical collection or publication."
        )

    # Online observation provider default
    if online_provider is None:
        def _default_online_provider(step: Any) -> Mapping[str, Any]:
            obs = step.observation if hasattr(step, "observation") else step.after.observation
            pts = obs.points
            t_w_e = obs.T_w_e
            return {
                "points": pts.astype(np.float32) if hasattr(pts, "astype") else np.array(pts, dtype=np.float32),
                "point_valid": np.ones(len(pts), dtype=bool),
                "T_w_e": t_w_e.astype(np.float32) if hasattr(t_w_e, "astype") else np.array(t_w_e, dtype=np.float32),
                "grip": float(obs.grip),
            }
        online_provider = _default_online_provider

    # Validate Drive and HF targets when publishers are not explicitly supplied
    if episode_publisher is None:
        if drive_dir is None or hf_repo_id is None:
            _reject("Phase 3-4 publication requires both an explicit mounted Drive root and HF dataset repository.")
        from icgs.data.collection.uploader import ExploratoryUploader
        hf_token = _read_secure_hf_token_file(hf_token_file) if hf_token_file else None
        uploader = ExploratoryUploader(
            output_dir,
            drive_root=drive_dir,
            hf_repo_id=hf_repo_id,
            hf_client=hf_client,
            hf_token=hf_token,
        )
        del hf_token
        # Explicit preflight before any simulator construction, demo, or local disk writes:
        # Verifies Drive containment/probe and HF authenticated capability without creating dummy commits.
        uploader.verify_publication_preflight()
        episode_publisher = uploader.sync_episode
        if index_publisher is None:
            index_publisher = uploader.sync_manifest
    elif hasattr(episode_publisher, "__self__") and hasattr(episode_publisher.__self__, "verify_publication_preflight"):
        episode_publisher.__self__.verify_publication_preflight()

    # Ensure dataset_manifest.json exists with valid exploratory schema
    manifest_path = output_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_manifest = {
            "manifest_version": 1,
            "dataset_track": "exploratory",
            "episodes": [],
            "lineage": [
                {"lineage_id": "exploratory_rlbench_dev", "split": "dev", "parent_ids": []}
            ],
            "asset_families": [
                {"asset_family_id": "rlbench-exploratory", "split": "dev"}
            ],
        }
        manifest_path.write_text(json.dumps(initial_manifest, indent=2), encoding="utf-8")

    # Concrete AttemptSpecs composition
    if specs is None:
        from icgs.data.collection.runner import AttemptSpec
        specs = [
            AttemptSpec(
                episode_id=f"e01-ep-{idx:04d}",
                program_id="E01",
                source_lineage_id="exploratory_rlbench_dev",
                asset_family_id="rlbench-exploratory",
                split="dev",
                generator_seed=1000 + idx,
                reset_seed=2000 + idx,
                action_seed=3000 + idx,
                calibration_id="rlbench-wrist-depth-raw-v1",
                observation_origin="measured",
                raw_commands_id="rlbench-live-expert-v1",
                materialized_commands_id=f"rlbench-live-mat-{idx:04d}",
            )
            for idx in range(num_attempts)
        ]

    # Concrete Environment Factory composition
    if environment_factory is None:
        def _default_env_factory(spec: Any) -> Any:
            return make_concrete_rlbench_environment(task_name=task_name, headless=True)
        environment_factory = _default_env_factory

    # Concrete Command Factory with live expert demonstration orchestration
    if command_factory is None:
        from icgs.environments.rlbench.teleop_oracle import create_live_expert_command_factory
        command_factory = create_live_expert_command_factory(
            source_protocol_id="rlbench-live-expert-v1",
            max_intervals=max_intervals,
        )

    # Concrete Limits composition
    if limits is None:
        from icgs.data.collection.runner import CollectionLimits
        limits = CollectionLimits(
            max_attempts=num_attempts,
            max_intervals=max_intervals * num_attempts,
            max_wall_time_s=wall_time_s,
            max_disk_bytes=disk_limit_bytes,
        )

    # Concrete MethodConfig composition
    if config is None:
        from icgs.configuration.method import MethodConfig
        config = MethodConfig(
            collection={
                "max_episode_intervals": max_intervals,
                "wall_limit_s": wall_time_s,
                "disk_limit_bytes": disk_limit_bytes,
            },
        )

    # Concrete Protocol Manifest
    if protocol_manifest is None:
        protocol_manifest = {
            "environment_protocol_id": "rlbench-live-clock-v1",
            "controller_protocol_id": "rlbench-timed-ik-v1",
            "sensor_protocol_id": "rlbench-wrist-depth-raw-v1",
            "asset_protocol_id": "rlbench-assets-pinned-v1",
            "split_protocol_id": "exploratory-split-dev-v1",
            "collection_protocol_id": "icgs-e01-exploratory-v1",
            "metadata": {
                "program_id": "E01",
                "track": "exploratory",
                "simulator": "CoppeliaSim-4.1.0",
                "task": task_name,
            },
        }

    if code_revision is None:
        code_revision, is_dirty_computed, dirty_patch_digest_computed = _get_git_info()
        if is_dirty is None:
            is_dirty = is_dirty_computed
        if dirty_patch_digest is None:
            dirty_patch_digest = dirty_patch_digest_computed
    if generator_version is None:
        generator_version = "0.1.0-e01"

    from icgs.data.collection.runner import run_collection
    report = run_collection(
        specs,
        environment_factory,
        dataset_root=output_dir,
        dataset_manifest_path=manifest_path,
        limits=limits,
        config=config,
        protocol_manifest=protocol_manifest,
        code_revision=code_revision,
        is_dirty=bool(is_dirty),
        dirty_patch_digest=dirty_patch_digest,
        generator_version=generator_version,
        command_factory=command_factory,
        episode_publisher=episode_publisher,
        index_publisher=index_publisher,
        online_provider=online_provider,
        dataset_track="exploratory",
    )
    if report.status not in ("completed", "attempt_limit", "interval_limit", "wall_limit"):
        details = []
        for r in report.attempt_records:
            err = r.get("error") if isinstance(r, dict) else getattr(r, "error", None)
            details.append(f"{r.get('episode_id') if isinstance(r, dict) else getattr(r, 'episode_id', '?')}: {err}")
        raise RuntimeError(
            f"E01 exploratory collection failed with status={report.status!r}; attempt_records={details}"
        )

    index_report: dict[str, Any] = {}
    if report.episodes_published > 0 and index_publisher is not None:
        try:
            index_report = index_publisher()
        except Exception as exc:
            raise RuntimeError(f"E01 isolated index publication failed closed: {exc}") from exc
        if (
            not isinstance(index_report, dict)
            or index_report.get("drive_verified") is not True
            or index_report.get("hf_verified") is not True
        ):
            raise RuntimeError("E01 isolated index publication did not verify Drive and HF bytes")

    manifest_sha256: str | None = None
    if manifest_path.is_file():
        import hashlib
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    cleanup_status = "COMPLETED"
    if report.status != "completed" or any(getattr(r, "cleanup_error", None) for r in report.attempt_records):
        cleanup_status = "UNKNOWN"

    if result_file is not None:
        res_path = Path(result_file).resolve()
        try:
            res_path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Result file '{res_path}' must be strictly under output_dir '{output_dir}'") from exc

        worker_result = {
            "schema_version": "icgs_worker_result_v1",
            "kind": "worker_result",
            "nonce": nonce,
            "terminal_status": report.status,
            "episodes_published": report.episodes_published,
            "manifest_sha256": manifest_sha256,
            "index_publication": {
                "drive_verified": bool(index_report.get("drive_verified", False)),
                "hf_verified": bool(index_report.get("hf_verified", False)),
                "index_commit": index_report.get("index_commit"),
                "manifest_sha256": index_report.get("manifest_sha256") or manifest_sha256,
            },
            "attempt_records": [
                {
                    "episode_id": getattr(r, "episode_id", None) if hasattr(r, "episode_id") else r.get("episode_id") if isinstance(r, dict) else None,
                    "status": getattr(r, "status", None) if hasattr(r, "status") else r.get("status") if isinstance(r, dict) else None,
                    "transitions": getattr(r, "transitions", None) if hasattr(r, "transitions") else r.get("transitions") if isinstance(r, dict) else None,
                }
                for r in report.attempt_records
            ],
            "cleanup": cleanup_status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_atomic_json(res_path, worker_result)

    return report.episodes_published


def _write_atomic_json(target_path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON data to file with fsync."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".tmp_{target_path.name}_{secrets.token_hex(6)}"
    payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    with open(tmp_path, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def _write_durable_outcome(
    outcome: dict[str, Any],
    local_path: Path,
    drive_path: Path | None = None,
) -> None:
    """Persist durable JSON outcome atomically with fsync to local path and Drive.

    Fails loudly if neither local path nor Drive outcome could be persisted.
    """
    local_err: Exception | None = None
    drive_err: Exception | None = None
    try:
        _write_atomic_json(local_path, outcome)
    except Exception as exc:
        local_err = exc
        logger.error("Failed to write local outcome JSON to %s: %s", local_path, exc)

    if drive_path is not None:
        try:
            _write_atomic_json(drive_path, outcome)
        except Exception as exc:
            drive_err = exc
            logger.error("Failed to write Drive outcome JSON to %s: %s", drive_path, exc)

    if local_err is not None:
        if drive_path is None or drive_err is not None:
            raise IOError(
                f"Failed to persist durable outcome to any location. Local error: {local_err}; Drive error: {drive_err}"
            ) from local_err


def _read_bounded_log_tail(log_path: Path, max_bytes: int = 4096) -> str:
    """Read bounded tail bytes from a log file into memory."""
    if not log_path.is_file():
        return ""
    try:
        size = log_path.stat().st_size
        offset = max(0, size - max_bytes)
        with open(log_path, "rb") as f:
            if offset > 0:
                f.seek(offset)
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def run_supervised_exploratory_collection(
    output_dir: Path,
    *,
    g1_receipt_path: Path | str | None = None,
    drive_dir: Path | None = None,
    hf_repo_id: str | None = None,
    hf_token_file: Path | str | None = None,
    num_attempts: int = 1,
    max_intervals: int = 16,
    wall_time_s: float = 60.0,
    disk_limit_bytes: int = 1024 * 1024 * 1024,
    supervisor_timeout_s: float | None = None,
    task_name: str = G1_DEFAULT_TASK,
    worker_cmd: Sequence[str] | None = None,
    outcome_file: Path | None = None,
    poll_interval_s: float = 0.05,
    kill_grace_s: float = 2.0,
) -> dict[str, Any]:
    """Execute E01 collection in an isolated subprocess supervised by hard wall timeout.

    Process-group termination with SIGTERM followed by bounded SIGKILL ensures
    no surviving simulator child processes on timeout. File-backed bounded logs
    prevent pipe buffer deadlocks. Honest diagnostic outcome is written atomically
    with fsync to local disk and Drive with cleanup marked UNKNOWN on forced kill.
    """
    _validate_e01_parameters(
        num_attempts=num_attempts,
        max_intervals=max_intervals,
        wall_time_s=wall_time_s,
        disk_limit_bytes=disk_limit_bytes,
        supervisor_timeout_s=supervisor_timeout_s,
    )
    verify_g1_receipt(g1_receipt_path, expected_task=task_name)
    constrained_drive = _validate_drive_dir_constraint(drive_dir)

    timeout_s = supervisor_timeout_s if supervisor_timeout_s is not None else wall_time_s
    outcome_path = outcome_file or (output_dir / "supervisor-outcome.json")
    drive_outcome_path = (constrained_drive / "supervisor-outcome.json") if constrained_drive else None

    nonce = secrets.token_hex(16)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = (output_dir / f".worker_result_{nonce}.json").resolve()

    try:
        result_path.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Result path '{result_path}' is not strictly within output_dir '{output_dir}'") from exc

    if result_path.exists():
        result_path.unlink()

    stdout_log = output_dir / f".worker_stdout_{nonce}.log"
    stderr_log = output_dir / f".worker_stderr_{nonce}.log"
    if stdout_log.exists():
        stdout_log.unlink()
    if stderr_log.exists():
        stderr_log.unlink()

    if worker_cmd is None:
        cmd: list[str] = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--output-dir",
            str(output_dir),
            "--g1-receipt",
            str(g1_receipt_path),
            "--task",
            task_name,
            "--num-attempts",
            str(num_attempts),
            "--max-intervals",
            str(max_intervals),
            "--wall-time-s",
            str(wall_time_s),
            "--nonce",
            nonce,
            "--result-file",
            str(result_path),
        ]
        if constrained_drive is not None:
            cmd.extend(["--drive-dir", str(constrained_drive)])
        if hf_repo_id is not None:
            cmd.extend(["--hf-repo-id", str(hf_repo_id)])
        if hf_token_file is not None:
            cmd.extend(["--hf-token-file", str(hf_token_file)])
    else:
        cmd = list(worker_cmd)
        has_result_file = any(
            c == "--result-file" or (isinstance(c, str) and c.startswith("--result-file="))
            for c in cmd
        )
        if not has_result_file:
            cmd.extend(["--result-file", str(result_path), "--nonce", nonce])

    t0 = time.monotonic()
    with open(stdout_log, "wb") as f_out, open(stderr_log, "wb") as f_err:
        proc = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=f_err,
            start_new_session=True,
        )

        timed_out = False
        deadline = t0 + timeout_s
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                break
            time.sleep(min(poll_interval_s, max(0.001, deadline - now)))

        if timed_out:
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                pgid = proc.pid

            # Step 1: SIGTERM to process group
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

            # Step 2: Bounded grace period
            try:
                proc.wait(timeout=kill_grace_s)
            except subprocess.TimeoutExpired:
                # Step 3: SIGKILL to process group
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    proc.wait(timeout=kill_grace_s)
                except subprocess.TimeoutExpired:
                    pass

            surviving = False
            try:
                os.killpg(pgid, 0)
                surviving = True
            except ProcessLookupError:
                surviving = False
            except OSError:
                pass

            elapsed_s = time.monotonic() - t0
            stdout_tail = _read_bounded_log_tail(stdout_log, max_bytes=4096)
            stderr_tail = _read_bounded_log_tail(stderr_log, max_bytes=4096)

            outcome = {
                "status": "FORCED_TIMEOUT",
                "forced_timeout": True,
                "wall_time_s": elapsed_s,
                "timeout_cap_s": timeout_s,
                "cleanup": "UNKNOWN",
                "outer_env_shutdown": "UNKNOWN",
                "surviving_children": surviving,
                "exit_code": proc.returncode,
                "error": f"Supervisor hard wall timeout ({timeout_s}s) exceeded; worker process group forcefully terminated",
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
            raise TimeoutError(f"Supervisor hard wall timeout ({timeout_s}s) exceeded; worker forcefully killed")

    elapsed_s = time.monotonic() - t0
    stdout_tail = _read_bounded_log_tail(stdout_log, max_bytes=4096)
    stderr_tail = _read_bounded_log_tail(stderr_log, max_bytes=4096)

    if proc.returncode != 0:
        outcome = {
            "status": "WORKER_FAILED",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": proc.returncode,
            "error": f"Worker process exited with nonzero code {proc.returncode}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker process failed with code {proc.returncode}")

    # Process exited with code 0: strictly verify worker-result JSON!
    if not result_path.is_file() or result_path.is_symlink():
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Worker exited 0 but result file is missing or invalid: {result_path}. stderr: {stderr_tail!r} stdout: {stdout_tail!r}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker exited 0 but result file is missing or invalid: {result_path}. stderr: {stderr_tail!r}")

    try:
        res_data = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Failed to parse worker result JSON: {exc}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker result JSON is malformed: {exc}") from exc

    if res_data.get("schema_version") != "icgs_worker_result_v1" or res_data.get("kind") != "worker_result":
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": "Worker result schema_version or kind invalid",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError("Worker result schema_version or kind invalid")

    if res_data.get("nonce") != nonce:
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Worker result nonce mismatch: got {res_data.get('nonce')}, expected {nonce}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker result nonce mismatch: got {res_data.get('nonce')}, expected {nonce}")

    if res_data.get("episodes_published") != 1:
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Worker did not publish exactly one episode: {res_data.get('episodes_published')}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker result did not publish exactly one episode: {res_data.get('episodes_published')}")

    idx_pub = res_data.get("index_publication", {})
    if not isinstance(idx_pub, dict) or not idx_pub.get("drive_verified") or not idx_pub.get("hf_verified"):
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Worker result index publication unverified: {idx_pub}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker result index publication unverified: {idx_pub}")

    if res_data.get("cleanup") != "COMPLETED":
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Worker result cleanup incomplete: {res_data.get('cleanup')}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker result indicates incomplete cleanup: {res_data.get('cleanup')}")

    if res_data.get("terminal_status") != "completed":
        outcome = {
            "status": "WORKER_RESULT_INVALID",
            "forced_timeout": False,
            "wall_time_s": elapsed_s,
            "timeout_cap_s": timeout_s,
            "cleanup": "UNKNOWN",
            "outer_env_shutdown": "UNKNOWN",
            "surviving_children": False,
            "exit_code": 0,
            "error": f"Worker result terminal status is {res_data.get('terminal_status')}",
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
        raise RuntimeError(f"Worker result terminal status is not completed: {res_data.get('terminal_status')}")

    outcome = {
        "status": "SUCCESS",
        "forced_timeout": False,
        "wall_time_s": elapsed_s,
        "timeout_cap_s": timeout_s,
        "cleanup": "COMPLETED",
        "outer_env_shutdown": "COMPLETED",
        "surviving_children": False,
        "exit_code": 0,
        "episodes_published": 1,
        "index_verified": True,
        "manifest_sha256": res_data.get("manifest_sha256"),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "worker_result": res_data,
    }
    _write_durable_outcome(outcome, outcome_path, drive_outcome_path)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded E01 exploratory collection entrypoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/icgs_data/exploratory"))
    parser.add_argument("--task", type=str, default=G1_DEFAULT_TASK)
    parser.add_argument(
        "--g1-receipt",
        type=Path,
        required=True,
        help="Path to a strict valid PASS G1 receipt required before collection.",
    )
    parser.add_argument("--drive-dir", type=Path, default=None, help="Mounted Drive root directory for primary sync.")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="HuggingFace dataset repository ID for secondary sync.")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=Path("/content/.icgs_hf_token") if Path("/content/.icgs_hf_token").is_file() else None,
        help="Path to regular file containing Hugging Face token.",
    )
    parser.add_argument("--num-attempts", type=int, default=1, help="Bounded attempt budget (strictly 1).")
    parser.add_argument("--max-intervals", type=int, default=350, help="Max intervals per episode (1..1024).")
    parser.add_argument("--wall-time-s", type=float, default=300.0, help="Max collection wall time in seconds.")
    parser.add_argument("--worker", action="store_true", help="Internal worker mode executing the simulator.")
    parser.add_argument("--outcome-file", type=Path, default=None, help="Path for durable supervisor JSON outcome.")
    parser.add_argument("--result-file", type=Path, default=None, help="Path for worker result JSON (worker mode).")
    parser.add_argument("--nonce", type=str, default=None, help="Nonce for worker result verification (worker mode).")
    args = parser.parse_args()

    # Enforce parameter validation immediately so neither supervisor nor --worker can bypass it
    _validate_e01_parameters(
        num_attempts=args.num_attempts,
        max_intervals=args.max_intervals,
        wall_time_s=args.wall_time_s,
    )

    if args.worker:
        try:
            run_exploratory_collection(
                args.output_dir,
                g1_receipt_path=args.g1_receipt,
                drive_dir=args.drive_dir,
                hf_repo_id=args.hf_repo_id,
                hf_token_file=args.hf_token_file,
                num_attempts=args.num_attempts,
                max_intervals=args.max_intervals,
                wall_time_s=args.wall_time_s,
                task_name=args.task,
                result_file=args.result_file,
                nonce=args.nonce,
            )
        except Exception as exc:
            print(f"[WORKER ERROR] {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            run_supervised_exploratory_collection(
                args.output_dir,
                g1_receipt_path=args.g1_receipt,
                drive_dir=args.drive_dir,
                hf_repo_id=args.hf_repo_id,
                hf_token_file=args.hf_token_file,
                num_attempts=args.num_attempts,
                max_intervals=args.max_intervals,
                wall_time_s=args.wall_time_s,
                task_name=args.task,
                outcome_file=args.outcome_file,
            )
        except Exception as exc:
            print(f"[SUPERVISOR ERROR] {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
