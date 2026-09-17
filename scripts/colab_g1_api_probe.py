#!/usr/bin/env python3
"""Bounded, fail-closed G1 probe for pinned RLBench / PyRep / CoppeliaSim 4.1.

Authority: ADR0012 and ADR0007. This script creates only one Drive-side
feasibility receipt. It never creates episode archives, shard files, manifests,
or Hugging Face artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("colab_g1_api_probe")

G1_PROBE_PROTOCOL_ID = "rlbench-g1-coppeliasim-4.1-franka"
RECEIPT_FILE_NAME = "g1-receipt.json"
RECEIPT_KIND = "g1_api_probe_receipt"
RECEIPT_VERSION = 2
PROBE_IMPLEMENTATION_ID = "g1-api-probe-v2"
AUTHORITY = "ADR0012 and ADR0007"
DRIVE_ROOT = Path("/content/drive/MyDrive")
PROVISION_ROOT = Path("/content/icgs-simulator")
PINNED_RLBENCH_REVISION = "02720bba4c73fe02eb75df946b8791b806028a9d"
PINNED_PYREP_REVISION = "8f420be8064b1970aae18a9cfbc978dfb15747ef"
PINNED_COPPELIASIM_VERSION = "4.1.0"
PINNED_COPPELIASIM_ARCHIVE_SHA256 = "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8"
PINNED_COPPELIASIM_ARCHIVE_NAME = "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
EXPECTED_TIMESTEP_S = 0.05
PHYSICAL_STEPS = 2
GRIPPER_HOLD_VELOCITY = 0.04
GRIPPER_HOLD_ATOL = 0.001
JOINT_TARGET_ATOL_RAD = 0.01
DEFAULT_MAX_WALL_TIME_S = 120.0
MAX_WALL_TIME_S = 180.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a complete receipt; never partially overwrite one."""
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _failure_receipt(
    task_name: str,
    error: BaseException | str,
    *,
    cleanup_succeeded: bool,
    started_at_utc: str,
) -> dict[str, Any]:
    exception = error if isinstance(error, BaseException) else RuntimeError(error)
    return {
        "kind": RECEIPT_KIND,
        "receipt_version": RECEIPT_VERSION,
        "probe_implementation_id": PROBE_IMPLEMENTATION_ID,
        "status": "FAIL",
        "authority": AUTHORITY,
        "protocol_id": G1_PROBE_PROTOCOL_ID,
        "task_name": task_name,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_now(),
        "cleanup": {"outer_env_shutdown": cleanup_succeeded},
        "error": _safe_error(exception),
    }


def verify_drive_directory(drive_dir: Path | str | None) -> Path:
    """Require a real descendant of the mounted Google Drive root.

    No local-output CLI escape exists. ``Path.relative_to`` rejects prefix
    siblings such as ``/content/drive_evil`` that a string-prefix check accepts.
    """
    if drive_dir is None:
        raise ValueError("Drive output directory must be explicitly specified (authority: ADR0012).")

    drive_root = DRIVE_ROOT.resolve()
    if not drive_root.is_dir():
        raise RuntimeError(
            f"Google Drive is not mounted at '{drive_root}' (authority: ADR0012). "
            "Mount Drive before probing."
        )

    path = Path(drive_dir).resolve()
    try:
        path.relative_to(drive_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Output path '{path}' is not a descendant of mounted Google Drive '{drive_root}' "
            "(authority: ADR0012)."
        ) from exc

    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"Unable to access or create directory at '{path}'.")
    return path


def _run_git(source_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not inspect pinned source tree '{source_root}'.") from exc
    return completed.stdout.strip()


def _collect_pinned_upstream_evidence() -> dict[str, Any]:
    """Verify actual provisioned input trees instead of recording constants as facts."""
    archive = PROVISION_ROOT / PINNED_COPPELIASIM_ARCHIVE_NAME
    simulator_root = PROVISION_ROOT / "CoppeliaSim"
    configured_root = os.environ.get("COPPELIASIM_ROOT")
    if not archive.is_file() or not simulator_root.is_dir():
        raise RuntimeError("Pinned CoppeliaSim 4.1 archive/root is unavailable.")
    if configured_root is None or Path(configured_root).resolve() != simulator_root.resolve():
        raise RuntimeError(
            "COPPELIASIM_ROOT does not resolve to the provisioned pinned CoppeliaSim 4.1 root."
        )
    with archive.open("rb") as stream:
        archive_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive_digest != PINNED_COPPELIASIM_ARCHIVE_SHA256:
        raise RuntimeError("CoppeliaSim archive SHA256 does not match the pinned distribution.")

    sources = {
        "rlbench": (PROVISION_ROOT / "RLBench", PINNED_RLBENCH_REVISION),
        "pyrep": (PROVISION_ROOT / "PyRep", PINNED_PYREP_REVISION),
    }
    evidence: dict[str, Any] = {}
    for name, (source_root, expected_revision) in sources.items():
        if not source_root.is_dir():
            raise RuntimeError(f"Pinned {name} source tree is unavailable at '{source_root}'.")
        revision = _run_git(source_root, "rev-parse", "HEAD")
        if revision != expected_revision:
            raise RuntimeError(
                f"Pinned {name} revision mismatch: found {revision!r}, expected {expected_revision!r}."
            )
        if _run_git(source_root, "status", "--porcelain"):
            raise RuntimeError(f"Pinned {name} source tree is dirty; G1 evidence would be ambiguous.")
        evidence[name] = {
            "revision": revision,
            "source_tree_clean": True,
            "source_root": str(source_root),
        }

    evidence["coppeliasim"] = {
        "version": PINNED_COPPELIASIM_VERSION,
        "archive_sha256": archive_digest,
        "simulator_root": str(simulator_root),
    }
    return evidence


def _require_callable(value: Any, label: str) -> Any:
    if not callable(value):
        raise RuntimeError(f"Missing expected API: {label} is required.")
    return value


def _require_module_origin(module_name: str, module: Any, source_root: Path) -> None:
    """Reject a module whose executed bytes are outside the verified source tree."""
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"Pinned module {module_name!r} has no inspectable source origin.")
    root = source_root.resolve()
    try:
        Path(module_file).resolve().relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Pinned module {module_name!r} does not resolve beneath verified source tree '{root}'."
        ) from exc


def _import_pinned_module(module_name: str, source_root: Path) -> Any:
    module = importlib.import_module(module_name)
    _require_module_origin(module_name, module, source_root)
    return module


def _bind_pinned_imports() -> dict[str, Path]:
    """Make the verified source trees, rather than ambient package caches, execute."""
    source_roots = {
        "pyrep": (PROVISION_ROOT / "PyRep").resolve(),
        "rlbench": (PROVISION_ROOT / "RLBench").resolve(),
    }
    for package_name in source_roots:
        for module_name in tuple(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]
    for source_root in (source_roots["pyrep"], source_roots["rlbench"]):
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    for module_name, source_root in source_roots.items():
        _import_pinned_module(module_name, source_root)
    return source_roots


def _validate_max_wall_time_s(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("max_wall_time_s must be a positive finite number.")
    seconds = float(value)
    if not math.isfinite(seconds) or not 0.0 < seconds <= MAX_WALL_TIME_S:
        raise ValueError(
            f"max_wall_time_s must be finite and in (0, {MAX_WALL_TIME_S}]."
        )
    return seconds


class _ProbeDeadline:
    """Interrupt a hung probe so its existing failure/cleanup path can run."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._previous_handler: Any = None
        self._armed = False

    def arm(self) -> None:
        if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
            raise RuntimeError("G1 probe requires POSIX SIGALRM wall-time enforcement.")
        previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
        if previous_delay != 0.0 or previous_interval != 0.0:
            raise RuntimeError("G1 probe refuses to replace an existing process wall-time timer.")

        def expire(_signum: int, _frame: Any) -> None:
            raise TimeoutError(f"G1 probe exceeded its {self._seconds}-second wall-time limit.")

        self._previous_handler = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, self._seconds)
        self._armed = True

    def disarm(self) -> None:
        if self._armed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            self._armed = False


def _finite_vector(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric.") from exc
    if vector.ndim != 1 or vector.size == 0 or (size is not None and vector.size != size):
        expected = f"length {size}" if size is not None else "a non-empty vector"
        raise RuntimeError(f"{name} must be {expected}.")
    if not np.isfinite(vector).all():
        raise RuntimeError(f"{name} contains non-finite values.")
    return vector


def _extract_reset_observation(reset_result: Any) -> Any:
    """Decode the exact pinned TaskEnvironment.reset() return contract."""
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        raise RuntimeError(
            "Pinned TaskEnvironment.reset() must return (descriptions, observation); no fallback shape is accepted."
        )
    observation = reset_result[1]
    if observation is None:
        raise RuntimeError("Pinned TaskEnvironment.reset() returned no observation.")
    return observation


def _extract_wrist_points(raw_observation: Any) -> np.ndarray:
    points = raw_observation.get("wrist_point_cloud") if isinstance(raw_observation, dict) else getattr(
        raw_observation, "wrist_point_cloud", None
    )
    if points is None:
        raise RuntimeError("Missing expected API / data: raw wrist point cloud not present in reset observation.")
    try:
        array = np.asarray(points)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Raw wrist point cloud is not array-like.") from exc
    if array.ndim < 2 or array.shape[-1] != 3 or array.size == 0:
        raise RuntimeError("Raw wrist point cloud must have a non-empty trailing XYZ dimension.")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise RuntimeError("Raw wrist point cloud must have float32 or float64 floating dtype.")
    if not np.isfinite(array).all():
        raise RuntimeError("Raw point cloud contains non-finite (NaN or Inf) values.")
    return array


def _perform_target_hold(
    *,
    arm: Any,
    gripper: Any,
    scene: Any,
    sim: Any,
    timestep_s: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Command the current live tip pose through exactly two full Scene steps."""
    get_tip = _require_callable(getattr(arm, "get_tip", None), "robot.arm.get_tip()")
    solve_ik = _require_callable(
        getattr(arm, "solve_ik_via_jacobian", None), "robot.arm.solve_ik_via_jacobian()"
    )
    set_joint_targets = _require_callable(
        getattr(arm, "set_joint_target_positions", None), "robot.arm.set_joint_target_positions()"
    )
    get_joint_positions = _require_callable(
        getattr(arm, "get_joint_positions", None), "robot.arm.get_joint_positions()"
    )
    get_open_amount = _require_callable(
        getattr(gripper, "get_open_amount", None), "robot.gripper.get_open_amount()"
    )
    actuate = _require_callable(getattr(gripper, "actuate", None), "robot.gripper.actuate()")
    step_scene = _require_callable(getattr(scene, "step", None), "TaskEnvironment._scene.step()")
    get_sim_time = _require_callable(getattr(sim, "simGetSimulationTime", None), "simGetSimulationTime()")

    tip = get_tip()
    get_tip_pose = _require_callable(getattr(tip, "get_pose", None), "robot.arm.get_tip().get_pose()")
    target_pose = _finite_vector(get_tip_pose(), name="live tip pose", size=7)
    if not math.isclose(float(np.linalg.norm(target_pose[3:])), 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError("Live tip pose quaternion is not unit length.")
    joint_targets = _finite_vector(
        solve_ik(target_pose[:3].tolist(), quaternion=target_pose[3:].tolist()),
        name="IK joint solution",
    )
    set_joint_targets(joint_targets.tolist())

    open_amounts = _finite_vector(get_open_amount(), name="gripper open amount")
    if np.any(open_amounts < 0.0) or np.any(open_amounts > 1.0):
        raise RuntimeError("Gripper open amount must be in [0, 1].")
    hold_amount = float(open_amounts[0])
    if not np.allclose(open_amounts, hold_amount, rtol=0.0, atol=1e-3):
        raise RuntimeError("Gripper joint open amounts disagree; scalar target hold is ambiguous.")

    t0 = float(get_sim_time())
    if not math.isfinite(t0) or t0 < 0.0:
        raise RuntimeError(f"Invalid initial simulation time t0={t0}; expected non-negative finite float.")
    for _ in range(PHYSICAL_STEPS):
        actuate(hold_amount, GRIPPER_HOLD_VELOCITY)
        step_scene()
    t1 = float(get_sim_time())
    if not math.isfinite(t1) or not t1 > t0:
        raise RuntimeError(
            f"Simulator clock failed to advance: before={t0}, after={t1}. "
            "Authority: ADR0012 + ADR0007 require a live, advancing simulator clock."
        )
    elapsed = t1 - t0
    expected_elapsed = timestep_s * PHYSICAL_STEPS
    if not math.isclose(elapsed, expected_elapsed, rel_tol=1e-4, abs_tol=1e-6):
        raise RuntimeError(
            "Scene-step elapsed simulator time disagrees with the queried timestep: "
            f"elapsed={elapsed}, expected={expected_elapsed}."
        )

    observed_joints = _finite_vector(get_joint_positions(), name="observed arm joint positions")
    if observed_joints.shape != joint_targets.shape:
        raise RuntimeError("Observed arm joint vector shape differs from the commanded IK target.")
    joint_error = float(np.max(np.abs(observed_joints - joint_targets)))
    if joint_error > JOINT_TARGET_ATOL_RAD:
        raise RuntimeError(
            f"Target hold did not reach commanded joints: max error={joint_error}, "
            f"tolerance={JOINT_TARGET_ATOL_RAD}."
        )
    observed_open_amounts = _finite_vector(get_open_amount(), name="observed gripper open amount")
    if observed_open_amounts.shape != open_amounts.shape:
        raise RuntimeError("Observed gripper vector shape differs from the commanded hold target.")
    gripper_error = float(np.max(np.abs(observed_open_amounts - hold_amount)))
    if gripper_error > GRIPPER_HOLD_ATOL:
        raise RuntimeError(
            f"Target hold did not retain commanded gripper opening: max error={gripper_error}, "
            f"tolerance={GRIPPER_HOLD_ATOL}."
        )

    clock = {
        "dt_s": timestep_s,
        "sim_time_before_s": t0,
        "sim_time_after_s": t1,
        "clock_advanced": True,
        "physics_steps": PHYSICAL_STEPS,
        "scene_steps": PHYSICAL_STEPS,
        "elapsed_s": elapsed,
    }
    apis = {
        "task_scene_step": True,
        "robot_arm_get_tip": True,
        "robot_arm_joint_positions": True,
        "robot_arm_ik": True,
        "robot_arm_set_joint_targets": True,
        "robot_gripper": True,
        "robot_gripper_actuate": True,
    }
    target_hold = {
        "target_from_live_tip": True,
        "ik_solution_applied": True,
        "scene_steps": PHYSICAL_STEPS,
        "gripper_actuation_calls": PHYSICAL_STEPS,
        "target_hold_verified": True,
        "joint_target_error_linf_rad": joint_error,
        "joint_target_atol_rad": JOINT_TARGET_ATOL_RAD,
        "gripper_hold_target": hold_amount,
        "gripper_hold_error_linf": gripper_error,
        "gripper_hold_atol": GRIPPER_HOLD_ATOL,
    }
    return clock, apis, target_hold


def _configure_raw_wrist_observation(observation_config: Any) -> None:
    """Enable the minimum raw wrist stream required by pinned RLBench.

    In the pinned ``Scene.get_observation()``, point-cloud construction enters
    only when the camera has RGB or depth capture enabled.  The probe needs the
    unmodified wrist cloud, so it explicitly retains depth alongside the point
    cloud after disabling every other high-dimensional stream.
    """
    observation_config.set_all_high_dim(False)
    observation_config.set_all_low_dim(False)
    observation_config.wrist_camera.point_cloud = True
    observation_config.wrist_camera.depth = True
    observation_config.gripper_pose = True
    observation_config.gripper_open = True


def _load_pinned_rlbench_components(task_name: str, source_root: Path) -> tuple[Any, Any, Any]:
    action_mode_module = _import_pinned_module("rlbench.action_modes.action_mode", source_root)
    arm_action_module = _import_pinned_module("rlbench.action_modes.arm_action_modes", source_root)
    gripper_action_module = _import_pinned_module("rlbench.action_modes.gripper_action_modes", source_root)
    environment_module = _import_pinned_module("rlbench.environment", source_root)
    observation_module = _import_pinned_module("rlbench.observation_config", source_root)
    rlbench_tasks = _import_pinned_module("rlbench.tasks", source_root)

    MoveArmThenGripper = action_mode_module.MoveArmThenGripper
    EndEffectorPoseViaIK = arm_action_module.EndEffectorPoseViaIK
    Discrete = gripper_action_module.Discrete
    Environment = environment_module.Environment
    ObservationConfig = observation_module.ObservationConfig

    task_cls = getattr(rlbench_tasks, task_name, None)
    if task_cls is None:
        raise RuntimeError(f"Missing expected task class: rlbench.tasks.{task_name}")
    return Environment, ObservationConfig, (MoveArmThenGripper, EndEffectorPoseViaIK, Discrete, task_cls)


def _normalise_task_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def probe_g1_environment(
    output_dir: Path | str,
    *,
    task_name: str = "PickAndLift",
    run_live_demo: bool = False,
    environment_instance: Any = None,
    sim_api: Any = None,
    sim_api2: Any = None,
    max_wall_time_s: float = DEFAULT_MAX_WALL_TIME_S,
) -> dict[str, Any]:
    """Run the bounded probe with a process-local POSIX wall-time deadline."""
    wall_time_limit_s = _validate_max_wall_time_s(max_wall_time_s)
    deadline = _ProbeDeadline(wall_time_limit_s)
    started_monotonic = time.monotonic()
    deadline.arm()
    try:
        return _probe_g1_environment(
            output_dir,
            task_name=task_name,
            run_live_demo=run_live_demo,
            environment_instance=environment_instance,
            sim_api=sim_api,
            sim_api2=sim_api2,
            wall_time_limit_s=wall_time_limit_s,
            started_monotonic=started_monotonic,
        )
    finally:
        deadline.disarm()


def _probe_g1_environment(
    output_dir: Path | str,
    *,
    task_name: str,
    run_live_demo: bool,
    environment_instance: Any,
    sim_api: Any,
    sim_api2: Any,
    wall_time_limit_s: float,
    started_monotonic: float,
) -> dict[str, Any]:
    """Prove a pinned target hold and write PASS only after clean shutdown.

    This internal implementation runs under ``probe_g1_environment``'s deadline.
    Its injected environment/simulator parameters are unit-test seams only.
    """
    output_path = verify_drive_directory(output_dir)
    receipt_path = output_path / RECEIPT_FILE_NAME
    started_at_utc = _utc_now()
    rlb_env: Any = None
    task_env: Any = None
    failure: BaseException | None = None
    cleanup_succeeded = False
    result: dict[str, Any] | None = None

    # Invalidate any prior eligibility evidence before simulator construction.
    receipt_path.unlink(missing_ok=True)
    _write_json_atomically(
        receipt_path,
        _failure_receipt(
            task_name,
            "G1 probe started; no successful receipt is valid until clean shutdown completes.",
            cleanup_succeeded=False,
            started_at_utc=started_at_utc,
        ),
    )

    try:
        upstream = _collect_pinned_upstream_evidence()
        if environment_instance is None:
            source_roots = _bind_pinned_imports()
            Environment, ObservationConfig, components = _load_pinned_rlbench_components(
                task_name, source_roots["rlbench"]
            )
            MoveArmThenGripper, EndEffectorPoseViaIK, Discrete, task_cls = components
            obs_cfg = ObservationConfig()
            _configure_raw_wrist_observation(obs_cfg)
            action_mode = MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete()
            )
            rlb_env = Environment(action_mode, obs_config=obs_cfg, headless=True)
            shutdown = _require_callable(getattr(rlb_env, "shutdown", None), "Environment.shutdown()")
            del shutdown
            rlb_env.launch()
            task_env = rlb_env.get_task(task_cls)
        else:
            rlb_env = environment_instance
            shutdown = _require_callable(getattr(rlb_env, "shutdown", None), "Environment.shutdown()")
            del shutdown
            get_task = _require_callable(getattr(rlb_env, "get_task", None), "Environment.get_task()")
            task_env = get_task(task_name)

        pyrep = getattr(rlb_env, "_pyrep", None)
        if pyrep is None:
            raise RuntimeError("Missing expected API: Environment._pyrep instance not found.")
        get_timestep = _require_callable(
            getattr(pyrep, "get_simulation_timestep", None), "PyRep.get_simulation_timestep()"
        )
        timestep_s = float(get_timestep())
        if not math.isfinite(timestep_s) or timestep_s <= 0.0:
            raise RuntimeError(f"Invalid simulator timestep dt={timestep_s}; expected positive finite float.")
        if not math.isclose(timestep_s, EXPECTED_TIMESTEP_S, rel_tol=1e-4, abs_tol=1e-6):
            raise RuntimeError(
                f"Pinned G1 requires timestep {EXPECTED_TIMESTEP_S}; observed {timestep_s}."
            )

        sim = sim_api if sim_api is not None else sim_api2
        if sim is None:
            _import_pinned_module("pyrep.backend", source_roots["pyrep"])
            sim = _import_pinned_module("pyrep.backend.sim", source_roots["pyrep"])
        _require_callable(getattr(sim, "simGetSimulationTime", None), "simGetSimulationTime()")

        reset = _require_callable(getattr(task_env, "reset", None), "TaskEnvironment.reset()")
        get_demos = _require_callable(getattr(task_env, "get_demos", None), "TaskEnvironment.get_demos()")
        get_name = _require_callable(getattr(task_env, "get_name", None), "TaskEnvironment.get_name()")
        resolved_task_name = get_name()
        if not isinstance(resolved_task_name, str) or not resolved_task_name:
            raise RuntimeError("TaskEnvironment.get_name() did not return a non-empty task identity.")
        if _normalise_task_name(resolved_task_name) != _normalise_task_name(task_name):
            raise RuntimeError(
                f"Task identity mismatch: requested {task_name!r}, resolved {resolved_task_name!r}."
            )
        scene = getattr(task_env, "_scene", None)
        robot = getattr(task_env, "_robot", None)
        if scene is None:
            raise RuntimeError("Missing expected API: TaskEnvironment._scene instance not found.")
        if robot is None:
            raise RuntimeError("Missing expected API: TaskEnvironment._robot instance not found.")
        arm = getattr(robot, "arm", None)
        gripper = getattr(robot, "gripper", None)
        if arm is None:
            raise RuntimeError("Missing expected API: robot.arm not found.")
        if gripper is None:
            raise RuntimeError("Missing expected API: robot.gripper not found.")

        raw_observation = _extract_reset_observation(reset())
        points = _extract_wrist_points(raw_observation)
        clock, target_apis, target_hold = _perform_target_hold(
            arm=arm, gripper=gripper, scene=scene, sim=sim, timestep_s=timestep_s
        )

        live_demo_executed = False
        demo_count = 0
        if run_live_demo:
            demos = get_demos(1, live_demos=True)
            live_demo_executed = True
            demo_count = len(demos) if demos else 0

        result = {
            "kind": RECEIPT_KIND,
            "receipt_version": RECEIPT_VERSION,
            "probe_implementation_id": PROBE_IMPLEMENTATION_ID,
            "status": "PASS",
            "authority": AUTHORITY,
            "protocol_id": G1_PROBE_PROTOCOL_ID,
            "task_name": task_name,
            "resolved_task_name": resolved_task_name,
            "clock": clock,
            "environment": {
                "outer_env_class": rlb_env.__class__.__name__,
                "task_env_class": task_env.__class__.__name__,
            },
            "apis": {
                "outer_env_shutdown": True,
                "task_reset": True,
                "task_get_demos": True,
                **target_apis,
            },
            "target_hold": target_hold,
            "raw_cloud": {
                "points_found": True,
                "points_finite": True,
                "points_shape": list(points.shape),
                "points_dtype": str(points.dtype),
                "point_count": int(np.prod(points.shape[:-1])),
            },
            "demonstrations": {
                "get_demos_supported": True,
                "live_demo_executed": live_demo_executed,
                "demo_count": demo_count,
            },
            "upstream": upstream,
            "wall_time": {
                "max_s": wall_time_limit_s,
                "elapsed_s": time.monotonic() - started_monotonic,
            },
            "started_at_utc": started_at_utc,
        }
    except BaseException as exc:
        failure = exc
    finally:
        if rlb_env is not None:
            shutdown = getattr(rlb_env, "shutdown", None)
            if callable(shutdown):
                try:
                    logger.info("Shutting down outer RLBench Environment...")
                    shutdown()
                    cleanup_succeeded = True
                except BaseException as cleanup_error:
                    cleanup_succeeded = False
                    if failure is None:
                        failure = cleanup_error
                    else:
                        failure = RuntimeError(f"{failure}; outer Environment.shutdown() also failed: {cleanup_error}")

    if failure is not None:
        _write_json_atomically(
            receipt_path,
            _failure_receipt(
                task_name,
                failure,
                cleanup_succeeded=cleanup_succeeded,
                started_at_utc=started_at_utc,
            ),
        )
        raise failure
    if result is None or not cleanup_succeeded:
        failure = RuntimeError("G1 probe ended without a clean outer Environment shutdown.")
        _write_json_atomically(
            receipt_path,
            _failure_receipt(
                task_name,
                failure,
                cleanup_succeeded=cleanup_succeeded,
                started_at_utc=started_at_utc,
            ),
        )
        raise failure

    result["cleanup"] = {"outer_env_shutdown": True}
    result["completed_at_utc"] = _utc_now()
    _write_json_atomically(receipt_path, result)
    logger.info("Wrote G1 API probe PASS receipt to %s", receipt_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RLBench G1 pinned-API probe runner.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/ICGS-data-20260916/g1-probe"),
        help="Mounted Google Drive target directory for the probe receipt.",
    )
    parser.add_argument("--task", type=str, default="PickAndLift", help="RLBench task name to probe.")
    parser.add_argument(
        "--run-live-demo",
        action="store_true",
        default=False,
        help="Opt-in to execute a live demonstration (default: False).",
    )
    parser.add_argument(
        "--max-wall-time-s",
        type=float,
        default=DEFAULT_MAX_WALL_TIME_S,
        help=f"Hard in-process wall-time cap in seconds (0 < value <= {MAX_WALL_TIME_S}).",
    )
    args = parser.parse_args()
    try:
        receipt = probe_g1_environment(
            args.output_dir,
            task_name=args.task,
            run_live_demo=args.run_live_demo,
            max_wall_time_s=args.max_wall_time_s,
        )
        print(json.dumps(receipt, indent=2))
    except Exception as exc:
        logger.error("G1 API probe failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
