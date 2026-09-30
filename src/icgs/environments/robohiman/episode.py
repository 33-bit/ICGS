"""Collect one instrumented RoboHiMan episode into an ``icgs_stage1_episode_v2`` record."""

from __future__ import annotations

import re
import subprocess
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from icgs.data.stage1.labels import LABEL_PROTOCOL_ID, derive_monitor_states, label_consistency_report
from icgs.data.stage1.schema import ONLINE_FIELDS, PHYSICS_TRANSITION, SCHEMA_VERSION, perturbation_families
from icgs.environments.robohiman.camera import CAMERA_CONVENTION_ID, decode_mask_handles
from icgs.environments.robohiman.expert import MIRROR_ID, WaypointExpert, expert_control
from icgs.environments.robohiman.monitors import build_monitor
from icgs.environments.robohiman.pins import UPSTREAM
from icgs.environments.robohiman.recorder import PHASES, StepRecorder


NATIVE_COMMAND = (
    "per physics step: arm joint target positions (RML path point, PD control loop enabled) "
    "or arm joint target velocities; gripper finger joint target velocities from Gripper.actuate; "
    "kinematic attach/detach via Gripper.grasp/release"
)


def code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]

    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=False, timeout=30).stdout.strip()
    try:
        status = run("status", "--porcelain", "--", "src")
        return {"icgs_revision": run("rev-parse", "HEAD") or None, "icgs_src_dirty": bool(status)}
    except Exception:  # pragma: no cover
        return {"icgs_revision": None, "icgs_src_dirty": None}


def mask_legend(session: Any) -> dict[str, dict[str, Any]]:
    """Simulator handle -> object instance -> semantic role/category for every scene shape.

    ``0`` is the background (no shape rendered). Roles: ``robot`` (arm and
    gripper trees), ``task_object`` (graspables and their subtrees),
    ``task_fixture`` (other shapes of the task tree), ``scene_static``
    (everything else, e.g. table, walls, distractors). The category is the
    instance name without trailing digits/separators.
    """
    from pyrep.const import ObjectType

    def shapes(root: Any) -> list[Any]:
        return [root, *root.get_objects_in_tree(object_type=ObjectType.SHAPE)]

    legend: dict[str, dict[str, Any]] = {
        "0": {"object": None, "instance": None, "role": "background", "category": "background"}}
    owners: dict[int, tuple[str, str]] = {}
    for obj in session.task_obj.get_graspable_objects():
        for shape in shapes(obj):
            owners.setdefault(shape.get_handle(), (obj.get_name(), "task_object"))
    base = session.task_obj.get_base()
    for shape in shapes(base):
        owners.setdefault(shape.get_handle(), (base.get_name(), "task_fixture"))
    for root in (session.robot.arm, session.robot.gripper):
        for shape in shapes(root):
            owners[shape.get_handle()] = (root.get_name(), "robot")
    for shape in session.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
        handle, name = shape.get_handle(), shape.get_name()
        owner, role = owners.get(handle, (name, "scene_static"))
        legend[str(handle)] = {"object": owner, "instance": name, "role": role,
                               "category": re.sub(r"[_#\d]+$", "", name) or name}
    return legend


def frame_capture(session: Any, *, masks: bool = False, point_cloud: bool = False) -> Callable[[], dict[str, Any]]:
    cameras = session.cameras

    def capture() -> dict[str, Any] | None:
        try:
            obs = session.scene.get_observation()
        except RuntimeError as error:
            # RLBench reads joint forces, which CoppeliaSim only provides after a
            # physics step; right after a state restore there is no frame yet.
            if "No value available" in str(error):
                return None
            raise
        frame: dict[str, Any] = {
            "frame_gripper_pose": np.asarray(obs.gripper_pose, dtype=np.float64),
            "frame_gripper_open": np.float64(obs.gripper_open),
            "frame_joint_positions": np.asarray(obs.joint_positions, dtype=np.float64),
            "frame_task_low_dim_state": np.asarray(obs.task_low_dim_state, dtype=np.float64),
        }
        for name in cameras:
            frame[f"cam_{name}_rgb"] = np.asarray(getattr(obs, f"{name}_rgb"), dtype=np.uint8)
            frame[f"cam_{name}_depth"] = np.asarray(getattr(obs, f"{name}_depth"), dtype=np.float32)
            frame[f"cam_{name}_intrinsics"] = np.asarray(obs.misc[f"{name}_camera_intrinsics"], dtype=np.float64)
            frame[f"cam_{name}_extrinsics"] = np.asarray(obs.misc[f"{name}_camera_extrinsics"], dtype=np.float64)
            frame[f"cam_{name}_near"] = np.float64(obs.misc[f"{name}_camera_near"])
            frame[f"cam_{name}_far"] = np.float64(obs.misc[f"{name}_camera_far"])
            if masks:
                raw = np.asarray(getattr(obs, f"{name}_mask"))
                handles = decode_mask_handles(raw) if raw.ndim == 3 else np.rint(raw)
                frame[f"cam_{name}_mask_handles"] = handles.astype(np.int32)
            if point_cloud:
                frame[f"cam_{name}_point_cloud_live"] = np.asarray(getattr(obs, f"{name}_point_cloud"), dtype=np.float64)
        return frame
    return capture


def collect_episode(
    session: Any,
    *,
    episode_id: str,
    run_id: str,
    episode_index: int,
    variation: int,
    rng_state: tuple | None,
    perturbations: list[dict[str, Any]] | None = None,
    frame_stride: int = 1,
    max_steps: int = 3000,
    masks: bool = False,
    point_cloud: bool = False,
    stop_after: int | None = None,
    canonical_start: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Return (manifest, arrays, runtime) for one episode; raises on simulator errors.

    ``canonical_start`` restores the post-reset snapshot before execution so
    the episode starts without engine-internal state left by the upstream
    reset; only such episodes can be reproduced by snapshot + command replay.
    """
    reset = session.reset(variation, rng_state=rng_state)
    if canonical_start:
        from icgs.environments.robohiman.snapshot import capture_snapshot, restore_snapshot

        restore_snapshot(session, capture_snapshot(session, 0))
        session.scene._has_init_episode = True
    legend = mask_legend(session) if masks else None
    monitor = build_monitor(session.task, session.task_env, variation)
    recorder = StepRecorder(session, monitor, frame_stride=frame_stride,
                            capture_frame=frame_capture(session, masks=masks, point_cloud=point_cloud))
    expert = WaypointExpert(session, recorder, perturbations=perturbations, max_steps=max_steps)
    with recorder:
        recorder.arm()
        with expert_control(session):
            result = expert.run(stop_after=stop_after)
        recorder.capture_final_frame()
        recorder.disarm()
        trailing = recorder.trailing_command()
    arrays = recorder.arrays()
    arrays["rng_state_mt19937"] = reset["rng_state"]
    states = derive_monitor_states(arrays["step_predicates"], monitor.predicate_names, monitor.event_specs)
    for key, value in states.items():
        arrays[f"monitor_{key}"] = value
    physics_dt = float(session.pyrep.get_simulation_timestep())
    steps = int(arrays["cmd_phase"].shape[0])
    frames = int(arrays["frame_step"].shape[0]) if "frame_step" in arrays else 0
    status = "failure" if result.status == "anchor" and not result.task_success else result.status
    if result.status == "anchor" and result.task_success:
        status = "success"
    execution = []
    for item in result.applied_perturbations:
        execution.append({key: value for key, value in item.items()})
    factors = session.variation_factor_state()
    perturbation = {"benchmark_factors": factors, "execution": execution}
    perturbation["families"] = list(perturbation_families(perturbation))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "source": {
            "benchmark": "robohiman",
            "benchmark_revision": UPSTREAM["robohiman"]["revision"],
            "task": session.task,
            "task_family": session.family,
            "level": session.level(),
            "variation_index": int(variation),
            "collection_strategy": {"index": session.strategy_index,
                                    "name": session.strategy.get("variation_name")},
            "descriptions": reset["descriptions"],
        },
        "perturbation": perturbation,
        "outcome": {
            "status": status,
            "reason": result.reason,
            "task_success_final": bool(result.task_success),
            "success_first_step": result.success_first_step,
            "last_completed_waypoint": result.last_completed_waypoint,
            "stopped_at_anchor_waypoint": stop_after,
            "recoverability": "unknown",
            "recoverability_evidence": None,
        },
        "lineage": {
            "collection_run_id": run_id,
            "episode_index": int(episode_index),
            "parent_episode_id": None,
            "anchor_id": None,
            "branch_id": None,
            "reset_rng": {"kind": "numpy MT19937 before TaskEnvironment.reset",
                          "sha256": reset["rng_state_sha256"], "array": "rng_state_mt19937",
                          "cached_gaussian": reset["rng_cached_gaussian"],
                          "failed_placements_before_success": reset["failed_placements_before_success"],
                          "failed_placement_causes": reset["failed_placement_causes"]},
            "benchmark_factor_rng_before_reset": reset["factor_rng_before_reset"],
            "benchmark_factors_after_reset": reset["factors_after_reset"],
        },
        "environment": {**session.provenance(), "workspace_bounds": session.workspace_bounds(),
                        "canonical_start": {
                            "applied": bool(canonical_start),
                            "meaning": "post-reset configuration-tree snapshot restored before execution "
                                       "(drops engine-internal contact/solver state; deviation from native)"},
                        "robot": {"arm_base_pose": [float(v) for v in session.robot.arm.get_pose()],
                                  "tip": session.robot.arm.get_tip().get_name(),
                                  "arm_joint_names": [j.get_name() for j in session.robot.arm.joints]}},
        "controller": {
            "physics_dt": physics_dt,
            "native_command": NATIVE_COMMAND,
            "expert": MIRROR_ID,
            "declared_action_mode": "MoveArmThenGripper(JointVelocity, Discrete); unused by the waypoint expert",
            "arm_control_loop_enabled": True,
            "max_steps": int(max_steps),
            "phase_codes": dict(PHASES),
            "trailing_command_not_executed": {
                "arm_target_written": trailing["arm_target"] is not None,
                "grasp_event": int(trailing["grasp_event"]),
            },
            "grasp_events": recorder.events,
        },
        "timing": {
            "physics_dt_s": physics_dt,
            "transition": PHYSICS_TRANSITION,
            "frame_stride": int(frame_stride),
            "model_dt_s": int(frame_stride) * physics_dt,
            "clock": "step_sim_time is the float32 CoppeliaSim clock at each boundary",
            "d_dyn_row": "one physics-step transition; model-cadence transitions group frame_stride rows",
            "prof_step_wall_s": "host wall time spent per step; profiling only, never an action duration",
        },
        "code": code_identity(),
        "cameras": {
            "names": list(session.cameras),
            "image_size": list(session.image_size),
            "depth_encoding": "normalized between per-frame near/far clipping planes (float32)",
            "convention": CAMERA_CONVENTION_ID,
            "frame_stride": int(frame_stride),
            "frame_policy": "every frame_stride-th boundary plus every event boundary "
                            "(predicate/success change, grasp/release, phase or waypoint change)",
            "event_boundaries": recorder.event_boundaries,
            "masks_recorded": bool(masks),
            **({"mask_legend": legend} if masks else {}),
            "live_point_cloud_recorded": bool(point_cloud),
        },
        "predicates": {**monitor.describe(),
                       "object_names": recorder.object_names,
                       "task_joint_names": recorder.joint_names},
        "events": {
            "specs": monitor.event_specs,
            "label_protocol": LABEL_PROTOCOL_ID,
            "consistency": label_consistency_report(states, monitor.event_specs),
            "monitor_arrays": "monitor_* are per-episode monitor states; monitor_event_id is a diagnostic, "
                              "not the ICGS alignment target (built in D_task from an independent context)",
        },
        "arrays": {"sha256": "pending"},
        "counts": {"steps": steps, "frames": frames},
        "model_boundary": {
            "online_fields": list(ONLINE_FIELDS),
            "offline_only": ["step_predicates", "monitor_*", "step_object_poses", "step_task_success",
                             "frame_task_low_dim_state", "source", "events"],
        },
    }
    runtime = {"result": result, "monitor": monitor, "recorder": recorder}
    return manifest, arrays, runtime


def attempt_record(*, run_id: str, task: str, variation: int, episode_index: int, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": "icgs_stage1_attempt_v1",
        "collection_run_id": run_id,
        "task": task,
        "variation_index": int(variation),
        "episode_index": int(episode_index),
        "outcome": "simulator_error",
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
        "traceback": traceback.format_exc()[-4000:],
    }


__all__ = ["NATIVE_COMMAND", "attempt_record", "code_identity", "collect_episode", "frame_capture", "mask_legend"]
