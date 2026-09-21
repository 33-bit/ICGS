"""Materialize measured RLBench attempt prefixes into primary-v3 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback as traceback_module
from typing import Any, Mapping, Sequence

import numpy as np

from icgs.data.collection.v3.distributed_contracts import GenerationJob, WorkerResult
from icgs.data.collection.v3.episode_record import assemble_attempt_record, assemble_episode_v2
from icgs.data.collection.v3.perturbations import INVALID_OBSERVATION, SIMULATOR_CRASH, SUCCESS, VALID_FAILURE
from icgs.data.training_layout import LAYOUT_VERSION, write_training_episode_layout


@dataclass(frozen=True)
class RawAttempt:
    observations: Sequence[Any]
    actions: Sequence[np.ndarray]
    scene_states: Sequence[dict[str, Any]]
    collision_events: Sequence[dict[str, Any]]
    sim_time_s: float
    predicates_ok: bool
    terminal_reason: str
    simulator_crash: bool = False
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class MaterializedAttempt:
    outcome: str
    episode_record: dict[str, Any] | None
    attempt_record: dict[str, Any] | None
    online_observations: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]
    auxiliary: dict[str, Any]


def _pose_to_matrix(pose: Any) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("gripper_pose must be finite [x,y,z,qx,qy,qz,qw]")
    x, y, z, w = value[3:]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0:
        raise ValueError("gripper_pose quaternion must be nonzero")
    x, y, z, w = (np.asarray([x, y, z, w]) / norm).tolist()
    rotation = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = value[:3]
    return transform


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _observation_row(raw: Any) -> dict[str, Any]:
    points = np.asarray(getattr(raw, "wrist_point_cloud"), dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("wrist_point_cloud must contain finite XYZ points")
    transform = _pose_to_matrix(getattr(raw, "gripper_pose"))
    grip = 1 if float(getattr(raw, "gripper_open")) > 0.5 else 0
    return {
        "points": points,
        "point_valid": np.ones((len(points),), dtype=bool),
        "T_w_e": transform,
        "grip": grip,
    }


def _attempt_error(raw: RawAttempt) -> str:
    return raw.error or raw.terminal_reason or raw.error_type or "attempt failed"


def materialize_raw_attempt(
    raw: RawAttempt,
    job: GenerationJob,
    binding: Mapping[str, Any],
) -> MaterializedAttempt:
    try:
        observations = tuple(_observation_row(item) for item in raw.observations)
        actions = tuple(np.asarray(item, dtype=np.float64) for item in raw.actions)
        observation_valid = len(observations) >= 2 and len(actions) == len(observations) - 1
        if observation_valid and any(item.shape != (8,) or not np.isfinite(item).all() for item in actions):
            observation_valid = False
    except (AttributeError, TypeError, ValueError):
        observations = ()
        actions = ()
        observation_valid = False

    if raw.simulator_crash:
        outcome = SIMULATOR_CRASH
    elif not observation_valid:
        outcome = INVALID_OBSERVATION
    elif raw.predicates_ok:
        outcome = SUCCESS
    else:
        outcome = VALID_FAILURE

    if outcome in {SIMULATOR_CRASH, INVALID_OBSERVATION}:
        attempt = assemble_attempt_record(
            attempt_id=job.attempt_id,
            program_id=job.program_id,
            outcome=outcome,
            error=_attempt_error(raw),
            episode_id=None,
            episode_kind=job.plan.episode_kind,
        )
        attempt.update({
            "error_type": raw.error_type,
            "error": raw.error,
            "traceback": raw.traceback,
            "scene_seed": job.plan.scene_seed,
            "scene_signature": job.plan.randomization.get("scene_signature"),
        })
        return MaterializedAttempt(outcome, None, attempt, observations, (), {
            "raw_observations": tuple(raw.observations),
            "raw_actions": tuple(raw.actions),
        })

    transitions: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        transitions.append({
            "command": {
                "T_w_e": _pose_to_matrix(action[:7]),
                "grip": int(action[7] > 0.5),
                "duration_s": 0.05,
            },
            "achieved_duration_s": 0.05,
            "physics_substeps": 1,
            "before_boundary": index,
            "after_boundary": index + 1,
        })

    robot_states = []
    joint_positions = []
    joint_velocities = []
    rgb_frames = []
    depth_frames = []
    wrist_masks = []
    front_masks = []
    for source, observation in zip(raw.observations, observations):
        positions = np.asarray(getattr(source, "joint_positions"), dtype=np.float64)
        velocities = np.asarray(getattr(source, "joint_velocities"), dtype=np.float64)
        joint_positions.append(positions)
        joint_velocities.append(velocities)
        robot_states.append({
            "T_w_e": observation["T_w_e"],
            "grip": observation["grip"],
            "joint_positions": positions,
            "joint_velocities": velocities,
        })
        rgb_frames.append(np.asarray(getattr(source, "front_rgb"), dtype=np.uint8))
        depth_frames.append(np.asarray(getattr(source, "wrist_depth")))
        wrist_masks.append(np.asarray(getattr(source, "wrist_mask"), dtype=np.uint8))
        front_masks.append(np.asarray(getattr(source, "front_mask"), dtype=np.uint8))

    object_states = list(raw.scene_states)
    if len(object_states) != len(observations):
        object_states = [{} for _ in observations]
    episode = assemble_episode_v2(
        plan=job.plan,
        binding=binding,
        observations=observations,
        transitions=transitions,
        outcome=outcome,
        robot_states=robot_states,
        object_states=object_states,
        intervention=job.plan.intervention,
        terminal_reason=raw.terminal_reason,
    )
    auxiliary = {
        "actions": np.stack(actions),
        "joint_positions": np.stack(joint_positions),
        "joint_velocities": np.stack(joint_velocities),
        "front_rgb_frames": np.stack(rgb_frames),
        "wrist_depth_frames": np.stack(depth_frames),
        "wrist_mask_frames": np.stack(wrist_masks),
        "front_mask_frames": np.stack(front_masks),
        "ee_poses": np.stack([item["T_w_e"] for item in observations]),
        "gripper_states": np.asarray([item["grip"] for item in observations], dtype=np.float32),
        "collision_events": list(raw.collision_events),
        "sim_time_s": raw.sim_time_s,
    }
    return MaterializedAttempt(outcome, episode, None, observations, tuple(transitions), auxiliary)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_closed_attempt_result(
    materialized: MaterializedAttempt,
    job: GenerationJob,
    result_dir: str | Path,
) -> WorkerResult:
    target = Path(result_dir)
    partial = target.with_name(target.name + f".partial-{os.getpid()}-{time.time_ns()}")
    if target.exists():
        raise FileExistsError(f"result directory already exists: {target}")
    partial.mkdir(parents=True)
    try:
        if materialized.episode_record is not None:
            record = materialized.episode_record
            _write_json(partial / "episode.json", record)
            observations = materialized.online_observations
            auxiliary = materialized.auxiliary
            layout_payload: dict[str, Any] = {
                "episode": {
                    "episode_id": job.episode_id,
                    "program_id": job.program_id,
                    "split": record["provenance"]["split"],
                    "layout_version": LAYOUT_VERSION,
                    "outcome": materialized.outcome,
                },
                "observations": {
                    "pointcloud": [item["points"] for item in observations],
                    "rgb": auxiliary["front_rgb_frames"],
                    "depth": auxiliary["wrist_depth_frames"],
                    "masks": auxiliary["wrist_mask_frames"],
                },
                "robot": {
                    "ee_pose": auxiliary["ee_poses"],
                    "joint_state": np.concatenate([
                        auxiliary["joint_positions"], auxiliary["joint_velocities"]
                    ], axis=1),
                    "gripper": auxiliary["gripper_states"],
                },
                "actions": auxiliary["actions"],
                "task": {
                    "events": record.get("events", []),
                    "collisions": auxiliary["collision_events"],
                },
                "result": {
                    "success": materialized.outcome == SUCCESS,
                    "terminal_reason": record["provenance"]["terminal_reason"],
                    "outcome": materialized.outcome,
                },
            }
            write_training_episode_layout(partial / "layout", layout_payload)
            np.savez_compressed(
                partial / "telemetry.npz",
                joint_positions=auxiliary["joint_positions"],
                joint_velocities=auxiliary["joint_velocities"],
                ee_poses=auxiliary["ee_poses"],
                gripper_states=auxiliary["gripper_states"],
                wrist_depth_frames=auxiliary["wrist_depth_frames"],
                wrist_mask_frames=auxiliary["wrist_mask_frames"],
                front_mask_frames=auxiliary["front_mask_frames"],
            )
            timeline = {
                "actions": len(materialized.transitions),
                "observations": len(materialized.online_observations),
                "durations": len(record["dt"]),
            }
            episode_id: str | None = job.episode_id
        else:
            _write_json(partial / "attempt.json", materialized.attempt_record)
            raw_observations = materialized.auxiliary.get("raw_observations") or ()
            raw_actions = materialized.auxiliary.get("raw_actions") or ()
            prefix: dict[str, Any] = {"actions": np.asarray(raw_actions)}
            poses = [getattr(item, "gripper_pose", None) for item in raw_observations]
            if poses and all(item is not None for item in poses):
                prefix["gripper_pose"] = np.asarray(poses)
            np.savez_compressed(partial / "valid_prefix.npz", **prefix)
            timeline = None
            episode_id = None

        artifact_files = {
            str(path.relative_to(partial)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(partial.rglob("*")) if path.is_file()
        }
        _write_json(partial / "artifact_manifest.json", {
            "attempt_id": job.attempt_id,
            "episode_id": episode_id,
            "program_id": job.program_id,
            "outcome": materialized.outcome,
            "files": artifact_files,
        })

        file_sha256 = {
            str(path.relative_to(partial)): _sha256(path)
            for path in sorted(partial.rglob("*")) if path.is_file()
        }
        os.replace(partial, target)
        return WorkerResult(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            episode_id=episode_id,
            program_id=job.program_id,
            outcome=materialized.outcome,
            result_dir=str(target),
            file_sha256=file_sha256,
            timeline=timeline,
        )
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def raw_attempt_from_exception(task: Any, exc: BaseException) -> RawAttempt:
    state = getattr(task, "_icgs_attempt_state", None) or {}
    return RawAttempt(
        observations=tuple(state.get("observations") or ()),
        actions=tuple(state.get("actions") or ()),
        scene_states=tuple(state.get("scene_states") or ()),
        collision_events=tuple(state.get("collision_events") or ()),
        sim_time_s=float(state.get("sim_time", 0.0)),
        predicates_ok=False,
        terminal_reason="simulator_exception",
        simulator_crash=True,
        error_type=type(exc).__name__,
        error=str(exc),
        traceback="".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
    )


__all__ = [
    "MaterializedAttempt",
    "RawAttempt",
    "materialize_raw_attempt",
    "raw_attempt_from_exception",
    "write_closed_attempt_result",
]
