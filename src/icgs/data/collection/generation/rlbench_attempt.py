"""Materialize measured RLBench attempt prefixes into generation artifacts."""

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


ONLINE_OBSERVATION_FIELDS = frozenset({"points", "point_valid", "T_w_e", "grip"})


def online_observation_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the public online-observation contract fields."""
    missing = ONLINE_OBSERVATION_FIELDS - set(observation)
    if missing:
        raise ValueError(f"online observation missing fields: {sorted(missing)}")
    return {field: observation[field] for field in ("points", "point_valid", "T_w_e", "grip")}

from icgs.data.collection.generation.distributed_contracts import GenerationJob, WorkerResult
from icgs.data.collection.generation.diversity import train_subset_for_sample
from icgs.data.collection.generation.episode_archive import (
    EpisodeArchiveWriter,
    archive_profile_from_environment,
)
from icgs.data.collection.generation.episode_record import assemble_attempt_record, assemble_episode
from icgs.data.collection.generation.task_labels import materialize_task_labels
from icgs.data.collection.generation.perturbations import INVALID_OBSERVATION, SIMULATOR_CRASH, SUCCESS, VALID_FAILURE
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
    metadata: Mapping[str, Any] | None = None


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
    archive_enabled = archive_profile_from_environment() is not None
    try:
        observations = tuple(_observation_row(item) for item in raw.observations)
        actions = tuple(
            np.asarray(item) if archive_enabled else np.asarray(item, dtype=np.float64)
            for item in raw.actions
        )
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
            failure_type=raw.error_type or ("simulator_exception" if raw.simulator_crash else None),
            terminal_t=len(actions) or None,
            valid_observation_until=len(observations) - 1 if observations else None,
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
        if not archive_enabled:
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
            continue
        state = {
            "T_w_e": observation["T_w_e"],
            "grip": observation["grip"],
        }
        if hasattr(source, "joint_positions") and getattr(source, "joint_positions") is not None:
            positions = np.asarray(getattr(source, "joint_positions"))
            joint_positions.append(positions)
            state["joint_positions"] = positions
        if hasattr(source, "joint_velocities") and getattr(source, "joint_velocities") is not None:
            velocities = np.asarray(getattr(source, "joint_velocities"))
            joint_velocities.append(velocities)
            state["joint_velocities"] = velocities
        robot_states.append(state)
        optional_modalities = (
            ("front_rgb", rgb_frames, None),
            ("wrist_depth", depth_frames, None),
            ("wrist_mask", wrist_masks, None),
            ("front_mask", front_masks, None),
        )
        for attribute, destination, dtype in optional_modalities:
            if hasattr(source, attribute) and getattr(source, attribute) is not None:
                value = np.asarray(getattr(source, attribute), dtype=dtype)
                destination.append(value)

    def _stack_if_complete(values: list[np.ndarray], count: int) -> np.ndarray | None:
        if len(values) != count:
            return None
        try:
            return np.stack(values)
        except (TypeError, ValueError):
            return None

    joint_positions_array = _stack_if_complete(joint_positions, len(observations))
    joint_velocities_array = _stack_if_complete(joint_velocities, len(observations))
    front_rgb_array = _stack_if_complete(rgb_frames, len(observations))
    wrist_depth_array = _stack_if_complete(depth_frames, len(observations))
    wrist_mask_array = _stack_if_complete(wrist_masks, len(observations))
    front_mask_array = _stack_if_complete(front_masks, len(observations))

    scene_states = list(raw.scene_states)
    if archive_enabled:
        object_states = (
            scene_states
            if len(scene_states) == len(observations) and all(isinstance(item, Mapping) for item in scene_states)
            else None
        )
    else:
        object_states = scene_states
        if len(object_states) != len(observations):
            object_states = [{} for _ in observations]
    structured_steps = binding.get("structured_steps") or binding.get("events") or ()
    task_labels = (
        materialize_task_labels(structured_steps, object_states, robot_states)
        if structured_steps and object_states is not None else {}
    )
    episode = assemble_episode(
        plan=job.plan,
        binding=binding,
        observations=observations,
        transitions=transitions,
        outcome=outcome,
        robot_states=robot_states,
        object_states=object_states,
        intervention=job.plan.intervention,
        terminal_reason=raw.terminal_reason,
        task_labels=task_labels,
    )
    auxiliary = {
        "actions": np.stack(actions),
        "ee_poses": np.stack([item["T_w_e"] for item in observations]),
        "gripper_states": np.asarray([item["grip"] for item in observations], dtype=np.float32),
        "collision_events": list(raw.collision_events),
        "sim_time_s": raw.sim_time_s,
        "task_labels": task_labels,
        "metadata": dict(raw.metadata or {}),
        "object_states_available": object_states is not None,
    }
    for name, value in (
        ("joint_positions", joint_positions_array),
        ("joint_velocities", joint_velocities_array),
        ("front_rgb_frames", front_rgb_array),
        ("wrist_depth_frames", wrist_depth_array),
        ("wrist_mask_frames", wrist_mask_array),
        ("front_mask_frames", front_mask_array),
    ):
        if value is not None:
            auxiliary[name] = value
    if archive_enabled:
        modality_sources = (
            ("front_rgb", rgb_frames, "front_rgb_frame_boundaries"),
            ("wrist_depth", depth_frames, "wrist_depth_frame_boundaries"),
            ("wrist_mask", wrist_masks, "wrist_mask_frame_boundaries"),
            ("front_mask", front_masks, "front_mask_frame_boundaries"),
        )
        for _name, values, boundary_name in modality_sources:
            if not values:
                continue
            try:
                auxiliary_name = {
                    "front_rgb": "front_rgb_frames",
                    "wrist_depth": "wrist_depth_frames",
                    "wrist_mask": "wrist_mask_frames",
                    "front_mask": "front_mask_frames",
                }[_name]
                if auxiliary_name not in auxiliary:
                    auxiliary[auxiliary_name] = np.stack(values)
                if len(values) != len(raw.observations):
                    auxiliary[boundary_name] = np.asarray(
                        [index for index, source in enumerate(raw.observations)
                         if hasattr(source, _name) and getattr(source, _name) is not None],
                        dtype=np.int64,
                    )
            except (TypeError, ValueError):
                # A captured modality with incompatible frame shapes cannot be
                # represented as one numeric archive array; omit only that
                # modality while retaining all other captured source arrays.
                auxiliary.pop(auxiliary_name, None)
                auxiliary.pop(boundary_name, None)
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


def _archive_debug_metadata(materialized: MaterializedAttempt, job: GenerationJob) -> dict[str, Any]:
    auxiliary = materialized.auxiliary
    return {
        "source_run_id": job.run_id,
        "code_revision": job.code_revision,
        "preprocessing_identity": "rlbench_measured_v1",
        "job": job.as_dict(),
        "collision_events": auxiliary.get("collision_events", []),
        "sim_time_s": auxiliary.get("sim_time_s"),
        "task_labels": auxiliary.get("task_labels", {}),
        "metadata": auxiliary.get("metadata", {}),
        "outcome": materialized.outcome,
    }


def _archive_raw_arrays(materialized: MaterializedAttempt) -> dict[str, Any]:
    auxiliary = materialized.auxiliary
    names = (
        "actions", "joint_positions", "joint_velocities", "front_rgb_frames",
        "wrist_depth_frames", "wrist_mask_frames", "front_mask_frames",
        "front_rgb_frame_boundaries", "wrist_depth_frame_boundaries",
        "wrist_mask_frame_boundaries", "front_mask_frame_boundaries",
        "ee_poses", "gripper_states",
    )
    return {name: auxiliary[name] for name in names if name in auxiliary}


def _archive_episode_record(materialized: MaterializedAttempt) -> dict[str, Any]:
    record = dict(materialized.episode_record or {})
    if not materialized.auxiliary.get("object_states_available", True):
        record.pop("object_states", None)
    robot_states = record.get("robot_states")
    if isinstance(robot_states, list):
        record["robot_states"] = [
            {key: value for key, value in state.items() if value is not None}
            if isinstance(state, Mapping) else state
            for state in robot_states
        ]
    return record


def _archive_raw_gripper_pose_prefix(raw_observations: Sequence[Any]) -> dict[str, np.ndarray]:
    """Keep captured numeric source poses without making a ragged object array."""
    captured: list[tuple[int, np.ndarray]] = []
    for index, observation in enumerate(raw_observations):
        try:
            value = getattr(observation, "gripper_pose", None)
            if value is None:
                continue
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.dtype.hasobject or array.dtype.kind not in "biufc" or array.ndim == 0:
            continue
        captured.append((index, array))
    if not captured:
        return {}
    boundaries = np.asarray([index for index, _value in captured], dtype=np.int64)
    first = captured[0][1]
    if all(array.shape == first.shape and array.dtype == first.dtype for _index, array in captured):
        prefix: dict[str, np.ndarray] = {"gripper_pose": np.stack([array for _index, array in captured])}
        if len(captured) != len(raw_observations):
            prefix["gripper_pose_boundaries"] = boundaries
        return prefix
    prefix = {"gripper_pose_boundaries": boundaries}
    prefix.update({f"gripper_pose_{index}": array for index, array in captured})
    return prefix


def _archive_attempt_prefix(materialized: MaterializedAttempt) -> dict[str, Any]:
    auxiliary = materialized.auxiliary
    observations = materialized.online_observations
    raw_actions = auxiliary.get("raw_actions")
    prefix: dict[str, Any] = {
        "actions": np.asarray(() if raw_actions is None else raw_actions),
    }
    raw_observations = auxiliary.get("raw_observations") or ()
    prefix.update(_archive_raw_gripper_pose_prefix(raw_observations))
    if observations:
        point_values = np.concatenate([np.asarray(item["points"]) for item in observations], axis=0)
        point_offsets = np.zeros(len(observations) + 1, dtype=np.int64)
        for index, observation in enumerate(observations):
            point_offsets[index + 1] = point_offsets[index] + len(observation["points"])
        prefix.update({
            "points": point_values,
            "point_offsets": point_offsets,
            "T_w_e": np.asarray([item["T_w_e"] for item in observations]),
            "grip": np.asarray([item["grip"] for item in observations]),
            "point_valid": np.concatenate([np.asarray(item["point_valid"], dtype=bool) for item in observations]),
        })
    return prefix


def _write_archive_result(
    materialized: MaterializedAttempt,
    job: GenerationJob,
    target: Path,
    profile: Any,
) -> WorkerResult:
    writer = EpisodeArchiveWriter(profile)
    if materialized.episode_record is not None:
        record = _archive_episode_record(materialized)
        writer.write_episode(
            record,
            raw_arrays=_archive_raw_arrays(materialized),
            debug_metadata=_archive_debug_metadata(materialized, job),
            output_dir=target,
        )
        episode_id: str | None = job.episode_id
        timeline = {
            "actions": len(materialized.transitions),
            "observations": len(materialized.online_observations),
            "durations": len(record["dt"]),
        }
    else:
        attempt_record = dict(materialized.attempt_record or {})
        attempt_record["split"] = "dev" if job.plan.split == "development" else job.plan.split
        attempt_record["subset"] = (
            job.plan.randomization.get("train_subset")
            or train_subset_for_sample(job.plan.randomization)
            if job.plan.split == "train" else None
        )
        writer.write_attempt(
            attempt_record,
            prefix_arrays=_archive_attempt_prefix(materialized),
            debug_metadata=_archive_debug_metadata(materialized, job),
            output_dir=target,
        )
        episode_id = None
        timeline = None
    file_sha256 = {
        str(path.relative_to(target)): _sha256(path)
        for path in sorted(target.rglob("*")) if path.is_file()
    }
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


def write_closed_attempt_result(
    materialized: MaterializedAttempt,
    job: GenerationJob,
    result_dir: str | Path,
) -> WorkerResult:
    target = Path(result_dir)
    profile = archive_profile_from_environment()
    if profile is not None:
        if target.exists():
            raise FileExistsError(f"result directory already exists: {target}")
        return _write_archive_result(materialized, job, target, profile)
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
            observation_payload: dict[str, Any] = {
                "pointcloud": [item["points"] for item in observations],
            }
            for name, output_name in (
                ("front_rgb_frames", "rgb"),
                ("wrist_depth_frames", "depth"),
                ("wrist_mask_frames", "masks"),
            ):
                if name in auxiliary:
                    observation_payload[output_name] = auxiliary[name]
            robot_payload: dict[str, Any] = {
                "ee_pose": auxiliary["ee_poses"],
                "gripper": auxiliary["gripper_states"],
            }
            if "joint_positions" in auxiliary and "joint_velocities" in auxiliary:
                robot_payload["joint_state"] = np.concatenate([
                    auxiliary["joint_positions"], auxiliary["joint_velocities"]
                ], axis=1)
            layout_payload: dict[str, Any] = {
                "episode": {
                    "episode_id": job.episode_id,
                    "program_id": job.program_id,
                    "split": record["provenance"]["split"],
                    "layout_version": LAYOUT_VERSION,
                    "outcome": materialized.outcome,
                },
                "observations": observation_payload,
                "robot": robot_payload,
                "actions": auxiliary["actions"],
                "task": {
                    "events": record.get("events", []),
                    "collisions": auxiliary["collision_events"],
                    **(auxiliary.get("task_labels") or {}),
                },
                "result": {
                    "success": materialized.outcome == SUCCESS,
                    "terminal_reason": record["provenance"]["terminal_reason"],
                    "outcome": materialized.outcome,
                },
            }
            write_training_episode_layout(partial / "layout", layout_payload)
            telemetry = {
                name: auxiliary[name]
                for name in (
                    "joint_positions", "joint_velocities", "ee_poses", "gripper_states",
                    "wrist_depth_frames", "wrist_mask_frames", "front_mask_frames",
                )
                if name in auxiliary
            }
            np.savez_compressed(partial / "telemetry.npz", **telemetry)
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
    "ONLINE_OBSERVATION_FIELDS",
    "RawAttempt",
    "materialize_raw_attempt",
    "online_observation_view",
    "raw_attempt_from_exception",
    "write_closed_attempt_result",
]
