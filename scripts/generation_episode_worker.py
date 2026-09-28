"""Physics episodes for phase-1 pilot programs. One env, per-program isolation."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

_DEFAULT_MANIFEST = _REPO_ROOT / "artifacts" / "composition" / "approved_composition_manifest.json"


class _GenerationTimeoutSignal(TimeoutError):
    pass


def _raise_for_generation_timeout(signum, frame) -> None:
    raise _GenerationTimeoutSignal("generation worker received SIGTERM timeout; preserving captured prefix")


def _begin_archive_write_marker(write_root: Path, program_id: str, attempt_id: str | None) -> Path:
    write_root.mkdir(parents=True, exist_ok=True)
    marker = write_root / (
        f".{program_id}.archive-write-in-progress-{os.getpid()}-{time.time_ns()}"
    )
    payload = json.dumps({
        "program_id": program_id,
        "attempt_id": attempt_id,
        "pid": os.getpid(),
        "write_started_at_s": time.time(),
    }, sort_keys=True).encode("utf-8") + b"\n"
    with marker.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return marker

from icgs.data.collection.generation.compiler import compile_generation_catalog
from icgs.data.collection.generation.expert import plan_step
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


def terminal_settle_grip(observations: list[dict]) -> float:
    """Hold measured grip instead of adding an unplanned terminal release."""
    return float(observations[-1]["grip"]) if observations else 1.0


PLACEMENT_SETTLE_STEPS = 10
SCENE_SETTLE_STEPS = 5


def grasp_attachment_confirmed(gripper, obj) -> bool:
    """Proximity is not attachment; also reject unintended co-grasps."""
    return list(gripper.get_grasped_objects()) == [obj]


@contextmanager
def release_collision_guard(obj):
    """Temporarily make a held body kinematic while releasing and retreating.

    Opening the gripper while a dynamic body is still attached can launch it
    through finger contact.  Suspending dynamics for this short window avoids
    that impulse; the original dynamic state is restored after retreat.
    """
    if obj is None:
        yield
        return
    states = {}
    for name in ("dynamic",):
        getter = getattr(obj, f"is_{name}", None)
        setter = getattr(obj, f"set_{name}", None)
        if not callable(setter):
            continue
        try:
            states[name] = bool(getter()) if callable(getter) else True
            setter(False)
        except Exception:
            continue
    try:
        yield
    finally:
        for name, value in states.items():
            setter = getattr(obj, f"set_{name}", None)
            if callable(setter):
                try:
                    setter(value)
                except Exception:
                    pass


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("quaternion must be nonzero")
    return value / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = _normalize_quaternion(left)
    rx, ry, rz, rw = _normalize_quaternion(right)
    return _normalize_quaternion(np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=np.float64))


def rotate_quaternion(quaternion: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate an xyzw tool quaternion around the world vertical axis."""
    half_angle = np.deg2rad(float(yaw_deg)) / 2.0
    yaw = np.asarray([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)], dtype=np.float64)
    return _quaternion_multiply(yaw, np.asarray(quaternion, dtype=np.float64))


def interpolate_quaternion(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Hemisphere-safe normalized interpolation for incremental IK commands."""
    left = _normalize_quaternion(start)
    right = _normalize_quaternion(end)
    if float(np.dot(left, right)) < 0.0:
        right = -right
    amount = min(1.0, max(0.0, float(fraction)))
    return _normalize_quaternion((1.0 - amount) * left + amount * right)


def quaternion_angle_deg(start: np.ndarray, end: np.ndarray) -> float:
    """Return the shortest orientation change between two xyzw quaternions."""
    dot = abs(float(np.dot(_normalize_quaternion(start), _normalize_quaternion(end))))
    return float(np.rad2deg(2.0 * np.arccos(min(1.0, max(-1.0, dot)))))


def placement_recovery_step(routine, obj_name: str, target_name: str) -> dict:
    """Recover using the matching placement, not a later unrelated step."""
    step = next((step for step in reversed(routine)
                 if step.get("obj") == obj_name and step.get("target") == target_name), {})
    return {**step, "type": "pick_place", "obj": obj_name, "target": target_name,
            "grasp_z": float(step.get("grasp_z", 0.02)),
            "place_z": float(step.get("place_z", 0.0))}


def find_shape(name: str):
    for candidate in (name, f"{name}0", f"{name}#0"):
        try:
            return Object.get_object(candidate)
        except Exception:
            pass
        try:
            return Shape(candidate)
        except Exception:
            continue
    return None


def live_poses(names) -> dict:
    poses = {}
    for name in names:
        shape = find_shape(name)
        if shape is not None:
            poses[name] = [float(v) for v in shape.get_position()]
    return poses


def _timed_wrist_depth_arrays(timed: list[dict]) -> dict[str, np.ndarray]:
    """Return captured numeric wrist-depth frames with their source boundaries."""
    frames: list[np.ndarray] = []
    boundaries: list[int] = []
    for index, observation in enumerate(timed):
        value = observation.get("wrist_depth")
        if value is None:
            continue
        try:
            frame = np.asarray(value)
        except (TypeError, ValueError):
            return {}
        if frame.ndim == 0 or frame.dtype.hasobject or frame.dtype.kind not in "biufc":
            return {}
        frames.append(frame)
        boundaries.append(index)
    if not frames:
        return {}
    try:
        stacked = np.stack(frames)
    except (TypeError, ValueError):
        return {}
    if stacked.dtype.hasobject or stacked.dtype.kind not in "biufc":
        return {}
    return {
        "wrist_depth_frames": stacked,
        "wrist_depth_frame_boundaries": np.asarray(boundaries, dtype=np.int64),
    }


def _write_episode(write_dir: Path, row: dict) -> None:
    from icgs.data.collection.generation.batch import plan_program_attempts, bounds_from_row
    from icgs.data.collection.generation.episode_record import assemble_episode, classify_generation_outcome
    from icgs.data.training_layout import LAYOUT_VERSION, write_training_episode_layout
    from icgs.data.collection.generation.rlbench_attempt import online_observation_view
    from icgs.data.collection.generation.episode_archive import (
        EpisodeArchiveWriter,
        archive_profile_from_environment,
        archive_writer_cap_from_environment,
    )

    write_dir.mkdir(parents=True, exist_ok=True)
    archive_profile = archive_profile_from_environment()
    archive_job_identity = None
    if archive_profile is not None:
        identity_value = os.environ.get("ICGS_GENERATION_JOB_IDENTITY")
        if identity_value:
            archive_job_identity = json.loads(identity_value)
            if not isinstance(archive_job_identity, dict):
                raise ValueError("ICGS_GENERATION_JOB_IDENTITY must encode an object")
    binding_path = Path(os.environ.get("ICGS_GENERATION_BINDING_JSON", ""))
    if binding_path.is_file():
        row["_binding"] = json.loads(binding_path.read_text(encoding="utf-8"))
    timed = row.get("_timed_obs") or []
    transitions = []
    for index in range(len(timed) - 1):
        transitions.append({
            "command": {
                "T_w_e": timed[index + 1]["T_w_e"],
                "grip": timed[index + 1]["grip"],
                "duration_s": 0.05,
            },
            "achieved_duration_s": 0.05,
            "physics_substeps": 1,
            "before_boundary": index,
            "after_boundary": index + 1,
        })
    binding = row.get("_binding")
    if binding is None:
        manifest_path = Path(os.environ.get(
            "ICGS_GENERATION_APPROVED_MANIFEST",
            str(_DEFAULT_MANIFEST),
        ))
        if manifest_path.is_file():
            catalog = json.loads(manifest_path.read_text(encoding="utf-8")).get("catalog", [])
            binding = next((item for item in catalog if item.get("program_id") == row["program_id"]), None)
    binding = dict(binding or {
        "program_id": row["program_id"],
        "asset_family_id": f"icgs-train-{row.get('family', 'basic-manipulation')}-v1",
        "source_lineage_id": f"{row['program_id'].lower()}-seed-root-v1",
        "split": "train",
        "randomization": {
            "translation_m": {"x": [-0.012, 0.012], "y": [-0.012, 0.012], "z": [0.0, 0.0]},
            "yaw_deg": [-30.0, 30.0],
            "scale": [0.8, 1.2],
            "camera_profile_id": "rlbench-wrist-depth-v1",
        },
    })
    from icgs.data.collection.generation.batch import attempt_from_dict

    if row.get("_plan"):
        plan = attempt_from_dict(row["_plan"])
    else:
        plan = plan_program_attempts(
            row["program_id"],
            n_nominal=1,
            n_perturbed=0,
            bounds=bounds_from_row(binding),
            asset_family_id=binding.get("asset_family_id"),
        )[0]
    def _enc(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _enc(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_enc(v) for v in value]
        return value

    result_class = classify_generation_outcome(
        simulator_crash=row.get("result_class") == "simulator_crash",
        predicates_ok=bool(row.get("success")),
        observation_valid=row.get("result_class") != "invalid_observation",
    )

    def _has_captured_object_state(states) -> bool:
        if not states:
            return False
        for state in states:
            if not isinstance(state, dict):
                if state:
                    return True
                continue
            if state.get("objects"):
                return True
            for key, value in state.items():
                if key == "objects" or value is None:
                    continue
                if isinstance(value, (list, tuple, dict)) and not value:
                    continue
                return True
        return False

    def _archive_record(record_value: dict) -> dict:
        normalized = dict(record_value)
        object_states = normalized.get("object_states")
        if not _has_captured_object_state(object_states):
            normalized.pop("object_states", None)
        robot_states = normalized.get("robot_states")
        if isinstance(robot_states, list):
            normalized["robot_states"] = [
                {
                    key: value for key, value in state.items()
                    if value is not None
                }
                if isinstance(state, dict) else state
                for state in robot_states
            ]
        return normalized

    if result_class in {"simulator_crash", "invalid_observation"}:
        from icgs.data.collection.generation.episode_record import assemble_attempt_record
        valid_until = len(timed) - 1 if timed else None
        attempt = assemble_attempt_record(
            attempt_id=f"att-{plan.episode_id}",
            program_id=row["program_id"],
            outcome=result_class,
            error=row.get("error", row.get("terminal_reason", "observation_incomplete")),
            episode_id=None,
            episode_kind=plan.episode_kind,
            failure_type=row.get("error_type") or ("observation_schema" if result_class == "invalid_observation" else "simulator_exception"),
            terminal_t=len(row.get("_actions") or ()) or None,
            valid_observation_until=valid_until,
        )
        attempt.update({
            "scene_seed": plan.scene_seed,
            "scene_signature": plan.randomization.get("scene_signature"),
            "asset_instance_id": plan.randomization.get("asset_instance_id"),
            "asset_family_id": binding.get("asset_family_id"),
            "split": "dev" if binding.get("split") == "development" else binding.get("split"),
            "dataset_version": GENERATION_PROTOCOL.dataset_version,
            "program_manifest_version": GENERATION_PROTOCOL.program_manifest_version,
            "execution_mode": GENERATION_PROTOCOL.execution_mode,
            "execution_source": GENERATION_PROTOCOL.execution_mode,
            "error_type": row.get("error_type"),
            "traceback": row.get("traceback"),
        })
        if archive_profile is not None:
            from icgs.data.collection.generation.diversity import train_subset_for_sample

            attempt["split"] = "dev" if plan.split == "development" else plan.split
            attempt["subset"] = (
                plan.randomization.get("train_subset")
                or train_subset_for_sample(plan.randomization)
                if plan.split == "train" else None
            )
            actions_value = row.get("_actions")
            prefix: dict[str, np.ndarray] = {
                "actions": np.asarray(() if actions_value is None else actions_value),
            }
            if timed:
                point_frames = [np.asarray(item["points"]).reshape(-1, 3) for item in timed]
                point_offsets = np.zeros(len(point_frames) + 1, dtype=np.int64)
                for index, frame in enumerate(point_frames):
                    point_offsets[index + 1] = point_offsets[index] + len(frame)
                prefix.update({
                    "points": np.concatenate(point_frames, axis=0),
                    "point_offsets": point_offsets,
                    "T_w_e": np.asarray([item["T_w_e"] for item in timed]),
                    "grip": np.asarray([item["grip"] for item in timed]),
                    "point_valid": np.concatenate([
                        np.asarray(item["point_valid"]) for item in timed
                    ]),
                })
                prefix.update(_timed_wrist_depth_arrays(timed))
            debug = {
                key: value for key, value in row.items()
                if key not in {"_timed_obs", "_actions"}
            }
            if not _has_captured_object_state(debug.get("_object_states")):
                debug.pop("_object_states", None)
            debug.update({
                "source_run_id": os.environ.get("ICGS_GENERATION_RUN_ID") or row.get("run_id") or "script-local",
                "code_revision": os.environ.get("ICGS_GENERATION_CODE_REVISION") or row.get("code_revision") or "script-local",
                "preprocessing_identity": "rlbench_script_measured_v1",
            })
            if archive_job_identity is not None:
                debug["job_identity"] = archive_job_identity
            EpisodeArchiveWriter(
                archive_profile, max_result_bytes=archive_writer_cap_from_environment()
            ).write_attempt(
                attempt,
                prefix_arrays=prefix,
                debug_metadata=debug,
                output_dir=write_dir,
            )
            return
        (write_dir / "attempt.json").write_text(json.dumps(_enc(attempt), indent=2) + "\n")
        point_frames = [np.asarray(item["points"], dtype=np.float32).reshape(-1, 3) for item in timed]
        point_offsets = np.zeros(len(point_frames) + 1, dtype=np.int64)
        for index, frame in enumerate(point_frames):
            point_offsets[index + 1] = point_offsets[index] + len(frame)
        point_values = (
            np.concatenate(point_frames, axis=0)
            if point_frames else np.empty((0, 3), dtype=np.float32)
        )
        np.savez_compressed(
            write_dir / "valid_prefix.npz",
            actions=np.asarray(row.get("_actions") or (), dtype=np.float64),
            points=point_values,
            point_offsets=point_offsets,
            T_w_e=np.asarray([item["T_w_e"] for item in timed], dtype=np.float64),
            grip=np.asarray([item["grip"] for item in timed], dtype=np.float32),
        )
        files = {
            str(path.relative_to(write_dir)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for path in write_dir.iterdir() if path.is_file()
        }
        (write_dir / "artifact_manifest.json").write_text(json.dumps({**attempt, "files": files}, indent=2) + "\n")
        return
    online_timed = [online_observation_view(item) for item in timed]
    record = assemble_episode(
        plan=plan,
        binding=binding,
        observations=online_timed,
        transitions=transitions,
        result_class=result_class,
        intervention=row.get("intervention"),
        robot_states=row.get("_robot_states"),
        object_states=row.get("_object_states"),
        task_labels=row.get("_task_labels"),
    )
    if row.get("_sensor_randomization") is not None:
        record["sensor_randomization"] = row["_sensor_randomization"]
    execution = {
        k: v for k, v in row.items()
        if k not in {"_timed_obs", "_actions", "_robot_states", "_object_states", "_task_labels"}
    }
    execution["outcome"] = result_class
    execution.update({
        "source_run_id": os.environ.get("ICGS_GENERATION_RUN_ID") or row.get("run_id") or "script-local",
        "code_revision": os.environ.get("ICGS_GENERATION_CODE_REVISION") or row.get("code_revision") or "script-local",
        "preprocessing_identity": "rlbench_script_measured_v1",
    })
    if archive_profile is not None:
        actions_value = row.get("_actions")
        if archive_job_identity is not None:
            execution["job_identity"] = archive_job_identity
        EpisodeArchiveWriter(
            archive_profile, max_result_bytes=archive_writer_cap_from_environment()
        ).write_episode(
            _archive_record(record),
            raw_arrays={
                "actions": np.asarray(() if actions_value is None else actions_value),
                **_timed_wrist_depth_arrays(timed),
            },
            debug_metadata=execution,
            output_dir=write_dir,
        )
        return
    (write_dir / "episode.json").write_text(json.dumps(_enc(record), indent=2) + "\n")
    execution = {
        k: v for k, v in row.items()
        if k not in {"_timed_obs", "result_class"}
    }
    execution["outcome"] = result_class
    (write_dir / "execution.json").write_text(json.dumps(_enc(execution), indent=2) + "\n")

    def _pose_vector(transform) -> np.ndarray:
        matrix = np.asarray(transform, dtype=np.float64)
        rotation = matrix[:3, :3]
        trace = float(np.trace(rotation))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            diagonal = np.diag(rotation)
            index = int(np.argmax(diagonal))
            if index == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif index == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        return np.asarray([*matrix[:3, 3], qx, qy, qz, qw], dtype=np.float64)

    action_array = np.stack([
        np.concatenate([_pose_vector(item["command"]["T_w_e"]), [item["command"]["grip"]]])
        for item in transitions
    ])
    ee_poses = np.stack([np.asarray(item["T_w_e"], dtype=np.float64) for item in timed])
    gripper = np.asarray([item["grip"] for item in timed], dtype=np.float32)
    write_training_episode_layout(write_dir / "layout", {
        "episode": {
            "episode_id": plan.episode_id,
            "program_id": row["program_id"],
            "split": binding["split"],
            "layout_version": LAYOUT_VERSION,
            "outcome": result_class,
        },
        "observations": {"pointcloud": [item["points"] for item in timed]},
        "robot": {"ee_pose": ee_poses, "gripper": gripper},
        "actions": action_array,
        "task": {
            "events": binding.get("structured_steps") or binding.get("events") or [
                {"event_id": f"event_{index:02d}", "primitive": primitive}
                for index, primitive in enumerate(row.get("events") or ())
            ],
            "collisions": [],
            **(row.get("_task_labels") or {}),
        },
        "result": {
            "success": result_class == "success",
            "outcome": result_class,
            "terminal_reason": "predicate_satisfied" if result_class == "success" else "predicate_failed",
        },
    })
    from icgs.data.datasets.generation_views import build_generation_view
    views_dir = write_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    for view_name in ("D_geom", "D_temporal", "D_dyn", "D_task"):
        (views_dir / f"{view_name}.json").write_text(json.dumps({
            "view": view_name,
            "episode_id": plan.episode_id,
            "pointers": build_generation_view([record], view_name, role="all", mix=False),
        }, indent=2, default=_enc) + "\n")
    np.savez_compressed(
        write_dir / "telemetry.npz",
        ee_pose=ee_poses,
        gripper=gripper,
        action=action_array,
        achieved_dt=np.asarray(record["dt"], dtype=np.float64),
    )
    files = {}
    for path in sorted(candidate for candidate in write_dir.rglob("*") if candidate.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        files[str(path.relative_to(write_dir))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (write_dir / "artifact_manifest.json").write_text(json.dumps({
        "attempt_id": f"att-{plan.episode_id}",
        "episode_id": plan.episode_id,
        "program_id": row["program_id"],
        "outcome": result_class,
        "files": files,
    }, indent=2) + "\n")


def load_task_class(spec):
    module = importlib.import_module(f"rlbench.tasks.{spec.module}")
    return getattr(module, spec.class_name)


def apply_prepared_layout(objects: dict) -> None:
    import math
    from pyrep.const import PrimitiveShape
    from pyrep.objects.shape import Shape

    for name, item in objects.items():
        pos = item["pos"] if isinstance(item, dict) else item[0]
        found = find_shape(name)
        yaw = item.get("yaw_deg") if isinstance(item, dict) else None
        size = item.get("size") if isinstance(item, dict) else None
        if found is not None:
            try:
                found.set_position(list(pos))
            except Exception:
                pass
            if yaw is not None:
                try:
                    found.set_orientation([0.0, 0.0, math.radians(float(yaw))])
                except Exception:
                    pass
            if size is not None and isinstance(found, Shape):
                # Pinned PyRep exposes relative scale_object, not set_size.
                # Convert the prepared absolute dimensions using local bounds;
                # repeated application must not compound the scale.
                bounds = np.asarray(found.get_bounding_box(), dtype=float).reshape(3, 2)
                current_size = bounds[:, 1] - bounds[:, 0]
                desired_size = np.asarray(size, dtype=float)
                if (desired_size.shape != (3,) or not np.isfinite(desired_size).all()
                        or np.any(desired_size <= 0) or np.any(current_size <= 0)):
                    raise ValueError(f"invalid physical shape dimensions for {name}")
                found.scale_object(*(desired_size / current_size))
            continue
        if isinstance(item, dict) and (item.get("declared") or name == "inserted_blocker"):
            try:
                shape = Shape.create(
                    PrimitiveShape.CUBOID,
                    size=item.get("size") or [0.04, 0.04, 0.04],
                    mass=0.05,
                    static=False,
                    respondable=True,
                    position=list(pos),
                    color=item.get("color") or [0.1, 0.1, 0.1],
                )
                shape.set_name(name)
                if yaw is not None:
                    try:
                        shape.set_orientation([0.0, 0.0, math.radians(float(yaw))])
                    except Exception:
                        pass
            except Exception:
                pass


def run_program(env, spec, plan=None, *, capture: dict | None = None) -> dict:
    from icgs.data.collection.generation.attempt_prep import prepare_attempt
    from icgs.data.collection.generation.simulator_randomization import apply_sensor_randomization

    capture_wrist_depth = bool(os.environ.get("ICGS_GENERATION_ARCHIVE_PROFILE", "").strip())
    task_cls = load_task_class(spec)
    task = env.get_task(task_cls)
    desc, obs = task.reset()
    prepared = None
    routine = list(spec.routine)
    if plan is not None:
        prepared = prepare_attempt(plan, spec.objects, spec.routine)
        apply_prepared_layout(prepared["objects"])
        routine = list(prepared["routine"])
    sensor_randomization = None
    if plan is not None:
        from pyrep.objects.light import Light
        from pyrep.objects.vision_sensor import VisionSensor

        def find_object(cls, names):
            for name in names:
                try:
                    return cls(name)
                except Exception:
                    continue
            return None

        camera = find_object(VisionSensor, ("wrist_camera", "wrist_camera#0", "cam_wrist"))
        lights = []
        for name in ("DefaultLightA", "DefaultLightB", "DefaultLightC", "DefaultLightD"):
            found = find_object(Light, (name,))
            if found is not None and all(found is not item for item in lights):
                lights.append(found)
        from pyrep.backend import sim
        class AmbientTarget:
            @staticmethod
            def set_ambient_light(rgb):
                values = sim.ffi.new("simFloat[]", list(rgb))
                sim.simSetArrayParameter(sim.sim_arrayparam_ambient_light, values)

        sensor_randomization = apply_sensor_randomization(
            camera=camera,
            lights=lights,
            ambient_target=AmbientTarget(),
            camera_viewpoint=plan.randomization.get("camera_viewpoint_applied") or {},
            lighting_profile=plan.randomization.get("lighting_applied") or {},
        )
    n_obs = 1
    n_actions = 0
    quat = _normalize_quaternion(np.asarray(env._scene.robot.arm.get_tip().get_quaternion(), dtype=np.float64))
    command_quat = quat.copy()
    sim_time = 0.0
    dt = 0.05
    timed_obs: list[dict] = []
    actions_series: list[np.ndarray] = []
    robot_states: list[dict] = []
    object_states: list[dict] = []
    if capture is not None:
        capture.update({
            "_timed_obs": timed_obs,
            "_actions": actions_series,
            "_robot_states": robot_states,
            "_object_states": object_states,
            "_sensor_randomization": sensor_randomization,
        })

    def _matrix_from_pose(pose) -> np.ndarray:
        value = np.asarray(pose, dtype=np.float64)
        if value.shape == (4, 4):
            return value
        if value.shape != (7,):
            raise ValueError("gripper_pose must be [x,y,z,qx,qy,qz,qw]")
        x, y, z, qx, qy, qz, qw = value
        norm = float(np.linalg.norm([qx, qy, qz, qw]))
        if norm <= 0:
            raise ValueError("gripper_pose quaternion must be nonzero")
        qx, qy, qz, qw = np.asarray([qx, qy, qz, qw]) / norm
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray([
            [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
        ])
        matrix[:3, 3] = value[:3]
        return matrix

    def snapshot_obs(raw_obs, grip: float) -> dict:
        tip = env._scene.robot.arm.get_tip()
        pos = np.asarray(tip.get_position(), dtype=np.float64)
        q = np.asarray(tip.get_quaternion(), dtype=np.float64)
        rot = np.array([
            [1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[2]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
            [2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[0]*q[3])],
            [2*(q[0]*q[2] - q[1]*q[3]), 2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[0]**2 + q[1]**2)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        measured_pose = getattr(raw_obs, "gripper_pose", None)
        T = _matrix_from_pose(measured_pose) if measured_pose is not None else T
        measured_points = getattr(raw_obs, "wrist_point_cloud", None)
        cloud = np.asarray(measured_points, dtype=np.float32).reshape(-1, 3) if measured_points is not None else np.empty((0, 3), dtype=np.float32)
        measured_grip = getattr(raw_obs, "gripper_open", None)
        actual_grip = 0 if float(measured_grip if measured_grip is not None else grip) < 0.5 else 1
        row = {
            "points": cloud,
            "point_valid": np.ones((cloud.shape[0],), dtype=bool),
            "T_w_e": T,
            "grip": actual_grip,
        }
        if capture_wrist_depth:
            wrist_depth = getattr(raw_obs, "wrist_depth", None)
            if wrist_depth is not None:
                row["wrist_depth"] = wrist_depth
        for key in ("joint_positions", "joint_velocities"):
            value = getattr(raw_obs, key, None)
            if value is not None:
                row[key] = np.asarray(value, dtype=np.float64)
        return row

    def capture_scene_state() -> dict:
        objects = []
        for name in spec.objects:
            shape = find_shape(name)
            if shape is None:
                continue
            item = {"name": name, "position": [float(v) for v in shape.get_position()], "valid": True}
            try:
                item["orientation_xyzw"] = [float(v) for v in shape.get_quaternion()]
            except Exception:
                pass
            try:
                velocity = shape.get_velocity()
                item["linear_velocity"] = [float(v) for v in velocity[0]]
                item["angular_velocity"] = [float(v) for v in velocity[1]]
            except Exception:
                pass
            objects.append(item)
        return {"objects": objects}

    def advance(action, fallback_grip: float):
        nonlocal sim_time, n_actions, n_obs
        result = task.step(action)
        raw = result[0] if isinstance(result, tuple) else result
        sim_time += dt
        n_actions += 1
        n_obs += 1
        actions_series.append(np.asarray(action, dtype=np.float64).copy())
        observed = snapshot_obs(raw, fallback_grip)
        timed_obs.append(observed)
        robot_states.append({
            "T_w_e": observed["T_w_e"],
            "grip": observed["grip"],
            "joint_positions": observed.get("joint_positions"),
            "joint_velocities": observed.get("joint_velocities"),
        })
        object_states.append(capture_scene_state())
        return raw
    trace_enabled = os.environ.get("ICGS_GENERATION_TRACE", "").strip().lower() in {"1", "true", "yes"}
    controller_trace = []

    def trace_event(kind: str, **fields) -> None:
        if not trace_enabled:
            return
        event = {"kind": kind, **fields}
        try:
            event["tip"] = [float(v) for v in env._scene.robot.arm.get_tip().get_position()]
            event["grasped"] = [v.get_name() for v in env._scene.robot.gripper.get_grasped_objects()]
            event["gripper_open_amount"] = [float(v) for v in env._scene.robot.gripper.get_open_amount()]
            event["objects"] = live_poses(spec.objects)
            event["object_physics"] = {}
            for name in spec.objects:
                shape = find_shape(name)
                if isinstance(shape, Shape):
                    parent = shape.get_parent()
                    event["object_physics"][name] = {
                        "parent": parent.get_name() if parent is not None else None,
                        "dynamic": shape.is_dynamic(),
                        "respondable": shape.is_respondable(),
                        "local_bounds": [float(v) for v in shape.get_bounding_box()],
                    }
        except Exception as exc:
            event["trace_error"] = f"{type(exc).__name__}: {exc}"
        controller_trace.append(event)

    def _is_ik_error(exc: BaseException) -> bool:
        name = type(exc).__name__
        text = str(exc)
        return "IK" in name or "IK" in text or "Jacobian" in text or "InvalidAction" in name

    initial = snapshot_obs(obs, 1.0)
    timed_obs.append(initial)
    robot_states.append({"T_w_e": initial["T_w_e"], "grip": initial["grip"], "joint_positions": initial.get("joint_positions"), "joint_velocities": initial.get("joint_velocities")})
    object_states.append(capture_scene_state())

    if plan is not None:
        # Procedural assets are spawned slightly above the support surface. Let
        # them settle before taking the first scripted pose snapshot; otherwise
        # grasp commands target the stale spawn height and can push an object away.
        settle_tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
        for _ in range(SCENE_SETTLE_STEPS):
            try:
                advance(np.concatenate([settle_tip, command_quat, [1.0]]), 1.0)
            except Exception as exc:
                if _is_ik_error(exc):
                    break
                raise

        # Targets are visual/condition markers. Their catalog height is the
        # nominal spawn height, while scaled dynamic objects settle at a
        # scale-dependent support height. Align only the marker's vertical
        # coordinate to the measured settled object; preserve randomized x/y
        # and never alter articulated-handle targets.
        for obj_name, target_name, _tolerance in spec.conditions:
            if "handle" in obj_name or "handle" in target_name:
                continue
            obj = find_shape(obj_name)
            target = find_shape(target_name)
            if obj is None or target is None:
                continue
            try:
                target_pos = list(target.get_position())
                # Container/drawer targets are intentionally elevated above
                # the table; they already encode their support surface and
                # must not be lowered to the table-settled object height.
                if float(target_pos[2]) > 0.78:
                    continue
                target_pos[2] = float(obj.get_position()[2])
                target.set_position(target_pos)
            except Exception:
                pass

    def move_ik(target_pos, grip: float, orientation=None) -> None:
        nonlocal sim_time, n_actions, n_obs
        target = np.asarray(target_pos, dtype=np.float64)
        target_quat = command_quat if orientation is None else _normalize_quaternion(orientation)
        trace_event("move_begin", target=target.tolist(), grip=float(grip))
        max_step = 0.012
        for _ in range(120):
            curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            delta = target - curr
            dist = float(np.linalg.norm(delta))
            if dist < 0.007:
                action = np.concatenate([target, target_quat, [grip]])
                try:
                    advance(action, grip)
                except Exception as exc:
                    if not _is_ik_error(exc):
                        raise
                    trace_event("ik_error", error=str(exc))
                trace_event("move_end", target=target.tolist())
                return
            scale = min(1.0, max_step / dist)
            nxt = curr + delta * scale
            action = np.concatenate([nxt, target_quat, [grip]])
            try:
                advance(action, grip)
            except Exception as exc:
                if not _is_ik_error(exc):
                    raise
                max_step *= 0.5
                trace_event("ik_error", error=str(exc), max_step=max_step)
                if max_step < 0.002:
                    return
                continue
        trace_event("move_exhausted", target=target.tolist())

    def actuate(grip: float, obj=None) -> None:
        nonlocal sim_time, n_actions, n_obs
        tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
        action = np.concatenate([tip, command_quat, [grip]])
        trace_event("grip_before", grip=float(grip), obj=None if obj is None else str(obj))
        held = find_shape(str(grasped_name or "")) if grip > 0.5 and grasped_name else None
        guard = release_collision_guard(held) if held is not None else nullcontext()
        with guard:
            if grip > 0.5:
                try:
                    env._scene.robot.gripper.release()
                except Exception:
                    pass
            for _ in range(grip_hold_steps):
                try:
                    advance(action, grip)
                except Exception as exc:
                    if _is_ik_error(exc):
                        break
                    raise
        trace_event("grip_after", grip=float(grip), obj=None if obj is None else str(obj))
        if grip < 0.5 and obj is not None:
            try:
                attached = bool(env._scene.robot.gripper.grasp(obj))
                trace_event("explicit_grasp", obj=str(obj), attached=attached)
            except Exception:
                trace_event("explicit_grasp_error", obj=str(obj))

    offset = np.zeros(3, dtype=np.float64)
    grasped_name = None
    rotation_checks = []
    grip_hold_steps = 5
    for step in routine:
        delta = step.get("grip_timing_delta_intervals")
        if delta is not None:
            grip_hold_steps = max(1, 5 + 5 * int(delta))
            break

    def _is_mechanism(name: str | None) -> bool:
        return bool(name) and "handle" in str(name)

    def _freeze(obj) -> None:
        if obj is None:
            return
        for method, args in (
            ("set_parent", (None,)),
            ("set_dynamic", (False,)),
            ("set_respondable", (False,)),
        ):
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn(*args)
                except Exception:
                    pass

    push_objs = {s.get("obj") for s in spec.routine if s.get("type") == "push"}
    for step in routine:
        # Refresh after every completed primitive: dynamic objects and
        # articulated handles may move while the previous primitive runs.
        poses = live_poses(spec.objects)
        exec_step = dict(step)
        if exec_step.get("obj") and exec_step["type"] in {
            "grasp", "lift", "pick_place", "place", "temporary_place", "regrasp",
            "open_articulation", "close_articulation",
        }:
            grasped_name = grasped_name or exec_step.get("obj")
        for motion in plan_step(exec_step, poses):
            kind = motion["kind"]
            trace_event("motion", step_type=str(exec_step.get("type")), motion_kind=str(kind),
                        grip=float(motion.get("grip", 1.0)))
            if kind in {"move", "slide"}:
                xyz = np.asarray(motion["xyz"], dtype=np.float64)
                if motion.get("grasp") or motion.get("frame") == "object":
                    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                    held = find_shape(str(grasped_name or step.get("obj") or ""))
                    if held is not None:
                        offset = np.asarray(held.get_position(), dtype=np.float64) - tip
                    xyz = xyz - offset
                    place_like = exec_step.get("type") in {
                        "place", "pick_place", "temporary_place", "fit", "retrieve", "park", "restore",
                    }
                    if (place_like and float(exec_step.get("place_z", 0.0)) > 0.0
                            and xyz[2] < 0.86 and motion.get("kind") != "slide"):
                        xyz[2] -= float(GENERATION_PROTOCOL.place_ik_z_shortfall_m)
                    off = exec_step.get("place_offset_m") or {}
                    xyz[0] += float(off.get("dx_m", 0.0))
                    xyz[1] += float(off.get("dy_m", 0.0))
                move_ik(xyz, float(motion.get("grip", 1.0)))
            elif kind == "grip":
                if float(motion["grip"]) > 0.5 and grasped_name:
                    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                    for _ in range(8):
                        try:
                            advance(np.concatenate([tip, command_quat, [0.0]]), 0.0)
                        except Exception as exc:
                            if _is_ik_error(exc):
                                break
                            raise
                obj = find_shape(str(motion["grasp_obj"])) if motion.get("grasp_obj") else None
                if obj is not None and float(motion["grip"]) < 0.5:
                    # A previously placed object may be parked as a settled,
                    # non-respondable body. Re-enable its normal physics only
                    # when the next explicit grasp needs it.
                    for method, args in (
                        ("set_dynamic", (True,)),
                        ("set_respondable", (True,)),
                        ("set_collidable", (True,)),
                    ):
                        fn = getattr(obj, method, None)
                        if callable(fn):
                            try:
                                fn(*args)
                            except Exception:
                                pass
                actuate(float(motion["grip"]), obj)
                if float(motion["grip"]) < 0.5 and obj is not None:
                    grasped_name = str(motion.get("grasp_obj") or "")
                    obj_pos = np.asarray(obj.get_position(), dtype=np.float64)
                    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                    if not grasp_attachment_confirmed(env._scene.robot.gripper, obj):
                        # A failed grasp must not be followed by a long
                        # transport that produces a misleading placement
                        # failure. Give the simulator a few deterministic
                        # re-grasp opportunities at the measured pose.
                        for _ in range(2):
                            actuate(0.0, obj)
                            obj_pos = np.asarray(obj.get_position(), dtype=np.float64)
                            tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                            if grasp_attachment_confirmed(env._scene.robot.gripper, obj):
                                break
                    offset = obj_pos - tip
                    trace_event("post_grasp_check", obj=str(motion.get("grasp_obj")),
                                distance_m=float(np.linalg.norm(obj_pos - tip)),
                                confirmed=grasp_attachment_confirmed(env._scene.robot.gripper, obj))
                else:
                    held = find_shape(str(grasped_name or "")) if grasped_name else None
                    guard = (nullcontext() if held is None or _is_mechanism(grasped_name)
                             else release_collision_guard(held))
                    with guard:
                        try:
                            env._scene.robot.gripper.release()
                        except Exception:
                            pass
                        if held is not None and _is_mechanism(grasped_name):
                            target_name = exec_step.get("target")
                            marker = find_shape(str(target_name)) if target_name else None
                            if marker is not None:
                                try:
                                    held.set_position(list(marker.get_position()))
                                except Exception:
                                    pass
                            _freeze(held)
                        elif held is not None:
                            for method, args in (
                                ("set_parent", (None,)),
                                ("set_dynamic", (True,)),
                            ):
                                fn = getattr(held, method, None)
                                if callable(fn):
                                    try:
                                        fn(*args)
                                    except Exception:
                                        pass
                        tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                        for _ in range(PLACEMENT_SETTLE_STEPS):
                            try:
                                advance(np.concatenate([tip, command_quat, [1.0]]), 1.0)
                            except Exception as exc:
                                if _is_ik_error(exc):
                                    break
                                raise
                        retreat = np.array([tip[0], tip[1], min(0.92, float(tip[2]) + 0.08)])
                        move_ik(retreat, 1.0)
                        for _ in range(12):
                            try:
                                advance(np.concatenate([retreat, command_quat, [1.0]]), 1.0)
                            except Exception as exc:
                                if _is_ik_error(exc):
                                    break
                                raise
                    if held is not None and not _is_mechanism(grasped_name):
                        for method, args in (
                            ("set_dynamic", (False,)),
                            ("set_respondable", (False,)),
                            ("set_collidable", (False,)),
                        ):
                            fn = getattr(held, method, None)
                            if callable(fn):
                                try:
                                    fn(*args)
                                except Exception:
                                    pass
                    offset = np.zeros(3, dtype=np.float64)
                    grasped_name = None
                    trace_event("post_release", obj=None if held is None else str(held))
            elif kind == "pause":
                tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                hold = float(motion.get("grip", 1.0))
                for _ in range(int(motion["intervals"]) * 5):
                    try:
                        advance(np.concatenate([tip, command_quat, [hold]]), hold)
                    except Exception as exc:
                        if _is_ik_error(exc):
                            break
                        raise
            elif kind == "rotate":
                rotated_name = str(grasped_name or exec_step.get("obj") or "")
                rotated_obj = find_shape(rotated_name) if rotated_name else None
                before_orientation = None
                if rotated_obj is not None:
                    try:
                        before_orientation = np.asarray(rotated_obj.get_quaternion(), dtype=np.float64)
                    except Exception:
                        before_orientation = None
                start_quat = command_quat.copy()
                requested_yaw = float(motion.get("yaw_deg", 0.0))
                target_quat = rotate_quaternion(start_quat, requested_yaw)
                tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                increments = max(1, int(np.ceil(abs(requested_yaw) / 5.0)))
                for index in range(1, increments + 1):
                    move_ik(tip, 0.0, interpolate_quaternion(start_quat, target_quat, index / increments))
                command_quat = target_quat
                measured_deg = None
                if rotated_obj is not None and before_orientation is not None:
                    try:
                        measured_deg = quaternion_angle_deg(before_orientation, rotated_obj.get_quaternion())
                    except Exception:
                        measured_deg = None
                rotation_checks.append({
                    "object": rotated_name,
                    "requested_yaw_deg": requested_yaw,
                    "measured_angle_deg": measured_deg,
                    "tolerance_deg": 10.0,
                    "passed": measured_deg is not None and abs(measured_deg - abs(requested_yaw)) <= 10.0,
                })

    def _condition_distance(obj_a: str, obj_b: str) -> float | None:
        sa, sb = find_shape(obj_a), find_shape(obj_b)
        if sa is None or sb is None:
            return None
        return float(np.linalg.norm(np.asarray(sa.get_position()) - np.asarray(sb.get_position())))

    poses = live_poses(spec.objects)
    for item in spec.conditions:
        obj_a, obj_b = item[0], item[1]
        if "handle" in obj_a or "handle" in obj_b:
            continue
        dist = _condition_distance(obj_a, obj_b)
        if dist is None or dist <= 0.01:
            continue
        sa = find_shape(obj_a)
        if sa is not None and float(sa.get_position()[2]) < 0.55:
            continue
        if dist <= 0.01:
            continue
        if dist > 0.28:
            continue
        retry_count = 3 if obj_a in push_objs else 1
        for _retry_index in range(retry_count):
            retry = (
                {"type": "push", "obj": obj_a, "target": obj_b, "push_z": 0.02}
                if obj_a in push_objs
                else placement_recovery_step(routine, obj_a, obj_b)
            )
            grasped_name = obj_a
            for motion in plan_step(retry, live_poses(spec.objects)):
                kind = motion["kind"]
                if kind in {"move", "slide"}:
                    xyz = np.asarray(motion["xyz"], dtype=np.float64)
                    if motion.get("grasp") or motion.get("frame") == "object":
                        tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                        held = find_shape(obj_a)
                        if held is not None:
                            offset = np.asarray(held.get_position(), dtype=np.float64) - tip
                        xyz = xyz - offset
                        if (float(retry.get("place_z", 0.0)) > 0.0
                                and xyz[2] < 0.86):
                            xyz[2] -= float(GENERATION_PROTOCOL.place_ik_z_shortfall_m)
                    move_ik(xyz, float(motion.get("grip", 1.0)))
                elif kind == "grip":
                    obj = find_shape(obj_a) if float(motion.get("grip", 1)) < 0.5 else None
                    actuate(float(motion["grip"]), obj)
            if obj_a in push_objs:
                remaining = _condition_distance(obj_a, obj_b)
                if remaining is None or remaining <= 0.01:
                    break

    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
    terminal_grip = terminal_settle_grip(timed_obs)
    for _ in range(15):
        try:
            advance(np.concatenate([tip, command_quat, [terminal_grip]]), terminal_grip)
        except Exception as exc:
            if _is_ik_error(exc):
                break
            raise
    success, _ = env._scene.task.success()
    rotation_ok = all(item["passed"] for item in rotation_checks)
    distances = []
    for obj_a, obj_b, _tol in spec.conditions:
        sa, sb = find_shape(obj_a), find_shape(obj_b)
        if sa is None or sb is None:
            distances.append({"a": obj_a, "b": obj_b, "distance_m": None})
            continue
        dist = float(np.linalg.norm(np.asarray(sa.get_position()) - np.asarray(sb.get_position())))
        distances.append({"a": obj_a, "b": obj_b, "distance_m": dist})
    timeline_ok = n_actions > 0 and n_obs == n_actions + 1
    observation_valid = bool(timed_obs) and all(
        np.asarray(item.get("points")).ndim == 2
        and np.asarray(item.get("points")).shape[1] == 3
        and len(item.get("points")) > 0
        and np.isfinite(np.asarray(item.get("points"))).all()
        for item in timed_obs
    )
    from icgs.data.collection.generation.task_labels import materialize_task_labels
    task_labels = materialize_task_labels(spec.events, object_states, robot_states)
    intervention = None if plan is None or prepared is None else prepared.get("intervention")
    if isinstance(intervention, dict) and intervention.get("intervention_frame") is None:
        intervention = dict(intervention)
        intervention["intervention_frame"] = 0 if intervention.get("kind") in {
            "object_displacement", "blocker_insertion", "pause_hold",
        } else min(1, max(0, n_actions - 1))
    return {
        "program_id": spec.program_id,
        "family": spec.family,
        "description": desc,
        "success": bool(success and observation_valid and rotation_ok),
        "result_class": "success" if success and observation_valid and rotation_ok else ("valid_failure" if observation_valid else "invalid_observation"),
        "n_actions": n_actions,
        "n_obs": n_obs,
        "timeline_ok": timeline_ok,
        "sim_time_s": sim_time,
        "events": [event["primitive"] for event in spec.events],
        "routine": [step["type"] for step in spec.routine],
        "predicate_distances_m": distances,
        "rotation_checks": rotation_checks,
        "object_xyz": {
            name: [float(v) for v in find_shape(name).get_position()]
            for name in spec.objects
            if find_shape(name) is not None
        },
        "_timed_obs": timed_obs,
        "_actions": actions_series,
        "_robot_states": robot_states,
        "_object_states": object_states,
        "_task_labels": task_labels,
        "_sensor_randomization": sensor_randomization,
        "_plan": None if plan is None else plan.as_dict(),
        "episode_kind": None if plan is None else plan.episode_kind,
        "intervention": intervention,
        **({"controller_trace": controller_trace} if trace_enabled else {}),
    }


def main() -> int:
    from icgs.data.collection.generation.steps import GENERATION_PROGRAMS

    wanted = sys.argv[1:] or list(GENERATION_PROGRAMS)
    compiled = compile_generation_catalog()
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(False)
    obs_config.wrist_camera.point_cloud = True
    obs_config.wrist_camera.depth = True
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.joint_positions = True
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(collision_checking=False),
        # The expert explicitly grasps its named object in actuate(). Native
        # auto-grasp attaches every detected neighbour (e.g. a drawer handle).
        gripper_action_mode=Discrete(attach_grasped_objects=False),
    )
    results = []
    env = None
    write_root_value = os.environ.get("ICGS_GENERATION_WRITE_EPISODE")
    archive_profile_enabled = bool(os.environ.get("ICGS_GENERATION_ARCHIVE_PROFILE", "").strip())
    try:
        for program_id in wanted:
            spec = compiled[program_id]
            print(f"START {program_id}", flush=True)
            if env is not None:
                try:
                    env.shutdown()
                except Exception:
                    pass
                env = None
            env = Environment(action_mode, "./", obs_config=obs_config, headless=True)
            env.launch()
            plan = None
            plan_path = os.environ.get("ICGS_GENERATION_ATTEMPT_JSON")
            if plan_path:
                from icgs.data.collection.generation.batch import attempt_from_dict
                payload = json.loads(Path(plan_path).read_text())
                plan = attempt_from_dict(payload)
            captured_prefix = (
                {}
                if archive_profile_enabled
                else None
            )
            write_marker = None
            if archive_profile_enabled and write_root_value:
                write_marker = _begin_archive_write_marker(
                    Path(write_root_value),
                    program_id,
                    plan.episode_id if plan is not None else None,
                )
            previous_sigterm = None
            try:
                if archive_profile_enabled:
                    previous_sigterm = signal.signal(signal.SIGTERM, _raise_for_generation_timeout)
                try:
                    row = run_program(env, spec, plan=plan, capture=captured_prefix)
                except Exception as exc:
                    row = {
                        "program_id": program_id,
                        "family": spec.family,
                        "success": False,
                        "result_class": "simulator_crash",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                        "timeout": isinstance(exc, _GenerationTimeoutSignal),
                        "events": [event["primitive"] for event in spec.events],
                        "routine": [step["type"] for step in spec.routine],
                        "_plan": None if plan is None else plan.as_dict(),
                    }
                    if captured_prefix is not None:
                        row.update(captured_prefix)
                finally:
                    if previous_sigterm is not None:
                        signal.signal(signal.SIGTERM, previous_sigterm)
                        previous_sigterm = None
                results.append(row)
                public = {k: v for k, v in row.items() if not k.startswith("_")}
                print(json.dumps(public), flush=True)
                if write_root_value:
                    write_root = Path(write_root_value)
                    _write_episode(write_root / program_id, row)
                    if write_marker is not None:
                        write_marker.unlink(missing_ok=True)
            finally:
                if previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, previous_sigterm)
    finally:
        if env is not None:
            try:
                env.shutdown()
            except Exception:
                pass
    summary = {
        "n": len(results),
        "success": [r["program_id"] for r in results if r.get("success")],
        "valid_failure": [r["program_id"] for r in results if r.get("result_class") == "valid_failure"],
        "simulator_crash": [r["program_id"] for r in results if r.get("result_class") == "simulator_crash"],
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    if write_root_value:
        Path(write_root_value).mkdir(parents=True, exist_ok=True)
        (Path(write_root_value) / "pilot_episode_results.json").write_text(
            json.dumps({"summary": summary, "results": [
                {k: v for k, v in row.items() if not k.startswith("_")} for row in results
            ]}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
