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
from icgs.data.collection.generation.layout_clearance import ELONGATED_ASPECT_RATIO, grasp_yaw_delta_deg
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
# Declared controller interval (manifest ``controller.step_s``). It is the
# nominal command interval only; achieved durations are measured per action.
NOMINAL_CONTROL_INTERVAL_S = 0.05


def transition_timing_at(timing, index: int, expected: int) -> dict:
    """Return measured timing for one transition; never infer it from counts."""
    if timing is None or len(timing) != expected:
        raise ValueError(
            "measured transition timing must cover every recorded action "
            f"(expected {expected}, found {None if timing is None else len(timing)})"
        )
    item = timing[index]
    achieved = float(item["achieved_duration_s"])
    substeps = int(item["physics_substeps"])
    if not np.isfinite(achieved) or achieved <= 0.0 or substeps < 1:
        raise ValueError(f"invalid measured transition timing at {index}: {item}")
    return {"achieved_duration_s": achieved, "physics_substeps": substeps}


def simulation_clock(env):
    """Return a simulator-time reader and physics timestep for measured timing."""
    from pyrep.backend import sim as pyrep_sim

    pyrep = getattr(env, "_pyrep", None) or env._scene.pyrep
    physics_dt = float(pyrep.get_simulation_timestep())
    if not np.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError(f"invalid simulator timestep: {physics_dt}")
    return (lambda: float(pyrep_sim.simGetSimulationTime())), physics_dt


# RLBench's Discrete gripper treats a gripper whose fingers are all open above
# this fraction as open: a later close command enters its release branch and
# drops a body the explicit grasp had attached.
DISCRETE_OPEN_THRESHOLD = 0.9
def grasp_attachment_confirmed(gripper, obj) -> bool:
    """Attached alone, and closed enough for the action mode to keep holding it.

    Proximity is not attachment; unintended co-grasps are rejected.
    """
    if list(gripper.get_grasped_objects()) != [obj]:
        return False
    open_amount = getattr(gripper, "get_open_amount", None)
    if callable(open_amount):
        return not all(float(value) > DISCRETE_OPEN_THRESHOLD for value in open_amount())
    return True


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


# Carried bodies are released this far above their measured rest height and
# settle under physics; they are never pressed into, or snapped onto, a support.
PLACE_RELEASE_CLEARANCE_M = 0.004
# Success requires every final predicate on this many trailing boundaries.
FINAL_HOLD_BOUNDARIES = 5
# Bounded, executed corrective attempts for one failed routine step.
MAX_STEP_RETRIES = 2
PLACE_STEP_TYPES = frozenset({
    "place", "pick_place", "temporary_place", "fit", "retrieve", "park", "restore",
    "transport_through_aperture",
})
PLANAR_GOAL_STEP_TYPES = frozenset({"push", "open_articulation", "close_articulation"})
ARTICULATION_STEP_TYPES = frozenset({"open_articulation", "close_articulation"})
# Closed-loop push completion: forward-only fingertip advances while the
# measured along-push error exceeds the tolerance (never pulls the body back).
PUSH_COMPLETION_STEPS = 3
PUSH_COMPLETION_TOLERANCE_M = 0.002
PUSH_COMPLETION_MAX_ADVANCE_M = 0.03
# Measured closed Panda finger extent around the tool tip in world axes for the
# default tool yaw (x: -0.0196..0.0095 m, y: -0.021..0.021 m).
PANDA_CLOSED_FINGER_EXTENT_M = ((-0.0196, 0.0095), (-0.021, 0.021))


def pusher_center_offset(finger_axis_deg: float, finger_extent=PANDA_CLOSED_FINGER_EXTENT_M) -> list[float]:
    """Closed-finger box centre relative to the tool tip in world axes.

    The measured box is asymmetric about the tip (tool tilt), and it turns with
    the tool yaw (finger axis at 90 degrees for the default yaw).
    """
    (x0, x1), (y0, y1) = finger_extent
    local = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    turn = np.deg2rad(float(finger_axis_deg) - 90.0)
    c, s = np.cos(turn), np.sin(turn)
    return [float(c * local[0] - s * local[1]), float(s * local[0] + c * local[1])]


def push_contact_offset_m(direction_xy, size_xy, yaw_rad: float,
                          finger_extent=PANDA_CLOSED_FINGER_EXTENT_M) -> float:
    """Tool-tip to object-centre distance while pushing along ``direction_xy``.

    The closed-finger support along the push direction plus the pushed body's
    half extent along its local axis nearest that direction (a flat pusher
    turns the contact face flush with the fingers).
    """
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(direction))
    direction = np.asarray([1.0, 0.0]) if norm <= 1e-9 else direction / norm
    (x0, x1), (y0, y1) = finger_extent
    finger_front = max(x0 * direction[0], x1 * direction[0]) + max(y0 * direction[1], y1 * direction[1])
    local_x = np.asarray([np.cos(yaw_rad), np.sin(yaw_rad)])
    along_x = abs(float(direction @ local_x))
    along_y = float(np.sqrt(max(0.0, 1.0 - along_x * along_x)))
    size = np.asarray(size_xy, dtype=np.float64).reshape(2)
    half = size[0] / 2.0 if along_x >= along_y else size[1] / 2.0
    return float(finger_front + half)


def marker_pairs(conditions, routine) -> list[tuple[str, str]]:
    """Body/marker pairs whose marker height follows the settled body.

    Covers every final condition and every routine target (placements,
    pushes, handle slides), so transient targets such as a temporary pad are
    compared at the body's measured rest height too.
    """
    pairs: list[tuple[str, str]] = []
    for item in conditions:
        pair = (str(item[0]), str(item[1]))
        if pair not in pairs:
            pairs.append(pair)
    targets = {pair[1] for pair in pairs}
    for step in routine:
        if step.get("obj") and step.get("target") and step.get("type") != "lift":
            pair = (str(step["obj"]), str(step["target"]))
            if pair not in pairs and pair[1] not in targets:
                pairs.append(pair)
                targets.add(pair[1])
    return pairs


def step_retry(step: dict) -> dict | None:
    """Executed corrective retry for a routine step whose goal was not reached.

    Retries aim at the true target: perturbation offsets and waypoint noise
    apply only to the first attempt.  Held continuations (lift, rotate,
    unreleased transport) have no safe scripted retry.
    """
    kind = step["type"]
    if kind in {"place", "pick_place", "temporary_place", "fit"} or (
        kind == "transport_through_aperture" and step.get("release", True)
    ):
        return {
            "type": "pick_place",
            "obj": step["obj"],
            "target": step["target"],
            "grasp_z": float(step.get("grasp_z", 0.02)),
            "place_z": float(step.get("place_z", 0.0)),
        }
    if kind in {"grasp", "push", "reach"} | ARTICULATION_STEP_TYPES:
        return {
            key: value for key, value in step.items()
            if key not in {"approach_waypoint", "release_waypoint", "place_offset_m", "contact_offset_m"}
        }
    return None


def final_hold_satisfied(object_states, conditions, boundaries: int) -> tuple[bool, dict]:
    """Whether every final predicate holds on each of the trailing boundaries."""
    worst: dict[str, float] = {}
    if len(object_states) < boundaries:
        return False, worst
    ok = True
    for state in object_states[-boundaries:]:
        positions = {
            str(item["name"]): np.asarray(item["position"], dtype=np.float64)
            for item in (state or {}).get("objects", ())
            if isinstance(item, dict) and item.get("position") is not None
        }
        for obj_a, obj_b, tolerance in conditions:
            key = f"{obj_a}~{obj_b}"
            if obj_a not in positions or obj_b not in positions:
                ok = False
                continue
            distance = float(np.linalg.norm(positions[obj_a] - positions[obj_b]))
            worst[key] = max(worst.get(key, 0.0), distance)
            if distance > float(tolerance):
                ok = False
    return ok, worst


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
                if key not in {"_timed_obs", "_actions", "_transition_timing"}
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
    timing = row.get("_transition_timing")
    transitions = []
    for index in range(len(timed) - 1):
        measured = transition_timing_at(timing, index, len(timed) - 1)
        transitions.append({
            "command": {
                "T_w_e": timed[index + 1]["T_w_e"],
                "grip": timed[index + 1]["grip"],
                # Nominal control interval of the declared controller protocol;
                # RLBench's IK/gripper action runs until convergence, so the
                # achieved duration below is measured, never inferred.
                "duration_s": NOMINAL_CONTROL_INTERVAL_S,
            },
            "achieved_duration_s": measured["achieved_duration_s"],
            "physics_substeps": measured["physics_substeps"],
            "before_boundary": index,
            "after_boundary": index + 1,
        })
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
        if k not in {
            "_timed_obs", "_actions", "_transition_timing", "_robot_states", "_object_states", "_task_labels",
        }
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
    simulation_time_s, physics_dt = simulation_clock(env)
    timed_obs: list[dict] = []
    actions_series: list[np.ndarray] = []
    transition_timing: list[dict] = []
    robot_states: list[dict] = []
    object_states: list[dict] = []
    if capture is not None:
        capture.update({
            "_timed_obs": timed_obs,
            "_actions": actions_series,
            "_transition_timing": transition_timing,
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
        started_s = simulation_time_s()
        result = task.step(action)
        achieved_s = simulation_time_s() - started_s
        raw = result[0] if isinstance(result, tuple) else result
        sim_time += achieved_s
        n_actions += 1
        n_obs += 1
        # The executed command and its measured duration stay in a crash
        # prefix even if the following observation cannot be captured.
        actions_series.append(np.asarray(action, dtype=np.float64).copy())
        transition_timing.append({
            "achieved_duration_s": achieved_s,
            "physics_substeps": int(round(achieved_s / physics_dt)),
        })
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

        # Targets are invisible condition markers. Their catalog height is a
        # nominal spawn height, while scaled dynamic objects and handles settle
        # at a measured support height. Align only each marker's vertical
        # coordinate to the settled body it is compared with; x/y stay as
        # planned. Elevated container targets already encode their support.
        for obj_name, target_name in marker_pairs(spec.conditions, routine):
            obj = find_shape(obj_name)
            target = find_shape(target_name)
            if obj is None or target is None:
                continue
            try:
                target_pos = list(target.get_position())
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
    # World angle of the finger closing axis: world y for the default tool yaw.
    finger_axis_deg = 90.0
    rotation_checks = []
    step_outcomes: list[dict] = []
    grip_hold_steps = 5
    for step in routine:
        delta = step.get("grip_timing_delta_intervals")
        if delta is not None:
            grip_hold_steps = max(1, 5 + 5 * int(delta))
            break
    tolerance_m = min((float(item[2]) for item in spec.conditions), default=GENERATION_PROTOCOL.predicate_success_m)

    def hold_steps(position, grip: float, count: int) -> None:
        for _ in range(count):
            try:
                advance(np.concatenate([position, command_quat, [grip]]), grip)
            except Exception as exc:
                if _is_ik_error(exc):
                    break
                raise

    def tip_position() -> np.ndarray:
        return np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)

    def align_grasp_yaw(exec_step: dict, pending_yaw_deg: float) -> None:
        """Before a grasp, turn the tool above the body so the fingers close on
        faces (the nearest face; across the short side of an elongated body),
        never on the corners of a yawed body.

        An elongated body has two equivalent finger orientations; pick the one
        keeping the predicted wrist joint (including the program's pending held
        rotation) farthest from its limits.
        """
        nonlocal command_quat, finger_axis_deg
        grasps = exec_step["type"] in {"grasp", "pick_place", "open_articulation", "close_articulation"} or (
            exec_step["type"] in {"lift", "grasp_rotate", "transport_through_aperture"} and not exec_step.get("held")
        )
        shape = find_shape(str(exec_step.get("obj") or "")) if grasps else None
        if shape is None:
            return
        bounds = np.asarray(shape.get_bounding_box(), dtype=np.float64).reshape(3, 2)
        size = bounds[:2, 1] - bounds[:2, 0]
        delta = grasp_yaw_delta_deg(size, np.rad2deg(float(shape.get_orientation()[2])), finger_axis_deg)
        if abs(delta) < 2.0:
            return
        position = np.asarray(shape.get_position(), dtype=np.float64)
        above = np.asarray([position[0], position[1], 0.90], dtype=np.float64)
        move_ik(above, 1.0)
        if float(size.max()) >= ELONGATED_ASPECT_RATIO * float(size.min()):
            wrist = float(env._scene.robot.arm.get_joint_positions()[6])
            candidates = [value for value in (delta, delta - 180.0, delta + 180.0) if -180.0 <= value <= 180.0]
            # A world-z tool yaw of +d degrees turns the Panda wrist joint by -d.
            delta = min(candidates, key=lambda value: max(
                abs(wrist - np.deg2rad(value)), abs(wrist - np.deg2rad(value + pending_yaw_deg))))
        start_quat = command_quat.copy()
        target_quat = rotate_quaternion(start_quat, delta)
        increments = max(1, int(np.ceil(abs(delta) / 5.0)))
        trace_event("grasp_yaw_align", delta_deg=delta)
        for index in range(1, increments + 1):
            move_ik(above, 1.0, interpolate_quaternion(start_quat, target_quat, index / increments))
        command_quat = target_quat
        finger_axis_deg += delta

    def execute_step(exec_step: dict, pending_yaw_deg: float = 0.0) -> None:
        nonlocal offset, grasped_name, command_quat, finger_axis_deg
        align_grasp_yaw(exec_step, pending_yaw_deg)
        poses = live_poses(spec.objects)
        if exec_step["type"] == "push" and exec_step.get("obj") in poses:
            exec_step = {
                **exec_step,
                "contact_offset_m": measured_push_contact_offset(exec_step, poses),
                "pusher_center_offset_m": pusher_center_offset(finger_axis_deg),
            }
        place_like = exec_step.get("type") in PLACE_STEP_TYPES
        for motion in plan_step(exec_step, poses):
            kind = motion["kind"]
            trace_event("motion", step_type=str(exec_step.get("type")), motion_kind=str(kind),
                        grip=float(motion.get("grip", 1.0)))
            if kind in {"move", "slide"}:
                xyz = np.asarray(motion["xyz"], dtype=np.float64)
                if motion.get("grasp") or motion.get("frame") == "object":
                    tip = tip_position()
                    held = find_shape(str(grasped_name or exec_step.get("obj") or ""))
                    if held is not None:
                        offset = np.asarray(held.get_position(), dtype=np.float64) - tip
                    xyz = xyz - offset
                    if place_like and kind != "slide" and motion.get("frame") == "object" and xyz[2] < 0.86:
                        # Release the carried body just above its rest height
                        # instead of pressing it into the support surface.
                        xyz[2] += PLACE_RELEASE_CLEARANCE_M - float(exec_step.get("place_z", 0.0))
                    off = exec_step.get("place_offset_m") or {}
                    xyz[0] += float(off.get("dx_m", 0.0))
                    xyz[1] += float(off.get("dy_m", 0.0))
                move_ik(xyz, float(motion.get("grip", 1.0)))
                if motion.get("push_final") and exec_step.get("obj") and exec_step.get("target"):
                    complete_push(str(exec_step["obj"]), str(exec_step["target"]), motion["push_direction"], xyz)
            elif kind == "grip":
                if float(motion["grip"]) < 0.5:
                    obj = find_shape(str(motion["grasp_obj"])) if motion.get("grasp_obj") else None
                    actuate(0.0, obj)
                    if obj is not None:
                        grasped_name = str(motion.get("grasp_obj") or "")
                        if not grasp_attachment_confirmed(env._scene.robot.gripper, obj):
                            # Deterministic re-grasp opportunities at the measured
                            # pose before any transport is attempted.
                            for _ in range(2):
                                actuate(0.0, obj)
                                if grasp_attachment_confirmed(env._scene.robot.gripper, obj):
                                    break
                        offset = np.asarray(obj.get_position(), dtype=np.float64) - tip_position()
                        confirmed = grasp_attachment_confirmed(env._scene.robot.gripper, obj)
                        trace_event("post_grasp_check", obj=str(motion.get("grasp_obj")),
                                    distance_m=float(np.linalg.norm(offset)), confirmed=confirmed)
                        if not confirmed:
                            # Never transport an unconfirmed grasp: the step goal
                            # check decides whether an executed retry follows.
                            return
                else:
                    held_name = grasped_name
                    if held_name:
                        # Let the carried body come to rest in contact before opening.
                        hold_steps(tip_position(), 0.0, 8)
                    actuate(1.0, None)
                    tip = tip_position()
                    hold_steps(tip, 1.0, PLACEMENT_SETTLE_STEPS)
                    retreat = np.array([tip[0], tip[1], min(0.92, float(tip[2]) + 0.08)])
                    move_ik(retreat, 1.0)
                    hold_steps(retreat, 1.0, 12)
                    offset = np.zeros(3, dtype=np.float64)
                    grasped_name = None
                    trace_event("post_release", obj=held_name)
            elif kind == "pause":
                hold = float(motion.get("grip", 1.0))
                hold_steps(tip_position(), hold, int(motion["intervals"]) * 5)
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
                tip = tip_position()
                increments = max(1, int(np.ceil(abs(requested_yaw) / 5.0)))
                for index in range(1, increments + 1):
                    move_ik(tip, 0.0, interpolate_quaternion(start_quat, target_quat, index / increments))
                command_quat = target_quat
                finger_axis_deg += requested_yaw
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

    def complete_push(obj_name: str, target_name: str, direction, contact_xyz) -> None:
        """Close the push on the measured object position while still in contact.

        The planned contact offset assumes a face-flush contact; a yawed body
        touched at a corner stops short.  Advance the fingers along the push
        direction by the measured remaining distance (bounded, forward only).
        """
        obj, target = find_shape(obj_name), find_shape(target_name)
        if obj is None or target is None:
            return
        unit = np.asarray(direction, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
        tip_target = np.asarray(contact_xyz, dtype=np.float64).copy()
        for _ in range(PUSH_COMPLETION_STEPS):
            remaining = float(unit @ (np.asarray(target.get_position())[:2] - np.asarray(obj.get_position())[:2]))
            if remaining <= PUSH_COMPLETION_TOLERANCE_M:
                break
            tip_target[:2] += unit * min(remaining, PUSH_COMPLETION_MAX_ADVANCE_M)
            trace_event("push_completion", remaining_m=remaining)
            move_ik(tip_target, 0.0)

    def measured_push_contact_offset(exec_step: dict, poses: dict) -> float:
        obj_name = str(exec_step["obj"])
        aim = exec_step.get("via") if exec_step.get("via") in poses and exec_step.get("via") != exec_step.get("target") else exec_step.get("target")
        direction = np.asarray(poses[aim][:2], dtype=np.float64) - np.asarray(poses[obj_name][:2], dtype=np.float64)
        shape = find_shape(obj_name)
        bounds = np.asarray(shape.get_bounding_box(), dtype=np.float64).reshape(3, 2)
        yaw = float(shape.get_orientation()[2])
        return push_contact_offset_m(direction, bounds[:2, 1] - bounds[:2, 0], yaw)

    def goal_status(exec_step: dict, begin_index: int) -> tuple[bool | None, float | None]:
        kind = exec_step["type"]
        obj_name, target_name = exec_step.get("obj"), exec_step.get("target")
        obj = find_shape(str(obj_name)) if obj_name else None
        target = find_shape(str(target_name)) if target_name else None
        if kind == "pause_hold":
            return True, None
        if kind == "grasp":
            return obj is not None and grasp_attachment_confirmed(env._scene.robot.gripper, obj), None
        if kind == "grasp_rotate":
            attached = obj is not None and grasp_attachment_confirmed(env._scene.robot.gripper, obj)
            return attached and bool(rotation_checks) and bool(rotation_checks[-1]["passed"]), None
        if kind == "reach":
            if target is None:
                return False, None
            goal = np.asarray(target.get_position(), dtype=np.float64)
            goal[2] += float(exec_step.get("reach_z", 0.0))
            reached = [
                float(np.linalg.norm(np.asarray(state["T_w_e"], dtype=np.float64)[:3, 3] - goal))
                for state in robot_states[begin_index:]
            ]
            best = min(reached) if reached else None
            return best is not None and best <= tolerance_m, best
        if obj is None:
            return False, None
        attached = grasp_attachment_confirmed(env._scene.robot.gripper, obj)
        if kind in {"lift", "transport_through_aperture"} and (kind == "lift" or exec_step.get("release") is False):
            if kind == "lift" and target is not None:
                distance = float(np.linalg.norm(np.asarray(obj.get_position()) - np.asarray(target.get_position())))
                return attached and distance <= tolerance_m, distance
            return attached, None
        if target is None:
            return False, None
        delta = np.asarray(obj.get_position(), dtype=np.float64) - np.asarray(target.get_position(), dtype=np.float64)
        if kind in PLANAR_GOAL_STEP_TYPES:
            distance = float(np.linalg.norm(delta[:2]))
        else:
            distance = float(np.linalg.norm(delta))
        return (not attached) and distance <= tolerance_m, distance

    def recoverable(exec_step: dict) -> bool:
        obj_name = exec_step.get("obj")
        if not obj_name:
            return exec_step["type"] == "reach"
        pose = find_shape(str(obj_name))
        if pose is None:
            return False
        position = np.asarray(pose.get_position(), dtype=np.float64)
        return bool(position[2] >= 0.70 and abs(position[0] - 0.25) <= 0.35 and abs(position[1]) <= 0.45)

    for step_index, step in enumerate(routine):
        exec_step = dict(step)
        if exec_step.get("obj") and exec_step["type"] in {
            "grasp", "lift", "pick_place", "place", "temporary_place", "regrasp",
            "open_articulation", "close_articulation",
        }:
            grasped_name = grasped_name or exec_step.get("obj")
        begin_index = len(robot_states)
        pending_yaw_deg = sum(
            float(later.get("yaw_deg", 0.0)) for later in routine[step_index + 1:]
            if later.get("type") == "grasp_rotate" and later.get("held") and later.get("obj") == exec_step.get("obj")
        )
        execute_step(exec_step, pending_yaw_deg)
        achieved, distance = goal_status(exec_step, begin_index)
        retries = 0
        while achieved is False and retries < MAX_STEP_RETRIES and recoverable(exec_step):
            retry = step_retry(exec_step)
            if retry is None:
                break
            retries += 1
            trace_event("step_retry", step_index=step_index, retry=retries, step_type=str(exec_step["type"]))
            if retry.get("obj") and retry["type"] in {"pick_place", "open_articulation", "close_articulation"}:
                grasped_name = retry["obj"]
            execute_step(retry)
            achieved, distance = goal_status(exec_step, begin_index)
        step_outcomes.append({
            "routine_index": step_index,
            "type": exec_step["type"],
            "obj": exec_step.get("obj"),
            "target": exec_step.get("target"),
            "achieved": achieved,
            "distance_m": distance,
            "retries": retries,
            "first_action": begin_index,
            "last_action": len(robot_states) - 1,
        })
        if achieved is False:
            # A failed mandatory step cannot be satisfied later in program
            # order; stop instead of executing dependent steps on a broken state.
            break

    tip = tip_position()
    terminal_grip = terminal_settle_grip(timed_obs)
    hold_steps(tip, terminal_grip, 15)
    rlbench_success, _ = env._scene.task.success()
    rotation_ok = all(item["passed"] for item in rotation_checks)
    history_ok = len(step_outcomes) == len(routine) and all(item["achieved"] is not False for item in step_outcomes)
    final_hold_ok, hold_distances = final_hold_satisfied(object_states, spec.conditions, FINAL_HOLD_BOUNDARIES)
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
    success = bool(rlbench_success) and history_ok and final_hold_ok and rotation_ok and observation_valid
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
        "success": success,
        "result_class": "success" if success else ("valid_failure" if observation_valid else "invalid_observation"),
        "n_actions": n_actions,
        "n_obs": n_obs,
        "timeline_ok": timeline_ok,
        "sim_time_s": sim_time,
        "events": [event["primitive"] for event in spec.events],
        "routine": [step["type"] for step in spec.routine],
        "predicate_distances_m": distances,
        "rotation_checks": rotation_checks,
        "step_outcomes": step_outcomes,
        "success_criteria": {
            "predicate_protocol_id": GENERATION_PROTOCOL.predicate_protocol_id,
            "rlbench_conditions": bool(rlbench_success),
            "mandatory_history": history_ok,
            "final_hold_boundaries": FINAL_HOLD_BOUNDARIES,
            "final_hold": final_hold_ok,
            "final_hold_max_distance_m": hold_distances,
            "rotation": rotation_ok,
            "observation_valid": observation_valid,
        },
        "object_xyz": {
            name: [float(v) for v in find_shape(name).get_position()]
            for name in spec.objects
            if find_shape(name) is not None
        },
        "_timed_obs": timed_obs,
        "_actions": actions_series,
        "_transition_timing": transition_timing,
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
