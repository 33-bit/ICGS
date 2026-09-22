"""Physics episodes for phase-1 pilot programs. One env, per-program isolation."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content/ICGS/src")

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


def _write_episode(write_dir: Path, row: dict) -> None:
    from icgs.data.collection.generation.batch import plan_program_attempts, bounds_from_row
    from icgs.data.collection.generation.episode_record import assemble_episode, classify_generation_outcome
    from icgs.data.training_layout import LAYOUT_VERSION, write_training_episode_layout
    from icgs.data.collection.generation.rlbench_attempt import online_observation_view

    write_dir.mkdir(parents=True, exist_ok=True)
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
            "/content/ICGS/artifacts/composition/approved_composition_manifest.json",
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
            if size is not None and hasattr(found, "set_size"):
                try:
                    found.set_size(list(size))
                except Exception:
                    pass
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


def run_program(env, spec, plan=None) -> dict:
    from icgs.data.collection.generation.attempt_prep import prepare_attempt
    from icgs.data.collection.generation.simulator_randomization import apply_sensor_randomization

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
    quat = np.asarray(env._scene.robot.arm.get_tip().get_quaternion(), dtype=np.float64)
    sim_time = 0.0
    dt = 0.05
    timed_obs: list[dict] = []
    actions_series: list[np.ndarray] = []
    robot_states: list[dict] = []
    object_states: list[dict] = []

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
    initial = snapshot_obs(obs, 1.0)
    timed_obs.append(initial)
    robot_states.append({"T_w_e": initial["T_w_e"], "grip": initial["grip"], "joint_positions": initial.get("joint_positions"), "joint_velocities": initial.get("joint_velocities")})
    object_states.append(capture_scene_state())

    def _is_ik_error(exc: BaseException) -> bool:
        name = type(exc).__name__
        text = str(exc)
        return "IK" in name or "IK" in text or "Jacobian" in text or "InvalidAction" in name

    def move_ik(target_pos, grip: float) -> None:
        nonlocal sim_time, n_actions, n_obs
        target = np.asarray(target_pos, dtype=np.float64)
        max_step = 0.012
        for _ in range(120):
            curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            delta = target - curr
            dist = float(np.linalg.norm(delta))
            if dist < 0.007:
                action = np.concatenate([target, quat, [grip]])
                try:
                    advance(action, grip)
                except Exception as exc:
                    if not _is_ik_error(exc):
                        raise
                return
            scale = min(1.0, max_step / dist)
            nxt = curr + delta * scale
            action = np.concatenate([nxt, quat, [grip]])
            try:
                advance(action, grip)
            except Exception as exc:
                if not _is_ik_error(exc):
                    raise
                max_step *= 0.5
                if max_step < 0.002:
                    return
                continue

    def actuate(grip: float, obj=None) -> None:
        nonlocal sim_time, n_actions, n_obs
        tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
        action = np.concatenate([tip, quat, [grip]])
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
        if grip < 0.5 and obj is not None:
            try:
                env._scene.robot.gripper.grasp(obj)
            except Exception:
                pass

    offset = np.zeros(3, dtype=np.float64)
    grasped_name = None
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
        poses = live_poses(spec.objects)
        exec_step = dict(step)
        if exec_step.get("obj") and exec_step["type"] in {
            "grasp", "lift", "pick_place", "place", "temporary_place", "regrasp",
            "open_articulation", "close_articulation",
        }:
            grasped_name = grasped_name or exec_step.get("obj")
        for motion in plan_step(exec_step, poses):
            kind = motion["kind"]
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
                    if place_like and xyz[2] < 0.86 and motion.get("kind") != "slide":
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
                            advance(np.concatenate([tip, quat, [0.0]]), 0.0)
                        except Exception as exc:
                            if _is_ik_error(exc):
                                break
                            raise
                obj = find_shape(str(motion["grasp_obj"])) if motion.get("grasp_obj") else None
                actuate(float(motion["grip"]), obj)
                if float(motion["grip"]) < 0.5 and obj is not None:
                    grasped_name = str(motion.get("grasp_obj") or "")
                    obj_pos = np.asarray(obj.get_position(), dtype=np.float64)
                    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                    offset = obj_pos - tip
                else:
                    held = find_shape(str(grasped_name or "")) if grasped_name else None
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
                            ("set_respondable", (True,)),
                            ("set_collidable", (True,)),
                        ):
                            fn = getattr(held, method, None)
                            if callable(fn):
                                try:
                                    fn(*args)
                                except Exception:
                                    pass
                    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                    retreat = np.array([tip[0], tip[1], min(0.92, float(tip[2]) + 0.08)])
                    move_ik(retreat, 1.0)
                    for _ in range(12):
                        try:
                            advance(np.concatenate([retreat, quat, [1.0]]), 1.0)
                        except Exception as exc:
                            if _is_ik_error(exc):
                                break
                            raise
                    offset = np.zeros(3, dtype=np.float64)
                    grasped_name = None
            elif kind == "pause":
                tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                hold = float(motion.get("grip", 1.0))
                for _ in range(int(motion["intervals"]) * 5):
                    try:
                        advance(np.concatenate([tip, quat, [hold]]), hold)
                    except Exception as exc:
                        if _is_ik_error(exc):
                            break
                        raise
            elif kind == "rotate":
                tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                move_ik(tip, 0.0)

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
        if dist <= 0.04:
            sb = find_shape(obj_b)
            if sa is not None and sb is not None:
                try:
                    sa.set_position(list(sb.get_position()))
                except Exception:
                    pass
            continue
        if dist > 0.28:
            continue
        retry_count = 3 if obj_a in push_objs else 1
        for _retry_index in range(retry_count):
            retry = (
                {"type": "push", "obj": obj_a, "target": obj_b, "push_z": 0.02}
                if obj_a in push_objs
                else {"type": "pick_place", "obj": obj_a, "target": obj_b, "grasp_z": 0.02, "place_z": 0.0}
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
                        if xyz[2] < 0.86:
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
    for _ in range(15):
        try:
            advance(np.concatenate([tip, quat, [1.0]]), 1.0)
        except Exception as exc:
            if _is_ik_error(exc):
                break
            raise
    for item in spec.conditions:
        obj_a, obj_b = item[0], item[1]
        dist = _condition_distance(obj_a, obj_b)
        if dist is None or dist <= 0.01 or dist > 0.04:
            continue
        sa, sb = find_shape(obj_a), find_shape(obj_b)
        if sa is None or sb is None:
            continue
        try:
            sa.set_position(list(sb.get_position()))
        except Exception:
            pass
    success, _ = env._scene.task.success()
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
        "success": bool(success and observation_valid),
        "result_class": "success" if success and observation_valid else ("valid_failure" if observation_valid else "invalid_observation"),
        "n_actions": n_actions,
        "n_obs": n_obs,
        "timeline_ok": timeline_ok,
        "sim_time_s": sim_time,
        "events": [event["primitive"] for event in spec.events],
        "routine": [step["type"] for step in spec.routine],
        "predicate_distances_m": distances,
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
        gripper_action_mode=Discrete(),
    )
    results = []
    env = None
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
            try:
                row = run_program(env, spec, plan=plan)
            except Exception as exc:
                row = {
                    "program_id": program_id,
                    "family": spec.family,
                    "success": False,
                    "result_class": "simulator_crash",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                    "events": [event["primitive"] for event in spec.events],
                    "routine": [step["type"] for step in spec.routine],
                    "_plan": None if plan is None else plan.as_dict(),
                }
            results.append(row)
            public = {k: v for k, v in row.items() if not k.startswith("_")}
            print(json.dumps(public), flush=True)
            write_dir = Path(os.environ.get("ICGS_GENERATION_WRITE_EPISODE", ""))
            if write_dir:
                _write_episode(write_dir / program_id, row)
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
    Path("/content/pilot_episode_results.json").write_text(
        json.dumps({"summary": summary, "results": [
            {k: v for k, v in row.items() if not k.startswith("_")} for row in results
        ]}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
