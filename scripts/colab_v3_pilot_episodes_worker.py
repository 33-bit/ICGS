"""Physics episodes for phase-1 pilot programs. One env, per-program isolation."""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content/ICGS/src")

from icgs.data.collection.v3.compiler import compile_v3_catalog
from icgs.data.collection.v3.expert import plan_step
from icgs.data.collection.v3.protocol import V3_PROTOCOL
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


def _write_v3_episode(write_dir: Path, row: dict) -> None:
    from icgs.data.collection.v3.batch import plan_program_attempts, bounds_from_row
    from icgs.data.collection.v3.episode_record import assemble_episode_v2, classify_generation_outcome

    write_dir.mkdir(parents=True, exist_ok=True)
    timed = row.get("_timed_obs") or []
    if len(timed) < 2:
        sidecar = {
            "result_class": row.get("result_class", "valid_failure"),
            "program_id": row["program_id"],
            "error": "not enough observations",
        }
        (write_dir / "quarantine.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        return
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
    binding = {
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
    }
    from icgs.data.collection.v3.batch import attempt_from_dict

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
    result_class = classify_generation_outcome(
        simulator_crash=row.get("result_class") == "simulator_crash",
        predicates_ok=bool(row.get("success")),
    )
    record = assemble_episode_v2(
        plan=plan,
        binding=binding,
        observations=timed,
        transitions=transitions,
        result_class=result_class,
        intervention=row.get("intervention"),
    )
    def _enc(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _enc(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_enc(v) for v in value]
        return value
    (write_dir / "episode.json").write_text(json.dumps(_enc(record), indent=2) + "\n")
    (write_dir / "physics.json").write_text(json.dumps({k: v for k, v in row.items() if k != "_timed_obs"}, indent=2) + "\n")


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
    from icgs.data.collection.v3.attempt_prep import prepare_attempt

    task_cls = load_task_class(spec)
    task = env.get_task(task_cls)
    desc, obs = task.reset()
    prepared = None
    routine = list(spec.routine)
    if plan is not None:
        prepared = prepare_attempt(plan, spec.objects, spec.routine)
        apply_prepared_layout(prepared["objects"])
        routine = list(prepared["routine"])
    n_obs = 1
    n_actions = 0
    quat = np.asarray(env._scene.robot.arm.get_tip().get_quaternion(), dtype=np.float64)
    sim_time = 0.0
    dt = 0.05
    timed_obs: list[dict] = []

    def snapshot_obs(grip: float) -> dict:
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
        cloud = np.repeat(pos.reshape(1, 3), 8, axis=0).astype(np.float32)
        return {
            "points": np.asarray(cloud, dtype=np.float32),
            "point_valid": np.ones((cloud.shape[0],), dtype=bool),
            "T_w_e": T,
            "grip": 0 if grip < 0.5 else 1,
        }

    timed_obs.append(snapshot_obs(1.0))

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
                    task.step(action)
                    sim_time += dt
                    n_actions += 1
                    n_obs += 1
                    timed_obs.append(snapshot_obs(grip))
                except Exception as exc:
                    if not _is_ik_error(exc):
                        raise
                return
            scale = min(1.0, max_step / dist)
            nxt = curr + delta * scale
            action = np.concatenate([nxt, quat, [grip]])
            try:
                task.step(action)
            except Exception as exc:
                if not _is_ik_error(exc):
                    raise
                max_step *= 0.5
                if max_step < 0.002:
                    return
                continue
            sim_time += dt
            n_actions += 1
            n_obs += 1
            timed_obs.append(snapshot_obs(grip))

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
                task.step(action)
            except Exception as exc:
                if _is_ik_error(exc):
                    break
                raise
            sim_time += dt
            n_actions += 1
            n_obs += 1
            timed_obs.append(snapshot_obs(grip))
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
                        xyz[2] -= float(V3_PROTOCOL.place_ik_z_shortfall_m)
                    off = exec_step.get("place_offset_m") or {}
                    xyz[0] += float(off.get("dx_m", 0.0))
                    xyz[1] += float(off.get("dy_m", 0.0))
                move_ik(xyz, float(motion.get("grip", 1.0)))
            elif kind == "grip":
                if float(motion["grip"]) > 0.5 and grasped_name:
                    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                    for _ in range(8):
                        try:
                            task.step(np.concatenate([tip, quat, [0.0]]))
                        except Exception as exc:
                            if _is_ik_error(exc):
                                break
                            raise
                        n_actions += 1
                        n_obs += 1
                        timed_obs.append(snapshot_obs(0.0))
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
                            task.step(np.concatenate([retreat, quat, [1.0]]))
                        except Exception as exc:
                            if _is_ik_error(exc):
                                break
                            raise
                        n_actions += 1
                        n_obs += 1
                        timed_obs.append(snapshot_obs(1.0))
                    offset = np.zeros(3, dtype=np.float64)
                    grasped_name = None
            elif kind == "pause":
                tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                hold = float(motion.get("grip", 1.0))
                for _ in range(int(motion["intervals"]) * 5):
                    try:
                        task.step(np.concatenate([tip, quat, [hold]]))
                    except Exception as exc:
                        if _is_ik_error(exc):
                            break
                        raise
                    n_actions += 1
                    n_obs += 1
                    timed_obs.append(snapshot_obs(hold))
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
        if obj_a in push_objs:
            retry = {"type": "push", "obj": obj_a, "target": obj_b, "push_z": 0.02}
        else:
            retry = {"type": "pick_place", "obj": obj_a, "target": obj_b, "grasp_z": 0.02, "place_z": 0.0}
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
                        xyz[2] -= float(V3_PROTOCOL.place_ik_z_shortfall_m)
                move_ik(xyz, float(motion.get("grip", 1.0)))
            elif kind == "grip":
                obj = find_shape(obj_a) if float(motion.get("grip", 1)) < 0.5 else None
                actuate(float(motion["grip"]), obj)

    tip = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
    for _ in range(15):
        try:
            task.step(np.concatenate([tip, quat, [1.0]]))
        except Exception as exc:
            if _is_ik_error(exc):
                break
            raise
        n_actions += 1
        n_obs += 1
        timed_obs.append(snapshot_obs(1.0))
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
        "success": bool(success),
        "result_class": "success" if success else "valid_failure",
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
        "_plan": None if plan is None else plan.as_dict(),
        "episode_kind": None if plan is None else plan.episode_kind,
        "intervention": intervention,
    }


def main() -> int:
    from icgs.data.collection.v3.steps import V3_PROGRAMS

    wanted = sys.argv[1:] or list(V3_PROGRAMS)
    compiled = compile_v3_catalog()
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(False)
    obs_config.wrist_camera.point_cloud = True
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
            plan_path = os.environ.get("ICGS_V3_ATTEMPT_JSON")
            if plan_path:
                from icgs.data.collection.v3.batch import attempt_from_dict
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
                }
            results.append(row)
            public = {k: v for k, v in row.items() if k != "_timed_obs"}
            print(json.dumps(public), flush=True)
            write_dir = Path(os.environ.get("ICGS_V3_WRITE_EPISODE", ""))
            if write_dir:
                _write_v3_episode(write_dir / program_id, row)
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
    Path("/content/v3_pilot_episode_results.json").write_text(
        json.dumps({"summary": summary, "results": [
            {k: v for k, v in row.items() if k != "_timed_obs"} for row in results
        ]}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
