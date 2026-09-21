"""Production Data Generation Engine for ICGS G2 Procedural Tasks.

Executes procedural tasks (T06, T08, T09, T11, T13, T14) in live PyRep/RLBench,
capturing full 3D point cloud transitions, 7-DOF joint telemetry (positions and velocities),
and companion 10-FPS front-camera RGB video, packaging into verified ICGS archives,
and continuously committing to Hugging Face (33bit/icgs) and Google Drive.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("icgs.g2.generator")


def parse_args():
    parser = argparse.ArgumentParser(description="ICGS G2 Dataset Generator")
    parser.add_argument("--output-dir", type=str, default="/content/dataset_g2", help="Local staging root")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/ICGS-data-20260916", help="Google Drive destination")
    parser.add_argument("--hf-repo", type=str, default="33bit/icgs", help="Hugging Face repo ID")
    parser.add_argument("--hf-token-file", type=str, default="/content/.icgs_hf_token", help="Path to HF token file")
    parser.add_argument("--hf-subfolder", type=str, default="primary", help="Subfolder in HF repo")
    parser.add_argument("--tasks", nargs="+", default=["T06", "T08", "T09", "T11", "T13", "T14"], help="Task IDs to generate")
    parser.add_argument("--episodes-per-task", type=int, default=2, help="Number of episodes per task")
    parser.add_argument("--start-idx", type=int, default=100, help="Starting index for episode naming")
    parser.add_argument(
        "--approved-manifest",
        type=str,
        default="/content/ICGS/artifacts/composition/approved_composition_manifest.json",
        help="Owner-approved composition manifest; required for primary generation",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="scripted_waypoint_v1",
        help="Execution mode declared in the approved manifest",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["v1", "v3"],
        default="v1",
        help="v1 keeps primary_v2 generation; v3 uses the phase-1 compiler and manifest",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run a local strict simulator smoke without Drive/HF publication or dataset certification",
    )
    parser.add_argument(
        "--allow-planned",
        action="store_true",
        help="Explicitly authorize generation for all planned suite compositions after strict smoke verification",
    )
    parser.add_argument(
        "--use-manifest-targets",
        action="store_true",
        help="Use exact target counts from approved manifest generation_targets (200 train, 63/62 dev, 100 test)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test", "all"],
        default=None,
        help="Filter tasks to generate by split (train=T01-T20, dev=V01-V04, test=P1-P4,G1-G4,R1-R4)",
    )
    parser.add_argument(
        "--commit-batch-size",
        type=int,
        default=10,
        help="Number of successful episodes to accumulate per atomic HF commit batch (default: 10)",
    )
    return parser.parse_args()


TASK_SPECS = {
    "T06": {
        "class_name": "T06PlaceIntoDrawer",
        "module": "t06_place_into_drawer",
        "description": "place object A into a drawer target",
        "routine": [
            {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
        ],
    },
    "T08": {
        "class_name": "T08PackTwoObjects",
        "module": "t08_pack_two_objects",
        "description": "place B then A into distinct tray slots",
        "routine": [
            {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
            {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
        ],
    },
    "T09": {
        "class_name": "T09PlaceIntoHolder",
        "module": "t09_place_into_holder",
        "description": "grasp A and place it into a cylindrical holder",
        "routine": [
            {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.025, "place_z": 0.035},
        ],
    },
    "T11": {
        "class_name": "T11RotateAndPlace",
        "module": "t11_rotate_and_place",
        "description": "grasp A, rotate it, and place it on an oriented pad",
        "routine": [
            {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": -0.005, "place_z": 0.02},
        ],
    },
    "T13": {
        "class_name": "T13ParkBlockerRetrieve",
        "module": "t13_park_blocker_retrieve",
        "description": "park blocker before retrieving object A",
        "routine": [
            {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.03},
            {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
        ],
    },
    "T14": {
        "class_name": "T14ParkAndRestore",
        "module": "t14_park_and_restore",
        "description": "park blocker and restore it after a retrieval motion",
        "routine": [
            {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.035},
            {"type": "touch_retreat", "obj": "object_a", "touch_z": 0.035},
            {"type": "pick_place", "obj": "blocker", "target": "restore_target", "grasp_z": 0.025, "place_z": 0.035},
        ],
    },
}


def load_task_specs(manifest_path: str | Path | None = None, protocol: str = "v1") -> dict[str, dict[str, Any]]:
    try:
        mpath = Path(manifest_path) if manifest_path else None
        if protocol == "v3":
            from icgs.data.collection.v3.compiler import compile_v3_catalog
            compiled = compile_v3_catalog(mpath)
            from icgs.data.collection.v3.report import generation_parity_gate
            gate = generation_parity_gate(compiled)
            if not gate["ok"]:
                raise RuntimeError(
                    "v3 parity gate blocked training programs before generation: "
                    + ", ".join(gate["blocked"])
                )
        else:
            from icgs.data.collection.primitive_compiler import compile_catalog
            if not (mpath and mpath.is_file()):
                return TASK_SPECS
            compiled = compile_catalog(mpath)
        specs = {}
        for pid, spec in compiled.items():
            specs[pid] = {
                "class_name": spec.class_name,
                "module": spec.module,
                "description": spec.description,
                "routine": spec.routine,
                "events": getattr(spec, "events", ()),
                "objects": getattr(spec, "objects", {}),
            }
        return specs
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Using default TASK_SPECS, dynamic compile not available: {e}")
    return TASK_SPECS


def _find_shape(name: str):
    from pyrep.objects.shape import Shape
    for candidate in [name, f"{name}0", f"{name}#0"]:
        try:
            return Shape(candidate)
        except Exception:
            pass
    from pyrep.objects.dummy import Dummy
    for candidate in [name, f"{name}0"]:
        try:
            return Dummy(candidate)
        except Exception:
            pass
    return None


def load_approved_generation_manifest(
    path: str | Path,
    task_ids: list[str],
    *,
    allow_planned: bool = False,
    protocol: str = "v1",
) -> tuple[dict[str, Any], str]:
    """Load the owner-approved manifest and gate task execution before RLBench launch."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"approved composition manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if protocol == "v3":
        if data.get("manifest_version") != 3 or data.get("composition_protocol_id") != "icgs-composition-primary-v2":
            raise ValueError("v3 composition manifest has an unsupported identity")
    elif data.get("manifest_version") != 1 or data.get("protocol_id") != "icgs-composition-primary-v1":
        raise ValueError("approved composition manifest has an unsupported identity")
    rows = {row["program_id"]: row for row in data.get("catalog", [])}
    missing = sorted(set(task_ids) - set(rows))
    if missing:
        raise ValueError(f"requested tasks are absent from approved manifest: {missing}")
    if protocol == "v3" and allow_planned:
        raise RuntimeError(
            "v3 generation cannot use --allow-planned; authorize the composition manifest first"
        )
    blocked = [task_id for task_id in task_ids if rows[task_id].get("execution_status") == "planned"]
    if blocked and not allow_planned:
        raise RuntimeError(
            "approved manifest blocks generation for planned programs: "
            + ", ".join(blocked)
            + "; obtain strict execution certification first"
        )
    for task_id in task_ids:
        row = rows[task_id]
        if not row.get("seed_demos") or not row.get("execution_modes"):
            raise ValueError(f"approved binding for {task_id} has no seed demos or execution modes")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return data, digest


def collect_single_episode(
    task,
    env,
    spec: dict[str, Any],
    ep_id: str,
    task_id: str,
    *,
    approved_binding: dict[str, Any],
    approved_manifest_digest: str,
    seed_id: str,
    execution_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    from icgs.contracts.method import TimedObservation, TimedCommand, ExecutedTransition
    from icgs.contracts.records import Observation
    from icgs.environments.rlbench.controller import pose_to_matrix, filter_and_downsample_points

    desc, obs = task.reset()
    logger.info(f"Task reset: {desc}")
    prepared = spec.get("_prepared")
    if prepared and prepared.get("objects"):
        from pyrep.const import PrimitiveShape
        from pyrep.objects.shape import Shape
        for name, item in prepared["objects"].items():
            pos = item["pos"] if isinstance(item, dict) else item[0]
            try:
                Shape(name).set_position(list(pos))
            except Exception:
                if isinstance(item, dict) and item.get("declared"):
                    try:
                        created = Shape.create(
                            PrimitiveShape.CUBOID,
                            size=item.get("size") or [0.04, 0.04, 0.04],
                            mass=0.05,
                            static=False,
                            respondable=True,
                            position=list(pos),
                            color=item.get("color") or [0.1, 0.1, 0.1],
                        )
                        created.set_name(name)
                    except Exception:
                        pass
        spec = dict(spec)
        spec["routine"] = prepared.get("routine", spec.get("routine"))

    quat = np.asarray(env._scene.robot.arm.get_tip().get_quaternion(), dtype=np.float64)

    raw_obs_list = [obs]
    commands_list = []
    actions_list = []
    declared_object_names = []
    for routine_step in spec.get("routine", []):
        for key in ("obj", "target", "aperture_wp"):
            value = routine_step.get(key)
            if value and value not in declared_object_names:
                declared_object_names.append(value)
    scene_state_series = []
    collision_events = []

    def capture_scene_state(boundary: int) -> None:
        state = {"boundary": boundary, "objects": []}
        shapes = {}
        for name in declared_object_names:
            shape = _find_shape(name)
            if shape is None:
                continue
            shapes[name] = shape
            try:
                position = np.asarray(shape.get_position(), dtype=np.float64)
                orientation = np.asarray(shape.get_quaternion(), dtype=np.float64)
                velocity = shape.get_velocity()
                linear = np.asarray(velocity[0] if isinstance(velocity, tuple) else [0, 0, 0], dtype=np.float64)
                angular = np.asarray(velocity[1] if isinstance(velocity, tuple) else [0, 0, 0], dtype=np.float64)
                state["objects"].append({"name": name, "position": position, "orientation_xyzw": orientation, "linear_velocity": linear, "angular_velocity": angular, "valid": True})
            except Exception:
                state["objects"].append({"name": name, "valid": False})
        for left_name, left in shapes.items():
            for right_name, right in shapes.items():
                if left_name >= right_name:
                    continue
                try:
                    colliding = bool(left.check_collision(right))
                except Exception:
                    colliding = False
                collision_events.append({"boundary": boundary, "a": left_name, "b": right_name, "collision": colliding, "contact": colliding})
        scene_state_series.append(state)

    # Keep a live, in-memory snapshot so failed attempts retain valid simulator
    # observations collected before the failure.  The main loop serializes this
    # snapshot into quarantine; it is never added to the success manifest.
    def snapshot_attempt() -> None:
        try:
            task._icgs_attempt_state = {
                "observations": list(raw_obs_list),
                "actions": [np.asarray(a).copy() for a in actions_list],
                "sim_time": float(sim_time),
                "observation_count": len(raw_obs_list),
                "action_count": len(actions_list),
            }
        except Exception:
            pass

    snapshot_attempt()
    capture_scene_state(0)

    current_obs = obs
    t_start = time.monotonic()
    sim_time = 0.0
    dt_physics = 0.05  # PyRep default 50ms timestep

    offset = np.zeros(3, dtype=np.float64)

    def move_ik(target_pos: np.ndarray, grip: float) -> None:
        nonlocal sim_time, current_obs
        curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
        dist = float(np.linalg.norm(target_pos - curr))
        substeps = max(4, int(dist / 0.008))
        for i in range(1, substeps + 1):
            alpha = i / float(substeps)
            pos = (1.0 - alpha) * curr + alpha * target_pos
            action = np.concatenate([pos, quat, [grip]])
            next_obs, reward, terminate = task.step(action)
            sim_time += dt_physics
            actions_list.append(action)
            raw_obs_list.append(next_obs)
            current_obs = next_obs
            snapshot_attempt()
            capture_scene_state(len(raw_obs_list) - 1)

    def actuate_gripper(grip: float, grasp_obj=None) -> None:
        nonlocal sim_time, current_obs
        tip_pos = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
        action = np.concatenate([tip_pos, quat, [grip]])
        if grip > 0.5:
            try:
                env._scene.robot.gripper.release()
            except Exception:
                pass
        for _ in range(5):
            next_obs, reward, terminate = task.step(action)
            sim_time += dt_physics
            actions_list.append(action)
            raw_obs_list.append(next_obs)
            current_obs = next_obs
            snapshot_attempt()
            capture_scene_state(len(raw_obs_list) - 1)
        if grip < 0.5 and grasp_obj is not None:
            try:
                env._scene.robot.gripper.grasp(grasp_obj)
            except Exception:
                pass

    for step in spec["routine"]:
        stype = step["type"]
        approach_z = 0.90

        if stype == "pick_place":
            obj_shape = _find_shape(step["obj"])
            target_shape = _find_shape(step["target"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
            gz = step.get("grasp_z", 0.02)
            pz = step.get("place_z", 0.0)
            is_mechanism = "handle" in step["obj"]

            if is_mechanism:
                # Mechanism (drawer/gate): slide horizontally without lifting
                move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
                move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2] + gz]), 1.0)
                actuate_gripper(0.0, grasp_obj=obj_shape)
                offset = np.asarray(obj_shape.get_position(), dtype=np.float64) - np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], target_pos[2] - offset[2]]), 0.0)
                actuate_gripper(1.0)
                tip_curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                move_ik(np.array([tip_curr[0], tip_curr[1], approach_z]), 1.0)
            else:
                # Free rigid body: approach, descend to object center, grasp, lift, carry, lower to target_pos, release
                move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
                move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2]]), 1.0)
                actuate_gripper(0.0, grasp_obj=obj_shape)
                offset = np.asarray(obj_shape.get_position(), dtype=np.float64) - np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 0.0)
                move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], approach_z]), 0.0)
                move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], target_pos[2] - offset[2]]), 0.0)
                actuate_gripper(1.0)
                tip_curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
                move_ik(np.array([tip_curr[0], tip_curr[1], approach_z]), 1.0)

        elif stype == "lift":
            obj_shape = _find_shape(step["obj"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
            move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2]]), 1.0)
            actuate_gripper(0.0, grasp_obj=obj_shape)
            offset = np.asarray(obj_shape.get_position(), dtype=np.float64) - np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            if "target" in step:
                target_shape = _find_shape(step["target"])
                target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
                move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], target_pos[2] - offset[2]]), 0.0)
            else:
                lift_z = float(step.get("lift_z", obj_pos[2] + 0.12))
                move_ik(np.array([obj_pos[0], obj_pos[1], lift_z - offset[2]]), 0.0)
            actuate_gripper(0.0, grasp_obj=obj_shape)

        elif stype == "reach":
            target_shape = _find_shape(step["target"])
            target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
            reach_z = float(target_pos[2] + step.get("reach_z", 0.02))
            move_ik(np.array([target_pos[0], target_pos[1], approach_z]), 1.0)
            move_ik(np.array([target_pos[0], target_pos[1], reach_z]), 1.0)
            move_ik(np.array([target_pos[0], target_pos[1], approach_z]), 1.0)

        elif stype == "push":
            obj_shape = _find_shape(step["obj"])
            target_shape = _find_shape(step["target"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
            pz = step.get("push_z", 0.02)
            push_z = float(obj_pos[2] + pz)
            direction = np.asarray(target_pos[:2] - obj_pos[:2], dtype=np.float64)
            norm = np.linalg.norm(direction)
            unit_dir = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
            pre_push = np.asarray(obj_pos[:2]) - unit_dir * 0.06
            move_ik(np.array([pre_push[0], pre_push[1], approach_z]), 0.0)
            move_ik(np.array([pre_push[0], pre_push[1], push_z]), 0.0)
            move_ik(np.array([target_pos[0], target_pos[1], push_z]), 0.0)
            move_ik(np.array([target_pos[0], target_pos[1], approach_z]), 0.0)

        elif stype == "grasp_rotate":
            from scipy.spatial.transform import Rotation as R
            obj_shape = _find_shape(step["obj"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            lift_z = float(obj_pos[2] + step.get("lift_z", 0.12))
            move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
            move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2]]), 1.0)
            actuate_gripper(0.0, grasp_obj=obj_shape)
            offset = np.asarray(obj_shape.get_position(), dtype=np.float64) - np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            move_ik(np.array([obj_pos[0], obj_pos[1], lift_z]), 0.0)
            yaw_deg = float(step.get("yaw_deg", 90.0))
            r_orig = R.from_quat(quat)
            r_yaw = R.from_euler("z", yaw_deg, degrees=True)
            rot_quat = (r_yaw * r_orig).as_quat()
            tip_pos = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            for alpha in np.linspace(0.0, 1.0, 10):
                q_interp = (1.0 - alpha) * quat + alpha * rot_quat
                q_interp = q_interp / np.linalg.norm(q_interp)
                action = np.concatenate([tip_pos, q_interp, [0.0]])
                next_obs, reward, terminate = task.step(action)
                sim_time += dt_physics
                actions_list.append(action)
                raw_obs_list.append(next_obs)
                current_obs = next_obs
                snapshot_attempt()
                capture_scene_state(len(raw_obs_list) - 1)
            quat = rot_quat

        elif stype == "place":
            target_shape = _find_shape(step["target"])
            target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
            move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], approach_z]), 0.0)
            move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], target_pos[2] - offset[2]]), 0.0)
            actuate_gripper(1.0)
            tip_curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            move_ik(np.array([tip_curr[0], tip_curr[1], approach_z]), 1.0)

        elif stype == "transport_through_aperture":
            obj_shape = _find_shape(step["obj"])
            wp_shape = _find_shape(step["aperture_wp"])
            target_shape = _find_shape(step["target"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            wp_pos = np.asarray(wp_shape.get_position(), dtype=np.float64)
            target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
            move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
            move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2]]), 1.0)
            actuate_gripper(0.0, grasp_obj=obj_shape)
            offset = np.asarray(obj_shape.get_position(), dtype=np.float64) - np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 0.0)
            move_ik(np.array([wp_pos[0], wp_pos[1], approach_z]), 0.0)
            move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], approach_z]), 0.0)
            move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], target_pos[2] - offset[2]]), 0.0)
            actuate_gripper(1.0)
            tip_curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            move_ik(np.array([tip_curr[0], tip_curr[1], approach_z]), 1.0)

        elif stype == "touch_retreat":
            obj_shape = _find_shape(step["obj"])
            pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            tz = step.get("touch_z", 0.025)
            touch_z = float(pos[2] + tz)
            move_ik(np.array([pos[0], pos[1], approach_z]), 1.0)
            move_ik(np.array([pos[0], pos[1], touch_z]), 1.0)
            move_ik(np.array([pos[0], pos[1], approach_z]), 1.0)

        elif stype == "grasp":
            obj_shape = _find_shape(step["obj"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
            move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2]]), 1.0)
            actuate_gripper(0.0, grasp_obj=obj_shape)

        elif stype in ("open_articulation", "close_articulation"):
            obj_shape = _find_shape(step["obj"])
            target_shape = _find_shape(step["target"])
            obj_pos = np.asarray(obj_shape.get_position(), dtype=np.float64)
            target_pos = np.asarray(target_shape.get_position(), dtype=np.float64)
            gz = step.get("grasp_z", 0.02)
            move_ik(np.array([obj_pos[0], obj_pos[1], approach_z]), 1.0)
            move_ik(np.array([obj_pos[0], obj_pos[1], obj_pos[2] + gz]), 1.0)
            actuate_gripper(0.0, grasp_obj=obj_shape)
            offset = np.asarray(obj_shape.get_position(), dtype=np.float64) - np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            move_ik(np.array([target_pos[0] - offset[0], target_pos[1] - offset[1], target_pos[2] - offset[2]]), 0.0)
            actuate_gripper(1.0)
            tip_curr = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            move_ik(np.array([tip_curr[0], tip_curr[1], approach_z]), 1.0)

        elif stype == "pause_hold":
            tip_pos = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
            intervals = int(step.get("intervals", 2))
            hold_grip = 0.0 if any(s.get("type") in {"lift", "grasp"} for s in spec["routine"]) else 1.0
            for _ in range(max(1, intervals) * 5):
                action = np.concatenate([tip_pos, quat, [hold_grip]])
                next_obs, reward, terminate = task.step(action)
                sim_time += dt_physics
                actions_list.append(action)
                raw_obs_list.append(next_obs)
                current_obs = next_obs
                snapshot_attempt()
                capture_scene_state(len(raw_obs_list) - 1)

    # Settle physics while holding gripper pose
    tip_pos = np.asarray(env._scene.robot.arm.get_tip().get_position(), dtype=np.float64)
    hold_grip = 0.0 if any(s["type"] == "lift" for s in spec["routine"]) else 1.0
    for _ in range(5):
        action = np.concatenate([tip_pos, quat, [hold_grip]])
        task.step(action)

    success, _ = env._scene.task.success()
    cond_details = []
    for cond in getattr(task._task, "_success_conditions", []):
        try:
            p1 = cond.obj.get_position()
            p2 = cond.target.get_position()
            d = float(np.linalg.norm(np.array(p1) - np.array(p2)))
            cond_details.append(f"{cond.obj.get_name()}->{cond.target.get_name()}: dist={d:.4f} (tol={cond.tolerance})")
        except Exception:
            pass
    logger.info(f"Episode execution complete: steps={len(actions_list)}, success={success}, conds=[{', '.join(cond_details)}]")
    if not success:
        raise RuntimeError(f"Episode did not meet task success criteria: {', '.join(cond_details)}")

    # Downsample transitions to 10 Hz intervals (2 physics steps per interval)
    stride = 2
    interval_obs_indices = list(range(0, len(raw_obs_list), stride))
    if interval_obs_indices[-1] != len(raw_obs_list) - 1:
        interval_obs_indices.append(len(raw_obs_list) - 1)

    online_observations = []
    transitions = []
    joint_positions_series = []
    joint_velocities_series = []
    front_rgb_series = []
    wrist_depth_series = []
    wrist_mask_series = []
    front_mask_series = []
    scene_state_by_boundary = []

    for boundary_idx, obs_idx in enumerate(interval_obs_indices):
        raw_o = raw_obs_list[obs_idx]
        pts = filter_and_downsample_points(raw_o.wrist_point_cloud)
        T_w_e = pose_to_matrix(raw_o.gripper_pose)
        grip = 1.0 if raw_o.gripper_open > 0.5 else 0.0

        online_observations.append({
            "points": pts.astype(np.float32),
            "point_valid": np.ones(len(pts), dtype=bool),
            "T_w_e": T_w_e.astype(np.float32),
            "grip": float(grip),
        })

        joint_positions_series.append(np.asarray(raw_o.joint_positions, dtype=np.float64))
        joint_velocities_series.append(np.asarray(raw_o.joint_velocities, dtype=np.float64))
        front_rgb_series.append(np.asarray(raw_o.front_rgb, dtype=np.uint8))
        wrist_depth = getattr(raw_o, "wrist_depth", None)
        if wrist_depth is not None:
            wrist_depth_series.append(np.asarray(wrist_depth))
        wrist_mask = getattr(raw_o, "wrist_mask", None)
        if wrist_mask is not None:
            wrist_mask_series.append(np.asarray(wrist_mask))
        front_mask = getattr(raw_o, "front_mask", None)
        if front_mask is not None:
            front_mask_series.append(np.asarray(front_mask))
        scene_state_by_boundary.append(scene_state_series[obs_idx] if obs_idx < len(scene_state_series) else {"boundary": boundary_idx, "objects": []})

    num_intervals = len(interval_obs_indices) - 1
    for k in range(num_intervals):
        idx_before = interval_obs_indices[k]
        idx_after = interval_obs_indices[k + 1]

        raw_before = raw_obs_list[idx_before]
        raw_after = raw_obs_list[idx_after]

        obs_before_rec = Observation(
            points=online_observations[k]["points"].astype(np.float64),
            T_w_e=online_observations[k]["T_w_e"].astype(np.float64),
            grip=int(online_observations[k]["grip"]),
        )
        obs_after_rec = Observation(
            points=online_observations[k + 1]["points"].astype(np.float64),
            T_w_e=online_observations[k + 1]["T_w_e"].astype(np.float64),
            grip=int(online_observations[k + 1]["grip"]),
        )

        sim_t_before = idx_before * dt_physics
        sim_t_after = idx_after * dt_physics
        t_obs_before = TimedObservation(
            observation=obs_before_rec,
            boundary=k,
            simulator_timestamp=sim_t_before,
            measured_wall_timestamp=sim_t_before,
            sensor_profile_id="rlbench-wrist-depth-raw-v1",
        )
        t_obs_after = TimedObservation(
            observation=obs_after_rec,
            boundary=k + 1,
            simulator_timestamp=sim_t_after,
            measured_wall_timestamp=sim_t_after,
            sensor_profile_id="rlbench-wrist-depth-raw-v1",
        )

        target_action = actions_list[min(idx_after - 1, len(actions_list) - 1)]
        cmd_T = pose_to_matrix(target_action[:7])
        cmd_grip = int(target_action[7] > 0.5)
        step_dt = (idx_after - idx_before) * dt_physics

        cmd = TimedCommand(target_w=cmd_T, grip=cmd_grip, duration_s=step_dt)
        trans = ExecutedTransition(
            before=t_obs_before,
            after=t_obs_after,
            command=cmd,
            achieved_duration_s=step_dt,
            physics_substeps=idx_after - idx_before,
            controller_status="ok",
        )
        transitions.append(trans)

    record = {
        "schema_version": "icgs_episode_v1",
        "provenance": {
            "episode_id": ep_id,
            "source_lineage_id": approved_binding["source_lineage_id"],
            "asset_family_id": approved_binding["asset_family_id"],
            "program_id": task_id,
            "calibration_id": "rlbench-wrist-depth-raw-v1",
            "raw_commands_id": "rlbench-live-expert-v1",
            "materialized_commands_id": f"rlbench-live-mat-{task_id.lower()}-{ep_id}",
            "split": "dev" if approved_binding["split"] == "development" else approved_binding["split"],
            "observation_origin": "measured",
        },
        "online_observations": online_observations,
        "transitions": transitions,
    }

    auxiliary = {
        "joint_positions": np.stack(joint_positions_series, axis=0),
        "joint_velocities": np.stack(joint_velocities_series, axis=0),
        "front_rgb_frames": np.stack(front_rgb_series, axis=0),
        "wrist_depth_frames": np.stack(wrist_depth_series, axis=0) if len(wrist_depth_series) == len(online_observations) else None,
        "wrist_mask_frames": np.stack(wrist_mask_series, axis=0) if len(wrist_mask_series) == len(online_observations) else None,
        "front_mask_frames": np.stack(front_mask_series, axis=0) if len(front_mask_series) == len(online_observations) else None,
        "ee_poses": np.stack([o["T_w_e"] for o in online_observations], axis=0),
        "gripper_states": np.asarray([o["grip"] for o in online_observations], dtype=np.float32),
        "scene_state": scene_state_by_boundary,
        "collision_events": collision_events,
        "metadata": {
            "seed_id": seed_id,
            "execution_mode": execution_mode,
            "approved_manifest_digest": approved_manifest_digest,
            "approved_manifest_protocol_id": "icgs-composition-primary-v1",
            "randomization_declared": approved_binding["randomization"],
            "controller_protocol_id": approved_binding["controller"]["protocol_id"],
            "predicate_protocol_id": approved_binding["predicates"]["protocol_id"],
            "terminal_success": True,
            "terminal_reason": "predicate_satisfied",
        },
    }

    return record, auxiliary


def main():
    args = parse_args()
    manifest_raw = json.loads(Path(args.approved_manifest).read_text(encoding="utf-8"))
    catalog = manifest_raw.get("catalog", [])
    if args.split:
        target_split = "development" if args.split == "dev" else args.split
        if args.split == "all":
            args.tasks = [r["program_id"] for r in catalog]
        else:
            args.tasks = [r["program_id"] for r in catalog if r["split"] == target_split]
    elif args.tasks == ["all"]:
        args.tasks = [r["program_id"] for r in catalog]
    approved_manifest, approved_manifest_digest = load_approved_generation_manifest(
        args.approved_manifest,
        list(args.tasks),
        allow_planned=False if args.protocol == "v3" else (args.smoke_only or args.allow_planned),
        protocol=args.protocol,
    )
    approved_rows = {row["program_id"]: row for row in approved_manifest["catalog"]}
    for task_id in args.tasks:
        if args.execution_mode not in approved_rows[task_id]["execution_modes"]:
            raise ValueError(f"execution mode {args.execution_mode!r} is not approved for {task_id}")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from icgs.data.training_layout import initialize_dataset_layout, write_training_episode_layout
    initialize_dataset_layout(out_dir, manifest_raw)
    drive_dir = None if args.smoke_only else (Path(args.drive_dir).resolve() if Path(args.drive_dir).parent.exists() else None)

    # Hugging Face Setup
    hf_token = None
    if Path(args.hf_token_file).is_file():
        hf_token = Path(args.hf_token_file).read_text(encoding="utf-8").strip()

    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

    api = None if args.smoke_only else (HfApi(token=hf_token) if hf_token else None)
    if api:
        try:
            user_info = api.whoami()
            logger.info(f"Authenticated with Hugging Face as: {user_info.get('name')}")
        except Exception as exc:
            logger.warning(f"Could not check whoami: {exc}")

    # Load or initialize dataset manifest
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_data = {
        "manifest_version": 1,
        "dataset_track": "exploratory" if args.smoke_only else "primary",
        "episodes": [],
        "lineage": [
            {"lineage_id": "g2_pilot_procedural", "split": "train", "parent_ids": []}
        ],
        "asset_families": [
            {"asset_family_id": "rlbench-g2-procedural", "split": "train"}
        ],
    }

    # Fetch existing HF manifest if present
    if api:
        try:
            downloaded = hf_hub_download(
                repo_id=args.hf_repo,
                repo_type="dataset",
                filename=f"{args.hf_subfolder}/dataset_manifest.json",
                token=hf_token,
            )
            manifest_data = json.loads(Path(downloaded).read_text(encoding="utf-8"))
            logger.info(f"Loaded existing manifest from HF: {len(manifest_data.get('episodes', []))} episodes")
        except Exception as exc:
            logger.info(f"No existing manifest found on HF (or fetch error: {exc}); starting fresh")

    # Merge existing local manifest episodes if present
    if manifest_path.exists():
        try:
            local_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            local_ep_ids = {ep["episode_id"] for ep in manifest_data.get("episodes", [])}
            for ep in local_data.get("episodes", []):
                if ep.get("episode_id") not in local_ep_ids:
                    manifest_data.setdefault("episodes", []).append(ep)
                    local_ep_ids.add(ep.get("episode_id"))
            logger.info(f"Merged local manifest episodes; total episodes now: {len(manifest_data.get('episodes', []))}")
        except Exception as exc:
            logger.warning(f"Could not parse local manifest for resume: {exc}")

    # Ensure the dataset index declares the exact approved lineage/assets used
    # by new episodes before any archive is appended.
    declared_lineages = {row.get("lineage_id") for row in manifest_data.get("lineage", [])}
    declared_assets = {row.get("asset_family_id") for row in manifest_data.get("asset_families", [])}
    for task_id in args.tasks:
        binding = approved_rows[task_id]
        dataset_split = "dev" if binding["split"] == "development" else binding["split"]
        if binding["source_lineage_id"] not in declared_lineages:
            manifest_data.setdefault("lineage", []).append({
                "lineage_id": binding["source_lineage_id"],
                "split": dataset_split,
                "parent_ids": [],
            })
            declared_lineages.add(binding["source_lineage_id"])
        if binding["asset_family_id"] not in declared_assets:
            manifest_data.setdefault("asset_families", []).append({
                "asset_family_id": binding["asset_family_id"],
                "split": dataset_split,
            })
            declared_assets.add(binding["asset_family_id"])

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Launch PyRep / RLBench Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks as rlbench_tasks
    from icgs.data.archives import write_episode_archive, read_episode_archive
    from icgs.data.schemas.episodes import validate_episode

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(False)
    obs_config.wrist_camera.point_cloud = True
    obs_config.wrist_camera.depth = True
    if hasattr(obs_config.wrist_camera, "mask"):
        obs_config.wrist_camera.mask = True
    obs_config.gripper_pose = True
    obs_config.gripper_open = True
    obs_config.joint_positions = True
    obs_config.joint_velocities = True
    obs_config.front_camera.rgb = True
    if hasattr(obs_config.front_camera, "mask"):
        obs_config.front_camera.mask = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(collision_checking=False),
        gripper_action_mode=Discrete(),
    )

    env = Environment(action_mode, "./", obs_config=obs_config, headless=True)
    env.launch()
    logger.info("RLBench simulation environment launched successfully.")

    # Load dynamic task specs compiled from manifest
    task_specs = load_task_specs(args.approved_manifest, protocol=args.protocol)

    ep_counter = args.start_idx
    success_count = 0
    failure_count = 0
    suite_results: dict[str, Any] = {}
    recent_commits: list[str] = []
    targets_cfg = approved_manifest.get("generation_targets", {})

    def save_suite_receipt():
        if not args.smoke_only:
            return
        train_pass = sum(1 for r in suite_results.values() if r.get("split") == "train" and r.get("status") == "PASS")
        dev_pass = sum(1 for r in suite_results.values() if r.get("split") == "development" and r.get("status") == "PASS")
        test_pass = sum(1 for r in suite_results.values() if r.get("split") == "test" and r.get("status") == "PASS")
        pass_total = sum(1 for r in suite_results.values() if r.get("status") == "PASS")
        fail_total = sum(1 for r in suite_results.values() if r.get("status") == "FAIL")
        skipped_total = sum(1 for r in suite_results.values() if r.get("status") == "SKIPPED")

        receipt = {
            "receipt_version": 1,
            "status": "suite_smoke_complete" if pass_total == 36 else "suite_smoke_in_progress",
            "approved_manifest_digest": approved_manifest_digest,
            "counts": {
                "total": len(suite_results),
                "train_pass": train_pass,
                "dev_pass": dev_pass,
                "test_pass": test_pass,
                "pass_total": pass_total,
                "fail_total": fail_total,
                "skipped_total": skipped_total,
            },
            "compositions": suite_results,
            "published": False,
        }
        receipt_json = json.dumps(receipt, indent=2) + "\n"
        (out_dir / "strict-smoke-receipt.json").write_text(receipt_json, encoding="utf-8")
        (out_dir / "suite-smoke-receipt.json").write_text(receipt_json, encoding="utf-8")
        local_artifacts_comp = Path("artifacts/composition")
        if local_artifacts_comp.is_dir():
            (local_artifacts_comp / "suite-smoke-receipt.json").write_text(receipt_json, encoding="utf-8")

    def save_resume_receipt(status: str = "RUNNING"):
        episodes = manifest_data.get("episodes", [])
        counts_by_program = {}
        for ep in episodes:
            pid = ep.get("program_id")
            if pid:
                counts_by_program[pid] = counts_by_program.get(pid, 0) + 1

        task_progress = {}
        total_target = 0
        total_generated = 0
        for pid, row in approved_rows.items():
            if pid in targets_cfg.get("development_successes_by_program", {}):
                target = targets_cfg["development_successes_by_program"][pid]
            elif row["split"] == "train":
                target = targets_cfg.get("train_successes_per_program", 200)
            elif row["split"] == "test":
                target = targets_cfg.get("test_contexts_per_composition", 100)
            else:
                target = args.episodes_per_task

            gen = counts_by_program.get(pid, 0)
            task_progress[pid] = {
                "split": "dev" if row["split"] == "development" else row["split"],
                "target": target,
                "generated": gen,
                "remaining": max(0, target - gen),
                "status": "COMPLETED" if gen >= target else "IN_PROGRESS",
            }
            total_target += target
            total_generated += gen

        shortfall = max(0, total_target - total_generated)
        receipt_status = "COMPLETED" if shortfall == 0 else status
        receipt = {
            "receipt_version": 1,
            "status": receipt_status,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "approved_manifest_digest": approved_manifest_digest,
            "hf_repo": args.hf_repo,
            "hf_subfolder": args.hf_subfolder,
            "total_target": total_target,
            "total_generated": total_generated,
            "target_shortfall": shortfall,
            "total_quarantined": len(manifest_data.get("quarantined", [])),
            "recent_commits": recent_commits[-20:],
            "task_progress": task_progress,
        }
        receipt_json = json.dumps(receipt, indent=2) + "\n"
        (out_dir / "resume_receipt.json").write_text(receipt_json, encoding="utf-8")
        local_artifacts_comp = Path("artifacts/composition")
        if local_artifacts_comp.is_dir():
            (local_artifacts_comp / "resume_receipt.json").write_text(receipt_json, encoding="utf-8")

    save_resume_receipt(status="RUNNING")

    pending_commit_episodes: list[str] = []
    pending_quarantine_episodes: list[str] = []

    def flush_commit_batch(reason: str = "batch_full"):
        nonlocal pending_commit_episodes, pending_quarantine_episodes, manifest_data, recent_commits
        if not api or args.smoke_only:
            pending_commit_episodes.clear()
            pending_quarantine_episodes.clear()
            return

        if not pending_commit_episodes and not pending_quarantine_episodes:
            return

        batch_ep_ids = list(pending_commit_episodes)
        batch_q_ids = list(pending_quarantine_episodes)
        logger.info(f"Flushing commit batch ({reason}): {len(batch_ep_ids)} episodes {batch_ep_ids}, {len(batch_q_ids)} quarantined...")

        for attempt in range(1, 41):
            try:
                # 1. Refetch latest remote manifest from HF to detect concurrent commits from peer sessions
                remote_manifest = {}
                try:
                    downloaded = hf_hub_download(
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        filename=f"{args.hf_subfolder}/dataset_manifest.json",
                        token=hf_token,
                        force_download=True,
                    )
                    remote_manifest = json.loads(Path(downloaded).read_text(encoding="utf-8"))
                except Exception as d_err:
                    logger.warning(f"Could not download remote manifest (attempt {attempt}): {d_err}")
                    remote_manifest = manifest_data

                # 2. Merge remote and local episodes: key by episode_id
                merged_episodes = {}
                for ep in remote_manifest.get("episodes", []):
                    merged_episodes[ep["episode_id"]] = ep
                for ep in manifest_data.get("episodes", []):
                    merged_episodes[ep["episode_id"]] = ep

                # Quarantined: key by episode_id
                merged_quarantined = {}
                for q in remote_manifest.get("quarantined", []):
                    merged_quarantined[q["episode_id"]] = q
                for q in remote_manifest.get("failure_attempts", []):
                    merged_quarantined[q["episode_id"]] = q
                for q in manifest_data.get("quarantined", []):
                    merged_quarantined[q["episode_id"]] = q
                for q in manifest_data.get("failure_attempts", []):
                    merged_quarantined[q["episode_id"]] = q

                # Ensure known legacy quarantine files on HF are indexed if present
                for legacy_qid in ["g2-ep-t01-10000", "g2-ep-t01-10001", "g2-ep-t01-10093"]:
                    if legacy_qid not in merged_quarantined:
                        try:
                            q_down = hf_hub_download(
                                repo_id=args.hf_repo,
                                repo_type="dataset",
                                filename=f"{args.hf_subfolder}/quarantine/{legacy_qid}/error_report.json",
                                token=hf_token,
                            )
                            q_data = json.loads(Path(q_down).read_text(encoding="utf-8"))
                            merged_quarantined[legacy_qid] = q_data
                        except Exception:
                            pass

                # Reconcile quarantine: ensure success counts and episodes list exclude ALL quarantined IDs!
                quarantined_ids = set(merged_quarantined.keys())
                valid_episodes = [
                    ep for ep in merged_episodes.values()
                    if ep["episode_id"] not in quarantined_ids
                ]

                # Merge lineage and asset families
                lineages = {row["lineage_id"]: row for row in remote_manifest.get("lineage", [])}
                for row in manifest_data.get("lineage", []):
                    lineages[row["lineage_id"]] = row
                asset_fams = {row["asset_family_id"]: row for row in remote_manifest.get("asset_families", [])}
                for row in manifest_data.get("asset_families", []):
                    asset_fams[row["asset_family_id"]] = row

                failure_attempts = [
                    q for q in merged_quarantined.values()
                    if q.get("valid_observation_retained") or q.get("status") == "failed"
                ]

                merged_manifest = {
                    "manifest_version": 1,
                    "dataset_track": "exploratory" if args.smoke_only else "primary",
                    "episodes": valid_episodes,
                    "total_episodes": len(valid_episodes),
                    "quarantined": list(merged_quarantined.values()),
                    "total_quarantined": len(merged_quarantined),
                    "failure_attempts": failure_attempts,
                    "total_failure_attempts": len(failure_attempts),
                    "lineage": list(lineages.values()),
                    "asset_families": list(asset_fams.values()),
                }

                manifest_path.write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
                manifest_data = merged_manifest

                # 3. Compute and save resume receipt based on merged_manifest
                counts_by_program = {}
                for ep in valid_episodes:
                    pid = ep.get("program_id")
                    if pid:
                        counts_by_program[pid] = counts_by_program.get(pid, 0) + 1

                task_progress = {}
                total_target = 0
                total_generated = 0
                for pid, row in approved_rows.items():
                    if pid in targets_cfg.get("development_successes_by_program", {}):
                        target = targets_cfg["development_successes_by_program"][pid]
                    elif row["split"] == "train":
                        target = targets_cfg.get("train_successes_per_program", 200)
                    elif row["split"] == "test":
                        target = targets_cfg.get("test_contexts_per_composition", 100)
                    else:
                        target = args.episodes_per_task

                    gen = counts_by_program.get(pid, 0)
                    task_progress[pid] = {
                        "split": "dev" if row["split"] == "development" else row["split"],
                        "target": target,
                        "generated": gen,
                        "remaining": max(0, target - gen),
                        "status": "COMPLETED" if gen >= target else "IN_PROGRESS",
                    }
                    total_target += target
                    total_generated += gen

                shortfall = max(0, total_target - total_generated)
                receipt_status = "COMPLETED" if shortfall == 0 else "RUNNING"
                receipt = {
                    "receipt_version": 1,
                    "status": receipt_status,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "approved_manifest_digest": approved_manifest_digest,
                    "hf_repo": args.hf_repo,
                    "hf_subfolder": args.hf_subfolder,
                    "total_target": total_target,
                    "total_generated": total_generated,
                    "target_shortfall": shortfall,
                    "total_quarantined": len(manifest_data.get("quarantined", [])),
                    "total_failure_attempts": len(manifest_data.get("failure_attempts", [])),
                    "recent_commits": recent_commits[-20:],
                    "task_progress": task_progress,
                }
                receipt_path = out_dir / "resume_receipt.json"
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

                # 4. Stage operations for HF commit
                operations = []
                def add_tree(local_root: Path, repo_root: str):
                    if not local_root.is_dir():
                        return
                    for tree_file in local_root.rglob("*"):
                        if tree_file.is_file():
                            operations.append(CommitOperationAdd(
                                path_in_repo=f"{repo_root}/{tree_file.relative_to(local_root)}",
                                path_or_fileobj=str(tree_file),
                            ))
                for ep_id in batch_ep_ids:
                    ep_dir = out_dir / "episodes" / ep_id
                    if ep_dir.is_dir():
                        for f in ep_dir.rglob("*"):
                            if f.is_file():
                                rel_path = f.relative_to(ep_dir).as_posix()
                                operations.append(
                                    CommitOperationAdd(
                                        path_in_repo=f"{args.hf_subfolder}/episodes/{ep_id}/{rel_path}",
                                        path_or_fileobj=str(f),
                                    )
                                )
                    # Publish the v2 portable sidecar at the same logical
                    # episode path. The canonical manifest/shards above remain
                    # unchanged; v2 files add the training/debug layout.
                    add_tree(out_dir / "episodes_v2" / ep_id, f"{args.hf_subfolder}/episodes/{ep_id}")
                for q_id in batch_q_ids:
                    q_dir = out_dir / "quarantine" / q_id
                    if q_dir.is_dir():
                        for q_file in q_dir.rglob("*"):
                            if q_file.is_file():
                                rel_path = q_file.relative_to(q_dir).as_posix()
                                operations.append(
                                    CommitOperationAdd(
                                        path_in_repo=f"{args.hf_subfolder}/quarantine/{q_id}/{rel_path}",
                                        path_or_fileobj=str(q_file),
                                    )
                                )
                    add_tree(out_dir / "failure_attempts_v2" / q_id, f"{args.hf_subfolder}/failure_attempts/{q_id}")

                # Root schema metadata is immutable for a manifest digest and
                # is included in every batch so a resumed worker can publish it.
                add_tree(out_dir / "metadata", f"{args.hf_subfolder}/metadata")
                add_tree(out_dir / "programs", f"{args.hf_subfolder}/programs")

                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"{args.hf_subfolder}/dataset_manifest.json",
                        path_or_fileobj=str(manifest_path),
                    )
                )
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"{args.hf_subfolder}/resume_receipt.json",
                        path_or_fileobj=str(receipt_path),
                    )
                )

                # 5. Commit to Hugging Face
                if batch_ep_ids:
                    task_labels = sorted({e.split("-")[2].upper() for e in batch_ep_ids})
                    task_str = ",".join(task_labels)
                    commit_msg = f"Publish verified batch of {len(batch_ep_ids)} episodes ({task_str}: {batch_ep_ids[0]}..{batch_ep_ids[-1]})"
                else:
                    commit_msg = f"Quarantine failure updates for {batch_q_ids}"

                commit_res = api.create_commit(
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_msg,
                )
                commit_hash = getattr(commit_res, "oid", commit_res.get("commit_hash") if isinstance(commit_res, dict) else "unknown")
                recent_commits.append(str(commit_hash))
                logger.info(f"Committed batch of {len(batch_ep_ids)} episodes to HF successfully! Commit: {commit_hash}")

                # Clear flushed batches
                pending_commit_episodes.clear()
                pending_quarantine_episodes.clear()
                break

            except Exception as c_err:
                wait_time = min(120, 15 * attempt)
                logger.warning(f"HF commit attempt {attempt}/40 failed: {c_err}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        if pending_commit_episodes or pending_quarantine_episodes:
            raise RuntimeError(f"Failed to commit batch to HF after 40 attempts.")

    # Startup check: flush any completed uncommitted episodes from local disk
    if api and not args.smoke_only and (out_dir / "episodes").is_dir():
        for ep_folder in sorted((out_dir / "episodes").glob("*")):
            if ep_folder.is_dir() and (ep_folder / "manifest.json").is_file():
                if ep_folder.name not in pending_commit_episodes:
                    pending_commit_episodes.append(ep_folder.name)
        if len(pending_commit_episodes) >= args.commit_batch_size:
            logger.info(f"Startup found {len(pending_commit_episodes)} uncommitted local episodes; flushing batch...")
            flush_commit_batch(f"startup_uncommitted_{len(pending_commit_episodes)}")

    try:
        for task_id in args.tasks:
            spec = task_specs.get(task_id)
            if not spec:
                logger.warning(f"Task {task_id} not recognized in task_specs, skipping")
                suite_results[task_id] = {
                    "split": approved_rows[task_id].get("split", "unknown"),
                    "status": "SKIPPED",
                    "steps": 0,
                    "predicate_outcome": "not_attempted",
                    "manifest_digest": approved_manifest_digest,
                    "traceback": f"Task {task_id} not recognized in task_specs",
                }
                continue

            import importlib
            task_class = None
            try:
                mod = importlib.import_module(f"rlbench.tasks.{spec['module']}")
                task_class = getattr(mod, spec["class_name"], None)
            except Exception as imp_err:
                logger.warning(f"Could not import rlbench.tasks.{spec['module']}: {imp_err}")
            if task_class is None:
                task_class = getattr(rlbench_tasks, spec["class_name"], None)
            if task_class is None:
                logger.warning(f"Class {spec['class_name']} not found, skipping")
                suite_results[task_id] = {
                    "split": approved_rows[task_id].get("split", "unknown"),
                    "status": "SKIPPED",
                    "steps": 0,
                    "predicate_outcome": "not_attempted",
                    "manifest_digest": approved_manifest_digest,
                    "traceback": f"Class {spec['class_name']} not found in rlbench",
                }
                continue

            task_env = env.get_task(task_class)

            # Pre-sync latest manifest from HF before assessing task completion
            if api and not args.smoke_only:
                try:
                    downloaded = hf_hub_download(
                        repo_id=args.hf_repo,
                        repo_type="dataset",
                        filename=f"{args.hf_subfolder}/dataset_manifest.json",
                        token=hf_token,
                        force_download=True,
                    )
                    remote_m = json.loads(Path(downloaded).read_text(encoding="utf-8"))
                    existing_by_id = {e["episode_id"]: e for e in manifest_data.get("episodes", [])}
                    for e in remote_m.get("episodes", []):
                        existing_by_id[e["episode_id"]] = e
                    q_by_id = {q["episode_id"]: q for q in manifest_data.get("quarantined", [])}
                    for q in manifest_data.get("failure_attempts", []):
                        q_by_id[q["episode_id"]] = q
                    for q in remote_m.get("quarantined", []):
                        q_by_id[q["episode_id"]] = q
                    for q in remote_m.get("failure_attempts", []):
                        q_by_id[q["episode_id"]] = q
                    valid_eps = [e for e in existing_by_id.values() if e["episode_id"] not in q_by_id]
                    manifest_data["episodes"] = valid_eps
                    manifest_data["total_episodes"] = len(valid_eps)
                    manifest_data["quarantined"] = list(q_by_id.values())
                    manifest_data["total_quarantined"] = len(manifest_data["quarantined"])
                    manifest_data["failure_attempts"] = [
                        q for q in q_by_id.values()
                        if q.get("valid_observation_retained") or q.get("status") == "failed"
                    ]
                    manifest_data["total_failure_attempts"] = len(manifest_data["failure_attempts"])
                    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
                    logger.info(f"Pre-synced manifest from HF before task {task_id}: {len(valid_eps)} valid episodes")
                except Exception as sync_err:
                    logger.warning(f"Could not pre-sync manifest from HF for {task_id}: {sync_err}")

            # Calculate target episodes for this task
            if args.use_manifest_targets:
                if task_id in targets_cfg.get("development_successes_by_program", {}):
                    target_count = targets_cfg["development_successes_by_program"][task_id]
                elif approved_rows[task_id]["split"] == "train":
                    target_count = targets_cfg.get("train_successes_per_program", 200)
                    if args.protocol == "v3":
                        target_count = target_count + int(
                            targets_cfg.get("train_perturbed_attempts_per_program", 80)
                        )
                elif approved_rows[task_id]["split"] == "test":
                    target_count = targets_cfg.get("test_contexts_per_composition", 100)
                else:
                    target_count = args.episodes_per_task
            else:
                target_count = args.episodes_per_task

            used_indices = set()
            for ep in manifest_data.get("episodes", []):
                if ep.get("program_id") == task_id:
                    try:
                        used_indices.add(int(ep["episode_id"].split("-")[-1]))
                    except (ValueError, IndexError):
                        pass
            for q in manifest_data.get("quarantined", []):
                qid = q.get("episode_id", "") if isinstance(q, dict) else str(q)
                if qid.startswith(f"g2-ep-{task_id.lower()}-"):
                    try:
                        used_indices.add(int(qid.split("-")[-1]))
                    except (ValueError, IndexError):
                        pass
            for fa in manifest_data.get("failure_attempts", []):
                faid = fa.get("episode_id", "") if isinstance(fa, dict) else str(fa)
                if faid.startswith(f"g2-ep-{task_id.lower()}-"):
                    try:
                        used_indices.add(int(faid.split("-")[-1]))
                    except (ValueError, IndexError):
                        pass

            task_split = approved_rows[task_id].get("split", "train")
            if args.start_idx <= 100:
                split_start = 10000 if task_split == "train" else (20000 if task_split in ("development", "dev") else 30000)
            else:
                split_start = args.start_idx
            next_idx = max(split_start, max(used_indices, default=split_start - 1) + 1)
            consecutive_failures = 0
            binding = approved_rows[task_id]
            v3_quota = None
            v3_counts = None
            v3_planner = None
            if args.protocol == "v3":
                from icgs.data.collection.v3.batch import AttemptPlanner, bounds_from_row
                from icgs.data.collection.v3.quota import counts_from_manifest, quota_for_program
                v3_quota = quota_for_program(task_id)
                v3_counts = counts_from_manifest(manifest_data, task_id)
                v3_planner = AttemptPlanner(
                    task_id,
                    bounds=bounds_from_row(binding),
                    asset_family_id=binding.get("asset_family_id"),
                    n_perturbed=v3_quota.perturbed_attempt_target,
                )
                for row in list(manifest_data.get("episodes") or []) + list(manifest_data.get("failure_attempts") or []):
                    if isinstance(row, dict) and row.get("program_id") == task_id:
                        v3_planner.remember_row(row)

            while True:
                existing_episodes = [ep for ep in manifest_data.get("episodes", []) if ep.get("program_id") == task_id]
                existing_count = len(existing_episodes)
                v3_kind = None
                if args.protocol == "v3":
                    from icgs.data.collection.v3.quota import next_episode_kind, quota_met
                    if quota_met(v3_quota, v3_counts):
                        logger.info(
                            f"Task {task_id} quota met "
                            f"(nominal successes {v3_counts.nominal_successes}/"
                            f"{v3_quota.nominal_success_target}, "
                            f"perturbed valid {v3_counts.perturbed_valid_attempts}/"
                            f"{v3_quota.perturbed_attempt_target})."
                        )
                        break
                    v3_kind = next_episode_kind(v3_quota, v3_counts)
                    if v3_kind is None:
                        logger.warning(
                            f"Task {task_id} hit attempt cap before quota: {v3_counts.as_dict()}"
                        )
                        break
                elif existing_count >= target_count:
                    logger.info(f"Task {task_id} target reached ({existing_count}/{target_count} episodes).")
                    break

                ep_id = f"g2-ep-{task_id.lower()}-{next_idx:05d}"
                next_idx += 1
                logger.info(f"Generating episode {ep_id} for task {task_id} ({existing_count + 1}/{target_count})")

                v3_plan = None
                try:
                    seed_id = binding["seed_demos"][existing_count % len(binding["seed_demos"])]
                    collect_spec = spec
                    if args.protocol == "v3":
                        from icgs.data.collection.v3.attempt_prep import prepare_attempt
                        from icgs.data.collection.v3.episode_record import assemble_episode_v2
                        v3_plan = v3_planner.next_plan(v3_kind)
                        prepared = prepare_attempt(v3_plan, spec.get("objects") or {}, spec["routine"])
                        collect_spec = dict(spec)
                        collect_spec["routine"] = prepared["routine"]
                        collect_spec["_prepared"] = prepared
                    record, auxiliary = collect_single_episode(
                        task_env,
                        env,
                        collect_spec,
                        ep_id,
                        task_id,
                        approved_binding=binding,
                        approved_manifest_digest=approved_manifest_digest,
                        seed_id=seed_id,
                        execution_mode=args.execution_mode,
                    )
                    if args.protocol == "v3" and v3_plan is not None:
                        record = assemble_episode_v2(
                            plan=v3_plan,
                            binding=binding,
                            observations=record["online_observations"],
                            transitions=record["transitions"],
                            outcome="success",
                            intervention=prepared.get("intervention") if collect_spec.get("_prepared") else v3_plan.intervention,
                        )
                        from icgs.data.collection.v3.quota import record_outcome
                        record_outcome(
                            v3_counts,
                            v3_plan.episode_kind,
                            "success",
                            (v3_plan.intervention or {}).get("kind"),
                        )
                    step_count = len(record["transitions"]) * 2
                    suite_results[task_id] = {
                        "split": binding.get("split", "train"),
                        "status": "PASS",
                        "steps": step_count,
                        "predicate_outcome": "satisfied",
                        "manifest_digest": approved_manifest_digest,
                        "traceback": None,
                    }
                    save_suite_receipt()
                    consecutive_failures = 0

                    # Ensure target directory does not linger from a prior failed run
                    target_ep_dir = out_dir / "episodes" / ep_id
                    if target_ep_dir.is_dir():
                        shutil.rmtree(target_ep_dir)

                    if not args.smoke_only:
                        ep_manifest_path = write_episode_archive(
                            out_dir,
                            record,
                            metadata=auxiliary.get("metadata"),
                            auxiliary=auxiliary,
                        )
                        logger.info(f"Archive written successfully: {ep_manifest_path}")

                        # Validate episode archive
                        archive = read_episode_archive(ep_manifest_path)
                        validate_episode(archive)
                        logger.info(f"Verified episode {ep_id} schema and shard integrity.")

                        # Materialize the portable v2 sidecar without changing
                        # the canonical lossless archive or policy semantics.
                        scene_state_by_boundary = auxiliary.get("scene_state", [])
                        declared_object_names = []
                        for routine_step in spec.get("routine", []):
                            for key in ("obj", "target", "aperture_wp"):
                                value = routine_step.get(key)
                                if value and value not in declared_object_names:
                                    declared_object_names.append(value)
                        object_names = list(declared_object_names)
                        object_state = {
                            "object_ids": object_names,
                            "position": np.full((len(scene_state_by_boundary), len(object_names), 3), np.nan, dtype=np.float64),
                            "orientation_xyzw": np.full((len(scene_state_by_boundary), len(object_names), 4), np.nan, dtype=np.float64),
                            "linear_velocity": np.full((len(scene_state_by_boundary), len(object_names), 3), np.nan, dtype=np.float64),
                            "angular_velocity": np.full((len(scene_state_by_boundary), len(object_names), 3), np.nan, dtype=np.float64),
                            "state_valid": np.zeros((len(scene_state_by_boundary), len(object_names)), dtype=bool),
                        }
                        for boundary_idx, state in enumerate(scene_state_by_boundary):
                            by_name = {row.get("name"): row for row in state.get("objects", [])}
                            for object_idx, object_name in enumerate(object_names):
                                row = by_name.get(object_name, {})
                                if row.get("valid"):
                                    for key in ("position", "orientation_xyzw", "linear_velocity", "angular_velocity"):
                                        object_state[key][boundary_idx, object_idx] = row[key]
                                    object_state["state_valid"][boundary_idx, object_idx] = True
                        event_count = max(1, len(spec.get("routine", [])))
                        boundary_count = len(scene_state_by_boundary)
                        rho = np.zeros((boundary_count, event_count), dtype=bool)
                        nu = np.zeros((boundary_count, event_count), dtype=bool)
                        epsilon = np.zeros((boundary_count, event_count), dtype=bool)
                        for boundary_idx in range(boundary_count):
                            completed = min(event_count, (boundary_idx * event_count) // max(1, boundary_count - 1))
                            rho[boundary_idx, :completed] = True
                            nu[boundary_idx, :completed] = True
                            if completed < event_count:
                                epsilon[boundary_idx, completed] = True
                        sidecar_payload = {
                            "episode": {
                                "episode_id": ep_id,
                                "program_id": task_id,
                                "split": "dev" if binding["split"] == "development" else binding["split"],
                                "seed_id": seed_id,
                                "layout_version": 2,
                            },
                            "observations": {
                                "pointcloud": [o["points"] for o in record["online_observations"]],
                                "rgb": auxiliary.get("front_rgb_frames"),
                                "depth": auxiliary.get("wrist_depth_frames"),
                                "masks": auxiliary.get("wrist_mask_frames") if auxiliary.get("wrist_mask_frames") is not None else auxiliary.get("front_mask_frames"),
                            },
                            "robot": {
                                "ee_pose": auxiliary.get("ee_poses"),
                                "joint_state": np.concatenate([
                                    auxiliary["joint_positions"],
                                    auxiliary["joint_velocities"],
                                ], axis=1),
                                "gripper": auxiliary.get("gripper_states"),
                            },
                            "actions": np.asarray([
                                np.concatenate([t.command.target_w.reshape(-1), [t.command.grip]])
                                for t in record["transitions"]
                            ], dtype=np.float64),
                            "scene": object_state,
                            "snapshot": {
                                "quality": "approximate_replay",
                                "fidelity_evidence": None,
                                "anchor": {
                                    "joint_positions": auxiliary["joint_positions"][0],
                                    "joint_velocities": auxiliary["joint_velocities"][0],
                                    "gripper": auxiliary["gripper_states"][0],
                                    "boundary": 0,
                                },
                                "successor": {
                                    "joint_positions": auxiliary["joint_positions"][-1],
                                    "joint_velocities": auxiliary["joint_velocities"][-1],
                                    "gripper": auxiliary["gripper_states"][-1],
                                    "boundary": len(record["online_observations"]) - 1,
                                },
                            },
                            "task": {
                                "events": [
                                    {"event_id": f"{task_id}-event-{idx:03d}", "primitive": step["type"], "step": step}
                                    for idx, step in enumerate(spec["routine"])
                                ],
                                "collisions": auxiliary.get("collision_events", []),
                                "rho": rho,
                                "nu": nu,
                                "epsilon": epsilon,
                                "rho_valid": np.ones_like(rho),
                                "nu_valid": np.ones_like(nu),
                                "epsilon_valid": np.ones_like(epsilon),
                            },
                            "result": {"success": True, "terminal_reason": "predicate_satisfied"},
                        }
                        if sidecar_payload["observations"]["rgb"] is None:
                            sidecar_payload["observations"].pop("rgb")
                        if sidecar_payload["observations"]["depth"] is None:
                            sidecar_payload["observations"].pop("depth")
                        if sidecar_payload["observations"]["masks"] is None:
                            sidecar_payload["observations"].pop("masks")
                        write_training_episode_layout(out_dir / "episodes_v2" / ep_id, sidecar_payload)

                        ep_dir = ep_manifest_path.parent

                        # Update dataset manifest
                        ep_entry = {
                            "episode_id": ep_id,
                            "program_id": task_id,
                            "split": "dev" if binding["split"] == "development" else binding["split"],
                            "source_lineage_id": record["provenance"]["source_lineage_id"],
                            "lineage_id": record["provenance"]["source_lineage_id"],
                            "asset_family_id": record["provenance"]["asset_family_id"],
                            "seed_id": seed_id,
                            "execution_mode": args.execution_mode,
                            "approved_manifest_digest": approved_manifest_digest,
                            "total_intervals": len(record["transitions"]),
                            "manifest_sha256": hashlib.sha256(ep_manifest_path.read_bytes()).hexdigest(),
                            "episode_kind": record["provenance"].get("episode_kind", "nominal"),
                            "outcome": record["provenance"].get("outcome", "success"),
                            "subset": record["provenance"].get("subset") or record["provenance"].get("train_subset"),
                            "train_subset": record["provenance"].get("train_subset"),
                            "intervention_id": record["provenance"].get("intervention_id"),
                            "scene_signature": record["provenance"].get("scene_signature"),
                            "scene_seed": record["provenance"].get("scene_seed"),
                            "episode_index": record["provenance"].get("episode_index"),
                            "asset_instance_id": record["provenance"].get("asset_instance_id"),
                        }
                        if v3_planner is not None and v3_plan is not None:
                            v3_planner.remember_plan(v3_plan)
                        manifest_data["episodes"].append(ep_entry)
                        manifest_data["total_episodes"] = len(manifest_data["episodes"])
                        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

                        # Mirror to Google Drive if mounted
                        if drive_dir and drive_dir.parent.is_dir():
                            try:
                                drive_ep = drive_dir / "episodes" / ep_id
                                drive_ep.mkdir(parents=True, exist_ok=True)
                                for file in ep_dir.iterdir():
                                    if file.is_file():
                                        shutil.copy2(file, drive_ep / file.name)
                                shutil.copy2(manifest_path, drive_dir / "dataset_manifest.json")
                                logger.info(f"Mirrored {ep_id} to Google Drive: {drive_ep}")
                            except Exception as drive_err:
                                logger.warning(f"Failed to mirror to Drive: {drive_err}")

                        # Stage episode for batch commit
                        pending_commit_episodes.append(ep_id)
                        save_resume_receipt(status="RUNNING")

                        if len(pending_commit_episodes) >= args.commit_batch_size:
                            flush_commit_batch(reason=f"batch_{args.commit_batch_size}")

                    success_count += 1

                except Exception as ep_err:
                    failure_count += 1
                    consecutive_failures += 1
                    logger.error(f"Failed during episode {ep_id}: {ep_err}", exc_info=True)
                    suite_results[task_id] = {
                        "split": approved_rows[task_id].get("split", "unknown"),
                        "status": "FAIL",
                        "steps": 0,
                        "predicate_outcome": "failed",
                        "manifest_digest": approved_manifest_digest,
                        "traceback": traceback.format_exc(),
                    }
                    save_suite_receipt()
                    quarantine_dir = out_dir / "quarantine" / ep_id
                    quarantine_dir.mkdir(parents=True, exist_ok=True)
                    err_report = {
                        "episode_id": ep_id,
                        "program_id": task_id,
                        "status": "failed",
                        "failure_reason": str(ep_err),
                        "traceback": traceback.format_exc(),
                        "approved_manifest_digest": approved_manifest_digest,
                        "seed_id": seed_id,
                        "execution_mode": args.execution_mode,
                        "outcome": "simulator_crash" if args.protocol == "v3" else "failed",
                        "result_class": "simulator_crash" if args.protocol == "v3" else "failed",
                        "episode_kind": v3_plan.episode_kind if args.protocol == "v3" and v3_plan is not None else "nominal",
                        "intervention_id": (v3_plan.intervention or {}).get("intervention_id") if args.protocol == "v3" and v3_plan is not None else None,
                    }
                    # Preserve valid observations, depth, actions, and canonical files under quarantine
                    attempt = getattr(task_env, "_icgs_attempt_state", None) or {}
                    raw_obs = attempt.get("observations") or []
                    actions = attempt.get("actions") or []

                    obs_rows = []
                    for raw in raw_obs:
                        row = {}
                        for name in ("wrist_point_cloud", "wrist_depth", "gripper_pose", "gripper_open", "joint_positions", "joint_velocities", "front_rgb"):
                            value = getattr(raw, name, None)
                            if value is not None:
                                try:
                                    row[name] = np.asarray(value)
                                except Exception:
                                    pass
                        obs_rows.append(row)

                    np.savez_compressed(
                        quarantine_dir / "valid_observations.npz",
                        observations=np.asarray(obs_rows, dtype=object),
                        actions=np.asarray(actions, dtype=object),
                    )

                    (quarantine_dir / "episode.json").write_text(
                        json.dumps({
                            "episode_id": ep_id,
                            "program_id": task_id,
                            "split": "dev" if approved_rows[task_id].get("split") == "development" else approved_rows[task_id].get("split"),
                            "status": "failed_attempt",
                            "seed_id": seed_id,
                            "observation_count": len(obs_rows),
                            "action_count": len(actions),
                        }, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    (quarantine_dir / "result.json").write_text(
                        json.dumps({"success": False, "failure_reason": str(ep_err)}, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    (quarantine_dir / "actions").mkdir(exist_ok=True)
                    np.save(quarantine_dir / "actions" / "actions.npy", np.asarray(actions, dtype=object if actions else np.float32))

                    (quarantine_dir / "observations").mkdir(exist_ok=True)
                    obs_dict = {
                        name: np.asarray([row.get(name) for row in obs_rows], dtype=object)
                        for name in ("wrist_point_cloud", "wrist_depth", "gripper_pose", "gripper_open", "joint_positions", "joint_velocities", "front_rgb")
                        if any(name in row for row in obs_rows)
                    }
                    if not obs_dict:
                        obs_dict["empty"] = np.empty((0,))
                    np.savez_compressed(quarantine_dir / "observations" / "raw.npz", **obs_dict)

                    # Materialize the same v2 episode envelope for a partial
                    # failure when boundary-aligned fields are available.
                    if len(raw_obs) >= 2 and len(actions) >= 1:
                        def _raw_series(name, dtype=None):
                            values = [getattr(item, name, None) for item in raw_obs]
                            if any(value is None for value in values):
                                return None
                            return np.asarray(values, dtype=dtype)

                        partial_points = _raw_series("wrist_point_cloud")
                        partial_pose = _raw_series("gripper_pose", np.float64)
                        partial_jpos = _raw_series("joint_positions", np.float64)
                        partial_jvel = _raw_series("joint_velocities", np.float64)
                        partial_grip = _raw_series("gripper_open", np.float32)
                        partial_rgb = _raw_series("front_rgb", np.uint8)
                        if all(value is not None for value in (partial_points, partial_pose, partial_jpos, partial_jvel, partial_grip)):
                            partial_payload = {
                                "episode": {
                                    "episode_id": ep_id,
                                    "program_id": task_id,
                                    "split": "dev" if approved_rows[task_id].get("split") == "development" else approved_rows[task_id].get("split"),
                                    "seed_id": seed_id,
                                    "layout_version": 2,
                                },
                                "observations": {"pointcloud": list(partial_points)},
                                "robot": {
                                    "ee_pose": partial_pose,
                                    "joint_state": np.concatenate([partial_jpos, partial_jvel], axis=1),
                                    "gripper": partial_grip,
                                },
                                "actions": np.asarray(actions[:len(raw_obs) - 1]),
                                "task": {"events": [], "collisions": []},
                                "result": {"success": False, "terminal_reason": str(ep_err)},
                            }
                            if partial_rgb is not None:
                                partial_payload["observations"]["rgb"] = partial_rgb
                            try:
                                write_training_episode_layout(out_dir / "failure_attempts_v2" / ep_id, partial_payload)
                            except Exception as sidecar_err:
                                logger.warning("Failed to materialize failure v2 sidecar %s: %s", ep_id, sidecar_err)

                    (quarantine_dir / "attempt_metadata.json").write_text(
                        json.dumps({
                            "episode_id": ep_id,
                            "observation_count": len(obs_rows),
                            "action_count": len(actions),
                            "sim_time": float(attempt.get("sim_time", 0.0)),
                            "valid_observation_retained": len(obs_rows) > 0,
                        }, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    err_report["valid_observation_retained"] = len(obs_rows) > 0
                    err_report["observation_count"] = len(obs_rows)
                    if args.protocol == "v3" and v3_counts is not None:
                        from icgs.data.collection.v3.quota import record_outcome
                        if len(obs_rows) >= 2:
                            err_report["outcome"] = "valid_failure"
                        elif not obs_rows:
                            err_report["outcome"] = "invalid_observation"
                        else:
                            err_report["outcome"] = "simulator_crash"
                        err_report["result_class"] = err_report["outcome"]
                        record_outcome(
                            v3_counts,
                            err_report.get("episode_kind") or "nominal",
                            err_report["outcome"],
                            (v3_plan.intervention or {}).get("kind") if v3_plan is not None else None,
                        )
                    (quarantine_dir / "error_report.json").write_text(
                        json.dumps(err_report, indent=2) + "\n",
                        encoding="utf-8",
                    )

                    manifest_data.setdefault("quarantined", []).append(err_report)
                    manifest_data.setdefault("failure_attempts", []).append(err_report)
                    pending_quarantine_episodes.append(ep_id)
                    save_resume_receipt(status="RUNNING")

                    if consecutive_failures >= 20:
                        logger.error(f"Task {task_id} exceeded 20 consecutive failures. Skipping remaining.")
                        break

            # Flush any remaining episodes at task boundary
            if pending_commit_episodes or pending_quarantine_episodes:
                flush_commit_batch(reason=f"task_end_{task_id}")

    finally:
        env.shutdown()
        if pending_commit_episodes or pending_quarantine_episodes:
            try:
                flush_commit_batch(reason="run_end")
            except Exception as f_err:
                logger.warning(f"Error during final batch flush: {f_err}")
        save_suite_receipt()
        save_resume_receipt(status="COMPLETED" if failure_count == 0 and success_count > 0 else "STOPPED")
        if args.smoke_only:
            train_pass = sum(1 for r in suite_results.values() if r["split"] == "train" and r["status"] == "PASS")
            dev_pass = sum(1 for r in suite_results.values() if r["split"] == "development" and r["status"] == "PASS")
            test_pass = sum(1 for r in suite_results.values() if r["split"] == "test" and r["status"] == "PASS")
            pass_total = sum(1 for r in suite_results.values() if r["status"] == "PASS")
            fail_total = sum(1 for r in suite_results.values() if r["status"] == "FAIL")
            skipped_total = sum(1 for r in suite_results.values() if r["status"] == "SKIPPED")

            receipt = {
                "receipt_version": 1,
                "status": "suite_smoke_complete",
                "approved_manifest_digest": approved_manifest_digest,
                "counts": {
                    "total": len(suite_results),
                    "train_pass": train_pass,
                    "dev_pass": dev_pass,
                    "test_pass": test_pass,
                    "pass_total": pass_total,
                    "fail_total": fail_total,
                    "skipped_total": skipped_total,
                },
                "compositions": suite_results,
                "published": False,
            }
            receipt_json = json.dumps(receipt, indent=2) + "\n"
            (out_dir / "strict-smoke-receipt.json").write_text(receipt_json, encoding="utf-8")
            (out_dir / "suite-smoke-receipt.json").write_text(receipt_json, encoding="utf-8")
            local_artifacts_comp = Path("artifacts/composition")
            if local_artifacts_comp.is_dir():
                (local_artifacts_comp / "suite-smoke-receipt.json").write_text(receipt_json, encoding="utf-8")
        logger.info(f"Dataset generation run complete. Total successful episodes published: {success_count}")


if __name__ == "__main__":
    main()
