"""Anchor snapshot/restore candidates and open-loop command replay.

Nothing here is assumed exact. ``capture_snapshot`` stores what PyRep exposes
(configuration trees, joint positions/targets, attachments, numpy RNG); the
replay gate measures what restoring it actually reproduces. Fields CoppeliaSim
does not expose (Bullet contact/solver caches, PD integrators, internal RNG)
are reported as unavailable, never as restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from icgs.environments.robohiman.recorder import PHASES


SNAPSHOT_ID = "robohiman-configtree-snapshot-v1"
REPLAY_ID = "robohiman-openloop-command-replay-v1"

# Coverage for icgs.environments.rlbench.replay.validate_replay_report.
SNAPSHOT_FIELDS = {
    "stored": ("joints", "velocities", "articulations", "attachments", "monitor_history", "rng",
               "object_poses", "grip", "controller_state", "task_history", "command_history"),
    "restored": ("joints", "articulations", "attachments", "monitor_history", "rng",
                 "object_poses", "grip", "controller_state", "task_history"),
    "unavailable": ("constraints", "integrators", "solver"),
}


@dataclass
class Snapshot:
    task_tree: bytes
    task_object_count: int
    arm_tree: bytes
    gripper_tree: bytes
    arm_joint_positions: np.ndarray
    arm_joint_targets: np.ndarray
    arm_joint_velocities: np.ndarray
    gripper_joint_positions: np.ndarray
    grasped: list[tuple[str, np.ndarray]]
    graspable_poses: dict[str, np.ndarray]
    gripper_prev_positions: list[Any]
    gripper_prev_vels: list[Any]
    control_loop: bool
    numpy_rng: tuple
    sim_time: float
    boundary: int


def capture_snapshot(session: Any, boundary: int) -> Snapshot:
    from pyrep.backend import sim

    robot = session.robot
    task_obj = session.task_obj
    state, count = task_obj.get_state()
    gripper = robot.gripper
    return Snapshot(
        task_tree=state,
        task_object_count=count,
        arm_tree=robot.arm.get_configuration_tree(),
        gripper_tree=gripper.get_configuration_tree(),
        arm_joint_positions=np.asarray(robot.arm.get_joint_positions(), dtype=np.float64),
        arm_joint_targets=np.asarray(robot.arm.get_joint_target_positions(), dtype=np.float64),
        arm_joint_velocities=np.asarray(robot.arm.get_joint_velocities(), dtype=np.float64),
        gripper_joint_positions=np.asarray(gripper.get_joint_positions(), dtype=np.float64),
        grasped=[(obj.get_name(), np.asarray(obj.get_pose(), dtype=np.float64))
                 for obj in gripper.get_grasped_objects()],
        graspable_poses={obj.get_name(): np.asarray(obj.get_pose(), dtype=np.float64)
                         for obj in task_obj.get_graspable_objects()},
        gripper_prev_positions=list(gripper._prev_positions),
        gripper_prev_vels=list(gripper._prev_vels),
        control_loop=bool(robot.arm.joints[0].is_control_loop_enabled()),
        numpy_rng=np.random.get_state(),
        sim_time=float(sim.simGetSimulationTime()),
        boundary=int(boundary),
    )


def restore_snapshot(session: Any, snap: Snapshot) -> None:
    """Restore the stored fields without stepping physics."""
    from pyrep.backend import sim
    from pyrep.objects.shape import Shape

    robot = session.robot
    gripper = robot.gripper
    pyrep = session.pyrep
    gripper.release()  # detach anything currently held, restoring task parents
    pyrep.set_configuration_tree(snap.task_tree)
    for name, pose in snap.graspable_poses.items():
        Shape(name).set_pose(pose)
    pyrep.set_configuration_tree(snap.arm_tree)
    pyrep.set_configuration_tree(snap.gripper_tree)
    robot.arm.set_control_loop_enabled(snap.control_loop)
    robot.arm.set_joint_target_positions(snap.arm_joint_targets)
    for joint in gripper.joints:
        joint.set_joint_target_velocity(0.0)
    for name, pose in snap.grasped:
        obj = Shape(name)
        obj.set_pose(pose)
        gripper.grasp(obj)
    held = {obj.get_name() for obj in gripper.get_grasped_objects()}
    if held != {name for name, _ in snap.grasped}:
        raise RuntimeError(f"attachment restore mismatch: {sorted(held)}")
    gripper._prev_positions = list(snap.gripper_prev_positions)
    gripper._prev_vels = list(snap.gripper_prev_vels)
    for shape in session.task_shapes():
        if shape.is_dynamic():
            sim.simResetDynamicObject(shape.get_handle())
    np.random.set_state(snap.numpy_rng)


def replay_commands(session: Any, arrays: dict[str, np.ndarray], start: int, stop: int) -> None:
    """Re-issue recorded command rows ``start..stop-1`` open loop.

    Each row sets exactly the channels that were written originally, applies
    the recorded attach/detach event, then advances physics with the same call
    the original phase used (``scene.step`` for path rows, ``pyrep.step`` +
    ``task.step`` for gripper/settle rows).
    """
    scene = session.scene
    robot = session.robot
    graspables = session.task_obj.get_graspable_objects()
    for s in range(start, stop):
        if arrays["cmd_grasp_event"][s] < 0:
            robot.gripper.release()
        if arrays["cmd_arm_joint_target_valid"][s]:
            robot.arm.set_joint_target_positions(arrays["cmd_arm_joint_target"][s])
        if arrays["cmd_arm_joint_velocity_valid"][s]:
            robot.arm.set_joint_target_velocities(arrays["cmd_arm_joint_velocity"][s])
        for index, joint in enumerate(robot.gripper.joints):
            if arrays["cmd_gripper_joint_velocity_valid"][s, index]:
                joint.set_joint_target_velocity(float(arrays["cmd_gripper_joint_velocity"][s, index]))
        if arrays["cmd_grasp_event"][s] > 0:
            for obj in graspables:
                robot.gripper.grasp(obj)
        phase = int(arrays["cmd_phase"][s])
        if phase == PHASES["path"]:
            scene.step()
        elif phase == PHASES["initial"]:
            scene.pyrep.step()
        else:
            scene.pyrep.step()
            scene.task.step()


def trajectory_discrepancy(a: dict[str, np.ndarray], b: dict[str, np.ndarray],
                           a_start: int, b_start: int, length: int) -> dict[str, float]:
    """Max absolute differences between two aligned boundary windows."""
    def window(arrays: dict[str, np.ndarray], key: str, start: int) -> np.ndarray:
        return arrays[key][start:start + length + 1]

    result: dict[str, float] = {}
    for key, name in (("step_joint_positions", "joint_rad"), ("step_gripper_joint_positions", "finger_m"),
                      ("step_task_joint_positions", "articulation_m")):
        result[f"max_{name}"] = float(np.abs(window(a, key, a_start) - window(b, key, b_start)).max(initial=0.0))
    tip = np.linalg.norm(window(a, "step_tip_pose", a_start)[:, :3] - window(b, "step_tip_pose", b_start)[:, :3], axis=-1)
    result["max_tip_position_m"] = float(tip.max(initial=0.0))
    result["final_tip_position_m"] = float(tip[-1]) if tip.size else 0.0
    objects = np.linalg.norm(window(a, "step_object_poses", a_start)[..., :3]
                             - window(b, "step_object_poses", b_start)[..., :3], axis=-1)
    result["max_object_position_m"] = float(objects.max(initial=0.0))
    result["final_object_position_m"] = float(objects[-1].max(initial=0.0)) if objects.size else 0.0
    pa = window(a, "step_predicates", a_start)
    pb = window(b, "step_predicates", b_start)
    result["predicate_disagreement_fraction"] = float((pa != pb).mean()) if pa.size else 0.0
    result["final_success_equal"] = float(bool(window(a, "step_task_success", a_start)[-1])
                                          == bool(window(b, "step_task_success", b_start)[-1]))
    return result


__all__ = ["REPLAY_ID", "SNAPSHOT_FIELDS", "SNAPSHOT_ID", "Snapshot", "capture_snapshot",
           "replay_commands", "restore_snapshot", "trajectory_discrepancy"]
