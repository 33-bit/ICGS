"""Per-physics-step command/state recorder installed as instance wrappers.

The recorder never edits upstream files. ``install`` shadows a few bound
methods on the live PyRep/robot *instances* (``pyrep.step``, arm target
setters, gripper joint velocity setters, ``gripper.actuate/grasp/release``);
``uninstall`` deletes those instance attributes so the class methods are
visible again. Wrappers only copy arguments and forward the call unchanged.

Commands written between two physics steps form command row ``s``; the state
read after the step forms boundary row ``s + 1``. Grasp/release events that
happen after the last step of a phase (RLBench attaches after the close loop)
belong to the next command row, matching when they first affect physics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# "initial" = the lone pyrep.step() get_demo issues before the first waypoint;
# "settle"/"gripper" rows are pyrep.step() + task.step(); "path" rows are scene.step().
PHASES = {"settle": 0, "path": 1, "gripper": 2, "policy": 3, "replay": 4, "other": 5, "initial": 6}


@dataclass
class _Pending:
    arm_target: np.ndarray | None = None
    arm_velocity: np.ndarray | None = None
    gripper_velocity: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    gripper_amount: float | None = None
    grasp_event: int = 0
    grasped_names: tuple[str, ...] = ()
    teleport_calls: int = 0


class StepRecorder:
    def __init__(self, session: Any, monitor: Any, *, frame_stride: int = 1,
                 capture_frame: Callable[[], Any] | None = None,
                 wall_clock: Callable[[], float] = time.perf_counter) -> None:
        self.session = session
        self.monitor = monitor
        self.frame_stride = int(frame_stride)
        self._capture_frame = capture_frame
        self._clock = wall_clock
        self._installed: list[tuple[Any, str]] = []
        self._pending = _Pending()
        self.phase = "other"
        self.waypoint_index = -1
        self.armed = False
        self.steps: list[dict[str, Any]] = []
        self.commands: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self.frame_steps: list[int] = []
        self.events: list[dict[str, Any]] = []
        self.object_names: list[str] = []
        self.joint_names: list[str] = []

    # ------------------------------------------------------------- wrappers
    def _wrap(self, owner: Any, name: str, make: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
        if name in vars(owner):
            raise RuntimeError(f"{type(owner).__name__}.{name} already shadowed")
        original = getattr(owner, name)
        setattr(owner, name, make(original))
        self._installed.append((owner, name))

    def install(self) -> None:
        pyrep = self.session.pyrep
        robot = self.session.robot
        pending = lambda: self._pending  # noqa: E731

        def step_wrapper(original):
            def step(*args, **kwargs):
                if not self.armed:
                    return original(*args, **kwargs)
                row = self._close_command_row()
                start = self._clock()
                result = original(*args, **kwargs)
                row["wall_s"] = self._clock() - start
                self.commands.append(row)
                self._capture_state()
                return result
            return step

        def arm_target(original):
            def call(positions, *args, **kwargs):
                pending().arm_target = np.asarray(positions, dtype=np.float64).copy()
                return original(positions, *args, **kwargs)
            return call

        def arm_velocity(original):
            def call(velocities, *args, **kwargs):
                pending().arm_velocity = np.asarray(velocities, dtype=np.float64).copy()
                return original(velocities, *args, **kwargs)
            return call

        def arm_teleport(original):
            def call(*args, **kwargs):
                pending().teleport_calls += 1
                return original(*args, **kwargs)
            return call

        def actuate(original):
            def call(amount, velocity, *args, **kwargs):
                pending().gripper_amount = float(amount)
                return original(amount, velocity, *args, **kwargs)
            return call

        def grasp(original):
            def call(obj, *args, **kwargs):
                detected = original(obj, *args, **kwargs)
                if detected:
                    pending().grasp_event = 1
                    pending().grasped_names += (obj.get_name(),)
                    self.events.append({"kind": "grasp", "object": obj.get_name(), "before_step": len(self.commands)})
                return detected
            return call

        def release(original):
            def call(*args, **kwargs):
                held = [o.get_name() for o in robot.gripper.get_grasped_objects()]
                result = original(*args, **kwargs)
                if held:
                    pending().grasp_event = -1
                    self.events.append({"kind": "release", "objects": held, "before_step": len(self.commands)})
                return result
            return call

        self._wrap(pyrep, "step", step_wrapper)
        self._wrap(robot.arm, "set_joint_target_positions", arm_target)
        self._wrap(robot.arm, "set_joint_target_velocities", arm_velocity)
        self._wrap(robot.arm, "set_joint_positions", arm_teleport)
        self._wrap(robot.gripper, "actuate", actuate)
        self._wrap(robot.gripper, "grasp", grasp)
        self._wrap(robot.gripper, "release", release)
        for index, joint in enumerate(robot.gripper.joints):
            def joint_velocity(original, index=index):
                def call(value, *args, **kwargs):
                    pending().gripper_velocity[index] = float(value)
                    return original(value, *args, **kwargs)
                return call
            self._wrap(joint, "set_joint_target_velocity", joint_velocity)

    def uninstall(self) -> None:
        while self._installed:
            owner, name = self._installed.pop()
            delattr(owner, name)

    def __enter__(self) -> "StepRecorder":
        self.install()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()

    # ------------------------------------------------------------- capture
    def arm(self) -> None:
        """Start the window: boundary 0 is the state now."""
        task_objects = self.session.task_shapes()
        self.object_names = [shape.get_name() for shape in task_objects]
        self._objects = task_objects
        self._joints = self.session.task_joints()
        self.joint_names = [joint.get_name() for joint in self._joints]
        self._pending = _Pending()
        self.armed = True
        self._capture_state()

    def disarm(self) -> None:
        self.armed = False

    def _close_command_row(self) -> dict[str, Any]:
        p = self._pending
        self._pending = _Pending()
        return {
            "arm_target": p.arm_target,
            "arm_velocity": p.arm_velocity,
            "gripper_velocity": p.gripper_velocity,
            "gripper_amount": p.gripper_amount,
            "grasp_event": p.grasp_event,
            "teleport_calls": p.teleport_calls,
            "phase": PHASES[self.phase],
            "waypoint": self.waypoint_index,
        }

    @staticmethod
    def _forces(arm: Any) -> np.ndarray | None:
        """Joint forces are unavailable until a physics step follows a state restore."""
        try:
            return np.asarray(arm.get_joint_forces(), dtype=np.float64)
        except RuntimeError:
            return None

    def _capture_state(self) -> None:
        from pyrep.backend import sim

        robot = self.session.robot
        arm = robot.arm
        gripper = robot.gripper
        tip = arm.get_tip()
        grasped = {o.get_name() for o in gripper.get_grasped_objects()}
        poses = np.array([shape.get_pose() for shape in self._objects], dtype=np.float64).reshape(-1, 7)
        velocities = np.array([np.concatenate(shape.get_velocity()) for shape in self._objects],
                              dtype=np.float64).reshape(-1, 6)
        row = {
            "sim_time": float(sim.simGetSimulationTime()),
            "joint_positions": np.asarray(arm.get_joint_positions(), dtype=np.float64),
            "joint_velocities": np.asarray(arm.get_joint_velocities(), dtype=np.float64),
            "joint_forces": self._forces(arm),
            "joint_target_positions": np.asarray(arm.get_joint_target_positions(), dtype=np.float64),
            "tip_pose": np.asarray(tip.get_pose(), dtype=np.float64),
            "gripper_joint_positions": np.asarray(gripper.get_joint_positions(), dtype=np.float64),
            "gripper_open_amount": np.asarray(gripper.get_open_amount(), dtype=np.float64),
            "grasped": np.array([name in grasped for name in self.object_names], dtype=bool),
            "object_poses": poses,
            "object_velocities": velocities,
            "task_joint_positions": np.array([j.get_joint_position() for j in self._joints], dtype=np.float64),
            "predicates": self.monitor.evaluate(),
            "task_success": bool(self.session.task_obj.success()[0]),
        }
        self.steps.append(row)
        boundary = len(self.steps) - 1
        if self._capture_frame is not None and self.frame_stride > 0 and boundary % self.frame_stride == 0:
            self.frames.append(self._capture_frame())
            self.frame_steps.append(boundary)

    def capture_final_frame(self) -> None:
        boundary = len(self.steps) - 1
        if self._capture_frame is not None and (not self.frame_steps or self.frame_steps[-1] != boundary):
            self.frames.append(self._capture_frame())
            self.frame_steps.append(boundary)

    def trailing_command(self) -> dict[str, Any]:
        """Commands written after the final physics step (never executed)."""
        return self._close_command_row()

    # ------------------------------------------------------------- export
    def arrays(self) -> dict[str, np.ndarray]:
        steps, commands = self.steps, self.commands
        count = len(commands)
        arm_dof = steps[0]["joint_positions"].shape[0]

        def stack(key: str, dtype: Any = np.float64) -> np.ndarray:
            return np.stack([row[key] for row in steps]).astype(dtype)

        arm_target = np.zeros((count, arm_dof))
        arm_target_valid = np.zeros(count, dtype=bool)
        arm_velocity = np.zeros((count, arm_dof))
        arm_velocity_valid = np.zeros(count, dtype=bool)
        gripper_velocity = np.zeros((count, 2))
        gripper_velocity_valid = np.zeros((count, 2), dtype=bool)
        gripper_amount = np.zeros(count)
        gripper_amount_valid = np.zeros(count, dtype=bool)
        for index, row in enumerate(commands):
            if row["arm_target"] is not None:
                arm_target[index] = row["arm_target"]
                arm_target_valid[index] = True
            if row["arm_velocity"] is not None:
                arm_velocity[index] = row["arm_velocity"]
                arm_velocity_valid[index] = True
            written = np.isfinite(row["gripper_velocity"])
            gripper_velocity[index, written] = row["gripper_velocity"][written]
            gripper_velocity_valid[index] = written
            if row["gripper_amount"] is not None:
                gripper_amount[index] = row["gripper_amount"]
                gripper_amount_valid[index] = True
        result = {
            "step_sim_time": np.array([row["sim_time"] for row in steps]),
            "step_joint_positions": stack("joint_positions"),
            "step_joint_velocities": stack("joint_velocities"),
            "step_joint_forces": np.stack([row["joint_forces"] if row["joint_forces"] is not None
                                           else np.zeros(arm_dof) for row in steps]),
            "step_joint_forces_valid": np.array([row["joint_forces"] is not None for row in steps], dtype=bool),
            "step_joint_target_positions": stack("joint_target_positions"),
            "step_tip_pose": stack("tip_pose"),
            "step_gripper_joint_positions": stack("gripper_joint_positions"),
            "step_gripper_open_amount": np.stack([np.atleast_1d(r["gripper_open_amount"]) for r in steps]),
            "step_grasped": stack("grasped", bool),
            "step_object_poses": stack("object_poses"),
            "step_object_velocities": stack("object_velocities"),
            "step_task_joint_positions": stack("task_joint_positions"),
            "step_predicates": stack("predicates", bool),
            "step_task_success": np.array([row["task_success"] for row in steps], dtype=bool),
            "cmd_arm_joint_target": arm_target,
            "cmd_arm_joint_target_valid": arm_target_valid,
            "cmd_arm_joint_velocity": arm_velocity,
            "cmd_arm_joint_velocity_valid": arm_velocity_valid,
            "cmd_gripper_joint_velocity": gripper_velocity,
            "cmd_gripper_joint_velocity_valid": gripper_velocity_valid,
            "cmd_gripper_amount": gripper_amount,
            "cmd_gripper_amount_valid": gripper_amount_valid,
            "cmd_grasp_event": np.array([row["grasp_event"] for row in commands], dtype=np.int8),
            "cmd_arm_teleport_calls": np.array([row["teleport_calls"] for row in commands], dtype=np.int16),
            "cmd_phase": np.array([row["phase"] for row in commands], dtype=np.int8),
            "cmd_waypoint": np.array([row["waypoint"] for row in commands], dtype=np.int16),
            "cmd_wall_s": np.array([row.get("wall_s", 0.0) for row in commands]),
        }
        if self.frames:
            result["frame_step"] = np.array(self.frame_steps, dtype=np.int64)
            for key in self.frames[0]:
                result[key] = np.stack([frame[key] for frame in self.frames])
        return result


__all__ = ["PHASES", "StepRecorder"]
