"""ICGS mirror of RLBench ``Scene.get_demo`` (buttomnutstoast/RLBench@587a6a0).

The loop below calls the same upstream primitives in the same order as
``rlbench/backend/scene.py:get_demo`` (collidable toggling, ``point.get_path``,
``path.visualize``, ``path.step`` + ``scene.step``, gripper ``actuate`` loops,
kinematic ``grasp``, and the final 10 settle steps). It adds only:

* the ability to start at / stop after a waypoint (anchors, continuations);
* declared execution perturbations applied through public PyRep calls;
* failure retention: planning errors and unsuccessful endings become outcomes
  instead of exceptions, and a step budget produces ``timeout``.

``scripts/robohiman_gates.py parity`` compares this mirror against upstream
``get_demo`` step by step before it may be used for collection.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


MIRROR_ID = "icgs-rlbench-get_demo-mirror-v1 (RLBench 587a6a0 scene.py:323-447)"


@dataclass
class ExpertResult:
    status: str  # success | failure | invalid_execution | timeout
    reason: str | None
    task_success: bool
    last_completed_waypoint: int
    success_first_step: int | None = None
    applied_perturbations: list[dict[str, Any]] = field(default_factory=list)


class _Timeout(Exception):
    pass


@contextmanager
def expert_control(session: Any) -> Iterator[None]:
    """``TaskEnvironmentExt.get_demos(live_demos=True)`` enables the arm control loop."""
    arm = session.robot.arm
    previous = arm.joints[0].is_control_loop_enabled()
    arm.set_control_loop_enabled(True)
    try:
        yield
    finally:
        arm.set_control_loop_enabled(previous)


def _rotation(rpy_deg: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", np.asarray(rpy_deg, dtype=np.float64), degrees=True).as_matrix()


class WaypointExpert:
    def __init__(self, session: Any, recorder: Any, *, perturbations: list[dict[str, Any]] | None = None,
                 max_steps: int = 3000) -> None:
        self.session = session
        self.recorder = recorder
        self.perturbations = [dict(item) for item in (perturbations or [])]
        self.max_steps = int(max_steps)
        self.applied: list[dict[str, Any]] = []
        self._success_first_step: int | None = None

    # ----------------------------------------------------------- utilities
    def _steps(self) -> int:
        return len(self.recorder.commands)

    def _check_budget(self) -> None:
        if self._steps() >= self.max_steps:
            raise _Timeout(f"step budget {self.max_steps} exhausted")

    def _note_success(self) -> None:
        if self._success_first_step is None and self.recorder.steps and self.recorder.steps[-1]["task_success"]:
            self._success_first_step = len(self.recorder.steps) - 1

    def _scene_step(self) -> None:
        self._check_budget()
        self.session.scene.step()
        self._note_success()

    def _physics_step(self) -> None:
        """Gripper/settle loops in get_demo call pyrep.step() then task.step()."""
        self._check_budget()
        self.session.scene.pyrep.step()
        self.session.scene.task.step()
        self._note_success()

    def _for_waypoint(self, family: str, index: int) -> list[dict[str, Any]]:
        return [item for item in self.perturbations
                if item["family"] == family and int(item.get("waypoint", -1)) == index]

    def _apply_before_path(self, index: int, point: Any) -> None:
        from pyrep.objects.shape import Shape

        step = len(self.recorder.steps) - 1
        for item in self._for_waypoint("execution_pose_offset", index):
            dummy = point._waypoint
            before = np.asarray(dummy.get_pose(), dtype=np.float64)
            dummy.set_position(before[:3] + np.asarray(item.get("delta_m", (0, 0, 0)), dtype=np.float64))
            if any(item.get("delta_rpy_deg", (0, 0, 0))):
                from scipy.spatial.transform import Rotation

                rotation = _rotation(item["delta_rpy_deg"]) @ Rotation.from_quat(before[3:]).as_matrix()
                dummy.set_quaternion(Rotation.from_matrix(rotation).as_quat())
            after = np.asarray(dummy.get_pose(), dtype=np.float64)
            self.applied.append({**item, "applied_at_step": step, "target": dummy.get_name(),
                                 "pose_before": before.tolist(), "pose_after": after.tolist()})
        for item in self._for_waypoint("object_displacement", index):
            shape = Shape(item["object"])
            before = np.asarray(shape.get_position(), dtype=np.float64)
            shape.set_position(before + np.asarray(item["delta_m"], dtype=np.float64))
            after = np.asarray(shape.get_position(), dtype=np.float64)
            self.applied.append({**item, "applied_at_step": step, "external_intervention": True,
                                 "position_before": before.tolist(), "position_after": after.tolist()})

    # ----------------------------------------------------------- main loop
    def run(self, *, start: int = 0, stop_after: int | None = None, initial_step: bool = True) -> ExpertResult:
        """Execute waypoints ``start..``; mirrors get_demo when start=0 and stop_after=None."""
        from pyrep.const import ObjectType
        from pyrep.errors import ConfigurationPathError

        scene = self.session.scene
        task = scene.task
        robot = scene.robot
        pyrep = scene.pyrep
        last_completed = start - 1
        self.recorder.phase = "settle"
        try:
            if start == 0 and initial_step:
                if not scene._has_init_task:
                    scene.init_task()
                if not scene._has_init_episode:
                    scene.init_episode(scene._variation_index, randomly_place=True)
                scene._has_init_episode = False
            waypoints = task.get_waypoints()
            if len(waypoints) == 0:
                return ExpertResult("invalid_execution", "no_waypoints", False, last_completed)
            if start == 0 and initial_step:
                self._check_budget()
                self.recorder.phase = "initial"
                pyrep.step()  # upstream: "Need this here or get_force doesn't work..."
            success = False
            while True:
                success = False
                scene._ignore_collisions_for_current_waypoint = False
                for i, point in enumerate(waypoints):
                    if i < start:
                        continue
                    self.recorder.waypoint_index = i
                    scene._ignore_collisions_for_current_waypoint = point._ignore_collisions
                    point.start_of_path()
                    if point.skip:
                        continue
                    grasped_objects = robot.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)
                                        if s not in grasped_objects and s not in scene._robot_shapes
                                        and s.is_collidable() and robot.arm.check_arm_collision(s)]
                    [s.set_collidable(False) for s in colliding_shapes]
                    self._apply_before_path(i, point)
                    try:
                        path = point.get_path()
                        [s.set_collidable(True) for s in colliding_shapes]
                    except ConfigurationPathError:
                        [s.set_collidable(True) for s in colliding_shapes]
                        return self._finish("invalid_execution", f"no_path_waypoint_{i}", last_completed)
                    ext = point.get_ext()
                    path.visualize()
                    early = self._for_waypoint("gripper_timing", i)
                    early_distance = next((float(p["close_early_at_distance_m"]) for p in early
                                           if "close_early_at_distance_m" in p), None)
                    self.recorder.phase = "path"
                    done = False
                    success = False
                    while not done:
                        done = path.step()
                        self._scene_step()
                        success, _ = task.success()
                        if early_distance is not None and not done:
                            gap = np.linalg.norm(robot.arm.get_tip().get_position() - point._waypoint.get_position())
                            if gap <= early_distance:
                                self.applied.append({**early[0], "applied_at_step": len(self.recorder.steps) - 1,
                                                     "tip_to_waypoint_m": float(gap)})
                                break
                    point.end_of_path()
                    path.clear_visualization()
                    for item in early:
                        for _ in range(int(item.get("delay_steps", 0))):
                            self._scene_step()
                        if item.get("delay_steps"):
                            self.applied.append({**item, "applied_at_step": len(self.recorder.steps) - 1})
                    if len(ext) > 0:
                        self.recorder.phase = "gripper"
                        contains_param = False
                        start_of_bracket = -1
                        gripper = robot.gripper
                        if "open_gripper(" in ext:
                            gripper.release()
                            start_of_bracket = ext.index("open_gripper(") + 13
                            contains_param = ext[start_of_bracket] != ")"
                            if not contains_param:
                                done = False
                                while not done:
                                    done = gripper.actuate(1.0, 0.04)
                                    self._physics_step()
                        elif "close_gripper(" in ext:
                            start_of_bracket = ext.index("close_gripper(") + 14
                            contains_param = ext[start_of_bracket] != ")"
                            if not contains_param:
                                done = False
                                while not done:
                                    done = gripper.actuate(0.0, 0.04)
                                    self._physics_step()
                        if contains_param:
                            rest = ext[start_of_bracket:]
                            num = float(rest[:rest.index(")")])
                            done = False
                            while not done:
                                done = gripper.actuate(num, 0.04)
                                self._physics_step()
                        if "close_gripper(" in ext:
                            for g_obj in task.get_graspable_objects():
                                gripper.grasp(g_obj)
                    last_completed = i
                    if stop_after is not None and i >= stop_after:
                        return self._finish("anchor", None, last_completed)
                if not task.should_repeat_waypoints() or success:
                    break
            self.recorder.phase = "settle"
            if not success:
                for _ in range(10):
                    self._physics_step()
                    success, _ = task.success()
                    if success:
                        break
            success, _ = task.success()
            return self._finish("success" if success else "failure",
                                None if success else "task_success_false_after_expert", last_completed)
        except _Timeout as exc:
            return self._finish("timeout", str(exc), last_completed)

    def _finish(self, status: str, reason: str | None, last_completed: int) -> ExpertResult:
        success = bool(self.session.task_obj.success()[0])
        return ExpertResult(status, reason, success, last_completed, self._success_first_step, list(self.applied))


__all__ = ["ExpertResult", "MIRROR_ID", "WaypointExpert", "expert_control"]
