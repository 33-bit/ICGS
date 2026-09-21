"""Compile v3 programs from structured catalog steps, not hardcoded routines."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.scenes import get_scene_layout
from icgs.data.collection.v3.steps import ProgramStep, get_v3_program


@dataclass(frozen=True)
class CompiledV3Task:
    program_id: str
    class_name: str
    module: str
    split: str
    family: str
    ordered_steps: tuple[str, ...]
    events: tuple[dict[str, Any], ...]
    objects: dict[str, Any]
    conditions: tuple[tuple[str, str, float], ...]
    waypoints: list[list[float]]
    routine: list[dict[str, Any]]
    py_source: str
    semantic_status: str
    compiler_routine_id: str
    future_dependency: dict[str, Any] | None
    training_eligible: bool
    deferred_reason: str | None
    description: str


def events_from_steps(steps: tuple[ProgramStep, ...]) -> tuple[dict[str, Any], ...]:
    events = []
    for step in steps:
        events.append({
            "step_id": step.step_id,
            "primitive": step.primitive,
            "object_role": step.object_role,
            "target_role": step.target_role,
            "aperture_role": step.aperture_role,
            "precondition": step.precondition,
            "success_predicate": step.success_predicate,
            "postcondition": step.postcondition,
            "articulation": step.articulation,
            "kind": step.kind,
        })
    return tuple(events)


def _routine_from_steps(steps: tuple[ProgramStep, ...]) -> list[dict[str, Any]]:
    grasped: str | None = None
    routine: list[dict[str, Any]] = []
    for step in steps:
        primitive = step.primitive
        if primitive == "grasp":
            routine.append({"type": "grasp", "obj": step.object_role, "grasp_z": 0.02})
            grasped = step.object_role
        elif primitive == "lift":
            op: dict[str, Any] = {"type": "lift", "obj": step.object_role, "grasp_z": 0.02, "lift_z": 0.88}
            if step.target_role:
                op["target"] = step.target_role
            routine.append(op)
            grasped = step.object_role
        elif primitive in {"place", "temporary_place", "fit"}:
            if grasped == step.object_role:
                routine.append({"type": "place", "obj": step.object_role, "target": step.target_role, "place_z": 0.03})
            else:
                routine.append({
                    "type": "pick_place",
                    "obj": step.object_role,
                    "target": step.target_role,
                    "grasp_z": 0.02,
                    "place_z": 0.03,
                })
            grasped = None
        elif primitive in {"retrieve", "park", "restore"}:
            routine.append({
                "type": "pick_place",
                "obj": step.object_role,
                "target": step.target_role,
                "grasp_z": 0.02,
                "place_z": 0.03,
            })
            grasped = None
        elif primitive == "open":
            routine.append({
                "type": "open_articulation",
                "obj": step.object_role,
                "target": step.target_role,
                "axis": "y" if step.object_role and "drawer" in step.object_role else "x",
            })
            grasped = None
        elif primitive == "close":
            routine.append({
                "type": "close_articulation",
                "obj": step.object_role,
                "target": step.target_role,
                "axis": "y" if step.object_role and "drawer" in step.object_role else "x",
            })
            grasped = None
        elif primitive == "push":
            routine.append({"type": "push", "obj": step.object_role, "target": step.target_role, "push_z": 0.02})
        elif primitive == "push_through_aperture":
            routine.append({
                "type": "push",
                "obj": step.object_role,
                "target": step.target_role,
                "push_z": 0.02,
                "via": step.aperture_role or step.target_role,
            })
        elif primitive == "reach":
            routine.append({"type": "reach", "target": step.target_role, "reach_z": 0.03})
        elif primitive == "rotate":
            routine.append({
                "type": "grasp_rotate",
                "obj": step.object_role,
                "grasp_z": -0.005,
                "lift_z": 0.15,
                "yaw_deg": 90.0,
            })
            grasped = step.object_role
        elif primitive == "regrasp":
            routine.append({"type": "grasp", "obj": step.object_role, "grasp_z": 0.02, "kind": "nominal"})
            grasped = step.object_role
            if step.target_role:
                routine.append({
                    "type": "place",
                    "obj": step.object_role,
                    "target": step.target_role,
                    "place_z": 0.0,
                    "kind": "nominal",
                })
                grasped = None
        elif primitive == "transport_through_aperture":
            op = {
                "type": "transport_through_aperture",
                "obj": step.object_role,
                "target": step.target_role,
                "grasp_z": 0.02,
                "place_z": 0.03,
            }
            if step.aperture_role:
                op["aperture_wp"] = step.aperture_role
            routine.append(op)
            grasped = None
        else:
            raise ValueError(f"unsupported primitive {primitive!r} at {step.step_id}")
    return routine


_TASK_HEADER = '''
import numpy as np
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition
from rlbench.backend.task import Task


class NearCondition(Condition):
    def __init__(self, obj, target, tolerance):
        self.obj = obj
        self.target = target
        self.tolerance = float(tolerance)

    def condition_met(self):
        distance = float(np.linalg.norm(self.obj.get_position() - self.target.get_position()))
        return distance <= self.tolerance, False


class ArticulationCondition(Condition):
    def __init__(self, handle, target, axis, tolerance):
        self.handle = handle
        self.target = target
        self.axis = {"x": 0, "y": 1, "z": 2}[axis]
        self.tolerance = float(tolerance)

    def condition_met(self):
        delta = self.handle.get_position() - self.target.get_position()
        return abs(float(delta[self.axis])) <= self.tolerance, False


class ProceduralTask(Task):
    OBJECTS = ()
    CONDITIONS = ()
    ARTICULATION = None
    WAYPOINT_COUNT = 0
    DESCRIPTION = ""

    def init_task(self):
        self._handles = {name: Object.get_object(name) for name in self.OBJECTS}
        graspable = [
            obj for name, obj in self._handles.items()
            if name.startswith("object") or name.startswith("blocker") or name.startswith("spacer") or "handle" in name
        ]
        if graspable:
            self.register_graspable_objects(graspable)
        conditions = []
        for item in self.CONDITIONS:
            kind = item[0] if item and item[0] in {"near", "articulation"} else "near"
            if kind == "articulation":
                _, handle, target, axis, tol = item
                conditions.append(ArticulationCondition(self._handles[handle], self._handles[target], axis, tol))
            else:
                if kind == "near":
                    _, obj_a, obj_b, tol = item
                else:
                    obj_a, obj_b, tol = item
                conditions.append(NearCondition(self._handles[obj_a], self._handles[obj_b], tol))
        self.register_success_conditions(conditions)

    def init_episode(self, index):
        self._variation_index = int(index)
        return [self.DESCRIPTION]

    def variation_count(self):
        return 1

    def is_static_workspace(self):
        return True
'''


def _generate_py_source(class_name: str, objects: tuple[str, ...], conditions: tuple, waypoint_count: int, description: str) -> str:
    return _TASK_HEADER + f'''

class {class_name}(ProceduralTask):
    OBJECTS = {objects!r}
    CONDITIONS = {conditions!r}
    WAYPOINT_COUNT = {waypoint_count}
    DESCRIPTION = {description!r}
'''


def _object_tuple(layout: dict[str, Any]) -> dict[str, tuple[list[float], list[float], list[float]]]:
    objects = {}
    for name, spec in layout["objects"].items():
        objects[name] = (list(spec["pos"]), list(spec["size"]), list(spec["color"]))
    return objects


def _conditions(layout: dict[str, Any], steps: tuple[ProgramStep, ...], tolerance: float) -> tuple:
    conditions: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for obj_a, obj_b, _tol in layout.get("conditions", ()):
        pair = (obj_a, obj_b)
        if pair not in seen:
            conditions.append(("near", obj_a, obj_b, float(tolerance)))
            seen.add(pair)
    articulation = layout.get("articulation")
    if articulation:
        handle = articulation["handle"]
        close_target = articulation.get("close_target")
        axis = articulation.get("axis", "y")
        last = steps[-1]
        if last.articulation == "closed" and close_target:
            conditions.append(("articulation", handle, close_target, axis, float(tolerance)))
    return tuple(conditions)


def compile_program(program_id: str, *, predicate_tolerance_m: float = 0.01) -> CompiledV3Task:
    spec = get_v3_program(program_id)
    layout = get_scene_layout(program_id)
    steps = spec.steps
    events = events_from_steps(steps)
    objects = _object_tuple(layout)
    conditions = _conditions(layout, steps, predicate_tolerance_m)
    waypoints = [list(item) for item in layout.get("waypoints", ())]
    routine = _routine_from_steps(steps)
    py_source = _generate_py_source(
        layout["class_name"],
        tuple(objects),
        conditions,
        len(waypoints),
        layout["description"],
    )
    return CompiledV3Task(
        program_id=program_id,
        class_name=layout["class_name"],
        module=layout["module"],
        split=spec.split,
        family=spec.family,
        ordered_steps=tuple(step.text for step in steps),
        events=events,
        objects=objects,
        conditions=tuple(
            (item[1], item[2], item[3]) if item[0] == "near" else (item[1], item[2], item[4])
            for item in conditions
        ),
        waypoints=waypoints,
        routine=routine,
        py_source=py_source,
        semantic_status=spec.semantic_status,
        compiler_routine_id=spec.compiler_routine_id,
        future_dependency=spec.future_dependency or layout.get("future_dependency"),
        training_eligible=spec.training_eligible,
        deferred_reason=spec.deferred_reason,
        description=layout["description"],
    )


def compile_v3_catalog(manifest_path: str | Path | None = None) -> dict[str, CompiledV3Task]:
    from icgs.data.collection.v3.steps import V3_PROGRAMS

    tolerances: dict[str, float] = {}
    if manifest_path is not None:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if data.get("composition_protocol_id") not in {None, V3_PROTOCOL.composition_protocol_id}:
            raise ValueError("v3 compiler requires composition protocol icgs-composition-primary-v2")
        for row in data.get("catalog", ()):
            predicates = row.get("predicates") or {}
            if "position_m" in predicates:
                tolerances[row["program_id"]] = float(predicates["position_m"])
    compiled = {}
    for program_id in V3_PROGRAMS:
        compiled[program_id] = compile_program(program_id, predicate_tolerance_m=tolerances.get(program_id, 0.01))
    return compiled
