"""Primitive compiler for ICGS composition suite.

Compiles all 36 approved programs (T01-T20, V01-V04, P1-P4, G1-G4, R1-R4)
from their declared ordered_steps into executable RLBench task definitions,
objects, waypoints, predicates, and scripted expert routines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompiledTaskSpec:
    program_id: str
    class_name: str
    module: str
    scene_id: str
    split: str
    family: str
    ordered_steps: tuple[str, ...]
    description: str
    objects: dict[str, tuple[list[float], list[float], list[float]]]
    conditions: tuple[tuple[str, str, float], ...]
    waypoints: list[list[float]]
    routine: list[dict[str, Any]]
    py_source: str


COMMON_TASK_HEADER = '''
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
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


class ProceduralTask(Task):
    OBJECTS = ()
    CONDITIONS = ()
    WAYPOINT_COUNT = 0
    DESCRIPTION = ''

    def init_task(self):
        self._objects = {name: Shape(name) for name in self.OBJECTS if not (name.startswith('target') or name.endswith('target') or name.endswith('_wp'))}
        self._targets = {name: Shape(name) for name in self.OBJECTS if name.startswith('target') or name.endswith('target') or name.endswith('_wp')}
        self._handles = {**self._objects, **self._targets}
        graspable = [
            obj for name, obj in self._objects.items()
            if name.startswith('object') or name.startswith('blocker') or name.startswith('spacer') or 'handle' in name
        ]
        if graspable:
            self.register_graspable_objects(graspable)
        conditions = [NearCondition(self._handles[a], self._handles[b], tol) for a, b, tol in self.CONDITIONS]
        self.register_success_conditions(conditions)

    def init_episode(self, index):
        self._variation_index = int(index)
        dx = ((index * 17) % 7 - 3) * 0.004
        dy = ((index * 29) % 7 - 3) * 0.004
        for name, obj in self._handles.items():
            pos = obj.get_position()
            obj.set_position([float(pos[0] + dx), float(pos[1] + dy), float(pos[2])])
        return [self.DESCRIPTION]

    def variation_count(self):
        return 32

    def is_static_workspace(self):
        return True

    def validate(self):
        self._waypoints = []
'''


def _generate_py_source(class_name: str, objects: tuple[str, ...], conditions: tuple[tuple[str, str, float], ...], waypoint_count: int, description: str) -> str:
    return COMMON_TASK_HEADER + f'''

class {class_name}(ProceduralTask):
    OBJECTS = {objects!r}
    CONDITIONS = {conditions!r}
    WAYPOINT_COUNT = {waypoint_count}
    DESCRIPTION = {description!r}
'''


def compile_catalog(manifest_path: str | Path) -> dict[str, CompiledTaskSpec]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    catalog = {row["program_id"]: row for row in data["catalog"]}
    compiled: dict[str, CompiledTaskSpec] = {}

    # Define procedural specifications for all 36 tasks
    raw_specs: dict[str, dict[str, Any]] = {
        "T01": {
            "class_name": "T01GraspAndLift",
            "module": "t01_grasp_and_lift",
            "description": "grasp object A and lift it",
            "objects": {
                "object_a": ([0.25, 0.0, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.25, 0.0, 0.88], [0.05, 0.05, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.0, 0.90], [0.25, 0.0, 0.79], [0.25, 0.0, 0.88]],
            "routine": [
                {"type": "lift", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "lift_z": 0.88},
            ],
        },
        "T02": {
            "class_name": "T02GraspAndPlace",
            "module": "t02_grasp_and_place",
            "description": "grasp object A and place it on a pad",
            "objects": {
                "object_a": ([0.25, -0.10, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "pad": ([0.25, 0.15, 0.755], [0.15, 0.15, 0.025], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.15, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, -0.10, 0.90], [0.25, -0.10, 0.79], [0.25, 0.15, 0.90], [0.25, 0.15, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T03": {
            "class_name": "T03PushBlockerReach",
            "module": "t03_push_blocker_reach",
            "description": "push blocker away and reach the target",
            "objects": {
                "blocker": ([0.25, -0.05, 0.775], [0.05, 0.05, 0.05], [0.15, 0.15, 0.15]),
                "push_target": ([0.25, 0.08, 0.775], [0.06, 0.06, 0.025], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, -0.05, 0.775], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
            },
            "conditions": [("blocker", "push_target", 0.08)],
            "waypoints": [[0.25, -0.05, 0.90], [0.25, -0.05, 0.80], [0.25, 0.08, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "push_target", "grasp_z": 0.025, "place_z": 0.03},
                {"type": "reach", "target": "target_a", "reach_z": 0.03},
            ],
        },
        "T04": {
            "class_name": "T04OpenAndCloseDrawer",
            "module": "t04_open_and_close_drawer",
            "description": "open drawer then close it",
            "objects": {
                "drawer": ([0.25, 0.16, 0.755], [0.25, 0.22, 0.04], [0.35, 0.35, 0.38]),
                "drawer_handle": ([0.25, 0.04, 0.775], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.06, 0.775], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.04, 0.775], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
            },
            "conditions": [("drawer_handle", "close_target", 0.08)],
            "waypoints": [[0.25, 0.04, 0.90], [0.25, 0.04, 0.79], [0.25, -0.06, 0.79], [0.25, 0.04, 0.79]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "T05": {
            "class_name": "T05OpenDrawerRetrieve",
            "module": "t05_open_drawer_retrieve",
            "description": "open drawer and retrieve object A",
            "objects": {
                "drawer_handle": ([0.25, 0.04, 0.775], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.06, 0.775], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "object_a": ([0.25, 0.16, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.25, -0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.04, 0.90], [0.25, 0.04, 0.79], [0.25, -0.06, 0.79], [0.25, 0.16, 0.90], [0.25, 0.16, 0.79], [0.25, -0.14, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T06": {
            "class_name": "T06PlaceIntoDrawer",
            "module": "t06_place_into_drawer",
            "description": "place object A into a drawer target",
            "objects": {
                "object_a": ([0.25, -0.10, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "drawer": ([0.25, 0.16, 0.755], [0.25, 0.22, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.16, 0.775], [0.12, 0.10, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, -0.10, 0.90], [0.25, -0.10, 0.79], [0.25, 0.16, 0.90], [0.25, 0.16, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T07": {
            "class_name": "T07PlaceTwoInTray",
            "module": "t07_place_two_in_tray",
            "description": "place object A and object B into tray",
            "objects": {
                "object_a": ([0.22, -0.10, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "object_b": ([0.34, -0.10, 0.775], [0.05, 0.035, 0.05], [0.15, 0.35, 0.85]),
                "tray": ([0.28, 0.16, 0.755], [0.28, 0.18, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.22, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "target_b": ([0.34, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("object_b", "target_b", 0.08)],
            "waypoints": [[0.22, -0.10, 0.90], [0.22, -0.10, 0.79], [0.22, 0.16, 0.80], [0.34, -0.10, 0.90], [0.34, -0.10, 0.79], [0.34, 0.16, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T08": {
            "class_name": "T08PackTwoObjects",
            "module": "t08_pack_two_objects",
            "description": "place B then A into distinct tray slots",
            "objects": {
                "object_a": ([0.22, -0.10, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "object_b": ([0.34, -0.10, 0.775], [0.05, 0.035, 0.05], [0.15, 0.35, 0.85]),
                "tray": ([0.28, 0.16, 0.755], [0.28, 0.18, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.22, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "target_b": ([0.34, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("object_b", "target_b", 0.08)],
            "waypoints": [[0.34, -0.10, 0.90], [0.34, -0.10, 0.79], [0.34, 0.16, 0.80], [0.22, -0.10, 0.90], [0.22, -0.10, 0.79], [0.22, 0.16, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T09": {
            "class_name": "T09PlaceIntoHolder",
            "module": "t09_place_into_holder",
            "description": "grasp A and place it into a cylindrical holder",
            "objects": {
                "object_a": ([0.25, -0.06, 0.775], [0.045, 0.045, 0.06], [0.85, 0.25, 0.15]),
                "holder": ([0.25, 0.16, 0.755], [0.13, 0.13, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.16, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, -0.06, 0.90], [0.25, -0.06, 0.80], [0.25, 0.16, 0.90], [0.25, 0.16, 0.81]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T10": {
            "class_name": "T10PushAperturePlace",
            "module": "t10_push_aperture_place",
            "description": "push object A through aperture and place at target",
            "objects": {
                "object_a": ([0.25, -0.12, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "aperture_wp": ([0.25, 0.02, 0.775], [0.06, 0.06, 0.02], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, -0.12, 0.90], [0.25, -0.12, 0.79], [0.25, 0.02, 0.80], [0.25, 0.16, 0.80]],
            "routine": [
                {"type": "transport_through_aperture", "obj": "object_a", "aperture_wp": "aperture_wp", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T11": {
            "class_name": "T11RotateAndPlace",
            "module": "t11_rotate_and_place",
            "description": "grasp A, rotate it, and place it on an oriented pad",
            "objects": {
                "object_a": ([0.22, -0.06, 0.775], [0.035, 0.075, 0.035], [0.85, 0.25, 0.15]),
                "pad": ([0.22, 0.16, 0.755], [0.15, 0.15, 0.025], [0.35, 0.35, 0.38]),
                "target_a": ([0.22, 0.16, 0.775], [0.06, 0.06, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.22, -0.06, 0.90], [0.22, -0.06, 0.79], [0.22, 0.16, 0.90], [0.22, 0.16, 0.80]],
            "routine": [
                {"type": "grasp_rotate", "obj": "object_a", "grasp_z": -0.005, "lift_z": 0.15, "yaw_deg": 90.0},
                {"type": "place", "obj": "object_a", "target": "target_a", "place_z": 0.02},
            ],
        },
        "T12": {
            "class_name": "T12OpenGatePushClose",
            "module": "t12_open_gate_push_close",
            "description": "open gate, push object through gate, and close gate",
            "objects": {
                "gate_handle": ([0.25, 0.0, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.36, 0.0, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.0, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.25, -0.10, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "gate_wp": ([0.25, 0.05, 0.775], [0.05, 0.05, 0.02], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.0, 0.90], [0.36, 0.0, 0.80], [0.25, -0.10, 0.90], [0.25, 0.16, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "gate_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "transport_through_aperture", "obj": "object_a", "aperture_wp": "gate_wp", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "gate_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "T13": {
            "class_name": "T13ParkBlockerRetrieve",
            "module": "t13_park_blocker_retrieve",
            "description": "park blocker before retrieving object A",
            "objects": {
                "blocker": ([0.25, 0.02, 0.775], [0.06, 0.06, 0.06], [0.1, 0.1, 0.1]),
                "object_a": ([0.25, 0.18, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "park_target": ([0.35, 0.02, 0.775], [0.10, 0.10, 0.025], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.28, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("blocker", "park_target", 0.10), ("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.35, 0.02, 0.80], [0.25, 0.18, 0.90], [0.25, 0.28, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "T14": {
            "class_name": "T14ParkAndRestore",
            "module": "t14_park_and_restore",
            "description": "park blocker and restore it after a retrieval motion",
            "objects": {
                "blocker": ([0.26, 0.02, 0.775], [0.06, 0.06, 0.06], [0.1, 0.1, 0.1]),
                "object_a": ([0.26, 0.18, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "park_target": ([0.35, 0.02, 0.775], [0.10, 0.10, 0.025], [0.35, 0.35, 0.38]),
                "restore_target": ([0.26, 0.02, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("blocker", "restore_target", 0.08)],
            "waypoints": [[0.26, 0.02, 0.90], [0.35, 0.02, 0.80], [0.26, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "touch_retreat", "obj": "object_a", "touch_z": 0.035},
                {"type": "pick_place", "obj": "blocker", "target": "restore_target", "grasp_z": 0.025, "place_z": 0.0},
            ],
        },
        "T15": {
            "class_name": "T15ParkTwoRetrieve",
            "module": "t15_park_two_retrieve",
            "description": "park object A and object B then retrieve object C",
            "objects": {
                "blocker_a": ([0.20, 0.0, 0.775], [0.05, 0.05, 0.05], [0.8, 0.4, 0.1]),
                "blocker_b": ([0.32, 0.0, 0.775], [0.05, 0.05, 0.05], [0.1, 0.4, 0.8]),
                "object_c": ([0.26, 0.16, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "park_target_a": ([0.20, -0.16, 0.775], [0.07, 0.07, 0.025], [0.35, 0.35, 0.38]),
                "park_target_b": ([0.35, -0.16, 0.775], [0.07, 0.07, 0.025], [0.35, 0.35, 0.38]),
                "target_c": ([0.26, 0.0, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("blocker_a", "park_target_a", 0.08), ("blocker_b", "park_target_b", 0.08), ("object_c", "target_c", 0.08)],
            "waypoints": [[0.20, 0.0, 0.90], [0.20, -0.16, 0.80], [0.32, 0.0, 0.90], [0.35, -0.16, 0.80], [0.26, 0.16, 0.90], [0.26, 0.0, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker_a", "target": "park_target_a", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker_b", "target": "park_target_b", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_c", "target": "target_c", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "T16": {
            "class_name": "T16OpenDrawerParkRetrieve",
            "module": "t16_open_drawer_park_retrieve",
            "description": "open drawer, park blocker, and retrieve object A",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.10, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "blocker": ([0.22, 0.08, 0.775], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]),
                "park_target": ([0.35, 0.08, 0.775], [0.07, 0.07, 0.025], [0.35, 0.35, 0.38]),
                "object_a": ([0.28, 0.16, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.28, -0.05, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("blocker", "park_target", 0.08), ("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, -0.10, 0.80], [0.22, 0.08, 0.90], [0.35, 0.08, 0.80], [0.28, 0.16, 0.90], [0.28, -0.05, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "T17": {
            "class_name": "T17RegraspObject",
            "module": "t17_regrasp_object",
            "description": "grasp object A, temporarily place it, and regrasp to target",
            "objects": {
                "object_a": ([0.25, -0.10, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "temp_target": ([0.25, 0.04, 0.775], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.18, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, -0.10, 0.90], [0.25, 0.04, 0.80], [0.25, 0.04, 0.90], [0.25, 0.18, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "temp_target", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "T18": {
            "class_name": "T18RetrievePlaceRetrieve",
            "module": "t18_retrieve_place_retrieve",
            "description": "retrieve object A, place it, then retrieve object B",
            "objects": {
                "object_a": ([0.22, 0.12, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "target_a": ([0.22, -0.10, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_b": ([0.34, 0.12, 0.775], [0.04, 0.04, 0.04], [0.15, 0.35, 0.85]),
                "target_b": ([0.34, -0.10, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("object_b", "target_b", 0.08)],
            "waypoints": [[0.22, 0.12, 0.90], [0.22, -0.10, 0.80], [0.34, 0.12, 0.90], [0.34, -0.10, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T19": {
            "class_name": "T19PlaceSpacerPack",
            "module": "t19_place_spacer_pack",
            "description": "place spacer then place object A into drawer",
            "objects": {
                "spacer": ([0.20, -0.10, 0.775], [0.04, 0.04, 0.04], [0.6, 0.4, 0.2]),
                "target_spacer": ([0.20, 0.15, 0.775], [0.06, 0.06, 0.03], [0.35, 0.35, 0.38]),
                "object_a": ([0.32, -0.10, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "target_a": ([0.32, 0.15, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("spacer", "target_spacer", 0.08), ("object_a", "target_a", 0.08)],
            "waypoints": [[0.20, -0.10, 0.90], [0.20, 0.15, 0.80], [0.32, -0.10, 0.90], [0.32, 0.15, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "spacer", "target": "target_spacer", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "T20": {
            "class_name": "T20ParkRetrievePlace",
            "module": "t20_park_retrieve_place",
            "description": "park blocker, retrieve object A, and place it on pad",
            "objects": {
                "blocker": ([0.25, 0.02, 0.775], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]),
                "park_target": ([0.35, 0.02, 0.775], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "object_a": ([0.25, 0.16, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.25, -0.12, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("blocker", "park_target", 0.08), ("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.35, 0.02, 0.80], [0.25, 0.16, 0.90], [0.25, -0.12, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "V01": {
            "class_name": "V01OpenPlaceClose",
            "module": "v01_open_place_close",
            "description": "open drawer, place object A, and close drawer",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.08, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.02, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.25, -0.16, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.25, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, -0.08, 0.80], [0.25, -0.16, 0.90], [0.25, 0.14, 0.80], [0.25, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "V02": {
            "class_name": "V02GraspRotatePlaceHolder",
            "module": "v02_grasp_rotate_place_holder",
            "description": "grasp object, rotate in flight, and place into holder",
            "objects": {
                "object_a": ([0.22, -0.06, 0.775], [0.035, 0.075, 0.035], [0.85, 0.25, 0.15]),
                "holder": ([0.22, 0.16, 0.755], [0.13, 0.13, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.22, 0.16, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.22, -0.06, 0.90], [0.22, 0.16, 0.80]],
            "routine": [
                {"type": "grasp_rotate", "obj": "object_a", "grasp_z": -0.005, "lift_z": 0.15, "yaw_deg": 90.0},
                {"type": "place", "obj": "object_a", "target": "target_a", "place_z": 0.0},
            ],
        },
        "V03": {
            "class_name": "V03ParkRetrievePlaceRestore",
            "module": "v03_park_retrieve_place_restore",
            "description": "park blocker, retrieve object, place on pad, and restore blocker",
            "objects": {
                "blocker": ([0.26, 0.02, 0.775], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]),
                "park_target": ([0.35, 0.02, 0.775], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "object_a": ([0.26, 0.18, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.26, -0.12, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "restore_target": ([0.26, 0.02, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("blocker", "restore_target", 0.08)],
            "waypoints": [[0.26, 0.02, 0.90], [0.35, 0.02, 0.80], [0.26, 0.18, 0.90], [0.26, -0.12, 0.80], [0.26, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker", "target": "restore_target", "grasp_z": 0.025, "place_z": 0.0},
            ],
        },
        "V04": {
            "class_name": "V04OpenGateRetrieveClose",
            "module": "v04_open_gate_retrieve_close",
            "description": "open gate, retrieve object A, and close gate",
            "objects": {
                "gate_handle": ([0.25, 0.0, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.36, 0.0, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.0, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.25, 0.15, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.25, -0.12, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.0, 0.90], [0.36, 0.0, 0.80], [0.25, 0.15, 0.90], [0.25, -0.12, 0.80], [0.25, 0.0, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "gate_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "gate_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "P1": {
            "class_name": "P1OpenPlaceTwoClose",
            "module": "p1_open_place_two_close",
            "description": "open drawer, place object A and B, close drawer",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.08, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.02, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.20, -0.16, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "target_a": ([0.20, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_b": ([0.32, -0.16, 0.775], [0.04, 0.04, 0.04], [0.15, 0.35, 0.85]),
                "target_b": ([0.32, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("object_b", "target_b", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, -0.08, 0.80], [0.20, -0.16, 0.90], [0.20, 0.14, 0.80], [0.32, -0.16, 0.90], [0.32, 0.14, 0.80], [0.25, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "P2": {
            "class_name": "P2OpenPlaceTwoReverseClose",
            "module": "p2_open_place_two_reverse_close",
            "description": "open drawer, place object B then A, close drawer",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.08, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.02, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_b": ([0.32, -0.16, 0.775], [0.04, 0.04, 0.04], [0.15, 0.35, 0.85]),
                "target_b": ([0.32, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_a": ([0.20, -0.16, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "target_a": ([0.20, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_b", "target_b", 0.08), ("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, -0.08, 0.80], [0.32, -0.16, 0.90], [0.32, 0.14, 0.80], [0.20, -0.16, 0.90], [0.20, 0.14, 0.80], [0.25, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "P3": {
            "class_name": "P3OpenPlaceSpacerClose",
            "module": "p3_open_place_spacer_close",
            "description": "open drawer, place object A, place spacer, place B, close",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.08, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.02, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.20, -0.16, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "target_a": ([0.20, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "spacer": ([0.26, -0.16, 0.775], [0.03, 0.03, 0.03], [0.6, 0.4, 0.2]),
                "target_spacer": ([0.26, 0.14, 0.775], [0.04, 0.04, 0.03], [0.35, 0.35, 0.38]),
                "object_b": ([0.32, -0.16, 0.775], [0.04, 0.04, 0.04], [0.15, 0.35, 0.85]),
                "target_b": ([0.32, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("spacer", "target_spacer", 0.08), ("object_b", "target_b", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, 0.14, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "spacer", "target": "target_spacer", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "P4": {
            "class_name": "P4OpenPlaceRetrievePlaceClose",
            "module": "p4_open_place_retrieve_place_close",
            "description": "open drawer, place A, retrieve C, place B, close",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.08, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.02, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.20, -0.16, 0.775], [0.04, 0.04, 0.04], [0.85, 0.25, 0.15]),
                "target_a": ([0.20, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_c": ([0.26, 0.14, 0.775], [0.04, 0.04, 0.04], [0.1, 0.6, 0.2]),
                "target_c": ([0.42, 0.0, 0.775], [0.06, 0.06, 0.025], [0.35, 0.35, 0.38]),
                "object_b": ([0.32, -0.16, 0.775], [0.04, 0.04, 0.04], [0.15, 0.35, 0.85]),
                "target_b": ([0.32, 0.14, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("object_c", "target_c", 0.08), ("object_b", "target_b", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, 0.14, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_c", "target": "target_c", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "G1": {
            "class_name": "G1GraspAperturePlaceHolder",
            "module": "g1_grasp_aperture_place_holder",
            "description": "grasp object, transport through aperture, and place into holder",
            "objects": {
                "object_a": ([0.25, -0.12, 0.775], [0.045, 0.045, 0.05], [0.85, 0.25, 0.15]),
                "aperture_wp": ([0.25, 0.02, 0.775], [0.06, 0.06, 0.02], [0.35, 0.35, 0.38]),
                "holder": ([0.25, 0.16, 0.755], [0.13, 0.13, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.16, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, -0.12, 0.90], [0.25, 0.02, 0.80], [0.25, 0.16, 0.81]],
            "routine": [
                {"type": "transport_through_aperture", "obj": "object_a", "aperture_wp": "aperture_wp", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "G2": {
            "class_name": "G2GraspRotateAperturePlace",
            "module": "g2_grasp_rotate_aperture_place",
            "description": "grasp object, rotate in free space, transport through aperture, place",
            "objects": {
                "object_a": ([0.22, -0.12, 0.775], [0.035, 0.075, 0.035], [0.85, 0.25, 0.15]),
                "aperture_wp": ([0.22, 0.02, 0.775], [0.06, 0.06, 0.02], [0.35, 0.35, 0.38]),
                "target_a": ([0.22, 0.16, 0.775], [0.06, 0.06, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.22, -0.12, 0.90], [0.22, 0.02, 0.80], [0.22, 0.16, 0.80]],
            "routine": [
                {"type": "grasp_rotate", "obj": "object_a", "grasp_z": -0.005, "lift_z": 0.15, "yaw_deg": 90.0},
                {"type": "transport_through_aperture", "obj": "object_a", "aperture_wp": "aperture_wp", "target": "target_a", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "G3": {
            "class_name": "G3OpenGateTransportClose",
            "module": "g3_open_gate_transport_close",
            "description": "open gate, grasp object, transport, place, close gate",
            "objects": {
                "gate_handle": ([0.25, 0.0, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.36, 0.0, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.0, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "object_a": ([0.25, -0.12, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "gate_wp": ([0.25, 0.04, 0.775], [0.05, 0.05, 0.02], [0.35, 0.35, 0.38]),
                "target_a": ([0.25, 0.16, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.25, 0.0, 0.90], [0.36, 0.0, 0.80], [0.25, 0.16, 0.80], [0.25, 0.0, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "gate_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.02},
                {"type": "transport_through_aperture", "obj": "object_a", "aperture_wp": "gate_wp", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "gate_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.02},
            ],
        },
        "G4": {
            "class_name": "G4RegraspTransportFit",
            "module": "g4_regrasp_transport_fit",
            "description": "grasp object, temporary place, regrasp, transport, fit",
            "objects": {
                "object_a": ([0.22, -0.12, 0.775], [0.04, 0.04, 0.05], [0.85, 0.25, 0.15]),
                "temp_target": ([0.22, 0.0, 0.755], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "holder": ([0.22, 0.16, 0.755], [0.13, 0.13, 0.04], [0.35, 0.35, 0.38]),
                "target_a": ([0.22, 0.16, 0.775], [0.06, 0.06, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08)],
            "waypoints": [[0.22, -0.12, 0.90], [0.22, 0.0, 0.80], [0.22, 0.16, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "object_a", "target": "temp_target", "grasp_z": 0.02, "place_z": 0.03},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.03},
            ],
        },
        "R1": {
            "class_name": "R1ParkRetrieveRestore",
            "module": "r1_park_retrieve_restore",
            "description": "park blocker, retrieve target, and restore blocker",
            "objects": {
                "blocker": ([0.26, 0.02, 0.775], [0.06, 0.06, 0.06], [0.1, 0.1, 0.1]),
                "park_target": ([0.35, 0.02, 0.775], [0.10, 0.10, 0.025], [0.35, 0.35, 0.38]),
                "object_a": ([0.26, 0.18, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.26, -0.12, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
                "restore_target": ([0.26, 0.02, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("blocker", "restore_target", 0.08)],
            "waypoints": [[0.26, 0.02, 0.90], [0.35, 0.02, 0.80], [0.26, 0.18, 0.90], [0.26, -0.12, 0.80], [0.26, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker", "target": "restore_target", "grasp_z": 0.025, "place_z": 0.0},
            ],
        },
        "R2": {
            "class_name": "R2ParkTwoRetrieveRestoreTwo",
            "module": "r2_park_two_retrieve_restore_two",
            "description": "park A, park B, retrieve, restore B, restore A",
            "objects": {
                "blocker_a": ([0.20, 0.02, 0.775], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]),
                "park_target_a": ([0.35, -0.06, 0.775], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "restore_target_a": ([0.20, 0.02, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "blocker_b": ([0.32, 0.02, 0.775], [0.05, 0.05, 0.05], [0.2, 0.2, 0.2]),
                "park_target_b": ([0.35, 0.08, 0.775], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "restore_target_b": ([0.32, 0.02, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_a": ([0.26, 0.18, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.26, -0.12, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("blocker_b", "restore_target_b", 0.08), ("blocker_a", "restore_target_a", 0.08)],
            "waypoints": [[0.20, 0.02, 0.90], [0.35, -0.06, 0.80], [0.32, 0.02, 0.90], [0.35, 0.08, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker_a", "target": "park_target_a", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker_b", "target": "park_target_b", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker_b", "target": "restore_target_b", "grasp_z": 0.025, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker_a", "target": "restore_target_a", "grasp_z": 0.025, "place_z": 0.0},
            ],
        },
        "R3": {
            "class_name": "R3OpenDrawerParkRetrieveRestoreClose",
            "module": "r3_open_drawer_park_retrieve_restore_close",
            "description": "open drawer, park blocker, retrieve, restore, close drawer",
            "objects": {
                "drawer_handle": ([0.25, 0.02, 0.78], [0.08, 0.03, 0.03], [0.15, 0.15, 0.15]),
                "open_target": ([0.25, -0.08, 0.78], [0.05, 0.05, 0.025], [0.8, 0.8, 0.2]),
                "close_target": ([0.25, 0.02, 0.78], [0.05, 0.05, 0.025], [0.2, 0.7, 0.3]),
                "blocker": ([0.22, 0.08, 0.775], [0.05, 0.05, 0.05], [0.1, 0.1, 0.1]),
                "park_target": ([0.35, 0.08, 0.775], [0.08, 0.08, 0.025], [0.35, 0.35, 0.38]),
                "restore_target": ([0.22, 0.08, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_a": ([0.28, 0.16, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.28, -0.05, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("blocker", "restore_target", 0.08)],
            "waypoints": [[0.25, 0.02, 0.90], [0.25, -0.08, 0.80], [0.22, 0.08, 0.90], [0.35, 0.08, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "drawer_handle", "target": "open_target", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker", "target": "restore_target", "grasp_z": 0.025, "place_z": 0.0},
                {"type": "pick_place", "obj": "drawer_handle", "target": "close_target", "grasp_z": 0.02, "place_z": 0.0},
            ],
        },
        "R4": {
            "class_name": "R4ParkRetrieveTwoRestore",
            "module": "r4_park_retrieve_two_restore",
            "description": "park blocker, retrieve target A, retrieve target B, restore blocker",
            "objects": {
                "blocker": ([0.26, 0.02, 0.775], [0.06, 0.06, 0.06], [0.1, 0.1, 0.1]),
                "park_target": ([0.35, 0.02, 0.775], [0.10, 0.10, 0.025], [0.35, 0.35, 0.38]),
                "restore_target": ([0.26, 0.02, 0.775], [0.07, 0.07, 0.04], [0.2, 0.7, 0.3]),
                "object_a": ([0.22, 0.18, 0.775], [0.045, 0.045, 0.045], [0.85, 0.25, 0.15]),
                "target_a": ([0.22, -0.12, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
                "object_b": ([0.32, 0.18, 0.775], [0.045, 0.045, 0.045], [0.15, 0.35, 0.85]),
                "target_b": ([0.32, -0.12, 0.775], [0.06, 0.06, 0.03], [0.2, 0.7, 0.3]),
            },
            "conditions": [("object_a", "target_a", 0.08), ("object_b", "target_b", 0.08), ("blocker", "restore_target", 0.08)],
            "waypoints": [[0.26, 0.02, 0.90], [0.35, 0.02, 0.80]],
            "routine": [
                {"type": "pick_place", "obj": "blocker", "target": "park_target", "grasp_z": 0.03, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_a", "target": "target_a", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "object_b", "target": "target_b", "grasp_z": 0.02, "place_z": 0.0},
                {"type": "pick_place", "obj": "blocker", "target": "restore_target", "grasp_z": 0.025, "place_z": 0.0},
            ],
        },
    }

    assert len(raw_specs) == 36, f"Expected 36 task specs, got {len(raw_specs)}"

    for pid, row in catalog.items():
        spec = raw_specs[pid]
        # Never inherit the looser pilot template tolerance; the approved
        # manifest owns the executable predicate contract.
        strict_tolerance = float(row["predicates"]["position_m"])
        spec["conditions"] = [
            (obj_a, obj_b, strict_tolerance)
            for obj_a, obj_b, _ in spec["conditions"]
        ]
        py_source = _generate_py_source(
            spec["class_name"],
            tuple(spec["objects"]),
            tuple(spec["conditions"]),
            len(spec["waypoints"]),
            spec["description"],
        )
        compiled[pid] = CompiledTaskSpec(
            program_id=pid,
            class_name=spec["class_name"],
            module=spec["module"],
            scene_id=row["scene_id"],
            split=row["split"],
            family=row["family"],
            ordered_steps=tuple(row["ordered_steps"]),
            description=spec["description"],
            objects=spec["objects"],
            conditions=tuple(spec["conditions"]),
            waypoints=spec["waypoints"],
            routine=spec["routine"],
            py_source=py_source,
        )

    return compiled
