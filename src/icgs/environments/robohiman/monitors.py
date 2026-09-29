"""Task monitors: raw simulator predicates plus ICGS event specs.

Every predicate is evaluated with an *upstream* condition class
(``DrawerCondition``, ``DetectedCondition``, ``GraspedCondition``) on live
simulator handles, or is one of the task's own registered success conditions.
Thresholds are copied from the upstream task that registers them and the
source is recorded. Event specs (which relation marks an event, its
prerequisites) are ICGS adapter decisions and are versioned here; they are
offline supervision only and never a model input.

Only tasks with a reviewed monitor are supported; others raise instead of
guessing from task names or language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


MONITOR_VERSION = "robohiman-monitor-v1"
_DRAWER_OPTIONS = ("bottom", "middle", "top")
# Thresholds registered by upstream task code at the pinned commit.
_OPEN_THRESHOLD = (0.15, "colosseum/rlbench/atomic_tasks/open_drawer.py: DrawerCondition(joint, 0.15, 'open')")
_CLOSE_THRESHOLD = (0.03, "colosseum/rlbench/compositional_tasks/put_in_and_close.py: DrawerCondition(joint, 0.03, 'close')")


@dataclass
class LiveMonitor:
    task: str
    predicate_names: list[str]
    predicate_sources: dict[str, str]
    event_specs: list[dict[str, Any]]
    _probes: list[Callable[[], bool]] = field(repr=False, default_factory=list)
    tracked_joints: list[Any] = field(repr=False, default_factory=list)

    def evaluate(self) -> np.ndarray:
        return np.array([bool(probe()) for probe in self._probes], dtype=bool)

    def describe(self) -> dict[str, Any]:
        return {
            "monitor_version": MONITOR_VERSION,
            "names": list(self.predicate_names),
            "sources": dict(self.predicate_sources),
        }


def _condition_probe(condition: Any) -> Callable[[], bool]:
    return lambda: bool(condition.condition_met()[0])


def _registered_success_probes(task_obj: Any) -> tuple[list[str], dict[str, str], list[Callable[[], bool]]]:
    names, sources, probes = [], {}, []
    for index, condition in enumerate(task_obj._success_conditions):
        kind = type(condition).__name__
        if kind == "ConditionSet" and getattr(condition, "_order_matters", False):
            raise RuntimeError("stateful ordered ConditionSet: per-step evaluation is not neutral")
        name = f"success_condition_{index}"
        names.append(name)
        sources[name] = f"task._success_conditions[{index}] ({kind}), registered by upstream init_episode"
        probes.append(_condition_probe(condition))
    return names, sources, probes


def _drawer_parts(task_obj: Any, variation: int) -> tuple[str, Any]:
    from pyrep.objects.joint import Joint

    option = _DRAWER_OPTIONS[variation % 3]
    return option, Joint(f"drawer_joint_{option}")


def build_monitor(task: str, task_env: Any, variation: int) -> LiveMonitor:
    """Construct a live monitor after ``task_env.reset`` for the given variation."""
    from colosseum.rlbench.extensions.conditions import DrawerCondition, GraspedCondition
    from pyrep.objects.proximity_sensor import ProximitySensor
    from pyrep.objects.shape import Shape
    from rlbench.backend.conditions import DetectedCondition

    task_obj = task_env._task
    scene = task_env._scene
    names, sources, probes = _registered_success_probes(task_obj)
    joints: list[Any] = []

    def add(name: str, source: str, condition: Any) -> None:
        names.append(name)
        sources[name] = source
        probes.append(_condition_probe(condition))

    if task in ("open_drawer", "close_drawer", "put_in_without_close", "put_in_and_close"):
        option, joint = _drawer_parts(task_obj, variation)
        joints.append(joint)
        add("drawer_open", f"DrawerCondition(drawer_joint_{option}, {_OPEN_THRESHOLD[0]}, 'open'); threshold from {_OPEN_THRESHOLD[1]}",
            DrawerCondition(joint, _OPEN_THRESHOLD[0], "open"))
        add("drawer_closed", f"DrawerCondition(drawer_joint_{option}, {_CLOSE_THRESHOLD[0]}, 'close'); threshold from {_CLOSE_THRESHOLD[1]}",
            DrawerCondition(joint, _CLOSE_THRESHOLD[0], "close"))
    if task in ("put_in_without_close", "put_in_and_close"):
        item = Shape("item")
        add("item_grasped", "GraspedCondition(robot.gripper, item): kinematic attachment list",
            GraspedCondition(scene.robot.gripper, item))
        add("item_in_drawer", f"DetectedCondition(item, ProximitySensor('success_{option}')): upstream success sensor",
            DetectedCondition(item, ProximitySensor(f"success_{option}")))

    if task == "open_drawer":
        events = [{"event_id": "drawer_opened", "relation": "drawer_open"}]
    elif task == "close_drawer":
        events = [{"event_id": "drawer_closed", "relation": "drawer_closed", "count_only_when_eligible": True,
                   "current_requirements": []}]
    elif task == "put_in_without_close":
        events = [
            {"event_id": "drawer_opened", "relation": "drawer_open"},
            {"event_id": "item_grasped", "relation": "item_grasped"},
            {"event_id": "item_placed_in_drawer", "relation": "item_in_drawer",
             "prerequisites": ["item_grasped"], "current_requirements": ["drawer_open"]},
        ]
    elif task == "put_in_and_close":
        events = [
            {"event_id": "drawer_opened", "relation": "drawer_open"},
            {"event_id": "item_grasped", "relation": "item_grasped"},
            {"event_id": "item_placed_in_drawer", "relation": "item_in_drawer",
             "prerequisites": ["item_grasped"], "current_requirements": ["drawer_open"]},
            {"event_id": "drawer_closed_after_placement", "relation": "drawer_closed",
             "prerequisites": ["item_placed_in_drawer"], "count_only_when_eligible": True},
        ]
    else:
        raise NotImplementedError(f"no reviewed ICGS monitor for RoboHiMan task {task!r}")
    for spec in events:
        spec.setdefault("prerequisites", [])
        spec.setdefault("current_requirements", [])
        spec.setdefault("hold_steps", 1)
        spec.setdefault("count_only_when_eligible", False)
    monitor = LiveMonitor(task, names, sources, events)
    monitor._probes = probes
    monitor.tracked_joints = joints
    return monitor


SUPPORTED_TASKS = ("open_drawer", "close_drawer", "put_in_without_close", "put_in_and_close")

__all__ = ["LiveMonitor", "MONITOR_VERSION", "SUPPORTED_TASKS", "build_monitor"]
