"""Task monitors: raw simulator predicates plus ICGS event specs.

Two layers, kept separate on purpose:

* **Generic** (this module's functions): predicate *kinds* built only from
  upstream condition classes on live simulator handles (``DetectedCondition``,
  ``GraspedCondition``, colosseum ``DrawerCondition``, a count over
  ``DetectedCondition``), probes of the task's own registered success
  conditions, template resolution of object names from the task instance, and
  label derivation (``icgs.data.stage1.labels``).
* **Task-specific** (``TASK_SPECS``): which objects/sensors/joints a task uses,
  which upstream file a threshold comes from, and the event specs (relation,
  prerequisites, current requirements). These are reviewed ICGS adapter
  decisions, versioned by ``MONITOR_VERSION``, offline supervision only.

Tasks without a reviewed spec raise instead of guessing from names or language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


MONITOR_VERSION = "robohiman-monitor-v2"
_DRAWER_OPTIONS = ("bottom", "middle", "top")
# Thresholds registered by upstream task code at the pinned commit.
_OPEN = (0.15, "colosseum/rlbench/atomic_tasks/open_drawer.py: DrawerCondition(joint, 0.15, 'open')")
_CLOSE = (0.03, "colosseum/rlbench/compositional_tasks/put_in_and_close.py: DrawerCondition(joint, 0.03, 'close')")


@dataclass
class LiveMonitor:
    task: str
    predicate_names: list[str]
    predicate_sources: dict[str, str]
    event_specs: list[dict[str, Any]]
    family: str = ""
    _probes: list[Callable[[], bool]] = field(repr=False, default_factory=list)
    tracked_joints: list[Any] = field(repr=False, default_factory=list)

    def evaluate(self) -> np.ndarray:
        return np.array([bool(probe()) for probe in self._probes], dtype=bool)

    def describe(self) -> dict[str, Any]:
        return {
            "monitor_version": MONITOR_VERSION,
            "semantic_family": self.family,
            "names": list(self.predicate_names),
            "sources": dict(self.predicate_sources),
            "generic_layer": ["upstream condition classes", "registered success-condition probes",
                              "template resolution", "icgs.data.stage1.labels derivation"],
            "task_specific_layer": f"TASK_SPECS[{self.task!r}] (objects, sensors, thresholds, events)",
        }


# --------------------------------------------------------------------- generic

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


def _template_values(task: str, task_obj: Any, variation: int) -> dict[str, str]:
    """Names the upstream task binds to a variation index (read from the task instance)."""
    values: dict[str, str] = {}
    if hasattr(task_obj, "_options"):
        values["option"] = _DRAWER_OPTIONS[variation % 3]
    groceries = getattr(task_obj, "groceries", None)
    if groceries:
        values["grocery"] = groceries[variation % len(groceries)].get_name()
    return values


def _build_predicate(kind: str, params: dict[str, Any], scene: Any) -> tuple[Any, str]:
    """Return (condition-like object with condition_met, source text) for one kind."""
    from colosseum.rlbench.extensions.conditions import DrawerCondition, GraspedCondition
    from pyrep.objects.joint import Joint
    from pyrep.objects.proximity_sensor import ProximitySensor
    from pyrep.objects.shape import Shape
    from rlbench.backend.conditions import DetectedCondition

    if kind == "detected":
        condition = DetectedCondition(Shape(params["object"]), ProximitySensor(params["sensor"]))
        return condition, f"DetectedCondition({params['object']}, ProximitySensor('{params['sensor']}'))"
    if kind == "grasped":
        condition = GraspedCondition(scene.robot.gripper, Shape(params["object"]))
        return condition, f"GraspedCondition(gripper, {params['object']}): kinematic attachment list"
    if kind == "grasped_any":
        conditions = [GraspedCondition(scene.robot.gripper, Shape(name)) for name in params["objects"]]

        class _Any:
            def condition_met(self):
                return any(c.condition_met()[0] for c in conditions), False
        return _Any(), f"any GraspedCondition over {params['objects']}"
    if kind == "detected_count_ge":
        sensor = ProximitySensor(params["sensor"])
        conditions = [DetectedCondition(Shape(name), sensor) for name in params["objects"]]
        need = int(params["k"])

        class _Count:
            def condition_met(self):
                return sum(c.condition_met()[0] for c in conditions) >= need, False
        return _Count(), f"count(DetectedCondition({params['objects']}, '{params['sensor']}')) >= {need}"
    if kind in ("drawer_open", "drawer_closed"):
        threshold, origin = _OPEN if kind == "drawer_open" else _CLOSE
        mode = "open" if kind == "drawer_open" else "close"
        joint = Joint(params["joint"])
        return DrawerCondition(joint, threshold, mode), f"DrawerCondition({params['joint']}, {threshold}, '{mode}'); from {origin}"
    raise ValueError(f"unknown predicate kind {kind!r}")


def _resolve(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**values)
    if isinstance(value, list):
        return [_resolve(item, values) for item in value]
    return value


# --------------------------------------------------------------- task-specific

_DRAWER = [("drawer_open", "drawer_open", {"joint": "drawer_joint_{option}"}),
           ("drawer_closed", "drawer_closed", {"joint": "drawer_joint_{option}"})]
_DIRT = [f"dirt{i}" for i in range(5)]

TASK_SPECS: dict[str, dict[str, Any]] = {
    "open_drawer": {"family": "articulated_drawer", "predicates": _DRAWER, "events": [
        {"event_id": "drawer_opened", "relation": "drawer_open"}]},
    "close_drawer": {"family": "articulated_drawer", "predicates": _DRAWER, "events": [
        {"event_id": "drawer_closed", "relation": "drawer_closed", "count_only_when_eligible": True}]},
    "put_in_without_close": {"family": "articulated_drawer", "predicates": _DRAWER + [
        ("item_grasped", "grasped", {"object": "item"}),
        ("item_in_drawer", "detected", {"object": "item", "sensor": "success_{option}"})], "events": [
        {"event_id": "drawer_opened", "relation": "drawer_open"},
        {"event_id": "item_grasped", "relation": "item_grasped"},
        {"event_id": "item_placed_in_drawer", "relation": "item_in_drawer",
         "prerequisites": ["item_grasped"], "current_requirements": ["drawer_open"]}]},
    "put_in_and_close": {"family": "articulated_drawer", "predicates": _DRAWER + [
        ("item_grasped", "grasped", {"object": "item"}),
        ("item_in_drawer", "detected", {"object": "item", "sensor": "success_{option}"})], "events": [
        {"event_id": "drawer_opened", "relation": "drawer_open"},
        {"event_id": "item_grasped", "relation": "item_grasped"},
        {"event_id": "item_placed_in_drawer", "relation": "item_in_drawer",
         "prerequisites": ["item_grasped"], "current_requirements": ["drawer_open"]},
        {"event_id": "drawer_closed_after_placement", "relation": "drawer_closed",
         "prerequisites": ["item_placed_in_drawer"], "count_only_when_eligible": True}]},
    "put_two_in_same": {"family": "articulated_drawer", "predicates": _DRAWER + [
        ("block_grasped", "grasped_any", {"objects": ["item0", "item1"]}),
        ("blocks_in_drawer_ge1", "detected_count_ge", {"objects": ["item0", "item1"], "sensor": "success_{option}", "k": 1}),
        ("blocks_in_drawer_ge2", "detected_count_ge", {"objects": ["item0", "item1"], "sensor": "success_{option}", "k": 2})],
        # Order-invariant events: which block goes first is the expert's choice.
        "events": [
            {"event_id": "drawer_opened", "relation": "drawer_open"},
            {"event_id": "first_block_placed", "relation": "blocks_in_drawer_ge1",
             "current_requirements": ["drawer_open"]},
            {"event_id": "second_block_placed", "relation": "blocks_in_drawer_ge2",
             "prerequisites": ["first_block_placed"], "current_requirements": ["drawer_open"]}]},
    "box_in_cupboard": {"family": "container_pick_place", "predicates": [
        ("target_grasped", "grasped", {"object": "{grocery}"}),
        ("target_in_cupboard", "detected", {"object": "{grocery}", "sensor": "success"})], "events": [
        {"event_id": "target_grasped", "relation": "target_grasped"},
        {"event_id": "target_placed_in_cupboard", "relation": "target_in_cupboard",
         "prerequisites": ["target_grasped"], "count_only_when_eligible": True}]},
    "box_exchange": {"family": "container_pick_place", "predicates": [
        ("sugar_grasped", "grasped", {"object": "sugar"}),
        ("sugar_on_table", "detected", {"object": "sugar", "sensor": "success_ground"}),
        ("sugar_in_cupboard", "detected", {"object": "sugar", "sensor": "success_cupboard"}),
        ("spam_grasped", "grasped", {"object": "spam"}),
        ("spam_in_cupboard", "detected", {"object": "spam", "sensor": "success_cupboard"}),
        ("spam_on_table", "detected", {"object": "spam", "sensor": "success_ground"})], "events": [
        {"event_id": "sugar_grasped", "relation": "sugar_grasped"},
        {"event_id": "sugar_placed_on_table", "relation": "sugar_on_table",
         "prerequisites": ["sugar_grasped"], "count_only_when_eligible": True},
        {"event_id": "spam_grasped", "relation": "spam_grasped"},
        {"event_id": "spam_placed_in_cupboard", "relation": "spam_in_cupboard",
         "prerequisites": ["spam_grasped"], "count_only_when_eligible": True}]},
    "rubbish_in_dustpan": {"family": "dustpan_tool_use", "predicates": [
        ("rubbish_grasped", "grasped", {"object": "rubbish"}),
        ("rubbish_in_dustpan", "detected", {"object": "rubbish", "sensor": "success"})], "events": [
        {"event_id": "rubbish_grasped", "relation": "rubbish_grasped"},
        {"event_id": "rubbish_dropped_in_dustpan", "relation": "rubbish_in_dustpan",
         "prerequisites": ["rubbish_grasped"], "count_only_when_eligible": True}]},
    "sweep_to_dustpan": {"family": "dustpan_tool_use", "predicates": [
        ("broom_grasped", "grasped", {"object": "broom"}),
        ("dirt_in_dustpan_ge1", "detected_count_ge", {"objects": _DIRT, "sensor": "success", "k": 1}),
        ("dirt_in_dustpan_all", "detected_count_ge", {"objects": _DIRT, "sensor": "success", "k": 5})], "events": [
        {"event_id": "broom_grasped", "relation": "broom_grasped"},
        {"event_id": "all_dirt_swept", "relation": "dirt_in_dustpan_all", "prerequisites": ["broom_grasped"],
         "count_only_when_eligible": True}]},
    "sweep_and_drop": {"family": "dustpan_tool_use", "predicates": [
        ("rubbish_grasped", "grasped", {"object": "rubbish"}),
        ("rubbish_in_dustpan", "detected", {"object": "rubbish", "sensor": "success"}),
        ("broom_grasped", "grasped", {"object": "broom"}),
        ("dirt_in_dustpan_ge1", "detected_count_ge", {"objects": _DIRT, "sensor": "success", "k": 1}),
        ("dirt_in_dustpan_all", "detected_count_ge", {"objects": _DIRT, "sensor": "success", "k": 5})], "events": [
        {"event_id": "rubbish_grasped", "relation": "rubbish_grasped"},
        {"event_id": "rubbish_dropped_in_dustpan", "relation": "rubbish_in_dustpan",
         "prerequisites": ["rubbish_grasped"], "count_only_when_eligible": True},
        {"event_id": "broom_grasped", "relation": "broom_grasped"},
        {"event_id": "all_dirt_swept", "relation": "dirt_in_dustpan_all", "prerequisites": ["broom_grasped"],
         "count_only_when_eligible": True}]},
}
SUPPORTED_TASKS = tuple(TASK_SPECS)


def build_monitor(task: str, task_env: Any, variation: int) -> LiveMonitor:
    """Construct a live monitor after ``task_env.reset`` for the given variation."""
    if task not in TASK_SPECS:
        raise NotImplementedError(f"no reviewed ICGS monitor for RoboHiMan task {task!r}")
    from pyrep.objects.joint import Joint

    spec = TASK_SPECS[task]
    task_obj = task_env._task
    scene = task_env._scene
    values = _template_values(task, task_obj, variation)
    names, sources, probes = _registered_success_probes(task_obj)
    joints: list[Any] = []
    for name, kind, params in spec["predicates"]:
        resolved = {key: _resolve(value, values) for key, value in params.items()}
        condition, source = _build_predicate(kind, resolved, scene)
        names.append(name)
        sources[name] = source
        probes.append(_condition_probe(condition))
        if kind in ("drawer_open", "drawer_closed") and not joints:
            joints.append(Joint(resolved["joint"]))
    events = [dict(event) for event in spec["events"]]
    for event in events:
        event.setdefault("prerequisites", [])
        event.setdefault("current_requirements", [])
        event.setdefault("hold_steps", 1)
        event.setdefault("count_only_when_eligible", False)
    monitor = LiveMonitor(task, names, sources, events, family=spec["family"])
    monitor._probes = probes
    monitor.tracked_joints = joints
    return monitor


__all__ = ["LiveMonitor", "MONITOR_VERSION", "SUPPORTED_TASKS", "TASK_SPECS", "build_monitor"]
