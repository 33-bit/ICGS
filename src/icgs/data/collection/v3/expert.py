"""Scripted expert motions for v3 routine types.

``plan_step`` is simulator-free.  ``execute_step`` runs the same motions through
injected IK/gripper callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


ROUTINE_TYPES = frozenset({
    "grasp",
    "lift",
    "place",
    "pick_place",
    "push",
    "reach",
    "open_articulation",
    "close_articulation",
    "grasp_rotate",
    "transport_through_aperture",
    "pause_hold",
})

GRASP_TYPES = frozenset({"grasp", "lift", "place", "pick_place", "grasp_rotate", "transport_through_aperture"})
CONTACT_PUSH_TYPES = frozenset({"push"})
ARTICULATION_TYPES = frozenset({"open_articulation", "close_articulation"})


def assert_routine_supported(routine: Sequence[Mapping[str, Any]]) -> None:
    for index, step in enumerate(routine):
        stype = step.get("type")
        if stype not in ROUTINE_TYPES:
            raise ValueError(f"unsupported routine type at step {index}: {stype!r}")


def plan_step(step: Mapping[str, Any], poses: Mapping[str, Sequence[float]], *, approach_z: float = 0.90) -> list[dict[str, Any]]:
    """Return an ordered list of {kind, ...} motions. No simulator import."""
    stype = step["type"]
    if stype not in ROUTINE_TYPES:
        raise ValueError(f"unsupported routine type: {stype!r}")

    def xyz(name: str) -> list[float]:
        if name not in poses:
            raise KeyError(f"missing pose for {name}")
        return [float(v) for v in poses[name]]

    def waypoint(key: str) -> tuple[float, float, float]:
        payload = step.get(key) or {}
        return (
            float(payload.get("dx_m", 0.0)),
            float(payload.get("dy_m", 0.0)),
            float(payload.get("dz_m", 0.0)),
        )

    ax, ay, az = waypoint("approach_waypoint")
    rx, ry, rz = waypoint("release_waypoint")
    motions: list[dict[str, Any]] = []
    if stype == "pause_hold":
        intervals = int(step.get("intervals", 2))
        motions.append({"kind": "pause", "intervals": intervals, "grasp": False})
        return motions

    if stype == "grasp":
        pos = xyz(step["obj"])
        motions.append({"kind": "move", "xyz": [pos[0] + ax, pos[1] + ay, approach_z + az], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [pos[0], pos[1], pos[2]], "grip": 1.0, "grasp": False})
        motions.append({"kind": "grip", "grip": 0.0, "grasp_obj": step["obj"], "grasp": True})
        return motions

    if stype == "push":
        obj = xyz(step["obj"])
        target = xyz(step["target"])
        via = xyz(step["via"]) if step.get("via") and step["via"] in poses else None
        push_z = obj[2] + float(step.get("push_z", 0.02))
        dx, dy = target[0] - obj[0], target[1] - obj[1]
        norm = (dx * dx + dy * dy) ** 0.5
        ux, uy = (dx / norm, dy / norm) if norm > 1e-6 else (1.0, 0.0)
        pre = [obj[0] - ux * 0.06 + ax, obj[1] - uy * 0.06 + ay]
        motions.append({"kind": "move", "xyz": [pre[0], pre[1], approach_z + az], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [pre[0], pre[1], push_z], "grip": 1.0, "grasp": False})
        if via is not None:
            motions.append({"kind": "move", "xyz": [via[0], via[1], push_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [target[0], target[1], push_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [target[0] + rx, target[1] + ry, approach_z + rz], "grip": 1.0, "grasp": False})
        return motions

    if stype in ARTICULATION_TYPES:
        handle = xyz(step["obj"])
        target = xyz(step["target"])
        motions.append({"kind": "move", "xyz": [handle[0], handle[1], approach_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [handle[0], handle[1], handle[2]], "grip": 1.0, "grasp": False})
        motions.append({"kind": "grip", "grip": 0.0, "grasp_obj": step["obj"], "grasp": True})
        motions.append({
            "kind": "slide",
            "xyz": [target[0], target[1], target[2]],
            "grip": 0.0,
            "axis": step.get("axis", "y"),
            "grasp": True,
        })
        motions.append({"kind": "grip", "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [target[0], target[1], approach_z], "grip": 1.0, "grasp": False})
        return motions

    if stype == "reach":
        target = xyz(step["target"])
        reach_z = target[2] + float(step.get("reach_z", 0.03))
        motions.append({"kind": "move", "xyz": [target[0], target[1], approach_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [target[0], target[1], reach_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [target[0], target[1], approach_z], "grip": 1.0, "grasp": False})
        return motions

    if stype == "lift":
        obj = xyz(step["obj"])
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], approach_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], obj[2]], "grip": 1.0, "grasp": False})
        motions.append({"kind": "grip", "grip": 0.0, "grasp_obj": step["obj"], "grasp": True})
        if step.get("target") and step["target"] in poses:
            tgt = xyz(step["target"])
            motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], tgt[2]], "grip": 0.0, "grasp": True, "frame": "object"})
        else:
            lift_z = float(step.get("lift_z", obj[2] + 0.12))
            motions.append({"kind": "move", "xyz": [obj[0], obj[1], lift_z], "grip": 0.0, "grasp": True, "frame": "object"})
        return motions

    if stype == "place":
        tgt = xyz(step["target"])
        place_z = float(step.get("place_z", 0.0))
        motions.append({"kind": "move", "xyz": [tgt[0] + ax, tgt[1] + ay, approach_z + az], "grip": 0.0, "grasp": True, "frame": "object"})
        motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], tgt[2] + place_z], "grip": 0.0, "grasp": True, "frame": "object"})
        motions.append({"kind": "grip", "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [tgt[0] + rx, tgt[1] + ry, approach_z + rz], "grip": 1.0, "grasp": False})
        return motions

    if stype == "pick_place":
        obj = xyz(step["obj"])
        tgt = xyz(step["target"])
        motions.append({"kind": "move", "xyz": [obj[0] + ax, obj[1] + ay, approach_z + az], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], obj[2]], "grip": 1.0, "grasp": False})
        motions.append({"kind": "grip", "grip": 0.0, "grasp_obj": step["obj"], "grasp": True})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], approach_z], "grip": 0.0, "grasp": True, "frame": "object"})
        motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], approach_z], "grip": 0.0, "grasp": True, "frame": "object"})
        motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], tgt[2] + float(step.get("place_z", 0.0))], "grip": 0.0, "grasp": True, "frame": "object"})
        motions.append({"kind": "grip", "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [tgt[0] + rx, tgt[1] + ry, approach_z + rz], "grip": 1.0, "grasp": False})
        return motions

    if stype == "grasp_rotate":
        obj = xyz(step["obj"])
        lift_z = obj[2] + float(step.get("lift_z", 0.15))
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], approach_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], obj[2]], "grip": 1.0, "grasp": False})
        motions.append({"kind": "grip", "grip": 0.0, "grasp_obj": step["obj"], "grasp": True})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], lift_z], "grip": 0.0, "grasp": True})
        motions.append({"kind": "rotate", "yaw_deg": float(step.get("yaw_deg", 90.0)), "grip": 0.0, "grasp": True})
        return motions

    if stype == "transport_through_aperture":
        obj = xyz(step["obj"])
        tgt = xyz(step["target"])
        aperture = xyz(step["aperture_wp"]) if step.get("aperture_wp") in poses else None
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], approach_z], "grip": 1.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], obj[2]], "grip": 1.0, "grasp": False})
        motions.append({"kind": "grip", "grip": 0.0, "grasp_obj": step["obj"], "grasp": True})
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], approach_z], "grip": 0.0, "grasp": True})
        if aperture is not None:
            motions.append({"kind": "move", "xyz": [aperture[0], aperture[1], approach_z], "grip": 0.0, "grasp": True})
        motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], approach_z], "grip": 0.0, "grasp": True})
        motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], tgt[2] + float(step.get("place_z", 0.0))], "grip": 0.0, "grasp": True, "frame": "object"})
        motions.append({"kind": "grip", "grip": 1.0, "grasp": False})
        return motions

    raise ValueError(f"unsupported routine type: {stype!r}")


def plan_routine(routine: Sequence[Mapping[str, Any]], poses: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    assert_routine_supported(routine)
    planned: list[dict[str, Any]] = []
    for step in routine:
        planned.extend(plan_step(step, poses))
    return planned


@dataclass
class ExpertRuntime:
    find_shape: Callable[[str], Any]
    move_ik: Callable[[Any, float], None]
    actuate_gripper: Callable[..., None]
    hold_pose: Callable[[int], None]
    get_tip_pos: Callable[[], Any]
    approach_z: float = 0.90


def execute_step(step: Mapping[str, Any], runtime: ExpertRuntime, poses: Mapping[str, Sequence[float]] | None = None) -> None:
    """Execute one compiled routine step via injected callbacks."""
    if poses is None:
        poses = {}
        names = [step.get("obj"), step.get("target"), step.get("via"), step.get("aperture_wp")]
        for name in names:
            if not name:
                continue
            shape = runtime.find_shape(str(name))
            if shape is not None:
                poses[str(name)] = list(shape.get_position())
    for motion in plan_step(step, poses, approach_z=runtime.approach_z):
        kind = motion["kind"]
        if kind in {"move", "slide"}:
            runtime.move_ik(motion["xyz"], float(motion.get("grip", 1.0)))
        elif kind == "grip":
            obj = None
            if motion.get("grasp_obj"):
                obj = runtime.find_shape(str(motion["grasp_obj"]))
            runtime.actuate_gripper(float(motion["grip"]), obj)
        elif kind == "pause":
            runtime.hold_pose(int(motion["intervals"]))
        elif kind == "rotate":
            runtime.move_ik(list(runtime.get_tip_pos()), 0.0)
