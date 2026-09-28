"""Scripted expert motions for generation routine types.

``plan_step`` is simulator-free.  ``execute_step`` runs the same motions through
injected IK/gripper callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL


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


def _approach_and_grasp(
    name: str,
    pos: Sequence[float],
    approach_z: float,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 0.0,
) -> list[dict[str, Any]]:
    """Open approach above an object, descend to its centre and close on it."""
    return [
        {"kind": "move", "xyz": [pos[0] + ax, pos[1] + ay, approach_z + az], "grip": 1.0, "grasp": False},
        {"kind": "move", "xyz": [pos[0], pos[1], pos[2]], "grip": 1.0, "grasp": False},
        {"kind": "grip", "grip": 0.0, "grasp_obj": name, "grasp": True},
    ]


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
        motions.extend(_approach_and_grasp(step["obj"], xyz(step["obj"]), approach_z, ax, ay, az))
        return motions

    if stype == "push":
        obj = xyz(step["obj"])
        target = xyz(step["target"])
        via_name = step.get("via")
        via = xyz(via_name) if via_name and via_name != step["target"] and via_name in poses else None
        # Push at the object's centre height: the closed fingertips end only
        # a few millimetres below the tool tip, so a higher push only grazes
        # the top edge of small objects.
        push_z = obj[2] + float(step.get("push_z", 0.0))
        # Distance from the tool tip to the closed-finger contact face plus
        # the object's half extent along the push direction.  The worker
        # supplies the measured value; the protocol constant is the nominal
        # unit-scale fallback.
        contact_offset_m = float(step.get("contact_offset_m", GENERATION_PROTOCOL.push_contact_offset_m))
        overshoot_m = float(step.get("overshoot_m", 0.0))
        clearance_m = float(step.get("pre_push_clearance_m", 0.03))
        back_off_m = float(step.get("push_back_off_m", 0.01))

        def unit(src, dst) -> tuple[float, float]:
            dx, dy = dst[0] - src[0], dst[1] - src[1]
            norm = (dx * dx + dy * dy) ** 0.5
            return (dx / norm, dy / norm) if norm > 1e-6 else (1.0, 0.0)

        path = [via, target] if via is not None else [target]
        ux, uy = unit(obj, path[0])
        # No lateral waypoint noise behind the object: an off-centre start
        # turns the push into a rotation.  Only the approach height varies.
        pre = [obj[0] - ux * (contact_offset_m + clearance_m), obj[1] - uy * (contact_offset_m + clearance_m)]
        motions.append({"kind": "move", "xyz": [pre[0], pre[1], approach_z + az], "grip": 0.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [pre[0], pre[1], push_z], "grip": 0.0, "grasp": False})
        previous = obj
        end = pre
        for index, point in enumerate(path):
            ux, uy = unit(previous, point)
            extra = overshoot_m if index == len(path) - 1 else 0.0
            end = [point[0] - ux * (contact_offset_m - extra), point[1] - uy * (contact_offset_m - extra)]
            motions.append({"kind": "move", "xyz": [end[0], end[1], push_z], "grip": 0.0, "grasp": False, "push_contact": True})
            previous = point
        back = [end[0] - ux * back_off_m, end[1] - uy * back_off_m]
        motions.append({"kind": "move", "xyz": [back[0], back[1], push_z], "grip": 0.0, "grasp": False})
        motions.append({"kind": "move", "xyz": [back[0] + rx, back[1] + ry, approach_z + rz], "grip": 0.0, "grasp": False})
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
        if not step.get("held"):
            motions.extend(_approach_and_grasp(step["obj"], obj, approach_z))
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
        if not step.get("held"):
            motions.extend(_approach_and_grasp(step["obj"], obj, approach_z))
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], lift_z], "grip": 0.0, "grasp": True})
        motions.append({"kind": "rotate", "yaw_deg": float(step.get("yaw_deg", 90.0)), "grip": 0.0, "grasp": True})
        return motions

    if stype == "transport_through_aperture":
        obj = xyz(step["obj"])
        tgt = xyz(step["target"])
        aperture = xyz(step["aperture_wp"]) if step.get("aperture_wp") in poses else None
        if not step.get("held"):
            motions.extend(_approach_and_grasp(step["obj"], obj, approach_z))
        motions.append({"kind": "move", "xyz": [obj[0], obj[1], approach_z], "grip": 0.0, "grasp": True})
        if aperture is not None:
            motions.append({"kind": "move", "xyz": [aperture[0], aperture[1], approach_z], "grip": 0.0, "grasp": True})
        motions.append({"kind": "move", "xyz": [tgt[0], tgt[1], approach_z], "grip": 0.0, "grasp": True})
        if step.get("release", True):
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
