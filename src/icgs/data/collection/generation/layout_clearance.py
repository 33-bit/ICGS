"""Simulator-free physical clearance audit of prepared generation scenes.

Uses the measured Panda finger extent around the tool tip (world axes, default
tool yaw): open fingers span x in [-0.0196, 0.0095] m and y in +/-0.0608 m,
closed fingers y in +/-0.021 m, fingertips 0.0016 m below the tip.  A prepared
attempt is physically well posed when no solid body spawns interpenetrating
another, no placement lands inside a present body, and the fingers never sweep
through a neighbour while grasping, releasing, pushing or sliding a handle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

TABLE_TOP_Z_M = 0.752
DEFAULT_FINGER_AXIS_DEG = 90.0  # fingers close along world y at the default tool yaw
ELONGATED_ASPECT_RATIO = 1.3
FINGERTIP_BELOW_TIP_M = 0.0016
OPEN_FINGERS_M = ((-0.0196, 0.0095), (-0.0608, 0.0608))
CLOSED_FINGERS_M = ((-0.0196, 0.0095), (-0.021, 0.021))
SUPPORT_NAMES = frozenset({"pad", "tray", "holder", "drawer"})
_GRASP_TYPES = frozenset({"grasp", "pick_place", "open_articulation", "close_articulation"})
_HELD_CONTINUATION_TYPES = frozenset({"lift", "grasp_rotate", "transport_through_aperture"})
_RELEASE_TYPES = frozenset({"place", "pick_place", "open_articulation", "close_articulation"})


@dataclass(frozen=True)
class ClearanceViolation:
    phase: str
    body: str
    other: str
    depth_m: float


def grasp_yaw_delta_deg(size_xy, object_yaw_deg: float, finger_axis_deg: float) -> float:
    """Tool yaw change aligning the finger closing axis with a body face normal.

    Near-square footprints use the nearest face (|delta| <= 45 deg); elongated
    footprints close across the short side (|delta| <= 90 deg).  Both keep the
    fingers flush with faces instead of landing on corners of a yawed body.
    """
    width, depth = float(size_xy[0]), float(size_xy[1])
    if max(width, depth) >= ELONGATED_ASPECT_RATIO * min(width, depth):
        short_axis = float(object_yaw_deg) + (0.0 if width <= depth else 90.0)
        return (short_axis - float(finger_axis_deg) + 90.0) % 180.0 - 90.0
    return (float(object_yaw_deg) - float(finger_axis_deg) + 45.0) % 90.0 - 45.0


def is_marker(name: str) -> bool:
    return name.startswith("target") or name.endswith("_wp") or name.endswith("_target") or "_target_" in name


def is_solid(name: str) -> bool:
    if is_marker(name):
        return False
    movable = (
        name.startswith("object") or name.startswith("blocker") or name.startswith("spacer")
        or "handle" in name or name == "inserted_blocker"
    )
    return movable or name in SUPPORT_NAMES


def _rectangle(center, half_x: float, half_y: float, yaw_deg: float = 0.0) -> list[tuple[float, float]]:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [
        (center[0] + c * x - s * y, center[1] + s * x + c * y)
        for x, y in ((half_x, half_y), (-half_x, half_y), (-half_x, -half_y), (half_x, -half_y))
    ]


def _box(center, extent, margin: float, turn_deg: float = 0.0) -> list[tuple[float, float]]:
    """Finger box around the tool tip, turned by the tool yaw about the tip."""
    (x0, x1), (y0, y1) = extent
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    corners = ((x0 - margin, y0 - margin), (x1 + margin, y0 - margin), (x1 + margin, y1 + margin), (x0 - margin, y1 + margin))
    return [(center[0] + c * x - s * y, center[1] + s * x + c * y) for x, y in corners]


def _sweep(start, end, extent, margin: float) -> list[tuple[float, float]]:
    (x0, x1), (y0, y1) = extent
    xs = [start[0] + x0, start[0] + x1, end[0] + x0, end[0] + x1]
    ys = [start[1] + y0, start[1] + y1, end[1] + y0, end[1] + y1]
    return [(min(xs) - margin, min(ys) - margin), (max(xs) + margin, min(ys) - margin),
            (max(xs) + margin, max(ys) + margin), (min(xs) - margin, max(ys) + margin)]


def _penetration(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    best = float("inf")
    for polygon in (a, b):
        for index in range(len(polygon)):
            x1, y1 = polygon[index]
            x2, y2 = polygon[(index + 1) % len(polygon)]
            nx, ny = y1 - y2, x2 - x1
            norm = math.hypot(nx, ny)
            nx, ny = nx / norm, ny / norm
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            best = min(best, min(max(pa), max(pb)) - max(min(pa), min(pb)))
    return best


def _spec(item: Any) -> tuple[list[float], list[float], float]:
    if isinstance(item, Mapping):
        return list(item["pos"]), list(item.get("size") or [0.04, 0.04, 0.04]), float(item.get("yaw_deg") or 0.0)
    position, size, _color = item
    return list(position), list(size), 0.0


def _rest_center(xy, size, supports) -> list[float]:
    z = TABLE_TOP_Z_M + size[2] / 2.0
    for position, support_size in supports:
        if abs(xy[0] - position[0]) <= support_size[0] / 2.0 and abs(xy[1] - position[1]) <= support_size[1] / 2.0:
            z = max(z, position[2] + support_size[2] / 2.0 + size[2] / 2.0)
    return [float(xy[0]), float(xy[1]), z]


def audit_prepared_attempt(
    objects: Mapping[str, Any],
    routine: Iterable[Mapping[str, Any]],
    *,
    margin_m: float = 0.003,
) -> list[ClearanceViolation]:
    """Return every clearance violation of one prepared attempt (empty when clean)."""
    specs = {name: _spec(item) for name, item in objects.items()}
    supports = [(position, size) for name, (position, size, _yaw) in specs.items() if name in SUPPORT_NAMES]
    state: dict[str, tuple[list[float], list[float], float]] = {}
    for name, (position, size, yaw) in specs.items():
        if is_solid(name):
            state[name] = (position if name in SUPPORT_NAMES else _rest_center(position, size, supports), size, yaw)
    violations: list[ClearanceViolation] = []

    def footprint(name: str) -> list[tuple[float, float]]:
        position, size, yaw = state[name]
        return _rectangle(position, size[0] / 2.0, size[1] / 2.0, yaw)

    def z_range(name: str) -> tuple[float, float]:
        position, size, _yaw = state[name]
        return position[2] - size[2] / 2.0, position[2] + size[2] / 2.0

    def check(phase: str, body: str, polygon, bottom_z: float, top_z: float = float("inf")) -> None:
        for other in state:
            if other == body:
                continue
            low, high = z_range(other)
            if high <= bottom_z + 1e-9 or low >= top_z - 1e-9:
                continue
            if other in SUPPORT_NAMES and phase.startswith(("grasp", "release", "push", "slide")):
                # Fingers stay above the support top whenever the tip is at a
                # body centre resting on that support.
                if bottom_z >= high - 1e-9:
                    continue
            depth = _penetration(polygon, footprint(other))
            if depth > 0.0:
                violations.append(ClearanceViolation(phase, body, other, depth))

    for name in list(state):
        if name in SUPPORT_NAMES:
            continue
        low, high = z_range(name)
        check("spawn", name, footprint(name), low, high)

    finger_axis = DEFAULT_FINGER_AXIS_DEG
    for step in routine:
        kind = step.get("type")
        body, target = step.get("obj"), step.get("target")
        held = bool(step.get("held"))
        if body in state and (kind in _GRASP_TYPES or (kind in _HELD_CONTINUATION_TYPES and not held)):
            position, size, yaw = state[body]
            finger_axis += grasp_yaw_delta_deg(size[:2], yaw, finger_axis)
            check(f"grasp:{kind}", body, _box(position, OPEN_FINGERS_M, margin_m, finger_axis - DEFAULT_FINGER_AXIS_DEG),
                  position[2] - FINGERTIP_BELOW_TIP_M)
        if kind == "grasp_rotate":
            finger_axis += float(step.get("yaw_deg", 90.0))
            if body in state:
                position, size, yaw = state[body]
                state[body] = (position, size, yaw + float(step.get("yaw_deg", 90.0)))
        if kind == "push" and body in state and target in specs:
            position, size, yaw = state[body]
            goal = specs[target][0]
            via_name = step.get("via")
            path = [specs[via_name][0], goal] if via_name in specs and via_name != target else [goal]
            contact = CLOSED_FINGERS_M[1][1] + size[1] / 2.0
            dx, dy = path[0][0] - position[0], path[0][1] - position[1]
            norm = math.hypot(dx, dy) or 1.0
            start = (position[0] - dx / norm * (contact + 0.03), position[1] - dy / norm * (contact + 0.03))
            previous = position
            for point in path:
                ux, uy = point[0] - previous[0], point[1] - previous[1]
                length = math.hypot(ux, uy) or 1.0
                end = (point[0] - ux / length * contact, point[1] - uy / length * contact)
                check("push", body, _sweep(start, end, CLOSED_FINGERS_M, margin_m), position[2] - FINGERTIP_BELOW_TIP_M)
                start, previous = end, point
            state[body] = ([goal[0], goal[1], position[2]], size, yaw)
        if kind in {"open_articulation", "close_articulation"} and body in state and target in specs:
            position, size, yaw = state[body]
            goal = specs[target][0]
            handle_extent = ((-size[0] / 2.0, size[0] / 2.0), (-size[1] / 2.0, size[1] / 2.0))
            bottom = position[2] - FINGERTIP_BELOW_TIP_M
            check("slide:fingers", body, _sweep(position, goal, CLOSED_FINGERS_M, margin_m), bottom)
            check("slide:handle", body, _sweep(position, goal, handle_extent, 0.0), position[2] - size[2] / 2.0)
        releases = kind in _RELEASE_TYPES or (kind == "transport_through_aperture" and step.get("release", True))
        if releases and body in state and target in specs:
            goal = specs[target][0]
            position, size, yaw = state[body]
            placed = [goal[0], goal[1], position[2]] if "handle" in body else _rest_center(goal, size, supports)
            state[body] = (placed, size, yaw)
            low, high = z_range(body)
            check(f"place:{kind}", body, footprint(body), low, high)
            check(f"release:{kind}", body, _box(goal, OPEN_FINGERS_M, margin_m, finger_axis - DEFAULT_FINGER_AXIS_DEG),
                  placed[2] - FINGERTIP_BELOW_TIP_M)
    return violations
