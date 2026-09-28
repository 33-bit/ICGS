"""Phase-1 perturbation kinds and attempt outcome classes."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.collection.generation.steps import PROGRAM_FAMILIES


SUCCESS = "success"
VALID_FAILURE = "valid_failure"
SIMULATOR_CRASH = "simulator_crash"
INVALID_OBSERVATION = "invalid_observation"
PHYSICAL_FAILURE = VALID_FAILURE

_BLOCKER_FAMILIES = {
    "obstacle-access",
    "gated-access",
    "park-retrieve",
    "park-restore",
    "park-retrieve-restore",
    "grasp-fit",
}


def applicable_perturbations(program_id: str) -> tuple[str, ...]:
    kinds = list(GENERATION_PROTOCOL.perturbation_kinds)
    family = PROGRAM_FAMILIES.get(program_id)
    if family not in _BLOCKER_FAMILIES:
        kinds.remove("blocker_insertion")
    return tuple(kinds)


def perturbation_quota(program_id: str, total: int | None = None) -> dict[str, int | str]:
    total = GENERATION_PROTOCOL.train_perturbed_attempts_per_program if total is None else total
    applicable = applicable_perturbations(program_id)
    all_kinds = GENERATION_PROTOCOL.perturbation_kinds
    quota: dict[str, int | str] = {kind: "not_applicable" for kind in all_kinds if kind not in applicable}
    share, remainder = divmod(total, len(applicable))
    for index, kind in enumerate(applicable):
        quota[kind] = share + (1 if index < remainder else 0)
    return quota


def apply_perturbation(
    kind: str,
    params: Mapping[str, Any],
    objects: Mapping[str, Any],
    routine: list[dict[str, Any]],
) -> dict[str, Any]:
    if kind not in GENERATION_PROTOCOL.perturbation_kinds:
        raise ValueError(f"unsupported perturbation kind: {kind}")
    next_objects = deepcopy(dict(objects))
    next_routine = deepcopy(list(routine))
    scope = intervention_application_scope(kind)
    meta: dict[str, Any] = {
        "episode_kind": "perturbed",
        "intervention_type": kind,
        "intervention_id": f"{kind}_v1",
        "intervention_params": dict(params),
        "application_scope": scope,
        "application_t": None,
        "intervention_frame": None,
        "source_episode_id": None,
        "source_frame": None,
        "pre_state_hash": None,
        "post_state_hash": None,
        "external_intervention": False,
        "episode_has_external_intervention": False,
        "held_out": False,
    }
    if kind == "action_pose_offset":
        dx = float(params.get("dx_m", 0.0))
        dy = float(params.get("dy_m", 0.0))
        ddeg = float(params.get("d_yaw_deg", 0.0))
        if abs(dx) > GENERATION_PROTOCOL.target_offset_m or abs(dy) > GENERATION_PROTOCOL.target_offset_m:
            raise ValueError("action pose offset exceeds 2 cm")
        if abs(ddeg) > GENERATION_PROTOCOL.target_rotation_offset_deg:
            raise ValueError("action pose offset exceeds 10 deg")
        target_name = params.get("target_role", "target_a")
        if target_name in next_objects and "pos" in next_objects[target_name]:
            pos = list(next_objects[target_name]["pos"])
            pos[0] += dx
            pos[1] += dy
            next_objects[target_name]["pos"] = pos
        for step in next_routine:
            if step.get("target") == target_name or step.get("type") in {"place", "pick_place"}:
                step["place_offset_m"] = {"dx_m": dx, "dy_m": dy, "d_yaw_deg": ddeg}
    elif kind == "gripper_timing":
        delta = int(params.get("delta_intervals", 0))
        if abs(delta) > GENERATION_PROTOCOL.grip_timing_offset_intervals:
            raise ValueError("gripper timing offset exceeds 1 interval")
        for step in next_routine:
            if step.get("type") != "pause_hold":
                step["grip_timing_delta_intervals"] = delta
                break
    elif kind == "object_displacement":
        dx = float(params.get("dx_m", 0.0))
        dy = float(params.get("dy_m", 0.0))
        if math_hypot(dx, dy) > GENERATION_PROTOCOL.object_shift_m + 1e-9:
            raise ValueError("object displacement exceeds 3 cm")
        obj = params.get("object_role", "object_a")
        if obj in next_objects and "pos" in next_objects[obj]:
            pos = list(next_objects[obj]["pos"])
            pos[0] += dx
            pos[1] += dy
            next_objects[obj]["pos"] = pos
        meta["external_intervention"] = True
        meta["episode_has_external_intervention"] = True
    elif kind == "blocker_insertion":
        blocker = params.get("blocker_role", "inserted_blocker")
        declared = list(params.get("pos", [0.30, 0.00, 0.775]))
        applied = free_blocker_position(declared, next_objects, size=_INSERTED_BLOCKER_SIZE_M)
        next_objects[blocker] = {
            "pos": applied,
            "size": [_INSERTED_BLOCKER_SIZE_M] * 3,
            "color": [0.1, 0.1, 0.1],
            "declared": True,
        }
        meta["declared_position"] = declared
        meta["applied_position"] = applied
        meta["external_intervention"] = True
        meta["episode_has_external_intervention"] = True
    elif kind == "pause_hold":
        lo, hi = GENERATION_PROTOCOL.pause_intervals
        intervals = int(params.get("intervals", lo))
        if intervals < lo or intervals > hi:
            raise ValueError("pause/hold must be 2-10 intervals")
        next_routine.insert(0, {"type": "pause_hold", "intervals": intervals})
    return {"objects": next_objects, "routine": next_routine, "intervention": meta}


_INSERTED_BLOCKER_SIZE_M = 0.04
_SUPPORT_NAMES = frozenset({"pad", "tray", "holder", "drawer"})


def _is_marker(name: str) -> bool:
    return (
        name.startswith("target")
        or name.endswith("_wp")
        or name.endswith("_target")
        or "_target_" in name
    )


def _footprint(pos, size, yaw_deg: float, margin: float) -> list[tuple[float, float]]:
    angle = math.radians(float(yaw_deg or 0.0))
    c, s = math.cos(angle), math.sin(angle)
    hx, hy = float(size[0]) / 2.0 + margin, float(size[1]) / 2.0 + margin
    return [
        (pos[0] + c * x - s * y, pos[1] + s * x + c * y)
        for x, y in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
    ]


def _separated(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    for polygon in (a, b):
        for index in range(len(polygon)):
            x1, y1 = polygon[index]
            x2, y2 = polygon[(index + 1) % len(polygon)]
            nx, ny = y1 - y2, x2 - x1
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return True
    return False


def free_blocker_position(
    declared,
    objects: Mapping[str, Any],
    *,
    size: float,
    margin: float = 0.005,
    step_m: float = 0.01,
    max_shift_m: float = 0.20,
) -> list[float]:
    """Nearest declared-row position whose footprint clears every solid body.

    The declared insertion point is kept when it is free.  Otherwise the
    blocker slides along the declared row (world y) in 1 cm steps, nearest
    first, so an inserted obstacle never spawns interpenetrating a scene body.
    """
    bodies = []
    for name, spec in objects.items():
        if _is_marker(name) or not isinstance(spec, Mapping) or "pos" not in spec:
            continue
        bodies.append(_footprint(spec["pos"], spec.get("size") or [0.04, 0.04, 0.04], spec.get("yaw_deg") or 0.0, 0.0))
    shifts = [0.0]
    for index in range(1, int(round(max_shift_m / step_m)) + 1):
        shifts.extend((index * step_m, -index * step_m))
    for shift in shifts:
        candidate = [float(declared[0]), float(declared[1]) + shift, float(declared[2])]
        footprint = _footprint(candidate, [size, size, size], 0.0, margin)
        if all(_separated(footprint, body) for body in bodies):
            return candidate
    raise ValueError(f"no collision-free inserted blocker position near {declared}")


def math_hypot(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5


def intervention_application_scope(kind: str) -> str:
    if kind in {"object_displacement", "blocker_insertion"}:
        return "initial_scene"
    if kind == "gripper_timing":
        return "timestep"
    return "event"


def classify_attempt_outcome(*, simulator_crash: bool, predicates_ok: bool, observation_valid: bool = True) -> str:
    if simulator_crash:
        return SIMULATOR_CRASH
    if not observation_valid:
        return INVALID_OBSERVATION
    if predicates_ok:
        return SUCCESS
    return VALID_FAILURE
