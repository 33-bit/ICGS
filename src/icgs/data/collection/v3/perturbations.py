"""Phase-1 perturbation kinds and attempt outcome classes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.steps import PROGRAM_FAMILIES


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
    kinds = list(V3_PROTOCOL.perturbation_kinds)
    family = PROGRAM_FAMILIES.get(program_id)
    if family not in _BLOCKER_FAMILIES:
        kinds.remove("blocker_insertion")
    return tuple(kinds)


def perturbation_quota(program_id: str, total: int | None = None) -> dict[str, int | str]:
    total = V3_PROTOCOL.train_perturbed_attempts_per_program if total is None else total
    applicable = applicable_perturbations(program_id)
    all_kinds = V3_PROTOCOL.perturbation_kinds
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
    if kind not in V3_PROTOCOL.perturbation_kinds:
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
        if abs(dx) > V3_PROTOCOL.target_offset_m or abs(dy) > V3_PROTOCOL.target_offset_m:
            raise ValueError("action pose offset exceeds 2 cm")
        if abs(ddeg) > V3_PROTOCOL.target_rotation_offset_deg:
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
        if abs(delta) > V3_PROTOCOL.grip_timing_offset_intervals:
            raise ValueError("gripper timing offset exceeds 1 interval")
        for step in next_routine:
            if step.get("type") != "pause_hold":
                step["grip_timing_delta_intervals"] = delta
                break
    elif kind == "object_displacement":
        dx = float(params.get("dx_m", 0.0))
        dy = float(params.get("dy_m", 0.0))
        if math_hypot(dx, dy) > V3_PROTOCOL.object_shift_m + 1e-9:
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
        next_objects[blocker] = {
            "pos": list(params.get("pos", [0.30, 0.00, 0.775])),
            "size": [0.04, 0.04, 0.04],
            "color": [0.1, 0.1, 0.1],
            "declared": True,
        }
        meta["external_intervention"] = True
        meta["episode_has_external_intervention"] = True
    elif kind == "pause_hold":
        lo, hi = V3_PROTOCOL.pause_intervals
        intervals = int(params.get("intervals", lo))
        if intervals < lo or intervals > hi:
            raise ValueError("pause/hold must be 2-10 intervals")
        next_routine.insert(0, {"type": "pause_hold", "intervals": intervals})
    return {"objects": next_objects, "routine": next_routine, "intervention": meta}


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
