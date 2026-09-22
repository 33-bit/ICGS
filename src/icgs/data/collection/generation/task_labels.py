"""Measured task-memory labels for phase-1 event views."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _objects(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("name")): item
        for item in state.get("objects", ())
        if isinstance(item, Mapping) and item.get("name")
    }


def _position(objects: Mapping[str, Mapping[str, Any]], name: str | None) -> np.ndarray | None:
    item = objects.get(str(name)) if name else None
    if not item or not item.get("valid", True) or item.get("position") is None:
        return None
    value = np.asarray(item["position"], dtype=np.float64)
    return value if value.shape == (3,) and np.isfinite(value).all() else None


def _ee_position(robot_state: Mapping[str, Any]) -> np.ndarray | None:
    transform = np.asarray(robot_state.get("T_w_e"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None
    return transform[:3, 3]


def _relation_observed(
    relation: str,
    step: Mapping[str, Any],
    state: Mapping[str, Any],
    robot_state: Mapping[str, Any],
    initial_state: Mapping[str, Any],
) -> bool | None:
    """Evaluate only geometric postconditions with measured object states."""
    post = str(relation or "")
    objects = _objects(state)
    obj = _position(objects, step.get("object_role"))
    target = _position(objects, step.get("target_role") or step.get("aperture_role"))
    ee = _ee_position(robot_state)
    grip = robot_state.get("grip")
    if post.endswith("_free") and obj is not None and ee is not None and grip is not None:
        return bool(float(grip) > 0.5 or np.linalg.norm(obj - ee) > 0.10)
    if post.endswith("_grasped") and obj is not None and ee is not None and grip is not None:
        return bool(float(grip) <= 0.5 and np.linalg.norm(obj - ee) <= 0.10)
    if post == "ee_at_target" and target is not None and ee is not None:
        return bool(np.linalg.norm(ee - target) <= 0.06)
    if post.endswith("_lifted") and obj is not None:
        initial_obj = _position(_objects(initial_state), step.get("object_role"))
        if target is not None:
            return bool(np.linalg.norm(obj - target) <= 0.04)
        if initial_obj is not None:
            return bool(obj[2] - initial_obj[2] >= 0.05)
    if obj is not None and target is not None and any(token in post for token in ("in_target", "placed", "fitted", "retrieved", "parked", "restored", "at_push_target", "past_", "cleared")):
        return bool(np.linalg.norm(obj - target) <= 0.04)
    return None


def materialize_task_labels(
    structured_steps: Sequence[Mapping[str, Any]],
    object_states: Sequence[Mapping[str, Any]],
    robot_states: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Return ``rho/nu/epsilon`` and masks aligned to T+1 boundaries.

    ``nu`` is only marked valid when its postcondition is geometrically
    observable from measured object states.  Unknown grasp/articulation
    predicates remain explicitly invalid instead of being inferred from final
    success. ``rho`` records historical occurrence of a valid observed
    postcondition; ``epsilon`` records known prerequisite satisfaction.
    """
    boundaries = len(object_states)
    events = len(structured_steps)
    shape = (boundaries, events)
    rho = np.zeros(shape, dtype=np.float32)
    nu = np.zeros(shape, dtype=np.float32)
    epsilon = np.zeros(shape, dtype=np.float32)
    rho_valid = np.zeros(shape, dtype=bool)
    nu_valid = np.zeros(shape, dtype=bool)
    epsilon_valid = np.zeros(shape, dtype=bool)
    robots = list(robot_states or ({} for _ in range(boundaries)))
    if len(robots) != boundaries:
        raise ValueError("robot_states must align with object_states")
    initial_state = object_states[0] if object_states else {}
    for t, state in enumerate(object_states):
        for index, step in enumerate(structured_steps):
            observed = _relation_observed(
                str(step.get("postcondition") or ""), step, state, robots[t], initial_state
            )
            if observed is not None:
                nu[t, index] = float(observed)
                nu_valid[t, index] = True
                rho[t, index] = float(observed or (t > 0 and rho[t - 1, index] > 0.5))
                rho_valid[t, index] = True
            prerequisite = step.get("precondition")
            if not prerequisite:
                epsilon[t, index] = 1.0
                epsilon_valid[t, index] = True
            else:
                source = next(
                    (
                        prior for prior in range(index - 1, -1, -1)
                        if prerequisite in {
                            structured_steps[prior].get("postcondition"),
                            structured_steps[prior].get("success_predicate"),
                        }
                    ),
                    None,
                )
                if source is not None and rho_valid[t, source]:
                    epsilon[t, index] = rho[t, source]
                    epsilon_valid[t, index] = True
                elif source is None:
                    initial = _relation_observed(
                        str(prerequisite), step, state, robots[t], initial_state
                    )
                    if initial is not None:
                        epsilon[t, index] = float(initial)
                        epsilon_valid[t, index] = True
    return {
        "rho": rho, "rho_valid": rho_valid,
        "nu": nu, "nu_valid": nu_valid,
        "epsilon": epsilon, "epsilon_valid": epsilon_valid,
    }


__all__ = ["materialize_task_labels"]
