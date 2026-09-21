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


def _postcondition_observed(step: Mapping[str, Any], state: Mapping[str, Any]) -> bool | None:
    """Evaluate only geometric postconditions with measured object states."""
    post = str(step.get("postcondition") or "")
    objects = _objects(state)
    obj = _position(objects, step.get("object_role"))
    target = _position(objects, step.get("target_role") or step.get("aperture_role"))
    if obj is None or target is None:
        return None
    if any(token in post for token in ("in_target", "placed", "fitted", "retrieved", "parked", "restored", "at_target", "past_")):
        return bool(np.linalg.norm(obj - target) <= 0.04)
    return None


def materialize_task_labels(
    structured_steps: Sequence[Mapping[str, Any]],
    object_states: Sequence[Mapping[str, Any]],
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
    for t, state in enumerate(object_states):
        for index, step in enumerate(structured_steps):
            observed = _postcondition_observed(step, state)
            if observed is not None:
                nu[t, index] = float(observed)
                nu_valid[t, index] = True
                rho[t, index] = float(observed or (t > 0 and rho[t - 1, index] > 0.5))
                rho_valid[t, index] = True
            prerequisite = step.get("precondition")
            if not prerequisite:
                epsilon[t, index] = 1.0
                epsilon_valid[t, index] = True
            elif index > 0 and rho_valid[t, index - 1]:
                epsilon[t, index] = rho[t, index - 1]
                epsilon_valid[t, index] = True
    return {
        "rho": rho, "rho_valid": rho_valid,
        "nu": nu, "nu_valid": nu_valid,
        "epsilon": epsilon, "epsilon_valid": epsilon_valid,
    }


__all__ = ["materialize_task_labels"]
