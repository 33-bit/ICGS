"""Materialize native proposals as timed, absolute world-frame commands."""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np

from icgs.contracts.method import CommandPrefix, TimedCommand


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _array(value: Any, name: str) -> np.ndarray:
    """Copy NumPy-like or tensor-like values without importing a tensor runtime."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _candidate_parts(candidate: Any) -> tuple[Any, Any, str, Any]:
    trajectory = getattr(candidate, "trajectory", candidate)
    transforms = getattr(trajectory, "transforms", None)
    grips = getattr(trajectory, "grips", None)
    if transforms is None or grips is None:
        raise ValueError("candidate must provide a trajectory with transforms and grips")

    root = getattr(candidate, "root_pose", None)
    if root is None:
        root = getattr(candidate, "proposal_root", None)
    if root is None:
        raise ValueError("candidate must provide one proposal root pose")

    raw_id = getattr(candidate, "raw_candidate_id", None)
    if raw_id is None:
        raw_id = getattr(candidate, "index", None)
    if (raw_id is None or isinstance(raw_id, (bool, np.bool_))
            or not isinstance(raw_id, (str, int, np.integer))):
        raise ValueError("candidate must provide a raw candidate ID or index")
    raw_id = str(raw_id)
    if not raw_id:
        raise ValueError("candidate must provide a raw candidate ID or index")
    return transforms, grips, raw_id, root


def materialize_grips(grips: Any, r: int) -> tuple[int, ...]:
    """Convert native grip logits/states to binary commands held for each group."""
    r = _integer(r, "r")
    if r < 1:
        raise ValueError("r must be a positive integer")
    values = _array(grips, "grips")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("grips must be a nonempty one-dimensional sequence")

    result: list[int] = []
    for start in range(0, len(values), r):
        held = int(values[start] > 0)
        result.extend([held] * min(r, len(values) - start))
    return tuple(result)


def materialize_prefix(
    candidate: Any,
    *,
    h: int,
    r: int,
    duration_s: float,
) -> CommandPrefix:
    """Convert one B=1 native candidate into absolute timed commands.

    Relative targets are all anchored at the candidate's single proposal root.
    The raw trajectory remains owned by the caller; the returned command records
    contain independent read-only target arrays.
    """
    h = _integer(h, "h")
    r = _integer(r, "r")
    if h < 1 or r < 1 or r > h:
        raise ValueError("materialization requires 1 <= r <= h")
    try:
        duration = float(duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_s must be finite and positive") from exc
    if not isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be finite and positive")

    transforms, grips, raw_id, root_value = _candidate_parts(candidate)
    transforms = _array(transforms, "candidate transforms")
    if transforms.ndim != 4 or transforms.shape[0] != 1 or transforms.shape[-2:] != (4, 4):
        raise ValueError("candidate transforms must have batch size 1 and shape [B,P,4,4]")
    horizon = transforms.shape[1]
    if h > horizon:
        raise ValueError("h must not exceed the candidate prediction horizon")

    grips = _array(grips, "candidate grips")
    if grips.shape != (1, horizon, 1):
        raise ValueError("candidate grips must have shape [1,P,1]")

    root = _array(root_value, "proposal root")
    if root.shape != (4, 4):
        raise ValueError("one proposal root pose required")

    materialized_grips = materialize_grips(grips[0, :h, 0], r)
    world_targets = np.matmul(root, transforms[0, :h])
    commands = tuple(
        TimedCommand(target, grip, duration)
        for target, grip in zip(world_targets, materialized_grips)
    )
    return CommandPrefix(commands, root, raw_id)


__all__ = ["materialize_grips", "materialize_prefix"]
