"""Replay provenance validation; engine restoration belongs to P08.

The validator checks the reported structure and finite scalar measurements.  It
does not restore state, compare trajectories, choose tolerances, or certify G5
equivalence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any, Literal, TypedDict


REPLAY_MODES = (
    "exact-snapshot",
    "deterministic-replay",
    "approximate-reset",
)

# Every category must be accounted for by a report.  A field may be stored but
# not restored; that is how unavailable hidden state remains visible without an
# exact-snapshot claim.
REQUIRED_REPLAY_FIELDS = (
    "joints",
    "velocities",
    "articulations",
    "constraints",
    "attachments",
    "integrators",
    "monitor_history",
    "rng",
    "solver",
    "object_poses",
    "grip",
    "controller_state",
    "task_history",
)
_KNOWN_REPLAY_FIELDS = frozenset(REQUIRED_REPLAY_FIELDS + ("command_history",))
_REQUIRED_DISCREPANCIES = ("pose", "cloud", "outcome")


class ReplayReport(TypedDict, total=False):
    mode: Literal["exact-snapshot", "deterministic-replay", "approximate-reset"]
    stored_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    restored_fields: tuple[str, ...]
    discrepancies: Mapping[str, float]
    protocol_id: str
    repetitions: int
    uncertainty: Mapping[str, float]


def _fields(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Iterable):
        raise ValueError(f"report {name} must list replay fields")
    result = tuple(value)
    if any(not isinstance(field, str) or not field for field in result):
        raise ValueError(f"report {name} must contain nonempty field names")
    if len(set(result)) != len(result):
        raise ValueError(f"report {name} must not contain duplicate fields")
    unknown = set(result) - _KNOWN_REPLAY_FIELDS
    if unknown:
        raise ValueError(f"report contains unknown replay fields: {sorted(unknown)}")
    return result


def _finite_mapping(value: Any, name: str, *, nonempty: bool = False) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"report {name} must be a mapping")
    if nonempty and not value:
        raise ValueError(f"report {name} must be nonempty")
    result: dict[str, float] = {}
    for name, measurement in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("report measurement names must be nonempty strings")
        if not isinstance(measurement, Real) or isinstance(measurement, bool):
            raise ValueError("report measurements must be finite scalar values")
        try:
            measurement = float(measurement)
        except (TypeError, ValueError) as exc:
            raise ValueError("report measurements must be finite scalar values") from exc
        if not math.isfinite(measurement):
            raise ValueError("report measurements must be finite scalar values")
        result[name] = measurement
    return result


def _discrepancies(value: Any) -> dict[str, float]:
    result = _finite_mapping(value, "discrepancies")
    missing = set(_REQUIRED_DISCREPANCIES) - set(result)
    if missing:
        raise ValueError(
            "report discrepancies must include finite measured pose, cloud, and outcome entries"
        )
    return result


def _repetitions(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 2:
        raise ValueError("report repetitions must be an integer >= 2")
    return int(value)


def validate_replay_report(report: Mapping[str, Any]) -> ReplayReport:
    """Validate and copy a replay capability report without restoring state."""
    if isinstance(report, (bool, bytes, str)) or not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")

    required = {
        "mode",
        "stored_fields",
        "unavailable_fields",
        "restored_fields",
        "discrepancies",
        "protocol_id",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"report fields are missing: {sorted(missing)}")
    conditional = {"repetitions", "uncertainty"}
    unknown = set(report) - required - conditional
    if unknown:
        raise ValueError(f"report has unknown fields: {sorted(unknown)}")

    mode = report["mode"]
    if not isinstance(mode, str) or mode not in REPLAY_MODES:
        raise ValueError(f"report mode must be one of {REPLAY_MODES}")
    if mode != "approximate-reset" and set(report) & conditional:
        raise ValueError("report repetitions and uncertainty are only valid for approximate-reset")
    protocol_id = report["protocol_id"]
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("report protocol_id must be nonempty")

    stored = _fields(report["stored_fields"], "stored_fields")
    unavailable = _fields(report["unavailable_fields"], "unavailable_fields")
    restored = _fields(report["restored_fields"], "restored_fields")
    stored_set = set(stored)
    unavailable_set = set(unavailable)
    restored_set = set(restored)
    overlap = stored_set & unavailable_set
    if overlap:
        raise ValueError(
            f"report stored_fields and unavailable_fields must be disjoint: {sorted(overlap)}"
        )
    missing_coverage = set(REQUIRED_REPLAY_FIELDS) - stored_set - unavailable_set
    if missing_coverage:
        raise ValueError(
            f"report fields must explicitly cover: {sorted(missing_coverage)}"
        )
    if not restored_set <= stored_set:
        raise ValueError("report restored_fields must be a subset of stored_fields")
    if mode == "exact-snapshot":
        missing_restored = set(REQUIRED_REPLAY_FIELDS) - restored_set
        unavailable_required = set(REQUIRED_REPLAY_FIELDS) & unavailable_set
        if missing_restored or unavailable_required:
            raise ValueError(
                "exact-snapshot requires every required replay field to be restored "
                "and no required field to be unavailable"
            )

    discrepancies = _discrepancies(report["discrepancies"])
    validated: dict[str, Any] = dict(report)
    validated["mode"] = mode
    validated["stored_fields"] = stored
    validated["unavailable_fields"] = unavailable
    validated["restored_fields"] = restored
    validated["discrepancies"] = discrepancies
    validated["protocol_id"] = protocol_id
    if mode == "approximate-reset":
        if "repetitions" not in report or "uncertainty" not in report:
            raise ValueError(
                "approximate-reset requires repetitions and uncertainty"
            )
        validated["repetitions"] = _repetitions(report["repetitions"])
        validated["uncertainty"] = _finite_mapping(
            report["uncertainty"], "uncertainty", nonempty=True
        )
    return validated  # type: ignore[return-value]


__all__ = [
    "REPLAY_MODES",
    "REQUIRED_REPLAY_FIELDS",
    "ReplayReport",
    "validate_replay_report",
]
