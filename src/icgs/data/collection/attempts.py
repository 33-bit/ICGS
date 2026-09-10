"""Validation and aggregate counts for recorded generation attempts.

The helper preserves every supplied attempt status in the caller-owned record;
it deliberately does not construct or execute a ``TimedEnvironment``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def attempt_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count attempts, successes, and invalid inputs without dropping failures.

    ``status`` remains recorder-owned provenance.  This boundary requires a
    nonempty canonical string with no surrounding whitespace.  It recognizes
    ``success`` and ``invalid-input`` for aggregate counters and leaves other
    valid statuses, such as ``timeout``, represented only in ``attempts``.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("attempt records must be a sequence")

    counts = {"attempts": len(records), "successes": 0, "invalid": 0}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"attempt record {index} must be a mapping")
        status = record.get("status")
        if not isinstance(status, str) or not status.strip():
            raise ValueError(f"attempt record {index} must have a nonempty status")
        if status != status.strip():
            raise ValueError(f"attempt record {index} status must be canonical")
        counts["successes"] += int(status == "success")
        counts["invalid"] += int(status == "invalid-input")
    return counts


__all__ = ["attempt_counts"]
