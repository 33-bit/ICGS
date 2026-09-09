"""Pure split and causal-history helpers for executed episode records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from icgs.contracts.method import ExecutedTransition
from icgs.data.schemas.episodes import validate_episode


_SPLITS = frozenset({"train", "dev", "test"})


def validate_split_lineage(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require each declared literal ``lineage_id`` to remain in one split.

    P02 requires descendant-aware splitting too, but no approved root/parent or
    ancestry-closure field currently exists.  This helper intentionally does not
    infer ancestry from identifiers, asset names, or augmentation naming.
    """

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("split lineage rows must be a sequence")
    splits_by_lineage: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"split lineage row {index} must be a mapping")
        if "lineage_id" not in row or "split" not in row:
            raise ValueError(f"split lineage row {index} must include lineage_id and split")
        lineage_id, split = row["lineage_id"], row["split"]
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError(f"split lineage row {index} has an invalid lineage_id")
        if split not in _SPLITS:
            raise ValueError(f"split lineage row {index} has an invalid split")
        previous = splits_by_lineage.setdefault(lineage_id, split)
        if previous != split:
            raise ValueError("source lineage crosses splits")


def causal_prefix(episode: Mapping[str, Any], boundary: int) -> tuple[ExecutedTransition, ...]:
    """Return only executed transitions whose successor is known at ``boundary``.

    ``boundary`` is an integer boundary index, not a transition-array position.
    Consequently the returned history ends at the current boundary and cannot
    expose the command or measured observation from a later interval.
    """

    if isinstance(boundary, bool) or not isinstance(boundary, int):
        raise ValueError("boundary must be an integer")
    validate_episode(episode)
    transitions = tuple(episode["transitions"])
    first_boundary = transitions[0].before.boundary
    final_boundary = transitions[-1].after.boundary
    if not first_boundary <= boundary <= final_boundary:
        raise ValueError("boundary must be within the episode's executed boundaries")
    return tuple(transition for transition in transitions if transition.after.boundary <= boundary)


__all__ = ["causal_prefix", "validate_split_lineage"]
