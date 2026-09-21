"""Attempt-based collection quota for phase 1.

Nominal collection runs until ``nominal_success_target`` successful contexts.
Perturbed collection runs until ``perturbed_attempt_target`` valid attempts.
Crashes and invalid observations do not count. The 70/30 mixture is a
training-time sampler over D_temporal/D_dyn transitions, not this quota.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from icgs.data.collection.v3.perturbations import (
    INVALID_OBSERVATION,
    SIMULATOR_CRASH,
    SUCCESS,
    VALID_FAILURE,
)
from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.steps import get_v3_program

_VALID = frozenset({SUCCESS, VALID_FAILURE})


@dataclass(frozen=True)
class CollectionQuota:
    program_id: str
    split: str
    nominal_success_target: int
    perturbed_attempt_target: int
    max_nominal_attempts: int
    max_perturbed_attempts: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuotaCounts:
    nominal_successes: int = 0
    nominal_valid_failures: int = 0
    nominal_crashes: int = 0
    nominal_invalid: int = 0
    perturbed_successes: int = 0
    perturbed_valid_failures: int = 0
    perturbed_crashes: int = 0
    perturbed_invalid: int = 0
    perturbed_by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def nominal_valid_attempts(self) -> int:
        return self.nominal_successes + self.nominal_valid_failures

    @property
    def perturbed_valid_attempts(self) -> int:
        return self.perturbed_successes + self.perturbed_valid_failures

    @property
    def nominal_launched(self) -> int:
        return (
            self.nominal_successes
            + self.nominal_valid_failures
            + self.nominal_crashes
            + self.nominal_invalid
        )

    @property
    def perturbed_launched(self) -> int:
        return (
            self.perturbed_successes
            + self.perturbed_valid_failures
            + self.perturbed_crashes
            + self.perturbed_invalid
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["nominal_valid_attempts"] = self.nominal_valid_attempts
        payload["perturbed_valid_attempts"] = self.perturbed_valid_attempts
        payload["nominal_launched"] = self.nominal_launched
        payload["perturbed_launched"] = self.perturbed_launched
        return payload


def quota_for_program(program_id: str) -> CollectionQuota:
    spec = get_v3_program(program_id)
    if spec.split == "train":
        nominal = V3_PROTOCOL.train_nominal_successes_per_program
        perturbed = V3_PROTOCOL.train_perturbed_attempts_per_program
    else:
        nominal = V3_PROTOCOL.eval_contexts_per_composition
        perturbed = V3_PROTOCOL.eval_perturbed_attempts_per_program
    return CollectionQuota(
        program_id=program_id,
        split=spec.split,
        nominal_success_target=nominal,
        perturbed_attempt_target=perturbed,
        max_nominal_attempts=nominal * V3_PROTOCOL.quota_attempt_cap_multiplier,
        max_perturbed_attempts=max(perturbed, 1) * V3_PROTOCOL.quota_attempt_cap_multiplier if perturbed else 0,
    )


def record_outcome(
    counts: QuotaCounts,
    episode_kind: str,
    result_class: str,
    intervention_kind: str | None = None,
) -> QuotaCounts:
    kind = "perturbed" if episode_kind == "perturbed" else "nominal"
    if result_class == SUCCESS:
        if kind == "nominal":
            counts.nominal_successes += 1
        else:
            counts.perturbed_successes += 1
    elif result_class == VALID_FAILURE:
        if kind == "nominal":
            counts.nominal_valid_failures += 1
        else:
            counts.perturbed_valid_failures += 1
    elif result_class == SIMULATOR_CRASH:
        if kind == "nominal":
            counts.nominal_crashes += 1
        else:
            counts.perturbed_crashes += 1
    else:
        if kind == "nominal":
            counts.nominal_invalid += 1
        else:
            counts.perturbed_invalid += 1
    if kind == "perturbed" and result_class in _VALID and intervention_kind:
        counts.perturbed_by_kind[intervention_kind] = counts.perturbed_by_kind.get(intervention_kind, 0) + 1
    return counts


def quota_met(quota: CollectionQuota, counts: QuotaCounts) -> bool:
    return (
        counts.nominal_successes >= quota.nominal_success_target
        and counts.perturbed_valid_attempts >= quota.perturbed_attempt_target
    )


def next_episode_kind(quota: CollectionQuota, counts: QuotaCounts) -> str | None:
    """Return the next attempt kind, or None if the quota is met or the cap is hit."""
    if counts.nominal_successes < quota.nominal_success_target:
        if counts.nominal_launched >= quota.max_nominal_attempts:
            return None
        return "nominal"
    if counts.perturbed_valid_attempts < quota.perturbed_attempt_target:
        if counts.perturbed_launched >= quota.max_perturbed_attempts:
            return None
        return "perturbed"
    return None


def valid_attempt_mixture(counts: QuotaCounts) -> dict[str, Any]:
    valid = counts.nominal_valid_attempts + counts.perturbed_valid_attempts
    nominal_fraction = counts.nominal_valid_attempts / valid if valid else 0.0
    perturbed_fraction = counts.perturbed_valid_attempts / valid if valid else 0.0
    return {
        "nominal": counts.nominal_valid_attempts,
        "perturbed": counts.perturbed_valid_attempts,
        "nominal_fraction": nominal_fraction,
        "perturbed_fraction": perturbed_fraction,
        "measured_on": "collection_valid_attempts",
        "training_mixture_target": list(V3_PROTOCOL.warmup_mixture),
        "training_mixture_measured_on": V3_PROTOCOL.mixture_measured_on,
        "training_mixture_views": list(V3_PROTOCOL.mixture_views),
    }


def counts_from_manifest(manifest: Mapping[str, Any], program_id: str) -> QuotaCounts:
    counts = QuotaCounts()
    seen: set[str] = set()

    def _ingest(row: Mapping[str, Any], *, default_result: str) -> None:
        if row.get("program_id") != program_id:
            return
        episode_id = str(row.get("episode_id") or row.get("attempt_id") or "")
        if episode_id:
            if episode_id in seen:
                return
            seen.add(episode_id)
        kind = str(row.get("episode_kind") or "nominal")
        result = str(row.get("outcome") or row.get("result_class") or default_result)
        if row.get("status") == "failed" and result not in {SIMULATOR_CRASH, INVALID_OBSERVATION, VALID_FAILURE, SUCCESS}:
            result = SIMULATOR_CRASH if not row.get("valid_observation_retained") else VALID_FAILURE
        intervention = row.get("intervention_kind") or row.get("intervention_id")
        if isinstance(intervention, str) and intervention.endswith("_v1"):
            intervention = intervention[: -len("_v1")]
        record_outcome(counts, kind, result, intervention if isinstance(intervention, str) else None)

    for row in list(manifest.get("episodes") or []):
        if isinstance(row, Mapping):
            _ingest(row, default_result=SUCCESS)
    for row in list(manifest.get("failure_attempts") or []) + list(manifest.get("quarantined") or []):
        if isinstance(row, Mapping):
            _ingest(row, default_result=VALID_FAILURE)
    return counts
