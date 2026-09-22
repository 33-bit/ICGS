"""Central quota and uniqueness authority for distributed generation collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from icgs.data.collection.generation.batch import AttemptPlanner, bounds_from_row
from icgs.data.collection.generation.distributed_contracts import GenerationJob, RunConfig, WorkerResult
from icgs.data.collection.generation.quota import (
    QuotaCounts,
    counts_from_manifest,
    quota_for_program,
    quota_met,
    record_outcome,
)


@dataclass(frozen=True)
class PlannerSnapshot:
    run_id: str
    status: str
    counts_by_program: dict[str, dict[str, Any]]
    inflight_job_ids: tuple[str, ...]
    completed_job_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DistributedPlanner:
    def __init__(
        self,
        run: RunConfig,
        approved_rows: Mapping[str, Mapping[str, Any]],
        manifest: Mapping[str, Any],
        inflight_jobs: Sequence[GenerationJob],
    ) -> None:
        if not approved_rows:
            raise ValueError("approved_rows must be nonempty")
        self.run = run
        self.approved_rows = {str(key): dict(value) for key, value in approved_rows.items()}
        self._program_ids = tuple(sorted(self.approved_rows))
        self._counts = {
            program_id: counts_from_manifest(manifest, program_id)
            for program_id in self._program_ids
        }
        self._planners: dict[str, AttemptPlanner] = {}
        for program_id in self._program_ids:
            row = self.approved_rows[program_id]
            quota = quota_for_program(program_id)
            planner = AttemptPlanner(
                program_id,
                bounds=bounds_from_row(row),
                asset_family_id=row.get("asset_family_id"),
                n_perturbed=quota.perturbed_attempt_target,
            )
            for existing in list(manifest.get("episodes") or []) + list(manifest.get("failure_attempts") or []):
                if isinstance(existing, Mapping) and existing.get("program_id") == program_id:
                    planner.remember_row(existing)
            self._planners[program_id] = planner
        self._inflight: dict[str, GenerationJob] = {}
        for job in inflight_jobs:
            if job.program_id not in self._planners:
                raise ValueError(f"in-flight job has unapproved program: {job.program_id}")
            if job.job_id in self._inflight:
                raise ValueError(f"duplicate in-flight job: {job.job_id}")
            self._planners[job.program_id].remember_plan(job.plan)
            self._inflight[job.job_id] = job
        self._completed: set[str] = set()
        self._cursor = 0

    @classmethod
    def from_manifest(
        cls,
        run: RunConfig,
        approved_rows: Mapping[str, Mapping[str, Any]],
        manifest: Mapping[str, Any],
        inflight_jobs: Sequence[GenerationJob] = (),
    ) -> "DistributedPlanner":
        return cls(run, approved_rows, manifest, inflight_jobs)

    def _inflight_count(self, program_id: str, episode_kind: str) -> int:
        return sum(
            job.program_id == program_id and job.plan.episode_kind == episode_kind
            for job in self._inflight.values()
        )

    def _candidate(self, program_id: str) -> tuple[str, float] | None:
        quota = quota_for_program(program_id)
        counts = self._counts[program_id]
        nominal_inflight = self._inflight_count(program_id, "nominal")
        if counts.nominal_successes < quota.nominal_success_target:
            remaining = quota.nominal_success_target - counts.nominal_successes - nominal_inflight
            cap_remaining = quota.max_nominal_attempts - counts.nominal_launched - nominal_inflight
            if remaining > 0 and cap_remaining > 0:
                return "nominal", (quota.nominal_success_target - counts.nominal_successes) / quota.nominal_success_target
            return None
        perturbed_inflight = self._inflight_count(program_id, "perturbed")
        if counts.perturbed_valid_attempts < quota.perturbed_attempt_target:
            remaining = quota.perturbed_attempt_target - counts.perturbed_valid_attempts - perturbed_inflight
            cap_remaining = quota.max_perturbed_attempts - counts.perturbed_launched - perturbed_inflight
            if remaining > 0 and cap_remaining > 0:
                denominator = max(quota.perturbed_attempt_target, 1)
                return "perturbed", (quota.perturbed_attempt_target - counts.perturbed_valid_attempts) / denominator
        return None

    def _select_program(self) -> tuple[str, str] | None:
        candidates: list[tuple[float, int, str, str]] = []
        total = len(self._program_ids)
        for offset in range(total):
            position = (self._cursor + offset) % total
            program_id = self._program_ids[position]
            candidate = self._candidate(program_id)
            if candidate is not None:
                episode_kind, deficit = candidate
                candidates.append((deficit, -offset, program_id, episode_kind))
        if not candidates:
            return None
        _, _, program_id, episode_kind = max(candidates)
        self._cursor = (self._program_ids.index(program_id) + 1) % total
        return program_id, episode_kind

    def next_job(self) -> GenerationJob | None:
        selected = self._select_program()
        if selected is None:
            return None
        program_id, episode_kind = selected
        plan = self._planners[program_id].next_plan(episode_kind)
        job_id = f"job-{self.run.run_id}-{plan.episode_id}"
        job = GenerationJob.create(
            job_id=job_id,
            run_id=self.run.run_id,
            attempt_id=f"att-{plan.episode_id}",
            episode_id=plan.episode_id,
            program_id=program_id,
            plan=plan,
            code_revision=self.run.code_revision,
            manifest_sha256=self.run.approved_manifest_sha256,
            output_root=str(Path(self.run.run_root) / "staging"),
        )
        if job_id in self._inflight or job_id in self._completed:
            raise ValueError(f"duplicate generated job ID: {job_id}")
        self._inflight[job_id] = job
        return job

    def record_result(self, result: WorkerResult, provenance: Mapping[str, Any]) -> None:
        if result.job_id in self._completed:
            return
        job = self._inflight.get(result.job_id)
        if job is None:
            raise ValueError(f"result does not match an in-flight job: {result.job_id}")
        if (result.attempt_id, result.program_id) != (job.attempt_id, job.program_id):
            raise ValueError("result identity does not match planned job")
        intervention = provenance.get("intervention_type") or provenance.get("intervention_id")
        if not intervention and job.plan.intervention:
            intervention = job.plan.intervention.get("kind") or job.plan.intervention.get("intervention_type")
        if isinstance(intervention, str) and intervention.endswith("_v1"):
            intervention = intervention[:-3]
        record_outcome(
            self._counts[job.program_id],
            job.plan.episode_kind,
            result.outcome,
            intervention if isinstance(intervention, str) else None,
        )
        del self._inflight[result.job_id]
        self._completed.add(result.job_id)

    def counts(self, program_id: str) -> QuotaCounts:
        if program_id not in self._counts:
            raise KeyError(program_id)
        return self._counts[program_id]

    def quota_complete(self) -> bool:
        return all(quota_met(quota_for_program(pid), self._counts[pid]) for pid in self._program_ids)

    def terminal_status(self) -> str:
        if self.quota_complete():
            return "COMPLETE"
        if self._inflight:
            return "RUNNING"
        if any(self._candidate(program_id) is not None for program_id in self._program_ids):
            return "RUNNING"
        return "INCOMPLETE"

    def snapshot(self) -> PlannerSnapshot:
        return PlannerSnapshot(
            run_id=self.run.run_id,
            status=self.terminal_status(),
            counts_by_program={pid: self._counts[pid].as_dict() for pid in self._program_ids},
            inflight_job_ids=tuple(sorted(self._inflight)),
            completed_job_ids=tuple(sorted(self._completed)),
        )


__all__ = ["DistributedPlanner", "PlannerSnapshot"]
