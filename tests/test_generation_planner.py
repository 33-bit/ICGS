from __future__ import annotations

import json
from pathlib import Path

from icgs.data.collection.generation.distributed_contracts import RunConfig, WorkerResult
from icgs.data.collection.generation.distributed_planner import DistributedPlanner


def _run() -> RunConfig:
    return RunConfig(
        run_id="run-1", run_root="/content/run",
        code_revision="a" * 40, approved_manifest_sha256="b" * 64,
    )


def _rows() -> dict[str, dict]:
    data = json.loads(Path("artifacts/composition/approved_composition_manifest.json").read_text())
    return {row["program_id"]: row for row in data["catalog"]}


def _result(job, outcome: str) -> WorkerResult:
    episode = job.episode_id if outcome in {"success", "valid_failure"} else None
    timeline = {"actions": 2, "observations": 3, "durations": 2} if episode else None
    return WorkerResult(
        job_id=job.job_id, attempt_id=job.attempt_id, episode_id=episode,
        program_id=job.program_id, outcome=outcome, result_dir=f"/results/{job.job_id}",
        file_sha256={}, timeline=timeline,
    )


def test_two_hundred_job_fill_has_unique_signatures_and_ids():
    planner = DistributedPlanner.from_manifest(_run(), _rows(), {"episodes": [], "failure_attempts": []})
    jobs = [planner.next_job() for _ in range(200)]
    assert all(job is not None for job in jobs)
    assert len({job.job_id for job in jobs}) == 200
    assert len({job.plan.randomization["scene_signature"] for job in jobs}) == 200


def test_valid_failure_counts_only_for_perturbed_quota():
    rows = {"T01": _rows()["T01"]}
    planner = DistributedPlanner.from_manifest(_run(), rows, {"episodes": [], "failure_attempts": []})
    nominal = planner.next_job()
    assert nominal is not None and nominal.plan.episode_kind == "nominal"
    planner.record_result(_result(nominal, "valid_failure"), {"episode_kind": "nominal"})
    assert planner.counts("T01").nominal_successes == 0

    manifest = {
        "episodes": [
            {
                "episode_id": f"old-{index}", "program_id": "T01", "outcome": "success",
                "episode_kind": "nominal", "scene_signature": f"old-sig-{index}",
                "scene_seed": index, "episode_index": index,
            }
            for index in range(200)
        ],
        "failure_attempts": [],
    }
    resumed = DistributedPlanner.from_manifest(_run(), rows, manifest)
    perturbed = resumed.next_job()
    assert perturbed is not None and perturbed.plan.episode_kind == "perturbed"
    resumed.record_result(_result(perturbed, "valid_failure"), {
        "episode_kind": "perturbed", "intervention_id": "pause_hold_v1",
    })
    assert resumed.counts("T01").perturbed_valid_attempts == 1


def test_resume_never_reuses_remote_or_inflight_signature():
    rows = {"T01": _rows()["T01"]}
    initial = DistributedPlanner.from_manifest(_run(), rows, {"episodes": [], "failure_attempts": []})
    first = initial.next_job()
    manifest = {
        "episodes": [{
            "episode_id": "old-0", "program_id": "T01", "outcome": "success",
            "episode_kind": "nominal", "scene_signature": "known-remote",
            "scene_seed": 4, "episode_index": 0,
        }],
        "failure_attempts": [],
    }
    resumed = DistributedPlanner.from_manifest(_run(), rows, manifest, inflight_jobs=(first,))
    second = resumed.next_job()
    assert second is not None
    assert second.plan.randomization["scene_signature"] not in {
        "known-remote", first.plan.randomization["scene_signature"],
    }
    assert second.plan.episode_index > first.plan.episode_index


def test_resume_restores_serialized_nominal_plan_for_perturbation_source():
    rows = {"T01": _rows()["T01"]}
    initial = DistributedPlanner.from_manifest(_run(), rows, {"episodes": [], "failure_attempts": []})
    nominal = initial.next_job()
    assert nominal is not None
    manifest = {
        "episodes": [{
            "episode_id": nominal.episode_id,
            "attempt_id": nominal.attempt_id,
            "program_id": nominal.program_id,
            "outcome": "success",
            "episode_kind": "nominal",
            "attempt_plan": nominal.plan.as_dict(),
        }] + [{
            "episode_id": f"old-{index}",
            "attempt_id": f"att-old-{index}",
            "program_id": "T01",
            "outcome": "success",
            "episode_kind": "nominal",
            "scene_signature": f"old-signature-{index}",
            "scene_seed": index,
            "episode_index": index,
        } for index in range(1, 200)],
        "failure_attempts": [],
    }

    resumed = DistributedPlanner.from_manifest(_run(), rows, manifest)
    perturbed = resumed.next_job()

    assert perturbed is not None
    assert perturbed.plan.episode_kind == "perturbed"
    assert perturbed.plan.intervention is not None
    assert perturbed.plan.intervention["source_episode_id"] == nominal.episode_id


def test_inflight_jobs_bound_optimistic_quota_overplanning():
    rows = {"T01": _rows()["T01"]}
    manifest = {
        "episodes": [
            {
                "episode_id": f"old-{index}", "program_id": "T01", "outcome": "success",
                "episode_kind": "nominal", "scene_signature": f"old-sig-{index}",
                "scene_seed": index, "episode_index": index,
            }
            for index in range(199)
        ],
        "failure_attempts": [],
    }
    planner = DistributedPlanner.from_manifest(_run(), rows, manifest)
    only_remaining = planner.next_job()
    assert only_remaining is not None
    assert planner.next_job() is None
    planner.record_result(_result(only_remaining, "valid_failure"), {"episode_kind": "nominal"})
    assert planner.next_job() is not None


def test_completed_manifest_reports_complete():
    rows = {"V01": _rows()["V01"]}
    episodes = [
        {
            "episode_id": f"n-{index}", "program_id": "V01", "outcome": "success",
            "episode_kind": "nominal", "scene_signature": f"n-sig-{index}",
            "scene_seed": index, "episode_index": index,
        }
        for index in range(100)
    ] + [
        {
            "episode_id": f"p-{index}", "program_id": "V01", "outcome": "valid_failure",
            "episode_kind": "perturbed", "intervention_id": "pause_hold_v1",
            "scene_signature": f"p-sig-{index}", "scene_seed": 1000 + index,
            "episode_index": 100 + index,
        }
        for index in range(20)
    ]
    planner = DistributedPlanner.from_manifest(_run(), rows, {"episodes": episodes, "failure_attempts": []})
    assert planner.quota_complete() is True
    assert planner.terminal_status() == "COMPLETE"
    assert planner.next_job() is None
