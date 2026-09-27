from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import os
import hashlib
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import generation_launch
from scripts import generation_watchdog
from scripts import generation_coordinator as generation_coordinator_module
from scripts import generation_launch
from scripts.generation_launch import validate_smoke_receipt
from scripts.generation_coordinator import (
    CoordinatorControlPlane,
    CoordinatorLock,
    _credential_path,
    _inflight_jobs_from_queue,
    load_remote_manifest,
    _merge_manifests,
    _validate_resume_manifest,
)
from scripts.generation_watchdog import (
    _coordinator_lock_is_free,
    _pid_matches,
    heartbeat_health,
    reconcile_processes,
)
from icgs.data.collection.generation.distributed_contracts import (
    ArchiveProfileConfig,
    GenerationJob,
    GenerationRuntimeConfig,
    RunConfig,
    WorkerResult,
)
from icgs.data.collection.generation.batch import AttemptPlan
from icgs.data.collection.generation.distributed_planner import DistributedPlanner
from icgs.data.collection.generation.distributed_queue import FilesystemJobQueue
from icgs.data.collection.generation.distributed_validation import ValidatedResult, validate_closed_result


def _runtime_config(tmp_path: Path, *, publication_enabled: bool = False):
    repo_root = tmp_path / "repo"
    simulator_root = tmp_path / "simulator"
    rlbench_root = tmp_path / "rlbench"
    run_root = tmp_path / "run"
    python_executable = tmp_path / "venv" / "bin" / "python"
    for directory in (repo_root, simulator_root, rlbench_root, run_root, python_executable.parent):
        directory.mkdir(parents=True, exist_ok=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    return GenerationRuntimeConfig.from_dict({
        "machine": {
            "repo_root": str(repo_root),
            "python_executable": str(python_executable),
            "simulator_root": str(simulator_root),
            "rlbench_root": str(rlbench_root),
            "display_base": 41,
            "display_width": 1366,
            "display_height": 768,
            "simulator_slots": 2,
            "worker_timeout_s": 47,
        },
        "run": {
            "run_id": "control-test",
            "run_root": str(run_root),
            "worker_count": 2,
            "publish_interval_s": 91,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "validation/control-test",
            "publication_enabled": publication_enabled,
            "validation_mode": True,
        },
    })


def _required_api(name: str):
    assert hasattr(generation_launch, name), f"generation_launch.{name} is required"
    return getattr(generation_launch, name)


def _archive_resume_profile() -> ArchiveProfileConfig:
    return ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="receipt_only")


@pytest.mark.parametrize("workers_only", [False, True])
def test_archive_launch_requires_positive_writer_cap_before_spawn(tmp_path: Path, monkeypatch, workers_only: bool):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["archive_profile"] = ArchiveProfileConfig().as_dict()
    path = tmp_path / "archive-runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spawned = []
    monkeypatch.setattr(generation_launch.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    args = ["--runtime-config", str(path), "--approved-manifest", str(tmp_path / "approved.json")]
    if workers_only:
        args.extend(["--workers-only", "--run-config", str(tmp_path / "run.json")])
    with pytest.raises(ValueError, match="archive.*max_result_bytes.*positive"):
        generation_launch.main(args)
    assert spawned == []


def _archive_resume_plan(*, episode_id: str, episode_kind: str = "nominal") -> AttemptPlan:
    return AttemptPlan(
        program_id="T01",
        split="train",
        episode_index=1,
        episode_id=episode_id,
        episode_kind=episode_kind,
        scene_seed=17,
        collection_seed=20260920,
        randomization={
            "scene_signature": f"signature-{episode_id}",
            "asset_instance_id": "T01-asset-17",
            "asset_family_id": "family-1",
            "scene_seed": 17,
        },
        intervention=(
            {"kind": "pause_hold", "intervention_id": "pause_hold_v1"}
            if episode_kind == "perturbed" else None
        ),
    )


def _archive_resume_row(
    profile: ArchiveProfileConfig,
    *,
    episode_id: str,
    outcome: str = "success",
    episode_kind: str = "nominal",
) -> dict:
    from icgs.data.collection.generation.diversity import train_subset_for_sample

    plan = _archive_resume_plan(episode_id=episode_id, episode_kind=episode_kind)
    is_episode = outcome in {"success", "valid_failure"}
    attempt_id = f"att-{plan.episode_id}"
    identity = episode_id if is_episode else attempt_id
    archive_ref = f"episodes/T01/{episode_id}" if is_episode else f"attempts/T01/{attempt_id}"
    manifest_name = "episode.manifest.json" if is_episode else "attempt.manifest.json"
    subset = train_subset_for_sample(plan.randomization)
    return {
        "archive_ref": archive_ref,
        "archive_manifest": f"{archive_ref}/{manifest_name}",
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "dataset_identity": profile.dataset_identity,
        "job_id": f"job-old-{identity}",
        "run_id": "old-run",
        "manifest_sha256": "b" * 64,
        "retry_generation": 0,
        "source_run_id": "old-run",
        "code_revision": "a" * 40,
        "preprocessing_identity": "resume_fixture_v1",
        "file_sha256": {
            manifest_name: "1" * 64,
            "artifact_manifest.json": "2" * 64,
            "debug.json": "3" * 64,
            "data/chunk-00000.npz": "4" * 64,
        },
            "attempt_id": attempt_id,
        "episode_id": episode_id if is_episode else None,
        "program_id": "T01",
        "split": "train",
        "subset": subset,
        "episode_kind": episode_kind,
        "scene_signature": plan.randomization["scene_signature"],
        "scene_seed": plan.scene_seed,
        "episode_index": plan.episode_index,
        "asset_instance_id": "T01-asset-17",
        "asset_family_id": "family-1",
        "source_lineage_id": "lineage-1" if is_episode else None,
        "outcome": outcome,
        "intervention_id": (plan.intervention or {}).get("intervention_id"),
        "source_episode_id": (plan.intervention or {}).get("source_episode_id"),
        "base_episode_id": (plan.intervention or {}).get("base_episode_id"),
        "attempt_plan": plan.as_dict(),
    }


def _archive_resume_manifest(profile: ArchiveProfileConfig) -> dict:
    return {
        "manifest_version": 3,
        "source_run_ids": ["old-run"],
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "view_status": "PROVISIONAL",
        "episodes": [
            _archive_resume_row(profile, episode_id="episode-old-success", outcome="success"),
            _archive_resume_row(profile, episode_id="episode-old-failure", outcome="valid_failure"),
        ],
        "failure_attempts": [
            _archive_resume_row(profile, episode_id="episode-old-crash", outcome="simulator_crash"),
        ],
    }


def test_launcher_builds_workers_from_configured_limits_and_paths(tmp_path: Path):
    config = _runtime_config(tmp_path)
    approved_manifest = str(tmp_path / "approved.json")
    try:
        commands = _required_api("build_worker_commands")(config, approved_manifest)
    except TypeError as exc:
        pytest.fail(f"build_worker_commands must accept a runtime config: {exc}")

    assert len(commands) == 2
    assert [command[command.index("--worker-id") + 1] for command in commands] == ["000", "001"]
    assert [command[command.index("--server-num") + 1] for command in commands] == ["41", "42"]
    assert all(
        command[command.index("-s") + 1].startswith("-screen 0 1366x768x24")
        for command in commands
    )
    assert all("+extension GLX" in command[command.index("-s") + 1] for command in commands)
    assert all("+render" in command[command.index("-s") + 1] for command in commands)
    assert all(config.machine.python_executable in command for command in commands)
    assert all(
        str(Path(config.machine.repo_root) / "scripts" / "generation_worker.py") in command
        for command in commands
    )
    assert all(
        command[command.index("--runtime-config") + 1]
        == str(Path(config.run.run_root) / "control" / "runtime_config.json")
        for command in commands
    )
    assert all(command[command.index("--approved-manifest") + 1] == approved_manifest for command in commands)
    assert all("/content" not in " ".join(command) for command in commands)


def test_launcher_worker_commands_can_use_host_local_runtime_snapshot(tmp_path: Path):
    config = _runtime_config(tmp_path)
    host_runtime_path = tmp_path / "worker-a-runtime.json"
    commands = _required_api("build_worker_commands")(
        config, tmp_path / "approved.json", runtime_config_path=host_runtime_path,
    )

    assert all(
        command[command.index("--runtime-config") + 1] == str(host_runtime_path)
        for command in commands
    )


def test_launcher_script_runs_as_portable_repository_entrypoint():
    result = subprocess.run(
        [sys.executable, "-B", "scripts/generation_launch.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--runtime-config" in result.stdout


def test_worker_environment_does_not_receive_hf_credentials(tmp_path: Path):
    config = _runtime_config(tmp_path)
    environment = _required_api("build_process_environment")(
        config,
        base={"ICGS_HF_TOKEN_PATH": "/secret/token", "HF_TOKEN": "secret", "KEEP": "yes"},
    )

    assert "ICGS_HF_TOKEN_PATH" not in environment
    assert "HF_TOKEN" not in environment
    assert environment["KEEP"] == "yes"


def test_launcher_persists_runtime_config_digest_before_starting_children(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    runtime_config_path = tmp_path / "runtime.json"
    runtime_config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    approved_manifest = tmp_path / "approved.json"
    approved_manifest.write_text("{}\n", encoding="utf-8")
    smoke_receipt = tmp_path / "smoke.json"
    validation_plan = Path("tests/fixtures/generation_validation_config.json")
    from icgs.data.collection.generation.steps import GENERATION_PROGRAMS

    smoke_receipt.write_text(json.dumps({
        "results": [
            {"program_id": program_id, "result_class": "success", "timeline_ok": True}
            for program_id in GENERATION_PROGRAMS
        ],
    }), encoding="utf-8")
    run_json = Path(config.run.run_root) / "control" / "run.json"
    runtime_snapshot = run_json.parent / "runtime_config.json"
    calls = []

    class Process:
        def __init__(self, pid: int):
            self.pid = pid

        def wait(self):
            return 0

    def popen(command, **kwargs):
        assert run_json.is_file()
        assert runtime_snapshot.is_file()
        payload = json.loads(run_json.read_text(encoding="utf-8"))
        snapshot_bytes = runtime_snapshot.read_bytes()
        assert payload["runtime_config"] == json.loads(snapshot_bytes)
        assert payload["runtime_config_sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
        calls.append(command)
        return Process(1000 + len(calls))

    monkeypatch.setattr(generation_launch.subprocess, "Popen", popen)
    args = [
        "--runtime-config", str(runtime_config_path),
        "--approved-manifest", str(approved_manifest),
        "--code-revision", "a" * 40,
        "--smoke-receipt", str(smoke_receipt),
        "--validation-plan", str(validation_plan),
        "--detach",
    ]
    try:
        result = generation_launch.main(args)
    except TypeError as exc:
        pytest.fail(f"generation_launch.main must accept runtime-config arguments: {exc}")
    except SystemExit as exc:
        pytest.fail(f"generation_launch rejected runtime-config arguments: {exc}")

    assert result == 0
    assert len(calls) == 4
    coordinator_calls = [
        command
        for command in calls
        if "generation_coordinator.py" in " ".join(command)
    ]
    assert len(coordinator_calls) == 1
    coordinator_command = coordinator_calls[0]
    assert coordinator_command[coordinator_command.index("--runtime-config") + 1] == str(runtime_snapshot)
    stored = json.loads(run_json.read_text(encoding="utf-8"))
    assert stored["run"]["worker_count"] == 2
    assert stored["run"]["validation_max_jobs"] == 7
    assert stored["runtime_config"] == config.as_dict()


def test_launcher_resume_preflight_reads_manifest_before_starting_workers(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    payload = config.as_dict()
    payload["run"]["resume_from_hf"] = True
    config = GenerationRuntimeConfig.from_dict(payload)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    approved_manifest = tmp_path / "approved.json"
    approved_manifest.write_text(
        Path("artifacts/composition/approved_composition_manifest.json").read_text(),
        encoding="utf-8",
    )
    smoke_receipt = tmp_path / "smoke.json"
    from icgs.data.collection.generation.steps import GENERATION_PROGRAMS
    smoke_receipt.write_text(json.dumps({"results": [
        {"program_id": item, "result_class": "success", "timeline_ok": True}
        for item in GENERATION_PROGRAMS
    ]}), encoding="utf-8")
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    remote = tmp_path / "dataset_manifest.json"
    remote.write_text(json.dumps({
        "manifest_version": 3,
        "source_run_ids": ["old-run"],
        "episodes": [],
        "failure_attempts": [],
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(generation_launch, "hf_hub_download", lambda **kwargs: str(remote))
    monkeypatch.setattr(
        generation_launch.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(pid=9000, wait=lambda: 0),
    )

    result = generation_launch.main([
        "--runtime-config", str(runtime_path),
        "--approved-manifest", str(approved_manifest),
        "--code-revision", "a" * 40,
        "--smoke-receipt", str(smoke_receipt),
        "--hf-token-path", str(token),
        "--detach",
    ])

    assert result == 0
    assert calls
    bootstrap = json.loads(
        (Path(config.run.run_root) / "control" / "resume_bootstrap.json").read_text()
    )
    assert bootstrap["source_run_ids"] == ["old-run"]


def test_launcher_resume_preflight_fails_before_spawning_workers(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    payload = config.as_dict()
    payload["run"]["resume_from_hf"] = True
    config = GenerationRuntimeConfig.from_dict(payload)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    approved_manifest = tmp_path / "approved.json"
    approved_manifest.write_text(
        Path("artifacts/composition/approved_composition_manifest.json").read_text(),
        encoding="utf-8",
    )
    smoke_receipt = tmp_path / "smoke.json"
    from icgs.data.collection.generation.steps import GENERATION_PROGRAMS
    smoke_receipt.write_text(json.dumps({"results": [
        {"program_id": item, "result_class": "success", "timeline_ok": True}
        for item in GENERATION_PROGRAMS
    ]}), encoding="utf-8")
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        generation_launch,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(OSError("manifest unavailable")),
    )
    monkeypatch.setattr(
        generation_launch.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append(command),
    )

    with pytest.raises(RuntimeError, match="resume_from_hf"):
        generation_launch.main([
            "--runtime-config", str(runtime_path),
            "--approved-manifest", str(approved_manifest),
            "--code-revision", "a" * 40,
            "--smoke-receipt", str(smoke_receipt),
            "--hf-token-path", str(token),
            "--detach",
        ])
    assert calls == []


def test_workers_only_launcher_starts_host_scope_without_coordinator(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path)
    payload = config.as_dict()
    payload["machine"].update({"host_id": "worker-a", "worker_ids": ["000"]})
    payload["machine"]["simulator_slots"] = 1
    payload["run"].update({"worker_count": 2, "distribution_mode": "shared_filesystem"})
    config = GenerationRuntimeConfig.from_dict(payload)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    run_config = Path(config.run.run_root) / "control" / "run.json"
    run_config.parent.mkdir(parents=True, exist_ok=True)
    run_config.write_text(json.dumps({
        "run": {"run_root": config.run.run_root},
        "coordinator_worker_ids": ["001"],
    }), encoding="utf-8")
    approved = tmp_path / "approved.json"
    approved.write_text("{}\n", encoding="utf-8")
    calls = []

    class Process:
        def __init__(self, pid):
            self.pid = pid

    monkeypatch.setattr(
        generation_launch.subprocess,
        "Popen",
        lambda command, **kwargs: (calls.append(command) or Process(5000 + len(calls))),
    )
    result = generation_launch.main([
        "--runtime-config", str(runtime_path),
        "--approved-manifest", str(approved),
        "--run-config", str(run_config),
        "--workers-only",
    ])

    assert result == 0
    assert len(calls) == 2
    assert all("generation_coordinator.py" not in " ".join(command) for command in calls)
    receipt = json.loads(
        (Path(config.run.run_root) / "control" / "worker-launch-worker-a.json").read_text()
    )
    assert receipt["worker_ids"] == ["000"]
    assert receipt["runtime_config_path"] == str(
        Path(config.run.run_root) / "control" / "runtime_config-worker-a.json"
    )
    assert Path(receipt["runtime_config_path"]).is_file()


def test_remote_resume_manifest_merge_rejects_immutable_conflict():
    remote = {"manifest_version": 3, "episodes": [{"episode_id": "e1", "program_id": "T01"}], "failure_attempts": []}
    local = {"manifest_version": 3, "episodes": [{"episode_id": "e1", "program_id": "T02"}], "failure_attempts": []}

    with pytest.raises(ValueError, match="immutable conflict"):
        _merge_manifests(remote, local)


def test_archive_resume_manifest_validates_identity_and_remote_file_hashes(tmp_path: Path, monkeypatch):
    _, _, manifest, prefix_root, _ = _complete_archive_remote(tmp_path, monkeypatch)
    profile = ArchiveProfileConfig.from_dict(manifest["archive_profile"])
    run = RunConfig(
        run_id="new-run",
        run_root="/tmp/new-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        resume_from_hf=True,
    )
    row = manifest["episodes"][0]
    remote_files = {
        f"{row['archive_ref']}/{relative}": (prefix_root / row["archive_ref"] / relative).read_bytes()
        for relative in row["file_sha256"]
    }

    loaded = load_remote_manifest(
        manifest,
        run,
        archive_profile=profile,
        approved_program_ids={"T01"},
        remote_files=remote_files,
    )

    assert loaded["episodes"][0]["archive_ref"] == row["archive_ref"]
    assert loaded["failure_attempts"] == []

    remote_files["episodes/T01/unlisted/extra.bin"] = b"unlisted"
    with pytest.raises(ValueError, match="outside the declared inventory"):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
            remote_files=remote_files,
        )


def test_coordinator_open_accepts_perturbed_archive_row_lineage(tmp_path: Path, monkeypatch):
    from test_generation_publication import _job

    nominal_id = "episode-t01-00000"
    perturbed_id = "episode-t01-00001"
    plan = _archive_resume_plan(episode_id=perturbed_id, episode_kind="perturbed")
    plan = replace(
        plan,
        intervention={
            **(plan.intervention or {}),
            "intervention_type": "pause_hold",
            "intervention_params": {"intervals": 4},
            "intervention_seed": 17,
            "application_scope": "event",
            "application_t": None,
            "intervention_frame": None,
            "source_episode_id": nominal_id,
            "base_episode_id": nominal_id,
        },
    )
    fixture_job = replace(
        _job(),
        job_id="job-perturbed-lineage",
        attempt_id=f"att-{perturbed_id}",
        episode_id=perturbed_id,
        plan=plan,
    )

    run_json, token, manifest, _prefix_root, _downloads = _complete_archive_remote(
        tmp_path,
        monkeypatch,
        fixture_job=fixture_job,
    )
    row = manifest["episodes"][0]
    assert row["source_episode_id"] == plan.intervention["source_episode_id"]
    assert row["base_episode_id"] == plan.intervention["base_episode_id"]
    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: object(),
        token_path=token,
    )
    assert control.manifest["episodes"][0]["source_episode_id"] == nominal_id
    assert control.manifest["episodes"][0]["base_episode_id"] == nominal_id


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda manifest: manifest.update({"archive_profile": {}}), "archive.*profile"),
        (
            lambda manifest: manifest["episodes"][0].update({"archive_ref": "../escape"}),
            "archive_ref",
        ),
        (
            lambda manifest: manifest["episodes"][0].update({
                "archive_ref": "episodes//T01/episode-old-success",
            }),
            "archive_ref",
        ),
        (
            lambda manifest: manifest["episodes"][0]["file_sha256"].pop("data/chunk-00000.npz"),
            "chunk",
        ),
        (
            lambda manifest: manifest["episodes"].append(dict(manifest["episodes"][0])),
            "duplicate",
        ),
    ],
)
def test_archive_resume_manifest_rejects_malformed_or_missing_archive_references(
    mutator, message,
):
    profile = _archive_resume_profile()
    run = RunConfig(
        run_id="new-run",
        run_root="/tmp/new-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        resume_from_hf=True,
    )
    manifest = _archive_resume_manifest(profile)
    mutator(manifest)

    with pytest.raises(ValueError, match=message):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
        )


def test_archive_resume_manifest_rejects_conflicting_row_and_current_source_run():
    profile = _archive_resume_profile()
    run = RunConfig(
        run_id="new-run",
        run_root="/tmp/new-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        resume_from_hf=True,
    )
    manifest = _archive_resume_manifest(profile)
    conflicting = dict(manifest["episodes"][0])
    conflicting["episode_id"] = manifest["episodes"][1]["episode_id"]
    manifest["episodes"].append(conflicting)
    with pytest.raises(ValueError, match="duplicate|conflict"):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
        )

    manifest = _archive_resume_manifest(profile)
    manifest["source_run_ids"] = [run.run_id]
    with pytest.raises(ValueError, match="disjoint"):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
        )


def test_archive_resume_manifest_rejects_duplicate_attempt_ids_across_row_kinds():
    profile = _archive_resume_profile()
    run = RunConfig(
        run_id="new-run",
        run_root="/tmp/new-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        resume_from_hf=True,
    )
    manifest = _archive_resume_manifest(profile)
    manifest["failure_attempts"][0]["attempt_id"] = manifest["episodes"][0]["attempt_id"]

    with pytest.raises(ValueError, match="duplicate attempt"):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
        )


def test_archive_resume_manifest_rejects_remote_manifest_missing_chunk_reference():
    profile = _archive_resume_profile()
    run = RunConfig(
        run_id="new-run",
        run_root="/tmp/new-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        resume_from_hf=True,
    )
    manifest = _archive_resume_manifest(profile)
    row = manifest["episodes"][0]
    archive_payload = {
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "dataset_identity": profile.dataset_identity,
        "archive_profile": profile.as_dict(),
        "archive_kind": "episode",
        "episode_id": row["episode_id"],
        "attempt_id": row["attempt_id"],
        "program_id": row["program_id"],
        "outcome": row["outcome"],
        "source_run_id": row["source_run_id"],
        "code_revision": row["code_revision"],
        "preprocessing_identity": row["preprocessing_identity"],
        "chunk_inventory": [{"path": "data/chunk-00001.npz", "sha256": "4" * 64}],
    }
    archive_content = json.dumps(archive_payload).encode("utf-8")
    row["file_sha256"]["episode.manifest.json"] = hashlib.sha256(archive_content).hexdigest()
    row["file_sha256"]["artifact_manifest.json"] = hashlib.sha256(b"artifact").hexdigest()
    row["file_sha256"]["debug.json"] = hashlib.sha256(b"debug").hexdigest()
    row["file_sha256"]["data/chunk-00000.npz"] = hashlib.sha256(b"chunk").hexdigest()
    remote_files = {
        row["archive_manifest"]: archive_content,
        f"{row['archive_ref']}/artifact_manifest.json": b"artifact",
        f"{row['archive_ref']}/debug.json": b"debug",
        f"{row['archive_ref']}/data/chunk-00000.npz": b"chunk",
    }

    with pytest.raises(ValueError, match="chunk reference"):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
            remote_files=remote_files,
        )


def test_archive_resume_manifest_rejects_missing_remote_archive_file():
    profile = _archive_resume_profile()
    run = RunConfig(
        run_id="new-run",
        run_root="/tmp/new-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        resume_from_hf=True,
    )
    manifest = _archive_resume_manifest(profile)
    row = manifest["episodes"][0]

    with pytest.raises(ValueError, match="remote archive file is missing"):
        load_remote_manifest(
            manifest,
            run,
            archive_profile=profile,
            approved_program_ids={"T01"},
            remote_files={
                f"{row['archive_ref']}/data/chunk-00000.npz": b"only one file",
            },
        )


def test_resume_manifest_rejects_source_run_collision_and_duplicate_ids():
    run = RunConfig(
        run_id="resume-run",
        run_root="/tmp/resume-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="disjoint"):
        _validate_resume_manifest(
            {"manifest_version": 3, "run_id": "resume-run", "episodes": [], "failure_attempts": []},
            run,
        )
    with pytest.raises(ValueError, match="duplicate"):
        _validate_resume_manifest(
            {
                "manifest_version": 3,
                "episodes": [
                    {"episode_id": "e1", "program_id": "T01", "outcome": "success"},
                    {"episode_id": "e1", "program_id": "T01", "outcome": "success"},
                ],
                "failure_attempts": [],
            },
            run,
        )


def test_coordinator_resume_requires_and_consumes_remote_manifest(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    payload = config.as_dict()
    payload["run"].update({
        "resume_from_hf": True,
        "publication_batch_size": 8,
        "publication_upload_threads": 4,
    })
    config = GenerationRuntimeConfig.from_dict(payload)
    approved = Path("artifacts/composition/approved_composition_manifest.json")
    run_json = generation_launch.persist_run_config(
        config,
        approved,
        code_revision="a" * 40,
    )
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    remote = tmp_path / "remote-manifest.json"
    remote.write_text(json.dumps({
        "manifest_version": 3,
        "source_run_ids": ["previous-run"],
        "episodes": [{
            "episode_id": "remote-t01-00000",
            "attempt_id": "att-remote-t01-00000",
            "program_id": "T01",
            "episode_kind": "nominal",
            "outcome": "success",
            "scene_signature": "remote-signature",
            "scene_seed": 99,
            "episode_index": 0,
        }],
        "failure_attempts": [],
    }), encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")

    monkeypatch.setattr(
        generation_coordinator_module,
        "hf_hub_download",
        lambda **kwargs: str(remote),
    )
    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: object(),
        token_path=token,
    )

    assert control.planner.counts("T01").nominal_successes == 1
    assert control.publisher.publication_config.batch_size == 8
    assert control.publisher.publication_config.upload_threads == 4


def test_persisted_run_keeps_publication_concurrency_controls(tmp_path: Path):
    config = _runtime_config(tmp_path, publication_enabled=True)
    payload = config.as_dict()
    payload["run"].update({
        "publication_batch_size": 8,
        "publication_upload_threads": 4,
    })
    config = GenerationRuntimeConfig.from_dict(payload)

    run_json = generation_launch.persist_run_config(
        config,
        Path("artifacts/composition/approved_composition_manifest.json"),
        code_revision="a" * 40,
    )

    persisted = json.loads(run_json.read_text(encoding="utf-8"))
    assert persisted["run"]["publication_batch_size"] == 8
    assert persisted["run"]["publication_upload_threads"] == 4


def _complete_archive_remote(
    tmp_path: Path,
    monkeypatch,
    *,
    remote_revision: str = "d1cc82c27bc57602bf3fac40f55c6dbb59766d45",
    fixture_job: GenerationJob | None = None,
    archive_outcome: str = "success",
):
    from test_generation_publication import _archive_queue, _job

    supplied_fixture = fixture_job is not None
    fixture_job = _job() if fixture_job is None else fixture_job
    fixture_job = replace(fixture_job, plan=replace(
        fixture_job.plan,
        randomization={**fixture_job.plan.randomization,
                       "scene_seed": fixture_job.plan.scene_seed,
                       "asset_instance_id": (
                           fixture_job.plan.randomization.get("asset_instance_id")
                           if supplied_fixture
                           else "T01-asset-1"
                       )},
    ))
    if archive_outcome == "success":
        _source_queue, job, result, profile = _archive_queue(
            tmp_path / "source", retention="receipt_only", job=fixture_job,
        )
        row = validate_closed_result(job, result, archive_profile=profile).episode_entry
        episode_rows, failure_attempt_rows = [row], []
    elif archive_outcome in {"simulator_crash", "invalid_observation"}:
        from scripts.generation_worker import _file_hashes, _write_archive_worker_attempt

        profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention="receipt_only")
        config_payload = _runtime_config(tmp_path).as_dict()
        config_payload["run"]["validation_mode"] = False
        config_payload["archive_profile"] = profile.as_dict()
        worker_config = GenerationRuntimeConfig.from_dict(config_payload)
        result_dir = tmp_path / "source" / "archive-worker-attempt"
        _write_archive_worker_attempt(
            result_dir,
            fixture_job,
            worker_config,
            outcome=archive_outcome,
            error="worker startup failed",
        )
        result = WorkerResult(
            job_id=fixture_job.job_id,
            attempt_id=fixture_job.attempt_id,
            episode_id=None,
            program_id=fixture_job.program_id,
            outcome=archive_outcome,
            result_dir=str(result_dir),
            file_sha256=_file_hashes(result_dir),
            timeline=None,
        )
        job = fixture_job
        row = validate_closed_result(job, result, archive_profile=profile).attempt_entry
        episode_rows, failure_attempt_rows = [], [row]
    else:
        raise ValueError(f"unsupported archive fixture outcome: {archive_outcome}")
    assert row is not None
    manifest = {
        "manifest_version": 3, "source_run_ids": [job.run_id],
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(), "view_status": "PROVISIONAL",
        "episodes": episode_rows, "failure_attempts": failure_attempt_rows,
    }
    config_payload = _runtime_config(tmp_path, publication_enabled=True).as_dict()
    config_payload["run"].update({
        "run_id": "new-archive-run",
        "validation_mode": False,
        "resume_from_hf": True,
    })
    config_payload["archive_profile"] = profile.as_dict()
    config_payload.update(max_result_bytes=1_000_000, max_staging_bytes=20_000_000, staging_reserve_bytes=2_000_000)
    config = GenerationRuntimeConfig.from_dict(config_payload)
    approved_payload = json.loads(Path("artifacts/composition/approved_composition_manifest.json").read_text())
    next(item for item in approved_payload["catalog"] if item["program_id"] == "T01")["asset_family_id"] = "family-1"
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(approved_payload), encoding="utf-8")
    run_json = generation_launch.persist_run_config(
        config,
        approved,
        code_revision="a" * 40,
    )
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    data_revision = "740e07d6648cdf099c78bca78a9b7195d7da5d6a"
    receipt_revision = "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
    verified_revision = "d1cc82c27bc57602bf3fac40f55c6dbb59766d45"
    run_prefix = config.run.hf_subfolder
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    dataset_sha = hashlib.sha256(manifest_bytes).hexdigest()
    artifact_hashes = {
        f"{row['job_id']}/{relative}": digest
        for relative, digest in row["file_sha256"].items()
    }
    resume_payload = {
        "run_id": job.run_id, "source_run_id": job.run_id,
        "source_run_ids": [job.run_id],
        "episodes": len(episode_rows), "failure_attempts": len(failure_attempt_rows),
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "dataset_manifest_sha256": dataset_sha,
        "dataset_manifest_revision": None,
    }
    publication_payload = {
        "receipt_version": 2, "run_id": job.run_id, "source_run_id": job.run_id,
        "job_ids": [job.job_id], "status": "PREPARED",
        "data_commit_oid": None, "commit_oid": None,
        "artifact_hashes": artifact_hashes,
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "dataset_manifest_sha256": dataset_sha,
        "prefix": run_prefix,
    }
    data_snapshot = {
        f"{run_prefix}/dataset_manifest.json": manifest_bytes,
        f"{run_prefix}/resume_receipt.json": json.dumps(resume_payload).encode("utf-8"),
        f"{run_prefix}/publication_receipt.json": json.dumps(publication_payload).encode("utf-8"),
    }
    archive_root = Path(result.result_dir)
    for path in archive_root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(archive_root).as_posix()
            data_snapshot[f"{run_prefix}/{row['archive_ref']}/{relative}"] = path.read_bytes()
    if episode_rows:
        from icgs.data.datasets.generation_view_index import build_provisional_episode_pointers

        episode_manifest = json.loads(
            (archive_root / "episode.manifest.json").read_text(encoding="utf-8")
        )
        for pointer in build_provisional_episode_pointers(episode_manifest):
            pointer_filename = (
                f"{run_prefix}/views/provisional/episodes/{row['program_id']}/"
                f"{row['episode_id']}/{pointer['view']}.json"
            )
            data_snapshot[pointer_filename] = (
                json.dumps(pointer, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
    receipt_snapshot = dict(data_snapshot)
    receipt_resume = dict(resume_payload)
    receipt_resume["dataset_manifest_revision"] = data_revision
    receipt_publication = dict(publication_payload)
    receipt_publication.update({
        "status": "DATA_COMMITTED",
        "data_commit_oid": data_revision,
        "commit_oid": None,
    })
    receipt_snapshot[f"{run_prefix}/resume_receipt.json"] = json.dumps(
        receipt_resume
    ).encode("utf-8")
    receipt_snapshot[f"{run_prefix}/publication_receipt.json"] = json.dumps(
        receipt_publication
    ).encode("utf-8")
    verified_snapshot = dict(receipt_snapshot)
    verified_publication = dict(receipt_publication)
    verified_publication.update({
        "status": "VERIFIED",
        "commit_oid": receipt_revision,
    })
    verified_snapshot[f"{run_prefix}/publication_receipt.json"] = json.dumps(
        verified_publication
    ).encode("utf-8")
    remote_snapshots = {
        data_revision: data_snapshot,
        receipt_revision: receipt_snapshot,
        verified_revision: verified_snapshot,
    }
    if remote_revision not in remote_snapshots:
        remote_snapshots[remote_revision] = dict(verified_snapshot)
        remote_snapshots[remote_revision]["unrelated/other-prefix/marker.json"] = b"unrelated"
    snapshot_roots = {}
    for snapshot_oid, files in remote_snapshots.items():
        root = tmp_path / "snapshots" / snapshot_oid
        snapshot_roots[snapshot_oid] = root
        for filename, content in files.items():
            target = root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    prefix_root = snapshot_roots[remote_revision] / run_prefix

    bootstrap = Path(config.run.run_root) / "control" / "resume_bootstrap.json"
    bootstrap.write_text(json.dumps({
        "run_id": config.run.run_id, "remote_revision": remote_revision,
        "remote_manifest_sha256": dataset_sha,
    }), encoding="utf-8")
    downloads = []
    def download(**kwargs):
        downloads.append(kwargs)
        pinned = kwargs.get("revision")
        selected_revision = pinned or remote_revision
        if pinned is not None:
            assert pinned in remote_snapshots
        path = snapshot_roots[selected_revision] / kwargs["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if not kwargs.get("cache_dir"):
            return str(path)
        cached = Path(kwargs["cache_dir"]) / "snapshots" / selected_revision / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, cached)
        return str(cached)
    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    return run_json, token, manifest, prefix_root, downloads


def _archive_resume_fixture_job(run_id: str, index: int, output_root: Path) -> GenerationJob:
    episode_id = f"episode-t01-{run_id}-{index:05d}"
    plan = AttemptPlan(
        program_id="T01",
        split="train",
        episode_index=index,
        episode_id=episode_id,
        episode_kind="nominal",
        scene_seed=index,
        collection_seed=20260920,
        randomization={
            "scene_signature": f"resume-{run_id}-{index}",
            "scene_seed": index,
            "asset_instance_id": f"T01-asset-{index}",
            "asset_family_id": "family-1",
        },
        intervention=None,
    )
    return GenerationJob.create(
        job_id=f"job-{run_id}-{index:05d}",
        run_id=run_id,
        attempt_id=f"att-{episode_id}",
        episode_id=episode_id,
        program_id="T01",
        plan=plan,
        code_revision="a" * 40,
        manifest_sha256="b" * 64,
        output_root=str(output_root),
    )


def _multi_batch_archive_resume_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    retention: str = "receipt_only",
):
    from test_generation_publication import _archive_queue
    from icgs.data.collection.generation.distributed_publication import PublicationReceipt

    profile = ArchiveProfileConfig(chunk_boundaries=2, local_artifact_retention=retention)
    hf_prefix = "icgs-extension-test-archive-v1"
    source_rows = []
    source_files = {}
    for index in range(1, 4):
        job = _archive_resume_fixture_job("old-source", index, tmp_path / "origin-staging")
        source_queue, job, result, source_profile = _archive_queue(
            tmp_path / "origin-fixtures" / f"{index:05d}",
            retention=retention,
            job=job,
        )
        assert source_profile == profile
        row = validate_closed_result(job, result, archive_profile=profile).episode_entry
        assert row is not None
        source_rows.append(row)
        for path in Path(result.result_dir).rglob("*"):
            if path.is_file():
                relative = path.relative_to(result.result_dir).as_posix()
                source_files[
                    f"{hf_prefix}/{row['archive_ref']}/{relative}"
                ] = path.read_bytes()
        from icgs.data.datasets.generation_view_index import build_provisional_episode_pointers

        episode_manifest = json.loads(
            (Path(result.result_dir) / "episode.manifest.json").read_text(encoding="utf-8")
        )
        for pointer in build_provisional_episode_pointers(episode_manifest):
            pointer_filename = (
                f"{hf_prefix}/views/provisional/episodes/{row['program_id']}/"
                f"{row['episode_id']}/{pointer['view']}.json"
            )
            source_files[pointer_filename] = (
                json.dumps(pointer, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")

    source_manifest = {
        "manifest_version": 3,
        "source_run_ids": ["old-source"],
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "view_status": "PROVISIONAL",
        "episodes": source_rows,
        "failure_attempts": [],
    }
    source_manifest_bytes = json.dumps(
        source_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    source_manifest_sha = hashlib.sha256(source_manifest_bytes).hexdigest()
    source_data_oid = hashlib.sha1(b"resume-origin-data").hexdigest()
    source_receipt_oid = hashlib.sha1(b"resume-origin-receipt").hexdigest()
    source_head_oid = hashlib.sha1(b"resume-origin-verified").hexdigest()
    source_artifact_hashes = {
        f"{row['job_id']}/{relative}": digest
        for row in source_rows
        for relative, digest in row["file_sha256"].items()
    }
    source_resume = {
        "run_id": "old-source",
        "source_run_id": "old-source",
        "source_run_ids": ["old-source"],
        "episodes": len(source_rows),
        "failure_attempts": 0,
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "dataset_manifest_sha256": source_manifest_sha,
        "dataset_manifest_revision": source_data_oid,
    }
    source_data_snapshot = {
        **source_files,
        f"{hf_prefix}/dataset_manifest.json": source_manifest_bytes,
    }
    source_data_committed = PublicationReceipt(
        run_id="old-source",
        source_run_id="old-source",
        job_ids=tuple(row["job_id"] for row in source_rows),
        status="DATA_COMMITTED",
        data_commit_oid=source_data_oid,
        commit_oid=None,
        artifact_hashes=source_artifact_hashes,
        dataset_identity=profile.dataset_identity,
        archive_format_id=profile.archive_format_id,
        episode_schema_version=profile.episode_schema_version,
        archive_profile=profile.as_dict(),
        dataset_manifest_sha256=source_manifest_sha,
        prefix=hf_prefix,
    )
    source_receipt_snapshot = {
        **source_data_snapshot,
        f"{hf_prefix}/resume_receipt.json": json.dumps(
            source_resume, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        f"{hf_prefix}/publication_receipt.json": json.dumps(
            source_data_committed.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    }
    source_verified = replace(
        source_data_committed,
        status="VERIFIED",
        commit_oid=source_receipt_oid,
    )
    source_verified_snapshot = {
        **source_receipt_snapshot,
        f"{hf_prefix}/publication_receipt.json": json.dumps(
            source_verified.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    }
    snapshots = {
        source_data_oid: source_data_snapshot,
        source_receipt_oid: source_receipt_snapshot,
        source_head_oid: source_verified_snapshot,
    }

    config_payload = _runtime_config(tmp_path, publication_enabled=True).as_dict()
    config_payload["run"].update({
        "run_id": "archive-extension-run",
        "validation_mode": False,
        "resume_from_hf": True,
        "hf_subfolder": hf_prefix,
    })
    config_payload["archive_profile"] = profile.as_dict()
    config_payload["max_result_bytes"] = 1_000_000
    config_payload.update(max_staging_bytes=20_000_000, staging_reserve_bytes=2_000_000)
    config = GenerationRuntimeConfig.from_dict(config_payload)
    approved_payload = json.loads(
        Path("artifacts/composition/approved_composition_manifest.json").read_text()
    )
    next(row for row in approved_payload["catalog"] if row["program_id"] == "T01")[
        "asset_family_id"
    ] = "family-1"
    approved_path = tmp_path / "approved.json"
    approved_path.write_text(json.dumps(approved_payload), encoding="utf-8")
    run_json = generation_launch.persist_run_config(
        config, approved_path, code_revision="c" * 40,
    )
    token = Path(config.run.run_root) / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    bootstrap_path = Path(config.run.run_root) / "control" / "resume_bootstrap.json"
    bootstrap_path.write_text(json.dumps({
        "run_id": config.run.run_id,
        "remote_manifest_sha256": source_manifest_sha,
        "remote_revision": source_head_oid,
        "source_run_ids": ["old-source"],
    }), encoding="utf-8")

    class SnapshotApi:
        def __init__(self):
            self.head = source_head_oid
            self.tree = dict(snapshots[source_head_oid])
            self.commit_count = 0

        def create_commit(self, **kwargs):
            updated = dict(self.tree)
            for operation in kwargs["operations"]:
                source = operation.path_or_fileobj
                content = (
                    Path(source).read_bytes()
                    if isinstance(source, (str, Path))
                    else source.read()
                )
                updated[operation.path_in_repo] = content
            self.commit_count += 1
            oid = hashlib.sha1(f"extension-commit-{self.commit_count}".encode()).hexdigest()
            self.tree = updated
            snapshots[oid] = dict(updated)
            self.head = oid
            return SimpleNamespace(oid=oid)

    api = SnapshotApi()

    def download(**kwargs):
        revision = kwargs.get("revision") or api.head
        files = snapshots.get(revision)
        if files is None or kwargs["filename"] not in files:
            raise FileNotFoundError(f"missing fake HF file at {revision}: {kwargs['filename']}")
        cache = Path(kwargs["cache_dir"])
        target = cache / "snapshots" / revision / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[kwargs["filename"]])
        return str(target)

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    monkeypatch.setattr(generation_launch, "hf_hub_download", download)
    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: api,
        token_path=token,
    )
    return {
        "api": api,
        "approved_path": approved_path,
        "bootstrap_path": bootstrap_path,
        "config": config,
        "control": control,
        "download": download,
        "manifest": source_manifest,
        "profile": profile,
        "run_json": run_json,
        "run_payload": run_payload,
        "snapshots": snapshots,
        "source_head_oid": source_head_oid,
        "source_manifest_sha": source_manifest_sha,
        "source_rows": source_rows,
        "tmp_path": tmp_path,
        "token": token,
    }


def _stage_resume_extension_batch(fixture: dict, index: int) -> GenerationJob:
    from test_generation_publication import _archive_queue
    from scripts.generation_coordinator import _manifest_from_closed_queue

    control = fixture["control"]
    run = fixture["config"].run
    job = _archive_resume_fixture_job(
        run.run_id,
        index,
        Path(run.run_root) / "staging",
    )
    source_queue, job, _result, profile = _archive_queue(
        Path(run.run_root) / "fixtures" / job.job_id,
        retention=fixture["profile"].local_artifact_retention,
        job=job,
    )
    assert profile == fixture["profile"]
    shutil.copytree(
        source_queue.root / "ingested" / job.job_id,
        control.queue.root / "ingested" / job.job_id,
    )
    local = _manifest_from_closed_queue(
        control.queue,
        states=("ingested",),
        archive_profile=fixture["profile"],
    )
    control.manifest = _merge_manifests(control.manifest, local)
    return job


def _publish_resume_extension_batch(fixture: dict, index: int) -> GenerationJob:
    job = _stage_resume_extension_batch(fixture, index)
    control = fixture["control"]
    completed = control.publisher.publish_due(
        now_s=300.0 + index,
        force=True,
        local_manifest=control.manifest,
    )
    assert completed is not None and completed.status == "COMPLETE"
    return job


def test_coordinator_resume_accepts_prepared_batch_when_head_is_prior_local_verified_batch(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)
    previous_head = fixture["api"].head

    def reject_data_commit(**_kwargs):
        raise RuntimeError("simulated interruption before data commit")

    fixture["api"].create_commit = reject_data_commit
    with pytest.raises(RuntimeError, match="before data commit"):
        fixture["control"].publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )
    assert fixture["api"].head == previous_head
    local_receipt = json.loads(
        (fixture["control"].queue.root / "publication_receipt.json").read_text()
    )
    assert local_receipt["status"] == "PREPARED"

    runtime_path = Path(
        json.loads(fixture["run_json"].read_text())["runtime_config_path"]
    )
    config = GenerationRuntimeConfig.from_file(runtime_path, check_paths=False)
    launcher_bootstrap = generation_launch.preflight_resume_manifest(
        config,
        token_path=fixture["token"],
        run_root=config.run.run_root,
        approved_manifest=fixture["approved_path"],
        downloader=fixture["download"],
    )
    assert launcher_bootstrap["latest_remote_revision"] == previous_head
    assert json.loads(
        (fixture["control"].queue.root / "publication_receipt.json").read_text()
    )["status"] == "PREPARED"

    commit_count = fixture["api"].commit_count
    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: fixture["api"],
        token_path=fixture["token"],
    )

    assert fixture["api"].commit_count == commit_count
    assert len(resumed.publisher.remote_manifest["episodes"]) == 4
    assert json.loads(
        (resumed.queue.root / "publication_receipt.json").read_text()
    )["status"] == "PREPARED"


def test_coordinator_pending_resume_fails_closed_without_prior_keep_receipt(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(
        tmp_path, monkeypatch, retention="keep",
    )
    first_job = _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)
    first_published = fixture["control"].queue.root / "published" / first_job.job_id
    assert not (first_published / "publication_receipt.json").exists()

    def reject_data_commit(**_kwargs):
        raise RuntimeError("simulated interruption before data commit")

    fixture["api"].create_commit = reject_data_commit
    with pytest.raises(RuntimeError, match="before data commit"):
        fixture["control"].publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            fixture["run_json"],
            api_factory=lambda: api_calls.append(True),
            token_path=fixture["token"],
        )

    assert "pending same-run recovery under keep retention lacks a durable local prior VERIFIED receipt OID" in str(
        failure.value.__cause__
    )
    assert api_calls == []
    assert not (first_published / "publication_receipt.json").exists()


def test_coordinator_keep_mode_stable_batches_resume_from_latest_verified_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(
        tmp_path, monkeypatch, retention="keep",
    )
    _publish_resume_extension_batch(fixture, 4)
    runtime_path = Path(
        json.loads(fixture["run_json"].read_text())["runtime_config_path"]
    )
    config = GenerationRuntimeConfig.from_file(runtime_path, check_paths=False)
    launcher_bootstrap = generation_launch.preflight_resume_manifest(
        config,
        token_path=fixture["token"],
        run_root=config.run.run_root,
        approved_manifest=fixture["approved_path"],
        downloader=fixture["download"],
    )
    assert launcher_bootstrap["latest_remote_revision"] == fixture["api"].head

    restarted = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: fixture["api"],
        token_path=fixture["token"],
    )
    assert len(restarted.publisher.remote_manifest["episodes"]) == 4
    fixture["control"] = restarted
    second_job = _publish_resume_extension_batch(fixture, 5)
    final_manifest = json.loads(
        fixture["snapshots"][fixture["api"].head][
            f"{fixture['config'].run.hf_subfolder}/dataset_manifest.json"
        ]
    )

    assert second_job.episode_id in {row["episode_id"] for row in final_manifest["episodes"]}
    assert len(final_manifest["episodes"]) == 5


def test_coordinator_resume_recovers_data_commit_with_local_prepared_receipt(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)
    api = fixture["api"]
    original_create_commit = api.create_commit

    def interrupt_after_remote_data_commit(**kwargs):
        if kwargs["commit_message"].startswith("Add generation batch"):
            original_create_commit(**kwargs)
            raise RuntimeError("simulated stop after remote data commit")
        return original_create_commit(**kwargs)

    api.create_commit = interrupt_after_remote_data_commit
    with pytest.raises(RuntimeError, match="after remote data commit"):
        fixture["control"].publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )
    pending = json.loads(
        (fixture["control"].queue.root / "publication_receipt.json").read_text()
    )
    assert pending["status"] == "PREPARED"
    assert pending["data_commit_oid"] is None
    assert json.loads(
        api.tree[f"{fixture['config'].run.hf_subfolder}/publication_receipt.json"]
    )["status"] == "PREPARED"

    commit_count = api.commit_count
    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: api,
        token_path=fixture["token"],
    )

    recovered = json.loads(
        (resumed.queue.root / "publication_receipt.json").read_text()
    )
    assert recovered["status"] == "DATA_COMMITTED"
    assert recovered["data_commit_oid"] == api.head
    assert api.commit_count == commit_count
    assert len(resumed.publisher.remote_manifest["episodes"]) == 5


def test_coordinator_resume_recovers_first_batch_data_commit_before_local_oid_write(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _stage_resume_extension_batch(fixture, 4)
    api = fixture["api"]
    original_create_commit = api.create_commit

    def interrupt_after_remote_data_commit(**kwargs):
        if kwargs["commit_message"].startswith("Add generation batch"):
            original_create_commit(**kwargs)
            raise RuntimeError("simulated stop after remote data commit")
        return original_create_commit(**kwargs)

    api.create_commit = interrupt_after_remote_data_commit
    with pytest.raises(RuntimeError, match="after remote data commit"):
        fixture["control"].publisher.publish_due(
            now_s=304.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )
    assert fixture["control"].queue.counts().published == 0

    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: api,
        token_path=fixture["token"],
    )

    receipt = json.loads(
        (resumed.queue.root / "publication_receipt.json").read_text()
    )
    assert receipt["status"] == "DATA_COMMITTED"
    assert receipt["data_commit_oid"] == api.head
    assert resumed.planner.counts("T01").nominal_successes == 4


def test_coordinator_resume_accepts_data_committed_pending_head(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)

    def fail_remote_verification(*_args):
        raise RuntimeError("simulated remote verification interruption")

    fixture["control"].publisher.remote_verify = fail_remote_verification
    with pytest.raises(RuntimeError, match="verification interruption"):
        fixture["control"].publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )
    pending = json.loads(
        (fixture["control"].queue.root / "publication_receipt.json").read_text()
    )
    remote = json.loads(
        fixture["api"].tree[
            f"{fixture['config'].run.hf_subfolder}/publication_receipt.json"
        ]
    )
    assert pending["status"] == "DATA_COMMITTED"
    assert remote["status"] == "DATA_COMMITTED"
    assert remote["data_commit_oid"] == pending["data_commit_oid"]

    commit_count = fixture["api"].commit_count
    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: fixture["api"],
        token_path=fixture["token"],
    )

    assert fixture["api"].commit_count == commit_count
    assert len(resumed.publisher.remote_manifest["episodes"]) == 5


def test_coordinator_resume_recovers_local_data_commit_with_remote_prepared_receipt(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)
    api = fixture["api"]
    publisher = fixture["control"].publisher
    original_write_receipt = publisher._write_publication_receipt

    def interrupt_after_local_data_receipt(receipt):
        original_write_receipt(receipt)
        if receipt.status == "DATA_COMMITTED" and receipt.data_commit_oid is not None:
            local = json.loads(
                (fixture["control"].queue.root / "publication_receipt.json").read_text()
            )
            remote = json.loads(
                api.tree[
                    f"{fixture['config'].run.hf_subfolder}/publication_receipt.json"
                ]
            )
            assert local["status"] == "DATA_COMMITTED"
            assert local["data_commit_oid"] == api.head
            assert remote["status"] == "PREPARED"
            assert remote["data_commit_oid"] is None
            raise RuntimeError("simulated stop after local data receipt persistence")

    publisher._write_publication_receipt = interrupt_after_local_data_receipt
    with pytest.raises(RuntimeError, match="after local data receipt persistence"):
        publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )

    local = json.loads(
        (fixture["control"].queue.root / "publication_receipt.json").read_text()
    )
    assert local["status"] == "DATA_COMMITTED"
    assert local["commit_oid"] is None
    local_resume = json.loads(
        (fixture["control"].queue.root / "resume_receipt.json").read_text()
    )
    assert local_resume["dataset_manifest_revision"] is None
    data_oid = local["data_commit_oid"]
    api.create_commit(operations=[], commit_message="unrelated repository change")
    assert api.head != data_oid
    commit_count = api.commit_count
    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: api,
        token_path=fixture["token"],
    )

    assert local["status"] == "DATA_COMMITTED"
    assert local["commit_oid"] is None
    assert local["data_commit_oid"] == data_oid
    assert api.commit_count == commit_count
    assert len(resumed.publisher.remote_manifest["episodes"]) == 5


def test_coordinator_resume_recovers_data_committed_before_metadata_receipt_commit(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)
    api = fixture["api"]
    original_create_commit = api.create_commit

    def interrupt_receipt_metadata_commit(**kwargs):
        if kwargs["commit_message"].startswith("Record committed generation archive"):
            local = json.loads(
                (fixture["control"].queue.root / "publication_receipt.json").read_text()
            )
            local_resume = json.loads(
                (fixture["control"].queue.root / "resume_receipt.json").read_text()
            )
            remote = json.loads(
                api.tree[
                    f"{fixture['config'].run.hf_subfolder}/publication_receipt.json"
                ]
            )
            assert local["status"] == "DATA_COMMITTED"
            assert local["data_commit_oid"] == api.head
            assert local_resume["dataset_manifest_revision"] == api.head
            assert remote["status"] == "PREPARED"
            raise RuntimeError("simulated stop before receipt metadata commit")
        return original_create_commit(**kwargs)

    api.create_commit = interrupt_receipt_metadata_commit
    with pytest.raises(RuntimeError, match="before receipt metadata commit"):
        fixture["control"].publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )
    commit_count = api.commit_count

    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: api,
        token_path=fixture["token"],
    )

    assert api.commit_count == commit_count
    assert len(resumed.publisher.remote_manifest["episodes"]) == 5


def test_coordinator_resume_recovers_deferred_receipt_after_data_commit(
    tmp_path: Path,
    monkeypatch,
):
    from icgs.data.collection.generation.distributed_publication import PublicationConfig

    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)
    api = fixture["api"]
    publisher = fixture["control"].publisher
    publisher.publication_config = PublicationConfig(
        retry_attempts=3,
        retry_base_s=0.0,
        retry_cooldown_s=0.0,
        rate_limit_cooldown_s=0.0,
    )
    original_create_commit = api.create_commit
    receipt_attempts = []

    def defer_receipt_metadata_commit(**kwargs):
        if kwargs["commit_message"].startswith("Record committed generation archive"):
            receipt_attempts.append(True)
            raise TimeoutError("simulated receipt metadata timeout")
        return original_create_commit(**kwargs)

    api.create_commit = defer_receipt_metadata_commit
    completed = publisher.publish_due(
        now_s=305.0,
        force=True,
        local_manifest=fixture["control"].manifest,
    )
    local = json.loads(
        (fixture["control"].queue.root / "publication_receipt.json").read_text()
    )
    remote = json.loads(
        api.tree[f"{fixture['config'].run.hf_subfolder}/publication_receipt.json"]
    )
    assert completed is None
    assert len(receipt_attempts) == 3
    assert local["status"] == "DEFERRED"
    assert local["data_commit_oid"] == api.head
    assert local["commit_oid"] is None
    assert remote["status"] == "PREPARED"

    commit_count = api.commit_count
    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: api,
        token_path=fixture["token"],
    )

    assert api.commit_count == commit_count
    assert len(resumed.publisher.remote_manifest["episodes"]) == 5


def test_coordinator_resume_rejects_wrong_local_data_receipt_oid(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    _stage_resume_extension_batch(fixture, 5)

    def fail_remote_verification(*_args):
        raise RuntimeError("simulated remote verification interruption")

    fixture["control"].publisher.remote_verify = fail_remote_verification
    with pytest.raises(RuntimeError, match="verification interruption"):
        fixture["control"].publisher.publish_due(
            now_s=305.0,
            force=True,
            local_manifest=fixture["control"].manifest,
        )
    local_receipt_path = fixture["control"].queue.root / "publication_receipt.json"
    local_receipt = json.loads(local_receipt_path.read_text())
    assert local_receipt["status"] == "DATA_COMMITTED"
    assert local_receipt["commit_oid"] is not None
    local_receipt["commit_oid"] = local_receipt["data_commit_oid"]
    local_receipt_path.write_text(json.dumps(local_receipt), encoding="utf-8")

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            fixture["run_json"],
            api_factory=lambda: api_calls.append(True),
            token_path=fixture["token"],
        )

    assert "local DATA_COMMITTED receipt OID does not bind its exact remote receipt snapshot" in str(
        failure.value.__cause__
    )
    assert api_calls == []


def test_coordinator_resume_rejects_latest_head_with_another_batch_receipt(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    first_job = _publish_resume_extension_batch(fixture, 4)
    _publish_resume_extension_batch(fixture, 5)
    prefix = fixture["config"].run.hf_subfolder
    latest_snapshot = fixture["snapshots"][fixture["api"].head]
    latest_manifest = json.loads(latest_snapshot[f"{prefix}/dataset_manifest.json"])
    other_batch_row = next(
        row for row in latest_manifest["episodes"] if row["job_id"] == first_job.job_id
    )
    publication = json.loads(latest_snapshot[f"{prefix}/publication_receipt.json"])
    receipt_revision = publication["commit_oid"]
    other_batch_hashes = {
        f"{first_job.job_id}/{relative}": digest
        for relative, digest in other_batch_row["file_sha256"].items()
    }
    prior_snapshot = fixture["snapshots"][receipt_revision]
    for snapshot, status in ((prior_snapshot, "DATA_COMMITTED"), (latest_snapshot, "VERIFIED")):
        other_batch_receipt = dict(publication)
        other_batch_receipt.update({
            "job_ids": [first_job.job_id],
            "artifact_hashes": other_batch_hashes,
            "status": status,
            "commit_oid": None if status == "DATA_COMMITTED" else receipt_revision,
        })
        snapshot[f"{prefix}/publication_receipt.json"] = json.dumps(
            other_batch_receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            fixture["run_json"],
            api_factory=lambda: api_calls.append(True),
            token_path=fixture["token"],
        )

    assert "remote same-run resume/publication receipt identity mismatch" in str(
        failure.value.__cause__
    )
    assert api_calls == []


def _remove_row_from_fake_hf_head(fixture: dict, episode_id: str) -> None:
    api = fixture["api"]
    files = fixture["snapshots"][api.head]
    filename = f"{fixture['config'].run.hf_subfolder}/dataset_manifest.json"
    manifest = json.loads(files[filename])
    manifest["episodes"] = [
        row for row in manifest["episodes"] if row["episode_id"] != episode_id
    ]
    files[filename] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _tamper_fake_hf_head_archive(fixture: dict, row: dict) -> None:
    api = fixture["api"]
    archive_relative = next(
        relative for relative in row["file_sha256"]
        if relative.startswith("data/chunk-")
    )
    filename = (
        f"{fixture['config'].run.hf_subfolder}/{row['archive_ref']}/{archive_relative}"
    )
    fixture["snapshots"][api.head][filename] += b"tampered remote archive bytes"


def _third_hf_only_resume_control(fixture: dict):
    third_root = Path(fixture["tmp_path"]) / "third-run"
    third_root.mkdir(parents=True, exist_ok=True)
    payload = fixture["config"].as_dict()
    payload["run"].update({
        "run_id": "archive-extension-third-run",
        "run_root": str(third_root),
        "resume_from_hf": True,
    })
    config = GenerationRuntimeConfig.from_dict(payload)
    run_json = generation_launch.persist_run_config(
        config,
        fixture["approved_path"],
        code_revision="e" * 40,
    )
    token = third_root / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    bootstrap = generation_launch.preflight_resume_manifest(
        config,
        token_path=token,
        run_root=third_root,
        approved_manifest=fixture["approved_path"],
        downloader=fixture["download"],
    )
    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: fixture["api"],
        token_path=token,
    )
    return bootstrap, control


def test_archive_resume_keeps_origin_and_all_local_batches_at_latest_head(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    control = fixture["control"]
    source_ids = {row["episode_id"] for row in fixture["source_rows"]}

    first_job = _publish_resume_extension_batch(fixture, 4)
    first_manifest = json.loads(
        fixture["snapshots"][fixture["api"].head][
            f"{fixture['config'].run.hf_subfolder}/dataset_manifest.json"
        ]
    )
    assert {row["episode_id"] for row in first_manifest["episodes"]} == (
        source_ids | {first_job.episode_id}
    )

    initial_bootstrap = json.loads(fixture["bootstrap_path"].read_text())
    assert initial_bootstrap["remote_revision"] == fixture["source_head_oid"]
    assert initial_bootstrap["remote_manifest_sha256"] == fixture["source_manifest_sha"]
    restart_config = GenerationRuntimeConfig.from_file(
        json.loads(fixture["run_json"].read_text())["runtime_config_path"],
        check_paths=False,
    )
    launcher_bootstrap = generation_launch.preflight_resume_manifest(
        restart_config,
        token_path=fixture["token"],
        run_root=restart_config.run.run_root,
        approved_manifest=fixture["approved_path"],
        downloader=fixture["download"],
    )
    assert launcher_bootstrap["remote_revision"] == fixture["source_head_oid"]
    assert launcher_bootstrap["remote_manifest_sha256"] == fixture["source_manifest_sha"]
    assert launcher_bootstrap["latest_remote_revision"] == fixture["api"].head

    control = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: fixture["api"],
        token_path=fixture["token"],
    )
    assert len(control.publisher.remote_manifest["episodes"]) == 4
    second_job = _publish_resume_extension_batch(fixture, 5)
    final_manifest = json.loads(
        fixture["snapshots"][fixture["api"].head][
            f"{fixture['config'].run.hf_subfolder}/dataset_manifest.json"
        ]
    )
    expected_ids = source_ids | {first_job.episode_id, second_job.episode_id}
    assert {row["episode_id"] for row in final_manifest["episodes"]} == expected_ids

    third_bootstrap, third_control = _third_hf_only_resume_control(fixture)

    assert third_bootstrap["source_run_ids"] == ["archive-extension-run", "old-source"]
    assert len(third_control.manifest["episodes"]) == 5
    assert third_control.planner.counts("T01").nominal_successes == 5


def test_coordinator_resume_accepts_unrelated_newer_repo_head_with_unchanged_prefix(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    prior_prefix_manifest = fixture["snapshots"][fixture["api"].head][
        f"{fixture['config'].run.hf_subfolder}/dataset_manifest.json"
    ]
    unrelated = fixture["api"].create_commit(
        operations=[], commit_message="unrelated repository change",
    )

    resumed = CoordinatorControlPlane.open(
        fixture["run_json"],
        api_factory=lambda: fixture["api"],
        token_path=fixture["token"],
    )

    assert fixture["api"].head == unrelated.oid
    assert fixture["snapshots"][fixture["api"].head][
        f"{fixture['config'].run.hf_subfolder}/dataset_manifest.json"
    ] == prior_prefix_manifest
    assert len(resumed.publisher.remote_manifest["episodes"]) == 4


def test_coordinator_resume_rejects_latest_head_missing_local_published_row(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    published_job = _publish_resume_extension_batch(fixture, 4)
    _remove_row_from_fake_hf_head(fixture, published_job.episode_id)

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            fixture["run_json"],
            api_factory=lambda: api_calls.append(True),
            token_path=fixture["token"],
        )

    assert "locally published archive row is missing from the latest remote manifest" in str(
        failure.value.__cause__
    )
    assert "inspect pinned revisions" in str(failure.value)
    assert api_calls == []


def test_coordinator_resume_rejects_missing_local_publication_receipt_after_publish(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    (fixture["control"].queue.root / "publication_receipt.json").unlink()

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            fixture["run_json"],
            api_factory=lambda: api_calls.append(True),
            token_path=fixture["token"],
        )

    assert "same-run restart requires a regular local publication receipt" in str(
        failure.value.__cause__
    )
    assert api_calls == []


def test_launcher_resume_preflight_rejects_missing_local_publication_receipt(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    _publish_resume_extension_batch(fixture, 4)
    (fixture["control"].queue.root / "publication_receipt.json").unlink()
    runtime_path = Path(
        json.loads(fixture["run_json"].read_text())["runtime_config_path"]
    )
    config = GenerationRuntimeConfig.from_file(runtime_path, check_paths=False)

    with pytest.raises(
        RuntimeError, match="remote resume manifest failed immutable validation"
    ) as failure:
        generation_launch.preflight_resume_manifest(
            config,
            token_path=fixture["token"],
            run_root=config.run.run_root,
            approved_manifest=fixture["approved_path"],
            downloader=fixture["download"],
        )

    assert "same-run restart requires a regular local publication receipt" in str(
        failure.value.__cause__
    )


def test_coordinator_resume_rejects_changed_archive_bytes_at_latest_head(
    tmp_path: Path,
    monkeypatch,
):
    from scripts.generation_coordinator import _manifest_from_closed_queue

    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    published_job = _publish_resume_extension_batch(fixture, 4)
    published_rows = _manifest_from_closed_queue(
        fixture["control"].queue,
        states=("published",),
        archive_profile=fixture["profile"],
    )
    published_row = next(
        row for row in published_rows["episodes"]
        if row["episode_id"] == published_job.episode_id
    )
    _tamper_fake_hf_head_archive(fixture, published_row)

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            fixture["run_json"],
            api_factory=lambda: api_calls.append(True),
            token_path=fixture["token"],
        )

    assert "remote archive hash mismatch" in str(failure.value.__cause__)
    assert api_calls == []


def test_launcher_resume_preflight_rejects_dropped_published_row_before_workers(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _multi_batch_archive_resume_fixture(tmp_path, monkeypatch)
    published_job = _publish_resume_extension_batch(fixture, 4)
    _remove_row_from_fake_hf_head(fixture, published_job.episode_id)
    smoke_receipt = tmp_path / "smoke.json"
    from icgs.data.collection.generation.steps import GENERATION_PROGRAMS

    smoke_receipt.write_text(json.dumps({"results": [
        {"program_id": program_id, "result_class": "success", "timeline_ok": True}
        for program_id in GENERATION_PROGRAMS
    ]}), encoding="utf-8")
    runtime_path = Path(
        json.loads(fixture["run_json"].read_text())["runtime_config_path"]
    )
    spawned = []
    monkeypatch.setattr(
        generation_launch.subprocess,
        "Popen",
        lambda *args, **kwargs: spawned.append(args) or SimpleNamespace(pid=9001, wait=lambda: 0),
    )

    with pytest.raises(
        RuntimeError,
        match="remote resume manifest failed immutable validation; inspect pinned revisions",
    ) as failure:
        generation_launch.main([
            "--runtime-config", str(runtime_path),
            "--approved-manifest", str(fixture["approved_path"]),
            "--code-revision", "e" * 40,
            "--smoke-receipt", str(smoke_receipt),
            "--hf-token-path", str(fixture["token"]),
            "--detach",
        ])

    assert spawned == []
    assert "locally published archive row is missing from the latest remote manifest" in str(
        failure.value.__cause__
    )


def _same_run_archive_restart_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    remote_revision: str = "d1cc82c27bc57602bf3fac40f55c6dbb59766d45",
):
    from test_generation_publication import _archive_queue, _job
    from icgs.data.collection.generation.distributed_publication import (
        HuggingFaceBatchPublisher,
        PublicationConfig,
    )

    profile = ArchiveProfileConfig(
        chunk_boundaries=2,
        local_artifact_retention="receipt_only",
    )
    config_payload = _runtime_config(tmp_path, publication_enabled=True).as_dict()
    config_payload["run"].update({
        "run_id": "run-1",
        "validation_mode": False,
        "resume_from_hf": False,
    })
    config_payload["archive_profile"] = profile.as_dict()
    config_payload.update(max_result_bytes=1_000_000, max_staging_bytes=20_000_000, staging_reserve_bytes=2_000_000)
    config = GenerationRuntimeConfig.from_dict(config_payload)
    approved_payload = json.loads(
        Path("artifacts/composition/approved_composition_manifest.json").read_text()
    )
    next(item for item in approved_payload["catalog"] if item["program_id"] == "T01")[
        "asset_family_id"
    ] = "family-1"
    approved_path = tmp_path / "approved.json"
    approved_path.write_text(json.dumps(approved_payload), encoding="utf-8")
    run_json = generation_launch.persist_run_config(
        config, approved_path, code_revision="a" * 40,
    )
    token = Path(config.run.run_root) / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")

    first_job = _job()
    first_job = replace(first_job, plan=replace(
        first_job.plan,
        randomization={
            **first_job.plan.randomization,
            "scene_seed": 1,
            "asset_instance_id": "T01-asset-1",
            "asset_family_id": "family-1",
        },
    ))
    first_source, first_job, first_result, _ = _archive_queue(
        Path(config.run.run_root) / "staging" / "first-local",
        retention="receipt_only",
        job=first_job,
    )
    queue = FilesystemJobQueue(Path(config.run.run_root) / "queue")
    queue.configure_runtime_staging(config)
    shutil.copytree(
        first_source.root / "ingested" / first_job.job_id,
        queue.root / "ingested" / first_job.job_id,
    )

    data_revision = "740e07d6648cdf099c78bca78a9b7195d7da5d6a"
    receipt_revision = "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
    verified_revision = "d1cc82c27bc57602bf3fac40f55c6dbb59766d45"

    class SnapshotApi:
        def __init__(self):
            self.tree = {}
            self.snapshots = {}
            self.commit_oids = [data_revision, receipt_revision, verified_revision]

        def create_commit(self, **kwargs):
            updated = dict(self.tree)
            for operation in kwargs["operations"]:
                updated[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
            oid = self.commit_oids[len(self.snapshots)]
            self.tree = updated
            self.snapshots[oid] = dict(updated)
            return SimpleNamespace(oid=oid)

    api = SnapshotApi()
    verified_batches = []
    publisher = HuggingFaceBatchPublisher(
        RunConfig.from_dict(run_payload["run"]),
        api,
        "secret",
        queue,
        last_success_s=0.0,
        remote_verify=lambda job_ids, revision, _token: verified_batches.append(
            (job_ids, revision)
        ),
        publication_config=PublicationConfig(batch_size=1),
        archive_profile=profile,
    )
    completed = publisher.publish_due(now_s=300.0, force=True)
    assert completed is not None and completed.status == "COMPLETE"
    assert verified_batches == [((first_job.job_id,), receipt_revision)]
    assert queue.counts().published == 1

    second_plan = replace(
        first_job.plan,
        episode_index=2,
        episode_id="episode-t01-00002",
        scene_seed=2,
        randomization={
            **first_job.plan.randomization,
            "scene_signature": "sig-2",
            "scene_seed": 2,
            "asset_instance_id": "T01-asset-2",
        },
    )
    second_job = GenerationJob.create(
        job_id="job-run-1-episode-t01-00002",
        run_id=first_job.run_id,
        attempt_id="att-episode-t01-00002",
        episode_id=second_plan.episode_id,
        program_id=first_job.program_id,
        plan=second_plan,
        code_revision=first_job.code_revision,
        manifest_sha256=first_job.manifest_sha256,
        output_root=first_job.output_root,
    )
    second_source, second_job, _second_result, _ = _archive_queue(
        Path(config.run.run_root) / "staging" / "second-local",
        retention="receipt_only",
        job=second_job,
    )
    shutil.copytree(
        second_source.root / "ingested" / second_job.job_id,
        queue.root / "ingested" / second_job.job_id,
    )

    remote_manifest = json.loads(
        (queue.root / "publication_manifest.json").read_text(encoding="utf-8")
    )
    assert len(remote_manifest["episodes"]) == 1
    remote_snapshots = {revision: dict(files) for revision, files in api.snapshots.items()}
    if remote_revision not in remote_snapshots:
        remote_snapshots[remote_revision] = dict(remote_snapshots[verified_revision])
        remote_snapshots[remote_revision]["unrelated/other-prefix/marker.json"] = b"unrelated"
    snapshot_roots = {}
    for revision, files in remote_snapshots.items():
        root = tmp_path / "hf-snapshot" / "snapshots" / revision
        snapshot_roots[revision] = root
        for filename, content in files.items():
            target = root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def download(**kwargs):
        pinned = kwargs.get("revision")
        selected_revision = pinned or remote_revision
        source = snapshot_roots[selected_revision] / kwargs["filename"]
        if not source.is_file():
            raise FileNotFoundError(source)
        cached = Path(kwargs["cache_dir"]) / "snapshots" / selected_revision / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, cached)
        return str(cached)

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    return run_json, token, queue, remote_manifest


def _archive_initial_discovery_fixture(tmp_path: Path, monkeypatch):
    resume_run_json, _resume_token, manifest, prefix_root, _ = _complete_archive_remote(
        tmp_path / "source", monkeypatch
    )
    resume_payload = json.loads(resume_run_json.read_text(encoding="utf-8"))
    config_payload = _runtime_config(
        tmp_path / "initial", publication_enabled=True
    ).as_dict()
    config_payload["run"].update({
        "run_id": "initial-archive-discovery",
        "validation_mode": False,
        "resume_from_hf": False,
    })
    config_payload["archive_profile"] = manifest["archive_profile"]
    config_payload.update(max_result_bytes=1_000_000, max_staging_bytes=20_000_000, staging_reserve_bytes=2_000_000)
    config = GenerationRuntimeConfig.from_dict(config_payload)
    run_json = generation_launch.persist_run_config(
        config,
        Path(resume_payload["approved_manifest"]),
        code_revision="a" * 40,
    )
    token = Path(config.run.run_root) / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    revision = prefix_root.parents[1].name
    return run_json, token, manifest, prefix_root, revision


def test_coordinator_same_run_restart_accepts_published_receipt_and_ingested_rows(
    tmp_path: Path, monkeypatch,
):
    run_json, token, queue, remote_manifest = _same_run_archive_restart_fixture(
        tmp_path, monkeypatch
    )

    control = CoordinatorControlPlane.open(
        run_json, api_factory=lambda: object(), token_path=token,
    )

    assert control.queue.counts().published == 1
    assert control.queue.counts().ingested == 1
    assert len(control.manifest["episodes"]) == 2
    assert control.planner.counts("T01").nominal_successes == 2
    assert control.publisher.remote_manifest["episodes"] == remote_manifest["episodes"]
    assert control.publisher.remote_manifest["episodes"][0]["job_id"] in {
        path.name for path in (queue.root / "published").iterdir()
    }
    bootstrap = json.loads(
        (Path(control.run.run_root) / "control" / "resume_bootstrap.json").read_text()
    )
    assert bootstrap["remote_revision"] == "d1cc82c27bc57602bf3fac40f55c6dbb59766d45"


def test_coordinator_same_run_restart_accepts_unrelated_repo_head_with_verified_prefix(
    tmp_path: Path, monkeypatch,
):
    run_json, token, _queue, _remote_manifest = _same_run_archive_restart_fixture(
        tmp_path,
        monkeypatch,
        remote_revision="f" * 40,
    )

    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: object(),
        token_path=token,
    )

    assert len(control.manifest["episodes"]) == 2
    assert control.publisher.remote_manifest["source_run_ids"] == ["run-1"]
    bootstrap = json.loads(
        (Path(control.run.run_root) / "control" / "resume_bootstrap.json").read_text()
    )
    assert bootstrap["remote_revision"] == "f" * 40


def test_coordinator_same_run_restart_accepts_data_committed_receipt_after_full_hash_check(
    tmp_path: Path, monkeypatch,
):
    receipt_revision = "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
    run_json, token, _queue, _remote_manifest = _same_run_archive_restart_fixture(
        tmp_path,
        monkeypatch,
        remote_revision=receipt_revision,
    )

    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: object(),
        token_path=token,
    )

    assert len(control.manifest["episodes"]) == 2
    bootstrap = json.loads(
        (Path(control.run.run_root) / "control" / "resume_bootstrap.json").read_text()
    )
    assert bootstrap["remote_revision"] == receipt_revision


def test_coordinator_same_run_restart_rejects_newer_head_with_changed_archive_bytes(
    tmp_path: Path, monkeypatch,
):
    head_revision = "f" * 40
    run_json, token, _queue, _remote_manifest = _same_run_archive_restart_fixture(
        tmp_path,
        monkeypatch,
        remote_revision=head_revision,
    )
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    runtime = json.loads(Path(run_payload["runtime_config_path"]).read_text(encoding="utf-8"))
    prefix_root = (
        tmp_path
        / "hf-snapshot"
        / "snapshots"
        / head_revision
        / runtime["run"]["hf_subfolder"]
    )
    chunk = next((prefix_root / "episodes").rglob("chunk-00000.npz"))
    chunk.write_bytes(chunk.read_bytes() + b"tampered")

    api_calls = []
    with pytest.raises(RuntimeError, match="archive startup requires a valid remote dataset manifest") as failure:
        CoordinatorControlPlane.open(
            run_json,
            api_factory=lambda: api_calls.append(True),
            token_path=token,
        )

    assert "remote archive hash mismatch" in str(failure.value.__cause__)
    assert api_calls == []


@pytest.mark.parametrize("local_damage", ["missing", "conflicting_local_row"])
def test_coordinator_same_run_restart_rejects_remote_rows_without_matching_local_evidence(
    tmp_path: Path, monkeypatch, local_damage: str,
):
    run_json, token, queue, remote_manifest = _same_run_archive_restart_fixture(
        tmp_path, monkeypatch
    )
    remote_job_id = remote_manifest["episodes"][0]["job_id"]
    if local_damage == "missing":
        shutil.rmtree(queue.root / "published" / remote_job_id)
    else:
        receipt_path = (
            queue.root / "published" / remote_job_id / "publication_receipt.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["manifest_row"]["preprocessing_identity"] = "conflicting_local_receipt"
        receipt["manifest_row_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    receipt["manifest_row"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    api_calls = []
    with pytest.raises(RuntimeError, match="archive startup requires a valid remote dataset manifest") as failure:
        CoordinatorControlPlane.open(
            run_json,
            api_factory=lambda: api_calls.append(True),
            token_path=token,
        )

    assert "same-run remote archive row" in str(failure.value.__cause__)
    assert api_calls == []


def test_coordinator_hf_only_resume_rejects_same_source_run_id(tmp_path: Path, monkeypatch):
    run_json, token, _manifest, _prefix_root, _downloads = _complete_archive_remote(
        tmp_path, monkeypatch
    )
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    payload["run"]["run_id"] = "run-1"
    runtime_path = Path(payload["runtime_config_path"])
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_payload["run"]["run_id"] = "run-1"
    runtime_bytes = (
        json.dumps(runtime_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    runtime_path.write_bytes(runtime_bytes)
    payload["runtime_config"] = runtime_payload
    payload["runtime_config_sha256"] = hashlib.sha256(runtime_bytes).hexdigest()
    bootstrap_path = Path(payload["run"]["run_root"]) / "control" / "resume_bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap["run_id"] = "run-1"
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    run_json.write_text(json.dumps(payload), encoding="utf-8")

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            run_json,
            api_factory=lambda: api_calls.append(True),
            token_path=token,
        )

    assert "resumed run_id must be disjoint" in str(failure.value.__cause__)
    assert api_calls == []


def test_coordinator_archive_initial_manifest_discovery_uses_owned_cache_and_cleans_it(
    tmp_path: Path, monkeypatch,
):
    run_json, token, manifest, prefix_root, revision = _archive_initial_discovery_fixture(
        tmp_path, monkeypatch
    )
    remote_manifest = prefix_root / "dataset_manifest.json"
    calls = []
    cache_roots = []
    returned_paths = []

    def download(**kwargs):
        calls.append(kwargs)
        cache_arg = kwargs.get("cache_dir")
        if cache_arg is None:
            return str(remote_manifest)
        cache = Path(cache_arg)
        cache_roots.append(cache)
        cached = cache / "snapshots" / revision / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(remote_manifest.read_bytes())
        returned_paths.append(cached)
        return str(cached)

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    control = CoordinatorControlPlane.open(
        run_json, api_factory=lambda: object(), token_path=token
    )

    assert len(calls) == 1
    assert calls[0]["filename"].endswith("/dataset_manifest.json")
    assert calls[0].get("cache_dir")
    assert calls[0].get("revision") is None
    assert control.publisher.remote_manifest["episodes"] == manifest["episodes"]
    assert len(cache_roots) == 1 and not cache_roots[0].exists()
    assert len(returned_paths) == 1 and not returned_paths[0].exists()
    assert generation_coordinator_module._snapshot_revision(returned_paths[0]) == revision


def test_coordinator_archive_initial_manifest_discovery_rejects_outside_cache_path(
    tmp_path: Path, monkeypatch,
):
    run_json, token, _manifest, prefix_root, _revision = _archive_initial_discovery_fixture(
        tmp_path, monkeypatch
    )
    remote_manifest = prefix_root / "dataset_manifest.json"
    calls = []
    cache_roots = []

    def download(**kwargs):
        calls.append(kwargs)
        cache_arg = kwargs.get("cache_dir")
        if cache_arg is not None:
            cache = Path(cache_arg)
            cache_roots.append(cache)
            temporary_copy = cache / "downloaded-control.tmp"
            temporary_copy.write_bytes(remote_manifest.read_bytes())
        return str(remote_manifest)

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    with pytest.raises(
        RuntimeError, match="archive startup requires a valid remote dataset manifest"
    ) as failure:
        CoordinatorControlPlane.open(
            run_json, api_factory=lambda: object(), token_path=token
        )

    assert len(calls) == 1
    assert calls[0].get("cache_dir")
    assert isinstance(failure.value.__cause__, ValueError)
    assert "outside owned scratch" in str(failure.value.__cause__)
    assert len(cache_roots) == 1 and not cache_roots[0].exists()


def test_coordinator_archive_initial_manifest_allows_confirmed_remote_absence(
    tmp_path: Path, monkeypatch,
):
    from huggingface_hub.errors import EntryNotFoundError

    run_json, token, _manifest, _prefix_root, _revision = _archive_initial_discovery_fixture(
        tmp_path, monkeypatch
    )
    calls = []
    cache_roots = []

    def download(**kwargs):
        calls.append(kwargs)
        cache = Path(kwargs["cache_dir"])
        cache_roots.append(cache)
        (cache / "partial-download.tmp").write_bytes(b"temporary cache bytes")
        raise EntryNotFoundError("dataset manifest is absent")

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    control = CoordinatorControlPlane.open(
        run_json, api_factory=lambda: object(), token_path=token
    )

    assert len(calls) == 1
    assert calls[0].get("cache_dir")
    assert calls[0].get("revision") is None
    assert len(cache_roots) == 1 and not cache_roots[0].exists()
    assert control.publisher.remote_manifest["episodes"] == []


def test_coordinator_archive_resume_restores_hf_rows_without_local_payload(tmp_path: Path, monkeypatch):
    run_json, token, manifest, prefix_root, downloads = _complete_archive_remote(tmp_path, monkeypatch)
    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: object(),
        token_path=token,
    )

    assert control.queue.counts().ingested == 0
    assert control.queue.counts().published == 0
    assert len(control.manifest["episodes"]) == 1
    assert len(control.manifest["failure_attempts"]) == 0
    assert control.planner.counts("T01").nominal_successes == 1
    current_receipt = json.loads((prefix_root / "publication_receipt.json").read_text())
    assert current_receipt["status"] == "VERIFIED"
    assert current_receipt["commit_oid"] == "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
    assert current_receipt["data_commit_oid"] == "740e07d6648cdf099c78bca78a9b7195d7da5d6a"
    assert any(
        item.get("revision") == "d1cc82c27bc57602bf3fac40f55c6dbb59766d45"
        and item["filename"].endswith("/publication_receipt.json")
        for item in downloads
    )
    assert any(
        item.get("revision") == "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
        and item["filename"].endswith("/publication_receipt.json")
        for item in downloads
    )
    assert any(
        item.get("revision") == "740e07d6648cdf099c78bca78a9b7195d7da5d6a"
        and item["filename"].endswith("/dataset_manifest.json")
        for item in downloads
    )
    assert {item["filename"] for item in downloads} == {
        f"{prefix_root.relative_to(prefix_root.parents[1])}/{relative}"
        for relative in ("dataset_manifest.json", "resume_receipt.json", "publication_receipt.json")
    } | {
        f"{prefix_root.relative_to(prefix_root.parents[1])}/{manifest['episodes'][0]['archive_ref']}/{relative}"
        for relative in manifest["episodes"][0]["file_sha256"]
    } | {
        f"{prefix_root.relative_to(prefix_root.parents[1])}/views/provisional/episodes/"
        f"{manifest['episodes'][0]['program_id']}/{manifest['episodes'][0]['episode_id']}/{view}.json"
        for view in ("D_geom", "D_temporal", "D_dyn", "D_task")
    }


def test_coordinator_archive_resume_reopens_bound_crash_attempt(tmp_path: Path, monkeypatch):
    run_json, token, manifest, _prefix_root, _downloads = _complete_archive_remote(
        tmp_path,
        monkeypatch,
        archive_outcome="simulator_crash",
    )

    control = CoordinatorControlPlane.open(
        run_json,
        api_factory=lambda: object(),
        token_path=token,
    )

    assert control.manifest["episodes"] == []
    assert len(control.manifest["failure_attempts"]) == 1
    row = control.manifest["failure_attempts"][0]
    assert row["split"] == "train"
    assert row["subset"] == manifest["failure_attempts"][0]["subset"]
    assert control.planner.counts("T01").nominal_crashes == 1


def test_coordinator_archive_resume_requires_verified_receipt_to_bind_data_committed_snapshot(
    tmp_path: Path, monkeypatch,
):
    run_json, token, _manifest, latest_prefix, _downloads = _complete_archive_remote(
        tmp_path, monkeypatch,
    )
    latest_snapshot = latest_prefix.parents[1]
    relative_prefix = latest_prefix.relative_to(latest_snapshot)
    prior_prefix = (
        latest_snapshot.parent
        / "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
        / relative_prefix
    )
    prior_receipt_path = prior_prefix / "publication_receipt.json"
    prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    prior_receipt["status"] = "PREPARED"
    prior_receipt_path.write_text(json.dumps(prior_receipt), encoding="utf-8")

    api_calls = []
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            run_json,
            api_factory=lambda: api_calls.append(True),
            token_path=token,
        )

    assert "DATA_COMMITTED receipt" in str(failure.value.__cause__)
    assert api_calls == []


def test_coordinator_archive_resume_accepts_data_committed_receipt_at_pinned_receipt_revision(
    tmp_path: Path, monkeypatch,
):
    receipt_revision = "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
    run_json, token, manifest, prefix_root, downloads = _complete_archive_remote(
        tmp_path,
        monkeypatch,
        remote_revision=receipt_revision,
    )

    control = CoordinatorControlPlane.open(
        run_json, api_factory=lambda: object(), token_path=token,
    )

    assert control.publisher.remote_manifest["episodes"] == manifest["episodes"]
    assert control.planner.counts("T01").nominal_successes == 1
    remote_receipt = json.loads((prefix_root / "publication_receipt.json").read_text())
    assert remote_receipt["status"] == "DATA_COMMITTED"
    assert remote_receipt["data_commit_oid"] == "740e07d6648cdf099c78bca78a9b7195d7da5d6a"
    assert remote_receipt["commit_oid"] is None
    assert any(
        item.get("revision") == "740e07d6648cdf099c78bca78a9b7195d7da5d6a"
        and item["filename"].endswith("/dataset_manifest.json")
        for item in downloads
    )


def test_coordinator_archive_resume_rejects_data_committed_receipt_from_other_snapshot_oid(
    tmp_path: Path, monkeypatch,
):
    run_json, token, _manifest, prefix_root, _downloads = _complete_archive_remote(
        tmp_path, monkeypatch
    )
    ordinary_download = generation_coordinator_module.hf_hub_download
    wrong_receipt_revision = "0" * 40

    def download(**kwargs):
        source = Path(ordinary_download(**kwargs))
        if kwargs["filename"].endswith("/publication_receipt.json"):
            target = (
                Path(kwargs["cache_dir"])
                / "snapshots"
                / wrong_receipt_revision
                / kwargs["filename"]
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return str(target)
        return str(source)

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)

    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            run_json, api_factory=lambda: object(), token_path=token,
        )

    assert "snapshot revision mismatch" in str(failure.value.__cause__)


def test_coordinator_data_committed_resume_requires_every_archive_file_at_pinned_revision(
    tmp_path: Path, monkeypatch,
):
    receipt_revision = "99bb4ee1fc49b3614b6ada358f1d02dcbe568990"
    run_json, token, manifest, prefix_root, _downloads = _complete_archive_remote(
        tmp_path,
        monkeypatch,
        remote_revision=receipt_revision,
    )
    missing_chunk = (
        prefix_root
        / manifest["episodes"][0]["archive_ref"]
        / "data"
        / "chunk-00000.npz"
    )
    missing_chunk.unlink()

    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            run_json, api_factory=lambda: object(), token_path=token,
        )

    assert isinstance(failure.value.__cause__, FileNotFoundError)


def test_coordinator_archive_resume_rejects_data_commit_oid_mismatch(tmp_path: Path, monkeypatch):
    run_json, token, _manifest, prefix_root, _downloads = _complete_archive_remote(
        tmp_path, monkeypatch
    )
    receipt_path = prefix_root / "publication_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "DATA_COMMITTED"
    receipt["commit_oid"] = None
    receipt["data_commit_oid"] = "0" * 40
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(
            run_json, api_factory=lambda: object(), token_path=token,
        )

    assert "resume/publication receipt identity mismatch" in str(failure.value.__cause__)


def test_archive_preflight_pins_local_fake_manifest(tmp_path: Path, monkeypatch):
    run_json, token, _, _, downloads = _complete_archive_remote(tmp_path, monkeypatch)
    run_payload = json.loads(run_json.read_text())
    config = GenerationRuntimeConfig.from_file(run_payload["runtime_config_path"], check_paths=False)
    (Path(config.run.run_root) / "control" / "resume_bootstrap.json").unlink()
    bootstrap = generation_launch.preflight_resume_manifest(
        config, token_path=token, run_root=config.run.run_root,
        approved_manifest=run_payload["approved_manifest"],
        downloader=generation_coordinator_module.hf_hub_download,
    )
    assert bootstrap["remote_revision"] == "d1cc82c27bc57602bf3fac40f55c6dbb59766d45"
    assert downloads[0].get("revision") is None
    assert all(
        call.get("revision") == "d1cc82c27bc57602bf3fac40f55c6dbb59766d45"
        for call in downloads[1:]
    )
    assert len(downloads) == 2


def _refresh_archive_remote_hashes(manifest: dict, prefix_root: Path) -> None:
    row = manifest["episodes"][0]
    archive = prefix_root / row["archive_ref"]
    manifest_path = archive / "episode.manifest.json"
    artifact_path = archive / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["files"]["episode.manifest.json"]["bytes"] = manifest_path.stat().st_size
    artifact["files"]["episode.manifest.json"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    row["file_sha256"] = {
        str(path.relative_to(archive)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in archive.rglob("*") if path.is_file()
    }
    dataset_path = prefix_root / "dataset_manifest.json"
    dataset_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    for receipt_name in ("resume_receipt.json", "publication_receipt.json"):
        path = prefix_root / receipt_name
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["dataset_manifest_sha256"] = dataset_sha
        if receipt_name == "publication_receipt.json":
            receipt["artifact_hashes"] = {
                f"{row['job_id']}/{relative}": digest
                for relative, digest in row["file_sha256"].items()
            }
        path.write_text(json.dumps(receipt), encoding="utf-8")


@pytest.mark.parametrize("missing", [
    "episode.manifest.json", "artifact_manifest.json", "debug.json", "data/chunk-00000.npz",
])
def test_archive_resume_rejects_missing_pinned_file(tmp_path: Path, monkeypatch, missing: str):
    run_json, token, manifest, prefix_root, _ = _complete_archive_remote(tmp_path, monkeypatch)
    (prefix_root / manifest["episodes"][0]["archive_ref"] / missing).unlink()
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)
    assert isinstance(failure.value.__cause__, FileNotFoundError)


def test_archive_resume_rejects_unpinned_revision(tmp_path: Path, monkeypatch):
    run_json, token, _, _, downloads = _complete_archive_remote(tmp_path, monkeypatch)
    bootstrap_path = Path(json.loads(run_json.read_text())["run"]["run_root"]) / "control" / "resume_bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    bootstrap["remote_revision"] = None
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned lowercase HF commit OID"):
        CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)
    assert downloads == []


@pytest.mark.parametrize("receipt_name,field", [
    ("resume_receipt.json", "dataset_manifest_sha256"),
    ("resume_receipt.json", "dataset_manifest_revision"),
    ("publication_receipt.json", "dataset_manifest_sha256"),
    ("publication_receipt.json", "artifact_hashes"),
])
def test_archive_resume_rejects_receipt_mismatch(tmp_path: Path, monkeypatch, receipt_name: str, field: str):
    run_json, token, _, prefix_root, _ = _complete_archive_remote(tmp_path, monkeypatch)
    receipt_path = prefix_root / receipt_name
    receipt = json.loads(receipt_path.read_text())
    if field == "artifact_hashes":
        receipt[field] = {}
    elif field == "dataset_manifest_revision":
        receipt[field] = "0" * 40
    else:
        receipt[field] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)
    assert "receipt" in str(failure.value.__cause__)


@pytest.mark.parametrize("malformation", ["duplicate_chunk", "omitted_chunk", "omitted_array_spec"])
def test_archive_resume_rejects_self_consistent_malformed_archive(
    tmp_path: Path, monkeypatch, malformation: str,
):
    run_json, token, manifest, prefix_root, _ = _complete_archive_remote(tmp_path, monkeypatch)
    archive_path = prefix_root / manifest["episodes"][0]["archive_ref"] / "episode.manifest.json"
    archive = json.loads(archive_path.read_text())
    if malformation == "duplicate_chunk":
        archive["chunk_inventory"].append(dict(archive["chunk_inventory"][0]))
    elif malformation == "omitted_chunk":
        archive["chunk_inventory"].clear()
    else:
        archive["array_specs"].pop(next(iter(archive["array_specs"])))
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    _refresh_archive_remote_hashes(manifest, prefix_root)
    bootstrap_path = Path(json.loads(run_json.read_text())["run"]["run_root"]) / "control" / "resume_bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    bootstrap["remote_manifest_sha256"] = hashlib.sha256((prefix_root / "dataset_manifest.json").read_bytes()).hexdigest()
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)
    assert any(word in str(failure.value.__cause__) for word in ("chunk", "array"))


@pytest.mark.parametrize("field,value", [
    ("local_artifact_retention", "keep"),
    ("view_status", "final"),
    ("max_chunk_bytes", 134217728),
])
def test_archive_resume_rejects_rehashed_archive_profile_mismatch(
    tmp_path: Path, monkeypatch, field: str, value: object,
):
    run_json, token, manifest, prefix_root, _ = _complete_archive_remote(tmp_path, monkeypatch)
    archive_path = prefix_root / manifest["episodes"][0]["archive_ref"] / "episode.manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["archive_profile"][field] = value
    if field == "max_chunk_bytes":
        archive["local_write_limits"]["max_chunk_bytes"] = value
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    _refresh_archive_remote_hashes(manifest, prefix_root)
    bootstrap_path = Path(json.loads(run_json.read_text())["run"]["run_root"]) / "control" / "resume_bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text())
    bootstrap["remote_manifest_sha256"] = hashlib.sha256((prefix_root / "dataset_manifest.json").read_bytes()).hexdigest()
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume_from_hf") as failure:
        CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)
    assert "archive_profile" in str(failure.value.__cause__)


@pytest.mark.parametrize("field,value,message", [
    ("split", "test", "split disagrees with catalog"),
    ("episode_kind", "unknown", "episode_kind is invalid"),
])
def test_archive_resume_rejects_self_consistent_invalid_plan(field: str, value: str, message: str):
    profile = _archive_resume_profile()
    run = RunConfig(run_id="new-run", run_root="/tmp/new-run", code_revision="a" * 40,
                    approved_manifest_sha256="b" * 64, resume_from_hf=True)
    manifest = _archive_resume_manifest(profile)
    row = manifest["episodes"][0]
    row[field] = value
    row["attempt_plan"][field] = value
    with pytest.raises(ValueError, match=message):
        load_remote_manifest(manifest, run, archive_profile=profile, approved_program_ids={"T01"})


def test_archive_resume_rejects_asset_family_outside_approved_catalog():
    profile = _archive_resume_profile()
    run = RunConfig(run_id="new-run", run_root="/tmp/new-run", code_revision="a" * 40,
                    approved_manifest_sha256="b" * 64, resume_from_hf=True)
    manifest = _archive_resume_manifest(profile)
    with pytest.raises(ValueError, match="asset_family_id disagrees with approved catalog"):
        load_remote_manifest(manifest, run, archive_profile=profile,
                             approved_rows={"T01": {"asset_family_id": "approved-family"}})


def test_archive_resume_rejects_self_consistent_invalid_asset_instance():
    profile = _archive_resume_profile()
    run = RunConfig(run_id="new-run", run_root="/tmp/new-run", code_revision="a" * 40,
                    approved_manifest_sha256="b" * 64, resume_from_hf=True)
    manifest = _archive_resume_manifest(profile)
    row = manifest["episodes"][0]
    row["asset_instance_id"] = "unrelated-asset"
    row["attempt_plan"]["randomization"]["asset_instance_id"] = "unrelated-asset"
    with pytest.raises(ValueError, match="asset_instance_id disagrees with planner seed"):
        load_remote_manifest(manifest, run, archive_profile=profile, approved_program_ids={"T01"})


def test_archive_resume_restores_success_valid_failure_and_crash_quota_without_payload():
    profile = _archive_resume_profile()
    run = RunConfig(run_id="new-run", run_root="/tmp/new-run", code_revision="a" * 40,
                    approved_manifest_sha256="b" * 64, resume_from_hf=True)
    manifest = load_remote_manifest(
        _archive_resume_manifest(profile), run, archive_profile=profile,
        approved_program_ids={"T01"},
    )
    approved = json.loads(Path("artifacts/composition/approved_composition_manifest.json").read_text())
    row = next(item for item in approved["catalog"] if item["program_id"] == "T01")
    planner = DistributedPlanner.from_manifest(run, {"T01": row}, manifest)
    counts = planner.counts("T01")
    assert counts.nominal_successes == 1
    assert counts.nominal_valid_failures == 1
    assert counts.nominal_crashes == 1


def test_coordinator_resume_reuses_launcher_pinned_revision(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    payload = config.as_dict()
    payload["run"].update({"resume_from_hf": True, "run_id": "resume-pinned"})
    config = GenerationRuntimeConfig.from_dict(payload)
    approved = Path("artifacts/composition/approved_composition_manifest.json")
    run_json = generation_launch.persist_run_config(config, approved, code_revision="a" * 40)
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    remote = tmp_path / "remote-manifest.json"
    remote.write_text(json.dumps({
        "manifest_version": 3,
        "source_run_ids": ["old-run"],
        "episodes": [],
        "failure_attempts": [],
    }), encoding="utf-8")
    bootstrap = Path(config.run.run_root) / "control" / "resume_bootstrap.json"
    bootstrap.write_text(json.dumps({
        "run_id": config.run.run_id,
        "remote_revision": "pinned-revision",
    }), encoding="utf-8")
    revisions = []

    def download(**kwargs):
        revisions.append(kwargs.get("revision"))
        return str(remote)

    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)

    assert revisions == ["pinned-revision"]
    assert json.loads(bootstrap.read_text())["remote_revision"] == "pinned-revision"


def test_coordinator_resume_fails_closed_when_remote_manifest_is_unavailable(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    payload = config.as_dict()
    payload["run"]["resume_from_hf"] = True
    config = GenerationRuntimeConfig.from_dict(payload)
    approved = Path("artifacts/composition/approved_composition_manifest.json")
    run_json = generation_launch.persist_run_config(config, approved, code_revision="a" * 40)
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    monkeypatch.setattr(
        generation_coordinator_module,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(OSError("remote unavailable")),
    )

    with pytest.raises(RuntimeError, match="resume_from_hf"):
        CoordinatorControlPlane.open(
            run_json,
            api_factory=lambda: object(),
            token_path=token,
        )


def test_coordinator_open_rejects_tampered_runtime_snapshot(tmp_path: Path, monkeypatch):
    config = _runtime_config(tmp_path, publication_enabled=True)
    run_json = generation_launch.persist_run_config(
        config,
        Path("artifacts/composition/approved_composition_manifest.json"),
        code_revision="a" * 40,
    )
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    runtime_path = Path(run_payload["runtime_config_path"])
    runtime_path.write_text(runtime_path.read_text(encoding="utf-8").replace("control-test", "tampered"), encoding="utf-8")
    monkeypatch.setattr(
        generation_coordinator_module,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(OSError("remote unavailable")),
    )

    with pytest.raises(ValueError, match="runtime config digest"):
        CoordinatorControlPlane.open(
            run_json,
            api_factory=lambda: object(),
            token_path=token,
            runtime_config_path=runtime_path,
        )


def test_coordinator_open_rejects_run_config_validation_mode_mismatch_before_remote_io(
    tmp_path: Path, monkeypatch,
):
    config = _runtime_config(tmp_path, publication_enabled=True)
    run_json = generation_launch.persist_run_config(
        config,
        Path("artifacts/composition/approved_composition_manifest.json"),
        code_revision="a" * 40,
    )
    token = tmp_path / "hf-token"
    token.write_text("secret\n", encoding="utf-8")
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    run_payload["hf_token_path"] = str(token)
    run_payload["run"]["validation_mode"] = False
    run_json.write_text(json.dumps(run_payload), encoding="utf-8")
    remote = tmp_path / "remote-manifest.json"
    remote.write_text(json.dumps({
        "manifest_version": 3,
        "source_run_ids": [],
        "episodes": [],
        "failure_attempts": [],
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        generation_coordinator_module,
        "hf_hub_download",
        lambda **kwargs: calls.append(kwargs) or str(remote),
    )

    with pytest.raises(ValueError, match="validation_mode.*run config"):
        CoordinatorControlPlane.open(run_json, api_factory=lambda: object(), token_path=token)

    assert calls == []


def test_no_stop_call_in_control_sources():
    for name in (
        "scripts/generation_launch.py",
        "scripts/generation_coordinator.py",
        "scripts/generation_watchdog.py",
    ):
        assert "colab stop" not in Path(name).read_text(encoding="utf-8")


def test_coordinator_uses_configured_credential_path(tmp_path: Path):
    token_path = tmp_path / "credentials" / "hf-token"
    token_path.parent.mkdir()
    token_path.write_text("secret", encoding="utf-8")

    resolved = _credential_path(
        tmp_path / "control" / "run.json",
        {"hf_token_path": str(token_path)},
    )

    assert resolved == token_path


def test_validation_plan_is_persisted_as_machine_readable_receipt(tmp_path: Path):
    plan_path = Path("tests/fixtures/generation_validation_config.json")
    receipt_path = tmp_path / "control" / "validation_receipt.json"

    assert hasattr(generation_launch, "persist_validation_receipt")
    receipt = generation_launch.persist_validation_receipt(plan_path, receipt_path)

    assert receipt_path.is_file()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_RUN"
    assert payload["plan_id"] == "generation-validation-cpu-20260922"
    assert set(payload["gates"]) == {
        "success",
        "valid_failure",
        "invalid_observation",
        "infrastructure_failure",
        "malformed_result",
        "coordinator_restart",
        "publication",
    }
    assert all(row["status"] == "NOT_RUN" for row in payload["gates"].values())
    assert all(row["reason"] for row in payload["gates"].values())
    assert receipt.status == "NOT_RUN"


def test_launcher_exports_pinned_simulator_environment_to_workers(tmp_path: Path):
    config = _runtime_config(tmp_path)
    environment = _required_api("build_process_environment")(config, base={})
    assert environment["COPPELIASIM_ROOT"] == config.machine.simulator_root
    assert environment["LD_LIBRARY_PATH"] == config.machine.simulator_root
    assert environment["QT_QPA_PLATFORM_PLUGIN_PATH"] == config.machine.simulator_root
    assert str(Path(config.machine.repo_root) / "src") in environment["PYTHONPATH"]


def test_coordinator_bounds_ingestion_per_tick():
    text = Path("scripts/generation_coordinator.py").read_text(encoding="utf-8")
    assert "MAX_READY_PER_TICK = 100" in text
    assert "self.queue.iter_ready()[:MAX_READY_PER_TICK]" in text


@pytest.mark.parametrize(
    "malformed_result_json",
    [False, True],
    ids=["validation-failure", "malformed-result-json"],
)
def test_tick_quarantines_bad_result_and_ingests_following_valid_result(
    tmp_path: Path,
    monkeypatch,
    malformed_result_json: bool,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    run = RunConfig(
        run_id="quarantine-run",
        run_root=str(run_root),
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        worker_count=2,
    )
    queue = FilesystemJobQueue(run_root / "queue")

    def make_job(job_id: str, index: int) -> GenerationJob:
        plan = AttemptPlan(
            program_id="T01", split="train", episode_index=index,
            episode_id=f"episode-t01-{index:06d}", episode_kind="nominal",
            scene_seed=index, collection_seed=20260920,
            randomization={"scene_signature": f"sig-{index}", "asset_instance_id": f"asset-{index}"},
            intervention=None,
        )
        return GenerationJob.create(
            job_id=job_id, run_id=run.run_id, attempt_id=f"att-{plan.episode_id}",
            episode_id=plan.episode_id, program_id="T01", plan=plan,
            code_revision="a" * 40, manifest_sha256="b" * 64,
            output_root=str(run_root / "staging"),
        )

    malformed_job = make_job("job-a-malformed", 1)
    valid_job = make_job("job-b-valid", 2)
    ready_results = []
    for worker_id, job, content in (
        ("000", malformed_job, b"malformed artifact"),
        ("001", valid_job, b"valid artifact"),
    ):
        result_dir = run_root / "staging" / "worker-results" / job.job_id / "T01"
        result_dir.mkdir(parents=True)
        (result_dir / "artifact.bin").write_bytes(content)
        queue.enqueue(job)
        assert queue.claim(worker_id) == job
        result = WorkerResult(
            job_id=job.job_id, attempt_id=job.attempt_id, episode_id=job.episode_id,
            program_id=job.program_id, outcome="success", result_dir=str(result_dir),
            file_sha256={}, timeline={"actions": 0, "observations": 1, "durations": 0},
        )
        ready_path = queue.publish_ready(worker_id, result)
        if malformed_result_json and job.job_id == malformed_job.job_id:
            (ready_path / "result.json").write_text("{ malformed", encoding="utf-8")
        ready_results.append(result)

    class Planner:
        def __init__(self):
            self.recorded = []

        def next_job(self):
            return None

        def record_result(self, result, provenance):
            self.recorded.append((result, provenance))

        def quota_complete(self):
            return False

        def snapshot(self):
            return SimpleNamespace(as_dict=lambda: {})

    class Publisher:
        def __init__(self):
            self.heartbeat_during_publish = []

        def publish_due(self, **kwargs):
            heartbeat_path = run_root / "control" / "coordinator-heartbeat.json"
            self.heartbeat_during_publish.append(json.loads(heartbeat_path.read_text()))
            return None

    planner = Planner()
    publisher = Publisher()
    control = CoordinatorControlPlane(run, queue, planner, publisher)

    def validate(job, result):
        if job.job_id == malformed_job.job_id:
            raise ValueError("invalid artifact inventory")
        return ValidatedResult(
            job=job,
            result=result,
            provenance={"episode_kind": "nominal"},
            episode_entry={
                "attempt_id": job.attempt_id,
                "episode_id": job.episode_id,
                "program_id": job.program_id,
                "outcome": "success",
            },
            attempt_entry=None,
        )

    def ingest(manifest, validated):
        updated = dict(manifest)
        updated["episodes"] = list(updated.get("episodes") or []) + [validated.episode_entry]
        updated["failure_attempts"] = list(updated.get("failure_attempts") or [])
        return updated

    monkeypatch.setattr(generation_coordinator_module, "validate_closed_result", validate)
    monkeypatch.setattr(generation_coordinator_module, "ingest_validated_result", ingest)
    assert hasattr(control, "process_ready_result"), "process_ready_result is required"
    process_ready_result = control.process_ready_result
    outcomes = []

    def record_processing(result):
        outcome = process_ready_result(result)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(control, "process_ready_result", record_processing)
    control.tick(now_s=10.0)

    assert outcomes == ["quarantined", "ingested"]
    assert control.status == "RUNNING"
    assert [result.job_id for result, _ in planner.recorded] == [valid_job.job_id]
    assert [row["episode_id"] for row in control.manifest["episodes"]] == [valid_job.episode_id]
    assert control.manifest["failure_attempts"] == []
    counts = queue.counts()
    assert counts.quarantined == 1
    assert counts.ingested == 1
    heartbeat = json.loads(
        (run_root / "control" / "coordinator-heartbeat.json").read_text(encoding="utf-8")
    )
    assert heartbeat["phase"] == "tick_end"
    assert heartbeat["tick_started_at_s"] == 10.0
    assert heartbeat["last_progress_at_s"] == 10.0
    assert heartbeat["last_progress_kind"] == "result_ingested"
    assert heartbeat["last_validation_error"]["job_id"] == malformed_job.job_id
    assert heartbeat["publication"]["phase"] == "idle"
    assert heartbeat["queue"]["quarantined"] == 1
    assert heartbeat["planner"] == {}
    assert publisher.heartbeat_during_publish[0]["phase"] == "publication_in_progress"
    assert publisher.heartbeat_during_publish[0]["publication"]["phase"] == "publication_in_progress"
    control.tick(now_s=20.0)
    later_heartbeat = json.loads(
        (run_root / "control" / "coordinator-heartbeat.json").read_text(encoding="utf-8")
    )
    assert later_heartbeat["last_progress_at_s"] == 10.0
    assert later_heartbeat["last_progress_kind"] == "result_ingested"
    assert tuple(job.job_id for job in _inflight_jobs_from_queue(queue)) == (
        malformed_job.job_id,
    )
    failure = json.loads((queue.root / "quarantined" / malformed_job.job_id / "validation_failure.json").read_text())
    if malformed_result_json:
        assert failure["result_identity"]["job_id"] == malformed_job.job_id
        assert failure["result_identity"]["result_json_sha256"] == hashlib.sha256(
            b"{ malformed"
        ).hexdigest()
        assert failure["exception_type"] == "JSONDecodeError"
        assert "artifact.bin" in failure["actual_file_inventory"]
    else:
        assert failure["result_identity"]["outcome"] == "success"
        assert failure["exception_message"] == "invalid artifact inventory"


def test_launcher_requires_real_code_revision_for_run_contract():
    text = Path("scripts/generation_launch.py").read_text(encoding="utf-8")
    assert '"--code-revision"' in text
    assert '"code_revision": "unknown"' not in text


def test_launcher_detaches_child_output_from_control_pipe():
    text = Path("scripts/generation_launch.py").read_text(encoding="utf-8")
    assert "stdout=subprocess.DEVNULL" in text
    assert "coordinator.log" in text


def test_coordinator_lock_rejects_second_owner_and_becomes_free(tmp_path):
    lock_path = tmp_path / "control" / "coordinator.lock"
    with CoordinatorLock(lock_path):
        assert _coordinator_lock_is_free(lock_path) is False
        import pytest
        with pytest.raises(RuntimeError, match="coordinator lock is already held"):
            with CoordinatorLock(lock_path):
                pass
    assert _coordinator_lock_is_free(lock_path) is True


def test_pid_identity_requires_every_expected_cmdline_token():
    reader = lambda pid: b"python\0worker.py\0--worker-id\0" + b"007" + b"\0--run-root\0/content/run\0"
    assert _pid_matches(12, ("worker.py", "007", "/content/run"), cmdline_reader=reader)
    assert not _pid_matches(12, ("worker.py", "008", "/content/run"), cmdline_reader=reader)
    assert not _pid_matches(12, ("worker.py", "007", "/content/other"), cmdline_reader=reader)


def test_pid_identity_accepts_configured_xvfb_wrapper_command_line(tmp_path: Path):
    python = str(tmp_path / "venv" / "bin" / "python")
    worker = str(tmp_path / "repo" / "scripts" / "generation_worker.py")
    runtime = str(tmp_path / "run" / "control" / "runtime_config.json")
    manifest = str(tmp_path / "approved.json")
    wrapper_command = (
        "xvfb-run",
        "--server-num",
        "41",
        "-s",
        "-screen 0 1366x768x24 +extension GLX +render -noreset",
    )
    worker_command = (
        python,
        "-B",
        worker,
        "--worker-id",
        "001",
        "--runtime-config",
        runtime,
        "--approved-manifest",
        manifest,
    )
    wrapped = ("/bin/sh", "/usr/bin/xvfb-run", *wrapper_command[1:], *worker_command)

    assert _pid_matches(
        12,
        expected_command=worker_command,
        wrapper_command=wrapper_command,
        cmdline_reader=lambda pid: shlex.join(wrapped),
    )


def test_pid_identity_accepts_normal_worker_and_rejects_embedded_command(tmp_path: Path):
    python = str(tmp_path / "venv" / "bin" / "python")
    worker = str(tmp_path / "repo" / "scripts" / "generation_worker.py")
    runtime = str(tmp_path / "run" / "control" / "runtime_config.json")
    manifest = str(tmp_path / "approved.json")
    worker_command = (
        python,
        "-B",
        worker,
        "--worker-id",
        "001",
        "--runtime-config",
        runtime,
        "--approved-manifest",
        manifest,
    )

    assert _pid_matches(
        12,
        expected_command=worker_command,
        wrapper_command=("xvfb-run", "--server-num", "41"),
        cmdline_reader=lambda pid: shlex.join(worker_command),
    )
    assert not _pid_matches(
        12,
        expected_command=worker_command,
        wrapper_command=("xvfb-run", "--server-num", "41"),
        cmdline_reader=lambda pid: shlex.join(
            (python, "-c", shlex.join(worker_command))
        ),
    )


@pytest.mark.parametrize(
    ("payload", "now_s", "stale_after_s", "expected"),
    [
        (
            {
                "status": "RUNNING",
                "phase": "tick_end",
                "timestamp_s": 100.0,
                "tick_started_at_s": 95.0,
                "last_progress_at_s": 100.0,
                "last_progress_kind": "result_ingested",
                "last_validation_error": None,
                "publication": {"phase": "idle"},
            },
            110.0,
            30.0,
            "healthy",
        ),
        (
            {
                "status": "RUNNING",
                "phase": "publication_in_progress",
                "timestamp_s": 100.0,
                "tick_started_at_s": 95.0,
                "last_progress_at_s": 80.0,
                "last_progress_kind": "result_ingested",
                "last_validation_error": None,
                "publication": {"phase": "publication_in_progress"},
            },
            1005.0,
            30.0,
            "busy",
        ),
        (
            {
                "status": "RUNNING",
                "phase": "tick_end",
                "timestamp_s": 10.0,
                "tick_started_at_s": 5.0,
                "last_progress_at_s": 10.0,
                "last_progress_kind": "result_ingested",
                "last_validation_error": None,
                "publication": {"phase": "idle"},
            },
            100.0,
            30.0,
            "stale",
        ),
        (
            {
                "status": "COMPLETE",
                "phase": "terminal",
                "timestamp_s": 10.0,
                "tick_started_at_s": 5.0,
                "last_progress_at_s": 10.0,
                "last_progress_kind": "publication_complete",
                "last_validation_error": None,
                "publication": {"phase": "complete"},
            },
            100.0,
            30.0,
            "terminal",
        ),
    ],
)
def test_heartbeat_health_classifies_progress_and_publication(
    payload, now_s, stale_after_s, expected
):
    assert heartbeat_health(payload, now_s=now_s, stale_after_s=stale_after_s) == expected


def _write_watchdog_runtime_files(tmp_path: Path, *, heartbeat: dict, coordinator_pid: int = 900):
    config = _runtime_config(tmp_path)
    root = Path(config.run.run_root)
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    runtime_path = control / "runtime_config.json"
    runtime_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    approved = tmp_path / "approved.json"
    approved.write_text("{}", encoding="utf-8")
    (control / "run.json").write_text(json.dumps({
        "run": {"worker_count": config.run.worker_count},
        "approved_manifest": str(approved),
        "runtime_config": config.as_dict(),
        "runtime_config_path": str(runtime_path),
    }), encoding="utf-8")
    (control / "coordinator-heartbeat.json").write_text(
        json.dumps(heartbeat), encoding="utf-8"
    )
    worker_pids = {f"{index:03d}": 1000 + index for index in range(config.run.worker_count)}
    launch = {
        "workers": config.run.worker_count,
        "worker_pids": worker_pids,
        "coordinator_pid": coordinator_pid,
        "watchdog_pid": os.getpid(),
        "restart_counts": {
            "coordinator": 0,
            "workers": {worker_id: 0 for worker_id in worker_pids},
        },
    }
    (control / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    return config, root, control, worker_pids


def test_watchdog_uses_runtime_config_and_restarts_stale_coordinator_with_reason(tmp_path: Path):
    heartbeat = {
        "status": "RUNNING",
        "phase": "tick_end",
        "timestamp_s": 10.0,
        "tick_started_at_s": 5.0,
        "last_progress_at_s": 10.0,
        "last_progress_kind": "result_ingested",
        "last_validation_error": None,
        "publication": {"phase": "idle"},
    }
    config, root, control, worker_pids = _write_watchdog_runtime_files(
        tmp_path, heartbeat=heartbeat
    )
    runtime_path = control / "runtime_config.json"
    coordinator_command = generation_watchdog._coordinator_command(
        config, runtime_path, control / "run.json"
    )
    approved_manifest = json.loads(
        (control / "run.json").read_text(encoding="utf-8")
    )["approved_manifest"]

    def reader(pid: int) -> bytes:
        if pid == 900:
            return b"\0".join(item.encode() for item in coordinator_command) + b"\0"
        worker_id = f"{pid - 1000:03d}"
        worker_command = generation_watchdog._worker_command(
            worker_id, config, runtime_path, approved_manifest
        )
        wrapped = ("/bin/sh", "/usr/bin/xvfb-run", *worker_command[1:])
        return b"\0".join(item.encode() for item in wrapped) + b"\0"

    class Process:
        pid = 7777

    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    updated = reconcile_processes(
        root,
        popen=popen,
        cmdline_reader=reader,
        now_s=100.0,
        stale_after_s=30.0,
    )

    assert len(calls) == 1
    command = calls[0][0]
    assert command[command.index("--runtime-config") + 1] == str(control / "runtime_config.json")
    assert all("/content" not in item for item in command)
    assert updated["coordinator_pid"] == 7777
    assert updated["restart_counts"]["coordinator"] == 1
    assert updated["restart_history"][-1]["reason"] == "heartbeat_stale"
    assert not list(control.glob("launch.json.partial-*"))
    assert config.machine.python_executable in command
    assert str(Path(config.machine.repo_root) / "scripts" / "generation_coordinator.py") in command


def test_watchdog_coordinator_restart_preserves_coordinator_only_token_path(tmp_path: Path):
    heartbeat = {
        "status": "RUNNING", "phase": "tick_end", "timestamp_s": 10.0,
        "last_progress_at_s": 10.0, "publication": {"phase": "idle"},
    }
    config, root, control, _ = _write_watchdog_runtime_files(tmp_path, heartbeat=heartbeat)
    token = tmp_path / "hf-token"
    token.write_text("secret", encoding="utf-8")
    launch_path = control / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch["hf_token_path"] = str(token)
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    runtime_path = control / "runtime_config.json"
    command = generation_watchdog._coordinator_command(
        config, runtime_path, control / "run.json", token,
    )

    calls = []
    class Process:
        pid = 7002
    updated = reconcile_processes(
        root,
        popen=lambda item, **kwargs: (calls.append(item) or Process()),
        cmdline_reader=lambda pid: b"",
        now_s=100.0,
        stale_after_s=30.0,
    )

    assert updated["coordinator_pid"] == 7002
    assert calls[0] == command


def test_watchdog_waits_for_initial_coordinator_heartbeat(tmp_path: Path):
    config, root, control, worker_pids = _write_watchdog_runtime_files(
        tmp_path, heartbeat={}
    )
    runtime_path = control / "runtime_config.json"
    coordinator_command = generation_watchdog._coordinator_command(
        config, runtime_path, control / "run.json"
    )
    approved_manifest = json.loads(
        (control / "run.json").read_text(encoding="utf-8")
    )["approved_manifest"]

    def reader(pid: int) -> bytes:
        if pid == 900:
            return b"\0".join(item.encode() for item in coordinator_command) + b"\0"
        worker_id = f"{pid - 1000:03d}"
        worker_command = generation_watchdog._worker_command(
            worker_id, config, runtime_path, approved_manifest
        )
        wrapped = ("/bin/sh", "/usr/bin/xvfb-run", *worker_command[1:])
        return b"\0".join(item.encode() for item in wrapped) + b"\0"

    calls = []
    updated = reconcile_processes(
        root,
        popen=lambda *args, **kwargs: calls.append((args, kwargs)),
        cmdline_reader=reader,
        now_s=100.0,
    )

    assert calls == []
    assert updated["coordinator_pid"] == 900
    assert updated["restart_counts"]["coordinator"] == 0


def test_worker_only_watchdog_replaces_only_its_host_worker(tmp_path: Path):
    heartbeat = {
        "status": "RUNNING",
        "phase": "tick_end",
        "timestamp_s": 10.0,
        "last_progress_at_s": 10.0,
        "publication": {"phase": "idle"},
    }
    config, root, control, _ = _write_watchdog_runtime_files(tmp_path, heartbeat=heartbeat)
    runtime_payload = json.loads((control / "runtime_config.json").read_text())
    runtime_payload["machine"].update({"host_id": "worker-a", "worker_ids": ["000"], "simulator_slots": 1})
    runtime_payload["run"].update({"distribution_mode": "shared_filesystem"})
    (control / "runtime_config.json").write_text(json.dumps(runtime_payload))
    worker_launch = {
        "host_id": "worker-a",
        "worker_pids": {"000": 1000},
        "restart_counts": {"workers": {"000": 0}},
        "restart_history": [],
    }
    (control / "worker-launch-worker-a.json").write_text(json.dumps(worker_launch))
    calls = []

    class Process:
        pid = 7000

    updated = reconcile_processes(
        root,
        workers_only=True,
        host_id="worker-a",
        popen=lambda command, **kwargs: (calls.append(command) or Process()),
        cmdline_reader=lambda pid: b"",
        now_s=100.0,
        stale_after_s=30.0,
    )

    assert len(calls) == 1
    assert "generation_coordinator.py" not in " ".join(calls[0])
    assert updated["worker_pids"]["000"] == 7000
    assert all(item["component"] == "worker:000" for item in updated["restart_history"])


def test_worker_only_watchdog_uses_host_runtime_snapshot_not_coordinator_config(tmp_path: Path):
    config = _runtime_config(tmp_path)
    root = Path(config.run.run_root)
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    coordinator_runtime = control / "runtime_config.json"
    coordinator_runtime.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    host_payload = config.as_dict()
    host_payload["machine"].update({"host_id": "worker-a", "worker_ids": ["001"], "simulator_slots": 1})
    host_payload["run"]["distribution_mode"] = "shared_filesystem"
    host_config = GenerationRuntimeConfig.from_dict(host_payload)
    host_runtime = control / "runtime_config-worker-a.json"
    host_runtime.write_text(json.dumps(host_config.as_dict()), encoding="utf-8")
    host_digest = hashlib.sha256(host_runtime.read_bytes()).hexdigest()
    approved = tmp_path / "approved.json"
    approved.write_text("{}", encoding="utf-8")
    (control / "run.json").write_text(json.dumps({
        "run": {"run_root": config.run.run_root},
        "approved_manifest": str(approved),
        "runtime_config_path": str(coordinator_runtime),
    }), encoding="utf-8")
    (control / "worker-launch-worker-a.json").write_text(json.dumps({
        "host_id": "worker-a",
        "runtime_config_path": str(host_runtime),
        "runtime_config_sha256": host_digest,
        "approved_manifest": str(approved),
        "worker_pids": {"001": 1000},
        "restart_counts": {"workers": {"001": 0}},
        "restart_history": [],
    }), encoding="utf-8")
    calls = []

    class Process:
        pid = 7001

    updated = reconcile_processes(
        root,
        workers_only=True,
        host_id="worker-a",
        popen=lambda command, **kwargs: (calls.append(command) or Process()),
        cmdline_reader=lambda pid: b"",
        now_s=100.0,
        stale_after_s=30.0,
    )

    assert len(calls) == 1
    assert calls[0][calls[0].index("--worker-id") + 1] == "001"
    assert calls[0][calls[0].index("--runtime-config") + 1] == str(host_runtime)
    assert updated["worker_pids"]["001"] == 7001


def test_watchdog_does_not_replace_stale_coordinator_while_lock_is_held(tmp_path: Path):
    heartbeat = {
        "status": "RUNNING",
        "phase": "tick_end",
        "timestamp_s": 10.0,
        "tick_started_at_s": 5.0,
        "last_progress_at_s": 10.0,
        "last_progress_kind": "result_ingested",
        "last_validation_error": None,
        "publication": {"phase": "idle"},
    }
    _, root, control, _ = _write_watchdog_runtime_files(tmp_path, heartbeat=heartbeat)
    config = GenerationRuntimeConfig.from_file(control / "runtime_config.json", check_paths=False)
    coordinator_command = generation_watchdog._coordinator_command(
        config, control / "runtime_config.json", control / "run.json"
    )
    reader = lambda pid: b"\0".join(item.encode() for item in coordinator_command) + b"\0"
    calls = []
    with CoordinatorLock(control / "coordinator.lock"):
        updated = reconcile_processes(
            root,
            popen=lambda *args, **kwargs: calls.append((args, kwargs)),
            cmdline_reader=reader,
            now_s=100.0,
            stale_after_s=30.0,
        )
    assert calls == []
    assert updated["coordinator_pid"] == 900
    assert updated["restart_counts"]["coordinator"] == 0


def test_watchdog_enforces_bounded_coordinator_restarts(tmp_path: Path):
    heartbeat = {
        "status": "RUNNING",
        "phase": "tick_end",
        "timestamp_s": 10.0,
        "tick_started_at_s": 5.0,
        "last_progress_at_s": 10.0,
        "last_progress_kind": "result_ingested",
        "last_validation_error": None,
        "publication": {"phase": "idle"},
    }
    _, root, control, _ = _write_watchdog_runtime_files(tmp_path, heartbeat=heartbeat)
    launch_path = control / "launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["restart_counts"]["coordinator"] = 1
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    config = GenerationRuntimeConfig.from_file(control / "runtime_config.json", check_paths=False)
    coordinator_command = generation_watchdog._coordinator_command(
        config, control / "runtime_config.json", control / "run.json"
    )
    reader = lambda pid: b"\0".join(item.encode() for item in coordinator_command) + b"\0"

    with pytest.raises(RuntimeError, match="coordinator restart limit exceeded"):
        reconcile_processes(
            root,
            popen=lambda *args, **kwargs: None,
            cmdline_reader=reader,
            now_s=100.0,
            stale_after_s=30.0,
            max_restarts=1,
        )


def test_watchdog_restarts_only_missing_slots_and_updates_receipt(tmp_path):
    config = _runtime_config(tmp_path)
    root = Path(config.run.run_root)
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    runtime_path = control / "runtime_config.json"
    runtime_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    approved = tmp_path / "approved.json"
    approved.write_text("{}", encoding="utf-8")
    (control / "run.json").write_text(json.dumps({
        "run": {"worker_count": config.run.worker_count},
        "approved_manifest": str(approved),
        "runtime_config": config.as_dict(),
        "runtime_config_path": str(runtime_path),
    }), encoding="utf-8")
    (control / "coordinator-heartbeat.json").write_text(json.dumps({
        "status": "RUNNING",
        "phase": "tick_end",
        "timestamp_s": 100.0,
        "tick_started_at_s": 95.0,
        "last_progress_at_s": 100.0,
        "last_progress_kind": "result_ingested",
        "last_validation_error": None,
        "publication": {"phase": "idle"},
    }), encoding="utf-8")
    worker_pids = {f"{index:03d}": 1000 + index for index in range(config.run.worker_count)}
    launch = {
        "workers": config.run.worker_count,
        "worker_pids": worker_pids,
        "coordinator_pid": 900,
        "watchdog_pid": os.getpid(),
        "restart_counts": {
            "coordinator": 0,
            "workers": {f"{index:03d}": 0 for index in range(config.run.worker_count)},
        },
    }
    (control / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    approved_manifest = str(approved)

    def reader(pid: int) -> bytes:
        if pid == 900:
            coordinator_command = generation_watchdog._coordinator_command(
                config, runtime_path, control / "run.json"
            )
            return b"\0".join(item.encode() for item in coordinator_command) + b"\0"
        if pid == worker_pids["001"]:
            return b""
        worker_id = f"{pid - 1000:03d}"
        worker_command = generation_watchdog._worker_command(
            worker_id, config, runtime_path, approved_manifest
        )
        wrapped = ("/bin/sh", "/usr/bin/xvfb-run", *worker_command[1:])
        return b"\0".join(item.encode() for item in wrapped) + b"\0"

    class Process:
        pid = 7777

    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    updated = reconcile_processes(root, popen=popen, cmdline_reader=reader, now_s=100.0)
    assert len(calls) == 1
    command = calls[0][0]
    assert command[command.index("--worker-id") + 1] == "001"
    assert command[command.index("--server-num") + 1] == "42"
    server_args = command[command.index("-s") + 1]
    assert "+extension GLX" in server_args
    assert "+render" in server_args
    assert command[command.index("--runtime-config") + 1] == str(runtime_path)
    assert updated["worker_pids"]["001"] == 7777
    assert updated["restart_counts"]["workers"]["001"] == 1
    assert updated["coordinator_pid"] == 900


def test_launch_smoke_accepts_retained_valid_failure(tmp_path):
    import json

    programs = [f"P{index:02d}" for index in range(36)]
    payload = {
        "summary": {"n": 36},
        "results": [
            {
                "program_id": program_id,
                "result_class": "valid_failure" if index == 0 else "success",
                "timeline_ok": True,
            }
            for index, program_id in enumerate(programs)
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    validate_smoke_receipt(path, expected_program_ids=programs)


def test_launch_smoke_rejects_crash_or_missing_timeline(tmp_path):
    import json
    import pytest

    programs = [f"P{index:02d}" for index in range(36)]
    payload = {
        "summary": {"n": 36},
        "results": [
            {
                "program_id": program_id,
                "result_class": "simulator_crash" if index == 0 else "success",
                "timeline_ok": index != 1,
            }
            for index, program_id in enumerate(programs)
        ],
    }
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="blocking smoke outcomes"):
        validate_smoke_receipt(path, expected_program_ids=programs)


def test_coordinator_resume_loads_pending_and_ready_jobs(tmp_path):
    import hashlib
    import json

    manifest_path = Path("artifacts/composition/approved_composition_manifest.json")
    rows = {row["program_id"]: row for row in json.loads(manifest_path.read_text())["catalog"]}
    run = RunConfig(
        run_id="resume-test", run_root=str(tmp_path), code_revision="a" * 40,
        approved_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    planner = DistributedPlanner.from_manifest(run, rows, {"episodes": [], "failure_attempts": []})
    queue = FilesystemJobQueue(tmp_path / "queue")
    first, second = planner.next_job(), planner.next_job()
    queue.enqueue(first)
    queue.enqueue(second)
    assert queue.claim("000") == first
    loaded = _inflight_jobs_from_queue(queue)
    assert {job.job_id for job in loaded} == {first.job_id, second.job_id}


def test_validation_mode_refill_never_enqueues_more_than_plan_max_jobs():
    from types import SimpleNamespace

    from icgs.data.collection.generation.distributed_contracts import QueueCounts
    from scripts.generation_coordinator import CoordinatorControlPlane

    class Queue:
        def __init__(self):
            self.jobs = []

        def counts(self):
            return QueueCounts(
                pending=len(self.jobs),
                claimed=0,
                ready=0,
                ingested=0,
                published=0,
            )

        def enqueue(self, job):
            self.jobs.append(job)

    class Planner:
        def __init__(self):
            self.index = 0

        def next_job(self):
            self.index += 1
            return f"job-{self.index}"

    queue = Queue()
    planner = Planner()
    run = SimpleNamespace(validation_mode=True, validation_max_jobs=7)
    coordinator = CoordinatorControlPlane(run, queue, planner, publisher=None, manifest={})

    coordinator._refill()

    assert len(queue.jobs) == 7


def test_validation_max_jobs_persists_across_coordinator_restart_and_open(tmp_path):
    manifest_path = Path("artifacts/composition/approved_composition_manifest.json")
    config = _runtime_config(tmp_path)
    run_path = generation_launch.persist_run_config(
        config,
        manifest_path,
        code_revision="a" * 40,
        validation_max_jobs=7,
    )
    token_path = Path(config.run.run_root) / "hf-token"
    token_path.write_text("token\n", encoding="utf-8")

    first = CoordinatorControlPlane.open(
        run_path,
        api_factory=lambda: object(),
        token_path=token_path,
    )
    assert json.loads(run_path.read_text(encoding="utf-8"))["run"]["validation_max_jobs"] == 7
    assert first._refill() == 7
    assert first.queue.counts().pending == 7
    assert first.queue.claim("000") is not None

    restarted = CoordinatorControlPlane.open(
        run_path,
        api_factory=lambda: object(),
        token_path=token_path,
    )
    assert restarted._refill() == 0
    counts = restarted.queue.counts()
    assert sum((counts.pending, counts.claimed, counts.ready, counts.ingested, counts.published)) == 7


def test_restarted_coordinator_refuses_to_publish_ingested_rows_after_safety_stop(tmp_path: Path):
    run_root = tmp_path / "stopped-run"
    queue = FilesystemJobQueue(run_root / "queue")
    plan = AttemptPlan(
        program_id="T01", split="train", episode_index=1,
        episode_id="episode-t01-stopped", episode_kind="nominal", scene_seed=7,
        collection_seed=20260920,
        randomization={"scene_signature": "stopped-signature", "asset_instance_id": "stopped-asset"},
        intervention=None,
    )
    job = GenerationJob.create(
        job_id="job-stopped-ingested", run_id="stopped-run",
        attempt_id="att-episode-t01-stopped", episode_id=plan.episode_id,
        program_id="T01", plan=plan, code_revision="a" * 40,
        manifest_sha256="b" * 64, output_root=str(run_root),
    )
    queue.enqueue(job)
    assert queue.claim("000") == job
    result_dir = run_root / "result"
    result_dir.mkdir(parents=True)
    (result_dir / "episode.json").write_text("{}", encoding="utf-8")
    result = WorkerResult(
        job_id=job.job_id, attempt_id=job.attempt_id, episode_id=job.episode_id,
        program_id=job.program_id, outcome="success", result_dir=str(result_dir),
        file_sha256={"episode.json": "c" * 64},
        timeline={"actions": 0, "observations": 1, "durations": 0},
    )
    queue.publish_ready("000", result)
    queue.mark_ingested(result)
    queue.trip_safety_stop({
        "reason": "orphan_archive_scratch_after_timeout",
        "recovery_action": "Preserve scratch files until HF fate is verified.",
    })

    class Planner:
        def quota_complete(self):
            return True

        def snapshot(self):
            return SimpleNamespace(as_dict=lambda: {})

    class Publisher:
        def __init__(self):
            self.calls = []

        def publish_due(self, **kwargs):
            self.calls.append(kwargs)

    run = RunConfig(
        run_id="stopped-run", run_root=str(run_root), code_revision="a" * 40,
        approved_manifest_sha256="b" * 64, worker_count=1,
        hf_subfolder="validation/stopped-run",
        validation_mode=True, validation_max_jobs=1,
    )
    publisher = Publisher()
    coordinator = CoordinatorControlPlane(
        run, queue, Planner(), publisher,
        {"manifest_version": 3, "episodes": [], "failure_attempts": []},
    )

    with pytest.raises(RuntimeError, match="generation safety stop"):
        coordinator.tick(now_s=10.0)

    assert publisher.calls == []
    assert coordinator.status == "FAILED"
    assert queue.counts().ingested == 1
    assert queue.counts().pending == 0
    heartbeat = json.loads((run_root / "control" / "coordinator-heartbeat.json").read_text())
    assert heartbeat["status"] == "FAILED"
    assert heartbeat["phase"] == "safety_stop"
