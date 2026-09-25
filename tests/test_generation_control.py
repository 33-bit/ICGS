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
    payload["run"]["resume_from_hf"] = True
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


def _complete_archive_remote(tmp_path: Path, monkeypatch):
    from test_generation_publication import _archive_queue, _job

    fixture_job = _job()
    fixture_job = replace(fixture_job, plan=replace(
        fixture_job.plan,
        randomization={**fixture_job.plan.randomization,
                       "scene_seed": fixture_job.plan.scene_seed,
                       "asset_instance_id": "T01-asset-1"},
    ))
    _source_queue, job, result, profile = _archive_queue(
        tmp_path / "source", retention="receipt_only", job=fixture_job,
    )
    row = validate_closed_result(job, result, archive_profile=profile).episode_entry
    manifest = {
        "manifest_version": 3, "source_run_ids": [job.run_id],
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(), "view_status": "PROVISIONAL",
        "episodes": [row], "failure_attempts": [],
    }
    config_payload = _runtime_config(tmp_path, publication_enabled=True).as_dict()
    config_payload["run"].update({
        "run_id": "new-archive-run",
        "validation_mode": False,
        "resume_from_hf": True,
    })
    config_payload["archive_profile"] = profile.as_dict()
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
    revision = "c" * 40
    remote_root = tmp_path / "snapshots" / revision
    prefix_root = remote_root / config.run.hf_subfolder
    prefix_root.mkdir(parents=True)
    dataset_path = prefix_root / "dataset_manifest.json"
    dataset_path.write_text(json.dumps(manifest), encoding="utf-8")
    archive_root = prefix_root / row["archive_ref"]
    shutil.copytree(result.result_dir, archive_root)
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    (prefix_root / "resume_receipt.json").write_text(json.dumps({
        "run_id": job.run_id, "source_run_id": job.run_id,
        "source_run_ids": [job.run_id], "episodes": 1, "failure_attempts": 0,
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "dataset_manifest_sha256": dataset_sha,
        "dataset_manifest_revision": "d" * 40,
    }), encoding="utf-8")
    (prefix_root / "publication_receipt.json").write_text(json.dumps({
        "receipt_version": 2, "run_id": job.run_id, "source_run_id": job.run_id,
        "job_ids": [job.job_id], "status": "VERIFIED",
        "data_commit_oid": "d" * 40, "commit_oid": "e" * 40,
        "artifact_hashes": {f"{job.job_id}/{relative}": digest for relative, digest in row["file_sha256"].items()},
        "dataset_identity": profile.dataset_identity,
        "archive_format_id": profile.archive_format_id,
        "episode_schema_version": profile.episode_schema_version,
        "archive_profile": profile.as_dict(),
        "dataset_manifest_sha256": dataset_sha,
        "prefix": config.run.hf_subfolder,
    }), encoding="utf-8")
    bootstrap = Path(config.run.run_root) / "control" / "resume_bootstrap.json"
    bootstrap.write_text(json.dumps({
        "run_id": config.run.run_id, "remote_revision": revision,
        "remote_manifest_sha256": dataset_sha,
    }), encoding="utf-8")
    downloads = []
    def download(**kwargs):
        downloads.append(kwargs)
        pinned = kwargs.get("revision")
        if "revision" in kwargs:
            assert pinned == revision
        path = remote_root / kwargs["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if pinned is None:
            return str(path)
        assert kwargs.get("cache_dir")
        cached = Path(kwargs["cache_dir"]) / "snapshots" / revision / kwargs["filename"]
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, cached)
        return str(cached)
    monkeypatch.setattr(generation_coordinator_module, "hf_hub_download", download)
    return run_json, token, manifest, prefix_root, downloads


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
    assert {item["filename"] for item in downloads} == {
        f"{prefix_root.relative_to(prefix_root.parents[1])}/{relative}"
        for relative in ("dataset_manifest.json", "resume_receipt.json", "publication_receipt.json")
    } | {
        f"{prefix_root.relative_to(prefix_root.parents[1])}/{manifest['episodes'][0]['archive_ref']}/{relative}"
        for relative in manifest["episodes"][0]["file_sha256"]
    }


def test_archive_preflight_pins_local_fake_manifest(tmp_path: Path, monkeypatch):
    run_json, token, _, _, downloads = _complete_archive_remote(tmp_path, monkeypatch)
    run_payload = json.loads(run_json.read_text())
    config = GenerationRuntimeConfig.from_file(run_payload["runtime_config_path"], check_paths=False)
    bootstrap = generation_launch.preflight_resume_manifest(
        config, token_path=token, run_root=config.run.run_root,
        approved_manifest=run_payload["approved_manifest"],
        downloader=generation_coordinator_module.hf_hub_download,
    )
    assert bootstrap["remote_revision"] == "c" * 40
    assert downloads[0].get("revision") is None
    assert all(call.get("revision") == "c" * 40 for call in downloads[1:])
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
