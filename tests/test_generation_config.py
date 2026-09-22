from __future__ import annotations

import json
from pathlib import Path

import pytest

from icgs.data.collection.generation import distributed_contracts as contracts


def _config_api(name: str):
    assert hasattr(contracts, name), f"distributed_contracts.{name} is required"
    return getattr(contracts, name)


def _portable_payload(tmp_path: Path) -> dict:
    machine_root = tmp_path / "machine"
    repo_root = machine_root / "repo"
    simulator_root = machine_root / "simulator"
    rlbench_root = machine_root / "rlbench"
    for directory in (repo_root, simulator_root, rlbench_root):
        directory.mkdir(parents=True)
    python_executable = machine_root / "venv" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    run_root = tmp_path / "runs" / "validation-test"
    run_root.mkdir(parents=True)
    return {
        "machine": {
            "repo_root": str(repo_root),
            "python_executable": str(python_executable),
            "simulator_root": str(simulator_root),
            "rlbench_root": str(rlbench_root),
            "display_base": 17,
            "display_width": 1280,
            "display_height": 1024,
            "simulator_slots": 2,
            "worker_timeout_s": 1200,
        },
        "run": {
            "run_id": "validation-test",
            "run_root": str(run_root),
            "worker_count": 2,
            "publish_interval_s": 300,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "validation/validation-test",
            "publication_enabled": False,
            "validation_mode": True,
        },
    }


def _load_payload(tmp_path: Path, payload: dict, *, check_paths: bool = True):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return _config_api("GenerationRuntimeConfig").from_file(path, check_paths=check_paths)


def test_runtime_config_types_are_available():
    assert _config_api("MachineConfig")
    assert _config_api("GenerationRuntimeConfig")


def test_portable_config_accepts_two_workers_and_two_simulator_slots(tmp_path: Path):
    runtime = _load_payload(tmp_path, _portable_payload(tmp_path))

    assert runtime.run.worker_count == 2
    assert runtime.machine.simulator_slots == 2
    assert runtime.machine.python_executable.endswith("/venv/bin/python")


def test_machine_config_rejects_relative_paths(tmp_path: Path):
    payload = _portable_payload(tmp_path)["machine"]
    payload["repo_root"] = "relative/repo"

    with pytest.raises(ValueError, match="absolute"):
        _config_api("MachineConfig").from_dict(payload)


@pytest.mark.parametrize("field", ["repo_root", "simulator_root", "rlbench_root"])
def test_runtime_config_rejects_nonexistent_required_directories(tmp_path: Path, field: str):
    payload = _portable_payload(tmp_path)
    payload["machine"][field] = str(tmp_path / "missing" / field)

    with pytest.raises(ValueError, match="directory"):
        _load_payload(tmp_path, payload)


def test_runtime_config_rejects_nonexistent_python_executable(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    payload["machine"]["python_executable"] = str(tmp_path / "missing-python")

    with pytest.raises(ValueError, match="executable"):
        _load_payload(tmp_path, payload)


def test_runtime_config_rejects_nonexistent_run_root_when_checking_paths(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    payload["run"]["run_root"] = str(tmp_path / "runs" / "not-created")

    with pytest.raises(ValueError, match="run_root.*directory"):
        _load_payload(tmp_path, payload)


def test_nonexistent_run_root_is_allowed_without_path_checks(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    payload["run"]["run_root"] = str(tmp_path / "runs" / "not-created")

    config_type = _config_api("GenerationRuntimeConfig")
    from_dict = config_type.from_dict(payload)
    from_file = _load_payload(tmp_path, payload, check_paths=False)

    assert from_dict.run.run_root == payload["run"]["run_root"]
    assert from_file.run.run_root == payload["run"]["run_root"]


def test_runtime_config_rejects_non_executable_python_file(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    executable = Path(payload["machine"]["python_executable"])
    executable.chmod(0o644)

    with pytest.raises(ValueError, match="executable"):
        _load_payload(tmp_path, payload)


def test_runtime_config_rejects_worker_count_below_one(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    payload["run"]["worker_count"] = 0

    with pytest.raises(ValueError, match="worker_count"):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


@pytest.mark.parametrize("slots", [0, 3])
def test_runtime_config_rejects_slots_outside_worker_count(tmp_path: Path, slots: int):
    payload = _portable_payload(tmp_path)
    payload["machine"]["simulator_slots"] = slots

    with pytest.raises(ValueError, match="simulator_slots"):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("machine", "worker_timeout_s", 0),
        ("machine", "display_width", 0),
        ("machine", "display_height", -1),
    ],
)
def test_runtime_config_rejects_nonpositive_machine_limits(
    tmp_path: Path, section: str, field: str, value: int
):
    payload = _portable_payload(tmp_path)
    payload[section][field] = value

    with pytest.raises(ValueError, match=field):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


def test_runtime_config_rejects_nonpositive_publication_interval(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    payload["run"]["publish_interval_s"] = 0

    with pytest.raises(ValueError, match="publish_interval_s"):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


def test_validation_mode_rejects_non_validation_hf_prefix(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    payload["run"]["hf_subfolder"] = "generation/full-run"

    with pytest.raises(ValueError, match="validation"):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


@pytest.mark.parametrize("missing", ["hf_repo", "hf_subfolder"])
def test_enabled_publication_requires_repository_and_subfolder(tmp_path: Path, missing: str):
    payload = _portable_payload(tmp_path)
    payload["run"]["publication_enabled"] = True
    payload["run"].pop(missing)

    with pytest.raises(ValueError, match="publication"):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [(None, "unknown"), ("machine", "unknown"), ("run", "unknown")],
)
def test_runtime_config_rejects_unknown_json_fields(
    tmp_path: Path, section: str | None, field: str
):
    payload = _portable_payload(tmp_path)
    target = payload if section is None else payload[section]
    target[field] = "unexpected"

    with pytest.raises(ValueError, match="unknown"):
        _config_api("GenerationRuntimeConfig").from_dict(payload)


def test_resolved_environment_merges_without_mutating_base(tmp_path: Path):
    payload = _portable_payload(tmp_path)
    runtime = _config_api("GenerationRuntimeConfig").from_dict(payload)
    base = {
        "LD_LIBRARY_PATH": "/opt/base-libraries",
        "PYTHONPATH": "/opt/base-python",
        "UNCHANGED": "present",
    }

    environment = runtime.resolved_environment(base=base)

    assert environment["COPPELIASIM_ROOT"] == payload["machine"]["simulator_root"]
    assert environment["LD_LIBRARY_PATH"] == (
        payload["machine"]["simulator_root"] + ":/opt/base-libraries"
    )
    assert environment["PYTHONPATH"] == ":".join(
        [payload["machine"]["repo_root"] + "/src", payload["machine"]["rlbench_root"], "/opt/base-python"]
    )
    assert environment["ICGS_SIMULATOR_SLOTS"] == "2"
    assert environment["ICGS_HF_REPO"] == "33bit/icgs"
    assert environment["ICGS_HF_SUBFOLDER"] == "validation/validation-test"
    assert environment["UNCHANGED"] == "present"
    assert base["LD_LIBRARY_PATH"] == "/opt/base-libraries"


def test_checked_in_generation_runtime_profile_parses_without_path_checks():
    profile = Path(__file__).resolve().parents[1] / "src/icgs/configuration/profiles/generation_runtime.json"

    runtime = _config_api("GenerationRuntimeConfig").from_file(profile, check_paths=False)

    assert runtime.run.worker_count == 2
    assert runtime.machine.simulator_slots == 2
    assert runtime.run.publication_enabled is False
    assert runtime.run.validation_mode is True


def test_run_config_accepts_configurable_publication_and_validation_fields():
    run = contracts.RunConfig(
        run_id="validation-run",
        run_root="/tmp/validation-run",
        code_revision="a" * 40,
        approved_manifest_sha256="b" * 64,
        worker_count=2,
        publish_interval_s=45,
        hf_repo="33bit/icgs",
        hf_subfolder="validation/validation-run",
        publication_enabled=True,
        validation_mode=True,
    )

    assert run.worker_count == 2
    assert run.publish_interval_s == 45
    assert run.publication_enabled is True
    assert run.validation_mode is True


def test_run_config_from_dict_preserves_default_worker_count_when_omitted():
    run = contracts.RunConfig.from_dict({
        "run_id": "legacy-run",
        "run_root": "/tmp/legacy-run",
        "code_revision": "a" * 40,
        "approved_manifest_sha256": "b" * 64,
    })

    assert run.worker_count == 200


def test_run_config_rejects_validation_subfolder_parent_traversal():
    with pytest.raises(ValueError, match="validation"):
        contracts.RunConfig(
            run_id="validation-run",
            run_root="/tmp/validation-run",
            code_revision="a" * 40,
            approved_manifest_sha256="b" * 64,
            hf_repo="33bit/icgs",
            hf_subfolder="validation/../generation",
            validation_mode=True,
        )
