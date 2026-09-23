from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _setup_module():
    return importlib.import_module("scripts.setup_environment")


def _verify_module():
    return importlib.import_module("scripts.verify_environment")


def test_project_declares_portable_profiles_and_python_floor():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["requires-python"] == ">=3.10,<3.13"
    extras = payload["project"]["optional-dependencies"]
    assert {"cpu", "cuda118", "generation"}.issubset(extras)
    assert any("torch" in requirement for requirement in extras["cpu"])
    assert any("torch" in requirement for requirement in extras["cuda118"])
    assert {
        "pyg-lib==0.4.0+pt22cu118",
        "torch-cluster==1.6.3+pt22cu118",
        "torch-scatter==2.1.2+pt22cu118",
    }.issubset(extras["cuda118"])
    assert any("gymnasium" in requirement for requirement in extras["generation"])
    assert "torch" not in payload["project"]["dependencies"]
    assert payload["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu"},
        {"index": "pytorch-cu118", "extra": "cuda118"},
    ]
    assert payload["tool"]["uv"]["sources"]["torch-scatter"] == {
        "index": "pyg-cu118",
        "extra": "cuda118",
    }


def test_stale_conda_export_is_not_a_canonical_environment():
    assert not (ROOT / "environment.yml").exists()
    canonical = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "README.md", ROOT / "docs" / "README.md")
    )
    assert "environment.yml" not in canonical
    assert "ip_env" not in canonical


def test_clean_clone_instructions_have_no_fixed_content_or_conda_commands():
    documents = [
        ROOT / "README.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "components" / "environment.md",
        ROOT / "docs" / "components" / "generation.md",
        ROOT / "tests" / "README.md",
    ]
    for path in documents:
        in_fence = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                command = line.lower()
                assert "conda" not in command
                assert "environment.yml" not in command
                assert "ip_env" not in command
                assert "/content" not in command


def test_secret_values_and_credential_paths_are_redacted_from_commands():
    setup = _setup_module()
    command = (
        "uv",
        "pip",
        "install",
        "--token",
        "hf_very_secret_value",
        "--env",
        "HF_TOKEN=hf_another_secret",
        "--token-path",
        "/home/user/.icgs_hf_token",
    )
    redacted = setup.redact_command(command)
    rendered = " ".join(redacted)
    assert "very_secret" not in rendered
    assert "another_secret" not in rendered
    assert "/home/user/.icgs_hf_token" not in rendered
    assert "<redacted>" in rendered


def test_profile_command_plan_is_dry_run_safe(tmp_path: Path):
    setup = _setup_module()
    commands = setup.plan_setup("cpu", repo_root=tmp_path, venv_root=tmp_path / ".venv")
    assert commands
    assert commands[0][:3] == ("uv", "venv", "--python")
    assert any(command[:3] == ("uv", "sync", "--locked") for command in commands)
    assert all("environment.yml" not in " ".join(command) for command in commands)
    assert all("conda" not in " ".join(command).lower() for command in commands)
    generation = setup.plan_setup("generation", repo_root=tmp_path, venv_root=tmp_path / ".venv")
    sync = next(command for command in generation if command[:3] == ("uv", "sync", "--locked"))
    assert sync.count("--extra") == 3
    assert {sync[index + 1] for index, part in enumerate(sync) if part == "--extra"} == {"cpu", "generation", "test"}
    with pytest.raises(ValueError, match="Python version"):
        setup.plan_setup("cpu", repo_root=tmp_path, venv_root=tmp_path / ".venv", python_version="3.9")


def test_setup_receipt_write_is_idempotent_and_redacted(tmp_path: Path):
    setup = _setup_module()
    receipt = {
        "schema_version": 1,
        "profile": "cpu",
        "commands": [
            ["uv", "sync", "HF_TOKEN=hf_secret"],
            ["cat", "/home/user/.icgs_hf_token"],
        ],
        "credential_path": "/run/secrets/hf",
    }
    target = tmp_path / "setup_receipt.json"
    setup.write_receipt(target, receipt)
    first = target.read_bytes()
    setup.write_receipt(target, receipt)
    second = target.read_bytes()
    assert first == second
    encoded = target.read_text(encoding="utf-8")
    assert "hf_secret" not in encoded
    assert "/run/secrets/hf" not in encoded


def test_verifier_reports_missing_dependency_without_importing_secret_values(tmp_path: Path):
    verify = _verify_module()
    calls: list[tuple[str, ...]] = []

    def runner(command, **kwargs):
        normalized = tuple(str(part) for part in command)
        calls.append(normalized)
        if any("import" in part for part in normalized) and "torch" in " ".join(normalized):
            return subprocess.CompletedProcess(normalized, 1, stdout="", stderr="No module named torch")
        return subprocess.CompletedProcess(normalized, 0, stdout="Python 3.11.0\n", stderr="")

    result = verify.verify_environment(
        "cpu",
        repo_root=tmp_path,
        python_executable=sys.executable,
        runner=runner,
        credential_paths=(tmp_path / "hf-token",),
    )
    assert result["profile"] == "cpu"
    assert result["checks"]["core_dependencies"]["status"] == "FAIL"
    assert result["checks"]["credentials"]["status"] == "FAIL"
    rendered = json.dumps(result, sort_keys=True)
    assert "secret" not in rendered.lower()
    assert all("hf-token" not in " ".join(command) for command in calls)


def test_verifier_emits_machine_readable_statuses():
    verify = _verify_module()
    result = verify.verify_environment("cpu", python_executable=sys.executable)
    assert set(result["checks"]) >= {
        "python",
        "icgs_imports",
        "core_dependencies",
        "pyg_abi",
        "renderer",
        "simulator",
        "rlbench_pyrep",
        "credentials",
    }
    assert all(value["status"] in {"PASS", "FAIL", "SKIPPED", "NOT_RUN"} for value in result["checks"].values())


def test_verifier_accepts_matching_redacted_receipt(tmp_path: Path):
    verify = _verify_module()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"schema_version": 1, "profile": "cpu"}),
        encoding="utf-8",
    )
    result = verify.verify_environment(
        "cpu",
        repo_root=tmp_path,
        python_executable=sys.executable,
        receipt_path=receipt,
    )
    assert result["checks"]["setup_receipt"]["status"] == "PASS"
