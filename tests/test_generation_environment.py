from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

from icgs.data.collection.generation.distributed_contracts import (
    GenerationRuntimeConfig,
)
from scripts import generation_launch


def _required_api(module, name: str):
    assert hasattr(module, name), f"{module.__name__}.{name} is required"
    return getattr(module, name)


def _environment_module():
    assert importlib.util.find_spec("scripts.generation_environment") is not None, (
        "scripts.generation_environment is required"
    )
    return importlib.import_module("scripts.generation_environment")


def _runtime_config(tmp_path: Path, *, create_python: bool = True):
    repo_root = tmp_path / "repo"
    simulator_root = tmp_path / "CoppeliaSim"
    rlbench_root = tmp_path / "RLBench"
    run_root = tmp_path / "runs" / "portable"
    for directory in (repo_root, simulator_root, rlbench_root, run_root):
        directory.mkdir(parents=True, exist_ok=True)
    python_executable = tmp_path / "venv" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    if create_python:
        python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python_executable.chmod(0o755)
    return GenerationRuntimeConfig.from_dict({
        "machine": {
            "repo_root": str(repo_root),
            "python_executable": str(python_executable),
            "simulator_root": str(simulator_root),
            "rlbench_root": str(rlbench_root),
            "display_base": 73,
            "display_width": 1440,
            "display_height": 900,
            "simulator_slots": 2,
            "worker_timeout_s": 47,
        },
        "run": {
            "run_id": "portable",
            "run_root": str(run_root),
            "worker_count": 2,
            "publish_interval_s": 91,
            "hf_repo": "33bit/icgs",
            "hf_subfolder": "validation/portable",
            "publication_enabled": False,
            "validation_mode": True,
        },
    })


def test_process_environment_uses_config_and_preserves_base(tmp_path: Path):
    config = _runtime_config(tmp_path)
    base = {
        "LD_LIBRARY_PATH": "/opt/base-libraries",
        "PYTHONPATH": "/opt/base-python",
        "UNCHANGED": "present",
    }

    environment = _required_api(generation_launch, "build_process_environment")(
        config, base=base
    )

    assert environment["COPPELIASIM_ROOT"] == config.machine.simulator_root
    assert environment["LD_LIBRARY_PATH"] == os.pathsep.join(
        [config.machine.simulator_root, "/opt/base-libraries"]
    )
    assert environment["QT_QPA_PLATFORM_PLUGIN_PATH"] == config.machine.simulator_root
    assert environment["QT_QPA_PLATFORM"] == "xcb"
    assert environment["QT_LOGGING_RULES"] == "*.debug=false"
    assert environment["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert environment["PYTHONPATH"] == os.pathsep.join([
        str(Path(config.machine.repo_root) / "src"),
        config.machine.rlbench_root,
        "/opt/base-python",
    ])
    assert environment["ICGS_SIMULATOR_SLOTS"] == "2"
    assert environment["ICGS_HF_SUBFOLDER"] == "validation/portable"
    assert environment["UNCHANGED"] == "present"
    assert base["LD_LIBRARY_PATH"] == "/opt/base-libraries"


def test_environment_keeps_pinned_sources_and_simulator_checksum():
    environment = _environment_module()

    assert environment.COPPELIASIM_SHA256 == (
        "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8"
    )
    assert environment.UPSTREAM == {
        "PyRep": "8f420be8064b1970aae18a9cfbc978dfb15747ef",
        "RLBench": "02720bba4c73fe02eb75df946b8791b806028a9d",
    }


def test_provision_environment_records_configured_commands_without_external_execution(
    tmp_path: Path, monkeypatch
):
    environment = _environment_module()
    config = _runtime_config(tmp_path, create_python=False)
    Path(config.machine.rlbench_root).rmdir()
    archive_bytes = b"test simulator archive"
    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
    monkeypatch.setattr(environment, "COPPELIASIM_SHA256", archive_digest)
    calls: list[tuple[str, ...]] = []

    def runner(command, **kwargs):
        normalized = tuple(str(part) for part in command)
        calls.append(normalized)
        if normalized[:2] == ("uv", "venv"):
            python = Path(config.machine.python_executable)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
        if normalized[0] == "curl":
            archive = Path(normalized[normalized.index("--output") + 1])
            archive.write_bytes(archive_bytes)
        if normalized[:3] == ("git", "clone", "--no-checkout"):
            Path(normalized[-1]).mkdir(parents=True)
        if normalized[:2] == ("git", "-C") and normalized[-2:] == ("rev-parse", "HEAD"):
            source_name = Path(normalized[2]).name
            return SimpleNamespace(stdout=environment.UPSTREAM[source_name] + "\n")
        return SimpleNamespace(stdout="")

    receipt = _required_api(environment, "provision_environment")(
        config,
        runner=runner,
    )

    assert receipt.commands == tuple(calls)
    assert receipt.simulator_sha256 == archive_digest
    assert receipt.upstream_revisions == environment.UPSTREAM
    assert receipt.python_executable == config.machine.python_executable
    assert receipt.simulator_root == config.machine.simulator_root
    assert any(command[:2] == ("apt-get", "update") for command in receipt.commands)
    apt_install = next(
        command
        for command in receipt.commands
        if command[:3] == ("apt-get", "install", "-y")
    )
    assert {
        "libfontconfig1",
        "libxcb-icccm4",
        "libxcb-image0",
        "libxcb-keysyms1",
        "libxcb-render-util0",
    }.issubset(apt_install)
    assert "libgl1-mesa-dri" in apt_install
    assert any(command[0] == "curl" for command in receipt.commands)
    assert any(command[:3] == ("uv", "pip", "install") for command in receipt.commands)
    assert any(config.machine.python_executable in command for command in receipt.commands)
    assert all("/content" not in " ".join(command) for command in receipt.commands)


def test_provision_commands_do_not_receive_credential_environment(tmp_path: Path, monkeypatch):
    environment = _environment_module()
    config = _runtime_config(tmp_path, create_python=False)
    Path(config.machine.rlbench_root).rmdir()
    archive_bytes = b"test simulator archive"
    monkeypatch.setattr(
        environment,
        "COPPELIASIM_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )

    def runner(command, **kwargs):
        assert "HF_TOKEN" not in kwargs["env"]
        assert "WANDB_API_KEY" not in kwargs["env"]
        normalized = tuple(str(part) for part in command)
        if normalized[:2] == ("uv", "venv"):
            python = Path(config.machine.python_executable)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
        if normalized[0] == "curl":
            archive = Path(normalized[normalized.index("--output") + 1])
            archive.write_bytes(archive_bytes)
        if normalized[:3] == ("git", "clone", "--no-checkout"):
            Path(normalized[-1]).mkdir(parents=True)
        if normalized[:2] == ("git", "-C") and normalized[-2:] == ("rev-parse", "HEAD"):
            source_name = Path(normalized[2]).name
            return SimpleNamespace(stdout=environment.UPSTREAM[source_name] + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setenv("HF_TOKEN", "hf_do_not_leak")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_do_not_leak")
    _required_api(environment, "provision_environment")(config, runner=runner)
