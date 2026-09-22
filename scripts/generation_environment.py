"""Provision the pinned generation environment from a runtime configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from icgs.data.collection.generation.distributed_contracts import GenerationRuntimeConfig


COPPELIASIM_SHA256 = "512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8"
COPPELIASIM_URL = (
    "https://downloads.coppeliarobotics.com/V4_1_0/"
    "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
)
UPSTREAM = {
    "PyRep": "8f420be8064b1970aae18a9cfbc978dfb15747ef",
    "RLBench": "02720bba4c73fe02eb75df946b8791b806028a9d",
}


@dataclass(frozen=True)
class ProvisionReceipt:
    repo_root: str
    python_executable: str
    simulator_root: str
    rlbench_root: str
    simulator_sha256: str
    upstream_revisions: dict[str, str]
    commands: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "python_executable": self.python_executable,
            "simulator_root": self.simulator_root,
            "rlbench_root": self.rlbench_root,
            "simulator_sha256": self.simulator_sha256,
            "upstream_revisions": dict(self.upstream_revisions),
            "commands": [list(command) for command in self.commands],
        }


def _run(
    command: list[str],
    *,
    timeout: int,
    env: dict[str, str],
    runner,
    commands: list[tuple[str, ...]],
    capture_output: bool = False,
    text: bool = False,
):
    normalized = tuple(str(part) for part in command)
    commands.append(normalized)
    options: dict[str, Any] = {"check": True, "timeout": timeout, "env": env}
    if capture_output:
        options["capture_output"] = True
    if text:
        options["text"] = True
    return runner(list(normalized), **options)


def provision_environment(
    config: GenerationRuntimeConfig,
    *,
    runner=subprocess.run,
) -> ProvisionReceipt:
    """Install pinned simulator dependencies, recording each external command."""
    if not isinstance(config, GenerationRuntimeConfig):
        raise TypeError("config must be a GenerationRuntimeConfig")

    machine = config.machine
    repo_root = Path(machine.repo_root)
    python_executable = Path(machine.python_executable)
    simulator_root = Path(machine.simulator_root)
    rlbench_root = Path(machine.rlbench_root)
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {repo_root}")
    if python_executable.exists() and (
        not python_executable.is_file() or not os.access(python_executable, os.X_OK)
    ):
        raise ValueError(f"python_executable must be executable: {python_executable}")

    commands: list[tuple[str, ...]] = []
    environment = config.resolved_environment(base=os.environ)
    environment.update({
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_TERMINAL_PROMPT": "0",
    })

    if not python_executable.is_file():
        venv_root = python_executable.parent.parent
        venv_root.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["uv", "venv", "--python", sys.executable, str(venv_root)],
            timeout=180,
            env=environment,
            runner=runner,
            commands=commands,
        )
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise ValueError(f"python_executable must be executable: {python_executable}")

    _run(
        ["apt-get", "update", "-qq"],
        timeout=300,
        env=environment,
        runner=runner,
        commands=commands,
    )
    _run(
        [
            "apt-get", "install", "-y", "-qq", "libgl1", "libglu1-mesa",
            "libxcb-xinerama0", "libxkbcommon-x11-0", "libxcb-cursor0",
            "libegl1", "xauth", "xvfb", "libxrender1", "libxi6", "libxrandr2",
        ],
        timeout=600,
        env=environment,
        runner=runner,
        commands=commands,
    )
    _run(
        [
            "uv", "pip", "install", "--python", str(python_executable),
            "numpy==1.26.4", "scipy==1.14.1", "cffi", "setuptools", "wheel",
            "pillow", "pyquaternion", "natsort", "transforms3d",
            "gymnasium==1.2.3", "huggingface-hub==0.26.2",
        ],
        timeout=900,
        env=environment,
        runner=runner,
        commands=commands,
    )

    archive = simulator_root.parent / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "curl", "--fail", "--location", "--retry", "2", "--max-time", "900",
            "--output", str(archive), COPPELIASIM_URL,
        ],
        timeout=1000,
        env=environment,
        runner=runner,
        commands=commands,
    )
    simulator_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if simulator_digest != COPPELIASIM_SHA256:
        raise RuntimeError(f"CoppeliaSim checksum mismatch: {simulator_digest}")

    simulator_root.mkdir(parents=True, exist_ok=True)
    _run(
        ["tar", "-xJf", str(archive), "-C", str(simulator_root), "--strip-components=1"],
        timeout=300,
        env=environment,
        runner=runner,
        commands=commands,
    )

    source_root = rlbench_root.parent
    source_root.mkdir(parents=True, exist_ok=True)
    for name, revision in UPSTREAM.items():
        source = source_root / name if name == "PyRep" else rlbench_root
        _run(
            ["git", "clone", "--no-checkout", f"https://github.com/stepjam/{name}.git", str(source)],
            timeout=600,
            env=environment,
            runner=runner,
            commands=commands,
        )
        _run(
            ["git", "-C", str(source), "fetch", "--depth", "1", "origin", revision],
            timeout=600,
            env=environment,
            runner=runner,
            commands=commands,
        )
        _run(
            ["git", "-C", str(source), "checkout", "--detach", revision],
            timeout=120,
            env=environment,
            runner=runner,
            commands=commands,
        )
        result = _run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            timeout=120,
            env=environment,
            runner=runner,
            commands=commands,
            capture_output=True,
            text=True,
        )
        actual_revision = (getattr(result, "stdout", "") or "").strip()
        if actual_revision != revision:
            raise RuntimeError(f"{name} revision mismatch: {actual_revision}")
        _run(
            ["uv", "pip", "install", "--python", str(python_executable), "--editable", str(source)],
            timeout=600,
            env=environment,
            runner=runner,
            commands=commands,
        )

    _run(
        [
            "uv", "pip", "install", "--python", str(python_executable),
            "--no-deps", "--editable", str(repo_root),
        ],
        timeout=600,
        env=environment,
        runner=runner,
        commands=commands,
    )
    machine.validate_paths()
    return ProvisionReceipt(
        repo_root=str(repo_root),
        python_executable=str(python_executable),
        simulator_root=str(simulator_root),
        rlbench_root=str(rlbench_root),
        simulator_sha256=simulator_digest,
        upstream_revisions=dict(UPSTREAM),
        commands=tuple(commands),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    args = parser.parse_args(argv)
    config = GenerationRuntimeConfig.from_file(args.runtime_config, check_paths=False)
    receipt = provision_environment(config)
    print(json.dumps(receipt.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
