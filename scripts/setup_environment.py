"""Install one of the supported ICGS dependency profiles.

The command planner is intentionally separate from execution.  Tests and
operators can inspect the exact ``uv``/simulator commands before running them,
and callers can inject a command runner without touching a real host.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_NAMES = ("cpu", "cuda118", "generation")
PROFILE_EXTRAS: dict[str, tuple[str, ...]] = {
    "cpu": ("cpu", "test"),
    "cuda118": ("cuda118", "test"),
    # The generation profile includes the CPU runtime because the pinned
    # simulator acceptance path is CPU rendered unless a host opts into CUDA.
    "generation": ("cpu", "generation", "test"),
}
_SECRET_NAME = re.compile(
    r"(?:token|secret|password|passwd|api[_-]?key|access[_-]?key)", re.IGNORECASE
)
_SECRET_VALUE = re.compile(
    r"(?:hf|gh[opurs]|sk)-[A-Za-z0-9_.-]{8,}|(?:AKIA|ASIA)[A-Z0-9]{12,}",
    re.IGNORECASE,
)
_CREDENTIAL_PATH = re.compile(
    r"(?:^|/)(?:\.env(?:\.[^/]*)?|[^/]*token[^/]*)$|/(?:secrets|credentials)/",
    re.IGNORECASE,
)


def _as_tuple(command: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(part) for part in command)


def _validate_profile(profile: str) -> str:
    if profile not in PROFILE_NAMES:
        choices = ", ".join(PROFILE_NAMES)
        raise ValueError(f"profile must be one of: {choices}")
    return profile


def _validate_python_version(value: str) -> str:
    match = re.fullmatch(r"3\.(10|11|12)(?:\.\d+)?", value.strip())
    if match is None:
        raise ValueError("Python version must be 3.10, 3.11 or 3.12")
    return value.strip()


def _redact_string(value: str) -> str:
    value = _SECRET_VALUE.sub("<redacted>", value)
    if "=" in value:
        name, remainder = value.split("=", 1)
        if _SECRET_NAME.search(name):
            return f"{name}=<redacted>"
        value = f"{name}={remainder}"
    if _CREDENTIAL_PATH.search(value):
        return "<redacted-credential-path>"
    return value


def redact_command(command: Sequence[object]) -> tuple[str, ...]:
    """Return a command with secret values and credential paths removed."""
    result: list[str] = []
    redact_next = False
    for raw in command:
        value = str(raw)
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        lowered = value.lower()
        if lowered in {"--token", "--token-path", "--password", "--secret", "--api-key"}:
            result.append(value)
            redact_next = True
            continue
        result.append(_redact_string(value))
    return tuple(result)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically write a deterministic, redacted setup receipt."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _redact_value(dict(receipt))
    encoded = json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
    if target.is_file() and target.read_text(encoding="utf-8") == encoded:
        return target
    temporary = target.with_name(f".{target.name}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target


def _python_executable(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def plan_setup(
    profile: str,
    *,
    repo_root: str | Path,
    venv_root: str | Path,
    python_version: str = "3.10",
    provision_simulator: bool = False,
    runtime_config: str | Path | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Build an idempotent command plan without executing anything."""
    profile = _validate_profile(profile)
    root = Path(repo_root).resolve()
    venv = Path(venv_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repo_root must be an existing directory: {root}")
    python_version = _validate_python_version(python_version)
    commands: list[tuple[str, ...]] = [
        ("uv", "venv", "--python", python_version, str(venv)),
    ]
    sync_command: list[str] = ["uv", "sync", "--locked"]
    for extra in PROFILE_EXTRAS[profile]:
        sync_command.extend(("--extra", extra))
    commands.append(tuple(sync_command))
    if profile == "generation" and provision_simulator:
        if runtime_config is None:
            raise ValueError("runtime_config is required with --provision-simulator")
        commands.append(
            (
                str(_python_executable(venv)),
                "-B",
                "scripts/generation_environment.py",
                "--runtime-config",
                str(Path(runtime_config).resolve()),
            )
        )
    return tuple(commands)


@dataclass(frozen=True)
class SetupReceipt:
    schema_version: int
    profile: str
    repo_root: str
    venv_root: str
    python_executable: str
    platform: str
    python_version: str
    status: str
    commands: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "repo_root": self.repo_root,
            "venv_root": self.venv_root,
            "python_executable": self.python_executable,
            "platform": self.platform,
            "python_version": self.python_version,
            "status": self.status,
            "commands": [list(redact_command(command)) for command in self.commands],
        }


def _safe_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if base is None else base)
    clean: dict[str, str] = {}
    for key, value in source.items():
        if _SECRET_NAME.search(key):
            continue
        clean[str(key)] = str(value)
    return clean


Runner = Callable[..., Any]


def setup_environment(
    profile: str,
    *,
    repo_root: str | Path = ".",
    venv_root: str | Path = ".venv",
    python_version: str = "3.10",
    receipt_path: str | Path | None = None,
    dry_run: bool = False,
    provision_simulator: bool = False,
    runtime_config: str | Path | None = None,
    runner: Runner = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> SetupReceipt:
    """Plan and, unless ``dry_run``, execute a selected environment profile."""
    profile = _validate_profile(profile)
    root = Path(repo_root).resolve()
    venv = Path(venv_root)
    if not venv.is_absolute():
        venv = (root / venv).resolve()
    commands = plan_setup(
        profile,
        repo_root=root,
        venv_root=venv,
        python_version=python_version,
        provision_simulator=provision_simulator,
        runtime_config=runtime_config,
    )
    status = "DRY_RUN" if dry_run else "PASS"
    clean_env = _safe_environment(environment)
    clean_env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    clean_env["GIT_TERMINAL_PROMPT"] = "0"
    if not dry_run:
        for command in commands:
            runner(
                list(command),
                check=True,
                cwd=str(root),
                env=clean_env,
            )
    receipt = SetupReceipt(
        schema_version=1,
        profile=profile,
        repo_root=str(root),
        venv_root=str(venv),
        python_executable=str(_python_executable(venv)),
        platform=platform.platform(),
        python_version=python_version,
        status=status,
        commands=commands,
    )
    target = Path(receipt_path) if receipt_path is not None else root / ".icgs" / "setup_receipt.json"
    write_receipt(target, receipt.as_dict())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--venv-root", default=".venv")
    parser.add_argument("--python-version", default="3.10")
    parser.add_argument("--receipt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provision-simulator", action="store_true")
    parser.add_argument("--runtime-config")
    args = parser.parse_args(argv)
    receipt = setup_environment(
        args.profile,
        repo_root=args.repo_root,
        venv_root=args.venv_root,
        python_version=args.python_version,
        receipt_path=args.receipt,
        dry_run=args.dry_run,
        provision_simulator=args.provision_simulator,
        runtime_config=args.runtime_config,
    )
    print(json.dumps(receipt.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
