"""Verify an installed ICGS environment without exposing credentials."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# When invoked as ``python scripts/verify_environment.py`` Python places the
# scripts directory on sys.path, so explicitly expose the clone root for the
# shared redaction helpers before importing the sibling module.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.setup_environment import (
    PROFILE_NAMES,
    _redact_string,
    _safe_environment,
    _validate_profile,
)

Runner = Callable[..., Any]
_IMPORT_PROBES = {
    "icgs_imports": "import icgs; import icgs.data.collection.generation",
    "core_dependencies": "import numpy; import scipy; import torch",
    "pyg_abi": "import torch; import torch_geometric",
}


def _status(status: str, detail: str = "") -> dict[str, str]:
    result = {"status": status}
    if detail:
        result["detail"] = _redact_string(detail)[:500]
    return result


def _probe(
    command: Sequence[str],
    *,
    runner: Runner,
    cwd: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    try:
        options: dict[str, Any] = {
            "check": False,
            "capture_output": True,
            "text": True,
            "cwd": str(cwd) if cwd is not None else None,
        }
        if environment is not None:
            options["env"] = dict(environment)
        result = runner([str(part) for part in command], **options)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return_code = int(getattr(result, "returncode", 1))
    output = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return return_code == 0, _redact_string(output.strip())


def _python_check(
    python_executable: str,
    runner: Runner,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    ok, output = _probe(
        [python_executable, "-c", "import sys; print(sys.version.split()[0])"],
        runner=runner,
        environment=environment,
    )
    if not ok:
        return _status("FAIL", output or "python executable could not run")
    try:
        major, minor = (int(part) for part in output.split(".", 2)[:2])
    except (TypeError, ValueError):
        return _status("FAIL", f"unparseable Python version: {output}")
    if (major, minor) < (3, 10) or (major, minor) >= (3, 13):
        return _status("FAIL", f"Python {major}.{minor} is outside >=3.10,<3.13")
    return _status("PASS", f"Python {major}.{minor}")


def _generation_probe_environment(
    repo_root: Path,
    simulator_root: str | Path | None,
    rlbench_root: str | Path | None,
) -> dict[str, str]:
    """Build the headless simulator environment without credential variables."""
    environment = _safe_environment()
    if simulator_root is not None:
        simulator = str(Path(simulator_root).resolve())
        environment["COPPELIASIM_ROOT"] = simulator
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (simulator, environment.get("LD_LIBRARY_PATH", "")))
        )
        environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = simulator
        environment["QT_QPA_PLATFORM"] = "xcb"
    paths = [str(repo_root / "src")]
    if rlbench_root is not None:
        paths.append(str(Path(rlbench_root).resolve()))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _credential_check(paths: Sequence[str | Path]) -> dict[str, str]:
    if not paths:
        return _status("NOT_RUN", "no credential path was supplied")
    failures: list[str] = []
    for raw in paths:
        path = Path(raw)
        try:
            mode = path.stat().st_mode & 0o777
        except OSError:
            failures.append("missing credential path")
            continue
        if mode & 0o077:
            failures.append("credential path must not be group/world readable")
    if failures:
        return _status("FAIL", "; ".join(failures))
    return _status("PASS", f"{len(paths)} credential path(s) present with restricted mode")


def verify_environment(
    profile: str,
    *,
    repo_root: str | Path = ".",
    python_executable: str | Path | None = None,
    simulator_root: str | Path | None = None,
    rlbench_root: str | Path | None = None,
    credential_paths: Sequence[str | Path] = (),
    receipt_path: str | Path | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Return machine-readable checks for a profile.

    Probe output is reduced to a short, redacted diagnostic.  In particular,
    credential files are only stat-ed; their contents are never opened.
    """
    profile = _validate_profile(profile)
    root = Path(repo_root).resolve()
    python = str(python_executable or sys.executable)
    probe_environment = (
        _generation_probe_environment(root, simulator_root, rlbench_root)
        if profile == "generation"
        else None
    )
    checks: dict[str, dict[str, str]] = {}
    checks["python"] = _python_check(python, runner, environment=probe_environment)
    for name, expression in _IMPORT_PROBES.items():
        ok, output = _probe(
            [python, "-c", expression],
            runner=runner,
            cwd=root,
            environment=probe_environment,
        )
        checks[name] = _status("PASS" if ok else "FAIL", output or ("import probe failed" if not ok else ""))

    if profile == "cuda118":
        ok, output = _probe(
            [
                python,
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda == '11.8' else 1)",
            ],
            runner=runner,
        )
        checks["cuda"] = _status("PASS" if ok else "FAIL", output or "CUDA 11.8 runtime is unavailable")
    else:
        checks["cuda"] = _status("NOT_RUN", "CUDA profile not selected")

    if profile == "generation":
        ok, output = _probe(
            ["sh", "-c", "command -v Xvfb"],
            runner=runner,
            environment=probe_environment,
        )
        checks["renderer"] = _status("PASS", output or "Xvfb found") if ok else _status("FAIL", "Xvfb is not installed")
        if simulator_root is None:
            checks["simulator"] = _status("FAIL", "simulator_root was not supplied")
        else:
            simulator = Path(simulator_root)
            checks["simulator"] = _status("PASS", "simulator root exists") if simulator.is_dir() else _status("FAIL", "simulator root is missing")
        if rlbench_root is None:
            checks["rlbench_pyrep"] = _status("FAIL", "rlbench_root was not supplied")
        else:
            rlbench = Path(rlbench_root)
            if not rlbench.is_dir():
                checks["rlbench_pyrep"] = _status("FAIL", "RLBench root is missing")
            else:
                ok, output = _probe(
                    [python, "-c", "import pyrep; import rlbench"],
                    runner=runner,
                    cwd=rlbench,
                    environment=probe_environment,
                )
                checks["rlbench_pyrep"] = _status("PASS" if ok else "FAIL", output or "PyRep/RLBench imports failed")
    else:
        checks["renderer"] = _status("NOT_RUN", "generation profile not selected")
        checks["simulator"] = _status("NOT_RUN", "generation profile not selected")
        checks["rlbench_pyrep"] = _status("NOT_RUN", "generation profile not selected")

    checks["credentials"] = _credential_check(credential_paths)
    if receipt_path is None:
        checks["setup_receipt"] = _status("NOT_RUN", "no setup receipt was supplied")
    else:
        receipt = Path(receipt_path)
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            checks["setup_receipt"] = _status("FAIL", f"setup receipt is unreadable: {exc}")
        else:
            valid = isinstance(payload, dict) and payload.get("schema_version") == 1 and payload.get("profile") == profile
            checks["setup_receipt"] = _status("PASS" if valid else "FAIL", "redacted setup receipt matches profile" if valid else "setup receipt schema/profile mismatch")
    statuses = [entry["status"] for entry in checks.values()]
    overall = "FAIL" if "FAIL" in statuses else "PASS" if all(status in {"PASS", "NOT_RUN", "SKIPPED"} for status in statuses) else "NOT_RUN"
    return {
        "schema_version": 1,
        "profile": profile,
        "repo_root": str(root),
        "python_executable": python,
        "platform": platform.platform(),
        "checks": checks,
        "status": overall,
    }


def render_summary(result: Mapping[str, Any]) -> str:
    lines = [f"ICGS environment: {result.get('status', 'UNKNOWN')} ({result.get('profile', 'unknown')})"]
    for name, check in result.get("checks", {}).items():
        detail = f": {check['detail']}" if check.get("detail") else ""
        lines.append(f"{check.get('status', 'UNKNOWN'):8} {name}{detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--simulator-root")
    parser.add_argument("--rlbench-root")
    parser.add_argument("--credential-path", action="append", default=[])
    parser.add_argument("--receipt")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)
    result = verify_environment(
        args.profile,
        repo_root=args.repo_root,
        python_executable=args.python_executable,
        simulator_root=args.simulator_root,
        rlbench_root=args.rlbench_root,
        credential_paths=args.credential_path,
        receipt_path=args.receipt,
    )
    if not args.json:
        print(render_summary(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
