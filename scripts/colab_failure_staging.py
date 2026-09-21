"""Discover locally staged failure attempts for the CPU commit coordinators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def find_ready_failures(
    staging_dirs: Iterable[Path],
    committed_ids: set[str],
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Return uncommitted quarantine attempts with their error reports.

    Workers do not have HF credentials.  They therefore materialize failures
    under ``quarantine/<episode_id>`` and optionally
    ``failure_attempts_v2/<episode_id>``; the coordinator publishes both trees.
    """
    ready: list[tuple[str, Path, dict[str, Any]]] = []
    for sdir in staging_dirs:
        root = sdir / "quarantine"
        if not root.is_dir():
            continue
        for ep_dir in sorted(root.iterdir()):
            if not ep_dir.is_dir() or ep_dir.name in committed_ids:
                continue
            report_path = ep_dir / "error_report.json"
            if not report_path.is_file():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(report, dict):
                continue
            report.setdefault("episode_id", ep_dir.name)
            report.setdefault("status", "failed")
            ready.append((ep_dir.name, sdir, report))
    return ready


def plan_failure_sidecar_copies(
    quarantine_files: Iterable[str],
    existing_failure_files: Iterable[str],
    *,
    quarantine_prefix: str = "primary_v2/quarantine",
    failure_prefix: str = "primary_v2/failure_attempts",
) -> list[tuple[str, str]]:
    """Plan idempotent repository copies for missing failure-attempt files.

    Older commits may contain a complete ``quarantine/<episode_id>`` record but
    no corresponding ``failure_attempts/<episode_id>`` tree.  This pure helper
    keeps a backfill limited to files that are actually missing and makes the
    source/destination mapping testable before any Hub mutation occurs.
    """
    q_prefix = quarantine_prefix.rstrip("/") + "/"
    f_prefix = failure_prefix.rstrip("/") + "/"
    existing = {str(path).lstrip("/") for path in existing_failure_files}
    planned: list[tuple[str, str]] = []
    for raw_path in sorted({str(path).lstrip("/") for path in quarantine_files}):
        if not raw_path.startswith(q_prefix):
            continue
        relative = raw_path[len(q_prefix) :]
        if not relative or relative.endswith("/"):
            continue
        destination = f_prefix + relative
        if destination not in existing:
            planned.append((raw_path, destination))
    return planned


def select_failure_publish_root(quarantine_dir: Path, sidecar_dir: Path) -> Path:
    """Choose a complete failure tree, falling back to quarantine artifacts.

    A simulator can fail before enough boundary data exists for the v2 layout
    writer.  The raw quarantine record is still a valid failure attempt and
    must be published under ``failure_attempts`` rather than silently omitted.
    """
    if sidecar_dir.is_dir() and any(path.is_file() for path in sidecar_dir.rglob("*")):
        return sidecar_dir
    return quarantine_dir
