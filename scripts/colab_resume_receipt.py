"""Build the shared primary_v2 resume receipt for CPU coordinators.

Workers intentionally run without HF credentials.  The coordinator is therefore
the single writer for both the merged dataset manifest and its receipt.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_approved_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the approved composition manifest and return its content hash."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("manifest_version") != 1:
        raise ValueError("approved composition manifest has an unsupported version")
    if data.get("protocol_id") != "icgs-composition-primary-v1":
        raise ValueError("approved composition manifest has an unsupported protocol")
    return data, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _target_for(program_id: str, row: dict[str, Any], targets: dict[str, Any]) -> int:
    development = targets.get("development_successes_by_program", {})
    if program_id in development:
        return int(development[program_id])
    if row.get("split") == "train":
        return int(targets.get("train_successes_per_program", 200))
    if row.get("split") == "test":
        return int(targets.get("test_contexts_per_composition", 100))
    return 0


def build_resume_receipt(
    manifest: dict[str, Any],
    approved_manifest: dict[str, Any],
    *,
    approved_manifest_digest: str,
    hf_repo: str,
    hf_subfolder: str,
    recent_commits: Iterable[str] = (),
    status: str = "RUNNING",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create a receipt whose counts are derived from the merged manifest."""
    counts = Counter(
        ep.get("program_id")
        for ep in manifest.get("episodes", [])
        if ep.get("program_id")
    )
    targets = approved_manifest.get("generation_targets", {})
    task_progress: dict[str, dict[str, Any]] = {}
    total_target = 0
    total_generated = 0
    for row in approved_manifest.get("catalog", []):
        program_id = row["program_id"]
        target = _target_for(program_id, row, targets)
        generated = int(counts.get(program_id, 0))
        split = "dev" if row.get("split") == "development" else row.get("split", "train")
        task_progress[program_id] = {
            "split": split,
            "target": target,
            "generated": generated,
            "remaining": max(0, target - generated),
            "status": "COMPLETED" if generated >= target else "IN_PROGRESS",
        }
        total_target += target
        total_generated += generated

    shortfall = max(0, total_target - total_generated)
    receipt = {
        "receipt_version": 1,
        "status": "COMPLETED" if shortfall == 0 else status,
        "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "approved_manifest_digest": approved_manifest_digest,
        "hf_repo": hf_repo,
        "hf_subfolder": hf_subfolder,
        "total_target": total_target,
        "total_generated": total_generated,
        "target_shortfall": shortfall,
        "total_quarantined": len(manifest.get("quarantined", [])),
        "total_failure_attempts": len(manifest.get("failure_attempts", [])),
        "recent_commits": list(recent_commits)[-20:],
        "task_progress": task_progress,
    }
    return receipt


def write_resume_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    """Write a complete receipt payload as UTF-8 JSON."""
    Path(path).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
