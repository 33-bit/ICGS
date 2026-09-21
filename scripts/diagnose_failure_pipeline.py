"""Read-only diagnostics for worker failure staging and coordinator state."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def main() -> None:
    print(json.dumps({"pid": os.getpid(), "cwd": os.getcwd()}))
    roots = sorted(Path("/content").glob("staging*"))
    for root in roots:
        if not root.is_dir():
            continue
        quarantine = root / "quarantine"
        failure = root / "failure_attempts_v2"
        q_ids = sorted(p.name for p in quarantine.iterdir() if p.is_dir()) if quarantine.is_dir() else []
        f_ids = sorted(p.name for p in failure.iterdir() if p.is_dir()) if failure.is_dir() else []
        reports = []
        for eid in q_ids:
            report = quarantine / eid / "error_report.json"
            if report.is_file():
                data = _read_json(report)
                reports.append({"episode_id": eid, "status": data.get("status"), "mtime": report.stat().st_mtime})
        print(json.dumps({"root": str(root), "quarantine_ids": q_ids, "failure_ids": f_ids, "reports": reports}))
    for state in sorted(Path("/content").glob("coordinator*_state.json")):
        data = _read_json(state)
        print(json.dumps({"state": str(state), "total_committed": data.get("total_committed"), "total_failure_attempts": data.get("total_failure_attempts"), "last_commits": data.get("commits", [])[-3:]}))


if __name__ == "__main__":
    main()
