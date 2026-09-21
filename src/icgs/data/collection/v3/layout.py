"""Phase-1 dataset tree. No anchors/branches/continuations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from icgs.data.collection.v3.camera import CAMERA_PROFILES
from icgs.data.collection.v3.lighting import LIGHTING_PROFILES
from icgs.data.collection.v3.protocol import V3_PROTOCOL
from icgs.data.collection.v3.steps import V3_PROGRAMS


def initialize_v3_dataset_layout(root: str | Path, *, approved_manifest: Mapping[str, Any] | None = None) -> Path:
    root = Path(root).resolve()
    for name in ("metadata", "programs", "episodes", "views", "reports", "quarantine"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        (root / "programs" / split).mkdir(parents=True, exist_ok=True)
        (root / "episodes" / split).mkdir(parents=True, exist_ok=True)
    catalog = list((approved_manifest or {}).get("catalog", ()))
    programs = {pid: spec.__dict__ for pid, spec in V3_PROGRAMS.items()}
    if catalog:
        programs = {row["program_id"]: row for row in catalog}
    splits = {pid: ("dev" if spec.split == "development" else spec.split) for pid, spec in V3_PROGRAMS.items()}
    metadata = {
        "layout_version": V3_PROTOCOL.layout_version,
        "dataset_version": V3_PROTOCOL.dataset_version,
        "phase": V3_PROTOCOL.phase,
        "views": list(V3_PROTOCOL.views),
        "preserves": V3_PROTOCOL.preserves,
    }
    _write(root / "metadata" / "dataset.yaml", metadata)
    _write(root / "metadata" / "programs.json", {pid: {
        "program_id": spec.program_id,
        "split": spec.split,
        "family": spec.family,
        "semantic_status": spec.semantic_status,
        "training_eligible": spec.training_eligible,
        "compiler_routine_id": spec.compiler_routine_id,
    } for pid, spec in V3_PROGRAMS.items()})
    _write(root / "metadata" / "splits.json", splits)
    _write(root / "metadata" / "calibrations.json", {key: dict(value) for key, value in CAMERA_PROFILES.items()})
    _write(root / "metadata" / "lighting.json", {key: dict(value) for key, value in LIGHTING_PROFILES.items()})
    _write(root / "metadata" / "controller_versions.json", {
        V3_PROTOCOL.controller_protocol_id: {"protocol_id": V3_PROTOCOL.controller_protocol_id}
    })
    _write(root / "metadata" / "assets.json", {})
    (root / "metadata" / "README.md").write_text(
        "ICGS primary v3 phase 1 layout. Views are pointer files. Simulator crashes go to quarantine.\n",
        encoding="utf-8",
    )
    for view in V3_PROTOCOL.views:
        (root / "views" / f"{view}.jsonl").touch()
    return root


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
