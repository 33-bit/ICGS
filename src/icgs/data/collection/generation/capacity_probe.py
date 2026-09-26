"""Strict bounded configuration for real-generation capacity probes.

The probe is intentionally separate from production quota planning: it limits
total jobs and wall time, and it can only publish below the validation prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Mapping


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class CapacityStage:
    name: str
    worker_count: int
    simulator_slots: int
    max_jobs: int
    worker_timeout_s: int = 180
    max_result_bytes: int | None = None
    max_staging_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_ID.fullmatch(self.name):
            raise ValueError("stage name must be a safe identifier")
        if type(self.worker_count) is not int or not 1 <= self.worker_count <= 200:
            raise ValueError("stage worker_count must be in 1..200")
        if type(self.simulator_slots) is not int or not 1 <= self.simulator_slots <= self.worker_count:
            raise ValueError("stage simulator_slots must be within worker_count")
        if type(self.max_jobs) is not int or not 1 <= self.max_jobs <= 400:
            raise ValueError("stage max_jobs must be in 1..400")
        if type(self.worker_timeout_s) is not int or not 30 <= self.worker_timeout_s <= 600:
            raise ValueError("stage worker_timeout_s must be in 30..600")
        if self.max_result_bytes is not None:
            if type(self.max_result_bytes) is not int or self.max_result_bytes <= 0:
                raise ValueError("stage max_result_bytes must be a positive integer")
        if self.max_staging_bytes is not None:
            if type(self.max_staging_bytes) is not int or self.max_staging_bytes <= 0:
                raise ValueError("stage max_staging_bytes must be a positive integer")
        if self.max_result_bytes is not None and self.max_staging_bytes is not None:
            if self.max_result_bytes > self.max_staging_bytes:
                raise ValueError("stage max_result_bytes cannot exceed max_staging_bytes")


@dataclass(frozen=True)
class CapacityProbeConfig:
    run_id: str
    hf_subfolder: str
    max_total_jobs: int
    max_runtime_s: int
    stages: tuple[CapacityStage, ...]
    max_result_bytes: int | None = None
    max_total_staging_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not _SAFE_ID.fullmatch(self.run_id):
            raise ValueError("run_id must be a safe identifier")
        if not self.hf_subfolder.startswith("validation/") or ".." in Path(self.hf_subfolder).parts:
            raise ValueError("hf_subfolder must remain under validation/")
        if type(self.max_total_jobs) is not int or not 1 <= self.max_total_jobs <= 400:
            raise ValueError("max_total_jobs must be in 1..400")
        if type(self.max_runtime_s) is not int or not 60 <= self.max_runtime_s <= 5400:
            raise ValueError("max_runtime_s must be in 60..5400")
        if not self.stages:
            raise ValueError("at least one stage is required")
        if sum(stage.max_jobs for stage in self.stages) > self.max_total_jobs:
            raise ValueError("stage job caps exceed max_total_jobs")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        if self.max_result_bytes is not None:
            if type(self.max_result_bytes) is not int or self.max_result_bytes <= 0:
                raise ValueError("max_result_bytes must be a positive integer")
        if self.max_total_staging_bytes is not None:
            if type(self.max_total_staging_bytes) is not int or self.max_total_staging_bytes <= 0:
                raise ValueError("max_total_staging_bytes must be a positive integer")
        if self.max_result_bytes is not None and self.max_total_staging_bytes is not None:
            if self.max_result_bytes > self.max_total_staging_bytes:
                raise ValueError("max_result_bytes cannot exceed max_total_staging_bytes")
        for stage in self.stages:
            if stage.max_staging_bytes is not None and self.max_total_staging_bytes is not None:
                if stage.max_staging_bytes > self.max_total_staging_bytes:
                    raise ValueError(f"stage {stage.name} max_staging_bytes cannot exceed max_total_staging_bytes")
            if stage.max_result_bytes is not None and self.max_total_staging_bytes is not None:
                if stage.max_result_bytes > self.max_total_staging_bytes:
                    raise ValueError(f"stage {stage.name} max_result_bytes cannot exceed max_total_staging_bytes")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapacityProbeConfig":
        allowed = {
            "run_id", "hf_subfolder", "max_total_jobs", "max_runtime_s", "stages",
            "max_result_bytes", "max_total_staging_bytes", "max_staging_bytes",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("unknown capacity probe field(s): " + ", ".join(sorted(unknown)))
        missing = {"run_id", "hf_subfolder", "max_total_jobs", "max_runtime_s", "stages"} - set(payload)
        if missing:
            raise ValueError("missing capacity probe field(s): " + ", ".join(sorted(missing)))
        stages = payload["stages"]
        if not isinstance(stages, list):
            raise ValueError("stages must be a list")
        allowed_stage_fields = {
            "name", "worker_count", "simulator_slots", "max_jobs",
            "worker_timeout_s", "max_result_bytes", "max_staging_bytes",
        }
        required_stage_fields = {"name", "worker_count", "simulator_slots", "max_jobs"}
        parsed = []
        for item in stages:
            if not isinstance(item, Mapping) or not (required_stage_fields <= set(item) <= allowed_stage_fields):
                raise ValueError("each stage must contain at least name, worker_count, simulator_slots, max_jobs")
            parsed.append(CapacityStage(**item))

        max_total_staging_bytes = payload.get("max_total_staging_bytes")
        if max_total_staging_bytes is None:
            max_total_staging_bytes = payload.get("max_staging_bytes")

        return cls(
            payload["run_id"],
            payload["hf_subfolder"],
            payload["max_total_jobs"],
            payload["max_runtime_s"],
            tuple(parsed),
            max_result_bytes=payload.get("max_result_bytes"),
            max_total_staging_bytes=max_total_staging_bytes,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CapacityProbeConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = [asdict(stage) for stage in self.stages]
        return data


def categorize_artifact_bytes(result_dir: str | Path) -> dict[str, int]:
    """Categorize files in a result directory into archive chunks, manifests, debug, views, and receipts."""
    categories: dict[str, int] = {
        "chunks": 0,
        "manifests": 0,
        "debug": 0,
        "views": 0,
        "receipts": 0,
    }
    path = Path(result_dir)
    if not path.is_dir():
        return categories
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(path).as_posix()
        size = item.stat().st_size
        if rel.startswith("data/") or rel.endswith(".npz"):
            categories["chunks"] += size
        elif rel.endswith(".manifest.json") or rel in {"manifest.json", "artifact_manifest.json", "episode.json", "attempt.json"}:
            categories["manifests"] += size
        elif rel == "debug.json" or rel.startswith("debug/"):
            categories["debug"] += size
        elif rel.startswith("views/") or rel.endswith(".view.json"):
            categories["views"] += size
        elif "receipt" in rel.lower() and rel.endswith(".json"):
            categories["receipts"] += size
        else:
            categories.setdefault("other", 0)
            categories["other"] += size
    return categories


def extract_result_metrics(
    result_dir: str | Path,
    *,
    fallback_timeline: Mapping[str, int] | None = None,
    fallback_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract per-result size, boundaries, raw point counts, and peak bytes without loading all prior episodes."""
    path = Path(result_dir)
    categories = categorize_artifact_bytes(path)
    total_bytes = sum(categories.values())

    manifest_payload: dict[str, Any] | None = None
    for candidate in ("episode.manifest.json", "attempt.manifest.json", "manifest.json"):
        m_path = path / candidate
        if m_path.is_file():
            try:
                manifest_payload = json.loads(m_path.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    boundaries = 0
    raw_points = 0
    local_peak_bytes = None
    archive_profile = fallback_profile

    if manifest_payload is not None:
        if "archive_profile" in manifest_payload and isinstance(manifest_payload["archive_profile"], dict):
            archive_profile = manifest_payload["archive_profile"]

        local_write_limits = manifest_payload.get("local_write_limits")
        if isinstance(local_write_limits, dict) and "local_peak_bytes_upper_bound" in local_write_limits:
            local_peak_bytes = int(local_write_limits["local_peak_bytes_upper_bound"])

        timeline = manifest_payload.get("timeline")
        if isinstance(timeline, dict) and "observations" in timeline:
            boundaries = int(timeline["observations"])
        elif fallback_timeline and "observations" in fallback_timeline:
            boundaries = int(fallback_timeline["observations"])

        array_specs = manifest_payload.get("array_specs")
        if isinstance(array_specs, dict):
            if "online_points" in array_specs and isinstance(array_specs["online_points"], dict):
                shape = array_specs["online_points"].get("shape")
                if isinstance(shape, list) and len(shape) >= 1:
                    raw_points = int(shape[0])
            else:
                for spec in array_specs.values():
                    if isinstance(spec, dict):
                        role = str(spec.get("semantic_role", ""))
                        if role in {"online_observations/points", "raw_arrays/measured_points", "raw_arrays/prefix_points"} or role.endswith("/points"):
                            shape = spec.get("shape")
                            if isinstance(shape, list) and len(shape) >= 2 and shape[-1] == 3:
                                raw_points = int(shape[0])
                                break
    else:
        if fallback_timeline and "observations" in fallback_timeline:
            boundaries = int(fallback_timeline["observations"])

    return {
        "artifact_bytes": total_bytes,
        "bytes_by_category": categories,
        "boundaries": boundaries,
        "raw_points": raw_points,
        "archive_profile": archive_profile,
        "local_peak_bytes": local_peak_bytes,
        "local_peak_bytes_semantic": "local_peak_bytes_upper_bound",
    }


def preflight_stage_capacity(
    stage: CapacityStage,
    probe: CapacityProbeConfig,
    output_dir: str | Path,
    *,
    disk_usage_fn: Callable[[Path], Any] | None = None,
) -> None:
    """Fail-closed preflight to ensure disk capacity and caps before launching any workers."""
    if disk_usage_fn is None:
        disk_usage_fn = shutil.disk_usage
    path = Path(output_dir)
    check_dir = path
    while not check_dir.exists() and check_dir != check_dir.parent:
        check_dir = check_dir.parent
    if not check_dir.exists():
        check_dir = Path(".")

    effective_max_staging = (
        stage.max_staging_bytes
        if stage.max_staging_bytes is not None
        else probe.max_total_staging_bytes
    )
    effective_max_result = (
        stage.max_result_bytes
        if stage.max_result_bytes is not None
        else probe.max_result_bytes
    )

    if effective_max_result is not None and effective_max_staging is not None:
        if effective_max_result > effective_max_staging:
            raise ValueError(
                f"stage max_result_bytes ({effective_max_result}) exceeds "
                f"staging cap ({effective_max_staging})"
            )

    required_staging_bytes = 0
    if effective_max_staging is not None:
        required_staging_bytes = effective_max_staging
    elif effective_max_result is not None:
        required_staging_bytes = effective_max_result * stage.max_jobs

    if required_staging_bytes > 0:
        usage = disk_usage_fn(check_dir)
        free_bytes = getattr(usage, "free", 0)
        if free_bytes < required_staging_bytes:
            raise RuntimeError(
                f"preflight capacity insufficient: required {required_staging_bytes} bytes "
                f"staging capacity, but only {free_bytes} bytes are free on {check_dir}"
            )
