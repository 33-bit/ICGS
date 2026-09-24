"""Strict bounded configuration for real-generation capacity probes.

The probe is intentionally separate from production quota planning: it limits
total jobs and wall time, and it can only publish below the validation prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class CapacityStage:
    name: str
    worker_count: int
    simulator_slots: int
    max_jobs: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_ID.fullmatch(self.name):
            raise ValueError("stage name must be a safe identifier")
        if type(self.worker_count) is not int or not 1 <= self.worker_count <= 200:
            raise ValueError("stage worker_count must be in 1..200")
        if type(self.simulator_slots) is not int or not 1 <= self.simulator_slots <= self.worker_count:
            raise ValueError("stage simulator_slots must be within worker_count")
        if type(self.max_jobs) is not int or not 1 <= self.max_jobs <= 400:
            raise ValueError("stage max_jobs must be in 1..400")


@dataclass(frozen=True)
class CapacityProbeConfig:
    run_id: str
    hf_subfolder: str
    max_total_jobs: int
    max_runtime_s: int
    stages: tuple[CapacityStage, ...]

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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapacityProbeConfig":
        allowed = {"run_id", "hf_subfolder", "max_total_jobs", "max_runtime_s", "stages"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("unknown capacity probe field(s): " + ", ".join(sorted(unknown)))
        missing = allowed - set(payload)
        if missing:
            raise ValueError("missing capacity probe field(s): " + ", ".join(sorted(missing)))
        stages = payload["stages"]
        if not isinstance(stages, list):
            raise ValueError("stages must be a list")
        stage_fields = {"name", "worker_count", "simulator_slots", "max_jobs"}
        parsed = []
        for item in stages:
            if not isinstance(item, Mapping) or set(item) != stage_fields:
                raise ValueError("each stage must contain exactly name, worker_count, simulator_slots, max_jobs")
            parsed.append(CapacityStage(**item))
        return cls(payload["run_id"], payload["hf_subfolder"], payload["max_total_jobs"], payload["max_runtime_s"], tuple(parsed))

    @classmethod
    def from_file(cls, path: str | Path) -> "CapacityProbeConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = [asdict(stage) for stage in self.stages]
        return data
