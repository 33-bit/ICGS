"""Strict JSON contracts for distributed primary-v3 collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from icgs.data.collection.v3.batch import AttemptPlan, attempt_from_dict


_OUTCOMES = frozenset({"success", "valid_failure", "simulator_crash", "invalid_observation"})


def _nonblank(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _sha(value: str, name: str, lengths: tuple[int, ...]) -> str:
    _nonblank(value, name)
    if len(value) not in lengths or any(ch not in "0123456789abcdef" for ch in value.lower()):
        expected = " or ".join(str(item) for item in lengths)
        raise ValueError(f"{name} must be {expected} hexadecimal characters")
    return value.lower()


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    run_root: str
    code_revision: str
    approved_manifest_sha256: str
    worker_count: int = 200
    publish_interval_s: int = 300
    hf_repo: str = "33bit/icgs"
    hf_subfolder: str = "primary_v3"

    def __post_init__(self) -> None:
        _nonblank(self.run_id, "run_id")
        _nonblank(self.run_root, "run_root")
        _sha(self.code_revision, "code_revision", (40, 64))
        _sha(self.approved_manifest_sha256, "approved_manifest_sha256", (64,))
        if self.worker_count != 200:
            raise ValueError("worker_count must be 200")
        if self.publish_interval_s != 300:
            raise ValueError("publish_interval_s must be 300")
        _nonblank(self.hf_repo, "hf_repo")
        if self.hf_subfolder != "primary_v3":
            raise ValueError("hf_subfolder must be primary_v3")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunConfig":
        return cls(**dict(payload))


@dataclass(frozen=True)
class GenerationJob:
    job_id: str
    run_id: str
    attempt_id: str
    episode_id: str
    program_id: str
    plan: AttemptPlan
    code_revision: str
    manifest_sha256: str
    output_root: str
    retry_generation: int = 0

    def __post_init__(self) -> None:
        for name in ("job_id", "run_id", "attempt_id", "episode_id", "program_id", "output_root"):
            _nonblank(getattr(self, name), name)
        _sha(self.code_revision, "code_revision", (40, 64))
        _sha(self.manifest_sha256, "manifest_sha256", (64,))
        if self.program_id != self.plan.program_id:
            raise ValueError("program_id must match plan.program_id")
        if self.retry_generation < 0:
            raise ValueError("retry_generation must be nonnegative")

    @classmethod
    def create(cls, **values: Any) -> "GenerationJob":
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plan"] = self.plan.as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationJob":
        values = dict(payload)
        values["plan"] = attempt_from_dict(values["plan"])
        return cls(**values)


@dataclass(frozen=True)
class WorkerResult:
    job_id: str
    attempt_id: str
    episode_id: str | None
    program_id: str
    outcome: str
    result_dir: str
    file_sha256: dict[str, str]
    timeline: dict[str, int] | None

    def __post_init__(self) -> None:
        for name in ("job_id", "attempt_id", "program_id", "result_dir"):
            _nonblank(getattr(self, name), name)
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"unsupported outcome: {self.outcome!r}")
        if self.outcome in {"simulator_crash", "invalid_observation"}:
            if self.episode_id is not None:
                raise ValueError(f"episode_id must be null for {self.outcome}")
            if self.timeline is not None:
                raise ValueError(f"timeline must be null for {self.outcome}")
        else:
            _nonblank(self.episode_id or "", "episode_id")
            if not isinstance(self.timeline, dict):
                raise ValueError("episode outcomes require timeline")
            actions = self.timeline.get("actions")
            observations = self.timeline.get("observations")
            durations = self.timeline.get("durations")
            if not all(isinstance(item, int) and item >= 0 for item in (actions, observations, durations)):
                raise ValueError("timeline counts must be nonnegative integers")
            if observations != actions + 1 or durations != actions:
                raise ValueError("episode timeline must have T actions, T+1 observations and T durations")
        for relative, digest in self.file_sha256.items():
            _nonblank(relative, "file_sha256 path")
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("file_sha256 paths must be relative and contained")
            _sha(digest, f"file_sha256[{relative}]", (64,))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerResult":
        values = dict(payload)
        values["file_sha256"] = dict(values.get("file_sha256") or {})
        values["timeline"] = dict(values["timeline"]) if values.get("timeline") is not None else None
        return cls(**values)


@dataclass(frozen=True)
class QueueCounts:
    pending: int
    claimed: int
    ready: int
    ingested: int
    published: int


__all__ = ["GenerationJob", "QueueCounts", "RunConfig", "WorkerResult"]
