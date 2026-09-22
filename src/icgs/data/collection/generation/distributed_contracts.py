"""Strict JSON contracts for distributed generation collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

from icgs.data.collection.generation.batch import AttemptPlan, attempt_from_dict


_OUTCOMES = frozenset({"success", "valid_failure", "simulator_crash", "invalid_observation"})
_MACHINE_FIELDS = frozenset({
    "repo_root",
    "python_executable",
    "simulator_root",
    "rlbench_root",
    "display_base",
    "display_width",
    "display_height",
    "simulator_slots",
    "worker_timeout_s",
})
_RUNTIME_RUN_FIELDS = frozenset({
    "run_id",
    "run_root",
    "worker_count",
    "publish_interval_s",
    "hf_repo",
    "hf_subfolder",
    "publication_enabled",
    "validation_mode",
})


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


def _mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(payload)


def _reject_unknown(payload: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        fields = ", ".join(sorted(str(item) for item in unknown))
        raise ValueError(f"{name} has unknown field(s): {fields}")


def _require_fields(payload: Mapping[str, Any], required: frozenset[str], name: str) -> None:
    missing = required - set(payload)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required field(s): {fields}")


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _absolute_path(value: str, name: str) -> None:
    _nonblank(value, name)
    if not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")


def _validate_hf_subfolder(value: str, *, validation_mode: bool) -> None:
    _nonblank(value, "hf_subfolder")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        if validation_mode:
            raise ValueError("validation_mode hf_subfolder must remain contained under validation/")
        raise ValueError("hf_subfolder must be a relative contained path")
    if validation_mode and not value.startswith("validation/"):
        raise ValueError("validation_mode requires an hf_subfolder under validation/")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("runtime config must contain a JSON object")
    return payload


@dataclass(frozen=True)
class MachineConfig:
    repo_root: str
    python_executable: str
    simulator_root: str
    rlbench_root: str
    display_base: int
    display_width: int
    display_height: int
    simulator_slots: int
    worker_timeout_s: int

    def __post_init__(self) -> None:
        for name in ("repo_root", "python_executable", "simulator_root", "rlbench_root"):
            _absolute_path(getattr(self, name), name)
        if type(self.display_base) is not int or self.display_base < 0:
            raise ValueError("display_base must be a nonnegative integer")
        for name in ("display_width", "display_height", "simulator_slots", "worker_timeout_s"):
            _positive_int(getattr(self, name), name)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MachineConfig":
        values = _mapping(payload, "machine")
        _reject_unknown(values, _MACHINE_FIELDS, "machine")
        _require_fields(values, _MACHINE_FIELDS, "machine")
        return cls(**values)

    def validate_paths(self) -> None:
        for name in ("repo_root", "simulator_root", "rlbench_root"):
            path = Path(getattr(self, name))
            if not path.is_dir():
                raise ValueError(f"{name} must be an existing directory: {path}")
        executable = Path(self.python_executable)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(
                "python_executable must be an existing executable file: "
                f"{executable}"
            )


@dataclass(frozen=True)
class RuntimeRunConfig:
    run_id: str
    run_root: str
    worker_count: int = 1
    publish_interval_s: int = 300
    hf_repo: str | None = None
    hf_subfolder: str | None = None
    publication_enabled: bool = False
    validation_mode: bool = False

    def __post_init__(self) -> None:
        _nonblank(self.run_id, "run_id")
        _absolute_path(self.run_root, "run_root")
        _positive_int(self.worker_count, "worker_count")
        _positive_int(self.publish_interval_s, "publish_interval_s")
        if type(self.publication_enabled) is not bool:
            raise ValueError("publication_enabled must be a boolean")
        if type(self.validation_mode) is not bool:
            raise ValueError("validation_mode must be a boolean")
        if self.hf_repo is not None:
            _nonblank(self.hf_repo, "hf_repo")
        if self.hf_subfolder is not None:
            _validate_hf_subfolder(
                self.hf_subfolder,
                validation_mode=self.validation_mode,
            )
        if self.publication_enabled and not (self.hf_repo and self.hf_subfolder):
            raise ValueError("publication requires hf_repo and hf_subfolder")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeRunConfig":
        values = _mapping(payload, "run")
        _reject_unknown(values, _RUNTIME_RUN_FIELDS, "run")
        _require_fields(values, frozenset({"run_id", "run_root"}), "run")
        return cls(**values)


@dataclass(frozen=True)
class GenerationRuntimeConfig:
    machine: MachineConfig
    run: RuntimeRunConfig

    def __post_init__(self) -> None:
        if not isinstance(self.machine, MachineConfig):
            raise ValueError("machine must be a MachineConfig")
        if not isinstance(self.run, RuntimeRunConfig):
            raise ValueError("run must be a RuntimeRunConfig")
        if self.machine.simulator_slots > self.run.worker_count:
            raise ValueError("simulator_slots must be within 1..worker_count")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationRuntimeConfig":
        values = _mapping(payload, "runtime config")
        allowed = frozenset({"machine", "run"})
        _reject_unknown(values, allowed, "runtime config")
        _require_fields(values, allowed, "runtime config")
        return cls(
            machine=MachineConfig.from_dict(values["machine"]),
            run=RuntimeRunConfig.from_dict(values["run"]),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        check_paths: bool = True,
    ) -> "GenerationRuntimeConfig":
        if type(check_paths) is not bool:
            raise ValueError("check_paths must be a boolean")
        config = cls.from_dict(_read_json_object(path))
        if check_paths:
            config.machine.validate_paths()
        return config

    def as_dict(self) -> dict[str, Any]:
        return {"machine": asdict(self.machine), "run": asdict(self.run)}

    def resolved_environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        if base is None:
            environment = os.environ.copy()
        else:
            environment = dict(base)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()):
            raise ValueError("base environment keys and values must be strings")

        pathsep = os.pathsep
        simulator_root = self.machine.simulator_root
        environment.update({
            "COPPELIASIM_ROOT": simulator_root,
            "LD_LIBRARY_PATH": pathsep.join(filter(None, (
                simulator_root,
                environment.get("LD_LIBRARY_PATH", ""),
            ))),
            "QT_QPA_PLATFORM_PLUGIN_PATH": simulator_root,
            "PYTHONPATH": pathsep.join(filter(None, (
                str(Path(self.machine.repo_root) / "src"),
                self.machine.rlbench_root,
                environment.get("PYTHONPATH", ""),
            ))),
            "ICGS_SIMULATOR_SLOTS": str(self.machine.simulator_slots),
            "ICGS_HF_REPO": self.run.hf_repo or "",
            "ICGS_HF_SUBFOLDER": self.run.hf_subfolder or "",
            "ICGS_PUBLICATION_ENABLED": "1" if self.run.publication_enabled else "0",
            "ICGS_VALIDATION_MODE": "1" if self.run.validation_mode else "0",
        })
        return environment


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    run_root: str
    code_revision: str
    approved_manifest_sha256: str
    worker_count: int = 1
    publish_interval_s: int = 300
    hf_repo: str = "33bit/icgs"
    hf_subfolder: str = "generation"
    publication_enabled: bool = False
    validation_mode: bool = False

    def __post_init__(self) -> None:
        _nonblank(self.run_id, "run_id")
        _nonblank(self.run_root, "run_root")
        _sha(self.code_revision, "code_revision", (40, 64))
        _sha(self.approved_manifest_sha256, "approved_manifest_sha256", (64,))
        _positive_int(self.worker_count, "worker_count")
        _positive_int(self.publish_interval_s, "publish_interval_s")
        _nonblank(self.hf_repo, "hf_repo")
        if type(self.publication_enabled) is not bool:
            raise ValueError("publication_enabled must be a boolean")
        if type(self.validation_mode) is not bool:
            raise ValueError("validation_mode must be a boolean")
        _validate_hf_subfolder(self.hf_subfolder, validation_mode=self.validation_mode)
        if self.publication_enabled and not (self.hf_repo and self.hf_subfolder):
            raise ValueError("publication requires hf_repo and hf_subfolder")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunConfig":
        values = _mapping(payload, "run config")
        allowed = frozenset(cls.__dataclass_fields__)
        _reject_unknown(values, allowed, "run config")
        required = frozenset({
            "run_id", "run_root", "code_revision", "approved_manifest_sha256",
        })
        _require_fields(values, required, "run config")
        return cls(**values)


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
        if self.episode_id != self.plan.episode_id:
            raise ValueError("episode_id must match plan.episode_id")
        if self.attempt_id != f"att-{self.plan.episode_id}":
            raise ValueError("attempt_id must match att-<plan.episode_id>")
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


__all__ = [
    "GenerationJob",
    "GenerationRuntimeConfig",
    "MachineConfig",
    "QueueCounts",
    "RunConfig",
    "RuntimeRunConfig",
    "WorkerResult",
]
