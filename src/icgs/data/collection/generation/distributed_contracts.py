"""Strict JSON contracts for distributed generation collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
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
_VALIDATION_GATE_NAMES = (
    "success",
    "valid_failure",
    "invalid_observation",
    "infrastructure_failure",
    "malformed_result",
    "coordinator_restart",
    "publication",
)
_VALIDATION_STATUSES = frozenset({"PASS", "FAIL", "NOT_RUN"})


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
            run_root = Path(config.run.run_root)
            if not run_root.is_dir():
                raise ValueError(f"run_root must be an existing directory: {run_root}")
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
    worker_count: int = 200
    publish_interval_s: int = 300
    hf_repo: str = "33bit/icgs"
    hf_subfolder: str = "generation"
    publication_enabled: bool = False
    validation_mode: bool = False
    validation_max_jobs: int | None = None

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
        if self.validation_max_jobs is not None:
            if not self.validation_mode:
                raise ValueError("validation_max_jobs requires validation_mode=true")
            if type(self.validation_max_jobs) is not int or self.validation_max_jobs <= 0:
                raise ValueError("validation_max_jobs must be a positive integer")
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


@dataclass(frozen=True)
class CoordinatorHeartbeat:
    status: str
    phase: str
    timestamp_s: float
    pid: int
    tick_started_at_s: float | None
    last_progress_at_s: float | None
    last_progress_kind: str | None
    last_validation_error: dict[str, Any] | None
    publication: dict[str, Any]
    queue: dict[str, Any]
    planner: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("status", "phase"):
            _nonblank(getattr(self, name), name)
        for name in ("timestamp_s", "tick_started_at_s", "last_progress_at_s"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number or null")
        if type(self.pid) is not int or self.pid < 0:
            raise ValueError("pid must be a nonnegative integer")
        if self.last_progress_kind is not None:
            _nonblank(self.last_progress_kind, "last_progress_kind")
        if self.last_validation_error is not None and not isinstance(
            self.last_validation_error, dict
        ):
            raise ValueError("last_validation_error must be an object or null")
        for name in ("publication", "queue", "planner"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoordinatorHeartbeat":
        values = _mapping(payload, "coordinator heartbeat")
        fields = frozenset(cls.__dataclass_fields__)
        _reject_unknown(values, fields, "coordinator heartbeat")
        _require_fields(values, fields, "coordinator heartbeat")
        return cls(**values)


@dataclass(frozen=True)
class ValidationGateSpec:
    name: str
    required: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationGateSpec":
        values = _mapping(payload, "validation gate")
        _reject_unknown(values, frozenset({"name", "required"}), "validation gate")
        _require_fields(values, frozenset({"name", "required"}), "validation gate")
        if values["name"] not in _VALIDATION_GATE_NAMES:
            raise ValueError(f"unknown validation gate: {values['name']}")
        if type(values["required"]) is not bool:
            raise ValueError("validation gate required must be a boolean")
        return cls(name=str(values["name"]), required=values["required"])


@dataclass(frozen=True)
class ValidationPlan:
    plan_id: str
    validation_mode: bool
    worker_count: int
    max_jobs: int
    max_episodes: int
    max_attempts: int
    hf_subfolder: str
    gates: tuple[ValidationGateSpec, ...]
    fault_injection: dict[str, bool]

    def __post_init__(self) -> None:
        _nonblank(self.plan_id, "plan_id")
        if self.validation_mode is not True:
            raise ValueError("validation plan must set validation_mode=true")
        if type(self.worker_count) is not int or not 1 <= self.worker_count <= 2:
            raise ValueError("worker_count must be bounded to 1..2")
        for name, maximum in (("max_jobs", 8), ("max_episodes", 2), ("max_attempts", 4)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be bounded to 1..{maximum}")
        if self.max_attempts > self.max_jobs:
            raise ValueError("max_attempts cannot exceed max_jobs")
        _validate_hf_subfolder(self.hf_subfolder, validation_mode=True)
        if not self.hf_subfolder.startswith("validation/validation-cpu-20260922/"):
            raise ValueError(
                "validation plan hf_subfolder must remain under "
                "validation/validation-cpu-20260922/"
            )
        names = tuple(gate.name for gate in self.gates)
        if names != _VALIDATION_GATE_NAMES:
            raise ValueError("validation plan must name the required gates exactly once")
        if any(type(value) is not bool for value in self.fault_injection.values()):
            raise ValueError("fault_injection values must be booleans")
        allowed_faults = frozenset(_VALIDATION_GATE_NAMES) - {"publication", "success", "valid_failure", "invalid_observation"}
        if set(self.fault_injection) - allowed_faults:
            raise ValueError("fault_injection contains an unsupported gate")

    @property
    def gate_names(self) -> tuple[str, ...]:
        return tuple(gate.name for gate in self.gates)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationPlan":
        values = _mapping(payload, "validation plan")
        allowed = frozenset({
            "plan_id", "validation_mode", "worker_count", "max_jobs",
            "max_episodes", "max_attempts", "hf_subfolder", "gates",
            "fault_injection",
        })
        _reject_unknown(values, allowed, "validation plan")
        _require_fields(values, allowed, "validation plan")
        gates = values["gates"]
        if not isinstance(gates, list):
            raise ValueError("validation plan gates must be a list")
        fault_injection = values["fault_injection"]
        if not isinstance(fault_injection, Mapping):
            raise ValueError("fault_injection must be an object")
        return cls(
            plan_id=str(values["plan_id"]),
            validation_mode=values["validation_mode"],
            worker_count=values["worker_count"],
            max_jobs=values["max_jobs"],
            max_episodes=values["max_episodes"],
            max_attempts=values["max_attempts"],
            hf_subfolder=str(values["hf_subfolder"]),
            gates=tuple(ValidationGateSpec.from_dict(item) for item in gates),
            fault_injection=dict(fault_injection),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "validation_mode": self.validation_mode,
            "worker_count": self.worker_count,
            "max_jobs": self.max_jobs,
            "max_episodes": self.max_episodes,
            "max_attempts": self.max_attempts,
            "hf_subfolder": self.hf_subfolder,
            "gates": [{"name": gate.name, "required": gate.required} for gate in self.gates],
            "fault_injection": dict(self.fault_injection),
        }


@dataclass(frozen=True)
class ValidationGateResult:
    status: str
    reason: str
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in _VALIDATION_STATUSES:
            raise ValueError(f"unsupported validation gate status: {self.status}")
        if not isinstance(self.reason, str):
            raise ValueError("validation gate reason must be a string")
        if not isinstance(self.evidence, dict):
            raise ValueError("validation gate evidence must be an object")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReceipt:
    plan_id: str
    status: str
    gates: dict[str, ValidationGateResult]
    created_at_s: float | None = None
    updated_at_s: float | None = None
    runtime: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALIDATION_STATUSES:
            raise ValueError(f"unsupported validation receipt status: {self.status}")
        if not isinstance(self.gates, dict):
            raise ValueError("validation receipt gates must be an object")
        if any(not isinstance(value, ValidationGateResult) for value in self.gates.values()):
            raise ValueError("validation receipt gates must contain ValidationGateResult values")

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": 1,
            "plan_id": self.plan_id,
            "status": self.status,
            "gates": {name: value.as_dict() for name, value in self.gates.items()},
            "created_at_s": self.created_at_s,
            "updated_at_s": self.updated_at_s,
            "runtime": dict(self.runtime or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationReceipt":
        values = _mapping(payload, "validation receipt")
        _reject_unknown(
            values,
            frozenset({"receipt_version", "plan_id", "status", "gates", "created_at_s", "updated_at_s", "runtime"}),
            "validation receipt",
        )
        gates = values.get("gates")
        if not isinstance(gates, Mapping):
            raise ValueError("validation receipt gates must be an object")
        parsed_gates: dict[str, ValidationGateResult] = {}
        for name, item in gates.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"validation receipt gate must be an object: {name}")
            parsed_gates[str(name)] = ValidationGateResult(
                status=str(item.get("status")),
                reason=item.get("reason", ""),
                evidence=dict(item.get("evidence") or {}),
            )
        return cls(
            plan_id=str(values["plan_id"]),
            status=str(values["status"]),
            gates=parsed_gates,
            created_at_s=values.get("created_at_s"),
            updated_at_s=values.get("updated_at_s"),
            runtime=dict(values.get("runtime") or {}),
        )


__all__ = [
    "ValidationGateResult",
    "ValidationGateSpec",
    "ValidationPlan",
    "ValidationReceipt",
    "GenerationJob",
    "GenerationRuntimeConfig",
    "CoordinatorHeartbeat",
    "MachineConfig",
    "QueueCounts",
    "RunConfig",
    "RuntimeRunConfig",
    "WorkerResult",
]
