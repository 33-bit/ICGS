"""Frozen, JSON-shaped configuration for the planned ICGS method.

This is deliberately separate from the native :mod:`ExperimentConfig`.  It
validates method metadata before any component or artifact is constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


NATIVE_ACTION_HORIZON = 8
NATIVE_DEMO_WAYPOINTS = 10
PRIMARY_HORIZON = 512
KNOWN_NATIVE_PROFILES = frozenset(
    {
        "instant-policy-original-65dc94e",
        "instant_policy_original",
        "instant_policy_published_vv19_119fa871",
    }
)
KNOWN_COMPONENT_IDS = {
    "geometry": "geometry",
    "event": "event",
    "memory": "memory",
    "dynamics": "dynamics",
    "evaluator": "evaluator",
    "planning": "planning",
    "control": "control",
    "dataset": "dataset",
    "training": "training",
}


def _json_value(value: Any, name: str) -> None:
    """Reject values that cannot have come from strict JSON."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} has a non-string JSON key")
            _json_value(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_value(item, f"{name}[{index}]")
        return
    raise ValueError(f"{name} must contain JSON values, not executable objects")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _component_id(value: Any, section: str) -> str:
    expected = KNOWN_COMPONENT_IDS[section]
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"unknown component ID for {section}: {value!r}")
    return value


def _merge_known(base: dict[str, Any], update: Any, prefix: str) -> None:
    if not isinstance(update, Mapping):
        raise ValueError(f"{prefix or 'config'} must be an object")
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge_known(base[key], value, f"{prefix}{key}.")
        else:
            _json_value(value, f"{prefix}{key}")
            base[key] = value


def _resolve_paths(config: dict[str, Any], directory: Path) -> None:
    for section, key in (("dataset", "manifest_path"), ("training", "output_dir")):
        value = config[section][key]
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"{section}.{key} must be a nonempty path or null")
        path = Path(value).expanduser()
        config[section][key] = str(path if path.is_absolute() else (directory / path).resolve())


def _optional_path(value: Any, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a nonempty path or null")


def _section(cls: type, payload: Any, name: str):
    if payload is None:
        return cls()
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    values = {item.name: getattr(cls(), item.name) for item in fields(cls)}
    values.update(payload)
    return cls(**values)


@dataclass(frozen=True)
class GeometryConfig:
    component_id: str = "geometry"
    voxel_size_m: float = 0.005
    num_anchors: int = 128
    width: int = 256
    point_dim: int = 3
    neighbors: int = 32
    ell0_m: float = 1.0


@dataclass(frozen=True)
class EventConfig:
    component_id: str = "event"
    width: int = 256
    max_interactions: int = 30
    landmarks: int = 2


@dataclass(frozen=True)
class MemoryConfig:
    component_id: str = "memory"
    width: int = 256
    slots: int = 2
    descriptor_dim: int = 8


@dataclass(frozen=True)
class DynamicsConfig:
    component_id: str = "dynamics"
    width: int = 256
    heads: int = 3
    H: int = PRIMARY_HORIZON


@dataclass(frozen=True)
class EvaluatorConfig:
    component_id: str = "evaluator"
    width: int = 256
    heads: int = 3
    H: int = PRIMARY_HORIZON


@dataclass(frozen=True)
class PlanningConfig:
    component_id: str = "planning"
    h: int = 2
    r: int = 2
    L: int = 32
    H: int = PRIMARY_HORIZON


@dataclass(frozen=True)
class ControlConfig:
    component_id: str = "control"
    dt0: float = 0.1


@dataclass(frozen=True)
class DatasetConfig:
    component_id: str = "dataset"
    schema_id: str = "icgs_episode_v1"
    shard_intervals: int = 256
    manifest_path: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    component_id: str = "training"
    H: int = PRIMARY_HORIZON
    heads: int = 3
    output_dir: str | None = None


@dataclass(frozen=True)
class MethodConfig:
    schema_version: int = 1
    native_profile: str = "instant-policy-original-65dc94e"
    geometry: GeometryConfig = GeometryConfig()
    event: EventConfig = EventConfig()
    memory: MemoryConfig = MemoryConfig()
    dynamics: DynamicsConfig = DynamicsConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    planning: PlanningConfig = PlanningConfig()
    control: ControlConfig = ControlConfig()
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()

    @classmethod
    def from_dict(cls, payload: dict) -> "MethodConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("method config must be a JSON object")
        _json_value(payload, "method config")
        section_types = {
            "geometry": GeometryConfig,
            "event": EventConfig,
            "memory": MemoryConfig,
            "dynamics": DynamicsConfig,
            "evaluator": EvaluatorConfig,
            "planning": PlanningConfig,
            "control": ControlConfig,
            "dataset": DatasetConfig,
            "training": TrainingConfig,
        }
        allowed = {"schema_version", "native_profile", *section_types}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown method config fields: {sorted(unknown)}")
        values: dict[str, Any] = {
            "schema_version": payload.get("schema_version", 1),
            "native_profile": payload.get("native_profile", cls().native_profile),
        }
        for name, section_type in section_types.items():
            values[name] = _section(section_type, payload.get(name), name)
        return cls(**values).validate()

    @classmethod
    def from_file(cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> "MethodConfig":
        json_path = Path(path).expanduser().resolve()
        try:
            document = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid method config file: {json_path}") from exc
        if not isinstance(document, Mapping):
            raise ValueError("method config document must be an object")
        _json_value(document, "method config document")
        unknown = set(document) - {"schema_version", "config"}
        if unknown:
            raise ValueError(f"unknown method config document fields: {sorted(unknown)}")
        if document.get("schema_version", 1) != 1 or isinstance(document.get("schema_version", 1), bool):
            raise ValueError("unsupported method config schema")
        merged = cls().to_dict()
        _merge_known(merged, document.get("config", {}), "config.")
        for dotted, value in (overrides or {}).items():
            if not isinstance(dotted, str) or not dotted:
                raise ValueError("CLI override keys must be nonempty dotted strings")
            update: dict[str, Any] = value
            for bit in reversed(dotted.split(".")):
                update = {bit: update}
            _merge_known(merged, update, "")
        _resolve_paths(merged, json_path.parent)
        return cls.from_dict(merged)

    def validate(self) -> "MethodConfig":
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported method config schema_version")
        if not isinstance(self.native_profile, str) or self.native_profile not in KNOWN_NATIVE_PROFILES:
            raise ValueError(f"unknown native profile: {self.native_profile!r}")

        geometry = self.geometry
        if type(geometry) is not GeometryConfig:
            raise ValueError("geometry must be GeometryConfig")
        _component_id(geometry.component_id, "geometry")
        if geometry.num_anchors != 128 or geometry.width != 256 or geometry.point_dim != 3:
            raise ValueError("geometry width/anchors/point dimension are incompatible")
        if geometry.neighbors != 32:
            raise ValueError("geometry neighbor count is incompatible")
        _finite_positive(geometry.voxel_size_m, "geometry.voxel_size_m")
        _finite_positive(geometry.ell0_m, "geometry.ell0_m")
        if geometry.voxel_size_m != 0.005 or geometry.ell0_m != 1.0:
            raise ValueError("geometry preprocessing scale is incompatible")

        event = self.event
        if type(event) is not EventConfig or event.width != 256:
            raise ValueError("event width is incompatible")
        _component_id(event.component_id, "event")
        _positive_int(event.max_interactions, "event.max_interactions")
        if event.max_interactions > 30 or event.landmarks != 2:
            raise ValueError("event interaction/landmark inventory is incompatible")

        memory = self.memory
        if type(memory) is not MemoryConfig or memory.width != 256 or memory.slots != 2:
            raise ValueError("memory width/slot inventory is incompatible")
        _component_id(memory.component_id, "memory")
        if memory.descriptor_dim != 8:
            raise ValueError("memory descriptor width is incompatible")

        for name, section, section_type in (("dynamics", self.dynamics, DynamicsConfig),
                                            ("evaluator", self.evaluator, EvaluatorConfig)):
            if type(section) is not section_type:
                raise ValueError(f"{name} section is invalid")
            _component_id(section.component_id, name)
            if section.width != 256 or section.heads != 3:
                raise ValueError(f"{name} width/head inventory is incompatible")
            if not 1 <= _positive_int(section.H, f"{name}.H") <= PRIMARY_HORIZON:
                raise ValueError(f"{name} horizon coverage is incompatible")

        planning = self.planning
        if type(planning) is not PlanningConfig:
            raise ValueError("planning section is invalid")
        _component_id(planning.component_id, "planning")
        for name, value in (("planning.h", planning.h), ("planning.r", planning.r),
                            ("planning.L", planning.L), ("planning.H", planning.H)):
            _positive_int(value, name)
        if not 1 <= planning.r <= planning.h <= NATIVE_ACTION_HORIZON:
            raise ValueError("commit/edge horizon is incompatible")
        if planning.L < planning.h or planning.H > PRIMARY_HORIZON:
            raise ValueError("planning horizon/lookahead is incompatible")
        if planning.H > self.dynamics.H or planning.H > self.evaluator.H or planning.H > self.training.H:
            raise ValueError("planning horizon exceeds trained horizon coverage")

        if type(self.control) is not ControlConfig:
            raise ValueError("control section is invalid")
        _component_id(self.control.component_id, "control")
        _finite_positive(self.control.dt0, "control.dt0")
        if type(self.dataset) is not DatasetConfig:
            raise ValueError("dataset section is invalid")
        _component_id(self.dataset.component_id, "dataset")
        if self.dataset.schema_id != "icgs_episode_v1":
            raise ValueError("unknown dataset schema")
        _positive_int(self.dataset.shard_intervals, "dataset.shard_intervals")
        _optional_path(self.dataset.manifest_path, "dataset.manifest_path")
        if type(self.training) is not TrainingConfig:
            raise ValueError("training section is invalid")
        _component_id(self.training.component_id, "training")
        _optional_path(self.training.output_dir, "training.output_dir")
        if self.training.heads != 3 or not 1 <= _positive_int(self.training.H, "training.H") <= PRIMARY_HORIZON:
            raise ValueError("training head/horizon inventory is incompatible")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-shaped metadata; no executable serialization is used."""
        self.validate()
        return asdict(self)


__all__ = [
    "ControlConfig", "DatasetConfig", "DynamicsConfig", "EventConfig",
    "EvaluatorConfig", "GeometryConfig", "MemoryConfig", "MethodConfig",
    "PlanningConfig", "TrainingConfig",
]
