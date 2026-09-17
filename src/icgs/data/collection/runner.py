"""Bounded collection runner with preflight validation, resource caps, and clean lifecycle management.

Composes collect_attempt, episode archive persistence, quarantine, and explicit
injected factories with finite attempt, interval, wall, and disk limits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import TimedCommand
from icgs.data.archives import (
    StorageLimitExceeded,
    _json_safe,
    check_and_write_bytes,
    quarantine_attempt,
)
from icgs.data.collection.attempts import (
    AttemptExecutionError,
    AttemptResult,
    collect_attempt,
    persist_attempt,
)
from icgs.data.collection.programs import program_catalog


def _get_dir_size(path: Path) -> int:
    """Calculate total size in bytes of all regular files in a directory tree."""
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = Path(root) / f
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                pass
    return total


@dataclass(frozen=True)
class CollectionLimits:
    """Explicit, finite resource bounds for a collection run."""

    max_attempts: int
    max_intervals: int
    max_wall_time_s: float
    max_disk_bytes: int

    def validate(self) -> None:
        if (
            self.max_attempts is None
            or isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError(f"max_attempts must be an explicit positive integer, got {self.max_attempts!r}")
        if (
            self.max_intervals is None
            or isinstance(self.max_intervals, bool)
            or not isinstance(self.max_intervals, int)
            or self.max_intervals < 1
        ):
            raise ValueError(f"max_intervals must be an explicit positive integer, got {self.max_intervals!r}")
        if (
            self.max_wall_time_s is None
            or isinstance(self.max_wall_time_s, bool)
            or not isinstance(self.max_wall_time_s, (int, float))
            or self.max_wall_time_s <= 0
            or not math.isfinite(self.max_wall_time_s)
        ):
            raise ValueError(f"max_wall_time_s must be an explicit positive finite float, got {self.max_wall_time_s!r}")
        if (
            self.max_disk_bytes is None
            or isinstance(self.max_disk_bytes, bool)
            or not isinstance(self.max_disk_bytes, int)
            or self.max_disk_bytes < 1
        ):
            raise ValueError(f"max_disk_bytes must be an explicit positive integer, got {self.max_disk_bytes!r}")


def _validate_spec_metadata(val: Any) -> None:
    from icgs.data.archives import _json_safe

    if isinstance(val, (bytes, bytearray)):
        raise TypeError(f"bytes and bytearray not supported in metadata: {type(val).__name__}")
    if isinstance(val, Mapping):
        for k, v in val.items():
            if not isinstance(k, str):
                raise TypeError(f"metadata keys must be str, got {type(k).__name__}")
            if k in ("attempt_status", "annotations", "attempt_id", "status", "transition_count", "provenance"):
                raise ValueError(f"metadata contains reserved key: {k!r}")
            _validate_spec_metadata(v)
    elif isinstance(val, (list, tuple)):
        for item in val:
            _validate_spec_metadata(item)
    else:
        _json_safe(val)


@dataclass(frozen=True)
class AttemptSpec:
    """Explicit parameters specifying a single collection attempt."""

    episode_id: str
    program_id: str
    source_lineage_id: str
    asset_family_id: str
    split: str
    generator_seed: int
    reset_seed: int
    action_seed: int
    calibration_id: str
    observation_origin: str
    raw_commands_id: str
    materialized_commands_id: str
    commands: Sequence[TimedCommand] | None = None
    metadata: Mapping[str, Any] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for s_name, s_val in (
            ("generator_seed", self.generator_seed),
            ("reset_seed", self.reset_seed),
            ("action_seed", self.action_seed),
        ):
            if s_val is None or isinstance(s_val, bool) or not isinstance(s_val, int):
                raise ValueError(f"{s_name} must be an explicit integer, got {s_val!r}")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError(f"seed must be an integer, got {self.seed!r}")


@dataclass(frozen=True)
class MaterializedCommands:
    """Typed result from a command factory carrying commands and bound provenance identities."""

    commands: Sequence[TimedCommand]
    raw_commands_id: str
    materialized_commands_id: str
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw_commands_id, str) or not self.raw_commands_id.strip():
            raise ValueError(f"raw_commands_id must be a non-empty string, got {self.raw_commands_id!r}")
        if not isinstance(self.materialized_commands_id, str) or not self.materialized_commands_id.strip():
            raise ValueError(f"materialized_commands_id must be a non-empty string, got {self.materialized_commands_id!r}")
        if not isinstance(self.commands, Sequence):
            raise TypeError(f"commands must be a Sequence of TimedCommand, got {type(self.commands).__name__}")


@dataclass(frozen=True)
class CollectionReport:
    """Summary of a completed or bounded collection execution."""

    status: str
    attempts_requested: int
    attempts_executed: int
    episodes_published: int
    reports_written: int
    quarantined_count: int
    total_intervals: int
    elapsed_time_s: float
    disk_bytes_used: int
    attempt_records: tuple[dict[str, Any], ...]
    provenance: Mapping[str, Any] | None = None


def run_collection(
    specs: Sequence[AttemptSpec],
    environment_factory: Callable[[AttemptSpec], Any],
    *,
    dataset_root: str | Path,
    dataset_manifest_path: str | Path,
    limits: CollectionLimits,
    config: MethodConfig,
    protocol_manifest: Mapping[str, Any],
    code_revision: str,
    is_dirty: bool,
    generator_version: str,
    dirty_patch_digest: str | None = None,
    command_factory: Callable[[AttemptSpec, Any], Iterable[TimedCommand] | MaterializedCommands] | None = None,
    monitor_factory: Callable[[AttemptSpec, Any], Any] | None = None,
    online_provider: Callable[[Any], Mapping[str, Any]] | None = None,
    episode_publisher: Callable[[str], Mapping[str, Any]] | None = None,
    index_publisher: Callable[[str | Path], Mapping[str, Any]] | None = None,
    dataset_track: str = "primary",
    clock: Callable[[], float] = time.monotonic,
) -> CollectionReport:
    """Execute bounded episode collection over explicit attempt specifications.

    Preflights all identities, paths, split assignments, protocol manifests,
    and resource caps before constructing any environment.
    """
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence) or len(specs) == 0:
        raise ValueError("specs must be a nonempty sequence of AttemptSpec")

    if config is None or not hasattr(config, "validate"):
        raise TypeError("config must be an explicit MethodConfig instance")
    config.validate()

    if not isinstance(limits, CollectionLimits):
        raise TypeError(f"limits must be a CollectionLimits instance, got {type(limits).__name__}")
    limits.validate()

    # Preflight checks before environment_factory
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("code_revision must be an explicit non-empty string")
    if code_revision.strip().upper() == "HEAD":
        raise ValueError("code_revision cannot fabricate 'HEAD'; explicit commit or version required")

    if type(is_dirty) is not bool:
        raise TypeError(f"is_dirty must be an explicit bool, got {type(is_dirty).__name__}")

    if is_dirty:
        if not isinstance(dirty_patch_digest, str) or not dirty_patch_digest.strip():
            raise ValueError("dirty_patch_digest is required when is_dirty is True")

    if not isinstance(generator_version, str) or not generator_version.strip():
        raise ValueError("generator_version must be an explicit non-empty string")

    if not isinstance(protocol_manifest, Mapping):
        raise TypeError(f"protocol_manifest must be a Mapping, got {type(protocol_manifest).__name__}")

    required_protocol_identities = (
        "environment_protocol_id",
        "controller_protocol_id",
        "sensor_protocol_id",
        "asset_protocol_id",
        "split_protocol_id",
        "collection_protocol_id",
    )
    for pkey in required_protocol_identities:
        pval = protocol_manifest.get(pkey)
        if not isinstance(pval, str) or not pval.strip():
            raise ValueError(f"protocol_manifest missing required non-empty identity: {pkey!r}")

    proto_meta = protocol_manifest.get("metadata")
    if not isinstance(proto_meta, Mapping) or len(proto_meta) == 0:
        raise ValueError("protocol_manifest requires non-empty metadata mapping")
    for mk, mv in proto_meta.items():
        if isinstance(mv, (float, int)) and not isinstance(mv, bool):
            if not math.isfinite(mv):
                raise ValueError(f"protocol_manifest metadata key {mk!r} has non-finite value: {mv}")
    if dataset_track not in ("primary", "exploratory"):
        raise ValueError("dataset_track must be 'primary' or 'exploratory'")

    root_path = Path(dataset_root).resolve()
    manifest_path = Path(dataset_manifest_path).resolve()

    # Preflight dataset manifest
    from icgs.data.datasets.episodes import validate_dataset_manifest
    manifest_data = validate_dataset_manifest(manifest_path, dataset_root=root_path)

    # Lineage and asset family lookup
    valid_lineages = {entry["lineage_id"]: entry["split"] for entry in manifest_data.get("lineage", ())}
    valid_asset_fams = {entry["asset_family_id"]: entry["split"] for entry in manifest_data.get("asset_families", ())}
    catalog = program_catalog()

    # Compute stable run provenance
    cfg_dict = getattr(config, "to_dict", None)
    cfg_content = cfg_dict() if callable(cfg_dict) else str(config)
    config_digest = hashlib.sha256(
        json.dumps(cfg_content, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    run_provenance: dict[str, Any] = {
        "generator_version": generator_version,
        "code_revision": code_revision,
        "is_dirty": is_dirty,
        "dirty_patch_digest": dirty_patch_digest,
        "config_digest": config_digest,
        "protocol_manifest": _json_safe(dict(protocol_manifest)),
        "protocol_identities": {
            "environment_protocol_id": protocol_manifest["environment_protocol_id"],
            "controller_protocol_id": protocol_manifest["controller_protocol_id"],
            "sensor_protocol_id": protocol_manifest["sensor_protocol_id"],
            "asset_protocol_id": protocol_manifest["asset_protocol_id"],
            "split_protocol_id": protocol_manifest["split_protocol_id"],
            "collection_protocol_id": protocol_manifest["collection_protocol_id"],
        },
    }

    # Preflight specs before constructing any environment
    seen_ids: set[str] = set()
    episodes_dir = root_path / "episodes"
    for idx, spec in enumerate(specs):
        if not isinstance(spec, AttemptSpec):
            raise TypeError(f"spec {idx} must be an AttemptSpec, got {type(spec).__name__}")
        if (
            not isinstance(spec.episode_id, str)
            or not spec.episode_id.strip()
            or Path(spec.episode_id).is_absolute()
            or spec.episode_id in (".", "..")
            or "/" in spec.episode_id
            or "\\" in spec.episode_id
        ):
            raise ValueError(f"spec {idx} has invalid episode_id: {spec.episode_id!r}")
        if spec.episode_id in seen_ids:
            raise ValueError(f"Duplicate episode_id in specs: {spec.episode_id}")
        seen_ids.add(spec.episode_id)

        # Preflight metadata recursively (Issue 5)
        if spec.metadata is not None:
            _validate_spec_metadata(spec.metadata)

        # Check provenance fields are explicit non-empty strings
        for field_name in ("calibration_id", "observation_origin", "raw_commands_id", "materialized_commands_id"):
            val = getattr(spec, field_name, None)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"spec {idx} {field_name} must be a non-empty string, got {val!r}")

        # Check generator, reset, and action seeds are non-negative integers
        for s_name, s_val in (
            ("generator_seed", spec.generator_seed),
            ("reset_seed", spec.reset_seed),
            ("action_seed", spec.action_seed),
        ):
            if not isinstance(s_val, int) or isinstance(s_val, bool) or s_val < 0:
                raise ValueError(f"spec {idx} {s_name} must be a non-negative integer, got {s_val!r}")

        # Check collision with already published episode
        if (episodes_dir / spec.episode_id).exists():
            raise FileExistsError(f"Episode {spec.episode_id} already exists in dataset root")

        # Program catalog check
        if spec.program_id not in catalog:
            raise ValueError(f"spec {idx} program_id {spec.program_id!r} not in program catalog")
        cat_prog = catalog[spec.program_id]
        expected_split = "dev" if cat_prog.split == "development" else cat_prog.split
        if spec.split != expected_split:
            raise ValueError(f"spec {idx} program {spec.program_id} catalog split {expected_split} != spec split {spec.split}")

        # Program and track mixing guard
        if dataset_track == "exploratory":
            if not spec.program_id.startswith("E"):
                raise ValueError(
                    f"spec {idx} program_id {spec.program_id!r} is not an exploratory program; exploratory track only allows E-series programs"
                )
            if spec.split != "dev":
                raise ValueError(
                    f"spec {idx} split {spec.split!r} must be 'dev' for exploratory track"
                )
        elif dataset_track == "primary":
            if spec.program_id.startswith("E"):
                raise ValueError(
                    f"spec {idx} program_id {spec.program_id!r} is an exploratory program; primary track forbids E-series programs"
                )

        # Lineage split check
        if spec.source_lineage_id not in valid_lineages:
            raise ValueError(f"spec {idx} source_lineage_id {spec.source_lineage_id!r} not in manifest lineage")
        if valid_lineages[spec.source_lineage_id] != spec.split:
            raise ValueError(f"spec {idx} lineage split {valid_lineages[spec.source_lineage_id]} != spec split {spec.split}")

        # Asset family check
        if spec.asset_family_id not in valid_asset_fams:
            raise ValueError(f"spec {idx} asset_family_id {spec.asset_family_id!r} not in manifest asset families")
        if valid_asset_fams[spec.asset_family_id] != spec.split:
            raise ValueError(f"spec {idx} asset family split {valid_asset_fams[spec.asset_family_id]} != spec split {spec.split}")

        # Commands check
        if spec.commands is None and command_factory is None:
            raise ValueError(f"spec {idx} has no commands and no command_factory provided")

    if dataset_track == "exploratory":
        if episode_publisher is None:
            raise ValueError("Exploratory track requires an explicit episode_publisher")
        if index_publisher is None:
            raise ValueError("Exploratory track requires an explicit index_publisher")

    start_time = clock()
    attempts_executed = 0
    episodes_published = 0
    reports_written = 0
    quarantined_count = 0
    total_intervals = 0
    status = "completed"
    attempt_records: list[dict[str, Any]] = []

    target_manifest_path = Path(manifest_path).resolve() if manifest_path is not None else root_path / "dataset_manifest.json"
    manifest_data: dict[str, Any] = {}
    if target_manifest_path.is_file():
        with open(target_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    else:
        manifest_data = {
            "manifest_version": 1,
            "episodes": [],
            "lineage": [
                {"lineage_id": lid, "split": sp, "parent_ids": []}
                for lid, sp in valid_lineages.items()
            ],
            "asset_families": [
                {"asset_family_id": afid, "split": sp}
                for afid, sp in valid_asset_fams.items()
            ],
        }

    # Preflight initial disk cap exhaustion (Sol Fix 1)
    initial_disk = _get_dir_size(root_path)
    if limits.max_disk_bytes is not None and initial_disk >= limits.max_disk_bytes:
        return CollectionReport(
            status="disk_limit",
            attempts_requested=len(specs),
            attempts_executed=0,
            episodes_published=0,
            reports_written=0,
            quarantined_count=0,
            total_intervals=0,
            elapsed_time_s=clock() - start_time,
            disk_bytes_used=initial_disk,
            attempt_records=(),
            provenance=run_provenance,
        )

    published_specs: list[AttemptSpec] = []
    last_verified_manifest_bytes: bytes | None = (
        target_manifest_path.read_bytes() if target_manifest_path.is_file() else None
    )

    def _build_candidate_manifest_bytes(
        candidate_entry: dict[str, Any] | None = None,
        candidate_published_specs: Sequence[AttemptSpec] | None = None,
    ) -> bytes:
        eps = [
            e for e in manifest_data.get("episodes", [])
            if isinstance(e, Mapping)
        ]
        if candidate_entry is not None:
            eps = [e for e in eps if e.get("episode_id") != candidate_entry.get("episode_id")]
            eps.append(candidate_entry)
        specs_for_prov = candidate_published_specs if candidate_published_specs is not None else published_specs
        candidate_prov = {
            "generator_version": generator_version,
            "code_revision": code_revision,
            "is_dirty": is_dirty,
            "dirty_patch_digest": dirty_patch_digest,
            "generator_seeds": [s.generator_seed for s in specs_for_prov],
            "reset_seeds": [s.reset_seed for s in specs_for_prov],
            "action_seeds": [s.action_seed for s in specs_for_prov],
            "protocol_manifest": _json_safe(dict(protocol_manifest)),
            "config_digest": config_digest,
            "protocol_identities": {
                "environment_protocol_id": protocol_manifest["environment_protocol_id"],
                "controller_protocol_id": protocol_manifest["controller_protocol_id"],
                "sensor_protocol_id": protocol_manifest["sensor_protocol_id"],
                "asset_protocol_id": protocol_manifest["asset_protocol_id"],
                "split_protocol_id": protocol_manifest["split_protocol_id"],
                "collection_protocol_id": protocol_manifest["collection_protocol_id"],
            },
        }
        m_copy = dict(manifest_data)
        m_copy["dataset_track"] = dataset_track
        m_copy["episodes"] = eps
        m_copy["provenance"] = candidate_prov
        return json.dumps(m_copy, indent=2, allow_nan=False).encode("utf-8")

    for spec in specs:
        # Check limits before starting next attempt
        if limits.max_attempts is not None and attempts_executed >= limits.max_attempts:
            status = "attempt_limit"
            break
        remaining_intervals = limits.max_intervals - total_intervals if limits.max_intervals is not None else None
        if remaining_intervals is not None and remaining_intervals <= 0:
            status = "interval_limit"
            break
        elapsed = clock() - start_time
        remaining_wall_s = limits.max_wall_time_s - elapsed if limits.max_wall_time_s is not None else None
        if remaining_wall_s is not None and remaining_wall_s <= 0:
            status = "wall_limit"
            break

        current_disk = _get_dir_size(root_path)
        if limits.max_disk_bytes is not None and current_disk >= limits.max_disk_bytes:
            status = "disk_limit"
            break

        # Compute per-attempt bound enforcing remaining interval and wall budgets (HIGH4)
        cfg_max_intervals = config.collection.max_episode_intervals
        attempt_intervals = min(cfg_max_intervals, remaining_intervals) if remaining_intervals is not None else cfg_max_intervals

        cfg_wall_s = config.collection.wall_limit_s
        if remaining_wall_s is not None:
            attempt_wall_s = min(cfg_wall_s, remaining_wall_s) if cfg_wall_s is not None else remaining_wall_s
        else:
            attempt_wall_s = cfg_wall_s

        attempt_config = MethodConfig(
            dataset=config.dataset,
            stages=config.stages,
            collection={
                "max_episode_intervals": attempt_intervals,
                "wall_limit_s": attempt_wall_s,
                "disk_limit_bytes": limits.max_disk_bytes,
            },
        )

        prov = {
            "episode_id": spec.episode_id,
            "program_id": spec.program_id,
            "source_lineage_id": spec.source_lineage_id,
            "asset_family_id": spec.asset_family_id,
            "split": spec.split,
            "calibration_id": spec.calibration_id,
            "observation_origin": spec.observation_origin,
            "raw_commands_id": spec.raw_commands_id,
            "materialized_commands_id": spec.materialized_commands_id,
        }

        attempt_metadata = dict(spec.metadata) if spec.metadata is not None else {}
        attempt_metadata["run_provenance"] = {
            "generator_seed": spec.generator_seed,
            "reset_seed": spec.reset_seed,
            "action_seed": spec.action_seed,
            "generator_version": generator_version,
            "code_revision": code_revision,
            "is_dirty": is_dirty,
            "dirty_patch_digest": dirty_patch_digest,
            "environment_protocol_id": protocol_manifest["environment_protocol_id"],
            "controller_protocol_id": protocol_manifest["controller_protocol_id"],
            "sensor_protocol_id": protocol_manifest["sensor_protocol_id"],
            "asset_protocol_id": protocol_manifest["asset_protocol_id"],
            "split_protocol_id": protocol_manifest["split_protocol_id"],
            "collection_protocol_id": protocol_manifest["collection_protocol_id"],
        }

        # Environment construction with single lifecycle ownership (Issue 2)
        env = None
        try:
            env = environment_factory(spec)
        except Exception as env_err:
            attempts_executed += 1
            status = "failed"
            attempt_records.append({
                "episode_id": spec.episode_id,
                "status": "failed",
                "transitions": 0,
                "error": str(env_err),
            })
            break

        env_closed = False
        def _close_env() -> BaseException | None:
            nonlocal env_closed
            if not env_closed and env is not None:
                env_closed = True
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except BaseException as c_err:
                        return c_err
            return None

        # Factories execution with exact-once close on error (Issue 2)
        mat_res: MaterializedCommands | None = None
        try:
            if spec.commands is not None:
                commands = spec.commands
            elif command_factory is not None:
                res = command_factory(spec, env)  # type: ignore[misc]
                if isinstance(res, MaterializedCommands):
                    mat_res = res
                    commands = res.commands
                else:
                    commands = res
            else:
                commands = ()

            monitor = monitor_factory(spec, env) if monitor_factory is not None else None
        except Exception as factory_err:
            attempts_executed += 1
            status = "failed"
            c_err = _close_env()
            from icgs.data.archives import build_quarantine_evidence
            q_evidence = build_quarantine_evidence()
            diagnostics = {
                "primary_error_type": type(factory_err).__name__,
                "primary_error_message": str(factory_err),
                "cleanup_error": (
                    {"error_type": type(c_err).__name__, "message": str(c_err)}
                    if c_err is not None else None
                ),
            }
            current_disk = _get_dir_size(root_path)
            remaining_for_q = limits.max_disk_bytes - current_disk if limits.max_disk_bytes is not None else None
            q_written = False
            q_path_str = None
            try:
                if remaining_for_q is not None and remaining_for_q <= 0:
                    raise StorageLimitExceeded(f"No disk allowance remaining for quarantine ({remaining_for_q} bytes)")
                q_path = quarantine_attempt(
                    root_path,
                    spec.episode_id,
                    diagnostics=diagnostics,
                    evidence=q_evidence,
                    staging_byte_cap=remaining_for_q,
                )
                quarantined_count += 1
                q_written = True
                q_path_str = str(q_path)
            except (StorageLimitExceeded, OSError, ValueError):
                pass

            if q_written:
                attempt_records.append({
                    "episode_id": spec.episode_id,
                    "status": "quarantined",
                    "transitions": 0,
                    "output_path": q_path_str,
                })
            else:
                attempt_records.append({
                    "episode_id": spec.episode_id,
                    "status": "quarantine_unwritten",
                    "transitions": 0,
                    "output_path": None,
                    "primary_error": diagnostics.get("primary_error_message"),
                    "unsaved_diagnostics": diagnostics,
                    "unsaved_evidence": q_evidence,
                    "retention_reason": "disk_limit_exceeded",
                })
                status = "disk_limit"
            break

        if mat_res is not None:
            prov["raw_commands_id"] = mat_res.raw_commands_id
            prov["materialized_commands_id"] = mat_res.materialized_commands_id
            if mat_res.metadata:
                attempt_metadata.update(mat_res.metadata)
        else:
            if dataset_track == "exploratory":
                if command_factory is not None:
                    _close_env()
                    raise RuntimeError(
                        f"spec {spec.episode_id}: exploratory track requires command_factory to return a concrete MaterializedCommands instance with bound raw/materialized identities"
                    )
                if spec.raw_commands_id in ("rlbench-live-expert-v1", "") or spec.materialized_commands_id.startswith("rlbench-live-mat-"):
                    _close_env()
                    raise ValueError(
                        f"spec {spec.episode_id}: exploratory track requires concrete bound raw_commands_id and materialized_commands_id, got placeholders"
                    )

        # Attempt execution (Issue 1)
        try:
            attempt_res = collect_attempt(
                env,
                commands,
                config=attempt_config,
                monitor=monitor,
                seed=spec.reset_seed,
                clock=clock,
                online_provider=online_provider,
            )
            env_closed = True
        except AttemptExecutionError as exec_err:
            env_closed = True
            attempts_executed += 1
            total_intervals += len(exec_err.transitions)
            from icgs.data.archives import build_quarantine_evidence
            q_evidence = build_quarantine_evidence(
                transitions=exec_err.transitions,
                annotations=exec_err.annotations,
                rejected_transition=exec_err.rejected_transition,
                initial_observation=exec_err.initial_observation,
            )
            diagnostics = {
                "primary_error_type": type(exec_err.cause).__name__ if exec_err.cause else type(exec_err).__name__,
                "primary_error_message": str(exec_err.cause) if exec_err.cause else str(exec_err),
                "cleanup_error": (
                    {"error_type": type(exec_err.cleanup_error).__name__, "message": str(exec_err.cleanup_error)}
                    if exec_err.cleanup_error else None
                ),
            }
            current_disk = _get_dir_size(root_path)
            remaining_for_q = limits.max_disk_bytes - current_disk if limits.max_disk_bytes is not None else None
            q_written = False
            q_path_str = None
            try:
                if remaining_for_q is not None and remaining_for_q <= 0:
                    raise StorageLimitExceeded(f"No disk allowance remaining for quarantine ({remaining_for_q} bytes)")
                q_path = quarantine_attempt(
                    root_path,
                    spec.episode_id,
                    diagnostics=diagnostics,
                    evidence=q_evidence,
                    staging_byte_cap=remaining_for_q,
                )
                quarantined_count += 1
                q_written = True
                q_path_str = str(q_path)
            except (StorageLimitExceeded, OSError, ValueError):
                pass

            if q_written:
                attempt_records.append({
                    "episode_id": spec.episode_id,
                    "status": "quarantined",
                    "transitions": len(exec_err.transitions),
                    "output_path": q_path_str,
                })
                status = "failed"
            else:
                attempt_records.append({
                    "episode_id": spec.episode_id,
                    "status": "quarantine_unwritten",
                    "transitions": len(exec_err.transitions),
                    "output_path": None,
                    "primary_error": diagnostics.get("primary_error_message"),
                    "unsaved_diagnostics": diagnostics,
                    "unsaved_evidence": q_evidence,
                    "retention_reason": "disk_limit_exceeded",
                })
                status = "disk_limit"
            break
        finally:
            _close_env()

        # Account attempted work (Issue 1)
        attempts_executed += 1
        total_intervals += len(attempt_res.transitions)

        # Check disk budget before persist (Exact pre-write sizing)
        current_disk = _get_dir_size(root_path)
        if limits.max_disk_bytes is not None:
            if len(attempt_res.transitions) > 0:
                dummy_entry = {
                    "episode_id": spec.episode_id,
                    "manifest_path": f"episodes/{spec.episode_id}/manifest.json",
                    "sha256": "0" * 64,
                    "split": spec.split,
                    "program_id": spec.program_id,
                    "source_lineage_id": spec.source_lineage_id,
                    "asset_family_id": spec.asset_family_id,
                }
                needed_for_index = len(_build_candidate_manifest_bytes(dummy_entry))
            else:
                needed_for_index = 0

            remaining_for_persist = limits.max_disk_bytes - current_disk - needed_for_index
            if remaining_for_persist <= 0:
                status = "disk_limit"
                break
        else:
            remaining_for_persist = None

        try:
            out_path = persist_attempt(
                root_path,
                attempt_res,
                provenance=prov,
                config=config,
                metadata=attempt_metadata,
                staging_byte_cap=remaining_for_persist,
                online_provider=online_provider,
            )
        except (StorageLimitExceeded, OSError, ValueError) as err:
            if isinstance(err, StorageLimitExceeded) or "staging_byte_cap" in str(err) or "staging cap" in str(err) or "exceeds" in str(err):
                status = "disk_limit"
                break
            raise

        if len(attempt_res.transitions) == 0:
            reports_written += 1
        else:
            # Publication is part of the episode close boundary. A failed or
            # unverified Drive/HF transfer stops the run before any index entry.
            publication_report: Mapping[str, Any] | None = None
            if episode_publisher is not None:
                try:
                    publication_report = episode_publisher(spec.episode_id)
                except Exception as publish_err:
                    status = "publication_failed"
                    attempt_records.append({
                        "episode_id": spec.episode_id,
                        "status": status,
                        "transitions": len(attempt_res.transitions),
                        "output_path": str(out_path),
                        "error": str(publish_err),
                    })
                    break
                if (
                    not isinstance(publication_report, Mapping)
                    or publication_report.get("drive_verified") is not True
                    or publication_report.get("hf_verified") is not True
                ):
                    status = "publication_unverified"
                    attempt_records.append({
                        "episode_id": spec.episode_id,
                        "status": status,
                        "transitions": len(attempt_res.transitions),
                        "output_path": str(out_path),
                        "publication": _json_safe(dict(publication_report or {})),
                    })
                    break

            ep_manifest_file = Path(out_path)
            hasher = hashlib.sha256()
            with open(ep_manifest_file, "rb") as mf:
                while chunk := mf.read(65536):
                    hasher.update(chunk)
            ep_sha = hasher.hexdigest()
            try:
                rel_path = str(ep_manifest_file.relative_to(root_path))
            except ValueError:
                rel_path = f"episodes/{spec.episode_id}/manifest.json"

            ep_entry = {
                "episode_id": spec.episode_id,
                "manifest_path": rel_path,
                "sha256": ep_sha,
                "split": spec.split,
                "program_id": spec.program_id,
                "source_lineage_id": spec.source_lineage_id,
                "asset_family_id": spec.asset_family_id,
            }
            # Serialize updated manifest and check peak root + new tmp bytes before atomic write
            candidate_specs = published_specs + [spec]
            candidate_episodes = [
                e for e in manifest_data.get("episodes", [])
                if isinstance(e, Mapping) and e.get("episode_id") != spec.episode_id
            ] + [ep_entry]
            updated_manifest_bytes = _build_candidate_manifest_bytes(
                ep_entry,
                candidate_published_specs=candidate_specs,
            )
            current_disk_now = _get_dir_size(root_path)
            remaining_for_index_tmp = limits.max_disk_bytes - current_disk_now if limits.max_disk_bytes is not None else None
            try:
                check_and_write_bytes(
                    target_manifest_path,
                    updated_manifest_bytes,
                    remaining_bytes=remaining_for_index_tmp,
                    atomic=True,
                )
            except StorageLimitExceeded:
                status = "disk_limit"
                break

            # Per-episode index publication before next physical attempt
            if index_publisher is not None:
                try:
                    index_report = index_publisher(target_manifest_path)
                except Exception as index_err:
                    status = "index_publication_failed"
                    if last_verified_manifest_bytes is not None:
                        target_manifest_path.write_bytes(last_verified_manifest_bytes)
                    elif target_manifest_path.exists():
                        target_manifest_path.unlink()
                    attempt_records.append({
                        "episode_id": spec.episode_id,
                        "status": status,
                        "transitions": len(attempt_res.transitions),
                        "output_path": str(out_path),
                        "error": str(index_err),
                    })
                    break
                if (
                    not isinstance(index_report, Mapping)
                    or index_report.get("drive_verified") is not True
                    or index_report.get("hf_verified") is not True
                ):
                    status = "index_publication_unverified"
                    if last_verified_manifest_bytes is not None:
                        target_manifest_path.write_bytes(last_verified_manifest_bytes)
                    elif target_manifest_path.exists():
                        target_manifest_path.unlink()
                    attempt_records.append({
                        "episode_id": spec.episode_id,
                        "status": status,
                        "transitions": len(attempt_res.transitions),
                        "output_path": str(out_path),
                        "publication": _json_safe(dict(index_report or {})),
                    })
                    break

            last_verified_manifest_bytes = updated_manifest_bytes
            manifest_data["episodes"] = candidate_episodes
            published_specs.append(spec)
            episodes_published += 1

        attempt_records.append({
            "episode_id": spec.episode_id,
            "status": attempt_res.status,
            "transitions": len(attempt_res.transitions),
            "output_path": str(out_path),
            "publication": _json_safe(dict(publication_report or {})) if len(attempt_res.transitions) > 0 else None,
        })

    # Atomically publish/update dataset manifest linking published episodes and run provenance
    if index_publisher is not None:
        if last_verified_manifest_bytes is not None:
            if not target_manifest_path.is_file() or target_manifest_path.read_bytes() != last_verified_manifest_bytes:
                target_manifest_path.write_bytes(last_verified_manifest_bytes)
    elif episodes_published > 0 or not target_manifest_path.is_file():
        final_manifest_bytes = _build_candidate_manifest_bytes(candidate_published_specs=published_specs)
        current_disk_now = _get_dir_size(root_path)
        remaining_for_index = limits.max_disk_bytes - current_disk_now if limits.max_disk_bytes is not None else None
        try:
            check_and_write_bytes(
                target_manifest_path,
                final_manifest_bytes,
                remaining_bytes=remaining_for_index,
                atomic=True,
            )
        except StorageLimitExceeded:
            status = "disk_limit"

    final_disk = _get_dir_size(root_path)
    total_elapsed = clock() - start_time

    return CollectionReport(
        status=status,
        attempts_requested=len(specs),
        attempts_executed=attempts_executed,
        episodes_published=episodes_published,
        reports_written=reports_written,
        quarantined_count=quarantined_count,
        total_intervals=total_intervals,
        elapsed_time_s=total_elapsed,
        disk_bytes_used=final_disk,
        attempt_records=tuple(attempt_records),
        provenance=run_provenance,
    )


def reconcile_episode(
    dataset_manifest_path: str | Path,
    episode_manifest_path: str | Path,
    *,
    config: MethodConfig | None = None,
) -> Path:
    """Reconcile an existing published episode into the dataset manifest without rescanning.

    Verifies checksum and schema of the episode manifest, checks for conflicting
    episode_id, validates proposed manifest before write, and atomically updates
    the dataset manifest through check_and_write_bytes.
    """
    raw_ds_path = Path(dataset_manifest_path)
    if raw_ds_path.is_symlink():
        raise ValueError(f"Dataset manifest path cannot be a symlink: {dataset_manifest_path}")
    ds_path = raw_ds_path.resolve()
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {ds_path}")

    raw_ep_path = Path(episode_manifest_path)
    if raw_ep_path.is_symlink():
        raise ValueError(f"Episode manifest path cannot be a symlink: {episode_manifest_path}")

    from icgs.data.archives import StorageLimitExceeded, check_and_write_bytes, read_episode_archive
    from icgs.data.datasets.episodes import validate_dataset_manifest

    # Validate existing dataset manifest
    manifest_data = validate_dataset_manifest(ds_path, dataset_root=ds_path.parent)
    dataset_root = ds_path.parent.resolve()

    # Reject episode paths outside dataset root (do not fabricate fallback)
    try:
        if raw_ep_path.is_absolute():
            rel_path = raw_ep_path.relative_to(dataset_root)
        else:
            rel_path = raw_ep_path
    except ValueError:
        raise ValueError(f"Episode manifest path {episode_manifest_path} is outside dataset root {dataset_root}")

    # Check for symlinks in path components before resolve per existing validator
    curr_chk = dataset_root
    for part in rel_path.parts:
        curr_chk = curr_chk / part
        if curr_chk.is_symlink():
            raise ValueError(f"Episode manifest path component is a symlink: {curr_chk}")

    ep_path = curr_chk.resolve()
    if not ep_path.is_file():
        raise FileNotFoundError(f"Episode manifest not found: {ep_path}")

    if dataset_root != ep_path and dataset_root not in ep_path.parents:
        raise ValueError(f"Episode manifest {ep_path} is outside dataset root {dataset_root}")

    # Validate episode archive
    ep_record = read_episode_archive(ep_path)
    prov = ep_record["provenance"]
    ep_id = prov["episode_id"]

    # Refuse conflict / duplicate
    existing_ids = {e["episode_id"] for e in manifest_data.get("episodes", []) if isinstance(e, Mapping)}
    if ep_id in existing_ids:
        raise ValueError(f"Episode {ep_id} already exists in dataset manifest")

    # Compute sha256 of episode manifest
    hasher = hashlib.sha256()
    with open(ep_path, "rb") as mf:
        while chunk := mf.read(65536):
            hasher.update(chunk)
    ep_sha = hasher.hexdigest()

    ep_entry = {
        "episode_id": ep_id,
        "manifest_path": str(rel_path),
        "sha256": ep_sha,
        "split": prov["split"],
        "program_id": prov["program_id"],
        "source_lineage_id": prov["source_lineage_id"],
        "asset_family_id": prov["asset_family_id"],
    }

    # Construct proposed mapping
    proposed = dict(manifest_data)
    proposed["episodes"] = list(manifest_data.get("episodes", [])) + [ep_entry]

    # Validate proposed mapping BEFORE ANY index write
    validate_dataset_manifest(proposed, dataset_root=dataset_root)

    # Route final replacement through check_and_write_bytes with actual current-root allowance
    new_manifest_bytes = json.dumps(proposed, indent=2, allow_nan=False).encode("utf-8")
    if config is not None and config.collection.disk_limit_bytes is not None:
        cur_disk = _get_dir_size(dataset_root)
        rem_bytes = config.collection.disk_limit_bytes - cur_disk
    else:
        rem_bytes = None

    check_and_write_bytes(
        ds_path,
        new_manifest_bytes,
        remaining_bytes=rem_bytes,
        atomic=True,
    )
    return ds_path


__all__ = [
    "AttemptSpec",
    "CollectionLimits",
    "CollectionReport",
    "MaterializedCommands",
    "reconcile_episode",
    "run_collection",
]
