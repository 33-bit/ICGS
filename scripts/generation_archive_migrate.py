#!/usr/bin/env python3
"""Migrate one explicitly selected immutable v2 artifact into the lossless archive.

The local API is fixture-friendly. The HF command requires a pinned commit OID,
source/target prefixes, program ID, and one episode or attempt ID. It verifies the
complete v2 artifact inventory before conversion and publishes only to the exact
new target paths supplied by the migration identity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from icgs.data.collection.generation.distributed_contracts import ArchiveProfileConfig
from icgs.data.collection.generation.episode_archive import (
    EpisodeArchiveReader,
    EpisodeArchiveWriter,
    _canonical_json,
    _npz_uncompressed_member_bytes,
    _redact_sensitive,
    validate_archive_manifest,
)
from icgs.data.schemas.episode_records import validate_episode
from icgs.data.training_layout import validate_training_episode_layout


MIGRATION_RECEIPT_VERSION = "icgs_generation_migration_receipt_v1"
_COMMIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_COPY_BLOCK_BYTES = 1 << 20
_DEFAULT_MAX_DECODED_SIDECAR_BYTES = 1024**3


class MigrationError(RuntimeError):
    """A fail-closed migration error with the path to its local failure receipt."""

    def __init__(self, message: str, *, failure_receipt_path: Path | None = None) -> None:
        super().__init__(message)
        self.failure_receipt_path = failure_receipt_path


@dataclass(frozen=True)
class MigrationReceipt:
    """Integrity and fidelity record for one v2-to-archive conversion."""

    source_identity: Mapping[str, Any]
    target_identity: Mapping[str, Any]
    source_artifact_hashes: Mapping[str, Mapping[str, Any]]
    target_archive_hashes: Mapping[str, Mapping[str, Any]]
    field_counts: Mapping[str, int]
    timeline_counts: Mapping[str, int]
    conversion_warnings: tuple[str, ...]
    status: str = "complete"
    receipt_version: str = MIGRATION_RECEIPT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "status": self.status,
            "source_identity": dict(self.source_identity),
            "target_identity": dict(self.target_identity),
            "source_artifact_hashes": {
                key: dict(value) for key, value in sorted(self.source_artifact_hashes.items())
            },
            "target_archive_hashes": {
                key: dict(value) for key, value in sorted(self.target_archive_hashes.items())
            },
            "field_counts": dict(self.field_counts),
            "timeline_counts": dict(self.timeline_counts),
            "conversion_warnings": list(self.conversion_warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MigrationReceipt":
        if payload.get("receipt_version") != MIGRATION_RECEIPT_VERSION:
            raise ValueError("migration receipt version mismatch")
        if payload.get("status") != "complete":
            raise ValueError("existing migration receipt is not complete")
        warnings = payload.get("conversion_warnings")
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise ValueError("migration receipt conversion_warnings must be a list of strings")
        return cls(
            source_identity=_require_mapping(payload.get("source_identity"), "source_identity"),
            target_identity=_require_mapping(payload.get("target_identity"), "target_identity"),
            source_artifact_hashes=_require_mapping(
                payload.get("source_artifact_hashes"), "source_artifact_hashes"
            ),
            target_archive_hashes=_require_mapping(
                payload.get("target_archive_hashes"), "target_archive_hashes"
            ),
            field_counts=_int_mapping(payload.get("field_counts"), "field_counts"),
            timeline_counts=_int_mapping(payload.get("timeline_counts"), "timeline_counts"),
            conversion_warnings=tuple(warnings),
        )


@dataclass(frozen=True)
class MigrationTargetWriter:
    """Bind the canonical archive writer to one new local staging destination.

    The wrapper owns paths and receipt placement only. Episode and attempt
    serialization always goes through ``EpisodeArchiveWriter``.
    """

    archive_writer: EpisodeArchiveWriter
    output_dir: Path
    receipt_path: Path
    failure_receipt_path: Path

    def __post_init__(self) -> None:
        output = Path(self.output_dir).absolute()
        receipt = Path(self.receipt_path).absolute()
        failure = Path(self.failure_receipt_path).absolute()
        if output == receipt or output in receipt.parents:
            raise ValueError("migration receipt must be outside the archive directory")
        if output == failure or output in failure.parents:
            raise ValueError("failure receipt must be outside the archive directory")
        if receipt == failure:
            raise ValueError("success and failure receipts require separate paths")
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(self, "receipt_path", receipt)
        object.__setattr__(self, "failure_receipt_path", failure)


@dataclass(frozen=True)
class _SourceRecord:
    kind: str
    record: Mapping[str, Any]
    prefix_arrays: Mapping[str, np.ndarray]
    raw_arrays: Mapping[str, Any]
    execution: Mapping[str, Any]
    sidecar_metadata: Mapping[str, Any]
    source_artifact_hashes: Mapping[str, Mapping[str, Any]]
    field_counts: Mapping[str, int]
    timeline_counts: Mapping[str, int]
    warnings: tuple[str, ...]
    artifact_manifest: Mapping[str, Any]


class V2ArtifactReader:
    """Read and validate one local copy of one immutable ``icgs_episode_v2`` artifact."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        max_decoded_sidecar_bytes: int = _DEFAULT_MAX_DECODED_SIDECAR_BYTES,
    ) -> None:
        if type(max_decoded_sidecar_bytes) is not int or max_decoded_sidecar_bytes <= 0:
            raise ValueError("max_decoded_sidecar_bytes must be a positive integer")
        self.root = Path(artifact_dir).absolute()
        self.max_decoded_sidecar_bytes = max_decoded_sidecar_bytes
        self._loaded: _SourceRecord | None = None
        self._partial_hashes: dict[str, dict[str, Any]] = {}
        self._declared_hashes: dict[str, dict[str, Any]] = {}

    @property
    def source_artifact_hashes(self) -> Mapping[str, Mapping[str, Any]]:
        if self._loaded is not None:
            return self._loaded.source_artifact_hashes
        return dict(self._partial_hashes)

    @property
    def declared_artifact_hashes(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._declared_hashes)

    def read(self) -> _SourceRecord:
        if self._loaded is not None:
            return self._loaded
        root = self.root
        if root.is_symlink() or not root.is_dir():
            raise ValueError("v2 artifact root must be a real directory")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"v2 artifact must not contain symlinks: {path.relative_to(root)}")
        artifact_path = root / "artifact_manifest.json"
        if not artifact_path.is_file():
            raise ValueError("v2 artifact is missing artifact_manifest.json")
        artifact_manifest = _read_json(artifact_path, name="artifact_manifest.json")
        files = artifact_manifest.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError("v2 artifact_manifest.json must contain a nonempty files inventory")
        expected: dict[str, dict[str, Any]] = {}
        for relative, info in files.items():
            relative = _safe_relative_path(relative, "artifact inventory path")
            if relative == "artifact_manifest.json":
                raise ValueError("v2 artifact inventory must not list artifact_manifest.json itself")
            info = _require_mapping(info, f"artifact inventory entry {relative}")
            digest = info.get("sha256")
            size = info.get("bytes")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"v2 artifact inventory has an invalid SHA256 for {relative}")
            if type(size) is not int or size < 0:
                raise ValueError(f"v2 artifact inventory has an invalid byte count for {relative}")
            expected[relative] = {"sha256": digest, "bytes": size}

        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root).as_posix() != "artifact_manifest.json"
        }
        if actual_paths != set(expected):
            missing = sorted(set(expected) - actual_paths)
            extra = sorted(actual_paths - set(expected))
            raise ValueError(f"v2 artifact inventory is incomplete (missing={missing}, extra={extra})")
        self._declared_hashes = dict(expected)
        for relative, item in expected.items():
            path = root.joinpath(*PurePosixPath(relative).parts)
            actual = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            self._partial_hashes[relative] = actual
            if actual != item:
                raise ValueError(f"v2 source artifact hash/size mismatch: {relative}")
        manifest_bytes = artifact_path.read_bytes()
        self._partial_hashes["artifact_manifest.json"] = {
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "bytes": len(manifest_bytes),
        }
        _preflight_decoded_sidecars(
            root,
            actual_paths,
            max_decoded_sidecar_bytes=self.max_decoded_sidecar_bytes,
        )

        episode_path, attempt_path = root / "episode.json", root / "attempt.json"
        if episode_path.is_file() == attempt_path.is_file():
            raise ValueError("v2 artifact must contain exactly one of episode.json or attempt.json")
        kind = "episode" if episode_path.is_file() else "attempt"
        record_path = episode_path if kind == "episode" else attempt_path
        record = _read_json(record_path, name=record_path.name)
        if kind == "episode":
            validate_episode(record)
        else:
            _validate_v2_attempt(record)

        warnings: list[str] = []
        raw_arrays: dict[str, Any] = {}
        prefix_arrays: dict[str, np.ndarray] = {}
        sidecar_metadata: dict[str, Any] = {}
        execution: Mapping[str, Any] = {}
        candidates: dict[str, list[tuple[str, np.ndarray]]] = {}

        layout_root = root / "layout"
        layout_manifest: Mapping[str, Any] | None = None
        if layout_root.exists():
            layout_manifest = _validate_layout_inventory(layout_root)
            _validate_layout_identity(layout_root, layout_manifest, record, kind)
        elif kind == "episode":
            warnings.append("typed_layout_sidecars_unavailable")

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in {"artifact_manifest.json", "episode.json", "attempt.json"}:
                continue
            if relative == "execution.json":
                value = _read_json(path, name=relative)
                execution = _require_mapping(value, "execution.json")
                continue
            if relative == "valid_prefix.npz":
                prefix_arrays = _load_npz(path, relative)
                continue
            if relative == "telemetry.npz":
                telemetry = _load_npz(path, relative)
                for name, value in telemetry.items():
                    _add_raw_array(raw_arrays, _array_key("telemetry", name), value)
                    semantic = {
                        "ee_pose": "online_poses",
                        "gripper": "online_grips",
                        "achieved_dt": "dt",
                    }.get(name)
                    if semantic:
                        candidates.setdefault(semantic, []).append((relative + ":" + name, value))
                continue
            if relative == "layout/layout_manifest.json":
                sidecar_metadata[relative] = dict(layout_manifest or {})
                continue
            if relative.startswith("layout/"):
                _read_layout_sidecar(
                    path,
                    relative,
                    raw_arrays=raw_arrays,
                    sidecar_metadata=sidecar_metadata,
                    candidates=candidates,
                )
                continue
            if path.suffix == ".npz":
                for name, value in _load_npz(path, relative).items():
                    _add_raw_array(raw_arrays, _array_key("source", relative, name), value)
                continue
            if path.suffix == ".npy":
                value = _load_npy(path, relative)
                _add_raw_array(raw_arrays, _array_key("source", relative), value)
                continue
            if path.suffix == ".json":
                sidecar_metadata[relative] = _read_json(path, name=relative)
                continue
            if path.suffix == ".jsonl":
                lines = path.read_text(encoding="utf-8").splitlines()
                sidecar_metadata[relative] = [
                    _strict_json(line, f"{relative} line {index + 1}")
                    for index, line in enumerate(lines)
                    if line.strip()
                ]
                if any(not line.strip() for line in lines):
                    raise ValueError(f"blank JSONL row in v2 sidecar: {relative}")
                continue
            _add_raw_array(raw_arrays, _array_key("source_bytes", relative), np.fromfile(path, dtype=np.uint8))

        typed_paths: set[str] = set()
        if kind == "episode":
            typed_paths = _apply_typed_sidecars(record, candidates, warnings)
            validate_episode(record)
            if "online_points" not in candidates:
                warnings.append("dtype_ambiguous:online_observations.points (JSON arrays have no dtype metadata)")
            if "online_poses" not in candidates:
                warnings.append("dtype_ambiguous:online_observations.T_w_e (JSON arrays have no dtype metadata)")
            if "online_grips" not in candidates:
                warnings.append("dtype_ambiguous:online_observations.grip (JSON numbers have no dtype metadata)")
            warnings.extend(_numeric_dtype_warnings(record, typed_paths))
        elif not prefix_arrays:
            warnings.append("attempt_has_no_typed_valid_prefix_arrays")

        _validate_artifact_identity(artifact_manifest, record, kind)
        if kind == "episode":
            field_counts = _episode_field_counts(record, execution, raw_arrays, sidecar_metadata, warnings)
            timeline_counts = {
                "observations": len(record["online_observations"]),
                "transitions": len(record["transitions"]),
            }
        else:
            observations, transitions = _attempt_timeline(record, prefix_arrays)
            field_counts = {
                "record_top_level": len(record),
                "attempt_fields": len(record),
                "execution_fields": len(execution),
                "prefix_array_fields": len(prefix_arrays),
                "sidecar_metadata_files": len(sidecar_metadata),
                "raw_array_fields": len(raw_arrays),
                "numeric_dtype_ambiguities": 0,
            }
            timeline_counts = {"observations": observations, "transitions": transitions}

        self._loaded = _SourceRecord(
            kind=kind,
            record=record,
            prefix_arrays=prefix_arrays,
            raw_arrays=raw_arrays,
            execution=execution,
            sidecar_metadata=sidecar_metadata,
            source_artifact_hashes=dict(self._partial_hashes),
            field_counts=field_counts,
            timeline_counts=timeline_counts,
            warnings=tuple(sorted(set(warnings))),
            artifact_manifest=artifact_manifest,
        )
        return self._loaded


def migrate_episode(
    source_reader: V2ArtifactReader,
    target_writer: MigrationTargetWriter,
    *,
    source_identity: Mapping[str, Any],
    target_profile: ArchiveProfileConfig,
) -> MigrationReceipt:
    """Convert exactly one validated v2 episode or attempt into a canonical archive.

    ``target_writer`` binds an ``EpisodeArchiveWriter`` to one new local target
    directory. The supplied source identity contains the immutable HF source
    repo/revision/prefix and the separate target repo/prefix paths.
    """

    source_hashes: Mapping[str, Mapping[str, Any]] = {}
    identity_payload: dict[str, Any] = {}
    try:
        if not isinstance(source_reader, V2ArtifactReader):
            raise TypeError("source_reader must be a V2ArtifactReader")
        if not isinstance(target_writer, MigrationTargetWriter):
            raise TypeError("target_writer must be a MigrationTargetWriter")
        if not isinstance(target_profile, ArchiveProfileConfig):
            raise TypeError("target_profile must be an ArchiveProfileConfig")
        if target_writer.archive_writer.profile != target_profile:
            raise ValueError("target writer profile disagrees with target_profile")
        identity_payload = _validate_identity(source_identity)
        source = source_reader.read()
        source_hashes = source.source_artifact_hashes
        _validate_source_identity(identity_payload, source)
        source_record = _redact_sensitive(dict(source.record))
        if not _semantic_equal(source.record, source_record):
            raise ValueError("security redaction would alter v2 semantics; refusing non-lossless migration")
        target_identity = _target_identity(identity_payload, target_profile)

        if target_writer.output_dir.exists() or target_writer.receipt_path.exists():
            return _reuse_existing_migration(
                source,
                target_writer,
                source_identity=identity_payload,
                target_identity=target_identity,
            )

        warnings = list(source.warnings)
        safe_execution = _redact_sensitive(dict(source.execution))
        safe_sidecars = _redact_sensitive(dict(source.sidecar_metadata))
        identity_values, identity_warnings = _required_archive_identity(
            identity_payload,
            source.record,
            source.execution,
        )
        warnings.extend(identity_warnings)

        migration_metadata = {
            "source_identity": identity_payload,
            "target_identity": target_identity,
            "original_identity_fields_unavailable": [
                name for name in ("source_run_id", "code_revision")
                if identity_values["original_values"].get(name) is None
            ],
            "original_identity_values": dict(identity_values["original_values"]),
            "identity_placeholders": dict(identity_values["placeholders"]),
            "source_artifact_manifest_sha256": source.source_artifact_hashes["artifact_manifest.json"]["sha256"],
        }
        debug_metadata = {
            **identity_values["archive_identity"],
            "v2_migration": migration_metadata,
            "source_execution": safe_execution,
            "source_sidecars": safe_sidecars,
            "source_artifact_manifest": _redact_sensitive(dict(source.artifact_manifest)),
        }
        if source.kind == "episode":
            target_writer.archive_writer.write_episode(
                source_record,
                raw_arrays=source.raw_arrays,
                debug_metadata=debug_metadata,
                output_dir=target_writer.output_dir,
            )
        else:
            overlap = set(source.prefix_arrays) & set(source.raw_arrays)
            if overlap:
                raise ValueError(f"attempt prefix/sidecar array names collide: {sorted(overlap)}")
            target_writer.archive_writer.write_attempt(
                source_record,
                prefix_arrays={**source.prefix_arrays, **source.raw_arrays},
                debug_metadata=debug_metadata,
                output_dir=target_writer.output_dir,
            )

        roundtrip = _validate_and_read_target(target_writer.output_dir, source.kind)
        _verify_source_roundtrip(
            source, roundtrip, source_record, safe_execution, safe_sidecars,
            identity_payload, target_identity,
        )
        target_hashes = _file_inventory(target_writer.output_dir)
        receipt = MigrationReceipt(
            source_identity=identity_payload,
            target_identity=target_identity,
            source_artifact_hashes=source.source_artifact_hashes,
            target_archive_hashes=target_hashes,
            field_counts={
                **dict(source.field_counts),
                "target_archive_array_fields": len(roundtrip.manifest.payload["array_specs"]),
                "conversion_warning_count": len(set(warnings)),
            },
            timeline_counts=dict(roundtrip.manifest.payload["timeline"] | {
                key: value for key, value in source.timeline_counts.items()
            }),
            conversion_warnings=tuple(sorted(set(warnings))),
        )
        _write_exclusive_json(target_writer.receipt_path, receipt.as_dict())
        return receipt
    except Exception as error:
        failure_path = getattr(target_writer, "failure_receipt_path", None)
        failure_payload = _failure_receipt(
            source_identity=identity_payload or _safe_partial_identity(source_identity),
            source_artifact_hashes=source_hashes or source_reader.source_artifact_hashes if isinstance(source_reader, V2ArtifactReader) else {},
            declared_source_artifact_hashes=(
                source_reader.declared_artifact_hashes if isinstance(source_reader, V2ArtifactReader) else {}
            ),
            target_profile=target_profile.as_dict() if isinstance(target_profile, ArchiveProfileConfig) else None,
            error=error,
        )
        if failure_path is not None:
            try:
                _write_exclusive_json(Path(failure_path), failure_payload)
            except Exception as receipt_error:
                raise MigrationError(
                    f"migration failed: {error}; failure receipt could not be written: {receipt_error}",
                    failure_receipt_path=Path(failure_path),
                ) from error
        raise MigrationError(
            f"migration failed: {error}",
            failure_receipt_path=Path(failure_path) if failure_path is not None else None,
        ) from error


def _validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(_require_mapping(value, "source_identity"))
    required = (
        "source_repo_id", "source_revision", "source_prefix", "source_artifact_path",
        "source_kind", "program_id", "record_id", "target_repo_id", "target_prefix",
        "target_artifact_path", "target_receipt_path",
    )
    missing = [name for name in required if not identity.get(name)]
    if missing:
        raise ValueError(f"migration requires explicit source/target identities: missing {missing}")
    revision = identity["source_revision"]
    if not isinstance(revision, str) or not _COMMIT_OID.fullmatch(revision.lower()):
        raise ValueError("source_revision must be a full 40- or 64-character immutable HF commit OID")
    identity["source_revision"] = revision.lower()
    for field in ("source_repo_id", "target_repo_id"):
        if not isinstance(identity[field], str) or not identity[field].strip():
            raise ValueError(f"{field} must be a nonblank Hugging Face dataset repo ID")
    for field in ("source_kind",):
        if identity[field] not in {"episode", "attempt"}:
            raise ValueError("source_kind must be episode or attempt")
    for field in ("program_id", "record_id"):
        if not isinstance(identity[field], str) or not _SAFE_ID.fullmatch(identity[field]) or identity[field] in {".", ".."}:
            raise ValueError(f"{field} must be a safe single path component")
    for field in ("source_prefix", "target_prefix"):
        identity[field] = _prefix(identity[field], field)
    _ensure_disjoint_prefixes(identity["source_prefix"], identity["target_prefix"])
    expected_record_path = _artifact_path(
        identity["source_prefix"], identity["source_kind"], identity["program_id"], identity["record_id"]
    )
    expected_target_path = _artifact_path(
        identity["target_prefix"], identity["source_kind"], identity["program_id"], identity["record_id"]
    )
    if _safe_repo_path(identity["source_artifact_path"], "source_artifact_path") != expected_record_path:
        raise ValueError("source_artifact_path must identify the explicitly selected v2 record under source_prefix")
    if _safe_repo_path(identity["target_artifact_path"], "target_artifact_path") != expected_target_path:
        raise ValueError("target_artifact_path must identify the selected record under the separate target_prefix")
    receipt_path = _safe_repo_path(identity["target_receipt_path"], "target_receipt_path")
    if not receipt_path.startswith(identity["target_prefix"] + "/"):
        raise ValueError("target_receipt_path must be inside the explicit target_prefix")
    if receipt_path.startswith(expected_target_path + "/"):
        raise ValueError("target migration receipt must be outside the archive directory")
    return identity


def _safe_partial_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}
    for key in (
        "source_repo_id", "source_revision", "source_prefix", "source_artifact_path",
        "source_kind", "program_id", "record_id", "target_repo_id", "target_prefix",
        "target_artifact_path", "target_receipt_path",
    ):
        item = value.get(key)
        if isinstance(item, (str, int, bool)) or item is None:
            output[key] = item
    return output


def _validate_source_identity(identity: Mapping[str, Any], source: _SourceRecord) -> None:
    if source.kind != identity["source_kind"]:
        raise ValueError("source artifact kind differs from the explicitly selected source_kind")
    record_id = source.record.get("provenance", {}).get("episode_id") if source.kind == "episode" else source.record.get("attempt_id")
    program_id = source.record.get("provenance", {}).get("program_id") if source.kind == "episode" else source.record.get("program_id")
    if record_id != identity["record_id"] or program_id != identity["program_id"]:
        raise ValueError("source artifact identity differs from the explicitly selected record path")


def _target_identity(identity: Mapping[str, Any], profile: ArchiveProfileConfig) -> dict[str, Any]:
    target = {
        "repo_id": identity["target_repo_id"],
        "prefix": identity["target_prefix"],
        "artifact_path": identity["target_artifact_path"],
        "receipt_path": identity["target_receipt_path"],
        "archive_profile": profile.as_dict(),
    }
    if "target_branch" in identity:
        target["branch"] = identity["target_branch"]
    return target


def _required_archive_identity(
    identity: Mapping[str, Any],
    record: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    provenance = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
    original_values: dict[str, Any] = {}
    archive_identity: dict[str, str] = {}
    placeholders: dict[str, str] = {}
    warnings: list[str] = []
    for field in ("source_run_id", "code_revision"):
        found: Any = None
        found_present = False
        for mapping in (execution, record, provenance):
            if field in mapping:
                found_present = True
                value = mapping[field]
                if isinstance(value, str) and value.strip():
                    found = value
                    break
                if value is not None:
                    raise ValueError(f"source identity field {field} must be a nonblank string when present")
        original_values[field] = found
        if found is not None:
            archive_identity[field] = found
            continue
        if field == "source_run_id":
            stable = json.dumps(
                [identity["source_repo_id"], identity["source_prefix"], identity["source_revision"]],
                separators=(",", ":"),
            ).encode("utf-8")
            placeholder = "v2-migration:" + hashlib.sha256(stable).hexdigest()[:24]
            archive_identity[field] = placeholder
            placeholders[field] = placeholder
            warnings.append("source_run_id_unavailable:migration_namespace_used")
        else:
            placeholder = "unknown-v2-generator-revision"
            archive_identity[field] = placeholder
            placeholders[field] = placeholder
            warnings.append("code_revision_unavailable:unknown_v2_generator_revision_used")
        if found_present:
            warnings.append(f"{field}_present_but_unavailable:placeholder_used")
    preprocessing = None
    for mapping in (execution, record, provenance):
        candidate = mapping.get("preprocessing_identity")
        if candidate is not None:
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("preprocessing_identity must be a nonblank string when present")
            preprocessing = candidate
            break
    if preprocessing is None:
        preprocessing = "v2-migration-preserved-measured-values"
        placeholders["preprocessing_identity"] = preprocessing
        warnings.append("preprocessing_identity_unavailable:migration_label_used")
    archive_identity["preprocessing_identity"] = preprocessing
    return {
        "archive_identity": archive_identity,
        "original_values": original_values,
        "placeholders": placeholders,
    }, warnings


def _reuse_existing_migration(
    source: _SourceRecord,
    target: MigrationTargetWriter,
    *,
    source_identity: Mapping[str, Any],
    target_identity: Mapping[str, Any],
) -> MigrationReceipt:
    output = target.output_dir
    receipt_path = target.receipt_path
    if output.is_symlink() or receipt_path.is_symlink():
        raise ValueError("existing migration output or receipt must not be a symlink")
    if not output.is_dir() or not receipt_path.is_file():
        raise FileExistsError("partial migration target exists without both archive and complete receipt")
    receipt = MigrationReceipt.from_dict(_read_json(receipt_path, name="migration receipt"))
    if dict(receipt.source_identity) != dict(source_identity):
        raise FileExistsError("existing migration target is bound to a different immutable v2 source")
    if dict(receipt.target_identity) != dict(target_identity):
        raise FileExistsError("existing migration target is bound to a different target prefix/profile")
    if dict(receipt.source_artifact_hashes) != dict(source.source_artifact_hashes):
        raise ValueError("existing migration receipt source hashes differ from the selected v2 artifact")
    current_hashes = _file_inventory(output)
    if dict(receipt.target_archive_hashes) != current_hashes:
        raise ValueError("existing migration archive hashes differ from its immutable receipt")
    roundtrip = _validate_and_read_target(output, source.kind)
    _, warnings = _required_archive_identity(source_identity, source.record, source.execution)
    safe_record = _redact_sensitive(dict(source.record))
    _verify_source_roundtrip(
        source,
        roundtrip,
        safe_record,
        _redact_sensitive(dict(source.execution)),
        _redact_sensitive(dict(source.sidecar_metadata)),
        source_identity,
        target_identity,
    )
    if not set(source.warnings).issubset(receipt.conversion_warnings) or not set(warnings).issubset(receipt.conversion_warnings):
        raise ValueError("existing migration receipt omits a source conversion warning")
    return receipt


def _validate_and_read_target(root: Path, kind: str) -> EpisodeArchiveReader:
    manifest_name = "episode.manifest.json" if kind == "episode" else "attempt.manifest.json"
    manifest_path = root / manifest_name
    validate_archive_manifest(manifest_path)
    reader = EpisodeArchiveReader(manifest_path)
    if reader.manifest.payload["archive_kind"] != kind:
        raise ValueError("converted archive kind disagrees with the v2 source")
    return reader


def _verify_source_roundtrip(
    source: _SourceRecord,
    reader: EpisodeArchiveReader,
    source_record: Mapping[str, Any],
    safe_execution: Mapping[str, Any],
    safe_sidecars: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    target_identity: Mapping[str, Any],
) -> None:
    restored = reader.to_episode_record()
    if not _semantic_equal(source_record, restored):
        raise ValueError("reconstructed archive record does not match the v2 semantic record")
    if source.kind == "episode":
        validate_episode(restored)
        for name, expected in source.raw_arrays.items():
            actual = reader.raw_arrays.get(name)
            if actual is None or not _semantic_equal(expected, actual):
                raise ValueError(f"reconstructed archive lost source raw array {name}")
    else:
        for name, expected in {**source.prefix_arrays, **source.raw_arrays}.items():
            actual = reader.raw_arrays.get(name)
            if actual is None or not _semantic_equal(expected, actual):
                raise ValueError(f"reconstructed archive lost attempt prefix/sidecar array {name}")
    debug = reader.debug_metadata
    if not _semantic_equal(debug.get("source_execution", {}), safe_execution):
        raise ValueError("reconstructed archive changed unique execution/debug metadata")
    if not _semantic_equal(debug.get("source_sidecars", {}), safe_sidecars):
        raise ValueError("reconstructed archive changed unique v2 sidecar metadata")
    migration = debug.get("v2_migration", {})
    if migration.get("source_identity") != dict(source_identity):
        raise ValueError("reconstructed archive is not bound to the selected immutable v2 source")
    if migration.get("target_identity") != dict(target_identity):
        raise ValueError("reconstructed archive is not bound to the selected target prefix/profile")


def _validate_artifact_identity(manifest: Mapping[str, Any], record: Mapping[str, Any], kind: str) -> None:
    provenance = record.get("provenance", {}) if kind == "episode" else record
    for name in ("program_id", "outcome"):
        if manifest.get(name) != provenance.get(name):
            raise ValueError(f"artifact_manifest {name} disagrees with {kind} record")
    episode_id = provenance.get("episode_id")
    if manifest.get("episode_id") != episode_id:
        raise ValueError("artifact_manifest episode_id disagrees with source record")
    if kind == "episode":
        if manifest.get("attempt_id") is not None and manifest.get("attempt_id") != provenance.get("attempt_id"):
            raise ValueError("artifact_manifest attempt_id disagrees with source episode")
    elif manifest.get("attempt_id") != provenance.get("attempt_id"):
        raise ValueError("artifact_manifest attempt_id disagrees with source attempt")


def _validate_v2_attempt(record: Mapping[str, Any]) -> None:
    if record.get("episode_id") is not None:
        raise ValueError("v2 crash/invalid attempts require episode_id=null")
    if record.get("outcome") not in {"simulator_crash", "invalid_observation"}:
        raise ValueError("v2 attempt outcome must be simulator_crash or invalid_observation")
    for field in ("attempt_id", "program_id"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"v2 attempt is missing required {field}")
    valid_until = record.get("valid_observation_until")
    if valid_until is not None and (type(valid_until) is not int or valid_until < 0):
        raise ValueError("v2 attempt valid_observation_until must be a nonnegative integer or null")


def _validate_layout_inventory(layout_root: Path) -> Mapping[str, Any]:
    manifest_path = layout_root / "layout_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("v2 layout directory is missing layout_manifest.json")
    manifest = _read_json(manifest_path, name="layout/layout_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("v2 layout_manifest.json must contain a nonempty files inventory")
    expected: dict[str, dict[str, Any]] = {}
    for relative, item in files.items():
        relative = _safe_relative_path(relative, "layout inventory path")
        if relative == "layout_manifest.json":
            raise ValueError("layout inventory must not list layout_manifest.json itself")
        item = _require_mapping(item, f"layout inventory entry {relative}")
        digest, size = item.get("sha256"), item.get("bytes")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"layout inventory has an invalid SHA256 for {relative}")
        if type(size) is not int or size < 0:
            raise ValueError(f"layout inventory has an invalid byte count for {relative}")
        expected[relative] = {"sha256": digest, "bytes": size}
    actual = {
        path.relative_to(layout_root).as_posix()
        for path in layout_root.rglob("*")
        if path.is_file() and path.name != "layout_manifest.json"
    }
    if actual != set(expected):
        raise ValueError("v2 layout inventory is incomplete or contains unlisted files")
    for relative, item in expected.items():
        path = layout_root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"v2 layout file is not a regular file: {relative}")
        if path.stat().st_size != item["bytes"] or _sha256_file(path) != item["sha256"]:
            raise ValueError(f"v2 layout sidecar hash/size mismatch: {relative}")
    return manifest


def _validate_layout_identity(
    layout_root: Path,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    kind: str,
) -> None:
    if kind != "episode":
        raise ValueError("attempt artifacts must not carry an episode training layout")
    if manifest.get("layout_version") != 2:
        raise ValueError("only the existing v2 training layout can supply migration dtypes")
    validated = validate_training_episode_layout(layout_root)
    if validated["boundaries"] != len(record["online_observations"]):
        raise ValueError("v2 layout boundary count disagrees with episode.json")
    provenance = record["provenance"]
    if validated["episode_id"] != provenance.get("episode_id") or manifest.get("episode_id") != provenance.get("episode_id"):
        raise ValueError("v2 layout episode identity disagrees with episode.json")
    expected_success = provenance.get("outcome") == "success"
    if validated["success"] is not expected_success or manifest.get("success") is not expected_success:
        raise ValueError("v2 layout success flag disagrees with episode.json")


def _read_layout_sidecar(
    path: Path,
    relative: str,
    *,
    raw_arrays: dict[str, Any],
    sidecar_metadata: dict[str, Any],
    candidates: dict[str, list[tuple[str, np.ndarray]]],
) -> None:
    subpath = relative.removeprefix("layout/")
    if path.suffix == ".npy":
        value = _load_npy(path, relative)
        _add_raw_array(raw_arrays, _array_key("layout", subpath), value)
        if subpath == "robot/ee_pose.npy":
            candidates.setdefault("online_poses", []).append((relative, value))
        elif subpath == "robot/gripper.npy":
            candidates.setdefault("online_grips", []).append((relative, value))
        elif subpath.startswith("task/"):
            label = Path(subpath).stem
            if label in {"rho", "nu", "epsilon", "rho_valid", "nu_valid", "epsilon_valid"}:
                candidates.setdefault("record:" + label, []).append((relative, value))
        return
    if path.suffix == ".npz":
        values = _load_npz(path, relative)
        if subpath == "observations/pointcloud/frames.npz":
            if "points" not in values or "offsets" not in values:
                raise ValueError("v2 pointcloud layout NPZ requires points and offsets arrays")
            for name, value in values.items():
                key = "measured_points" if name == "points" else _array_key("layout", subpath, name)
                _add_raw_array(raw_arrays, key, value)
            candidates.setdefault("online_points", []).append((relative, values["points"]))
            candidates.setdefault("online_point_offsets", []).append((relative, values["offsets"]))
        else:
            for name, value in values.items():
                _add_raw_array(raw_arrays, _array_key("layout", subpath, name), value)
        return
    if path.suffix == ".json":
        sidecar_metadata[relative] = _read_json_value(path, name=relative)
        return
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError(f"blank JSONL row in v2 sidecar: {relative}")
        sidecar_metadata[relative] = [
            _strict_json(line, f"{relative} line {index + 1}") for index, line in enumerate(lines)
        ]
        return
    _add_raw_array(raw_arrays, _array_key("layout_bytes", subpath), np.fromfile(path, dtype=np.uint8))


def _apply_typed_sidecars(
    record: dict[str, Any],
    candidates: Mapping[str, list[tuple[str, np.ndarray]]],
    warnings: list[str],
) -> set[str]:
    observations = record["online_observations"]
    observation_count = len(observations)
    typed_paths: set[str] = set()

    points_candidates = candidates.get("online_points", [])
    offsets_candidates = candidates.get("online_point_offsets", [])
    if points_candidates or offsets_candidates:
        if len(points_candidates) != 1 or len(offsets_candidates) != 1:
            raise ValueError("v2 typed point-cloud sidecar has an incomplete points/offsets pair")
        (points_path, points), (offsets_path, offsets) = points_candidates[0], offsets_candidates[0]
        if points.dtype.kind != "f" or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"typed point sidecar has an invalid schema: {points_path}")
        if offsets.dtype.kind not in "iu" or offsets.shape != (observation_count + 1,):
            raise ValueError(f"typed point offsets have an invalid schema: {offsets_path}")
        offsets64 = offsets.astype(np.int64, copy=False)
        if offsets64[0] != 0 or offsets64[-1] != len(points) or np.any(offsets64[1:] <= offsets64[:-1]):
            raise ValueError("typed point offsets do not cover the v2 observation timeline")
        for index, observation in enumerate(observations):
            frame = points[offsets64[index]:offsets64[index + 1]]
            _require_sidecar_matches(frame, observation["points"], points_path, f"online_observations.{index}.points")
            observation["points"] = np.array(frame, copy=True)
            typed_paths.add(f"online_observations.{index}.points")
    elif observation_count:
        warnings.append("dtype_ambiguous:online_observations.points (JSON arrays have no dtype metadata)")

    _apply_sequence_candidate(
        observations,
        candidates.get("online_poses", []),
        field="T_w_e",
        semantic_path="online_observations",
        dtype_kind="f",
        typed_paths=typed_paths,
        warning=warnings,
    )
    _apply_sequence_candidate(
        observations,
        candidates.get("online_grips", []),
        field="grip",
        semantic_path="online_observations",
        dtype_kind="biuf",
        typed_paths=typed_paths,
        warning=warnings,
    )
    dt_candidates = candidates.get("dt", [])
    if dt_candidates:
        if "dt" not in record:
            warnings.append("typed_telemetry_achieved_dt_without_episode_json_field")
        else:
            value = _choose_typed_candidate(dt_candidates, "dt")
            if value.ndim != 1 or value.shape[0] != len(record["transitions"]):
                raise ValueError("typed telemetry achieved_dt does not align with v2 transitions")
            _require_sidecar_matches(value, record["dt"], dt_candidates[0][0], "dt")
            record["dt"] = [value[index].copy() for index in range(len(value))]
            typed_paths.add("dt")
    elif "dt" in record:
        warnings.append("dtype_ambiguous:dt (no matching typed sidecar; JSON numbers have no dtype metadata)")

    for name in ("rho", "rho_valid", "nu", "nu_valid", "epsilon", "epsilon_valid"):
        options = candidates.get("record:" + name, [])
        if not options:
            continue
        if name not in record:
            warnings.append(f"typed_layout_sidecar_without_episode_json_field:{name}")
            continue
        value = _choose_typed_candidate(options, name)
        _require_sidecar_matches(value, record[name], options[0][0], name)
        record[name] = np.array(value, copy=True)
        typed_paths.add(name)
    return typed_paths


def _apply_sequence_candidate(
    sequence: list[dict[str, Any]],
    options: list[tuple[str, np.ndarray]],
    *,
    field: str,
    semantic_path: str,
    dtype_kind: str,
    typed_paths: set[str],
    warning: list[str],
) -> None:
    if not options:
        warning.append(f"dtype_ambiguous:{semantic_path}.{field} (no matching typed sidecar)")
        return
    value = _choose_typed_candidate(options, field)
    if value.dtype.kind not in dtype_kind or value.shape[0] != len(sequence):
        raise ValueError(f"typed sidecar has an invalid {field} schema: {options[0][0]}")
    expected_tail = (4, 4) if field == "T_w_e" else ()
    if value.shape[1:] != expected_tail:
        raise ValueError(f"typed sidecar has an invalid {field} shape: {value.shape}")
    for index, item in enumerate(sequence):
        _require_sidecar_matches(value[index], item[field], options[0][0], f"{semantic_path}.{index}.{field}")
        item[field] = np.array(value[index], copy=True) if expected_tail else np.asarray(value[index]).copy()
        typed_paths.add(f"{semantic_path}.{index}.{field}")


def _choose_typed_candidate(options: list[tuple[str, np.ndarray]], name: str) -> np.ndarray:
    first_path, first = options[0]
    for other_path, other in options[1:]:
        if first.dtype != other.dtype:
            raise ValueError(f"typed sidecars disagree on {name} dtype: {first_path}, {other_path}")
        if not np.array_equal(first, other):
            raise ValueError(f"typed sidecars disagree on {name} values: {first_path}, {other_path}")
    return first


def _require_sidecar_matches(array: np.ndarray, expected: Any, sidecar_path: str, semantic_path: str) -> None:
    expected_array = np.asarray(expected)
    if array.shape != expected_array.shape or not np.array_equal(array, expected_array):
        raise ValueError(f"typed v2 sidecar disagrees with episode.json at {semantic_path}: {sidecar_path}")


def _numeric_dtype_warnings(record: Mapping[str, Any], typed_paths: set[str]) -> list[str]:
    paths: list[str] = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, np.ndarray):
            if value.dtype.kind in "iufc" and value.ndim > 0:
                paths.append(".".join(path))
            return
        if isinstance(value, (list, tuple)) and value:
            array = np.asarray(value)
            if not array.dtype.hasobject and array.dtype.kind in "iufc" and array.ndim > 0:
                paths.append(".".join(path))
                return
            for index, item in enumerate(value):
                walk(item, path + (str(index),))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, path + (str(key),))

    walk(record, ())
    output = []
    for path in sorted(set(paths)):
        if path in typed_paths:
            continue
        # Root observation and label arrays are counted at their semantic leaves.
        if any(path.startswith(prefix + ".") for prefix in typed_paths):
            continue
        output.append(f"dtype_ambiguous:{path} (numeric JSON array has no dtype metadata)")
    return output


def _npy_uncompressed_array_bytes(path: Path, name: str) -> int:
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            elif version == (3, 0):
                # NumPy has no public generic header reader for the UTF-8 v3 format.
                shape, _fortran_order, dtype = np.lib.format._read_array_header(stream, version)
            else:
                raise ValueError("unsupported NPY format version")
            header_bytes = stream.tell()
    except Exception as error:
        raise ValueError(f"v2 sidecar is not a safe NPY file: {name}") from error

    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.kind not in "biufc":
        raise ValueError(f"v2 binary sidecar must contain numeric/boolean arrays only: {name}")
    element_count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError(f"v2 NPY sidecar has an invalid array shape: {name}")
        element_count *= dimension
    return header_bytes + element_count * dtype.itemsize


def _preflight_decoded_sidecars(
    root: Path,
    relative_paths: set[str],
    *,
    max_decoded_sidecar_bytes: int,
) -> int:
    decoded_bytes = 0
    for relative in sorted(relative_paths):
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.suffix == ".npz":
            try:
                size = _npz_uncompressed_member_bytes(path)
            except Exception as error:
                raise ValueError(f"cannot preflight v2 NPZ sidecar: {relative}") from error
        elif path.suffix == ".npy":
            size = _npy_uncompressed_array_bytes(path, relative)
        else:
            continue
        decoded_bytes += size
        if decoded_bytes > max_decoded_sidecar_bytes:
            raise ValueError(
                "decoded sidecar aggregate exceeds max-decoded-sidecar-bytes "
                f"({decoded_bytes} > {max_decoded_sidecar_bytes}): {relative}"
            )
    return decoded_bytes


def _load_npz(path: Path, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            values = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except Exception as error:
        raise ValueError(f"v2 sidecar is not a safe NPZ file: {name}") from error
    for key, value in values.items():
        _require_numeric_array(value, f"{name}:{key}")
    return values


def _load_npy(path: Path, name: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"v2 sidecar is not a safe NPY file: {name}") from error
    _require_numeric_array(value, name)
    return np.array(value, copy=True)


def _require_numeric_array(value: np.ndarray, name: str) -> None:
    if value.dtype.hasobject or value.dtype.kind not in "biufc":
        raise ValueError(f"v2 binary sidecar must contain numeric/boolean arrays only: {name}")


def _array_key(*parts: str) -> str:
    normalized = "_".join(re.sub(r"[^a-zA-Z0-9]+", "_", part).strip("_").lower() for part in parts)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if len(normalized) > 54:
        normalized = normalized[:45] + "_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return normalized


def _add_raw_array(target: dict[str, Any], name: str, value: np.ndarray) -> None:
    if name in target:
        raise ValueError(f"v2 sidecar arrays collide after key normalization: {name}")
    target[name] = value


def _episode_field_counts(
    record: Mapping[str, Any],
    execution: Mapping[str, Any],
    raw_arrays: Mapping[str, Any],
    sidecar_metadata: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, int]:
    provenance = record.get("provenance", {})
    obs_fields = sum(len(item) for item in record.get("online_observations", []))
    transition_fields = sum(len(item) for item in record.get("transitions", []))
    return {
        "record_top_level": len(record),
        "provenance": len(provenance),
        "observation_fields": obs_fields,
        "transition_fields": transition_fields,
        "execution_fields": len(execution),
        "sidecar_metadata_files": len(sidecar_metadata),
        "raw_array_fields": len(raw_arrays),
        "numeric_dtype_ambiguities": sum(warning.startswith("dtype_ambiguous:") for warning in warnings),
    }


def _attempt_timeline(record: Mapping[str, Any], prefix_arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    observations = 0
    if "T_w_e" in prefix_arrays:
        value = prefix_arrays["T_w_e"]
        if value.ndim != 3 or value.shape[1:] != (4, 4):
            raise ValueError("valid_prefix.T_w_e must be [observations,4,4]")
        observations = int(value.shape[0])
    elif "point_offsets" in prefix_arrays:
        offsets = prefix_arrays["point_offsets"]
        if offsets.ndim != 1 or len(offsets) == 0 or offsets.dtype.kind not in "iu":
            raise ValueError("valid_prefix.point_offsets must be a one-dimensional integer array")
        observations = len(offsets) - 1
    valid_until = record.get("valid_observation_until")
    if valid_until is not None and observations != valid_until + 1:
        raise ValueError("attempt valid_observation_until disagrees with typed valid_prefix")
    actions = prefix_arrays.get("actions")
    transitions = int(actions.shape[0]) if actions is not None and actions.ndim > 0 else max(0, observations - 1)
    return observations, transitions


def _semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_semantic_equal(left[key], right[key]) for key in left)
    if isinstance(left, (np.ndarray, list, tuple)) and isinstance(right, (np.ndarray, list, tuple)):
        left_array, right_array = np.asarray(left), np.asarray(right)
        if not left_array.dtype.hasobject and not right_array.dtype.hasobject:
            if left_array.shape != right_array.shape:
                return False
            if left_array.dtype.kind == "b" or right_array.dtype.kind == "b":
                return left_array.dtype.kind == right_array.dtype.kind and np.array_equal(left_array, right_array)
            if left_array.dtype.kind in "iufc" and right_array.dtype.kind in "iufc":
                return bool(np.array_equal(left_array, right_array))
        return len(left) == len(right) and all(_semantic_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return bool(left == right)
    return left == right


def _failure_receipt(
    *,
    source_identity: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Mapping[str, Any]],
    declared_source_artifact_hashes: Mapping[str, Mapping[str, Any]],
    target_profile: Mapping[str, Any] | None,
    error: Exception,
) -> dict[str, Any]:
    return {
        "receipt_version": MIGRATION_RECEIPT_VERSION,
        "status": "failed",
        "source_identity": dict(source_identity),
        "source_artifact_hashes": {
            key: dict(value) for key, value in sorted(source_artifact_hashes.items())
        },
        "declared_source_artifact_hashes": {
            key: dict(value) for key, value in sorted(declared_source_artifact_hashes.items())
        },
        "target_profile": None if target_profile is None else dict(target_profile),
        "error": {
            "type": type(error).__name__,
            "message": _redact_sensitive(str(error)),
        },
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json(_redact_sensitive(dict(payload)))
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"receipt path is not a regular file: {path}")
        if path.read_bytes() != raw:
            raise FileExistsError(f"refusing to overwrite a different immutable migration receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{os.urandom(4).hex()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    payload = _read_json_value(path, name=name)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _read_json_value(path: Path, *, name: str) -> Any:
    try:
        payload = _strict_json(path.read_text(encoding="utf-8"), name)
    except OSError as error:
        raise ValueError(f"cannot read {name}") from error
    return payload


def _strict_json(value: str, name: str) -> Any:
    def reject_constant(item: str) -> None:
        raise ValueError(f"non-finite JSON value {item} in {name}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except Exception as error:
        raise ValueError(f"{name} is malformed JSON") from error


def _safe_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonblank relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or path.as_posix() != value
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a normalized contained relative path")
    return value


def _safe_repo_path(value: Any, name: str) -> str:
    return _safe_relative_path(value, name)


def _prefix(value: Any, name: str) -> str:
    path = _safe_relative_path(value, name)
    if path.endswith("/"):
        path = path[:-1]
    if not path:
        raise ValueError(f"{name} must be a nonempty repo-relative prefix")
    return path


def _artifact_path(prefix: str, kind: str, program_id: str, record_id: str) -> str:
    collection = "episodes" if kind == "episode" else "attempts"
    return f"{prefix}/{collection}/{program_id}/{record_id}"


def _ensure_disjoint_prefixes(source: str, target: str) -> None:
    if source == target or source.startswith(target + "/") or target.startswith(source + "/"):
        raise ValueError("target_prefix must be separate from the immutable source_prefix")


def _file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"archive output must be a real directory: {root}")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archive output must not contain symlinks: {path.relative_to(root)}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_COPY_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _int_mapping(value: Any, name: str) -> dict[str, int]:
    mapping = _require_mapping(value, name)
    if any(type(item) is not int or item < 0 for item in mapping.values()):
        raise ValueError(f"{name} must contain nonnegative integer values")
    return {str(key): int(item) for key, item in mapping.items()}


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _temporary_path(directory: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-id", required=True, help="HF dataset repo containing the immutable v2 artifact")
    parser.add_argument("--source-revision", required=True, help="Full immutable 40- or 64-character HF commit OID")
    parser.add_argument("--source-prefix", required=True, help="Exact v2 HF prefix; no repository scan is performed")
    parser.add_argument("--kind", required=True, choices=("episode", "attempt"))
    parser.add_argument("--program-id", required=True, help="One exact program ID")
    parser.add_argument("--record-id", required=True, help="One exact episode ID or attempt ID")
    parser.add_argument("--target-repo-id", required=True, help="HF dataset repo for the new archive")
    parser.add_argument("--target-prefix", required=True, help="New, separate HF prefix for the migrated archive")
    parser.add_argument("--target-branch", default="main", help="Existing target repo branch to append to")
    parser.add_argument("--receipt-dir", required=True, type=Path, help="Local directory for durable success/failure receipts")
    parser.add_argument("--scratch-root", type=Path, help="Parent for the owned temporary download/cache/staging tree")
    parser.add_argument("--max-source-bytes", type=_positive_int, default=2 * 1024**3)
    parser.add_argument("--max-target-bytes", type=_positive_int, default=2 * 1024**3)
    parser.add_argument("--max-scratch-bytes", type=_positive_int, default=8 * 1024**3)
    parser.add_argument(
        "--max-decoded-sidecar-bytes",
        type=_positive_int,
        default=_DEFAULT_MAX_DECODED_SIDECAR_BYTES,
        help="aggregate uncompressed NPZ/NPY array bytes allowed before sidecar loading",
    )
    parser.add_argument("--chunk-boundaries", type=_positive_int, default=64)
    parser.add_argument("--max-chunk-bytes", type=_positive_int, default=256 * 1024**2)
    return parser


def _remote_file_sizes(api: Any, *, repo_id: str, revision: str, prefix: str) -> dict[str, int]:
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=prefix,
            recursive=True,
            expand=False,
        )
    except Exception as error:
        if _is_remote_not_found(error):
            return {}
        raise
    files: dict[str, int] = {}
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str) or not (path == prefix or path.startswith(prefix + "/")):
            raise ValueError("HF tree listing returned a path outside the exact selected prefix")
        size = getattr(entry, "size", None)
        if size is None:
            # RepoFolder entries do not have byte sizes; files must always have them.
            continue
        if type(size) is not int or size < 0:
            raise ValueError(f"HF tree returned an invalid size for {path}")
        if path in files:
            raise ValueError(f"HF tree returned a duplicate path: {path}")
        files[path] = size
    return files


def _is_remote_not_found(error: Exception) -> bool:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(error, "status_code", None)
    return status == 404


def _copy_hf_file(
    hf_hub_download: Any,
    *,
    repo_id: str,
    revision: str,
    filename: str,
    repo_type: str,
    cache_dir: Path,
    destination: Path,
    expected_bytes: int,
    byte_limit: int,
) -> None:
    downloaded = Path(hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir),
    ))
    if not downloaded.is_file():
        raise ValueError(f"HF download did not produce a regular file for {filename}")
    try:
        downloaded.resolve(strict=True).relative_to(cache_dir.resolve())
    except (OSError, ValueError) as error:
        raise ValueError(f"HF download escaped its owned cache: {filename}") from error
    if downloaded.stat().st_size != expected_bytes or expected_bytes > byte_limit:
        raise ValueError(f"HF file size is incomplete or exceeds the configured cap: {filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with downloaded.open("rb") as source, destination.open("xb") as target:
        for block in iter(lambda: source.read(_COPY_BLOCK_BYTES), b""):
            written += len(block)
            if written > min(expected_bytes, byte_limit):
                raise ValueError(f"HF download exceeded its preflight byte limit: {filename}")
            target.write(block)
        target.flush()
        os.fsync(target.fileno())
    if written != expected_bytes:
        destination.unlink(missing_ok=True)
        raise ValueError(f"HF download byte count mismatch for {filename}")


def _download_source_artifact(
    api: Any,
    hf_hub_download: Any,
    *,
    identity: Mapping[str, Any],
    revision: str,
    root: Path,
    cache_dir: Path,
    max_source_bytes: int,
    max_scratch_bytes: int,
    max_target_bytes: int,
) -> tuple[Path, dict[str, int]]:
    remote_files = _remote_file_sizes(
        api,
        repo_id=identity["source_repo_id"],
        revision=revision,
        prefix=identity["source_artifact_path"],
    )
    manifest_repo_path = identity["source_artifact_path"] + "/artifact_manifest.json"
    if manifest_repo_path not in remote_files:
        raise ValueError("selected v2 HF artifact has no artifact_manifest.json")
    total_bytes = sum(remote_files.values())
    if total_bytes > max_source_bytes:
        raise ValueError(f"selected v2 artifact exceeds max-source-bytes ({total_bytes} > {max_source_bytes})")
    if 2 * total_bytes + 2 * max_target_bytes > max_scratch_bytes:
        raise ValueError("configured max-scratch-bytes is below the preflight source/cache/target bound")

    root.mkdir(parents=True, exist_ok=False)
    manifest_destination = root / "artifact_manifest.json"
    _copy_hf_file(
        hf_hub_download,
        repo_id=identity["source_repo_id"],
        revision=revision,
        filename=manifest_repo_path,
        repo_type="dataset",
        cache_dir=cache_dir,
        destination=manifest_destination,
        expected_bytes=remote_files[manifest_repo_path],
        byte_limit=max_source_bytes,
    )
    artifact_manifest = _read_json(manifest_destination, name="artifact_manifest.json")
    inventory = artifact_manifest.get("files")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("source artifact manifest has no complete files inventory")
    expected = {manifest_repo_path}
    for relative, info in inventory.items():
        relative = _safe_relative_path(relative, "source artifact inventory path")
        if relative == "artifact_manifest.json":
            raise ValueError("source artifact inventory must not list its own manifest")
        size = _require_mapping(info, f"source inventory entry {relative}").get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError(f"source artifact inventory has invalid bytes for {relative}")
        remote_path = identity["source_artifact_path"] + "/" + relative
        expected.add(remote_path)
        if remote_files.get(remote_path) != size:
            raise ValueError(f"source HF tree is incomplete or byte counts disagree for {relative}")
    if set(remote_files) != expected:
        raise ValueError("selected source HF prefix has extra or missing files relative to artifact_manifest.json")

    copied: dict[str, int] = {manifest_repo_path: remote_files[manifest_repo_path]}
    for remote_path in sorted(expected - {manifest_repo_path}):
        relative = remote_path.removeprefix(identity["source_artifact_path"] + "/")
        destination = root.joinpath(*PurePosixPath(relative).parts)
        _copy_hf_file(
            hf_hub_download,
            repo_id=identity["source_repo_id"],
            revision=revision,
            filename=remote_path,
            repo_type="dataset",
            cache_dir=cache_dir,
            destination=destination,
            expected_bytes=remote_files[remote_path],
            byte_limit=max_source_bytes,
        )
        copied[remote_path] = remote_files[remote_path]
    return root, copied


def _download_existing_target(
    api: Any,
    hf_hub_download: Any,
    *,
    identity: Mapping[str, Any],
    target_revision: str,
    target_root: Path,
    receipt_path: Path,
    cache_dir: Path,
    source_hashes: Mapping[str, Mapping[str, Any]],
    max_target_bytes: int,
) -> bool:
    remote_files = _remote_file_sizes(
        api,
        repo_id=identity["target_repo_id"],
        revision=target_revision,
        prefix=identity["target_artifact_path"],
    )
    receipt_exists = api.file_exists(
        repo_id=identity["target_repo_id"],
        repo_type="dataset",
        revision=target_revision,
        filename=identity["target_receipt_path"],
    )
    if not remote_files and not receipt_exists:
        return False
    if not remote_files or not receipt_exists:
        raise ValueError("partial immutable migration target already exists")
    receipt_remote = _temporary_path(cache_dir, "remote-receipt-")
    try:
        receipt_size = _remote_file_sizes(
            api,
            repo_id=identity["target_repo_id"],
            revision=target_revision,
            prefix=identity["target_receipt_path"],
        ).get(identity["target_receipt_path"])
        if receipt_size is None or receipt_size > _MAX_RECEIPT_BYTES:
            raise ValueError("existing migration receipt is missing or exceeds its byte cap")
        _copy_hf_file(
            hf_hub_download,
            repo_id=identity["target_repo_id"],
            revision=target_revision,
            filename=identity["target_receipt_path"],
            repo_type="dataset",
            cache_dir=cache_dir,
            destination=receipt_remote,
            expected_bytes=receipt_size,
            byte_limit=_MAX_RECEIPT_BYTES,
        )
        payload = _read_json(receipt_remote, name="existing migration receipt")
        receipt = MigrationReceipt.from_dict(payload)
        if dict(receipt.source_identity) != dict(identity):
            raise ValueError("existing target is bound to a different immutable source identity")
        if dict(receipt.source_artifact_hashes) != dict(source_hashes):
            raise ValueError("existing target receipt source hashes differ from selected v2 artifact")
        expected_files = set(receipt.target_archive_hashes)
        expected_remote = {identity["target_artifact_path"] + "/" + name for name in expected_files}
        if set(remote_files) != expected_remote:
            raise ValueError("existing migration target file inventory is incomplete or contains extra files")
        total = sum(remote_files.values()) + receipt_size
        if total > max_target_bytes:
            raise ValueError(f"existing migration target exceeds max-target-bytes ({total} > {max_target_bytes})")
        target_root.mkdir(parents=True, exist_ok=False)
        for relative, info in sorted(receipt.target_archive_hashes.items()):
            relative = _safe_relative_path(relative, "existing target archive path")
            remote_path = identity["target_artifact_path"] + "/" + relative
            if remote_files.get(remote_path) != info.get("bytes"):
                raise ValueError(f"existing migration target byte count mismatch: {relative}")
            local_path = target_root.joinpath(*PurePosixPath(relative).parts)
            _copy_hf_file(
                hf_hub_download,
                repo_id=identity["target_repo_id"],
                revision=target_revision,
                filename=remote_path,
                repo_type="dataset",
                cache_dir=cache_dir,
                destination=local_path,
                expected_bytes=info["bytes"],
                byte_limit=max_target_bytes,
            )
            if _sha256_file(local_path) != info.get("sha256"):
                raise ValueError(f"existing migration target SHA256 mismatch: {relative}")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(receipt_remote, receipt_path)
        return True
    finally:
        receipt_remote.unlink(missing_ok=True)


def _target_file_pairs(target_root: Path, receipt_path: Path, target_path: str, remote_receipt: str) -> list[tuple[str, Path]]:
    pairs = [
        (target_path + "/" + relative, path)
        for relative, _info in sorted(_file_inventory(target_root).items())
        for path in [target_root.joinpath(*PurePosixPath(relative).parts)]
    ]
    pairs.append((remote_receipt, receipt_path))
    return pairs


def _verify_remote_files(
    api: Any,
    hf_hub_download: Any,
    *,
    repo_id: str,
    revision: str,
    file_pairs: list[tuple[str, Path]],
    cache_dir: Path,
    max_bytes: int,
) -> bool:
    existence = [
        api.file_exists(repo_id=repo_id, repo_type="dataset", revision=revision, filename=remote)
        for remote, _local in file_pairs
    ]
    if not any(existence):
        return False
    if not all(existence):
        raise ValueError("partial immutable migration snapshot already exists")
    total_bytes = sum(local.stat().st_size for _remote, local in file_pairs)
    if total_bytes > max_bytes:
        raise ValueError(f"migration target exceeds max-target-bytes ({total_bytes} > {max_bytes})")
    for remote, local in file_pairs:
        size = local.stat().st_size
        temporary = _temporary_path(cache_dir, "verify-")
        try:
            _copy_hf_file(
                hf_hub_download,
                repo_id=repo_id,
                revision=revision,
                filename=remote,
                repo_type="dataset",
                cache_dir=cache_dir,
                destination=temporary,
                expected_bytes=size,
                byte_limit=max_bytes,
            )
            if _sha256_file(temporary) != _sha256_file(local):
                raise ValueError(f"HF migration target verification hash mismatch: {remote}")
        finally:
            temporary.unlink(missing_ok=True)
    return True


def _publish_target(
    api: Any,
    hf_hub_download: Any,
    *,
    identity: Mapping[str, Any],
    target_branch: str,
    target_head: str,
    target_root: Path,
    receipt_path: Path,
    cache_dir: Path,
    max_target_bytes: int,
) -> str:
    from huggingface_hub import CommitOperationAdd

    file_pairs = _target_file_pairs(
        target_root,
        receipt_path,
        identity["target_artifact_path"],
        identity["target_receipt_path"],
    )
    if _verify_remote_files(
        api,
        hf_hub_download,
        repo_id=identity["target_repo_id"],
        revision=target_head,
        file_pairs=file_pairs,
        cache_dir=cache_dir,
        max_bytes=max_target_bytes,
    ):
        return target_head
    current = api.repo_info(
        repo_id=identity["target_repo_id"], repo_type="dataset", revision=target_branch
    ).sha
    if current != target_head:
        raise ValueError("target HF branch changed after immutable conflict preflight; refusing to retry")
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for remote, local in file_pairs
    ]
    commit = api.create_commit(
        repo_id=identity["target_repo_id"],
        repo_type="dataset",
        revision=target_branch,
        parent_commit=target_head,
        operations=operations,
        commit_message=(
            f"Migrate {identity['source_kind']} {identity['record_id']} from immutable v2 artifact"
        ),
    )
    commit_oid = getattr(commit, "oid", None)
    if not isinstance(commit_oid, str) or not _COMMIT_OID.fullmatch(commit_oid.lower()):
        raise ValueError("HF migration commit did not return an immutable commit OID")
    commit_oid = commit_oid.lower()
    if not _verify_remote_files(
        api,
        hf_hub_download,
        repo_id=identity["target_repo_id"],
        revision=commit_oid,
        file_pairs=file_pairs,
        cache_dir=cache_dir,
        max_bytes=max_target_bytes,
    ):
        raise ValueError("HF migration commit verification found no uploaded archive/receipt files")
    return commit_oid


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    failure_receipt_path: Path | None = None
    identity: dict[str, Any] = {}
    source_reader: V2ArtifactReader | None = None
    try:
        revision = args.source_revision.lower()
        if not _COMMIT_OID.fullmatch(revision):
            raise ValueError("source-revision must be a full immutable 40- or 64-character HF commit OID")
        if not isinstance(args.target_branch, str) or not _SAFE_ID.fullmatch(args.target_branch):
            raise ValueError("target-branch must be a safe existing branch name")
        identity = _validate_identity({
            "source_repo_id": args.source_repo_id,
            "source_revision": revision,
            "source_prefix": args.source_prefix,
            "source_artifact_path": _artifact_path(args.source_prefix.rstrip("/"), args.kind, args.program_id, args.record_id),
            "source_kind": args.kind,
            "program_id": args.program_id,
            "record_id": args.record_id,
            "target_repo_id": args.target_repo_id,
            "target_prefix": args.target_prefix,
            "target_artifact_path": _artifact_path(args.target_prefix.rstrip("/"), args.kind, args.program_id, args.record_id),
            "target_receipt_path": (
                f"{args.target_prefix.rstrip('/')}/migration-receipts/{args.kind}/"
                f"{args.program_id}/{args.record_id}.json"
            ),
            "target_branch": args.target_branch,
        })
        local_name = f"{args.kind}-{args.program_id}-{args.record_id}"
        args.receipt_dir.mkdir(parents=True, exist_ok=True)
        failure_receipt_path = args.receipt_dir / f"{local_name}.failure.json"
        local_success_receipt = args.receipt_dir / f"{local_name}.json"
        if local_success_receipt.exists() and local_success_receipt.is_symlink():
            raise ValueError("local success receipt must not be a symlink")
        scratch_parent = args.scratch_root
        if scratch_parent is not None:
            scratch_parent.mkdir(parents=True, exist_ok=True)
            if scratch_parent.is_symlink() or not scratch_parent.is_dir():
                raise ValueError("scratch-root must be a real directory")
        with tempfile.TemporaryDirectory(prefix="icgs-v2-migration-", dir=scratch_parent) as temporary:
            scratch = Path(temporary)
            cache_dir = scratch / "hf-cache"
            cache_dir.mkdir()
            source_root = scratch / "source-record"
            target_root = scratch / "target-archive"
            staged_receipt = scratch.joinpath(*PurePosixPath(identity["target_receipt_path"]).parts)

            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi()
            source_info = api.repo_info(
                repo_id=identity["source_repo_id"], repo_type="dataset", revision=revision
            )
            if str(getattr(source_info, "sha", "")).lower() != revision:
                raise ValueError("HF did not resolve the requested source revision to the same immutable commit")
            _download_source_artifact(
                api,
                hf_hub_download,
                identity=identity,
                revision=revision,
                root=source_root,
                cache_dir=cache_dir,
                max_source_bytes=args.max_source_bytes,
                max_scratch_bytes=args.max_scratch_bytes,
                max_target_bytes=args.max_target_bytes,
            )
            source_reader = V2ArtifactReader(
                source_root,
                max_decoded_sidecar_bytes=args.max_decoded_sidecar_bytes,
            )
            source = source_reader.read()
            _validate_source_identity(identity, source)
            target_head = api.repo_info(
                repo_id=identity["target_repo_id"],
                repo_type="dataset",
                revision=args.target_branch,
            ).sha

            target_present = _download_existing_target(
                api,
                hf_hub_download,
                identity=identity,
                target_revision=target_head,
                target_root=target_root,
                receipt_path=staged_receipt,
                cache_dir=cache_dir,
                source_hashes=source.source_artifact_hashes,
                max_target_bytes=args.max_target_bytes,
            )
            target_profile = ArchiveProfileConfig(
                chunk_boundaries=args.chunk_boundaries,
                max_chunk_bytes=args.max_chunk_bytes,
                local_artifact_retention="keep",
            )
            bound_writer = MigrationTargetWriter(
                EpisodeArchiveWriter(target_profile),
                output_dir=target_root,
                receipt_path=staged_receipt,
                failure_receipt_path=failure_receipt_path,
            )
            receipt = migrate_episode(
                source_reader,
                bound_writer,
                source_identity=identity,
                target_profile=target_profile,
            )
            if target_present:
                _write_exclusive_json(local_success_receipt, receipt.as_dict())
                print(json.dumps({"migration_receipt": receipt.as_dict(), "publication": "already-present-and-verified"}, indent=2, sort_keys=True))
                return 0

            target_bytes = sum(item["bytes"] for item in _file_inventory(target_root).values()) + staged_receipt.stat().st_size
            source_bytes = sum(item["bytes"] for item in source.source_artifact_hashes.values())
            if target_bytes > args.max_target_bytes:
                raise ValueError(f"new migration target exceeds max-target-bytes ({target_bytes} > {args.max_target_bytes})")
            if 2 * source_bytes + 2 * target_bytes > args.max_scratch_bytes:
                raise ValueError("actual migration staging exceeded max-scratch-bytes")
            commit_oid = _publish_target(
                api,
                hf_hub_download,
                identity=identity,
                target_branch=args.target_branch,
                target_head=target_head,
                target_root=target_root,
                receipt_path=staged_receipt,
                cache_dir=cache_dir,
                max_target_bytes=args.max_target_bytes,
            )
            _write_exclusive_json(local_success_receipt, receipt.as_dict())
            print(json.dumps({
                "migration_receipt": receipt.as_dict(),
                "publication": {"repo_id": identity["target_repo_id"], "revision": commit_oid},
            }, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        safe_error_text = _redact_sensitive(str(error))
        if failure_receipt_path is None and getattr(args, "receipt_dir", None):
            try:
                args.receipt_dir.mkdir(parents=True, exist_ok=True)
                safe_program = re.sub(r"[^A-Za-z0-9._-]", "_", getattr(args, "program_id", "unknown"))
                safe_record = re.sub(r"[^A-Za-z0-9._-]", "_", getattr(args, "record_id", "unknown"))
                failure_receipt_path = args.receipt_dir / f"migration-{safe_program}-{safe_record}.failure.json"
            except Exception:
                failure_receipt_path = None
        hashes = source_reader.source_artifact_hashes if source_reader is not None else {}
        payload = _failure_receipt(
            source_identity=identity or _safe_partial_identity({
                "source_repo_id": getattr(args, "source_repo_id", None),
                "source_revision": getattr(args, "source_revision", None),
                "source_prefix": getattr(args, "source_prefix", None),
                "source_kind": getattr(args, "kind", None),
                "program_id": getattr(args, "program_id", None),
                "record_id": getattr(args, "record_id", None),
                "target_repo_id": getattr(args, "target_repo_id", None),
                "target_prefix": getattr(args, "target_prefix", None),
            }),
            source_artifact_hashes=hashes,
            declared_source_artifact_hashes=(
                source_reader.declared_artifact_hashes if source_reader is not None else {}
            ),
            target_profile=None,
            error=error,
        )
        if failure_receipt_path is not None:
            try:
                _write_exclusive_json(failure_receipt_path, payload)
            except Exception as receipt_error:
                safe_receipt_error_text = _redact_sensitive(str(receipt_error))
                print(
                    "migration failed: "
                    f"{safe_error_text}; failure receipt write also failed: {safe_receipt_error_text}"
                )
                return 2
        print(f"migration failed: {safe_error_text}")
        if failure_receipt_path is not None:
            print(f"failure receipt: {failure_receipt_path}")
        return 2


__all__ = [
    "MIGRATION_RECEIPT_VERSION",
    "MigrationError",
    "MigrationReceipt",
    "MigrationTargetWriter",
    "V2ArtifactReader",
    "main",
    "migrate_episode",
]


if __name__ == "__main__":
    raise SystemExit(main())
