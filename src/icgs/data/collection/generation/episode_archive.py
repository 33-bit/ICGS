"""Lossless, bounded-chunk archives for generated ICGS episodes and attempts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np

from icgs.data.collection.generation.distributed_contracts import (
    ARCHIVE_DATASET_IDENTITY,
    ARCHIVE_EPISODE_SCHEMA_VERSION,
    ARCHIVE_FORMAT_ID,
    ArchiveProfileConfig,
)
from icgs.data.schemas.episode_records import validate_episode


_MANIFEST_FIELDS = frozenset({
    "archive_format_id",
    "episode_schema_version",
    "dataset_identity",
    "archive_kind",
    "episode_id",
    "attempt_id",
    "program_id",
    "split",
    "subset",
    "outcome",
    "source_run_id",
    "code_revision",
    "preprocessing_identity",
    "archive_profile",
    "local_write_limits",
    "timeline",
    "record_metadata",
    "raw_arrays",
    "array_specs",
    "array_aliases",
    "chunk_inventory",
    "debug_path",
    "debug_bytes",
    "debug_sha256",
    "artifact_manifest_path",
})
_SAFE_ARRAY_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_NPZ_MEMBER = re.compile(r"^[A-Za-z0-9_]{1,128}\.npy$")
_MAX_DEBUG_TEXT = 8192
_SENSITIVE_METADATA_KEYS = frozenset({
    "access_key", "access_token", "api_key", "apikey", "authorization", "client_secret",
    "credential", "credentials", "hf_token", "id_token", "password", "password_path",
    "passphrase", "passwd", "private_key", "proxy_authorization", "refresh_token",
    "secret", "secret_key", "token", "token_file", "token_path",
})
_PIECE_FIELDS = frozenset({
    "key", "path", "shape", "byte_count", "sha256",
    "boundary_start", "boundary_stop", "point_start", "point_stop", "validity_point_count",
    "transition_start", "transition_stop", "timeline_start", "timeline_stop",
    "array_start", "array_stop", "timeline_index",
})


class ArchiveWriterCapExceeded(RuntimeError):
    """A writer write would exceed the per-result retry-root byte limit."""


class _BudgetedFile:
    def __init__(self, stream: Any, budget: "_WriterByteBudget", path: Path) -> None:
        self._stream = stream
        self._budget = budget
        self._path = path
        self._size = path.stat().st_size

    def write(self, payload: bytes) -> int:
        end = self._stream.tell() + len(payload)
        growth = max(0, end - self._size)
        attempted = self._budget.used + growth
        if attempted > self._budget.limit:
            raise ArchiveWriterCapExceeded(
                f"max_result_bytes limit={self._budget.limit} current={self._budget.used} "
                f"attempted={attempted} preserved_path={self._path} "
                f"retry_root={self._budget.root}"
            )
        written = self._stream.write(payload)
        new_size = max(self._size, self._stream.tell())
        self._budget.used += new_size - self._size
        self._size = new_size
        return written

    def __enter__(self) -> "_BudgetedFile":
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _WriterByteBudget:
    def __init__(self, root: Path, limit: int) -> None:
        self.root = root
        self.limit = limit
        self.used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    def open(self, path: Path) -> _BudgetedFile:
        return _BudgetedFile(path.open("wb", buffering=0), self, path)


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_array_digest(digest: Any, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
        return
    for block in np.nditer(
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=65536,
    ):
        digest.update(np.asarray(block).tobytes(order="C"))


def _array_digest(dtype: np.dtype, shape: Sequence[int], chunks: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(item) for item in shape]))
    for value in chunks:
        _update_array_digest(digest, value)
    return digest.hexdigest()


def _require_safe_relative(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a contained relative path")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized relative path")
    return value


def _as_numeric_array(value: Any, *, path: str) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden: {path}")
        if value.dtype.kind in "biufc":
            return value
        if value.dtype.kind in "SU":
            return None
        raise ValueError(f"unsupported NumPy dtype {value.dtype} at {path}")
    if isinstance(value, (list, tuple)) and value:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if array.dtype.hasobject:
            return None
        if array.dtype.kind in "biufc":
            return array
    return None


def _json_scalar(value: Any, *, path: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return value
    raise TypeError(f"unsupported archive metadata value at {path}: {type(value).__name__}")


def _redact_debug_text(value: str) -> str:
    text = value[:_MAX_DEBUG_TEXT]
    text = re.sub(r"\bhf_[A-Za-z0-9]{16,}\b", "[REDACTED_HF_TOKEN]", text)
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(token=)[^&\s]+", r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)\s*[:=]\s*)[^\s&,;]+",
        r"\1[REDACTED]",
        text,
    )
    return text


def _is_sensitive_metadata_key(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    if normalized in _SENSITIVE_METADATA_KEYS:
        return True
    return normalized.endswith((
        "_access_key", "_access_token", "_api_key", "_authorization", "_credential",
        "_credentials", "_password", "_password_path", "_passphrase", "_passwd",
        "_private_key", "_refresh_token", "_secret", "_secret_key", "_token",
        "_token_file", "_token_path",
    ))


def _redact_sensitive(value: Any) -> Any:
    """Redact credentials recursively before attempt metadata reaches disk."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return _redact_debug_text(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            safe_key = _redact_debug_text(str(key))
            if safe_key in result:
                raise ValueError("attempt metadata keys collide after secret redaction")
            result[safe_key] = "[REDACTED]" if _is_sensitive_metadata_key(key) else _redact_sensitive(item)
        return result
    if isinstance(value, tuple):
        return tuple(_redact_sensitive(item) for item in value)
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _npy_member_uncompressed_bytes(value: np.ndarray) -> int:
    """Return the exact uncompressed .npy member size, including its header."""
    buffer = io.BytesIO()
    header = np.lib.format.header_data_from_array_1_0(value)
    np.lib.format.write_array_header_1_0(buffer, header)
    return int(value.nbytes) + buffer.tell()


def _npz_uncompressed_member_bytes(path: Path) -> int:
    """Inspect ZIP metadata only; do not inflate members to enforce archive caps."""
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [item.filename for item in entries]
            if not entries or len(names) != len(set(names)):
                raise ValueError(f"NPZ chunk has an empty or duplicate member inventory: {path}")
            for item in entries:
                if (
                    not _SAFE_NPZ_MEMBER.fullmatch(item.filename)
                    or "/" in item.filename
                    or "\\" in item.filename
                    or item.filename in {".", ".."}
                    or item.file_size < 10
                ):
                    raise ValueError(f"NPZ chunk has an unsafe member name or size: {path}")
            return sum(int(item.file_size) for item in entries)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"chunk is not a safe NPZ archive: {path}") from error


def _check_npz_uncompressed_cap(path: Path, *, max_chunk_bytes: int, relative: str) -> int:
    size = _npz_uncompressed_member_bytes(path)
    if size > max_chunk_bytes:
        raise ValueError(
            f"uncompressed archive chunk exceeds max_chunk_bytes ({size} > {max_chunk_bytes}): {relative}"
        )
    return size


def _array_references(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        if "$archive_array" in value:
            name = value.get("$archive_array")
            if not isinstance(name, str) or not name:
                raise ValueError("archive array reference must be a nonblank name")
            yield name
        for item in value.values():
            yield from _array_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _array_references(item)
    elif isinstance(value, tuple):
        for item in value:
            yield from _array_references(item)


def _array_references_with_paths(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(value, Mapping):
        if "$archive_array" in value:
            name = value.get("$archive_array")
            if not isinstance(name, str) or not name:
                raise ValueError("archive array reference must be a nonblank name")
            yield path, name
            return
        for key, item in value.items():
            yield from _array_references_with_paths(item, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _array_references_with_paths(item, path + (str(index),))


def _reference_semantic_role(reader: "EpisodeArchiveReader", name: str) -> str:
    for alias in reader.manifest.payload["array_aliases"]:
        if alias["name"] == name:
            return alias["semantic_role"]
    resolved = reader._resolved_name(name)
    spec = reader.manifest.payload["array_specs"].get(resolved)
    if spec is None:
        raise ValueError(f"archive array reference is missing: {name}")
    return spec["semantic_role"]


def _validate_reference_semantic_roles(
    reader: "EpisodeArchiveReader",
    payload: Mapping[str, Any],
    debug_metadata: Mapping[str, Any],
) -> None:
    def check(section: str, value: Any, *, attempt_prefix: bool = False) -> None:
        for path, name in _array_references_with_paths(value):
            semantic_role = _reference_semantic_role(reader, name)
            if section == "record_metadata":
                if path and path[0] == "dt":
                    expected_role = "transitions/dt"
                elif path and path[0] == "_archive_transition_metadata":
                    expected_role = "/".join(("transition_metadata", *path[1:]))
                else:
                    expected_role = "/".join(path)
            elif section == "raw_arrays":
                if not path:
                    raise ValueError("raw archive array references require a named field")
                prefix = f"prefix_{path[0]}" if attempt_prefix else path[0]
                expected_role = "/".join(("raw_arrays", prefix, *path[1:]))
            else:
                expected_role = "/".join(("debug", *path))
            if semantic_role != expected_role:
                raise ValueError(
                    f"archive array reference semantic role mismatch: {name} has {semantic_role!r}, expected {expected_role!r}"
                )

    check("record_metadata", payload["record_metadata"])
    check("raw_arrays", payload["raw_arrays"], attempt_prefix=payload["archive_kind"] == "attempt")
    check("debug", debug_metadata)


@dataclass(frozen=True)
class ArchiveManifest:
    """Validated immutable view of one canonical archive manifest."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = dict(self.payload)
        unknown = set(values) - _MANIFEST_FIELDS
        missing = _MANIFEST_FIELDS - set(values)
        if unknown:
            raise ValueError(f"archive manifest has unknown field(s): {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"archive manifest is missing field(s): {', '.join(sorted(missing))}")
        if values["archive_format_id"] != ARCHIVE_FORMAT_ID:
            raise ValueError("archive_format_id mismatch")
        if values["episode_schema_version"] != ARCHIVE_EPISODE_SCHEMA_VERSION:
            raise ValueError("episode_schema_version mismatch")
        if values["dataset_identity"] != ARCHIVE_DATASET_IDENTITY:
            raise ValueError("dataset_identity mismatch")
        if values["archive_kind"] not in {"episode", "attempt"}:
            raise ValueError("archive_kind must be episode or attempt")
        if values["archive_kind"] == "episode":
            for key in ("episode_id", "attempt_id", "program_id"):
                if not isinstance(values[key], str) or not values[key].strip():
                    raise ValueError(f"{key} must be a nonblank string for episode archives")
            if values["outcome"] not in {"success", "valid_failure"}:
                raise ValueError("episode archive outcome must be success or valid_failure")
        else:
            if values["episode_id"] is not None:
                raise ValueError("attempt archive episode_id must be null")
            for key in ("attempt_id", "program_id"):
                if not isinstance(values[key], str) or not values[key].strip():
                    raise ValueError(f"{key} must be a nonblank string for attempt archives")
            if values["outcome"] not in {"simulator_crash", "invalid_observation"}:
                raise ValueError("attempt archive outcome must be simulator_crash or invalid_observation")
        for key in ("source_run_id", "code_revision", "preprocessing_identity"):
            if not isinstance(values[key], str) or not values[key].strip():
                raise ValueError(f"{key} must be a nonblank string")
        if not isinstance(values["archive_profile"], Mapping):
            raise ValueError("archive_profile must be an object")
        profile = ArchiveProfileConfig.from_dict(values["archive_profile"])
        if not isinstance(values["local_write_limits"], Mapping):
            raise ValueError("local_write_limits must be an object")
        write_limits = values["local_write_limits"]
        for key in ("max_chunk_bytes", "spool_peak_bytes", "local_peak_bytes_upper_bound"):
            if type(write_limits.get(key)) is not int or write_limits[key] < 0:
                raise ValueError(f"local_write_limits.{key} must be a nonnegative integer")
        if write_limits["max_chunk_bytes"] != profile.max_chunk_bytes:
            raise ValueError("local_write_limits max_chunk_bytes disagrees with archive profile")
        if write_limits["spool_peak_bytes"] > write_limits["local_peak_bytes_upper_bound"]:
            raise ValueError("local write peak bound is smaller than its numeric spool peak")
        if not isinstance(values["timeline"], Mapping):
            raise ValueError("timeline must be an object")
        for field in ("record_metadata", "raw_arrays"):
            if not isinstance(values[field], Mapping):
                raise ValueError(f"{field} must be an object")
        if not isinstance(values["array_specs"], Mapping):
            raise ValueError("array_specs must be an object")
        if not isinstance(values["array_aliases"], list):
            raise ValueError("array_aliases must be a list")
        if not isinstance(values["chunk_inventory"], list):
            raise ValueError("chunk_inventory must be a list")
        chunk_paths: set[str] = set()
        for item in values["chunk_inventory"]:
            if not isinstance(item, Mapping):
                raise ValueError("chunk_inventory entries must be objects")
            path = _require_safe_relative(item.get("path"), "chunk path")
            if path in chunk_paths:
                raise ValueError(f"duplicate chunk path: {path}")
            chunk_paths.add(path)
            if type(item.get("bytes")) is not int or item["bytes"] < 0:
                raise ValueError(f"chunk bytes must be a nonnegative integer: {path}")
            if not _is_sha256(item.get("sha256")):
                raise ValueError(f"chunk SHA256 is invalid: {path}")
        if values["debug_path"] != "debug.json":
            raise ValueError("debug_path must be debug.json")
        if type(values["debug_bytes"]) is not int or values["debug_bytes"] < 0:
            raise ValueError("debug_bytes must be a nonnegative integer")
        if not _is_sha256(values["debug_sha256"]):
            raise ValueError("debug_sha256 is invalid")
        if values["artifact_manifest_path"] != "artifact_manifest.json":
            raise ValueError("artifact_manifest_path must be artifact_manifest.json")

        specs = dict(values["array_specs"])
        member_refs: set[tuple[str, str]] = set()
        for name, spec in specs.items():
            if not isinstance(name, str) or not _SAFE_ARRAY_NAME.fullmatch(name):
                raise ValueError(f"invalid archive array name: {name!r}")
            if not isinstance(spec, Mapping):
                raise ValueError(f"array spec must be an object: {name}")
            dtype = np.dtype(spec.get("dtype"))
            if dtype.hasobject or dtype.kind not in "biufc":
                raise ValueError(f"array dtype is unsafe: {name}")
            shape = spec.get("shape")
            if not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape):
                raise ValueError(f"array shape is invalid: {name}")
            if not isinstance(spec.get("semantic_role"), str) or not spec["semantic_role"]:
                raise ValueError(f"array semantic_role is required: {name}")
            if type(spec.get("byte_count")) is not int or spec["byte_count"] < 0:
                raise ValueError(f"array byte_count is invalid: {name}")
            if not _is_sha256(spec.get("sha256")):
                raise ValueError(f"array SHA256 is invalid: {name}")
            pieces = spec.get("pieces")
            if not isinstance(pieces, list) or not pieces:
                raise ValueError(f"array pieces are required: {name}")
            for piece in pieces:
                if not isinstance(piece, Mapping):
                    raise ValueError(f"array piece must be an object: {name}")
                unknown_piece = set(piece) - _PIECE_FIELDS
                if unknown_piece:
                    raise ValueError(f"array piece has unknown field(s): {', '.join(sorted(unknown_piece))}")
                path = _require_safe_relative(piece.get("path"), "array piece path")
                if path not in chunk_paths:
                    raise ValueError(f"array piece references missing chunk: {name}")
                if not isinstance(piece.get("key"), str) or not _SAFE_NPZ_MEMBER.fullmatch(piece["key"] + ".npy"):
                    raise ValueError(f"array piece key is required: {name}")
                member_ref = (path, piece["key"])
                if member_ref in member_refs:
                    raise ValueError(f"array piece is referenced more than once: {name}")
                member_refs.add(member_ref)
                piece_shape = piece.get("shape")
                if not isinstance(piece_shape, list) or any(type(item) is not int or item < 0 for item in piece_shape):
                    raise ValueError(f"array piece shape is invalid: {name}")
                if type(piece.get("byte_count")) is not int or piece["byte_count"] < 0:
                    raise ValueError(f"array piece byte_count is invalid: {name}")
                expected_bytes = math.prod(piece_shape) * dtype.itemsize
                if piece["byte_count"] != expected_bytes:
                    raise ValueError(f"array piece byte count disagrees with shape: {name}")
                if not _is_sha256(piece.get("sha256")):
                    raise ValueError(f"array piece SHA256 is invalid: {name}")
                if len(piece_shape) != len(shape) or piece_shape[1:] != shape[1:]:
                    raise ValueError(f"array piece rank or trailing shape mismatch: {name}")
                for start_key, stop_key in (
                    ("boundary_start", "boundary_stop"),
                    ("point_start", "point_stop"),
                    ("transition_start", "transition_stop"),
                    ("timeline_start", "timeline_stop"),
                    ("array_start", "array_stop"),
                ):
                    has_start, has_stop = start_key in piece, stop_key in piece
                    if has_start != has_stop:
                        raise ValueError(f"array piece has an incomplete {start_key[:-6]} range: {name}")
                    if has_start and (
                        type(piece[start_key]) is not int
                        or type(piece[stop_key]) is not int
                        or piece[start_key] < 0
                        or piece[stop_key] <= piece[start_key]
                    ):
                        raise ValueError(f"array piece has an invalid {start_key[:-6]} range: {name}")
                for field in ("validity_point_count", "timeline_index"):
                    if field in piece and (type(piece[field]) is not int or piece[field] < 0):
                        raise ValueError(f"array piece {field} is invalid: {name}")
            if len(shape) == 0 and len(pieces) != 1:
                raise ValueError(f"scalar archive arrays must have exactly one piece: {name}")
            if type(spec["byte_count"]) is int:
                expected_bytes = math.prod(shape) * dtype.itemsize
                if spec["byte_count"] != expected_bytes:
                    raise ValueError(f"array byte count disagrees with shape: {name}")
        aliases: dict[str, str] = {}
        for alias in values["array_aliases"]:
            if not isinstance(alias, Mapping):
                raise ValueError("array_aliases entries must be objects")
            name, target = alias.get("name"), alias.get("target")
            if not isinstance(name, str) or not _SAFE_ARRAY_NAME.fullmatch(name):
                raise ValueError("array alias name is invalid")
            if not isinstance(target, str) or target not in specs:
                raise ValueError(f"array alias target is missing: {target!r}")
            if name in specs or name in aliases:
                raise ValueError(f"array alias name is duplicated: {name}")
            if alias.get("dtype") != specs[target]["dtype"] or alias.get("shape") != specs[target]["shape"]:
                raise ValueError(f"array alias target metadata mismatch: {name}")
            if alias.get("sha256") != specs[target]["sha256"]:
                raise ValueError(f"array alias digest mismatch: {name}")
            if not isinstance(alias.get("semantic_role"), str) or not alias["semantic_role"]:
                raise ValueError(f"array alias semantic_role is required: {name}")
            target_role = specs[target]["semantic_role"]
            alias_role = alias["semantic_role"]
            explicit_measured_alias = (
                target_role == "online_observations/points"
                and alias_role == "raw_arrays/measured_points"
            )
            if alias_role != target_role and not explicit_measured_alias:
                raise ValueError(f"array alias crosses incompatible semantic roles: {name}")
            if alias.get("pieces") != specs[target]["pieces"]:
                raise ValueError(f"array alias logical piece mapping mismatch: {name}")
            aliases[name] = target
        known_array_names = set(specs) | set(aliases)
        for section in (values["record_metadata"], values["raw_arrays"]):
            for reference in _array_references(section):
                if reference not in known_array_names:
                    raise ValueError(f"archive array reference is missing: {reference}")
        object.__setattr__(self, "payload", values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchiveManifest":
        if not isinstance(payload, Mapping):
            raise ValueError("archive manifest must be an object")
        return cls(dict(payload))

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dict(self.payload), sort_keys=True, allow_nan=False))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def archive_profile_from_environment(environment: Mapping[str, str] | None = None) -> ArchiveProfileConfig | None:
    """Decode the opt-in worker profile without changing legacy callers."""
    values = os.environ if environment is None else environment
    raw = values.get("ICGS_GENERATION_ARCHIVE_PROFILE", "")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("ICGS_GENERATION_ARCHIVE_PROFILE must contain a JSON object") from error
    return ArchiveProfileConfig.from_dict(payload)


def archive_writer_cap_from_environment(environment: Mapping[str, str] | None = None) -> int | None:
    values = os.environ if environment is None else environment
    raw = values.get("ICGS_GENERATION_MAX_RESULT_BYTES")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal() or int(raw) <= 0:
        raise ValueError("ICGS_GENERATION_MAX_RESULT_BYTES must be a positive integer")
    return int(raw)


class _ArchiveArrays:
    def __init__(self, scratch_root: Path, *, max_chunk_bytes: int, budget: _WriterByteBudget | None = None) -> None:
        self.scratch_root = scratch_root
        if self.scratch_root.is_symlink() or not self.scratch_root.is_dir() or any(self.scratch_root.iterdir()):
            raise ValueError("archive scratch directory must be a new, empty real directory")
        self.max_chunk_bytes = max_chunk_bytes
        self.budget = budget
        self.chunk_files: dict[int, list[tuple[str, Path]]] = {}
        self.chunk_data_bytes: dict[int, int] = {}
        self.spool_bytes = 0
        self.spool_peak_bytes = 0
        self.specs: dict[str, dict[str, Any]] = {}
        self.aliases: list[dict[str, Any]] = []
        self.signatures: dict[tuple[str, str, tuple[int, ...], str, str], str] = {}
        self.counter = 0

    def ensure_chunk_capacity(self, chunk_index: int, additional_bytes: int) -> None:
        if additional_bytes < 0:
            raise ValueError("chunk byte estimate must be nonnegative")
        used = self.chunk_data_bytes.get(chunk_index, 0)
        if used + additional_bytes > self.max_chunk_bytes:
            raise ValueError(
                f"uncompressed archive chunk including .npy headers exceeds max_chunk_bytes "
                f"({used + additional_bytes} > {self.max_chunk_bytes})"
            )

    @staticmethod
    def _alias_role(semantic_role: str) -> str:
        # These two names are the one explicit cross-role alias accepted by ADR0015.
        if semantic_role in {"online_observations/points", "raw_arrays/measured_points"}:
            return "lossless_measured_point_values"
        return semantic_role

    def _alias(self, name: str, target: str, dtype: np.dtype, shape: Sequence[int], digest: str, semantic_role: str) -> str:
        target_spec = self.specs[target]
        self.aliases.append({
            "name": name,
            "target": target,
            "dtype": dtype.str,
            "shape": [int(item) for item in shape],
            "sha256": digest,
            "semantic_role": semantic_role,
            "pieces": [dict(piece) for piece in target_spec["pieces"]],
        })
        return name

    @staticmethod
    def _range_signature(pieces: Sequence[Mapping[str, Any]]) -> str:
        range_keys = (
            "boundary_start", "boundary_stop", "point_start", "point_stop", "validity_point_count",
            "transition_start", "transition_stop", "timeline_start", "timeline_stop",
            "array_start", "array_stop", "timeline_index",
        )
        ranges = [{key: piece[key] for key in range_keys if key in piece} for piece in pieces]
        return _canonical_json(ranges).decode("utf-8")

    def _measured_point_alias_target(self, semantic_role: str, signature: tuple[str, str, tuple[int, ...], str, str]) -> str | None:
        if semantic_role != "raw_arrays/measured_points":
            return None
        content_signature = signature[:4]
        for candidate, target in self.signatures.items():
            if candidate[:4] == content_signature and self.specs[target]["semantic_role"] == "online_observations/points":
                return target
        return None

    def _store_array_file(self, key: str, chunk_index: int, value: np.ndarray) -> Path:
        chunk_root = self.scratch_root / f"chunk-{chunk_index:05d}"
        chunk_root.mkdir(parents=True, exist_ok=True)
        path = chunk_root / f"{key}.npy"
        with (self.budget.open(path) if self.budget is not None else path.open("wb")) as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            # This is temporary spool input. The durable boundary is the final
            # compressed NPZ chunk below; syncing every member makes large
            # episodes serialize thousands of journal commits.
        self.spool_bytes += path.stat().st_size
        self.spool_peak_bytes = max(self.spool_peak_bytes, self.spool_bytes)
        return path

    def add(
        self,
        name: str,
        dtype: np.dtype,
        shape: Sequence[int],
        semantic_role: str,
        pieces: Iterable[tuple[int, np.ndarray, Mapping[str, int]]],
    ) -> str:
        if name in self.specs or any(item["name"] == name for item in self.aliases):
            raise ValueError(f"duplicate logical archive array: {name}")
        digest_builder = hashlib.sha256()
        digest_builder.update(dtype.str.encode("ascii"))
        digest_builder.update(b"\0")
        digest_builder.update(_canonical_json([int(item) for item in shape]))
        byte_count = 0
        piece_specs: list[dict[str, Any]] = []
        stored_files: list[tuple[int, str, Path, int]] = []
        try:
            for piece_index, (chunk_index, raw_value, ranges) in enumerate(pieces):
                value = np.asarray(raw_value)
                if value.dtype.hasobject:
                    raise ValueError(f"object arrays are forbidden: {semantic_role}")
                if value.dtype.kind not in "biufc":
                    raise ValueError(f"unsupported array dtype at {semantic_role}: {value.dtype}")
                if value.dtype != dtype:
                    raise ValueError(f"array dtype changed between chunks at {semantic_role}")
                member_bytes = _npy_member_uncompressed_bytes(value)
                self.ensure_chunk_capacity(chunk_index, member_bytes)
                key = f"a{self.counter:06d}_{piece_index:03d}"
                self.counter += 1
                piece_digest = _array_digest(value.dtype, value.shape, [value])
                _update_array_digest(digest_builder, value)
                byte_count += int(value.nbytes)
                path = self._store_array_file(key, chunk_index, value)
                stored_files.append((chunk_index, key, path, member_bytes))
                self.chunk_data_bytes[chunk_index] = self.chunk_data_bytes.get(chunk_index, 0) + member_bytes
                piece_specs.append({
                    "key": key,
                    "chunk_index": chunk_index,
                    "shape": [int(item) for item in value.shape],
                    "byte_count": int(value.nbytes),
                    "sha256": piece_digest,
                    **dict(ranges),
                })
        except Exception as error:
            if not isinstance(error, ArchiveWriterCapExceeded):
                self._discard_files(stored_files)
            raise
        if not piece_specs:
            raise ValueError(f"archive array has no payload pieces: {semantic_role}")
        digest = digest_builder.hexdigest()
        normalized_shape = tuple(int(item) for item in shape)
        signature = (
            self._alias_role(semantic_role),
            dtype.str,
            normalized_shape,
            digest,
            self._range_signature(piece_specs),
        )
        existing = self.signatures.get(signature)
        if existing is None:
            existing = self._measured_point_alias_target(semantic_role, signature)
        if existing is not None:
            self._discard_files(stored_files)
            return self._alias(name, existing, dtype, shape, digest, semantic_role)
        for chunk_index, key, path, _data_bytes in stored_files:
            self.chunk_files.setdefault(chunk_index, []).append((key, path))
        self.specs[name] = {
            "dtype": dtype.str,
            "shape": [int(item) for item in shape],
            "semantic_role": semantic_role,
            "byte_count": byte_count,
            "sha256": digest,
            "pieces": piece_specs,
        }
        self.signatures[signature] = name
        return name

    def add_array(
        self,
        name: str,
        value: np.ndarray,
        semantic_role: str,
        *,
        chunk_index: int = 0,
        ranges: Mapping[str, int] | None = None,
    ) -> str:
        if value.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden: {semantic_role}")
        array = np.asarray(value)
        digest = _array_digest(array.dtype, array.shape, [array])
        normalized_ranges = ranges or {}
        signature = (
            self._alias_role(semantic_role),
            array.dtype.str,
            tuple(int(item) for item in array.shape),
            digest,
            self._range_signature([normalized_ranges]),
        )
        existing = self.signatures.get(signature)
        if existing is None:
            existing = self._measured_point_alias_target(semantic_role, signature)
        if existing is not None:
            return self._alias(name, existing, array.dtype, array.shape, digest, semantic_role)
        return self.add(
            name,
            array.dtype,
            array.shape,
            semantic_role,
            [(chunk_index, array, normalized_ranges)],
        )

    def _discard_files(self, files: Sequence[tuple[int, str, Path, int]]) -> None:
        for chunk_index, _key, path, data_bytes in files:
            try:
                file_bytes = path.stat().st_size
            except FileNotFoundError:
                file_bytes = 0
            path.unlink(missing_ok=True)
            self.spool_bytes -= file_bytes
            if self.budget is not None:
                self.budget.used -= file_bytes
            self.chunk_data_bytes[chunk_index] = max(0, self.chunk_data_bytes.get(chunk_index, 0) - data_bytes)

    def write_chunks(self, destination: Path) -> list[dict[str, Any]]:
        inventory = []
        for chunk_index in sorted(self.chunk_files):
            relative = f"data/chunk-{chunk_index:05d}.npz"
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with (self.budget.open(path) if self.budget is not None else path.open("wb")) as stream:
                    archive = zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED)
                    try:
                        for key, source in sorted(self.chunk_files[chunk_index]):
                            try:
                                value = np.load(source, allow_pickle=False, mmap_mode="r")
                            except ValueError:
                                # NumPy cannot mmap zero-dimensional .npy scalars; these are tiny metadata values.
                                value = np.load(source, allow_pickle=False)
                            try:
                                with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                                    np.lib.format.write_array(member, value, allow_pickle=False)
                            finally:
                                mmap = getattr(value, "_mmap", None)
                                if mmap is not None:
                                    mmap.close()
                        archive.close()
                    except ArchiveWriterCapExceeded:
                        # A ZIP central directory would add forbidden bytes; leave the
                        # partial chunk as forensic scratch without a destructor retry.
                        archive.fp = None
                        raise
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as error:
                if not isinstance(error, ArchiveWriterCapExceeded):
                    path.unlink(missing_ok=True)
                raise
            _check_npz_uncompressed_cap(
                path,
                max_chunk_bytes=self.max_chunk_bytes,
                relative=f"data/chunk-{chunk_index:05d}.npz",
            )
            inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        return inventory

    def specs_as_dict(self, chunk_paths: Mapping[int, str]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for name, spec in sorted(self.specs.items()):
            normalized = dict(spec)
            normalized_pieces = []
            for piece in spec["pieces"]:
                item = dict(piece)
                chunk_index = item.pop("chunk_index")
                item["path"] = chunk_paths[chunk_index]
                normalized_pieces.append(item)
            normalized["pieces"] = normalized_pieces
            output[name] = normalized
        return output

    def aliases_as_list(self, chunk_paths: Mapping[int, str]) -> list[dict[str, Any]]:
        output = []
        for alias in self.aliases:
            normalized = dict(alias)
            pieces = []
            for piece in alias["pieces"]:
                item = dict(piece)
                chunk_index = item.pop("chunk_index")
                item["path"] = chunk_paths[chunk_index]
                pieces.append(item)
            normalized["pieces"] = pieces
            output.append(normalized)
        return sorted(output, key=lambda item: item["name"])

    def cleanup(self) -> None:
        shutil.rmtree(self.scratch_root, ignore_errors=True)


class _MetadataEncoder:
    def __init__(
        self,
        arrays: _ArchiveArrays,
        *,
        transition_count: int,
        observation_count: int,
        chunk_boundaries: int,
    ) -> None:
        self.arrays = arrays
        self.transition_count = transition_count
        self.observation_count = observation_count
        self.chunk_boundaries = chunk_boundaries
        self.counter = 0

    def _next_name(self) -> str:
        name = f"a{self.counter + 100000:06d}"
        self.counter += 1
        return name

    @staticmethod
    def _indexed_path(path: tuple[str, ...], roots: frozenset[str]) -> int | None:
        if len(path) < 2 or path[0] not in roots:
            return None
        try:
            return int(path[1])
        except ValueError:
            return None

    def _add_general_array(self, array: np.ndarray, path: tuple[str, ...], original: Any) -> dict[str, str]:
        name = self._next_name()
        indexed = self._indexed_path(
            path,
            frozenset({"robot_states", "object_states", "event_states", "events", "timestamps"}),
        )
        if indexed is not None:
            chunk = indexed // self.chunk_boundaries
            self.arrays.add_array(name, array, "/".join(path), chunk_index=chunk, ranges={"timeline_index": indexed})
        elif array.ndim > 0 and array.shape[0] in {self.transition_count, self.observation_count}:
            self.arrays.add(
                name,
                array.dtype,
                array.shape,
                "/".join(path),
                _series_pieces(array, chunk_boundaries=self.chunk_boundaries, index_key="array"),
            )
        else:
            self.arrays.add_array(name, array, "/".join(path), chunk_index=0)
        reference: dict[str, str] = {"$archive_array": name}
        if isinstance(original, tuple):
            reference["$container"] = "tuple"
        elif isinstance(original, list):
            reference["$container"] = "list"
        return reference

    def encode(self, value: Any, path: tuple[str, ...]) -> Any:
        array = _as_numeric_array(value, path="/".join(path))
        if array is not None:
            return self._add_general_array(array, path, value)
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key in sorted(value, key=lambda item: str(item)):
                name = str(key)
                result[name] = self.encode(value[key], path + (name,))
            return result
        if isinstance(value, (list, tuple)):
            return [self.encode(item, path + (str(index),)) for index, item in enumerate(value)]
        return _json_scalar(value, path="/".join(path))


def _sequence_array_metadata(
    values: Sequence[Any],
    *,
    name: str,
    element_shape: tuple[int, ...] | None = None,
) -> tuple[np.dtype, tuple[int, ...]]:
    arrays = [np.asarray(item) for item in values]
    if not arrays:
        raise ValueError(f"{name} must contain at least one value")
    dtype = arrays[0].dtype
    if dtype.hasobject or dtype.kind not in "biufc":
        raise ValueError(f"{name} must use a numeric or boolean dtype")
    if any(item.dtype != dtype for item in arrays):
        raise ValueError(f"{name} must preserve a single source dtype")
    expected_shape = element_shape if element_shape is not None else arrays[0].shape
    if any(item.shape != expected_shape for item in arrays):
        raise ValueError(f"{name} entries must have shape {expected_shape}")
    return dtype, (len(arrays), *expected_shape)


def _sequence_pieces(
    values: Sequence[Any],
    *,
    chunk_boundaries: int,
    index_key: str,
) -> Iterator[tuple[int, np.ndarray, Mapping[str, int]]]:
    for start in range(0, len(values), chunk_boundaries):
        stop = min(len(values), start + chunk_boundaries)
        yield (
            start // chunk_boundaries,
            np.stack([np.asarray(item) for item in values[start:stop]], axis=0),
            {f"{index_key}_start": start, f"{index_key}_stop": stop},
        )


def _series_pieces(
    array: np.ndarray,
    *,
    chunk_boundaries: int,
    index_key: str,
) -> Iterator[tuple[int, np.ndarray, Mapping[str, int]]]:
    for start in range(0, int(array.shape[0]), chunk_boundaries):
        stop = min(int(array.shape[0]), start + chunk_boundaries)
        yield (start // chunk_boundaries, array[start:stop], {f"{index_key}_start": start, f"{index_key}_stop": stop})


class EpisodeArchiveWriter:
    """Write one lossless episode/attempt archive using bounded NPZ chunks."""

    def __init__(self, profile: ArchiveProfileConfig, *, max_result_bytes: int | None = None) -> None:
        if not isinstance(profile, ArchiveProfileConfig):
            raise TypeError("profile must be an ArchiveProfileConfig")
        if max_result_bytes is not None and (type(max_result_bytes) is not int or max_result_bytes <= 0):
            raise ValueError("max_result_bytes must be a positive integer")
        self.profile = profile
        self.max_result_bytes = max_result_bytes

    def _new_array_store(self, output_dir: str | Path) -> _ArchiveArrays:
        target = Path(output_dir).absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        budget = (
            _WriterByteBudget(target.parent, self.max_result_bytes)
            if self.max_result_bytes is not None else None
        )
        scratch = Path(tempfile.mkdtemp(
            prefix=f".{target.name}.archive-spool-",
            dir=target.parent,
        ))
        return _ArchiveArrays(scratch, max_chunk_bytes=self.profile.max_chunk_bytes, budget=budget)

    def write_episode(
        self,
        record: Mapping[str, Any],
        *,
        raw_arrays: Mapping[str, Any],
        debug_metadata: Mapping[str, Any],
        output_dir: str | Path,
    ) -> ArchiveManifest:
        arrays = self._new_array_store(output_dir)
        try:
            result = self._write_episode_body(
                record,
                raw_arrays=raw_arrays,
                debug_metadata=debug_metadata,
                output_dir=output_dir,
                arrays=arrays,
            )
        except ArchiveWriterCapExceeded:
            raise
        except BaseException:
            arrays.cleanup()
            raise
        else:
            arrays.cleanup()
            return result

    def _write_episode_body(
        self,
        record: Mapping[str, Any],
        *,
        raw_arrays: Mapping[str, Any],
        debug_metadata: Mapping[str, Any],
        output_dir: str | Path,
        arrays: _ArchiveArrays,
    ) -> ArchiveManifest:
        validate_episode(record)
        outcome = record["provenance"]["outcome"]
        if outcome not in {"success", "valid_failure"}:
            raise ValueError("crash and invalid observation must be written as attempt archives")
        observations = record["online_observations"]
        transitions = record["transitions"]
        if not isinstance(raw_arrays, Mapping) or not isinstance(debug_metadata, Mapping):
            raise ValueError("raw_arrays and debug_metadata must be mappings")
        debug = dict(debug_metadata)
        identity = self._identity(debug, record["provenance"])
        boundaries = self.profile.chunk_boundaries
        n_transitions = len(transitions)
        n_observations = len(observations)
        if n_observations != n_transitions + 1:
            raise ValueError("archive episodes require T+1 observations for T transitions")

        point_arrays: list[np.ndarray] = []
        validity_arrays: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        grips: list[np.ndarray] = []
        offsets = [0]
        for index, observation in enumerate(observations):
            points = np.asarray(observation["points"])
            validity = np.asarray(observation["point_valid"])
            pose = np.asarray(observation["T_w_e"])
            grip = np.asarray(observation["grip"])
            if points.dtype.hasobject or points.dtype.kind not in "f" or points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"online observation {index} points must be a floating Nx3 array")
            if len(points) == 0:
                raise ValueError(f"online observation {index} points cannot be empty")
            if not np.isfinite(points).all():
                raise ValueError(f"online observation {index} points contain non-finite values")
            if validity.dtype != np.bool_ or validity.shape != (len(points),):
                raise ValueError(f"online observation {index} point_valid must be boolean with one value per point")
            if pose.dtype.hasobject or pose.dtype.kind not in "f" or pose.shape != (4, 4):
                raise ValueError(f"online observation {index} T_w_e must be a floating 4x4 array")
            if grip.ndim != 0 or grip.dtype.hasobject or grip.dtype.kind not in "biuf":
                raise ValueError(f"online observation {index} grip must be a numeric scalar")
            point_arrays.append(points)
            validity_arrays.append(validity)
            poses.append(pose)
            grips.append(grip)
            offsets.append(offsets[-1] + len(points))
        point_dtype = point_arrays[0].dtype
        validity_dtype = validity_arrays[0].dtype
        pose_dtype = poses[0].dtype
        grip_dtype = grips[0].dtype
        for name, values, dtype in (
            ("online points", point_arrays, point_dtype),
            ("online point validity", validity_arrays, validity_dtype),
            ("online poses", poses, pose_dtype),
            ("online grips", grips, grip_dtype),
        ):
            if any(item.dtype != dtype for item in values):
                raise ValueError(f"{name} must preserve a single source dtype")

        def online_point_pieces() -> Iterator[tuple[int, np.ndarray, Mapping[str, int]]]:
            for start in range(0, n_observations, boundaries):
                stop = min(n_observations, start + boundaries)
                chunk_index = start // boundaries
                additional = sum(int(item.nbytes) for item in point_arrays[start:stop])
                arrays.ensure_chunk_capacity(chunk_index, additional)
                block_points = np.concatenate(point_arrays[start:stop], axis=0)
                yield chunk_index, block_points, {
                    "boundary_start": start,
                    "boundary_stop": stop,
                    "point_start": offsets[start],
                    "point_stop": offsets[stop],
                }

        def online_validity_pieces() -> Iterator[tuple[int, np.ndarray, Mapping[str, int]]]:
            for start in range(0, n_observations, boundaries):
                stop = min(n_observations, start + boundaries)
                chunk_index = start // boundaries
                validity_count = sum(len(item) for item in validity_arrays[start:stop])
                packed_bytes = (validity_count + 7) // 8
                arrays.ensure_chunk_capacity(chunk_index, packed_bytes)
                block_validity = np.concatenate(validity_arrays[start:stop], axis=0)
                packed = np.packbits(block_validity, bitorder="little")
                yield chunk_index, packed, {
                    "boundary_start": start,
                    "boundary_stop": stop,
                    "point_start": offsets[start],
                    "point_stop": offsets[stop],
                    "validity_point_count": validity_count,
                }

        arrays.add("online_points", point_dtype, (offsets[-1], 3), "online_observations/points", online_point_pieces())
        arrays.add(
            "online_validity_packed",
            np.dtype(np.uint8),
            (sum((offsets[stop] - offsets[start] + 7) // 8 for start in range(0, n_observations, boundaries) for stop in [min(n_observations, start + boundaries)]),),
            "online_observations/point_valid_bitpacked_little",
            online_validity_pieces(),
        )
        arrays.add(
            "online_points_offsets",
            np.dtype(np.int64),
            (n_observations + 1,),
            "online_observations/point_offsets",
            [(0, np.asarray(offsets, dtype=np.int64), {})],
        )
        arrays.add(
            "online_poses",
            pose_dtype,
            (n_observations, 4, 4),
            "online_observations/T_w_e",
            _sequence_pieces(poses, chunk_boundaries=boundaries, index_key="boundary"),
        )
        arrays.add(
            "online_grips",
            grip_dtype,
            (n_observations,),
            "online_observations/grip",
            _sequence_pieces(grips, chunk_boundaries=boundaries, index_key="boundary"),
        )

        command_poses: list[np.ndarray] = []
        command_grips: list[np.ndarray] = []
        command_durations: list[np.ndarray] = []
        achieved_durations: list[np.ndarray] = []
        substeps: list[np.ndarray] = []
        transition_extras: list[dict[str, Any]] = []
        transition_count = len(transitions)
        for index, transition in enumerate(transitions):
            command = transition.get("command")
            if not isinstance(command, Mapping) or "T_w_e" not in command or "grip" not in command:
                raise ValueError(f"transition {index} must contain command.T_w_e and command.grip")
            command_poses.append(np.asarray(command["T_w_e"]))
            command_grips.append(np.asarray(command["grip"]))
            command_durations.append(np.asarray(command.get("duration_s", transition["achieved_duration_s"])))
            achieved_durations.append(np.asarray(transition["achieved_duration_s"]))
            substeps.append(np.asarray(transition["physics_substeps"]))
            extras = {key: value for key, value in transition.items() if key not in {"command", "achieved_duration_s", "physics_substeps"}}
            extras["command"] = {key: value for key, value in command.items() if key not in {"T_w_e", "grip", "duration_s"}}
            transition_extras.append(extras)
        sequence_specs = {
            "commands": _sequence_array_metadata(command_poses, name="commands", element_shape=(4, 4)),
            "command_grips": _sequence_array_metadata(command_grips, name="command_grips", element_shape=()),
            "command_durations": _sequence_array_metadata(command_durations, name="command_durations", element_shape=()),
            "achieved_durations": _sequence_array_metadata(achieved_durations, name="achieved_durations", element_shape=()),
            "substeps": _sequence_array_metadata(substeps, name="substeps", element_shape=()),
        }
        dt_source = np.asarray(record.get("dt", [item.item() for item in achieved_durations]))
        if dt_source.dtype.hasobject or dt_source.ndim != 1 or dt_source.shape[0] != transition_count:
            raise ValueError("dt must have T numeric values")
        if dt_source.dtype.kind not in "biuf" or not np.isfinite(dt_source).all():
            raise ValueError("dt must contain finite numeric values")
        transition_sequences = {
            "commands": command_poses,
            "command_grips": command_grips,
            "command_durations": command_durations,
            "achieved_durations": achieved_durations,
            "substeps": substeps,
        }
        for name, sequence in transition_sequences.items():
            dtype, shape = sequence_specs[name]
            arrays.add(
                name,
                dtype,
                shape,
                f"transitions/{name}",
                _sequence_pieces(sequence, chunk_boundaries=boundaries, index_key="transition"),
            )
        arrays.add(
            "dt",
            dt_source.dtype,
            dt_source.shape,
            "transitions/dt",
            _series_pieces(dt_source, chunk_boundaries=boundaries, index_key="transition"),
        )

        encoder = _MetadataEncoder(
            arrays,
            transition_count=transition_count,
            observation_count=n_observations,
            chunk_boundaries=boundaries,
        )
        record_metadata: dict[str, Any] = {}
        for key in sorted(record):
            if key == "online_observations":
                record_metadata[key] = {"$archive_series": "observations"}
            elif key == "transitions":
                record_metadata[key] = {"$archive_series": "transitions"}
            elif key == "dt":
                record_metadata[key] = {
                    "$archive_array": "dt",
                    "$container": "tuple" if isinstance(record[key], tuple) else "list" if isinstance(record[key], list) else "ndarray",
                }
            else:
                record_metadata[key] = encoder.encode(record[key], (key,))
        for index, extras in enumerate(transition_extras):
            transition_extras[index] = encoder.encode(extras, ("transition_metadata", str(index)))
        record_metadata["_archive_transition_metadata"] = transition_extras
        encoded_raw_arrays: dict[str, Any] = {}
        for key in sorted(raw_arrays, key=str):
            name = str(key)
            encoded_raw_arrays[name] = self._encode_raw_value(
                raw_arrays[key],
                name=name,
                arrays=arrays,
                transition_count=transition_count,
                observation_count=n_observations,
                chunk_boundaries=boundaries,
                encoder=encoder,
            )
        encoded_debug = encoder.encode(debug, ("debug",))
        return self._write_archive(
            target=Path(output_dir),
            archive_kind="episode",
            episode_id=str(record["provenance"]["episode_id"]),
            attempt_id=str(record["provenance"].get("attempt_id") or f"att-{record['provenance']['episode_id']}"),
            program_id=str(record["provenance"]["program_id"]),
            split=str(record["provenance"]["split"]),
            subset=record["provenance"].get("subset"),
            outcome=outcome,
            identity=identity,
            timeline={"observations": n_observations, "transitions": transition_count},
            record_metadata=record_metadata,
            raw_arrays=encoded_raw_arrays,
            debug_metadata=encoded_debug,
            arrays=arrays,
        )

    def write_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        prefix_arrays: Mapping[str, Any],
        debug_metadata: Mapping[str, Any],
        output_dir: str | Path,
    ) -> ArchiveManifest:
        arrays = self._new_array_store(output_dir)
        try:
            result = self._write_attempt_body(
                attempt,
                prefix_arrays=prefix_arrays,
                debug_metadata=debug_metadata,
                output_dir=output_dir,
                arrays=arrays,
            )
        except ArchiveWriterCapExceeded:
            raise
        except BaseException:
            arrays.cleanup()
            raise
        else:
            arrays.cleanup()
            return result

    def _write_attempt_body(
        self,
        attempt: Mapping[str, Any],
        *,
        prefix_arrays: Mapping[str, Any],
        debug_metadata: Mapping[str, Any],
        output_dir: str | Path,
        arrays: _ArchiveArrays,
    ) -> ArchiveManifest:
        if not isinstance(attempt, Mapping) or not isinstance(prefix_arrays, Mapping) or not isinstance(debug_metadata, Mapping):
            raise ValueError("attempt, prefix_arrays and debug_metadata must be mappings")
        attempt = _redact_sensitive(dict(attempt))
        prefix_arrays = _redact_sensitive(dict(prefix_arrays))
        outcome = attempt.get("outcome")
        if outcome not in {"simulator_crash", "invalid_observation"}:
            raise ValueError("write_attempt requires simulator_crash or invalid_observation")
        if attempt.get("episode_id") is not None:
            raise ValueError("attempt archives require a null episode_id")
        attempt_id = attempt.get("attempt_id")
        program_id = attempt.get("program_id")
        if not isinstance(attempt_id, str) or not attempt_id.strip() or not isinstance(program_id, str) or not program_id.strip():
            raise ValueError("attempt_id and program_id are required")
        debug = _redact_sensitive(dict(debug_metadata))
        for field in ("error_type", "error", "traceback", "terminal_reason", "exit_code", "timeout", "stderr", "stdout"):
            if field not in debug and field in attempt:
                debug[field] = attempt[field]
        debug = _redact_sensitive(debug)
        identity = self._identity(debug, attempt)

        valid_until = attempt.get("valid_observation_until")
        if valid_until is not None and (type(valid_until) is not int or valid_until < 0):
            raise ValueError("valid_observation_until must be a nonnegative integer or null")
        if ("points" in prefix_arrays) != ("point_offsets" in prefix_arrays):
            raise ValueError("attempt prefix points and point_offsets must be provided together")
        observation_count = 0
        for key in ("observations", "gripper_pose", "T_w_e"):
            value = prefix_arrays.get(key)
            if value is not None:
                candidate = np.asarray(value)
                if candidate.ndim > 0:
                    observation_count = int(candidate.shape[0])
                    break
        point_offsets = prefix_arrays.get("point_offsets")
        if observation_count == 0 and point_offsets is not None:
            offsets = np.asarray(point_offsets)
            if offsets.ndim == 1 and len(offsets) > 0:
                observation_count = int(offsets.shape[0]) - 1
        if observation_count == 0 and valid_until is not None:
            observation_count = valid_until + 1
        if valid_until is not None and observation_count != valid_until + 1:
            raise ValueError("valid_observation_until must identify the final archived observation boundary")
        if valid_until is not None and not any(
            key in prefix_arrays for key in ("observations", "gripper_pose", "T_w_e", "points", "point_offsets")
        ):
            raise ValueError("valid_observation_until requires archived measured prefix arrays")
        action_values = prefix_arrays.get("actions")
        action_array = None if action_values is None else np.asarray(action_values)
        transition_count = (
            int(action_array.shape[0])
            if action_array is not None and action_array.ndim > 0
            else max(0, observation_count - 1)
        )
        encoder = _MetadataEncoder(
            arrays,
            transition_count=transition_count,
            observation_count=observation_count,
            chunk_boundaries=self.profile.chunk_boundaries,
        )
        encoded_attempt = encoder.encode(dict(attempt), ("attempt",))
        encoded_prefix: dict[str, Any] = {}
        for key in sorted(prefix_arrays, key=str):
            encoded_prefix[str(key)] = self._encode_raw_value(
                prefix_arrays[key],
                name=f"prefix_{key}",
                arrays=arrays,
                transition_count=transition_count,
                observation_count=observation_count,
                chunk_boundaries=self.profile.chunk_boundaries,
                encoder=encoder,
            )
        encoded_debug = encoder.encode(debug, ("debug",))
        return self._write_archive(
            target=Path(output_dir),
            archive_kind="attempt",
            episode_id=None,
            attempt_id=attempt_id,
            program_id=program_id,
            split=attempt.get("split"),
            subset=attempt.get("subset"),
            outcome=str(outcome),
            identity=identity,
            timeline={
                "observations": observation_count,
                "transitions": transition_count,
                "valid_observation_until": valid_until,
            },
            record_metadata={"attempt": encoded_attempt},
            raw_arrays=encoded_prefix,
            debug_metadata=encoded_debug,
            arrays=arrays,
        )

    @staticmethod
    def _identity(debug: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, str]:
        source_run_id = debug.get("source_run_id") or provenance.get("source_run_id") or provenance.get("run_id")
        code_revision = debug.get("code_revision") or provenance.get("code_revision")
        preprocessing_identity = debug.get("preprocessing_identity") or provenance.get("preprocessing_identity") or "measured_raw_v1"
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise ValueError("debug_metadata.source_run_id is required for archive identity")
        if not isinstance(code_revision, str) or not code_revision.strip():
            raise ValueError("debug_metadata.code_revision is required for archive identity")
        if not isinstance(preprocessing_identity, str) or not preprocessing_identity.strip():
            raise ValueError("preprocessing_identity is required for archive identity")
        return {
            "source_run_id": source_run_id,
            "code_revision": code_revision,
            "preprocessing_identity": preprocessing_identity,
        }

    @staticmethod
    def _encode_raw_value(
        value: Any,
        *,
        name: str,
        arrays: _ArchiveArrays,
        transition_count: int,
        observation_count: int,
        chunk_boundaries: int,
        encoder: _MetadataEncoder,
    ) -> Any:
        array = _as_numeric_array(value, path=f"raw_arrays/{name}")
        if array is not None:
            safe_name = re.sub(r"[^a-z0-9_]", "_", name.lower())
            if not safe_name or not safe_name[0].isalpha():
                safe_name = f"array_{safe_name}"
            logical_name = f"raw_{safe_name}"
            if len(logical_name) > 64:
                suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
                logical_name = f"{logical_name[:55]}_{suffix}"
            if array.ndim > 0 and array.shape[0] in {transition_count, observation_count} and array.shape[0] > 0:
                pieces = _series_pieces(array, chunk_boundaries=chunk_boundaries, index_key="timeline")
                arrays.add(logical_name, array.dtype, array.shape, f"raw_arrays/{name}", pieces)
            else:
                arrays.add_array(logical_name, array, f"raw_arrays/{name}")
            reference = {"$archive_array": logical_name}
            if isinstance(value, tuple):
                reference["$container"] = "tuple"
            elif isinstance(value, list):
                reference["$container"] = "list"
            return reference
        return encoder.encode(value, ("raw_arrays", name))

    def _write_archive(
        self,
        *,
        target: Path,
        archive_kind: str,
        episode_id: str | None,
        attempt_id: str,
        program_id: str,
        split: Any,
        subset: Any,
        outcome: str,
        identity: Mapping[str, str],
        timeline: Mapping[str, int],
        record_metadata: Mapping[str, Any],
        raw_arrays: Mapping[str, Any],
        debug_metadata: Any,
        arrays: _ArchiveArrays,
    ) -> ArchiveManifest:
        target = target.absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError(f"archive output directory must not be a symlink: {target}")
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError(f"archive output directory must be new or empty: {target}")
        owns_staging = True
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
        try:
            data_dir = staging / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            chunk_inventory = arrays.write_chunks(staging)
            chunk_paths = {
                chunk_index: f"data/chunk-{chunk_index:05d}.npz"
                for chunk_index in arrays.chunk_files
            }
            debug_path = staging / "debug.json"
            debug_bytes = _canonical_json(debug_metadata)
            self._atomic_write(debug_path, debug_bytes, budget=arrays.budget)
            debug_inventory = {
                "path": "debug.json",
                "bytes": len(debug_bytes),
                "sha256": _sha256_bytes(debug_bytes),
            }
            manifest_name = "episode.manifest.json" if archive_kind == "episode" else "attempt.manifest.json"
            manifest_payload: dict[str, Any] = {
                "archive_format_id": ARCHIVE_FORMAT_ID,
                "episode_schema_version": ARCHIVE_EPISODE_SCHEMA_VERSION,
                "dataset_identity": ARCHIVE_DATASET_IDENTITY,
                "archive_kind": archive_kind,
                "episode_id": episode_id,
                "attempt_id": attempt_id,
                "program_id": program_id,
                "split": split,
                "subset": subset,
                "outcome": outcome,
                **dict(identity),
                "archive_profile": self.profile.as_dict(),
                "local_write_limits": {
                    "max_chunk_bytes": self.profile.max_chunk_bytes,
                    "spool_peak_bytes": arrays.spool_peak_bytes,
                    "local_peak_bytes_upper_bound": 0,
                },
                "timeline": dict(timeline),
                "record_metadata": dict(record_metadata),
                "raw_arrays": dict(raw_arrays),
                "array_specs": arrays.specs_as_dict(chunk_paths),
                "array_aliases": arrays.aliases_as_list(chunk_paths),
                "chunk_inventory": chunk_inventory,
                "debug_path": debug_inventory["path"],
                "debug_bytes": debug_inventory["bytes"],
                "debug_sha256": debug_inventory["sha256"],
                "artifact_manifest_path": "artifact_manifest.json",
            }
            compressed_bytes = sum(item["bytes"] for item in chunk_inventory)
            local_peak_bound = arrays.spool_peak_bytes + compressed_bytes + len(debug_bytes) + 2_000_000
            for _iteration in range(4):
                manifest_payload["local_write_limits"]["local_peak_bytes_upper_bound"] = local_peak_bound
                manifest_bytes = _canonical_json(manifest_payload)
                all_file_inventory: dict[str, dict[str, Any]] = {
                    entry["path"]: {"sha256": entry["sha256"], "bytes": entry["bytes"]}
                    for entry in chunk_inventory
                }
                all_file_inventory["debug.json"] = debug_inventory | {"path": "debug.json"}
                all_file_inventory[manifest_name] = {
                    "path": manifest_name,
                    "bytes": len(manifest_bytes),
                    "sha256": _sha256_bytes(manifest_bytes),
                }
                artifact_bytes = _canonical_json({
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "program_id": program_id,
                    "outcome": outcome,
                    "dataset_identity": ARCHIVE_DATASET_IDENTITY,
                    "archive_format_id": ARCHIVE_FORMAT_ID,
                    "files": all_file_inventory,
                })
                updated_bound = (
                    arrays.spool_peak_bytes
                    + compressed_bytes
                    + len(debug_bytes)
                    + len(manifest_bytes)
                    + len(artifact_bytes)
                )
                if updated_bound <= local_peak_bound:
                    break
                local_peak_bound = updated_bound
            manifest_payload["local_write_limits"]["local_peak_bytes_upper_bound"] = local_peak_bound
            manifest_bytes = _canonical_json(manifest_payload)
            all_file_inventory[manifest_name] = {
                "bytes": len(manifest_bytes),
                "sha256": _sha256_bytes(manifest_bytes),
            }
            artifact_bytes = _canonical_json({
                "attempt_id": attempt_id,
                "episode_id": episode_id,
                "program_id": program_id,
                "outcome": outcome,
                "dataset_identity": ARCHIVE_DATASET_IDENTITY,
                "archive_format_id": ARCHIVE_FORMAT_ID,
                "files": all_file_inventory,
            })
            actual_local_bound = (
                arrays.spool_peak_bytes
                + compressed_bytes
                + len(debug_bytes)
                + len(manifest_bytes)
                + len(artifact_bytes)
            )
            if actual_local_bound > local_peak_bound:
                manifest_payload["local_write_limits"]["local_peak_bytes_upper_bound"] = actual_local_bound
                manifest_bytes = _canonical_json(manifest_payload)
                all_file_inventory[manifest_name] = {
                    "bytes": len(manifest_bytes),
                    "sha256": _sha256_bytes(manifest_bytes),
                }
                artifact_bytes = _canonical_json({
                    "attempt_id": attempt_id,
                    "episode_id": episode_id,
                    "program_id": program_id,
                    "outcome": outcome,
                    "dataset_identity": ARCHIVE_DATASET_IDENTITY,
                    "archive_format_id": ARCHIVE_FORMAT_ID,
                    "files": all_file_inventory,
                })
            self._atomic_write(staging / "artifact_manifest.json", artifact_bytes, budget=arrays.budget)
            self._atomic_write(staging / manifest_name, manifest_bytes, budget=arrays.budget)
            self._fsync_directory(data_dir)
            self._fsync_directory(staging)
            if owns_staging:
                if target.exists():
                    if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
                        raise FileExistsError(f"archive output directory changed during write: {target}")
                    target.rmdir()
                os.replace(staging, target)
                self._fsync_directory(target.parent)
            return ArchiveManifest.from_dict(manifest_payload)
        except Exception as error:
            if owns_staging and not isinstance(error, ArchiveWriterCapExceeded):
                shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _atomic_write(path: Path, payload: bytes, *, budget: _WriterByteBudget | None = None) -> None:
        temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{os.urandom(4).hex()}")
        try:
            with (budget.open(temporary) if budget is not None else temporary.open("wb")) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except ArchiveWriterCapExceeded:
            raise
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _read_manifest(path: str | Path) -> tuple[Path, ArchiveManifest]:
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ValueError(f"archive manifest must not be a symlink: {manifest_path}")
    if manifest_path.is_dir():
        candidates = (manifest_path / "episode.manifest.json", manifest_path / "attempt.manifest.json")
        existing = [item for item in candidates if item.is_file()]
        if len(existing) != 1:
            raise ValueError("archive directory must contain exactly one episode/attempt manifest")
        manifest_path = existing[0]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"archive manifest must be a regular file: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"archive manifest is unreadable: {manifest_path}") from error
    return manifest_path, ArchiveManifest.from_dict(payload)


def _safe_file(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError(f"archive root must not be a symlink: {root}")
    relative = _require_safe_relative(relative, "archive file path")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"archive path must not traverse symlinks: {relative}")
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"archive path escapes its root: {relative}")
    if not candidate.is_file():
        raise ValueError(f"archive file is missing: {relative}")
    return candidate


def _manifest_alias_map(manifest: ArchiveManifest) -> dict[str, str]:
    return {item["name"]: item["target"] for item in manifest.payload["array_aliases"]}


def _validate_piece_ranges(payload: Mapping[str, Any]) -> None:
    timeline = payload["timeline"]
    observations = timeline.get("observations")
    transitions = timeline.get("transitions")
    for name, spec in payload["array_specs"].items():
        pieces = spec["pieces"]
        for prefix in ("boundary", "point", "transition", "timeline", "array"):
            start_key, stop_key = f"{prefix}_start", f"{prefix}_stop"
            if not any(start_key in piece for piece in pieces):
                continue
            if any(start_key not in piece or stop_key not in piece for piece in pieces):
                raise ValueError(f"{name} has an incomplete {prefix} range mapping")
            cursor = 0
            for piece in pieces:
                start, stop = piece[start_key], piece[stop_key]
                if start != cursor:
                    raise ValueError(f"{name} {prefix} range has a gap, overlap, or reordered piece")
                cursor = stop
                if prefix in {"transition", "timeline", "array"} and piece["shape"]:
                    if piece["shape"][0] != stop - start:
                        raise ValueError(f"{name} {prefix} range does not align with its array rows")
            expected_stop = None
            if prefix == "boundary" and payload["archive_kind"] == "episode":
                expected_stop = observations
            elif prefix == "transition":
                expected_stop = transitions
            elif prefix in {"timeline", "array"}:
                shape = spec["shape"]
                expected_stop = shape[0] if shape else None
            elif prefix == "point" and name == "online_points":
                expected_stop = spec["shape"][0]
            elif prefix == "point" and name == "online_validity_packed":
                expected_stop = sum(int(piece.get("validity_point_count", 0)) for piece in pieces)
            if expected_stop is not None and cursor != expected_stop:
                raise ValueError(f"{name} {prefix} ranges do not cover the declared logical array/timeline")

        for piece in pieces:
            if "timeline_index" in piece:
                index = piece["timeline_index"]
                role = spec["semantic_role"]
                if role.startswith(("robot_states/", "object_states/", "event_states/", "timestamps/")) and index >= (observations or 0):
                    raise ValueError(f"{name} timeline_index is outside the declared timeline")


def _require_array_spec(reader: "EpisodeArchiveReader", name: str) -> Mapping[str, Any]:
    resolved = reader._resolved_name(name)
    spec = reader.manifest.payload["array_specs"].get(resolved)
    if spec is None:
        raise ValueError(f"required archive array is missing: {name}")
    return spec


def _require_array(reader: "EpisodeArchiveReader", name: str) -> tuple[Mapping[str, Any], np.ndarray]:
    spec = _require_array_spec(reader, name)
    return spec, reader.read_array(name)


def _require_complete_range(spec: Mapping[str, Any], *, name: str, prefix: str, count: int) -> None:
    start_key, stop_key = f"{prefix}_start", f"{prefix}_stop"
    cursor = 0
    for piece in spec["pieces"]:
        if start_key not in piece or stop_key not in piece or piece[start_key] != cursor:
            raise ValueError(f"{name} {prefix} ranges do not cover the timeline")
        cursor = piece[stop_key]
    if cursor != count:
        raise ValueError(f"{name} {prefix} ranges do not cover the timeline")


def _validate_episode_semantics(reader: "EpisodeArchiveReader", payload: Mapping[str, Any]) -> None:
    required = {
        "online_points", "online_points_offsets", "online_validity_packed", "online_poses", "online_grips",
        "commands", "command_grips", "command_durations", "achieved_durations", "dt", "substeps",
    }
    available = set(payload["array_specs"]) | set(_manifest_alias_map(reader.manifest))
    missing = required - available
    if missing:
        raise ValueError(f"episode archive is missing required arrays: {', '.join(sorted(missing))}")
    timeline = payload["timeline"]
    transitions, observations = timeline.get("transitions"), timeline.get("observations")
    if type(transitions) is not int or transitions <= 0 or type(observations) is not int or observations != transitions + 1:
        raise ValueError("episode archive timeline must contain T transitions and T+1 observations")

    point_spec = _require_array_spec(reader, "online_points")
    offsets_spec, offsets = _require_array(reader, "online_points_offsets")
    validity_spec = _require_array_spec(reader, "online_validity_packed")
    if (
        offsets.shape != (observations + 1,)
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or offsets[-1] != point_spec["shape"][0]
        or np.any(offsets[1:] <= offsets[:-1])
        or point_spec["shape"][1:] != [3]
        or np.dtype(point_spec["dtype"]).kind != "f"
    ):
        raise ValueError("online point offsets must be monotonic, strictly increasing, and end at the point count")
    if tuple(point_spec["shape"]) != (int(offsets[-1]), 3):
        raise ValueError("online points must have Nx3 values aligned with offsets")
    if np.dtype(validity_spec["dtype"]) != np.dtype(np.uint8):
        raise ValueError("packed online validity mask must use uint8 storage")
    point_pieces = point_spec["pieces"]
    validity_pieces = validity_spec["pieces"]
    if len(point_pieces) != len(validity_pieces):
        raise ValueError("online point and validity chunk mappings do not align")
    _require_complete_range(point_spec, name="online_points", prefix="boundary", count=observations)
    _require_complete_range(validity_spec, name="online_validity_packed", prefix="boundary", count=observations)
    for point_piece, validity_piece in zip(point_pieces, validity_pieces):
        start = point_piece["boundary_start"]
        stop = point_piece["boundary_stop"]
        point_start, point_stop = int(offsets[start]), int(offsets[stop])
        if (
            (validity_piece["boundary_start"], validity_piece["boundary_stop"]) != (start, stop)
            or point_piece.get("point_start") != point_start
            or point_piece.get("point_stop") != point_stop
            or validity_piece.get("point_start") != point_start
            or validity_piece.get("point_stop") != point_stop
            or point_piece["shape"][0] != point_stop - point_start
        ):
            raise ValueError("online point and validity piece ranges do not align with offsets")
        expected_count = point_stop - point_start
        if validity_piece.get("validity_point_count") != expected_count:
            raise ValueError("packed online validity point count does not align with offsets")
        packed = reader._load_chunk(validity_piece["path"])[validity_piece["key"]]
        if packed.shape != ((expected_count + 7) // 8,):
            raise ValueError("packed online validity chunk has the wrong byte count")
        if expected_count % 8:
            padding = np.unpackbits(packed[-1:], bitorder="little")[expected_count % 8:]
            if np.any(padding):
                raise ValueError("packed online validity padding bits must be zero")

    for name, shape, prefix, count in (
        ("online_poses", (observations, 4, 4), "boundary", observations),
        ("online_grips", (observations,), "boundary", observations),
        ("commands", (transitions, 4, 4), "transition", transitions),
        ("command_grips", (transitions,), "transition", transitions),
        ("command_durations", (transitions,), "transition", transitions),
        ("achieved_durations", (transitions,), "transition", transitions),
        ("dt", (transitions,), "transition", transitions),
        ("substeps", (transitions,), "transition", transitions),
    ):
        spec, value = _require_array(reader, name)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have the required T/T+1 shape")
        _require_complete_range(spec, name=name, prefix=prefix, count=count)

    metadata = payload["record_metadata"]
    for field in ("robot_states", "robot_state", "object_states", "object_state"):
        value = metadata.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            if len(value) != observations:
                raise ValueError(f"{field} must align with T+1 observation boundaries")
        elif isinstance(value, Mapping) and "$archive_array" in value:
            state_spec = _require_array_spec(reader, str(value["$archive_array"]))
            if not state_spec["shape"] or state_spec["shape"][0] != observations:
                raise ValueError(f"{field} must align with T+1 observation boundaries")
        else:
            raise ValueError(f"{field} must be a boundary sequence or archive-array reference")
    for name in ("rho", "nu", "epsilon"):
        value = metadata.get(name)
        valid_value = metadata.get(f"{name}_valid")
        if value is None:
            if valid_value is not None:
                raise ValueError(f"{name}_valid is present without {name}")
            continue
        if not isinstance(value, Mapping) or "$archive_array" not in value:
            raise ValueError(f"{name} must reference an archived task-label array")
        label_spec = _require_array_spec(reader, str(value["$archive_array"]))
        if len(label_spec["shape"]) != 2 or label_spec["shape"][0] != observations:
            raise ValueError(f"{name} must align to T+1 task-label rows")
        if valid_value is not None:
            if not isinstance(valid_value, Mapping) or "$archive_array" not in valid_value:
                raise ValueError(f"{name}_valid must reference an archived boolean array")
            valid_spec = _require_array_spec(reader, str(valid_value["$archive_array"]))
            if np.dtype(valid_spec["dtype"]) != np.dtype(np.bool_) or valid_spec["shape"] != label_spec["shape"]:
                raise ValueError(f"{name}_valid must match {name} shape and use boolean dtype")


def _validate_attempt_semantics(reader: "EpisodeArchiveReader", payload: Mapping[str, Any]) -> None:
    timeline = payload["timeline"]
    observations, transitions = timeline.get("observations"), timeline.get("transitions")
    if type(observations) is not int or observations < 0 or type(transitions) is not int or transitions < 0:
        raise ValueError("attempt timeline observations/transitions must be nonnegative integers")
    valid_until = timeline.get("valid_observation_until")
    if valid_until is not None and (
        type(valid_until) is not int or valid_until < 0 or observations != valid_until + 1
    ):
        raise ValueError("attempt valid_observation_until must identify its final archived observation")
    raw_arrays = payload["raw_arrays"]
    measured_prefix_reference = False
    for key in ("observations", "gripper_pose", "T_w_e"):
        for reference in _array_references(raw_arrays.get(key)):
            spec = _require_array_spec(reader, reference)
            if spec["shape"] and spec["shape"][0] == observations:
                measured_prefix_reference = True
    point_values_ref, point_offsets_ref = raw_arrays.get("points"), raw_arrays.get("point_offsets")
    if isinstance(point_values_ref, Mapping) and "$archive_array" in point_values_ref and isinstance(point_offsets_ref, Mapping) and "$archive_array" in point_offsets_ref:
        measured_prefix_reference = True
    if valid_until is not None and not measured_prefix_reference:
        raise ValueError("attempt valid_observation_until requires a measured-prefix reference")
    for key, expected_count in (("actions", transitions), ("commands", transitions), ("T_w_e", observations), ("gripper_pose", observations), ("grip", observations)):
        reference = raw_arrays.get(key)
        if isinstance(reference, Mapping) and "$archive_array" in reference:
            value = reader.read_array(str(reference["$archive_array"]))
            if not value.shape or value.shape[0] != expected_count:
                raise ValueError(f"attempt prefix {key} does not align with its timeline count")
    points_ref, offsets_ref = raw_arrays.get("points"), raw_arrays.get("point_offsets")
    if (points_ref is None) != (offsets_ref is None):
        raise ValueError("attempt prefix points and point_offsets must be provided together")
    if isinstance(offsets_ref, Mapping) and "$archive_array" in offsets_ref:
        offsets = reader.read_array(str(offsets_ref["$archive_array"]))
        if offsets.shape != (observations + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0:
            raise ValueError("attempt point_offsets must contain one extra T+1 boundary")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("attempt point_offsets must be monotonic")
        if isinstance(points_ref, Mapping) and "$archive_array" in points_ref:
            point_spec = _require_array_spec(reader, str(points_ref["$archive_array"]))
            if point_spec["shape"][1:] != [3] or offsets[-1] != point_spec["shape"][0]:
                raise ValueError("attempt points and point_offsets do not align")
    attempt = payload["record_metadata"].get("attempt")
    if not isinstance(attempt, Mapping):
        raise ValueError("attempt manifest must retain its source attempt metadata")
    if attempt.get("attempt_id") != payload["attempt_id"] or attempt.get("program_id") != payload["program_id"]:
        raise ValueError("attempt source metadata identity mismatch")
    if attempt.get("valid_observation_until") != valid_until:
        raise ValueError("attempt source valid_observation_until disagrees with timeline")
    if attempt.get("outcome") != payload["outcome"] or attempt.get("episode_id") is not None:
        raise ValueError("attempt source metadata outcome/episode identity mismatch")


def validate_archive_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify all local archive hashes, chunk arrays, identity and timeline offsets."""
    path, manifest = _read_manifest(manifest_path)
    root = path.parent
    payload = manifest.payload
    chunk_inventory = {item["path"]: item for item in payload["chunk_inventory"]}
    profile = ArchiveProfileConfig.from_dict(payload["archive_profile"])
    _validate_piece_ranges(payload)
    artifact_path = _safe_file(root, payload["artifact_manifest_path"])
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("artifact_manifest.json is unreadable") from error
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("files"), Mapping):
        raise ValueError("artifact_manifest.json files must be an object")
    expected_inventory = dict(artifact["files"])
    for item in root.rglob("*"):
        if item.is_symlink():
            relative = item.relative_to(root).as_posix()
            raise ValueError(f"archive paths must not traverse symlinks: {relative}")
    actual_paths = {
        str(item.relative_to(root).as_posix())
        for item in root.rglob("*")
        if item.is_file() and item.name != payload["artifact_manifest_path"]
    }
    if actual_paths != set(expected_inventory):
        raise ValueError("artifact manifest file inventory mismatch")
    for relative, entry in expected_inventory.items():
        relative = _require_safe_relative(relative, "artifact manifest path")
        file_path = _safe_file(root, relative)
        if not isinstance(entry, Mapping):
            raise ValueError(f"artifact manifest entry must be an object: {relative}")
        if file_path.stat().st_size != entry.get("bytes") or _sha256_file(file_path) != entry.get("sha256"):
            raise ValueError(f"artifact manifest checksum mismatch: {relative}")
    manifest_name = path.name
    if manifest_name not in expected_inventory:
        raise ValueError("archive manifest is missing from artifact inventory")
    debug_entry = expected_inventory.get(payload["debug_path"])
    if not isinstance(debug_entry, Mapping):
        raise ValueError("debug metadata is missing from artifact inventory")
    if debug_entry.get("bytes") != payload["debug_bytes"] or debug_entry.get("sha256") != payload["debug_sha256"]:
        raise ValueError("debug metadata identity mismatch")
    if artifact.get("archive_format_id") != ARCHIVE_FORMAT_ID or artifact.get("dataset_identity") != ARCHIVE_DATASET_IDENTITY:
        raise ValueError("artifact manifest archive identity mismatch")
    if (
        artifact.get("episode_id") != payload["episode_id"]
        or artifact.get("attempt_id") != payload["attempt_id"]
        or artifact.get("program_id") != payload["program_id"]
        or artifact.get("outcome") != payload["outcome"]
    ):
        raise ValueError("artifact manifest identity mismatch")

    expected_chunks: dict[str, set[str]] = {}
    pieces_by_chunk: dict[str, list[tuple[str, int, Mapping[str, Any], Mapping[str, Any]]]] = {}
    spec_state: dict[str, dict[str, Any]] = {}
    chunk_rank = {relative: index for index, relative in enumerate(sorted(chunk_inventory))}
    for relative, entry in chunk_inventory.items():
        artifact_entry = expected_inventory.get(relative)
        if not isinstance(artifact_entry, Mapping):
            raise ValueError(f"chunk is missing from artifact inventory: {relative}")
        if artifact_entry.get("bytes") != entry["bytes"] or artifact_entry.get("sha256") != entry["sha256"]:
            raise ValueError(f"chunk identity disagrees with artifact inventory: {relative}")
        chunk_path = _safe_file(root, relative)
        if chunk_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"chunk checksum mismatch: {relative}")
    for name, spec in payload["array_specs"].items():
        digest = hashlib.sha256()
        digest.update(spec["dtype"].encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(spec["shape"]))
        spec_state[name] = {"digest": digest, "byte_count": 0, "shapes": []}
        previous_rank = -1
        for piece_index, piece in enumerate(spec["pieces"]):
            relative = piece["path"]
            rank = chunk_rank[relative]
            if rank < previous_rank:
                raise ValueError(f"array pieces are not in chunk order: {name}")
            previous_rank = rank
            expected_chunks.setdefault(relative, set()).add(piece["key"])
            pieces_by_chunk.setdefault(relative, []).append((name, piece_index, spec, piece))

    for relative in sorted(chunk_inventory):
        chunk_path = _safe_file(root, relative)
        _check_npz_uncompressed_cap(
            chunk_path,
            max_chunk_bytes=profile.max_chunk_bytes,
            relative=relative,
        )
        if _sha256_file(chunk_path) != chunk_inventory[relative]["sha256"]:
            raise ValueError(f"chunk checksum mismatch: {relative}")
        try:
            with np.load(chunk_path, allow_pickle=False) as loaded:
                actual_keys = set(loaded.files)
                if actual_keys != expected_chunks.get(relative, set()):
                    raise ValueError(f"chunk array inventory mismatch: {relative}")
                for name, _piece_index, spec, piece in sorted(
                    pieces_by_chunk.get(relative, []), key=lambda item: (item[0], item[1])
                ):
                    value = np.asarray(loaded[piece["key"]])
                    if value.dtype.str != spec["dtype"] or list(value.shape) != piece["shape"]:
                        raise ValueError(f"array piece dtype/shape mismatch: {name}")
                    if int(value.nbytes) != piece["byte_count"] or _array_digest(value.dtype, value.shape, [value]) != piece["sha256"]:
                        raise ValueError(f"array piece checksum mismatch: {name}")
                    role = spec["semantic_role"]
                    if role in {"online_observations/points", "raw_arrays/measured_points", "raw_arrays/prefix_points"}:
                        if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
                            raise ValueError(f"point array piece must contain finite Nx3 values: {name}")
                    if role in {"rho", "nu", "epsilon"}:
                        if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
                            raise ValueError(f"{role} values must be finite probabilities in [0,1]")
                    if role in {"rho_valid", "nu_valid", "epsilon_valid"} and value.dtype != np.bool_:
                        raise ValueError(f"{role} pieces must use boolean dtype")
                    state = spec_state[name]
                    _update_array_digest(state["digest"], value)
                    state["byte_count"] += int(value.nbytes)
                    state["shapes"].append(list(value.shape))
                    del value
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"chunk is not a safe NPZ archive: {relative}") from error
    for name, spec in payload["array_specs"].items():
        state = spec_state[name]
        shapes = state["shapes"]
        if state["byte_count"] != spec["byte_count"] or state["digest"].hexdigest() != spec["sha256"]:
            raise ValueError(f"array checksum or byte count mismatch: {name}")
        shape = spec["shape"]
        if len(shapes) == 1 and shapes[0] != shape:
            raise ValueError(f"array logical shape mismatch: {name}")
        if len(shapes) > 1:
            if not shape or sum(item[0] for item in shapes) != shape[0]:
                raise ValueError(f"array logical shape mismatch: {name}")
            if any(item[1:] != shape[1:] for item in shapes):
                raise ValueError(f"array logical shape mismatch: {name}")

    known_array_names = set(payload["array_specs"]) | set(_manifest_alias_map(manifest))
    try:
        debug_metadata = json.loads(_safe_file(root, payload["debug_path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("debug metadata is unreadable") from error
    if not isinstance(debug_metadata, Mapping):
        raise ValueError("debug metadata must be an object")
    for reference in _array_references(debug_metadata):
        if reference not in known_array_names:
            raise ValueError(f"debug archive array reference is missing: {reference}")

    reader = EpisodeArchiveReader(path, validate_files=False)
    _validate_reference_semantic_roles(reader, payload, debug_metadata)
    if payload["archive_kind"] == "episode":
        _validate_episode_semantics(reader, payload)
    else:
        if _canonical_json(_redact_sensitive(payload["record_metadata"])) != _canonical_json(payload["record_metadata"]):
            raise ValueError("attempt manifest metadata contains unredacted or oversized secret text")
        if _canonical_json(_redact_sensitive(debug_metadata)) != _canonical_json(debug_metadata):
            raise ValueError("attempt debug metadata contains unredacted or oversized secret text")
        sensitive_identity = {
            name: payload.get(name)
            for name in ("attempt_id", "program_id", "split", "subset", "source_run_id", "code_revision", "preprocessing_identity")
        }
        if _canonical_json(_redact_sensitive(sensitive_identity)) != _canonical_json(sensitive_identity):
            raise ValueError("attempt archive identity contains unredacted or oversized secret text")
        _validate_attempt_semantics(reader, payload)
    return {
        "valid": True,
        "archive_kind": payload["archive_kind"],
        "episode_id": payload["episode_id"],
        "attempt_id": payload["attempt_id"],
        "outcome": payload["outcome"],
        "array_count": len(payload["array_specs"]) + len(payload["array_aliases"]),
        "chunk_count": len(chunk_inventory),
        "file_count": len(actual_paths) + 1,
        "manifest_sha256": _sha256_file(path),
    }


class EpisodeArchiveReader:
    """Lazy reader with a bounded uncompressed NPZ chunk cache."""

    DEFAULT_CACHE_BYTES = 128 * 1024 * 1024

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
        validate_files: bool = False,
        resolved_preprocessing_identity: str | None = None,
        resolved_preprocessing_sha256: str | None = None,
    ) -> None:
        if type(cache_bytes) is not int or cache_bytes <= 0:
            raise ValueError("cache_bytes must be a positive integer")
        self.manifest_path, self.manifest = _read_manifest(manifest_path)
        self.root = self.manifest_path.parent
        self.cache_bytes = cache_bytes
        self.max_chunk_bytes = ArchiveProfileConfig.from_dict(self.manifest.payload["archive_profile"]).max_chunk_bytes
        if resolved_preprocessing_identity is not None and (
            not isinstance(resolved_preprocessing_identity, str) or not resolved_preprocessing_identity.strip()
        ):
            raise ValueError("resolved_preprocessing_identity must be a nonblank string when provided")
        if resolved_preprocessing_sha256 is not None and not _is_sha256(resolved_preprocessing_sha256):
            raise ValueError("resolved_preprocessing_sha256 must be a lowercase SHA256 digest")
        if (resolved_preprocessing_identity is None) != (resolved_preprocessing_sha256 is None):
            raise ValueError("resolved preprocessing identity and SHA256 must be supplied together")
        source_preprocessing_identity = str(self.manifest.payload["preprocessing_identity"])
        self.resolved_preprocessing_identity = resolved_preprocessing_identity or source_preprocessing_identity
        self.resolved_preprocessing_sha256 = resolved_preprocessing_sha256 or _sha256_bytes(
            source_preprocessing_identity.encode("utf-8")
        )
        identity_bytes = _canonical_json({
            "archive_manifest_sha256": _sha256_file(self.manifest_path),
            "archive_preprocessing_identity": source_preprocessing_identity,
            "resolved_preprocessing_identity": self.resolved_preprocessing_identity,
            "resolved_preprocessing_sha256": self.resolved_preprocessing_sha256,
        })
        self.cache_identity = _sha256_bytes(identity_bytes)
        self._cache: OrderedDict[tuple[str, str], tuple[dict[str, np.ndarray], int]] = OrderedDict()
        self._cache_size = 0
        self._aliases = _manifest_alias_map(self.manifest)
        self._chunk_entries = {item["path"]: item for item in self.manifest.payload["chunk_inventory"]}
        if validate_files:
            validate_archive_manifest(self.manifest_path)
        self._debug_metadata: Any | None = None

    def iter_boundaries(self) -> Iterator[int]:
        count = int(self.manifest.payload["timeline"].get("observations", 0))
        return iter(range(count))

    def _load_chunk(self, relative: str) -> dict[str, np.ndarray]:
        cache_key = (self.cache_identity, relative)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached[0]
        entry = self._chunk_entries.get(relative)
        if entry is None:
            raise ValueError(f"chunk is not declared by the archive manifest: {relative}")
        path = _safe_file(self.root, relative)
        _check_npz_uncompressed_cap(path, max_chunk_bytes=self.max_chunk_bytes, relative=relative)
        if path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"chunk checksum mismatch: {relative}")
        try:
            with np.load(path, allow_pickle=False) as loaded:
                values = {key: np.array(loaded[key], copy=True) for key in loaded.files}
        except Exception as error:
            raise ValueError(f"chunk is not a safe NPZ archive: {relative}") from error
        size = sum(int(value.nbytes) for value in values.values())
        if size <= self.cache_bytes:
            while self._cache and self._cache_size + size > self.cache_bytes:
                _evicted_key, (_evicted_values, evicted_size) = self._cache.popitem(last=False)
                self._cache_size -= evicted_size
            self._cache[cache_key] = (values, size)
            self._cache_size += size
        return values

    def _resolved_name(self, name: str) -> str:
        seen: set[str] = set()
        current = name
        while current in self._aliases:
            if current in seen:
                raise ValueError(f"cyclic array alias: {name}")
            seen.add(current)
            current = self._aliases[current]
        return current

    def read_array(self, name: str) -> np.ndarray:
        resolved = self._resolved_name(name)
        spec = self.manifest.payload["array_specs"].get(resolved)
        if spec is None:
            raise KeyError(f"archive array does not exist: {name}")
        values = []
        for piece in spec["pieces"]:
            chunk = self._load_chunk(piece["path"])
            try:
                values.append(chunk[piece["key"]])
            except KeyError as error:
                raise ValueError(f"array piece is missing from chunk: {name}") from error
        if len(values) == 1:
            return np.array(values[0], copy=True)
        return np.concatenate(values, axis=0)

    def read_array_row(self, name: str, index: int) -> np.ndarray:
        if type(index) is not int or index < 0:
            raise IndexError(f"array row index out of range: {index}")
        resolved = self._resolved_name(name)
        spec = self.manifest.payload["array_specs"].get(resolved)
        if spec is None:
            raise KeyError(f"archive array does not exist: {name}")
        if not spec["shape"] or index >= spec["shape"][0]:
            raise IndexError(f"array row index out of range: {index}")
        for piece in spec["pieces"]:
            start = piece.get(
                "array_start",
                piece.get("timeline_start", piece.get("boundary_start", piece.get("transition_start", 0))),
            )
            stop = piece.get(
                "array_stop",
                piece.get(
                    "timeline_stop",
                    piece.get("boundary_stop", piece.get("transition_stop", spec["shape"][0])),
                ),
            )
            if start <= index < stop:
                value = self._load_chunk(piece["path"])[piece["key"]]
                return np.array(value[index - start], copy=True)
        raise IndexError(f"{name} has no chunk for row {index}")

    def _piece_at(self, name: str, *, key_start: str, index: int) -> tuple[Mapping[str, Any], np.ndarray]:
        resolved = self._resolved_name(name)
        spec = self.manifest.payload["array_specs"].get(resolved)
        if spec is None:
            raise KeyError(f"archive array does not exist: {name}")
        for piece in spec["pieces"]:
            start = piece.get(f"{key_start}_start")
            stop = piece.get(f"{key_start}_stop")
            if isinstance(start, int) and isinstance(stop, int) and start <= index < stop:
                return piece, self._load_chunk(piece["path"])[piece["key"]]
        raise IndexError(f"{name} has no chunk for index {index}")

    def observation(self, boundary: int) -> Mapping[str, Any]:
        count = int(self.manifest.payload["timeline"].get("observations", 0))
        if type(boundary) is not int or boundary < 0 or boundary >= count:
            raise IndexError(f"observation boundary out of range: {boundary}")
        point_piece, point_values = self._piece_at("online_points", key_start="boundary", index=boundary)
        validity_piece, validity_values = self._piece_at("online_validity_packed", key_start="boundary", index=boundary)
        offsets = self.read_array("online_points_offsets")
        point_start = int(offsets[boundary]) - int(point_piece["point_start"])
        point_stop = int(offsets[boundary + 1]) - int(point_piece["point_start"])
        _pose_piece, pose_values = self._piece_at("online_poses", key_start="boundary", index=boundary)
        _grip_piece, grip_values = self._piece_at("online_grips", key_start="boundary", index=boundary)
        pose_piece = _pose_piece
        local_boundary = boundary - int(pose_piece["boundary_start"])
        validity_block = np.unpackbits(
            validity_values,
            count=int(validity_piece["validity_point_count"]),
            bitorder="little",
        )
        validity_start = int(offsets[boundary]) - int(validity_piece["point_start"])
        validity_stop = int(offsets[boundary + 1]) - int(validity_piece["point_start"])
        return {
            "points": np.array(point_values[point_start:point_stop], copy=True),
            "point_valid": np.array(validity_block[validity_start:validity_stop], dtype=np.bool_, copy=True),
            "T_w_e": np.array(pose_values[local_boundary], copy=True),
            "grip": np.array(grip_values[local_boundary], copy=True).item(),
        }

    def transition(self, index: int) -> Mapping[str, Any]:
        count = int(self.manifest.payload["timeline"].get("transitions", 0))
        if type(index) is not int or index < 0 or index >= count:
            raise IndexError(f"transition index out of range: {index}")
        piece, command_values = self._piece_at("commands", key_start="transition", index=index)
        local = index - int(piece["transition_start"])
        grip_piece, grip_values = self._piece_at("command_grips", key_start="transition", index=index)
        duration_piece, duration_values = self._piece_at("command_durations", key_start="transition", index=index)
        achieved_piece, achieved_values = self._piece_at("achieved_durations", key_start="transition", index=index)
        substeps_piece, substeps_values = self._piece_at("substeps", key_start="transition", index=index)
        extras = self.manifest.payload["record_metadata"].get("_archive_transition_metadata", [])[index]
        decoded = self._decode_value(extras)
        command = dict(decoded.pop("command", {}))
        command.update({
            "T_w_e": np.array(command_values[local], copy=True),
            "grip": np.asarray(grip_values[index - int(grip_piece["transition_start"])]).item(),
            "duration_s": np.asarray(duration_values[index - int(duration_piece["transition_start"])]).item(),
        })
        decoded["command"] = command
        decoded["achieved_duration_s"] = np.asarray(achieved_values[index - int(achieved_piece["transition_start"])]).item()
        decoded["physics_substeps"] = np.asarray(substeps_values[index - int(substeps_piece["transition_start"])]).item()
        return decoded

    def _decode_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            if "$archive_array" in value:
                array = self.read_array(str(value["$archive_array"]))
                container = value.get("$container")
                if container == "list":
                    return array.tolist()
                if container == "tuple":
                    return tuple(array.tolist())
                return array
            series = value.get("$archive_series")
            if series == "observations":
                return [self.observation(index) for index in self.iter_boundaries()]
            if series == "transitions":
                return [self.transition(index) for index in range(int(self.manifest.payload["timeline"]["transitions"]))]
            return {key: self._decode_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode_value(item) for item in value]
        return value

    def to_episode_record(self) -> dict[str, Any]:
        metadata = self.manifest.payload["record_metadata"]
        if self.manifest.payload["archive_kind"] == "attempt":
            return self._decode_value(metadata["attempt"])
        record = self._decode_value(metadata)
        record.pop("_archive_transition_metadata", None)
        return record

    @property
    def raw_arrays(self) -> Mapping[str, Any]:
        return self._decode_value(self.manifest.payload["raw_arrays"])

    @property
    def debug_metadata(self) -> Mapping[str, Any]:
        if self._debug_metadata is None:
            path = _safe_file(self.root, self.manifest.payload["debug_path"])
            if path.stat().st_size != self.manifest.payload["debug_bytes"] or _sha256_file(path) != self.manifest.payload["debug_sha256"]:
                raise ValueError("debug metadata checksum mismatch")
            self._debug_metadata = self._decode_value(json.loads(path.read_text(encoding="utf-8")))
        return self._debug_metadata

    def task_labels(self, boundary: int) -> Mapping[str, Any]:
        observation_count = int(self.manifest.payload["timeline"].get("observations", 0))
        if type(boundary) is not int or boundary < 0 or boundary >= observation_count:
            raise IndexError(f"task label boundary out of range: {boundary}")
        record_metadata = self.manifest.payload["record_metadata"]
        labels = {}
        for name in ("rho", "rho_valid", "nu", "nu_valid", "epsilon", "epsilon_valid"):
            value = record_metadata.get(name)
            if isinstance(value, Mapping) and "$archive_array" in value:
                labels[name] = self.read_array_row(str(value["$archive_array"]), boundary)
            elif isinstance(value, list) and boundary < len(value):
                labels[name] = np.asarray(self._decode_value(value[boundary]))
        return labels


__all__ = [
    "ArchiveManifest",
    "EpisodeArchiveReader",
    "EpisodeArchiveWriter",
    "archive_profile_from_environment",
    "validate_archive_manifest",
]
