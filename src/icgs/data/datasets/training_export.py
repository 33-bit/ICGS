"""Compact training exports and lazy readers for HF generation archives.

The export is a revision-bound index of the already-finalized generation views.
It contains no numeric archive chunks; :class:`TrainingExportReader` resolves
those chunks lazily from a local source tree or a pinned Hugging Face revision.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

import numpy as np

from icgs.data.collection.generation.distributed_contracts import (
    ARCHIVE_DATASET_IDENTITY,
    ARCHIVE_EPISODE_SCHEMA_VERSION,
    ARCHIVE_FORMAT_ID,
)
from icgs.data.collection.generation.episode_archive import EpisodeArchiveReader
from icgs.data.datasets.generation_archive import (
    SampleRef,
    _PREPROCESSING_IDENTITY,
    _PREPROCESSING_SHA256,
    _remove_statistical_outliers,
    _sample_seed,
)
from icgs.data.datasets.generation_view_index import DEFAULT_ROLE_SPECS, validate_source_revision
from icgs.data.datasets.episodes import adapter_a0, adapter_a1
from icgs.geometry.transforms import transform_pcd


TRAINING_EXPORT_SCHEMA = "icgs_training_export_v1"
TRAINING_PREFIX = "training"
_ROLES = ("train", "validation", "evaluation")
_VIEWS = ("D_geom", "D_temporal", "D_dyn", "D_task")
_VIEW_IDS = tuple(f"{role}/{view}" for role in _ROLES for view in _VIEWS)
_SAMPLE_REF_FIELDS = frozenset({
    "episode_id",
    "view",
    "t",
    "archive_manifest",
    "archive_manifest_sha256",
    "dataset_manifest_sha256",
    "archive_preprocessing_identity",
    "preprocessing_identity",
    "preprocessing_sha256",
    "sample_seed",
})


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _loads_json(content: bytes, *, field: str) -> Any:
    """Decode JSON without accepting non-standard NaN/Infinity values."""

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{field} contains non-finite JSON constant {value}")

    try:
        return json.loads(content, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid JSON") from error


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank relative path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a contained relative path")
    if path.as_posix() != value or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized relative path")
    return value


def _prefix(value: Any, field: str) -> str:
    return _safe_relative(value, field).rstrip("/")


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


def _cache_bytes(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("cache_bytes must be a positive integer")
    return value


def _validate_sha256_field(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return str(value)


def _validate_pointer(value: Any, *, field: str) -> tuple[str, str, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must be a three-element pointer")
    owner, kind, boundary = value
    _nonblank(owner, f"{field} owner")
    _nonblank(kind, f"{field} kind")
    if type(boundary) is not int or boundary < 0:
        raise ValueError(f"{field} boundary must be a nonnegative integer")
    return str(owner), str(kind), boundary


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _validate_sample_row(
    row: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    role: str,
    view: str,
    index: int,
    source_prefix: str,
) -> None:
    label = f"{role}/{view} sample {index}"
    _nonblank(row.get("episode_id"), f"{label} episode_id")
    if row.get("view") != view:
        raise ValueError(f"{label} view identity mismatch")
    if type(row.get("t")) is not int or row["t"] < 0:
        raise ValueError(f"{label} boundary must be a nonnegative integer")
    archive_manifest = row.get("archive_manifest")
    _safe_relative(archive_manifest, f"{label} archive_manifest")
    if not archive_manifest.endswith("/episode.manifest.json"):
        raise ValueError(f"{label} must reference an episode manifest")
    archive_path = row.get("archive_manifest_path")
    _safe_relative(archive_path, f"{label} archive_manifest_path")
    if archive_path != f"{source_prefix}/{archive_manifest}":
        raise ValueError(f"{label} archive path is not source-bound")
    _validate_sha256_field(row.get("archive_manifest_sha256"), f"{label} archive manifest hash")
    if row.get("dataset_manifest_sha256") != payload.get("source_manifest_sha256"):
        raise ValueError(f"{label} dataset manifest hash mismatch")
    _nonblank(row.get("archive_preprocessing_identity"), f"{label} archive preprocessing identity")
    if row.get("preprocessing_identity") != payload.get("preprocessing_identity"):
        raise ValueError(f"{label} preprocessing identity mismatch")
    if row.get("preprocessing_sha256") != payload.get("preprocessing_sha256"):
        raise ValueError(f"{label} preprocessing hash mismatch")
    if type(row.get("sample_seed")) is not int or row["sample_seed"] < 0:
        raise ValueError(f"{label} sample seed must be a nonnegative integer")
    expected_seed = _sample_seed(int(payload["mixture_seed"]), str(row["episode_id"]), view, int(row["t"]))
    if row["sample_seed"] != expected_seed:
        raise ValueError(f"{label} sample seed is not bound to the snapshot mixture seed")
    if row.get("episode_kind") not in {"nominal", "perturbed"}:
        raise ValueError(f"{label} episode_kind is unsupported")

    if view == "D_temporal":
        observations = row.get("observations")
        actions = row.get("actions")
        if not isinstance(observations, list) or not isinstance(actions, list):
            raise ValueError(f"{label} temporal pointers are missing")
        start = max(0, row["t"] - 3)
        if observations != [[row["episode_id"], "observation", t] for t in range(start, row["t"] + 1)]:
            raise ValueError(f"{label} temporal observations are not a canonical causal window")
        if actions != [[row["episode_id"], "action", t] for t in range(start, row["t"])]:
            raise ValueError(f"{label} temporal actions are not a canonical causal window")
    elif view == "D_dyn":
        history = row.get("history")
        if not isinstance(history, list) or any(type(item) is not int or item < 0 or item >= row["t"] for item in history):
            raise ValueError(f"{label} dynamics history is invalid")
        if history != sorted(set(history)):
            raise ValueError(f"{label} dynamics history must be ordered and unique")
        if row.get("action_t") != row["t"] or row.get("observation_t") != row["t"]:
            raise ValueError(f"{label} dynamics boundary binding is invalid")
    elif view == "D_task":
        events = row.get("events")
        if not isinstance(events, list):
            raise ValueError(f"{label} task events must be a list")
        for event_index, event in enumerate(events):
            if (not isinstance(event, list) or len(event) != 3
                    or event[:2] != [row["episode_id"], "event"]
                    or not isinstance(event[2], (str, int, type(None)))):
                raise ValueError(f"{label} event {event_index} is invalid")


def _resolve_contained(root: Path, relative: str, *, field: str) -> Path:
    _safe_relative(relative, field)
    if root.is_symlink():
        raise ValueError(f"{field} root must not be a symlink")
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{field} must not traverse symlinks: {relative}")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError(f"{field} escapes its root: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} is not a regular file: {relative}")
    return resolved


def _immutable_write(path: Path, payload: bytes) -> None:
    _assert_safe_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"immutable training export conflict: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"immutable training export conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _assert_safe_target(path: Path) -> None:
    """Reject symlinked ancestors before a target is created or inspected."""

    system_aliases = {
        Path("/var"): Path("/private/var"),
        Path("/tmp"): Path("/private/tmp"),
    }
    cursor = path
    existing: list[Path] = []
    while cursor != cursor.parent:
        existing.append(cursor)
        cursor = cursor.parent
    for item in reversed(existing):
        if item.is_symlink():
            if item in system_aliases and item.resolve() == system_aliases[item]:
                continue
            raise ValueError(f"immutable training export path must not traverse a symlink: {item}")


def _prepare_immutable_output(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    """Preflight every file and reject partial/extra output before any write."""

    _assert_safe_target(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("training export output_dir must be a directory")
    expected = {PurePosixPath(relative).as_posix() for relative in payloads}
    if output_dir.exists():
        actual = set()
        for item in output_dir.rglob("*"):
            relative = item.relative_to(output_dir).as_posix()
            if item.is_symlink():
                raise ValueError(f"training export output must not contain symlinks: {relative}")
            if item.is_file() and relative not in expected:
                raise ValueError(f"training export output contains an unexpected file: {relative}")
            if item.is_file():
                actual.add(relative)
            if item.is_dir() and not any(
                candidate == relative or candidate.startswith(relative + "/") for candidate in expected
            ):
                raise ValueError(f"training export output contains an unexpected directory: {relative}")
        if actual and actual != expected:
            raise ValueError("partial training export output already exists")
    for relative, content in payloads.items():
        path = output_dir / PurePosixPath(relative)
        _assert_safe_target(path)
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError(f"immutable training export conflict: {path}")


def _validate_view_snapshot(
    payload: Mapping[str, Any],
    *,
    role: str,
    view: str,
    source_revision: str,
    source_prefix: str,
) -> None:
    view_id = f"{role}/{view}"
    if payload.get("manifest_version") != 1 or payload.get("snapshot_kind") != "generation_training_view":
        raise ValueError(f"{view_id} is not a generation training view snapshot")
    if payload.get("view_id") != view_id or payload.get("view") != view or payload.get("role") != role:
        raise ValueError(f"{view_id} view identity mismatch")
    if payload.get("status") != "FINAL" or payload.get("view_status") != "FINAL":
        raise ValueError(f"{view_id} must be FINAL")
    if payload.get("source_revision") != source_revision:
        raise ValueError(f"{view_id} source revision mismatch")
    if payload.get("source_prefix") != source_prefix:
        raise ValueError(f"{view_id} source prefix mismatch")
    if payload.get("dataset_identity") != ARCHIVE_DATASET_IDENTITY:
        raise ValueError(f"{view_id} dataset identity mismatch")
    if payload.get("archive_format_id") != ARCHIVE_FORMAT_ID:
        raise ValueError(f"{view_id} archive format mismatch")
    if payload.get("episode_schema_version") != ARCHIVE_EPISODE_SCHEMA_VERSION:
        raise ValueError(f"{view_id} episode schema mismatch")
    _validate_sha256_field(payload.get("source_manifest_sha256"), f"{view_id} source manifest hash")
    if not isinstance(payload.get("preprocessing_identity"), str) or not payload["preprocessing_identity"].strip():
        raise ValueError(f"{view_id} preprocessing identity is missing")
    _validate_sha256_field(payload.get("preprocessing_sha256"), f"{view_id} preprocessing hash")
    if type(payload.get("mixture_seed")) is not int or payload["mixture_seed"] < 0:
        raise ValueError(f"{view_id} mixture_seed must be a nonnegative integer")
    archive_identities = payload.get("archive_preprocessing_identities")
    if not isinstance(archive_identities, list) or any(
        not isinstance(item, str) or not item.strip() for item in archive_identities
    ):
        raise ValueError(f"{view_id} archive preprocessing identities are missing")
    role_spec = payload.get("role_spec")
    if not isinstance(role_spec, Mapping) or set(role_spec) != {"splits", "subsets"}:
        raise ValueError(f"{view_id} role_spec is invalid")
    for key in ("splits", "subsets"):
        if role_spec[key] != list(DEFAULT_ROLE_SPECS[role][key]):
            raise ValueError(f"{view_id} role_spec.{key} is invalid")
    mixture = payload.get("mixture")
    if not isinstance(mixture, Mapping) or mixture.get("algorithm") != "sha256-rank-exact-7-3-training-transitions-v1":
        raise ValueError(f"{view_id} mixture identity is invalid")
    mixed = role == "train" and view in {"D_temporal", "D_dyn"}
    if mixture.get("applied") is not mixed or mixture.get("seed") != (payload["mixture_seed"] if mixed else None):
        raise ValueError(f"{view_id} mixture seed differs from snapshot seed")
    samples = payload.get("samples")
    if not isinstance(samples, list) or type(payload.get("sample_count")) is not int or payload["sample_count"] != len(samples):
        raise ValueError(f"{view_id} sample count is inconsistent")
    for index, row in enumerate(samples):
        if not isinstance(row, Mapping):
            raise ValueError(f"{view_id} sample {index} must be an object")
        _validate_sample_row(
            row,
            payload=payload,
            role=role,
            view=view,
            index=index,
            source_prefix=source_prefix,
        )
    if len({(row["episode_id"], row["t"]) for row in samples}) != len(samples):
        raise ValueError(f"{view_id} contains duplicate samples")
    kinds = Counter(row["episode_kind"] for row in samples)
    if dict(kinds) != payload.get("sample_counts_by_episode_kind"):
        raise ValueError(f"{view_id} episode kind counts mismatch")
    if sorted({row["archive_preprocessing_identity"] for row in samples}) != archive_identities:
        raise ValueError(f"{view_id} archive preprocessing identities mismatch")
    if mixed:
        units = len(samples) // 10
        if units <= 0 or kinds != {"nominal": 7 * units, "perturbed": 3 * units}:
            raise ValueError(f"{view_id} must preserve complete 7:3 mixture units")
        if (mixture.get("achieved_transition_counts") != dict(kinds)
                or mixture.get("complete_7_to_3_units") != units
                or mixture.get("target_ratio") != {"nominal": 7, "perturbed": 3}):
            raise ValueError(f"{view_id} mixture counts mismatch")


@dataclass(frozen=True)
class TrainingExportReceipt:
    repo_id: str
    target_prefix: str
    source_prefix: str
    source_revision: str
    views_revision: str
    source_manifest_sha256: str
    view_manifest_sha256: str
    view_ids: tuple[str, ...]
    counts: Mapping[str, int]
    status: str
    export_manifest_sha256: str
    output_directory: str = ""

    def __post_init__(self) -> None:
        if self.status != "FINAL":
            raise ValueError("training export receipt must be FINAL")
        object.__setattr__(self, "counts", dict(self.counts))

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": 1,
            "schema_version": TRAINING_EXPORT_SCHEMA,
            "status": self.status,
            "repo_id": self.repo_id,
            "target_prefix": self.target_prefix,
            "source_prefix": self.source_prefix,
            "source_revision": self.source_revision,
            "views_revision": self.views_revision,
            "source_manifest_sha256": self.source_manifest_sha256,
            "view_manifest_sha256": self.view_manifest_sha256,
            "view_ids": list(self.view_ids),
            "counts": dict(self.counts),
            "export_manifest_sha256": self.export_manifest_sha256,
            **({"output_directory": self.output_directory} if self.output_directory else {}),
        }


def export_training_views(
    view_root: str | Path,
    *,
    repo_id: str,
    source_revision: str,
    views_revision: str,
    source_prefix: str,
    output_dir: str | Path,
    source_manifest: str | Path | None = None,
    target_prefix: str = TRAINING_PREFIX,
) -> TrainingExportReceipt:
    """Materialize compact export metadata from twelve FINAL view snapshots."""
    _nonblank(repo_id, "repo_id")
    source_revision = validate_source_revision(source_revision)
    views_revision = validate_source_revision(views_revision)
    source_prefix = _prefix(source_prefix, "source_prefix")
    if target_prefix != TRAINING_PREFIX:
        raise ValueError(f"target prefix must be exactly {TRAINING_PREFIX!r}")
    view_root = Path(view_root)
    output_dir = Path(output_dir)
    if view_root.is_symlink() or not view_root.is_dir():
        raise ValueError("view_root must be a regular directory")
    if output_dir.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    if source_prefix == target_prefix or source_prefix.startswith(target_prefix + "/") or target_prefix.startswith(source_prefix + "/"):
        raise ValueError("source and training prefixes must be disjoint")
    if output_dir.exists() and any(path.suffix == ".npz" for path in output_dir.rglob("*") if path.is_file()):
        raise ValueError("training export must not contain numeric archive chunks")
    source_manifest_path = None if source_manifest is None else Path(source_manifest)
    source_payload: Mapping[str, Any] | None = None
    if source_manifest_path is not None:
        if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
            raise ValueError("source_manifest must be a regular, non-symlink file")
        source_manifest_path = source_manifest_path.resolve()
        source_payload = _loads_json(source_manifest_path.read_bytes(), field="source dataset manifest")
        if not isinstance(source_payload, Mapping):
            raise ValueError("source dataset manifest must be an object")
        source_bytes = source_manifest_path.read_bytes()
        for key, expected in (
            ("manifest_version", 3),
            ("dataset_identity", ARCHIVE_DATASET_IDENTITY),
            ("archive_format_id", ARCHIVE_FORMAT_ID),
            ("episode_schema_version", ARCHIVE_EPISODE_SCHEMA_VERSION),
        ):
            if source_payload.get(key) != expected:
                raise ValueError(f"source dataset manifest {key} mismatch")
        if source_payload.get("hf_prefix") not in {None, source_prefix}:
            raise ValueError("source dataset manifest prefix mismatch")
        recorded_revision = source_payload.get("source_revision") or source_payload.get("pinned_hf_revision")
        if recorded_revision not in {None, source_revision}:
            raise ValueError("source revision mismatch")
        source_hash = _sha256_bytes(source_bytes)
    else:
        source_hash = None
    source_rows = None
    if source_payload is not None:
        source_rows = _source_episode_rows(source_payload, source_prefix=source_prefix)

    source_manifest_sha256: str | None = None
    snapshot_identity: tuple[str, str, str, str, str] | None = None
    sample_seed: int | None = None
    snapshot_bytes: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    view_hashes: dict[str, str] = {}
    for role in _ROLES:
        for view in _VIEWS:
            view_id = f"{role}/{view}"
            path = _resolve_contained(view_root, f"{role}/{view}.json", field=f"{view_id} snapshot")
            content = path.read_bytes()
            payload = _loads_json(content, field=f"{view_id} snapshot")
            if not isinstance(payload, Mapping):
                raise ValueError(f"{view_id} snapshot must be a JSON object")
            _validate_view_snapshot(
                payload,
                role=role,
                view=view,
                source_revision=source_revision,
                source_prefix=source_prefix,
            )
            identity = (
                str(payload["dataset_identity"]),
                str(payload["archive_format_id"]),
                str(payload["episode_schema_version"]),
                str(payload["preprocessing_identity"]),
                str(payload["preprocessing_sha256"]),
            )
            if snapshot_identity is None:
                snapshot_identity = identity
            elif identity != snapshot_identity:
                raise ValueError(f"{view_id} identity differs from another view snapshot")
            if source_manifest_sha256 is None:
                source_manifest_sha256 = str(payload["source_manifest_sha256"])
            elif payload["source_manifest_sha256"] != source_manifest_sha256:
                raise ValueError(f"{view_id} source manifest hash differs from another view snapshot")
            if source_hash is not None and payload["source_manifest_sha256"] != source_hash:
                raise ValueError("final view source hash does not match source_manifest")
            if sample_seed is None:
                sample_seed = int(payload["mixture_seed"])
            elif payload["mixture_seed"] != sample_seed:
                raise ValueError(f"{view_id} mixture seed differs from another view snapshot")
            snapshot_bytes[view_id] = content
            counts[view_id] = int(payload["sample_count"])
            view_hashes[view_id] = _sha256_bytes(content)

    assert source_manifest_sha256 is not None and snapshot_identity is not None and sample_seed is not None
    if source_rows is not None:
        snapshots_for_membership = {
            view_id: _loads_json(content, field=f"{view_id} snapshot")
            for view_id, content in snapshot_bytes.items()
        }
        _validate_source_membership(snapshots_for_membership, source_rows)
    view_manifest_sha256 = _sha256_bytes(_canonical_json({item: view_hashes[item] for item in _VIEW_IDS}))
    manifest = {
        "manifest_version": 1,
        "schema_version": TRAINING_EXPORT_SCHEMA,
        "target_prefix": TRAINING_PREFIX,
        "repo_id": repo_id,
        "status": "FINAL",
        "source": {
            "archive_revision": source_revision,
            "archive_prefix": source_prefix,
            "views_revision": views_revision,
            "dataset_manifest_sha256": source_manifest_sha256,
            "dataset_identity": snapshot_identity[0],
            "archive_format_id": snapshot_identity[1],
            "episode_schema_version": snapshot_identity[2],
            "preprocessing_identity": snapshot_identity[3],
            "preprocessing_sha256": snapshot_identity[4],
            "sample_seed": sample_seed,
            "sample_seed_algorithm": "sha256-base-seed-episode-view-boundary-u32-v1",
        },
        "view_manifest_sha256": view_manifest_sha256,
        "views": [
            {
                "view_id": view_id,
                "role": view_id.split("/", 1)[0],
                "view": view_id.split("/", 1)[1],
                "path": f"views/{view_id}.json",
                "sha256": view_hashes[view_id],
                "sample_count": counts[view_id],
            }
            for view_id in _VIEW_IDS
        ],
    }
    manifest_bytes = _canonical_json(manifest)
    output_payloads = {
        **{f"views/{view_id}.json": content for view_id, content in snapshot_bytes.items()},
        "manifest.json": manifest_bytes,
    }
    _prepare_immutable_output(output_dir, output_payloads)
    for relative, content in output_payloads.items():
        _immutable_write(output_dir / relative, content)
    return TrainingExportReceipt(
        repo_id=repo_id,
        target_prefix=TRAINING_PREFIX,
        source_prefix=source_prefix,
        source_revision=source_revision,
        views_revision=views_revision,
        source_manifest_sha256=source_manifest_sha256,
        view_manifest_sha256=view_manifest_sha256,
        view_ids=_VIEW_IDS,
        counts=counts,
        status="FINAL",
        export_manifest_sha256=_sha256_bytes(manifest_bytes),
        output_directory=str(output_dir),
    )


def _load_export_root(export_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = _resolve_contained(export_root, "manifest.json", field="training export manifest")
    manifest = _loads_json(manifest_path.read_bytes(), field="training export manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("training export manifest must be an object")
    if manifest.get("manifest_version") != 1 or manifest.get("schema_version") != TRAINING_EXPORT_SCHEMA:
        raise ValueError("training export schema_version is unsupported")
    if manifest.get("target_prefix") != TRAINING_PREFIX:
        raise ValueError("training export target prefix must be exactly 'training'")
    if manifest.get("status") != "FINAL":
        raise ValueError("training export must be FINAL")
    _nonblank(manifest.get("repo_id"), "training export repo_id")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("training export source is missing")
    source_revision = validate_source_revision(source.get("archive_revision"))
    views_revision = validate_source_revision(source.get("views_revision"))
    source_prefix = _prefix(source.get("archive_prefix"), "training export archive_prefix")
    source_manifest_sha256 = source.get("dataset_manifest_sha256")
    _validate_sha256_field(source_manifest_sha256, "training export dataset manifest hash")
    for key, expected in (
        ("dataset_identity", ARCHIVE_DATASET_IDENTITY),
        ("archive_format_id", ARCHIVE_FORMAT_ID),
        ("episode_schema_version", ARCHIVE_EPISODE_SCHEMA_VERSION),
    ):
        if source.get(key) != expected:
            raise ValueError(f"training export source {key} mismatch")
    if not isinstance(source.get("preprocessing_identity"), str) or not source["preprocessing_identity"].strip():
        raise ValueError("training export preprocessing identity is missing")
    _validate_sha256_field(source.get("preprocessing_sha256"), "training export preprocessing hash")
    if source.get("preprocessing_identity") != _PREPROCESSING_IDENTITY:
        raise ValueError("training export preprocessing identity is not the native archive identity")
    if source.get("preprocessing_sha256") != _PREPROCESSING_SHA256:
        raise ValueError("training export preprocessing hash is not the native archive identity")
    if type(source.get("sample_seed")) is not int or source["sample_seed"] < 0:
        raise ValueError("training export sample_seed must be a nonnegative integer")
    if source.get("sample_seed_algorithm") != "sha256-base-seed-episode-view-boundary-u32-v1":
        raise ValueError("training export sample_seed_algorithm is unsupported")
    views = manifest.get("views")
    if (
        not isinstance(views, list)
        or len(views) != len(_VIEW_IDS)
        or any(not isinstance(item, Mapping) for item in views)
        or len({item.get("view_id") for item in views}) != len(_VIEW_IDS)
        or {item.get("view_id") for item in views} != set(_VIEW_IDS)
    ):
        raise ValueError("training export must contain exactly twelve role/view files")
    view_hashes: dict[str, str] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for item in views:
        if not isinstance(item, Mapping):
            raise ValueError("training export view entry must be an object")
        view_id = item.get("view_id")
        if view_id not in _VIEW_IDS or item.get("path") != f"views/{view_id}.json":
            raise ValueError(f"training export view path is invalid: {view_id!r}")
        digest = item.get("sha256")
        if not _is_sha256(digest):
            raise ValueError(f"training export view hash is invalid: {view_id}")
        path = _resolve_contained(export_root, item["path"], field=f"training export {view_id}")
        content = path.read_bytes()
        if _sha256_bytes(content) != digest:
            raise ValueError(f"training export view hash mismatch: {view_id}")
        payload = _loads_json(content, field=f"training export {view_id}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"training export {view_id} must be an object")
        role, view = view_id.split("/", 1)
        _validate_view_snapshot(payload, role=role, view=view, source_revision=source_revision, source_prefix=source_prefix)
        if payload["source_manifest_sha256"] != source_manifest_sha256:
            raise ValueError(f"training export {view_id} source manifest hash mismatch")
        if payload["preprocessing_identity"] != source["preprocessing_identity"]:
            raise ValueError(f"training export {view_id} preprocessing identity mismatch")
        if payload["preprocessing_sha256"] != source["preprocessing_sha256"]:
            raise ValueError(f"training export {view_id} preprocessing hash mismatch")
        if payload["mixture_seed"] != source["sample_seed"]:
            raise ValueError(f"training export {view_id} sample seed mismatch")
        if item.get("sample_count") != payload["sample_count"]:
            raise ValueError(f"training export {view_id} sample count mismatch")
        snapshots[view_id] = dict(payload)
        view_hashes[view_id] = digest
    expected_view_hash = _sha256_bytes(_canonical_json({item: view_hashes[item] for item in _VIEW_IDS}))
    _validate_sha256_field(manifest.get("view_manifest_sha256"), "training export view manifest hash")
    if manifest.get("view_manifest_sha256") != expected_view_hash:
        raise ValueError("training export view manifest hash mismatch")
    return dict(manifest), snapshots


class _HFFileStore:
    """Small cache scoped to one repository and immutable HF revision."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        root: Path,
        cache_bytes: int,
        download_file: Callable[[str, str, str], bytes | str | Path] | None = None,
    ) -> None:
        _nonblank(repo_id, "repo_id")
        revision = validate_source_revision(revision)
        cache_bytes = _cache_bytes(cache_bytes)
        _assert_safe_target(root)
        if root.is_symlink():
            raise ValueError("HF cache root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(f"{repo_id}\0{revision}".encode("utf-8")).hexdigest()
        scoped_root = root / "revisions" / identity
        _assert_safe_target(scoped_root)
        scoped_root.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.revision = revision
        self.root = scoped_root.resolve()
        self.cache_bytes = cache_bytes
        self.download_file = download_file
        self._entries: OrderedDict[Path, int] = OrderedDict()
        self._pinned: set[Path] = set()

    def _download(self, filename: str) -> bytes:
        if self.download_file is not None:
            result = self.download_file(self.repo_id, filename, self.revision)
            if isinstance(result, (str, Path)):
                return Path(result).read_bytes()
            if isinstance(result, bytes):
                return result
            raise TypeError("download_file must return bytes or a file path")
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory(prefix="icgs-training-hf-") as temporary:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                revision=self.revision,
                cache_dir=temporary,
            )
            return Path(downloaded).read_bytes()

    def fetch(
        self,
        filename: str,
        *,
        expected_sha256: str | None = None,
        destination: Path | None = None,
        pinned: bool = False,
    ) -> Path:
        filename = _safe_relative(filename, "HF filename")
        if expected_sha256 is not None:
            _validate_sha256_field(expected_sha256, "expected HF file SHA256")
        path = destination or (self.root / PurePosixPath(filename))
        _assert_safe_target(path)
        resolved_root = self.root
        if path.is_symlink():
            raise ValueError(f"HF cache file must not be a symlink: {path}")
        resolved = path.resolve(strict=False)
        if resolved_root != resolved and resolved_root not in resolved.parents:
            raise ValueError("HF destination escapes cache root")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            data = path.read_bytes()
        else:
            data = self._download(filename)
            if len(data) > self.cache_bytes:
                raise ValueError(f"HF file exceeds cache_bytes: {filename}")
            if expected_sha256 is not None and _sha256_bytes(data) != expected_sha256:
                raise ValueError(f"HF file hash mismatch: {filename}")
            pinned_bytes = sum(size for cached_path, size in self._entries.items() if cached_path in self._pinned)
            if pinned and pinned_bytes + len(data) > self.cache_bytes:
                raise ValueError("pinned HF files exceed cache_bytes")
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        if expected_sha256 is not None and _sha256_bytes(data) != expected_sha256:
            raise ValueError(f"HF cached file hash mismatch: {filename}")
        if len(data) > self.cache_bytes:
            raise ValueError(f"HF cached file exceeds cache_bytes: {filename}")
        if pinned and path not in self._pinned:
            pinned_bytes = sum(size for cached_path, size in self._entries.items() if cached_path in self._pinned)
            if pinned_bytes + len(data) > self.cache_bytes:
                raise ValueError("pinned HF files exceed cache_bytes")
        self._entries.pop(path, None)
        self._entries[path] = len(data)
        if pinned:
            self._pinned.add(path)
        self._evict(keep=path)
        return path

    def release(self, path: Path) -> None:
        self._pinned.discard(path)
        self._evict()

    def _evict(self, *, keep: Path | None = None) -> None:
        total = sum(self._entries.values())
        while total > self.cache_bytes and self._entries:
            candidate, size = next(iter(self._entries.items()))
            if candidate == keep or candidate in self._pinned:
                self._entries.move_to_end(candidate)
                if all(item == keep or item in self._pinned for item in self._entries):
                    break
                continue
            self._entries.pop(candidate)
            candidate.unlink(missing_ok=True)
            total -= size


class _RemoteEpisodeArchiveReader(EpisodeArchiveReader):
    def __init__(self, manifest_path: Path, *, store: _HFFileStore, remote_archive_root: str, **kwargs: Any) -> None:
        self._store = store
        self._remote_archive_root = _prefix(remote_archive_root, "remote archive root")
        super().__init__(manifest_path, **kwargs)

    def _load_chunk(self, relative: str) -> dict[str, np.ndarray]:
        cache_key = (self.cache_identity, relative)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached[0]
        entry = self._chunk_entries.get(relative)
        if entry is None:
            raise ValueError(f"chunk is not declared by the archive manifest: {relative}")
        local = self.root / PurePosixPath(relative)
        path = self._store.fetch(
            f"{self._remote_archive_root}/{relative}",
            expected_sha256=entry["sha256"],
            destination=local,
        )
        try:
            return super()._load_chunk(relative)
        finally:
            self._store.release(path)

    @property
    def debug_metadata(self) -> Mapping[str, Any]:
        if self._debug_metadata is None:
            relative = self.manifest.payload["debug_path"]
            entry = self._store.fetch(
                f"{self._remote_archive_root}/{relative}",
                expected_sha256=self.manifest.payload["debug_sha256"],
                destination=self.root / PurePosixPath(relative),
            )
            try:
                content = entry.read_bytes()
                if len(content) != self.manifest.payload["debug_bytes"]:
                    raise ValueError("debug metadata byte count mismatch")
                self._debug_metadata = self._decode_value(_loads_json(content, field="remote archive debug metadata"))
            finally:
                self._store.release(entry)
        return self._debug_metadata


def _source_episode_rows(payload: Mapping[str, Any], *, source_prefix: str) -> dict[str, dict[str, Any]]:
    """Build a lightweight source-manifest identity index without reading chunks."""

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or payload.get("total_episodes") not in {None, len(episodes)}:
        raise ValueError("source dataset manifest episode inventory is invalid")
    rows: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(episodes):
        if not isinstance(item, Mapping):
            raise ValueError(f"source dataset episode row {index} must be an object")
        episode_id = item.get("episode_id")
        _nonblank(episode_id, f"source episode row {index} episode_id")
        if episode_id in rows:
            raise ValueError(f"source dataset manifest contains duplicate episode_id: {episode_id}")
        archive_manifest = item.get("archive_manifest")
        _safe_relative(archive_manifest, f"source episode {episode_id} archive_manifest")
        if not archive_manifest.endswith("/episode.manifest.json"):
            raise ValueError(f"source episode {episode_id} must reference an episode manifest")
        file_hashes = item.get("file_sha256")
        if not isinstance(file_hashes, Mapping):
            raise ValueError(f"source episode {episode_id} file_sha256 inventory is missing")
        manifest_hash = item.get("archive_manifest_sha256")
        if manifest_hash is None:
            candidates = (
                archive_manifest,
                PurePosixPath(archive_manifest).name,
                "episode.manifest.json",
            )
            manifest_hash = next((file_hashes.get(candidate) for candidate in candidates if candidate in file_hashes), None)
        _validate_sha256_field(manifest_hash, f"source episode {episode_id} archive manifest hash")
        if item.get("outcome") not in {"success", "valid_failure"}:
            raise ValueError(f"source episode {episode_id} has an ineligible outcome")
        if item.get("split") not in {"train", "dev", "development", "test"}:
            raise ValueError(f"source episode {episode_id} split is invalid")
        if item.get("episode_kind") not in {"nominal", "perturbed"}:
            raise ValueError(f"source episode {episode_id} episode_kind is invalid")
        bound = dict(item)
        bound["archive_manifest_sha256"] = str(manifest_hash)
        rows[str(episode_id)] = bound
    return rows


def _validate_source_membership(
    snapshots: Mapping[str, Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    for view_id, snapshot in snapshots.items():
        role = view_id.split("/", 1)[0]
        for index, row in enumerate(snapshot["samples"]):
            source = source_rows.get(str(row["episode_id"]))
            if source is None:
                raise ValueError(f"{view_id} sample {index} is absent from the source dataset")
            if (
                source.get("archive_manifest") != row.get("archive_manifest")
                or source.get("archive_manifest_sha256") != row.get("archive_manifest_sha256")
                or source.get("episode_kind") != row.get("episode_kind")
            ):
                raise ValueError(f"{view_id} sample {index} archive identity differs from source dataset")
            split = source.get("split")
            subset = source.get("subset") or source.get("train_subset")
            expected = DEFAULT_ROLE_SPECS[role]
            if split not in expected["splits"] or subset not in expected["subsets"]:
                raise ValueError(f"{view_id} sample {index} violates frozen split membership")


class TrainingExportReader:
    """Read FINAL export references and lazily resolve source archive data."""

    def __init__(
        self,
        *,
        export_root: Path,
        manifest: Mapping[str, Any],
        snapshots: Mapping[str, Mapping[str, Any]],
        source_root: Path | None,
        store: _HFFileStore | None,
        source_prefix: str,
        cache_bytes: int,
        source_rows: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        self.export_root = export_root
        self.manifest = dict(manifest)
        self.snapshots = {key: dict(value) for key, value in snapshots.items()}
        self.repo_id = str(manifest["repo_id"])
        source = manifest["source"]
        self.source_revision = str(source["archive_revision"])
        self.views_revision = str(source["views_revision"])
        self.source_prefix = source_prefix
        self.source_manifest_sha256 = str(source["dataset_manifest_sha256"])
        self.preprocessing_identity = str(source["preprocessing_identity"])
        self.preprocessing_sha256 = str(source["preprocessing_sha256"])
        self.sample_seed = int(source["sample_seed"])
        self._source_root = source_root
        self._store = store
        self._cache_bytes = _cache_bytes(cache_bytes)
        self._source_rows = None if source_rows is None else {key: dict(value) for key, value in source_rows.items()}
        self._sample_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for view_id, snapshot in self.snapshots.items():
            view = view_id.split("/", 1)[1]
            for row in snapshot["samples"]:
                self._sample_rows.setdefault((view, str(row["episode_id"]), int(row["t"])), []).append(dict(row))
        self._readers: OrderedDict[str, EpisodeArchiveReader] = OrderedDict()
        self._reader_manifests: dict[str, Path] = {}
        self._reader_limit = max(1, cache_bytes // EpisodeArchiveReader.DEFAULT_CACHE_BYTES)

    @classmethod
    def from_local(
        cls,
        export_root: str | Path,
        *,
        source_root: str | Path | None = None,
        cache_bytes: int = EpisodeArchiveReader.DEFAULT_CACHE_BYTES,
    ) -> "TrainingExportReader":
        root = Path(export_root)
        if root.is_file():
            root = root.parent
        if root.is_symlink() or not root.is_dir():
            raise ValueError("training export root must be a regular directory")
        cache_bytes = _cache_bytes(cache_bytes)
        manifest, snapshots = _load_export_root(root)
        resolved_source = None if source_root is None else Path(source_root)
        if resolved_source is not None:
            _assert_safe_target(resolved_source)
            if resolved_source.is_symlink() or not resolved_source.is_dir():
                raise ValueError("source_root must be a regular directory")
            resolved_source = resolved_source.resolve()
        source_rows = None
        if resolved_source is not None:
            source_payload = cls._read_source_manifest(
                resolved_source,
                manifest["source"],
            )
            source_rows = _source_episode_rows(source_payload, source_prefix=str(manifest["source"]["archive_prefix"]))
        return cls(
            export_root=root.resolve(),
            manifest=manifest,
            snapshots=snapshots,
            source_root=resolved_source,
            store=None,
            source_prefix=str(manifest["source"]["archive_prefix"]),
            cache_bytes=cache_bytes,
            source_rows=source_rows,
        )

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        revision: str,
        *,
        prefix: str = TRAINING_PREFIX,
        cache_dir: str | Path,
        cache_bytes: int = EpisodeArchiveReader.DEFAULT_CACHE_BYTES,
        download_file: Callable[[str, str, str], bytes | str | Path] | None = None,
    ) -> "TrainingExportReader":
        if prefix != TRAINING_PREFIX:
            raise ValueError(f"training export prefix must be exactly {TRAINING_PREFIX!r}")
        revision = validate_source_revision(revision)
        cache_bytes = _cache_bytes(cache_bytes)
        cache_root = Path(cache_dir)
        _assert_safe_target(cache_root)
        export_store = _HFFileStore(
            repo_id=repo_id,
            revision=revision,
            root=cache_root,
            cache_bytes=cache_bytes,
            download_file=download_file,
        )
        export_root = export_store.root / "export"
        manifest_local = export_store.fetch(f"{prefix}/manifest.json", destination=export_root / "manifest.json", pinned=True)
        manifest = _loads_json(manifest_local.read_bytes(), field="training export manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("training export manifest must be an object")
        if manifest.get("repo_id") != repo_id:
            raise ValueError("training export repo_id differs from requested repository")
        # Validate the manifest header and the complete view inventory before
        # asking HF for any child file.
        if manifest.get("manifest_version") != 1 or manifest.get("schema_version") != TRAINING_EXPORT_SCHEMA:
            raise ValueError("training export manifest schema_version is unsupported")
        if manifest.get("target_prefix") != TRAINING_PREFIX or manifest.get("status") != "FINAL":
            raise ValueError("training export manifest target/status is invalid")
        manifest_source = manifest.get("source")
        if not isinstance(manifest_source, Mapping):
            raise ValueError("training export manifest source is missing")
        source_revision = validate_source_revision(manifest_source.get("archive_revision"))
        source_prefix = _prefix(manifest_source.get("archive_prefix"), "archive_prefix")
        if not isinstance(manifest.get("views"), list) or len(manifest["views"]) != len(_VIEW_IDS):
            raise ValueError("training export manifest view inventory is incomplete")
        view_entries = {item.get("view_id"): item for item in manifest["views"] if isinstance(item, Mapping)}
        if set(view_entries) != set(_VIEW_IDS):
            raise ValueError("training export manifest view inventory is invalid")
        fetched_export_paths: list[Path] = [manifest_local]
        for view_id in _VIEW_IDS:
            item = view_entries[view_id]
            if item.get("path") != f"views/{view_id}.json" or not _is_sha256(item.get("sha256")):
                raise ValueError(f"training export manifest view entry is invalid: {view_id}")
            export_store.fetch(
                f"{prefix}/{item['path']}",
                expected_sha256=item["sha256"],
                destination=export_root / item["path"],
                pinned=True,
            )
            fetched_export_paths.append(export_root / item["path"])
        manifest, snapshots = _load_export_root(export_root)
        source = manifest["source"]
        if source_revision != source["archive_revision"] or source_prefix != source["archive_prefix"]:
            raise ValueError("training export source binding changed during load")
        source_store = _HFFileStore(
            repo_id=repo_id,
            revision=source_revision,
            root=cache_root,
            cache_bytes=cache_bytes,
            download_file=download_file,
        )
        source_root = source_store.root / source_prefix
        source_manifest = source_store.fetch(
            f"{source_prefix}/dataset_manifest.json",
            expected_sha256=source["dataset_manifest_sha256"],
            destination=source_root / "dataset_manifest.json",
            pinned=True,
        )
        source_payload = _loads_json(source_manifest.read_bytes(), field="source dataset manifest")
        if not isinstance(source_payload, Mapping):
            raise ValueError("source dataset manifest must be an object")
        source_rows = _source_episode_rows(source_payload, source_prefix=source_prefix)
        for path in fetched_export_paths:
            export_store.release(path)
        source_store.release(source_manifest)
        return cls(
            export_root=export_root.resolve(),
            manifest=manifest,
            snapshots=snapshots,
            source_root=source_root.resolve(),
            store=source_store,
            source_prefix=source_prefix,
            cache_bytes=cache_bytes,
            source_rows=source_rows,
        )

    @staticmethod
    def _read_source_manifest(source_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
        path = _resolve_contained(source_root, "dataset_manifest.json", field="source dataset manifest")
        expected_sha256 = source.get("dataset_manifest_sha256")
        if _sha256_bytes(path.read_bytes()) != expected_sha256:
            raise ValueError("source dataset manifest hash mismatch")
        payload = _loads_json(path.read_bytes(), field="source dataset manifest")
        if not isinstance(payload, Mapping):
            raise ValueError("source dataset manifest must be an object")
        for key, expected in (
            ("dataset_identity", ARCHIVE_DATASET_IDENTITY),
            ("archive_format_id", ARCHIVE_FORMAT_ID),
            ("episode_schema_version", ARCHIVE_EPISODE_SCHEMA_VERSION),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"source dataset manifest {key} mismatch")
        recorded_prefix = payload.get("hf_prefix")
        if recorded_prefix is not None and recorded_prefix != source["archive_prefix"]:
            raise ValueError("source dataset manifest prefix mismatch")
        recorded_revision = payload.get("source_revision") or payload.get("pinned_hf_revision")
        if recorded_revision is not None and recorded_revision != source["archive_revision"]:
            raise ValueError("source dataset manifest revision mismatch")
        return dict(payload)

    def sample_refs(self, view: str, role: str = "train") -> Iterator[SampleRef]:
        if view not in _VIEWS:
            raise ValueError(f"unsupported training view: {view}")
        if role not in _ROLES:
            raise ValueError(f"unsupported training role: {role}")
        snapshot = self.snapshots[f"{role}/{view}"]
        for row in snapshot["samples"]:
            yield self._sample_ref(row, view=view)

    def _sample_ref(self, row: Mapping[str, Any], *, view: str) -> SampleRef:
        if row.get("view") != view or type(row.get("t")) is not int or row["t"] < 0:
            raise ValueError("training sample reference has invalid view or boundary")
        fields = {key: value for key, value in row.items() if key not in _SAMPLE_REF_FIELDS}
        return SampleRef(
            episode_id=str(row["episode_id"]),
            view=view,
            t=int(row["t"]),
            archive_manifest=str(row["archive_manifest"]),
            archive_manifest_sha256=str(row["archive_manifest_sha256"]),
            dataset_manifest_sha256=str(row["dataset_manifest_sha256"]),
            archive_preprocessing_identity=str(row["archive_preprocessing_identity"]),
            preprocessing_identity=str(row["preprocessing_identity"]),
            preprocessing_sha256=str(row["preprocessing_sha256"]),
            sample_seed=int(row["sample_seed"]),
            fields=fields,
        )

    def _validate_ref(self, ref: SampleRef) -> None:
        if not isinstance(ref, SampleRef):
            raise TypeError("training reader expects a SampleRef")
        if (
            ref.view not in _VIEWS
            or ref.preprocessing_identity != self.preprocessing_identity
            or ref.preprocessing_sha256 != self.preprocessing_sha256
            or ref.dataset_manifest_sha256 != self.source_manifest_sha256
        ):
            raise ValueError("sample reference preprocessing identity does not belong to this export")
        canonical = self._sample_rows.get((ref.view, ref.episode_id, ref.t), ())
        ref_payload = _jsonable(ref.as_dict())
        if not any(dict(row) == ref_payload for row in canonical):
            raise ValueError("sample reference does not belong to this training export")
        if self._source_rows is not None:
            source_row = self._source_rows.get(ref.episode_id)
            if source_row is None:
                raise ValueError("sample reference episode is absent from the source dataset")
            if (
                source_row.get("archive_manifest") != ref.archive_manifest
                or source_row.get("archive_manifest_sha256") != ref.archive_manifest_sha256
                or source_row.get("preprocessing_identity") != ref.archive_preprocessing_identity
            ):
                raise ValueError("sample reference archive identity does not belong to the source dataset")
            if source_row.get("outcome") not in {"success", "valid_failure"}:
                raise ValueError("sample reference points to an ineligible source outcome")

    def reader_for(self, ref: SampleRef) -> EpisodeArchiveReader:
        self._validate_ref(ref)
        cached = self._readers.get(ref.episode_id)
        if cached is not None:
            self._readers.move_to_end(ref.episode_id)
            return cached
        if self._source_root is None:
            raise RuntimeError("this training reader has no source archive root; use from_hf or pass source_root")
        if self._store is None:
            manifest_path = _resolve_contained(self._source_root, ref.archive_manifest, field="archive manifest")
            if _sha256_bytes(manifest_path.read_bytes()) != ref.archive_manifest_sha256:
                raise ValueError("archive manifest hash mismatch")
            reader: EpisodeArchiveReader = EpisodeArchiveReader(
                manifest_path,
                cache_bytes=min(self._cache_bytes, EpisodeArchiveReader.DEFAULT_CACHE_BYTES),
                resolved_preprocessing_identity=self.preprocessing_identity,
                resolved_preprocessing_sha256=self.preprocessing_sha256,
            )
        else:
            remote_path = f"{self.source_prefix}/{ref.archive_manifest}"
            manifest_path = self._store.fetch(
                remote_path,
                expected_sha256=ref.archive_manifest_sha256,
                destination=self._source_root / PurePosixPath(ref.archive_manifest),
                pinned=True,
            )
            archive_root = f"{self.source_prefix}/{PurePosixPath(ref.archive_manifest).parent.as_posix()}"
            reader = _RemoteEpisodeArchiveReader(
                manifest_path,
                store=self._store,
                remote_archive_root=archive_root,
                cache_bytes=min(self._cache_bytes, EpisodeArchiveReader.DEFAULT_CACHE_BYTES),
                resolved_preprocessing_identity=self.preprocessing_identity,
                resolved_preprocessing_sha256=self.preprocessing_sha256,
            )
        self._readers[ref.episode_id] = reader
        if self._store is not None:
            self._reader_manifests[ref.episode_id] = manifest_path
        while len(self._readers) > self._reader_limit:
            evicted_episode, _ = self._readers.popitem(last=False)
            evicted_manifest = self._reader_manifests.pop(evicted_episode, None)
            if evicted_manifest is not None:
                self._store.release(evicted_manifest)
        return reader

    def read_observation(self, ref: SampleRef, boundary: int | None = None) -> Mapping[str, Any]:
        target = ref.t if boundary is None else boundary
        return self.reader_for(ref).observation(target)

    def read_transition(self, ref: SampleRef, index: int | None = None) -> Mapping[str, Any]:
        target = ref.t if index is None else index
        return self.reader_for(ref).transition(target)

    def _provenance(self, ref: SampleRef) -> dict[str, Any]:
        provenance = dict(ref.fields)
        if self._source_rows is not None:
            source = self._source_rows.get(ref.episode_id)
            if source is not None:
                for field in ("outcome", "split", "subset", "program_id"):
                    provenance.setdefault(field, source.get(field))
        return provenance

    def preprocess_observation(self, ref: SampleRef, *, boundary: int | None = None) -> np.ndarray:
        target = ref.t if boundary is None else boundary
        observation = self.reader_for(ref).observation(target)
        filtered, _ = _remove_statistical_outliers(observation["points"], nb_neighbors=20, std_ratio=2.0)
        filtered = np.asarray(filtered)
        if len(filtered) == 0:
            raise ValueError("native 2,048-point preprocessing does not support empty point clouds")
        seed = ref.sample_seed if target == ref.t else _sample_seed(
            self.sample_seed, ref.episode_id, ref.view, target
        )
        rng = np.random.RandomState(seed)
        selected = filtered[rng.choice(len(filtered), 2048, replace=len(filtered) < 2048)]
        return transform_pcd(selected, np.linalg.inv(observation["T_w_e"]))

    @staticmethod
    def _transition_sample(archive: EpisodeArchiveReader, index: int) -> dict[str, Any]:
        transition = dict(archive.transition(index))
        before_boundary = transition.get("before_boundary", index)
        after_boundary = transition.get("after_boundary", index + 1)
        transition["before"] = {
            "observation": archive.observation(int(before_boundary)),
            "boundary": int(before_boundary),
        }
        transition["after"] = {
            "observation": archive.observation(int(after_boundary)),
            "boundary": int(after_boundary),
        }
        return transition

    def read_sample(
        self,
        ref: SampleRef,
        *,
        supervised_intervals: int = 1,
        burnin: int = 8,
    ) -> dict[str, Any]:
        self._validate_ref(ref)
        if type(supervised_intervals) is not int or supervised_intervals <= 0:
            raise ValueError("supervised_intervals must be a positive integer")
        if type(burnin) is not int or burnin < 0:
            raise ValueError("burnin must be a nonnegative integer")
        archive = self.reader_for(ref)
        if ref.view == "D_geom":
            observation = archive.observation(ref.t)
            return {
                "episode_id": ref.episode_id,
                "boundary": ref.t,
                "points": observation["points"],
                "valid": observation["point_valid"],
                "T_w_e": observation["T_w_e"],
                "grip": observation["grip"],
                "provenance": self._provenance(ref),
            }
        if ref.view == "D_temporal":
            observation_indices = [item[2] for item in ref.fields.get("observations", [])]
            action_indices = [item[2] for item in ref.fields.get("actions", [])]
            transitions = tuple(self._transition_sample(archive, index) for index in action_indices)
            return {
                "episode_id": ref.episode_id,
                "boundary": ref.t,
                "observations": tuple(archive.observation(index) for index in observation_indices),
                "transitions": transitions,
                "history_transitions": transitions,
                "target_transitions": tuple(),
                "dt": archive.transition(ref.t)["achieved_duration_s"],
                "provenance": self._provenance(ref),
            }
        if ref.view == "D_dyn":
            transition_count = int(archive.manifest.payload["timeline"]["transitions"])
            target_stop = ref.t + supervised_intervals
            if target_stop > transition_count:
                raise ValueError(
                    f"D_dyn sample at boundary {ref.t} has only {transition_count - ref.t} supervised intervals; "
                    f"requested window {supervised_intervals}"
                )
            history = tuple(self._transition_sample(archive, index) for index in range(ref.t))
            targets = tuple(self._transition_sample(archive, index) for index in range(ref.t, target_stop))
            return {
                "episode_id": ref.episode_id,
                "boundary": ref.t,
                "history_transitions": history,
                "target_transitions": targets,
                "transitions": history + targets,
                "observation_t": targets[0]["before"]["observation"],
                "observation_t1": targets[0]["after"]["observation"],
                "supervised_intervals": supervised_intervals,
                "burnin": min(burnin, len(history)),
                "provenance": self._provenance(ref),
            }
        metadata = archive.manifest.payload.get("record_metadata", {})
        events = metadata.get("events") if isinstance(metadata, Mapping) else None
        resolved_events = [dict(event) for event in (events or ()) if isinstance(event, Mapping)]
        return {
            "episode_id": ref.episode_id,
            "boundary": ref.t,
            "labels": dict(archive.task_labels(ref.t)),
            "events": resolved_events,
            "provenance": self._provenance(ref),
        }

    def a0(self, ref: SampleRef) -> dict[str, Any]:
        if ref.view != "D_geom":
            raise ValueError("A0 adapter requires a D_geom sample")
        return adapter_a0(self.read_sample(ref))

    def a1(self, ref: SampleRef, *, supervised_intervals: int = 1, burnin: int = 8) -> dict[str, Any]:
        if ref.view not in {"D_temporal", "D_dyn"}:
            raise ValueError("A1 adapter requires a D_temporal or D_dyn sample")
        return adapter_a1(self.read_sample(ref, supervised_intervals=supervised_intervals, burnin=burnin))


__all__ = [
    "TRAINING_EXPORT_SCHEMA",
    "TRAINING_PREFIX",
    "TrainingExportReceipt",
    "TrainingExportReader",
    "export_training_views",
]
