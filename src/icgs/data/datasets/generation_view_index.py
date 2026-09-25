"""Revision-bound provisional pointers and final generation view snapshots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from types import MappingProxyType
from typing import Any

from icgs.data.collection.generation.distributed_contracts import (
    ARCHIVE_DATASET_IDENTITY,
    ARCHIVE_EPISODE_SCHEMA_VERSION,
    ARCHIVE_FORMAT_ID,
)
from icgs.data.collection.generation.episode_archive import ArchiveManifest
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL
from icgs.data.datasets.generation_archive import ArchiveDatasetIndex, SampleRef


_ROLE_ORDER = ("train", "validation", "evaluation")
_VIEW_ORDER = tuple(GENERATION_PROTOCOL.views)
_MIXTURE_VIEWS = frozenset(GENERATION_PROTOCOL.mixture_views)
_PINNED_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DEFAULT_ROLE_SPEC_DATA = {
    "train": {"splits": ("train",), "subsets": (GENERATION_PROTOCOL.train_on,)},
    "validation": {"splits": ("train",), "subsets": (GENERATION_PROTOCOL.validation,)},
    "evaluation": {"splits": ("development", "dev", "test"), "subsets": (None,)},
}
DEFAULT_ROLE_SPECS: Mapping[str, Mapping[str, tuple[Any, ...]]] = MappingProxyType({
    role: MappingProxyType(dict(spec))
    for role, spec in _DEFAULT_ROLE_SPEC_DATA.items()
})


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _compact_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_source_revision(source_revision: str) -> str:
    """Require a full immutable HF commit identifier, never a branch or tag."""
    if not isinstance(source_revision, str) or not _PINNED_REVISION.fullmatch(source_revision):
        raise ValueError("source_revision must be a pinned lowercase 40- or 64-character HF commit OID")
    return source_revision


def _safe_prefix(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank HF-relative prefix")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or any(part in {"", "."} for part in path.parts)
        or path.as_posix() != value.rstrip("/")
    ):
        raise ValueError(f"{field_name} must be a normalized HF-relative prefix")
    return value.rstrip("/")


def _safe_segment(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be a safe path segment")
    return value


def _episode_payload(episode_manifest: Mapping[str, Any] | ArchiveManifest) -> dict[str, Any]:
    if isinstance(episode_manifest, ArchiveManifest):
        archive = episode_manifest
    else:
        archive = ArchiveManifest.from_dict(episode_manifest)
    payload = archive.as_dict()
    if payload.get("archive_kind") != "episode":
        raise ValueError("provisional episode pointers require an episode archive manifest")
    if payload.get("outcome") not in {"success", "valid_failure"}:
        raise ValueError("crash and invalid attempts cannot have provisional training views")
    for key, expected in (
        ("dataset_identity", ARCHIVE_DATASET_IDENTITY),
        ("archive_format_id", ARCHIVE_FORMAT_ID),
        ("episode_schema_version", ARCHIVE_EPISODE_SCHEMA_VERSION),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"provisional episode pointer {key} mismatch")
    for key in ("episode_id", "program_id"):
        _safe_segment(payload.get(key), key)
    provenance = payload.get("record_metadata", {}).get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("provisional episode pointer provenance is missing")
    episode_kind = provenance.get("episode_kind")
    if episode_kind not in {"nominal", "perturbed"}:
        raise ValueError("provisional episode pointer episode_kind is unsupported")
    if payload.get("split") not in {"train", "dev", "development", "test"}:
        raise ValueError("provisional episode pointer split is unsupported")
    timeline = payload.get("timeline")
    if not isinstance(timeline, Mapping):
        raise ValueError("provisional episode pointer timeline is missing")
    observations, transitions = timeline.get("observations"), timeline.get("transitions")
    if (
        type(observations) is not int
        or type(transitions) is not int
        or observations != transitions + 1
        or transitions <= 0
    ):
        raise ValueError("provisional episode pointer timeline must contain T+1 observations and T transitions")
    return payload


def _provisional_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    observations = int(payload["timeline"]["observations"])
    transitions = int(payload["timeline"]["transitions"])
    metadata = payload["record_metadata"]
    events = metadata.get("events") or ()
    preferred: set[int] = set()
    for event in events:
        if isinstance(event, Mapping):
            start = int(event.get("start_t", 0))
            preferred.update(range(max(0, start - 1), min(transitions, start + 2)))
    if not preferred:
        preferred = set(range(transitions))
    return {
        "D_geom": observations,
        "D_temporal": len(preferred),
        "D_dyn": transitions,
        "D_task": observations,
    }


def build_provisional_episode_pointers(
    episode_manifest: Mapping[str, Any] | ArchiveManifest,
) -> tuple[dict[str, Any], ...]:
    """Build four metadata-only per-episode discovery pointers.

    These pointers intentionally expose role ``all`` with mixing disabled. They
    are provisional references to an immutable episode archive and never claim
    to be a frozen, role-filtered training snapshot.
    """
    payload = _episode_payload(episode_manifest)
    episode_id = str(payload["episode_id"])
    program_id = str(payload["program_id"])
    archive_ref = f"episodes/{program_id}/{episode_id}"
    archive_manifest = f"{archive_ref}/episode.manifest.json"
    archive_manifest_sha256 = _sha256(_compact_json_bytes(payload))
    counts = _provisional_counts(payload)
    provenance = payload["record_metadata"]["provenance"]
    return tuple({
        "manifest_version": 1,
        "view": view,
        "view_id": f"provisional/{program_id}/{episode_id}/{view}",
        "status": "PROVISIONAL",
        "view_status": "PROVISIONAL",
        "role": "all",
        "mix": False,
        "dataset_identity": payload["dataset_identity"],
        "archive_format_id": payload["archive_format_id"],
        "episode_schema_version": payload["episode_schema_version"],
        "episode_id": episode_id,
        "program_id": program_id,
        "split": payload["split"],
        "subset": payload.get("subset"),
        "episode_kind": provenance.get("episode_kind"),
        "outcome": payload["outcome"],
        "archive_ref": archive_ref,
        "archive_manifest": archive_manifest,
        "archive_manifest_sha256": archive_manifest_sha256,
        "preprocessing_identity": payload["preprocessing_identity"],
        "sample_count": counts[view],
        "pointers_only": True,
    } for view in _VIEW_ORDER)


def _normalized_role_specs(role_specs: Mapping[str, Any]) -> dict[str, dict[str, list[Any]]]:
    if not isinstance(role_specs, Mapping) or set(role_specs) != set(_ROLE_ORDER):
        raise ValueError("role specification must define exactly train, validation, and evaluation")
    normalized: dict[str, dict[str, list[Any]]] = {}
    for role in _ROLE_ORDER:
        value = role_specs[role]
        if not isinstance(value, Mapping) or set(value) != {"splits", "subsets"}:
            raise ValueError(f"role specification for {role} must contain splits and subsets")
        fields: dict[str, list[Any]] = {}
        for field_name in ("splits", "subsets"):
            values = value[field_name]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise ValueError(f"role specification {role}.{field_name} must be a nonempty sequence")
            fields[field_name] = list(values)
        expected = _DEFAULT_ROLE_SPEC_DATA[role]
        if (
            set(fields["splits"]) != set(expected["splits"])
            or set(fields["subsets"]) != set(expected["subsets"])
            or len(fields["splits"]) != len(expected["splits"])
            or len(fields["subsets"]) != len(expected["subsets"])
        ):
            raise ValueError(f"role specification for {role} conflicts with the frozen generation split contract")
        normalized[role] = {
            field_name: list(expected[field_name])
            for field_name in ("splits", "subsets")
        }
    return normalized


def _source_timestamp(manifest: Mapping[str, Any]) -> int | float:
    value = manifest.get("source_snapshot_created_at_s")
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError("source snapshot timestamp is missing or invalid")
    return value


def _canonical_role_refs(
    index: ArchiveDatasetIndex,
    *,
    view: str,
    role: str,
    mixture_seed: int,
) -> tuple[list[SampleRef], bool, bool]:
    refs = list(index.sample_refs(view, role))
    apply_mixture = role == "train" and view in _MIXTURE_VIEWS
    if not apply_mixture:
        return refs, False, False
    from icgs.data.collection.generation.sampler import mix_training_transitions

    by_key = {(ref.episode_id, ref.t): ref for ref in refs}
    selected_rows = mix_training_transitions(
        [ref.as_dict() for ref in refs],
        view=view,
        seed=mixture_seed,
    )
    selected = [by_key[(str(row["episode_id"]), int(row["t"]))] for row in selected_rows]
    return selected, True, len(selected) != len(refs)


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.parent.is_symlink():
        raise ValueError(f"final view output directory must not be a symlink: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"final view manifest must not be a symlink: {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"final view manifest must be a regular file: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"immutable final view conflict: {path}")
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
                raise ValueError(f"immutable final view conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ViewFinalizationReceipt:
    source_manifest_sha256: str
    source_revision: str
    view_ids: tuple[str, ...]
    counts: Mapping[str, int]
    mixture_identity: Mapping[str, Any]
    status: str
    finalization_timestamp_s: int | float
    view_manifest_sha256: str
    manifest_sha256s: Mapping[str, str] = field(default_factory=dict)
    output_directory: str = ""

    def __post_init__(self) -> None:
        if self.status != "FINAL":
            raise ValueError("view finalization receipt status must be FINAL")
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "mixture_identity", MappingProxyType(dict(self.mixture_identity)))
        object.__setattr__(self, "manifest_sha256s", MappingProxyType(dict(self.manifest_sha256s)))

    @property
    def source_hf_revision(self) -> str:
        return self.source_revision

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": 1,
            "status": self.status,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_revision": self.source_revision,
            "view_ids": list(self.view_ids),
            "counts": dict(self.counts),
            "mixture_identity": dict(self.mixture_identity),
            "finalization_timestamp_s": self.finalization_timestamp_s,
            "finalization_timestamp_source": "source_snapshot_created_at_s",
            "view_manifest_sha256": self.view_manifest_sha256,
            "manifest_sha256s": dict(self.manifest_sha256s),
        }


def finalize_generation_views(
    dataset_manifest: str | Path,
    *,
    source_revision: str,
    role_specs: Mapping[str, Any],
    mixture_seed: int,
    output_dir: str | Path | None = None,
) -> ViewFinalizationReceipt:
    """Write immutable, role-specific pointer snapshots for a frozen HF revision.

    ``dataset_manifest`` must be the dataset manifest materialized from the
    supplied immutable ``source_revision``. The publisher stores the source
    snapshot timestamp in that manifest; this function never substitutes wall
    clock time when replaying finalization.
    """
    revision = validate_source_revision(source_revision)
    if type(mixture_seed) is not int or mixture_seed < 0:
        raise ValueError("mixture_seed must be a nonnegative integer")
    normalized_specs = _normalized_role_specs(role_specs)
    manifest_path = Path(dataset_manifest)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("dataset_manifest must be a regular, non-symlink file from the pinned HF snapshot")
    manifest_path = manifest_path.resolve()
    source_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset manifest must be an object")
    for key, expected in (
        ("dataset_identity", ARCHIVE_DATASET_IDENTITY),
        ("archive_format_id", ARCHIVE_FORMAT_ID),
        ("episode_schema_version", ARCHIVE_EPISODE_SCHEMA_VERSION),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"final view source {key} mismatch")
    if manifest.get("view_status") != "PROVISIONAL":
        raise ValueError("final views require a PROVISIONAL source dataset manifest")
    source_prefix = _safe_prefix(manifest.get("hf_prefix"), "hf_prefix")
    for field_name in ("source_revision", "pinned_hf_revision"):
        recorded_revision = manifest.get(field_name)
        if recorded_revision is not None and recorded_revision != revision:
            raise ValueError(f"source revision conflicts with dataset manifest {field_name}")
    timestamp = _source_timestamp(manifest)
    episodes = manifest.get("episodes")
    failure_attempts = manifest.get("failure_attempts")
    if not isinstance(episodes, list) or not isinstance(failure_attempts, list):
        raise ValueError("complete dataset manifest must contain episode and failure_attempt lists")
    if type(manifest.get("total_episodes")) is not int or manifest["total_episodes"] != len(episodes):
        raise ValueError("complete dataset manifest total_episodes binding is missing or inconsistent")
    if (
        type(manifest.get("total_failure_attempts")) is not int
        or manifest["total_failure_attempts"] != len(failure_attempts)
    ):
        raise ValueError("complete dataset manifest total_failure_attempts binding is missing or inconsistent")

    index = ArchiveDatasetIndex.from_manifest(manifest_path, sample_seed=mixture_seed)
    source_manifest_sha256 = index.dataset_manifest_sha256
    base_output = (
        Path(output_dir)
        if output_dir is not None
        else manifest_path.parent / "views" / "final" / revision / f"seed-{mixture_seed}"
    )
    if base_output.is_symlink():
        raise ValueError("final view output directory must not be a symlink")

    counts: dict[str, int] = {}
    manifest_hashes: dict[str, str] = {}
    mixture_identity = {
        "algorithm": "sha256-rank-70-30-training-transitions-v1",
        "target": list(GENERATION_PROTOCOL.warmup_mixture),
        "measured_on": GENERATION_PROTOCOL.mixture_measured_on,
        "views": list(GENERATION_PROTOCOL.mixture_views),
        "seed": mixture_seed,
    }
    for role in _ROLE_ORDER:
        for view in _VIEW_ORDER:
            view_id = f"{role}/{view}"
            selected, mixture_requested, mixture_changed = _canonical_role_refs(
                index,
                view=view,
                role=role,
                mixture_seed=mixture_seed,
            )
            kind_counts = Counter(
                "perturbed" if ref.fields.get("episode_kind") == "perturbed" else "nominal"
                for ref in selected
            )
            sample_rows: list[dict[str, Any]] = []
            for ref in selected:
                row = ref.as_dict()
                row["archive_manifest_path"] = f"{source_prefix}/{ref.archive_manifest}"
                sample_rows.append(row)
            mixture = {
                **mixture_identity,
                "applied": mixture_requested,
                "selection_changed": mixture_changed,
                "seed": mixture_seed if mixture_requested else None,
            }
            snapshot = {
                "manifest_version": 1,
                "snapshot_kind": "generation_training_view",
                "view_id": view_id,
                "view": view,
                "role": role,
                "role_spec": normalized_specs[role],
                "status": "FINAL",
                "view_status": "FINAL",
                "source_revision": revision,
                "source_prefix": source_prefix,
                "source_manifest_sha256": source_manifest_sha256,
                "dataset_identity": index.dataset_identity,
                "archive_format_id": manifest["archive_format_id"],
                "episode_schema_version": manifest["episode_schema_version"],
                "archive_preprocessing_identities": sorted({
                    ref.archive_preprocessing_identity for ref in selected
                }),
                "preprocessing_identity": index.preprocessing_identity,
                "preprocessing_sha256": index.preprocessing_sha256,
                "mixture_seed": mixture_seed,
                "mixture": mixture,
                "finalization_timestamp_s": timestamp,
                "finalization_timestamp_source": "source_snapshot_created_at_s",
                "source_snapshot_created_at_s": timestamp,
                "finalization_timestamp_definition": "time the pinned dataset manifest was materialized",
                "sample_count": len(sample_rows),
                "sample_counts_by_episode_kind": dict(sorted(kind_counts.items())),
                "samples": sample_rows,
            }
            content = _canonical_json_bytes(snapshot)
            relative_path = Path(role) / f"{view}.json"
            _write_immutable(base_output / relative_path, content)
            counts[view_id] = len(sample_rows)
            manifest_hashes[view_id] = _sha256(content)

    ordered_ids = tuple(f"{role}/{view}" for role in _ROLE_ORDER for view in _VIEW_ORDER)
    view_manifest_sha256 = _sha256(_canonical_json_bytes({
        view_id: manifest_hashes[view_id] for view_id in ordered_ids
    }))
    receipt = ViewFinalizationReceipt(
        source_manifest_sha256=source_manifest_sha256,
        source_revision=revision,
        view_ids=ordered_ids,
        counts=counts,
        mixture_identity=mixture_identity,
        status="FINAL",
        finalization_timestamp_s=timestamp,
        view_manifest_sha256=view_manifest_sha256,
        manifest_sha256s=manifest_hashes,
        output_directory=str(base_output),
    )
    _write_immutable(base_output / "receipt.json", _canonical_json_bytes(receipt.as_dict()))
    return receipt


__all__ = [
    "DEFAULT_ROLE_SPECS",
    "ViewFinalizationReceipt",
    "build_provisional_episode_pointers",
    "finalize_generation_views",
    "validate_source_revision",
]
