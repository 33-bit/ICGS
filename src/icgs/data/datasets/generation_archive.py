"""Lazy indexing and sample references for canonical generation archives.

The index reads the frozen dataset manifest and compact episode manifests only.
Point and transition chunks remain unopened until a caller resolves a reference
through :meth:`ArchiveDatasetIndex.reader_for`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from icgs.data.collection.generation.distributed_contracts import (
    ARCHIVE_DATASET_IDENTITY,
    ARCHIVE_EPISODE_SCHEMA_VERSION,
    ARCHIVE_FORMAT_ID,
)
from icgs.data.collection.generation.episode_archive import (
    ArchiveManifest,
    EpisodeArchiveReader,
)
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL


_PREPROCESSING_IDENTITY = "icgs_ip_native_sor20_std2_mt19937_2048_local_v1"
_PREPROCESSING_SPEC = {
    "identity": _PREPROCESSING_IDENTITY,
    "input": "online_observations.points",
    "point_valid_mask": "not-applied-to-match-native-ip-preprocessing",
    "outlier_filter": {
        "implementation": "open3d.geometry.PointCloud.remove_statistical_outlier",
        "nb_neighbors": 20,
        "std_ratio": 2.0,
    },
    "sampling": {
        "implementation": "numpy.random.RandomState.choice",
        "num_points": 2048,
        "replace_when_source_points_below_num_points": True,
        "seed_derivation": "sha256-base-seed-episode-view-boundary-u32-v1",
    },
    "frame": "transform_pcd(points, inverse(T_w_e))",
}
_PREPROCESSING_SHA256 = hashlib.sha256(
    (json.dumps(_PREPROCESSING_SPEC, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
).hexdigest()
_OPTIONAL_MODALITIES = ("rgb", "depth", "mask")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a contained relative path")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized relative path")
    return value


def _resolve_contained_file(root: Path, relative: str, *, field: str) -> Path:
    relative = _safe_relative_path(relative, field)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{field} path component must not be a symlink: {path}")
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{field} traverses outside the dataset manifest directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} is not a regular file: {resolved}")
    return resolved


def _sample_seed(base_seed: int, episode_id: str, view: str, boundary: int) -> int:
    payload = f"{base_seed}|{episode_id}|{view}|{boundary}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _remove_statistical_outliers(points: np.ndarray, *, nb_neighbors: int, std_ratio: float):
    """Load the native Open3D operation only when a point cloud is resolved."""
    from icgs.data.preprocessing.native import remove_statistical_outliers

    return remove_statistical_outliers(points, nb_neighbors=nb_neighbors, std_ratio=std_ratio)


@dataclass(frozen=True)
class SampleRef(Mapping[str, Any]):
    """Immutable, metadata-only reference to one archive-backed training sample."""

    episode_id: str
    view: str
    t: int
    archive_manifest: str
    archive_manifest_sha256: str
    dataset_manifest_sha256: str
    archive_preprocessing_identity: str
    preprocessing_identity: str
    preprocessing_sha256: str
    sample_seed: int
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze(self.fields))

    def as_dict(self) -> dict[str, Any]:
        return {
            **dict(self.fields),
            "episode_id": self.episode_id,
            "view": self.view,
            "t": self.t,
            "archive_manifest": self.archive_manifest,
            "archive_manifest_sha256": self.archive_manifest_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "archive_preprocessing_identity": self.archive_preprocessing_identity,
            "preprocessing_identity": self.preprocessing_identity,
            "preprocessing_sha256": self.preprocessing_sha256,
            "sample_seed": self.sample_seed,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


@dataclass(frozen=True)
class _EpisodeEntry:
    episode_id: str
    archive_manifest: str
    archive_manifest_path: Path
    archive_manifest_sha256: str
    payload: Mapping[str, Any]
    provenance: Mapping[str, Any]
    observations: int
    transitions: int


class ArchiveDatasetIndex:
    """Index archive metadata without inflating any NPZ array chunks."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        dataset_manifest_sha256: str,
        dataset_identity: str,
        source_revision: str | None,
        entries: tuple[_EpisodeEntry, ...],
        sample_seed: int,
        cache_bytes: int,
    ) -> None:
        self.dataset_root = dataset_root
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.dataset_identity = dataset_identity
        self.source_revision = source_revision
        self.sample_seed = sample_seed
        self.cache_bytes = cache_bytes
        self.preprocessing_identity = _PREPROCESSING_IDENTITY
        self.preprocessing_sha256 = _PREPROCESSING_SHA256
        self._entries = entries
        self._entries_by_episode = {entry.episode_id: entry for entry in entries}
        self._reader_cache_limit = max(1, cache_bytes // EpisodeArchiveReader.DEFAULT_CACHE_BYTES)
        self._reader_cache_bytes = cache_bytes // self._reader_cache_limit
        self._readers: OrderedDict[str, EpisodeArchiveReader] = OrderedDict()

    @classmethod
    def from_manifest(
        cls,
        dataset_manifest_path: str | Path,
        *,
        sample_seed: int = GENERATION_PROTOCOL.collection_seed,
        cache_bytes: int = EpisodeArchiveReader.DEFAULT_CACHE_BYTES,
    ) -> "ArchiveDatasetIndex":
        if type(sample_seed) is not int or sample_seed < 0:
            raise ValueError("sample_seed must be a nonnegative integer")
        if type(cache_bytes) is not int or cache_bytes <= 0:
            raise ValueError("cache_bytes must be a positive integer")
        manifest_path = Path(dataset_manifest_path)
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("dataset manifest must be a regular, non-symlink file")
        manifest_path = manifest_path.resolve()
        manifest_bytes = manifest_path.read_bytes()
        try:
            dataset = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("dataset manifest is not valid JSON") from error
        if not isinstance(dataset, Mapping) or dataset.get("manifest_version") != 3:
            raise ValueError("archive dataset manifest_version must be 3")
        if dataset.get("dataset_identity") != ARCHIVE_DATASET_IDENTITY:
            raise ValueError("archive dataset identity mismatch")
        if dataset.get("archive_format_id") != ARCHIVE_FORMAT_ID:
            raise ValueError("archive format identity mismatch")
        if dataset.get("episode_schema_version") != ARCHIVE_EPISODE_SCHEMA_VERSION:
            raise ValueError("archive episode schema identity mismatch")
        episode_rows = dataset.get("episodes")
        if not isinstance(episode_rows, list):
            raise ValueError("dataset manifest episodes must be a list")

        dataset_root = manifest_path.parent.resolve()
        seen_episode_ids: set[str] = set()
        entries: list[_EpisodeEntry] = []
        for row_index, row in enumerate(episode_rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"dataset manifest episode {row_index} must be an object")
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise ValueError(f"dataset manifest episode {row_index} has an invalid episode_id")
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode_id in dataset manifest: {episode_id}")
            seen_episode_ids.add(episode_id)
            if row.get("outcome") not in {"success", "valid_failure"}:
                raise ValueError(f"episode {episode_id} has a non-training outcome")

            relative_manifest = _safe_relative_path(row.get("archive_manifest"), "archive_manifest")
            episode_manifest_path = _resolve_contained_file(
                dataset_root,
                relative_manifest,
                field=f"episode {episode_id} archive_manifest",
            )
            manifest_bytes = episode_manifest_path.read_bytes()
            manifest_digest = _sha256_bytes(manifest_bytes)
            file_sha256 = row.get("file_sha256")
            if not isinstance(file_sha256, Mapping) or file_sha256.get(PurePosixPath(relative_manifest).name) != manifest_digest:
                raise ValueError(f"episode {episode_id} archive manifest SHA256 mismatch")
            try:
                archive_payload = json.loads(manifest_bytes)
                archive_manifest = ArchiveManifest.from_dict(archive_payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"episode {episode_id} archive manifest is not valid JSON") from error
            payload = archive_manifest.payload
            if payload.get("archive_kind") != "episode":
                raise ValueError(f"episode {episode_id} points to a non-episode archive")
            if payload.get("episode_id") != episode_id or payload.get("outcome") != row.get("outcome"):
                raise ValueError(f"episode {episode_id} archive identity differs from dataset manifest")
            for key, expected in (
                ("archive_format_id", ARCHIVE_FORMAT_ID),
                ("episode_schema_version", ARCHIVE_EPISODE_SCHEMA_VERSION),
                ("dataset_identity", ARCHIVE_DATASET_IDENTITY),
            ):
                if row.get(key) != expected or payload.get(key) != expected:
                    raise ValueError(f"episode {episode_id} {key} mismatch")

            record_metadata = payload["record_metadata"]
            provenance = record_metadata.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"episode {episode_id} archive provenance is missing")
            for field in ("program_id", "split", "subset", "episode_kind"):
                recorded = provenance.get(field)
                if field == "subset" and recorded is None:
                    recorded = provenance.get("train_subset")
                if row.get(field) != recorded:
                    raise ValueError(f"episode {episode_id} {field} differs from its archive manifest")
            if row.get("preprocessing_identity") != payload.get("preprocessing_identity"):
                raise ValueError(f"episode {episode_id} preprocessing identity differs from its archive manifest")

            timeline = payload.get("timeline")
            if not isinstance(timeline, Mapping):
                raise ValueError(f"episode {episode_id} timeline is missing")
            observations = timeline.get("observations")
            transitions = timeline.get("transitions")
            if (
                type(observations) is not int
                or type(transitions) is not int
                or observations != transitions + 1
                or transitions <= 0
            ):
                raise ValueError(f"episode {episode_id} timeline must contain T transitions and T+1 observations")
            entries.append(_EpisodeEntry(
                episode_id=episode_id,
                archive_manifest=relative_manifest,
                archive_manifest_path=episode_manifest_path,
                archive_manifest_sha256=manifest_digest,
                payload=payload,
                provenance=provenance,
                observations=observations,
                transitions=transitions,
            ))

        source_revision = dataset.get("dataset_manifest_revision") or dataset.get("source_revision")
        if source_revision is not None and not isinstance(source_revision, str):
            raise ValueError("dataset manifest source revision must be a string when present")
        return cls(
            dataset_root=dataset_root,
            dataset_manifest_sha256=_sha256_bytes(manifest_bytes),
            dataset_identity=ARCHIVE_DATASET_IDENTITY,
            source_revision=source_revision,
            entries=tuple(entries),
            sample_seed=sample_seed,
            cache_bytes=cache_bytes,
        )

    def metadata_payload(self) -> dict[str, Any]:
        """Return compact reproducibility metadata for a caller-owned run record."""
        payload: dict[str, Any] = {
            "dataset_identity": self.dataset_identity,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "preprocessing_identity": self.preprocessing_identity,
            "preprocessing_sha256": self.preprocessing_sha256,
            "sample_seed": self.sample_seed,
            "sample_seed_algorithm": "sha256-base-seed-episode-view-boundary-u32-v1",
        }
        if self.source_revision is not None:
            payload["dataset_manifest_revision"] = self.source_revision
        return payload

    def sample_refs(self, view: str, role: str = "train") -> Iterator[SampleRef]:
        """Yield metadata-only references for one view and split role."""
        if view not in GENERATION_PROTOCOL.views:
            raise ValueError(f"unsupported phase-1 view: {view}")
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a nonblank string")
        for entry in self._entries:
            if not _include_record(entry.provenance, role):
                continue
            yield from self._entry_refs(entry, view)

    def _entry_refs(self, entry: _EpisodeEntry, view: str) -> Iterator[SampleRef]:
        metadata = entry.payload["record_metadata"]
        provenance = entry.provenance
        kind = provenance.get("episode_kind", "nominal")
        subset = provenance.get("subset") or provenance.get("train_subset")
        if view == "D_geom":
            for boundary in range(entry.observations):
                optional = {
                    name: (entry.episode_id, name, boundary)
                    if self._has_observation_array(entry, name)
                    else None
                    for name in _OPTIONAL_MODALITIES
                }
                fields = {
                    "rgb": optional["rgb"],
                    "depth": optional["depth"],
                    "pointcloud": (entry.episode_id, "pointcloud", boundary),
                    "mask": optional["mask"],
                    "calibration_id": provenance.get("calibration_id"),
                    "episode_kind": kind,
                    "subset": subset,
                }
                yield self._make_ref(entry, view, boundary, fields)
            return

        if view == "D_temporal":
            events = metadata.get("events") or ()
            preferred: set[int] = set()
            for event in events:
                if isinstance(event, Mapping):
                    start = int(event.get("start_t", 0))
                    preferred.update(range(max(0, start - 1), min(entry.transitions, start + 2)))
            if not preferred:
                preferred = set(range(entry.transitions))
            for boundary in sorted(preferred):
                history_start = max(0, boundary - 4 + 1)
                fields = {
                    "observations": [
                        (entry.episode_id, "observation", index)
                        for index in range(history_start, boundary + 1)
                    ],
                    "actions": [
                        (entry.episode_id, "action", index)
                        for index in range(history_start, boundary)
                    ],
                    "dt": (entry.episode_id, "dt", boundary),
                    "around": "contact_or_event",
                    "episode_kind": kind,
                }
                yield self._make_ref(entry, view, boundary, fields)
            return

        if view == "D_dyn":
            intervention = metadata.get("intervention") or {}
            scope = intervention.get("application_scope", provenance.get("application_scope"))
            application_t = intervention.get("application_t", provenance.get("application_t"))
            if application_t is None:
                application_t = intervention.get("intervention_frame", provenance.get("intervention_frame"))
            episode_external = bool(
                provenance.get(
                    "episode_has_external_intervention",
                    intervention.get(
                        "episode_has_external_intervention",
                        intervention.get("external_intervention", provenance.get("external_intervention")),
                    ),
                )
            )
            intervention_id = intervention.get("intervention_id") or provenance.get("intervention_id")
            for transition in range(entry.transitions):
                history = list(range(max(0, transition - 4), transition))
                transition_external = (
                    episode_external
                    and scope in {"timestep", "event"}
                    and application_t is not None
                    and int(application_t) == transition
                )
                fields = {
                    "history": history,
                    "action_t": transition,
                    "observation_t": transition,
                    "observation_t1": transition + 1,
                    "robot_state_t": transition,
                    "object_state_t": transition,
                    "object_state_t1": transition + 1,
                    "dt": (entry.episode_id, "dt", transition),
                    "episode_kind": kind,
                    "episode_has_external_intervention": episode_external,
                    "transition_has_external_intervention": transition_external,
                    "external_intervention": transition_external,
                    "initial_scene_intervention_id": intervention_id if scope == "initial_scene" else None,
                    "intervention_id": intervention_id if transition_external else None,
                }
                yield self._make_ref(entry, view, transition, fields)
            return

        events = metadata.get("events") or ()
        for boundary in range(entry.observations):
            fields = {
                "events": [
                    (entry.episode_id, "event", event.get("step_id"))
                    for event in events
                    if isinstance(event, Mapping)
                ],
                "rho": (entry.episode_id, "rho", boundary) if metadata.get("rho") is not None else None,
                "nu": (entry.episode_id, "nu", boundary) if metadata.get("nu") is not None else None,
                "epsilon": (entry.episode_id, "epsilon", boundary) if metadata.get("epsilon") is not None else None,
            }
            yield self._make_ref(entry, view, boundary, fields)

    def _make_ref(
        self,
        entry: _EpisodeEntry,
        view: str,
        boundary: int,
        fields: Mapping[str, Any],
    ) -> SampleRef:
        return SampleRef(
            episode_id=entry.episode_id,
            view=view,
            t=boundary,
            archive_manifest=entry.archive_manifest,
            archive_manifest_sha256=entry.archive_manifest_sha256,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            archive_preprocessing_identity=str(entry.payload["preprocessing_identity"]),
            preprocessing_identity=self.preprocessing_identity,
            preprocessing_sha256=self.preprocessing_sha256,
            sample_seed=_sample_seed(self.sample_seed, entry.episode_id, view, boundary),
            fields=fields,
        )

    @staticmethod
    def _resolved_array_name(entry: _EpisodeEntry, name: str) -> str | None:
        raw_arrays = entry.payload["raw_arrays"]
        if name not in raw_arrays:
            return None
        value = raw_arrays[name]
        if not isinstance(value, Mapping) or not isinstance(value.get("$archive_array"), str):
            raise ValueError(f"optional modality {name} must reference an archived numeric array")
        return str(value["$archive_array"])

    def _has_observation_array(self, entry: _EpisodeEntry, name: str) -> bool:
        array_name = self._resolved_array_name(entry, name)
        if array_name is None:
            return False
        resolved = array_name
        aliases = {item["name"]: item["target"] for item in entry.payload["array_aliases"]}
        visited: set[str] = set()
        while resolved in aliases:
            if resolved in visited:
                raise ValueError(f"cyclic archive array alias: {array_name}")
            visited.add(resolved)
            resolved = aliases[resolved]
        spec = entry.payload["array_specs"].get(resolved)
        if spec is None or not spec.get("shape") or spec["shape"][0] != entry.observations:
            raise ValueError(f"optional modality {name} is not aligned to observation boundaries")
        return True

    def reader_for(self, sample_ref: SampleRef | str) -> EpisodeArchiveReader:
        """Return a bounded reader for a reference or episode ID."""
        if isinstance(sample_ref, SampleRef):
            episode_id = sample_ref.episode_id
            entry = self._entry_for_ref(sample_ref)
        elif isinstance(sample_ref, str):
            episode_id = sample_ref
            entry = self._entries_by_episode.get(episode_id)
            if entry is None:
                raise KeyError(f"episode is not present in this dataset index: {episode_id}")
        else:
            raise TypeError("reader_for expects a SampleRef or episode ID")
        reader = self._readers.get(episode_id)
        if reader is not None:
            self._readers.move_to_end(episode_id)
        else:
            reader = EpisodeArchiveReader(
                entry.archive_manifest_path,
                cache_bytes=self._reader_cache_bytes,
                resolved_preprocessing_identity=self.preprocessing_identity,
                resolved_preprocessing_sha256=self.preprocessing_sha256,
            )
            self._readers[episode_id] = reader
            while len(self._readers) > self._reader_cache_limit:
                self._readers.popitem(last=False)
        return reader

    def _entry_for_ref(self, sample_ref: SampleRef) -> _EpisodeEntry:
        entry = self._entries_by_episode.get(sample_ref.episode_id)
        if entry is None:
            raise ValueError("sample reference does not belong to this dataset snapshot")
        boundary_count = entry.transitions if sample_ref.view in {"D_temporal", "D_dyn"} else entry.observations
        expected_seed = _sample_seed(self.sample_seed, sample_ref.episode_id, sample_ref.view, sample_ref.t)
        if (
            sample_ref.view not in GENERATION_PROTOCOL.views
            or type(sample_ref.t) is not int
            or not 0 <= sample_ref.t < boundary_count
            or sample_ref.archive_manifest != entry.archive_manifest
            or sample_ref.archive_manifest_sha256 != entry.archive_manifest_sha256
            or sample_ref.dataset_manifest_sha256 != self.dataset_manifest_sha256
            or sample_ref.archive_preprocessing_identity != entry.payload["preprocessing_identity"]
            or sample_ref.preprocessing_identity != self.preprocessing_identity
            or sample_ref.preprocessing_sha256 != self.preprocessing_sha256
            or sample_ref.sample_seed != expected_seed
        ):
            raise ValueError("sample reference does not belong to this dataset snapshot")
        return entry

    def read_optional_modality(self, sample_ref: SampleRef, modality: str) -> np.ndarray | None:
        if modality not in _OPTIONAL_MODALITIES:
            raise ValueError(f"unsupported optional observation modality: {modality}")
        entry = self._entry_for_ref(sample_ref)
        array_name = self._resolved_array_name(entry, modality)
        if array_name is None:
            return None
        if not self._has_observation_array(entry, modality):
            raise ValueError(f"optional modality {modality} is not aligned to observation boundaries")
        return self.reader_for(sample_ref).read_array_row(array_name, sample_ref.t)

    def preprocess_observation(self, sample_ref: SampleRef, *, boundary: int | None = None) -> np.ndarray:
        """Derive the frozen 2,048-point local-frame representation lazily."""
        target_boundary = sample_ref.t if boundary is None else boundary
        if type(target_boundary) is not int:
            raise ValueError("boundary must be an integer")
        reader = self.reader_for(sample_ref)
        observation = reader.observation(target_boundary)
        from icgs.geometry.transforms import transform_pcd

        filtered, _ = _remove_statistical_outliers(observation["points"], nb_neighbors=20, std_ratio=2.0)
        filtered = np.asarray(filtered)
        if len(filtered) == 0:
            raise ValueError("native 2,048-point preprocessing does not support empty point clouds")
        seed = sample_ref.sample_seed if target_boundary == sample_ref.t else _sample_seed(
            self.sample_seed, sample_ref.episode_id, sample_ref.view, target_boundary
        )
        rng = np.random.RandomState(seed)
        selected = filtered[rng.choice(
            len(filtered),
            2048,
            replace=True if len(filtered) < 2048 else False,
        )]
        return transform_pcd(selected, np.linalg.inv(observation["T_w_e"]))


def _include_record(provenance: Mapping[str, Any], role: str) -> bool:
    outcome = str(provenance.get("outcome") or provenance.get("result_class") or "success")
    if outcome in {"simulator_crash", "invalid_observation"}:
        return False
    split = provenance.get("split")
    subset = provenance.get("subset") or provenance.get("train_subset")
    if role == "train":
        return split not in {"dev", "test", "development"} and subset != GENERATION_PROTOCOL.validation
    if role == "validation":
        return split == "train" and subset == GENERATION_PROTOCOL.validation
    if role == "evaluation":
        return split in {"dev", "test", "development"}
    return True


def build_generation_view(
    index: ArchiveDatasetIndex,
    view: str,
    *,
    role: str = "train",
    mix: bool | None = None,
) -> list[SampleRef]:
    """Build archive-backed lazy references and preserve the legacy train mixture."""
    refs = list(index.sample_refs(view, role))
    apply_mix = GENERATION_PROTOCOL.mixture_measured_on == "training_transitions" if mix is None else mix
    if role != "train" or not apply_mix:
        return refs
    from icgs.data.collection.generation.sampler import mix_training_transitions

    mixed = mix_training_transitions(refs, view=view)
    by_key = {(ref.episode_id, ref.t): ref for ref in refs}
    return [by_key[(row["episode_id"], row["t"])] for row in mixed]


__all__ = ["ArchiveDatasetIndex", "SampleRef", "build_generation_view"]
