"""Lossless durable episode archives, strict verification, and quarantine.

Implements ADR0011: JSON manifest + compressed NPZ shards (allow_pickle=False),
atomic directory publication, SHA256 integrity verification, safe path enforcement,
and diagnostic quarantine without dropping task failures or partial evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation
from icgs.data.schemas.episodes import validate_episode


ARCHIVE_VERSION = 1
DEFAULT_SHARD_INTERVALS = 256


class StorageLimitExceeded(ValueError, OSError):
    """Raised when a storage write would exceed the explicit byte allowance."""
    pass


def check_and_write_bytes(
    target_path: str | Path,
    data: bytes,
    *,
    remaining_bytes: int | None = None,
    atomic: bool = True,
) -> int:
    """Verify data length does not exceed remaining allowance (allowing exact equality) then write bytes.

    Returns the number of bytes written.
    Raises StorageLimitExceeded if remaining_bytes is not None and len(data) > remaining_bytes.
    """
    length = len(data)
    if remaining_bytes is not None and length > remaining_bytes:
        raise StorageLimitExceeded(
            f"Write of {length} bytes exceeds staging cap (staging_byte_cap allowance) of {remaining_bytes} bytes"
        )
    dest = Path(target_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp_path = dest.parent / f".{dest.name}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, dest)
    else:
        with open(dest, "wb") as f:
            f.write(data)
    return length


def _validate_safe_id(identifier: Any, name: str = "id") -> str:
    """Ensure identifier is a nonempty safe single path component without dot-components."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"{name} must be a nonempty string")
    clean = identifier.strip()
    if clean != identifier:
        raise ValueError(f"{name} must have no leading or trailing whitespace")
    if clean in (".", "..") or "/" in clean or "\\" in clean or "\x00" in clean:
        raise ValueError(f"{name} must be a safe single path component: {identifier!r}")
    return clean


MAX_QUARANTINE_TRANSITIONS = 128
MAX_QUARANTINE_POINTS = 512


def _json_safe(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types with allow_nan=False."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float value not permitted in JSON: {obj}")
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite float value not permitted in JSON: {val}")
        return val
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, BaseException):
        return {"error_type": type(obj).__name__, "message": str(obj)}
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Unsupported object type for JSON encoding: {type(obj).__name__}")


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _serialize_timed_observation(obs: TimedObservation | None) -> dict[str, Any] | None:
    if obs is None:
        return None
    inner = obs.observation
    pts_summary: dict[str, Any] = {
        "num_points": int(len(inner.points)),
        "shape": list(inner.points.shape),
        "dtype": str(inner.points.dtype),
    }
    if len(inner.points) <= MAX_QUARANTINE_POINTS:
        pts_summary["points"] = inner.points.tolist()
    else:
        pts_summary["points_head"] = inner.points[:16].tolist()
        pts_summary["points_tail"] = inner.points[-16:].tolist()
    return {
        "boundary": int(obs.boundary),
        "simulator_timestamp": float(obs.simulator_timestamp),
        "measured_wall_timestamp": float(obs.measured_wall_timestamp),
        "sensor_profile_id": str(obs.sensor_profile_id),
        "T_w_e": np.asarray(inner.T_w_e, dtype=float).tolist(),
        "grip": float(inner.grip),
        "points": pts_summary,
    }


def _serialize_timed_command(cmd: TimedCommand) -> dict[str, Any]:
    return {
        "target_w": np.asarray(cmd.target_w, dtype=float).tolist(),
        "grip": int(cmd.grip),
        "duration_s": float(cmd.duration_s),
    }


def _serialize_executed_transition(trans: ExecutedTransition | None) -> dict[str, Any] | None:
    if trans is None:
        return None
    return {
        "before": _serialize_timed_observation(trans.before),
        "after": _serialize_timed_observation(trans.after),
        "command": _serialize_timed_command(trans.command),
        "achieved_duration_s": float(trans.achieved_duration_s),
        "physics_substeps": int(trans.physics_substeps),
        "controller_status": str(trans.controller_status),
    }


def build_quarantine_evidence(
    *,
    transitions: Sequence[ExecutedTransition] = (),
    annotations: Sequence[Mapping[str, Any]] = (),
    rejected_transition: ExecutedTransition | None = None,
    initial_observation: TimedObservation | None = None,
    max_transitions: int = MAX_QUARANTINE_TRANSITIONS,
) -> dict[str, Any]:
    """Serialize partial execution evidence for quarantine with explicit size policy."""
    num_trans = len(transitions)
    truncated = num_trans > max_transitions
    selected_trans = transitions[:max_transitions]
    selected_ann = annotations[:max_transitions]

    return {
        "total_transitions_recorded": num_trans,
        "transitions_retained_count": len(selected_trans),
        "is_truncated": truncated,
        "max_transitions_policy": max_transitions,
        "initial_observation": _serialize_timed_observation(initial_observation),
        "rejected_transition": _serialize_executed_transition(rejected_transition),
        "partial_transitions": [_serialize_executed_transition(t) for t in selected_trans],
        "annotations": [_json_safe(ann) for ann in selected_ann],
    }


def quarantine_attempt(
    root: str | Path,
    attempt_id: str,
    *,
    diagnostics: Mapping[str, Any],
    evidence: Any = None,
    staging_byte_cap: int | None = None,
) -> Path:
    """Quarantine an attempt with diagnostic metadata and optional retained evidence."""
    root_path = Path(root).resolve()
    safe_attempt_id = _validate_safe_id(attempt_id, "attempt_id")
    q_dir = root_path / "quarantine" / safe_attempt_id

    report_data = {
        "archive_version": ARCHIVE_VERSION,
        "attempt_id": safe_attempt_id,
        "diagnostics": _json_safe(diagnostics),
        "evidence": _json_safe(evidence) if evidence is not None else None,
    }
    report_bytes = json.dumps(report_data, indent=2, allow_nan=False).encode("utf-8")
    report_path = q_dir / "error_report.json"
    check_and_write_bytes(report_path, report_bytes, remaining_bytes=staging_byte_cap, atomic=True)
    return report_path


def write_attempt_report(
    root: str | Path,
    attempt_id: str,
    report: Mapping[str, Any],
    *,
    staging_byte_cap: int | None = None,
) -> Path:
    """Write an attempt report (e.g. for empty attempts or operational status)."""
    root_path = Path(root).resolve()
    safe_attempt_id = _validate_safe_id(attempt_id, "attempt_id")
    r_dir = root_path / "reports" / safe_attempt_id

    report_bytes = json.dumps(_json_safe(report), indent=2, allow_nan=False).encode("utf-8")
    report_path = r_dir / "report.json"
    check_and_write_bytes(report_path, report_bytes, remaining_bytes=staging_byte_cap, atomic=True)
    return report_path


def read_attempt_report(report_path: str | Path) -> dict[str, Any]:
    """Read an attempt report from JSON."""
    path = Path(report_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Attempt report not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_archive_metadata(manifest_path: str | Path) -> dict[str, Any] | None:
    """Read caller-attached metadata from a published episode archive manifest."""
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("archive_version") != ARCHIVE_VERSION:
        raise ValueError(f"Unsupported archive_version: {manifest.get('archive_version')!r}")
    if manifest.get("schema_version") != "icgs_episode_v1":
        raise ValueError(f"Unsupported schema_version: {manifest.get('schema_version')!r}")
    return manifest.get("metadata")


def _pack_ragged_points(
    points_seq: Sequence[np.ndarray],
    valid_seq: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pack a sequence of N+1 point clouds into concatenated array and int64 offsets."""
    offsets = [0]
    pts_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []

    target_dtype = points_seq[0].dtype
    for idx, pts in enumerate(points_seq):
        if pts.dtype != target_dtype:
            raise TypeError(f"Point cloud dtype mismatch at index {idx}: {pts.dtype} vs {target_dtype}")
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Point cloud at index {idx} must have shape [M, 3], got {pts.shape}")
        if not np.isfinite(pts).all():
            raise ValueError(f"Non-finite coordinates at index {idx}")
        pts_list.append(pts)
        offsets.append(offsets[-1] + len(pts))
        if valid_seq is not None:
            val = valid_seq[idx]
            if val.dtype != bool or val.shape != (len(pts),):
                raise ValueError(f"Point valid mask at index {idx} must be boolean matching length of points")
            val_list.append(val)

    if pts_list:
        concatenated_pts = np.concatenate(pts_list, axis=0)
    else:
        concatenated_pts = np.empty((0, 3), dtype=target_dtype)

    concatenated_val = np.concatenate(val_list, axis=0) if valid_seq is not None else None
    return concatenated_pts, np.array(offsets, dtype=np.int64), concatenated_val


def write_episode_archive(
    root: str | Path,
    record: Mapping[str, Any],
    *,
    config: MethodConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    staging_byte_cap: int | None = None,
    auxiliary: Mapping[str, Any] | None = None,
) -> Path:
    """Losslessly write an episode record to disk with atomic publication and SHA256 hashes.

    Parameters
    ----------
    root : str | Path
        Root dataset directory.
    record : Mapping[str, Any]
        In-memory episode record conforming to ``icgs_episode_v1``.
    config : MethodConfig | None
        Method configuration, used to resolve ``shard_intervals`` and disk cap.
    metadata : Mapping[str, Any] | None
        Optional caller-supplied metadata attached to the archive envelope.
    staging_byte_cap : int | None
        Optional byte cap for temporary staging and disk usage.

    Returns
    -------
    Path
        Path to the published ``manifest.json``.
    """
    validate_episode(record)

    episode_id = _validate_safe_id(record["provenance"]["episode_id"], "episode_id")
    root_path = Path(root).resolve()
    episodes_dir = root_path / "episodes"
    target_dir = episodes_dir / episode_id

    if target_dir.exists():
        raise FileExistsError(f"Episode directory already exists: {target_dir}")

    shard_intervals = DEFAULT_SHARD_INTERVALS
    if config is not None:
        shard_intervals = config.dataset.shard_intervals
    if isinstance(shard_intervals, bool) or not isinstance(shard_intervals, int) or shard_intervals < 1:
        raise ValueError(f"shard_intervals must be a positive integer, got {shard_intervals!r}")

    effective_staging_cap = staging_byte_cap
    if effective_staging_cap is None and config is not None:
        collection_cfg = getattr(config, "collection", None)
        if collection_cfg is not None:
            effective_staging_cap = getattr(collection_cfg, "disk_limit_bytes", None)

    transitions = tuple(record["transitions"])
    online_obs = tuple(record["online_observations"])
    num_transitions = len(transitions)

    # Prepare temporary staging directory under root
    episodes_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".tmp_ep_", dir=episodes_dir))

    staged_bytes = 0
    try:
        shards_manifest: list[dict[str, Any]] = []
        num_shards = max(1, math.ceil(num_transitions / shard_intervals))

        for shard_idx in range(num_shards):
            start_t = shard_idx * shard_intervals
            end_t = min(num_transitions, (shard_idx + 1) * shard_intervals)
            shard_transitions = transitions[start_t:end_t]
            shard_online_obs = online_obs[start_t : end_t + 1]
            k = len(shard_transitions)

            # Online observations: preserve exact dtypes and masks
            online_pts = [o["points"] for o in shard_online_obs]
            online_val = [o["point_valid"] for o in shard_online_obs]
            cat_online_pts, online_offsets, cat_online_val = _pack_ragged_points(online_pts, online_val)
            online_t_w_e = np.stack([o["T_w_e"] for o in shard_online_obs], axis=0)
            online_grip_raw = [o["grip"] for o in shard_online_obs]
            online_grip = np.asarray(online_grip_raw)

            # Measured boundary observations from transitions
            meas_obs_list: list[Observation] = []
            meas_bounds: list[int] = []
            meas_sim_times: list[float] = []
            meas_wall_times: list[float] = []
            meas_sensor_ids: list[str] = []

            for i, trans in enumerate(shard_transitions):
                if i == 0:
                    meas_obs_list.append(trans.before.observation)
                    meas_bounds.append(trans.before.boundary)
                    meas_sim_times.append(trans.before.simulator_timestamp)
                    meas_wall_times.append(trans.before.measured_wall_timestamp)
                    meas_sensor_ids.append(trans.before.sensor_profile_id)
                meas_obs_list.append(trans.after.observation)
                meas_bounds.append(trans.after.boundary)
                meas_sim_times.append(trans.after.simulator_timestamp)
                meas_wall_times.append(trans.after.measured_wall_timestamp)
                meas_sensor_ids.append(trans.after.sensor_profile_id)

            cat_meas_pts, meas_offsets, _ = _pack_ragged_points([m.points for m in meas_obs_list])
            meas_t_w_e = np.stack([m.T_w_e for m in meas_obs_list], axis=0)
            meas_grip = np.array([m.grip for m in meas_obs_list], dtype=np.float64)

            # Transition commands and achieved quantities
            cmd_target_w = np.stack([t.command.target_w for t in shard_transitions], axis=0) if k > 0 else np.empty((0, 4, 4), dtype=np.float64)
            cmd_grip = np.array([t.command.grip for t in shard_transitions], dtype=np.int32)
            cmd_duration_s = np.array([t.command.duration_s for t in shard_transitions], dtype=np.float64)
            achieved_duration_s = np.array([t.achieved_duration_s for t in shard_transitions], dtype=np.float64)
            physics_substeps = np.array([t.physics_substeps for t in shard_transitions], dtype=np.int64)
            controller_statuses = [t.controller_status for t in shard_transitions]

            shard_filename = f"shard_{shard_idx:04d}.npz"
            shard_path = staging_dir / shard_filename

            shard_arrays = {
                "online_points": cat_online_pts,
                "online_point_offsets": online_offsets,
                "online_point_valid": cat_online_val,
                "online_T_w_e": online_t_w_e,
                "online_grip": online_grip,
                "meas_points": cat_meas_pts,
                "meas_point_offsets": meas_offsets,
                "meas_T_w_e": meas_t_w_e,
                "meas_grip": meas_grip,
                "meas_boundary": np.array(meas_bounds, dtype=np.int64),
                "meas_sim_time": np.array(meas_sim_times, dtype=np.float64),
                "meas_wall_time": np.array(meas_wall_times, dtype=np.float64),
                "cmd_target_w": cmd_target_w,
                "cmd_grip": cmd_grip,
                "cmd_duration_s": cmd_duration_s,
                "achieved_duration_s": achieved_duration_s,
                "physics_substeps": physics_substeps,
            }

            # Encode shard into single in-memory buffer to verify exact length vs remaining allowance BEFORE write
            shard_buf = io.BytesIO()
            np.savez_compressed(shard_buf, **shard_arrays)
            shard_bytes = shard_buf.getvalue()
            shard_buf.close()

            remaining = effective_staging_cap - staged_bytes if effective_staging_cap is not None else None
            check_and_write_bytes(shard_path, shard_bytes, remaining_bytes=remaining, atomic=False)
            staged_bytes += len(shard_bytes)

            array_schema = {
                k_arr: {"dtype": str(v_arr.dtype), "shape": list(v_arr.shape)}
                for k_arr, v_arr in shard_arrays.items()
            }

            sha256 = hashlib.sha256(shard_bytes).hexdigest()
            shards_manifest.append({
                "index": shard_idx,
                "filename": shard_filename,
                "sha256": sha256,
                "start_boundary": start_t,
                "end_boundary": end_t,
                "interval_count": k,
                "controller_statuses": controller_statuses,
                "sensor_profile_ids": meas_sensor_ids,
                "array_schema": array_schema,
            })

        meta_dict: dict[str, Any] = dict(metadata) if metadata is not None else {}
        if auxiliary is not None:
            # 1. Telemetry: joint positions & joint velocities
            if auxiliary.get("joint_positions") is not None or auxiliary.get("joint_velocities") is not None:
                telem_arrays: dict[str, np.ndarray] = {}
                if auxiliary.get("joint_positions") is not None:
                    telem_arrays["joint_positions"] = np.asarray(auxiliary["joint_positions"], dtype=np.float64)
                if auxiliary.get("joint_velocities") is not None:
                    telem_arrays["joint_velocities"] = np.asarray(auxiliary["joint_velocities"], dtype=np.float64)
                telem_buf = io.BytesIO()
                np.savez_compressed(telem_buf, **telem_arrays)
                telem_bytes = telem_buf.getvalue()
                telem_buf.close()
                telem_path = staging_dir / "telemetry.npz"
                remaining = effective_staging_cap - staged_bytes if effective_staging_cap is not None else None
                check_and_write_bytes(telem_path, telem_bytes, remaining_bytes=remaining, atomic=False)
                staged_bytes += len(telem_bytes)
                meta_dict["telemetry"] = "telemetry.npz"

            # 2. RGB Video: front camera frames
            if auxiliary.get("front_rgb_frames") is not None and len(auxiliary["front_rgb_frames"]) > 0:
                video_path = staging_dir / "video_front.mp4"
                from icgs.data.collection.video import encode_frames_to_mp4
                if encode_frames_to_mp4(auxiliary["front_rgb_frames"], video_path, fps=10):
                    staged_bytes += video_path.stat().st_size
                    meta_dict["video_front"] = "video_front.mp4"

        manifest_data = {
            "archive_version": ARCHIVE_VERSION,
            "schema_version": record["schema_version"],
            "provenance": _json_safe(record["provenance"]),
            "shards": shards_manifest,
            "total_intervals": num_transitions,
            "metadata": _json_safe(meta_dict) if meta_dict else None,
        }

        manifest_bytes = json.dumps(manifest_data, indent=2, allow_nan=False).encode("utf-8")
        remaining = effective_staging_cap - staged_bytes if effective_staging_cap is not None else None
        manifest_path = staging_dir / "manifest.json"
        check_and_write_bytes(manifest_path, manifest_bytes, remaining_bytes=remaining, atomic=False)
        staged_bytes += len(manifest_bytes)

        # Pre-publication verification: reconstruct and validate
        restored = read_episode_archive(manifest_path)
        validate_episode(restored)

        # Atomic publish
        if target_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {target_dir}")
        os.replace(staging_dir, target_dir)
        return target_dir / "manifest.json"

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def read_episode_archive(manifest_path: str | Path) -> dict[str, Any]:
    """Read a published episode archive, verify shard hashes, and restore in-memory record."""
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("archive_version") != ARCHIVE_VERSION:
        raise ValueError(f"Unsupported archive_version: {manifest.get('archive_version')!r}")
    if manifest.get("schema_version") != "icgs_episode_v1":
        raise ValueError(f"Unsupported schema_version: {manifest.get('schema_version')!r}")

    episode_dir = manifest_file.parent
    shards_meta = manifest.get("shards")
    if not isinstance(shards_meta, Sequence) or len(shards_meta) == 0:
        raise ValueError("Manifest must contain a nonempty sequence of shards")

    online_obs_list: list[dict[str, Any]] = []
    transitions_list: list[ExecutedTransition] = []
    expected_next_boundary = 0

    required_keys = {
        "online_points", "online_point_offsets", "online_point_valid", "online_T_w_e", "online_grip",
        "meas_points", "meas_point_offsets", "meas_T_w_e", "meas_grip", "meas_boundary",
        "meas_sim_time", "meas_wall_time", "cmd_target_w", "cmd_grip", "cmd_duration_s",
        "achieved_duration_s", "physics_substeps",
    }

    for s_idx, s_info in enumerate(shards_meta):
        if not isinstance(s_info, Mapping):
            raise ValueError(f"Shard metadata {s_idx} must be a mapping")

        filename = s_info.get("filename")
        _validate_safe_id(filename, f"shard filename {s_idx}")
        shard_path = episode_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
        if shard_path.is_symlink():
            raise ValueError(f"Shard must not be a symlink: {shard_path}")

        # Check SHA256 digest
        computed_sha = _compute_sha256(shard_path)
        expected_sha = s_info.get("sha256")
        if computed_sha != expected_sha:
            raise ValueError(
                f"Shard checksum mismatch for {filename}: computed {computed_sha}, expected {expected_sha}"
            )

        start_b = s_info["start_boundary"]
        end_b = s_info["end_boundary"]
        k = s_info["interval_count"]
        if end_b - start_b != k:
            raise ValueError(f"Shard {s_idx} boundary mismatch: end ({end_b}) - start ({start_b}) != interval_count ({k})")
        if start_b != expected_next_boundary:
            raise ValueError(f"Shard {s_idx} start boundary {start_b} != expected {expected_next_boundary}")
        expected_next_boundary = end_b

        controller_statuses = s_info["controller_statuses"]
        sensor_profile_ids = s_info["sensor_profile_ids"]
        if len(controller_statuses) != k:
            raise ValueError(f"Shard {s_idx} controller_statuses count {len(controller_statuses)} != {k}")
        if len(sensor_profile_ids) != k + 1:
            raise ValueError(f"Shard {s_idx} sensor_profile_ids count {len(sensor_profile_ids)} != {k + 1}")

        # Load NPZ with allow_pickle=False and strict exact key inventory
        with np.load(shard_path, allow_pickle=False) as npz:
            npz_files = set(npz.files)
            if npz_files != required_keys:
                missing = required_keys - npz_files
                extras = npz_files - required_keys
                raise ValueError(
                    f"Shard {filename} array keys mismatch: missing={missing}, unexpected_extras={extras}"
                )

            online_pts = npz["online_points"]
            online_offsets = npz["online_point_offsets"]
            online_val = npz["online_point_valid"]
            online_t_w_e = npz["online_T_w_e"]
            online_grip = npz["online_grip"]

            meas_pts = npz["meas_points"]
            meas_offsets = npz["meas_point_offsets"]
            meas_t_w_e = npz["meas_T_w_e"]
            meas_grip = npz["meas_grip"]
            meas_bounds = npz["meas_boundary"]
            meas_sim = npz["meas_sim_time"]
            meas_wall = npz["meas_wall_time"]

            cmd_target_w = npz["cmd_target_w"]
            cmd_grip = npz["cmd_grip"]
            cmd_dur = npz["cmd_duration_s"]
            ach_dur = npz["achieved_duration_s"]
            substeps = npz["physics_substeps"]

            # Verify stored array_schema in shard manifest (Issue 9)
            if "array_schema" not in s_info or not isinstance(s_info["array_schema"], dict):
                raise ValueError(f"Shard manifest for {filename} is missing required 'array_schema'")

            stored_schema = s_info["array_schema"]
            required_shard_keys = (
                "online_points",
                "online_point_valid",
                "online_point_offsets",
                "online_T_w_e",
                "online_grip",
                "meas_points",
                "meas_point_offsets",
                "meas_T_w_e",
                "meas_grip",
                "meas_boundary",
                "meas_sim_time",
                "meas_wall_time",
                "cmd_target_w",
                "cmd_grip",
                "cmd_duration_s",
                "achieved_duration_s",
                "physics_substeps",
            )
            missing_keys = [k for k in required_shard_keys if k not in stored_schema]
            if missing_keys:
                raise ValueError(f"Shard array_schema incomplete in {filename}; missing keys: {missing_keys}")

            for arr_k, arr_spec in stored_schema.items():
                if arr_k not in npz_files:
                    raise ValueError(f"Array {arr_k} from schema missing in shard {filename}")
                act_arr = npz[arr_k]
                if str(act_arr.dtype) != arr_spec["dtype"]:
                    raise TypeError(
                        f"Array {arr_k} in {filename} has dtype {act_arr.dtype}, expected {arr_spec['dtype']} from schema"
                    )
                if list(act_arr.shape) != arr_spec["shape"]:
                    raise ValueError(
                        f"Array {arr_k} in {filename} has shape {act_arr.shape}, expected {arr_spec['shape']} from schema"
                    )

            # Per-array shape, exact dtype, finiteness, and length validations (MED1)
            if online_pts.ndim != 2 or online_pts.shape[1] != 3:
                raise ValueError(f"online_points in {filename} must have shape [N, 3], got {online_pts.shape}")
            if online_pts.dtype not in (np.float32, np.float64):
                raise TypeError(f"online_points in {filename} must have float32 or float64 dtype, got {online_pts.dtype}")
            if not np.all(np.isfinite(online_pts)):
                raise ValueError(f"online_points in {filename} contains non-finite coordinates")

            if online_val.dtype != bool:
                raise TypeError(f"online_point_valid in {filename} must have bool dtype, got {online_val.dtype}")
            if online_val.shape != (len(online_pts),):
                raise ValueError(f"online_point_valid in {filename} must have shape ({len(online_pts)},)")

            if online_offsets.dtype != np.int64:
                raise TypeError(f"online_point_offsets in {filename} must be int64 dtype, got {online_offsets.dtype}")
            if len(online_offsets) != k + 2 or int(online_offsets[0]) != 0 or int(online_offsets[-1]) != len(online_pts):
                raise ValueError(f"Invalid online_point_offsets in shard {filename}")
            if not np.all(np.diff(online_offsets) >= 0):
                raise ValueError(f"online_point_offsets must be monotonic in shard {filename}")

            if online_t_w_e.shape != (k + 1, 4, 4):
                raise ValueError(f"online_T_w_e in {filename} must have shape ({k + 1}, 4, 4), got {online_t_w_e.shape}")
            if online_t_w_e.dtype not in (np.float32, np.float64):
                raise TypeError(f"online_T_w_e in {filename} must have float32 or float64 dtype, got {online_t_w_e.dtype}")
            if not np.all(np.isfinite(online_t_w_e)):
                raise ValueError(f"online_T_w_e in {filename} contains non-finite values")

            if online_grip.shape != (k + 1,):
                raise ValueError(f"online_grip in {filename} must have shape ({k + 1},), got {online_grip.shape}")
            if not np.all(np.isfinite(online_grip)):
                raise ValueError(f"online_grip in {filename} contains non-finite values")

            if meas_pts.ndim != 2 or meas_pts.shape[1] != 3:
                raise ValueError(f"meas_points in {filename} must have shape [M, 3], got {meas_pts.shape}")
            if meas_pts.dtype != np.float64:
                raise TypeError(f"meas_points in {filename} must have float64 dtype, got {meas_pts.dtype}")
            if not np.all(np.isfinite(meas_pts)):
                raise ValueError(f"meas_points in {filename} contains non-finite coordinates")

            if meas_offsets.dtype != np.int64:
                raise TypeError(f"meas_point_offsets in {filename} must be int64 dtype, got {meas_offsets.dtype}")
            if len(meas_offsets) != k + 2 or int(meas_offsets[0]) != 0 or int(meas_offsets[-1]) != len(meas_pts):
                raise ValueError(f"Invalid meas_point_offsets in shard {filename}")
            if not np.all(np.diff(meas_offsets) >= 0):
                raise ValueError(f"meas_point_offsets must be monotonic in shard {filename}")

            if meas_t_w_e.shape != (k + 1, 4, 4):
                raise ValueError(f"meas_T_w_e in {filename} must have shape ({k + 1}, 4, 4), got {meas_t_w_e.shape}")
            if meas_t_w_e.dtype != np.float64:
                raise TypeError(f"meas_T_w_e in {filename} must have float64 dtype, got {meas_t_w_e.dtype}")
            if not np.all(np.isfinite(meas_t_w_e)):
                raise ValueError(f"meas_T_w_e in {filename} contains non-finite values")

            if meas_grip.shape != (k + 1,):
                raise ValueError(f"meas_grip in {filename} must have shape ({k + 1},), got {meas_grip.shape}")
            if meas_grip.dtype != np.float64:
                raise TypeError(f"meas_grip in {filename} must have float64 dtype, got {meas_grip.dtype}")
            if not np.all(np.isfinite(meas_grip)):
                raise ValueError(f"meas_grip in {filename} contains non-finite values")

            if meas_bounds.shape != (k + 1,):
                raise ValueError(f"meas_boundary in {filename} must have shape ({k + 1},)")
            if meas_bounds.dtype != np.int64:
                raise TypeError(f"meas_boundary in {filename} must be int64 dtype, got {meas_bounds.dtype}")

            if meas_sim.shape != (k + 1,) or not np.all(np.isfinite(meas_sim)):
                raise ValueError(f"meas_sim_time in {filename} must be finite shape ({k + 1},)")
            if meas_sim.dtype != np.float64:
                raise TypeError(f"meas_sim_time in {filename} must have float64 dtype, got {meas_sim.dtype}")

            if meas_wall.shape != (k + 1,) or not np.all(np.isfinite(meas_wall)):
                raise ValueError(f"meas_wall_time in {filename} must be finite shape ({k + 1},)")
            if meas_wall.dtype != np.float64:
                raise TypeError(f"meas_wall_time in {filename} must have float64 dtype, got {meas_wall.dtype}")

            if cmd_target_w.shape != (k, 4, 4) or not np.all(np.isfinite(cmd_target_w)):
                raise ValueError(f"cmd_target_w in {filename} must be finite shape ({k}, 4, 4)")
            if cmd_target_w.dtype != np.float64:
                raise TypeError(f"cmd_target_w in {filename} must have float64 dtype, got {cmd_target_w.dtype}")

            if cmd_grip.shape != (k,) or not np.all(np.isfinite(cmd_grip)):
                raise ValueError(f"cmd_grip in {filename} must be finite shape ({k},)")
            if cmd_grip.dtype != np.int32:
                raise TypeError(f"cmd_grip in {filename} must have int32 dtype, got {cmd_grip.dtype}")

            if cmd_dur.shape != (k,) or not np.all(np.isfinite(cmd_dur)) or np.any(cmd_dur <= 0):
                raise ValueError(f"cmd_duration_s in {filename} must be positive finite shape ({k},)")
            if cmd_dur.dtype != np.float64:
                raise TypeError(f"cmd_duration_s in {filename} must have float64 dtype, got {cmd_dur.dtype}")

            if ach_dur.shape != (k,) or not np.all(np.isfinite(ach_dur)) or np.any(ach_dur < 0):
                raise ValueError(f"achieved_duration_s in {filename} must be non-negative finite shape ({k},)")
            if ach_dur.dtype != np.float64:
                raise TypeError(f"achieved_duration_s in {filename} must have float64 dtype, got {ach_dur.dtype}")

            if substeps.shape != (k,) or np.any(substeps < 1):
                raise ValueError(f"physics_substeps in {filename} must be integers >= 1")
            if substeps.dtype != np.int64:
                raise TypeError(f"physics_substeps in {filename} must have int64 dtype, got {substeps.dtype}")

            # Reconstruct observations in this shard
            shard_timed_obs: list[TimedObservation] = []
            for b_idx in range(k + 1):
                # Measured observation
                m_start = int(meas_offsets[b_idx])
                m_end = int(meas_offsets[b_idx + 1])
                m_p = meas_pts[m_start:m_end]
                m_obs = Observation(m_p, meas_t_w_e[b_idx], float(meas_grip[b_idx]))
                t_obs = TimedObservation(
                    m_obs,
                    int(meas_bounds[b_idx]),
                    float(meas_sim[b_idx]),
                    float(meas_wall[b_idx]),
                    sensor_profile_ids[b_idx],
                )
                shard_timed_obs.append(t_obs)

                # Validate duplicated measured shared boundaries across shards exactly
                if s_idx > 0 and b_idx == 0:
                    prev_meas = transitions_list[-1].after
                    if prev_meas.boundary != t_obs.boundary:
                        raise ValueError(
                            f"Shared measured boundary mismatch: prev={prev_meas.boundary}, current={t_obs.boundary}"
                        )
                    if prev_meas.simulator_timestamp != t_obs.simulator_timestamp or prev_meas.measured_wall_timestamp != t_obs.measured_wall_timestamp:
                        raise ValueError(f"Shared boundary timestamp mismatch at boundary {t_obs.boundary}")
                    if prev_meas.sensor_profile_id != t_obs.sensor_profile_id:
                        raise ValueError(f"Shared boundary sensor_profile_id mismatch at boundary {t_obs.boundary}")
                    if prev_meas.observation.grip != t_obs.observation.grip:
                        raise ValueError(f"Shared boundary grip mismatch at boundary {t_obs.boundary}")
                    if not np.array_equal(prev_meas.observation.T_w_e, t_obs.observation.T_w_e):
                        raise ValueError(f"Shared boundary pose T_w_e mismatch at boundary {t_obs.boundary}")
                    if prev_meas.observation.points.dtype != t_obs.observation.points.dtype:
                        raise TypeError(f"Shared boundary points dtype mismatch at boundary {t_obs.boundary}")
                    if not np.array_equal(prev_meas.observation.points, t_obs.observation.points):
                        raise ValueError(f"Shared boundary points coordinate mismatch at boundary {t_obs.boundary}")

                # Online observation: validate exact agreement if shared boundary
                if s_idx > 0 and b_idx == 0:
                    prev_online = online_obs_list[-1]
                    o_start = int(online_offsets[b_idx])
                    o_end = int(online_offsets[b_idx + 1])
                    if (
                        prev_online["points"].dtype != online_pts[o_start:o_end].dtype
                        or not np.array_equal(prev_online["points"], online_pts[o_start:o_end])
                        or not np.array_equal(prev_online["point_valid"], online_val[o_start:o_end])
                        or not np.array_equal(prev_online["T_w_e"], online_t_w_e[b_idx])
                        or prev_online["grip"] != online_grip[b_idx]
                    ):
                        raise ValueError(f"Shared boundary {start_b} online observation disagreement between shards")
                else:
                    o_start = int(online_offsets[b_idx])
                    o_end = int(online_offsets[b_idx + 1])
                    online_obs_list.append({
                        "points": online_pts[o_start:o_end],
                        "point_valid": online_val[o_start:o_end],
                        "T_w_e": online_t_w_e[b_idx],
                        "grip": online_grip[b_idx].item() if isinstance(online_grip[b_idx], np.generic) else online_grip[b_idx],
                    })

            # Reconstruct transitions
            for t_idx in range(k):
                before_obs = shard_timed_obs[t_idx]
                after_obs = shard_timed_obs[t_idx + 1]
                cmd = TimedCommand(cmd_target_w[t_idx], int(cmd_grip[t_idx]), float(cmd_dur[t_idx]))
                trans = ExecutedTransition(
                    before_obs,
                    after_obs,
                    cmd,
                    float(ach_dur[t_idx]),
                    int(substeps[t_idx]),
                    controller_statuses[t_idx],
                )
                transitions_list.append(trans)

    record = {
        "schema_version": manifest["schema_version"],
        "provenance": manifest["provenance"],
        "online_observations": online_obs_list,
        "transitions": transitions_list,
    }
    validate_episode(record)
    return record


__all__ = [
    "ARCHIVE_VERSION",
    "MAX_QUARANTINE_POINTS",
    "MAX_QUARANTINE_TRANSITIONS",
    "StorageLimitExceeded",
    "build_quarantine_evidence",
    "check_and_write_bytes",
    "quarantine_attempt",
    "read_archive_metadata",
    "read_attempt_report",
    "read_episode_archive",
    "write_attempt_report",
    "write_episode_archive",
]
