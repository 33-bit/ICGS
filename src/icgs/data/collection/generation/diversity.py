"""Independent episode seeds, stratified randomization, and duplicate rejection."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import hashlib
import math
from typing import Any

from icgs.data.collection.generation.lighting import get_lighting_profile, lighting_profile_ids
from icgs.data.collection.generation.protocol import GENERATION_PROTOCOL


def episode_seed(
    program_id: str,
    collection_seed: int,
    episode_index: int,
    *,
    dataset_version: str | None = None,
) -> int:
    version = dataset_version or GENERATION_PROTOCOL.dataset_version
    payload = f"{version}|{program_id}|{collection_seed}|{episode_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _unit_interval(seed: int, channel: int) -> float:
    raw = hashlib.sha256(f"{seed}:{channel}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64)


def _lerp(bounds: tuple[float, float], u: float) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return lo + (hi - lo) * u


def sample_randomization(
    program_id: str,
    collection_seed: int,
    episode_index: int,
    bounds: Mapping[str, Any],
    *,
    scene_bucket: str = "all",
) -> dict[str, Any]:
    seed = episode_seed(program_id, collection_seed, episode_index)
    n_pos = GENERATION_PROTOCOL.position_strata
    n_yaw = GENERATION_PROTOCOL.yaw_strata
    n_scale = GENERATION_PROTOCOL.scale_strata
    reserved = min(GENERATION_PROTOCOL.eval_held_out_position_bin, n_pos - 1)
    stratum_id = episode_index % (n_pos * n_yaw * n_scale)
    if scene_bucket == "eval":
        pos_bin = reserved
    elif scene_bucket == "train":
        pos_bin = stratum_id % max(1, n_pos - 1)
        if pos_bin >= reserved:
            pos_bin += 1
    else:
        pos_bin = stratum_id % n_pos
    yaw_bin = (stratum_id // n_pos) % n_yaw
    scale_bin = (stratum_id // (n_pos * n_yaw)) % n_scale

    translation = bounds["translation_m"]
    yaw_bounds = tuple(bounds["yaw_deg"])
    scale_bounds = tuple(bounds["scale"])

    def bin_sample(pair: tuple[float, float], bins: int, index: int, channel: int) -> float:
        lo, hi = float(pair[0]), float(pair[1])
        if hi <= lo:
            return lo
        width = (hi - lo) / bins
        inner = _unit_interval(seed, channel)
        return lo + index * width + inner * width

    x = bin_sample(tuple(translation["x"]), n_pos, pos_bin, 1)
    y = bin_sample(tuple(translation["y"]), n_pos, pos_bin, 2)
    z = _lerp(tuple(translation["z"]), _unit_interval(seed, 3))
    yaw = bin_sample(yaw_bounds, n_yaw, yaw_bin, 4)
    scale = bin_sample(scale_bounds, n_scale, scale_bin, 5)
    gap = GENERATION_PROTOCOL.inter_object_gap_m
    wp = GENERATION_PROTOCOL.waypoint_delta_m
    lighting = _sample_lighting(seed)
    approach = {
        "dx_m": (_unit_interval(seed, 10) - 0.5) * wp,
        "dy_m": (_unit_interval(seed, 11) - 0.5) * wp,
        "dz_m": _unit_interval(seed, 12) * wp,
    }
    release = {
        "dx_m": (_unit_interval(seed, 13) - 0.5) * wp,
        "dy_m": (_unit_interval(seed, 14) - 0.5) * wp,
        "dz_m": _unit_interval(seed, 15) * wp,
    }
    camera_viewpoint = {
        "d_yaw_deg": (_unit_interval(seed, 16) - 0.5) * 10.0,
        "d_pitch_deg": (_unit_interval(seed, 17) - 0.5) * 6.0,
    }
    sample = {
        "scene_seed": seed,
        "asset_instance_id": f"{program_id}-asset-{seed}",
        "object_translation_m": {"x": x, "y": y, "z": z},
        "object_pose_applied": {"x": x, "y": y, "z": z},
        "object_yaw_deg": yaw,
        "object_scale_applied": scale,
        "scale": scale,
        "inter_object_gap_m": {
            "dx_m": (_unit_interval(seed, 8) - 0.5) * 2.0 * gap,
            "dy_m": (_unit_interval(seed, 9) - 0.5) * 2.0 * gap,
        },
        "approach_waypoint": approach,
        "release_waypoint": release,
        "waypoint_noise_applied": {"dx_m": approach["dx_m"], "dy_m": approach["dy_m"]},
        "camera_profile_id": bounds.get("camera_profile_id", GENERATION_PROTOCOL.camera_profile_id),
        "camera_viewpoint_applied": camera_viewpoint,
        "lighting_profile_id": lighting["lighting_profile_id"],
        "lighting_applied": lighting,
        "stratum": {"position": pos_bin, "yaw": yaw_bin, "scale": scale_bin, "id": stratum_id},
        "scene_bucket": scene_bucket,
    }
    sample["train_subset"] = train_subset_for_sample(sample)
    return sample


def _sample_lighting(seed: int) -> dict[str, Any]:
    ids = lighting_profile_ids()
    index = int(_unit_interval(seed, 18) * len(ids)) % len(ids)
    profile = get_lighting_profile(ids[index])
    profile["azimuth_deg"] = float(profile["azimuth_deg"]) + (_unit_interval(seed, 19) - 0.5) * 8.0
    return profile


def train_subset_for_sample(sample: Mapping[str, Any], *, core_fraction: float | None = None) -> str:
    fraction = GENERATION_PROTOCOL.train_core_fraction if core_fraction is None else core_fraction
    signature = scene_signature(sample)
    bucket = int(signature[:2], 16) / 255.0
    return "train_core" if bucket < fraction else "train_val"


def _quantize(value: float, step: float) -> float:
    return round(value / step) * step


def scene_signature(sample: Mapping[str, Any]) -> str:
    translation = sample.get("object_translation_m") or sample.get("object_pose_applied") or {}
    gap = sample.get("inter_object_gap_m") or {}
    approach = sample.get("approach_waypoint") or sample.get("waypoint_noise_applied") or {}
    viewpoint = sample.get("camera_viewpoint_applied") or {}
    payload = {
        "program": sample.get("program_id"),
        "split": sample.get("split"),
        "scene_bucket": sample.get("scene_bucket"),
        "asset": sample.get("asset_family_id") or sample.get("asset_instance_id"),
        "asset_instance": sample.get("asset_instance_id"),
        "camera": sample.get("camera_profile_id"),
        "lighting": sample.get("lighting_profile_id"),
        "tx": _quantize(float(translation.get("x", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "ty": _quantize(float(translation.get("y", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "tz": _quantize(float(translation.get("z", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "yaw": _quantize(float(sample.get("object_yaw_deg", 0.0)), 1.0),
        "scale": _quantize(float(sample.get("scale") or sample.get("object_scale_applied") or 1.0), 0.02),
        "gap_x": _quantize(float(gap.get("dx_m", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "gap_y": _quantize(float(gap.get("dy_m", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "approach_x": _quantize(float(approach.get("dx_m", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "approach_y": _quantize(float(approach.get("dy_m", 0.0)), GENERATION_PROTOCOL.pose_duplicate_threshold_m),
        "cam_yaw": _quantize(float(viewpoint.get("d_yaw_deg", 0.0)), 1.0),
    }
    raw = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()
    return raw[:16]


def pose_distance_m(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ta = a.get("object_translation_m") or a.get("object_pose_applied") or {}
    tb = b.get("object_translation_m") or b.get("object_pose_applied") or {}
    dx = float(ta.get("x", 0.0)) - float(tb.get("x", 0.0))
    dy = float(ta.get("y", 0.0)) - float(tb.get("y", 0.0))
    dz = float(ta.get("z", 0.0)) - float(tb.get("z", 0.0))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_duplicate_episode(
    sample: Mapping[str, Any],
    seen: MutableMapping[str, set[int]] | Mapping[str, set[int]],
    *,
    previous: list[Mapping[str, Any]] | None = None,
) -> bool:
    signature = scene_signature(sample)
    known = seen.get(signature, set())
    if known:
        return True
    if previous:
        threshold = GENERATION_PROTOCOL.pose_duplicate_threshold_m
        same_asset = [
            item for item in previous
            if item.get("asset_family_id") == sample.get("asset_family_id")
            and item.get("camera_profile_id") == sample.get("camera_profile_id")
        ]
        if any(pose_distance_m(sample, item) < threshold for item in same_asset):
            return True
    return False


def nearest_signature_distance(sample: Mapping[str, Any], previous: list[Mapping[str, Any]]) -> float | None:
    if not previous:
        return None
    return min(pose_distance_m(sample, item) for item in previous)


def register_episode(sample: Mapping[str, Any], seen: MutableMapping[str, set[int]]) -> str:
    signature = scene_signature(sample)
    seen.setdefault(signature, set()).add(int(sample["scene_seed"]))
    return signature
