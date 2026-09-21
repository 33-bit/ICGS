"""Lighting profiles whose applied values are stored on each attempt."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from icgs.data.collection.v3.protocol import V3_PROTOCOL


LIGHTING_PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "rlbench-base-lighting-v1": {
        "lighting_profile_id": "rlbench-base-lighting-v1",
        "ambient": 0.60,
        "directional": 0.80,
        "azimuth_deg": 45.0,
        "elevation_deg": 60.0,
    },
    "rlbench-dim-lighting-v1": {
        "lighting_profile_id": "rlbench-dim-lighting-v1",
        "ambient": 0.35,
        "directional": 0.50,
        "azimuth_deg": 30.0,
        "elevation_deg": 50.0,
    },
    "rlbench-high-key-lighting-v1": {
        "lighting_profile_id": "rlbench-high-key-lighting-v1",
        "ambient": 0.75,
        "directional": 1.00,
        "azimuth_deg": 60.0,
        "elevation_deg": 70.0,
    },
})


def get_lighting_profile(profile_id: str | None = None) -> dict[str, Any]:
    key = profile_id or V3_PROTOCOL.lighting_profile_id
    if key not in LIGHTING_PROFILES:
        raise KeyError(f"unknown lighting profile: {key}")
    return dict(LIGHTING_PROFILES[key])


def lighting_profile_ids() -> tuple[str, ...]:
    return tuple(LIGHTING_PROFILES)
