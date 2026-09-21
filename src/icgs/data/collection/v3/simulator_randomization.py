"""Runtime application of approved camera and lighting randomization."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class SimulatorRandomizationError(RuntimeError):
    pass


def _call_first(target: Any, names: Sequence[str], *args: Any) -> str:
    for name in names:
        method = getattr(target, name, None)
        if callable(method):
            method(*args)
            return name
    raise SimulatorRandomizationError(
        f"{type(target).__name__} does not support any of: {', '.join(names)}"
    )


def apply_camera_viewpoint(camera: Any, viewpoint: Mapping[str, Any]) -> dict[str, Any]:
    if camera is None:
        raise SimulatorRandomizationError("wrist camera object is unavailable")
    yaw = math.radians(float(viewpoint.get("d_yaw_deg", 0.0)))
    pitch = math.radians(float(viewpoint.get("d_pitch_deg", 0.0)))
    getter = getattr(camera, "get_orientation", None)
    if not callable(getter):
        raise SimulatorRandomizationError("camera has no get_orientation()")
    orientation = list(getter())
    if len(orientation) != 3:
        raise SimulatorRandomizationError("camera orientation must be XYZ Euler [3]")
    orientation[0] += pitch
    orientation[1] += yaw
    method = _call_first(camera, ("set_orientation",), orientation)
    return {
        "applied": True,
        "d_yaw_deg": float(viewpoint.get("d_yaw_deg", 0.0)),
        "d_pitch_deg": float(viewpoint.get("d_pitch_deg", 0.0)),
        "setter": method,
    }


def apply_lighting_profile(
    lights: Sequence[Any], profile: Mapping[str, Any], *, ambient_target: Any = None
) -> dict[str, Any]:
    if not lights:
        raise SimulatorRandomizationError("no simulator lights were found")
    directional = float(profile["directional"])
    azimuth = math.radians(float(profile["azimuth_deg"]))
    elevation = math.radians(float(profile["elevation_deg"]))
    setters: list[dict[str, str]] = []
    if ambient_target is None:
        raise SimulatorRandomizationError("ambient-light runtime target is unavailable")
    ambient_setter = None
    for name, args in (
        ("set_ambient_light", ([float(profile["ambient"])] * 3,)),
        ("set_ambient_intensity", (float(profile["ambient"]),)),
    ):
        method = getattr(ambient_target, name, None)
        if callable(method):
            method(*args)
            ambient_setter = name
            break
    if ambient_setter is None:
        raise SimulatorRandomizationError("runtime has no ambient-light setter")
    for light in lights:
        intensity_setter = _call_first(light, ("set_intensity",), directional)
        orientation = [elevation, 0.0, azimuth]
        orientation_setter = _call_first(light, ("set_orientation",), orientation)
        setters.append({"intensity": intensity_setter, "orientation": orientation_setter})
    return {
        "applied": True,
        "lighting_profile_id": profile.get("lighting_profile_id"),
        "ambient": float(profile["ambient"]),
        "directional": directional,
        "azimuth_deg": float(profile["azimuth_deg"]),
        "elevation_deg": float(profile["elevation_deg"]),
        "setters": setters,
        "ambient_setter": ambient_setter,
    }


def apply_sensor_randomization(
    *,
    camera: Any,
    lights: Sequence[Any],
    ambient_target: Any = None,
    camera_viewpoint: Mapping[str, Any],
    lighting_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply both nuisance factors or raise without claiming partial success."""
    camera_receipt = apply_camera_viewpoint(camera, camera_viewpoint)
    try:
        lighting_receipt = apply_lighting_profile(lights, lighting_profile, ambient_target=ambient_target)
    except Exception:
        # Do not leave a camera-only partial claim in a closed attempt.
        raise
    return {"camera": camera_receipt, "lighting": lighting_receipt, "applied": True}


__all__ = [
    "SimulatorRandomizationError",
    "apply_camera_viewpoint",
    "apply_lighting_profile",
    "apply_sensor_randomization",
]
