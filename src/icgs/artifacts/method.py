"""Reference-lineage metadata for the planned method.

Only JSON metadata is handled here.  Tensor and checkpoint IO belongs to the
outer artifact/composition boundary and is intentionally absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from icgs.configuration.method import KNOWN_NATIVE_PROFILES, MethodConfig


REFERENCE_FIELDS = (
    "ip_checksum",
    "native_profile",
    "geometry",
    "physical_weights",
    "event_weights",
    "task_weights",
    "segmentation",
    "router",
    "preprocessing",
    "calibration",
    "camera",
    "gravity",
    "workspace",
    "cadence",
    "rng_protocol",
)
EXCLUDED_REFERENCE_FIELDS = frozenset(
    {"dynamics", "evaluator", "stopping", "learned_stopping", "search"}
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "reference_id",
        "reference_payload",
        "evaluator_reference_id",
        "dynamics_artifact_id",
        "evaluator_artifact_id",
        "learned_stopping_artifact_id",
    }
)

_PREPROCESSING_FIELDS = (
    "voxel_size_m", "num_anchors", "num_points", "neighbors", "ell0_m",
    "fps_start", "tie_break",
)
_RNG_CONFIG_FIELDS = ("generator_seed", "reset_seed", "action_seed")


def _json_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"reference {name} must contain finite JSON numbers")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"reference {name} has a non-string key")
            _json_value(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_value(item, f"{name}[{index}]")
        return
    raise ValueError(f"reference {name} must contain JSON values")


def _required(value: Any, name: str) -> None:
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"reference field {name} is required and nonempty")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"reference {name} must be a positive integer")
    return value


def _validate_cadence(cadence: Any) -> None:
    if not isinstance(cadence, Mapping):
        raise ValueError("reference cadence must be an object")
    for key in ("dt0", "h", "r"):
        if key not in cadence:
            raise ValueError(f"reference cadence missing {key}")
    dt0 = cadence["dt0"]
    if isinstance(dt0, bool) or not isinstance(dt0, (int, float)) or not math.isfinite(float(dt0)) or float(dt0) <= 0:
        raise ValueError("reference cadence dt0 must be finite and positive")
    h, r = _positive_int(cadence["h"], "cadence.h"), _positive_int(cadence["r"], "cadence.r")
    if not 1 <= r <= h <= 8:
        raise ValueError("reference cadence commit/edge horizon is incompatible")
    if "H" in cadence and (not isinstance(cadence["H"], int) or isinstance(cadence["H"], bool) or not 1 <= cadence["H"] <= 512):
        raise ValueError("reference cadence H is incompatible")


def _canonical_reference_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only documented reference aliases before fingerprinting.

    ``anchors`` predates the central ``geometry.num_anchors`` spelling.  It is
    a provenance alias, not an extra tuning knob; conflicting spellings fail
    rather than producing different identities for the same reference.
    """
    canonical = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))

    def normalize_anchors(value: Any, name: str) -> None:
        if not isinstance(value, dict) or "anchors" not in value:
            return
        anchors = value.pop("anchors")
        if "num_anchors" in value and value["num_anchors"] != anchors:
            raise ValueError(f"reference {name} has conflicting anchors aliases")
        value["num_anchors"] = anchors

    for field in ("geometry", "preprocessing"):
        normalize_anchors(canonical.get(field), field)
    physical = canonical.get("physical_weights")
    if isinstance(physical, dict) and isinstance(physical.get("config"), dict):
        normalize_anchors(physical["config"].get("geometry"), "physical geometry")
    return canonical


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"reference {name}/config mismatch")


def _reference_config_projection(config: MethodConfig) -> dict[str, Any]:
    """Return the resolved fields that define the frozen reference path.

    This deliberately does not use ``MethodConfig.fingerprint()``: that hash
    includes evaluator, stopping and search settings which ADR0008 excludes
    from ``pi_ref``.  The selected values cover the P03 physical profile, P04
    causal-memory inputs, and P01 commitment cadence already consumed at the
    outer construction boundary.
    """
    resolved = config.to_dict()
    geometry = resolved["geometry"]
    return {
        "geometry": geometry,
        "preprocessing": {name: geometry[name] for name in _PREPROCESSING_FIELDS},
        "router": resolved["router"],
        "physical": {
            "geometry": geometry,
            "memory": resolved["memory"],
            "neural": resolved["neural"],
            "decoder": resolved["decoder"],
            "numerics": resolved["numerics"],
            "sensors": resolved["sensors"],
            "control": {"dt0": resolved["control"]["dt0"]},
        },
        "rng": {name: resolved["stages"][name] for name in _RNG_CONFIG_FIELDS},
    }


def _validate_reference_config(payload: Mapping[str, Any], config: MethodConfig) -> None:
    """Reject manifests whose selected reference inputs differ from config."""
    expected = _reference_config_projection(config)
    _require_equal(payload["geometry"], expected["geometry"], "geometry")
    _require_equal(payload["preprocessing"], expected["preprocessing"], "preprocessing")
    router = payload["router"]
    if not isinstance(router, Mapping) or not isinstance(router.get("config"), Mapping):
        raise ValueError("reference router/config metadata is required")
    _require_equal(dict(router["config"]), expected["router"], "router")
    physical = payload["physical_weights"]
    if not isinstance(physical, Mapping) or not isinstance(physical.get("config"), Mapping):
        raise ValueError("reference physical/config metadata is required")
    _require_equal(dict(physical["config"]), expected["physical"], "physical")
    rng = payload["rng_protocol"]
    if not isinstance(rng, Mapping) or not isinstance(rng.get("config"), Mapping):
        raise ValueError("reference rng/config metadata is required")
    _require_equal(dict(rng["config"]), expected["rng"], "rng")


def reference_fingerprint(payload: dict) -> str:
    """Return the canonical SHA-256 identity of the frozen reference path."""
    if not isinstance(payload, Mapping):
        raise ValueError("reference payload must be an object")
    _json_value(payload, "payload")
    payload = _canonical_reference_payload(payload)
    allowed = set(REFERENCE_FIELDS) | EXCLUDED_REFERENCE_FIELDS
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"reference payload has unknown fields: {sorted(unknown)}")
    missing = [name for name in REFERENCE_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"reference payload missing fields: {missing}")
    for name in REFERENCE_FIELDS:
        _required(payload[name], name)

    checksum = payload["ip_checksum"]
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checksum) is None:
        raise ValueError("reference ip_checksum must be a SHA-256 hex digest")
    profile = payload["native_profile"]
    if profile not in KNOWN_NATIVE_PROFILES:
        raise ValueError(f"reference native profile is unknown: {profile!r}")
    _validate_cadence(payload["cadence"])

    canonical = {name: payload[name] for name in REFERENCE_FIELDS}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config(value: MethodConfig | Mapping[str, Any]) -> MethodConfig:
    if isinstance(value, MethodConfig):
        return value.validate()
    if isinstance(value, Mapping):
        return MethodConfig.from_dict(dict(value))
    raise ValueError("reference manifest requires MethodConfig metadata")


def validate_method_manifest(
    manifest: Mapping[str, Any],
    config: MethodConfig | Mapping[str, Any],
    *,
    artifact_loader: Any = None,
) -> str:
    """Validate lineage metadata before an outer caller reads any artifact."""
    if not isinstance(manifest, Mapping):
        raise ValueError("reference manifest must be an object")
    _json_value(manifest, "manifest")
    unknown = set(manifest) - _MANIFEST_FIELDS
    if unknown:
        raise ValueError(f"reference manifest has unknown fields: {sorted(unknown)}")
    required = {
        "schema_version", "reference_id", "reference_payload", "evaluator_reference_id",
        "dynamics_artifact_id", "evaluator_artifact_id", "learned_stopping_artifact_id",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"reference manifest missing fields: {sorted(missing)}")
    if manifest["schema_version"] != 1 or isinstance(manifest["schema_version"], bool):
        raise ValueError("unsupported reference manifest schema")
    expected = reference_fingerprint(manifest["reference_payload"])
    if manifest["reference_id"] != expected:
        raise ValueError("reference manifest fingerprint mismatch")
    if manifest["evaluator_reference_id"] != expected:
        raise ValueError("evaluator/reference lineage mismatch")
    if artifact_loader is not None and not callable(artifact_loader):
        raise ValueError("reference artifact_loader must be callable")

    method_config = _config(config)
    payload = _canonical_reference_payload(manifest["reference_payload"])
    if payload["native_profile"] != method_config.native_profile:
        raise ValueError("reference native profile/config mismatch")
    cadence = payload["cadence"]
    if cadence["h"] != method_config.planning.h or cadence["r"] != method_config.planning.r:
        raise ValueError("reference cadence/config mismatch")
    if not math.isclose(float(cadence["dt0"]), method_config.control.dt0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reference dt0/config mismatch")
    if "H" in cadence and cadence["H"] != method_config.planning.H:
        raise ValueError("reference horizon/config mismatch")
    _validate_reference_config(payload, method_config)
    for role, name in (("dynamics", "dynamics_artifact_id"),
                       ("evaluator", "evaluator_artifact_id"),
                       ("learned_stopping", "learned_stopping_artifact_id")):
        if name in manifest and (not isinstance(manifest[name], str) or not manifest[name]):
            raise ValueError(f"reference {name} must be a nonempty ID")
        if artifact_loader is not None:
            artifact_loader(role, manifest[name])
    return expected


__all__ = ["EXCLUDED_REFERENCE_FIELDS", "REFERENCE_FIELDS", "reference_fingerprint", "validate_method_manifest"]
