"""Explicit checkpoint translation and loading at the trusted IO boundary."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import torch


_ALIASES = {
    "graph_rep": ("model", "graph"),
    "scene_encoder": ("model", "scene_encoder"),
    "local_encoder": ("model", "local_encoder"),
    "cond_encoder": ("model", "cond_encoder"),
    "action_encoder": ("model", "action_encoder"),
    "action_head_trans": ("model", "prediction_head"),
    "action_head_rot": ("model", "prediction_head_rot"),
    "action_head_grip": ("model", "prediction_head_g"),
}
_RAW_MODEL_OWNERS = {
    "graph",
    "prediction_head",
    "prediction_head_rot",
    "prediction_head_g",
}


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Complete diagnostics for one state-dict translation."""

    renamed: tuple[tuple[str, str], ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    shape_mismatches: tuple[
        tuple[str, str, tuple[int, ...] | None, tuple[int, ...] | None], ...
    ] = ()
    conflicts: tuple[tuple[str, tuple[str, ...]], ...] = ()


class CheckpointCompatibilityError(ValueError):
    """Raised when checkpoint translation would discard incompatible state."""

    def __init__(self, report: CheckpointLoadReport):
        self.report = report
        super().__init__(f"Checkpoint compatibility failure: {report!r}")


def _remove_compiled_segments(key: str) -> str:
    return ".".join(segment for segment in key.split(".") if segment != "_orig_mod")


def _canonical_key(key: str) -> str:
    parts = _remove_compiled_segments(key).split(".")
    if parts and parts[0] in _ALIASES:
        parts = [*_ALIASES[parts[0]], *parts[1:]]
    elif parts and parts[0] in _RAW_MODEL_OWNERS:
        parts = ["model", *parts]
    return ".".join(parts)


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def _equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    try:
        result = left == right
        return bool(result) if not isinstance(result, torch.Tensor) else bool(result.all())
    except (RuntimeError, TypeError, ValueError):
        return False


def normalize_state_dict(state, expected):
    """Translate explicit legacy names into exactly the keys ``expected`` owns.

    Equal registered aliases may collapse or be synthesized. Unequal aliases and
    shape mismatches raise :class:`CheckpointCompatibilityError` regardless of
    the eventual load strictness.
    """
    if not isinstance(state, Mapping) or not isinstance(expected, Mapping):
        raise TypeError("state and expected must be mappings")
    if not all(isinstance(key, str) for key in state):
        raise TypeError("state-dict keys must be strings")
    if not all(isinstance(key, str) for key in expected):
        raise TypeError("expected state-dict keys must be strings")

    source_groups = {}
    for source_key, value in state.items():
        source_groups.setdefault(_canonical_key(source_key), []).append(
            (source_key, value)
        )
    expected_groups = {}
    for expected_key in expected:
        expected_groups.setdefault(_canonical_key(expected_key), []).append(expected_key)

    conflicts = []
    for canonical_key, entries in source_groups.items():
        reference = entries[0][1]
        if any(not _equal(reference, value) for _, value in entries[1:]):
            conflicts.append((canonical_key, tuple(key for key, _ in entries)))

    missing = tuple(
        expected_key
        for expected_key in expected
        if _canonical_key(expected_key) not in source_groups
    )
    unexpected = tuple(
        source_key
        for source_key in state
        if _canonical_key(source_key) not in expected_groups
    )

    translated = OrderedDict()
    renamed = []
    renamed_seen = set()
    shape_mismatches = []
    for canonical_key, expected_keys in expected_groups.items():
        entries = source_groups.get(canonical_key)
        if not entries:
            continue
        for expected_key in expected_keys:
            for source_key, _ in entries:
                pairing = (source_key, expected_key)
                if source_key != expected_key and pairing not in renamed_seen:
                    renamed.append(pairing)
                    renamed_seen.add(pairing)
            source_key, value = next(
                (
                    entry
                    for entry in entries
                    if entry[0] == expected_key
                ),
                entries[0],
            )
            actual_shape = _shape(value)
            expected_shape = _shape(expected[expected_key])
            if actual_shape != expected_shape:
                shape_mismatches.append(
                    (source_key, expected_key, actual_shape, expected_shape)
                )
                continue
            translated[expected_key] = value

    report = CheckpointLoadReport(
        renamed=tuple(renamed),
        missing=missing,
        unexpected=unexpected,
        shape_mismatches=tuple(shape_mismatches),
        conflicts=tuple(conflicts),
    )
    if report.conflicts or report.shape_mismatches:
        raise CheckpointCompatibilityError(report)
    return translated, report


def load_state_dict_compatible(module, state, strict=True):
    """Translate and load ``state`` into a real torch module with diagnostics."""
    translated, report = normalize_state_dict(state, module.state_dict())
    if strict and (report.missing or report.unexpected):
        raise CheckpointCompatibilityError(report)
    if not strict:
        warnings.warn(
            f"Non-strict checkpoint load: {report!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    module.load_state_dict(translated, strict=strict)
    return report


def load_checkpoint_state(path, module, strict=True, map_location="cpu"):
    """Load a raw or Lightning state dict from a *trusted* checkpoint file.

    ``torch.load(..., weights_only=False)`` can execute pickle payloads. Only pass
    artifacts whose source you trust. This function reads but never rewrites the
    artifact; Lightning metadata outside ``state_dict`` remains untouched.
    """
    artifact = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(artifact, Mapping):
        raise TypeError("checkpoint must contain a state-dict mapping")
    if "state_dict" in artifact:
        state = artifact["state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint 'state_dict' must be a mapping")
    else:
        state = artifact
    return load_state_dict_compatible(module, state, strict=strict)
