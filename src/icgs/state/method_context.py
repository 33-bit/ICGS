"""Owned event-memory lineage and externally prepared method context state."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import Tensor

from icgs.contracts.method import SegmentRef
from icgs.data.preprocessing.events import TimedDemoInput
from icgs.models.encoders.event import EventEncoding
from icgs.state.context_cache import PreparedContext


_EVENT_WIDTH = 256


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _raw_hash_row(raw_hashes: Any) -> tuple[str, ...]:
    if not isinstance(raw_hashes, tuple):
        raise TypeError("raw_hashes must be a tuple")
    if not raw_hashes:
        raise ValueError("raw_hashes must be nonempty")
    for value in raw_hashes:
        _identifier(value, "raw_hashes entry")
    if len(set(raw_hashes)) != len(raw_hashes):
        raise ValueError(
            "raw_hashes entries must be unique within each EventMemory batch row"
        )
    return raw_hashes


def context_fingerprint(
    raw_hashes: tuple[str, ...],
    encoder_id: str,
    segmentation_id: str,
) -> str:
    """Hash one ordered demonstration-representation lineage."""

    raw_hashes = _raw_hash_row(raw_hashes)
    encoder_id = _identifier(encoder_id, "encoder_id")
    segmentation_id = _identifier(segmentation_id, "segmentation_id")
    payload = {
        "domain": "icgs.event-memory",
        "schema": 1,
        "raw_hashes": list(raw_hashes),
        "encoder_id": encoder_id,
        "segmentation_id": segmentation_id,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_refs(
    refs: Any,
    raw_hashes: Any,
    valid: Tensor,
) -> tuple[tuple[SegmentRef | None, ...], ...]:
    batch, events = valid.shape
    if not isinstance(refs, tuple) or len(refs) != batch:
        raise ValueError("refs must have exactly the EventMemory batch dimension")
    if not isinstance(raw_hashes, tuple) or len(raw_hashes) != batch:
        raise ValueError("raw_hashes must have exactly the EventMemory batch dimension")
    for row_index, (row_refs, row_hashes) in enumerate(zip(refs, raw_hashes)):
        if not isinstance(row_refs, tuple) or len(row_refs) != events:
            raise ValueError("each refs row must have exactly L entries")
        row_hashes = _raw_hash_row(row_hashes)
        for event_index, reference in enumerate(row_refs):
            is_valid = bool(valid[row_index, event_index].item())
            if is_valid and reference is None:
                raise ValueError("every valid event must have one ref")
            if not is_valid and reference is not None:
                raise ValueError("padding or invalid events must have ref=None")
            if reference is None:
                continue
            if not isinstance(reference, SegmentRef):
                raise TypeError("valid refs must be SegmentRef records")
            if reference.demo_content_hash not in row_hashes:
                raise ValueError("SegmentRef hash must belong to its raw_hashes row")
        referenced_hashes = tuple(
            reference.demo_content_hash
            for reference in row_refs
            if reference is not None
        )
        if any(content_hash not in referenced_hashes for content_hash in row_hashes):
            raise ValueError(
                "every declared raw demo hash must be represented by a valid ref"
            )
        block_order = []
        for content_hash in referenced_hashes:
            if not block_order or block_order[-1] != content_hash:
                block_order.append(content_hash)
        if tuple(block_order) != row_hashes:
            raise ValueError(
                "valid refs must form one contiguous demo block per raw_hashes "
                "entry in declared order"
            )
    return refs


@dataclass(frozen=True)
class EventMemory:
    """Batched owned event tokens with exact per-token provenance."""

    tokens: Tensor
    valid: Tensor
    refs: tuple[tuple[SegmentRef | None, ...], ...]
    raw_hashes: tuple[tuple[str, ...], ...]
    encoder_id: str
    segmentation_id: str
    fingerprints: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.tokens) or self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,L,256]")
        if self.tokens.shape[0] <= 0 or self.tokens.shape[1] <= 0:
            raise ValueError("tokens must have nonempty B and L dimensions")
        if self.tokens.shape[2] != _EVENT_WIDTH:
            raise ValueError("tokens must have shape [B,L,256]")
        if not self.tokens.is_floating_point():
            raise TypeError("tokens must use a floating dtype")
        if not torch.is_tensor(self.valid) or self.valid.shape != self.tokens.shape[:2]:
            raise ValueError("valid must have shape [B,L]")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must use torch.bool dtype")
        if self.valid.device != self.tokens.device:
            raise ValueError("valid must share the tokens device")
        if not bool(torch.isfinite(self.tokens).all().item()):
            raise ValueError("tokens must contain only finite values")
        invalid = (~self.valid)[..., None].expand_as(self.tokens)
        if bool((self.tokens.masked_select(invalid) != 0).any().item()):
            raise ValueError("invalid or padding token rows must be exactly zero")

        refs = _validate_refs(self.refs, self.raw_hashes, self.valid)
        encoder_id = _identifier(self.encoder_id, "encoder_id")
        segmentation_id = _identifier(self.segmentation_id, "segmentation_id")
        raw_hashes = tuple(_raw_hash_row(row) for row in self.raw_hashes)
        fingerprints = tuple(
            context_fingerprint(row, encoder_id, segmentation_id)
            for row in raw_hashes
        )
        object.__setattr__(self, "tokens", self.tokens.clone())
        object.__setattr__(self, "valid", self.valid.clone())
        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "raw_hashes", raw_hashes)
        object.__setattr__(self, "encoder_id", encoder_id)
        object.__setattr__(self, "segmentation_id", segmentation_id)
        object.__setattr__(self, "fingerprints", fingerprints)


def build_event_memory(
    encoding: EventEncoding,
    segment_refs: tuple[tuple[SegmentRef | None, ...], ...],
    raw_hashes: tuple[tuple[str, ...], ...],
    encoder_id: str,
    segmentation_id: str,
) -> EventMemory:
    """Build owned state from an already computed tensor encoding."""

    if not isinstance(encoding, EventEncoding):
        raise TypeError("encoding must be an EventEncoding")
    return EventMemory(
        encoding.tokens,
        encoding.valid,
        segment_refs,
        raw_hashes,
        encoder_id,
        segmentation_id,
    )


def _shares_mutable_storage(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if left is right:
        return True
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.device != right.device or left.numel() == 0 or right.numel() == 0:
            return False
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.shares_memory(left, right))
    return False


def _validate_native_ownership(
    native_full: PreparedContext,
    native_windows: tuple[PreparedContext | None, ...],
) -> None:
    contexts = [native_full, *(window for window in native_windows if window is not None)]
    if len({id(context) for context in contexts}) != len(contexts):
        raise ValueError("full and window PreparedContext objects must be distinct; alias detected")
    for left_index, left in enumerate(contexts):
        left_buffers = (left.embeddings, left.positions)
        for right in contexts[left_index + 1:]:
            right_buffers = (right.embeddings, right.positions)
            if any(
                _shares_mutable_storage(left_buffer, right_buffer)
                for left_buffer in left_buffers
                for right_buffer in right_buffers
            ):
                raise ValueError("PreparedContext mutable buffers must not alias")


@dataclass(frozen=True)
class MethodContext:
    """Online context retaining external native state without copying it."""

    raw_demos: tuple[TimedDemoInput, ...]
    events: EventMemory
    native_full: PreparedContext
    native_windows: tuple[PreparedContext | None, ...]
    reference_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw_demos, tuple) or not self.raw_demos:
            raise TypeError("raw_demos must be a nonempty tuple")
        if not all(isinstance(demo, TimedDemoInput) for demo in self.raw_demos):
            raise TypeError("raw_demos must contain only TimedDemoInput records")
        if not isinstance(self.events, EventMemory):
            raise TypeError("events must be an EventMemory")
        if self.events.tokens.shape[0] != 1:
            raise ValueError("MethodContext supports online B=1 event memory")
        ordered_hashes = tuple(demo.demo_content_hash for demo in self.raw_demos)
        if ordered_hashes != self.events.raw_hashes[0]:
            raise ValueError("raw demo hash order must exactly match EventMemory raw_hashes")
        if not isinstance(self.native_full, PreparedContext):
            raise TypeError("native_full must be a valid injected PreparedContext")
        if not isinstance(self.native_windows, tuple):
            raise TypeError("native_windows must be a tuple")
        event_count = self.events.tokens.shape[1]
        if len(self.native_windows) != event_count:
            raise ValueError("native_windows length must equal EventMemory L")
        if not all(
            window is None or isinstance(window, PreparedContext)
            for window in self.native_windows
        ):
            raise TypeError("native_windows entries must be PreparedContext or None")
        for index, window in enumerate(self.native_windows):
            if window is None:
                continue
            reference = self.events.refs[0][index]
            if (
                not bool(self.events.valid[0, index].item())
                or reference is None
                or reference.kind != "interaction"
                or not reference.valid_action_window
            ):
                raise ValueError(
                    "a native window requires a valid structurally eligible interaction event"
                )
        _validate_native_ownership(self.native_full, self.native_windows)
        _identifier(self.reference_id, "reference_id")

    @property
    def native_window_valid(self) -> Tensor:
        """Derive final availability; no caller-owned mask is stored."""

        values = []
        for index, window in enumerate(self.native_windows):
            reference = self.events.refs[0][index]
            values.append(
                bool(self.events.valid[0, index].item())
                and reference is not None
                and reference.kind == "interaction"
                and reference.valid_action_window
                and window is not None
            )
        return torch.tensor(
            [values],
            dtype=torch.bool,
            device=self.events.valid.device,
        )


__all__ = [
    "EventMemory",
    "MethodContext",
    "build_event_memory",
    "context_fingerprint",
]
