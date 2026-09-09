"""Explicit bounded NumPy capture; no tensor/device conversion is attempted."""

from __future__ import annotations

import hashlib
import os
from collections import deque
from pathlib import Path
import shutil
import threading
import uuid
from typing import Any, Mapping

import numpy as np

from .records import CaptureResult, SCHEMA_VERSION
from .serialization import json_bytes, safe_value


_MAX_ARRAY_ENTRIES = 256
_MAX_ARRAY_KEY_CHARS = 256
_HASH_CHUNK_BYTES = 64 * 1024


class _CaptureQuotaExceeded(Exception):
    """A bounded capture would exceed its configured stored-byte quota."""


class _BoundedArchiveWriter:
    """Non-seekable ZIP output which refuses bytes beyond a finite budget."""

    def __init__(self, stream, limit: int):
        self.stream = stream
        self.limit = limit
        self.written = 0

    def write(self, data):
        size = len(data)
        if self.written + size > self.limit:
            raise _CaptureQuotaExceeded("capture archive quota exceeded")
        written = self.stream.write(data)
        if written != size:
            raise OSError(
                f"observability archive short write: expected {size} bytes, got {written}"
            )
        self.written += written
        return written

    def tell(self):
        return self.written

    def flush(self):
        return self.stream.flush()

    def read(self, *args, **kwargs):
        return self.stream.read(*args, **kwargs)

    def writable(self):
        return True

    def seekable(self):
        return False

    def seek(self, *args, **kwargs):
        raise OSError("observability capture archive is not seekable")


def _hash_file(path: Path, *, max_bytes: int | None = None) -> tuple[int, str]:
    """Hash a bounded file incrementally without materializing its contents."""
    size = path.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise _CaptureQuotaExceeded("capture archive quota exceeded")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise OSError("observability capture archive was truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    if path.stat().st_size != size:
        raise OSError("observability capture archive changed while hashing")
    return size, digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json_bytes(value) + b"\n"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError("observability manifest short write")
                stream.flush()
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _open_private(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        stream = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise
    os.chmod(path, 0o600)
    return stream


class CaptureManager:
    def __init__(self, run_dir: Path, config: Any, *, rank: int = 0):
        self.run_dir = Path(run_dir)
        self.config = config
        self.rank = rank
        self.count = 0
        self.total_bytes = 0
        self._quota_lock = threading.Lock()
        self.captures_dir = self.run_dir / "captures"
        _private_dir(self.captures_dir)

    def _common_rejection(self, name: Any) -> CaptureResult | None:
        if self.count >= self.config.capture_max_count:
            return CaptureResult(False, reason="capture_count_quota")
        if not isinstance(name, str) or not name.strip():
            return CaptureResult(False, reason="invalid_capture_name")
        return None

    def _cleanup(self, capture_dir: Path) -> None:
        try:
            if capture_dir.is_symlink():
                capture_dir.unlink()
            elif capture_dir.exists():
                # The directory name is freshly generated and unpublished for
                # this operation; keep cleanup scoped to that exact bundle.
                shutil.rmtree(capture_dir)
        except OSError:
            pass

    def _write_metadata_manifest(
        self,
        name: str,
        *,
        context: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        recent_events: list[Mapping[str, Any]],
    ) -> CaptureResult:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": uuid.uuid4().hex,
            "name": name,
            "archive": None,
            "arrays": {},
            "recent_events": safe_value(recent_events, max_depth=32),
            "context": safe_value(context or {}, max_depth=32),
            "metadata": safe_value(metadata or {}, max_depth=32),
        }
        payload = json_bytes(manifest) + b"\n"
        if len(payload) > self.config.capture_max_bytes_each:
            return CaptureResult(False, reason="capture_size_quota")
        if self.total_bytes + len(payload) > self.config.capture_total_bytes:
            return CaptureResult(False, reason="capture_total_quota")
        capture_dir = self.captures_dir / manifest["capture_id"]
        try:
            _private_dir(capture_dir)
            _atomic_json(capture_dir / "manifest.json", manifest)
        except Exception as exc:
            self._cleanup(capture_dir)
            return CaptureResult(False, reason=f"capture_write_failed:{type(exc).__name__}")
        self.count += 1
        self.total_bytes += len(payload)
        return CaptureResult(True, capture_id=manifest["capture_id"],
                             path=str(capture_dir), bytes_written=len(payload))

    def capture_metadata(
        self,
        name: str,
        *,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        recent_events: list[Mapping[str, Any]] | None = None,
    ) -> CaptureResult:
        """Persist a bounded context/recent-event bundle without raw arrays."""
        if self.rank != 0:
            return CaptureResult(False, reason="non_rank0_capture_owner")
        if self.config.mode != "capture":
            return CaptureResult(False, reason="capture_mode_required")
        with self._quota_lock:
            rejection = self._common_rejection(name)
            if rejection is not None:
                return rejection
            if isinstance(recent_events, list):
                bounded_recent = list(recent_events[-self.config.recent_event_capacity:])
            else:
                bounded_recent = list(deque(
                    recent_events or (), maxlen=self.config.recent_event_capacity
                ))
            return self._write_metadata_manifest(
                name, context=context, metadata=metadata,
                recent_events=bounded_recent,
            )

    def capture(self, name: str, arrays: Mapping[str, Any], *, context: Mapping[str, Any] | None = None,
                metadata: Mapping[str, Any] | None = None) -> CaptureResult:
        if self.rank != 0:
            return CaptureResult(False, reason="non_rank0_capture_owner")
        if self.config.mode != "capture" or not self.config.capture_arrays:
            return CaptureResult(False, reason="capture_arrays_not_enabled")
        with self._quota_lock:
            rejection = self._common_rejection(name)
            if rejection is not None:
                return rejection
            if not isinstance(arrays, Mapping) or not arrays:
                return CaptureResult(False, reason="numeric_array_mapping_required")

            if len(name) > min(self.config.max_field_string_chars, _MAX_ARRAY_KEY_CHARS):
                return CaptureResult(False, reason="invalid_capture_name")

            # Admit only a bounded list of mapping entries before inspecting or
            # copying any array.  This prevents a large/custom Mapping from
            # turning the capture path into an unbounded serializer.
            entries: list[tuple[Any, Any]] = []
            for index, entry in enumerate(arrays.items()):
                if index >= _MAX_ARRAY_ENTRIES:
                    return CaptureResult(False, reason="capture_entry_quota")
                try:
                    key, value = entry
                except (TypeError, ValueError):
                    return CaptureResult(False, reason="invalid_array_name")
                entries.append((key, value))

            raw_bytes = 0
            array_metadata: dict[str, dict[str, Any]] = {}
            for key, value in entries:
                if (not isinstance(key, str) or not key or len(key) > _MAX_ARRAY_KEY_CHARS
                        or len(key) > self.config.max_field_string_chars
                        or "/" in key or "\\" in key):
                    return CaptureResult(False, reason="invalid_array_name")
                if not isinstance(value, np.ndarray):
                    return CaptureResult(False, reason="cpu_numpy_arrays_required")
                if value.dtype.kind not in "buiufc":
                    return CaptureResult(False, reason="object_or_non_numeric_array")
                array_bytes = int(value.nbytes)
                if raw_bytes + array_bytes > self.config.capture_max_bytes_each:
                    return CaptureResult(False, reason="capture_size_quota")
                raw_bytes += array_bytes
                array_metadata[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "nbytes": array_bytes,
                }

            safe_context = safe_value(
                context or {}, max_depth=32,
                max_string_chars=self.config.max_field_string_chars,
            )
            safe_metadata = safe_value(
                metadata or {}, max_depth=32,
                max_string_chars=self.config.max_field_string_chars,
            )
            capture_id = uuid.uuid4().hex
            manifest_bound = {
                "schema_version": SCHEMA_VERSION,
                "capture_id": capture_id,
                "name": name,
                "archive": "arrays.npz",
                "archive_sha256": "0" * 64,
                # The configured capture bound has no more decimal digits
                # than any archive size admitted below.
                "bytes": self.config.capture_max_bytes_each,
                "arrays": array_metadata,
                "context": safe_context,
                "metadata": safe_metadata,
            }
            manifest_bound_bytes = len(json_bytes(manifest_bound)) + 1
            if manifest_bound_bytes >= self.config.capture_max_bytes_each:
                return CaptureResult(False, reason="capture_size_quota")
            total_remaining = self.config.capture_total_bytes - self.total_bytes
            if total_remaining <= manifest_bound_bytes:
                return CaptureResult(False, reason="capture_total_quota")
            archive_limit = min(
                self.config.capture_max_bytes_each - manifest_bound_bytes,
                total_remaining - manifest_bound_bytes,
            )
            if raw_bytes > archive_limit:
                if raw_bytes > self.config.capture_max_bytes_each - manifest_bound_bytes:
                    return CaptureResult(False, reason="capture_size_quota")
                return CaptureResult(False, reason="capture_total_quota")

            # All source checks are complete before the explicit deep copy.
            copied: dict[str, np.ndarray] = {}
            for key, value in entries:
                array = np.array(value, copy=True, order="C")
                copied_bytes = int(array.nbytes)
                if copied_bytes != int(value.nbytes):
                    return CaptureResult(False, reason="capture_size_quota")
                copied[key] = array

            capture_dir = self.captures_dir / capture_id
            try:
                _private_dir(capture_dir)
                archive_path = capture_dir / "arrays.npz"
                temporary = capture_dir / ".arrays.npz.tmp"
                with _open_private(temporary) as stream:
                    bounded = _BoundedArchiveWriter(stream, archive_limit)
                    np.savez_compressed(bounded, **copied)
                    bounded.flush()
                    stream.flush()
                    os.fsync(stream.fileno())
                archive_bytes, archive_sha256 = _hash_file(
                    temporary, max_bytes=archive_limit
                )
                os.replace(temporary, archive_path)
                os.chmod(archive_path, 0o600)
                manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "capture_id": capture_id,
                    "name": name,
                    "archive": "arrays.npz",
                    "archive_sha256": archive_sha256,
                    "bytes": archive_bytes,
                    "arrays": array_metadata,
                    "context": safe_context,
                    "metadata": safe_metadata,
                }
                manifest_bytes = len(json_bytes(manifest)) + 1
                bundle_bytes = archive_bytes + manifest_bytes
                if bundle_bytes > self.config.capture_max_bytes_each:
                    raise _CaptureQuotaExceeded("capture bundle quota exceeded")
                if self.total_bytes + bundle_bytes > self.config.capture_total_bytes:
                    raise _CaptureQuotaExceeded("capture total quota exceeded")
                _atomic_json(capture_dir / "manifest.json", manifest)
            except _CaptureQuotaExceeded as exc:
                self._cleanup(capture_dir)
                reason = "capture_total_quota" if "total" in str(exc) else "capture_size_quota"
                return CaptureResult(False, reason=reason)
            except Exception as exc:
                self._cleanup(capture_dir)
                return CaptureResult(False, reason=f"capture_write_failed:{type(exc).__name__}")

            bundle_bytes = archive_bytes + manifest_bytes
            self.count += 1
            self.total_bytes += bundle_bytes
            return CaptureResult(True, capture_id=capture_id, path=str(capture_dir), bytes_written=bundle_bytes)


__all__ = ["CaptureManager"]
