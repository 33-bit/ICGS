"""Bounded per-process local JSONL writer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Callable

from .serialization import json_bytes


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _open_private_stream(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        stream = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise
    os.chmod(path, 0o600)
    return stream


@dataclass(frozen=True)
class PendingRecord:
    kind: str
    payload: dict[str, Any]
    critical: bool


class _ShortWriteError(OSError):
    """The stream accepted fewer bytes than the complete JSONL record."""


class LocalWriter:
    """Write copied records without blocking the producer on disk IO."""

    def __init__(self, run_dir: Path, config: Any, *, process_id: int, rank: int,
                 on_warning: Callable[[str], None] | None = None):
        self.run_dir = Path(run_dir)
        self.events_dir = self.run_dir / "events"
        self.metrics_dir = self.run_dir / "metrics"
        _private_dir(self.events_dir)
        _private_dir(self.metrics_dir)
        self.config = config
        self.process_id = process_id
        self.rank = rank
        self.on_warning = on_warning or (lambda message: None)
        self.queue: Queue[PendingRecord] = Queue(maxsize=config.queue_capacity)
        self._detail_capacity = config.queue_capacity - config.reserved_critical_slots
        self._queued = 0
        self._queue_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"icgs-observability-{process_id}", daemon=True)
        self._file_lock = threading.Lock()
        self._event_index = 1
        self._event_bytes = 0
        self._critical_event_bytes = 0
        self._emergency_capacity = (config.max_record_bytes + 1) * config.reserved_critical_slots
        self._metrics_bytes = 0
        self._event_stream = None
        self._metric_stream = None
        self._sink_disabled = False
        self._warned_sink = False
        self._stats = {
            "events": 0,
            "metrics": 0,
            "dropped": 0,
            "dropped_critical": 0,
            "metrics_skipped": 0,
            "sink_errors": 0,
            "record_errors": 0,
            "short_writes": 0,
            "flush_errors": 0,
            "close_errors": 0,
            "event_chunks": 0,
        }
        self._open_streams()

    def _open_streams(self) -> None:
        prefix = f"p{self.process_id}-r{self.rank}-"
        event_path = self.events_dir / f"{prefix}{self._event_index:06d}.jsonl"
        self._event_stream = _open_private_stream(event_path)
        self._stats["event_chunks"] = 1
        self._metric_stream = _open_private_stream(self.metrics_dir / f"{prefix}000001.jsonl")

    @property
    def stats(self) -> dict[str, int]:
        with self._file_lock:
            return dict(self._stats)

    @property
    def sink_disabled(self) -> bool:
        return self._sink_disabled

    def enqueue(self, kind: str, payload: dict[str, Any], *, critical: bool = False) -> bool:
        item = PendingRecord(kind, dict(payload), critical)
        with self._queue_lock:
            if self._sink_disabled:
                self._stats["dropped"] += 1
                if critical:
                    self._stats["dropped_critical"] += 1
                return False
            if not critical and self._queued >= self._detail_capacity:
                self._stats["dropped"] += 1
                return False
            try:
                self.queue.put_nowait(item)
                self._queued += 1
                return True
            except Full:
                self._stats["dropped"] += 1
                if critical:
                    self._stats["dropped_critical"] += 1
                    self._warn("observability critical queue is full; preserving computation outcome")
                return False

    def note_metric_skipped(self) -> None:
        with self._file_lock:
            self._stats["metrics_skipped"] += 1

    def note_internal_failure(self) -> None:
        with self._file_lock:
            self._stats["record_errors"] += 1
            self._stats["sink_errors"] += 1

    def _warn(self, message: str) -> None:
        try:
            self.on_warning(message)
        except Exception:
            pass

    def _mark_sink_error(self, error: BaseException, *, phase: str) -> None:
        self._sink_disabled = True
        self._stats["sink_errors"] += 1
        if phase == "short_write":
            self._stats["short_writes"] += 1
        elif phase == "flush":
            self._stats["flush_errors"] += 1
        elif phase == "close":
            self._stats["close_errors"] += 1
        if not self._warned_sink:
            self._warned_sink = True
            self._warn(
                f"observability local sink disabled after {phase} failure: "
                f"{type(error).__name__}"
            )

    def _mark_record_error(self, error: BaseException) -> None:
        self._stats["record_errors"] += 1
        self._stats["sink_errors"] += 1
        self._warn(f"observability record was not written: {type(error).__name__}")

    def _write_exact(self, stream, line: bytes) -> None:
        written = stream.write(line)
        if written != len(line):
            raise _ShortWriteError(
                f"observability short write: expected {len(line)} bytes, got {written}"
            )

    def _close_stream(self, stream) -> None:
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            stream.flush()
        except Exception as exc:
            self._mark_sink_error(exc, phase="flush")
        try:
            stream.close()
        except Exception as exc:
            self._mark_sink_error(exc, phase="close")

    def _write_line(self, item: PendingRecord) -> None:
        try:
            line = json_bytes(item.payload, max_bytes=self.config.max_record_bytes) + b"\n"
        except Exception as exc:
            with self._file_lock:
                self._mark_record_error(exc)
            return
        with self._file_lock:
            if self._sink_disabled:
                self._stats["dropped"] += 1
                return
            try:
                if item.kind == "metric":
                    if self._metrics_bytes + len(line) > self.config.metrics_total_bytes_per_process:
                        self._stats["metrics_skipped"] += 1
                        return
                    self._write_exact(self._metric_stream, line)
                    self._metrics_bytes += len(line)
                    self._stats["metrics"] += 1
                    return
                if not item.critical and len(line) > self.config.event_chunk_bytes:
                    self._stats["dropped"] += 1
                    return
                overflow = self._event_bytes + len(line) > self.config.event_chunk_bytes
                if overflow and item.critical:
                    if self._critical_event_bytes + len(line) > self._emergency_capacity:
                        self._stats["dropped"] += 1
                        self._stats["dropped_critical"] += 1
                        return
                if overflow and not item.critical:
                    if self._event_index >= self.config.max_event_chunks_per_process:
                        self._stats["dropped"] += 1
                        return
                    try:
                        self._event_stream.flush()
                    except Exception as exc:
                        self._mark_sink_error(exc, phase="flush")
                        return
                    try:
                        self._event_stream.close()
                    except Exception as exc:
                        self._mark_sink_error(exc, phase="close")
                        return
                    self._event_index += 1
                    event_path = self.events_dir / f"p{self.process_id}-r{self.rank}-{self._event_index:06d}.jsonl"
                    try:
                        self._event_stream = _open_private_stream(event_path)
                    except Exception as exc:
                        self._mark_sink_error(exc, phase="open")
                        return
                    self._event_bytes = 0
                    self._stats["event_chunks"] = self._event_index
                self._write_exact(self._event_stream, line)
                self._event_bytes += len(line)
                if overflow and item.critical:
                    self._critical_event_bytes += len(line)
                self._stats["events"] += 1
            except Exception as exc:
                phase = "short_write" if isinstance(exc, _ShortWriteError) else "write"
                self._mark_sink_error(exc, phase=phase)

    def _flush(self) -> None:
        with self._file_lock:
            if self._sink_disabled:
                return
            for stream in (self._event_stream, self._metric_stream):
                if stream is None or getattr(stream, "closed", False):
                    continue
                try:
                    stream.flush()
                except Exception as exc:
                    self._mark_sink_error(exc, phase="flush")
                    return

    def _run(self) -> None:
        last_flush = time.monotonic()
        try:
            while not self._stop.is_set() or self._queued:
                try:
                    item = self.queue.get(timeout=self.config.flush_interval_s)
                except Empty:
                    self._flush()
                    last_flush = time.monotonic()
                    continue
                try:
                    self._write_line(item)
                except Exception as exc:
                    with self._file_lock:
                        self._mark_record_error(exc)
                finally:
                    with self._queue_lock:
                        self._queued = max(0, self._queued - 1)
                    self.queue.task_done()
                if time.monotonic() - last_flush >= self.config.flush_interval_s:
                    self._flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            with self._file_lock:
                self._mark_record_error(exc)
        finally:
            self._finalize_streams()

    def _finalize_streams(self) -> None:
        self._flush()
        with self._file_lock:
            self._close_stream(self._event_stream)
            self._close_stream(self._metric_stream)

    def close(self, timeout_s: float) -> bool:
        self._stop.set()
        if self._thread.ident is None:
            finalizer = threading.Thread(
                target=self._finalize_streams,
                name=f"icgs-observability-close-{self.process_id}",
                daemon=True,
            )
            finalizer.start()
            finalizer.join(timeout=max(0.0, float(timeout_s)))
            return not finalizer.is_alive()
        self._thread.join(timeout=max(0.0, float(timeout_s)))
        return not self._thread.is_alive()


__all__ = ["LocalWriter", "PendingRecord"]
