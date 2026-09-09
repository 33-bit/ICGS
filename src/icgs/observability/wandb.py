"""Lazy optional W&B mirror with a separate bounded queue."""

from __future__ import annotations

import importlib
import os
from queue import Empty, Full, Queue
import threading
from typing import Any, Mapping


class WandbAdapter:
    def __init__(self, run: Any, config: Any, *, recorder=None,
                 timeout_s: float | None = None):
        self.run = run
        self.config = config
        self.recorder = recorder
        if timeout_s is None:
            timeout_s = getattr(config, "shutdown_timeout_s", None)
        if timeout_s is None:
            # Direct adapter tests/callers do not have the enclosing central
            # config; use the same packaged default rather than a second
            # shutdown policy.
            from .config import default_config
            timeout_s = default_config().shutdown_timeout_s
        self.timeout_s = max(0.0, float(timeout_s))
        self.queue: Queue[tuple[dict[str, float], int | None]] = Queue(maxsize=256)
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._disabled = False
        self._thread = threading.Thread(target=self._worker, name="icgs-wandb", daemon=True)
        self._thread.start()

    @property
    def disabled(self) -> bool:
        with self._state_lock:
            return self._disabled

    def _disable(self, error: BaseException) -> None:
        with self._state_lock:
            if self._disabled:
                return
            self._disabled = True
        if self.recorder is not None:
            try:
                self.recorder.event(
                    "wandb.mirror.error", level="ERROR", component="wandb",
                    fields={"type": type(error).__name__},
                )
            except Exception:
                pass

    def _worker(self) -> None:
        while True:
            try:
                values, step = self.queue.get(timeout=0 if self._stop.is_set() else 0.1)
            except Empty:
                if self._stop.is_set():
                    break
                continue
            try:
                if self.disabled:
                    continue
                kwargs = {"step": step} if step is not None else {}
                self.run.log(values, **kwargs)
            except Exception as exc:
                self._disable(exc)
            finally:
                self.queue.task_done()
            if self._stop.is_set() and self.queue.empty():
                break
        # Finish is serialized in this same worker, after every queued log.
        # A timeout in close() disables the mirror, so a late unblock cannot
        # perform a second, racy SDK operation after local shutdown returned.
        if not self.disabled:
            try:
                self.run.finish()
            except Exception as exc:
                self._disable(exc)

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        if self.disabled or self._closed:
            return
        filtered = {}
        for name, value in values.items():
            if not isinstance(name, str) or not (
                name == "Train_Loss" or name.startswith("Val_")
                or any(name.startswith(prefix) for prefix in self.config.metric_prefixes)
            ):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                continue
            filtered[name] = value
        if not filtered:
            return
        with self._close_lock:
            if self._closed or self.disabled:
                return
            try:
                self.queue.put_nowait((filtered, step))
            except Full:
                if self.recorder is not None:
                    self.recorder.event("wandb.mirror.dropped", level="WARNING", component="wandb")

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=self.timeout_s)
        if self._thread.is_alive():
            self._disable(TimeoutError("W&B mirror shutdown deadline exceeded"))


def start(config: Any, *, run_id: str, metadata: Mapping[str, Any] | None = None,
          recorder=None) -> WandbAdapter | None:
    if not config.enabled:
        return None
    rank = getattr(recorder, "rank", None)
    if rank is None:
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
    if rank != 0:
        if recorder is not None:
            try:
                recorder.event("wandb.skipped", level="WARNING", component="wandb",
                               fields={"reason": "non_rank0", "rank": rank})
            except Exception:
                pass
        return None
    try:
        sdk = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError("W&B was explicitly requested but the wandb package is unavailable") from exc
    kwargs = {"project": config.project, "mode": config.mode, "id": run_id}
    if config.entity is not None:
        kwargs["entity"] = config.entity
    if config.tags:
        kwargs["tags"] = list(config.tags)
    if config.send_config and metadata:
        kwargs["config"] = dict(metadata)
    run = sdk.init(**kwargs)
    shutdown_timeout_s = getattr(getattr(recorder, "config", None),
                                 "shutdown_timeout_s", None)
    return WandbAdapter(run, config, recorder=recorder,
                        timeout_s=shutdown_timeout_s)


__all__ = ["WandbAdapter", "start"]
