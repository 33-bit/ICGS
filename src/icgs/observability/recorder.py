"""Run recorder, no-op capability, correlated spans and bounded local files."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
import hashlib
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Iterator, Mapping

from .config import default_config
from .context import TraceContext, _CURRENT, bind_context, current_context
from .local import LocalWriter
from .records import CaptureResult, RunSummary, SCHEMA_VERSION
from .serialization import exception_payload, json_bytes, metric_value, safe_value


def _new_id() -> str:
    return uuid.uuid4().hex


def _rank() -> int:
    value = os.environ.get("RANK", "0")
    try:
        return int(value)
    except ValueError:
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _level(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("observability level must be a string")
    value = value.upper()
    if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"unknown observability level: {value!r}")
    return value


_LEVEL_RANK = {name: rank for rank, name in enumerate(
    ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
)}


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{_new_id()}.tmp")
    encoded = json_bytes(payload) + b"\n"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                written = stream.write(encoded)
                if written != len(encoded):
                    raise OSError("observability metadata short write")
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


def _private_root(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for created in reversed(missing):
        created.mkdir(mode=0o700, exist_ok=True)
        os.chmod(created, 0o700)


class _OwnedConsoleHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(f"[{record.levelname}] {record.name}: {record.getMessage()}", file=sys.stderr)
        except Exception:
            pass


class _OwnedEventHandler(logging.Handler):
    def __init__(self, recorder: "RunRecorder"):
        super().__init__()
        self.recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.recorder.event(record.getMessage(), level=record.levelname,
                                component=record.name or "logging")
        except Exception:
            pass


class NoopSpan:
    def __init__(self):
        self.span_id: str | None = None

    def __enter__(self) -> "NoopSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None


class BoundRecorder:
    def __init__(self, recorder: "RunRecorder | NoopRecorder", updates: Mapping[str, Any]):
        self.recorder = recorder
        self.updates = dict(updates)
        self._token = None

    def __enter__(self) -> "BoundRecorder":
        values = dict(self.updates)
        if self.recorder.run_id is not None:
            values.setdefault("run_id", self.recorder.run_id)
        self._token = _CURRENT.set(current_context().update(**values))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            try:
                _CURRENT.reset(self._token)
            except Exception as failure:
                self.recorder._note_observability_failure("context restore", failure)
            self._token = None
        return False

    @contextmanager
    def _bound(self) -> Iterator[None]:
        if self._token is not None:
            yield
            return
        values = dict(self.updates)
        if self.recorder.run_id is not None:
            values.setdefault("run_id", self.recorder.run_id)
        with bind_context(**values):
            yield

    def bind(self, **updates: Any) -> "BoundRecorder":
        merged = dict(self.updates)
        merged.update(updates)
        return BoundRecorder(self.recorder, merged)

    def event(self, name: str, *, level: str = "INFO", component: str = "runtime",
              fields: Mapping[str, Any] | None = None) -> None:
        with self._bound():
            self.recorder.event(name, level=level, component=component, fields=fields)

    def span(self, name: str, *, component: str = "runtime",
             fields: Mapping[str, Any] | None = None):
        if not self.recorder.enabled:
            return NoopSpan()
        return _Span(self.recorder, name, component, fields, self.updates)

    def metrics(self, values: Mapping[str, Any], *, axis: str, step: int,
                component: str, units: Mapping[str, str] | None = None) -> None:
        with self._bound():
            self.recorder.metrics(values, axis=axis, step=step, component=component, units=units)

    def exception(self, error: BaseException, *, component: str,
                  fields: Mapping[str, Any] | None = None) -> None:
        with self._bound():
            self.recorder.exception(error, component=component, fields=fields)

    def capture(self, name: str, arrays: Mapping[str, Any], *,
                context: Mapping[str, Any] | None = None,
                metadata: Mapping[str, Any] | None = None) -> CaptureResult:
        with self._bound():
            return self.recorder.capture(name, arrays, context=context, metadata=metadata)


class _Span:
    def __init__(self, recorder: "RunRecorder | NoopRecorder", name: str, component: str,
                 fields: Mapping[str, Any] | None, updates: Mapping[str, Any]):
        self.recorder = recorder
        self.name = name
        self.component = component
        self.fields = fields
        self.updates = dict(updates)
        self.span_id: str | None = None
        self._token = None
        self._start_ns = 0
        self._active = False

    def __enter__(self) -> "_Span":
        if not self.recorder.enabled:
            return self
        try:
            previous = current_context()
            values = dict(self.updates)
            if self.recorder.run_id is not None:
                values.setdefault("run_id", self.recorder.run_id)
            values["span_id"] = _new_id()
            values["parent_span_id"] = previous.span_id
            self.span_id = values["span_id"]
            self._token = _CURRENT.set(previous.update(**values))
            self._start_ns = time.monotonic_ns()
            self._active = True
            try:
                self.recorder._emit_span_start(self.name, self.component, self.fields)
            except Exception as failure:
                self.recorder._note_observability_failure("span enter", failure)
        except Exception as failure:
            if self._token is not None:
                try:
                    _CURRENT.reset(self._token)
                except Exception as restore_failure:
                    self.recorder._note_observability_failure("context restore", restore_failure)
                self._token = None
            self._active = False
            self.span_id = None
            self.recorder._note_observability_failure("span enter", failure)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.recorder.enabled or not self._active:
            return False
        try:
            try:
                outcome = "error" if exc is not None else "ok"
                if exc is not None:
                    try:
                        self.recorder.exception(exc, component=self.component,
                                                fields={"operation": self.name})
                    except Exception as failure:
                        self.recorder._note_observability_failure("span exception", failure)
                duration = max(0, time.monotonic_ns() - self._start_ns)
                try:
                    self.recorder._emit(
                        "span_end", "span.end",
                        level="ERROR" if exc is not None else "DEBUG",
                        component=self.component,
                        fields={"operation": self.name, "duration_ns": duration,
                                "outcome": outcome},
                        critical=exc is not None,
                    )
                except Exception as failure:
                    self.recorder._note_observability_failure("span exit", failure)
            except Exception as failure:
                self.recorder._note_observability_failure("span exit", failure)
        finally:
            if self._token is not None:
                try:
                    _CURRENT.reset(self._token)
                except Exception as failure:
                    self.recorder._note_observability_failure("context restore", failure)
                self._token = None
            self._active = False
        return False

    def event(self, name: str, *, level: str = "INFO", component: str | None = None,
              fields: Mapping[str, Any] | None = None) -> None:
        self.recorder.event(name, level=level, component=component or self.component, fields=fields)


class NoopRecorder:
    """Drop-in capability that performs no field inspection, IO, or threading."""

    enabled = False
    run_id = None
    path = None

    def bind(self, **updates: Any) -> BoundRecorder:
        return BoundRecorder(self, updates)

    def _note_observability_failure(self, *args: Any, **kwargs: Any) -> None:
        return None

    def recent_events(self) -> list[dict[str, Any]]:
        return []

    def span(self, name: str, *, component: str = "runtime",
             fields: Mapping[str, Any] | None = None) -> NoopSpan:
        return NoopSpan()

    def write_resolved_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def metrics(self, *args: Any, **kwargs: Any) -> None:
        return None

    def exception(self, *args: Any, **kwargs: Any) -> None:
        return None

    def capture(self, *args: Any, **kwargs: Any) -> CaptureResult:
        return CaptureResult(False, reason="recorder_disabled")

    def close(self, *, status: str = "succeeded", error: BaseException | None = None) -> RunSummary:
        return RunSummary(None, status, True, 0, 0, 0, 0, 0, 0)


class RunRecorder:
    enabled = True

    @classmethod
    def start(cls, config: Any = None, *, command: str = "unknown",
              metadata: Mapping[str, Any] | None = None,
              parent_run_id: str | None = None) -> "RunRecorder | NoopRecorder":
        if config is None:
            config = default_config()
        logging_config = getattr(config, "observability", config)
        logging_config.validate()
        if logging_config.mode == "off":
            return NoopRecorder()
        return cls(logging_config, command=command, metadata=metadata or {}, parent_run_id=parent_run_id)

    def __init__(self, config: Any, *, command: str, metadata: Mapping[str, Any], parent_run_id: str | None):
        self.config = config
        self.command = command
        self.rank = _rank()
        self.process_id = os.getpid()
        self.run_id = _new_id()
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()
        self._recent: list[dict[str, Any]] = []
        self._recent_lock = threading.Lock()
        self._capture_manager = None
        self._capture_init_lock = threading.Lock()
        self._capture_stats_lock = threading.Lock()
        self._capture_skipped = 0
        self._resolved_config_attempted = False
        self._resolved_config_written = False
        self._resolved_config_lock = threading.Lock()
        self._logger = logging.getLogger("icgs")
        self._owned_handlers: list[logging.Handler] = []
        self._previous_logger_level = self._logger.level
        self._previous_logger_propagate = self._logger.propagate
        self._effective_file_level = (
            "DEBUG" if config.mode in {"debug", "capture"} else config.file_level
        )
        root = Path(config.output_dir).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        _private_root(root)
        self.path = self._make_run_dir(root)
        self.parent_run_id = parent_run_id or metadata.get("parent_run_id")
        config_payload = {"observability": config.to_dict()}
        config_bytes = json_bytes(config_payload)
        startup = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "command": command,
            "started_at": _utc_now(),
            "process_id": self.process_id,
            "rank": self.rank,
            "python": sys.version.split()[0],
            "package": "icgs",
            "platform": platform.platform(),
            "device": metadata.get("device"),
            "code_revision": self._code_revision(),
            "metadata": safe_value(metadata, max_string_chars=config.max_field_string_chars),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "declared_capabilities": {"local": True, "wandb": bool(config.wandb.enabled)},
        }
        _atomic_write(self.path / "run.json", startup)
        self.writer = LocalWriter(self.path, config, process_id=self.process_id, rank=self.rank,
                                  on_warning=self._warning)
        self.writer._thread.start()
        self._attach_handlers()

    @staticmethod
    def _make_run_dir(root: Path) -> Path:
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            path = root / f"{stamp}-{_new_id()}"
            try:
                path.mkdir(mode=0o700, parents=True, exist_ok=False)
                os.chmod(path, 0o700)
                return path
            except FileExistsError:
                continue

    @staticmethod
    def _code_revision() -> dict[str, Any]:
        checkout = Path.cwd() / ".git"
        if not checkout.exists():
            return {"revision": None, "dirty": None, "reason": "not_a_checkout"}
        try:
            revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path.cwd(),
                                     capture_output=True, text=True, timeout=1, check=True).stdout.strip()
            dirty = subprocess.run(["git", "diff", "--quiet", "--exit-code"], cwd=Path.cwd(),
                                   timeout=1).returncode != 0
            return {"revision": revision or None, "dirty": dirty}
        except (OSError, subprocess.SubprocessError):
            return {"revision": None, "dirty": None, "reason": "git_unavailable"}

    def _warning(self, message: str) -> None:
        print(message, file=sys.stderr)

    def _attach_handlers(self) -> None:
        for handler in list(self._logger.handlers):
            if getattr(handler, "_icgs_observability_owned", False):
                self._logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        console = _OwnedConsoleHandler()
        console._icgs_observability_owned = True
        console.setLevel(getattr(logging, self.config.console_level))
        event_handler = _OwnedEventHandler(self)
        event_handler._icgs_observability_owned = True
        event_handler.setLevel(getattr(logging, self._effective_file_level))
        self._logger.addHandler(console)
        self._logger.addHandler(event_handler)
        self._owned_handlers = [console, event_handler]
        self._logger.propagate = False
        self._logger.setLevel(min(self._logger.level or logging.WARNING,
                                  console.level, event_handler.level))

    def _note_observability_failure(self, operation: str, error: BaseException) -> None:
        """Account for a diagnostic failure without raising it into runtime work."""
        try:
            self.writer.note_internal_failure()
        except Exception:
            pass
        try:
            self._warning(
                f"observability {operation} failed: {type(error).__name__}; "
                "continuing without that diagnostic"
            )
        except Exception:
            pass

    def _level_admitted(self, level: str) -> bool:
        return _LEVEL_RANK[level] >= _LEVEL_RANK[self._effective_file_level]

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            self._sequence += 1
            return self._sequence

    def _should_record(self, component: str, context: TraceContext, level: str) -> bool:
        if self.config.trace_components and component not in self.config.trace_components:
            if level not in {"WARNING", "ERROR", "CRITICAL"}:
                return False
        if self.config.trace_episodes and context.episode_id not in self.config.trace_episodes:
            if level not in {"WARNING", "ERROR", "CRITICAL"}:
                return False
        return True

    def _emit(self, kind: str, name: str, *, level: str, component: str,
              fields: Mapping[str, Any] | None, critical: bool) -> None:
        try:
            if self._closed:
                return
            severity = _level(level)
            if not self._level_admitted(severity):
                return
            context = current_context()
            if not self._should_record(component, context, severity):
                return
            try:
                safe_fields = safe_value(fields or {}, max_string_chars=self.config.max_field_string_chars)
                if not isinstance(safe_fields, dict):
                    raise TypeError("observability fields must be a mapping")
            except Exception:
                if not critical:
                    self.writer.note_metric_skipped()
                    return
                safe_fields = {"serialization_error": "fields_rejected"}
            payload = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "timestamp_utc": _utc_now(),
                "timestamp_ns": time.monotonic_ns(),
                "process_id": self.process_id,
                "rank": self.rank,
                "sequence": self._next_sequence(),
                "severity": severity,
                "component": component,
                "event": name,
                "run_id": context.run_id or self.run_id,
                "span_id": context.span_id,
                "parent_span_id": context.parent_span_id,
                "episode_id": context.episode_id,
                "decision_id": context.decision_id,
                "candidate_id": context.candidate_id,
                "branch_id": context.branch_id,
                "transition_id": context.transition_id,
                "boundary_index": context.boundary_index,
                "head_id": context.head_id,
                "origin": context.origin,
                "fields": safe_fields,
            }
            with self._recent_lock:
                self._recent.append(payload)
                if len(self._recent) > self.config.recent_event_capacity:
                    del self._recent[:len(self._recent) - self.config.recent_event_capacity]
            self.writer.enqueue("metric" if kind == "metric" else "event", payload, critical=critical)
        except Exception as failure:
            self._note_observability_failure("emit", failure)

    def _emit_span_start(self, operation: str, component: str,
                         fields: Mapping[str, Any] | None) -> None:
        """Admit the level before touching caller-supplied span fields."""
        if not self._level_admitted("DEBUG"):
            return
        span_fields: dict[str, Any] = {"operation": operation}
        if fields:
            span_fields.update(fields)
        self._emit("span_start", "span.start", level="DEBUG", component=component,
                    fields=span_fields, critical=False)

    def _runtime_config_document(self, runtime_config: Any) -> dict[str, Any]:
        if runtime_config is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "config": {"observability": self.config.to_dict()},
                "config_sha256": hashlib.sha256(
                    json_bytes({"observability": self.config.to_dict()})
                ).hexdigest(),
            }
        if callable(getattr(runtime_config, "resolved_config", None)):
            document = runtime_config.resolved_config()
        elif callable(getattr(runtime_config, "to_dict", None)):
            document = {"schema_version": SCHEMA_VERSION,
                        "config": runtime_config.to_dict()}
        elif is_dataclass(runtime_config):
            document = {"schema_version": SCHEMA_VERSION,
                        "config": asdict(runtime_config)}
        elif isinstance(runtime_config, Mapping):
            document = dict(runtime_config)
        else:
            raise TypeError("resolved runtime config must be a mapping or typed config")
        if not isinstance(document, Mapping):
            raise TypeError("resolved runtime config document must be a mapping")
        config_value = document.get("config", document)
        if not isinstance(config_value, Mapping):
            raise TypeError("resolved runtime config must contain a mapping config")
        config_value = dict(config_value)
        digest = document.get("config_sha256")
        if not isinstance(digest, str) or not digest:
            digest = hashlib.sha256(json_bytes(config_value)).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "config": config_value,
            "config_sha256": digest,
        }

    def write_resolved_config(
        self,
        runtime_config: Any = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        unavailable: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the outer-boundary resolved record exactly once.

        The startup manifest is immutable.  An outer boundary calls this once,
        after it has resolved its runtime configuration and known identities but
        before model or controller work.  A later call is a programming error.
        """
        with self._resolved_config_lock:
            if self._resolved_config_attempted:
                raise RuntimeError("resolved runtime config was already attempted")
            self._resolved_config_attempted = True
            document = self._runtime_config_document(runtime_config)
            details = dict(metadata or {})
            identities = details.pop("identities", {})
            reasons = dict(unavailable or details.pop("unavailable", {}) or {})
            if runtime_config is None:
                reasons.setdefault("runtime_config", "outer boundary did not provide a resolved config")
            if not isinstance(identities, Mapping) or not isinstance(reasons, Mapping):
                raise TypeError("resolved identities and unavailable reasons must be mappings")
            document.update({
                "identities": dict(identities),
                "unavailable": dict(reasons),
            "metadata": details,
            })
            safe_document = safe_value(
                document, max_depth=32,
                max_string_chars=self.config.max_field_string_chars,
            )
            _atomic_write(self.path / "resolved_config.json", safe_document)
            self._resolved_config_written = True

    def recent_events(self) -> list[dict[str, Any]]:
        """Return a bounded snapshot for metadata-only diagnostic capture."""
        with self._recent_lock:
            return [dict(event) for event in self._recent]

    def bind(self, **updates: Any) -> BoundRecorder:
        return BoundRecorder(self, updates)

    def event(self, name: str, *, level: str = "INFO", component: str = "runtime",
              fields: Mapping[str, Any] | None = None) -> None:
        self._emit("event", name, level=level, component=component, fields=fields,
                   critical=_level(level) in {"WARNING", "ERROR", "CRITICAL"})

    def span(self, name: str, *, component: str = "runtime",
             fields: Mapping[str, Any] | None = None) -> _Span:
        return _Span(self, name, component, fields, {})

    def metrics(self, values: Mapping[str, Any], *, axis: str, step: int,
                component: str, units: Mapping[str, str] | None = None) -> None:
        if self._closed:
            return
        if not isinstance(values, Mapping) or not values:
            return
        try:
            step_int = int(step)
        except (TypeError, ValueError):
            self.event("metrics.invalid", level="ERROR", component=component,
                       fields={"reason": "step_not_integer"})
            return
        if step_int % self.config.metric_every_steps:
            self.writer.note_metric_skipped()
            return
        encoded = {}
        nonfinite = False
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                self.event("metrics.invalid", level="ERROR", component=component,
                           fields={"reason": "metric_name_not_string"})
                return
            try:
                encoded[name] = metric_value(value)
            except TypeError:
                self.event("metrics.invalid", level="ERROR", component=component,
                           fields={"metric": name, "reason": "metric_not_real"})
                return
            nonfinite = nonfinite or encoded[name]["nonfinite"]
        self._emit("metric", "metrics.sample", level="ERROR" if nonfinite else "INFO",
                   component=component,
                   fields={"axis": axis, "step": step_int, "values": encoded,
                           "units": units or {}, "nonfinite": nonfinite},
                   critical=nonfinite)

    def exception(self, error: BaseException, *, component: str,
                  fields: Mapping[str, Any] | None = None) -> None:
        try:
            try:
                payload = exception_payload(
                    error, max_string_chars=self.config.max_field_string_chars
                )
            except Exception as failure:
                self._note_observability_failure("exception serialization", failure)
                error_type = type(error)
                payload = {
                    "type": (
                        f"{getattr(error_type, '__module__', 'builtins')}."
                        f"{getattr(error_type, '__qualname__', 'BaseException')}"
                    ),
                    "message": "<unavailable>",
                    "serialization_error": type(failure).__name__,
                }
            details = {"exception": payload}
            if fields:
                details.update(fields)
            self._emit("exception", "exception", level="ERROR", component=component,
                       fields=details, critical=True)
        except Exception as failure:
            self._note_observability_failure("exception record", failure)

    def capture(self, name: str, arrays: Mapping[str, Any], *,
                context: Mapping[str, Any] | None = None,
                metadata: Mapping[str, Any] | None = None) -> CaptureResult:
        try:
            with self._capture_init_lock:
                if self._capture_manager is None:
                    from .capture import CaptureManager
                    self._capture_manager = CaptureManager(self.path, self.config, rank=self.rank)
            context_value = context or current_context().as_dict()
            if self.config.mode == "capture" and not self.config.capture_arrays:
                result = self._capture_manager.capture_metadata(
                    name, context=context_value, metadata=metadata,
                    recent_events=self.recent_events(),
                )
            else:
                result = self._capture_manager.capture(name, arrays, context=context_value, metadata=metadata)
        except Exception as failure:
            self._note_observability_failure("capture", failure)
            result = CaptureResult(False, reason=f"capture_failed:{type(failure).__name__}")
        if result.captured:
            self._emit("event", "capture.created", level="INFO", component="capture",
                       fields={"capture_id": result.capture_id, "bytes": result.bytes_written}, critical=False)
        else:
            with self._capture_stats_lock:
                self._capture_skipped += 1
            self._emit("event", "capture.skipped", level="WARNING", component="capture",
                       fields={"reason": result.reason}, critical=True)
        return result

    def close(self, *, status: str = "succeeded", error: BaseException | None = None) -> RunSummary:
        with self._close_lock:
            if self._closed:
                return self._summary
            if status not in {"succeeded", "failed", "interrupted"}:
                raise ValueError("run status must be succeeded, failed, or interrupted")
            if error is not None and status == "succeeded":
                status = "failed"
            if not self._resolved_config_written:
                if self._resolved_config_attempted:
                    resolved_config_failed = True
                else:
                    try:
                        self.write_resolved_config(
                            metadata={"command": self.command},
                            unavailable={
                                "runtime_config": "outer boundary did not provide a resolved config",
                            },
                        )
                    except Exception as failure:
                        self._note_observability_failure("resolved config", failure)
                        resolved_config_failed = True
                    else:
                        resolved_config_failed = False
            else:
                resolved_config_failed = False
            if error is not None:
                try:
                    self.exception(error, component="run", fields={"status": status})
                except Exception as failure:
                    self._note_observability_failure("run exception", failure)
            try:
                self._emit("event", "run.end", level="ERROR" if status == "failed" else "INFO",
                           component="run", fields={"status": status}, critical=True)
            except Exception as failure:
                self._note_observability_failure("run end", failure)
            try:
                writer_complete = self.writer.close(self.config.shutdown_timeout_s)
            except Exception as failure:
                self._note_observability_failure("writer close", failure)
                writer_complete = False
            try:
                stats = self.writer.stats
            except Exception as failure:
                self._note_observability_failure("writer statistics", failure)
                stats = {
                    "events": 0, "metrics": 0, "dropped": 0,
                    "dropped_critical": 0, "metrics_skipped": 0,
                    "sink_errors": 1,
                }
            skipped_captures = self._capture_skipped
            try:
                with self._capture_stats_lock:
                    skipped_captures = self._capture_skipped
            except Exception as failure:
                self._note_observability_failure("capture statistics", failure)
            capture_count = 0
            capture_bytes = 0
            if self._capture_manager is not None:
                try:
                    capture_count = int(getattr(self._capture_manager, "count", 0))
                    capture_bytes = int(getattr(self._capture_manager, "total_bytes", 0))
                except Exception as failure:
                    self._note_observability_failure("capture totals", failure)
                    capture_count = capture_bytes = 0
            logs_complete = (writer_complete and stats["sink_errors"] == 0
                             and stats["dropped"] == 0
                             and stats["metrics_skipped"] == 0
                             and skipped_captures == 0
                             and not resolved_config_failed)
            try:
                error_payload = (
                    exception_payload(error, max_string_chars=self.config.max_field_string_chars)
                    if error else None
                )
            except Exception as failure:
                self._note_observability_failure("final exception serialization", failure)
                error_payload = {
                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "message": "<unavailable>",
                    "serialization_error": type(failure).__name__,
                } if error else None
            summary_payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": status,
                "logs_complete": logs_complete,
                "event_count": stats["events"],
                "metric_count": stats["metrics"],
                "captures": {"count": capture_count,
                             "bytes": capture_bytes,
                             "skipped": skipped_captures},
                "dropped": {"count": stats["dropped"], "critical": stats["dropped_critical"],
                            "metrics_skipped": stats["metrics_skipped"]},
                "sink_errors": stats["sink_errors"],
                "finished_at": _utc_now(),
                "error": error_payload,
            }
            summary_write_failed = False
            try:
                _atomic_write(self.path / "summary.json", summary_payload)
            except Exception as exc:
                logs_complete = False
                summary_write_failed = True
                self._warning(f"observability summary write failed: {type(exc).__name__}")
            if summary_write_failed:
                summary_payload["logs_complete"] = False
            try:
                self._detach_handlers()
            except Exception as failure:
                self._note_observability_failure("handler cleanup", failure)
            self._closed = True
            self._summary = RunSummary(self.run_id, status, logs_complete, stats["events"],
                                        stats["metrics"], summary_payload["captures"]["count"],
                                        stats["dropped"], skipped_captures,
                                        stats["sink_errors"] + int(summary_write_failed),
                                        str(self.path), error_payload)
            return self._summary

    def _detach_handlers(self) -> None:
        for handler in self._owned_handlers:
            self._logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        self._owned_handlers = []
        self._logger.setLevel(self._previous_logger_level)
        self._logger.propagate = self._previous_logger_propagate


__all__ = ["RunRecorder", "NoopRecorder", "BoundRecorder"]
