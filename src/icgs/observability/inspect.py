"""Bounded, offline-only run inspection."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .records import SCHEMA_VERSION


_MAX_JSONL_LINE_BYTES = 1024 * 1024


class _ReadLimitError(ValueError):
    """Inspection exceeded one of its explicit bounded-read limits."""


class _DirectoryLimitError(_ReadLimitError):
    """A directory contained more entries than the inspection admission bound."""


class _ReadBudget:
    def __init__(self, limit: int):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("observability read limit must be positive")
        self.limit = limit
        self.consumed = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.consumed

    def consume(self, amount: int) -> None:
        if amount < 0 or self.consumed + amount > self.limit:
            raise _ReadLimitError("observability read limit exceeded")
        self.consumed += amount


def _validate_limit(value: int, message: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(message)


def _bounded_entries(directory: Path, *, limit: int) -> tuple[list[Path], bool]:
    """Collect at most ``limit`` entries and inspect one more for overflow."""
    _validate_limit(limit, "observability directory entry limit must be positive")
    entries: list[Path] = []
    overflow = False
    for index, path in enumerate(directory.iterdir()):
        if index >= limit:
            overflow = True
            break
        entries.append(path)
    return entries, overflow


def _read_json(path: Path, *, max_bytes: int, budget: _ReadBudget | None = None) -> dict[str, Any]:
    _validate_limit(max_bytes, "observability JSON read limit must be positive")
    budget = budget or _ReadBudget(max_bytes)
    try:
        size = path.stat().st_size
        if size > max_bytes or size > budget.remaining:
            raise _ReadLimitError(f"observability JSON read limit exceeded: {path}")
        with path.open("rb") as stream:
            raw = stream.read(min(max_bytes + 1, budget.remaining + 1))
        budget.consume(len(raw))
        if len(raw) > max_bytes:
            raise _ReadLimitError(f"observability JSON read limit exceeded: {path}")
        value = json.loads(raw.decode("utf-8"))
    except _ReadLimitError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid observability JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"observability JSON must be an object: {path}")
    return value


def _iter_jsonl(path: Path, *, max_bytes: int, max_records: int,
                max_line_bytes: int, warnings: list[str], budget: _ReadBudget):
    _validate_limit(max_bytes, "observability JSONL read limit must be positive")
    _validate_limit(max_records, "observability record limit must be positive")
    total = 0
    count = 0
    with path.open("rb") as stream:
        while True:
            remaining_file = max_bytes - total
            if remaining_file < 1 or budget.remaining < 1:
                # An empty file may end exactly at the budget boundary; a
                # nonempty next byte must be reported as a bounded-read error.
                raw = stream.readline(1)
            else:
                raw = stream.readline(min(max_line_bytes + 1,
                                          remaining_file + 1,
                                          budget.remaining + 1))
            if not raw:
                break
            budget.consume(len(raw))
            total += len(raw)
            if total > max_bytes:
                raise ValueError(f"observability JSONL line limit exceeded: {path}")
            if len(raw) > max_line_bytes:
                raise _ReadLimitError(f"observability JSONL line limit exceeded: {path}")
            if count >= max_records:
                raise _ReadLimitError("observability record limit exceeded")
            if not raw.endswith(b"\n"):
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    warnings.append(f"ignored partial final JSONL line: {path.name}")
                    return
            else:
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"interior observability JSONL corruption: {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"observability JSONL record must be an object: {path}")
            count += 1
            yield value


def read_records(run_dir: str | Path, *, kind: str = "events", max_bytes: int = 64 * 1024 * 1024,
                 max_records: int = 200_000,
                 _budget: _ReadBudget | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(run_dir).expanduser().resolve()
    if kind not in {"events", "metrics"}:
        raise ValueError("observability record kind must be events or metrics")
    _validate_limit(max_bytes, "observability read limit must be positive")
    _validate_limit(max_records, "observability record limit must be positive")
    budget = _budget or _ReadBudget(max_bytes)
    directory = root / kind
    if not directory.is_dir():
        return [], [f"missing {kind} directory"]
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    entries, overflow = _bounded_entries(directory, limit=max_records)
    if overflow:
        raise _DirectoryLimitError("observability directory entry limit exceeded")
    paths = sorted(path for path in entries if path.name.endswith(".jsonl"))
    for path in paths:
        file_size = path.stat().st_size
        if file_size > max_bytes or file_size > budget.remaining:
            raise _ReadLimitError("observability read limit exceeded")
        records.extend(_iter_jsonl(
            path, max_bytes=file_size or 1, max_records=max_records - len(records),
            max_line_bytes=min(max_bytes, _MAX_JSONL_LINE_BYTES), warnings=warnings,
            budget=budget,
        ))
    return records, warnings


def _reject_symlink_components(base: Path, parts: tuple[str, ...], *, label: str) -> None:
    current = base
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} references a symlink")


def _hash_archive(path: Path, *, budget: _ReadBudget) -> tuple[int, str]:
    size = path.stat().st_size
    if size > budget.remaining:
        raise _ReadLimitError("observability read limit exceeded while hashing capture archive")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                raise ValueError("capture archive was truncated while hashing")
            budget.consume(len(chunk))
            digest.update(chunk)
            remaining -= len(chunk)
    if path.stat().st_size != size:
        raise ValueError("capture archive changed while hashing")
    return size, digest.hexdigest()


def load_capture_manifest(run_dir: str | Path, reference: str,
                          *, max_bytes: int = 64 * 1024 * 1024,
                          _budget: _ReadBudget | None = None) -> dict[str, Any]:
    """Read one capture manifest while refusing path traversal."""
    _validate_limit(max_bytes, "observability capture read limit must be positive")
    budget = _budget or _ReadBudget(max_bytes)
    root = Path(run_dir).expanduser().resolve()
    captures = root / "captures"
    if captures.is_symlink():
        raise ValueError("capture directory is a symlink")
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError("capture reference must be a relative capture ID")
    relative = Path(reference)
    if relative.parts and relative.parts[0] == "captures":
        relative = Path(*relative.parts[1:])
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("capture reference escapes run directory")
    _reject_symlink_components(captures, relative.parts, label="capture reference")
    capture_dir = captures.joinpath(*relative.parts)
    if not capture_dir.is_dir():
        raise ValueError("capture reference does not name a directory")
    manifest_path = capture_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("capture manifest is a symlink")
    manifest = _read_json(manifest_path, max_bytes=max_bytes, budget=budget)
    # Check the referenced object before deeper schema validation so a
    # malformed manifest cannot turn an unsafe in-run symlink into a benign
    # schema error.
    raw_archive = manifest.get("archive")
    if (isinstance(raw_archive, str) and not Path(raw_archive).is_absolute()
            and not any(part in {".", ".."} for part in Path(raw_archive).parts)):
        raw_archive_parts = Path(raw_archive).parts
        _reject_symlink_components(capture_dir, raw_archive_parts, label="capture archive")
    if (type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("capture manifest has an unsupported schema version")
    capture_id = manifest.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id or capture_id != capture_dir.name:
        raise ValueError("capture manifest identity is invalid")
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("capture manifest name is invalid")
    if "arrays" not in manifest or not isinstance(manifest["arrays"], Mapping):
        raise ValueError("capture manifest arrays field is invalid")
    arrays = manifest["arrays"]
    for key, descriptor in arrays.items():
        if (not isinstance(key, str) or not key or len(key) > 256
                or "/" in key or "\\" in key):
            raise ValueError("capture manifest array key is invalid")
        if not isinstance(descriptor, Mapping):
            raise ValueError("capture manifest array descriptor is invalid")
        shape = descriptor.get("shape")
        dtype = descriptor.get("dtype")
        nbytes = descriptor.get("nbytes")
        if (not isinstance(shape, list)
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
                or not isinstance(dtype, str) or not dtype
                or isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0):
            raise ValueError("capture manifest array metadata is invalid")
    if "archive" not in manifest:
        raise ValueError("capture manifest archive field is missing")
    archive = manifest["archive"]
    if archive is None:
        if arrays:
            raise ValueError("metadata-only capture cannot contain arrays")
        return manifest
    if (not isinstance(archive, str) or Path(archive).is_absolute()
            or any(part in {".", ".."} for part in Path(archive).parts)):
        raise ValueError("capture archive reference is unsafe")
    archive_parts = Path(archive).parts
    _reject_symlink_components(capture_dir, archive_parts, label="capture archive")
    archive_path = capture_dir.joinpath(*archive_parts)
    if not archive_path.is_file():
        raise ValueError("capture archive escapes capture directory")
    archive_size = manifest.get("bytes")
    archive_sha256 = manifest.get("archive_sha256")
    if (isinstance(archive_size, bool) or not isinstance(archive_size, int) or archive_size < 0
            or not isinstance(archive_sha256, str) or len(archive_sha256) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in archive_sha256)):
        raise ValueError("capture manifest archive size or SHA256 is invalid")
    actual_size, actual_sha256 = _hash_archive(archive_path, budget=budget)
    if actual_size != archive_size:
        raise ValueError("capture archive size does not match manifest")
    if actual_sha256 != archive_sha256.lower():
        raise ValueError("capture archive SHA256 does not match manifest")
    return manifest


def inspect_run(run_dir: str | Path, *, max_bytes: int = 64 * 1024 * 1024,
                max_records: int = 200_000) -> dict[str, Any]:
    _validate_limit(max_bytes, "observability read limit must be positive")
    _validate_limit(max_records, "observability record limit must be positive")
    budget = _ReadBudget(max_bytes)
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    run = _read_json(root / "run.json", max_bytes=max_bytes, budget=budget)
    summary_path = root / "summary.json"
    summary = (_read_json(summary_path, max_bytes=max_bytes, budget=budget)
               if summary_path.exists() else None)
    events, event_warnings = read_records(
        root, kind="events", max_bytes=max_bytes, max_records=max_records, _budget=budget
    )
    metrics, metric_warnings = read_records(
        root, kind="metrics", max_bytes=max_bytes, max_records=max_records, _budget=budget
    )
    warnings = event_warnings + metric_warnings
    errors = [record for record in events if record.get("severity") in {"ERROR", "CRITICAL"}
              or (isinstance(record.get("event"), str) and record["event"].endswith(".error"))]
    missing = []
    if summary is None:
        missing.append("summary.json")
    if not events:
        missing.append("events")
    if summary is not None:
        if summary.get("event_count") != len(events):
            missing.append("events:summary-count-mismatch")
        if summary.get("metric_count") != len(metrics):
            missing.append("metrics:summary-count-mismatch")
        capture_summary = summary.get("captures", {})
        if not isinstance(capture_summary, Mapping):
            expected_captures = -1
        else:
            expected_captures = capture_summary.get("count", 0)
        try:
            expected_captures = int(expected_captures)
        except (TypeError, ValueError):
            expected_captures = -1
        captures_dir = root / "captures"
        if expected_captures < 0:
            missing.append("captures:summary-invalid")
        if expected_captures > 0 or captures_dir.exists() or captures_dir.is_symlink():
            valid_captures = 0
            if not captures_dir.is_dir() or captures_dir.is_symlink():
                missing.append("captures")
            else:
                children, overflow = _bounded_entries(captures_dir, limit=max_records)
                if overflow:
                    missing.append("captures:read-limit")
                valid_captures = 0
                invalid_capture = False
                for child in sorted(children):
                    if child.is_symlink():
                        warnings.append(f"invalid capture evidence {child.name}: symlink")
                        invalid_capture = True
                        continue
                    if not child.is_dir():
                        warnings.append(f"invalid capture evidence {child.name}: not a directory")
                        invalid_capture = True
                        continue
                    try:
                        load_capture_manifest(
                            root, child.name, max_bytes=max_bytes, _budget=budget
                        )
                    except _ReadLimitError:
                        raise
                    except ValueError as exc:
                        warnings.append(f"invalid capture evidence {child.name}: {exc}")
                        invalid_capture = True
                        continue
                    valid_captures += 1
                if invalid_capture:
                    missing.append("captures:invalid-entry")
                if (valid_captures != expected_captures
                        or len(children) != expected_captures):
                    missing.append("captures:summary-count-mismatch")
        dropped = summary.get("dropped", {})
        if isinstance(dropped, Mapping):
            for key in ("count", "critical", "metrics_skipped"):
                try:
                    if int(dropped.get(key, 0)) > 0:
                        missing.append(f"dropped:{key}")
                except (TypeError, ValueError):
                    missing.append(f"dropped:{key}")
        try:
            if int(summary.get("sink_errors", 0)) > 0:
                missing.append("sink_errors")
        except (TypeError, ValueError):
            missing.append("sink_errors")
    if event_warnings or metric_warnings:
        missing.append("partial_record_evidence")
    logs_complete = False if summary is None else bool(summary.get("logs_complete", False))
    if missing or event_warnings or metric_warnings:
        logs_complete = False
    return {
        "run": run,
        "summary": summary,
        "status": None if summary is None else summary.get("status"),
        "logs_complete": logs_complete,
        "counts": {"events": len(events), "metrics": len(metrics), "errors": len(errors)},
        "drops": {} if summary is None else summary.get("dropped", {}),
        "captures": {} if summary is None else summary.get("captures", {}),
        "missing_evidence": missing,
        "warnings": warnings,
    }


def _matches(record: Mapping[str, Any], filters: Mapping[str, str | None]) -> bool:
    for key, value in filters.items():
        if value is not None and str(record.get(key)) != str(value):
            return False
    return True


def trace_run(run_dir: str | Path, *, episode_id: str | None = None,
              decision_id: str | None = None, candidate_id: str | None = None,
              max_bytes: int = 64 * 1024 * 1024, max_records: int = 200_000) -> dict[str, Any]:
    records, warnings = read_records(run_dir, kind="events", max_bytes=max_bytes, max_records=max_records)
    filters = {"episode_id": episode_id, "decision_id": decision_id, "candidate_id": candidate_id}
    selected = [record for record in records if _matches(record, filters)]
    starts = {record.get("span_id"): record for record in selected
              if record.get("kind") == "span_start" and record.get("span_id")}
    ends = {record.get("span_id"): record for record in selected
            if record.get("kind") == "span_end" and record.get("span_id")}
    spans = []
    for span_id, start in starts.items():
        end = ends.get(span_id)
        spans.append({"span_id": span_id, "start": start, "end": end,
                      "unfinished": end is None})
    return {"events": selected, "spans": spans,
            "unfinished_spans": [item["span_id"] for item in spans if item["unfinished"]],
            "warnings": warnings}


__all__ = ["read_records", "load_capture_manifest", "inspect_run", "trace_run"]
