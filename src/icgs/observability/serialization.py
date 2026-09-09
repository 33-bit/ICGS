"""Bounded, non-executable serialization for observability records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import numbers
from traceback import extract_tb
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SENSITIVE_PARTS = ("password", "token", "secret", "api_key", "authorization", "credential")
_REDACTED = "<redacted>"


def _sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return any(part in lowered for part in _SENSITIVE_PARTS)


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.query:
        return value
    query = []
    for key, _ in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, _REDACTED))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _array_metadata(value: Any) -> dict[str, Any] | None:
    """Describe known numeric containers without reading their values."""
    module = type(value).__module__
    if module.startswith("numpy") and hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"type": "numeric_array", "shape": list(value.shape), "dtype": str(value.dtype)}
    if module.startswith("torch") and hasattr(value, "shape") and hasattr(value, "dtype"):
        result: dict[str, Any] = {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
        for name in ("device", "requires_grad"):
            if hasattr(value, name):
                result[name] = str(getattr(value, name)) if name == "device" else bool(getattr(value, name))
        return result
    return None


def safe_value(value: Any, *, max_depth: int = 8, max_string_chars: int = 2048,
               _depth: int = 0, _key: str | None = None) -> Any:
    """Return only bounded JSON values; reject arbitrary objects and ``repr``."""
    if _key is not None and _sensitive_key(_key):
        return _REDACTED
    if _depth > max_depth:
        raise ValueError("observability field nesting exceeds the configured bound")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observability fields must contain finite numbers")
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("observability fields must contain finite numbers")
        return value
    if isinstance(value, str):
        value = _safe_url(value)
        if len(value) > max_string_chars:
            return value[:max_string_chars] + "..."
        return value
    metadata = _array_metadata(value)
    if metadata is not None:
        return metadata
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("observability mapping keys must be strings")
            result[key] = safe_value(item, max_depth=max_depth,
                                     max_string_chars=max_string_chars,
                                     _depth=_depth + 1, _key=key)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [safe_value(item, max_depth=max_depth, max_string_chars=max_string_chars,
                           _depth=_depth + 1) for item in value]
    raise TypeError(f"unsupported observability field type: {type(value).__name__}")


def metric_value(value: Any) -> dict[str, Any]:
    """Encode one metric while making nonfinite values explicit and invalid."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("metrics must be real numbers")
    number = float(value)
    if not math.isfinite(number):
        return {"value": None, "nonfinite": True}
    return {"value": number, "nonfinite": False}


def exception_payload(error: BaseException, *, max_string_chars: int = 2048,
                      max_depth: int = 8) -> dict[str, Any]:
    """Serialize exception causes and traceback locations, never frame locals."""
    seen: set[int] = set()

    def one(current: BaseException, depth: int) -> dict[str, Any]:
        if id(current) in seen or depth >= max_depth:
            return {"type": "<cycle-or-depth-limit>"}
        seen.add(id(current))
        frames = [
            {"filename": str(frame.filename), "line": int(frame.lineno), "name": str(frame.name)}
            for frame in extract_tb(current.__traceback__)[-32:]
        ]
        error_type = type(current)
        type_name = (
            f"{getattr(error_type, '__module__', 'builtins')}."
            f"{getattr(error_type, '__qualname__', 'BaseException')}"
        )
        result: dict[str, Any] = {
            "type": type_name,
            "traceback": frames,
        }
        try:
            message = str(current)
        except Exception as formatting_error:
            result["message"] = "<unavailable>"
            result["message_error"] = type(formatting_error).__name__
        else:
            result["message"] = safe_value(message, max_string_chars=max_string_chars)
        cause = current.__cause__ or current.__context__
        if cause is not None:
            result["cause"] = one(cause, depth + 1)
        return result

    return one(error, 0)


def json_bytes(value: Any, *, max_bytes: int | None = None) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if max_bytes is not None and len(payload) > max_bytes:
        raise ValueError("observability record exceeds max_record_bytes")
    return payload


__all__ = ["safe_value", "metric_value", "exception_payload", "json_bytes"]
