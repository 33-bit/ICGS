"""Logging-only configuration adapter backed by the packaged method defaults."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

from icgs.configuration.method import MethodConfig, ObservabilityConfig
from icgs.configuration.method_schema import strict_json


def default_config() -> ObservabilityConfig:
    """Return a fresh central logging configuration."""
    return MethodConfig().observability


def _merge_known(base: dict[str, Any], update: Mapping[str, Any], prefix: str = "") -> None:
    if not isinstance(update, Mapping):
        raise ValueError(f"{prefix or 'observability'} must be an object")
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge_known(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def from_mapping(update: Mapping[str, Any] | None = None) -> ObservabilityConfig:
    base = default_config().to_dict()
    if update is not None:
        _merge_known(base, update)
    return MethodConfig.from_dict({"observability": base}).observability


def load_config(path: str | Path | None = None, *, overrides: Mapping[str, Any] | None = None,
                output_dir: str | Path | None = None, log_level: str | None = None,
                trace_mode: str | None = None) -> ObservabilityConfig:
    """Load an explicit logging envelope and apply only explicit CLI overrides."""
    base = default_config().to_dict()
    directory = Path.cwd()
    if path is not None:
        file_path = Path(path).expanduser().resolve()
        directory = file_path.parent
        try:
            document = strict_json(file_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid logging configuration file: {file_path}") from exc
        if not isinstance(document, Mapping) or set(document) != {"schema_version", "observability"}:
            raise ValueError("logging configuration requires schema_version/observability envelope")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("unsupported logging configuration schema")
        _merge_known(base, document["observability"], "observability.")
    for key, value in (overrides or {}).items():
        if key.startswith("observability."):
            key = key[len("observability."):]
        if not key:
            raise ValueError("logging override key must be nonempty")
        update: Any = value
        for bit in reversed(key.split(".")):
            update = {bit: update}
        _merge_known(base, update)
    if output_dir is not None:
        base["output_dir"] = str(output_dir)
    if log_level is not None:
        base["console_level"] = log_level
        base["file_level"] = log_level
    if trace_mode is not None:
        base["mode"] = trace_mode
    config = from_mapping(base)
    if path is not None and output_dir is None and not Path(config.output_dir).is_absolute():
        config = replace(config, output_dir=str((directory / config.output_dir).resolve()))
    elif output_dir is not None and not Path(config.output_dir).is_absolute():
        config = replace(config, output_dir=str((Path.cwd() / config.output_dir).resolve()))
    return config.validate()


__all__ = ["default_config", "from_mapping", "load_config"]
