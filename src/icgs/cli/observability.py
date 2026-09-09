"""Explicit CLI boundary setup for the local recorder."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

from icgs.observability.config import load_config
from icgs.observability.recorder import NoopRecorder, RunRecorder


_DOTENV_MAX_BYTES = 64 * 1024
_DOTENV_KEYS = {
    "ICGS_WANDB_ENABLED", "ICGS_WANDB_MODE", "ICGS_WANDB_PROJECT",
    "ICGS_WANDB_ENTITY", "ICGS_WANDB_TAGS", "ICGS_WANDB_SEND_CONFIG",
    "ICGS_WANDB_UPLOAD_ARTIFACTS", "WANDB_API_KEY",
}


def _load_project_dotenv() -> None:
    """Load the limited user-local CLI settings without executing shell syntax."""
    path = Path.cwd() / ".env"
    try:
        if not path.is_file() or path.stat().st_size > _DOTENV_MAX_BYTES:
            return
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in _DOTENV_KEYS:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _dotenv_observability_overrides() -> dict:
    _load_project_dotenv()
    mapping = {
        "ICGS_WANDB_ENABLED": ("enabled", lambda value: _parse_bool(value, "ICGS_WANDB_ENABLED")),
        "ICGS_WANDB_MODE": ("mode", str),
        "ICGS_WANDB_PROJECT": ("project", str),
        "ICGS_WANDB_ENTITY": ("entity", str),
        "ICGS_WANDB_TAGS": ("tags", lambda value: tuple(tag.strip() for tag in value.split(",") if tag.strip())),
        "ICGS_WANDB_SEND_CONFIG": ("send_config", lambda value: _parse_bool(value, "ICGS_WANDB_SEND_CONFIG")),
        "ICGS_WANDB_UPLOAD_ARTIFACTS": ("upload_artifacts", lambda value: _parse_bool(value, "ICGS_WANDB_UPLOAD_ARTIFACTS")),
    }
    wandb = {target: convert(os.environ[source]) for source, (target, convert) in mapping.items()
             if source in os.environ}
    return {"wandb": wandb} if wandb else {}


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be true or false")


def start_cli_run(args, command: str, *, metadata: dict | None = None,
                  enable_wandb: bool = False):
    overrides = {}
    logging_path = getattr(args, "logging_config", None)
    log_dir = getattr(args, "log_dir", None)
    log_level = getattr(args, "log_level", None)
    trace_mode = getattr(args, "trace_mode", None)
    config = load_config(logging_path, base_overrides=_dotenv_observability_overrides(),
                         output_dir=log_dir, log_level=log_level,
                         trace_mode=trace_mode)
    if enable_wandb and not config.wandb.enabled:
        config = replace(config, wandb=replace(config.wandb, enabled=True))
    details = dict(metadata or {})
    details.setdefault("command", command)
    recorder = RunRecorder.start(config, command=command, metadata=details)
    return recorder


def close_cli_run(recorder, *, status: str, error: BaseException | None = None):
    return recorder.close(status=status, error=error)


__all__ = ["start_cli_run", "close_cli_run"]
