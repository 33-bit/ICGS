"""Explicit CLI boundary setup for the local recorder."""

from __future__ import annotations

from dataclasses import replace

from icgs.observability.config import load_config
from icgs.observability.recorder import NoopRecorder, RunRecorder


def start_cli_run(args, command: str, *, metadata: dict | None = None,
                  enable_wandb: bool = False):
    overrides = {}
    logging_path = getattr(args, "logging_config", None)
    log_dir = getattr(args, "log_dir", None)
    log_level = getattr(args, "log_level", None)
    trace_mode = getattr(args, "trace_mode", None)
    config = load_config(logging_path, output_dir=log_dir, log_level=log_level,
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
