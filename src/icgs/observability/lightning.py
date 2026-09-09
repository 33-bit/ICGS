"""Optional Lightning logger bridge; the recorder remains the local authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from lightning.pytorch.loggers.logger import Logger as _LightningLogger
except ImportError:  # pragma: no cover - exercised on inference-only installs
    _LightningLogger = object


class RecorderLightningLogger(_LightningLogger):
    def __init__(self, recorder, remote=None):
        if _LightningLogger is not object:
            super().__init__()
        self.recorder = recorder
        self.remote = remote
        self._name = "icgs-local"
        self._version = recorder.run_id or "disabled"

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def experiment(self):
        return self

    def log_hyperparams(self, params: Any) -> None:
        fields = dict(params) if isinstance(params, Mapping) else {"available": False}
        self.recorder.event("training.config", component="training", fields=fields)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self.recorder.metrics(metrics, axis="optimizer_step", step=0 if step is None else step,
                              component="training")
        if self.remote is not None:
            self.remote.log(metrics, step=step)

    def save(self) -> None:
        return None

    def finalize(self, status: str) -> None:
        self.recorder.event("training.logger.finalize", component="training",
                            fields={"status": status})


__all__ = ["RecorderLightningLogger"]
