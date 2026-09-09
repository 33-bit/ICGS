"""Optional, local-first observability for the ICGS outer boundaries.

The package deliberately has no import-time handlers, files, threads, model
imports, simulator imports, or network clients.  Library callers use the
no-op recorder unless they explicitly start a run.
"""

from .config import default_config, load_config
from .context import TraceContext, bind_context, current_context
from .recorder import NoopRecorder, RunRecorder

__all__ = [
    "TraceContext",
    "RunRecorder",
    "NoopRecorder",
    "bind_context",
    "current_context",
    "default_config",
    "load_config",
]
