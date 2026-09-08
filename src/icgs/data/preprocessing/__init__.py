"""Public native preprocessing functions with lazy optional-dependency loading."""

from importlib import import_module


_NATIVE_EXPORTS = (
    "save_sample",
    "sample_to_cond_demo",
    "sample_to_live",
    "subsample_traj",
    "extract_waypoints",
    "subsample_pcd",
    "remove_statistical_outliers",
    "downsample_pcd",
)


def __getattr__(name):
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    native = import_module(f"{__name__}.native")
    value = getattr(native, name)
    globals()[name] = value
    return value


__all__ = list(_NATIVE_EXPORTS)
