"""Strict JSON/resource mechanics for the explicitly declared method schema."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import lru_cache
from importlib import resources
import json
from math import isfinite
from types import UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def typed_value(value: Any, annotation: Any, name: str, *, complete=False) -> Any:
    """Check declared types (not example values) and freeze all JSON sequences."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for option in args:
            try:
                return typed_value(value, option, name, complete=complete)
            except ValueError:
                pass
        raise ValueError(f"{name} has an invalid type or value")
    if annotation is type(None):
        if value is None:
            return None
    elif annotation is bool:
        if type(value) is bool:
            return value
    elif annotation is int:
        if type(value) is int:
            return value
    elif annotation is float:
        if type(value) in (float, int):
            try:
                converted = float(value)
            except OverflowError as exc:
                raise ValueError(f"{name} must be a finite representable number") from exc
            if isfinite(converted):
                return converted
    elif annotation is str:
        if isinstance(value, str):
            return value
    elif origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(typed_value(item, args[0], f"{name}[{index}]", complete=complete)
                         for index, item in enumerate(value))
        if len(value) == len(args):
            return tuple(typed_value(item, kind, f"{name}[{index}]", complete=complete)
                         for index, (item, kind) in enumerate(zip(value, args)))
    elif isinstance(annotation, type) and is_dataclass(annotation):
        hints = get_type_hints(annotation)
        if type(value) is annotation:
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be {annotation.__name__} or an object")
        unknown = set(value) - set(hints)
        if unknown:
            raise ValueError(f"unknown configuration key in {name}: {sorted(unknown)}")
        if complete and set(value) != set(hints):
            raise ValueError(f"missing canonical defaults in {name}: {sorted(set(hints) - set(value))}")
        checked = {key: typed_value(item, hints[key], f"{name}.{key}", complete=complete)
                   for key, item in value.items()}
        # Complete validation is used while reading defaults; no constructors or
        # factories may run there, otherwise resource loading would recurse.
        return checked if complete else annotation(**checked)
    raise ValueError(f"{name} must match declared type {annotation}")


@lru_cache(maxsize=1)
def _checked_defaults(text: str, schema: type) -> dict[str, Any]:
    document = strict_json(text)
    if not isinstance(document, dict) or set(document) != {"schema_version", "config"}:
        raise ValueError("canonical defaults require schema_version/config envelope")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("canonical defaults require schema_version 1")
    return typed_value(document["config"], schema, "config", complete=True)


def defaults(schema: type) -> dict[str, Any]:
    try:
        resource = resources.files("icgs.configuration").joinpath("profiles/icgs_primary.json")
        return _checked_defaults(resource.read_text(encoding="utf-8"), schema)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("invalid or missing packaged canonical defaults "
                         "icgs.configuration/profiles/icgs_primary.json; "
                         f"reinstall a complete ICGS package: {exc}") from exc


class TypedSection:
    """Frozen dataclass mixin: direct constructors enforce types and freeze lists."""

    def __post_init__(self):
        hints = get_type_hints(type(self))
        for item in fields(self):
            value = typed_value(getattr(self, item.name), hints[item.name],
                                f"{type(self).__name__}.{item.name}")
            object.__setattr__(self, item.name, value)


def json_shape(value):
    if is_dataclass(value):
        return {item.name: json_shape(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [json_shape(item) for item in value]
    return value
