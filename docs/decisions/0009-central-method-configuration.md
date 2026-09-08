# 0009: Packaged JSON owns added-method hyperparameter defaults

Date: 2026-09-09. Status: accepted by owner approval of Centralize ICGS
Hyperparameters in JSON, including ICGS-only scope and retained shape locks.

## Context

MethodConfig provided typed metadata and partial JSON overrides but numeric
defaults lived in Python. Many target training/data/search settings lived only in
the proposal/specifications; parallel component work risked creating inconsistent
local constants. Native configuration/checkpoint identity must not change.

## Decision

Use one packaged `configuration/profiles/icgs_primary.json` as the authoritative
default source for the new ICGS method. Python defines immutable types, ranges,
compatibility checks and derived values, not an independent numerical fallback.
Missing/malformed resource fails explicitly. Explicit experiment JSON overlays
defaults; explicit overrides win; file-relative paths retain existing behavior.

Preserve MethodConfig entry points, schema1 and prior effective field values. Add
missing target settings without claiming their future consumers are implemented.
Existing shape locks and native constraints remain. Semantic dimensions and
dependent counts are computed, not independent tuning knobs. Unknown setup and
resource/seed values remain unset until the relevant operation requires them.

The full resolved config and its SHA256 belong in future run artifacts. That full
identity is not the frozen reference-policy identity: existing reference semantics
still exclude learned stopping/evaluator/search changes. No file/hash is evidence
of artifact trust, physical calibration, model quality or consumption of a setting.

## Alternatives and consequences

- Python-only defaults duplicate numeric source with experiment files.
- Documentation-only tables cannot supply installed configuration.
- Root-only JSON would break standalone installed use; the resource ships in wheels.
- Moving native IP too expands baseline risk; owner explicitly chose ICGS only.
- Unrestricted architecture tuning would conflict with existing shape validation;
  owner explicitly retained those locks.

All component plans now identify the sections to consume and require explicit
nondefault propagation tests as each component is integrated. No model is to read
docs or load JSON inside forward/step loops. See [parameters](../method/parameters.md).

## Compatibility and validation

Adds a default source and extended fields; does not supersede native baseline,
ADR0001/0004/0005 or reference/stopping ADR0008. Shape checks and original values
remain regression-tested. Validate immutable roundtrips, strict parsing, overrides,
paths, full identity, missing resource failures and installed wheel loading without
the checkout. No training/simulator/checkpoint workload is incidental to this change.
