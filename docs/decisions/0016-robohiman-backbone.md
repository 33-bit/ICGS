# 0016: RoboHiMan as primary task/environment backbone, ICGS-owned instrumentation

Date: 2026-09-30. Status: accepted as migration direction by explicit owner
request (2026-09-30 migration brief). Default-path switch remains gated by the
go/no-go gates below. The custom generator was later removed by
[ADR 0019](0019-remove-legacy-generation.md).

## Context

The custom 36-program generator ([classification](../audits/2026-09-30-robohiman-migration-classification.md))
has unresolved physical-abstraction gaps (handle-block drawers, marker
predicates) and a costly custom benchmark. RoboHiMan / HiMan-Bench
(`chenyt31/RoboHiMan@33f71d3`) provides atomic/compositional tasks, assets,
perturbation factors and machine-readable predicates, but its saved
demonstrations lack commanded actions, timing, failures, predicate traces and
validated anchors ([compatibility audit](../audits/2026-09-30-robohiman-icgs-compatibility.md)).

## Decision

1. RoboHiMan is an upstream dependency pinned as: RoboHiMan `33f71d30`,
   PyRep `stepjam@231a1ac6`, RLBench `buttomnutstoast@587a6a0e` (RVT pins that
   the RoboHiMan overlay files match), CoppeliaSim 4.1.0. Task code, TTMs,
   assets, success conditions, variation factors and collection strategy JSON
   are used unmodified.
2. ICGS instrumentation is owned by `icgs.environments.robohiman`. It uses
   runtime *instance-level* method wrappers (installed and removed per episode)
   to observe commands and physics steps; it never edits upstream files. A
   neutrality gate must show identical physics with and without wrappers.
3. RoboHiMan perturbation factors (AP/CP Colosseum variations) and ICGS
   execution perturbations are recorded as *perturbations*; execution result is
   recorded separately as *outcome*. Neither implies the other.
4. Stage-1 records use the source-neutral `icgs_stage1_episode_v1` schema in
   `icgs.data.stage1`; views are indexes into one store. `D_value`/`D_pair` are
   refused until a frozen `pi_ref` identity exists.
5. Oracle predicates, object poses and task identities are offline labels only;
   the online field allowlist of `icgs_episode_v1` stays the model boundary.
6. The custom generator becomes non-primary legacy/reference and is not scaled.
   (Superseded: removed from the tree by [ADR 0019](0019-remove-legacy-generation.md).)

Forbidden: editing RoboHiMan task/success/variation code to improve ICGS
results; labelling achieved next pose as a command; fabricating failures by
editing frames or labels; calling configuration-tree restoration “exact”
without the measured replay gate.

## Alternatives considered

- Keep scaling the custom benchmark: rejected by owner; physical gaps remain.
- Patch upstream `Scene.get_demo` to record commands: smallest diff, but forks
  benchmark code; runtime wrappers achieve the same without source edits.
- Re-implement the expert: needed only for mid-episode continuation; allowed
  only as an ICGS executor that calls upstream primitives in upstream order and
  passes a parity test against `Scene.get_demo`.
- Reuse the custom v2 archive writer: rejected for now; its manifest identity is
  bound to custom program/protocol fields.

## Consequences

Two collection paths coexist temporarily; the long-term target is one generic
ICGS layer over environment adapters. New dependencies (hydra-core, omegaconf,
colosseum) live only in the isolated simulator venv, not in `pyproject.toml`.

## Compatibility implications

No checkpoint, published-profile or IP preprocessing change. Legacy
`icgs_episode_v2` archives keep their identity. The RoboHiMan venv uses Python
3.11 (upstream INSTALL) and an unpinned `cffi` build dependency; this is
recorded in every episode manifest.
