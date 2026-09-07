# 0007: Named timed execution and executed-data lineage

Date: 2026-09-07. Status: accepted for target design; physical feasibility unverified.
Decision owner: repository owner; approval: approved documentation/component plan.

## Context

Existing AbsolutePrefix has optional duration metadata; existing RLBench step uses
native controller semantics. Neither proves a fixed-rate wrapper or exact replay.
Native PyG pseudo-context records cannot certify physical dynamics outcomes.

## Decision

Add a separately identified ICGS fixed-target-interval protocol with proposed
dt0=0.1 s and h=r=2. Record absolute target, commanded grip/planned duration,
achieved pose/grip/duration and physics substeps separately. Never infer achieved
timing from target count or future pose-reaching completion time.

Use the same common absolute commands across hypotheses; apply world/local
conversion once. Declare mask/crop/calibration, safe hold and wall-clock semantics
in the controller protocol. Keep native cadence/sensors as B0 and isolate wrapper
changes in B1. Gate timing/live-clock claims on measured controller evidence.

Create separately versioned executed episodes, numeric shards and manifests;
preserve native formats. Split programs/assets/source lineages before generation.
Privileged simulator/program/predicate data is annotation-only, not online input.
Snapshot/replay provenance records actual restored fields and measured discrepancy.
Exact, deterministic replay and approximate repeated-reset data are not conflated.

## Alternatives considered

- Treat native `step` as0.1 s: unmeasured and potentially false.
- Reuse geometric pseudo-demos as dynamics: violates executed-outcome semantics.
- Call every near-restored state an exact counterfactual: hides solver/controller
  differences and invalidates matched comparisons.
- Mix external simulator trajectories without lineage: introduces unmeasured domain
  changes; external sources remain separately versioned tracks.

## Consequences

[Contracts](../method/contracts.md) and [data/training](../method/data-training.md)
own details. Timing, replay and task-asset pilots precede large collection. Missing
exact restoration may permit explicitly approximate analysis, not an exact-intervention
claim. Unmet required gates block the dependent claim/workload, not docs completion.

## Compatibility implications

This adds a named protocol and new record schema; it does not replace native action,
normalization, PyG/inference schema or published profile. It does not supersede any
prior accepted ADR. Native public benchmark scoring stays unchanged. Changing new
cadence/preprocessing/controller/reference lineage requires new labels where affected.
