# 0006: Added ICGS components around a preserved native action prior

Date: 2026-09-07. Status: accepted for target design, not implementation evidence.
Decision owner: repository owner; approval: explicit approval of the ICGS Proposal
→ Documentation and Component Implementation Plans in this task.

## Context

[Current runtime](../ARCHITECTURE.md) implements native IP, K proposals, command
conversion and context/RNG lifecycle. The [proposal](../proposals/README.md) adds
physical history, task inference, world model, continuation evaluation and search.
Calling these implemented would contradict the source and validation evidence.

## Decision

Adopt the [method specification](../method/README.md) as the target. Keep canonical
`src/icgs`, native parameter ownership/checkpoint loading, graph/action/sampler
semantics and published preprocessing. Add components to ordinary model/state/
algorithm/training/environment owners, not a wrapped legacy subtree.

Physical state/prediction excludes task context. Task interpretation consumes
physical history and demonstration events. Search depends on narrow capabilities,
not concrete IP or simulator classes. Model tensor computation does not load
artifacts, run training/evaluation, or read docs/tests/root scripts.

Real-root native IP receives measured geometry. Added physical preprocessing and
imagined-cloud decode/re-encode are separate paths. Shared immutable context is
distinct from branch-local mutable physical/task memory and session scratch.

B3 is a separate history/context-conditioned diffusion student trained from the
same executed bank and outcome labels; native IP is not fine-tuned. Its precise
architecture/objective is an explicitly labeled implementation default in the
[control specification](../method/planning-evaluation.md).

## Alternatives considered

- Replacing IP internals would conflate planning gains with changed prior and
  invalidate the requested preservation comparison.
- A second wrapped legacy runtime would violate ADR0004 and duplicate ownership.
- Placeholder dynamics/constant evaluators would misleadingly appear implemented.
- B3 deterministic actor is cheaper but weaker for multimodal action control;
  the owner selected the separate diffusion student.

## Consequences

The [roadmap](../plans/active/icgs-method-implementation.md) decomposes P00–P13.
Contracts and independent test seams enable controlled ablations. More modules
require lineage and interface checks; specifications are not evidence of success.
Single/two-demo contexts and the decoded-cloud bridge need real native validation.

## Compatibility implications

Adds target ownership; does not supersede ADR0001, the remaining numerical/dependency
principles of ADR0002/0003, or ADR0004/0005. No installed APIs/data/checkpoint bytes
change in the documentation delivery. Runtime additions require original-path
regressions and mandatory published C1–C5 acceptance for the planned migration.
