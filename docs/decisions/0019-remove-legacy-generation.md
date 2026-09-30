# 0019: Remove the legacy custom-benchmark data stack

Date: 2026-09-30
Status: accepted
Decision owner and approval evidence: project owner, 2026-09-30 request to delete
all legacy data-generation code, tests and docs (committed and uncommitted),
choosing the whole custom-benchmark data stack and deletion of its history docs.

## Context

[ADR 0016](0016-robohiman-backbone.md) made RoboHiMan the task/environment
backbone and kept the custom 36-program generator as non-primary reference.
[ADR 0018](0018-robohiman-dataset-layout.md) defined the RoboHiMan dataset.
The legacy stack had no remaining role.

## Decision

The following were removed from the tree:
- `icgs.data.collection` (runner, programs, bindings, approved manifests and
  the canonical generator);
- `icgs.data.archives`, `icgs.data.training_layout`, the
  `icgs.data.datasets.{episodes,generation_*,training_export}` modules and
  `icgs.data.schemas.{episodes,episode_records}`;
- the RLBench teleop oracle;
- the `generation` environment profile and its runtime configs;
- `scripts/generation_*` and `scripts/plan_generation_batch.py`;
- `artifacts/{generation,composition}`;
- the tests for all of the above;
- the generation component docs, plans, audits, experiment evidence and specs,
  and ADRs 0011, 0014 and 0015.

The training split check now compares lineage-ID sets directly instead of
calling the removed `validate_split_lineage` helper.

## Alternatives considered

Keeping the stack as a reference was ADR 0016's position. The owner chose
removal; Git history is the record.

## Consequences

Recovery: `git revert` of the removal commit, or checkout of commit `df531e7`
(the last tree with the stack, including the owner's training-export
work-in-progress snapshotted in `29d4daa`/`df531e7`).

## Compatibility implications

- Checkpoints, the Instant Policy baseline, published fidelity and native PyG data
  preparation: unaffected.
- Datasets: legacy episode archives (local, VPS or HF) are no longer readable by
  this tree.
- RoboHiMan Stage-1: unaffected.
