# 0017: Frozen ICGS Stage-1 split over RoboHiMan (split-v1)

Date: 2026-09-30. Status: accepted — frozen by explicit owner request
(pre-flight/split-lock brief, 2026-09-30). Changing it requires a new
`split_id` and a superseding decision.

## Context

The native RoboHiMan protocol trains on A/AP (10 atomic tasks) and C/CP
(4 compositional tasks) and tests on the same atomic tasks plus 12
compositional tasks. Four of those 12 are the training compositions. The
[pre-flight evidence](../experiments/robohiman-validation/v3/README.md) shows
three things:
- TRAIN covers every primitive step and every manipulated asset of the other
  eight compositional tasks;
- native seeds are unrecoverable (`np.random.seed(None)`);
- perturbation families are shared between train and test.

The native split therefore supports composition claims only at the level of
task structure. It cannot support asset or primitive generalization.

## Decision

Artifact: `artifacts/robohiman/icgs_robohiman_stage1_split_v1.json` (SHA256
`2d1e7dcfd300480854d9d35e1c7abbf196863790be7ff6fcaa65317a73b032ef`), locked by
the `.lock` file next to it (SHA256). The collector refuses any manifest whose
hash differs from its lock.

**TRAIN (the only split that feeds gradients).**
- Tasks: the 10 atomic tasks and the 4 native C tasks (`put_in_without_close`,
  `sweep_and_drop`, `take_out_without_close`, `transfer_box`).
- Levels: strategy 0 (A/C) plus every enabled Colosseum strategy except
  `distractor`, `background_texture` and `all_mixed`.
- `rubbish_in_dustpan` has strategy 0 disabled upstream, so it contributes AP
  strategies only, matching native `train_AP`.
- Seeds: numpy seeds 100,000,000–199,999,999; factor seed 42.

**DEV (model selection only).**
- `DEV-config`: TRAIN tasks under DEV seeds 200,000,000–299,999,999 and factor
  seed 4242.
- `DEV-composition`: `take_out_and_close`, `take_two_out_of_same`,
  `take_two_out_of_different`. These are the "take" counterparts of three
  "put" test structures.

**TEST (reported claims).** Seeds 300,000,000–399,999,999; factor seed 244
(native test).

| Track | Tasks | Claim |
| --- | --- | --- |
| `TEST-config` | TRAIN tasks under TEST seeds | Known-task configuration robustness |
| `TEST-unseen-perturbation` | TRAIN tasks under the withheld strategies | Robustness to unseen perturbation families |
| `TEST-held-out-composition` | `box_exchange`, `put_in_and_close`, `put_two_in_different`, `put_two_in_same`, `retrieve_and_sweep` | Held-out task structure |
| `TEST-dependency-stress` | `put_in_and_close`, `put_two_in_different` | Prerequisite chains (a subset of held-out composition) |

- Every held-out task introduces at least one step transition unseen in TRAIN
  while reusing only TRAIN primitives and object geometry.
- Inference-time demonstration contexts for TEST come only from seeds
  390,000,000–399,999,999.
- Unsupported claims are named in the manifest: asset generalization, unseen
  primitives, and dependency-mechanism generalization.

**Excluded.** The custom 36-program generator (removed from the tree by
[ADR 0019](0019-remove-legacy-generation.md); recoverable from Git) was a possible
mechanistic diagnostic suite. There is no scientific reason to put it into
TRAIN: its dependency value is unproven, and it has known physical
abstractions. The original Colosseum 20 tasks are out of benchmark scope.

## Alternatives considered

- **Native split as-is.** Shares perturbation families and has no DEV, so
  test tasks would drive model selection.
- **All eight unseen compositions in TEST, DEV from seeds only.** Model
  selection would then carry no composition signal.
- **A grocery-held-out asset track.** Confounds asset and composition in
  `box_exchange`, and would remove training diversity; not adopted.

## Consequences

- Stage-1 generation must pass `--split-manifest/--split`; Stage-2 anchors
  inherit the split of their source episode.
- The misnamed always-on drawer-size factor is present in every split.
- The consequential-decision evidence is sparse and lives in TRAIN
  (`put_in_without_close`), so dependency claims on TEST are stress tests, not
  generalization claims.

## Compatibility implications

This defines a new dataset identity (`split_id`). Native RoboHiMan episodes and
the legacy custom archives are not members of the split.
