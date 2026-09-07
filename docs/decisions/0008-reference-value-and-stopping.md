# 0008: Reference continuation value excludes learned stopping

Date: 2026-09-07. Status: accepted target semantics.
Decision owner: repository owner. Approval evidence: owner selected “Separate
stopping rule” during planning and approved the complete implementation-plan brief.

## Context

The proposal collects continuation labels in C, trains completion S in D2, then
uses S for online stopping. Including that changing stop rule in the reference
controller would change the policy whose outcomes were collected and create a
training-order cycle.

## Decision

`pi_ref` means exactly the frozen router/IP action controller, without the learned
stop rule. It draws one prior sample, executes the same r2 interval commitment,
updates causal memories, and repeats without consulting H except timeout. Training
continuations stop on external task terminal/deadline. V_H is success probability
under this named action controller from an active state; V_0=0. It is neither
optimal feasibility nor success probability of the final search policy.

Learned stopping is a separately versioned deployment wrapper using calibrated
S>=0.95 for three consecutive measured boundaries. Share it across B1–B7, keep B0
native, report false stops and matched disabled-stop controls. Do not call deployed
gated B2 exactly the controller used to generate V targets. Its terminal rule is
outside the reference fingerprint but inside deployed-method provenance.

Physical/task/event encoder weights, preprocessing/calibration, routing, IP artifact,
cadence and seed protocol affecting reference behavior freeze before C. Changing
them requires new reference identity and recollection. Dynamics refinement and
evaluator/calibration training do not update this frozen reference path.

Search initializes root success mass once with S and accumulates first-terminal
hazards. Leaf return U+sum(active_mass*V_remaining) never adds completion twice.
Progress Phi is auxiliary, never added as reward. Value under a weak reference
does not establish global infeasibility.

## Alternatives considered

- Train/freeze stopping before collection: coherent but moves completion training
  and calibration ahead of C; owner chose to preserve the proposed ordering.
- Leave unspecified: allows silent target drift and misleading value claims.
- Recollect after every evaluator update: unnecessary when stopping is separate,
  with substantial simulator cost.

## Consequences

[Data/training](../method/data-training.md) owns freeze phases and target labels;
[planning/evaluation](../method/planning-evaluation.md) owns gated execution and
matched reporting. V need not equal actual gated-policy success. The difference
must remain visible in analysis and artifacts.

## Compatibility implications

Resolves the proposal's reference/stopping ambiguity without changing the archived
source or any earlier ADR. Native checkpoints and baseline behavior stay unchanged.
No existing outcome bank is claimed migrated or relabeled by this docs change.
