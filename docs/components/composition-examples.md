# ICGS method-level usage

## Published native IP

```python
from icgs.artifacts.published import load_published_policy
from icgs.data.schemas.inference import load_input
from icgs.state.randomness import scoped_seed

policy = load_published_policy("/content/icgs-artifacts/model.pt", device="cuda")
observation, demos = load_input("/content/icgs-evidence/input.npz")
with scoped_seed(17, device="cuda"):
    context = policy.prepare_context(demos)
    trajectory = policy.predict(observation, context)
```

The wheel requires no reference binary, old package, sidecar config/encoder or
checkout-relative defaults. Only the known hash is accepted by this target loader.

## IP as one method component

```python
from icgs.algorithms.planning.candidates import propose_candidates
from icgs.execution.commands import absolute_prefix

candidates = propose_candidates(policy, observation, context, count=3, seeds=[17,18,19])
prefix = absolute_prefix(candidates[0].trajectory, observation.T_w_e, length=3)
```

This produces proposals, not a planner score or a search decision. Your method may
later select/evaluate proposals through its own physical dynamics/task-value models.
Those components use shared contracts and are not imported by InstantPolicy.

A second environment implements the narrow observation/demo/action/step contract in
`icgs.contracts.records.EvaluationEnvironment`; common rollout lives in
`icgs.execution.rollout.evaluate_policy`. Different metric semantics require a
named evaluation protocol, not reuse disguised as benchmark equivalence.

World model/planner direction (designed, not implemented):
history/physical representation → dynamics/evaluator → search → selected absolute
command prefix. Search depends on proposer/predictor/evaluator capabilities, never
concrete IP classes. Grounding a new goal into native demo context remains explicit
future research work, not an assumed latent-goal API.

## Internal component ablations

For research that actually changes native components, use
`icgs.composition.ComponentFactories` and `dataclasses.replace` on
`icgs.configuration.defaults.instant_policy_original()`. Scene, graph, backbone,
codec, sampler, objective and scheduler have explicit factories; preserve semantic
compatibility or declare a matched experiment. This is optional internal capability,
not the organizing center of the ICGS method.

Runtime JSON metadata records identities, not executable custom factory code.
Custom factories must be supplied by the experiment's Python composition.

## Target method composition — planned APIs only

The [target signatures](../method/contracts.md#planned-python-interfaces) describe
future `build_method`, timed execution and predictor/evaluator capabilities.
They are not runnable imports. Existing examples above remain the supported native
path. [P00–P13 plans](../plans/active/icgs-method-implementation.md) explain how to
extend composition without replacing `build_policy` or weakening published loading.
