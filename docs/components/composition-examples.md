# Composition and controlled experiments

These examples use current APIs. They require the indicated research libraries and
trusted encoder/checkpoint assets; none is automatically run by validation.

## Original architecture

```python
from ip.configs.original import instant_policy_original
from ip.composition import build_policy, build_training_module

config = instant_policy_original()
policy = build_policy(config)  # CUDA + original scene encoder artifact required
training_module = build_training_module(config)  # independent model; requires Lightning
```

For inference from existing weights, use load_policy rather than treating an
uninitialized build as pretrained. From the original ip working directory:

```python
from ip.composition import load_policy
policy = load_policy("./checkpoints", mode="eval", num_demos=2)
```

Training/evaluation/deployment share construction. Do not build both examples above
when only one model is needed. Source defaults and entry profiles differ by design.

## Replace only the scene encoder

Supply your real nn.Module factory with the
[encoder contract](../ARCHITECTURE.md#public-component-contracts). This example
assumes make_scene_encoder is an implemented, imported experiment factory; it does
not invent the encoder architecture.

```python
from dataclasses import replace
from ip.composition import ComponentFactories, build_policy
from ip.configs.original import instant_policy_original

def build_encoder_experiment(make_scene_encoder):
    base = instant_policy_original()
    experiment = replace(base, name="encoder_ablation",
                         scene=replace(base.scene, kind="experiment_encoder",
                                       pretrained=False, freeze=False),
                         runtime=replace(base.runtime, compile_models=False))
    factories = ComponentFactories(scene={"experiment_encoder": make_scene_encoder})
    return build_policy(experiment, factories)
```

Only encoder identity/initialization and explicit compilation setting change;
record them as controls/confounders. Preserve width, node count and frames to keep
the original graph/backbone/codec/sampler. Recompute cached features. A semantic
object encoder with incompatible node meaning needs an explicit compatible graph
composition; shape matching alone is insufficient.

## Add a compatible sampler

The new sampler class must implement constructor(sampling,diffusion,codec,schedule)
and sample(network,data) returning ActionTrajectory.

```python
from dataclasses import replace
from ip.composition import ComponentFactories, build_policy
from ip.configs.original import instant_policy_original

def build_sampler_experiment(sampler_class):
    base = instant_policy_original()
    config = replace(base, name="sampler_ablation",
                     sampling=replace(base.sampling, kind="experiment_sampler"))
    return build_policy(config, ComponentFactories(sampler={"experiment_sampler": sampler_class}))
```

Scheduler identity is selected separately by diffusion.scheduler_kind; objective by
diffusion.kind. Flow matching generally changes training targets too and is not
merely a sampler swap. No new sampler is shipped here; tests inject controlled
collaborators to prove the seam.

## Second environment

Implement EvaluationEnvironment from ip.types using your benchmark's actual API:
launch, collect_demos returning prepared conditioning dictionaries, reset, observe
returning Observation, encode_action anchored to the observation, step returning
(reward,terminate), and shutdown. Then:

```python
from ip.evaluation import evaluate_policy

def evaluate_second_environment(policy, adapter):
    return evaluate_policy(policy, adapter, num_demos=2, num_rollouts=1,
                           max_execution_steps=30, execution_horizon=8,
                           num_traj_wp=10)
```

No policy internals or RLBench classes belong in this adapter. If the benchmark's
metric/termination protocol differs, provide its own evaluator rather than claiming
the RLBench success fraction applies universally. collect_demos is prepared=True:
it must supply local clouds/grips/world demo poses with the documented waypoint count.

## World model or planner later

Conceptual composition only; not implemented interfaces:

```text
outer agent owns history/representation
  → world model and value estimates
  → planner chooses a grounded low-level goal/context
  → existing InstantPolicy.predict(observation, context)
  → environment executes decoded trajectory
```

A world model never needs to be imported by the current policy. MCTS/MCGS/CEM own
their search state, rollout/value contracts and goal grounding outside it. How a
planner's goal becomes demonstrations/context is future research work, not an
already supported latent-goal API. Action-representation changes similarly require
a compatible codec and often graph/denoiser/objective; environment code can remain
unchanged only if the decoded physical action contract remains the same.

## Experiment recording and loading

Use the [research contract](../experiments/TEMPLATE.md). Frozen config sections and
dataclasses.replace isolate overrides. resolved_config(config) returns versioned
JSON-safe metadata; training writes resolved_config.json and legacy config.pkl.
On load, read_experiment_config prefers resolved JSON; restore any custom factories
explicitly. Unknown IDs, fields or schema versions fail instead of choosing defaults.
The config metadata is not a plugin installer or a metric database.
