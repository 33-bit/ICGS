# 0002: Explicit research composition inside ip

Date: 2026-09-06. Status: accepted by the owner with the modularization plan.

Layout/legacy-import clauses below are superseded by [ADR0004](0004-unified-icgs-v5.md).
They record the historical decision, not current namespace instructions. Numerical
and dependency principles remain unless specifically updated by that decision.

## Context

AGI constructs all components; GraphDiffusion combines algorithm and training;
evaluation/deployment manage model internals. Mutable configuration and hidden
state prevent isolated ablations. Source semantics must be preserved.

## Decision

Retain the ip namespace and legacy import/CLI paths. Use frozen dataclass config
sections, explicit component factories and a public policy facade. Preserve
original graph/action/sampler math, parameter owners and initialization order.
Separate pure geometry/actions, denoiser, objective/sampler, data IO, training,
environment conversion and evaluation. No speculative research components.

Reusable core is ip.types, ip.geometry, ip.actions, ip.policy, ip.algorithms,
ip.configs.structured, ip.configs.original, and model modules denoiser,
scene_encoder, graph_rep, graph_transformer, backbone and embeddings. Its local
dependency closure must not import RLBench, PyRep, WandB, Lightning, argparse,
training/evaluation/entry modules, checkpoint IO or composition. Compatibility
modules model.py/diffusion.py and stale occupancy_net.py are outside this core;
core must not import them. Pure model/geometry/action modules must not reach
Open3D. Policy preprocessing may use it. The evaluator uses a policy/environment
contract, not concrete model internals. Composition is an outer construction owner.

## Alternatives and consequences

Renaming to instant_policy or src/instant_policy adds import migration without
improving replaceability. Hydra/global registries add machinery without current
need. Preserve coherent modules and use protocols only at meaningful seams.
Graph/cache/scheduler state is per policy session, not implicitly thread-safe.

## Compatibility

Keep source baseline identity and old PyG data fields. New resolved config is
versioned, with explicit conversion and entry profiles. Preserve original CLI
defaults, mathematical quirks and benchmark conditions; fixes require separate
decisions. Checkpoint policy is in [0003](0003-checkpoint-compatibility.md).
The [migration record](../plans/completed/modularize-instant-policy.md) records implementation and evidence.
