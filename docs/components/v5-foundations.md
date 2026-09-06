# ICGS v5 ownership and implementation status

ICGS is the method-level project. InstantPolicy is one proposer/policy, not the
application's central abstraction. Unified composition connects shared contracts,
state, algorithms and execution; native IP neural code lives in ordinary model
owners, not a vendor/legacy backend.

| Area | Status | Owner / scope |
| --- | --- | --- |
| Native IP policy and published checkpoint | Implemented | icgs.policies.instant_policy, composition, artifacts, models, diffusion |
| Native K proposals | Implemented | algorithms.planning.candidates; sequential K calls, not search |
| Absolute command prefix | Implemented | execution.commands; root @ each target, never chained increments |
| Context/RNG lifecycle | Implemented | state.context_cache and randomness; owned immutable demo content, separate mutable buffers |
| Physical/task memory contracts | Designed | This document; no unused executable neural stubs |
| Spatial physical encoder, GRU, task tracker/router | Deferred | Future models.encoders/models.memory plus state owners |
| World model, continuation/event evaluators | Deferred | Future models.dynamics/models.evaluators; not benchmark evaluation |
| Shooting/MCTS/MCGS/CEM, planned/stage-aware policies | Deferred | Search uses narrow capabilities; no concrete InstantPolicy import |

## Implemented candidate and command contract

Candidate.index is K index; trajectory transforms/grips are [B,P,4,4]/[B,P,1].
The current observation API supports B=1, separate from K. Each proposal records
root pose, immutable demo-content hash, artifact identity, seed and elapsed time.
Seeds are scoped across Python/NumPy/torch, restoring affected RNG on exception;
global RNG consumers must not run concurrently. K=1 executes the same native call,
not an optimized batched variant. Published profile encodes inside sampling, while
the historical source profile caches demo features explicitly.

absolute_prefix takes h<=P and returns root-pose-anchored world targets with
commanded normalized grip. Duration, achieved duration and physics substeps stay
unknown unless actually measured/supplied. Environment.step is not assumed 20Hz.
Branch copies share immutable demo content but clone mutable cached features;
they do not clone model, graph or scheduler scratch state. No branch tree exists yet.

## Contracts for later learning/search work

Physical state b=(X,x,p,m) includes geometry and causal physical history. Demo event
memory M_C and task memory q are separate; changing a goal context must not silently
change physical transition predictions. Real memory updates from observed execution;
imagined branches update from predicted transitions. Decoded cloud must retain enough
geometry for the next native proposal and cross the world/local boundary once.

Phi is progress, S is already-completed probability, and V_H^pi_ref is active
continuation under a fixed reference controller with V_0=0. Physical/event outcome
atoms carry weights; search chooses one common absolute action, integrates outcomes,
counts accumulated success once, and assigns continuation only to active mass.

Prediction chunk T/P, search prefix h, physical lookahead L, execution commitment r
and remaining deadline H are distinct quantities. Exact cache precedes approximate
MCGS merging; no graph merging is required now. Snapshot restoration is an optional
future real capability, never a stub returning success.

## Data and training roadmap

Native PyG records keep their original schema. Future ICGS records need explicit
version/lineage for geometric pseudo-context, executed transitions, and restored-
anchor branches/reference outcomes. Snapshot IDs, task programs and success/contact
oracles belong to training/evaluation, not deployed Observation.

Learning order: geometry warm-up; temporal physical-memory warm-up; event/tracker/
router; freeze reference; collect outcomes; world model and learned evaluators;
optional frozen-teacher consistency; calibration/test. Models own tensors, losses
belong to their algorithm/stage, optimizers/runners to training. Changing reference
controller after collection changes value targets and requires a new lineage.

Only current native training has an executable runner. No random neural module,
constant value function, fake planner or speculative trainer is in production.
