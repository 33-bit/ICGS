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
| Method contracts/configuration P00 | Partially reusable / planned | Native contracts/config remain; [target contracts](../method/contracts.md) add physical/task/timed types |
| Timed execution P01 | Partially reusable / evidence-gated | Absolute targets reusable; fixed-duration controller/mask/live-clock pilot required |
| Executed episode data/tasks P02 | Planned / evidence-gated | Separate schema, task catalog, asset/monitor feasibility; native PyG unchanged |
| Physical encoder/decoder bridge P03 | Planned / evidence-gated | New geometry; same-seed reconstructed-cloud/native-action fidelity required |
| Physical GRU/history P04 | Planned | New causal state, branch isolation and history replay |
| Event encoder P05 | Planned | Observable segmentation, raw-window references, permutation-compatible tokens |
| Task tracker/router P06 | Planned / evidence-gated | New inference; native one/two-demo sessions require validation |
| Physical world model P07 | Planned | Three fixed bootstrap heads, decode–reencode; no task input |
| Replay/branch/reference data P08 | Planned / evidence-gated | Actual restoration/counterfactual labels, no fabricated snapshots |
| Continuation/event evaluators P09 | Planned | Value under frozen reference; separate learned stopping |
| Rerank/shooting/MCTS P10 | Planned | Capability-based common-command search; no concrete IP import |
| Staged method training P11 | Planned | A0/A1/B/C/D/E; existing native trainer remains separate |
| Controls/benchmark P12 | Planned / evidence-gated | B0–B7 and custom suite, no measured success claimed |
| Integrated method P13 | Planned / evidence-gated | Installed execution and six-program pilot before primary scaling |
| Approximate MCGS/CEM and external/robot tracks | Outside primary / optional | Not silently included in approved core or current runtime |

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

## Target contracts for later learning/search work

The authoritative [target specification](../method/README.md) and
[P00–P13 roadmap](../plans/active/icgs-method-implementation.md) now detail these
boundaries. Status above was checked against source on 2026-09-07; documentation
adoption itself implements none of the new runtime components.

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
and remaining deadline H are distinct quantities. Exact cache is primary;
approximate MCGS merging is excluded. Branch collection needs validated snapshot
or replay capability; arbitrary exact snapshot support is not assumed and must
never be represented by a stub returning success.

## Data and training roadmap

Native PyG records keep their original schema. Future ICGS records need explicit
version/lineage for geometric pseudo-context, executed transitions, and restored-
anchor branches/reference outcomes. Snapshot IDs, task programs and success/contact
oracles belong to training/evaluation, not deployed Observation.

Learning order: geometry warm-up; temporal physical-memory warm-up with pilot
dynamics; event/tracker/router; freeze reference; collect outcomes; dynamics
refinement and learned evaluators; calibration/test. Models own tensors, losses
belong to their algorithm/stage, optimizers/runners to training. Changing reference
controller after collection changes value targets and requires a new lineage.

Only current native training has an executable runner. No random neural module,
constant value function, fake planner or speculative trainer is in production.
