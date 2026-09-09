# Planned ICGS contracts

Status: PR/AD target plus explicitly marked IDs; active P10 search runtime implemented with deterministic fixtures (`icgs.algorithms.planning.{belief,budget,mcts,rerank,shooting}`). None of the other planned types below exists
yet unless called EI. See the [native contract](../components/policy-data-contract.md)
for unchanged installed behavior and the [method index](README.md) for authority.

## Online information and frame/time contract

**PR:** `Observation` stays world XYZ `[N,3]`, measured `T_w_e [4,4]`, grip
0 closed/1 open. World W is the robot base, meters/radians, scale ell0=1 m;
gravity is calibration. Demos have their own calibrated base frames. No per-object
normalization. Foreground includes every declared object/blocker, not a target ID.

**PR:** one interval holds one absolute target for dt0=0.1 s. T=8 proposed targets,
h=r=2 default edge/commit intervals, L=32 lookahead, H<=512 remaining intervals;
L analysis is 8/16/64. Length stress H<=1024 is separately trained. Native timing
remains unknown where not measured. No pose-reaching completion time is an action
input. Commanded and achieved quantities have separate fields.

**ID:** interval boundaries are indexed integers from reset. Initial observation
is boundary zero; each successful `advance` yields the next boundary observation.
Counters update once per boundary, not once per planning request. Replanning after
r intervals uses already updated memories; it does not encode/update the last
observation twice. A same-boundary request may read state but cannot advance it.

Native normalized grip maps to command `int(g_native > 0)` (zero closes).
**ID resolving gripper commitment:** at a replan take the selected first target's
grip and hold it for the entire r-interval commitment. Proposed future groups of r
intervals use their group's first grip. Apply this materialization identically in
collection, reference execution, model rollouts and added controls; store raw and
materialized commands. No mid-interval cancellation. Nondefault r must satisfy
1<=r<=h<=P for one edge. Native-gripper cadence is a separate control.

```text
world_goal[j] = proposal_root @ native_relative_target[j]
u_m = [Log(inv(achieved_pose_m) @ world_goal).translation / ell0,
       Log(...).rotation, commanded_grip, log(planned_dt / dt0)]  # 8
p   = [translation/ell0, first_two_rotation_columns, measured_grip, gravity] #13
```

`Log`/`Exp` here are translation-first SE(3) twist operations, not concatenated
raw translation and axis-angle. Implement stable small-angle and near-pi tests in
new geometry helpers; do not rewrite the native codec's different formulas.

## Type inventory and ownership

**ID:** model tensors below use leading batch B; online B=1. Data/storage uses
NumPy and metadata, neural modules use torch. Conversion occurs at composition/
data boundaries, not hidden in the simulator. Frozen records do not imply frozen
tensor storage. Constructors copy caller-owned arrays; branch-copy clones mutable
tensors. Shared immutable content uses non-writeable owned backing.

| Planned type | Fields / constraints | Owner |
| --- | --- | --- |
| `TimedCommand` | `target_w [4,4]`, `grip {0,1}`, `duration_s>0`; finite valid SE(3) | contracts |
| `CommandPrefix` | nonempty tuple of TimedCommand, proposal root, raw candidate ID; already absolute/materialized | contracts |
| `TimedObservation` | existing Observation, boundary index, simulator timestamp, measured wall timestamp, sensor-profile ID | contracts |
| `ExecutedTransition` | before/after TimedObservation, TimedCommand, achieved duration, physics substeps, controller status; no oracle labels | contracts |
| `PhysicalState` | `X [B,128,256]`, `x [B,128,3]`, `anchor_valid [B,128]`, `p [B,13]`, `memory [B,2,256]`, `T_w_e [B,4,4]`, `grip [B,1]`, cached world cloud/mask, boundary, encoder/physical-memory lineage, origin real/imagined | state |
| `SegmentRef` | demo content hash, boundary indices a,b, kind interaction/start/end, valid action-window flag; indices into immutable raw data | contracts |
| `EventMemory` | `tokens [B,Lc,256]`, `valid [B,Lc]`, SegmentRef sequence, demo-content/encoder/segmentation lineage | state |
| `TaskState` | `r [B,256]`, `alpha [B,Lc+1]` (last null), rho/nu/eligible `[B,Lc]`, boundary, context/tracker lineage | state |
| `MethodContext` | immutable raw demos + EventMemory + separately owned native full/window PreparedContexts, reference ID; no task state | state |
| `PhysicalPrediction` | next PhysicalState, grip logits `[B,1]`, head ID; contains no context/task output | contracts |
| `TerminalProbabilities` | success/failure/continue `[B,3]`, finite nonnegative sum 1, event-temperature artifact ID | contracts |
| `EvaluationOutput` | value/completion/progress logits, calibrated V/S and temperatures ID; V(H=0)=0 | contracts |
| `Hypothesis` | fixed head ID 0..2, PhysicalState, TaskState, scalar active mass >=0 | planning |
| `BeliefNode` | tau,H_root,U,F, three hypotheses, candidate edges, node visits; root-relative masses | planning |
| `PlanningResult` | selected CommandPrefix, completed evaluation/fallback reason, expected return if computed, call/timing/cache counters, immutable audit IDs | contracts |

The online records never contain program IDs, object-role IDs, predicate labels,
snapshot handles, future measured frames, score or oracle stage. Annotation records
and replay capabilities live in data/environment/benchmark owners and join by IDs.
Context hash changes invalidate TaskState; physical state remains reusable only
when its own causal/encoder lineage is identical. A context swap recomputes q by
replaying the same physical history, not by changing only M_C at the final step.

## Planned Python interfaces

These signatures are normative planning vocabulary, not importable examples.
Tensor output shapes are defined above. `H` is a nonnegative integer; horizons
beyond checkpoint training coverage fail validation instead of silent clipping.

```python
encode_cloud(points_w, point_valid) -> EncodedCloud  # X,x,anchor_valid
decode_cloud(encoded: EncodedCloud) -> DecodedCloud  # points_w,point_valid
physical_update(encoded, observation, previous_descriptor, memory) -> PhysicalState
segment_demo(raw_demo, segmentation_config) -> tuple[SegmentRef, ...]
encode_events(raw_demos, segments) -> EventMemory
track_task(previous: TaskState | None, state: PhysicalState,
           events: EventMemory) -> TaskState
sample_prior(observation, task: TaskState, context: MethodContext,
             *, seed: int) -> Candidate
materialize_prefix(candidate, *, h: int, r: int, duration_s: float) -> CommandPrefix
predict_step(state: PhysicalState, command: TimedCommand,
             *, head_id: int) -> PhysicalPrediction
evaluate_state(state: PhysicalState, task: TaskState,
               events: EventMemory, H: int) -> EvaluationOutput
predict_terminal(before: PhysicalState, q_before: TaskState,
                 after: PhysicalState, q_after: TaskState,
                 events: EventMemory, command: TimedCommand) -> TerminalProbabilities
plan(state: PhysicalState, task: TaskState, context: MethodContext,
     *, H: int, budget: PlanningBudget) -> PlanningResult
```

`EncodedCloud` is the X/x/anchor_valid subset of PhysicalState; `DecodedCloud` is
points_w/point_valid. `PlanningBudget` carries wall seconds, optional native-call
cap, optional model-interval cap, and clock track, with independent counters.
`previous_descriptor` is `[B,8]`, computed from the **before-transition achieved
pose** and absolute command before stepping. Store it with the transition or
reconstruct it from the before observation; do not recompute using the successor
pose. It is zero only at reset. Terminal probabilities at inference are
`softmax(raw_event_logits / Te)`; raw logits remain the D2 cross-entropy input.
`Candidate` reuses EI native trajectory/provenance fields, with added route/window
and seed provenance in a method wrapper; do not expand native Candidate semantics
silently. Capability protocols expose these operations; no planner imports a
concrete InstantPolicy. Neural forward methods may implement the operation as a
module call, but public adapters retain these argument/return meanings.

`TimedEnvironment.reset(seed)` returns TimedObservation;
`advance(command)` returns ExecutedTransition; `close()` is idempotent. Privileged
`TaskMonitor.annotate(transition)` returns a separate annotation.
`ReplayProvider.restore(anchor_ref)` returns a measured observation plus a
ReplayReport, never `True` as a fake restoration success. `ReplayReport` names
stored/restored fields, exact-snapshot/deterministic-replay/approximate-reset
mode, discrepancy measurements and protocol ID.

## Configuration, artifacts and failures

**EI/AD:** `MethodConfig` now loads the added-method defaults from the packaged
[primary JSON](../../src/icgs/configuration/profiles/icgs_primary.json), not Python
numeric defaults; native `ExperimentConfig` remains unchanged. `build_method` is
still a separate integration deliverable. The expanded immutable typed sections
cover all model/training/collection/search/evaluation parameters; see the
[key and consumer reference](parameters.md). Preserve existing constructor/JSON
entry points and old effective defaults/shape locks. Persist schema version and component IDs;
reject unknown keys/IDs, nonfinite values and incompatible shape/horizon settings
before loading tensors. No Hydra, plugin discovery, implicit relative root presets
or executable config serialization. Runtime defaults < explicit JSON < explicit
CLI overrides remains the precedence; file-relative paths resolve against JSON.
Only outer composition reads a JSON configuration; neural forward/update loops
receive explicit sections/arguments. `resolved_config()` returns a complete
JSON-ready envelope with a full-configuration SHA256 for future run storage;
this hash is distinct from the reference fingerprint below. New settings are not
claimed wired into existing physical modules merely because validation accepts them.

**PR/AD:** reference fingerprint includes IP checksum, native profile, geometry and
physical/task/event weights, routing/segmentation/preprocessing, calibration and
camera/gravity/workspace profile, cadence/r and RNG protocol. It excludes world
model, continuation/event evaluators and learned stopping. The latter get their
own artifacts; an evaluator references its training reference ID. Resume artifacts
include optimizer/scheduler, stage, dataset manifest, resolved config and RNG state.

**ID:** empty/invalid clouds or invalid poses reject the observation; terminate the
added-protocol rollout with a tagged invalid-input outcome rather than hallucinate
geometry. Missing artifacts, stale caches, invalid windows and unsupported control
capabilities are explicit diagnostics. Invalid individual windows get router
fallback; invalid full context aborts context setup. Nonfinite model output cannot
be cached/selected; record model error and use a reference proposal only if the
real observation/context remains valid. No unbounded retries in new paths.

**FG:** controller timing, crop/calibration/assets, replay equivalence, native
single-demo compatibility and decoded-cloud policy fidelity require measured
evidence. The [roadmap](../plans/active/icgs-method-implementation.md) owns the gates.
