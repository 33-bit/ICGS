# Executed data, labels and staged training

Status: **target**, not collected data or training evidence. PR values come from
proposal sections Losses and Gradient Flow, Training Data, and Training Procedure
and Configuration; ID choices resolve implementation details within
the [approved target](README.md). See [contracts](contracts.md) for online types.

## P02: episode repository and views

**PR/AD:** preserve native PyG and safe inference NPZ. Added episodes use a separate
schema `icgs_episode_v1`, JSON manifests plus numeric compressed NPZ shards loaded
with `allow_pickle=False`. Large artifacts stay outside Git. Simulator snapshots
are separately stored opaque artifacts with recorded format/version and trusted
loader; never embed simulator objects in a neural sample.

**ID:** one episode manifest references content-hashed shards; each shard contains
at most256 intervals, plus its start boundary observation. Adjacent shard boundary
observations share the same identity and are counted once. Store timestamps,
pose/grip arrays, command targets/planned duration, achieved duration/substeps, and
ragged cloud values with offsets (or calibrated depth references). Preserve raw
foreground depth/cloud sufficient for both preprocessing paths, not only FPS
clouds. JSON writes reject NaN/Infinity; absent measurement is explicit null plus
reason. Record writer uses temporary file + atomic rename and validates checksums
before publishing the episode manifest; incomplete episodes remain quarantined.

| Manifest group | Required provenance and fields |
| --- | --- |
| Identity | episode, program, source/seed lineage, asset families, split, generator version, code revision/dirty patch |
| Environment | engine/solver/assets/controller versions, action mode, timestep/substeps, robot, cameras, calibration, gravity, crop/mask protocol |
| Trajectory | boundary timestamps, measured observations, absolute commands, achieved state/duration, valid-point masks, controller status |
| Causal state | full-history pointer, anchor/replay spec, cached encoder/memory/context hash and causal end boundary |
| Annotations | event mapping, occurrence/current relation/eligibility, validity masks, local outcome, terminal precedence/times |
| Branch/outcome | anchor/context/candidate IDs, exact raw/materialized commands, prefix duration, remaining deadline, reference ID, independent seeds, trial traces, k(H),n |
| Matched/audit | physical scene-pair/suffix-pair IDs, intervention ID, dependency span, replay report, approximate/exact status, rejection reason, observed/imagined origin |

No program/snapshot/oracle fields enter online records or model input tensors.
Normalization statistics are training-only; object scale/clearance is never
normalized away. Cache keys include all generating model/preprocessing/reference
identities and causal history; an incompatible cache is rejected, not reused.

Views are selections of a shared repository, not additive dataset counts:

| View | Sample / target | Consumers |
| --- | --- | --- |
| geom | observed cloud/mask → itself | A0 |
| dyn | causal prefix + executed commands → K executed successor observations | A1/D1 |
| task | independent context + query prefix → alignment/rho/nu/eligibility/masks | B |
| terminal | active transition or observed state → first event/current completion | D2 |
| value | active anchor-context-H-reference + trials → count-weighted outcomes | D2 |
| pair | equal-deadline active successors + context → confident preference | D2 |
| calib | independent development random/planner-selected anchors → calibration only | E |
| audit | held-out fixed candidate pool + dense trials → empirical ranking/support | evaluation only |

## Programs, splits and predicates

**PR:** split before generation:20 training skeletons T01–T20,4 development V01–V04,
12 locked test P1–P4/G1–G4/R1–R4. The exact program catalog is in proposal section
Program Catalog for the Composition Split, and test skeletons in Primary Benchmark:
Cross-Stage Dependencies; [requirements](requirements.md) maps both to P02.
The generator manifest must transcribe those catalogs, not infer task IDs from
filenames. Asset families split70/15/15 before scaling/augmentation. A seed and
all descendants stay in one split. Physical anchors retain their split through
context swaps. Query episode differs from all context episodes.

**ID pilot selection:** T06/T08 (packing), T09/T11 (grasp/final orientation),
T13/T14 (parking/retrieval/restoration). These are six existing training catalog
entries, not a new held-out suite. Pilot certifies each selected program via expert
execution; if an asset/program is infeasible, record a protocol revision before
substitution. Pilot does not open test compositions.

Primary training uses2–4 stages, not strict2–3→8 length extrapolation. The separate
length track excludes all longer training episodes/branches/continuations and
tests4/6/8 stages under a predeclared1024 deadline. Dependency-mechanism holdout,
asset holdout and configuration generalization are reported independently.

**PR custom predicates:** grasp/lift >=3 cm and relative stability <1 cm/10 degrees
for5 intervals; stable placement released with support and speed<1 cm/s for5;
inside uses full oriented bounding volume with2 mm margin; drawer closed within
5 mm for5; fit/final/restored relation <=1 cm/10 degrees and required support;
full success all current and mandatory-history predicates for5 intervals.
Unsafe terminal includes leaving workspace/controller failure and validated force/
penetration limits. Proposed40 N for3 intervals and2 mm penetration for3 exclude
declared grasp/support solver artifacts only with measured justification.

**FG:** verify force/penetration observability and thresholds in the exact engine
before freezing the benchmark; do not fabricate unavailable force signals. Public
benchmarks retain official scoring. Failed grasp is recoverable, not automatically
terminal. Failure wins simultaneous first success/failure; absorbed success remains
terminal. Annotation monitors are external to deployed networks.

## Collection and event annotations

**PR:**5 seed demos/program, two valid execution modes when possible; primary200
successful contexts/program, retaining all valid failed attempts. Generation stops
at512 intervals. Re-execute transformed contact motion: object-relative waypoint
adaptation is `T_object_new @ inv(T_object_seed) @ T_ee_seed`; free-space connectors
use motion planning, never teleportation. IP geometric pseudo-contexts cannot
provide physical transition labels.

Dynamics episode mixture after reference freeze:50% executed demo attempts,
30% reference,20% bounded perturbation/recovery; warm-up70/30 demo/perturbed.
Perturbations: target <=2 cm/10 degrees, grip timing +/-1 interval, stationary
object displacement <=3 cm, declared blocker insertion; execute physics and record
intervention ID and pre/post observations. Pauses2–10 intervals and retries are
executed, not timestamp edits.

Event-to-primitive mapping takes greatest temporal overlap, requires>=50% token
interval overlap; ties **ID:** earlier primitive occurrence, with uncertain semantic
labels masked. Alignment is uniform over corresponding valid events across demos;
unrepresented recovery gets null. rho is historical, nu recomputes current state,
eligibility is current prerequisites. Unidentifiable/free-space postconditions are
masked, not invented. Include displacement with rho=1,nu=0. Progress pairs require
strict subset of satisfied requirements with no loss; time order alone is not a
progress label.

## P08: anchors, replay and continuation targets

**PR:** anchor contains physically executed state, full causal history, controller
state and deadline. Restore physics/controller/task monitor together; reconstruct
model memory from observation history. Snapshot inventory includes joints/velocities,
object/articulation poses, grip/attachments, integrators, task history, RNG and
solver state where available. Replay the same prefix twice and report cloud/pose/
outcome discrepancy before collecting counterfactuals. Missing exact state uses
reset/history replay; inequivalent restoration is approximate repeated-reset data.
Never mark approximate pairs as exact interventions.

Primary2000 anchor-context records:40% delayed-consequence decisions,20% boundaries,
20% recoverable errors,20% random. Matched-suffix quota30% overlaps this positional
stratification; it is not an extra600 anchors. **ID:** stratify these two dimensions
jointly and record achieved counts, not implied exact balance when cases are scarce.
K8 candidate prefixes, lengths2/8/16; **ID:** uniform per-anchor prefix-length draw
among feasible lengths, all comparison candidates at an anchor share that length.
Long prefixes resample IP using actual observations between committed groups.
Store actual commands; downstream counterfactual use replays those common commands.

Terminal prefix yields realized return1/0 with no active-state V successor label.
Active successor gets n4 independent pi_ref continuations with separate diffusion/
perturbation seeds, until first terminal or deadline. Construct applicable H values
32/64/128/256/512 from the same first-event trials. These labels are correlated
across H; they are not additional independent trials. **ID:** trials interrupted
by infrastructure failure are censored beyond their observed coverage; keep
reason/count and do not treat the interruption as a physical failure. Protocol
timeouts are observed failures at their collection deadline, not infrastructure
censoring. Store each trial's `observed_through` boundary and collection deadline.
A nonterminal trial observed only through32 supports timeout-as-failure at H32,
but supplies no label at H512. Include a trial in n(H) only if it was observed
through H or a physical first terminal was observed earlier; success contributes
only when its first-success time<=H. A protocol timeout is not an absorbing physical
failure that can be extrapolated to longer H. Expert and reference trials use
separate controller IDs/views.

For context swap reuse physical prefix only if feasible; recompute q from the full
same history and collect separate continuations for each C. An apparent ranking
reversal is only labeled when executed posterior evidence supports both directions.
Recovery requires at least one successful executed recovery under allowed budget;
distance from demos is not a recovery label.

## Loss definitions and pair independence

**PR:** symmetric CD, normalized by(0.01 m)^2. Physical loss adds translation MSE
with that same scale, geodesic rotation squared/(5*pi/180)^2, and grip BCE-logits.
Geometry reconstruction is CD of observed versus decode(encode(observed)).

```text
L_WM = sum_i,m bootstrap[i,m] * sum_k Lphys[i,m,k]
       / (Kroll * sum_i,m bootstrap[i,m]) + 0.1 * Lrec
L_task = masked_CE(alpha,y_alpha) + sum_z masked_BCE(z,y_z)
L_out = sum_i n_i * BCEWithLogits(logit_V[i],k_i/n_i) / sum_i n_i
L_S = BCEWithLogits(logit_S,y_complete)
L_E = CE(event_logits,first_event)
L_Phi = mean_valid softplus(Phi_earlier-Phi_later)
L_pair = sum_pair weight * BCEWithLogits(logit_Va-logit_Vb, preference)
         / sum_pair weight
L_eval = L_out + L_S + L_E + 0.1*L_Phi
         + 0.2*(L_branch + L_recovery + L_suffix)
```

Each task loss averages valid entries independently before summation. No-valid
auxiliary pool contributes differentiable zero. Value samples require n>0.
Episode bootstrap Bernoulli0.8, force at least one head per episode and supervised
samples per batch. **ID:** if all zero, choose head from episode-hash modulo3;
persist mask seed. Fixed head unroll K uses predicted inputs after the first step.
Reconstruction term is omitted after encoder/decoder freeze; a constant frozen
reconstruction term is not described as trained loss.

Independent branch counts use Beta(k+1,n-k+1); compute P(va>vb) by deterministic
SciPy quadrature (**ID**, absolute/relative numerical tolerance1e-8). Retain max(p,1-p)
>=0.9, weight2*abs(p-0.5). Equal distributions should yield0.5. Shared-random-number
trials instead use paired bootstrap with stored indices/seed, not independent Beta.
Pair pools are disjoint: **ID precedence** suffix first, then recovery, then general.
Full count likelihood still trains all eligible records. Do not interpret logit
difference as physical probability that one branch is better.

## P11: trainable/frozen phases

| Stage | Trainable | Frozen / data and selection |
| --- | --- | --- |
| A0 | physical point encoder + decoder | IP frozen; geom; reconstruction plus paired native-action fidelity |
| A1 | encoder/decoder + physical GRUs + pilot dynamics | IP frozen; executed dyn, K1/2/4 curriculum; temporal gradients train memory |
| B | event encoder + task GRU/heads | IP/physical encoder/GRUs/decoder frozen; task labels; masked loss and stage-aware execution |
| C | none | freeze entire reference; collect exact-reference outcomes |
| D1 | dynamics trunk/heads | encoders/decoder/all memories/IP frozen; K2/4/8/16 recursive executed dyn |
| D2 | evaluator and terminal heads | entire reference path frozen; observed outcomes/terminal/pair losses |
| E | positive scalar Tv,Ts,Te | all neural weights frozen; disjoint calibration partition only |
| Test | none | fixed context panels/resets, no test gradients or archive adaptation |

**AD:** learned completion stopping is not part of C's reference. Its neural head
can train at D2 without invalidating reference outcomes; deploying its stop rule
changes the executed wrapper and must be reported separately.

**PR proposed optimizer:** AdamW lr1e-4, betas(.9,.999), wd1e-2, warm-up2000 updates,
cosine to1e-5, global grad clip1. A0 batch64 max50k updates; A1 batch8 sequences,
8 burn-in +16 supervised max100k; B batch8 context/query sequences,16 supervised
max100k; D1 batch8 max200k; D2 batch128 anchor-context-H +64 terminal +up to128 pairs
max100k. Curriculum phases divide stage maximum updates equally. Eval every5k,
stop after5 evaluations without improvement. Three training seeds, separate
generator/reset/action seeds; all actual seeds are recorded.

History before burn-in is replayed from reset with current weights under no-grad;
detach hidden state before the supervised unroll. Frozen full-history states may
be cached with lineage. Do not reset memory at arbitrary sampled clip starts.
Outcome sampler chooses anchor-context first, then one valid horizon; retain
trial-dependence IDs. Class-balanced sampling must carry inverse inclusion weights
when recovering population likelihood/calibration.

**ID selection resolution:** use lexicographic metrics, no weighted blend selected
on test: A0 normalized CD then native translation/rotation/grip drift subject to
the bridge gate; A1/D1 normalized multi-step physical loss then development
branch-ranking diagnostic when available; B masked task loss then stage-aware
success; D2 binomial NLL then finite-pool regret. A1 before evaluators exist reports
ranking diagnostic NOT RUN and selects physical loss alone; record this phase
limitation rather than invent a value model. Ties require numerical equality at
recorded metric precision; earlier checkpoint wins. Record priority before runs.

Temperatures are exp(logT), optimized in FP64 NLL on calibration only; **ID:**
initialize logT0, optimize deterministic LBFGS max100 iterations, line search
strong-Wolfe, reject nonfinite result and retain identity with failure report.
H0 stays exactly0 after calibration. Observed, imagined and planner-selected
calibration are reported separately; fitting on observed states proves nothing
automatically about model-generated states.

## Pilot budgets and scaling gates

| Item | Pilot | Primary proposed ceiling/target |
| --- | --- | --- |
| Train programs / successful contexts each | 6 / 50 | 20 / 200 |
| Anchor-contexts / candidates / active continuations | 200 / 4 / 2 | 2000 / 8 / 4 |
| Maximum branch / continuation records | 800 / 1600 | 16000 / 64000 |
| Calibration | preliminary only | 250 anchors *8*16 =32000 trials |
| Dense audit | 20 anchors | 100 anchors *8*32 =25600 trials |

Terminal branches reduce trial counts. Development compositions split by lineage
into selection and calibration before training. Dense audit is never training.
Planner hard examples are collected only on training programs; maintain50% original
bank minibatches, retain frozen reference, record growth and retrain equal-data
controls. This is a separately scoped collection round.

Branch-step estimate Na*K*(mean_prefix+n*mean_continuation): primary example
mean_prefix8, mean_continuation128 gives8,320,000 intervals; length512 gives
32,896,000. Raw2048x3 float32 cloud is24,576 bytes; first example ~204.5 GB before
metadata/compression, excluding generation/replay/calibration/audit. Measure
physics/render/reset throughput, acceptance, storage and memory on pilot hardware;
do not infer machine-hours from another engine. Main-scale launch requires an
approved resource estimate and passed [roadmap gates](../plans/active/icgs-method-implementation.md).
