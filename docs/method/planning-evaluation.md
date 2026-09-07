# Planning, controls and research evaluation

Status: target, not implemented search or measured benchmark results. PR refers
to the proposal's Budgeted Search, Benchmark and Reproducibility sections. AD/ID
have the meanings in the [method index](README.md).

## P10: belief bookkeeping and search

**PR:** each node is an open-loop set of three fixed-head predictive hypotheses,
not a selectable outcome. Root `tau=0, U=S(root), F=0, w_m=(1-U)/3`. Each action is
sampled at a representative hypothesis, made absolute once, then evaluated on all
heads. Preserve the same physical command, not the same relative descriptor.

**ID medoid:** among hypotheses with positive active mass, minimize the unweighted
sum to other active hypotheses of `CD/(0.01 m)^2 + ||delta_t||^2/(0.01 m)^2 +
angle(delta_R)^2/(5*pi/180)^2`. Tie by lower head ID. This normalization resolves
the proposal's unspecified normalized pose/cloud metric; do not call it learned
or optimal. Poses/clouds use current achieved predictions, not target commands.

```python
def propagate_mass(U, F, weights, event_probabilities):
    success, failure, cont = event_probabilities.T
    return (U + (weights * success).sum(),
            F + (weights * failure).sum(), weights * cont)

def leaf_return(U, weights, active_values, remaining):
    return U if remaining == 0 else U + (weights * active_values).sum()
```

Every interval updates physical prediction, task memory, then terminal hazards.
Check U+F+sum(w)=1 in FP32 or higher; never add leaf S again, never maximize over
head outcomes. Reject invalid probabilities; do not silently normalize away a
model/API defect. Numerical assertion tolerance **ID:**1e-5 for FP32 mass sums;
tests in FP64 use1e-12. Terminal mass is exhausted when all active weights are
exactly zero, not an arbitrary pruning threshold. Deadline/cap truncation uses
`min(h,H_root-tau,L-tau)`; partial final edges carry their actual command count.

MCTS edge holds absolute prefix, cached child, visits N_e and total root-relative
return W_e. Progressive widening `max(1,floor(1.5*sqrt(1+N_node)))`; add one
candidate per expansion. Unvisited edges first, then
`W_e/N_e + sqrt(log(1+N_node)/N_e)`. Do not evaluate UCT at zero visits. Simulation
stops at a new leaf/depth cap, computes G once, backs it through every traversed
edge and increments each node exactly once. Caching does not create a visit.

**ID ties:** earlier candidate insertion ID for selection ties; final root choice
most visits, then greater mean return, then earlier ID. Preserve duplicate samples
in candidate logs; reuse transition computation only for byte-identical canonical
command arrays/durations and identical parent state/context/head/model/time IDs.
Exact cache is within a single planning root. No approximate latent merges, no
cross-root W reuse, no dropping deadline/task history from keys.

Root reranking evaluates h or full native T8 then leaf V. Shooting repeatedly
resamples the same prior at predicted nodes to L and chooses highest G. MCTS shares
the same prior, heads/evaluator, commitment, precision and budget accounting.
**ID shooting ties:** first completed sequence. Do not describe L32 as cross-stage
without reporting actual semantic interaction boundaries crossed.

## Budgets, fallback and stopping

**PR:** primary equal-wall-time budgets0.1/0.5/2 s, diagnostic native-call caps
16/64/256 plus separately reported physical model intervals. Count input encoding,
native preprocessing/denoising, rollout decode/re-encode, evaluator, transfer and
device synchronization. Cache preparation at episode start is reported separately
and included in total episode wall time. No claim that one IP call fits0.1 s.

**ID:** use monotonic wall clock; check before each noninterruptible operation and
after completion, logging overshoot. Native calls cannot be assumed preemptible.
Only evaluations completed by budget expiration are eligible as search results;
late results remain audit data. If none completed, use one reference sample;
reuse the earliest sampled reference-prior candidate if one exists, else make
one new sample and record fallback latency beyond budget. Invalid input/context
aborts rather than generating a fallback from fabricated geometry.

Execute at most r intervals and remaining episode deadline. The environment owns
external safety/task termination; online policy sees no test predicates. New real
observations update memories once per interval; after r build a fresh root.

**AD:** shared learned stopping applies to B1–B7 after `S>=0.95` at three consecutive
observed boundaries, never imagined nodes. **ID:** reset count0; reset the streak
on S<0.95 or invalid input, do not count repeated reads of one boundary. S is the
calibrated completion output; stop rule and temperature have their own IDs. The
proposal's threshold is the fixed primary default; any validation-driven threshold
change is a protocol revision fixed before test. The evaluator scores a false stop
as failure. Report the matched stopping-disabled control so B0/B1 cadence effects
are not mistaken for completion-head effects. Reference outcome collection never
uses learned stopping, regardless of the deployed B2 wrapper.

**FG live timing:** paused simulation and live-clock results are different tracks.
In quasi-static primary, hold a predetermined controller-safe target while waiting
and count measured waiting time toward completion time. Record physics intervals
and wall deadline independently. Freeze the wall-to-control-deadline/stale-command
policy after P01 controller pilot and before outcome collection; if the environment
cannot advance during planning, label it paused and do not claim live-clock proof.
No arbitrary robot-safe hold target is invented by the generic planner.

## P12: baseline ladder

| ID | Defined behavior | Controlled question |
| --- | --- | --- |
| B0 | existing published native IP, native cadence/context/sensors/evaluator | unchanged checkpoint reference |
| B1 | full-context IP, ICGS sensor/cadence; shared stop gate with disabled-gate control | wrapper versus native |
| B2 | frozen routed IP action policy, shared external learned stop gate | history/task routing |
| B3 | separate diffusion student below, no online sampling/ranking from IP | equal-data amortized action choice |
| B4 | direct action-value Q reranks the same native candidate pool | prediction-free action scoring |
| B5 | learned WM+V root rerank, L=h and L=T | value and single-chunk prediction |
| B6 | shooting/MPC, matched L and budget | sampling action sequences |
| B7 | progressive-widening MCTS, matched L and budget | tree allocation |
| B8 | separately reproduced ManiLong-Shot only with official assets/protocol | external competitor |

B2–B7 use the same sensors, context panels, physical/task encoder artifacts,
reference version and eligible executed data. B3 is a distinct learned policy,
not the same action prior; B4/B5–B7 use matched native pools where the experiment
requires one. All learned baseline training counts and supervised inputs are
reported. No hidden oracle-stage input or extra outcome collection for one method.

## B3 diffusion-student implementation default

**AD:** a separate history/context-conditioned diffusion student; frozen native
IP is unchanged. **ID design, not stated in the original proposal:**

- Input context = valid physical tokens S, event keys K and task r from the shared
  frozen encoders/tracker. No H or oracle labels at inference. Eight action tokens
  of width256, learned horizon-position embeddings and projected diffusion-time
  sinusoidal embedding (128 sine/cosine frequencies combined to256).
- Four pre-LN decoder blocks: action-token self-attention, cross-attention to
  `[S;K;r]`, FFN1024, width256/eight heads/dropout0. Prediction head256→256→10
  estimates diffusion epsilon. Blocks use the shared numerical conventions but
  have independent student weights.
- Per-target vector10 = translation3 relative to proposal root divided by0.08 m,
  rotation first-two-columns6, and normalized commanded grip +/-1. Rotation means
  root-relative rotation, not SE(3) log rotation. Decode Rot6 with Gram-Schmidt;
  reject degenerate norm<1e-6 or nonfinite output as a model failure, then use
  reference fallback and count the failure. No arbitrary decoded transform clamp.
- Carry `action_valid [B,8]` through training forward: zero missing command values
  before noising, zero their noise and token rows, mask them as self-attention
  keys, and zero invalid query rows after each block. Loss uses the same mask.
  Inference sets all eight positions valid. Perturbing stored invalid tails must
  not affect valid predictions under a fixed noise seed; loss masking alone is
  insufficient. At least one target must be valid in every training example.
- Train on physically executed branch commands (not achieved poses/unexecuted
  tail), first min(prefix length,8), with horizon validity mask. Available prefix
  outcomes are1/0 for terminal prefixes and k(H)/n for active successors under
  the frozen reference at the recorded remaining deadline. One valid H per
  anchor-context sample; H is a training-weight source, not a student input.
- Outcome-weighted behavior cloning: `weight=0.1+0.9*return_estimate`; weighted
  mean epsilon MSE over valid target elements, normalize by sum of per-example
  weights. This positive floor retains failed branches; no fictitious action
  ranking or expert continuation probability is added. Same bank/version as B4–B7.
- Independent 100-step squared-cosine DDPM noise schedule, epsilon prediction;
  deterministic DDIM eta0 with4 inference steps, no sample clipping. This is the
  student schedule, not a rewrite of IP's preserved sampler. Initial noise seed
  is recorded. Sample one trajectory, materialize the same grip/cadence rule and
  execute r2; no candidate reranking or world-model query online.
- AdamW/default schedule from [training](data-training.md), batch128 branch
  records, max100k updates, validation weighted denoising loss then matched
  development success. Report training/inference costs. Separate unweighted
  student ablation tests the outcome-weighting choice without extra data.

This is return-weighted imitation, not proof of optimal policy improvement or
calibrated student value. The student can generate actions outside the finite IP
pool; compare unconditional success/fallback as well as data/compute cost.

## B4 direct Q and fair comparison

**PR:** encode an h-command sequence with width256 GRU and combine with evaluator
query to output Q_H(b,q,C,A). **ID:** compute all action descriptors relative to
the observed root pose (no predicted future poses); include planned duration and
held commanded grip. Use last valid GRU state; concatenate the value query before
its final scalar head, then MLP512→256→1. Train BCE-logits on realized prefix+
reference return, weight active records by n and terminal records by1, normalize
by weight sum. Q denotes prefix return; terminal prefixes are legal Q labels even
though they are not active successor V labels. Use identical h/H/candidate pool
and temperature calibration on its own held-out logits, never test data.

For Q labels, distinguish `H_root`, actually executed prefix length `ell`, and
`H_successor=H_root-ell`. An active successor's `k(32)/n` after an eight-interval
prefix labels `Q_40`, not `Q_32`. Candidate execution truncates at H_root; terminal
prefix labels use the realized terminal outcome. Reject inconsistent records and
H_root beyond the checkpoint's trained coverage rather than giving B4 extra time.
The same root/successor conversion applies to B3 outcome-weight provenance and
WM+V comparisons, even though B3 does not consume H online.

B5 h2 versus T8, B6/B7 L8/16/32/64 use matched commitments r. Reporting tree
advantage requires equal-L and equal-time comparisons, not a longer horizon
given only to MCTS. Native-call and model-interval counters are separate because
equal policy calls do not imply equal model work.

## Experiments, ablations and reporting

**PR primary:** ICGS-Dependencies is a proposed custom suite, not a downloadable
existing benchmark.12 held-out compositions,100 matched resets each,2 demos,
H512,3 training seeds →3600 episodes/method; all three planner budgets →10800.
One-demo ablation50 matched resets/composition. Native-only methods share reset/
context/action seeds but are not replicated as independent trained models.

| Experiment | Required comparison / outputs |
| --- | --- |
| E1 regression/stage | B0/B1/B2 native/easy controls; local success, phase accuracy, premature advancement, stopping control |
| E2 value semantics | realized branches; temporal/local/outcome/full targets, correct/oracle-stage diagnostic; regret/NLL/calibration |
| E3 suffix dependence | same physical history/candidates, distinct identifiable suffixes; full/no/shuffled context, no suffix loss; executed reversal evidence |
| E4 lookahead | B4/B5/B6/B7 equal prior/data/evaluator where applicable, equal time/L; B7–B6 primary confirmatory contrast |
| E5 dependency span | fixed feasible J with varied delta; success/regret and actual imagined boundaries |
| E6 transfer | disjoint compositions/assets and separately trained strict-length track |
| E7 model exploitation | increasing budget, fixed held-out audit; predicted-minus-realized return/regret |
| E8 recovery | same executed perturbations; unconditional success, recovery success/latency |
| E9 latency | same hardware; paused versus live, p50/p95/stale/fallback/overshoot rates |

Diagnostic consequential-choice set uses K16 common candidates, at least two
locally successful branches, separated uncertainty intervals with expected gap>=0.2.
It does not filter difficult episodes from end-to-end success. Distinguish this
from the K8 dense-audit bank. Build feasible (J,delta) cells J4/6/8,delta1/2/3;
report impossible cells N/A. Independent-chain and memory-only controls prevent
stage recognition from masquerading as downstream search benefit.

Minimum retrained ablations: no physical history/pooled geometry/six-IP-token
diagnostic; pooled events/no task GRU/monotonic alignment; temporal/local/outcome/
suffix value targets; pair loss off versus suffix data removed at equal count;
persistence/one-step/recursive dynamics and1/3 heads; planners/L/r; observed versus
reconstructed versus predicted clouds. r1/2/8 only where r<=h is valid. No zeroing
unseen-at-training features and calling it a fair retrained architecture ablation.
Optimistic max-head planning is a misspecification diagnostic, never deployment.

Oracle stage, simulator transitions and finite-pool empirical returns are privileged
diagnostics. No oracle value exists for an arbitrary generated cloud without a
physically matched state. Report finite-pool support and rate with no locally
successful candidate. **ID primary-cause order:** perception/interface → task/stage
→ no supported action → dynamics ranking → evaluator ranking → execution deviation
→ false terminal/timeout. Permit secondary tags and unknown when evidence cannot
distinguish causes; do not assert causal attribution from a single learned score.

Metrics: macro full-task SR; per-task counts; unconditional and conditional
downstream/recovery SR; time/intervals; masked alignment/null/rollback; count-weighted
binomial NLL; trial-level Brier; ECE10 equal-count bins; finite-pool regret with
trial uncertainty; CD/pose/grip/moving-object/contact diagnostics; unique prefixes,
calls/cache hits/coverage; p50/p95 latency, peak GPU memory, collision/invalid/
empty-perception/false-stop/fallback rates. Invalid runs remain explicitly reported
in requested-run accounting, not silently removed from the denominator.

**PR:** paired hierarchical bootstrap10000 resamples of compositions then
reset/context IDs, shared indices across methods; report95% intervals plus
training-seed mean/spread without treating seeds/frames as independent trials.
**ID:** average matched learned-seed results within reset/context for the paired
suite contrast, then bootstrap tasks/reset units; report individual-seed SR beside
it. Missing/failed runs remain visible and comparisons use predeclared matched
panels. Holm adjustment within predeclared comparison family; choose baseline on
validation. Effect sizes remain reported when nonsignificant.

## Optional extension gates

- **B8:** verify official code/assets/version/license/protocol and reproduce
  demonstration count/sensors; otherwise NOT RUN, not a homemade equivalent.
- **LIBERO/CALVIN:** separately version depth/calibration/embodiment/action adapters,
  demo panels replacing language, splits and official evaluator; adapted tracks
  are not official language-conditioned headline scores.
- **ManiSkill/MimicGen:** record engine/controller/contact-domain conversion;
  re-execute/replay with exact versions. Released trajectories are not automatically
  compatible RLBench dynamics, nor new independent transitions when re-rendered.
- **Length/mechanism/segmentation:** independent training coverage and protocol IDs,
  no test-derived threshold changes; ambiguous contexts reported separately.
- **Real robot:** separate owner authorization, protective controller/calibration/
  sensing/recovery procedure. Three families*30 resets*B2/B6/B7 gives>=270 trials,
 2 independent demos/task; any robot fine-tuning uses disjoint training setups.
  No simulation threshold is claimed to be a reliable real-force safety limit.

These are prerequisite roadmaps, not permission to download, run a simulator,
train extra variants or issue robot motion.
