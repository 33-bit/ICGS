# ICGS method implementation roadmap

Status: **active — runtime implementation not started**.
Owner: repository owner / assigned implementer per component.
Category: C architecture/research migration.
Authority: owner-approved documentation/component-plan brief, 2026-09-07.
Spec: [target method](../../method/README.md); source: [proposal provenance](../../proposals/README.md).

## Goal and current/target states

Build the proposal's complete primary method and B0–B7 controls while preserving
native IP and its scoped published compatibility. Current source at adoption
`4282700` has native inference, proposals, command conversion and context/RNG
foundations, not trained ICGS models/search/collector. See [current architecture](../../ARCHITECTURE.md)
and [status matrix](../../components/v5-foundations.md).

This master owns dependency/release gates only. Each linked component owns its
implementation tasks and evidence. Creating the plans is not completing runtime
work. No duplicate story database or experimental results are created here.

## Component dependency index

| Plan | Deliverable | Prerequisites / release coupling |
| --- | --- | --- |
| [P00](../../plans/active/icgs-p00-contracts.md) | contracts/config/composition seams | current native contracts; no new runtime stub claims |
| [P01](../../plans/active/icgs-p01-timed-execution.md) | timed execution/sensors | P00; G1 controller protocol before C |
| [P02](../../plans/active/icgs-p02-episode-data.md) | executed episodes/tasks/labels | P00/P01; G2 assets/splits/monitors |
| [P03](../../plans/active/icgs-p03-geometry-bridge.md) | blocks/physical encoder/decoder | P00, synthetic geometry; observed data P02 for A0; G3 bridge |
| [P04](../../plans/active/icgs-p04-physical-memory.md) | causal physical history | P00/P03; A1 pilot dynamics from P07 |
| [P05](../../plans/active/icgs-p05-event-memory.md) | event segmentation/encoding | P00/P03; P02 annotations for B |
| [P06](../../plans/active/icgs-p06-task-router.md) | tracker/router/frozen reference | P04/P05; G4 native D1/D2; trained A1/B before freeze |
| [P07](../../plans/active/icgs-p07-world-model.md) | physical dynamics/recursive rollout | P03/P04; pilot dynamics available for A1; P06 task update only outside physical model |
| [P08](../../plans/active/icgs-p08-outcome-collection.md) | replay/branch/reference bank | P01/P02/P06 frozen; G5 replay; no dependency on D2 learned stopping |
| [P09](../../plans/active/icgs-p09-evaluators.md) | value/completion/hazard/calibration | P04/P05/P06 representations; P08 outcomes for D2 |
| [P10](../../plans/active/icgs-p10-search.md) | rerank/shooting/MCTS | P00 analytic tests first; actual P06/P07/P09 before learned deployment |
| [P11](../../plans/active/icgs-p11-training.md) | stages/optimizers/resume artifacts | build stage infrastructure alongside its first consumer, not after P10 |
| [P12](../../plans/active/icgs-p12-controls-evaluation.md) | B0–B7, metrics/benchmark analyses | metric/task fixtures early; trained models/data for real comparisons |
| [P13](../../plans/active/icgs-p13-integration.md) | installed policy/CLI/pilot release | all required components; G6–G8 release gates |

No dependency cycle: P07's physical pilot model requires P03/P04, not a trained
router/evaluator; it supplies A1 gradients. P11 A0/A1/B stages are implemented as
their components arrive. P08 restoration interfaces can be tested before reference
freeze, but actual C outcome labels require frozen P06. P10 toy collaborators test
search bookkeeping, never certify learned behavior.

## Invariants and approved decisions

Read [ADR0006](../../decisions/0006-icgs-target-boundaries.md),
[ADR0007](../../decisions/0007-icgs-timed-data-lineage.md), and
[ADR0008](../../decisions/0008-reference-value-and-stopping.md).

- Keep `src/icgs` canonical; no runtime dependencies on docs/tests/harness scripts.
- Preserve native IP weights, preprocessing, graph/action/sampler/RNG and strict
  published loading; no altered baseline hidden behind unchanged identifier.
- New physical preprocessing is separate from native IP; measured-root IP does
  not receive an autoencoded cloud.
- Real versus imagined memory, physical versus task state, commanded versus
  achieved pose/grip/time, and privileged annotations versus online input differ.
- Reference=router/IP action policy without learned stop. Freeze its complete
  lineage before C; changed reference requires new outcome labels.
- Learned stop is shared deployed wrapper B1–B7, with false-stop/disabled controls.
- B3 separate diffusion student uses matched data/outcomes; no IP fine-tuning.
- Search integrates common-command outcomes; no optimistic head choice, progress
  reward, double completion, approximate MCGS merge or cross-root W reuse.
- Native formats remain; method records/checkpoints/configs are versioned separately.
- No training/download/preprocessing/simulator/robot workloads as incidental tests.

## Release phases and acceptance gates

- [ ] Foundation: P00/P01/P02 synthetic tests; select supported Linux model environment;
  freeze G1/G2 protocols before any main collection.
- [ ] Representation: P03/P04/P07 pilot model plus P11 A0/A1; pass G3 bridge,
  train P05/P06 B, pass G4 and freeze reference manifest.
- [ ] Executed labels: pass G5 replay; authorize bounded pilot P08 collection.
- [ ] Models: D1/D2, observed-state evaluation/calibration, freeze artifacts;
  train matched B3/B4 on the same bank.
- [ ] Decision stack: P10/P12/P13 installed integration; pass G6 native regression
  and G7 six-program pilot. Preserve failed runs and negative scientific results.
- [ ] Scale: approve G8 resource report and frozen splits/metrics; only then launch
  primary collection/benchmarks under separately granted resource scope.

| Gate | Required measured artifact | Stop condition / next action |
| --- | --- | --- |
| G1 controller/time | exact engine/controller/camera/gravity/crop/mask manifest, target-hold trace with dt/substeps, grip commitment, safe hold, paused/live/wall-deadline policy | Missing capability or future completion time used as input: stop timed collection; implement/validate adapter or explicitly version a different protocol |
| G2 task/splits | concrete asset/version/parameter/seed catalog, family/lineage split, expert success per accepted instance, predicate observability and tolerance sensitivity | Infeasible/ambiguous instance, unsupported force signal or split overlap: quarantine; owner approves revised manifest before generation |
| G3 geometry bridge | observed vs reconstructed same-seed IP actions, CD/pose/grip and local-success discrepancy, sparse/disocclusion cases | Without approved development tolerances and passing evidence, no imagined-IP efficacy claim; fix/train bridge, do not silently loosen tolerance |
| G4 reference sessions | strict shared-checksum D1/D2 native runs, full/window waypoint fidelity, repeatable route/diffusion seeds, reference fingerprint | Duplication to satisfy D, stale session scratch or incompatible window: stop reference freeze and fix adapter |
| G5 replay | actual snapshot inventory, repeat-prefix cloud/pose/outcome discrepancies, exact/replay/approximate classification | Unsupported exact restore: try deterministic replay; otherwise label approximate and restrict exact-intervention claims, never fabricate equivalence |
| G6 compatibility | L1 executed counts plus C1 artifact/config,C2 strict load,C3 real inference,C4 reference fidelity,C5 installed independence | Required check SKIPPED/FAIL keeps migration active; use provisioned reference/native environments, not fake imports |
| G7 pilot | six training programs, contexts/attempt failures, timing/storage/replay/support/bridge/model/calibration/search traces, B0–B7 feasible-control reports | No meaningful supported actions, unbounded errors or unexplained optimistic exploitation: diagnose; do not scale to conceal the failure |
| G8 primary launch | measured throughput/render/reset cost, actual trial counts, storage/compute estimate, frozen protocol/splits/metrics, approved execution budget | Missing estimate/tolerances/owner scope: do not launch main collection; retain pilot evidence and request explicit scope |

Thresholds for G1/G2/G3/G5 that require physical calibration are **FG**, not blanks
for implementers to guess: the pilot first produces a development discrepancy/
feasibility report, then the repository owner approves numeric tolerances in a
versioned protocol **before the gate is judged or primary/test workloads run**.
Local numeric invariants (shapes, conservation, masking, units) have exact tests.
No target success rate is invented. Hypotheses may fail while correctly implemented
experiments succeed in producing honest evidence.

## Validation and experiments

Owner: [validation guide](../../../tests/README.md). Component commands use flat
`unittest discover`, installed ICGS and actual dependencies. L1 small tensors and
short optimizer steps do not authorize training. L2/C1–C5 need the trusted published
artifact and pinned Linux/CUDA environment; existing historical C1–C5 evidence is
not a fresh test of future changes. L3 timed/replay/task pilots and L4 benchmark
runs require explicit resource/task scope. No SKIPPED-as-PASS aggregation.

For each real run, create the [research record](../../experiments/TEMPLATE.md)
with code/dirty patch, command/cwd, resolved config, actual seeds, environment,
hardware, dataset/split/reference/checkpoint hashes, failures and comparison.
Do not create an experiment record merely because this roadmap names a gate.
Report planned versus actual episode counts, wall/sim time and incomplete runs.

## Compatibility and recovery

New components are opt-in named method profiles. Preserve baseline and original
artifact bytes; reject incompatible caches/checkpoints before model IO. Resolve
shared contract edits in P00 before downstream plans diverge. New files/packages
must not exist merely to satisfy harness shape checks; tests follow runtime owners.

If a component fails, keep the native path usable, disable only the new profile,
retain original/failure artifacts and reproducibility metadata. Revert only the
scoped component change after checking user work; never reset the whole checkout.
Changing a scientific choice requires a named configuration/protocol revision and
new labels/checkpoints where affected. Optional extension work is excluded from
primary completion unless later explicitly added.

## Documentation-delivery evidence

The documentation package is implemented separately from this still-active runtime
roadmap. See [delivery record](../../plans/completed/icgs-proposal-documentation.md)
for actual documentation validation. All P00–P13 runtime checkboxes stay unchecked
until their software/evidence exists. The roadmap cannot move to completed merely
because every plan file has been written.
