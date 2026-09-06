# Unified ICGS v5 migration

Status: COMPLETED — C1–C5 passed on the final native wheel. Owner authorization: ICGS_v5_refactor_prompt revision
2026-09-06. No C1–C5 gate is waived. Category C; this task is execution, not another
request for approval of the already specified src layout.

## Scope and authority

One installed runtime `src/icgs`, IP as an internal policy/components, shared
contracts/config/composition/execution. Supersede only conflicting legacy namespace
and optional checkpoint-acceptance clauses. Preserve numerical baseline, diagnostic
loading, harness independence and explicit evidence. No new compute rental, push,
training, broad benchmark or physical robot actions. Existing resources and isolated
environments/dependencies/reference downloads/inference are authorized.

## Audit and references

- FACT: checkout HEAD is 32e177f92b2c5611d30a62b4c9de65e1ba04f965; origin is
  github.com/33-bit/ICGS. Only ip/checkpoints/ is initially untracked.
- Pre-migration committed source snapshot: /tmp/icgs-v5-reference.mSfx4M, from
  git archive HEAD. Checkpoint remains separately preserved; never delete data
  when removing old Python package files.
- Target reference: vv19/instant_policy checkout under the same temporary directory,
  verify revision 72281736672a555e9c9834eb15c225e2f1baaf93. It is not native runtime.
- Cached downloaded model.pt: 471777552 bytes, SHA256
  119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5.
  Source Drive file 1TM_zU1pVOqPuWZL3E9knNp4w-p7EBwwt, author download script.
- Restricted torch.load(weights_only=True) succeeds: Lightning dict with 674
  state keys, optimizer/trainer metadata, no hyper_parameters config. Encoder
  tensors are embedded. Architecture cannot be inferred from shapes alone.
- Current runtime constructor depends on missing sidecar config/encoder. It is not
  yet a verified loader for the published artifact. ICGS snapshot is a regression
  reference, NOT automatically the target oracle.
- Host is macOS ARM64, no CUDA. Docker client installed but daemon initially off;
  testing local Docker startup for Linux reference. No Colab sessions exist. Do not
  allocate one (current request forbids renting compute).
- Detailed standalone ICGS_proposal_v5.md not supplied. Prompt contracts govern
  foundation; learned methods remain deferred.

## Ownership mapping and invariants

ip/types → contracts (records) + state/context_cache (mutable lifecycle).
geometry → geometry/transforms,rotations,pointcloud. actions → diffusion/codec.
model encoders/graphs/backbones/denoiser → same semantic owners under icgs/models.
diffusion → algorithms/diffusion; policy → policies/instant_policy.
composition → composition; checkpoint IO → artifacts; config → configuration.
preprocessing/dataset → data; RLBench → environments/rlbench.
rollout → execution; metrics/protocol wrapper → evaluation; Lightning → training.
CLI → cli and icgs console entry. Audit/preserve unsupported occupancy code in an
appropriate training owner, never disguise it as working.

Preserve world/local conversion exactly once; root-relative action chunks (not
increments); normalized/commanded/achieved grip distinctions; batch B, horizon P,
candidate K; owner/cardinality/cache rules; scoped RNG restoration on exceptions.
No IP/instant_policy.so imports, sys.modules aliases or runtime sys.path hacks.
Timing may be unknown; do not invent controller periods. Neural v5/planners are
not implemented before published native fidelity is proven.

## Tasks and mandatory gates

- [x] Audit source/instructions and capture separate source references.
- [x] C1: provenance manifest, restricted container inspection, reference-exposed
  configuration and auxiliary artifact evidence.
- [x] Capture reference public inference/cold-warm RNG evidence in compatible ABI
  before broad migration; record any genuine environment/access blocker.
- [x] Build one src package, real config parser/CLI, semantic owners and installed
  wheel/editable tests outside checkout; preserve historical source via Git.
- [x] C2: native strict published weights load with complete mapping report.
- [x] C3: real native inference with nondegenerate two-demo fixture, repeated calls,
  reset, finite/shape/SE3/grip checks and no gradients.
- [x] C4: public target/native matched input/weights/settings, multiple seeds/cache
  states; compare error against reference-repeatability-based tolerance.
- [x] C5: installed wheel outside checkout without old package/binary/reference path.
- [x] Migrate execution/training/eval/CLI/tests/docs and retain original semantics.
- [x] After native gates: K sequential candidates, root-absolute prefixes, provenance/
  accounting and method-level v5 designed/implemented/deferred contracts.
- [x] Re-run all gates/regressions; only mark completed if C1–C5 pass.

## Validation and recovery

Updated authorization: owner explicitly permits Colab CLI for all inference work
in the follow-up. A named T4 session is now requested for reference/native gates;
stop/release it after validation. This supersedes the no-rental constraint only
for the requested bounded inference checks, not training/benchmarks.

Prerequisite evidence: Docker x86 emulation on the ARM host runs torch2.2/PyG but
Open3D0.18 import exits SIGILL (-4). Isolated import probes identify Open3D, not
checkpoint/model code. Task-owned container stopped before using Colab. Native
Python3.10/torch2.2 strict-load probe on the old source loads 337 model keys with
no missing/unexpected/conflicts/shapes; this is not C2 for src/icgs or C4 fidelity.

Use stdlib L0/actual test discovery and negative fixtures; real dependency tests,
generated/checkpoint tests distinguished from target gates. Keep exact logs and
artifact hashes. Runtime source root checks must move to src/icgs without empty
scan success. Do not weaken semantic assertions to pass migration.

Recovery: scoped working-tree diff against HEAD, source snapshot outside runtime,
checkpoint bytes preserved. No git reset/clean or broad deletion. Before-image
does not authorize overwriting later user edits. If gates truly blocked, retain
active plan, useful independently verified work and precise INCOMPLETE handoff.

## Final handoff — 2026-09-06

### Implemented
- Canonical src/icgs package with pyproject discovery, console/module CLI, explicit
  JSON config and runtime defaults; no ip package/shim, binary runtime or sys.path/
  sys.modules aliases. Old scientific source is recoverable from HEAD32e177f and
  external snapshot. Published weights moved without modification to
  artifacts/checkpoints/vv19/model.pt; ignored by Git.
- Native IP code moved into geometry, models/layers/encoders/graphs/backbones/
  denoisers, algorithms/diffusion, policies, state, data, training, execution,
  evaluation and artifacts. No monolith vendor/backend wrapper.
- Implemented infer/train/evaluate/prepare-data CLI. Help is dependency-light.
  Safe NPZ schema/fixture and strict published loader work outside checkout.
- Sequential K candidates, root-anchored absolute prefixes, source/context/seed/
  timing metadata, owner/cardinality checks, immutable demo content, isolated
  branch feature copies and exception-safe RNG. No production fake planner/value.
- Method v5 physical/task/world-model/evaluator/planner contracts and training
  roadmap are designed in docs; neural/planning implementations deferred.

### Target evidence and numerical corrections
- Reference actual binary strict-load PASS674 state entries; native strict-load
  PASS337 model-owned parameters after explicit equal alias mapping. No missing,
  unexpected, shape mismatch, conflicting alias, auxiliary config/encoder download,
  random fill or retraining.
- Public reference config/normalizer/scheduler captured on Python3.10.21,
  torch2.2.0+cu118, PyG2.5, Diffusers0.31, Open3D0.18, TeslaT4.
- Initial source-vs-target comparison FAILED (translation up to0.0443, different
  grip). Hooks established live voxel preprocessing and pre-encoding/noise order
  differences. ADR0005 records an explicit published profile while historical source
  defaults remain separate. No guessed tolerance or formula rewrite.
- Final native/reference seeds17/29/41: max matrix2.2351741790771484e-7,
  max translation6.705522537231445e-8, max rotation2.657257508532552e-7 radians;
  gripper commands identical. Tolerance established from reference repeats before
  native comparison: matrix5.513429641723633e-6, translation1e-6, rotation1e-5.
- C3 repeated/cold/warm/reset and no-grad PASS. K=1 parity, K=3 indexing/RNG/prefix
  integration PASS. C5 installed wheel from /tmp without reference/old modules or
  unused training/environment/v5 networks PASS.
- Final wheel SHA2567509bcdb66f7da26b3a7bfebf414b410d7e3b9a25b6b34d0dca0e8abf3625d49.
  Final source .py bytes match that wheel; subsequent changes are documentation only.
- All C1–C5 PASS: [acceptance](../../experiments/vv19-validation/acceptance.json).
  Full actual output: [commands](../../experiments/vv19-validation/all-commands.log).

### Validation
- Full installed-wheel discovery on Colab: 88 selected, 87 PASS, 1 SKIPPED
  (real RLBench task-class resolution; simulator not installed), zero failures.
  This includes real published model integration, native original-source differential,
  actual Open3D, Lightning wrapper, CPU/CUDA RNG and all public seam regressions.
- L0 local and Colab no-site: 19 PASS, source-root discovery and intentional
  forbidden fixtures included. No hidden empty discovery success.
- Editable package import/help from /tmp PASS in isolated local Python3.10 env.
  Wheel native inference/CLI help from /tmp on Colab PASS.
- Tests remain flat; fixtures/regression script are explicit entry points. No test
  framework change. Full benchmark/training/robot operation NOT RUN; simulator
  rollout NOT RUN (not required to establish the supplied-fixture gates).
- macOS binary/reference prerequisites failed in emulated Open3D (SIGILL) and
  native OpenMP. These failed attempts were not called passed; Colab provided
  compatible actual inference instead. CUDA cu121/cu118 installation mismatch
  corrected by official direct cu118 wheel pin.
- Full-Colab archive first contained AppleDouble metadata and two static tests
  failed UTF-8 decoding. Recreated transfer with COPYFILE_DISABLE and no xattrs;
  rerun passed, without ignoring source checks.
- Independent review identified runtime-override discard, mutable prepared demos,
  schema range/type, CPU-scope CUDA RNG, missing ABI prerequisites and root-shape
  gaps. All fixed with regressions. Follow-up reviewer hit provider429; code fixes
  were verified by actual full suite/C1–C5 rather than claimed reviewed.
- Colab final file-download API failed after completed execution. Exact logs and
  result JSON recovered from exported CLI history (SHA256
  d5dff2f9c57de5660055fe2e29319e99ce082108ee20eaca55f8fd673256fa99);
  no evidence numbers fabricated. Final output NPZ not recovered, but checked-in
  original fixture and generator plus executed commands reproduce it.
- Both owned Colab assignments explicitly released; server reports no active
  sessions. Task-owned Docker container stopped. No push/commit/staging performed.

### Changes, compatibility and limitations
Old→new mapping above corresponds to actual semantic owner files. Geometry and
diffusion cohesive helpers are grouped rather than one-file-per-function. Training
runner split from Lightning module; evaluation wraps shared execution. Config
presets are inputs, packaged published profile is runtime artifact metadata.
Historical ip public imports are intentionally removed. Native PyG schema and
scientific formulas preserved except explicit target profile corrections.
Arbitrary whole-object pickle and optimizer-resume remain unverified claims.

Checkpoint bytes: 471777552, SHA256119fa871091c7082b98d8a795dd80eca38295c4b7ab454e1f88549194bd4a4a5.
Fixture: two geometric synthetic demos, ten poses each, nontrivial translation/
rotation, three point clusters; no task-success claim. Schema and exact commands:
[CLI/data](../../components/cli-and-data.md), [setup](../../../README.md).
Source default preset and published target preset are named separately.

Stale occupancy/pretraining code retained under training/stages/occupancy_unsupported,
explicitly unsupported because missing visualization/incompatible old constructor;
not imported by baseline. Legacy diagnostic scaffolding/import adapters were removed,
recoverable in Git. Original retry/logging quirks remain documented, not quietly
fixed. No random v5 network/constant evaluator/search placeholder is used.

Next method implementation milestone: specify and implement the real physical
representation/history component and its observed-transition data contract against
the designed v5 ownership, while keeping the tested native proposer untouched.
Do not automatically train or benchmark; that requires a separately scoped task.
