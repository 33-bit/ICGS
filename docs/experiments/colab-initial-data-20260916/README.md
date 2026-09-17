# Colab initial-data collection readiness

Status: Drive/HF and simulator readiness PASS; bounded pinned G1 API probe PASS;
training-data collection remains BLOCKED_IMPLEMENTATION, NOT STARTED.
Owner: repository owner; execution coordinator, 2026-09-16.

## Resume after user Drive authorization

Owner confirmed Drive authorization. The named T4 session now reports
`drive_mounted=true`; a second unnamed T4 assignment belongs to external/user
activity and is not modified by this job. Resume order is explicit:

1. Persist a clearly marked storage preflight file to mounted Drive and verify
   readback; commit only that safe file to HF and verify the committed bytes.
2. Provision isolated Python 3.11 and pinned upstream simulator dependencies,
   keeping setup logs/evidence on Drive. No native IP/model training changes.
3. Measure a bounded headless simulator check separately from approved-program
   collection. Missing program/expert/calibration bindings remain a blocker, not
   permission to label an unrelated benchmark demo as an ICGS program.
4. Only closed, validated numeric episode artifacts can enter the dataset prefix;
   push after each completed episode and stop on persistence/upload failure.

HF credentials travel only through the authenticated Colab file API to a private
ephemeral path, not notebook source/history, Drive, or Hugging Face. The complete
local `.env` is never transferred. A storage marker is not a training sample.

### Actual resumed evidence

- Named session uses T4 endpoint `gpu-t4-s-kkb-ass1c1-2guaua19dr83i`; the other
  unnamed assignment is external/user-owned and was left untouched.
- Drive mounted at `/content/drive/MyDrive`. Setup run:
  `ICGS-data-20260916/setup-20260916T030801Z`.
- Drive write/read and HF commit readback: PASS. Safe storage marker commit
  `dcc41bcea6561b40d1d54a151958dd6377e4e9f7` under
  `preflight/setup-20260916T030801Z/storage-preflight.json` in `33bit/icgs`.
  Marker SHA256 `fa2426266ec7110df7e8f0b9ea11ad00c98627fb87f602eeba2ef44848411629`.
- Isolated environment `/content/icgs-data-env`, Python 3.11, NumPy 1.26.4,
  SciPy 1.14.1. Exact installed package list is saved in `environment-setup.json`.
- Simulator: CoppeliaSim Edu 4.1.0 Ubuntu20.04 official archive; SHA256
  `512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8`.
  Source revisions: PyRep `8f420be8064b1970aae18a9cfbc978dfb15747ef`;
  RLBench `02720bba4c73fe02eb75df946b8791b806028a9d`.
- Setup used upstream installation guidance; no silent upgrade of ICGS native
  model dependencies. RLBench's missing gymnasium import required explicit
  `gymnasium==1.2.3`; dependency failure log preserved.
- Readiness probe corrections: use backend `simGetSimulationTime` (not a missing
  PyRep instance method), and the base scene's automatically handled camera.
  Failed probe logs/JSON remain under `attempt-history/`.
- Final bounded headless probe: PASS, two steps, measured timestep
  `0.05000000074505806` seconds, simulator clock 0 to `0.10000000149011612`, finite
  point cloud `[128,128,3]`, depth `[128,128]`. Capture stored as numeric NPZ on
  Drive. This is renderer/physics readiness, NOT G1/G2 or an executed task episode.
- Final HF evidence commit: `ca9743c2f13c64f86d61a3ac354a60178382f752` at
  `preflight/setup-20260916T030801Z`, including setup report/log, readiness
  numeric capture/report/log, collection-readiness blocker report and preserved
  failed attempt logs. Every uploaded file was downloaded at that exact commit
  and matched against its Drive SHA256. Receipts remain on Drive.
- Private Drive cache contains the 156375216-byte simulator distribution with
  verified SHA256 and exact upstream source archive. Neither binary/source cache
  nor credentials were published to the public dataset.
- Final local `python3 -B scripts/validate_fast.py`: PASS 19/19, zero skips;
  `git diff --check`: PASS. Remote setup/simulator checks are operational
  evidence, not a fresh C1–C5 or approved-task benchmark run.
- Local source snapshot: commit `2166acd630f832d4ab111d292320e2199b990935` plus
  current uncommitted source/test files, bundle SHA256
  `b9be731ce39269ac78988e4cc684ae438377233362f3c7078b125e4669847386`.
  Includes only src/tests/pyproject, excludes `.env`, credentials and user outputs.
  Source/simulator archives are cached on private Drive, not the public dataset.

Commands executed from the local repository through Colab CLI:

```bash
colab exec -s icgs-initial-data-20260916 -f scripts/colab_data_storage.py --timeout 90
colab exec -s icgs-initial-data-20260916 -f scripts/colab_data_environment.py --timeout 1530
colab exec -s icgs-initial-data-20260916 -f scripts/colab_fix_gymnasium.py --timeout 140
colab upload -s icgs-initial-data-20260916 scripts/colab_simulator_probe.py /content/colab_simulator_probe.py
colab exec -s icgs-initial-data-20260916 -f scripts/colab_simulator_probe.py --timeout 140
colab exec -s icgs-initial-data-20260916 -f scripts/colab_collection_handoff.py --timeout 240
colab exec -s icgs-initial-data-20260916 -f scripts/colab_publish_preflight.py --timeout 180
```

The preflight publication script only accepts explicit setup files, checks for
the token before upload, and verifies remote committed bytes against Drive.
No background/continuous episode uploader is running because no physical episode
generator exists yet. Initial data runner/archives being accepted did not implement
the concrete `TimedController` or the six custom tasks and their experts.

After publishing and verifying readiness artifacts, the private ephemeral HF
credential is removed and the named job VM is released to avoid idle charges.
The other unnamed T4 assignment is not this job's resource and is left untouched.
At this pre-G1 readiness point, training episodes produced: **0**. Full measured
G1/G2 and training were **NOT RUN**; the later bounded G1 result is recorded
below.

## Authority and scope

Owner requested a new Colab CLI session, allowed hardware/job specifications to
be chosen, and required all generated data on Google Drive with continuous
publication to the existing Hugging Face dataset `33bit/icgs`. This authorizes
provisioning and bounded collection setup, not fabricated G1/G2 measurements or
replacement of approved custom programs with unrelated benchmark demonstrations.
The dataset is public; publish only intended numeric episodes and safe provenance,
never credentials, source `.env`, OAuth caches, or unrelated local files.

## Chosen initial envelope

- Session: `icgs-initial-data-20260916`, one NVIDIA T4 requested.
- First workload is environment/storage readiness, followed only when prerequisites
  pass by a three-attempt, maximum 32 intervals per attempt smoke collection.
- Initial collection wall cap 1800 seconds, dataset cap 1 GiB; no training or
  primary-scale collection. Actual physics/task/program/seed manifests must be
  bound and validated before any attempt starts.
- Proposed persistent run directory: Google Drive `ICGS-data-20260916` with a
  unique run directory; immutable episode uploads to
  `https://huggingface.co/datasets/33bit/icgs` under a matching run prefix.
- Durability order: validate/checksum episode, persist to Drive, upload closed
  episode/shards to Hugging Face, publish the index last. Pause collection if
  durable persistence fails; a periodic sync is not a durability guarantee.

This record is an operational preflight, not model/baseline efficacy evidence.
No native policy/configuration semantics or checkpoint bytes change. Initial
physical generation still needs an explicit timed controller, custom scene/expert
bindings and calibration/predicate artifacts; an injected test double is not data.

## Initial attempt commands and evidence (historical)

Local cwd: `/Users/33bit/AI/Research/VLA/ICGS`.

- `colab version`: 0.6.0.
- `colab sessions`: no prior active assignments before provisioning.
- `colab new -s icgs-initial-data-20260916 --gpu T4`: session READY.
- `colab exec -s icgs-initial-data-20260916 -f scripts/colab_data_preflight.py`:
  PASS as a readiness inspection, not collection. Actual Tesla T4, 15360 MiB,
  driver 580.82.07; Linux x86_64; kernel Python 3.13.15 (outside ICGS's declared
  support, so use an isolated Python 3.11 environment before running ICGS).
  Free ephemeral disk 202175352832 bytes. NumPy/torch/huggingface_hub and
  Xvfb/git/uv present; RLBench/PyRep absent; Drive NOT mounted.
- HF token read in-process from `.env` key `HF_ACCESS_TOKEN`; authenticated
  account `33bit`, fine-grained repository write scope. Dataset exists, public,
  initial contents only `.gitattributes`. No token is embedded in this record.
- Google Drive REST read using existing Colab OAuth credentials: FAIL, HTTP 403
  `accessNotConfigured`; response requires an explicitly configured quota project.
  Do not use an unrelated third-party billing/quota project to bypass this.
- Matching-account local Google Drive folder inspection: FAIL, macOS
  `PermissionError: Operation not permitted`. No attempt to bypass this system
  permission was made. User authorization is required for either a Colab Drive
  mount or local app access to the mounted Drive.
- `colab stop -s icgs-initial-data-20260916`: PASS, session stopped after blocked
  preflight to avoid consuming T4 credits while waiting for authorization.
- Local `python3 -B scripts/validate_fast.py`: PASS, 19/19 L0 tests, zero skips
  (two existing host-Python SyntaxWarnings); `git diff --check`: PASS.

## Initial attempt gates (historical; Drive authorization now resolved)

Google Drive durability is required BEFORE producing training episodes. Existing
Colab CLI guidance says Drive mounting can require interactive consent; do not
hang a headless job on that prompt. No generation job or background upload daemon
has been started and no training episodes have been uploaded.

HF authentication/read access PASS and token scope includes writes; no test commit
was pushed because required Drive durability was unavailable. Continuous dual
publication is **NOT CONFIGURED / NOT RUN**, not implicitly proven by a valid token.

No ICGS/model dependencies were installed on this VM, no weights/assets downloaded,
no simulator launched, no collection/training performed. Concrete program/expert,
controller and calibration bindings remain missing in addition to storage access.

Resume by authorizing Drive for a fresh Colab session (`colab drivemount -s NAME`
is interactive and must be performed by the user), or granting this app normal
access to the matching mounted Drive location. Then create a fresh bounded
session, verify a persistent Drive write/read and HF commit, prepare supported
Python and physical bindings, and only then authorize measured smoke execution.

If storage authorization or physical task bindings block setup, stop the unused
Colab VM to avoid idle compute charges, retain this record, and report the exact
resume requirements. Never claim continuous uploads or collection is running
merely because a session was allocated.

## Bounded G1 API probe command and acceptance boundary

Authority: [ADR0012](../../decisions/0012-exploratory-rlbench-g1-track.md) and [ADR0007](../../decisions/0007-icgs-timed-data-lineage.md).

Before executing any RLBench exploratory collection (`scripts/run_rlbench_exploratory.py`) or creating exploratory dataset archives, a verified version-2 PASS receipt produced by the bounded G1 API probe (`scripts/colab_g1_api_probe.py`) is mandatory. Real collection launch without this receipt is strictly forbidden. A valid receipt is feasibility evidence only: the Phase 0 runner still refuses both real and mock collection until Phases 2--4 are separately implemented and reviewed.

### Exact G1 probe command

Run headlessly within the pinned Colab environment using Xvfb:

```bash
timeout --foreground 150s xvfb-run -a -s "-screen 0 1280x1024x24" /content/icgs-data-env/bin/python -B scripts/colab_g1_api_probe.py \
  --output-dir /content/drive/MyDrive/ICGS-data-20260916/g1-probe \
  --task PickAndLift \
  --max-wall-time-s 120
```

To execute an opt-in live demonstration check (strictly opt-in, not default):

```bash
timeout --foreground 150s xvfb-run -a -s "-screen 0 1280x1024x24" /content/icgs-data-env/bin/python -B scripts/colab_g1_api_probe.py \
  --output-dir /content/drive/MyDrive/ICGS-data-20260916/g1-probe \
  --task PickAndLift \
  --run-live-demo \
  --max-wall-time-s 120
```

### Acceptance boundary

The G1 probe output is accepted if and only if:
1. **Drive containment and lifecycle**: `g1-receipt.json` is under the mounted `/content/drive/MyDrive/...` subtree. The probe replaces any prior PASS with a durable FAIL before launch, and writes PASS only after clean outer-environment shutdown.
2. **Receipt identity**: `kind == "g1_api_probe_receipt"`, `receipt_version == 2`, `probe_implementation_id == "g1-api-probe-v2"`, and `protocol_id == "rlbench-g1-coppeliasim-4.1-franka"`.
3. **Pinned inputs**: the receipt records clean source trees at PyRep `8f420be8064b1970aae18a9cfbc978dfb15747ef` and RLBench `02720bba4c73fe02eb75df946b8791b806028a9d`, plus the verified CoppeliaSim 4.1 archive SHA256 `512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8`. Provisioning installs those detached trees editable, and the probe rejects any imported `pyrep` or `rlbench` module whose resolved source lies outside them.
4. **Task identity**: the requested task and `TaskEnvironment.get_name()` resolve to the same task identity.
5. **Live target hold**: after the pinned `(descriptions, observation)` reset contract, the probe reads the live arm-tip pose, solves Jacobian IK, applies joint targets, holds the measured gripper opening, and calls `TaskEnvironment._scene.step()` exactly twice. It records finite arm-target error within the pinned 0.01-radian tolerance and gripper-hold error within 0.001.
6. **Measured clock**: `clock.clock_advanced == true`, queried timestep is finite and equal to 0.05 seconds, exactly two scene/physics steps occur, and measured elapsed simulator time equals two timesteps without a synthetic fallback.
7. **Raw sensor evidence**: the raw reset observation contains a finite floating XYZ wrist cloud with self-consistent shape and point-count evidence.
8. **Clean shutdown**: `cleanup.outer_env_shutdown == true`; a shutdown or setup failure leaves a FAIL receipt rather than stale PASS evidence.
9. **Bounded execution and no collection/secret handling**: the in-process wall-time cap is at most 180 seconds and the command has an outer 150-second timeout. Zero episode archives, shard files, or dataset manifests are created; the script does not import `dotenv` or `huggingface_hub`, or read `.env` / secret environment variables.

### Actual bounded G1 result — 2026-09-17 local time

- Named session `icgs-e01-g1-20260916` was created as a Tesla T4, Drive was
  mounted interactively, and the prior zero-episode storage receipt was reused
  only to locate the existing Drive setup directory. No credential was read,
  transferred, or published during G1.
- The fresh VM provisioned `/content/icgs-data-env/bin/python` (Python 3.11),
  the detached clean PyRep revision `8f420be8064b1970aae18a9cfbc978dfb15747ef`,
  the detached clean RLBench revision
  `02720bba4c73fe02eb75df946b8791b806028a9d`, and CoppeliaSim 4.1 archive SHA256
  `512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8`.
- The first bounded non-demo attempt safely wrote `FAIL`: the pinned scene did
  not construct `wrist_point_cloud` because its `get_observation()` path requires
  RGB or depth capture in addition to the point-cloud flag. This was verified
  directly against the pinned source. The probe was corrected to retain raw wrist
  depth capture, and a new focused local regression test passed before retry.
- The one corrected bounded retry wrote a final `PASS` receipt at
  `/content/drive/MyDrive/ICGS-data-20260916/g1-probe/g1-receipt.json`. It used
  `PickAndLift` without `--run-live-demo`; API availability was recorded, but no
  expert demonstration was executed or claimed. A direct readback after the T4
  session stopped verified SHA256
  `ff8fdfb68a609f553c7a936b070ad4443c48929e28948524df378dca53080a25`.
- Measured receipt evidence: timestep `0.05000000074505806` s; simulator clock
  `0.699999988079071` to `0.800000011920929` s after exactly two scene steps;
  elapsed `0.10000002384185791` s; finite raw wrist cloud `float64 [128,128,3]`;
  joint hold error `0.0005646944046020508` rad (limit `0.01`); gripper hold error
  `0.000042188913573104614` (limit `0.001`); in-process elapsed wall time
  `6.32511466699998` s (limit `120`); outer `Environment.shutdown()` succeeded.
- The strict v2 verifier in `scripts/run_rlbench_exploratory.py` independently
  accepted the persisted receipt. `g1-probe/` contained only that receipt: no
  archive, shard, manifest, data upload, or Hugging Face action was performed.
- `colab stop -s icgs-e01-g1-20260916` succeeded and `colab sessions` then
  reported no active sessions. The PASS is G1 feasibility evidence only; it does
  not authorize E01 collection before Phases 2--4 are implemented and reviewed.
