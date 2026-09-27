# Generation launch readiness — 2026-09-27

Status: **PASS for the selected single-host collection readiness gates**.
No full collection or training was started.
This closes engineering follow-ups from Paseo
`15256c9e-1435-4982-ab45-98b6349bb43c`, without changing program quotas,
seeds, task semantics, data representation, actions, preprocessing or models.

## Scope and environment

All tests and live validation run over SSH on `vps-a`, with cwd
`/home/huy2325/ICGS`. No tests run on the local editing host. All new
environments, caches, simulator assets, credentials and evidence are inside
that clone. CPython 3.10.21 in `.venv`, Linux x86_64, CPU, two workers
`000`/`001`, two Xvfb simulator slots (displays 200/201), 8 CPUs and 14 GiB RAM.
About 161 GiB disk was free during the smoke. Pins remain CoppeliaSim 4.1
(checksum in `scripts/generation_environment.py`), PyRep
`8f420be8064b1970aae18a9cfbc978dfb15747ef`, RLBench
`02720bba4c73fe02eb75df946b8791b806028a9d`.

Base commit: `0252fca96a784d5cb88390ef3078bed4ca399ad8`. Runtime fixes were
copied to the VPS working tree before validation; live roots retain
`source.patch` and the smoke records its SHA256. These are dirty-tree tests,
not claims that the unmodified base commit passed. Helpers are retained as
`outputs/readiness_smoke.py` and `outputs/readiness_hf.py`; they are finite
acceptance drivers, not a replacement runtime or production scheduler.

Smoke source patch SHA256:
`24bbbf58aaee6d86cba5ad4d34af49119b4c26a52cb91bfff07e3b53929b24a8`.
The `scripts/` + `src/` portion was compared byte-for-byte against the final
runtime diff on the VPS: PASS, SHA256
`fc9da2e86878803964558d14d8c4646d30945adc3ffe1bdb25ebca30bdf1e613`.

## Corrections and preserved failures

1. Shared queue admission persists a policy and per-attempt reservation under
   the existing lock. Physical bytes and reservations are conservatively added;
   leases, ready/ingested backlog and keep retention never release capacity.
   Verified receipt-only pruning does. Production worker/coordinator/launcher
   entry points require the budget. HF scratch uses the reserved filesystem.
2. Fresh upstream provisioning lacked generated `rlbench.tasks.*` modules.
   The first live smoke failed for that reason. Setup now has an explicit
   all-program task-build option, and the non-simulator verifier rejects missing
   or stale task source/empty models. All 36 assets were built on this host.
3. An `fsync` for every temporary `.npy` spool member caused heavy ext4 journal
   waits. Temporary members now flush/close; completed NPZ chunks, manifests,
   directories and atomic publication retain their durability boundaries.
   A regression verifies zero spool syncs, one final chunk sync and exact values.

Initial storage tests failed before implementation, then passed. Review found
the missing core-entry guard and system-temp scratch placement; both were fixed
and regression-tested. Final read-only review found no outstanding important
issues. Production recovery fixtures were updated to install the policy before
claims rather than weakening the guard. The one-off HF helper uses stable worker
instance IDs for sequential one-shot processes; production leases are unchanged.

Intermediate logs remain under `outputs/readiness-*`. The initial missing-task
failure archives remain at `outputs/readiness-smoke`. One aborted restart's
unpublished `outputs/readiness-smoke-built` scratch was mistakenly deleted before
the current run; those bytes are not recoverable. This was disclosed to the user.
No published archive was deleted, and subsequent evidence is retained.

Extending the initial 30-minute smoke watchdog also terminated its helper
after 20 receipt rows, while validating V01/V02. Both workers had already
exited with closed ready archives. The resumed helper revalidated those
archives, then continued only unrun programs, preserving the original patch
and receipt. No completed capture was regenerated. Resume used
`timeout --signal=INT 1300 .venv/bin/python -B outputs/readiness_smoke.py --resume`
with the same environment and root; its log is `outputs/readiness-smoke-resume.log`.
The helper now unwinds owned worker groups on interruption.
After the longer V/P batches, the old timeout was paused and a separate
1,600-second process monitor installed for the remaining fixed programs;
there was no second helper restart or increase to the 36-job scope.

## Commands and gates

Caches/temp were directed into the clone with `TMPDIR=$PWD/outputs/tmp`,
`UV_CACHE_DIR=$PWD/outputs/cache/uv`, `PIP_CACHE_DIR=$PWD/outputs/cache/pip`,
`HF_HOME=$PWD/outputs/cache/hf`, `XDG_CACHE_HOME=$PWD/outputs/cache`.
The setup invocation used `PATH=/home/huy2325/.local/bin:$PATH`:

```bash
python3 -B scripts/setup_environment.py --profile generation --python-version 3.10 \
  --venv-root /home/huy2325/ICGS/.venv --provision-simulator \
  --runtime-config "$PWD/outputs/readiness-runtime.json" \
  --receipt "$PWD/outputs/setup_receipt.json"

COPPELIASIM_ROOT="$PWD/outputs/CoppeliaSim" \
LD_LIBRARY_PATH="$PWD/outputs/CoppeliaSim" \
QT_QPA_PLATFORM_PLUGIN_PATH="$PWD/outputs/CoppeliaSim" QT_QPA_PLATFORM=xcb \
LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=src timeout 300 xvfb-run --server-num 210 \
  .venv/bin/python -B scripts/generation_build_tasks.py --all-programs \
  --rlbench-root "$PWD/outputs/RLBench" --output-root "$PWD/outputs/generation-tasks"

.venv/bin/python -B scripts/verify_environment.py --profile generation \
  --repo-root "$PWD" --python "$PWD/.venv/bin/python" \
  --simulator-root "$PWD/outputs/CoppeliaSim" --rlbench-root "$PWD/outputs/RLBench" \
  --credential-path "$PWD/.env" --receipt "$PWD/outputs/setup_receipt.json" --json

LD_LIBRARY_PATH="$PWD/outputs/CoppeliaSim" PYTHONPATH=src \
  timeout 1200 .venv/bin/python -B -m pytest -q -rs -p no:cacheprovider \
  tests/test_generation*.py tests/test_capacity_probe.py tests/test_environment_setup.py
.venv/bin/python -B scripts/validate_fast.py
.venv/bin/python -B -S scripts/validate_fast.py
```

Setup and standalone all-task build: PASS. The new combined setup/build command
path is fixture-tested; this live rebuild used the separate commands above.
Environment verification: PASS for generation prerequisites, task assets,
credentials and setup receipt; CUDA NOT RUN (CPU generation host).

Full generation/capacity/setup fixture suite: **PASS, 650 passed, 0 skipped**,
771.64 seconds, exit 0 (`outputs/readiness-fixtures-final.log`). Before that,
the stopped full run reported 371 PASS and one recovery-fixture failure for a
missing budget; its corrected remainder passed all 304 tests in 407.39 seconds.
Both L0 commands above: **PASS, 22 tests each**, no skipped tests. L0 does not
execute a model. Environment verification's final JSON is retained at
`outputs/readiness-environment-final.json`.

Smoke uses 18 isolated batches of two explicit jobs, one per program, with
1 GiB writer cap, 16 GiB total and 4 GiB reserve, keep retention. Each worker
runs `--once` through canonical worker commands. The helper invokes
`validate_closed_result`, checks timelines, and stops on blocking outcomes.
No publisher or refill loop runs. The top-level command is:

```bash
LD_LIBRARY_PATH="$PWD/outputs/CoppeliaSim" PYTHONPATH=.:src \
  timeout 1800 .venv/bin/python -B outputs/readiness_smoke.py
```

HF rehearsal uses three explicit jobs, the same two-worker layout, 1 GiB writer
cap, a deliberately tight 4.5 GiB total and 2 GiB reserve, and receipt-only
retention. Its dedicated prefix is `validation/readiness-hf-20260927` in
`33bit/icgs`; it cannot refill the collection quota. The third claim must block,
then become available only after pinned remote verification and local pruning.
Credentials are read privately by the coordinator and omitted from worker
environments. Raw token values are not part of this record.

Smoke: **PASS, 36/36 programs**, 29 successes and seven valid failures
(T01, T06, V02, V04, R1, R3, R4), no crashes/invalid results or blocking
timelines. Canonical archive validation was performed for every row. The
launcher `validate_smoke_receipt` accepted the combined real receipt via
`PYTHONPATH=.:src LD_LIBRARY_PATH=$PWD/outputs/CoppeliaSim .venv/bin/python -B
outputs/readiness_validate.py` (exit 0). Its metrics are at
`outputs/readiness-smoke-built/metrics.json`, receipt at
`outputs/readiness-smoke-built/smoke.json`.
All writer-reported upper bounds were below 1 GiB; the largest was T06,
316,396,579 bytes (685 boundaries, 132,519,126 compressed artifact bytes).
Total retained archive inventory was 2,797,275,534 bytes. These are measured
closed inventories/writer bounds, not a continuously sampled aggregate peak
or proof about every future random seed.

HF rehearsal: **PASS**, exit 0, three published jobs, zero pending/claimed/
ready/ingested/quarantined jobs, zero remaining reservations, no result payload
directories and empty HF scratch. Before publication: 2,147,483,648 reserved
bytes, 71,985,036 local bytes, `can_claim=false`. After the first verified prune:
1,073,741,824 reserved bytes, 44,111,464 local bytes, `can_claim=true`. The third
one-shot worker then completed. Final local bytes before writing acceptance:
184,720 (receipts/configuration, including the private ignored credential file).
Pruned payloads remain recoverable from their pinned HF archives; they were
not discarded before remote verification.

| Program | Data commit OID | Pinned verification receipt OID |
| --- | --- | --- |
| T02 | `255d27a287148836de3bcfd8e240056c4d61e862` | `2d51da0d5b2c553bdc9b52997523c612804b3b12` |
| T03 | `ef52e8c130b76eccdd39047c7f91ec2ed9e2c2dc` | `0829aad5b66f57f3c8329c0d0a8e57d81b581269` |
| T04 | `678785f88ddde88775446f5f34ab01b6fcfb85b9` | `89ad5a180bc998e0c783a28a405b2ed2f4a74711` |

All three local publication receipts are `COMPLETE`; final verified-metadata
HF HEAD is `66da3899a3331600e73147ebd78c14633017917c`. Full receipt at
`outputs/readiness-hf/acceptance.json`, log at `outputs/readiness-hf.log`.
HF source patch SHA256:
`9093b45b4cc79b2a580f1a00e9b88787326fc5a624a474fabc08a48f8a30f4c1`.
Execution command (with the cache/temp variables above):

```bash
LD_LIBRARY_PATH="$PWD/outputs/CoppeliaSim" PYTHONPATH=.:src \
  timeout --signal=INT 1800 .venv/bin/python -B outputs/readiness_hf.py
```

## Prepared handoff, not executed

VPS `outputs/generation-vps-a-runtime.json` selects the validated two-worker/
two-slot host, 1 GiB per writer, 16 GiB total staging and 4 GiB fixed reserve.
Its fresh run ID is `generation-vps-a-20260927`, root under `outputs/runs/`,
HF prefix `generation/archive-v1-vps-a-20260927`, production publication and
receipt-only retention. Configuration/path validation passed; the run root
remains absent. Backlog admission is byte-bounded, not an extra episode quota.
The reserve is deliberately more conservative than the 2 GiB live rehearsal.

After synchronizing the final main commit, an operator can launch from the VPS
clone with the following command. **This command was NOT RUN**:

```bash
TMPDIR="$PWD/outputs/tmp" XDG_CACHE_HOME="$PWD/outputs/cache" \
HF_HOME="$PWD/outputs/cache/hf" .venv/bin/python -B scripts/generation_launch.py \
  --runtime-config outputs/generation-vps-a-runtime.json \
  --approved-manifest artifacts/composition/approved_composition_manifest.json \
  --code-revision "$(git rev-parse HEAD)" \
  --smoke-receipt outputs/readiness-smoke-built/smoke.json \
  --hf-token-path outputs/readiness-hf/credential --detach
```

The credential file is mode 0600 and ignored; do not print, commit or pass it
to workers. Keep it while this command references it. Inspect coordinator and
queue heartbeats after launch; a safety marker is a stop for investigation,
not permission to delete scratch. A different host/layout or changed runtime
requires its own bounded rehearsal. This is not a throughput or quota-completion
claim.

## Claim limits

Admission is cooperative, not a kernel filesystem quota. External writers and
unbounded historical remote metadata need operational headroom. A retained or
abandoned reservation fails closed and may require inspection or a fresh run;
never delete unpublished data merely to resume. A one-attempt-per-program smoke
does not establish success rate or guarantee that every random seed fits a cap.

Full collection, training, Open3D/model execution, L2/C1–C5 and L4 were NOT RUN
for this engineering change. Historical checkpoint evidence is unchanged, not
rerun or replaced by these checks. Archive-backed trainer integration remains a
separate downstream gap; collector readiness does not claim training readiness.
