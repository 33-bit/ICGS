# Generation Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical generation runtime portable, failure-isolated and observable, then prove it with a bounded two-worker Colab CPU run and isolated Hugging Face publication.

**Architecture:** One validated runtime config supplies machine paths, process limits and run/publication settings to the existing canonical scripts. Queue validation failures become durable quarantine transitions instead of coordinator-fatal exceptions; progress heartbeats expose blocked phases; publication reconciles partially completed remote commits before retry. No duplicate Colab/debug entry points are added.

**Tech Stack:** Python 3.11, dataclasses/JSON, POSIX `flock`, filesystem queue, RLBench/PyRep/CoppeliaSim, Hugging Face Hub, pytest/unittest, Colab CLI.

**Spec:** `docs/superpowers/specs/2026-09-22-generation-runtime-hardening-design.md`

## Global Constraints

- Modify only canonical runtime owners under `src/icgs/data/collection/generation/` and `scripts/generation_*`, plus their tests/docs.
- Do not create `debug_*`, `tmp_*`, `smoke_*`, `colab_*`, or machine-specific duplicate scripts.
- Do not hardcode `/content`, a fixed Python executable, 200 workers, display 200, or simulator concurrency in portable contracts.
- `success` and `valid_failure` remain episodes; `simulator_crash` and `invalid_observation` remain attempts.
- HF publication is disabled unless explicitly enabled and CPU validation uses only `validation/validation-cpu-20260922/`.
- Do not start full generation, TPU generation, training or benchmark evaluation.
- Every implementation task follows red-green-refactor and ends in an independently reviewable commit.

---

### Task 1: Define the portable runtime contract

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_contracts.py`
- Create: `src/icgs/configuration/profiles/generation_runtime.json`
- Modify: `pyproject.toml`
- Test: `tests/test_generation_config.py`

**Interfaces:**
- Produces: `MachineConfig.from_dict(payload) -> MachineConfig`
- Produces: `GenerationRuntimeConfig.from_dict(payload) -> GenerationRuntimeConfig`
- Produces: `GenerationRuntimeConfig.from_file(path) -> GenerationRuntimeConfig`
- Produces: `GenerationRuntimeConfig.resolved_environment(base=None) -> dict[str, str]`
- Extends: `RunConfig` with configurable worker/publication limits while preserving immutable run identity.

- [ ] **Step 1: Write failing contract tests**

Cover a portable temporary directory config with two workers/two slots; reject relative paths, nonexistent required executables/directories, worker count below 1, slots outside `1..worker_count`, nonpositive timeout/display dimensions, non-validation HF prefix when `validation_mode=true`, enabled publication without repo/subfolder, and unknown JSON fields. Assert the checked-in profile parses without touching the simulator.

- [ ] **Step 2: Run red tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_config.py
```

Expected: FAIL because `MachineConfig` and `GenerationRuntimeConfig` do not exist.

- [ ] **Step 3: Implement strict dataclasses and JSON loading**

Use frozen dataclasses. Keep validation pure except `from_file(..., check_paths=True)`; tests and config inspection can pass `check_paths=False`. Reject unknown keys explicitly. Resolve environment values without modifying `os.environ`.

Required defaults in the packaged example profile:

```json
{
  "machine": {
    "repo_root": "/workspace/ICGS",
    "python_executable": "/workspace/venv/bin/python",
    "simulator_root": "/workspace/CoppeliaSim",
    "rlbench_root": "/workspace/RLBench",
    "display_base": 200,
    "display_width": 1280,
    "display_height": 1024,
    "simulator_slots": 2,
    "worker_timeout_s": 1200
  },
  "run": {
    "run_id": "validation-example",
    "run_root": "/workspace/runs/validation-example",
    "worker_count": 2,
    "publish_interval_s": 300,
    "hf_repo": "33bit/icgs",
    "hf_subfolder": "validation/validation-example",
    "publication_enabled": false,
    "validation_mode": true
  }
}
```

- [ ] **Step 4: Run green tests and focused regressions**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_config.py tests/test_generation_queue.py \
  tests/test_generation_control.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/distributed_contracts.py \
  src/icgs/configuration/profiles/generation_runtime.json pyproject.toml \
  tests/test_generation_config.py
git commit -m "feat(data): add portable generation runtime config"
```

### Task 2: Make environment setup and launch config-driven

**Files:**
- Modify: `scripts/generation_environment.py`
- Modify: `scripts/generation_launch.py`
- Modify: `scripts/generation_worker.py`
- Test: `tests/test_generation_environment.py`
- Test: `tests/test_generation_control.py`
- Test: `tests/test_generation_worker.py`

**Interfaces:**
- Consumes: `GenerationRuntimeConfig`
- Produces: `build_worker_commands(config, approved_manifest) -> list[list[str]]`
- Produces: `build_process_environment(config, base=None) -> dict[str, str]`
- Produces: `provision_environment(config, *, runner=subprocess.run) -> ProvisionReceipt`
- Changes worker CLI to require `--runtime-config` and read display/timeout/slots from it.

- [ ] **Step 1: Write failing portability tests**

Use temporary absolute paths and injected runners. Assert two workers generate two commands, displays derive from `display_base`, screen size derives from config, executable and repository script paths derive from config, `ICGS_SIMULATOR_SLOTS` is exported, and worker subprocess timeout derives from config. Assert the environment provisioner never contains literal `/content`.

- [ ] **Step 2: Run red tests**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_environment.py tests/test_generation_control.py \
  tests/test_generation_worker.py
```

Expected: FAIL on fixed worker count/paths and missing config interfaces.

- [ ] **Step 3: Refactor existing scripts without adding entry points**

Replace module constants for root/python paths with config fields. Preserve pinned revisions and CoppeliaSim SHA256 in the environment owner. Make apt/download/install commands observable in a returned receipt. `generation_launch.py` writes the exact resolved runtime config and its SHA256 into `control/run.json` before starting children.

- [ ] **Step 4: Run green tests**

Use the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generation_environment.py scripts/generation_launch.py \
  scripts/generation_worker.py tests/test_generation_environment.py \
  tests/test_generation_control.py tests/test_generation_worker.py
git commit -m "refactor(data): drive generation processes from runtime config"
```

### Task 3: Quarantine malformed ready results without stopping progress

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_queue.py`
- Modify: `src/icgs/data/collection/generation/distributed_validation.py`
- Modify: `scripts/generation_coordinator.py`
- Test: `tests/test_generation_queue.py`
- Test: `tests/test_generation_validation.py`
- Test: `tests/test_generation_control.py`

**Interfaces:**
- Produces: queue state `quarantined/`.
- Produces: `ValidationFailure` with job/result identity, exception type/message, traceback and actual file inventory.
- Produces: `FilesystemJobQueue.quarantine_ready(job_id, failure) -> Path`.
- Produces: `CoordinatorControlPlane.process_ready_result(result) -> Literal["ingested", "quarantined"]`.

- [ ] **Step 1: Write failing isolation tests**

Create one malformed ready directory followed by one valid ready result. Assert one coordinator tick moves the malformed result to `quarantined`, writes `validation_failure.json`, ingests the valid result, records only the valid result in planner quota, and remains `RUNNING`. Assert partial directories remain ignored and quarantine is immutable/idempotent for identical content but rejects conflicts.

- [ ] **Step 2: Run red tests**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_queue.py tests/test_generation_validation.py \
  tests/test_generation_control.py
```

Expected: FAIL because malformed validation still escapes `tick()`.

- [ ] **Step 3: Implement per-result failure isolation**

Add `quarantined` to queue counts and state discovery. Move the complete ready directory atomically, then atomically write the diagnostic. Never convert malformed data to `valid_failure`. Do not call `planner.record_result` for a validation failure; its job remains accounted in the quarantine receipt for explicit retry policy.

- [ ] **Step 4: Run green tests**

Use Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/distributed_queue.py \
  src/icgs/data/collection/generation/distributed_validation.py \
  scripts/generation_coordinator.py tests/test_generation_queue.py \
  tests/test_generation_validation.py tests/test_generation_control.py
git commit -m "fix(data): isolate malformed generation results"
```

### Task 4: Make progress and watchdog liveness explicit

**Files:**
- Modify: `scripts/generation_coordinator.py`
- Modify: `scripts/generation_watchdog.py`
- Modify: `src/icgs/data/collection/generation/distributed_contracts.py`
- Test: `tests/test_generation_control.py`

**Interfaces:**
- Produces heartbeat fields: `phase`, `tick_started_at_s`, `last_progress_at_s`, `last_progress_kind`, `last_validation_error`, `publication`, `queue`, `planner`.
- Produces: `heartbeat_health(payload, *, now_s, stale_after_s) -> Literal["healthy", "busy", "stale", "terminal"]`.
- Watchdog uses PID identity plus heartbeat health and coordinator lock; it never starts a second coordinator while the lock is held.

- [ ] **Step 1: Write failing heartbeat/watchdog tests**

Test healthy progress, a visible long `publication_in_progress` phase, stale heartbeat with a live PID/free lock, stale heartbeat with a held lock, terminal status, bounded restart count, and atomic receipt update. Assert publication activity is not misreported as generic health.

- [ ] **Step 2: Run red tests**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation_control.py
```

Expected: FAIL because watchdog currently checks PID only.

- [ ] **Step 3: Implement progress-aware heartbeat semantics**

Update heartbeat before/after validation, refill and publication transitions. Preserve last successful progress time across ticks. A watchdog may replace a coordinator only when PID ownership is invalid or heartbeat is stale and the coordinator lock is free. Record the reason in `launch.json` restart history.

- [ ] **Step 4: Run green tests**

Use Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generation_coordinator.py scripts/generation_watchdog.py \
  src/icgs/data/collection/generation/distributed_contracts.py \
  tests/test_generation_control.py
git commit -m "fix(data): monitor generation progress not only pids"
```

### Task 5: Reconcile partially completed HF publication

**Files:**
- Modify: `src/icgs/data/collection/generation/distributed_publication.py`
- Modify: `scripts/generation_coordinator.py`
- Test: `tests/test_generation_publication.py`
- Test: `tests/test_generation_control.py`

**Interfaces:**
- Produces durable publication states: `PREPARED`, `DATA_COMMITTED`, `VERIFIED`, `COMPLETE`, `DEFERRED`.
- Produces: `reconcile_publication(receipt, remote_manifest) -> PublicationReceipt`.
- Publication config controls job batch size, upload threads, retry attempts and cooldowns; validation mode restricts prefix to `validation/validation-cpu-20260922/`.

- [ ] **Step 1: Write failing reconciliation tests**

Cover: data commit success followed by receipt commit timeout; coordinator restart with `DATA_COMMITTED`; matching remote hashes complete without re-upload; conflicting remote identity fails closed; 429/RequestTimeout remains `DEFERRED`; successful verification moves exactly the committed jobs to `published`.

- [ ] **Step 2: Run red tests**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_publication.py tests/test_generation_control.py
```

Expected: FAIL because current receipt has only pending/committed states and retries the two-commit sequence directly.

- [ ] **Step 3: Implement idempotent publication state machine**

Persist the receipt before every external boundary. On startup, reconcile `DATA_COMMITTED` against the remote manifest and immutable hashes. Do not mark queue entries published until verification completes. Surface cooldown and last error in coordinator heartbeat.

- [ ] **Step 4: Run green tests**

Use Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/icgs/data/collection/generation/distributed_publication.py \
  scripts/generation_coordinator.py tests/test_generation_publication.py \
  tests/test_generation_control.py
git commit -m "fix(data): reconcile interrupted generation publication"
```

### Task 6: Add canonical bounded validation mode and receipt

**Files:**
- Modify: `scripts/generation_launch.py`
- Modify: `scripts/generation_coordinator.py`
- Modify: `docs/components/generation.md`
- Create: `tests/fixtures/generation_validation_config.json`
- Test: `tests/test_generation_control.py`
- Test: `tests/test_generation_validation.py`

**Interfaces:**
- Produces: `control/validation_receipt.json` with required gates from the spec.
- Produces CLI `generation_launch.py --runtime-config PATH --validation-plan PATH` using the same worker/coordinator code paths as production.
- Validation plan is data, not executable Python; it declares bounded job/outcome/fault-injection cases.

- [ ] **Step 1: Write failing receipt tests**

Assert the fixture parses, is bounded to two workers and isolated HF prefix, names exactly success/valid-failure/invalid-observation/infrastructure-failure/malformed-result/coordinator-restart/publication gates, and refuses `PASS` without attached evidence fields.

- [ ] **Step 2: Run red tests**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_control.py tests/test_generation_validation.py
```

Expected: FAIL because validation plan/receipt support is missing.

- [ ] **Step 3: Implement bounded validation orchestration inside canonical owners**

Use JSON plans and normal queue records. Fault injection is allowed only when `validation_mode=true` and must be represented in the receipt. Production mode rejects fault injection fields. Do not add an executable validation script.

- [ ] **Step 4: Update canonical documentation and run focused suite**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation.py tests/test_generation_*.py
python3 -B scripts/validate_fast.py
git diff --check
```

Expected: focused generation tests and L0 PASS; no simulator is launched locally.

- [ ] **Step 5: Commit**

```bash
git add scripts/generation_launch.py scripts/generation_coordinator.py \
  docs/components/generation.md tests/fixtures/generation_validation_config.json \
  tests/test_generation_control.py tests/test_generation_validation.py
git commit -m "feat(data): add bounded generation validation mode"
```

### Task 7: Verify the implementation locally

**Files:**
- Modify: `docs/plans/active/generation-runtime-hardening.md`

**Interfaces:**
- Consumes all previous task interfaces.
- Produces local PASS/FAIL/SKIPPED/NOT RUN evidence before remote execution.

- [ ] **Step 1: Run focused generation tests**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation.py tests/test_generation_*.py
```

- [ ] **Step 2: Run the full repository-owned local suite**

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests
```

- [ ] **Step 3: Run L0 and static cleanup checks**

```bash
python3 -B scripts/validate_fast.py
git diff --check
find src scripts tests -type f \( -name '*.pyc' -o -name '*.pyo' \) -print
git status --short
```

- [ ] **Step 4: Record exact evidence**

Record interpreter/platform, pass/fail/skip counts and explicit `NOT RUN` for CPU simulator/HF gates in `docs/plans/active/generation-runtime-hardening.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/plans/active/generation-runtime-hardening.md
git commit -m "docs: record local generation hardening evidence"
```

### Task 8: Execute and audit the two-worker Colab CPU acceptance run

**Files:**
- Create: `docs/experiments/generation-validation/validation-cpu-20260922/README.md`
- Add: small JSON receipts/log summaries only under the same directory; do not commit episode binaries or credentials.
- Modify: `docs/plans/active/generation-runtime-hardening.md`

**Interfaces:**
- Consumes canonical config, launch, worker, coordinator, watchdog and publisher.
- Produces isolated HF prefix `validation/validation-cpu-20260922/` and a checked-in evidence summary.

- [ ] **Step 1: Create a CPU session and resolve machine config**

Use `colab new -s generation-validation-cpu` for a CPU runtime. Query server assignments before creating another session. Generate the runtime JSON on the remote machine from resolved absolute paths and record its SHA256; never store tokens in it.

- [ ] **Step 2: Provision and preflight with canonical commands**

Run environment provisioning from `scripts/generation_environment.py --runtime-config ...`, install the committed source, and run remote focused contract tests. Record all revisions and checksums.

- [ ] **Step 3: Run two persistent workers and bounded cases**

Execute the validation plan through `generation_launch.py`. Collect evidence for success, valid failure, invalid observation, infrastructure failure, malformed-result quarantine, continued ingestion and resource bounds.

- [ ] **Step 4: Exercise coordinator restart recovery**

Stop only the coordinator process, allow the canonical watchdog/recovery mechanism to restore it, and verify no duplicate job/attempt/episode IDs or quota increments.

- [ ] **Step 5: Enable isolated publication and verify remote hashes**

Publish only to `validation/validation-cpu-20260922/`. Verify dataset manifest, attempt/episode paths, view pointers, publication receipt and every remote SHA256. Record commit revisions.

- [ ] **Step 6: Stop the disposable CPU session and preserve evidence**

After receipts are downloaded, stop the CPU session explicitly. Confirm `colab sessions` no longer lists its endpoint. Do not delete the isolated HF validation evidence unless separately requested.

- [ ] **Step 7: Write experiment record and final verification**

The README must list exact commands/environment, per-gate PASS/FAIL/NOT RUN, remote prefix/revisions, file counts/hashes and remaining production risks. Rerun local focused tests, L0 and `git diff --check` after importing only small receipts.

- [ ] **Step 8: Commit and close the plan only if every required gate passed**

```bash
git add docs/experiments/generation-validation docs/plans/active/generation-runtime-hardening.md
git commit -m "test(data): record two-worker CPU generation validation"
```

Move the active plan to `docs/plans/completed/` only when all required CPU gates pass; otherwise keep it active with blockers recorded.
