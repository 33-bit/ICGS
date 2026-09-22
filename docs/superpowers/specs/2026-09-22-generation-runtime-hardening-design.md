# Generation Runtime Hardening Design

Date: 2026-09-22
Status: proposed for implementation
Scope: bounded CPU integration run with two workers, followed by general
distributed-runtime hardening

## Goal

Make the canonical generation runtime portable across Colab CPU/TPU VMs and
local Linux hosts, then use a two-worker CPU run to reproduce and close the
failure modes observed in the previous full TPU session.

The change must not add disposable debug scripts, must not start full
generation, and must not publish into the normal dataset prefix. The only
remote publication allowed by this design is a small, isolated validation
prefix under `validation/<run-id>/` in `33bit/icgs`.

## Root causes in scope

The previous run exposed four coupled classes of failures:

1. Simulator/process pressure was not a launch invariant. The worker slot pool
   existed, but the launch contract did not explicitly configure simulator
   concurrency or machine resources.
2. A malformed or incomplete ready result could terminate the coordinator tick,
   preventing refill and making the queue appear stalled.
3. Coordinator liveness was inferred mainly from PID existence. Synchronous HF
   upload and remote hash verification could block progress while the process
   remained alive.
4. The writer/publication contract was repaired after the first run, but the
   repaired contract was not proven through a portable, sustained, two-worker
   run with real publication.

## Architecture

### Portable machine configuration

Add one validated JSON machine/run configuration consumed by every canonical
entry point. It is explicit input, not a checked-in secret or a one-off script.
The configuration owns:

```json
{
  "machine": {
    "repo_root": "/absolute/path/ICGS",
    "python_executable": "/absolute/path/bin/python",
    "simulator_root": "/absolute/path/CoppeliaSim",
    "rlbench_root": "/absolute/path/RLBench",
    "display_base": 200,
    "display_width": 1280,
    "display_height": 1024,
    "simulator_slots": 2,
    "worker_timeout_s": 1200
  },
  "run": {
    "run_id": "validation-<timestamp>",
    "run_root": "/absolute/path/run",
    "worker_count": 2,
    "publish_interval_s": 300,
    "hf_repo": "33bit/icgs",
    "hf_subfolder": "validation/<run-id>",
    "publication_enabled": false
  }
}
```

No runtime owner may assume `/content`, a fixed Python environment, a fixed
display range, exactly 200 workers, or an implicit simulator slot count. The
production profile may still request 200 workers; the CPU profile requests two.
The approved manifest, code revision, and manifest digest remain immutable
run-contract fields.

### Queue and coordinator failure isolation

The coordinator processes each ready result independently:

```text
ready result
  ├─ valid → ingest → planner → publication queue
  └─ invalid → quarantine receipt + counted infrastructure attempt → continue
```

Validation errors never escape the main tick without a durable diagnostic. A
quarantined result retains job ID, attempt ID, program, outcome, exception,
traceback, file inventory and source path. It cannot be silently re-planned as
success, and it cannot kill the coordinator loop.

The coordinator heartbeat records phase, last successful progress timestamp,
last validation error, last publication result, retry cooldown and queue
counts. A long upload is visible as `publication_in_progress`, not as a fresh
healthy heartbeat.

### Publication isolation

HF publication is a bounded state transition. The coordinator must retain an
immutable local receipt until both the data commit and the receipt/hash
verification phase are resolved. A data commit followed by a failed receipt
commit is reconciled by remote immutable identity before retrying; it is never
blindly duplicated.

The CPU test uses exactly one or a few episodes and attempts under
`validation/<run-id>/`, never `generation/`, `primary_v3/`, or another normal
dataset prefix. Workers never receive the HF credential.

## CPU acceptance run

The run is intentionally small and bounded:

1. Provision one CPU Colab session using the portable machine config.
2. Install the pinned simulator/RLBench dependencies through the canonical
   environment owner.
3. Run two persistent workers with a configured simulator slot count.
4. Exercise one success, one valid failure, one invalid observation and one
   simulator/infrastructure failure.
5. Inject a malformed closed result through the canonical queue fixture path;
   prove it is quarantined and the coordinator continues processing another
   valid result.
6. Stop/restart only the coordinator process inside the run to prove queue
   recovery and disjoint job identity. Do not restart the simulator workers as
   part of this test unless a real failure requires recovery.
7. Publish a bounded validation batch to the isolated HF prefix and verify
   every published file hash, manifest row and outcome class.
8. Produce a machine-readable validation receipt and stop the disposable CPU
   session after evidence is collected.

This is not a smoke test that launches one job per worker and exits. It is a
short sustained queue test with real materialization, validation, recovery and
publication boundaries.

## Canonical interfaces

The implementation modifies existing owners only:

- `src/icgs/data/collection/generation/distributed_contracts.py`: portable
  machine/run configuration and explicit validation.
- `src/icgs/data/collection/generation/distributed_queue.py`: quarantine and
  durable diagnostic state transitions.
- `src/icgs/data/collection/generation/distributed_validation.py`: structured
  validation diagnostics without process-fatal control flow.
- `scripts/generation_launch.py`: config-driven environment construction and
  worker/coordinator launch.
- `scripts/generation_worker.py`: configured executable, timeout, display and
  simulator-slot handling.
- `scripts/generation_coordinator.py`: per-result isolation and progress
  heartbeat.
- `scripts/generation_watchdog.py`: heartbeat-progress/liveness checks with
  configurable thresholds.
- `src/icgs/data/collection/generation/distributed_publication.py`: isolated
  publication receipts and idempotent remote reconciliation.

No `debug_*.py`, `tmp_*.py`, `smoke_*.py`, or Colab-only duplicate entry point
will be created.

## Invariants

- `success` and `valid_failure` remain episodes with complete `T/T+1/T`
  timelines and full training layout.
- `simulator_crash` and `invalid_observation` remain attempt records with null
  episode IDs.
- A malformed result is quarantined, never relabeled as a valid failure.
- A retry/restart never creates a duplicate immutable job or episode identity.
- HF publication is disabled by default and enabled only by explicit config.
- Validation publication uses a disjoint prefix and never mutates historical
  prefixes.
- Existing serialized artifact identities are not changed by this runtime
  hardening.

## Failure evidence required

The final receipt must report each category separately:

| Gate | Required evidence |
|---|---|
| portability | config path, resolved executables, simulator roots, OS/Python |
| materialization | file inventory, layout manifest, timeline, measured origin |
| outcome taxonomy | counts for success/valid_failure/crash/invalid_observation |
| queue isolation | quarantined malformed result plus later valid result ingested |
| recovery | coordinator restart, no duplicate job IDs, resumed queue counts |
| publication | isolated prefix, commit revisions, remote hash verification |
| resource bound | worker count, simulator slots, timeout, peak observed queue |

## Out of scope

- Full quota generation.
- TPU provisioning or 200-worker production launch.
- Deleting or rewriting existing HF data.
- Changing serialized schema/protocol identity strings.
- Training, benchmark evaluation or robot motion.

## Acceptance criteria

The design is accepted only when the CPU receipt demonstrates every required
gate above, focused and full local tests pass, L0 passes, and no generated
one-off files remain in the repository. If the CPU session cannot satisfy a
gate, the receipt must mark it `FAIL` or `NOT RUN` with the exact blocker; it
must not be called a PASS by inference.
