# Data generation

ICGS has one supported data-generation system. Runtime owners live under
`src/icgs/data/collection/generation`; operational entry points use the
`scripts/generation_*` prefix.

Install and verify the host with the portable `generation` profile described in
[Environment setup](environment.md). Simulator provisioning is explicit and
idempotent; generation setup never starts a worker or coordinator by itself.

## Ownership

| Owner | Responsibility |
| --- | --- |
| `protocol.py` | Frozen dataset, schema, controller, quota and split identities |
| `steps.py`, `scenes.py`, `compiler.py` | Program semantics, scene layouts and executable task compilation |
| `batch.py`, `attempt_prep.py`, `quota.py`, `diversity.py` | Attempts, perturbations, seeds, uniqueness and stop conditions |
| `episode_record.py`, `rlbench_attempt.py`, `task_labels.py` | Measured episode/attempt materialization and labels |
| `distributed_contracts.py`, `distributed_queue.py` | Immutable jobs/results and atomic filesystem queue |
| `distributed_planner.py`, `distributed_validation.py` | Central quota authority and closed-result validation |
| `distributed_publication.py` | Conflict-safe Hugging Face commits and resumable receipts |
| `generation_worker.py` | Credential-free persistent worker |
| `generation_coordinator.py` | Single-owner ingestion, planning and publication control plane |
| `generation_watchdog.py` | Bounded replacement of missing worker/control processes |
| `generation_launch.py` | Preflight and launch of the configured distributed run |

`artifacts/composition/approved_composition_manifest.json` is the only approved
program manifest. `src/icgs/configuration/profiles/generation.json` owns
generation-specific method settings. There is no legacy protocol selection or
fallback collector.

## Record classes

`success` and `valid_failure` are valid episodes. Simulator crashes and invalid
observations are closed attempt records and must never be mislabeled as valid
failures. A valid episode has `T` actions, `T+1` causal online observations and
`T` durations. Every closed result carries an immutable file inventory and an
artifact manifest; publication starts only after validation.

Serialized records retain their existing `schema_version` and protocol identity
strings. Those values identify stored artifacts and are not source-module names.
Changing them requires a separate data migration.

## Distributed lifecycle

```text
planner → pending → claimed → ready → ingested → published
             worker ────────┘       coordinator ─────────┘
```

Workers have no Hugging Face token. The coordinator alone validates results,
updates quota state and publishes. Temporary directories containing `.partial-`
are not queue entries. Remote conflicts fail closed; matching immutable hashes
may be reconciled during an explicit resume.

### Resume after losing local state

Set `run.resume_from_hf` to `true` in the runtime profile and choose a new
`run_id` that is not present in the remote manifest's `source_run_ids`. The
coordinator reads `<hf_subfolder>/dataset_manifest.json` before constructing
the planner; the launcher performs the same read before starting any worker.
The manifest must be schema version 3, identify its source run(s), contain
unique immutable `episode_id`/`attempt_id` rows, and preserve any serialized
`attempt_plan`. Missing, malformed, conflicting, or unavailable remote state
fails closed. The fetched revision and SHA256 are written to
`control/resume_bootstrap.json` and reused by the coordinator.

The resume command requires a coordinator-only credential path:

```bash
python3 -B scripts/generation_launch.py \
  --runtime-config /runs/resume/runtime.json \
  --approved-manifest artifacts/composition/approved_composition_manifest.json \
  --code-revision "$(git rev-parse HEAD)" \
  --smoke-receipt /runs/resume/smoke.json \
  --hf-token-path /secure/credentials/hf-token \
  --detach
```

The token path is never placed in worker command lines or worker environments.
The coordinator persists queue and publication receipts so a process restart
continues from immutable queue/HF state rather than resetting quota.

### Multiple hosts on one shared filesystem

Use `run.distribution_mode: "shared_filesystem"`, an absolute `run_root` visible
on every host, a safe explicit `machine.host_id`, and a disjoint explicit
`machine.worker_ids` subset. The coordinator host launches normally. Each
additional host receives its own host-local runtime profile and attaches with:

```bash
python3 -B scripts/generation_launch.py \
  --runtime-config /shared/run/control/worker-a-runtime.json \
  --approved-manifest artifacts/composition/approved_composition_manifest.json \
  --run-config /shared/run/control/run.json \
  --workers-only
```

`--workers-only` writes `control/worker-launch-<host_id>.json`, starts no
coordinator, and starts a watchdog that reconciles only that host's worker
scope. POSIX `flock` serializes claim, lease, heartbeat, publish, recovery and
state transitions. A worker lease contains host and process-instance identity;
an active duplicate is rejected, an expired lease can be replaced, and a late
result from the old instance is fenced. Stale claims increment
`retry_generation`; retry artifacts use a disjoint staging directory, so an old
subprocess cannot overwrite a replacement attempt.

The coordinator remains the only planner, validator and HF publisher. Do not
run multiple coordinators for one `run_root`, and do not use shared-filesystem
mode when hosts cannot provide atomic rename and POSIX locking.

## Operational rules

- Use a new disjoint run ID after a dead Colab assignment.
- Resume from the remote manifest, never from remembered chat state.
- Preserve valid failures and closed infrastructure attempts.
- Do not count skipped validation as pass.
- Do not launch generation as incidental validation.
- Publication rate limits and transient upload timeouts defer work; they do not
  authorize dropping queue entries.

Bounded validation is data-driven. `tests/fixtures/generation_validation_config.json`
is a JSON plan consumed by the existing launcher and validation modules; it is not
an executable validation program. Plans are limited to at most two workers, eight
jobs, two episodes and four attempts, and validation receipts use explicit
`PASS`, `FAIL` or `NOT_RUN` states. A `PASS` requires gate evidence, while a
`FAIL` or `NOT_RUN` requires a reason. Validation receipts remain isolated under
the `validation/validation-cpu-20260922/` publication prefix.

Historical generation audits under `docs/audits/` record what happened during
earlier launches. They are evidence, not current commands.

## Validation

Cheap validation:

```bash
python3 -B scripts/validate_fast.py
PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation*.py
```

Simulator execution, full collection and large Hugging Face publication require
an explicitly provisioned environment and separate authorization.
