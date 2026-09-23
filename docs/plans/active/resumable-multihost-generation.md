# Resumable multi-host generation hardening

Status: active
Owner: ICGS data-generation runtime
Design record: `docs/superpowers/specs/2026-09-23-resumable-multihost-generation-design.md`
Motivation and observable acceptance criteria: resume quota and uniqueness from
the HF manifest after local state loss; attach disjoint workers on multiple
hosts without duplicate leases, stale-result corruption, credential leakage, or
coordinator duplication. No simulator/full-generation run is part of this
change.

## Current and target states

The canonical runtime is `src/icgs/data/collection/generation` with the
coordinator and worker entry points under `scripts/`. The target keeps all
serialized protocol/schema strings unchanged while adding strict runtime
configuration for `resume_from_hf`, `distribution_mode`, `machine.host_id` and
`machine.worker_ids`. A coordinator owns planning, validation and publication;
other hosts run workers and a host-scoped watchdog against one shared POSIX
filesystem.

## Invariants and scope

- Remote resume is fail-closed: manifest version, source-run identity, row
  identity, serialized attempt plans and immutable conflicts are checked before
  workers start.
- A resumed run uses a new disjoint run ID and keeps the configured HF prefix.
- Workers never receive HF credentials.
- Shared queue transitions use atomic rename plus POSIX `flock`.
- Active `(worker_id, host_id, worker_instance_id)` leases fence duplicates and
  late results; expired leases may be replaced.
- Stale retries increment `retry_generation` and use disjoint staging paths.
- Simulator execution, full generation, HF writes, and live multi-host
  provisioning are out of scope for local validation.

## Phases and progress

- [x] Runtime contract: strict host scope, distribution and resume fields;
  frozen-config normalization; planner restoration from `attempt_plan`.
- [x] Queue safety: serialized lease/claim/heartbeat/publish/recovery,
  heartbeat-aware stale recovery, retry fencing and host-local simulator slots.
- [x] Launch/watchdog roles: host-local runtime snapshots, credential-free
  workers, workers-only attach, digest/run identity checks and host-scoped
  reconciliation.
- [x] HF bootstrap: launcher preflight before child spawn, manifest validation,
  cached revision/SHA receipt and coordinator reuse of the pinned revision.
- [x] Documentation and no-secret profile example.
- [x] Bounded VPS acceptance with HF resume, pinned manifest, disjoint run ID and
  workers-only host attach is recorded in
  `docs/experiments/generation-validation/validation-vps-20260924/`.
  A true simultaneous multi-host production run remains outside bounded scope.

## Validation and experiment strategy

Run focused generation tests and L0 locally. Unit tests cover duplicate/expired
leases, concurrent lock serialization, stale-claim heartbeat behavior, late
result fencing, host-local command/config paths, credential filtering, strict
resume manifest validation, preflight-before-spawn, coordinator snapshot digest,
and retry output isolation. Do not treat skipped or unrun simulator/HF/live
checks as PASS.

## Compatibility

Existing single-host profiles omit new fields and normalize to all configured
workers. Existing episode/schema/protocol strings remain serialized exactly as
before. `retry_generation` is additive to internal job receipts and does not
change episode IDs or dataset record schemas.

## Risks, open questions and recovery

The shared-filesystem contract requires reliable atomic rename and POSIX locks;
NFS-like mounts without those guarantees are unsupported. If a live acceptance
check fails, stop only the new test workers, retain queue/HF receipts, and resume
from the recorded manifest with a fresh disjoint run ID. Do not delete prior
artifacts or rewrite published rows.

## Final evidence

Fresh local evidence on Python 3.11.15:

- `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests/test_generation*.py` —
  **PASS, 217 passed**.
- `PYTHONPATH=src .venv/bin/python -B -m pytest -q tests` — **PASS, 896 passed,
  10 skipped, 3 warnings**.
- `python3 -B scripts/validate_fast.py` — **PASS**, 22 harness tests; L1–L4
  explicitly **NOT RUN**.
- `git diff --check` — **PASS**; bytecode compilation — **PASS**.

Repository-wide discovery `pytest -q` remains **FAIL** because three generated
`output/icgs-figures*/src/test_typography.py` modules share one import name;
this is pre-existing output-tree collection pollution, not a runtime test
failure. Live simulator, multi-host, HF resume and HF write checks remain
**NOT RUN** until explicitly provisioned.
