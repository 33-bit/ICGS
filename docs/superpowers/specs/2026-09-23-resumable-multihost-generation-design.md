# Resumable Multi-Host Generation Design

Date: 2026-09-23. Status: accepted for implementation by the request to complete
HF resume and multi-host distribution.

## Goal

Allow a generation run to resume from its Hugging Face manifest after local run
state is lost, and allow workers on multiple hosts to share one run safely.

## Decision

run.resume_from_hf=true makes the remote dataset_manifest.json under the
configured HF prefix mandatory input to planner construction. The coordinator
merges the remote manifest with any surviving local closed queue state, restores
attempt planner uniqueness from serialized attempt_plan rows, and fails closed
when the remote manifest cannot be downloaded. A resumed run uses a new run ID
for new job IDs while retaining the configured HF prefix.

run.distribution_mode is either single_host or shared_filesystem.
Shared-filesystem runs use absolute paths visible on every host, atomic rename,
and POSIX flock semantics. Each runtime config names machine.host_id and an
explicit machine.worker_ids subset of the global worker range. A worker lease
records host and process instance identity; an active lease rejects a duplicate
logical worker ID. Expired leases can be replaced and stale claimed jobs are
recovered by the coordinator.

The normal launcher starts a coordinator, watchdog and the assigned local
workers. --workers-only --run-config ... attaches another host to the existing
run, starts only its assigned workers, writes a host-scoped launch receipt, and
starts a worker-only watchdog. The coordinator host remains the single planner,
validator and HF publisher.

## Invariants

- Existing serialized episode/schema/protocol strings are unchanged.
- Workers never receive the HF credential.
- A remote resume cannot silently start from an empty planner.
- Job IDs remain immutable and new resumed runs use disjoint run IDs.
- No two active hosts may hold the same logical worker lease.
- Queue state lives on one shared filesystem for multi-host mode.
- Coordinator remains the sole quota/publication owner.
- Single-host configs preserve current behavior when new fields are omitted.

## Acceptance

- Unit tests prove remote manifest is required and seeds planner counts/uniqueness.
- Unit tests prove attempt-plan restoration preserves perturbed source linkage.
- Unit tests prove duplicate active worker leases fail and expired leases can
  be replaced.
- Unit tests prove worker command generation honors host worker subsets.
- Unit tests prove workers-only launch writes a host receipt and does not spawn
  a coordinator.
- Unit tests prove worker-only watchdog does not alter coordinator state.
- L0 and generation tests pass; no simulator or full generation is launched.
