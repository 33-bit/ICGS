# Generation launch readiness

Approved scope: user agreement on 2026-09-27, following Paseo session
`15256c9e-1435-4982-ab45-98b6349bb43c`. Small engineering changes to the existing
queue/setup owners; no dataset, seeds, quota, action, or checkpoint changes.
Work directly on main. Execute tests and bounded live acceptance only on
`vps-a`, under `/home/huy2325/ICGS`. Full collection and training are not started.

## Design and acceptance

Use existing POSIX queue lock, not a service or alternate scheduler. Persist an
immutable run-wide policy (`max_result_bytes`, `max_staging_bytes`,
`staging_reserve_bytes`) and durable per-attempt reservations before claim rename.
Conservatively count all local bytes plus outstanding full writer reservations;
do not subtract changing active payload sizes. Reserve at least two result caps
for sequential HF verification/recovery. Check actual free disk too. Backpressure
returns no claim, letting publication drain. Keep reservations across leases,
retry, quarantine, ready, ingested and keep-retention publication. Only verified
receipt-only pruning releases the corresponding attempt. Abandoned reservations
fail closed; never erase unpublished forensic scratch to resume automatically.
HF scratch uses the reserved run filesystem. Legacy non-archive behavior remains.

Boundary record (encode-invariant): authority is the user's approved design
above and ADR0015's bounded-local-retention decision. Scope is production archive
runtime admission, not model/runtime package architecture. Allowed: a fresh queue
with valid, matching shared limits; bounded validation without a shared policy
is an explicit compatibility exception. Forbidden: production archive entry
without the budget, conflicting limits, claim beyond available capacity, or
release on lease expiry/unverified publication. Diagnostics name the missing
limit/conflict; supply positive limits or preserve the old run and choose a fresh
one. Native queue checks and `test_generation_storage.py` own enforcement;
they cannot police arbitrary external filesystem writers. No hooks, CI changes
or branch-protection settings are installed or claimed.

## Work and proof

- [x] Rebuild locked Python 3.10 generation environment in clone; cache/temp/assets
      remain in ignored clone paths. Setup and dependency verification passed.
- [x] RED/GREEN shared storage tests on VPS: first eight failed for absent API;
      then 106 storage/queue/config assertions passed. Worker entry regressions
      failed before integration; 11 storage tests then passed.
- [x] Close independent review: direct runtime-entry enforcement and reserved
      filesystem HF scratch; verify with targeted regressions and full suite.
- [x] Complete generation task build/setup verification. Live initial smoke
      found missing generated `rlbench.tasks.*` modules after fresh provisioning.
      Preserve failed attempts under `outputs/readiness-smoke` on VPS.
- [x] Run genuine 36-program smoke, two workers/two slots, one attempt/program,
      isolated no-refill batches; keep all failures. Validate canonical archives.
- [ ] Rehearse bounded HF publication, pinned verification and receipt-only
      pruning on disjoint validation prefix; prove a blocked claim resumes only
      after pruning. Credentials coordinator-only and never committed/logged.
- [ ] Record commands, revisions/patch hashes, environment, receipts and remaining
      limitations in a checked-in evidence record. Update current owner docs.
- [ ] Commit/push main, sync VPS, leave full-run config/launch instructions only
      if all required gates pass. Do not label partial smoke/full collection PASS.

## Recovery

No deletion of existing unpublished outputs. A process crash leaves durable
reservations and retry-fenced payloads. Inspect their HF fate before recovery;
use a new run after preserving evidence if uncertainty remains. No architecture
or research baseline redesign is included. Archive-backed training ingestion
remains separate downstream work.

## Evidence location

Commands, intermediate failures and final gate results belong in the
[readiness record](../../experiments/generation-validation/readiness-20260927/README.md).
The initial no-deletion rule was violated for one aborted, unpublished smoke
restart directory; this was disclosed and the irrecoverable loss is recorded
there. All subsequent evidence is preserved. The 36-program smoke continues
from closed archives after its first helper process was interrupted while
extending the watchdog; no completed episodes were regenerated.
