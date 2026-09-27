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

## Lossless HF archive profile

[ADR0015](../decisions/0015-generation-storage-and-view-snapshots.md) accepts an
opt-in archive identity `icgs-primary-v3-archive-v1` / `icgs_npz_chunked_v1` /
`icgs_episode_archive_v1`. Runtime configs without `archive_profile` retain the
existing v3 JSON behavior. The archive profile keeps full-resolution measured
clouds and every captured information-bearing debug modality in Hugging Face;
there is no raw-only local or undocumented cold-storage tier. A valid
`success` or `valid_failure` is a complete episode archive. A
`simulator_crash` or `invalid_observation` is an attempt archive with a null
`episode_id`, including any measured prefix needed for diagnosis. All four
records remain in the HF archive and retain their immutable file hashes; only
successes and valid failures can enter training views.

The archive tree for one record is:

```text
<hf-prefix>/episodes/<program>/<episode>/
├── episode.manifest.json       # identity, timeline, array pieces and profile
├── data/chunk-*.npz             # lossless numeric arrays; allow_pickle=False on read
├── debug.json                   # bounded scalar/string metadata and array references
└── artifact_manifest.json       # complete file/byte/SHA256 inventory
```

Crash and invalid records use the same tree under
`<hf-prefix>/attempts/<program>/<attempt>/` with `attempt.manifest.json`.
Ragged clouds use values plus offsets, validity masks are bit-packed and
losslessly unpacked, and byte-identical measured arrays may use an explicit
semantic alias. `max_chunk_bytes` defaults to 256 MiB of uncompressed NPZ member
bytes (including `.npy` headers); the manifest records the local write bound.

The archive writer/reader, both local materialization paths, archive-aware worker
result detection, profile-aware validation, HF-only resume bootstrap, lazy index,
and final-view builder are implemented and fixture-tested locally. The
coordinator validates the canonical archive inventory before ingestion and the
publisher revalidates it immediately before upload. Validated dataset rows
contain HF-relative archive references and complete file hash inventories, not
absolute local result paths. Legacy configs without `archive_profile` retain
their JSON/layout behavior. Verified receipt-only pruning retains a per-job
queue receipt, immutable manifest row, artifact hashes, source run identity and
remote commit OID; pruning is allowed only after pinned-revision byte verification
of every archive and control/view receipt, and is rejected in validation mode
whose checked-in profile remains `keep`. Live HF publication/finalization and
production-run authorization remain separate gates. See the [active
implementation plan](../plans/active/generation-storage-and-view-finalization.md)
for phase evidence and the [generation artifact inventory](generation-artifacts.md)
for the legacy/new tree comparison.

### Provisional and final views

Materialization may write four per-episode metadata-only pointers (`D_geom`,
`D_temporal`, `D_dyn`, `D_task`) with `status: "PROVISIONAL"`, `role: "all"`,
and `mix: false`. They are discovery indexes and do not represent the training
mixture. Each cumulative prefix discovery view written during publication is
also `status: "PROVISIONAL"` and `pointers_only: true`, but intentionally omits
the per-episode `role` and `mix` fields. The finalizer reads a complete `dataset_manifest.json`
from one pinned HF revision, verifies every selected episode manifest, then writes
twelve immutable role/view snapshots (`train`, `validation`, `evaluation` × the
four views) with `status: "FINAL"`. A final snapshot records the source revision,
dataset-manifest SHA256, archive/preprocessing identities, role, mixture seed,
sample counts and source-manifest timestamp. Re-running from remembered local
state is not valid. A source revision or manifest mismatch fails closed only
when reusing the same existing output target; a later source revision uses its
own `<output-prefix>/<source-revision>/seed-<mixture-seed>/` directory.
The 70/30 nominal/perturbed mixture is applied only to final train
`D_temporal`/`D_dyn` transition references, in complete 7:3 units.

The native `icgs train` command still consumes its existing PyG sample directory.
It does not automatically ingest archive-backed views and does not persist their
view metadata. Open3D statistical-outlier filtering (SOR, 20 neighbours and
standard ratio 2) was not executed locally; fixture tests cover archive/index
contracts and deterministic sampling only.

Production archive publication exposes two bounded runtime controls:
`run.publication_batch_size` (results per HF data commit) and
`run.publication_upload_threads` (Hugging Face multipart upload threads). Both
default to `1` for compatibility with the conservative validation profile. A
production run may raise them only after a bounded live benchmark verifies
archive hashes, pinned-revision verification, receipt-only pruning, and resume
identity. They are persisted in the runtime snapshot and run receipt, so a
resume cannot silently change publication behavior.

## Full-generation readiness

The checked-in `src/icgs/configuration/profiles/generation_runtime.json` is a
**validation/setup example**, not a production launch profile. A compatible
Linux host can be rebuilt from a clean clone using [Environment
setup](environment.md#rebuild-a-generation-host-from-a-clone); host-specific
runtime JSON and simulator assets may live under the clone's ignored
`outputs/` tree, while the HF credential stays outside Git. Rebuilding that
environment does not authorize the 7,520-attempt collection.

As of 2026-09-27, a one-job G1 no-HF capacity probe and live
HF publication/recovery of deliberately triggered crash and invalid attempts
**PASS**ed on `vps-a`. Run-wide admission now uses the queue's existing POSIX
lock and durable per-attempt reservations. Production archive runtimes require
explicit `max_result_bytes`, `max_staging_bytes`, and `staging_reserve_bytes`.
The reserve must be at least twice the writer cap and leave room for a writer.
Claims conservatively count all retained run bytes plus outstanding full writer
reservations, the next writer cap and the fixed reserve, and check real disk
free space. HF verification/recovery scratch stays inside the run filesystem.
Expired leases, retries, quarantine and ready/ingested/keep-retention backlog
retain their reservations. Only verified receipt-only pruning releases an
attempt's reservation. A conflicting policy or an old nonempty queue without
one is rejected: use a fresh run, not a retroactive budget declaration.

This is cooperative admission/backpressure, not a filesystem quota for arbitrary
external writers. Keep run paths on one filesystem, leave host headroom, and
preserve abandoned reservations for inspection rather than clearing them to
force progress. A 230,889,909-byte writer upper bound and
a 230,898,507-byte sampled staging peak were measured for one G1 job; neither
is a bound for concurrent workers or every program. See the [launch readiness
plan](../plans/completed/generation-launch-readiness.md), the [active storage
plan](../plans/active/generation-storage-and-view-finalization.md) and its
[pinned validation record](../experiments/generation-validation/archive-aa869b3-20260927/README.md).

Before preparing any full-run launch, choose the actual single- or multi-host
layout, worker IDs and simulator slots, available staging filesystem, and a
bounded queue backlog. Establish and test an enforced reservation covering
concurrent per-result writer caps, unpruned closed results, remote-verification
scratch, and recovery margin; prove that another claim cannot pass the cap
during a still-live lease and that verified receipt-only pruning releases
capacity. Then perform a new bounded rehearsal on that selected layout and
record its receipts and a real per-program smoke receipt. Only after those
checks pass should an operator make a new production runtime config with a
fresh `run_id`/`run_root` and disjoint HF prefix, explicit positive
`max_result_bytes`, `publication_enabled: true`, `validation_mode: false`,
archive profile and `receipt_only` local retention. Keep the HF token path
coordinator-only and outside the runtime JSON. The launcher requires the
approved composition manifest, pinned code revision, and an actual smoke
receipt; do not fabricate a receipt to satisfy its parser. Production runtime
JSON also requires `max_staging_bytes` and `staging_reserve_bytes`.

The current launcher validates a supplied smoke receipt but does not itself
produce one; obtaining and reviewing that live receipt belongs to the
selected-host rehearsal, not to environment setup.

The selected `vps-a` two-worker/two-slot layout now has a genuine **36/36-program
PASS** smoke and **three-job PASS** HF backpressure/verified-pruning rehearsal.
The [readiness record](../experiments/generation-validation/readiness-20260927/README.md)
contains exact limits, receipts, commands and an unexecuted production launch
command. It closes this layout's storage and smoke blockers; it does not certify
another host/layout or start the 7,520-attempt collection. Training ingestion of
archive-backed views is a separate downstream gap, not evidence that the
collector passed.

## Distributed lifecycle

```text
planner → pending → claimed → ready → ingested → published
             worker ────────┘       coordinator ─────────┘
```

Workers have no Hugging Face token. The coordinator alone validates results,
updates quota state and publishes. Temporary directories containing `.partial-`
are not queue entries. Remote conflicts fail closed; matching immutable hashes
may be reconciled during an explicit resume.

For the archive profile, receipt-only retention follows pinned-revision byte
verification of every archive file, the dataset manifest, resume receipt,
publication receipt snapshot and all four provisional views. Per-file downloads
use independent temporary caches. Interrupted commits are reconciled from the
remote revision and exact local receipt snapshot before another upload is allowed.

### Resume after losing local state

Set `run.resume_from_hf` to `true` in the runtime profile and choose a new
`run_id` that is not present in the remote manifest's `source_run_ids`. The
coordinator reads `<hf_subfolder>/dataset_manifest.json` before constructing
the planner; the launcher performs the same read before starting any worker.
The manifest must be schema version 3, identify its source run(s), contain
unique immutable `episode_id`/`attempt_id` rows, and preserve any serialized
`attempt_plan`. Missing, malformed, conflicting, or unavailable remote state
fails closed. The fetched revision and SHA256 are written to
`control/resume_bootstrap.json` and reused by the coordinator. Archive-profile
resume requires a lowercase 40- or 64-character HF commit OID. Before restoring
planner state, the coordinator checks both remote receipts against the manifest,
downloads and hashes every declared archive file at that revision through owned
temporary scratch, runs the canonical archive validator, and requires each archive
manifest's full profile to equal the dataset/runtime profile. Attempt plans must
also match the program catalog's split, allowed kind, and approved asset family.
Legacy resume behavior is unchanged.

If the same local run has already published archive rows, the latest pinned
prefix must also bind to its local publication receipt chain. A stable local
`COMPLETE`/`VERIFIED` receipt requires the exact local publication manifest and
matching remote receipt; an unrelated newer repository commit is acceptable
when those prefix controls are unchanged. Receipt-only retention can recover a
pending batch from its durable per-job verified receipts. With `keep` retention,
an interrupted later batch may have overwritten the run-level controls without
leaving per-job receipt OIDs; that state fails closed with an instruction to
inspect pinned HF revisions. Stable `keep` restarts and clean-machine resume
remain supported. Persisting prior run-level controls for this pending case is
deferred to a separate protocol change.

The clean-machine resume command requires a coordinator-only credential path.
The file contains the token, but the token value is never embedded in a command
or committed to a runtime snapshot:

```bash
export ICGS_HF_TOKEN_PATH=/secure/credentials/hf-token
python3 -B scripts/generation_launch.py \
  --runtime-config /runs/resume/runtime.json \
  --approved-manifest artifacts/composition/approved_composition_manifest.json \
  --code-revision "$(git rev-parse HEAD)" \
  --smoke-receipt /runs/resume/smoke.json \
  --hf-token-path "$ICGS_HF_TOKEN_PATH" \
  --detach
```

The token path is never placed in worker command lines or worker environments.
The coordinator persists queue and publication receipts so a process restart
continues from immutable queue/HF state rather than resetting quota.

### Clean-machine archive inspection

The archive reader is a local-file API, not a `generation` CLI. Keep inspection
bounded to one selected archive: fetch its manifest, artifact inventory, debug
metadata and chunks at one immutable revision into a temporary directory, then
run the canonical validator and reader. Configure Hugging Face authentication
outside this command (`HF_TOKEN` or the machine's normal HF login); no token is
printed or placed in the command line.

```bash
export HF_DATASET_REPO=ORG/DATASET
export HF_SOURCE_REVISION=0123456789abcdef0123456789abcdef01234567
export HF_ARCHIVE_PREFIX=icgs-primary-v3-archive-v1/episodes/T01/EPISODE_ID
export HF_ARCHIVE_KIND=episode
python3 -B - <<'PY'
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from huggingface_hub import hf_hub_download
from icgs.data.collection.generation.episode_archive import (
    EpisodeArchiveReader,
    validate_archive_manifest,
)

repo = os.environ["HF_DATASET_REPO"]
revision = os.environ["HF_SOURCE_REVISION"]
prefix = os.environ["HF_ARCHIVE_PREFIX"].rstrip("/")

with tempfile.TemporaryDirectory(prefix="icgs-archive-read-") as temporary, \
        tempfile.TemporaryDirectory(prefix="icgs-archive-hf-cache-") as cache_temporary:
    root = Path(temporary)
    cache = Path(cache_temporary)

    def fetch(relative: str) -> Path:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError(f"unsafe archive path: {relative}")
        destination = root.joinpath(*path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=f"{prefix}/{path.as_posix()}",
            revision=revision,
            cache_dir=str(cache),
        )
        destination.write_bytes(Path(downloaded).read_bytes())
        return destination

    # Fetch the selected record's manifest first; crashes/invalid attempts use
    # attempt.manifest.json instead of episode.manifest.json.
    archive_kind = os.environ.get("HF_ARCHIVE_KIND", "episode")
    manifest_name = {
        "episode": "episode.manifest.json",
        "attempt": "attempt.manifest.json",
    }.get(archive_kind)
    if manifest_name is None:
        raise ValueError("HF_ARCHIVE_KIND must be episode or attempt")
    manifest_path = fetch(manifest_name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("archive_kind") != archive_kind:
        raise ValueError("archive manifest kind does not match HF_ARCHIVE_KIND")
    artifact_path = fetch("artifact_manifest.json")
    inventory = json.loads(artifact_path.read_text(encoding="utf-8"))["files"]
    for relative in inventory:
        fetch(relative)
    validate_archive_manifest(manifest_path)
    reader = EpisodeArchiveReader(manifest_path)
    print(json.dumps({
        "episode_id": reader.manifest.payload["episode_id"],
        "outcome": reader.manifest.payload["outcome"],
        "timeline": reader.manifest.payload["timeline"],
        "archive_format_id": reader.manifest.payload["archive_format_id"],
    }, sort_keys=True))
PY
```

This intentionally downloads one archive, never a whole HF prefix. A
metadata-only index can fetch `dataset_manifest.json` and the selected compact
episode manifests first; use a caller-owned fetch adapter to materialize only
the chunks needed for a sample. The native `ArchiveDatasetIndex` and
`EpisodeArchiveReader` do not stream directly from HF.

### Clean-machine final-view publication

`generation_finalize_views.py` is the only finalizer command. It downloads the
dataset manifest and compact episode manifests from the explicitly supplied
source prefix/revision, writes snapshots under a separate output prefix, and
verifies the committed bytes at the returned revision. Configure HF credentials
through the machine's normal login or `HF_TOKEN`; do not substitute a branch,
tag or remembered local manifest for the full commit OID.

```bash
export HF_DATASET_REPO=ORG/DATASET
export HF_SOURCE_PREFIX=icgs-primary-v3-archive-v1
export HF_SOURCE_REVISION=0123456789abcdef0123456789abcdef01234567
export HF_FINAL_VIEWS_PREFIX=icgs-primary-v3-final-views
python3 -B scripts/generation_finalize_views.py \
  --repo-id "$HF_DATASET_REPO" \
  --source-prefix "$HF_SOURCE_PREFIX" \
  --source-revision "$HF_SOURCE_REVISION" \
  --output-prefix "$HF_FINAL_VIEWS_PREFIX" \
  --mixture-seed 20260920
```

The finalizer never downloads numeric chunks and never rebuilds from remembered
local state. Each source revision gets its own output directory at
`<output-prefix>/<source-revision>/seed-<mixture-seed>/`. A conflict occurs only
when that same target already exists with different bytes or recorded
source-revision/manifest identity; a later source revision is a distinct target
directory and may be finalized independently.

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

Focused archive/view/resume validation uses tiny local fixtures and fake HF
clients; none of these commands contacts Hugging Face or starts a simulator,
training job, preprocessing workload or robot motion:

```bash
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_archive.py
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_views.py tests/test_generation_view_finalization.py
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_generation_control.py tests/test_generation_planner.py \
  tests/test_generation_publication.py tests/test_generation_queue.py
PYTHONPATH=src .venv/bin/python -B -m pytest -q \
  tests/test_capacity_probe.py
python3 -B scripts/validate_fast.py
git diff --check
```

Report each selected command as `PASS`, `FAIL`, `SKIPPED` (with the missing
prerequisite) or `NOT RUN` (not selected). A skipped required gate is never a
pass. The focused tests prove archive round-trip/hash/offset behavior, final
view role/mixture/revision binding, remote-resume contract fixtures and local
capacity budgets. They do not prove live HF publication, clean-machine remote
resume, Open3D SOR execution, native `icgs train` archive ingestion, simulator
behavior, full generation or C1–C5; those remain explicit acceptance gates.

Simulator execution, full collection and large Hugging Face publication require
an explicitly provisioned environment and separate authorization.
