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

Materialization and each publication batch may write four metadata-only pointers
(`D_geom`, `D_temporal`, `D_dyn`, `D_task`) with `status: "PROVISIONAL"`,
`role: "all"`, and `mix: false`. They are discovery indexes and do not represent
the training mixture. The finalizer reads a complete `dataset_manifest.json`
from one pinned HF revision, verifies every selected episode manifest, then writes
twelve immutable role/view snapshots (`train`, `validation`, `evaluation` × the
four views) with `status: "FINAL"`. A final snapshot records the source revision,
dataset-manifest SHA256, archive/preprocessing identities, role, mixture seed,
sample counts and source-manifest timestamp. Re-running with remembered local
state is not valid; a changed source revision or manifest hash fails closed.
The 70/30 nominal/perturbed mixture is applied only to final train
`D_temporal`/`D_dyn` transition references, in complete 7:3 units.

The native `icgs train` command still consumes its existing PyG sample directory.
It does not automatically ingest archive-backed views and does not persist their
view metadata. Open3D statistical-outlier filtering (SOR, 20 neighbours and
standard ratio 2) was not executed locally; fixture tests cover archive/index
contracts and deterministic sampling only.

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

with tempfile.TemporaryDirectory(prefix="icgs-archive-read-") as temporary:
    root = Path(temporary)
    cache = root / "hf-cache"
    cache.mkdir()

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

    manifest_path = fetch("episode.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
local state. The output path is bound to
`<output-prefix>/<source-revision>/seed-<mixture-seed>/`; an existing snapshot
with different bytes or a source manifest whose revision/hash changed fails
closed.

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
