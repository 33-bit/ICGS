# 0015: HF source-of-truth generation archives and frozen view snapshots

Date: 2026-09-25. Status: accepted by explicit user agreement on the
storage/view redesign. Implementation is not yet complete.

## Context

The canonical generation runtime currently writes dense point clouds and other
numeric episode fields as indented JSON lists in `episode.json`, then writes
overlapping binary layout and telemetry sidecars. A measured 509-boundary
episode is approximately 1.1 GB, with the root JSON accounting for more than
94% of its bytes. The planned collection minimum is 7,520 valid attempts, so
the current representation projects to roughly 6.9–7.0 TB before retries.

The Hugging Face dataset is the system of record. Local worker/coordinator disks
are staging and cache only; losing local state must not lose raw observations,
valid failures, provenance, or debug evidence. Existing `icgs_episode_v2`
artifacts and the accepted generation protocol remain immutable historical
artifacts.

The current runtime also emits per-episode pointer views while materializing an
episode and rewrites cumulative prefix view files on each publication batch.
Those files are provisional indexes, not final training views. The 70/30
nominal/perturbed mixture and train/train_val/evaluation roles require a frozen
dataset-manifest snapshot.

## Decision

### 1. HF is the complete canonical archive

The next production storage profile uses a new serialized archive/schema
identity and a new HF prefix. It stores every information-bearing field needed
for training, debugging, replay diagnosis, and provenance in lossless binary
chunks plus compact JSON manifests. Full-resolution point clouds are retained
in HF; they are not moved to an undocumented local or external-only tier.

The canonical per-episode inventory is:

```text
episode.manifest.json
data/chunk-*.npz
debug.json
artifact_manifest.json
```

Chunks use `allow_pickle=False` and contain raw online/measured clouds (with
offsets and validity masks), poses, grips, commands, durations, substeps,
robot/object states, task labels, and any actually captured sensor modalities.
If online and measured arrays are byte-identical, the manifest may alias one
chunk rather than duplicate it. If they differ, both remain available.

`debug.json` retains scalar/string execution metadata, seeds, randomization,
intervention, controller/simulator/camera identities, task binding, errors and
traceback evidence. It must not contain a second copy of large numeric arrays.
Unique telemetry such as depth, masks, forces or simulator-specific state is
stored in binary chunks, not discarded.
Archive aliases preserve semantic roles and the target's full piece/range map;
content equality alone never aliases unrelated fields. The explicit measured
point-cloud alias is `raw_arrays/measured_points` →
`online_observations/points` when dtype, shape, and bytes match. Attempt error,
traceback, and diagnostic strings are bounded and recursively redacted in every
archived copy, including manifest metadata and `debug.json`.

### 2. Training views are derived, not a second source of truth

The training reader derives the fixed 2,048-point representation lazily from
the lossless HF chunks using a recorded preprocessing identity and hash. A
rebuildable derived cache may be published separately, but it is never the only
copy and never replaces the raw canonical archive.

Per-episode pointer views may be emitted during materialization for discovery.
They are explicitly `PROVISIONAL`. Final `D_geom`, `D_temporal`, `D_dyn` and
`D_task` snapshots are generated only after a dataset-manifest revision is
frozen. The final view manifest records the source HF revision/SHA256, role,
mixture seed, filter identity, and sample counts. The 70/30 mixture is applied
only during this finalization step (or to an explicitly named provisional
snapshot), never treated as a collection quota.

Crash and invalid-observation attempts remain out of training views. Success
and `valid_failure` episodes remain in HF and in eligible views according to
their split/subset rules.

### 3. Bounded local retention

New production profiles use an explicit receipt-only local retention mode:
workers stage one compact episode/chunk set, the coordinator publishes and
verifies remote hashes, then removes the large local result while retaining the
queue receipt, artifact hashes, and remote commit identity. Resume reconstructs
state from the HF manifest and episode manifests. A keep-local mode remains
available for bounded validation and forensic reproduction.

### 4. Compatibility boundary

Existing `icgs_episode_v2`, protocol identity strings, published prefixes and
historical HF revisions are not rewritten. The new archive has a new dataset,
archive and episode-schema identity. Native IP preprocessing, action/graph
semantics, program quotas, valid-failure retention, split rules and checkpoint
behavior remain unchanged unless a separate research decision records otherwise.

## Alternatives considered

1. **Delete debug sidecars only.** Rejected: the measured root JSON is the
   dominant cost, so this saves well below 1%.
2. **Publish only 2,048-point derived clouds.** Rejected as the sole archive:
   it would prevent future preprocessing changes and reduce forensic evidence.
3. **Keep raw clouds only in local/cold storage.** Rejected because HF is the
   source of truth and local state is not durable.
4. **Use Zarr/HDF5 immediately.** Deferred: chunked formats may improve random
   access, but they add dependency and remote-integration complexity. Existing
   compressed NPZ shards provide a lower-risk first migration. A later decision
   may select another chunk backend after measured training I/O evidence.

## Consequences

Lossless binary canonicalization is expected to reduce the current multi-terabyte
projection to roughly 0.4–0.5 TB, with a further optional reduction from
rebuildable 2,048-point caches. The archive remains inspectable from a clean
machine using HF alone. Publication commits become smaller and can batch more
episodes, but the reader, writer, validator, publisher, resume bootstrap and
view finalizer all require a coordinated migration.

The new writer must preserve exact timeline cardinality (`T+1` observations,
`T` actions, `T` durations), valid-failure semantics, immutable hashes and
causal view rules. Any lossy quantization or depth-only replacement is a later
experiment with its own fidelity evidence.

## Compatibility implications

- **Dataset:** new prefix/schema; old v2 data remains readable and immutable.
- **HF resume:** bootstrap validates the new dataset manifest and per-episode
  archive manifests; local episode binaries are not a resume prerequisite.
- **Training:** lazy archive reader and view adapters replace eager JSON-array
  loading; model/action semantics do not change.
- **Evaluation:** dev/test episodes remain evaluation-only; final view snapshots
  are revision-bound.
- **Validation:** focused archive/view tests and L0 are required locally; live
  simulator, HF publication and full quota runs remain explicit gates.
