# Training export and lazy HF reader

Status: approved design; implementation is intentionally bounded to the archive
training boundary. Full generation remains paused.

## Goal

Publish a compact, revision-bound training index under the exact Hugging Face
dataset prefix `training`, and read its samples lazily from the immutable lossless
archive without copying numeric archive chunks.

## Current and target states

The archive profile already writes lossless episode manifests and chunks, and the
final-view finalizer writes twelve role/view pointer snapshots. Those snapshots
are not currently a training dataset API: the lazy archive index expects a local
dataset manifest and the native `icgs train` command expects a PyG directory.

The target export is a small `training/manifest.json` plus twelve compact view
files under `training/views/<role>/<view>.json`. Every export records the source
HF repository, immutable archive revision, source prefix, final-view revision and
hashes, dataset-manifest hash, archive/preprocessing identities, split roles, and
sample counts. The export contains references only; source chunks remain under the
archive prefix.

The reader resolves a pinned export and returns the existing `SampleRef` contract
for `D_geom`, `D_temporal`, `D_dyn`, and `D_task`. It downloads only the selected
episode manifest and NPZ chunk into an owned bounded cache, verifies declared
bytes and SHA256 values, and delegates observation/transition decoding and the
existing native 2,048-point preprocessing to `EpisodeArchiveReader` and the
current geometry/preprocessing owners. A0/A1 adapters remain the existing
`episodes.py` adapters; this change does not alter native IP training semantics.

## Export identity and layout

```text
training/
├── manifest.json
└── views/
    ├── train/{D_geom,D_temporal,D_dyn,D_task}.json
    ├── validation/{D_geom,D_temporal,D_dyn,D_task}.json
    └── evaluation/{D_geom,D_temporal,D_dyn,D_task}.json
```

The export manifest is `icgs_training_export_v1`. Its source binding is the
archive dataset revision and manifest hash; its target prefix is always exactly
`training`. Re-running against the same source produces byte-identical files and
is accepted. A partial or conflicting `training` prefix fails closed.

## Interfaces

- `export_training_views(...) -> TrainingExportReceipt`
- `TrainingExportReader.from_hf(repo_id, revision, prefix="training", ...)`
- `TrainingExportReader.sample_refs(view, role) -> Iterator[SampleRef]`
- `TrainingExportReader.reader_for(ref)` and `read_sample(ref)`
- `export_training_views(...)` for local final-view snapshots
- `scripts/generation_training_export.py` for pinned HF export/publication

The live CLI uses the caller's normal Hugging Face authentication. It never
embeds or prints token values. The archive source revision and final-view
revision are explicit; the exporter downloads only the dataset manifest and
final-view JSON files while
the reader fetches numeric chunks on demand.

## Invariants

- `success` and eligible `valid_failure` episodes remain readable; crash and
  invalid-observation attempts are excluded by the existing final views.
- The final-view 7:3 nominal/perturbed selection is preserved; the exporter does
  not re-sample or reinterpret it.
- Source revision, dataset-manifest hash, view hashes, archive identities and
  preprocessing identity are checked before a reader returns samples.
- No raw archive chunk is copied into `training`.
- Existing archive, finalizer, v2 layout, native PyG reader, model/action
  semantics, and generation prefixes remain unchanged.
- Live HF publication and actual model training are validation steps, not import
  or unit-test side effects.

## Out of scope

This change does not teach the current native `icgs train` runner to accept an
archive reader, invent missing A0/A1/B model losses, regenerate final views, or
start training. Those are separate integration decisions after the reader's
validation receipt exists.

## Validation

Local fixture tests cover export identity, idempotency, role/view hashes, source
binding, lazy reference selection, remote-file hash checks, bounded cache
eviction, and A0/A1-compatible sample adapters. After implementation, the
operator will run the CLI against the pinned prior HF validation archive and
publish/read the derived `training` prefix. This task stops before that live test
by user instruction.
