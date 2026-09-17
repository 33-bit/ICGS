# 0011: Lossless episode archives and initial geometry/dynamics views

Date: 2026-09-15. Status: accepted by delegated owner decision.
Authority: owner requested completion and on 2026-09-15 explicitly delegated
evaluation and contract decisions to the coordinating agent.

## Context and scope

[P02](../plans/active/icgs-p02-episode-data.md) has an in-memory episode validator,
causal-prefix helpers and bounded `collect_attempt`, but no persistent dataset.
[Data/training](../method/data-training.md) already requires numeric shards,
JSON manifests, failure retention, split-before-generation and physical truth.
This decision supplies the missing storage and initial-view contract. It does
not change native IP, P00 timed records, model losses or training schedules.

## Decision

### Archive and publication

Use an explicitly supplied dataset root with `episodes/<episode_id>/manifest.json`
and numbered `shard_0000.npz` files. IDs must be nonempty safe single path
components, excluding dot components; never interpret IDs as arbitrary paths.
Published episode directories are immutable: refuse collisions, never replace
an existing episode. Use a unique temporary sibling directory, validate its
checksums and reconstructed record, then rename it into place. No readers see
partially published episodes. Refuse symlink/path traversal shard references.
No pickle, executable configuration, database, plugin registry or network store.

Each shard has at most `cfg.dataset.shard_intervals` executed transitions and
their N+1 boundary observations. Record explicit start/end boundary indices;
duplicate shared boundaries must agree exactly and views count them once.
Store SHA256 of each NPZ byte stream, plus array schema, in the manifest.
The archive envelope uses `archive_version=1` and nests the unchanged in-memory
`icgs_episode_v1` provenance rather than silently repurposing its allowlist.

The writer and loader must roundtrip ALL in-memory observation and transition
fields without numerical loss. Preserve numeric float32/float64 dtypes and values
of online observations; timed-record measurements already own float64 values.
Do not reconstruct measured transitions from the approximately matching padded
online observations: the existing validator explicitly permits their difference.
Store canonical measured clouds/poses separately from online arrays where needed.
Ragged clouds use concatenated values and int64 offsets (N+2 offsets for N+1
boundaries), with boolean point masks for online arrays. Offset origin is zero,
offsets are monotonic, final offset equals values length, shapes/dtypes/finite
values and array-key inventory must validate before record construction.
Unsupported dtypes are rejected explicitly, not silently converted.

Persist measured boundary index, simulator/wall timestamps and sensor-profile ID;
command target/grip/planned duration; achieved duration, substeps and controller
status. Metadata is strict finite JSON (`allow_nan=False`); unknown unavailable
measurements are null with a reason rather than guessed values. Persist attempt
status, per-step annotations and operational/cleanup errors separately from the
online data. Empty attempts get an attempt report, not an invented transition.

Valid task failures/timeouts are eligible physical training data. Corrupt or
incomplete records go under `quarantine/<attempt_id>/` with diagnostics and any
serializable retained evidence, not discarded or relabeled as physical failure.
Valid prefixes of interrupted attempts may be archived with explicit incomplete
status; never include rejected transitions in the accepted causal chain. Handle
non-JSON diagnostic payloads explicitly; never pickle them or silently lose them.

### Dataset manifest and lineage

A finite versioned dataset manifest lists episode manifest paths and SHA256s,
explicit lineage nodes and asset-family split assignments. Each lineage node has
`lineage_id`, `parent_ids`, and `split`; parents must exist, graph must be acyclic,
and every parent/descendant must keep the same split. Seed roots have no parents.
Every episode source lineage and asset family must resolve in this manifest;
program IDs must match the explicit catalog split (`development` maps to `dev`
at this boundary). Reject conflicting IDs, duplicate episodes, dangling parents,
cycles, split disagreement and train/dev/test family leakage before building views
or starting collection. Do not infer ancestry from filenames or caller-declared
root strings alone. A deterministic content hash identifies the dataset manifest.

Run provenance must record actual generator/reset/action seeds, resolved config
identity, code revision/dirty-patch identity, generator version, environment and
controller/sensor/asset/split protocol identities. Caller-owned protocol artifacts
remain explicit inputs; this software does not manufacture physical certifications.

### Initial views and consumer boundary

`build_view(dataset_manifest_path, view, *, split, config)` supplies only `geom`
and `dyn`; reject other view names explicitly. No duplicate copied dataset banks.
Each sample has provenance, stable episode/boundary or episode/transition IDs,
and separation of causal inputs from future supervision.

`geom` exposes each observed boundary and validity mask once, retaining pose/grip
metadata for later fidelity checks. `dyn` exposes reset-to-root causal history,
commands and executed successor targets, including sequences crossing shards.
Never reset memory at a shard/clip boundary or use a future observation as input.
Provide an explicit small adapter to existing P11 A0/A1 batch keys where needed;
read its actual API before choosing shapes. Do not modify P11 losses/training
semantics. A1 windows obey configured supervised coverage; short valid episodes
remain archived/usable for geometry, and excluded dynamics windows are counted.
Masks and unit/frame conventions remain unchanged. No automatic preprocessing
job runs while inspecting an archive; physical preprocessing, if needed for batch
conversion, is explicit and uses existing functions and resolved config.

### Bounded collection entry point

Provide a Python `run_collection` entry point under `icgs.data.collection` which
composes existing `collect_attempt` and persistence over explicitly supplied
finite attempt specifications and injected environment/command/monitor factories.
Preflight identities, paths and positive finite wall/disk/attempt/interval caps
before constructing any environment. Validate the supplied protocol manifests,
without interpreting approval as measured PASS. No arbitrary dynamic Python imports
from JSON. Stop cleanly at limits, preserve failures and close every constructed
environment. Report requested/actual counts, disk use, elapsed time and stop reason.
Document that nonpreemptible backend calls can overrun wall limits; do not claim
hard real-time enforcement. Refuse a new episode before exceeding the disk cap;
do not delete existing data to make room. Cap staging bytes as well as published
bytes and retain small diagnostic reports within the declared budget policy.

Ship a copyable Python API example and exact offline validation commands. Do not
present a fake-environment example or an unimplemented CLI as physical generation.
A concrete RLBench controller, scene/expert generation, calibrated sensor profiles
and measured predicate bindings are separate integration prerequisites. Report
their actual local availability; no blanket OS support claims without evidence.

## Alternatives and consequences

### Manager integration clarification — 2026-09-16

After three partial repair rounds, owner-delegated review retains the architecture
but makes lifecycle/publication semantics explicit. The next batch first adds
end-to-end failing regressions; isolated helper success is insufficient.

- Infrastructure/factory/cleanup failures stop this run after accounting and
  retained diagnostics; they are not ordinary task failures and are not retried
  implicitly. Valid unsuccessful task episodes remain training data.
- Charge observed executed work before persistence; a disk failure must not erase
  work from the run report. If an advance fails with unknown physical coverage,
  stop rather than claiming zero physical work and starting another attempt.
- Copy/validate online arrays at each collection boundary; later persistence
  cannot recover prior contents of a reused simulator buffer.
- Require all three seed fields explicitly (equal values are allowed), and a
  dirty-patch digest when a caller declares dirty source. Never invent a clean
  tree or seed aliases. Validate supplied metadata recursively before construction.
- A narrow explicit `reconcile_episode` operation may attach a checksum-valid
  orphan episode to its matching dataset manifest without re-executing physics.
  Reject conflicts; no directory scanning, auto-import, or silent overwrite.
- Enforce peak write allowance before writing bytes, not after deleting an
  over-budget staging file. Per-shard in-memory compressed serialization is
  acceptable for this bounded pilot; it must not buffer a whole dataset.
- Disk-enforcement refinement: serialize one NPZ shard or complete JSON payload
  to bytes and check its actual encoded length before its filesystem write.
  Fixed overhead estimates are not a correctness bound for unbounded strings.
  Include the full final index provenance and temporary index coexistence with
  the old index. Archive, empty-report, quarantine and index budget exhaustion
  share one explicit storage-limit outcome. Do not duplicate diagnostic files.
  If even diagnostics cannot fit, retain the unsaved evidence/error in the
  returned in-memory run report with an explicit retention reason; never claim
  quarantine was written or exceed the cap to log the failure.
- P11-facing records/adapters retain split/episode provenance and explicitly map
  source lineage to the existing consumer's `lineage_id`, leaving archive and
  P11 model/training semantics unchanged.

- Root-ID-only splitting is smaller but cannot validate inconsistent ancestry.
  An explicit finite parent table is the smallest auditable closure here.
- Reusing native NPZ/PyG would silently change its semantics; keep formats separate.
- A generic collection framework or database is unnecessary for the pilot.
- Lossless arrays cost more than quantized arrays; any later quantization needs a
  named format/protocol revision and measured fidelity evidence.

## Validation and limits

Deterministic L1 tests cover roundtrips, corruption/path safety, shard boundaries,
failure retention, lineage leakage, causal windows, existing consumer batches,
limits and cleanup. Neither fake environments nor G1/G2 owner approval establishes
physical feasibility. Required C1–C5 and physical L3/L4 gates remain separately
reported, never silently PASS. Full P02 stays active until its remaining required
views/integration/evidence exist; this initial slice is not all of P02.
