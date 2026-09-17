# 0012: Isolated RLBench G1 feasibility track

Date: 2026-09-16. Status: accepted by owner delegation for the bounded
exploratory track only.
Decision owner and approval evidence: the repository owner authorized a Colab
data-generation job, required Drive-first and continuous Hugging Face
publication, and delegated the concrete specification choice to the coordinator.

## Context

P02's archive, lineage, initial geometry and dynamics software slice is accepted,
but it does not supply an actual CoppeliaSim controller, assets, calibration, or
expert executions. The approved ICGS programs T01--T20 and their pilot subset
T06/T08/T09/T11/T13/T14 still lack their custom scene/expert bindings. Mapping an
upstream RLBench task to any of those programs would misstate the data lineage.

The first E01 implementation attempted to bridge an upstream RLBench task but was
audited before collection. It used a synthetic clock and command fallback, removed
canonical cloud content before archival, and published only after a full run. That
path is not permitted to generate data.

## Decision

`E01` is an **exploratory G1 feasibility track**, not an ICGS benchmark program
and not P11 training data. It may establish a physical controller/time/sensor and
durability trace only under all of the following rules:

- The dataset manifest declares a distinct `dataset_track` value and can contain
  only programs belonging to that track. E01 cannot share a manifest or a
  `geom`/`dyn` consumer view with the primary ICGS development, training, or test
  tracks.
- The track is stored under an isolated Google Drive run directory and the public
  dataset prefix `exploratory/`; it cannot update a primary dataset index.
- Every accepted interval uses a live CoppeliaSim clock and a queried simulator
  timestep. Missing APIs, no-op actuation, synthetic timestamps, and inferred
  duration are fail-closed errors.
- Canonical measured clouds retain the finite sensor cloud without crop,
  voxelization, or point-count truncation. A separately identified online
  projection may crop/downsample/mask its own arrays. The archive must retain the
  two representations independently.
- Commands come from an actual live RLBench expert demonstration and the exact
  materialization protocol. No coordinate fallback, random nominal trajectory,
  fixed-length padding, or fabricated pause is allowed. Any hold/pause recorded
  as a transition must be physically executed.
- For each closed archive: validate and checksum locally; copy and verify the
  complete episode to Drive; make and read back a Hugging Face commit for the
  complete episode; then publish the isolated manifest/index last. A transfer or
  verification failure stops further collection. Recovery may resume incomplete
  publication idempotently but never overwrites an immutable episode.
- Run provenance records actual revision, dirty-patch digest, requested seeds,
  exact upstream revisions, controller/sensor protocol, and Drive/HF receipts.
  No placeholder revision, clean-state assertion, or all-zero digest is allowed.

Before E01 archives exist, execute a bounded live pinned-API G1 probe that writes
only a clearly labelled feasibility receipt. It is not an episode, G2 task proof,
or training sample.

## Alternatives considered

- Label `PickAndLift` as T06--T14: rejected because the custom ICGS asset,
  predicate, calibration, and expert contracts do not exist.
- Treat E01 as a primary `train` dataset: rejected because it would make an
  upstream task a training-source claim before a separately approved data split
  and training protocol exist.
- Keep the present dev E01 in a shared manifest and rely on an HF prefix:
  rejected because `build_view` consumes local manifest content, not remote paths.
- Store only an online 2048-point cloud: rejected because it destroys the raw
  physical measurement required for alternate preprocessing.

## Consequences

This adds a small, explicit dataset-track guard and a G1-only operational plan.
It does not modify native Instant Policy semantics, checkpoints, main task
catalogue claims, P11 training data, G2 asset/predicate certification, or C1--C5.
The first successful result is a bounded exploratory feasibility record; it is not
evidence that the main ICGS collection or benchmark can start.

## Compatibility implications

Existing primary manifests retain their current track identity through a
backward-compatible explicit default. New exploratory manifests must declare their
track explicitly and are rejected if combined with primary programs. Archive
version and P02's numeric shard contract remain stable; the validator is corrected
to permit independently derived online arrays while retaining exact measured and
online boundary continuity within their respective streams.
