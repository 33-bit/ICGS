# Full Training Dataset Design

## Status and scope

Approved by the repository owner on 2026-09-19. This is a Category C data-contract
extension. It does not change Instant Policy preprocessing, action semantics,
checkpoint loading, or train/dev/test ownership.

## Compatibility

The existing `manifest.json` and lossless `shard_*.npz` files remain canonical.
A versioned training layout is added beside them. Existing readers continue to
work; new readers may require `layout_version == 2`.

## Dataset root

`metadata/` records the approved programs, asset families, splits, controllers,
dataset identity, modality policy, and README. Program definitions are copied into
`programs/{train,dev,test}`. Derived views and caches are optional and must identify
their source archive, preprocessing, model, and checkpoint hashes.

## Episode envelope

Every successful episode and every failed attempt with valid observations uses the
same logical envelope: episode identity/provenance, observations, robot state,
actions, object state, task events and labels, and terminal result. Missing sensor
modalities are omitted and declared unavailable; no synthetic values substitute
for missing measurements. Failure envelopes are indexed separately and never
contribute to successful target counts.

## Simulator-only supervision

Segmentation, object states, collisions, contacts, and program labels are
supervision/debug data and are not automatically policy inputs. `rho` is historical
occurrence, `nu` is the current postcondition, and `epsilon` is current prerequisite
eligibility. Each has an explicit valid mask.

## Snapshots

RLBench/PyRep replay packages contain robot/object/articulation state, task-monitor
state, RNG identity, and action history when available. They are labeled
`approximate_replay` unless repeated restore/replay satisfies declared fidelity
tolerances. Unsupported solver/contact state is never described as exact.

## Integrity and validation

All indexed files carry SHA256 and size metadata. Validators enforce boundary
alignment, unique IDs, split consistency, success/failure separation, valid-mask
shape agreement, source hashes for caches, and explicit snapshot quality.
