# 0014: One canonical primary generation runtime

Date: 2026-09-22. Status: proposed for implementation.

## Context

The repository accumulated several generation layers: historical v1/v2
collectors, a v3 primary collector, Colab launchers, and ad-hoc recovery and
publication scripts. The current distributed worker still imports execution
helpers from `scripts/colab_g2_dataset_generator.py`, so deleting the legacy
file without first moving those helpers would break the only working primary
generation path.

The user has authorized one canonical generation system and requested semantic
names without `v1`, `v2`, `v3`, or `g2` in file, module, or entry-point names.

## Decision

The canonical generation namespace is `icgs.data.collection.primary` and the
canonical operational scripts use semantic names under `scripts/primary_*`.
The distributed worker, coordinator, watchdog, planner, validation, queue and
publisher import only from this namespace or from shared runtime modules.

Legacy generation branches and operational tooling are removed after their
live dependencies are migrated:

- old v1/v2 CLI branches and primary-v2 publication/sidecar scripts;
- the compatibility collector `colab_g2_dataset_generator.py` after its
  execution/materialization helpers are moved into the primary package;
- exploratory generation launchers that are not part of the primary dataset;
- one-off Colab debug, drain, recovery, alias and bulk-upload scripts.

The following remain intentionally:

- historical baseline, checkpoint and architecture decisions as provenance;
- shared model/runtime code outside data generation;
- the generated data contract's numeric `schema_version` field and payload
  compatibility checks, because those identify serialized artifacts rather than
  source filenames;
- tests that exercise the canonical runtime, rewritten to import canonical
  modules.

## Naming map

| Current | Canonical |
| --- | --- |
| `src/icgs/data/collection/v3/` | `src/icgs/data/collection/primary/` |
| `V3_PROTOCOL`, `V3Protocol` | `PRIMARY_PROTOCOL`, `PrimaryProtocol` |
| `colab_v3_distributed_worker.py` | `primary_worker.py` |
| `colab_v3_distributed_coordinator.py` | `primary_coordinator.py` |
| `colab_v3_distributed_watchdog.py` | `primary_watchdog.py` |
| `colab_v3_distributed_launch.py` | `primary_launch.py` |
| `colab_v3_pilot_episodes_worker.py` | `primary_episode_worker.py` |
| `approved_composition_manifest_v3.json` | `approved_composition_manifest.json` |
| `episode_record.py` functions ending `_v2` | semantic names without the suffix |
| `episodes_v2.py` | merged into the canonical episode schema owner |

Serialized payload compatibility is preserved during this migration; changing
the schema value itself requires a separate data-contract decision and proof.

## Migration order

1. Add canonical package and semantic aliases with tests.
2. Move worker execution/materialization helpers out of the legacy collector.
3. Update all runtime scripts, tests, package data and documentation.
4. Remove legacy branches and untracked operational debris only after import and
   reference scans are clean.
5. Run L0 and focused generation-contract tests; report expensive simulator and
   full-data validation as not run unless explicitly authorized.

## Recovery and compatibility

This is a source-layout migration only. Existing HF artifacts and local research
data are not deleted. A future run must use a new run identity and the existing
HF manifest as its resume source. Historical documents remain readable but are
marked as historical evidence rather than operational instructions.

## Consequences

The generation entry surface becomes smaller and discoverable. Existing scripts
that import legacy module names will stop working by design; tests and checked-in
documentation are migrated in the same change. Serialized episode records remain
readable because their payload contract is not silently changed.
