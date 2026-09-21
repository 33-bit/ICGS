# Approved composition manifest

`approved_composition_manifest.json` is the owner-approved primary catalog for
the 20 training programs, 4 development programs, and 12 locked test
compositions. `E01` is intentionally excluded because it belongs to the
exploratory track.

Validate it without launching a simulator:

```bash
PYTHONPATH=src python3 -c \
  'from icgs.data.collection.approved_manifest import load_approved_manifest; \
   load_approved_manifest("artifacts/composition/approved_composition_manifest.json"); print("PASS")'
```

The historical G2 receipts cover `T06`, `T08`, `T09`, `T11`, `T13`, and `T14`,
but they used a looser pilot predicate/controller identity. These six entries
are now `generation_authorized`: they may be rerun by the manifest-aware
generator, but this status does not claim any successful episode. The remaining
primary entries are `planned`.

The manifest validator is metadata-only and does not imply RLBench execution.
Before publishing an episode, pass `approved_manifest_path` to the collection
runner so program, split, lineage, asset family, seed ID, and execution mode
are checked before environment construction.

Strict simulator smoke (local receipt only, no publication):

```bash
.../python scripts/colab_g2_dataset_generator.py \
  --smoke-only --tasks T06 --episodes-per-task 1 --start-idx 900
```

The smoke receipt must include the approved-manifest digest and a successful
task predicate before a program can be changed from `planned` to
`pilot_certified`.

Validation status for this change:

- Approved-manifest tests: PASS (`9` tests).
- Episode-provenance tests: PASS (`4` tests).
- Collection/binding regression tests: PASS (`45` tests).
- Simulator generation for planned programs: NOT RUN.
