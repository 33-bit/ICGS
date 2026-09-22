# Approved composition manifest

`approved_composition_manifest.json` is the owner-approved generation catalog for
the 20 training programs, 4 development programs, and 12 locked test
compositions. All 36 rows are explicitly generation-authorized.

Validate it without launching a simulator:

```bash
PYTHONPATH=src python3 -c \
  'from icgs.data.collection.approved_manifest import load_approved_manifest; \
   load_approved_manifest("artifacts/composition/approved_composition_manifest.json"); print("PASS")'
```

The manifest validator is metadata-only and does not imply RLBench execution.
Before publishing an episode, pass `approved_manifest_path` to the collection
runner so program, split, lineage, asset family, seed ID, and execution mode
are checked before environment construction.

Strict simulator probe (no episode generation):

```bash
colab exec -s SESSION -f scripts/generation_simulator_probe.py
```

This probe validates the renderer and simulator clock only. It does not certify
task success or authorize a large generation run.

Validation status for this change:

- Approved-manifest tests: PASS (`9` tests).
- Episode-provenance tests: PASS (`4` tests).
- Collection/binding regression tests: PASS (`45` tests).
- Full simulator generation: NOT RUN.
