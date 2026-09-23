# VPS generation acceptance closure - 2026-09-24

Status: **bounded runtime acceptance PASS; production generation NOT RUN**.

This record combines fresh evidence from `vps-a` at commit `f3ef53c` with the
previous isolated bounded receipts. It does not authorize or claim a production
quota run, benchmark, or model training job.

## Fresh VPS evidence

- Checkout: `/home/huy2325/ICGS`, commit `f3ef53c2faa9adaeaa5409116d562581610ac1c6`.
- Environment: Python 3.10.21, `torch==2.2.0+cpu`, PyG 2.5.0, pinned CoppeliaSim,
  PyRep `8f420be8064b1970aae18a9cfbc978dfb15747ef`, RLBench
  `02720bba4c73fe02eb75df946b8791b806028a9d`.
- Environment verifier: PASS, including credential permission, setup receipt,
  renderer, simulator and PyG ABI checks.
- Focused generation/environment tests: `243 passed`.
- Full repository tests with renderer variables exported:
  `915 passed, 9 skipped, 3 warnings`.
- L0: `scripts/validate_fast.py`, 22 harness tests PASS and local links PASS.
- Simulator readiness: PASS; 2 physics steps, advancing clock, finite `128x128x3`
  point cloud. This is readiness only and contains no training episode.

## Gate disposition

| Gate | Status | Evidence |
| --- | --- | --- |
| Portable environment | PASS | Current verifier and setup receipt |
| Success episode | PASS | VPS bounded smoke, G1-G4 receipts |
| Valid failure | PASS | VPS bounded smoke, T01/T11 receipts |
| Malformed-result quarantine | PASS | Corrected bounded r3 receipt plus current control tests |
| Infrastructure/crash materialization | PASS | Corrected bounded r3 receipt |
| Coordinator restart/recovery | PASS | Corrected bounded r3 receipt |
| Publication and remote hash verification | PASS | VPS isolated prefix `vps-6341777`, four verified batches |
| HF resume from pinned manifest | PASS | Disjoint run, pinned revision and SHA receipt |
| Workers-only multi-host attach | PASS | Host-scoped receipt and lease heartbeat |
| Invalid-observation live simulator path | NOT RUN | Only contract/unit evidence exists |
| Full production quota | NOT RUN | Explicitly outside bounded acceptance |
| L4 benchmark | NOT RUN | No benchmark workload authorized |

The combined receipt is machine-readable in `acceptance_receipt.json`. A gate is
not marked PASS merely because a unit test exists; live evidence and its source
run are named separately.
