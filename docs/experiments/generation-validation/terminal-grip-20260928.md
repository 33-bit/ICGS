# Terminal gripper preservation

Category B controller correction; authorized in the generation-readiness task.
This is an explicit collection-behavior change, not an archive-format change.
Existing archives remain immutable; new runs record their code revision.

## Diagnosis

Read-only inspection of the prior V6e-1 archives on `vps-a` showed both nominal
T01 seeds reaching their lift target within 1.164/1.052 mm with closed grip.
The worker then issued 15 unconditional open-grip settling commands before
evaluating success. The object dropped to the table, ending 107.5/123.3 mm away.
The failure therefore was not evidence that more seeds or workers were needed.

## Change

All programs retain their last measured gripper state through the existing
15-frame settling period. There is no T01 branch. Routine commands, scene seeds,
task predicates/tolerances, split/quota rules, collection timings, preprocessing,
and lossless archive format are unchanged. Explicit releases still occur in
task routines; settling no longer adds an unplanned release.

This does not correct existing object-position snapping or other placement
controller behavior. T06/T08/V02 still need their own quota-readiness evidence;
success in two T01 seeds does not prove full-catalog completion.

## Validation

Environment: `vps-a`, `/home/huy2325/ICGS`, `.venv/bin/python` 3.10.21.
Base revision `6ada0d853954b9865d554f462991dcd9749ab2a9` plus recorded dirty
patch SHA256 `b359d6885dc7a27c5db287afd3f3b392021ca6d51c3732ad22156d27a8a29eb4`.

- PASS: `PYTHONPATH=src .venv/bin/python -B -m pytest -q
  tests/test_generation_worker.py tests/test_generation_expert.py` — 70 tests.
  Includes measured-grip selection and actual terminal-loop command regression.
- PASS: `PYTHONPATH=src:. .venv/bin/python -B outputs/t01_replay.py` — two
  canonical nominal seeds, 416071343 and 689424400, both success; both archives
  passed `validate_closed_result`. No HF publication or quota refill.
- Proof and exact helper/patch: `outputs/t01-terminal-grip-replay/` on `vps-a`.
- NOT RUN in this experiment: training, published-model inference, complete
  36-program quota collection. No claim of full-generation readiness.
