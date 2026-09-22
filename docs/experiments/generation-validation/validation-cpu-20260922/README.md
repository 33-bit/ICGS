# CPU generation validation — 2026-09-22

Status: **FAIL — bounded acceptance stopped safely**

This was the first remote acceptance attempt for the portable generation
runtime. It used the canonical environment, launch, worker, coordinator and
watchdog owners on a disposable CPU Colab session with two workers. No
production dataset prefix was modified and no HF publication was performed.

## Environment and commands

- Session: `generation-validation-cpu` (CPU, endpoint `m-s-kkb-usw3b1-2y4u3f90bozrv`)
- Runtime config: `/content/generation-runtime.json`
- Runtime config SHA256: `efc92c7f29824b6dc5c0f81146ccf8c89d2b146b2b335bac1ee5b438c7b40329`
- Workers: 2; configured simulator slots: 2; display base: 20; screen: 1024×768
- Provisioning: PASS — pinned CoppeliaSim, PyRep and RLBench setup completed
- Contract preflight: PASS — 92 focused remote tests after setting the repository cwd
- Launch used canonical `scripts/generation_launch.py` and the JSON validation plan.

## Gate results

| Gate | Status | Evidence / reason |
| --- | --- | --- |
| success | FAIL | No success episode before safe stop |
| valid_failure | NOT RUN | Not reached |
| invalid_observation | NOT RUN | Not reached |
| infrastructure_failure | PASS | 27 `simulator_crash` attempts were materialized |
| malformed_result | NOT RUN | Not reached |
| coordinator_restart | FAIL | Watchdog restarted coordinator on `pid_invalid`; controlled restart gate was not completed |
| publication | NOT RUN | Zero published jobs; no HF commit or remote hash verification |
| resource bound | FAIL | Coordinator refilled to 400 jobs despite validation plan `max_jobs=7` |

The queue reached 380 pending jobs after the safe stop. Workers and the
coordinator/watchdog were terminated before unbounded generation could
continue. The validation receipt and launch/heartbeat summaries are stored
alongside this record; they contain no credentials or episode binaries.

## Follow-up required

The coordinator must derive its refill target from the active validation plan
and never enqueue more than `max_jobs`. A new CPU acceptance run is required
after that fix; this record remains FAIL and the active plan must not be moved
to completed until all required gates pass.

The session was explicitly stopped and `colab sessions` returned no active
assignments.
