# TPU v6e-1 capacity probe — 2026-09-24

Status: **completed bounded probe; no production data generated**.

Session: `icgs-capacity-v6e1-20260924-r3`, V6E1 Standard, endpoint was recovered
once after a local alias loss, then released normally. The probe used the
canonical worker and real RLBench/CoppeliaSim materialization, isolated run roots,
no HF publication and no worker credentials. Global probe limits were 400 jobs
and 5,400 seconds. Only 120 jobs were attempted successfully before the w128
timeout; no w200 stage was started.

| Stage | Sim slots | Result | Wall time | Throughput | Minimum available RAM | Artifact bytes |
|---:|---:|---|---:|---:|---:|---:|
| 8 workers | 2 | 8 success | 376.5 s | 76.5/h | 155.4 GiB | 8.46 GB |
| 16 workers | 4 | 14 success, 2 valid_failure | 375.4 s | 153.4/h | 154.9 GiB | 14.99 GB |
| 32 workers | 8 | 26 success, 6 valid_failure | 409.5 s | 281.3/h | 142.2 GiB | 27.99 GB |
| 64 workers | 16 | 49 success, 15 valid_failure | 500.7 s | **460.2/h** | 102.6 GiB | 58.99 GB |
| 128 workers | 24 | **TIMEOUT** | >1100 s | not accepted | partial claims cleaned | not retained |

All completed stages had zero `simulator_crash`, zero `invalid_observation`, zero
invalid result artifacts, and worker return code 0. The w128 stage hit the probe
deadline and was terminated by owned process-group cleanup; it is not evidence of
usable throughput.

## Recommendation and ETA

Use 64 workers with 16 simulator slots as the initial production configuration,
with a coordinator, watchdog and bounded publication. The measured ideal lower
bound for the 7,520 valid attempts is:

`7520 / 460.2 = 16.3 hours`

This excludes quota retries, nominal successes that become valid failures,
coordinator/HF publication, startup, and recovery pauses. A realistic planning
window is **20–30 hours**, with a hard operational checkpoint after the first
published batches. Artifact volume at the observed rate is roughly 0.92 GB per
attempt, so storage and HF upload bandwidth must be monitored; this is the main
uncertainty in the ETA.

The 64-worker result is a capacity recommendation, not a production-generation
authorization or a claim that full quota has been completed.
