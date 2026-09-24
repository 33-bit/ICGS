# TPU v6e-1 capacity probe — 2026-09-24

Status: **completed bounded probe; no production data generated**.

Sessions: `icgs-capacity-v6e1-20260924-r3` and a follow-up
`icgs-capacity-v6e1-w128-20260924`, both V6E1 Standard. The first endpoint was
recovered once after a local alias loss, then released normally. The probe used the
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
| 128 workers | 24 | **FAIL quality**: 63 success, 16 valid_failure, 1 malformed artifact | 509.4 s | **565.3/h** | 87.9 GiB | 73.39 GB |

The w08–w64 stages had zero `simulator_crash`, zero `invalid_observation`, zero
invalid result artifacts, and worker return code 0. The corrected w128 measurement
reached all 80 results, but one result (`T19`) lacked `artifact_manifest.json` and
was therefore rejected by closed-result validation. Most workers returned 0; one
worker was terminated after the result cap was satisfied. The earlier w128 run was
an outer-timeout artifact and is superseded by this result.

## Recommendation and ETA

Use **64 workers with 16 simulator slots** as the initial production configuration,
with a coordinator, watchdog and bounded publication. Although w128 measured
565.3 attempts/hour, it failed artifact-quality acceptance and consumed more
memory, so it is not the recommended production setting. The measured ideal lower
bound for the 7,520 valid attempts at the accepted w64 rate is:

`7520 / 460.2 = 16.3 hours`

This excludes quota retries, nominal successes that become valid failures,
coordinator/HF publication, startup, and recovery pauses. A realistic planning
window is **20–30 hours**, with a hard operational checkpoint after the first
published batches. Artifact volume at the observed rate is roughly 0.92 GB per
attempt, so storage and HF upload bandwidth must be monitored; this is the main
uncertainty in the ETA.

The 64-worker result is a capacity recommendation, not a production-generation
authorization or a claim that full quota has been completed.
