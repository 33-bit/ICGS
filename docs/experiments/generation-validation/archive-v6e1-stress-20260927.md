# Lossless archive collection on Colab V6e-1

Status: running; OAuth2 retry succeeded. No full collection or training started.

## Contract and reproduction

Category B operational component experiment. Baseline: current lossless archive
collector at `464cb7e0864e855922efea3fa2c8a94d5881fbf7`, not the legacy-format
September24 capacity probe. Hypothesis: extra simulator concurrency improves
capture until host CPU contention dominates; publication can independently limit
end-to-end throughput. Only worker/slot counts change in the matched sweep.
Raw observations, eight-boundary chunks, controller, task definitions, canonical
seeds/planner, archive validation and full-cloud retention stay unchanged.
No learned model/checkpoint is involved; the TPU accelerator is not used.

Session `icgs-archive-stress-v6e1-20260927`, explicit `colab --auth=oauth2`.
Actual host:44 logical CPUs,172GiB RAM, approximately205GiB initially free disk;
advertised4.08 compute units/hour. Work root `/content/ICGS`.
Locked generation environment: Python3.10, CPU dependencies, pinned
CoppeliaSim4.1/PyRep/RLBench and all36 generated task assets.

```bash
colab --auth=oauth2 new -s icgs-archive-stress-v6e1-20260927 --tpu v6e1
# On that VM, in /content/ICGS at the pinned revision:
python3 -B scripts/setup_environment.py --profile generation --python-version 3.10 \
  --venv-root /content/ICGS/.venv --provision-simulator --build-generation-tasks \
  --runtime-config /content/ICGS/outputs/stress-runtime.json \
  --receipt /content/ICGS/outputs/stress-setup-receipt.json
.venv/bin/python -B outputs/colab_stress.py --stage w08s08 --workers 8 --slots 8
.venv/bin/python -B outputs/colab_stress.py --stage w16s16 --workers 16 --slots 16
.venv/bin/python -B outputs/colab_stress.py --stage w32s16 --workers 32 --slots 16
.venv/bin/python -B outputs/colab_stress.py --stage w32s32 --workers 32 --slots 32
```

The finite operator helpers call canonical worker commands, shared reservations
and `validate_closed_result`; they never refill the production quota queue.
Each sweep stage repeats the same first nominal seed for each of36 programs.
These are matched repeats, not144 independent scenes. Confirmation uses a second
nominal scene and first perturbed attempt for each program (72 attempts).
Selection is the smallest worker/slot setting within5% of fastest passing capture
throughput. One36-attempt batch per setting limits statistical confidence;
the selected setting is best tested, not a global optimum.

At most240 jobs and three hours live testing after setup. Limits:1GiB/result,
120GiB shared staging,4GiB staging reserve; stop on RAM<12GiB, free disk<20GiB,
invalid archives, safety stop or timeout. Capture and serial archive validation
are timed separately. The eight-result publication probe additionally measures
canonical ingest, real HF upload, revision-bound byte verification and pruning.
Its production retention is `receipt_only` under a separate `validation/` prefix;
credentials are coordinator-only. No tokens are included in evidence.

## Configuration finding

The default publication batch size is1, upload threads1 and runtime interval300s.
This interval alone limits publication to at most12 results/hour while quotas
remain incomplete (and validation/upload can make it slower). The finite probe
sets the already exposed `run.publish_interval_s=1`, leaves batching/thread
defaults unchanged and calls `publish_due(force=False)` with the current
coordinator manifest. No runtime implementation change is required for this
configuration adjustment. The probe does not certify long-running quota refill,
large-manifest scaling or concurrent capture/publisher contention.

## Quota-aware estimates

Canonical quotas require5,600 nominal successes plus1,920 valid perturbed
attempts, an ideal minimum7,520 results. Nominal valid failures are retained but
do not meet nominal success quotas. Per-program estimate is
`sum(nominal_target_i / p_success_i) + 1920`, assuming valid perturbed attempts.
A program with no observed nominal success has no finite plug-in estimate;
pooled success rates must not conceal that missing evidence. Actual retry caps
can terminate collection incomplete. This small sample cannot establish that a
zero-observed-success program is impossible, nor certify quota completion.

Colab imposes variable resource/lifetime limits; paid compute does not guarantee
an uninterrupted whole run. Multi-session recovery is required if the workload
exceeds a session lifetime. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Results and evidence

| Workers / simulator slots | Capture seconds | Capture attempts/hour | Serial validation seconds | Valid archives | Outcomes |
| --- | ---: | ---: | ---: | ---: | --- |
| 8 / 8 | 278.76 | 464.9 | 320.28 | 36 / 36 PASS | 29 success, 7 valid failure |
| 16 / 16 | 199.88 | 648.4 | 321.06 | 36 / 36 PASS | 29 success, 7 valid failure |
| 32 / 16 | 214.81 | 603.3 | 323.14 | 36 / 36 PASS | 29 success, 7 valid failure |
| 32 / 32 | 126.31 | 1026.0 | 322.15 | 36 / 36 PASS | 29 success, 7 valid failure |

The sweep selects32 workers/32 slots. No invalid observation, simulator crash,
quarantine or resource stop occurred in these144 attempts. The32-slot run had
158.41GiB minimum available RAM and3.17GiB sampled peak staging. More workers
without more simulator slots did not help. Serial archive validation alone is
approximately400 results/hour for this workload, already below capture rate.
The seven nominal failures on the matched seed were R1,T01,T06,T08,T14,T17,V02
at every setting; these are valid negative artifacts, not nominal quota successes.

The confirmation at32/32 passed72/72 valid archives in248.43s capture plus
760.00s serial validation:57 successes and15 valid failures. It used a second
nominal seed and the first perturbed attempt per program; this is confirmation,
not exhaustive perturbation coverage. Per-program seed evidence shows zero
nominal successes in the two observed seeds for T01, T06, T08 and V02. Therefore
a finite quota ETA for those programs is not proven by this experiment.

The real eight-result publisher probe passed all checks. It ingested, uploaded,
verified remote bytes at the pinned revision and pruned local artifacts in573.47s:
76.39s ingest and496.88s publish/verify/prune,50.24 results/hour. The remote
dataset head was `8a07eca0c732b00bb0cae8467b2bb1ea7a46e794`;8/8 were published,
zero remained ready/claimed/quarantined. This is the current single-result
publisher path with `publish_interval_s=1`; it does not prove large-batch or
parallel-publisher scaling.

Quota-aware estimate, conditional on a common nominal-success probability and
the measured archive volume (not a per-program guarantee):

| Assumed nominal success probability | Attempts | Archive volume | Publisher-only wall time at50.24/hour |
| ---: | ---: | ---: | ---: |
| 1.0 (ideal minimum) | 7,520 | ~524 GB | ~150 h / 6.3 days |
| 0.8 | 8,920 | ~622 GB | ~178 h / 7.4 days |
| 0.6 | 11,253 | ~787 GB | ~224 h / 9.3 days |

Capture at the selected setting is ~1,026 attempts/hour and serial validation
is ~400 results/hour, so publication is the observed bottleneck. These estimates
assume the publisher runs continuously, HF service remains available, and each
nominal failure can eventually be retried into a success. Because T01/T06/T08/V02
had no observed nominal success in two seeds, the honest whole-run answer is
“not finitely certified”; use roughly6–9+ days only as a planning range after a
per-program pilot. Colab lifetime/resource interruptions and HF throttling can
extend it; multi-session resume is required. Do not start the full quota from
this benchmark alone.

Evidence package:16,583,577,600 bytes, SHA-256
`ebab288ade3b3da1246cac539827051c750e007c99a1b218cde306195795e72b`, streamed
to `vps-a:/home/huy2325/ICGS/outputs/colab-v6e1-archive-stress-20260927/`.
