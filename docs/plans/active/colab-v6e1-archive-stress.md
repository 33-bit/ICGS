# Colab V6E1 archive capacity experiment

User scope: create V6E1 through Colab CLI (OAuth2), stress-test collection and
recommend settings/whole-run ETA. This explicitly selects Colab rather than
the earlier VPS-only validation scope. No full quota run or training.
Category B operational component experiment; no runtime/data semantic changes.

## Design

Pin code `464cb7e0864e855922efea3fa2c8a94d5881fbf7`, pinned Python3.10
generation profile and simulator, all 36 generated tasks. Use the canonical
planner, worker commands, shared reservations and closed-result validator.
Compare the same 36 nominal program/seed plans at (workers,slots)=(8,8),
(16,16),(32,16),(32,32). Only concurrency changes; the archive retains raw
observations, default chunking, and valid failures. A finite confirmation at
the best passing setting covers additional seeds and perturbed attempts.
At most 240 jobs and three hours of live testing after setup. Stop a stage
on invalid archives, safety marker, RAM below 12 GiB, or disk headroom below
20 GiB. Retain failed evidence. Do not chase concurrency beyond a measured
plateau merely to maximize process count.

Each stage: writer cap1GiB, shared total120GiB/reserve4GiB, at most36 queued
jobs; monitor physical staging and memory, time capture separately from serial
validation. Prior stages remain counted in disk headroom. End-to-end publication
uses a separate finite prefix and production receipt-only mode, with credentials
only in the coordinator. Estimate collection and publication service rates
separately and expose the bottleneck. Quotas require 5,600 nominal successes
plus1,920 valid perturbed attempts, not just7,520 arbitrary valid results;
nominal valid failures require retries. Report per-program limitations, not
a guaranteed finite ETA if a program has no observed success.

## Checklist

- [x] Allocate via OAuth2; verify resources and cost. 44 CPUs,172GiB RAM,
      ~205GiB free disk;4.08 units/hour. Release delayed duplicate from first
      timed-out allocation; only one intended session remains.
- [x] Provision/verify locked environment; first bounded batch captured all36.
- [x] Execute comparable concurrency stages and quality validation: all four
      PASS,36/36 valid archives each; identical29 successes/7 valid failures.
      Capture rates465,648,603,1026 attempts/hour; selected32 workers/32 slots.
- [x] Confirm selected setting on additional nominal/perturbed seeds:72/72
      valid,57 success/15 valid failure, no invalid archives.
- [x] Measure real HF ingest/upload/verification/pruning overhead separately:
      8/8 published and verified in573.47s; service rate50.24 results/hour.
- [ ] Preserve evidence and outputs outside ephemeral VM, stop session,
      record PASS/FAIL/NOT RUN, best tested setting and conditional ETA.

Current evidence package is `/content/ICGS/outputs/colab-stress-evidence.tar`,
16,583,577,600 bytes, SHA-256
`ebab288ade3b3da1246cac539827051c750e007c99a1b218cde306195795e72b`;
streaming to `vps-a:/home/huy2325/ICGS/outputs/colab-v6e1-archive-stress-20260927/`.

Session: `icgs-archive-stress-v6e1-20260927`.
Root: `/content/ICGS`; local helpers/configs in ignored `outputs/`.
Setup artifacts and logs remain under the clone. No credentials in Git.

Publication configuration finding: the shipped runtime interval is300s with
single-result batches. The finite publication probe changes only the exposed
`publish_interval_s` to1 and uses `force=False` with the coordinator's current
manifest. This measures the deployable single-result path without the avoidable
five-minute wait; batching/upload thread defaults remain unchanged.
