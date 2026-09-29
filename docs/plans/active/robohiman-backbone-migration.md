# Plan: RoboHiMan backbone migration (Stage-1 instrumentation and gates)

Status: active
Owner: repository owner (migration brief 2026-09-30); implementation by agent session.
Motivation and observable acceptance criteria: replace the custom 36-program
generator as the primary task/environment source with RoboHiMan, keeping
RoboHiMan semantics unchanged and adding ICGS instrumentation. Acceptance is the
go/no-go gate list below, each backed by a recorded command and numeric result.

## Current and target states

Current: custom generator is the only collection path
([classification](../../audits/2026-09-30-robohiman-migration-classification.md)).
Target ([ADR 0016](../../decisions/0016-robohiman-backbone.md)):

```text
generic ICGS (contracts, icgs.data.stage1 record/labels/views, preprocessing, method)
        ↑
icgs.environments.robohiman (session, recorder, expert mirror, monitors, snapshot, camera)
        ↑
RoboHiMan@33f71d3 + RLBench@587a6a0 + PyRep@231a1ac + CoppeliaSim 4.1 (unmodified)
```

Unchanged: IP baseline, published profile, checkpoints, custom generator code and
archives, `icgs_episode_v1/v2` schemas.

## Invariants and scope

- No upstream RoboHiMan/RLBench/PyRep file is edited. Instrumentation is
  per-instance method shadowing, removed after each episode.
- Commands are recorded when issued; achieved state is read after the physics
  step. They are never derived from each other.
- Perturbation ≠ outcome; outcome labels come from simulator execution only.
- Oracle predicates/labels are offline-only (`model_boundary` in each manifest).
- No D_value/D_pair/D_terminal before a frozen pi_ref identity.
- Smoke scale only: `robohiman_collect.py` refuses >20 episodes per call.
- Out of scope: bulk collection, pi_ref construction, Stage-2 collection,
  training, split redesign.

## Phases and progress

- [x] P0 classification audit and ADR.
- [x] P1 pinned simulator venv on `vps-a` (`outputs/robohiman/.venv`, Python 3.11).
- [x] P2 source-neutral `icgs.data.stage1` (schema, store, labels, views).
- [x] P3 adapter modules (`session`, `recorder`, `expert`, `monitors`, `camera`,
      `snapshot`, `kinematics`, `episode`) and scripts.
- [ ] P4 gates on `vps-a`: parity/neutrality, A0, A1, B, failure retention,
      snapshot/replay, dependency audit, split overlap. Evidence goes to
      `docs/experiments/robohiman-validation/`.
- [ ] P5 report, README/generation docs switch notes (default switch only after gates).

ID decisions recorded here: alpha = first non-occurred eligible event, null
otherwise; drawer open/close thresholds are taken from the upstream tasks that
register them (0.15 open, 0.03 close); "drawer closed" events only count while
eligible because the relation holds at reset.

## Validation and experiment strategy

L0 locally; L1 unit tests for `icgs.data.stage1` and the camera/labels code;
L3 simulator gates on `vps-a` via `scripts/robohiman_gates.py` and
`scripts/robohiman_collect.py`; offline Stage-1 report via
`scripts/robohiman_stage1_report.py`. No L4 benchmark.

## Go / no-go gates (default-path switch)

1. Stage-1 episodes produce valid D_geom, D_dyn, D_task.
2. Task-monitor labels manually sanity-checked on representative trajectories.
3. Failed executions retained without changing benchmark semantics.
4. Snapshot/replay quantitatively evaluated.
5. Dependency audit establishes whether RoboHiMan has consequential decisions.
6. Custom pipeline remains recoverable until 1–5 pass.

## Compatibility

No checkpoint/config/evaluation change. New dataset identity
`icgs_stage1_episode_v1`; legacy archives untouched.

## Risks, open questions and recovery

- CoppeliaSim OMPL planning (RRTConnect) may be nondeterministic; parity and
  replay gates measure this instead of assuming it.
- Kinematic grasp attachment and `path.visualize()` arm excursions are upstream
  expert behaviour; recorded, not hidden.
- Recovery: delete `src/icgs/environments/robohiman`, `src/icgs/data/stage1`
  and `scripts/robohiman_*`; nothing else depends on them.

## Final evidence

Pending P4/P5.
