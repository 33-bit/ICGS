# G2 procedural pilot task bundle

This bundle was generated in Colab session `icgs-g2-pilot-2` with pinned
CoppeliaSim 4.1, PyRep `8f420be8064b1970aae18a9cfbc978dfb15747ef` and RLBench
`02720bba4c73fe02eb75df946b8791b806028a9d`.

It contains six custom procedural primitive task models and Python task files:
T06, T08, T09, T11, T13 and T14. The binding manifest validates locally against
the repository catalog and requires scene, asset/version, workspace,
randomization, seed, expert, controller, predicate and calibration identities.

Evidence:

- CoppeliaSim model import: **PASS** for T06 (`7` model objects).
- Task reset: **PASS** for T06.
- Local binding-manifest validation: **PASS**, six required pilot programs.
- Live executed expert demo: **NOT RUN/PENDING**. The first procedural waypoint
  controller failed RLBench IK/path feasibility; no episode was accepted.
- Hugging Face publication: **NOT RUN**. No training episodes were uploaded.

The `.ttm` files are generated simulator assets, not proof of G2 physical
feasibility. Do not start primary collection from this bundle until the direct
expert controller is repaired and a measured G2 receipt records accepted seeds,
success/failure outcomes and predicate observability.
