# Research engineering workflow

## Select the change

| Category | Required work | Durable plan |
| --- | --- | --- |
| A: small engineering change | Inspect affected behavior; make scoped change; select focused validation; report evidence | Not required |
| B: component experiment | Name baseline, hypothesis, changed/unchanged components, config delta and evaluation; follow [research policy](RESEARCH.md) | Only if boundaries, sessions, dependencies or recovery require one |
| C: architecture/research migration | Document current/target states, invariants, phases, compatibility, risks, recovery and proof | Required |

A bug fix is not automatically a small research change: if it changes action,
graph, preprocessing, checkpoint, dataset or benchmark semantics, treat the
scientific effect explicitly even when the patch is short.

## Before and during work

1. Inspect applicable instructions, Git state, affected source and existing proof.
   Preserve unrelated edits. An audit or diagnosis does not authorize fixing it.
2. Find the authoritative owner in the [map](README.md). Distinguish FACT
   (direct evidence), ASSUMPTION (inference), and UNKNOWN (missing evidence).
   An observed default is not permission to establish a new architectural rule.
3. Classify the task and state scope, expected compatibility and validation cost.
   Ask for direction when a material research choice is unresolved.
4. Implement the smallest approved change. Keep documentation owners current.
   Do not weaken validation or change a baseline to make a test pass.
5. Run the applicable tiers from the [validation guide](../tests/README.md).
   Stop or bound operations that can hang; do not start expensive work implicitly.
6. Report outcome, changed files, exact evidence and remaining risk. Another
   researcher must be able to resume from repository records, not chat history.

## Plans

Use one `docs/plans/active/<slug>.md` from the [template](plans/TEMPLATE.md)
for category C and work requiring durable memory. Keep progress, decisions,
open questions and recovery current after each meaningful implementation phase.
Fields not relevant to the task may be omitted with a reason.

Move the plan to `docs/plans/completed/<slug>.md` only after the outcome exists
and evidence and validation gaps are recorded. Unavailable expensive validation
may remain disclosed if it is not an acceptance requirement; an unmet required
criterion keeps the plan active. Do not create separate task databases or story
packets. Links from either lifecycle directory should remain valid after moving.

## Decisions

Use `docs/decisions/<number>-<slug>.md` and the
[template](decisions/TEMPLATE.md) for durable choices: package layout, component
boundaries, config mechanism, checkpoint/dataset compatibility, environment/core
dependency direction, coordinate frames, action representation, or changed
baseline-preservation policy. Record accepted alternatives and consequences.

Minor naming, utility selection and tiny refactors normally stay in the diff or
plan. Do not retroactively invent decisions for historical implementation details.
A proposed ADR is not accepted authority until the repository owner approves it.

## Encoding an invariant

Locate accepted authority and exact allowed/forbidden scope before adding a guard.
Reuse the smallest native check; require allowed and forbidden fixtures with an
actionable diagnostic. The [encode-invariant skill](../.agents/skills/encode-invariant/SKILL.md)
routes this specialized work. Report local execution, optional hooks, CI invocation
and external branch protection separately. None implies the others.
