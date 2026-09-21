# 0013: G2 task/split certification authorization

Date: 2026-09-18. Status: accepted for certification authorization; physical
gate **PENDING MEASURED EVIDENCE**.
Decision owner and approval evidence: repository owner approval recorded in the
task conversation on 2026-09-18.

## Context

G2 is the task/splits feasibility gate in the [implementation roadmap](../plans/active/icgs-method-implementation.md).
It protects the primary collection from ambiguous tasks, asset leakage and
unobservable predicates. The repository already defines the 20/4/12 program
skeletons and split rules, but the concrete asset/version/parameter/seed catalog,
expert executions, predicate observability and tolerance sensitivity have not
been measured. Existing simulator readiness evidence is explicitly not G2
evidence.

## Decision

Authorize preparation and bounded certification work for G2, including:

- a versioned concrete asset, parameter, seed, program and source-lineage
  manifest;
- family/lineage split checks before any generated data are admitted;
- expert feasibility executions for accepted instances;
- predicate observability and tolerance-sensitivity measurements; and
- an experiment record containing revisions, seeds, environment, failures and
  the resulting acceptance/quarantine decisions.

This authorization does **not** declare G2 passed. G2 remains
**AUTHORIZED FOR CERTIFICATION — PENDING MEASURED EVIDENCE** until the required
artifacts exist and the roadmap stop conditions are cleared. It does not
authorize primary-scale collection, training, benchmark execution or relabeling
unrelated RLBench tasks as ICGS programs. Any simulator workload remains a
separately bounded execution scope under the repository workflow.

The G2 acceptance record must show the concrete catalog, split/lineage closure,
expert success for each accepted instance, predicate observability and tolerance
sensitivity. Unsupported force signals, infeasible/ambiguous instances or split
overlap remain quarantine conditions; they are not silently resolved by changing
thresholds or substituting an upstream task.

## Alternatives considered

- Marking G2 **PASS** from owner approval alone was rejected: the roadmap
  requires measured physical evidence and ADR0011 states that owner approval is
  not physical-feasibility evidence.
- Treating the existing exploratory E01 RLBench track as G2 evidence was
  rejected by ADR0012; it is isolated G1 feasibility only.
- Waiting for all primary data before authorizing certification was rejected:
  it would prevent the bounded evidence needed to decide whether collection is
  feasible while still keeping the gate open.

## Consequences

P02 may prepare the concrete G2 manifests and certification record, but remains
active until measured evidence is recorded. No quota or target count becomes
permission to generate data. The owner must record a follow-up accepted result
or a revised/quarantined manifest after certification.

## Compatibility implications

No native IP, checkpoint, action, preprocessing, dataset schema or existing
exploratory-track identity changes. This decision changes authorization state
only; it does not create training data or alter benchmark claims.
