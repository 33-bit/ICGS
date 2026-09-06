# Research policy

## Baseline preservation

The [original baseline](baselines/instant_policy.md) identifies original Instant
Policy. Preserve its identity, configuration, preprocessing, graph and action
semantics, dataset/checkpoint assumptions and evaluation protocol. A bug in the
original is a documented limitation, not permission to silently redefine it.

Prefer baseline + controlled configuration/component change + comparable evaluation.
New code derives from instant_policy_original() with dataclasses.replace on frozen
sections; record exact overrides and resolved component identities. The legacy
base_config.config dictionary remains mutable for old imports; deep-copy it before
legacy experiments. There is no YAML/Hydra framework. Construction-driven component
selection and versioned resolved JSON are documented in the
[composition examples](components/composition-examples.md); arbitrary Python files
are not automatically discovered or consumed by the CLI.

A future runtime edit affecting the original route must either preserve a usable
original behavior path or explicitly declare a baseline-changing migration,
compatibility consequences, new identifier, recovery and comparison evidence.
Config changes alone cannot preserve arbitrary code changes; test preservation.
Do not duplicate the entire implementation merely to keep a baseline.

## Research contracts and ablations

For a real component experiment, copy the
[experiment template](experiments/TEMPLATE.md) to a named record alongside its
actual configuration. Reference authoritative metadata instead of maintaining
two copies. Required information is the scientific comparison: baseline,
hypothesis, changed component, intentionally unchanged components, dataset/split,
tasks, metric definitions, seeds, checkpoint initialization, config/command,
and expected comparison. An ablation isolates one controlled difference where
feasible; describe unavoidable confounders.

Record run identity, code revision and dirty patch, working directory, resolved
configuration, environment/package versions, hardware, dataset fingerprint/split,
preprocessing version, encoder/checkpoint checksums, simulator revisions and task
settings. Record requested and actually used seeds, including demonstration and
task selection. A seed helper or a CLI flag alone is not proof of determinism.

## Evidence and storage

Definitions, reproduction commands, configuration deltas, decisions and result
links belong in Git. Large datasets, checkpoints and metric histories belong in
appropriate artifact stores or existing tools such as WandB. Do not commit
credentials, private data, or large outputs. The original ignore file does not
provide a research-artifact retention policy; inspect additions before staging.

Use immutable artifact identifiers/checksums when possible. Cached scene embeddings
must identify their encoder and preprocessing provenance. Record compatibility
before reusing them with another encoder. Load pickle/PyTorch artifacts only from
trusted sources; checksum identity does not establish publisher trust.

Report uncertainty, number of trials, failed runs and evaluation conditions.
A one-rollout smoke test is not a benchmark, and a numerical baseline claim needs
matched protocol and measured evidence. [Validation reporting](../tests/README.md)
owns the PASS/FAIL/SKIPPED/NOT RUN vocabulary.

## Skills and future work

Encode a research workflow as a skill only when it is stable, repeated, has known
rules, and automation prevents recurring errors. No experiment, component,
baseline-reproduction, configuration-comparison or planner skill is installed now.
Consider harness-improvement work only after repeated observed friction, explicit
scope and a representative rerun; do not change guidance automatically after each task.
