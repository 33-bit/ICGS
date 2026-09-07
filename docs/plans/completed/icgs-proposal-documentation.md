# ICGS proposal documentation delivery

Status: completed — documentation delivery only; runtime remains planned.
Date: 2026-09-07. Owner: repository owner / documentation implementer.
Category C target migration documentation, approved explicitly before edits.
Baseline revision: `4282700`; checkout initially clean, current working directory
`/Users/33bit/AI/Research/VLA/ICGS`. No branch/commit/push operation performed.

## Delivered surface

- Immutable [proposal archive](../../proposals/README.md), byte identity verified
  against owner upload; SHA256 c88cf56cc12e26ed8a2e9ab4e141500bf85e056a40eed396cd051feddbb77c34.
- [Target specification](../../method/README.md), shared contracts, neural method,
  data/training, search/controls/evaluation and full requirements matrix.
- ADR0006–0008 record approved target boundaries, timed-data lineage and reference
  value versus learned stopping; previous ADRs and historical records preserved.
- [Master roadmap](../../plans/active/icgs-method-implementation.md) plus14 detailed
  P00–P13 plans. Those runtime plans remain active, with no runtime work checked off.
- Updated docs map/current architecture/foundation status/native-contract and
  composition cross-links. Existing native examples and semantic text preserved.

## Scope and decisions

No runtime, tests, package configuration, checkpoints, datasets or simulator files
were changed. No model training, data collection, external-source verification,
downloads, simulator execution or robot motion occurred. Uploaded TeX is archived,
not rendered to PDF. Native scientific evidence is linked, not reclassified as
newly executed. Python3.14 is used only for stdlib L0, not model compatibility.

The owner's accepted decisions are separate stopping from reference, full primary
method behind pilot gates, and a separate diffusion B3 student. Deterministic
implementation defaults (sampling/event ties, r-interval grip holding, native
D1/D2 sessions, medoid metric, storage, B3 objective) are labeled ID in owner specs.
They are proposed implementation choices, not retroactive claims from the archive.
Empirical timing/replay/asset/bridge thresholds remain named evidence gates with
owner approval before gate judgment/main/test runs; no numeric success rate is invented.

## Review corrections

Independent read-only review identified and corrected:

1. Physical memory must receive the previous descriptor computed at the
   before-transition pose, not infer it from successor pose.
2. Direct-Q prefix return uses H_root=executed_prefix_length+H_successor;
   k(32)/n after8 intervals labels Q40, not Q32.
3. Terminal hazards use softmax(raw_logits/Te) at inference; raw logits train D2.
4. Proposal section-number references were replaced by exact section titles.
5. Horizon counts include only trials with sufficient observed-through coverage
   or an observed physical terminal; a timeout at32 cannot label H512.
6. Invalid anchor aggregates are zeroed before LayerNorm; B3 invalid action tails
   are sanitized and attention-masked, not merely excluded from loss.
7. Replay declaration/implementation and task-monitor creation/consumption have
   single owners; P11 infrastructure arrives alongside its earliest stage consumer.

Final scoped independent rereview confirmed the horizon-coverage, sparse-anchor
and B3 masking fixes addressed, with no remaining findings in that scope. This is
documentation review, not execution of the planned runtime assertion snippets.

## Commands and evidence

All commands below run from the repository root. Actual documentation checks are
separate from the planned tests in the component documents.

| Command/check | Result | Scope / count |
| --- | --- | --- |
| `cmp docs/proposals/ICGS_proposal_English.tex` against original upload | PASS | byte-for-byte archive identity |
| `shasum -a 256 docs/proposals/ICGS_proposal_English.tex` | PASS | matches recorded source checksum |
| Initial `python3 -B scripts/validate_fast.py` while assembling docs | FAIL | only missing delivery-record link;19 harness tests passed; target added afterward |
| `python3 -B scripts/validate_fast.py` | PASS | syntax, links, boundaries;19 selected/executed,0 skipped |
| `python3 -B -S scripts/validate_fast.py` | PASS | isolated stdlib;19 selected/executed,0 skipped |
| Proposal equation-label coverage | PASS | all43 `eq:` labels mapped to specification/plan/test |
| Component-plan structure and test-first steps | PASS | all14 P00–P13 plans include required sections and RED/GREEN assertions |
| `git diff --check` and docs-only status inspection | PASS | all32 changed/new paths under docs; no whitespace errors |
| L1, L2/C1–C5, L3, L4 | NOT RUN | no runtime change or authorized model/simulator/benchmark workload in this deliverable |

Remaining risk: planned APIs have not been implemented; prototype/model/physics
feasibility and research hypotheses are unproven. Documentation completeness and
L0 PASS do not establish learned-method correctness. Runtime completion belongs
to the active roadmap and its measured gates, not this delivery record.

Environment: Darwin24.6.0 arm64, Python3.14.4. No dependencies were installed.
The model environment remains the separately provisioned supported/pinned stack.
An auxiliary Ruby whitespace-audit attempt used unsupported `filter_map` on the
host Ruby and failed; rerunning with portable `each_with_index` passed for all
new/changed Markdown. This was tooling compatibility, not a repository-test failure.

Reproduce equation-coverage audit without modifying the repository:

```sh
ruby -e 's=File.read("docs/proposals/ICGS_proposal_English.tex"); m=File.read("docs/method/requirements.md"); labels=s.scan(/\\label\{(eq:[^}]+)\}/).flatten; missing=labels.reject { |x| m.include?("`#{x}`") }; abort missing.join(",") unless missing.empty?; puts "PASS: #{labels.size} equation labels mapped"'
```

The structural audit checked all14 plan files for Goal/Architecture/Tech Stack/
Spec, constraints, ownership, acceptance, compatibility/recovery, execution evidence,
assertion snippets, and RED/GREEN steps. Semantic review additionally checked the
method against the archived equations and shared interfaces; string coverage alone
does not prove faithful scientific implementation.
