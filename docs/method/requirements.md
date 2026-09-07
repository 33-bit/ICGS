# Proposal coverage and acceptance ownership

Status: documentation traceability, not executed acceptance. Source is the
immutable [English proposal](../proposals/ICGS_proposal_English.tex). Exact section
titles and equation labels avoid ambiguity from rendered numbering. PR/AD/ID/EI/FG
are defined in the [index](README.md). Every test path below is a **planned** flat
unittest owner unless explicitly EI; none is claimed created by this docs task.

## Equation coverage

The following rows account for every `eq:` label in the archived source. The plan
column links to the owner; the acceptance column names concrete observable behavior,
not just “test the component.” Neural and training formulas remain in the archive
and are operationalized by the linked spec rather than silently altered.

| Source labels | Kind / specification | Plan | Planned test owner and assertion |
| --- | --- | --- | --- |
| `eq:obs`, `eq:objective` | PR [contracts](contracts.md) | [P00](../plans/active/icgs-p00-contracts.md), [P01](../plans/active/icgs-p01-timed-execution.md) | `test_method_contracts.py`, `test_timed_execution.py`: no oracle input; command versus achieved state; timeout0 |
| `eq:factor` | PR [separation](README.md#architecture-and-lifecycle) | [P04](../plans/active/icgs-p04-physical-memory.md), [P06](../plans/active/icgs-p06-task-router.md) | `test_physical_memory.py`, `test_task_router.py`: same physics history unchanged by context, task replay changes q |
| `eq:attn`, `eq:gru` | PR [shared blocks](neural.md#shared-blocks) | [P03](../plans/active/icgs-p03-geometry-bridge.md), [P04](../plans/active/icgs-p04-physical-memory.md) | `test_physical_geometry.py`, `test_physical_memory.py`: mask behavior, finite gradients and reset-after GRU |
| `eq:encoder`, `eq:decoder`, `eq:observationbridge` | PR/ID/FG [geometry](neural.md#p03-geometry-encoder-and-observation-bridge) | [P03](../plans/active/icgs-p03-geometry-bridge.md) | `test_physical_geometry.py`: shape/duplicate masks, one frame conversion; real native-action drift gate |
| `eq:proprio`, `eq:memory` | PR [history](neural.md#p04-physical-representation-and-memory) | [P04](../plans/active/icgs-p04-physical-memory.md) | `test_physical_memory.py`: dimensions13/277/131, causal replay and branch isolation |
| `eq:demoframe`, `eq:eventtoken` | PR/ID [events](neural.md#p05-segmentation-and-event-encoder) | [P05](../plans/active/icgs-p05-event-memory.md) | `test_event_memory.py`:269/776 dimensions, protected transitions, overflow and permutation |
| `eq:taskmemory`, `eq:eventheads`, `eq:router` | PR/ID [task routing](neural.md#p06-task-tracker-and-router) | [P06](../plans/active/icgs-p06-task-router.md) | `test_task_router.py`: null, rho1/nu0, invalid-window normalization, D1/D2 compatibility |
| `eq:actionbridge`, `eq:actiondesc` | PR/EI/ID [command contract](contracts.md#online-information-and-frametime-contract) | [P01](../plans/active/icgs-p01-timed-execution.md), [P07](../plans/active/icgs-p07-world-model.md) | `test_timed_execution.py`, `test_world_model.py`: common absolute target, distinct hypothesis u, translation-first SE(3) |
| `eq:wmtrunk`, `eq:wmpredict`, `eq:wmrecursive` | PR/ID [dynamics](neural.md#p07-physical-dynamics-and-closed-rollout) | [P07](../plans/active/icgs-p07-world-model.md) | `test_world_model.py`:132 tokens, fixed heads, decode once, predicted recursive input and frozen-path gradients |
| `eq:valuedef`, `eq:valuekeys`, `eq:valueforward` | PR/AD [value](neural.md#p09-continuation-completion-progress-and-terminal-hazards) | [P09](../plans/active/icgs-p09-evaluators.md) | `test_evaluators.py`: active V0=0, deadline-independent S/Phi, completed events retained |
| `eq:hazard` | PR [terminal](neural.md#p09-continuation-completion-progress-and-terminal-hazards) | [P09](../plans/active/icgs-p09-evaluators.md) | `test_evaluators.py`:1288 input, active-before first-event labels, mutually exclusive mass |
| `eq:cd`, `eq:physloss`, `eq:wmloss` | PR [physical loss](data-training.md#loss-definitions-and-pair-independence) | [P03](../plans/active/icgs-p03-geometry-bridge.md), [P07](../plans/active/icgs-p07-world-model.md), [P11](../plans/active/icgs-p11-training.md) | `test_physical_geometry.py`, `test_world_model.py`, `test_method_training.py`: loss scaling, stable rotation, recursive bootstrap and omission of frozen Lrec |
| `eq:taskloss` | PR [task labels/loss](data-training.md#collection-and-event-annotations) | [P06](../plans/active/icgs-p06-task-router.md), [P11](../plans/active/icgs-p11-training.md) | `test_task_router.py`, `test_method_training.py`: masked averages, null/recovery, gradient ownership |
| `eq:outloss`, `eq:evallossbasic`, `eq:prefuncertainty`, `eq:prefloss`, `eq:totaleval` | PR/ID [outcome losses](data-training.md#loss-definitions-and-pair-independence) | [P08](../plans/active/icgs-p08-outcome-collection.md), [P09](../plans/active/icgs-p09-evaluators.md) | `test_outcome_collection.py`, `test_evaluators.py`: n weighting, correlated-trial handling, empty pairs0, disjoint pools |
| `eq:node`, `eq:backupmass`, `eq:return`, `eq:uct` | PR/ID [search](planning-evaluation.md#p10-belief-bookkeeping-and-search) | [P10](../plans/active/icgs-p10-search.md) | `test_search.py`: analytic return, mass conservation, first-terminal once, UCT zero visits, exact cache not visits |
| `eq:program`, `eq:adaptation` | PR/FG [generation](data-training.md#programs-splits-and-predicates) | [P02](../plans/active/icgs-p02-episode-data.md) | `test_episode_data.py`: catalog/split lineage, re-execution metadata, annotation-only roles |
| `eq:suffixlabel` | PR [context-specific trials](data-training.md#p08-anchors-replay-and-continuation-targets) | [P08](../plans/active/icgs-p08-outcome-collection.md) | `test_outcome_collection.py`: context swap invalidates continuation labels, no forced reversal |
| `eq:datacost` | PR [budgets](data-training.md#pilot-budgets-and-scaling-gates) | [P08](../plans/active/icgs-p08-outcome-collection.md), [P13](../plans/active/icgs-p13-integration.md) | `test_outcome_collection.py`, `test_method_integration.py`: upper bounds versus actual terminal-reduced counts; shared-view deduplication |
| `eq:calibration` | PR/ID [stages](data-training.md#p11-trainablefrozen-phases) | [P09](../plans/active/icgs-p09-evaluators.md), [P11](../plans/active/icgs-p11-training.md) | `test_evaluators.py`, `test_method_training.py`: positive temperatures, calibration-only partition, V0 unaffected |
| `eq:metrics` | PR/ID [reporting](planning-evaluation.md#experiments-ablations-and-reporting) | [P12](../plans/active/icgs-p12-controls-evaluation.md) | `test_method_evaluation.py`: macro SR, trial Brier, finite-pool regret, paired bootstrap |

## Non-equation coverage

| Proposal section / substantive requirement | Kind / spec owner | Plan and acceptance |
| --- | --- | --- |
| Summary; Introduction and Research Questions; Related Work | PR [research claims](README.md#research-question-and-claims) | P12: H1/H2/H3 mapped to E2/E4/E3; no novelty-by-components claim; citations preserved as author claims |
| Observations, Context, and Allowed Information | PR/FG [contracts](contracts.md) | P00/P02: no language/program/oracle inputs; independent1–2 demos; foreground union and ambiguity split |
| Objective and Terminal Convention | PR/AD [execution](planning-evaluation.md#budgets-fallback-and-stopping) | P01/P09/P13: five-interval external confirmation, failure precedence, three-observation learned stopping kept distinct |
| Time and Coordinate Frames | PR/ID/FG [contracts](contracts.md) | P00/P01/P13: T/P versus demo count; metric units; measured0.1 s pilot; deadlines and waiting |
| Overall Architecture / input-output table | PR/AD [neural](neural.md) | P00/P03–P09: all tensor widths and dependency directions checked |
| Event segmentation and native window preprocessing | PR/ID [events/router](neural.md) | P05/P06: deterministic thresholds/ties, suffix overflow, mandatory-frame fit, no duplicated demos |
| Ensemble uncertainty and action/observation bridge fidelity | PR/FG [neural](neural.md) | P03/P07: empirical action fidelity; no posterior/contact-mode claim; real versus imagined paths |
| Losses and Gradient Flow; Algorithms3/4 | PR [training](data-training.md) | P07/P09/P11: frozen parameters versus transmitted gradients, predicted recursive inputs, observed calibration |
| Algorithm1; Reranking, Shooting, and Fair Horizons | PR/ID [planning](planning-evaluation.md) | P10/P12: visits/cache/deadline fixtures, equal-time/equal-L/call counts, fallback overshoot |
| Principles and Primary Simulator; Available Sources and Conditions for Use | PR/FG [data](data-training.md) | P02/P08: RLBench/Panda executed data, no mixed-engine/pseudo-physics labels; external sources optional |
| Task Grammar and Pre-Generation Data Splits; Program Catalog for the Composition Split | PR/ID/FG [programs](data-training.md#programs-splits-and-predicates) | P02/P12: T01–T20,V01–V04,P/G/R catalog, family/lineage split; concrete asset manifest gate |
| Data Views and Module Mapping; Minimum Record Schema | PR/ID [episode records](data-training.md#p02-episode-repository-and-views) | P00/P02: versioned shards, masks, no pickle tensors/oracle inputs, unique episodes not additive views |
| Collecting Successful Contexts and Executed Attempts; Creating Event/Task-Memory Labels | PR/ID/FG [collection](data-training.md#collection-and-event-annotations) | P02/P05/P06: retain failure/perturbation traces, overlap masks, history/current predicates differ |
| Branch Bank and Continuation Labels; Algorithm2; Snapshot Integrity | PR/ID/FG [outcomes](data-training.md#p08-anchors-replay-and-continuation-targets) | P08: exact/replay/approximate classes, frozen pi_ref, censoring, terminal-prefix and horizon handling |
| Matched-Suffix and Recovery Data | PR [outcomes](data-training.md#p08-anchors-replay-and-continuation-targets) | P08/P09/P12: unchanged physical pool where possible; trial-supported recovery/reversal |
| Calibration, Hard Examples, and Data Budget | PR/ID/FG [scale](data-training.md#pilot-budgets-and-scaling-gates) | P08/P11/P12/P13: train-only hard examples,50% original bank, calibration/audit no gradients, measured resource gate |
| Optimization Order; Optimizer, Sampling, and Early Stopping | PR/AD/ID [stages](data-training.md#p11-trainablefrozen-phases) | P11: A0/A1/B/C/D/E trainable groups, history burn-in, exact reference/cache IDs, resume evidence |
| Equal-Data Reactive Control; Baseline Ladder | PR/AD/ID [controls](planning-evaluation.md#p12-baseline-ladder) | P12: defined B3 student/B4 Q objectives and matched banks; B0/B1 wrapper separation |
| Primary Benchmark; Geometry, Predicates, and Precise Labels | PR/FG [predicates](data-training.md#programs-splits-and-predicates) | P02/P12:12 custom held-out compositions, expert-certified instances, threshold sensitivity and invalid rates |
| Verifying Downstream Reasoning | PR [experiments](planning-evaluation.md#experiments-ablations-and-reporting) | P12: K16 diagnostic pool not test filter; independent-chain/memory controls; feasible J/delta cells |
| Public Benchmarks and External Validity; Optional Real-Robot Evaluation | PR/FG [extensions](planning-evaluation.md#optional-extension-gates) | P12/P13: supported native regression mandatory; official/adapted regimes separate; no robot launch authority |
| Experiments; Priority Ablations; Oracles and Failure Attribution | PR/ID [reporting](planning-evaluation.md#experiments-ablations-and-reporting) | P12: E1–E9, retraining and equal-data controls, physical oracle matching, evidence-backed attribution |
| Metrics and Statistics | PR/ID [reporting](planning-evaluation.md#experiments-ablations-and-reporting) | P12:3600 episodes/method primary, one-demo50 resets,95% paired CI/10000 resamples/Holm/no seed pseudoreplication |
| Expected-Result Analysis and Limitations | PR [limitations](README.md#limitations-and-optional-tracks) | P12/P13: support, contact aliasing, decoded drift, weak-reference zeros, optimism, head correlation, no guaranteed positive result |
| Fixed Choices of Primary Method; Tables and Figures Planned for Empirical Paper | PR [target](README.md), [reporting](planning-evaluation.md) | P12/P13: resolved provenance, tableA/B/C and figureA–E input records; appendix reproduction; no fabricated results |
| References and Data Sources | PR [archive](../proposals/README.md) | Docs acceptance: byte-preserved bibliography; no independent external verification claimed |

## Existing proof versus future proof

**EI:** native candidate/prefix/context tests exist; C1–C5 scoped two-demo synthetic
published fidelity is recorded in [actual evidence](../experiments/vv19-validation/README.md).
Neither establishes learned ICGS, one-demo/window fidelity, fixed cadence, replay,
task success or scientific hypotheses. Planned component tests establish their
assertions only; L3/L4 acceptance requires authorized measured workloads.

**AD resolutions:** separate stopping/reference and B3 diffusion choice are in
ADR0008 and ADR0006 respectively. **ID resolutions** include deterministic tie
rules, gripper commitment, medoid distance, serialization and B3 architecture;
they are explicitly marked in owner specs, not retroactively attributed to the
proposal. **FG owners** and required evidence are in the master roadmap.
