# Minimal-edit plan cases: cascade impact analysis

> **Terminology note (2026-08-15):** This file name and some historical result
> labels retain `minimal edit` for artifact compatibility. In the paper, these
> diagnostics belong to **Editing Behavior**; Level 2 is reported as **Edit
> Correctness**, and no single minimal-edit score is treated as a third
> correctness gate.

> **Current paper pipeline (2026-07-14):** Section 9 supersedes the earlier
> DeepSeek-only cohort numbers below. Those earlier numbers are retained as a
> historical diagnostic, but must not be compared with the formal typed runs.

## 1. Operational question

The current benchmark Level3 is correctly gated on Level1 feasibility/origin
preservation and Level2 Edit Correctness. It reports the inferred edit scope,
atomic-operation counts, content retention, POI-sequence edit distance, and
activity change ratio. These quantities measure **edit volume**, but not the
paper's more specific claim: a small requested change can propagate through
temporally and structurally dependent itinerary items.

We therefore separate:

- **Direct target change**: a member of the smallest observable necessary-set
  proxy for its constraint type. Named entities are exact; numeric, temporal,
  semantic, and day/global targets use explicitly labeled conservative proxies.
- **Cascade change**: every other changed activity after that direct target is
  accounted for.
- **Cascade amplification**: `all changed activities / direct target changes`.
- **Spillover ratio**: `cascade changes / all changed activities`.
- **Cascade radius**: maximum shortest-path distance from a grounded target to a
  reachable changed activity in the origin dependency graph.
- **Cross-day spillover**: changed activities outside target days.
- **Disconnected changes**: changed activities not connected to a target by the
  observable graph; this is best interpreted as uncontrolled/global rewrite,
  not as a proven causal effect.
- **Rollback-required support**: a non-target change for which no valid
  counterfactual is found within the complete, versioned rollback family. This
  is not a universal necessity claim.
- **Verified removable**: a non-target change with a saved rollback witness
  that passes the complete gates. This is evaluator-relative evidence, not
  proof of global dispensability.

The observable dependency graph contains within-day consecutive-activity edges
(shared time/transport boundary) and cross-day same-entity edges (typically a
hotel continuity dependency). This is intentionally a lower-bound graph: a
missing edge never fabricates a cascade and instead produces a visible
`disconnected_change_count`.

## 2. Candidate schemes

| Scheme | Core quantity | Required input | Complexity | Interpretability | Main limitation |
|---|---|---|---|---|---|
| A. Flat edit-cost baseline | Weighted atomic edit count, retention, sequence distance | Origin and edited plans | Linear plus sequence edit distance | High as a minimality score | Cannot distinguish required edits from collateral edits; two plans with equal edit count can have very different propagation |
| B. Type-aware dependency cascade **(implemented)** | Direct vs cascade changes, amplification, graph radius, day spillover | Plans plus structured edit constraints | Linear graph construction and BFS after existing diff | High; exposes attribution mode and confidence per case | Non-entity modes are conservative proxies; graph edges indicate plausible dependence, not causal proof |
| C. Counterfactual repair delta | Compare the chosen edit with the minimum feasible plan under target ablation or alternative target placements | Deterministic solver/replanner, full constraints, multiple reruns | High to very high | Strongest causal reading: marginal edits induced by the target | Solver optimum and tie-breaking become part of the metric; expensive and currently unavailable for all constraints |
| D. Oracle cascade closure | Human labels the initial editable refs and each justified expansion trigger | Expert annotation plus proposed edits | High annotation cost | Best gold standard for scope precision/recall and justified cascade | Small scale, annotator disagreement, privileged diagnostic rather than a fair end-to-end metric |

Recommendation: use **B as the main scalable analysis**, retain **A as the
minimality baseline**, and annotate a stratified 30--50-case subset with **D**
to validate B. Use **C** only for a solver-supported subset as a mechanistic case
study. The repository's `oracle_scope_v1` schema already provides a natural
starting point for D (`initial_editable_refs`, bounded expansions, triggers, and
`cascade_required`).

## 3. Why this design is defensible

- Classical minimal-change work treats an update as selecting states closest
  to the original state, motivating the retained-content/edit-cost baseline,
  while also documenting that a single generic minimal-change rule can be
  problematic under richer information. See the primary AAAI discussion of
  Winslett-style possible-model updates: [Foo and Zhang, AAAI 1996](https://cdn.aaai.org/AAAI/1996/AAAI96-084.pdf).
- Graph edit distance formalizes transformation cost as a least-cost sequence of
  node/edge edit operations. This supports Scheme A but also shows why costs must
  be explicitly chosen: [Bunke, Pattern Recognition Letters 1997](https://doi.org/10.1016/S0167-8655(97)00060-3).
- Planning causal graphs expose implicit variable dependencies and decompose
  goal-distance computation into local dependency windows. This motivates
  measuring propagation over an explicit dependency graph rather than treating
  the itinerary as a flat string: [Helmert, JAIR 2006](https://ai.dmi.unibas.ch/papers/helmert-jair06.pdf).

These sources motivate the representation and metric families; they do not
establish that an itinerary adjacency edge is causal. Hence the report uses the
term *observable dependency cascade* and reports disconnected changes rather
than forcing all edits into a causal chain.

## 4. Implemented analysis and reproducibility

Implementation:

- `src/evaluation/cascade_analysis.py`: deterministic per-case target grounding,
  dependency graph, and metrics.
- `scripts/analyze_minimal_edit_cascade.py`: batch runner with JSON, CSV, and
  Markdown outputs.
- `tests/test_cascade_analysis.py`: direct/cascade separation, cross-day
  spillover, numeric, temporal, semantic, day-structure, forbidden-entity, and
  exact-view compatibility tests.

### Type-aware necessary sets

| Mode | Necessary modification set | Allowed impact surface | Confidence |
|---|---|---|---|
| `exact_entity` | Named target insertion/replacement/time change, or named forbidden-entity deletion | Target plus dependency-reachable changes | High |
| `neighborhood_constraint_proxy` | Changed distance-1 neighbor of an unchanged named anchor | Anchor neighborhood | Medium |
| `global_numeric_proxy` | Matched activity with reduced cost, or deletion in the requested activity class | Any plan activity may be inspected; only favorable cost deltas are direct | Medium |
| `global_numeric_fallback` | One virtual required unit when the plan changed but no cost delta is observable | Global | Low |
| `temporal_proxy` | Matched activity whose start/end time changed | Same-day schedule dependency chain | Medium |
| `semantic_fine_tag` | Inserted/replaced activity whose ChinaTravel concept scorer returns the requested semantic class | Same class and dependency neighborhood | Medium |
| `semantic_type_proxy` | Coarse-type fallback for semantic families without an applicable fine scorer | Same type and dependency neighborhood | Low |
| `day_structure_proxy` | One virtual unit per added/removed day boundary | Cross-day allocation | Medium |
| `day_allocation_proxy` | Activity actually moved across days | Source and destination day | Medium |

Fallback modes never mark a whole day or whole plan as direct. A virtual direct
unit keeps attribution measurable while making low-confidence assumptions
visible. `exact_entity_*` parallel fields preserve the original exact-name
metric even when another type-aware mode also applies.

Legacy `cascade_change_count` and `spillover_ratio` remain “all non-direct
impact.” The current five-way view reports direct target, rollback-required
support, scope-authorized completion, verified removable, and unresolved evidence.
It adds `target_satisfied`, `proof_coverage`, the four class counts/units, and
the four-way total. Three-way field names remain compatibility aliases. Virtual
day/global units occur only in the new total, so no activity masquerades as a
day container.

### Original coverage audit

Before type-aware attribution, the all-completed cohort covered 78/182 records.
The remaining 104 records were dominated by these mutually exclusive constraint
combinations: semantic-only 14, activity-budget 12, total-budget 10,
forbidden+semantic 10, forbidden-only 10, restaurant-type 9,
anchor-distance+required-name 9, day-count+daily-cap 8, ticket-budget 7, and
budget+day-count 6. Smaller groups account for the remainder. At the overlapping
constraint-type level, the largest unsupported counts were semantic requirement
26, forbidden attraction 20, day count 16, total budget 16, activity budget 12,
and required attraction 12.

The current paper workflow freezes the four formal ReAct runs and recomputes
Level3 eligibility with the repaired canonical IR. Strict analysis deliberately
fails without this manifest; it never falls back to historical evaluation
reports.

```bash
PYTHONPATH=.:src:scripts .venv/bin/python scripts/build_paper_cascade_manifest.py \
  --output experiments/main_analysis/cascade_paper_manifest_batch006_react_v1.json
```

One route cache is then built from the union of requests across all four models,
so a shared route uses the same ChinaTravel snapshot and is queried only once.

```bash
PYTHONPATH=.:src:scripts .venv/bin/python scripts/precompute_cascade_route_evidence.py \
  --paper-manifest experiments/main_analysis/cascade_paper_manifest_batch006_react_v1.json \
  --output experiments/main_analysis/cascade_route_evidence_batch006_react_four_models_v1.json

PYTHONPATH=.:src:scripts .venv/bin/python scripts/analyze_paper_cascade.py \
  --paper-manifest experiments/main_analysis/cascade_paper_manifest_batch006_react_v1.json \
  --route-evidence-cache experiments/main_analysis/cascade_route_evidence_batch006_react_four_models_v1.json \
  --output-dir experiments/main_analysis/cascade_analysis_batch006_react_four_models_v1
```

## 5. Results and paper-facing finding

### Strict, valid minimal-edit cohort

There are 12 Level3-eligible records. Type-aware attribution covers all 12,
improving coverage from **6/12 (50%) to 12/12 (100%)**. On these 12:

- mean cascade amplification is **2.682x** (11 records have a non-zero direct denominator);
- mean spillover ratio is **46.7%**;
- mean affected-day count is **1.167**;
- mean cascade radius is **0.917**;
- mean disconnected change count is **0.917**.

By mode, exact-entity-only cases average 3.083x amplification; three
global-numeric cases average 2.333x and 55.6% spillover; the one neighborhood
constraint case reaches 3x and spans two days. One eligible record has no
observable activity change and is reported as `no_observed_change`, with no
fabricated amplification denominator.

### Diagnostic completed cohort

All 182 records with both plans now have an attribution mode, improving coverage
from **78/182 (42.9%) to 182/182 (100%)**. This cohort is not gated for
correctness, but it is useful for mechanism inspection:

- success-only amplification **2.564x**, legacy spillover **38.3%**;
- parameter cases: **1.0x**, 0% spillover, 1.0 affected day;
- structural cases: **2.141x**, 34.4%, 1.229 affected days;
- compositional cases: **5.909x**, 80.3%, 2.619 affected days, and 8.714
  disconnected changes on average;
- failure records are excluded from all bullets above and reported separately.

The parameter/structural/compositional breakdown below is a legacy artifact and
must not be used as a paper-facing benchmark taxonomy or result grouping. Current
reporting uses concrete edit operations, affected-day count, retention, sequence
distance, activity change ratio, and rollback attribution. Do not frame this
diagnostic cohort as method performance because many records fail Level1/Level2.

The primary aggregate includes only target-satisfied records. Strict N=12 has
10.23% avoidable lower bound, 18.48% upper bound, and 91.74% adjudication
coverage. The all-completed file contains 162 successes and 20 failures; the
With the versioned ChinaTravel route-evidence cache, the success-only main view
has 17.80% lower, 20.64% upper, 45.60% legacy v1.0 hard-support share, 2.25% authorized
share, 2.84% unresolved share, and 94.91% coverage. The cache contains 240
fixed `(city, start, end, start_time)` requests and complete walk/metro/taxi
responses; analysis never performs an unrecorded live lookup.
The 20 failures are reported only under `failure_impact` and are never mixed
into these main means.

## 6. Presentation optimized for the cascade claim

Use one main figure with two panels:

1. A stacked bar per scope showing direct / distance-1 / distance-2+ /
   disconnected changed activities (strict cohort when enough records exist;
   otherwise the expanded diagnostic cohort clearly labeled).
2. A scatter plot of amplification vs. affected days, with point shape for
   Level3 eligibility and annotations for the 1x and 8x cases.

Report target-grounding coverage next to every aggregate. Avoid a single
unqualified `MED` number: it hides both required target work and propagation.
For the final paper, report strict-cohort counts beside every mean and keep the
interpretation descriptive. The current study does not claim scope-level
statistical significance and does not add bootstrap intervals or repeated
model runs.

## 7. Known risks and next data action

- Semantic and fallback modes are measurable but not exact; all comparisons must
  retain `attribution_mode` and `attribution_confidence`.
- Although `day_allocation_proxy` is implemented and regression-tested, no
  DeepSeek record in this 182-case cohort contains a matched named POI that
  actually moves across days; real `poi_day_binding` cases use exact named
  insertion/replacement attribution instead.
- Matching and atomic-op inference inherit the current Level3 heuristic,
  including replacement ambiguity.
- Consecutive-activity and repeated-entity edges are observable dependencies,
  not proof of causal necessity.
- Rollback-required evidence is validator- and family-relative. If the edited
  plan already fails a rule, only newly introduced violation codes count;
  coordinated repairs outside the versioned family may still exist.
- Four-way direct is intentionally narrower than legacy attribution: only an
  explicit named target, a day container, or the minimum semantic cardinality
  can be direct. Budget reductions and schedule changes must pass rollback
  adjudication.
- Twelve strict cases are enough for a concrete mechanism demonstration, not a
  statistically strong model comparison; per-mode Ns are smaller still.

Human gold necessary sets and global-minimum plans are not required by the
current operational claim. Future annotation of `direct_target_refs` could
improve low-confidence semantic proxies, but it is an optional extension rather
than a missing prerequisite for the paper.

## 8. Four-case validator audit

- `adjacent_travel_time_cap` sums all valid `(end-start)` legs in the inbound
  `transports` of the later filtered attraction. In
  `structural_temporal_overflow/sample_000003`, the reported edges are 18 and 14
  minutes. Removing 重庆润泽射击俱乐部 leaves a nominal 14-minute edge but
  leaves the inbound transport starting at the deleted POI; it is therefore
  unresolved/needs-repair, not proven avoidable. Final split: 1 explicit direct
  (红岩), 3 historical v1.0 hard-support labels, 0 unresolved units. Cached direct routes from
  红岩革命纪念馆 to 民俗文化村 take 258 minutes by walk, 59 by metro, and 32
  by taxi, so every supported mode violates the 25-minute cap.
- Fine semantic attribution reuses ChinaTravel's `attraction_type`. 汉口江滩
  resolves to `公园`, not `自然风光`, and the edited plan actually fails the
  semantic constraint. The record now uses low-confidence `semantic_fallback`
  rather than a false fine-tag direct claim, with `target_satisfied=false`; it
  is excluded from the success-only main aggregate.
- `compositional_structural_overflow/sample_000010` has one virtual day target,
  three historical v1.0 hard-support labels (return-flight move, accommodation, lunch), one authorized
  completion (new-day breakfast), and one unresolved new-day attraction whose
  removal requires repairing the following inbound transport. It has no legacy
  v1.0 avoidable-extra unit. Its baseline already has `duplicate_poi`.
- `compositional_semantic_discontinuity/sample_000006` has 2 direct targets, 5
  historical v1.0 hard-support labels, and one unresolved lunch replacement. Restoring the old lunch
  does not regress edit/hygiene/quality checks, but breaks the following inbound
  transport endpoint, so it is not proven avoidable. Its baseline idle-gap
  violations remain an evidence limitation.

## 9. Legacy four-model batch006 audit (not paper-facing)

### Cohort construction

The source of truth is `experiments/main_analysis/main_experiment_registry.jsonl`:
only rows with `include_in_main_table=true`, `framework=react`, and the four
paper models are selected. Gemini's current merged run uses the latest completed
evaluation attempt for each `(primary_conflict, sample_id)`, matching the paper
table recomputation. Every model is required to have exactly 196 unique tasks.

Task identity is the sample basename plus a hash of the full input after removing
run-local paths, IDs, and the physical conflict label. This proves that all four
models received the same task multiset. Category comes from the canonical
evaluation record's `primary_conflict`, not from a parent directory. This also
repairs the historical DeepSeek layout in which 13
`parameter_resource_overflow` cases were stored under
`structural_resource_overlap`.

Strict eligibility is recomputed by the current `BenchmarkEvaluator` after
overlaying the 29 frozen canonical-IR repairs, then checked against the final
recomputed paper Table 3. The resulting eligible counts are 56 DeepSeek, 57
Gemini, 23 GPT-4o-mini, and 16 Qwen3-32B. A missing eligibility value is a hard
error; historical per-run reports are not silently reused.

The union route cache contains 354 unique
`(city, start, end, start_time)` requests. Each request stores walk, metro, and
taxi results under one tool/data fingerprint. This makes route evidence shared
and reproducible across models, although ChinaTravel remains an offline
simulator rather than a real-time map service.

### Denominators and results

`Valid plan` means both origin and edited plans are present. `Supported` means
the request has an implemented attribution mode. `Target satisfied` is the
success-only main cohort. Rates below are case-level means over that cohort, not
ratios pooled over activity units.

| Model | Total tasks | Valid plan | Supported | Target satisfied | Strict eligible | Avoidable lower--upper, all | Evidence coverage, all | Avoidable lower--upper, strict | Evidence coverage, strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 196 | 178 | 178 | 148 | 56 | 20.18--21.81% | 96.48% | 16.56--16.82% | 99.22% |
| Gemini-3.1-Pro-Preview | 196 | 147 | 146 | 131 | 57 | 14.27--15.84% | 97.55% | 10.62--10.62% | 100.00% |
| GPT-4o-mini | 196 | 177 | 172 | 116 | 23 | 16.44--17.88% | 96.43% | 13.66--14.41% | 99.25% |
| Qwen3-32B | 196 | 124 | 124 | 98 | 16 | 18.09--20.50% | 96.91% | 0.00--0.00% | 100.00% |

The strict cohort should be the paper-facing comparison because it uses the same
feasibility, preservation, edit-success, and repaired-IR gate as the main paper.
The all-completed view remains useful for diagnosing model behavior but mixes in
plans that fail Level1 or Level2. Qwen's strict zero does not establish a general
absence of unnecessary edits: it is based on only 16 eligible cases and should
be accompanied by its cohort size.

### Remaining validity risks

- Local rollback defines removability or rollback-required support relative to
  the current validator, edited plan, and versioned candidate family. It is not
  an approximation to a claimed global-minimum solution.
- A route mode with a tool error is missing evidence, not proof that no route
  exists. The lower--upper interval preserves this uncertainty.
- Formal run paths in the generated manifest are absolute workspace paths. The
  manifest must be regenerated after moving the experiment mirror.
- The route cache fingerprint hashes code and compact data manifests, not every
  underlying POI and metro database file.
- Strict cohort counts are always reported beside means. The paper uses frozen
  single runs and makes descriptive comparisons only; it does not report
  confidence intervals, repeated-run variance, significance tests, or
  point-estimate winner claims.

## 10. Integration into the formal Level3 report schema

Cascade analysis is now reported as `level3.cascade`, while Level3 eligibility
keeps its original meaning: `eligible = Level1 pass AND Level2 pass`. No cascade
threshold or cascade pass/fail field is introduced.

For every v1.1 eligible sample, the nested result contains the five evidence
counts (`direct_target`, `rollback_required_support`,
`scope_authorized_completion`, `verified_removable`, and `unresolved`), total impact count, removable lower and
upper rates, proof coverage, and evidence shares. Eligible samples for which the
offline evidence enrichment was not run are marked `not_computed`. Ineligible
samples are explicitly `not_evaluated`, so they cannot enter any cascade mean.
Route/tool failures remain unresolved evidence and can only widen the
lower--upper interval.

The legacy four-model artifact uses schema `cascade-paper-comparison-v2` and
retains `hard_required_support` / `avoidable_extra` field names. It
contains a `level3_cascade` aggregate for each model and emits both JSONL and
CSV sample tables covering all 196 tasks per model. The official Level3 cascade
cohort is therefore 56/196 for DeepSeek, 57/196 for Gemini, 23/196 for
GPT-4o-mini, and 16/196 for Qwen3-32B; the remaining samples are
`not_evaluated`, not cascade failures. The eligible-only mean results remain:

| Model | L3 evaluated | Not evaluated | Legacy avoidable lower--upper | Proof coverage | Legacy hard-support share | Authorized-completion share | Unresolved share |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 56 | 140 | 16.56--16.82% | 99.22% | 54.19% | 0.52% | 0.26% |
| Gemini-3.1-Pro-Preview | 57 | 139 | 10.62--10.62% | 100.00% | 54.85% | 0.00% | 0.00% |
| GPT-4o-mini | 23 | 173 | 13.66--14.41% | 99.25% | 32.21% | 0.00% | 0.75% |
| Qwen3-32B | 16 | 180 | 0.00--0.00% | 100.00% | 32.29% | 0.00% | 0.00% |

These are descriptive Level3 submetrics. In particular, Qwen's zero avoidable
rate applies only to its 16 surviving eligible samples and is not an overall
model win.

## 11. IPM paper-facing cohort and shape-specific propagation

The IPM analysis uses only DeepSeek-V4-Pro and
Gemini-3.1-Pro-Preview. No model is rerun. Frozen outputs are filtered by the
final 188-task dataset after excluding eight candidates with proved
edit-solvability blockers. The strict L1+L2 cohorts remain 56 for DeepSeek and
57 for Gemini; none of the eight excluded candidates had entered the strict
cascade cohort.

Every shape reports total changed activities, total attributed impacts,
affected days, cross-day spillover, the removable lower--upper interval,
legacy hard-support share, unresolved share, and proof coverage. The paper then
emphasizes a shape-specific diagnostic:

- point: cascade radius and disconnected changes from the named seed;
- range: spillover ratio and cross-day spread beyond the local adjustment;
- set/predicate: rollback-required share and removable interval while satisfying the
  predicate;
- global: cascade amplification and affected-day coverage.

The resulting summary is stored in
`experiments/main_analysis/constraint_grounded_rq_analysis/shape_impact_summary.csv`.
These readouts describe observable propagation. They do not infer a hidden
causal graph, and local rollback is not presented as a proof that no globally
smaller feasible plan exists.
