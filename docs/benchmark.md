# Benchmark and Dataset Review

## Research task

Each benchmark example contains an original itinerary, the request that
produced it, and a new edit request. A system returns an edited itinerary.
Evaluation asks whether the result remains feasible, satisfies the new request,
preserves unrelated requirements, and limits unnecessary changes.

An example is not a free-form trip-planning prompt. The original itinerary is
part of the input and must remain recognizable unless the request requires a
larger revision.

## Dataset construction

The available construction pipeline:

1. selects an original itinerary and a change category;
2. creates an edit request grounded in the itinerary and travel data;
3. builds and validates structured requirements for that request;
4. rewrites the request into natural language without changing its meaning;
5. records why the request conflicts with or changes the original itinerary;
6. validates the complete example and separates accepted and rejected items.

An optional extended run also creates planning and mock-edit annotations for
analysis. Those annotations are construction aids, not model answers and not
evaluation ground truth.

The pipeline writes rich internal records for debugging. Release exports
separate the public task fields from the private construction and validation
record. Export checks consistency but does not create a new task category.

## Quality checks

Before release, verify:

- the edit request refers to real places and facts available to the project;
- the natural-language request agrees with its structured requirements;
- the edit-induced budget, route/time, transport, resource, and logical
  requirements have no supported deterministic blocker, unless impossibility
  is intentional;
- the original itinerary and request are preserved exactly;
- accepted and rejected counts are reported for every construction stage;
- private model traces and local paths are absent from the public export.

Edit solvability is checked against the hard requirements of the original
request and the edit request, not by freezing every attraction that happened
to appear in the original itinerary. A trip-length edit may reselect intercity
services. A commute-cap edit may retain origin-required places while replacing
other attractions with candidates near the newly required place. Budget checks
retain applicable intercity cost and include necessary nights, normal paid meal
windows, room/person requirements, and required tickets. The implementation
uses bounded checks rather than full-trip DFS.

`NO_BLOCKER_BY_EDIT_CHECKS` means that no supported check found a deterministic
blocker; it is not a constructive-plan certificate. Proved blockers and
`UNKNOWN` outcomes are automatic nonconforming results.

Passing automatic construction checks does not show that an editing method can
solve the example. Dataset validity and method performance are separate claims.

## Paper-facing release decision

The current paper-facing and Human Baseline cohort is
`batch006_four_types_natural_feasible_187_20260803_v1`. It is the complete
feasible, deduplicated pool rather than a class-balanced subsample. Its natural
intervention distribution is 40 point, 41 range, 67 set/predicate, and 39 global
tasks. The total denominator is 187.

The immediate source pool contains 200 tasks. Hard-gate filtering removes eight
tasks whose solver verdict is `PROVED_INFEASIBLE`; deduplication removes five
additional records with an identical origin-plan, canonical-IR, and canonical
edit-query fingerprint. No model score or experiment result is inspected during
this decision. All 187 retained records have `validation_disposition=passed`,
and the released cohort contains no `manual_review` or proved-infeasible task.
The exact inclusion, exclusion, and duplicate mapping is recorded in the
cohort's `selection_manifest.json`.

The cohort retains all 20 newly generated records: two named-POI point edits and
18 non-budget global transport edits covering total local-transport duration,
total walking distance, and total local-transport cost. The earlier ticket-cost
reaudit also restores the three nonduplicate ticket-budget records whose origin
plans genuinely violate the all-traveler ticket-total constraint.

Paper-facing experiment artifacts are consolidated under
`human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/`.
The matrix contains four models, three methods, and the same 187 task IDs in
every group: 2,244 run/evaluation slots with full coverage. Use
`main_three_methods_four_models/manifest_natural_feasible_187.json` and
`main_three_methods_four_models/natural_feasible_187_matrix_audit.json` for
identity and coverage,
`reports_natural_feasible_187_dcca_v1_1_20260818_v2/` for the 12 offline
Level 1/2/3 reports and their saved-witness proof artifacts, and
`edit_scope_analysis_dcca_v1_1_20260818_v2/edit_scope_analysis.json` for the
current aggregate analysis. The pre-supplement 167/187 coverage snapshot is retained only under
`experiment_results/audits/` as provenance.

The identity-plan check is now a blocking construction invariant. Both
`query_factual_validity` and `origin_requires_edit` must pass; the latter runs
the authoritative Level-2 evaluator with the origin plan as both the baseline
and candidate edit. The historical no-op exclusion manifest and recoverable
task/result archive are under
`human_baseline_data/_exclusions/noop_edit_conflicts_20260801/`. Superseded
fixed-size selection cohorts are recoverably archived under
`human_baseline_data/_superseded/20260803/`; they are not paper datasets.

The project does not claim that this release underwent a formal human
adjudication or risk-stratified spot-check. Such a protocol was proposed in an
earlier planning draft but was neither required by the engineering pipeline nor
completed for this paper. Consequently, reviewer background, agreement,
overrides, and spot-check error rates are not paper metrics. An optional future
sanity audit would be supplementary quality assurance, not part of the current
construction claim.

The same 187-task manifest is loaded by the Human Baseline workbench. Stable
task-ID and source-identity aliases preserve prior submissions and drafts. At
the 2026-08-03 release audit, 45 selected tasks had discovered completed
submissions. Study sessions
and dataset-review decisions remain different records and should be analyzed
separately.

## Release checklist

For a paper or artifact release, preserve:

- the code revision and dependency lock;
- the exact source data and exported example identifiers;
- construction configuration, random seed, model provider, and model name;
- counts before and after every filter and review decision;
- counts and reason codes for every deterministic inclusion/exclusion decision;
- the public benchmark, a private audit copy, and review summary;
- the commands and method settings described in
  [reproduction](reproduction.md);
- evaluation reports with their denominators and unsupported cases.

Do not describe planned generation features, historical batches, or mock edits
as part of the released benchmark unless they are present in the archived
artifact.

## Code and field mapping

| Reader-facing concept | Repository identifier |
|---|---|
| benchmark example | `sample_*.json` |
| change category | `target_bucket`, `primary_conflict` |
| four-type intervention class | `human_baseline_classification.query_class` |
| paper and Human Baseline cohort | `human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/` |
| four intervention classes | `batches/{point,range,set_predicate,global}` under the paper cohort |
| source candidate pool | `human_baseline_data/batch006_four_types_extended_200_20260803_v1/` |
| unified 4-model × 3-method results | `human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/main_three_methods_four_models/` |
| offline aggregate reports | `human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/reports_natural_feasible_187_20260803_v1/` |
| structured request requirements | `edit_target_constraints`, `edit_target_preferences` |
| automatically conforming candidates | `_validation_split/passed` |
| excluded candidates retained for audit | `_validation_split/manual_review` |
| revalidation audit | `_analysis/edit_solvability_v2_revalidation.json` |
| public/private release copies | `exports/public`, `exports/audit` |
| review correction | `data/reviews/.../*.patch.json` |

## Implementation references

- `data_generation/scripts/run_pipeline.py`
- `data_generation/scripts/19_revalidate_edit_solvability_dataset.py`
- `scripts/materialize_human_four_type_dataset.py`
- `scripts/build_natural_feasible_dataset.py --include-all-feasible`
- `scripts/build_natural_feasible_187_matrix.py`
- `scripts/recompute_matrix_reports_offline.py`
- `data_generation/scripts/export_benchmark_views.py`
- `data_generation/config/`
- `data_generation/utils/benchmark_views.py`
- `apps/reviewer-backend/reviewer_backend/`
- `apps/reviewer-frontend/src/`
- `tests/test_pipeline_without_strategy.py`
- `tests/test_benchmark_views.py`
- `tests/apps/test_human_baseline.py`
