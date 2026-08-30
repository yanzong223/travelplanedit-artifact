# Evaluation

The evaluation covers three complementary aspects of travel plan editing. Plan
Validity and Edit Correctness establish outcome success; Editing Behavior then
examines the transformation that produced that outcome. The third aspect is a
conditional diagnostic, not a pass/fail gate or a single minimal-edit score.

## What is measured

Evaluation separates three questions that should not be collapsed into one
score.

### Plan Validity

The edited itinerary must remain a valid ChinaTravel plan. Requirements from
the original request that were not changed by the edit request should still
hold. This gate measures the validity and preservation properties of the
revised plan.

### Edit Correctness

The edited itinerary is checked against explicit requirements created for the
edit request. These include supported hard conditions and supported directional
preferences, such as reducing cost or travel time. This gate measures whether
the requested transformation is realized.

The implementation and released artifacts retain the names `Request
Satisfaction`, `Edit Success`, and `level2` for compatibility.

### Editing Behavior

The canonical algorithm is Dependency-Closed Counterfactual Attribution
(DCCA). Its complete definition, result contract, route-repair procedure, and
code map are in [Edit Scope Analysis algorithm](edit_scope_analysis_algorithm.md).

For results eligible for comparison, the evaluator matches activities between
the original and edited itineraries, infers the changes, and reports:

- the broadest area changed;
- counts of changed parameters, activities, and larger structures;
- retained content;
- sequence distance and activity-change ratio;
- a lexicographic edit-cost tuple.

Parameter changes include activity time, nested `transports`, and other
activity attributes such as `cost`, `price`, `tickets`, `room_type`, and
`rooms`. Numeric and clock formatting are normalized before comparison, and
transport changes are split into route semantics (topology, mode, endpoints,
timing, distance, cost/capacity) versus equivalent format/unknown-metadata
rewrites.

These measurements describe the extent and distribution of observed changes.
They are paired with evidence-based attribution for non-target changes.

The inferred object is the **Actual Edit Scope** of the model output, which is
one component of Editing Behavior. Dataset
construction supplies request-side target types and machine-checkable revised-
plan requirements, but it does not supply a gold expected scope, allowed edit
set, or permissible dependency closure. Editing Behavior analysis therefore
characterizes and diagnoses observed changes; it does not determine whether an
output complies with a construction-time edit-scope setting.

## Inputs

An evaluation-ready record needs the original itinerary, edited itinerary,
original request information, edit request, and explicit edit requirements.
The command can enrich older records from a directory of source plans and
requests.

Natural-language text alone is not silently converted into evaluation
requirements at run time. If a requirement type is unsupported, it should be
reported as unsupported rather than counted as a pass or failure.

## Reporting results

Every rate should state its denominator. At minimum, report:

- result files discovered;
- records successfully joined with their source examples;
- records evaluated;
- requirements supported and unsupported;
- results eligible for edit-extent analysis;
- run failures separately from itinerary failures.

A zero rate with a zero denominator is not evidence of poor performance.
Sample-level failures should be preserved for audit.

An experimental attribution analysis further separates rollback-required
support within the tested family from verified removable changes. A changed activity is verified
removable only when dependency-closed local rollback preserves both Plan
Validity and Edit Correctness. For an inbound transport edit, the minimal
closure includes a changed predecessor when reverting the transport alone
would create an artificial endpoint mismatch. When a rollback affects multiple
route boundaries, evaluate the Cartesian product of the locally cached walk,
metro, and taxi repairs and rerun the complete gates on each joint candidate.
Confirmed `ok_no_route` evidence is conclusive; route query errors remain
unresolved and can never establish rollback-required support. A negative label
is allowed only when the versioned rollback family is complete and every tested
candidate fails. Report the resulting Verified Removable-Change Rate with its
upper bound, proof coverage, and unresolved rate. It is evaluator-relative,
eligible-only, and not a pass/fail metric.

## Paper checklist

- Define each metric in reader-facing language before naming its code field.
- State which results were excluded and why.
- Separate hard requirements from preferences.
- Report unsupported checks by type.
- Compare methods on the same-task matched eligible cohort and report its `n`.
- Keep all-eligible edit extent only as a description of each method's own
  successful-task coverage.
- Keep model/API errors out of plan-quality denominators.
- Declare any travel-data queries, verification helpers, retries, or
  deterministic repair used by each method.
- Archive the aggregate report and sample-level failure output.

## Code and field mapping

| Reader-facing measure | Repository identifier |
|---|---|
| Plan Validity | `level1` / Level 1 |
| Edit Correctness | `level2` / Level 2 |
| Editing Behavior | `level3` / Level 3 |
| supported hard request checks | `edit_logical_success` |
| supported preference checks | `edit_preference_success` |
| experimental change attribution | `level3.cascade` |

## Implementation references

- `scripts/evaluate_benchmark.py`
- `scripts/evaluate_edit_framework_run.py`
- `src/evaluation/benchmark/levels.py`
- `src/evaluation/benchmark/edit_adapters.py`
- `src/evaluation/benchmark/preference_scorers.py`
- `src/evaluation/benchmark/diffing.py`
- `src/evaluation/edit_scope.py`
- `src/evaluation/cascade_analysis.py`
- `src/evaluation/route_evidence.py`
- `src/evaluation/benchmark/level3.py`
- `src/evaluation/benchmark/reporting.py`
- `tests/test_benchmark_evaluation.py`
- `tests/test_soft_preference_scorers.py`
