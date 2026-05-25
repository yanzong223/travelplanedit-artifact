# Three-Level Benchmark Evaluation Design

This document records the current implementation of the benchmark evaluator for
ChinaTravel-format travel plan editing. It describes the mathematical target of
each level and the concrete code structure used in the repository.

## Overview

The evaluator is organized into three levels:

1. `Level1`: validate whether the edited plan is still a valid travel plan and
   whether untargeted origin hard constraints are preserved.
2. `Level2`: validate whether the edit request is actually satisfied, for both
   logical targets and soft/preference targets.
3. `Level3`: measure whether the model solved the edit with minimal changes.

Current code entrypoints:

- [src/evaluation/benchmark/levels.py](./src/evaluation/benchmark/levels.py)
- [src/evaluation/benchmark/origin_adapters.py](./src/evaluation/benchmark/origin_adapters.py)
- [src/evaluation/benchmark/edit_adapters.py](./src/evaluation/benchmark/edit_adapters.py)
- [src/evaluation/benchmark/verifiers.py](./src/evaluation/benchmark/verifiers.py)
- [src/evaluation/benchmark/preference_scorers.py](./src/evaluation/benchmark/preference_scorers.py)
- [src/evaluation/benchmark/diffing.py](./src/evaluation/benchmark/diffing.py)
- [src/evaluation/benchmark/level3.py](./src/evaluation/benchmark/level3.py)
- [src/evaluation/benchmark/reporting.py](./src/evaluation/benchmark/reporting.py)
- [scripts/evaluate_benchmark.py](./scripts/evaluate_benchmark.py)

The evaluator consumes ChinaTravel-style records with these core fields:

- `original_plan` or `origin_plan`
- `edited_plan`
- `origin_query` or `origin_query_structured`
- `edit_query`
- optional explicit labels:
  - `edit_target_constraints`
  - `edit_target_preferences`

## Shared Data Model

The evaluator standardizes constraints into two object families:

### Logical Constraints

Implemented in [src/evaluation/benchmark/models.py](./src/evaluation/benchmark/models.py):

```python
LogicalConstraintObject(
    id,
    source,
    type,
    scope,
    target,
    operator,
    value,
    params,
    is_hard,
    provenance,
)
```

These represent hard or executable logical conditions such as:

- `day_count`
- `people_count`
- `budget_total`
- `required_attraction_name`
- `required_innercity_transport_type`
- `required_room_count`

### Preference Constraints

```python
PreferenceConstraintObject(
    id,
    source,
    family,
    facet,
    direction,
    anchor,
    edit_mode,
    params,
    provenance,
)
```

These represent score-based soft targets such as:

- `experience_richness / daily_attractions_maximize`
- `route_compactness / transport_time_minimize`
- `cost_allocation_preference / hotel_cost_minimize`
- `anchor_proximity / distance_to_poi_minimize`

### Adapter Flow

- `origin_query -> adapt_origin_query()`:
  - parse `hard_logic_py` into `LogicalConstraintObject`
  - parse `preference_en/preference_py` into `PreferenceConstraintObject`
- `edit_record -> adapt_edit_record()`:
  - read explicit `edit_target_constraints`
  - read explicit `edit_target_preferences`
  - do not backfill from `edit_target_preference_tags`,
    `query_generation_trace`, or natural language heuristics at runtime

## Level1: Plan Validity

### Goal

`Level1` answers:

> Is the edited plan a valid travel plan, and did it preserve untargeted origin
> hard constraints?

It does not judge whether the edit request itself was achieved.

### Mathematical Definition

Let:

- origin query be `Q_o`
- edit query be `Q_e`
- edited plan be `P'`

Define plan feasibility:

$$
\mathrm{Feasible}(P') \in \{0,1\}
$$

and origin hard constraint extraction:

$$
C_o^{hard} = \mathrm{ExtractHard}(Q_o)
$$

Let the subset explicitly targeted by the edit be:

$$
C_{target} = \mathrm{Targeted}(Q_e, C_o^{hard})
$$

Then the constraints that must still hold are:

$$
C_{preserve} = C_o^{hard} \setminus C_{target}
$$

For each preserved constraint `c`, define:

$$
\mathrm{sat}(P', c) \in \{0,1\}
$$

The strict Level1 hard preservation pass is:

$$
\mathrm{HardPreserve}(P') =
\mathbf{1}\Big[\forall c \in C_{preserve}, \mathrm{sat}(P', c)=1\Big]
$$

Current `Level1` pass is:

$$
\mathrm{Level1Pass}(P') =
\mathrm{Feasible}(P') \land \mathrm{HardPreserve}(P')
$$

### Implementation

`Level1` is implemented in
[src/evaluation/benchmark/levels.py](./src/evaluation/benchmark/levels.py).

It has two parts:

1. `feasibility`
   - implemented by `data_clean.rules.evaluate_plan`
   - current pass criterion is `hygiene_pass and quality_pass`
   - both pass flags are still preserved in diagnostics for debugging
2. `origin_logical_preservation`
   - adapt origin query into logical constraints
   - remove edit-targeted constraints by `_is_targeted(...)`
   - verify each remaining constraint with `verify_constraints(...)`

The returned structure is:

```json
{
  "pass": true,
  "feasibility": {...},
  "origin_logical_preservation": {...},
  "diagnostics": {...}
}
```

### Verification Strategy

Constraint verification is handled in
[src/evaluation/benchmark/verifiers.py](./src/evaluation/benchmark/verifiers.py).

The verifier uses a two-stage strategy:

1. direct structured verification
2. symbolic `hard_logic_py` fallback

This design avoids making benchmark correctness depend on optional ChinaTravel
runtime dependencies when a structured check is already available.

## Level2: Edit Success

### Goal

`Level2` answers:

> Did the edited plan satisfy the edit request?

It evaluates both:

- logical edit targets
- soft/preference edit targets

### Mathematical Definition

Let:

- origin plan be `P`
- edited plan be `P'`
- explicit edit targets be:

$$
C_e^{hard} = \mathrm{EditLogicalTargets}(Q_e)
$$

$$
C_e^{soft} = \mathrm{EditPreferenceTargets}(Q_e)
$$

For logical targets:

$$
\mathrm{LogicalPass}(P') =
\mathbf{1}\Big[\forall c \in C_e^{hard}, \mathrm{sat}(P', c)=1\Big]
$$

For preference targets, let a scorer be:

$$
\mathrm{score}(P, c) \in [0,1]
$$

For `addition` mode:

$$
\mathrm{Pass}_{add}(P', c) =
\mathbf{1}\big[\mathrm{score}(P', c) \ge \tau_c\big]
$$

For `strengthen` mode:

$$
\Delta(P, P', c) = \mathrm{score}(P', c) - \mathrm{score}(P, c)
$$

$$
\mathrm{Pass}_{str}(P, P', c) =
\mathbf{1}\big[\Delta(P, P', c) \ge \epsilon\big]
$$

The current combined Level2 pass is:

$$
\mathrm{Level2Pass}(P, P') =
\mathrm{LogicalPass}(P') \land \mathrm{PreferencePass}(P, P')
$$

with an additional implementation rule:

- there must be at least one supported logical or preference target to make the
  sample evaluable

### Implementation

Implemented in
[src/evaluation/benchmark/levels.py](./src/evaluation/benchmark/levels.py).

#### Logical Edit Success

- edit targets come from `adapt_edit_record()`
- each logical target is verified with `verify_constraints(...)`
- `compiled_hard_logic_py` is emitted for compatibility/debugging, not as the
  primary source of truth

#### Preference Edit Success

Implemented by
[src/evaluation/benchmark/preference_scorers.py](./src/evaluation/benchmark/preference_scorers.py).

Currently implemented preference facets:

- `daily_attractions_maximize`
- `transport_time_minimize`
- `restaurant_transport_time_minimize`
- `food_cost_ratio_maximize`
- `hotel_cost_minimize`
- `distance_to_poi_minimize`

## Acceptance Matrix

This matrix is the compact contract for the benchmark surface we care about in
practice. Each row maps a natural-language edit request to the explicit object
family that should be materialized, the evaluator component that checks it, and
the expected outcome when the edited plan matches or violates the request.

| Aspect | Example query surface | Materialized object(s) | Verifier / scorer | Expected |
| --- | --- | --- | --- | --- |
| 时间 | `景点之间交通时间不超过 45 分钟` / `黄鹤楼最好上午去` | `adjacent_travel_time_cap`, `poi_clock_time_window`, `transport_time_window` | `verify_constraints(...)` | Match => pass; violate travel duration or time window => fail |
| 空间 | `围绕西湖安排活动，周边 5 公里内` / `把两景点之间距离控制在 3 公里` | `anchor_neighbor_commute_distance_cap`, `adjacent_travel_distance_cap`, `pairwise_transport_mode_distance_cap` | `verify_constraints(...)` | Match => pass; anchor radius or pairwise distance exceeded => fail |
| 主题 / vibe | `想要更偏文化主题` / `行程氛围轻松一点、晚上热闹一点` | `theme_alignment`, `vibe_alignment` | `score_preference_baseline(...)` via `edit_target_preferences` | Match => score meets threshold or improves enough for strengthen; otherwise fail |
| 预算 | `总预算不超过 2000 元` / `某景点+附近一餐控制在 300 元内` | `budget_total`, `ticket_budget_total`, `activity_budget_limit`, `anchor_bundle_budget_limit` | `verify_constraints(...)` | Match => pass; budget cap exceeded => fail |
| 天数 / 人数 | `改成 3 天游玩` / `同行改为 4 人` | `day_count`, `people_count` | `verify_constraints(...)` | Match => pass; day or people count mismatch => fail |
| 基本合理性 | `只要行程还能正常执行` | `evaluate_plan()` -> `hygiene_pass` + `quality_pass` | `Level1` feasibility gate | Both pass => Level1 may pass; either one fails => Level1 fails |

Two practical notes:

1. `Level2` only evaluates explicit edit truth objects or the explicit
   `edit_target_preference_tags -> edit_target_preferences` materialization path.
   It does not infer targets from raw edit text at evaluation time.
2. `diagnostics` are retained for debugging and reporting, but they do not
   decide pass/fail.

Current thresholds are static:

- addition thresholds come from `_ADDITION_THRESHOLDS`
- strengthen threshold comes from `_DELTA_THRESHOLD`

### Output

```json
{
  "pass": true,
  "edit_logical_success": {...},
  "edit_preference_success": {...}
}
```

## Level3: Edit Efficiency / Minimal Edit

### Goal

`Level3` answers:

> Assuming the model already produced a valid and successful edit, how small was
> the change?

`Level3` is only evaluated when both `Level1` and `Level2` pass.

### Mathematical Definition

Let:

- origin plan be `P`
- edited plan be `P'`
- inferred edit sequence be:

$$
\hat{E} = \mathrm{InferEditSequence}(P, P')
$$

The atomic operation vocabulary is:

$$
\Omega =
\{
\text{change\_time},
\text{insert},
\text{delete},
\text{replace},
\text{reorder}
\}
$$

We count:

$$
n_{time}(\hat{E}),\;
n_{ins}(\hat{E}),\;
n_{del}(\hat{E}),\;
n_{rep}(\hat{E}),\;
n_{reo}(\hat{E})
$$

Parameter and structural counts are:

$$
m_{param}(\hat{E}) = n_{time}(\hat{E})
$$

$$
m_{struct}(\hat{E}) =
n_{ins}(\hat{E}) +
n_{del}(\hat{E}) +
n_{rep}(\hat{E}) +
n_{reo}(\hat{E})
$$

The highest scope level is:

$$
s(\hat{E}) =
\begin{cases}
0 & \text{parameter-only edits} \\
1 & \text{single-day / local structural edits} \\
2 & \text{cross-day or multi-day compositional edits}
\end{cases}
$$

The benchmark comparison key is the lexicographic tuple:

$$
\mathrm{cost}(\hat{E}) =
\big(s(\hat{E}), m_{struct}(\hat{E}), m_{param}(\hat{E})\big)
$$

This means:

1. lower scope is better
2. for the same scope, fewer structural edits are better
3. for the same scope and structural count, fewer parameter edits are better

### Auxiliary Diagnostics

`Level3` also returns two diagnostic metrics.

#### Content Retention

Let `Retained(P, P')` be the set of origin activities matched as the same
activity after editing, excluding `replace`.

$$
\mathrm{Retention}(P, P') =
\frac{|Retained(P, P')|}{|A(P)|}
$$

Current implementation counts `change_time` and `reorder` as retained, but not
`replace`.

#### POI Sequence Edit Distance

Let `Seq(P)` be the sequence of activity tokens extracted day by day from the
plan. Then:

$$
\mathrm{MED}(P, P') =
\mathrm{LevenshteinDistance}(\mathrm{Seq}(P), \mathrm{Seq}(P'))
$$

#### Activity Change Ratio

$$
\mathrm{ChangeRatio}(P, P') =
\frac{\mathrm{changed\_activities}}{|A(P)|}
$$

This is used as a rewrite tendency diagnostic.

### Matching Strategy

Implemented in
[src/evaluation/benchmark/diffing.py](./src/evaluation/benchmark/diffing.py).

The matcher follows the current identity-first order:

1. exact same activity -> `unchanged`
2. same activity identity, same day, only time change -> `change_time`
3. same activity identity, different day -> compositional `reorder`
4. same day, same type, same slot, different POI -> `replace`
5. same matched set but different order -> `reorder`
6. unmatched origin -> `delete`
7. unmatched edited -> `insert`
8. any cross-day structural effect or day-count change upgrades scope to
   `compositional`

Activity identity is currently derived as:

- attractions / meals / accommodation: `type + position`
- train / airplane: `type + start + end + TrainID`
- fallback activities: `type + start + end`

This is intentionally independent of internal audit labels or strategy traces.

### Eligibility Rule

Implemented in
[src/evaluation/benchmark/level3.py](./src/evaluation/benchmark/level3.py).

`Level3` only runs when:

$$
\mathrm{Level1Pass}(P') = 1
\quad \land \quad
\mathrm{Level2Pass}(P, P') = 1
$$

Otherwise:

- `eligible = false`
- `reason = "requires_level1_and_level2_pass"`

### Output

Current `Level3` result:

```json
{
  "eligible": true,
  "reason": "ok",
  "scope_level": 1,
  "scope_name": "structural",
  "parameter_count": 1,
  "structural_count": 2,
  "compositional_count": 0,
  "atomic_counts": {...},
  "edit_cost_tuple": [1, 2, 1],
  "content_retention_rate": 0.75,
  "poi_seq_edit_distance": 2,
  "activity_change_ratio": 0.5,
  "matched_pairs": [...],
  "unmatched_origin": [...],
  "unmatched_edited": [...]
}
```

## Reporting and CLI

Dataset-level aggregation is implemented in
[src/evaluation/benchmark/reporting.py](./src/evaluation/benchmark/reporting.py).

Current summary fields include:

- `feasibility_pass_rate`
- `origin_logical_preservation_rate`
- `level1_pass_rate`
- `edit_logical_success_rate`
- `edit_preference_success_rate`
- `combined_edit_success_rate`
- `level3_evaluable_records`
- `avg_parameter_count`
- `avg_structural_count`
- `avg_compositional_count`
- `avg_content_retention_rate`
- `avg_poi_seq_edit_distance`
- `avg_activity_change_ratio`

The CLI is
[scripts/evaluate_benchmark.py](./scripts/evaluate_benchmark.py).

Supported modes:

- `--level 1`
- `--level 2`
- `--level 3`
- `--level all`

Example:

```bash
uv run python scripts/evaluate_benchmark.py \
  --results-dir results/batch_results \
  --base-plans-dir data/tpe_dataset/base_plans \
  --level all
```

## Current Design Choices

Important implementation choices in the current system:

- benchmark truth is object-based, not string-code-based
- structured verification is preferred over symbolic execution whenever
  possible
- `Level2` soft evaluation is score-based and parameterized by
  `family/facet/edit_mode`
- `Level3` does not use runtime cost, token cost, or internal audit metadata
- `Level3` uses inferred edit sequences, not real execution logs
- `edit_cost_tuple` is the primary comparison key for minimal-edit analysis

## Limitations

The current implementation is intentionally pragmatic and has known limits:

- `Level2` soft thresholds are static and not yet dataset-calibrated
- `Level3` infers a plausible edit sequence, not the true minimum program
- activity identity relies on plan fields currently present in ChinaTravel
  plans; unusual schemas may reduce matching quality
- `replace` and `reorder` are heuristic recognizers, not globally optimal graph
  alignment

These tradeoffs are acceptable for the current benchmark version because the
system prioritizes reproducibility, public evaluability, and compatibility with
ChinaTravel-format plan JSON.
