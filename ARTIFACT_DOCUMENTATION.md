# Artifact Documentation

## Scope

This artifact contains the anonymized TravelPlan Editing benchmark data, the data-generation pipeline, the evaluation code, the public experiment entrypoint for reproducing the main table with the `db_read_typed` tool setting, and the offline evidence-grounded edit-scope analysis code.

## Public Entrypoint

```bash
PYTHONPATH=src uv run python scripts/run_batch_by_category.py --help
```

The public entrypoint dispatches to internal runner modules under `src/artifact_runner/`. The retained model-side tool setting is `--tool-profile db_read_typed`.

## Benchmark Data

The benchmark split included in this artifact is:

```text
data/benchmark/<category>/sample_*.json
```

It contains the validation-passed benchmark samples used by the main experiments. Each sample stores the original travel plan, original user request, edit request, logical/preference constraints, metadata, and evaluation-facing fields needed by the benchmark evaluator.

## Data-Generation Pipeline

The retained generation scripts are:

```text
data_generation/scripts/run_pipeline.py
data_generation/scripts/01_sample_bucket.py
data_generation/scripts/02_generate_query.py
data_generation/scripts/03_freeze_edit_truth.py
data_generation/scripts/04_rewrite_query_surface.py
data_generation/scripts/05_analyze_conflict.py
data_generation/scripts/06_validate.py
```

Historical strategy-generation, mock-plan, diff-extraction, audit, backfill, repair, and smoke-test scripts are omitted because they were not part of the final public workflow.

## Auxiliary Data Modules

`data_clean/` provides validity and rule checks used by both generation and evaluation.

`data_classify/` is retained because `data_generation/utils/soft_constraint_control.py` imports its soft-constraint taxonomy heuristics during Step 1/2. It is not a separate experiment entrypoint.

## Main Analysis

`experiments/main_analysis/` contains only the main paper table:

```text
experiments/main_analysis/tables/main_results_table.csv
experiments/main_analysis/tables/main_results_table.md
```

Plotting scripts, rendered figures, raw per-run reports, and intermediate rollups are intentionally omitted.

## Edit-Scope Analysis

The stable attribution API is `src/evaluation/edit_scope.py`; the implementation engine is `src/evaluation/cascade_analysis.py`. Versioned route evidence and manifest helpers live next to them under `src/evaluation/`.

The public offline entrypoints are:

```text
scripts/analyze_natural_feasible_187_edit_scope.py
scripts/precompute_matrix_cascade_route_evidence.py
scripts/recompute_matrix_reports_offline.py
scripts/validate_itimo_request_satisfaction_portability.py
```

The corresponding regression tests are under `tests/`. Frozen private study submissions, model-provider outputs, and local route caches are intentionally excluded; the scripts accept those inputs by path when available.

## Anonymity

The repository is prepared for anonymous review. It avoids local absolute paths, usernames, institution-identifying strings, private remotes, API keys, and author-identifying metadata in tracked files.
