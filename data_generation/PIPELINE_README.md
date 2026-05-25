# Data Generation Pipeline

`data_generation/scripts/run_pipeline.py` is the public orchestrator for the dataset construction pipeline. It runs the retained step scripts in order and supports checkpoint-based resume.

## Retained Flow

```text
run_pipeline.py
  -> 01_sample_bucket.py
  -> 02_generate_query.py
  -> 03_freeze_edit_truth.py
  -> 04_rewrite_query_surface.py
  -> 05_analyze_conflict.py
  -> 06_validate.py
```

This artifact keeps the single flow used for the published benchmark data. The earlier strategy generation, mock-plan generation, and diff extraction steps are omitted.

## Example

```bash
PYTHONPATH=src uv run python data_generation/scripts/run_pipeline.py \
  --input /path/to/origin_plans \
  --query-input /path/to/origin_queries \
  --output data/batches/example_batch \
  --batch-size 100
```

Use `--resume` with the same output directory to continue from the saved checkpoint.

## Notes

Development-only scripts for auditing, backfilling, repairing, final benchmark curation, and API smoke tests are intentionally omitted from this artifact. The published benchmark data is already provided under `data/benchmark/`.
