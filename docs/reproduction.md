# Reproduction

Run commands from the repository root unless a section changes directory.

## Install

The project requires Python 3.11 or newer and includes a locked dependency
file.

```bash
uv sync --dev
uv run python --version
```

Keep provider credentials in an untracked `.env` file. Model-backed commands
use one of these sets:

```text
DMXAPI_API_KEY, DMXAPI_BASE_URL, DMXAPI_MODEL
SILICONCLOUD_API_KEY, SILICONCLOUD_BASE_URL, SILICONCLOUD_MODEL
```

Never include keys in logs, reports, or commits.

The multi-key pool in `pllm.yaml` stores environment-variable names only.
Set `SILICONCLOUD_API_KEY_1` through `SILICONCLOUD_API_KEY_10` for the
corresponding enabled pool entries. Entries whose variables are unset are
skipped. The single-provider commands continue to use
`SILICONCLOUD_API_KEY` or `DMXAPI_API_KEY`.

## Inspect current command options

These five help commands are the source of truth for accepted options:

```bash
uv run python data_generation/scripts/run_pipeline.py --help
uv run python data_generation/scripts/export_benchmark_views.py --help
uv run python scripts/run_edit_framework.py --help
uv run python scripts/evaluate_edit_framework_run.py --help
uv run python scripts/evaluate_benchmark.py --help
```

## Construct benchmark examples

```bash
uv run python data_generation/scripts/run_pipeline.py \
  --input path/to/original_itineraries \
  --query-input path/to/original_requests \
  --output data/batches/my_batch \
  --batch-size 10 \
  --sample-mode core \
  --model dmxapi \
  --step2-seed 42
```

The current code runs construction steps `1,2,3,4,5,9` in `core` mode and all
nine steps in `full` mode. The shorter description printed by `--help` for
`core` is a known documentation bug in the command itself.

Resume with the same request source and mode:

```bash
uv run python data_generation/scripts/run_pipeline.py \
  --query-input path/to/original_requests \
  --output data/batches/my_batch \
  --sample-mode core \
  --resume
```

Export public and private review copies:

```bash
uv run python data_generation/scripts/export_benchmark_views.py \
  --input data/batches/my_batch \
  --output-root data/exports/my_batch
```

## Run an editing method

One example:

```bash
PYTHONPATH=src uv run python scripts/run_edit_framework.py \
  --framework react \
  --sample-path path/to/sample.json \
  --output-dir results/edit_framework_runs
```

A directory of accepted examples:

```bash
PYTHONPATH=src uv run python scripts/run_edit_framework.py \
  --framework reflexion \
  --batch-dir path/to/passed_examples \
  --limit 10 \
  --output-dir results/edit_framework_runs
```

The default result layout separates method, model, settings, time, and the
final `run` directory. Preserve `summary.json` with the per-example results.

## Join and evaluate results

```bash
PYTHONPATH=src uv run python scripts/evaluate_edit_framework_run.py \
  --run-dir path/to/run \
  --samples-dir path/to/source_examples \
  --level all \
  --emit-experiment-view \
  --emit-error-summary \
  --emit-sample-failures
```

For records that are already joined with their source examples:

```bash
PYTHONPATH=src uv run python scripts/evaluate_benchmark.py \
  --results-dir path/to/results \
  --level all \
  --output path/to/benchmark_evaluation_report.json
```

## Recompute the current 187-task paper reports

The frozen task/result mapping is:

```text
human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/
  manifest.json
  experiment_results/main_three_methods_four_models/
    manifest_natural_feasible_187.json
```

Precompute the versioned route evidence needed by rollback families, without
model API calls:

```bash
PYTHONPATH=src:scripts uv run python scripts/precompute_matrix_cascade_route_evidence.py \
  --manifest human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/main_three_methods_four_models/manifest_natural_feasible_187.json \
  --reports-dir human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/reports_natural_feasible_187_fieldscope_multiboundary_irfix_20260815_v5 \
  --seed-cache human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/main_three_methods_four_models/cascade_route_evidence_fieldscope_multiboundary_irfix_20260815_v5.json \
  --output human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/main_three_methods_four_models/cascade_route_evidence_dcca_v1_1_20260818_v2.json
```

Recompute all 12 aggregate reports and full proof artifacts offline:

```bash
PYTHONPATH=src uv run python scripts/recompute_matrix_reports_offline.py \
  --manifest human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/main_three_methods_four_models/manifest_natural_feasible_187.json \
  --output-dir human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/reports_natural_feasible_187_dcca_v1_1_20260818_v2 \
  --compute-cascade \
  --route-evidence-cache human_baseline_data/batch006_four_types_natural_feasible_187_20260803_v1/experiment_results/main_three_methods_four_models/cascade_route_evidence_dcca_v1_1_20260818_v2.json
```

The final directory contains 12 benchmark reports and 12 line-addressable
`*_edit_scope_proofs.jsonl` files with saved witnesses and tested-family
evidence. Write future recomputes to a new versioned directory rather than
overwriting these artifacts.

The matrix audit requires 187 identical task IDs in each of four models by
three methods. Historical pre-supplement coverage files under
`experiment_results/audits/` are provenance records, not current results.

## Review application

Start the API:

```bash
PYTHONPATH=apps/reviewer-backend \
  uv run uvicorn reviewer_backend.main:app \
  --host 127.0.0.1 --port 8000
```

Start the interface in a second terminal:

```bash
cd apps/reviewer-frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5174`. Set `VITE_API_BASE` if the API uses another
address.

## Offline checks

```bash
PYTHONPATH=src:apps/reviewer-backend uv run pytest -q
cd apps/reviewer-frontend && npm test
```

Passing offline tests does not validate external model providers or local
ChinaTravel data.

## Experiment record

Archive:

- Git revision and whether the working tree was clean;
- data export identifier and checksums;
- Python version and dependency lock;
- model provider/name, random seed, temperature, and retry limits;
- editing method, prompt variant, available data/tools, and concurrency;
- numbers discovered, completed, joined, evaluated, supported, and eligible;
- aggregate reports and sample-level failures.

Known repository issue: the `tpe` console script declared in `pyproject.toml`
points to a missing module. Use the explicit Python commands above.
