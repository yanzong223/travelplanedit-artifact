# TravelPlan Editing Benchmark Artifact

This repository contains the code, data, and analysis scripts used for the TravelPlan Editing benchmark experiments.

## Contents

```text
.
├── configs/                 # model aliases and batch-processing config
├── data/benchmark/          # validation-passed benchmark data used by the main experiments
├── data_generation/         # dataset construction pipeline
├── data_clean/              # validity rules used by generation/evaluation
├── data_classify/           # small soft-constraint taxonomy dependency for generation
├── src/                     # editing frameworks, LLM client, evaluators, and runtime tools
├── scripts/                 # one public experiment entrypoint
├── Chinatravel/             # trimmed ChinaTravel database/tools dependency
└── experiments/main_analysis/
```

The public entrypoint is:

```bash
PYTHONPATH=src uv run python scripts/run_batch_by_category.py --help
```

Internal runner modules live under `src/artifact_runner/`.

## Setup

Use Python 3.11.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Copy `.env.example` to `.env` and set your OpenAI-compatible endpoint credentials.

## Smoke Check

Preview the default benchmark run without calling the model:

```bash
PYTHONPATH=src uv run python scripts/run_batch_by_category.py \
  --model-alias deepseek_v4_pro_guan \
  --framework react \
  --batch-id benchmark \
  --tool-profile db_read_typed \
  --dry-run
```

Run a small experiment:

```bash
PYTHONPATH=src uv run python scripts/run_batch_by_category.py \
  --model-alias deepseek_v4_pro_guan \
  --framework react \
  --batch-id benchmark \
  --tool-profile db_read_typed \
  --parallel \
  --max-workers 4 \
  --level all
```

Outputs are written under `experiments/<model>/<timestamp>.../`.

For artifact and data documentation, see `ARTIFACT_DOCUMENTATION.md`.

## Data

The benchmark data is under:

```text
data/benchmark/<category>/sample_*.json
```

The included ChinaTravel dependency is trimmed to the package code, environment tools, database, and symbolic verification utilities needed by generation and evaluation. Caches, old model outputs, images, and local model files are intentionally excluded.

## Analysis

The paper-facing main table is under:

```text
experiments/main_analysis/
```
