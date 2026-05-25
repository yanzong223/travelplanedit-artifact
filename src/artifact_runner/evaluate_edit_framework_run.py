#!/usr/bin/env python3
"""Merge a standalone edit-framework run and evaluate it in one command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from edit_framework.merge_results import merge_results_directory
from edit_framework.run_layout import resolve_default_evaluation_dir
from evaluation.benchmark import BenchmarkEvaluator, aggregate_results, attach_report_views


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_result_files(results_dir: Path) -> list[Path]:
    ignored_names = {
        "summary.json",
        "merge_summary.json",
    }
    return sorted(
        file_path
        for file_path in results_dir.rglob("*.json")
        if file_path.name not in ignored_names
        and not file_path.name.startswith("benchmark_evaluation_report")
    )


def _load_episode_map(episodes_dir: Path | None) -> dict[str, dict[str, Any]]:
    if not episodes_dir or not episodes_dir.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for file_path in episodes_dir.rglob("*.json"):
        try:
            payload = _load_json(file_path)
        except Exception:
            continue
        episode_id = payload.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            mapping[episode_id] = payload
    return mapping


def _load_base_query(base_plans_dir: Path | None, plan_id: str | None) -> dict[str, Any] | None:
    if not base_plans_dir or not plan_id:
        return None
    query_file = base_plans_dir / plan_id / "ct_query.json"
    if not query_file.exists():
        return None
    return _load_json(query_file)


def _load_base_plan(base_plans_dir: Path | None, plan_id: str | None) -> dict[str, Any] | None:
    if not base_plans_dir or not plan_id:
        return None
    plan_file = base_plans_dir / plan_id / "ct_plan.json"
    if not plan_file.exists():
        return None
    return _load_json(plan_file)


def _enrich_record(
    record: dict[str, Any],
    episode_map: dict[str, dict[str, Any]],
    base_plans_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    merged = dict(record)
    episode = episode_map.get(str(record.get("episode_id", "")), {})
    if episode:
        for key, value in episode.items():
            merged.setdefault(key, value)
    plan_id = merged.get("plan_id")
    origin_query = (
        merged.get("origin_query")
        or merged.get("origin_query_structured")
        or _load_base_query(base_plans_dir, plan_id)
    )
    if "original_plan" not in merged:
        base_plan = _load_base_plan(base_plans_dir, plan_id)
        if base_plan is not None:
            merged["original_plan"] = base_plan
    return merged, origin_query


def _infer_samples_dir(run_dir: Path) -> Path | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
    except Exception:
        return None
    batch_dir = summary.get("batch_dir")
    if not batch_dir:
        return None
    path = Path(batch_dir)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _default_merged_dir(run_dir: Path) -> Path:
    return resolve_default_evaluation_dir(PROJECT_ROOT, run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and evaluate a standalone edit-framework run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Standalone run directory, e.g. results/edit_framework_runs/<framework>/<model>/<timestamp>/run",
    )
    parser.add_argument("--samples-dir", type=Path, help="Batch sample directory. Defaults to batch_dir from run summary.json.")
    parser.add_argument("--merged-dir", type=Path, help="Directory for merged eval-ready files.")
    parser.add_argument("--episodes-dir", type=Path, help="Optional episode directory for enrichment.")
    parser.add_argument("--base-plans-dir", type=Path, help="Optional base plans directory containing ct_plan.json and ct_query.json.")
    parser.add_argument("--level", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--output", type=Path, help="Evaluation report path. Defaults to <merged-dir>/benchmark_evaluation_report.json.")
    parser.add_argument(
        "--emit-experiment-view",
        action="store_true",
        help="Write experiment_view JSON next to the main report.",
    )
    parser.add_argument(
        "--emit-error-summary",
        action="store_true",
        help="Write failure_summary JSON next to the main report.",
    )
    parser.add_argument(
        "--emit-sample-failures",
        action="store_true",
        help="Write sample_failures JSON next to the main report.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    samples_dir = (args.samples_dir.resolve() if args.samples_dir else _infer_samples_dir(run_dir))
    if samples_dir is None:
        parser.error("Could not infer --samples-dir from run summary.json; please pass --samples-dir explicitly.")

    merged_dir = args.merged_dir.resolve() if args.merged_dir else _default_merged_dir(run_dir)
    merge_summary = merge_results_directory(
        samples_dir=samples_dir,
        results_dir=run_dir,
        output_dir=merged_dir,
    )

    files = _discover_result_files(merged_dir)
    if args.max_results is not None:
        files = files[: args.max_results]

    evaluator = BenchmarkEvaluator(level=args.level)
    episode_map = _load_episode_map(args.episodes_dir.resolve() if args.episodes_dir else None)
    base_plans_dir = args.base_plans_dir.resolve() if args.base_plans_dir else None

    results = []
    evaluation_failures = []
    for file_path in files:
        record = _load_json(file_path)
        enriched, origin_query = _enrich_record(record, episode_map, base_plans_dir)
        try:
            results.append(evaluator.evaluate(enriched, origin_query=origin_query))
        except Exception as exc:
            evaluation_failures.append(
                {
                    "file_path": str(file_path),
                    "record_id": str(
                        enriched.get("episode_id")
                        or enriched.get("sample_id")
                        or enriched.get("plan_id")
                        or file_path.stem
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    report = aggregate_results(results)
    report["evaluation_failures"] = evaluation_failures
    report["evaluation_failure_count"] = len(evaluation_failures)
    report = attach_report_views(report)
    output_path = args.output.resolve() if args.output else merged_dir / "benchmark_evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.emit_experiment_view:
        experiment_view_path = output_path.with_name(f"{output_path.stem}_experiment_view.json")
        with experiment_view_path.open("w", encoding="utf-8") as handle:
            json.dump(report.get("experiment_view", {}), handle, ensure_ascii=False, indent=2)
    if args.emit_error_summary:
        error_summary_path = output_path.with_name(f"{output_path.stem}_error_summary.json")
        with error_summary_path.open("w", encoding="utf-8") as handle:
            json.dump(report.get("failure_summary", []), handle, ensure_ascii=False, indent=2)
    if args.emit_sample_failures:
        sample_failures_path = output_path.with_name(f"{output_path.stem}_sample_failures.json")
        with sample_failures_path.open("w", encoding="utf-8") as handle:
            json.dump(report.get("sample_failures", []), handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "samples_dir": str(samples_dir),
                "merged_dir": str(merged_dir),
                "report_path": str(output_path),
                "merged_count": merge_summary.get("merged_count", 0),
                "evaluated_count": len(results),
                "failed_count": len(evaluation_failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
