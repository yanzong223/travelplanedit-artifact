#!/usr/bin/env python3
"""Run one standalone edit framework across categories and optionally evaluate each run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CHINATRAVEL_ROOT = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
for candidate in (PROJECT_ROOT, SRC_ROOT, CHINATRAVEL_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

FRAMEWORK_CHOICES = [
    "react",
    "reflexion",
    "pter",
]

TOOL_PROFILE_CHOICES = [
    "db_read_typed",
]

DEFAULT_BATCH_ROOT = PROJECT_ROOT / "data" / "batches" / "batch002" / "batches"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "edit_framework_runs" / "batch002_by_category_20"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a framework over each category directory under one batch root."
    )
    parser.add_argument("--framework", default="react", choices=FRAMEWORK_CHOICES)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional category directory names to run. Defaults to all non-empty categories.",
    )
    parser.add_argument(
        "--validation-split",
        choices=["passed", "failed", "rejected"],
        default=None,
        help=(
            "Use samples from <category>/_validation_split/<split> while keeping "
            "the parent category name for output layout."
        ),
    )
    parser.add_argument(
        "--split-source",
        choices=["validation", "curated", "direct"],
        default=None,
        help=(
            "Where category samples are read from. Defaults to validation when "
            "--validation-split is set, otherwise direct category directories."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum samples to run for each category.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run every discovered sample in each category instead of forwarding --limit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Shared experiment root passed to run_edit_framework.py.",
    )
    parser.add_argument(
        "--output-layout",
        choices=["nested", "flat"],
        default="nested",
        help="Forwarded to run_edit_framework.py. flat treats --output-dir as the exact experiment dir.",
    )
    parser.add_argument(
        "--prompt-ablation",
        type=str,
        default="original",
        choices=["original", "unified_contract", "no_defensive_rules"],
        help="Prompt ablation mode (forwarded to run_edit_framework.py).",
    )
    parser.add_argument(
        "--annotation-scaffold-level",
        choices=["none", "target_grounding", "conflict_localization", "preservation"],
        default="none",
        help="Staged oracle-annotation scaffold level (forwarded to run_edit_framework.py).",
    )
    parser.add_argument("--level", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--episodes-dir", type=Path)
    parser.add_argument("--base-plans-dir", type=Path)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Only run the framework, skip merge+benchmark evaluation.",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--max-reflections", type=int, default=2)
    parser.add_argument("--guard-retries", type=int, default=None)
    parser.add_argument(
        "--exposure-mode",
        choices=["primitive_only"],
        default="primitive_only",
    )
    parser.add_argument(
        "--tool-profile",
        choices=TOOL_PROFILE_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first category whose run or evaluation fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    return parser


def count_category_samples(category_dir: Path) -> int:
    sample_files = sorted(category_dir.glob("sample_*.json"))
    if not sample_files:
        sample_files = [
            path
            for path in sorted(category_dir.glob("*.json"))
            if path.name not in {"summary.json", "manifest.json", "regeneration_summary.json"}
        ]
    return len(sample_files)


def discover_category_dirs(
    batch_root: Path,
    *,
    selected_categories: set[str] | None = None,
    validation_split: str | None = None,
    split_source: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    categories: list[dict[str, Any]] = []
    skipped_empty: list[str] = []

    for path in sorted(batch_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if selected_categories is not None and name not in selected_categories:
            continue
        sample_path = path
        resolved_source = split_source or ("validation" if validation_split else "direct")
        if validation_split and resolved_source == "validation":
            sample_path = path / "_validation_split" / validation_split
        elif validation_split and resolved_source == "curated":
            sample_path = path / "_curated" / validation_split
        sample_count = count_category_samples(sample_path) if sample_path.exists() else 0
        if sample_count <= 0:
            skipped_empty.append(name)
            continue
        categories.append(
            {
                "name": name,
                "path": sample_path,
                "sample_count": sample_count,
            }
        )

    return categories, skipped_empty


def _build_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{current}" if current else str(SRC_ROOT)
    return env


def _build_logs_dir(output_dir: Path, framework: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / "_orchestration" / f"{framework}_{timestamp}"


def _category_output_dir(args: argparse.Namespace, category_name: str) -> Path:
    if getattr(args, "output_layout", "nested") == "flat":
        return Path(args.output_dir) / category_name
    return Path(args.output_dir)


def _collect_run_summaries(output_dir: Path) -> set[Path]:
    if not output_dir.exists():
        return set()
    return {path.resolve() for path in output_dir.rglob("run/summary.json")}


def _pick_new_summary(before: set[Path], after: set[Path]) -> Path | None:
    new_paths = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if new_paths:
        return new_paths[-1]
    return None


def _pick_run_summary(
    before: set[Path],
    after: set[Path],
    *,
    expected_run_summary: Path,
) -> Path | None:
    new_summary = _pick_new_summary(before, after)
    if new_summary is not None:
        return new_summary
    if expected_run_summary.exists():
        return expected_run_summary.resolve()
    if after:
        return sorted(after, key=lambda path: path.stat().st_mtime)[-1]
    return None


def build_run_command(
    args: argparse.Namespace,
    category_dir: Path,
    *,
    category_name: str | None = None,
) -> list[str]:
    output_category_name = category_name or category_dir.name
    run_output_dir = _category_output_dir(args, output_category_name)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "artifact_runner" / "run_edit_framework.py"),
        "--framework",
        args.framework,
        "--batch-dir",
        str(category_dir),
        "--output-dir",
        str(run_output_dir),
        "--output-layout",
        str(args.output_layout),
        "--max-steps",
        str(args.max_steps),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--max-reflections",
        str(args.max_reflections),
        "--exposure-mode",
        str(args.exposure_mode),
    ]
    if not getattr(args, "all_samples", False) and args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.tool_profile:
        cmd.extend(["--tool-profile", str(args.tool_profile)])
    if getattr(args, "guard_retries", None) is not None:
        cmd.extend(["--guard-retries", str(args.guard_retries)])

    if args.parallel:
        cmd.extend(["--parallel", "--max-workers", str(args.max_workers)])
    if args.prompt_ablation and args.prompt_ablation != "original":
        cmd.extend(["--prompt-ablation", str(args.prompt_ablation)])
    if args.annotation_scaffold_level and args.annotation_scaffold_level != "none":
        cmd.extend(["--annotation-scaffold-level", str(args.annotation_scaffold_level)])
    return cmd


def build_evaluate_command(
    args: argparse.Namespace,
    *,
    category_dir: Path,
    run_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "artifact_runner" / "evaluate_edit_framework_run.py"),
        "--run-dir",
        str(run_dir),
        "--samples-dir",
        str(category_dir),
        "--level",
        str(args.level),
    ]
    if args.episodes_dir:
        cmd.extend(["--episodes-dir", str(args.episodes_dir)])
    if args.base_plans_dir:
        cmd.extend(["--base-plans-dir", str(args.base_plans_dir)])
    return cmd


def _write_command_logs(
    log_dir: Path,
    prefix: str,
    *,
    cmd: list[str],
    completed: subprocess.CompletedProcess[str],
) -> None:
    (log_dir / f"{prefix}.cmd.txt").write_text(
        " ".join(cmd),
        encoding="utf-8",
    )
    (log_dir / f"{prefix}.stdout.log").write_text(
        completed.stdout or "",
        encoding="utf-8",
    )
    (log_dir / f"{prefix}.stderr.log").write_text(
        completed.stderr or "",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compact_report(report_path: Path | None) -> dict[str, Any] | None:
    if report_path is None or not report_path.exists():
        return None
    report = _load_json(report_path)
    summary = report.get("summary", {})
    return {
        "total_records": summary.get("total_records"),
        "level1_pass_rate": summary.get("level1_pass_rate"),
        "edit_logical_success_rate": summary.get("edit_logical_success_rate"),
        "edit_logical_evaluable_records": summary.get("edit_logical_evaluable_records"),
        "edit_preference_success_rate": summary.get("edit_preference_success_rate"),
        "edit_preference_evaluable_records": summary.get("edit_preference_evaluable_records"),
        "combined_edit_success_rate": summary.get("combined_edit_success_rate"),
        "combined_edit_evaluable_records": summary.get("combined_edit_evaluable_records"),
        "level3_evaluable_records": summary.get("level3_evaluable_records"),
    }


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=_build_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _status(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.all_samples:
        args.limit = None

    batch_root = args.batch_root.resolve()
    output_dir = args.output_dir.resolve()

    if not batch_root.exists():
        parser.error(f"--batch-root does not exist: {batch_root}")

    selected = set(args.categories) if args.categories else None
    categories, skipped_empty = discover_category_dirs(
        batch_root,
        selected_categories=selected,
        validation_split=args.validation_split,
        split_source=args.split_source,
    )

    if selected:
        discovered_names = {item["name"] for item in categories} | set(skipped_empty)
        missing = sorted(selected - discovered_names)
        if missing:
            parser.error(f"Unknown category names: {', '.join(missing)}")

    if not categories:
        parser.error(f"No non-empty category directories found under {batch_root}")

    logs_dir = _build_logs_dir(output_dir, args.framework)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _status(
        f"Discovered {len(categories)} non-empty categories under {batch_root}. "
        f"Logs will be written to {logs_dir}"
    )
    if skipped_empty:
        _status(f"Skipping empty categories: {', '.join(skipped_empty)}")

    summary_payload: dict[str, Any] = {
        "framework": args.framework,
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "limit_per_category": args.limit,
        "skip_evaluation": bool(args.skip_evaluation),
        "level": args.level,
        "parallel": bool(args.parallel),
        "max_workers": args.max_workers if args.parallel else 1,
        "exposure_mode": args.exposure_mode,
        "tool_profile": args.tool_profile,
        "output_layout": args.output_layout,
        "validation_split": args.validation_split,
        "split_source": args.split_source or ("validation" if args.validation_split else "direct"),
        "tool_flags": {"db_read_typed": args.tool_profile == "db_read_typed"},
        "skipped_empty_categories": skipped_empty,
        "categories": [],
        "logs_dir": str(logs_dir),
    }

    if args.dry_run:
        for category in categories:
            run_cmd = build_run_command(
                args,
                category["path"],
                category_name=category["name"],
            )
            summary_payload["categories"].append(
                {
                    "category": category["name"],
                    "sample_count": category["sample_count"],
                    "planned_run_command": run_cmd,
                }
            )
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        return 0

    total_categories = len(categories)
    for index, category in enumerate(categories, start=1):
        category_name = category["name"]
        category_dir = Path(category["path"])
        log_prefix = category_name
        category_summary: dict[str, Any] = {
            "category": category_name,
            "sample_count": category["sample_count"],
            "limit": args.limit,
            "run_status": "pending",
            "evaluation_status": "skipped" if args.skip_evaluation else "pending",
        }
        summary_payload["categories"].append(category_summary)
        _status(
            f"[{index}/{total_categories}] Running {category_name} "
            f"(available={category['sample_count']}, limit={args.limit})"
        )

        category_output_dir = _category_output_dir(args, category_name).resolve()
        before = _collect_run_summaries(category_output_dir)
        run_cmd = build_run_command(args, category_dir, category_name=category_name)
        run_completed = _run_command(run_cmd)
        _write_command_logs(
            logs_dir, f"{log_prefix}.run", cmd=run_cmd, completed=run_completed
        )

        run_summary_path = _pick_run_summary(
            before,
            _collect_run_summaries(category_output_dir),
            expected_run_summary=category_output_dir / "run" / "summary.json",
        )
        run_dir = run_summary_path.parent if run_summary_path is not None else None
        run_summary = (
            _load_json(run_summary_path) if run_summary_path is not None else None
        )

        category_summary.update(
            {
                "run_command": run_cmd,
                "run_returncode": run_completed.returncode,
                "run_dir": str(run_dir) if run_dir else None,
                "run_summary_path": str(run_summary_path) if run_summary_path else None,
                "run_stdout_log": str(logs_dir / f"{log_prefix}.run.stdout.log"),
                "run_stderr_log": str(logs_dir / f"{log_prefix}.run.stderr.log"),
                "run_status": "ok" if run_completed.returncode == 0 else "failed",
                "runner_summary": run_summary,
            }
        )
        _status(
            f"[{index}/{total_categories}] Run {category_name} finished "
            f"with return code {run_completed.returncode}. "
            f"Logs: {logs_dir / f'{log_prefix}.run.stdout.log'}"
        )

        if run_completed.returncode != 0:
            if args.stop_on_error:
                summary_path = logs_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(summary_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
                return 1
            continue

        if args.skip_evaluation:
            continue

        if run_dir is None:
            category_summary["evaluation_status"] = "failed"
            category_summary["evaluation_error"] = (
                "run completed but run_dir could not be inferred"
            )
            _status(
                f"[{index}/{total_categories}] Evaluation skipped for {category_name}: "
                "run_dir could not be inferred"
            )
            if args.stop_on_error:
                summary_path = logs_dir / "summary.json"
                summary_path.write_text(
                    json.dumps(summary_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
                return 1
            continue

        _status(f"[{index}/{total_categories}] Evaluating {category_name}")
        eval_cmd = build_evaluate_command(
            args, category_dir=category_dir, run_dir=run_dir
        )
        eval_completed = _run_command(eval_cmd)
        _write_command_logs(
            logs_dir, f"{log_prefix}.eval", cmd=eval_cmd, completed=eval_completed
        )

        report_path = run_dir.parent / "evaluation" / "benchmark_evaluation_report.json"
        category_summary.update(
            {
                "evaluation_command": eval_cmd,
                "evaluation_returncode": eval_completed.returncode,
                "evaluation_status": (
                    "ok" if eval_completed.returncode == 0 else "failed"
                ),
                "report_path": str(report_path) if report_path.exists() else None,
                "report_summary": _compact_report(report_path),
                "eval_stdout_log": str(logs_dir / f"{log_prefix}.eval.stdout.log"),
                "eval_stderr_log": str(logs_dir / f"{log_prefix}.eval.stderr.log"),
            }
        )
        _status(
            f"[{index}/{total_categories}] Evaluation {category_name} finished "
            f"with return code {eval_completed.returncode}. "
            f"Report: {report_path if report_path.exists() else 'not generated'}"
        )

        if eval_completed.returncode != 0 and args.stop_on_error:
            summary_path = logs_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
            return 1

    summary_payload["run_ok_count"] = sum(
        1 for item in summary_payload["categories"] if item.get("run_status") == "ok"
    )
    summary_payload["run_failed_count"] = sum(
        1
        for item in summary_payload["categories"]
        if item.get("run_status") == "failed"
    )
    summary_payload["evaluation_ok_count"] = sum(
        1
        for item in summary_payload["categories"]
        if item.get("evaluation_status") == "ok"
    )
    summary_payload["evaluation_failed_count"] = sum(
        1
        for item in summary_payload["categories"]
        if item.get("evaluation_status") == "failed"
    )

    summary_path = logs_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))

    if (
        summary_payload["run_failed_count"] > 0
        or summary_payload["evaluation_failed_count"] > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
