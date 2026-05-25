#!/usr/bin/env python3
"""Offline runner for standalone edit baselines."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "src"
chinatravel_root = project_root / "Chinatravel" / "ChinaTravel"
for candidate in (src_root, project_root, chinatravel_root):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from edit_framework import create_edit_framework
from edit_framework.adapters import collect_sample_paths, load_edit_input_from_sample
from edit_framework.error_handling import (
    classify_terminal_error_categories,
    framework_error_handling_metadata,
)
from edit_framework.run_layout import resolve_experiment_dir, resolve_run_dir
from edit_framework.run_layout import build_tool_config_label
from edit_framework.runtime_tools.types import ExposureMode
from edit_framework.tool_profiles import (
    SUPPORTED_TOOL_PROFILES,
    TOOL_PROFILE_DB_READ_TYPED,
    resolve_tool_profile,
)
from llm.client import SiliconCloudClient
from tqdm import tqdm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        required=True,
        choices=[
            "react",
            "reflexion",
            "pter",
        ],
    )
    parser.add_argument("--sample-path")
    parser.add_argument("--batch-dir")
    parser.add_argument(
        "--latest-batch",
        action="store_true",
        help="Use the most recently modified directory under data/batches",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-dir",
        help="Base directory for experiment folders. Runs are saved under <framework>/<model>/<timestamp>/run.",
    )
    parser.add_argument(
        "--output-layout",
        choices=["nested", "flat"],
        default="nested",
        help="nested: <output>/<framework>/<model>/<timestamp>/run; flat: <output>/run.",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--max-reflections", type=int, default=2)
    parser.add_argument(
        "--guard-retries",
        type=int,
        default=None,
        help="Candidate-plan guard retry budget. Defaults to 0 for the artifact db_read_typed setting.",
    )
    parser.add_argument(
        "--reflection-strategy",
        choices=["reflexion", "last_attempt"],
        default="reflexion",
        help="Reflexion retry style: explicit verbal reflection memory or direct LAST_ATTEMPT-style feedback.",
    )
    parser.add_argument(
        "--context-prompt",
        action="store_true",
        help="Inject structured sample/constraint/plan context into ReAct and Reflexion prompts.",
    )
    parser.add_argument(
        "--database-prompt",
        action="store_true",
        help="Inject precomputed database/reference context from sample metadata when available.",
    )
    parser.add_argument(
        "--annotation-scaffold-level",
        choices=["none", "target_grounding", "conflict_localization", "preservation"],
        default="none",
        help=(
            "Staged oracle-annotation scaffold level for diagnostic experiments. "
            "none: no scaffold. target_grounding: reveal edit targets. "
            "conflict_localization: also reveal conflict locations. "
            "preservation: also reveal constraints to preserve."
        ),
    )
    parser.add_argument(
        "--exposure-mode",
        choices=[ExposureMode.PRIMITIVE_ONLY.value],
        default=ExposureMode.PRIMITIVE_ONLY.value,
        help="Runtime tool exposure mode for the published db_read_typed setting.",
    )
    parser.add_argument(
        "--tool-profile",
        choices=list(SUPPORTED_TOOL_PROFILES),
        default=None,
        help="Legacy compatibility profile. Exposure mode is the new primary switch.",
    )
    parser.add_argument(
        "--prompt-ablation",
        type=str,
        default="original",
        choices=["original", "unified_contract", "no_defensive_rules"],
        help="Prompt ablation mode: original, unified_contract, or no_defensive_rules.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run multiple samples concurrently",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent workers when --parallel is enabled",
    )
    return parser


def _resolve_tool_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    spec = resolve_tool_profile(
        tool_profile=getattr(args, "tool_profile", None),
    )
    guard_retries = getattr(args, "guard_retries", None)
    if guard_retries is None:
        guard_retries = 0
    semantic_tool_allowlist = None
    if args.framework == "pter" and spec.tool_profile == TOOL_PROFILE_DB_READ_TYPED:
        if getattr(args, "prompt_ablation", "original") == "original":
            args.prompt_ablation = "unified_contract"
    return {
        "exposure_mode": getattr(args, "exposure_mode", ExposureMode.PRIMITIVE_ONLY.value),
        "tool_profile": spec.tool_profile,
        "enable_ct_atoms": spec.enable_ct_atoms,
        "enable_ct_verify": spec.enable_ct_verify,
        "enable_ct_conflict_lift": spec.enable_ct_conflict_lift,
        "enable_ct_notepad": spec.enable_ct_notepad,
        "db_read_enabled": spec.db_read_enabled,
        "semantic_tool_allowlist": semantic_tool_allowlist,
        "guard_retries": int(guard_retries),
    }


def _build_world_env() -> Any:
    from chinatravel.environment.world_env import WorldEnv

    return WorldEnv()


def _resolve_experiment_dir(
    output_dir: str | None,
    framework: str,
    model_name: str,
    output_layout: str = "nested",
    tool_label: str | None = None,
) -> Path:
    return resolve_experiment_dir(
        project_root,
        framework=framework,
        model_name=model_name,
        tool_label=tool_label,
        output_dir=output_dir,
        layout_mode=output_layout,
    )


def _resolve_output_dir(
    output_dir: str | None,
    framework: str,
    model_name: str,
    output_layout: str = "nested",
    tool_label: str | None = None,
) -> Path:
    return resolve_run_dir(
        project_root,
        framework=framework,
        model_name=model_name,
        tool_label=tool_label,
        output_dir=output_dir,
        layout_mode=output_layout,
    )


def _resolve_latest_batch_dir() -> Path:
    batches_root = project_root / "data" / "batches"
    batch_dirs = [path for path in batches_root.iterdir() if path.is_dir()]
    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found under {batches_root}")
    return max(batch_dirs, key=lambda path: path.stat().st_mtime)


def _resolve_model_name(args: argparse.Namespace, llm_client: SiliconCloudClient | None) -> str:
    return llm_client.default_model


def _summarize_failures(results: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    failed_cases: list[Dict[str, Any]] = []
    for item in results:
        if item.get("success"):
            continue
        errors = item.get("errors") or []
        first_error = str(errors[0]) if errors else "unknown_error"
        failed_cases.append(
            {
                "case_id": str(item.get("case_id", "unknown_case")),
                "error": first_error,
            }
        )
    return failed_cases


def _is_runtime_success(item: Dict[str, Any]) -> bool:
    """Runtime success means the framework produced a valid executable result."""

    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    errors = [str(error) for error in (item.get("errors") or []) if error is not None]
    is_guard_reject = any(error.startswith("candidate_plan_guard_failed:") for error in errors)

    if not item.get("success") and not is_guard_reject:
        return False
    if isinstance(metrics.get("infeasible_detection"), dict):
        return True
    if isinstance(item.get("edited_plan"), dict):
        return True
    ops = item.get("ops")
    return isinstance(ops, list) and len(ops) > 0


def _summarize_guard_outcomes(results: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "guard_accept_count": 0,
        "guard_reject_count": 0,
        "guard_revise_count": 0,
        "guard_infeasible_count": 0,
    }
    for item in results:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        decision = metrics.get("guard_decision")
        if decision == "accept":
            counts["guard_accept_count"] += 1
        elif decision in {"revise", "infeasible"}:
            counts["guard_reject_count"] += 1
            if decision == "revise":
                counts["guard_revise_count"] += 1
            else:
                counts["guard_infeasible_count"] += 1
    return counts


def _summarize_terminal_error_categories(results: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in results:
        if _is_runtime_success(item):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        categories = metrics.get("terminal_error_categories") or classify_terminal_error_categories(
            item.get("errors"),
            metrics=metrics,
            special_backend=item.get("special_backend"),
        )
        for category in categories or ["unknown_runtime_failure"]:
            category = str(category)
            counts[category] = counts.get(category, 0) + 1
    return counts


def _metric_number(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value, key=str)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _summarize_resource_usage(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(results)
    sample_count = len(items)
    totals = {
        "execution_time_seconds": 0.0,
        "llm_request_time_seconds": 0.0,
        "llm_call_count": 0,
        "failed_llm_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tokens_used": 0,
    }
    per_case: list[Dict[str, Any]] = []

    for item in items:
        metrics = item.get("metrics") or {}
        total_tokens = int(_metric_number(metrics, "total_tokens"))
        if total_tokens == 0:
            total_tokens = int(_metric_number(metrics, "tokens_used"))
        case_summary = {
            "case_id": item.get("case_id"),
            "success": bool(item.get("success")),
            "execution_time_seconds": _metric_number(metrics, "execution_time_seconds"),
            "llm_request_time_seconds": _metric_number(metrics, "llm_request_time_seconds"),
            "llm_call_count": int(_metric_number(metrics, "llm_call_count")),
            "failed_llm_call_count": int(_metric_number(metrics, "failed_llm_call_count")),
            "prompt_tokens": int(_metric_number(metrics, "prompt_tokens")),
            "completion_tokens": int(_metric_number(metrics, "completion_tokens")),
            "total_tokens": total_tokens,
        }
        per_case.append(case_summary)
        for key in (
            "execution_time_seconds",
            "llm_request_time_seconds",
            "llm_call_count",
            "failed_llm_call_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            totals[key] += case_summary[key]
        totals["tokens_used"] += total_tokens

    averages = {
        "execution_time_seconds": totals["execution_time_seconds"] / sample_count
        if sample_count
        else 0.0,
        "llm_request_time_seconds": totals["llm_request_time_seconds"] / sample_count
        if sample_count
        else 0.0,
        "total_tokens": totals["total_tokens"] / sample_count if sample_count else 0.0,
    }
    return {
        "totals": totals,
        "averages": averages,
        "per_case": per_case,
    }


def _serialize_result(
    framework: str,
    model_name: str,
    edit_input: Any,
    result: Any,
) -> Dict[str, Any]:
    return {
        "case_id": edit_input.case_id,
        "framework": framework,
        "model": model_name,
        "tool_profile": result.tool_profile,
        "exposure_mode": result.exposure_mode,
        "db_read_enabled": result.db_read_enabled,
        "semantic_tool_allowlist": result.metrics.get("tool_flags", {}).get("semantic_tool_allowlist"),
        "enabled_semantic_tools": result.metrics.get("tool_flags", {}).get("enabled_semantic_tools"),
        "special_backend": result.special_backend,
        "input": edit_input.model_dump(),
        "success": result.success,
        "edited_plan": result.edited_plan,
        "ops": result.ops,
        "trace": result.trace,
        "metrics": result.metrics,
        "errors": result.errors,
    }


async def _run_case(
    framework_name: str,
    model_name: str,
    framework_factory: Any,
    sample_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    edit_input = load_edit_input_from_sample(sample_path)
    case_start = time.perf_counter()
    try:
        framework = framework_factory()
        result = await framework.run(edit_input)
        payload = _serialize_result(framework_name, model_name, edit_input, result)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            metrics.setdefault("execution_time_seconds", time.perf_counter() - case_start)
    except Exception as exc:
        payload = {
            "case_id": edit_input.case_id,
            "framework": framework_name,
            "model": model_name,
            "tool_profile": None,
            "exposure_mode": None,
            "db_read_enabled": None,
            "semantic_tool_allowlist": None,
            "enabled_semantic_tools": None,
            "special_backend": None,
            "input": edit_input.model_dump(),
            "success": False,
            "edited_plan": None,
            "ops": [],
            "trace": [],
            "metrics": {
                "runner_exception_type": type(exc).__name__,
                "execution_time_seconds": time.perf_counter() - case_start,
                **framework_error_handling_metadata(framework_name),
                "terminal_error_categories": classify_terminal_error_categories(
                    [f"{type(exc).__name__}: {exc}"],
                    metrics={"runner_exception_type": type(exc).__name__},
                ),
            },
            "errors": [f"{type(exc).__name__}: {exc}"],
            "runner_traceback": traceback.format_exception(exc),
        }
    output_path = output_root / f"{edit_input.case_id}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return payload


async def _run_cases_serial(
    *,
    framework_name: str,
    model_name: str,
    framework_factory: Any,
    sample_paths: Iterable[Path],
    output_root: Path,
) -> list[Dict[str, Any]]:
    results = []
    sample_paths = list(sample_paths)
    progress = tqdm(sample_paths, desc=f"{framework_name} serial", unit="sample")
    for sample_path in progress:
        results.append(
            await _run_case(framework_name, model_name, framework_factory, sample_path, output_root)
        )
    return results


async def _run_cases_parallel(
    *,
    framework_name: str,
    model_name: str,
    framework_factory: Any,
    sample_paths: list[Path],
    output_root: Path,
    max_workers: int,
) -> list[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_workers)
    results = []

    async def _guarded(sample_path: Path) -> Dict[str, Any]:
        async with semaphore:
            return await _run_case(
                framework_name,
                model_name,
                framework_factory,
                sample_path,
                output_root,
            )

    tasks = [asyncio.create_task(_guarded(sample_path)) for sample_path in sample_paths]
    progress = tqdm(total=len(sample_paths), desc=f"{framework_name} parallel", unit="sample")
    try:
        for task in asyncio.as_completed(tasks):
            results.append(await task)
            progress.update(1)
    finally:
        progress.close()
    return results


async def _async_main(args: argparse.Namespace) -> int:
    batch_dir = args.batch_dir
    if args.latest_batch:
        batch_dir = str(_resolve_latest_batch_dir())
    elif not args.sample_path and not batch_dir:
        batch_dir = str(_resolve_latest_batch_dir())

    sample_paths = collect_sample_paths(
        sample_path=args.sample_path,
        batch_dir=batch_dir,
        limit=args.limit,
    )
    llm_client = SiliconCloudClient(
        api_key=os.getenv("DMXAPI_API_KEY") or os.getenv("SILICONCLOUD_API_KEY"),
        base_url=os.getenv("DMXAPI_BASE_URL") or os.getenv("SILICONCLOUD_BASE_URL"),
    )
    model_name = _resolve_model_name(args, llm_client)
    tool_runtime = _resolve_tool_runtime_config(args)
    def framework_factory() -> Any:
        return create_edit_framework(
            args.framework,
            llm_client=llm_client,
            world_env=_build_world_env(),
            tool_profile=tool_runtime["tool_profile"],
            exposure_mode=tool_runtime["exposure_mode"],
            max_steps=args.max_steps,
            max_tool_calls=args.max_tool_calls,
            max_reflections=args.max_reflections,
            guard_retries=tool_runtime.get("guard_retries", 0),
            reflection_strategy=args.reflection_strategy,
            context_prompt=args.context_prompt,
            database_prompt=args.database_prompt,
            prompt_ablation=args.prompt_ablation,
            semantic_tool_allowlist=tool_runtime["semantic_tool_allowlist"],
            annotation_scaffold_level=args.annotation_scaffold_level,
        )

    tool_label = build_tool_config_label(
        tool_profile=tool_runtime["tool_profile"],
        exposure_mode=tool_runtime["exposure_mode"],
        enable_ct_atoms=bool(tool_runtime["enable_ct_atoms"]),
        enable_ct_verify=bool(tool_runtime["enable_ct_verify"]),
        enable_ct_conflict_lift=bool(tool_runtime["enable_ct_conflict_lift"]),
        enable_ct_notepad=tool_runtime["enable_ct_notepad"],
    )
    experiment_dir = _resolve_experiment_dir(
        args.output_dir,
        args.framework,
        model_name,
        args.output_layout,
        tool_label,
    )
    output_root = _resolve_output_dir(
        args.output_dir,
        args.framework,
        model_name,
        args.output_layout,
        tool_label,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        results = await _run_cases_parallel(
            framework_name=args.framework,
            model_name=model_name,
            framework_factory=framework_factory,
            sample_paths=sample_paths,
            output_root=output_root,
            max_workers=max(1, args.max_workers),
        )
    else:
        results = await _run_cases_serial(
            framework_name=args.framework,
            model_name=model_name,
            framework_factory=framework_factory,
            sample_paths=sample_paths,
            output_root=output_root,
        )

    resource_usage = _summarize_resource_usage(results)
    runtime_success_count = sum(1 for item in results if _is_runtime_success(item))
    runtime_failure_count = len(results) - runtime_success_count
    runtime_success_rate = runtime_success_count / len(results) if results else 0.0
    guard_outcomes = _summarize_guard_outcomes(results)
    enabled_semantic_tools = []
    summary = {
        "framework": args.framework,
        "batch_dir": batch_dir,
        "sample_count": len(results),
        "success_count": sum(1 for item in results if item["success"]),
        "failure_count": sum(1 for item in results if not item["success"]),
        "runtime_success_count": runtime_success_count,
        "runtime_failure_count": runtime_failure_count,
        "runtime_success_rate": runtime_success_rate,
        **guard_outcomes,
        "terminal_error_categories": _summarize_terminal_error_categories(results),
        "failed_cases": _summarize_failures(results),
        "parallel": bool(args.parallel),
        "max_workers": max(1, args.max_workers) if args.parallel else 1,
        "samples": [item["case_id"] for item in results],
        "model": model_name,
        "tool_profile": tool_runtime["tool_profile"],
        "tool_label": tool_label,
        "exposure_mode": tool_runtime["exposure_mode"],
        "db_read_enabled": tool_runtime["db_read_enabled"],
        "semantic_tool_allowlist": tool_runtime["semantic_tool_allowlist"],
        "guard_retries": tool_runtime.get("guard_retries", 0),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_root),
        "run_started_at": experiment_dir.name,
        "output_layout": args.output_layout,
        "resource_usage": resource_usage,
        "reflection_strategy": args.reflection_strategy if args.framework == "reflexion" else None,
        "prompt_modes": {
            "context_prompt": bool(args.context_prompt),
            "database_prompt": bool(args.database_prompt),
        },
        "total_execution_time_seconds": resource_usage["totals"]["execution_time_seconds"],
        "total_llm_request_time_seconds": resource_usage["totals"]["llm_request_time_seconds"],
        "total_tokens": resource_usage["totals"]["total_tokens"],
        "total_prompt_tokens": resource_usage["totals"]["prompt_tokens"],
        "total_completion_tokens": resource_usage["totals"]["completion_tokens"],
        "total_llm_call_count": resource_usage["totals"]["llm_call_count"],
        "tool_flags": {
            "enable_ct_atoms": bool(tool_runtime["enable_ct_atoms"]),
            "enable_ct_verify": bool(tool_runtime["enable_ct_verify"]),
            "enable_ct_conflict_lift": bool(tool_runtime["enable_ct_conflict_lift"]),
            "enable_ct_notepad": bool(tool_runtime["enable_ct_notepad"]),
            "db_read_enabled": bool(tool_runtime["db_read_enabled"]),
            "tool_profile": tool_runtime["tool_profile"],
            "exposure_mode": tool_runtime["exposure_mode"],
            "semantic_tool_allowlist": tool_runtime["semantic_tool_allowlist"],
            "enabled_semantic_tools": enabled_semantic_tools,
            "guard_retries": tool_runtime.get("guard_retries", 0),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
