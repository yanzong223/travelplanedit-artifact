#!/usr/bin/env python3
"""Run one model on one batch by category."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
CHINATRAVEL_ROOT = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
for candidate in (SRC_ROOT, PROJECT_ROOT, CHINATRAVEL_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from edit_framework.run_layout import build_tool_config_label, resolve_model_experiment_dir
from llm.client import SUPPORTED_MODEL_FORMATS
from llm.model_router import ModelRoute, get_model_router


DEFAULT_BATCH_ID = "benchmark"
DEFAULT_MODEL_ALIAS = "deepseek_v4_pro_guan"
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
RUN_BY_CATEGORY_PATH = PROJECT_ROOT / "src" / "artifact_runner" / "run_edit_framework_by_category.py"


def _load_run_by_category_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_edit_framework_by_category_for_single_model_batch",
        RUN_BY_CATEGORY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_RUN_BY_CATEGORY = _load_run_by_category_module()
discover_category_dirs = _RUN_BY_CATEGORY.discover_category_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one model over one batch by category."
    )
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--batch-root", type=Path, default=None)
    parser.add_argument("--validation-split", choices=["passed", "failed", "rejected"], default="passed")
    parser.add_argument(
        "--split-source",
        choices=["validation", "curated", "direct"],
        default="direct",
        help="Forwarded to run_edit_framework_by_category.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--framework", default="react")
    parser.add_argument("--tool-profile", default="db_read_typed", choices=["db_read_typed"])
    parser.add_argument("--exposure-mode", choices=["primitive_only"], default="primitive_only")
    parser.add_argument("--level", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=30)
    parser.add_argument("--max-reflections", type=int, default=2)
    parser.add_argument(
        "--prompt-ablation",
        type=str,
        default="original",
        choices=["original", "unified_contract", "no_defensive_rules"],
        help="Prompt ablation mode (forwarded to run_edit_framework_by_category.py).",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--guard-retries", type=int, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_batch_root(args: argparse.Namespace) -> Path:
    if args.batch_root is not None:
        return Path(args.batch_root).resolve()
    return (PROJECT_ROOT / "data" / str(args.batch_id)).resolve()


def _infer_direct_family(model_alias: str) -> str:
    normalized = model_alias.lower().replace("_", "-")
    if "deepseek" in normalized:
        return "deepseek"
    if "gemini" in normalized:
        return "gemini"
    if "qwen" in normalized:
        return "qwen"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")) or "openai" in normalized:
        return "openai"
    return ""


def resolve_model_route(model_alias: str) -> ModelRoute | None:
    return get_model_router().resolve(model_alias, provider="DMXAPI")


def resolve_model_family(model_alias: str) -> str:
    route = resolve_model_route(model_alias)
    if route is not None and route.family:
        return route.family.lower()
    return _infer_direct_family(model_alias)


def validate_model_format(model_alias: str) -> str:
    family = resolve_model_family(model_alias)
    if family not in SUPPORTED_MODEL_FORMATS:
        raise ValueError(
            "Unsupported model format "
            f"for model_alias={model_alias!r}, family={family or 'unknown'!r}. "
            f"Supported families: {', '.join(sorted(SUPPORTED_MODEL_FORMATS))}."
        )
    return family


def dataset_label(args: argparse.Namespace) -> str:
    split = str(args.validation_split or "all")
    source = str(args.split_source or "validation")
    suffix = "" if source == "validation" else f"_{source}"
    return f"{args.batch_id}_{split}_bycat{suffix}"


def effective_tool_profile(args: argparse.Namespace) -> str | None:
    return args.tool_profile


def default_output_dir(args: argparse.Namespace, *, started_at: str) -> Path:
    return resolve_model_experiment_dir(
        Path(args.experiments_root),
        started_at=started_at,
        model_name=str(args.model_alias),
        framework=str(args.framework),
        tool_label=build_tool_config_label(
            tool_profile=effective_tool_profile(args),
            exposure_mode=args.exposure_mode,
        ),
        dataset_label=dataset_label(args),
    )


def build_env(model_alias: str) -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    pythonpath_parts = [str(SRC_ROOT), str(PROJECT_ROOT), str(CHINATRAVEL_ROOT)]
    if current:
        pythonpath_parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["DMXAPI_MODEL"] = model_alias
    env["DMXAPI_MODEL_ALIAS"] = model_alias

    family = validate_model_format(model_alias)
    route = resolve_model_route(model_alias)
    if family == "deepseek" and (route is None or route.supports_thinking):
        env.setdefault("DMXAPI_DEEPSEEK_REASONING_EFFORT", "high")
        env.setdefault("DMXAPI_DEEPSEEK_THINKING_TYPE", "enabled")
        env.setdefault("DMXAPI_REACT_MAX_COMPLETION_TOKENS", "65536")
    return env


def build_command(args: argparse.Namespace, *, batch_root: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_BY_CATEGORY_PATH),
        "--framework",
        str(args.framework),
        "--batch-root",
        str(batch_root),
        "--output-dir",
        str(output_dir),
        "--output-layout",
        "flat",
        "--validation-split",
        str(args.validation_split),
        "--split-source",
        str(args.split_source),
        "--all-samples",
        "--level",
        str(args.level),
        "--max-steps",
        str(args.max_steps),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--max-reflections",
        str(args.max_reflections),
        "--exposure-mode",
        str(args.exposure_mode),
    ]
    if args.tool_profile:
        cmd.extend(["--tool-profile", str(args.tool_profile)])
    if args.guard_retries is not None:
        cmd.extend(["--guard-retries", str(args.guard_retries)])
    if args.parallel:
        cmd.extend(["--parallel", "--max-workers", str(args.max_workers)])
    if args.skip_evaluation:
        cmd.append("--skip-evaluation")
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    if args.prompt_ablation and args.prompt_ablation != "original":
        cmd.extend(["--prompt-ablation", str(args.prompt_ablation)])
    return cmd


def summarize_categories(batch_root: Path, validation_split: str, split_source: str) -> dict[str, Any]:
    categories, skipped_empty = discover_category_dirs(
        batch_root,
        validation_split=validation_split,
        split_source=split_source,
    )
    return {
        "category_count": len(categories),
        "sample_count": sum(int(item["sample_count"]) for item in categories),
        "categories": [
            {
                "name": item["name"],
                "path": str(Path(item["path"])),
                "sample_count": item["sample_count"],
            }
            for item in categories
        ],
        "skipped_empty_categories": skipped_empty,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model_family = validate_model_format(str(args.model_alias))
    except ValueError as exc:
        parser.error(str(exc))

    batch_root = resolve_batch_root(args)
    if not batch_root.exists():
        parser.error(f"--batch-root does not exist: {batch_root}")

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(
        args,
        started_at=started_at,
    )
    env = build_env(str(args.model_alias))
    command = build_command(args, batch_root=batch_root, output_dir=output_dir)
    category_summary = summarize_categories(
        batch_root,
        str(args.validation_split),
        str(args.split_source),
    )
    tool_label = build_tool_config_label(
        tool_profile=effective_tool_profile(args),
        exposure_mode=args.exposure_mode,
    )

    payload = {
        "started_at": started_at,
        "model_alias": args.model_alias,
        "model_family": model_family,
        "provider_env": {
            key: env[key]
            for key in sorted(env)
            if key.startswith("DMXAPI_MODEL")
            or key.startswith("DMXAPI_DEEPSEEK")
            or key.startswith("DMXAPI_QWEN")
            or key.startswith("DMXAPI_GEMINI")
            or key == "DMXAPI_REACT_MAX_COMPLETION_TOKENS"
        },
        "batch_id": args.batch_id,
        "batch_root": str(batch_root),
        "output_dir": str(output_dir),
        "validation_split": args.validation_split,
        "split_source": args.split_source,
        "framework": args.framework,
        "exposure_mode": args.exposure_mode,
        "tool_profile": args.tool_profile,
        "tool_label": tool_label,
        "dataset_label": dataset_label(args),
        "all_samples": True,
        "category_summary": category_summary,
        "command": command,
    }

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
