#!/usr/bin/env python3
"""
数据生成Pipeline Orchestrator
顺序执行步骤1-6，支持断点续传和迭代重试。
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_GEN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DATA_GEN_DIR / "utils"))
PROJECT_ROOT = DATA_GEN_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from origin_plan_resampler import select_replacement_origin_plan
from constraint_type_funnel import write_constraint_type_funnel
from conflict_feasibility import build_origin_plan_evidence as build_origin_plan_evidence_for_plan
from solver_feasibility import assess_solver_feasibility
from temporal_window_feasibility import assess_temporal_window_feasibility
from category_guard_validity import assess_category_guard_validity

# 步骤映射
PIPELINE_STEPS = [1, 2, 3, 4, 5, 6]

# 步骤脚本文件名
SCRIPT_FILES = {
    1: "01_sample_bucket.py",
    2: "02_generate_query.py",
    3: "03_freeze_edit_truth.py",
    4: "04_rewrite_query_surface.py",
    5: "05_analyze_conflict.py",
    6: "06_validate.py",
}

MAX_ORIGIN_SWITCHES = 3


def get_steps_for_mode(sample_mode: str) -> List[int]:
    return PIPELINE_STEPS


def get_project_root() -> Path:
    """项目根目录（project root）"""
    return Path(__file__).resolve().parent.parent.parent


def normalize_batch_dir(batch_dir: str) -> Path:
    """
    将 batch 目录标准化为绝对路径。
    相对路径统一按项目根目录解析，避免主进程和子步骤写到不同目录。
    """
    batch_path = Path(batch_dir).expanduser()
    if not batch_path.is_absolute():
        batch_path = get_project_root() / batch_path
    return batch_path.resolve()


def resolve_user_path(path_value: str) -> Path:
    """将用户输入路径解析为绝对路径（相对路径按当前 shell cwd 解析）。"""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_config_path(path_value: Optional[str], default_path: Path) -> str:
    """
    解析配置路径：
    - None: 使用默认路径
    - 绝对路径: 直接使用
    - 相对路径: 优先按项目根目录解析，若不存在再按 data_generation 目录解析
    """
    if path_value is None:
        return str(default_path.resolve())

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    project_candidate = (get_project_root() / path).resolve()
    if project_candidate.exists():
        return str(project_candidate)

    data_gen_candidate = (Path(__file__).resolve().parent.parent / path).resolve()
    return str(data_gen_candidate)


def _normalize_origin_plan(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "people_number": plan_data.get("people_number", 1),
        "start_city": plan_data.get("start_city", ""),
        "target_city": plan_data.get("target_city", ""),
        "itinerary": plan_data.get("itinerary", []),
    }


def load_origin_query_lookup(query_dir: str) -> Dict[str, Dict[str, Any]]:
    """加载 origin query 目录下所有 JSON（按文件名索引）。"""
    query_path = Path(query_dir)
    if not query_path.exists() or not query_path.is_dir():
        raise ValueError(f"Origin query directory not found: {query_dir}")

    lookup: Dict[str, Dict[str, Any]] = {}
    duplicate_files = 0
    for json_file in sorted(query_path.rglob("*.json")):
        file_name = json_file.name
        if file_name in lookup:
            duplicate_files += 1
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            lookup[file_name] = json.load(f)

    print(f"Loaded {len(lookup)} origin queries from {query_dir}")
    if duplicate_files > 0:
        print(f"  Warning: ignored {duplicate_files} duplicate query files by filename")
    return lookup


def extract_origin_query_text(origin_query: Dict[str, Any]) -> str:
    """从 origin query 结构中提取自然语言文本。"""
    if not isinstance(origin_query, dict):
        return ""
    for key in ("nature_language", "natural_language", "query", "text"):
        value = origin_query.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _plan_has_pois(origin_plan: Dict[str, Any]) -> bool:
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if activity.get("type") in {"attraction", "restaurant"}:
                return True
    return False


def _calculate_plan_costs(origin_plan: Dict[str, Any]) -> Tuple[float, float]:
    total_cost = 0.0
    ticket_cost = 0.0
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            cost = activity.get("cost")
            if not isinstance(cost, (int, float)):
                continue
            total_cost += float(cost)
            if activity.get("type") == "attraction":
                ticket_cost += float(cost)
    return total_cost, ticket_cost


def _calculate_plan_activity_costs(origin_plan: Dict[str, Any]) -> Tuple[float, float]:
    meal_cost = 0.0
    accommodation_cost = 0.0
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            cost = activity.get("cost")
            if not isinstance(cost, (int, float)):
                continue
            if activity.get("type") in {"breakfast", "lunch", "dinner"}:
                meal_cost += float(cost)
            elif activity.get("type") == "accommodation":
                accommodation_cost += float(cost)
    return meal_cost, accommodation_cost


def _min_numeric_option(values: Any) -> Optional[float]:
    if not isinstance(values, (list, tuple)):
        return None
    numeric_values: List[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric_values.append(float(value))
        elif isinstance(value, str):
            try:
                numeric_values.append(float(value))
            except ValueError:
                continue
    return min(numeric_values) if numeric_values else None


def _origin_has_budget_requirement(origin_logical_constraints: Any) -> bool:
    if not isinstance(origin_logical_constraints, list):
        return False
    for item in origin_logical_constraints:
        if not isinstance(item, dict):
            continue
        constraint_type = str(item.get("type", "")).strip().lower()
        if "budget" in constraint_type:
            return True
    return False


def _extract_origin_budget_total(origin_logical_constraints: Any) -> Optional[float]:
    if not isinstance(origin_logical_constraints, list):
        return None
    for item in origin_logical_constraints:
        if not isinstance(item, dict):
            continue
        constraint_type = str(item.get("type", "")).strip().lower()
        if constraint_type != "budget_total":
            continue
        value = item.get("value")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _template_is_applicable_for_plan(
    template: Dict[str, Any],
    origin_plan: Dict[str, Any],
    origin_logical_constraints: Any = None,
) -> bool:
    if template.get("poi_source") == "existing_attractions" and not _plan_has_pois(origin_plan):
        return False

    constraint_type = str(template.get("constraint_type", "")).strip()
    params = template.get("params", {}) if isinstance(template.get("params"), dict) else {}
    total_cost, ticket_cost = _calculate_plan_costs(origin_plan)
    has_origin_budget = _origin_has_budget_requirement(origin_logical_constraints)
    origin_budget_total = _extract_origin_budget_total(origin_logical_constraints)
    evidence = build_origin_plan_evidence_for_plan(origin_plan)

    if constraint_type == "budget_limit":
        min_budget = _min_numeric_option(params.get("budget", []))
        return min_budget is None or total_cost > min_budget

    if constraint_type in {"budget_cap_preference", "budget_target_update"}:
        return origin_budget_total is not None and total_cost > origin_budget_total

    if constraint_type == "ticket_budget_limit":
        if not has_origin_budget:
            return False
        min_budget = _min_numeric_option(params.get("budget", []))
        return min_budget is None or ticket_cost > min_budget

    if constraint_type == "activity_budget_limit":
        min_budget = _min_numeric_option(params.get("budget", []))
        if min_budget is None:
            return True
        meal_cost, accommodation_cost = _calculate_plan_activity_costs(origin_plan)
        template_text = " ".join(
            str(part)
            for part in [template.get("template", ""), template.get("description", "")]
            if part
        )
        if any(token in template_text for token in ["餐", "餐饮", "用餐"]):
            return meal_cost > min_budget
        if any(token in template_text for token in ["住宿", "酒店", "房"]):
            return accommodation_cost > min_budget
        return meal_cost > min_budget or accommodation_cost > min_budget

    if constraint_type == "resource_overlap":
        if not has_origin_budget:
            return False
        min_budget = _min_numeric_option(params.get("budget", []))
        return _plan_has_pois(origin_plan) and (min_budget is None or total_cost > min_budget)

    if constraint_type == "multi_day_budget_overflow":
        target_day_count = _min_numeric_option(params.get("day_count", []))
        current_day_count = int(evidence.get("day_count", 0) or 0)
        if target_day_count is not None and current_day_count >= int(target_day_count):
            return False
        if not bool(evidence.get("budget_baseline_reliable_for_multi_day", False)):
            return False
        return float(evidence.get("transferable_local_cost", 0.0) or 0.0) > 0.0

    return True


def _collect_configured_constraint_types(templates_path: str) -> List[str]:
    raw = yaml.safe_load(Path(templates_path).read_text(encoding="utf-8")) or {}
    seen: List[str] = []
    for bucket_cfg in raw.values():
        if not isinstance(bucket_cfg, dict) or not bucket_cfg.get("eligible_for_edit", False):
            continue
        for template in bucket_cfg.get("templates", []):
            if not isinstance(template, dict):
                continue
            constraint_type = str(template.get("constraint_type", "") or "").strip()
            if constraint_type and constraint_type not in seen:
                seen.append(constraint_type)
    return seen


def _sanitize_constraint_type(constraint_type: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(constraint_type or ""))


def run_pipeline_by_constraint_type(
    *,
    output_root: str,
    batch_size: int = 100,
    variants_per_plan: int = 1,
    max_retries: int = 3,
    config_path: Optional[str] = None,
    templates_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    query_input: Optional[str] = None,
    model: str = "dmxapi",
    step2_seed: Optional[int] = None,
    step2_llm_temperature: Optional[float] = None,
    step4_purity_threshold: float = 0.7,
    step1_sampling_mode: str = "weighted",
    step1_allow_violations: Optional[str] = None,
    constraint_type_whitelist: Optional[str] = None,
    sample_mode: str = "core",
    solver_valid_profile: str = "off",
) -> bool:
    if sample_mode != "core":
        raise ValueError("by_constraint_type mode currently supports core sample_mode only")

    output_path = normalize_batch_dir(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    data_gen_dir = Path(__file__).resolve().parent.parent
    config_default = data_gen_dir / "config" / "bucket_distribution.yaml"
    templates_default = data_gen_dir / "config" / "templates.yaml"
    config_path_abs = resolve_config_path(config_path, config_default)
    templates_path_abs = resolve_config_path(templates_path, templates_default)

    requested_types = [
        item.strip()
        for item in str(constraint_type_whitelist or "").split(",")
        if item.strip()
    ]
    constraint_types = requested_types or _collect_configured_constraint_types(templates_path_abs)

    overall_success = True
    runs: List[Dict[str, Any]] = []
    for constraint_type in constraint_types:
        sub_batch_dir = output_path / "batches" / _sanitize_constraint_type(constraint_type)
        success = run_pipeline(
            batch_dir=str(sub_batch_dir),
            batch_size=batch_size,
            variants_per_plan=variants_per_plan,
            max_retries=max_retries,
            config_path=config_path_abs,
            templates_path=templates_path_abs,
            input_dir=input_dir,
            query_input=query_input,
            model=model,
            step2_seed=step2_seed,
            step2_llm_temperature=step2_llm_temperature,
            step4_purity_threshold=step4_purity_threshold,
            start_step=1,
            step1_sampling_mode=step1_sampling_mode,
            step1_allow_violations=step1_allow_violations,
            step1_bucket_whitelist=None,
            constraint_type_whitelist=constraint_type,
            sample_mode=sample_mode,
            solver_valid_profile=solver_valid_profile,
        )
        overall_success = overall_success and success
        runs.append(
            {
                "constraint_type": constraint_type,
                "batch_dir": str(sub_batch_dir),
                "success": success,
            }
        )

    (output_path / "constraint_type_runs_manifest.json").write_text(
        json.dumps(
            {
                "output_root": str(output_path),
                "mode": "by_constraint_type",
                "constraint_types": constraint_types,
                "batch_size_per_type": batch_size,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return overall_success


def _load_template_bucket_index(templates_path: str) -> Tuple[set, dict]:
    try:
        import yaml
        import re
    except Exception:
        return set(), {}

    with open(templates_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    template_buckets = set()
    template_config: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, bucket_cfg in config.items():
        if key in {"poi_source_types", "city_constraint_types", "template_instructions"}:
            continue
        bucket_tuple = None
        if isinstance(key, str):
            match = re.match(r"\(([^)]+)\)", key)
            if match:
                parts = [part.strip().strip("\"'") for part in match.group(1).split(",")]
                if len(parts) == 3:
                    bucket_tuple = tuple(parts)
        elif isinstance(key, (list, tuple)) and len(key) == 3:
            bucket_tuple = tuple(key)
        if bucket_tuple is None:
            continue
        if not isinstance(bucket_cfg, dict) or not bucket_cfg.get("eligible_for_edit", False):
            continue
        template_buckets.add(bucket_tuple)
        template_config[bucket_tuple] = bucket_cfg
    return template_buckets, template_config


def _is_bucket_applicable_for_plan(
    bucket: tuple,
    plan_meta: Dict[str, Any],
    templates_index: set,
    template_config: dict,
) -> bool:
    if bucket not in templates_index:
        return False
    bucket_cfg = template_config.get(bucket)
    if not isinstance(bucket_cfg, dict):
        return False
    origin_plan = plan_meta.get("origin_plan", {})
    for template in bucket_cfg.get("templates", []):
        if not isinstance(template, dict):
            continue
        if _template_is_applicable_for_plan(
            template,
            origin_plan,
            plan_meta.get("origin_logical_constraints"),
        ):
            return True
    return False


def _clear_sample_after_origin_swap(sample: Dict[str, Any]):
    for key in [
        "origin_logical_constraints",
        "origin_query_text",
        "origin_query_structured",
        "edit_query",
        "constraints",
        "edit_target_constraints",
        "edit_target_preferences",
        "edit_target_preference_tags",
        "query_generation_trace",
        "conflict_facts",
        "conflict_labels",
        "conflict_set",
        "primary_conflict",
        "secondary_conflicts",
        "purity_score_rule",
        "match_type_rule",
        "resolver_trace",
        "strategy_plan",
        "mock_edited_plan",
        "op_tags",
        "diff_summary",
        "checks",
    ]:
        sample.pop(key, None)


def load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    """加载batch中的所有samples"""
    batch_path = Path(batch_dir)
    samples = []

    for sample_file in sorted(batch_path.glob("sample_*.json")):
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
                samples.append(sample)
        except Exception as e:
            print(f"  Warning: Failed to load {sample_file}: {e}")

    return samples


def save_sample(sample: Dict[str, Any], batch_dir: str):
    """保存sample到文件"""
    sample_file = Path(batch_dir) / f"{sample['sample_id']}.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


def collect_sample_state(batch_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    收集当前 batch 内 sample 文件状态，用于快照对比。
    key 使用 sample_id（如 sample_000001）。
    """
    batch_path = Path(batch_dir)
    state: Dict[str, Dict[str, Any]] = {}

    for sample_file in sorted(batch_path.glob("sample_*.json")):
        sample_id = sample_file.stem

        try:
            raw = sample_file.read_bytes()
        except Exception as e:
            state[sample_id] = {
                "path": str(sample_file),
                "status": f"<read_error:{type(e).__name__}>",
                "sha256": "",
                "size_bytes": 0,
                "mtime": sample_file.stat().st_mtime if sample_file.exists() else 0
            }
            continue

        digest = hashlib.sha256(raw).hexdigest()
        status = "<parse_error>"
        try:
            data = json.loads(raw.decode("utf-8"))
            status = data.get("meta", {}).get("status", "<missing_status>")
        except Exception:
            pass

        stat = sample_file.stat()
        state[sample_id] = {
            "path": str(sample_file),
            "status": status,
            "sha256": digest,
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime
        }

    return state


def diff_sample_states(before: Dict[str, Dict[str, Any]],
                       after: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """对比前后两个 sample 状态快照。"""
    before_ids = set(before.keys())
    after_ids = set(after.keys())

    added = sorted(after_ids - before_ids)
    removed = sorted(before_ids - after_ids)

    changed = sorted(
        sid for sid in (before_ids & after_ids)
        if before[sid].get("sha256") != after[sid].get("sha256")
    )

    unchanged = sorted((before_ids & after_ids) - set(changed))

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged
    }


def count_status_distribution(state: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """统计 sample 状态分布。"""
    counts: Dict[str, int] = {}
    for item in state.values():
        status = item.get("status", "<missing_status>")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def write_step_snapshot(step: int,
                        run_index: int,
                        batch_dir: str,
                        success: bool,
                        command: str,
                        before_state: Dict[str, Dict[str, Any]],
                        after_state: Dict[str, Dict[str, Any]]) -> Path:
    """
    保存步骤级快照，包含：
    - status_counts.json
    - changed_samples.txt
    - summary.json
    - samples/ 下本步骤新增或变化的 sample 文件副本
    """
    batch_path = Path(batch_dir)
    snapshot_dir = batch_path / "_snapshots" / f"step_{step:02d}_run_{run_index:03d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    diff = diff_sample_states(before_state, after_state)
    status_counts = count_status_distribution(after_state)
    changed_or_added = sorted(set(diff["added"] + diff["changed"]))

    status_changes = {}
    for sid in changed_or_added:
        status_changes[sid] = {
            "before": before_state.get(sid, {}).get("status", "<missing>"),
            "after": after_state.get(sid, {}).get("status", "<missing>")
        }

    summary = {
        "step": step,
        "run_index": run_index,
        "step_run_id": f"step_{step:02d}_run_{run_index:03d}",
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "command": command,
        "sample_count_before": len(before_state),
        "sample_count_after": len(after_state),
        "diff": {
            "added": len(diff["added"]),
            "removed": len(diff["removed"]),
            "changed": len(diff["changed"]),
            "unchanged": len(diff["unchanged"])
        },
        "status_distribution": status_counts,
        "status_changes": status_changes
    }

    with open(snapshot_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(snapshot_dir / "status_counts.json", "w", encoding="utf-8") as f:
        json.dump(status_counts, f, ensure_ascii=False, indent=2)

    with open(snapshot_dir / "changed_samples.txt", "w", encoding="utf-8") as f:
        f.write(f"step={step}\n")
        f.write(f"run_index={run_index}\n")
        f.write(f"success={success}\n\n")
        f.write(f"added ({len(diff['added'])}):\n")
        for sid in diff["added"]:
            f.write(f"  {sid}\n")
        f.write(f"\nchanged ({len(diff['changed'])}):\n")
        for sid in diff["changed"]:
            f.write(f"  {sid}\n")
        f.write(f"\nremoved ({len(diff['removed'])}):\n")
        for sid in diff["removed"]:
            f.write(f"  {sid}\n")

    if changed_or_added:
        samples_snapshot_dir = snapshot_dir / "samples"
        samples_snapshot_dir.mkdir(parents=True, exist_ok=True)
        for sid in changed_or_added:
            src_path = Path(after_state[sid]["path"])
            if src_path.exists():
                shutil.copy2(src_path, samples_snapshot_dir / src_path.name)

    return snapshot_dir


def cleanup_stale_outputs_for_step1(batch_dir: str):
    """
    当从 Step 1 重跑时，清理可能导致混淆的旧产物，避免样本混入。
    """
    batch_path = Path(batch_dir)
    removed_count = 0

    for sample_file in batch_path.glob("sample_*.json"):
        sample_file.unlink(missing_ok=True)
        removed_count += 1

    for file_name in ["checkpoint.json", "pipeline_report.json", "batch_metadata.json"]:
        file_path = batch_path / file_name
        if file_path.exists():
            file_path.unlink()

    failed_dir = batch_path / "_failed"
    if failed_dir.exists():
        shutil.rmtree(failed_dir)

    validation_split_dir = batch_path / "_validation_split"
    if validation_split_dir.exists():
        shutil.rmtree(validation_split_dir)

    if removed_count > 0:
        print(f"清理旧样本: 删除 {removed_count} 个 sample_*.json")


def is_terminal_failure_status(status: Any) -> bool:
    text = str(status or "").strip()
    if not text or "_failed" not in text:
        return False
    if "retryable" in text or "_pending" in text:
        return False
    return True


def archive_terminal_failed_samples(batch_dir: str) -> int:
    """
    将终态失败样本移出 batch 根目录，避免影响后续步骤扫描。
    """
    batch_path = Path(batch_dir)
    archived_count = 0

    for sample_file in sorted(batch_path.glob("sample_*.json")):
        try:
            sample = json.loads(sample_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        status = sample.get("meta", {}).get("status", "")
        if not is_terminal_failure_status(status):
            continue

        target_dir = batch_path / "_failed" / str(status)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sample_file), str(target_dir / sample_file.name))
        archived_count += 1

    if archived_count > 0:
        print(f"归档终态失败样本: {archived_count} -> {batch_path / '_failed'}")

    return archived_count


def export_validation_split(batch_dir: str) -> Dict[str, Any]:
    """
    导出验证结果视图：
    - _validation_split/passed: checks.all_pass == True
    - _validation_split/failed: checks.all_pass != True

    保留原始 sample 文件，不移动，只复制，避免影响 resume 和下游步骤。
    """
    batch_path = Path(batch_dir)
    split_root = batch_path / "_validation_split"
    passed_dir = split_root / "passed"
    failed_dir = split_root / "failed"

    if split_root.exists():
        shutil.rmtree(split_root)
    passed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    passed_count = 0
    failed_count = 0

    for sample_file in sorted(batch_path.glob("sample_*.json")):
        try:
            sample = json.loads(sample_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        checks = sample.get("checks", {})
        target_dir = passed_dir if checks.get("all_pass", False) else failed_dir
        shutil.copy2(sample_file, target_dir / sample_file.name)

        if target_dir == passed_dir:
            passed_count += 1
        else:
            failed_count += 1

    summary = {
        "root": str(split_root),
        "passed_dir": str(passed_dir),
        "failed_dir": str(failed_dir),
        "passed_count": passed_count,
        "failed_count": failed_count,
    }
    (split_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


_COMPACT_META_KEYS = [
    "target_bucket",
    "target_city",
    "start_city",
    "is_single_day",
    "day_count",
    "origin_plan_file",
    "origin_variant_index",
    "origin_variant_limit",
    "requested_constraint_type",
    "requested_bucket_names",
    "curated_batch_id",
    "curated_from_batch_id",
    "source_seed_sample_id",
    "source_seed_path",
    "diversity_augmented",
    "diversity_domain",
    "constraint_type",
    "intent_group",
    "status",
    "created_at",
    "target_confirmed",
    "target_confidence",
    "primary_label",
]


def _build_compact_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    compact_meta = {key: meta[key] for key in _COMPACT_META_KEYS if key in meta}
    compact: Dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "batch_id": sample.get("batch_id"),
        "origin_query_text": sample.get("origin_query_text"),
        "origin_query_structured": sample.get("origin_query_structured"),
        "origin_plan": sample.get("origin_plan"),
        "origin_logical_constraints": sample.get("origin_logical_constraints"),
        "edit_query": sample.get("edit_query"),
        "constraints": sample.get("constraints"),
        "edit_target_constraints": sample.get("edit_target_constraints"),
        "edit_target_preferences": sample.get("edit_target_preferences"),
        "origin_preference_tags": sample.get("origin_preference_tags"),
        "edit_target_preference_tags": sample.get("edit_target_preference_tags"),
        "conflict_set": sample.get("conflict_set"),
        "primary_conflict": sample.get("primary_conflict"),
        "secondary_conflicts": sample.get("secondary_conflicts"),
        "meta": compact_meta,
    }
    return {key: value for key, value in compact.items() if value is not None}


def export_compact_curated(batch_dir: str, validation_split: Dict[str, Any]) -> Dict[str, Any]:
    batch_path = Path(batch_dir)
    curated_root = batch_path / "_curated"
    curated_passed = curated_root / "passed"

    if curated_root.exists():
        shutil.rmtree(curated_root)
    curated_passed.mkdir(parents=True, exist_ok=True)

    dataset_jsonl = curated_root / "dataset.jsonl"
    exported_count = 0
    with dataset_jsonl.open("w", encoding="utf-8") as sink:
        for sample_file in sorted(Path(validation_split["passed_dir"]).glob("sample_*.json")):
            try:
                sample = json.loads(sample_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            compact = _build_compact_sample(sample)
            (curated_passed / sample_file.name).write_text(
                json.dumps(compact, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sink.write(json.dumps(compact, ensure_ascii=False) + "\n")
            exported_count += 1

    summary = {
        "root": str(curated_root),
        "passed_dir": str(curated_passed),
        "dataset_jsonl": str(dataset_jsonl),
        "exported_count": exported_count,
    }
    (curated_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_step(step: int, batch_dir: str, **kwargs) -> Tuple[bool, str]:
    """
    执行单个步骤。

    Returns:
        (是否成功, 实际命令字符串)
    """
    script_file = SCRIPT_FILES[step]
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / script_file
    project_root = get_project_root()
    batch_dir_abs = str(normalize_batch_dir(batch_dir))
    python_bin = sys.executable if sys.executable else "python3"

    # Step 1使用input_dir，其他步骤使用batch_dir
    if step == 1:
        input_dir = kwargs.get("input_dir")
        if not input_dir:
            print("错误: Step 1需要input_dir参数")
            return False, ""
        query_input = kwargs.get("query_input")
        if not query_input:
            print("错误: Step 1需要query_input参数")
            return False, ""
        cmd = [python_bin, str(script_path), "--input", str(input_dir)]
    else:
        cmd = [python_bin, str(script_path), "--input", batch_dir_abs]

    # Step 1需要额外参数
    if step == 1:
        batch_size = kwargs.get("batch_size", 100)
        variants_per_plan = kwargs.get("variants_per_plan", 1)
        config_path = kwargs.get("config_path")
        templates_path = kwargs.get("templates_path")
        step1_sampling_mode = kwargs.get("step1_sampling_mode")
        step1_allow_violations = kwargs.get("step1_allow_violations")
        step1_bucket_whitelist = kwargs.get("step1_bucket_whitelist")
        constraint_type_whitelist = kwargs.get("constraint_type_whitelist")
        query_input = kwargs.get("query_input")

        cmd.extend(["--output", batch_dir_abs, "--batch-size", str(batch_size)])
        cmd.extend(["--variants-per-plan", str(variants_per_plan)])
        cmd.extend(["--query-input", str(query_input)])

        if config_path:
            cmd.extend(["--config", str(config_path)])
        if templates_path:
            cmd.extend(["--templates", str(templates_path)])
        if step1_sampling_mode:
            cmd.extend(["--sampling-mode", str(step1_sampling_mode)])
        if step1_allow_violations:
            cmd.extend(["--allow-violations", str(step1_allow_violations)])
        if step1_bucket_whitelist:
            cmd.extend(["--bucket-whitelist", str(step1_bucket_whitelist)])
        if constraint_type_whitelist:
            cmd.extend(["--constraint-type-whitelist", str(constraint_type_whitelist)])

    # Step 2需要config参数（templates.yaml）
    if step == 2:
        templates_path = kwargs.get("templates_path")
        if templates_path:
            cmd.extend(["--config", str(templates_path)])
        step1_allow_violations = kwargs.get("step1_allow_violations")
        if step1_allow_violations:
            cmd.extend(["--allow-violations", str(step1_allow_violations)])
        step2_seed = kwargs.get("step2_seed")
        if step2_seed is not None:
            cmd.extend(["--seed", str(step2_seed)])
        step2_llm_temperature = kwargs.get("step2_llm_temperature")
        if step2_llm_temperature is not None:
            cmd.extend(["--llm-temperature", str(step2_llm_temperature)])
        constraint_type_whitelist = kwargs.get("constraint_type_whitelist")
        if constraint_type_whitelist:
            cmd.extend(["--constraint-type-whitelist", str(constraint_type_whitelist)])

    # Step 5强一致阈值
    if step == 5:
        step4_purity_threshold = kwargs.get("step4_purity_threshold")
        if step4_purity_threshold is not None:
            cmd.extend(["--purity-threshold", str(step4_purity_threshold)])

    if step == 6:
        solver_valid_profile = str(kwargs.get("solver_valid_profile", "off") or "off")
        if solver_valid_profile != "off":
            cmd.extend(["--solver-valid-profile", solver_valid_profile])

    # LLM步骤统一透传provider，避免auto模式误选
    if step in [2, 4, 5]:
        model = kwargs.get("model")
        if model:
            cmd.extend(["--model", model])

    step_run_id = kwargs.get("step_run_id", "")

    env = os.environ.copy()
    env["PIPELINE_BATCH_DIR"] = batch_dir_abs
    env["PIPELINE_LLM_DEBUG_ROOT"] = str(Path(batch_dir_abs) / "_snapshots" / "llm")
    if step_run_id:
        env["PIPELINE_STEP_RUN_ID"] = str(step_run_id)

    command_str = " ".join(cmd)

    print(f"\n{'='*60}")
    print(f"执行步骤 {step}: {script_file}")
    print(f"{'='*60}")
    print(f"命令: {command_str}")
    print(f"工作目录: {project_root}")

    try:
        subprocess.run(cmd, check=True, capture_output=False, cwd=str(project_root), env=env)
        print(f"✓ 步骤 {step} 完成")
        return True, command_str
    except subprocess.CalledProcessError as e:
        print(f"✗ 步骤 {step} 失败: {e}")
        return False, command_str


def execute_step_with_snapshot(step: int,
                               batch_dir: str,
                               step_run_counter: Dict[int, int],
                               **kwargs) -> bool:
    """执行步骤并保存本步骤前后对比快照。"""
    before_state = collect_sample_state(batch_dir)
    step_run_counter[step] = step_run_counter.get(step, 0) + 1
    run_index = step_run_counter[step]
    step_run_id = f"step_{step:02d}_run_{run_index:03d}"

    success, command = run_step(
        step,
        batch_dir,
        step_run_id=step_run_id,
        **kwargs
    )

    after_state = collect_sample_state(batch_dir)
    snapshot_dir = write_step_snapshot(
        step=step,
        run_index=run_index,
        batch_dir=batch_dir,
        success=success,
        command=command,
        before_state=before_state,
        after_state=after_state
    )
    print(f"✓ 步骤快照已保存: {snapshot_dir}")
    if success:
        archive_terminal_failed_samples(batch_dir)
    return success


def save_checkpoint(step: int, batch_dir: str):
    """保存checkpoint"""
    checkpoint = {
        "last_step": step,
        "timestamp": datetime.now().isoformat()
    }

    checkpoint_file = Path(batch_dir) / "checkpoint.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"✓ Checkpoint已保存 (步骤 {step})")


def load_checkpoint(batch_dir: str) -> int:
    """
    加载checkpoint，返回上一步骤号
    如果没有checkpoint，返回0
    """
    checkpoint_file = Path(batch_dir) / "checkpoint.json"

    if not checkpoint_file.exists():
        return 0

    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
            last_step = checkpoint.get("last_step", 0)
            print(f"从checkpoint恢复: 上次完成步骤 {last_step}")
            return last_step
    except Exception as e:
        print(f"Warning: 无法加载checkpoint: {e}")
        return 0


def get_next_step_after(last_step: int, sample_mode: str = "core") -> int:
    """根据 checkpoint 的 last_step 计算恢复时的下一步。"""
    steps = get_steps_for_mode(sample_mode)
    for step in steps:
        if step > last_step:
            return step
    return steps[-1] + 1


def find_step4_target_not_confirmed(batch_dir: str, purity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    找出Step 4中target_confirmed=False的samples
    """
    samples = load_samples(batch_dir)
    failed = []

    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        match_type = str(
            sample.get("match_type_rule", sample.get("meta", {}).get("match_type", ""))
        ).strip().lower()
        purity_score_raw = sample.get(
            "purity_score_rule", sample.get("meta", {}).get("purity_score", 0.0)
        )
        try:
            purity_score = float(purity_score_raw)
        except (TypeError, ValueError):
            purity_score = 0.0
        target_confirmed = bool(
            sample.get(
                "meta",
                {},
            ).get(
                "target_confirmed",
                match_type == "strong" and purity_score >= purity_threshold,
            )
        )

        needs_retry = (
            (not target_confirmed) or
            (match_type != "strong") or
            (purity_score < purity_threshold)
        )

        if status == "04_completed" and needs_retry:
            failed.append(sample)

    return failed


def retry_step2_for_failed_samples(batch_dir: str, max_retries: int = 3, purity_threshold: float = 0.7) -> int:
    """
    将Step 4中target未确认的samples标记为回到Step 2重试。
    """
    failed_samples = find_step4_target_not_confirmed(batch_dir, purity_threshold=purity_threshold)

    if not failed_samples:
        return 0

    retry_count = 0
    for sample in failed_samples:
        current_retry_count = sample["meta"].get("retry_count_step2", 0)

        if current_retry_count < max_retries:
            sample["meta"]["status"] = "02_pending"
            sample["meta"]["retry_count_step2"] = current_retry_count + 1
            sample["meta"]["last_failure_reason"] = (
                "step4_not_strong_match"
                f" (target_confirmed={sample['meta'].get('target_confirmed')},"
                f" match_type={sample.get('match_type_rule', sample['meta'].get('match_type'))},"
                f" purity_score={sample.get('purity_score_rule', sample['meta'].get('purity_score'))})"
            )
            save_sample(sample, batch_dir)
            retry_count += 1
            print(f"  Sample {sample['sample_id']}: 回到Step 2重试 (第{current_retry_count + 1}次)")
        else:
            print(f"  Sample {sample['sample_id']}: 重试次数已达上限")
            sample["meta"]["flagged_for_manual_review"] = True
            save_sample(sample, batch_dir)

    return retry_count


def find_solver_valid_guard_failed_samples(
    batch_dir: str,
    solver_valid_profile: str = "off",
) -> List[Dict[str, Any]]:
    profile = str(solver_valid_profile or "off").strip().lower()
    if profile == "off":
        return []

    failed: List[Dict[str, Any]] = []
    for sample in load_samples(batch_dir):
        if sample.get("meta", {}).get("status") != "05_completed":
            continue
        solver = assess_solver_feasibility(
            origin_plan=sample.get("origin_plan", {}),
            constraints=sample.get("constraints", {}),
            edit_target_constraints=sample.get("edit_target_constraints", []),
            origin_logical_constraints=sample.get("origin_logical_constraints", []),
            effective_logical_constraints=sample.get("effective_logical_constraints", []),
            canonical_constraint_ir=sample.get("canonical_constraint_ir", {}),
            query_generation_trace=sample.get("query_generation_trace", {}),
            conflict_set=sample.get("conflict_set") or sample.get("conflict_labels"),
            primary_conflict=sample.get("primary_conflict"),
            target_bucket=(sample.get("meta") or {}).get("target_bucket") if isinstance(sample.get("meta"), dict) else None,
        )
        temporal_window = assess_temporal_window_feasibility(
            origin_plan=sample.get("origin_plan", {}),
            constraints=sample.get("constraints", {}),
            edit_target_constraints=sample.get("edit_target_constraints", []),
            canonical_constraint_ir=sample.get("canonical_constraint_ir", {}),
            query_generation_trace=sample.get("query_generation_trace", {}),
        )
        guard = assess_category_guard_validity(
            origin_plan=sample.get("origin_plan", {}),
            constraints=sample.get("constraints", {}),
            edit_target_constraints=sample.get("edit_target_constraints", []),
            origin_logical_constraints=sample.get("origin_logical_constraints", []),
            query_generation_trace=sample.get("query_generation_trace", {}),
            conflict_set=sample.get("conflict_set") or sample.get("conflict_labels"),
            primary_conflict=sample.get("primary_conflict"),
            target_bucket=(sample.get("meta") or {}).get("target_bucket") if isinstance(sample.get("meta"), dict) else None,
            solver_feasibility=solver,
            profile=profile,
        )
        should_retry = not guard.get("pass", False)
        if profile == "normal" and not solver.get("pass", False):
            should_retry = True
        if profile == "normal" and temporal_window.get("pass") is False:
            should_retry = True
        if should_retry:
            sample["_solver_valid_guard"] = {
                "solver_feasibility": solver,
                "category_guard_validity": guard,
                "temporal_window_feasibility": temporal_window,
            }
            failed.append(sample)
    return failed


def retry_step2_for_solver_valid_guard_failures(
    batch_dir: str,
    solver_valid_profile: str = "off",
    max_retries: int = 3,
) -> int:
    failed_samples = find_solver_valid_guard_failed_samples(
        batch_dir,
        solver_valid_profile=solver_valid_profile,
    )
    if not failed_samples:
        return 0

    retry_count = 0
    for sample in failed_samples:
        meta = sample.setdefault("meta", {})
        current_retry_count = int(meta.get("retry_count_solver_valid", 0) or 0)
        guard_payload = sample.pop("_solver_valid_guard", {})
        guard = guard_payload.get("category_guard_validity", {}) if isinstance(guard_payload, dict) else {}
        solver = guard_payload.get("solver_feasibility", {}) if isinstance(guard_payload, dict) else {}
        temporal_window = guard_payload.get("temporal_window_feasibility", {}) if isinstance(guard_payload, dict) else {}
        guard_errors = guard.get("errors", []) if isinstance(guard, dict) else []
        solver_status = solver.get("status") if isinstance(solver, dict) else None
        temporal_reason = temporal_window.get("reason_code") if isinstance(temporal_window, dict) else None
        temporal_failed = isinstance(temporal_window, dict) and temporal_window.get("pass") is False

        if current_retry_count < max_retries:
            meta["status"] = "02_pending"
            meta["retry_count_solver_valid"] = current_retry_count + 1
            if temporal_failed:
                meta["temporal_window_guard_failed"] = {
                    "reason_code": temporal_reason,
                    "status": temporal_window.get("status"),
                    "blocking_reasons": temporal_window.get("blocking_reasons", []),
                }
                meta["last_failure_reason"] = (
                    "temporal_window_guard_failed"
                    f" (reason_code={temporal_reason}, solver_status={solver_status})"
                )
                meta["failure_reason_code"] = "temporal_window_guard_failed"
            else:
                meta["last_failure_reason"] = (
                    "solver_valid_guard_failed"
                    f" (solver_status={solver_status}, guard_errors={guard_errors[:3]})"
                )
                meta["failure_reason_code"] = "solver_valid_guard_failed"
            save_sample(sample, batch_dir)
            retry_count += 1
            print(
                f"  Sample {sample['sample_id']}: solver-valid guard failed, "
                f"回到 Step 2 重试 (第{current_retry_count + 1}次)"
            )
        else:
            meta["flagged_for_manual_review"] = True
            if temporal_failed:
                meta["temporal_window_guard_exhausted"] = {
                    "reason_code": temporal_reason,
                    "status": temporal_window.get("status"),
                    "blocking_reasons": temporal_window.get("blocking_reasons", []),
                }
            meta["solver_valid_guard_exhausted"] = {
                "solver_status": solver_status,
                "guard_errors": guard_errors,
            }
            save_sample(sample, batch_dir)
            print(f"  Sample {sample['sample_id']}: solver-valid guard 重试次数已达上限")

    return retry_count


def retry_step2_with_new_origin(
    batch_dir: str,
    input_dir: Optional[str],
    templates_path: Optional[str],
    origin_query_lookup: Optional[Dict[str, Dict[str, Any]]],
) -> int:
    if not input_dir or not templates_path:
        return 0
    if not origin_query_lookup:
        raise ValueError("origin_query_lookup is required for origin swaps")

    templates_index, template_config = _load_template_bucket_index(templates_path)
    if not templates_index:
        return 0

    samples = load_samples(batch_dir)
    swapped = 0
    from evaluation.benchmark.hard_truth import (
        build_origin_logical_constraints,
        serialize_logical_constraints,
    )
    for sample in samples:
        if sample.get("meta", {}).get("status") != "02_failed_retryable_new_origin":
            continue

        recovery = sample.get("meta", {}).get("step2_recovery", {})
        if not isinstance(recovery, dict):
            recovery = {}
        used_origin_plan_files = set()
        for item in recovery.get("used_origin_plan_files", []):
            if isinstance(item, str) and item:
                used_origin_plan_files.add(item)

        target_bucket = tuple(sample.get("meta", {}).get("target_bucket", []))
        target_city = sample.get("meta", {}).get("target_city")
        day_count = int(sample.get("meta", {}).get("day_count", 0))
        is_single_day = bool(sample.get("meta", {}).get("is_single_day", False))
        preferred_start_city = sample.get("meta", {}).get("start_city")

        replacement = None
        replacement_query = None
        replacement_logical_constraints = None
        while True:
            replacement = select_replacement_origin_plan(
                input_dir=input_dir,
                target_bucket=target_bucket,
                target_city=target_city,
                day_count=day_count,
                is_single_day=is_single_day,
                used_origin_plan_files=used_origin_plan_files,
                preferred_start_city=preferred_start_city,
                is_bucket_applicable_fn=_is_bucket_applicable_for_plan,
                templates_index=templates_index,
                template_config=template_config,
            )
            if replacement is None:
                break
            replacement_query = origin_query_lookup.get(replacement["origin_plan_file"])
            if isinstance(replacement_query, dict):
                replacement_logical_constraints = serialize_logical_constraints(
                    build_origin_logical_constraints(replacement_query)
                )
                replacement_with_query = dict(replacement)
                replacement_with_query["origin_logical_constraints"] = replacement_logical_constraints
                if _is_bucket_applicable_for_plan(
                    target_bucket,
                    replacement_with_query,
                    templates_index,
                    template_config,
                ):
                    break
            used_origin_plan_files.add(replacement["origin_plan_file"])
            print(
                f"  Sample {sample['sample_id']}: replacement plan missing compatible origin query/budget,"
                f" skip {replacement['origin_plan_file']}"
            )
            replacement = None

        if replacement is None:
            sample["meta"]["status"] = "02_failed_exhausted"
            sample["meta"]["failure_reason"] = sample.get("meta", {}).get("failure_reason", "No replacement origin plan available")
            sample["meta"]["failure_reason_code"] = sample.get("meta", {}).get("failure_reason_code", "origin_plan_exhausted")
            sample["meta"]["last_step"] = "02_generate_query"
            save_sample(sample, batch_dir)
            continue

        _clear_sample_after_origin_swap(sample)
        sample["origin_plan"] = replacement["origin_plan"]
        sample["origin_query_structured"] = replacement_query
        sample["origin_query_text"] = extract_origin_query_text(replacement_query)
        sample["origin_logical_constraints"] = replacement_logical_constraints or serialize_logical_constraints(
            build_origin_logical_constraints(replacement_query)
        )
        sample["meta"]["origin_plan_file"] = replacement["origin_plan_file"]
        sample["meta"]["start_city"] = replacement["start_city"]
        sample["meta"]["target_city"] = replacement["target_city"]
        sample["meta"]["day_count"] = replacement["day_count"]
        sample["meta"]["is_single_day"] = replacement["is_single_day"]
        sample["meta"]["status"] = "02_pending"
        sample["meta"]["last_step"] = "02_generate_query"
        sample["meta"]["failure_reason"] = ""
        sample["meta"]["failure_reason_code"] = ""
        step2_recovery = sample["meta"].get("step2_recovery", {})
        if not isinstance(step2_recovery, dict):
            step2_recovery = {}
        sample["meta"]["step2_recovery"] = step2_recovery
        if replacement["origin_plan_file"] not in step2_recovery.get("used_origin_plan_files", []):
            step2_recovery.setdefault("used_origin_plan_files", []).append(replacement["origin_plan_file"])
        save_sample(sample, batch_dir)
        swapped += 1
        print(f"  Sample {sample['sample_id']}: 替换 origin_plan -> {replacement['origin_plan_file']}")

    return swapped


def resolve_step2_origin_swaps(
    batch_dir: str,
    step_run_counter: Dict[int, int],
    **kwargs,
) -> bool:
    while True:
        swapped = retry_step2_with_new_origin(
            batch_dir=batch_dir,
            input_dir=kwargs.get("input_dir"),
            templates_path=kwargs.get("templates_path"),
            origin_query_lookup=kwargs.get("origin_query_lookup"),
        )
        if swapped == 0:
            return True
        print(f"\n发现 {swapped} 个 samples 需要替换 origin_plan，重新执行 Step 2...")
        success = execute_step_with_snapshot(2, batch_dir, step_run_counter, **kwargs)
        if not success:
            return False
        save_checkpoint(2, batch_dir)


def has_pending_samples(batch_dir: str) -> Tuple[bool, int]:
    """
    检查是否有pending状态的samples需要处理

    Returns:
        (是否有pending, 最早的pending步骤号)
    """
    samples = load_samples(batch_dir)

    for sample in samples:
        if sample.get("meta", {}).get("status") == "02_pending":
            return True, 2

    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        if "_pending" in status:
            try:
                step_num = int(status.split("_")[0])
                return True, step_num
            except Exception:
                pass

    return False, 0


def run_remaining_steps(from_step: int,
                        batch_dir: str,
                        step_run_counter: Dict[int, int],
                        **kwargs) -> bool:
    """
    从指定步骤开始执行后续步骤。
    """
    sample_mode = str(kwargs.get("sample_mode", "core")).strip().lower()
    steps = get_steps_for_mode(sample_mode)

    if from_step in steps:
        steps_to_run = [step for step in steps if step >= from_step]
    else:
        print(f"未知的起始步骤: {from_step}")
        return False

    print(f"\n重新执行步骤: {' → '.join(map(str, steps_to_run))}")

    all_success = True
    for step in steps_to_run:
        success = execute_step_with_snapshot(step, batch_dir, step_run_counter, **kwargs)
        if not success:
            print(f"警告: 步骤 {step} 执行失败，继续下一步")
            all_success = False
            continue
        save_checkpoint(step, batch_dir)
        if step == 2:
            if not resolve_step2_origin_swaps(batch_dir, step_run_counter, **kwargs):
                print("警告: Step 2 origin swap recovery failed")
                all_success = False
        if step == 5:
            retry_count = retry_step2_for_solver_valid_guard_failures(
                batch_dir,
                solver_valid_profile=kwargs.get("solver_valid_profile", "off"),
                max_retries=int(kwargs.get("max_retries", 3) or 3),
            )
            if retry_count > 0:
                print(f"\n发现{retry_count}个 samples 未通过 solver-valid guard，回到 Step 2 重试...")
                if not run_remaining_steps(2, batch_dir, step_run_counter, **kwargs):
                    all_success = False

    return all_success


def generate_report(batch_dir: str, templates_path: Optional[str] = None):
    """
    生成pipeline执行报告
    """
    samples = load_samples(batch_dir)

    total_samples = len(samples)
    archived_samples = 0
    archived_status_counts: Dict[str, int] = {}
    failed_root = Path(batch_dir) / "_failed"
    if failed_root.exists():
        for sample_file in failed_root.rglob("sample_*.json"):
            archived_samples += 1
            try:
                sample = json.loads(sample_file.read_text(encoding="utf-8"))
                status = sample.get("meta", {}).get("status", sample_file.parent.name)
            except Exception:
                status = sample_file.parent.name
            archived_status_counts[status] = archived_status_counts.get(status, 0) + 1

    status_counts: Dict[str, int] = {}
    retry_count_step2 = 0
    manual_review_count = 0
    validation_passed_count = 0
    target_confirmed_count = 0
    factually_valid_count = 0
    factually_invalid_count = 0
    surface_source_distribution: Dict[str, int] = {}
    template_variant_distribution: Dict[str, int] = {}
    constraint_type_pass_distribution: Dict[str, int] = {}
    distinct_queries = set()
    query_count = 0

    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

        retry_count_step2 += sample["meta"].get("retry_count_step2", 0)

        if sample["meta"].get("flagged_for_manual_review", False):
            manual_review_count += 1

        if sample["meta"].get("target_confirmed", False):
            target_confirmed_count += 1

        edit_query = sample.get("edit_query")
        if isinstance(edit_query, str) and edit_query.strip():
            query_count += 1
            distinct_queries.add(edit_query.strip())

        trace = sample.get("query_generation_trace", {})
        if isinstance(trace, dict):
            surface_source = str(trace.get("query_surface_source", "") or "").strip()
            if surface_source:
                surface_source_distribution[surface_source] = surface_source_distribution.get(surface_source, 0) + 1
            template_id = str(trace.get("template_id", "") or "").strip()
            if template_id:
                template_variant_distribution[template_id] = template_variant_distribution.get(template_id, 0) + 1

        checks = sample.get("checks", {})
        if checks.get("all_pass", False):
            validation_passed_count += 1
            if isinstance(trace, dict):
                constraint_type = str(trace.get("constraint_type", "") or "").strip()
                if constraint_type:
                    constraint_type_pass_distribution[constraint_type] = constraint_type_pass_distribution.get(constraint_type, 0) + 1
        factual_check = checks.get("query_factual_validity", {})
        if isinstance(factual_check, dict):
            label = str(factual_check.get("label", "") or "").strip()
            if label == "factually_valid":
                factually_valid_count += 1
            elif label == "factually_invalid":
                factually_invalid_count += 1

    report = {
        "batch_dir": batch_dir,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total_samples,
        "archived_failed_samples": archived_samples,
        "archived_failed_status_distribution": archived_status_counts,
        "status_distribution": status_counts,
        "target_confirmed": target_confirmed_count,
        "target_confirmed_rate": target_confirmed_count / total_samples if total_samples > 0 else 0,
        "validation_passed": validation_passed_count,
        "validation_passed_rate": validation_passed_count / total_samples if total_samples > 0 else 0,
        "factually_valid_queries": factually_valid_count,
        "factually_valid_rate": factually_valid_count / total_samples if total_samples > 0 else 0,
        "factually_invalid_queries": factually_invalid_count,
        "factually_invalid_rate": factually_invalid_count / total_samples if total_samples > 0 else 0,
        "total_retries_step2": retry_count_step2,
        "manual_review_count": manual_review_count,
        "surface_source_distribution": surface_source_distribution,
        "template_variant_distribution": template_variant_distribution,
        "distinct_query_rate": len(distinct_queries) / query_count if query_count > 0 else 0,
        "constraint_type_pass_distribution": constraint_type_pass_distribution,
    }

    validation_split = export_validation_split(batch_dir)
    report["validation_split"] = {
        "root": validation_split["root"],
        "passed_dir": validation_split["passed_dir"],
        "failed_dir": validation_split["failed_dir"],
        "passed_count": validation_split["passed_count"],
        "failed_count": validation_split["failed_count"],
    }
    compact_curated = export_compact_curated(batch_dir, validation_split)
    report["compact_curated"] = compact_curated
    if templates_path:
        try:
            funnel_report = write_constraint_type_funnel(batch_dir, templates_path)
            report["constraint_type_funnel"] = {
                "json_path": funnel_report.get("json_path"),
                "csv_path": funnel_report.get("csv_path"),
            }
        except Exception as exc:
            report["constraint_type_funnel"] = {"error": str(exc)}

    report_file = Path(batch_dir) / "pipeline_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Pipeline执行报告")
    print(f"{'='*60}")
    print(f"总样本数: {total_samples}")
    print(f"归档失败样本: {archived_samples}")
    print(f"Target确认: {target_confirmed_count} ({report['target_confirmed_rate']:.1%})")
    print(f"验证通过: {validation_passed_count} ({report['validation_passed_rate']:.1%})")
    print(f"事实有效 query: {factually_valid_count} ({report['factually_valid_rate']:.1%})")
    print(f"事实无效 query: {factually_invalid_count} ({report['factually_invalid_rate']:.1%})")
    print(f"Step 2重试次数: {retry_count_step2}")
    print(f"需要人工审查: {manual_review_count}")
    print(f"验证通过目录: {validation_split['passed_dir']} ({validation_split['passed_count']})")
    print(f"验证未通过目录: {validation_split['failed_dir']} ({validation_split['failed_count']})")
    print(f"精简数据集目录: {compact_curated['passed_dir']} ({compact_curated['exported_count']})")
    print(f"精简数据集 JSONL: {compact_curated['dataset_jsonl']}")
    funnel_meta = report.get("constraint_type_funnel", {})
    if isinstance(funnel_meta, dict) and funnel_meta.get("json_path"):
        print(f"类型漏斗报告 JSON: {funnel_meta['json_path']}")
        print(f"类型漏斗报告 CSV: {funnel_meta['csv_path']}")
    print(f"\n报告已保存到: {report_file}")


def run_pipeline(batch_dir: str,
                 batch_size: int = 100,
                 variants_per_plan: int = 1,
                 max_retries: int = 3,
                 config_path: Optional[str] = None,
                 templates_path: Optional[str] = None,
                 input_dir: Optional[str] = None,
                 query_input: Optional[str] = None,
                 model: str = "dmxapi",
                 step2_seed: Optional[int] = None,
                 step2_llm_temperature: Optional[float] = None,
                 step4_purity_threshold: float = 0.7,
                 start_step: int = 1,
                 step1_sampling_mode: str = "weighted",
                 step1_allow_violations: Optional[str] = None,
                 step1_bucket_whitelist: Optional[str] = None,
                 constraint_type_whitelist: Optional[str] = None,
                 sample_mode: str = "core",
                 solver_valid_profile: str = "off") -> bool:
    """
    运行完整pipeline。
    """
    batch_path = normalize_batch_dir(batch_dir)
    batch_path.mkdir(parents=True, exist_ok=True)
    batch_dir_abs = str(batch_path)

    print(f"\n{'='*60}")
    print("数据生成Pipeline启动")
    print(f"{'='*60}")
    print(f"输出目录: {batch_dir_abs}")
    print(f"Batch大小: {batch_size}")
    print(f"单个Origin最多复用次数: {max(int(variants_per_plan or 1), 1)}")
    print(f"最大重试次数: {max_retries}")
    print(f"LLM Provider: {model}")
    if step2_seed is not None:
        print(f"Step2 seed: {step2_seed}")
    if step2_llm_temperature is not None:
        print(f"Step2 temperature: {step2_llm_temperature}")
    print(f"Step4 purity阈值: {step4_purity_threshold}")
    print(f"Origin Query目录: {query_input}")
    print(f"Step1采样模式: {step1_sampling_mode}")
    if step1_allow_violations:
        print(f"Step1 violation白名单: {step1_allow_violations}")
    if step1_bucket_whitelist:
        print(f"Step1 bucket白名单: {step1_bucket_whitelist}")
    if constraint_type_whitelist:
        print(f"Constraint type白名单: {constraint_type_whitelist}")
    print(f"Sample模式: {sample_mode}")
    print(f"Solver-valid profile: {solver_valid_profile}")
    print(f"起始步骤: {start_step}")

    data_gen_dir = Path(__file__).resolve().parent.parent
    config_default = data_gen_dir / "config" / "bucket_distribution.yaml"
    templates_default = data_gen_dir / "config" / "templates.yaml"

    config_path_abs = resolve_config_path(config_path, config_default)
    templates_path_abs = resolve_config_path(templates_path, templates_default)
    if not query_input:
        raise ValueError("query_input is required")
    query_lookup = load_origin_query_lookup(query_input)
    if not query_lookup:
        raise ValueError("No origin queries loaded from query_input")

    kwargs = {
        "batch_size": batch_size,
        "variants_per_plan": max(int(variants_per_plan or 1), 1),
        "config_path": config_path_abs,
        "templates_path": templates_path_abs,
        "input_dir": input_dir,
        "query_input": query_input,
        "origin_query_lookup": query_lookup,
        "model": model,
        "step2_seed": step2_seed,
        "step2_llm_temperature": step2_llm_temperature,
        "step4_purity_threshold": step4_purity_threshold,
        "max_retries": max_retries,
        "step1_sampling_mode": step1_sampling_mode,
        "step1_allow_violations": step1_allow_violations,
        "step1_bucket_whitelist": step1_bucket_whitelist,
        "constraint_type_whitelist": constraint_type_whitelist,
        "sample_mode": sample_mode,
        "solver_valid_profile": solver_valid_profile,
    }

    step_run_counter: Dict[int, int] = {}

    steps = get_steps_for_mode(sample_mode)

    # 阶段1：主流程
    print(f"\n{'='*60}")
    print(f"阶段1: 执行主流程（步骤 {' → '.join(map(str, steps))}）")
    print(f"{'='*60}")

    steps_to_run = [step for step in steps if step >= start_step]
    if steps_to_run and steps_to_run[0] == 1:
        cleanup_stale_outputs_for_step1(batch_dir_abs)

    if not steps_to_run:
        print("主流程步骤已全部完成，跳过阶段1。")

    for step in steps_to_run:
        success = execute_step_with_snapshot(step, batch_dir_abs, step_run_counter, **kwargs)
        if not success:
            print(f"错误: 步骤 {step} 执行失败，终止pipeline")
            return False

        save_checkpoint(step, batch_dir_abs)
        if step == 2:
            if not resolve_step2_origin_swaps(batch_dir_abs, step_run_counter, **kwargs):
                print("错误: Step 2 origin swap recovery failed")
                return False

        if step == 4:
            failed_count = retry_step2_for_failed_samples(
                batch_dir_abs,
                max_retries=max_retries,
                purity_threshold=step4_purity_threshold
            )
            if failed_count > 0:
                print(f"\n发现{failed_count}个samples的target未确认，回到Step 2重试...")

                if not run_remaining_steps(2, batch_dir_abs, step_run_counter, **kwargs):
                    print("警告: 重试执行失败")
                    return False

                failed_count = retry_step2_for_failed_samples(
                    batch_dir_abs,
                    max_retries=max_retries,
                    purity_threshold=step4_purity_threshold
                )
                if failed_count > 0:
                    print(f"\n仍有{failed_count}个samples的target未确认，进入下一轮迭代...")

        if step == 5:
            guard_failed_count = retry_step2_for_solver_valid_guard_failures(
                batch_dir_abs,
                solver_valid_profile=solver_valid_profile,
                max_retries=max_retries,
            )
            if guard_failed_count > 0:
                print(f"\n发现{guard_failed_count}个 samples 未通过 solver-valid guard，回到 Step 2 重试...")
                if not run_remaining_steps(2, batch_dir_abs, step_run_counter, **kwargs):
                    print("警告: solver-valid guard 重试执行失败")
                    return False

    # 阶段2：迭代检查
    print(f"\n\n{'='*60}")
    print("阶段2: 迭代检查Step 4的target确认情况")
    print(f"{'='*60}")

    for iteration in range(max_retries):
        failed_samples = find_step4_target_not_confirmed(
            batch_dir_abs,
            purity_threshold=step4_purity_threshold
        )

        if not failed_samples:
            print("\n✓ 所有samples的target都已确认！")
            break

        print(f"\n迭代 {iteration + 1}/{max_retries}: 发现{len(failed_samples)}个samples的target未确认")

        retry_count = retry_step2_for_failed_samples(
            batch_dir_abs,
            max_retries=max_retries,
            purity_threshold=step4_purity_threshold
        )

        if retry_count == 0:
            print("所有samples重试次数已达上限")
            break

        if not run_remaining_steps(2, batch_dir_abs, step_run_counter, **kwargs):
            print("警告: 重试执行失败")
            return False

    # 阶段3：报告
    print(f"\n\n{'='*60}")
    print("阶段3: 生成执行报告")
    print(f"{'='*60}")
    generate_report(batch_dir_abs, templates_path=kwargs.get("templates_path"))

    print(f"\n{'='*60}")
    print("Pipeline执行完成！")
    print(f"{'='*60}")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据生成Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行数据生成 pipeline
  python run_pipeline.py --input /path/to/origin_plans --query-input /path/to/origin_query --output data/batches/batch_001 --batch-size 100

  # 从checkpoint恢复
  python run_pipeline.py --query-input /path/to/origin_query --output data/batches/batch_001 --resume
        """
    )

    parser.add_argument("--input", help="Origin plans目录（首次运行或从Step1恢复需要）")
    parser.add_argument("--query-input", required=True, help="Origin query目录（必需）")
    parser.add_argument("--output", required=True, help="Batch输出目录")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch大小（默认100）")
    parser.add_argument(
        "--variants-per-plan",
        type=int,
        default=1,
        help="Step1 中同一个 origin plan 最多复用次数（默认1）",
    )
    parser.add_argument("--resume", action="store_true", help="从checkpoint恢复")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数（默认3）")
    parser.add_argument("--step4-purity-threshold", type=float, default=0.7,
                        help="Step4强一致 purity 阈值（默认0.7）")
    parser.add_argument("--config", help="Bucket分布配置文件路径")
    parser.add_argument("--templates", help="Templates配置文件路径")
    parser.add_argument("--model", default="dmxapi", choices=["auto", "siliconcloud", "dmxapi"],
                        help="LLM provider for steps 2/4/5 (default: dmxapi)")
    parser.add_argument(
        "--step2-seed",
        type=int,
        default=None,
        help="Random seed for Step2 query generation (for reproducibility)",
    )
    parser.add_argument(
        "--step2-llm-temperature",
        type=float,
        default=None,
        help="LLM temperature override for Step2",
    )
    parser.add_argument(
        "--sample-mode",
        default="core",
        choices=["core"],
        help="Retained for compatibility; the artifact pipeline uses the single core flow.",
    )
    parser.add_argument(
        "--step1-sampling-mode",
        default="weighted",
        choices=["weighted", "uniform"],
        help="Step1 bucket sampling mode (default: weighted)",
    )
    parser.add_argument(
        "--step1-allow-violations",
        default=None,
        help="Optional comma-separated violation whitelist for Step1",
    )
    parser.add_argument(
        "--step1-bucket-whitelist",
        default=None,
        help="Optional comma-separated bucket names for Step1 (used with --step1-sampling-mode uniform)",
    )
    parser.add_argument(
        "--constraint-type-whitelist",
        default=None,
        help="Optional comma-separated constraint_type whitelist passed to Step1 and Step2",
    )
    parser.add_argument(
        "--constraint-type-run-mode",
        default="mixed",
        choices=["mixed", "by_constraint_type"],
        help="mixed: existing mixed-batch flow; by_constraint_type: run one core batch per constraint_type",
    )
    parser.add_argument(
        "--solver-valid-profile",
        default="off",
        choices=["off", "normal", "intentional_infeasible"],
        help="off: legacy behavior; normal: require static solver feasibility and category guards; intentional_infeasible: keep conflict-valid infeasible samples",
    )

    args = parser.parse_args()

    batch_path = normalize_batch_dir(args.output)
    input_dir_abs = str(resolve_user_path(args.input)) if args.input else None
    query_input_abs = str(resolve_user_path(args.query_input))

    start_step = 1

    if args.constraint_type_run_mode == "by_constraint_type" and args.resume:
        print("错误: by_constraint_type 模式暂不支持 --resume")
        sys.exit(1)

    if args.resume:
        print("恢复模式: 检查checkpoint...")
        if not batch_path.exists():
            print(f"错误: 恢复模式下输出目录不存在: {batch_path}")
            sys.exit(1)

        last_step = load_checkpoint(str(batch_path))
        if last_step == 0:
            print("未找到checkpoint，将从头开始")
        start_step = get_next_step_after(last_step, sample_mode=args.sample_mode)

        if start_step <= 1 and not input_dir_abs:
            print("错误: 从头开始需要--input参数指定origin plans目录")
            sys.exit(1)
    else:
        if not batch_path.exists():
            if not input_dir_abs:
                print("错误: 首次运行需要--input参数指定origin plans目录")
                sys.exit(1)
        else:
            existing_samples = list(batch_path.glob("sample_*.json"))
            if existing_samples:
                print(f"警告: Batch目录已存在 {len(existing_samples)} 个samples")
                print("如要恢复运行，请使用--resume参数")
                print("如要重新生成，请删除或重命名batch目录")
                sys.exit(1)
            if not input_dir_abs:
                print("错误: Batch目录为空，需要--input参数指定origin plans目录")
                sys.exit(1)

    if args.constraint_type_run_mode == "by_constraint_type":
        success = run_pipeline_by_constraint_type(
            output_root=str(batch_path),
            batch_size=args.batch_size,
            variants_per_plan=args.variants_per_plan,
            max_retries=args.max_retries,
            config_path=args.config,
            templates_path=args.templates,
            input_dir=input_dir_abs,
            query_input=query_input_abs,
            model=args.model,
            step2_seed=args.step2_seed,
            step2_llm_temperature=args.step2_llm_temperature,
            step4_purity_threshold=max(0.0, min(1.0, args.step4_purity_threshold)),
            step1_sampling_mode=args.step1_sampling_mode,
            step1_allow_violations=args.step1_allow_violations,
            constraint_type_whitelist=args.constraint_type_whitelist,
            sample_mode=args.sample_mode,
            solver_valid_profile=args.solver_valid_profile,
        )
    else:
        success = run_pipeline(
            batch_dir=str(batch_path),
            batch_size=args.batch_size,
            variants_per_plan=args.variants_per_plan,
            max_retries=args.max_retries,
            config_path=args.config,
            templates_path=args.templates,
            input_dir=input_dir_abs,
            query_input=query_input_abs,
            model=args.model,
            step2_seed=args.step2_seed,
            step2_llm_temperature=args.step2_llm_temperature,
            step4_purity_threshold=max(0.0, min(1.0, args.step4_purity_threshold)),
            start_step=start_step,
            step1_sampling_mode=args.step1_sampling_mode,
            step1_allow_violations=args.step1_allow_violations,
            step1_bucket_whitelist=args.step1_bucket_whitelist,
            constraint_type_whitelist=args.constraint_type_whitelist,
            sample_mode=args.sample_mode,
            solver_valid_profile=args.solver_valid_profile,
        )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
