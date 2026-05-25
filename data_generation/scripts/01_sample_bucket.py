#!/usr/bin/env python3
"""
Step 1: Sample Bucket
读取origin plans，采样目标conflict buckets，创建sample skeletons

这是数据生成pipeline的第一步，为每个origin plan分配目标conflict bucket。
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GEN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(DATA_GEN_DIR, "utils"))
PROJECT_ROOT = os.path.dirname(DATA_GEN_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from bucket_sampler import BucketSampler
from plan_parser import load_origin_plan
from conflict_feasibility import build_origin_plan_evidence as build_origin_plan_evidence_for_plan
from soft_constraint_control import (
    build_origin_soft_profile,
    build_soft_control_statistics,
    choose_soft_control_target,
    load_soft_control_config,
)
from evaluation.benchmark.hard_truth import (
    build_origin_logical_constraints,
    serialize_logical_constraints,
)
from evaluation.benchmark.soft_truth import (
    build_origin_preference_tags,
    compute_origin_preference_score_snapshot,
    serialize_preference_snapshots,
    serialize_preference_tags,
)

DEFAULT_ALLOWED_EDIT_VIOLATIONS = {"overflow", "overlap", "discontinuity", "incompatibility"}


class MissingOriginQueryError(RuntimeError):
    """当 origin plan 无法匹配到同名 origin query 时抛出。"""


def parse_allowed_violations(arg_value: str) -> set:
    """解析允许通过 edit 过滤的 violation 集合。"""
    if not isinstance(arg_value, str):
        return set(DEFAULT_ALLOWED_EDIT_VIOLATIONS)
    parsed = {item.strip().lower() for item in arg_value.split(",") if item.strip()}
    return parsed or set(DEFAULT_ALLOWED_EDIT_VIOLATIONS)


def parse_constraint_type_whitelist(raw: Optional[str]) -> Optional[Set[str]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    parsed = {item.strip() for item in raw.split(",") if item.strip()}
    return parsed or None


def load_origin_queries(query_dir: str) -> Dict[str, Dict[str, Any]]:
    """加载 origin query 目录下所有 JSON（按文件名索引）。"""
    query_path = Path(query_dir)
    if not query_path.exists() or not query_path.is_dir():
        raise ValueError(f"Origin query directory not found: {query_dir}")

    query_lookup: Dict[str, Dict[str, Any]] = {}
    duplicate_files = 0

    for json_file in sorted(query_path.rglob("*.json")):
        file_name = json_file.name
        if file_name in query_lookup:
            duplicate_files += 1
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                query_lookup[file_name] = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load query {json_file}: {e}")

    print(f"Loaded {len(query_lookup)} origin queries from {query_dir}")
    if duplicate_files > 0:
        print(f"  Warning: ignored {duplicate_files} duplicate query files by filename")
    return query_lookup


def extract_origin_query_text(origin_query: Dict[str, Any]) -> str:
    """从 origin query 结构中提取自然语言文本。"""
    if not isinstance(origin_query, dict):
        return ""
    for key in ("nature_language", "natural_language", "query", "text"):
        value = origin_query.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _bucket_has_whitelisted_constraint_type(
    bucket_cfg: Dict[str, Any],
    constraint_type_whitelist: Optional[Set[str]],
) -> bool:
    if not constraint_type_whitelist:
        return True
    templates = bucket_cfg.get("templates", []) if isinstance(bucket_cfg, dict) else []
    if not isinstance(templates, list):
        return False
    for template in templates:
        if not isinstance(template, dict):
            continue
        constraint_type = str(template.get("constraint_type", "")).strip()
        if constraint_type in constraint_type_whitelist:
            return True
    return False


def _filter_bucket_cfg_by_constraint_type(
    bucket_cfg: Dict[str, Any],
    constraint_type_whitelist: Optional[Set[str]],
) -> Dict[str, Any]:
    if not isinstance(bucket_cfg, dict):
        return {}
    if not constraint_type_whitelist:
        return dict(bucket_cfg)

    templates = bucket_cfg.get("templates", [])
    filtered_templates = []
    if isinstance(templates, list):
        for template in templates:
            if not isinstance(template, dict):
                continue
            constraint_type = str(template.get("constraint_type", "")).strip()
            if constraint_type in constraint_type_whitelist:
                filtered_templates.append(dict(template))

    filtered_cfg = dict(bucket_cfg)
    filtered_cfg["templates"] = filtered_templates
    return filtered_cfg


def load_template_buckets(
    templates_path: str,
    allowed_violations: set,
    constraint_type_whitelist: Optional[Set[str]] = None,
) -> tuple:
    """
    解析templates.yaml，提取所有定义的bucket tuples和完整配置

    Args:
        templates_path: templates.yaml文件路径

    Returns:
        (template_buckets, template_config) 元组
        - template_buckets: 定义的bucket tuple集合
        - template_config: 完整的模板配置字典
    """
    template_buckets = set()
    template_config = {}

    try:
        import yaml
        import re
        with open(templates_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        excluded_count = 0

        # 提取所有顶层key（每个key是一个bucket tuple）
        for key in config.keys():
            # 跳过非bucket的配置项
            if key in ["poi_source_types", "city_constraint_types", "template_instructions"]:
                continue

            # 提取bucket tuple
            bucket_tuple = None
            if isinstance(key, str):
                # YAML将tuple解析为字符串，例如 "(parameter, temporal, overflow)"
                # 使用正则提取tuple内容
                match = re.match(r'\(([^)]+)\)', key)
                if match:
                    parts = [p.strip().strip('"\'') for p in match.group(1).split(',')]
                    if len(parts) == 3:
                        bucket_tuple = tuple(parts)
            elif isinstance(key, list) or isinstance(key, tuple):
                # 如果YAML保留了列表/元组格式
                if len(key) == 3:
                    bucket_tuple = tuple(key)

            if not bucket_tuple:
                excluded_count += 1
                continue

            bucket_cfg = config[key]
            violation = bucket_tuple[2]
            eligible_for_edit = bool(bucket_cfg.get("eligible_for_edit", False))

            # 仅保留可用于 edit 的模板格子
            if violation not in allowed_violations or not eligible_for_edit:
                excluded_count += 1
                continue
            if not _bucket_has_whitelisted_constraint_type(bucket_cfg, constraint_type_whitelist):
                excluded_count += 1
                continue

            bucket_cfg = _filter_bucket_cfg_by_constraint_type(bucket_cfg, constraint_type_whitelist)
            template_buckets.add(bucket_tuple)
            template_config[bucket_tuple] = bucket_cfg

        print(f"  Loaded {len(template_buckets)} edit-eligible template buckets from {templates_path}")
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} buckets (ineligible or unsupported violation)")
        return template_buckets, template_config

    except Exception as e:
        print(f"  Warning: Failed to load templates: {e}")
        import traceback
        traceback.print_exc()
        return set(), {}


def is_bucket_applicable(bucket: tuple, plan_meta: dict, templates_index: set,
                        template_config: dict = None) -> bool:
    """
    检查bucket是否适用于当前plan

    Args:
        bucket: bucket tuple (scope, dimension, violation)
        plan_meta: plan元数据（包含target_city, is_single_day, origin_plan等）
        templates_index: 定义的模板bucket集合
        template_config: 完整的templates配置（用于检查依赖）

    Returns:
        是否适用
    """
    # 1. 检查是否有对应模板
    if bucket not in templates_index:
        return False

    # 2. 如果提供了template_config，检查特殊约束
    if template_config and bucket in template_config:
        bucket_config = template_config[bucket]
        templates = bucket_config.get("templates", [])

        # 只要 bucket 内存在一个对当前 plan 真正可用的模板，就认为 bucket applicable。
        for template in templates:
            if _template_is_applicable_for_plan(template, plan_meta):
                return True
        return False

    return True


def _check_plan_has_pois(origin_plan: dict) -> bool:
    """
    检查origin plan中是否有attraction或restaurant类型的活动

    Args:
        origin_plan: origin plan字典

    Returns:
        是否有POI（attraction或restaurant）
    """
    itinerary = origin_plan.get("itinerary", [])

    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        for activity in activities:
            act_type = activity.get("type", "")
            if act_type in ["attraction", "restaurant"]:
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
        if str(item.get("type", "")).strip().lower() != "budget_total":
            continue
        value = item.get("value")
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _min_numeric_option(values: Any) -> Optional[float]:
    if not isinstance(values, (list, tuple)):
        return None
    candidates: List[float] = []
    for value in values:
        try:
            candidates.append(float(value))
        except (TypeError, ValueError):
            continue
    return min(candidates) if candidates else None


def _template_is_applicable_for_plan(template: Dict[str, Any], plan_meta: Dict[str, Any]) -> bool:
    origin_plan = plan_meta.get("origin_plan", {}) if isinstance(plan_meta.get("origin_plan"), dict) else {}
    origin_logical_constraints = plan_meta.get("origin_logical_constraints")
    poi_source = str(template.get("poi_source", "") or "").strip()
    if poi_source == "existing_attractions" and not _check_plan_has_pois(origin_plan):
        return False

    constraint_type = str(template.get("constraint_type", "") or "").strip()
    params = template.get("params", {}) if isinstance(template.get("params"), dict) else {}
    total_cost, ticket_cost = _calculate_plan_costs(origin_plan)
    has_origin_budget = _origin_has_budget_requirement(origin_logical_constraints)
    origin_budget_total = _extract_origin_budget_total(origin_logical_constraints)
    evidence = build_origin_plan_evidence_for_plan(origin_plan)

    if constraint_type in {"budget_cap_preference", "budget_target_update"}:
        return origin_budget_total is not None and total_cost > origin_budget_total

    if constraint_type == "budget_limit":
        threshold = _min_numeric_option(params.get("budget", []))
        if threshold is None:
            return total_cost > 0
        return total_cost > threshold

    if constraint_type == "ticket_budget_limit":
        threshold = _min_numeric_option(params.get("budget", []))
        if threshold is None:
            return ticket_cost > 0
        return ticket_cost > threshold

    if constraint_type == "max_distance_constraint":
        return _check_plan_has_pois(origin_plan)

    if constraint_type == "multi_day_requirement":
        min_day_count = _min_numeric_option(params.get("day_count", []))
        if min_day_count is None:
            return bool(plan_meta.get("is_single_day", False)) or int(plan_meta.get("day_count", 0) or 0) < 2
        return int(plan_meta.get("day_count", 0) or 0) < int(min_day_count)

    if constraint_type == "activity_budget_limit":
        threshold = _min_numeric_option(params.get("budget", []))
        if threshold is None:
            return True
        meal_cost, accommodation_cost = _calculate_plan_activity_costs(origin_plan)
        template_text = " ".join(
            str(part)
            for part in (template.get("template", ""), template.get("description", ""))
            if part
        )
        if any(token in template_text for token in ["餐", "餐饮", "用餐"]):
            return meal_cost > threshold
        if any(token in template_text for token in ["住宿", "酒店", "房"]):
            return accommodation_cost > threshold
        return meal_cost > threshold or accommodation_cost > threshold

    if constraint_type == "ticket_price_cap_or_substitute":
        return _check_plan_has_pois(origin_plan) and ticket_cost > 0

    if constraint_type == "resource_overlap":
        threshold = _min_numeric_option(params.get("budget", []))
        return has_origin_budget and _check_plan_has_pois(origin_plan) and (threshold is None or total_cost > threshold)

    if constraint_type == "multi_day_budget_overflow":
        target_day_count = _min_numeric_option(params.get("day_count", []))
        current_day_count = int(plan_meta.get("day_count", 0) or 0)
        if target_day_count is not None and current_day_count >= int(target_day_count):
            return False
        if not bool(evidence.get("budget_baseline_reliable_for_multi_day", False)):
            return False
        return float(evidence.get("transferable_local_cost", 0.0) or 0.0) > 0.0

    return True


def expand_origin_plans_for_batch(
    origin_plans: List[Dict[str, Any]],
    *,
    batch_size: int,
    variants_per_plan: int = 1,
) -> List[Dict[str, Any]]:
    """Expand origin plans for batch generation while tracking reuse variants."""
    if batch_size <= 0 or not origin_plans:
        return []
    variants = max(int(variants_per_plan or 1), 1)
    expanded: List[Dict[str, Any]] = []
    total_plans = len(origin_plans)
    for index in range(batch_size):
        plan = dict(origin_plans[index % total_plans])
        plan["_origin_variant_index"] = index // total_plans + 1
        plan["_origin_variant_limit"] = variants
        expanded.append(plan)
    return expanded


def sample_valid_bucket(sampler: BucketSampler, plan_meta: dict,
                        templates_index: set, template_config: dict,
                        max_retry: int = 50,
                        allow_fallback: bool = True,
                        candidate_buckets: Optional[List[Tuple[str, str, str]]] = None) -> Optional[tuple]:
    """
    循环采样直到找到合法的bucket

    Args:
        sampler: BucketSampler实例
        plan_meta: plan元数据（包含origin_plan）
        templates_index: 定义的模板bucket集合
        template_config: 完整的模板配置
        max_retry: 最大重试次数

    Returns:
        有效的bucket tuple
    """
    filtered_candidates = [
        bucket for bucket in (candidate_buckets or [])
        if is_bucket_applicable(bucket, plan_meta, templates_index, template_config)
    ]
    if filtered_candidates:
        return random.choice(filtered_candidates)

    for _ in range(max_retry):
        bucket = sampler.sample_bucket()
        if candidate_buckets is not None and bucket not in candidate_buckets:
            continue
        if is_bucket_applicable(bucket, plan_meta, templates_index, template_config):
            return bucket

    if not allow_fallback:
        print(f"  Warning: Could not find valid filtered bucket after {max_retry} attempts")
        return None

    # 如果尝试max_retry次后仍未找到，返回最后一次采样的结果
    # 并记录警告
    print(f"  Warning: Could not find valid bucket after {max_retry} attempts, using last sampled")
    if candidate_buckets:
        return random.choice(candidate_buckets)
    return sampler.sample_bucket()


def parse_bucket_name(bucket_name: str) -> Optional[Tuple[str, str, str]]:
    if not isinstance(bucket_name, str):
        return None
    parts = [p.strip() for p in bucket_name.strip().split("_") if p.strip()]
    if len(parts) != 3:
        return None
    return (parts[0], parts[1], parts[2])


def parse_bucket_whitelist(raw: Optional[str]) -> Optional[set]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    buckets = set()
    invalid = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        bucket = parse_bucket_name(name)
        if bucket is None:
            invalid.append(name)
            continue
        buckets.add(bucket)
    if invalid:
        print(f"  Warning: Invalid bucket names ignored: {invalid}")
    return buckets if buckets else None


def choose_uniform_bucket_for_plan(
    plan_meta: Dict[str, Any],
    candidate_buckets: List[Tuple[str, str, str]],
    bucket_counts: Dict[Tuple[str, str, str], int],
    templates_index: set,
    template_config: dict,
) -> Optional[Tuple[str, str, str]]:
    applicable = [
        b for b in candidate_buckets
        if is_bucket_applicable(b, plan_meta, templates_index, template_config)
    ]
    if not applicable:
        return None

    min_count = min(bucket_counts.get(b, 0) for b in applicable)
    least_used = [b for b in applicable if bucket_counts.get(b, 0) == min_count]
    selected = random.choice(least_used) if least_used else None
    if selected is not None:
        bucket_counts[selected] = bucket_counts.get(selected, 0) + 1
    return selected


def _soft_family_minimums(soft_config: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not isinstance(soft_config, dict):
        return {}
    sampling_cfg = soft_config.get("sampling", {})
    if not isinstance(sampling_cfg, dict):
        return {}
    minimums = sampling_cfg.get("family_minimums", {})
    if not isinstance(minimums, dict):
        return {}
    normalized: Dict[str, int] = {}
    for family, raw_count in minimums.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            normalized[str(family).strip()] = count
    return normalized


def _choose_priority_soft_family(
    controlled_family_counts: Dict[str, int],
    soft_config: Optional[Dict[str, Any]],
) -> str:
    minimums = _soft_family_minimums(soft_config)
    for family, required in minimums.items():
        if controlled_family_counts.get(family, 0) < required:
            return family
    return ""


def _choose_bucket_for_soft_family(
    family: str,
    *,
    soft_config: Optional[Dict[str, Any]],
    plan_meta: Dict[str, Any],
    candidate_buckets: List[Tuple[str, str, str]],
    bucket_counts: Dict[Tuple[str, str, str], int],
    templates_index: set,
    template_config: dict,
    uniform_enabled: bool,
) -> Optional[Tuple[str, str, str]]:
    if not family or not isinstance(soft_config, dict):
        return None
    families_cfg = soft_config.get("families", {})
    family_cfg = families_cfg.get(family) if isinstance(families_cfg, dict) else None
    if not isinstance(family_cfg, dict):
        return None
    allowed_bucket_names = family_cfg.get("allowed_buckets", [])
    if not isinstance(allowed_bucket_names, list) or not allowed_bucket_names:
        return None
    applicable = [
        bucket for bucket in candidate_buckets
        if "_".join(bucket) in allowed_bucket_names
        and is_bucket_applicable(bucket, plan_meta, templates_index, template_config)
    ]
    if not applicable:
        return None
    if uniform_enabled:
        return choose_uniform_bucket_for_plan(
            plan_meta=plan_meta,
            candidate_buckets=applicable,
            bucket_counts=bucket_counts,
            templates_index=templates_index,
            template_config=template_config,
        )
    return random.choice(applicable)


def load_origin_plans(origin_plans_dir: str) -> List[Dict[str, Any]]:
    """
    加载所有origin plan文件

    Args:
        origin_plans_dir: origin plans目录路径

    Returns:
        origin plan字典列表
    """
    origin_plans = []
    origin_plans_path = Path(origin_plans_dir)
    skipped_non_plan = 0

    if not origin_plans_path.exists():
        raise ValueError(f"Origin plans directory not found: {origin_plans_dir}")

    # 遍历所有JSON文件
    for json_file in origin_plans_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                plan = json.load(f)
                if not _looks_like_origin_plan(plan):
                    skipped_non_plan += 1
                    continue
                plan["_source_file"] = json_file.name  # 记录源文件
                origin_plans.append(plan)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    print(f"Loaded {len(origin_plans)} origin plans from {origin_plans_dir}")
    if skipped_non_plan:
        print(f"  Skipped {skipped_non_plan} JSON files without start_city/target_city/itinerary")
    return origin_plans


def _looks_like_origin_plan(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    if not str(plan.get("start_city") or "").strip():
        return False
    if not str(plan.get("target_city") or "").strip():
        return False
    itinerary = plan.get("itinerary")
    if not isinstance(itinerary, list) or not itinerary:
        return False
    return True


def sample_buckets_for_plans(origin_plans: List[Dict[str, Any]],
                              config_path: str,
                              templates_path: str = None,
                              allowed_violations: set = None,
                              sampling_mode: str = "weighted",
                              bucket_whitelist: Optional[set] = None,
                              constraint_type_whitelist: Optional[Set[str]] = None,
                              origin_query_lookup: Dict[str, Dict[str, Any]] = None,
                              soft_config_path: Optional[str] = None,
                              variants_per_plan: int = 1,
                              max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    为每个plan采样target bucket（带模板校验）

    Args:
        origin_plans: origin plan列表
        config_path: bucket distribution配置文件路径
        templates_path: templates.yaml文件路径（可选，用于校验）

    Returns:
        samples列表（包含origin_plan和meta信息）
    """
    # 初始化bucket sampler
    sampler = BucketSampler(config_path)

    # 加载templates索引（用于校验）
    templates_index = set()
    template_config = {}
    allowed_violations = allowed_violations or set(DEFAULT_ALLOWED_EDIT_VIOLATIONS)
    if templates_path:
        templates_index, template_config = load_template_buckets(
            templates_path,
            allowed_violations=allowed_violations,
            constraint_type_whitelist=constraint_type_whitelist,
        )

    candidate_buckets: List[Tuple[str, str, str]] = sorted(templates_index)
    strict_filtered_sampling = bool(bucket_whitelist or constraint_type_whitelist)
    if bucket_whitelist:
        candidate_buckets = sorted([b for b in candidate_buckets if b in bucket_whitelist])
        print(f"  Bucket whitelist enabled: {len(candidate_buckets)} candidate buckets")
        if not candidate_buckets:
            print("  Warning: whitelist has no overlap with eligible templates; filtered sampling will skip unmatched plans")
    if strict_filtered_sampling and not candidate_buckets:
        print("  Warning: no candidate buckets remain after filtering")
    uniform_enabled = (
        sampling_mode == "uniform"
        and bool(candidate_buckets)
        and bool(templates_index)
        and bool(template_config)
    )
    if sampling_mode == "uniform" and not uniform_enabled:
        print("  Warning: uniform mode is unavailable (no eligible template buckets), fallback to weighted")
    bucket_counts: Dict[Tuple[str, str, str], int] = {b: 0 for b in candidate_buckets}
    if not isinstance(origin_query_lookup, dict) or not origin_query_lookup:
        raise ValueError("origin_query_lookup is required and cannot be empty")
    soft_config = None
    if isinstance(soft_config_path, str) and soft_config_path.strip():
        soft_config = load_soft_control_config(soft_config_path)

    samples = []
    sample_index = 0
    variant_limit = max(int(variants_per_plan or 1), 1)
    controlled_family_counts: Dict[str, int] = {}

    for i, plan_data in enumerate(origin_plans):
        if max_samples is not None and len(samples) >= max_samples:
            break
        try:
            # 解析plan（plan_data已经是dict）
            from plan_parser import OriginPlanParser
            parser = OriginPlanParser(plan_data)

            # 构建plan_meta（包含origin_plan用于检查existing_attractions）
            plan_meta = {
                "target_city": parser.get_target_city(),
                "start_city": parser.get_start_city(),
                "is_single_day": parser.is_single_day(),
                "day_count": parser.get_day_count(),
                "origin_plan": parser.get_plan_for_edit()
            }

            origin_plan_file = plan_data.get("_source_file", "")
            origin_query_structured = origin_query_lookup.get(origin_plan_file)
            if not isinstance(origin_query_structured, dict):
                raise MissingOriginQueryError(
                    f"Missing origin query for plan file: {origin_plan_file}"
                )
            try:
                origin_logical_constraints = serialize_logical_constraints(
                    build_origin_logical_constraints(origin_query_structured)
                )
            except Exception:
                origin_logical_constraints = []
            plan_meta["origin_logical_constraints"] = origin_logical_constraints
            origin_query_text = extract_origin_query_text(origin_query_structured)
            origin_soft_profile = build_origin_soft_profile(
                origin_query_structured,
                origin_query_text=origin_query_text,
            )
            try:
                origin_preference_tags = serialize_preference_tags(
                    build_origin_preference_tags(
                        origin_query_structured,
                        origin_query_text=origin_query_text,
                    )
                )
                origin_preference_score_snapshot = serialize_preference_snapshots(
                    compute_origin_preference_score_snapshot(
                        parser.get_plan_for_edit(),
                        origin_preference_tags,
                    )
                )
            except Exception:
                origin_preference_tags = []
                origin_preference_score_snapshot = []
            origin_plan_evidence = build_origin_plan_evidence_for_plan(parser.get_plan_for_edit())
            soft_target = None
            soft_control_status = "disabled"
            if soft_config is not None:
                pass

            for variant_idx in range(variant_limit):
                if max_samples is not None and len(samples) >= max_samples:
                    break

                prioritized_family = _choose_priority_soft_family(controlled_family_counts, soft_config)
                bucket = _choose_bucket_for_soft_family(
                    prioritized_family,
                    soft_config=soft_config,
                    plan_meta=plan_meta,
                    candidate_buckets=candidate_buckets,
                    bucket_counts=bucket_counts,
                    templates_index=templates_index,
                    template_config=template_config,
                    uniform_enabled=uniform_enabled,
                )
                if bucket is None:
                    if uniform_enabled:
                        bucket = choose_uniform_bucket_for_plan(
                            plan_meta=plan_meta,
                            candidate_buckets=candidate_buckets,
                            bucket_counts=bucket_counts,
                            templates_index=templates_index,
                            template_config=template_config,
                        )
                        if bucket is None:
                            bucket = sample_valid_bucket(
                                sampler,
                                plan_meta,
                                templates_index,
                                template_config,
                                allow_fallback=not strict_filtered_sampling,
                            )
                    elif templates_index:
                        bucket = sample_valid_bucket(
                            sampler,
                            plan_meta,
                            templates_index,
                            template_config,
                            candidate_buckets=candidate_buckets if strict_filtered_sampling else None,
                            allow_fallback=not strict_filtered_sampling,
                        )
                    else:
                        bucket = None if strict_filtered_sampling else sampler.sample_bucket()

                if bucket is None:
                    continue

                soft_target = None
                soft_control_status = "disabled"
                if soft_config is not None:
                    soft_target = choose_soft_control_target(
                        profile=origin_soft_profile,
                        bucket=bucket,
                        config=soft_config,
                        rng=random,
                        preferred_family=prioritized_family,
                        evidence=origin_plan_evidence,
                    )
                    soft_control_status = "controlled" if soft_target else "fallback_no_compatible_soft_target"
                    if soft_target is None and prioritized_family:
                        soft_target = choose_soft_control_target(
                            profile=origin_soft_profile,
                            bucket=bucket,
                            config=soft_config,
                            rng=random,
                            evidence=origin_plan_evidence,
                        )
                        soft_control_status = "controlled" if soft_target else "fallback_no_compatible_soft_target"
                if soft_target:
                    family = str(soft_target.get("family", "")).strip()
                    if family:
                        controlled_family_counts[family] = controlled_family_counts.get(family, 0) + 1

                sample_index += 1
                sample = {
                    "sample_id": f"sample_{sample_index:06d}",
                    "batch_id": "",  # 稍后设置
                    "origin_plan": parser.get_plan_for_edit(),
                    "origin_logical_constraints": origin_logical_constraints,
                    "origin_query_text": origin_query_text,
                    "origin_query_structured": origin_query_structured,
                    "origin_soft_profile": origin_soft_profile,
                    "origin_preference_tags": origin_preference_tags,
                    "origin_preference_score_snapshot": origin_preference_score_snapshot,
                    "meta": {
                        "target_bucket": bucket,
                        "target_city": parser.get_target_city(),
                        "start_city": parser.get_start_city(),
                        "is_single_day": parser.is_single_day(),
                        "day_count": parser.get_day_count(),
                        "origin_plan_file": origin_plan_file,
                        "origin_variant_index": variant_idx + 1,
                        "origin_variant_limit": variant_limit,
                        "requested_constraint_type": sorted(constraint_type_whitelist) if constraint_type_whitelist else [],
                        "requested_bucket_names": ["_".join(parts) for parts in candidate_buckets],
                        "soft_constraint_family": soft_target.get("family", "") if soft_target else "",
                        "soft_constraint_facet": soft_target.get("facet", "") if soft_target else "",
                        "soft_edit_mode": soft_target.get("edit_mode", "") if soft_target else "",
                        "soft_schema_version": soft_target.get("schema_version", "") if soft_target else "",
                        "soft_anchor_type": soft_target.get("anchor_type", "") if soft_target else "",
                        "origin_soft_profile": origin_soft_profile,
                        "soft_control_status": soft_control_status,
                        "status": "01_completed",
                        "created_at": get_timestamp()
                    }
                }

                samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(origin_plans)} plans...")

        except Exception as e:
            print(f"Error processing plan {i}: {e}")
            if isinstance(e, MissingOriginQueryError):
                raise
            continue

    print(f"\nSampled {len(samples)} buckets:")
    # 统计bucket分布
    bucket_counts = {}
    for sample in samples:
        bucket_name = sampler.get_bucket_name(sample["meta"]["target_bucket"])
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1

    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count}")

    return samples


def get_timestamp() -> str:
    """获取当前时间戳"""
    from datetime import datetime
    return datetime.now().isoformat()


def save_samples(samples: List[Dict[str, Any]], output_dir: str, batch_id: str):
    """
    保存samples到文件

    Args:
        samples: samples列表
        output_dir: 输出目录
        batch_id: batch ID
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 更新每个sample的batch_id
    for sample in samples:
        sample["batch_id"] = batch_id

        # 保存为单独的JSON文件
        sample_file = output_path / f"{sample['sample_id']}.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(samples)} samples to {output_dir}/")


def create_batch_manifest(samples: List[Dict[str, Any]], output_dir: str, batch_id: str):
    """
    创建batch级别的metadata文件

    Args:
        samples: samples列表
        output_dir: 输出目录
        batch_id: batch ID
    """
    sampler = BucketSampler()

    # 统计信息
    bucket_distribution = {}
    for sample in samples:
        bucket_name = sampler.get_bucket_name(sample["meta"]["target_bucket"])
        bucket_distribution[bucket_name] = bucket_distribution.get(bucket_name, 0) + 1

    city_distribution = {}
    for sample in samples:
        city = sample["meta"]["target_city"]
        city_distribution[city] = city_distribution.get(city, 0) + 1

    day_type_distribution = {
        "single_day": 0,
        "multi_day": 0
    }
    for sample in samples:
        if sample["meta"]["is_single_day"]:
            day_type_distribution["single_day"] += 1
        else:
            day_type_distribution["multi_day"] += 1

    manifest = {
        "batch_id": batch_id,
        "sample_count": len(samples),
        "status": "01_sample_bucket_completed",
        "created_at": get_timestamp(),
        "statistics": {
            "bucket_distribution": bucket_distribution,
            "city_distribution": city_distribution,
            "day_type_distribution": day_type_distribution,
            **build_soft_control_statistics(samples),
        }
    }

    manifest_file = Path(output_dir) / "batch_metadata.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Created batch manifest at {manifest_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sample buckets for origin plans")
    parser.add_argument("--input", required=True, help="Origin plans directory")
    parser.add_argument("--output", required=True, help="Output batch directory")
    parser.add_argument("--batch-id", default="batch_001", help="Batch ID (default: batch_001)")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of samples per batch (default: 100)")
    parser.add_argument(
        "--variants-per-plan",
        type=int,
        default=1,
        help="Maximum number of samples to derive from the same origin plan (default: 1)",
    )
    parser.add_argument("--config", default="data_generation/config/bucket_distribution.yaml",
                       help="Bucket distribution config file (default: data_generation/config/bucket_distribution.yaml)")
    parser.add_argument("--templates", default=None,
                       help="Templates YAML file for validation (optional)")
    parser.add_argument(
        "--query-input",
        required=True,
        help="Origin query目录（必需，按同名JSON匹配origin_plan_file）",
    )
    parser.add_argument(
        "--allow-violations",
        default="overflow,overlap,discontinuity,incompatibility",
        help="Comma-separated violation whitelist for edit sampling",
    )
    parser.add_argument(
        "--sampling-mode",
        default="weighted",
        choices=["weighted", "uniform"],
        help="Bucket sampling mode: weighted (default) or uniform (balanced over eligible buckets)",
    )
    parser.add_argument(
        "--bucket-whitelist",
        default=None,
        help="Optional comma-separated bucket names (e.g. structural_spatial_overflow,parameter_temporal_overflow)",
    )
    parser.add_argument(
        "--constraint-type-whitelist",
        default=None,
        help="Optional comma-separated constraint_type whitelist applied before bucket sampling",
    )
    parser.add_argument(
        "--soft-config",
        default="data_generation/config/soft_constraint_control.yaml",
        help="Soft constraint control config file (default: data_generation/config/soft_constraint_control.yaml)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Sample Buckets")
    print("=" * 60)

    # 1. 加载origin plans
    print(f"\n1. Loading origin plans from: {args.input}")
    origin_plans = load_origin_plans(args.input)

    if not origin_plans:
        print("No origin plans found. Exiting.")
        sys.exit(1)

    bucket_whitelist = parse_bucket_whitelist(args.bucket_whitelist)
    constraint_type_whitelist = parse_constraint_type_whitelist(args.constraint_type_whitelist)
    strict_filtered_sampling = bool(bucket_whitelist or constraint_type_whitelist)

    variant_limit = max(int(args.variants_per_plan or 1), 1)
    if args.batch_size:
        required_plan_count = min(len(origin_plans), max(1, (args.batch_size + variant_limit - 1) // variant_limit))
        if strict_filtered_sampling:
            print("  Strict filtered sampling enabled; keeping all origin plans to satisfy whitelist quota")
        elif len(origin_plans) > required_plan_count:
            print(f"  Limiting to first {required_plan_count} plans (from {len(origin_plans)} total)")
            origin_plans = origin_plans[:required_plan_count]

    # 2. 加载origin queries（强制）
    print(f"\n2. Loading origin queries from: {args.query_input}")
    origin_query_lookup = load_origin_queries(args.query_input)
    if not origin_query_lookup:
        print("No origin queries found. Exiting.")
        sys.exit(1)

    # 3. 采样buckets
    print(f"\n3. Sampling buckets using config: {args.config}")
    if args.templates:
        print(f"   With template validation: {args.templates}")
    allowed_violations = parse_allowed_violations(args.allow_violations)
    print(f"   Allowed violations: {sorted(allowed_violations)}")
    print(f"   Sampling mode: {args.sampling_mode}")
    print(f"   Variants per plan: {variant_limit}")
    if bucket_whitelist:
        printable = [f"{b[0]}_{b[1]}_{b[2]}" for b in sorted(bucket_whitelist)]
        print(f"   Bucket whitelist: {printable}")
    if constraint_type_whitelist:
        print(f"   Constraint type whitelist: {sorted(constraint_type_whitelist)}")

    try:
        samples = sample_buckets_for_plans(
            origin_plans,
            args.config,
            args.templates,
            allowed_violations=allowed_violations,
            sampling_mode=args.sampling_mode,
            bucket_whitelist=bucket_whitelist,
            constraint_type_whitelist=constraint_type_whitelist,
            origin_query_lookup=origin_query_lookup,
            soft_config_path=args.soft_config,
            variants_per_plan=variant_limit,
            max_samples=args.batch_size,
        )
    except MissingOriginQueryError as e:
        print(f"\nError: {e}")
        print("Step 1 aborted because origin query is mandatory.")
        sys.exit(1)

    # 4. 保存samples
    print(f"\n4. Saving samples to: {args.output}")
    save_samples(samples, args.output, args.batch_id)

    # 5. 创建manifest
    create_batch_manifest(samples, args.output, args.batch_id)

    print("\n" + "=" * 60)
    print("✓ Step 1 Complete!")
    print("=" * 60)
    print(f"\nOutput location: {args.output}/")
    print(f"Batch ID: {args.batch_id}")
    print(f"Sample count: {len(samples)}")
    print(f"\nNext step:")
    print(f"  python scripts/02_generate_query.py --input {args.output}")


if __name__ == "__main__":
    main()
