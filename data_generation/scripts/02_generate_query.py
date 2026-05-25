#!/usr/bin/env python3
"""
Step 2: Generate Query
为每个sample生成edit_query和constraints，使用LLM和POI数据库

这是数据生成pipeline的第二步，为核心数据生成逻辑。
"""

import os
import sys
import json
import argparse
import math
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GEN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(DATA_GEN_DIR)
sys.path.insert(0, os.path.join(DATA_GEN_DIR, "utils"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from llm_client import load_client_from_env
from llm_debug import create_llm_debug_logger
from poi_database import POIDatabase
from bucket_sampler import BucketSampler
from plan_parser import OriginPlanParser
from conflict_feasibility import (
    FeasibilityResult,
    build_origin_plan_evidence as build_origin_plan_evidence_for_plan,
    ensure_validator_coverage,
    validate_conflict_trigger as validate_conflict_trigger_proof,
)
from query_surface_parsing import (
    extract_anchor_poi_from_query as shared_extract_anchor_poi_from_query,
    extract_daily_poi_cap_from_query as shared_extract_daily_poi_cap_from_query,
    extract_day_poi_pairs_from_query as shared_extract_day_poi_pairs_from_query,
    extract_hotel_feature_from_query as shared_extract_hotel_feature_from_query,
    extract_hotel_name_from_query as shared_extract_hotel_name_from_query,
    extract_intercity_transport_modes_from_query as shared_extract_intercity_transport_modes_from_query,
    extract_preferred_poi_type_from_query as shared_extract_preferred_poi_type_from_query,
    extract_requested_day_count_from_query as shared_extract_requested_day_count_from_query,
    extract_requested_day_delta_from_query as shared_extract_requested_day_delta_from_query,
    extract_replace_source_poi_from_query as shared_extract_replace_source_poi_from_query,
    extract_restaurant_name_from_query as shared_extract_restaurant_name_from_query,
    extract_restaurant_type_from_query as shared_extract_restaurant_type_from_query,
    extract_room_count_from_query as shared_extract_room_count_from_query,
    extract_room_type_from_query as shared_extract_room_type_from_query,
    extract_semantic_target_from_query as shared_extract_semantic_target_from_query,
    extract_specific_poi_from_query as shared_extract_specific_poi_from_query,
    extract_transport_time_windows_from_query as shared_extract_transport_time_windows_from_query,
    merge_semantic_slots,
    normalize_intercity_transport_mode,
    normalize_preferred_poi_type,
    parse_query_semantic_slots,
)
from soft_constraint_control import (
    summarize_soft_profile,
    template_matches_soft_control,
    validate_soft_query_alignment,
)
from evaluation.benchmark.soft_truth import build_edit_target_preference_tags, build_edit_target_preferences, serialize_preference_constraints
from evaluation.benchmark.hard_truth import _SUPPORTED_DATASET_CITIES
from constraint_ir.compiler import build_canonical_constraint_ir_from_structured_inputs
from constraint_ir.template_semantics import (
    TEMPLATE_SEMANTIC_SPEC_VERSION,
    annotate_template_semantics,
)

try:
    from soft_constraint_control import edit_mode_is_compatible
except ImportError:
    def edit_mode_is_compatible(profile, family, facet, edit_mode):
        del profile, family, facet, edit_mode
        return False

DEFAULT_ALLOWED_EDIT_VIOLATIONS = {"overflow", "overlap", "discontinuity", "incompatibility"}
DEFAULT_TEMPLATE_ATTEMPTS = 5
DEFAULT_MAX_ORIGIN_SWITCHES = 3
DEFAULT_SURFACE_REWRITE_ATTEMPTS = 2
DEFAULT_SURFACE_REWRITE_TEMPERATURE = 0.2
EXTEND_PACING_SOFT_SUPPRESSION_REASON = "day_extension_daily_poi_cap_hard_only"
CONTROLLED_SURFACE_TYPES = {
    "budget_limit",
    "ticket_budget_limit",
    "travel_time_constraint",
    "ticket_price_cap_or_substitute",
}


def should_suppress_extend_with_pacing_edit_soft(
    *,
    template: Dict[str, Any],
    constraint_type: str,
    params: Optional[Dict[str, Any]] = None,
    query_spec: Optional[Dict[str, Any]] = None,
) -> bool:
    if str(constraint_type or "").strip() != "extend_with_pacing":
        return False
    if not (
        template.get("suppress_edit_soft_when_day_extension") is True
        or template.get("requires_day_extension") is True
    ):
        return False
    params = params if isinstance(params, dict) else {}
    query_spec = query_spec if isinstance(query_spec, dict) else {}
    day_count = query_spec.get("day_count", params.get("day_count"))
    daily_poi_cap = query_spec.get("daily_poi_cap", params.get("poi_count"))
    return day_count is not None and daily_poi_cap is not None
SEMANTIC_SLOT_FALLBACK_TYPES = {
    "semantic_and_specific",
    "category_overlap",
    "semantic_substitution",
    "type_filter",
    "preference_constraint",
    "required_restaurant_type",
    "required_hotel_feature",
    "resource_overlap",
    "multi_city_split",
    "remove_poi",
    "cross_day_restructure",
    "poi_local_radius_limit",
}

HIGH_FREQUENCY_ROTATION_BUCKETS = {
    ("compositional", "structural", "overflow"),
    ("structural", "spatial", "overflow"),
    ("structural", "resource", "overflow"),
}
SEMANTIC_DIVERSITY_CONSTRAINT_TYPES = {
    "type_filter",
    "preference_constraint",
    "semantic_and_specific",
    "category_overlap",
}
SEMANTIC_TYPE_EXCLUDE = {"其它"}
SEMANTIC_SOFT_FACET_TYPE_GROUPS = {
    "nature": ["自然风光", "公园", "园林"],
    "culture": ["历史古迹", "博物馆/纪念馆", "文化旅游区", "人文景观", "美术馆/艺术馆", "红色景点", "大学校园"],
}
SEMANTIC_TYPE_USAGE_REGISTRY: Dict[str, Dict[str, int]] = {}
SURFACE_STIFF_PHRASES = [
    "活动半径限制为",
    "周边活动半径",
    "总价不超过",
    "总预算不能超过",
    "扩展到",
    "每天只安排",
]
SURFACE_DIRECTIVE_PREFIXES = ("把", "将", "确保", "限制", "调整", "安排", "控制")
TEMPLATE_USAGE_REGISTRY: Dict[str, Dict[str, Dict[str, int]]] = {}
QUERY_SURFACE_REGISTRY: Dict[str, int] = {}
ORIGIN_LEAK_PATTERNS = [
    r"[零〇一二两三四五六七八九十\d]+个人",
    r"[零〇一二两三四五六七八九十\d]+间(?:房|房间)",
    r"双床房|大床房|标间|单人间",
    r"当前位置[^，。；]{1,12}",
    r"(?:坐|乘|搭)?(?:火车|高铁|飞机|地铁|公交|自驾)(?:往返)?",
    r"酒店[^，。；]{0,12}",
    r"(?:打算|计划|准备|想)去[^，。；]{1,16}玩[零〇一二两三四五六七八九十\d]+天",
    r"中途在[^，。；]{1,12}停留[零〇一二两三四五六七八九十\d]+[天晚日]?",
]
CN_NUMERAL_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _next_random_state(rng: random.Random) -> int:
    """从统一 RNG 派生 pandas 可用的 random_state。"""
    return rng.randint(0, 2**32 - 1)


def parse_allowed_violations(arg_value: str) -> set:
    """解析允许进入 Step 2 的 violation 集合。"""
    if not isinstance(arg_value, str):
        return set(DEFAULT_ALLOWED_EDIT_VIOLATIONS)
    parsed = {item.strip().lower() for item in arg_value.split(",") if item.strip()}
    return parsed or set(DEFAULT_ALLOWED_EDIT_VIOLATIONS)


def parse_constraint_type_whitelist(raw: Optional[str]) -> Optional[set[str]]:
    if not isinstance(raw, str):
        return None
    parsed = {item.strip() for item in raw.split(",") if item.strip()}
    return parsed or None


def load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    """加载batch中的所有samples"""
    batch_path = Path(batch_dir)
    samples = []

    for sample_file in batch_path.glob("sample_*.json"):
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
                samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load {sample_file}: {e}")

    print(f"Loaded {len(samples)} samples from {batch_dir}")
    return samples


def filter_pending_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤出需要处理的samples（还没有edit_query的）"""
    pending = []
    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        force_regenerate = status in {
            "02_pending",
            "02_failed_retryable",
            "02_failed_retryable_same_origin",
        }
        missing_query = ("edit_query" not in sample or not sample["edit_query"])
        missing_constraints = ("constraints" not in sample or not sample["constraints"])

        # 回退重试时（02_pending）必须重生 query；首次生成或缺字段也需要处理
        if force_regenerate or missing_query or missing_constraints:
            pending.append(sample)

    print(f"Found {len(pending)} samples needing query generation (from {len(samples)} total)")
    return pending


def list_templates_for_bucket(bucket: tuple, templates_config: Dict) -> List[Dict[str, Any]]:
    template_config = select_template_for_bucket(bucket, templates_config)
    if not isinstance(template_config, dict):
        return []
    templates = template_config.get("templates", [])
    if not isinstance(templates, list):
        return []
    bucket_key = _bucket_key(bucket)
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(templates):
        if not isinstance(item, dict):
            continue
        normalized_item = dict(item)
        normalized_item.setdefault("template_id", f"{bucket_key}__{normalized_item.get('constraint_type', 'general')}__{idx:02d}")
        surface_variants = normalized_item.get("surface_variants", [])
        if not isinstance(surface_variants, list):
            normalized_item["surface_variants"] = []
        normalized.append(
            annotate_template_semantics(
                normalized_item,
                constraint_type=str(normalized_item.get("constraint_type", "") or ""),
                bucket=bucket if isinstance(bucket, tuple) and len(bucket) == 3 else None,
            )
        )
    return normalized


def filter_intent_groups_for_sample(intent_groups: List[Dict[str, Any]], sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    if str(meta.get("soft_control_status", "")).strip() != "controlled":
        return list(intent_groups)

    family = str(meta.get("soft_constraint_family", "")).strip()
    facet = str(meta.get("soft_constraint_facet", "")).strip()
    edit_mode = str(meta.get("soft_edit_mode", "")).strip()
    if not family or not facet or not edit_mode:
        return list(intent_groups)

    filtered: List[Dict[str, Any]] = []
    for group in intent_groups:
        soft_control = group.get("soft_control", {}) if isinstance(group, dict) else {}
        if not isinstance(soft_control, dict):
            continue
        supported_modes = soft_control.get("supported_edit_modes", [])
        if not isinstance(supported_modes, list):
            supported_modes = []
        if (
            str(soft_control.get("family", "")).strip() == family
            and str(soft_control.get("facet", "")).strip() == facet
            and edit_mode in supported_modes
        ):
            filtered.append(group)
    return filtered


def _bucket_key(bucket: Any) -> str:
    if isinstance(bucket, (list, tuple)) and len(bucket) == 3:
        return "_".join(str(part).strip() for part in bucket)
    return str(bucket or "").strip()


def _template_usage_stats(bucket: Any, template_id: str) -> Dict[str, int]:
    bucket_key = _bucket_key(bucket)
    bucket_stats = TEMPLATE_USAGE_REGISTRY.setdefault(bucket_key, {})
    return bucket_stats.setdefault(template_id, {"attempted_count": 0, "success_count": 0})


def record_template_attempt(
    bucket: Any,
    template_id: str,
    *,
    success: bool = False,
    increment_attempted: bool = True,
) -> None:
    stats = _template_usage_stats(bucket, template_id)
    if increment_attempted:
        stats["attempted_count"] += 1
    if success:
        stats["success_count"] += 1


def _semantic_usage_key(bucket: Any, target_city: str) -> str:
    return f"{_bucket_key(bucket)}::{str(target_city or '').strip()}"


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _city_attraction_type_pool(
    poi_db: POIDatabase,
    target_city: str,
    *,
    allowed_types: Optional[List[str]] = None,
) -> List[str]:
    try:
        city_types = [
            normalize_preferred_poi_type(item) or str(item).strip()
            for item in poi_db.get_attraction_types(target_city)
        ]
    except Exception:
        city_types = []
    city_types = [
        item
        for item in _dedupe_keep_order(city_types)
        if item and item not in SEMANTIC_TYPE_EXCLUDE
    ]
    if allowed_types is None:
        return city_types
    allowed = _dedupe_keep_order(
        [
            normalize_preferred_poi_type(item) or str(item).strip()
            for item in allowed_types
            if str(item or "").strip()
        ]
    )
    return [item for item in city_types if item in set(allowed)]


def _origin_attraction_type_counts(sample: Dict[str, Any], poi_db: POIDatabase, target_city: str) -> Counter:
    counts: Counter = Counter()
    origin_plan = sample.get("origin_plan", {}) if isinstance(sample.get("origin_plan"), dict) else {}
    for day_plan in origin_plan.get("itinerary", []) or []:
        if not isinstance(day_plan, dict):
            continue
        for activity in day_plan.get("activities", []) or []:
            if not isinstance(activity, dict) or activity.get("type") != "attraction":
                continue
            poi_name = str(activity.get("position") or activity.get("end") or "").strip()
            category = str(activity.get("category") or activity.get("poi_type") or "").strip()
            if not category and poi_name:
                try:
                    row = poi_db.get_attraction_by_name(target_city, poi_name)
                    if row is not None:
                        category = str(row.get("type") or "").strip()
                except Exception:
                    category = ""
            category = normalize_preferred_poi_type(category) or category
            if category:
                counts[category] += 1
    return counts


def _choose_semantic_poi_type(
    *,
    template: Dict[str, Any],
    param_values: Any,
    target_bucket: Any,
    target_city: str,
    sample: Dict[str, Any],
    poi_db: POIDatabase,
    rng: random.Random,
    soft_family: str,
    soft_facet: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    options = list(param_values) if isinstance(param_values, (list, tuple)) else []
    normalized_options = _dedupe_keep_order(
        [
            normalize_preferred_poi_type(item) or str(item).strip()
            for item in options
            if str(item or "").strip()
        ]
    )
    source = "city_inventory"
    allowed_types = None
    if soft_family == "theme_alignment" and soft_facet in SEMANTIC_SOFT_FACET_TYPE_GROUPS:
        allowed_types = SEMANTIC_SOFT_FACET_TYPE_GROUPS[soft_facet]
        source = f"soft_facet:{soft_facet}"
    elif normalized_options and len(normalized_options) >= 3:
        allowed_types = normalized_options
        source = "template_options"

    city_pool = _city_attraction_type_pool(poi_db, target_city, allowed_types=allowed_types)
    if not city_pool and normalized_options:
        city_pool = [item for item in normalized_options if item not in SEMANTIC_TYPE_EXCLUDE]
        source = "template_options_fallback"
    if not city_pool:
        return None, {
            "source": "empty_pool",
            "template_id": template.get("template_id", ""),
            "target_city": target_city,
            "template_options": normalized_options,
        }

    origin_counts = _origin_attraction_type_counts(sample, poi_db, target_city)
    usage_key = _semantic_usage_key(target_bucket, target_city)
    usage_counts = SEMANTIC_TYPE_USAGE_REGISTRY.setdefault(usage_key, {})
    decorated = [
        (
            usage_counts.get(candidate, 0),
            origin_counts.get(candidate, 0),
            rng.random(),
            candidate,
        )
        for candidate in city_pool
    ]
    decorated.sort(key=lambda item: item[:3])
    selected = decorated[0][3]
    return selected, {
        "source": source,
        "template_id": template.get("template_id", ""),
        "target_city": target_city,
        "selected": selected,
        "candidate_count": len(city_pool),
        "candidates": city_pool[:12],
        "template_options": normalized_options,
        "batch_usage_before": usage_counts.get(selected, 0),
        "origin_type_count": origin_counts.get(selected, 0),
    }


def record_semantic_type_success(bucket: Any, target_city: str, preferred_type: Any) -> None:
    normalized = normalize_preferred_poi_type(preferred_type) or str(preferred_type or "").strip()
    if not normalized:
        return
    usage = SEMANTIC_TYPE_USAGE_REGISTRY.setdefault(_semantic_usage_key(bucket, target_city), {})
    usage[normalized] = usage.get(normalized, 0) + 1


def order_templates_for_bucket_rotation(
    templates: List[Dict[str, Any]],
    bucket: Any,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if len(templates) <= 1:
        return list(templates)

    bucket_tuple = tuple(bucket) if isinstance(bucket, (list, tuple)) else bucket
    prioritized_rotation = bucket_tuple in HIGH_FREQUENCY_ROTATION_BUCKETS
    decorated: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for template in templates:
        template_id = str(template.get("template_id", template.get("template", "")))
        stats = _template_usage_stats(bucket, template_id)
        tie_break = rng.random()
        sort_key = (
            stats["attempted_count"],
            stats["success_count"],
            tie_break,
        )
        if not prioritized_rotation:
            sort_key = (
                stats["attempted_count"],
                stats["success_count"],
                tie_break,
            )
        decorated.append((sort_key, template))
    decorated.sort(key=lambda item: item[0])
    return [template for _, template in decorated]


def order_templates_for_origin_diversity(
    templates: List[Dict[str, Any]],
    sample: Dict[str, Any],
    diversity_registry: Dict[str, Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    origin_file = str(sample.get("meta", {}).get("origin_plan_file", "") or "")
    origin_stats = diversity_registry.get(origin_file, {}) if isinstance(diversity_registry, dict) else {}
    used_templates = origin_stats.get("used_templates", set())
    used_constraint_types = origin_stats.get("used_constraint_types", set())
    if not isinstance(used_templates, set):
        used_templates = set(used_templates or [])
    if not isinstance(used_constraint_types, set):
        used_constraint_types = set(used_constraint_types or [])

    decorated: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
    for template in templates:
        template_text = str(template.get("template", "") or "")
        constraint_type = str(template.get("constraint_type", "") or "")
        decorated.append(
            (
                (
                    template_text in used_templates,
                    constraint_type in used_constraint_types,
                    rng.random(),
                ),
                template,
            )
        )
    decorated.sort(key=lambda item: item[0])
    return [template for _, template in decorated]


def _min_numeric_option(values: Any) -> Optional[float]:
    if not isinstance(values, (list, tuple)):
        return None
    parsed: List[float] = []
    for item in values:
        num = _parse_numeric(item)
        if num is not None:
            parsed.append(float(num))
    return min(parsed) if parsed else None


def _min_duration_option_minutes(values: Any) -> Optional[int]:
    if not isinstance(values, (list, tuple)):
        return None
    parsed: List[int] = []
    for item in values:
        num = _parse_duration_text_to_minutes(item)
        if num is not None:
            parsed.append(int(num))
    return min(parsed) if parsed else None


def _parse_compact_cn_numeral(token: Any) -> Optional[int]:
    text = str(token or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CN_NUMERAL_DIGITS:
        return CN_NUMERAL_DIGITS[text]
    digit_map = {"零": 0, "〇": 0, **CN_NUMERAL_DIGITS}
    unit_map = {"十": 10, "百": 100, "千": 1000}
    total = 0
    current = 0
    used_unit = False
    for char in text:
        if char in digit_map:
            current = digit_map[char]
            continue
        if char in unit_map:
            unit = unit_map[char]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            used_unit = True
            continue
        return None
    if used_unit:
        return total + current
    return None


def _round_to_step_below(value: float, *, step: int, upper_bound: float) -> int:
    rounded = int(math.ceil(value / step) * step)
    if rounded >= upper_bound:
        rounded -= step
    return rounded


def _choose_adaptive_duration_limit_minutes(baseline_minutes: Any) -> Optional[int]:
    baseline = _parse_numeric(baseline_minutes)
    if baseline is None or baseline <= 10:
        return None
    target = min(float(baseline) * 0.8, float(baseline) - max(5.0, float(baseline) * 0.12))
    target = max(10.0, target)
    chosen = _round_to_step_below(target, step=5, upper_bound=float(baseline))
    return chosen if chosen >= 10 else None


def _choose_adaptive_budget_cap(
    baseline_cost: Any,
    *,
    min_floor: int,
    rounding_step: int,
) -> Optional[int]:
    baseline = _parse_numeric(baseline_cost)
    if baseline is None or baseline <= min_floor:
        return None
    effective_step = rounding_step
    if baseline < 300:
        effective_step = min(effective_step, 10)
    elif baseline < 1000:
        effective_step = min(effective_step, 20)
    elif baseline < 3000:
        effective_step = min(effective_step, 50)
    target = float(baseline) - max(float(effective_step), float(baseline) * 0.03)
    target = max(float(min_floor), target)
    chosen = int(math.floor(target / effective_step) * effective_step)
    if chosen >= baseline:
        chosen -= effective_step
    return chosen if chosen >= min_floor else None


def _budget_rounding_step(amount: float) -> int:
    if amount < 300:
        return 10
    if amount < 1000:
        return 20
    if amount < 3000:
        return 50
    if amount < 8000:
        return 100
    return 200


def _extract_origin_budget_from_text(text: Any) -> Optional[float]:
    plain_text = str(text or "").strip()
    if not plain_text or "预算" not in plain_text:
        return None
    match = re.search(r"预算[^\d]{0,8}(\d+(?:\.\d+)?)", plain_text)
    if not match:
        return None
    return _parse_numeric(match.group(1))


def _extract_origin_budget_hint(sample: Dict[str, Any]) -> Optional[float]:
    origin_query_text = sample.get("origin_query_text", "")
    parsed = _extract_origin_budget_from_text(origin_query_text)
    if parsed is not None:
        return parsed
    structured = sample.get("origin_query_structured", {})
    if isinstance(structured, dict):
        for key in ("nature_language", "nature_language_en"):
            parsed = _extract_origin_budget_from_text(structured.get(key))
            if parsed is not None:
                return parsed
    return None


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


def _choose_adaptive_budget_relaxation(
    *,
    origin_budget: Any,
    baseline_total: Any,
) -> Optional[int]:
    origin_budget_num = _parse_numeric(origin_budget)
    baseline_total_num = _parse_numeric(baseline_total)
    if baseline_total_num is None:
        return None
    if origin_budget_num is not None and baseline_total_num <= origin_budget_num:
        return None
    reference = float(origin_budget_num) if origin_budget_num is not None else float(baseline_total_num)
    step = _budget_rounding_step(reference)
    targets = [
        float(baseline_total_num) + step,
        float(baseline_total_num) * 1.05,
    ]
    if origin_budget_num is not None:
        targets.append(float(origin_budget_num) * 1.12)
    target = max(targets)
    chosen = int(math.ceil(target / step) * step)
    if origin_budget_num is not None and chosen <= origin_budget_num:
        return int(origin_budget_num + step)
    if chosen <= baseline_total_num:
        return int(math.ceil((float(baseline_total_num) + step) / step) * step)
    return chosen


def _choose_budget_threshold_for_constraint(
    constraint_type: str,
    *,
    baseline_total_cost: Any,
    options: List[Any],
    origin_budget_total: Optional[Any] = None,
    min_ratio: float = 0.75,
) -> Tuple[str, float]:
    baseline = _parse_numeric(baseline_total_cost)
    if baseline is None or baseline <= 0:
        raise ValueError("baseline_total_cost must be positive")

    origin_budget = _parse_numeric(origin_budget_total)
    parsed_options: List[Tuple[str, float]] = []
    for option in options or []:
        value = _parse_numeric(option)
        if value is not None:
            parsed_options.append((str(option), float(value)))
    ctype = str(constraint_type or "").strip()

    if ctype in {"budget_cap_preference", "budget_target_update"} and origin_budget is not None:
        lower = float(origin_budget)
        upper = float(baseline)
        if upper <= lower:
            raise ValueError(
                "budget update requires baseline_total_cost greater than origin_budget_total"
            )
        candidates = [
            item for item in parsed_options
            if lower < item[1] <= upper
        ]
        if candidates:
            selected = max(candidates, key=lambda item: item[1])
            return selected[0], selected[1]

        step = _budget_rounding_step(lower)
        inferred = _round_to_step_below(upper * 0.95, step=step, upper_bound=upper)
        if inferred <= lower:
            inferred = int(lower + step)
        return str(inferred), float(inferred)

    if origin_budget is None:
        inferred = _choose_adaptive_budget_cap(
            baseline,
            min_floor=100,
            rounding_step=100,
        )
        if inferred is None:
            raise ValueError("could not infer budget threshold")
        return str(inferred), float(inferred)

    if origin_budget <= float(baseline) * 1.25:
        upper = min(float(baseline), float(origin_budget))
        lower = float(origin_budget) * float(min_ratio)
    else:
        upper = float(baseline)
        lower = float(baseline) * 0.9

    candidates = [
        item for item in parsed_options
        if lower <= item[1] < upper
    ]
    if candidates:
        selected = max(candidates, key=lambda item: item[1])
        return selected[0], selected[1]

    step = _budget_rounding_step(upper)
    inferred = _round_to_step_below(max(lower, upper * 0.9), step=step, upper_bound=upper)
    if inferred < lower:
        inferred = int(math.ceil(lower / step) * step)
        if inferred >= upper:
            inferred = int(math.floor((upper - 1) / step) * step)
    return str(inferred), float(inferred)


def _estimate_resource_overlap_budget_baseline(
    evidence: Dict[str, Any],
    *,
    anchor_poi: Optional[str] = None,
) -> Optional[float]:
    ticket_component = 0.0
    if isinstance(anchor_poi, str) and anchor_poi.strip():
        price_map = evidence.get("poi_ticket_price_by_name", {})
        if isinstance(price_map, dict):
            matched = price_map.get(anchor_poi.strip())
            if matched is None:
                for name, info in price_map.items():
                    if anchor_poi.strip() in str(name) or str(name) in anchor_poi.strip():
                        matched = info
                        break
            if isinstance(matched, dict):
                ticket_component = float(_parse_numeric(matched.get("ticket_price")) or 0.0)

    meal_component = 0.0
    meal_count = int(evidence.get("meal_count", 0) or 0)
    if meal_count > 0:
        meal_component = float(evidence.get("meal_cost", 0.0) or 0.0) / float(meal_count)

    baseline = ticket_component + meal_component
    return baseline if baseline > 0 else None


def _project_multi_day_budget_baseline(
    evidence: Dict[str, Any],
    *,
    requested_day_count: Optional[int],
) -> Optional[float]:
    current_day_count = max(int(evidence.get("day_count", 0) or 0), 1)
    local_cost = _parse_numeric(
        evidence.get("transferable_local_cost", 0.0)
        or evidence.get("local_core_cost", 0.0)
    )
    if requested_day_count is None or local_cost is None or local_cost <= 0:
        return None
    if int(requested_day_count) <= current_day_count:
        return float(local_cost)
    per_day_local = float(local_cost) / float(current_day_count)
    return max(float(local_cost), per_day_local * float(requested_day_count))


def filter_templates_for_sample(
    templates: List[Dict[str, Any]],
    sample: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    intent_group: Optional[Dict[str, Any]] = None,
    *,
    ignore_soft_control: bool = False,
) -> List[Dict[str, Any]]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    filtered = templates
    if evidence is None:
        evidence = build_origin_plan_evidence(sample)
    if not ignore_soft_control and str(meta.get("soft_control_status", "")).strip() == "controlled":
        family = str(meta.get("soft_constraint_family", "")).strip()
        facet = str(meta.get("soft_constraint_facet", "")).strip()
        edit_mode = str(meta.get("soft_edit_mode", "")).strip()
        if family and facet and edit_mode:
            filtered = [template for template in filtered if template_matches_soft_control(template, meta)]

    current_day_count = int(meta.get("day_count", 0) or 0)
    current_plan = sample.get("origin_plan", {}) if isinstance(sample.get("origin_plan"), dict) else {}
    day_attraction_count: Dict[int, int] = {}
    for day_plan in current_plan.get("itinerary", []):
        if not isinstance(day_plan, dict):
            continue
        day = day_plan.get("day")
        if not isinstance(day, int):
            continue
        count = 0
        for activity in day_plan.get("activities", []):
            if isinstance(activity, dict) and activity.get("type") == "attraction":
                count += 1
        day_attraction_count[day] = count
    max_day_cap = max(day_attraction_count.values()) if day_attraction_count else 0

    evidence_filtered: List[Dict[str, Any]] = []
    for template in filtered:
        constraint_type = str(template.get("constraint_type", "")).strip()
        params = template.get("params", {}) if isinstance(template.get("params"), dict) else {}
        if constraint_type == "time_limit":
            attraction_days = [
                day for day, count in (evidence or {}).get("day_attraction_count", {}).items()
                if int(count or 0) > 0
            ]
            if not attraction_days:
                continue
            evidence_filtered.append(template)
            continue
        if constraint_type == "budget_limit":
            if not bool((evidence or {}).get("budget_baseline_reliable_for_total_budget_edit", True)):
                continue
            baseline_total = float((evidence or {}).get("total_cost", 0.0) or 0.0)
            adaptive_budget = _choose_adaptive_budget_cap(
                baseline_total,
                min_floor=100,
                rounding_step=100,
            )
            if adaptive_budget is None:
                continue
            evidence_filtered.append(template)
            continue
        if constraint_type == "ticket_budget_limit":
            if not bool((evidence or {}).get("has_origin_budget_requirement", False)):
                continue
            baseline_ticket = float((evidence or {}).get("ticket_cost", 0.0) or 0.0)
            adaptive_ticket_budget = _choose_adaptive_budget_cap(
                baseline_ticket,
                min_floor=50,
                rounding_step=20,
            )
            if adaptive_ticket_budget is None:
                continue
            evidence_filtered.append(template)
            continue
        if constraint_type in {"budget_cap_preference", "budget_target_update"}:
            baseline_total = float((evidence or {}).get("total_cost", 0.0) or 0.0)
            origin_budget = (evidence or {}).get("origin_budget_total") or _extract_origin_budget_hint(sample)
            adaptive_relaxed_budget = _choose_adaptive_budget_relaxation(
                origin_budget=origin_budget,
                baseline_total=baseline_total,
            )
            if adaptive_relaxed_budget is None:
                continue
            evidence_filtered.append(template)
            continue
        if constraint_type == "multi_day_budget_overflow":
            if not bool((evidence or {}).get("budget_baseline_reliable_for_multi_day", False)):
                continue
            day_options = [
                int(value)
                for value in params.get("day_count", [])
                if isinstance(value, (int, float)) or (isinstance(value, str) and str(value).isdigit())
            ]
            valid_day_options = [option for option in day_options if option > current_day_count]
            if day_options and not valid_day_options:
                continue
            projected_candidates = [
                _project_multi_day_budget_baseline(evidence or {}, requested_day_count=option)
                for option in (valid_day_options or [current_day_count + 1])
            ]
            projected_candidates = [value for value in projected_candidates if value is not None]
            if not projected_candidates:
                continue
            adaptive_budget = _choose_adaptive_budget_cap(
                min(projected_candidates),
                min_floor=100,
                rounding_step=100,
            )
            if adaptive_budget is None:
                continue
            evidence_filtered.append(template)
            continue
        if constraint_type == "resource_overlap":
            if not bool((evidence or {}).get("has_origin_budget_requirement", False)):
                continue
            evidence_filtered.append(template)
            continue
        if constraint_type == "travel_time_constraint":
            baseline_transport = int((evidence or {}).get("max_transport_duration_min", 0) or 0)
            adaptive_duration = _choose_adaptive_duration_limit_minutes(baseline_transport)
            if adaptive_duration is None:
                continue
            evidence_filtered.append(template)
            continue
        requires_day_extension = template.get("requires_day_extension")
        day_options = [
            int(value)
            for value in params.get("day_count", [])
            if isinstance(value, (int, float)) or (isinstance(value, str) and str(value).isdigit())
        ]
        cap_options = [
            int(value)
            for value in params.get("poi_count", [])
            if isinstance(value, (int, float)) or (isinstance(value, str) and str(value).isdigit())
        ]
        has_stricter_day_option = any(option > current_day_count for option in day_options)
        has_stricter_cap_option = any(option < max_day_cap for option in cap_options) if max_day_cap > 0 else bool(cap_options)

        if requires_day_extension is True and not has_stricter_day_option:
            continue
        if requires_day_extension is False and not has_stricter_cap_option:
            continue
        evidence_filtered.append(template)
    return evidence_filtered


def filter_templates_by_constraint_type(
    templates: List[Dict[str, Any]],
    constraint_type_whitelist: Optional[set[str]],
) -> List[Dict[str, Any]]:
    if not constraint_type_whitelist:
        return list(templates)
    return [
        template
        for template in templates
        if str(template.get("constraint_type", "")).strip() in constraint_type_whitelist
    ]


def resolve_sample_constraint_type_filter(
    sample: Dict[str, Any],
    constraint_type_whitelist: Optional[set[str]],
) -> Optional[set[str]]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    raw_requested = meta.get("requested_constraint_type")
    requested: set[str] = set()
    if isinstance(raw_requested, list):
        requested = {str(item).strip() for item in raw_requested if str(item).strip()}
    elif isinstance(raw_requested, str) and raw_requested.strip():
        requested = {item.strip() for item in raw_requested.split(",") if item.strip()}
    if constraint_type_whitelist and requested:
        return set(constraint_type_whitelist).intersection(requested)
    if requested:
        return requested
    return constraint_type_whitelist


def filter_templates_by_domain_scope(
    templates: List[Dict[str, Any]],
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    domain = str(meta.get("diversity_domain", "") or "").strip()
    if not domain:
        return list(templates)
    matched = [
        template
        for template in templates
        if str(template.get("domain_scope", "") or "").strip() == domain
    ]
    return matched or list(templates)


def _retarget_or_fallback_soft_control(
    sample: Dict[str, Any],
    templates: List[Dict[str, Any]],
    evidence: Dict[str, Any],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    origin_profile = meta.get("origin_soft_profile", [])
    soft_candidates: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for template in templates:
        family = str(template.get("soft_family", "")).strip()
        facet = str(template.get("soft_facet", "")).strip()
        supported_modes = template.get("supported_edit_modes", [])
        if not family or not facet or not isinstance(supported_modes, list):
            continue
        for mode in supported_modes:
            edit_mode = str(mode or "").strip()
            key = (family, facet, edit_mode)
            if not edit_mode or key in seen:
                continue
            if not edit_mode_is_compatible(origin_profile, family, facet, edit_mode):
                continue
            probe_sample = {"meta": dict(meta), "origin_plan": sample.get("origin_plan", {})}
            probe_meta = probe_sample["meta"]
            probe_meta["soft_control_status"] = "controlled"
            probe_meta["soft_constraint_family"] = family
            probe_meta["soft_constraint_facet"] = facet
            probe_meta["soft_edit_mode"] = edit_mode
            if not filter_templates_for_sample(templates, probe_sample, evidence=evidence):
                continue
            seen.add(key)
            soft_candidates.append(
                {
                    "family": family,
                    "facet": facet,
                    "edit_mode": edit_mode,
                }
            )

    if soft_candidates:
        chosen = rng.choice(soft_candidates)
        meta["soft_control_status"] = "controlled"
        meta["soft_constraint_family"] = chosen["family"]
        meta["soft_constraint_facet"] = chosen["facet"]
        meta["soft_edit_mode"] = chosen["edit_mode"]
        meta["soft_schema_version"] = "v2"
        return filter_templates_for_sample(templates, sample, evidence=evidence)

    meta["soft_control_status"] = "fallback_no_compatible_soft_target"
    meta["soft_constraint_family"] = ""
    meta["soft_constraint_facet"] = ""
    meta["soft_edit_mode"] = ""
    meta["soft_schema_version"] = ""
    meta["soft_anchor_type"] = ""
    return filter_templates_for_sample(templates, sample, evidence=evidence, ignore_soft_control=True)


def _choose_stricter_int_option(
    options: Any,
    *,
    baseline: int,
    direction: str,
) -> Optional[int]:
    values: List[int] = []
    if isinstance(options, (list, tuple)):
        for item in options:
            if isinstance(item, (int, float)):
                values.append(int(item))
            elif isinstance(item, str) and item.strip().isdigit():
                values.append(int(item.strip()))
    if not values:
        return None
    if direction == "gt":
        candidates = sorted(value for value in values if value > baseline)
        return candidates[0] if candidates else None
    if direction == "lt":
        candidates = sorted((value for value in values if value < baseline), reverse=True)
        return candidates[0] if candidates else None
    return None


def select_template_for_bucket(bucket: tuple, templates_config: Dict) -> Optional[Dict]:
    """
    根据target_bucket选择模板

    Args:
        bucket: (scope, dimension, violation) tuple
        templates_config: templates配置字典

    Returns:
        选中的模板配置，如果没找到返回None
    """
    # JSON反序列化后可能是list，统一转tuple
    if isinstance(bucket, list):
        bucket = tuple(bucket)

    # 将bucket tuple转换为字符串key格式，例如 "(parameter, temporal, overflow)"
    bucket_str = str(bucket)

    # 直接在templates_config的顶层查找
    if bucket_str in templates_config:
        return templates_config[bucket_str]

    # 尝试其他格式
    bucket_key_underscore = "_".join(bucket)  # 例如: "parameter_temporal_overflow"

    for key in templates_config.keys():
        # 跳过非bucket的配置项
        if key in ["poi_source_types", "city_constraint_types", "template_instructions"]:
            continue

        # 尝试精确匹配
        if key == bucket_str or key == bucket_key_underscore:
            return templates_config[key]

    # 如果还没有匹配，尝试部分匹配（兼容性）
    scope, dimension, violation = bucket
    for key, value in templates_config.items():
        if key in ["poi_source_types", "city_constraint_types", "template_instructions"]:
            continue

        # 检查是否包含所有三个部分
        if scope in key and dimension in key and violation in key:
            return value

    # 没找到匹配的模板
    return None


def is_bucket_eligible_for_edit(
    bucket: Any,
    template_config: Dict[str, Any],
    allowed_violations: set,
) -> bool:
    """
    判断 bucket 是否允许进入 edit 生成：
    1) violation ∈ allowed_violations
    2) 模板格子标记 eligible_for_edit=true
    """
    if not isinstance(bucket, (list, tuple)) or len(bucket) != 3:
        return False

    violation = bucket[2]
    if violation not in allowed_violations:
        return False

    return bool(template_config.get("eligible_for_edit", False))


def sample_pois_for_template(
    template: Dict,
    target_city: str,
    sample: Dict,
    poi_db: POIDatabase,
    rng: random.Random,
) -> List[Dict]:
    """
    根据模板采样POI

    Args:
        template: 模板配置
        target_city: 目标城市
        sample: sample字典（包含origin_plan）
        poi_db: POI数据库
        n: 采样数量

    Returns:
        采样的POI列表
    """
    pois = []
    poi_source = template.get("poi_source", "attractions")
    poi_count = template.get("poi_count", 1)
    constraint_type = str(template.get("constraint_type", "") or "").strip()

    def _existing_activity_pois(category: str, *activity_types: str) -> List[Dict[str, Any]]:
        origin_plan = sample.get("origin_plan", {})
        itinerary = origin_plan.get("itinerary", [])
        names: List[str] = []
        for day_plan in itinerary:
            activities = day_plan.get("activities", [])
            for activity in activities:
                if activity.get("type") not in activity_types:
                    continue
                poi_name = activity.get("position", activity.get("end", ""))
                if poi_name and poi_name not in names:
                    names.append(poi_name)
        return [
            {
                "name": name,
                "city": target_city,
                "type": category,
                "price": 0,
            }
            for name in names
        ]

    if poi_source == "attractions":
        # 从attractions数据库采样
        attractions = poi_db.sample_attractions(
            target_city,
            n=poi_count,
            random_state=_next_random_state(rng),
        )
        for attr in attractions:
            pois.append({
                "name": attr["name"],
                "type": "attraction",
                "category": attr.get("type"),
                "city": target_city,
                "price": attr.get("price", 0),
                "lat": attr["lat"],
                "lon": attr["lon"]
            })

    elif poi_source == "existing_attractions":
        # 从origin plan中已有的景点选择
        origin_plan = sample.get("origin_plan", {})
        itinerary = origin_plan.get("itinerary", [])

        existing_pois = []
        for day_plan in itinerary:
            activities = day_plan.get("activities", [])
            for activity in activities:
                # 只选择attraction类型的活动
                if activity.get("type") == "attraction":
                    poi_name = activity.get("position", activity.get("end", ""))
                    if poi_name and poi_name not in existing_pois:
                        existing_pois.append(poi_name)

        # 从已有的POI中选择
        import pandas as pd
        if len(existing_pois) > 0:
            sampled_names = rng.sample(existing_pois, min(poi_count, len(existing_pois)))
            for name in sampled_names:
                # 尝试从数据库获取详细信息
                attr = poi_db.get_attraction_by_name(target_city, name)
                # 检查 attr 是否有效（pandas Series 或 dict）
                if attr is not None and not (isinstance(attr, pd.Series) and attr.empty):
                    # 如果是 Series，转换为字典
                    if isinstance(attr, pd.Series):
                        attr_dict = attr.to_dict()
                    else:
                        attr_dict = attr

                    pois.append({
                        "name": name,
                        "type": "attraction",
                        "category": attr_dict.get("type"),
                        "city": target_city,
                        "price": attr_dict.get("price", 0),
                        "lat": attr_dict.get("lat", 0),
                        "lon": attr_dict.get("lon", 0)
                    })
                else:
                    # 如果数据库中没有，使用基本信息
                    pois.append({
                        "name": name,
                        "type": "attraction",
                        "category": None,
                        "city": target_city,
                        "price": 0,
                        "lat": 0,
                        "lon": 0
                    })

    elif poi_source == "restaurants":
        existing_names = {
            item["name"]
            for item in _existing_activity_pois("restaurant", "breakfast", "lunch", "dinner")
            if isinstance(item.get("name"), str)
        }
        restaurants = poi_db.get_restaurants_by_city(target_city)
        if not restaurants.empty:
            if constraint_type == "required_restaurant_name" and "name" in restaurants.columns:
                candidates = restaurants[~restaurants["name"].astype(str).isin(existing_names)]
                if not candidates.empty:
                    restaurants = candidates
            sampled = restaurants.sample(
                n=min(poi_count, len(restaurants)),
                random_state=_next_random_state(rng),
            )
            for _, row in sampled.iterrows():
                pois.append({
                    "name": row["name"],
                    "type": "restaurant",
                    "city": target_city,
                    "price": row.get("price", 0) if hasattr(row, "get") else 0,
                })

    elif poi_source == "accommodations":
        existing_names = {
            item["name"]
            for item in _existing_activity_pois("accommodation", "accommodation")
            if isinstance(item.get("name"), str)
        }
        accommodations = poi_db.get_accommodations_by_city(target_city)
        if not accommodations.empty:
            if constraint_type == "required_hotel_name" and "name" in accommodations.columns:
                candidates = accommodations[~accommodations["name"].astype(str).isin(existing_names)]
                if not candidates.empty:
                    accommodations = candidates
            sampled = accommodations.sample(
                n=min(poi_count, len(accommodations)),
                random_state=_next_random_state(rng),
            )
            for _, row in sampled.iterrows():
                pois.append({
                    "name": row["name"],
                    "type": "accommodation",
                    "city": target_city,
                    "price": row.get("price", 0) if hasattr(row, "get") else 0,
                })

    return pois


def _sample_attraction_by_type(
    poi_db: POIDatabase,
    target_city: str,
    poi_type: str,
    exclude_names: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(poi_type, str) or not poi_type.strip():
        return None
    exclude_names = exclude_names or set()

    try:
        import pandas as pd
        typed_df = poi_db.get_attractions_by_city(target_city, type_filter=poi_type.strip())
        if typed_df is None or typed_df.empty:
            return None
        if "name" in typed_df.columns and exclude_names:
            typed_df = typed_df[~typed_df["name"].isin(list(exclude_names))]
            if typed_df.empty:
                return None

        row = typed_df.sample(n=1).iloc[0]
        price_val = row.get("price", 0) if hasattr(row, "get") else 0
        if pd.isna(price_val):
            price_val = 0
        return {
            "name": row.get("name", ""),
            "type": "attraction",
            "category": row.get("type", poi_type.strip()),
            "city": target_city,
            "price": float(price_val) if isinstance(price_val, (int, float)) else 0.0,
            "lat": float(row.get("lat", 0)) if isinstance(row.get("lat", 0), (int, float)) else 0.0,
            "lon": float(row.get("lon", 0)) if isinstance(row.get("lon", 0), (int, float)) else 0.0,
        }
    except Exception:
        return None


def _sample_any_attraction(
    poi_db: POIDatabase,
    target_city: str,
    exclude_names: Optional[set] = None,
    exclude_categories: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    exclude_names = exclude_names or set()
    exclude_categories = exclude_categories or set()

    try:
        import pandas as pd
        candidates = poi_db.get_attractions_by_city(target_city)
        if candidates is None or candidates.empty:
            return None
        if "name" in candidates.columns and exclude_names:
            candidates = candidates[~candidates["name"].isin(list(exclude_names))]
            if candidates.empty:
                return None
        if "type" in candidates.columns and exclude_categories:
            candidates = candidates[~candidates["type"].isin(list(exclude_categories))]
            if candidates.empty:
                return None

        row = candidates.sample(n=1).iloc[0]
        price_val = row.get("price", 0) if hasattr(row, "get") else 0
        if pd.isna(price_val):
            price_val = 0
        return {
            "name": row.get("name", ""),
            "type": "attraction",
            "category": row.get("type"),
            "city": target_city,
            "price": float(price_val) if isinstance(price_val, (int, float)) else 0.0,
            "lat": float(row.get("lat", 0)) if isinstance(row.get("lat", 0), (int, float)) else 0.0,
            "lon": float(row.get("lon", 0)) if isinstance(row.get("lon", 0), (int, float)) else 0.0,
        }
    except Exception:
        return None


def _align_pois_with_type_requirements(
    template_str: str,
    constraint_type: str,
    params: Dict[str, Any],
    target_city: str,
    poi_db: POIDatabase,
    fallback_pois: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(template_str, str):
        return fallback_pois

    # 语义替换类模板里的 {poi} 是 origin_plan 中的 source anchor，
    # 不能在“按类型对齐”阶段被城市 POI 库重新采样覆盖。
    if fallback_pois and template_refers_existing_poi_semantically(template_str, constraint_type):
        return fallback_pois

    # 单点类型模板：{poi} + {poi_type}
    if "{poi}" in template_str and "{poi_type}" in template_str:
        type_value = params.get("poi_type")
        target_type = str(type_value).strip() if type_value is not None else ""
        sampled = _sample_attraction_by_type(poi_db, target_city, target_type)
        if not sampled:
            sampled = _sample_any_attraction(poi_db, target_city)
        if sampled and isinstance(sampled.get("category"), str) and sampled.get("category").strip():
            params["poi_type"] = sampled.get("category").strip()
        if sampled:
            return [sampled]
        return fallback_pois

    # 双点类型模板：{poi1}/{poi2} + {poi_type1}/{poi_type2}
    if "{poi1}" in template_str and "{poi2}" in template_str:
        type1 = params.get("poi_type1")
        type2 = params.get("poi_type2")
        if not (isinstance(type1, str) and type1.strip() and isinstance(type2, str) and type2.strip()):
            return fallback_pois

        selected: List[Dict[str, Any]] = []
        used_names: set = set()
        p1 = _sample_attraction_by_type(poi_db, target_city, type1, exclude_names=used_names)
        if not p1:
            p1 = _sample_any_attraction(poi_db, target_city, exclude_names=used_names)
        if not p1:
            return fallback_pois
        selected.append(p1)
        used_names.add(p1["name"])
        if isinstance(p1.get("category"), str) and p1.get("category").strip():
            params["poi_type1"] = p1.get("category").strip()

        p2 = _sample_attraction_by_type(poi_db, target_city, type2, exclude_names=used_names)
        if not p2:
            exclude_categories = set()
            if isinstance(params.get("poi_type1"), str) and params.get("poi_type1").strip():
                exclude_categories.add(params.get("poi_type1").strip())
            p2 = _sample_any_attraction(
                poi_db,
                target_city,
                exclude_names=used_names,
                exclude_categories=exclude_categories,
            )
        if not p2:
            p2 = _sample_any_attraction(poi_db, target_city, exclude_names=used_names)
        if not p2:
            return fallback_pois
        if isinstance(p2.get("category"), str) and p2.get("category").strip():
            params["poi_type2"] = p2.get("category").strip()
        selected.append(p2)
        return selected

    return fallback_pois


def fill_template(template_str: str, target_city: str, pois: List[Dict],
                params: Dict[str, Any]) -> str:
    """
    填充模板参数

    Args:
        template_str: 模板字符串
        target_city: 目标城市
        pois: POI列表
        params: 参数字典

    Returns:
        填充后的字符串
    """
    result = template_str

    # 替换POI占位符
    if "{poi}" in result and pois:
        result = result.replace("{poi}", pois[0]["name"])
    if "{poi1}" in result and len(pois) >= 1:
        result = result.replace("{poi1}", pois[0]["name"])
    if "{poi2}" in result and len(pois) >= 2:
        result = result.replace("{poi2}", pois[1]["name"])

    # 替换城市占位符
    result = result.replace("{target_city}", target_city)
    if "start_city" in params:
        result = result.replace("{start_city}", str(params["start_city"]))

    # 替换时间占位符
    if "{day}" in result and "day" in params:
        result = result.replace("{day}", str(params["day"]))

    if "{day_count}" in result and "day_count" in params:
        result = result.replace("{day_count}", str(params["day_count"]))

    if "{day1}" in result and "day1" in params:
        result = result.replace("{day1}", str(params["day1"]))

    if "{day2}" in result and "day2" in params:
        result = result.replace("{day2}", str(params["day2"]))

    # 替换其他参数占位符
    for param_key, param_value in params.items():
        placeholder = "{" + param_key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(param_value))

    return result


def _find_unresolved_placeholders(rendered_text: str) -> List[str]:
    if not isinstance(rendered_text, str):
        return []
    return sorted(set(re.findall(r"\{[a-zA-Z0-9_]+\}", rendered_text)))


def template_uses_poi_placeholders(template_str: str) -> bool:
    """判断模板是否包含 POI 占位符。"""
    if not isinstance(template_str, str):
        return False
    return any(token in template_str for token in ["{poi}", "{poi1}", "{poi2}"])


def template_refers_existing_poi_semantically(template_str: str, constraint_type: str = "") -> bool:
    """
    仅当模板语义明确指向“原行程已有POI”时返回 True。
    不按 scope 做硬编码，避免误伤 structural/compositional 的新增采样场景。
    """
    text = str(template_str or "")
    ctype = str(constraint_type or "").strip().lower()

    if ctype in {
        "remove_poi",
        "substitute_poi",
        "semantic_substitution",
        "type_replacement",
        "duration_limit",
        "poi_local_radius_limit",
    }:
        return True

    existing_signals = [
        "从行程中删除",
        "去掉{poi}",
        "原行程",
        "原计划",
        "已有",
        "现有",
    ]
    return any(sig in text for sig in existing_signals)


def _parse_time_to_minutes(time_str: Any) -> Optional[int]:
    if not isinstance(time_str, str) or ":" not in time_str:
        return None
    parts = time_str.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour == 24 and minute == 0:
        return 24 * 60
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour * 60 + minute
    return None


def _format_minutes_to_clock(minutes: int) -> str:
    minutes = max(0, min(24 * 60, int(minutes)))
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _parse_duration_text_to_minutes(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(float(value))
    if not isinstance(value, str):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", value)
    if not m:
        return None
    num = float(m.group(1))
    text = value.lower()
    if "小时" in value or "h" in text:
        return int(num * 60)
    if "分钟" in value or "min" in text:
        return int(num)
    return int(num)


def _format_minutes_to_duration_text(minutes: int) -> str:
    minutes = int(max(1, minutes))
    if minutes % 60 == 0:
        return f"{minutes // 60}小时"
    if minutes > 60 and minutes % 30 == 0:
        hours = minutes / 60.0
        if abs(hours - round(hours)) < 1e-9:
            return f"{int(round(hours))}小时"
        return f"{hours:.1f}小时"
    return f"{minutes}分钟"


def _parse_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"(\d+(?:\.\d+)?)", value)
        if m:
            return float(m.group(1))
    return None


def _day_to_cn(day: int) -> str:
    cn = {1: "第一天", 2: "第二天", 3: "第三天", 4: "第四天", 5: "第五天", 6: "第六天", 7: "第七天"}
    return cn.get(day, f"第{day}天")


def _cn_to_day(day_text: Any) -> Optional[int]:
    if not isinstance(day_text, str):
        return None
    mapping = {
        "第一天": 1, "第二天": 2, "第三天": 3, "第四天": 4,
        "第五天": 5, "第六天": 6, "第七天": 7,
    }
    if day_text in mapping:
        return mapping[day_text]
    m = re.search(r"第(\d+)天", day_text)
    if m:
        return int(m.group(1))
    return None


def _coerce_day_value(day_value: Any) -> Optional[int]:
    if isinstance(day_value, int) and day_value > 0:
        return day_value
    if isinstance(day_value, str):
        return _cn_to_day(day_value)
    return None


def _normalize_temporal_list(temporal_raw: Any) -> List[Dict[str, Any]]:
    if isinstance(temporal_raw, dict):
        return [dict(temporal_raw)]
    if isinstance(temporal_raw, list):
        return [dict(item) for item in temporal_raw if isinstance(item, dict)]
    return []


def _extract_query_days(edit_query: str) -> List[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    days: List[int] = []
    for token in re.findall(r"第([一二三四五六七八九十\d]+)天", edit_query):
        day = _cn_to_day(f"第{token}天")
        if day and day not in days:
            days.append(day)
    return days


def _extract_day_poi_pairs_from_query(edit_query: str) -> List[Tuple[int, str]]:
    return shared_extract_day_poi_pairs_from_query(edit_query)


def _extract_replace_source_poi(edit_query: str) -> Optional[str]:
    return shared_extract_replace_source_poi_from_query(edit_query)


def _extract_specific_poi_from_query(edit_query: str) -> Optional[str]:
    return shared_extract_specific_poi_from_query(edit_query)


def _extract_ticket_price_cap_source_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None

    patterns = [
        r"(?:第[一二三四五六七八九十\d]+天)?去([^，。；;]+?)，但如果门票",
        r"安排([^，。；;]+?)，但如果门票",
        r"游玩([^，。；;]+?)，但如果门票",
    ]
    for pattern in patterns:
        m = re.search(pattern, edit_query)
        if not m:
            continue
        name = str(m.group(1)).strip().strip("，。；;,. ")
        if name:
            return name
    return None


def _extract_preferred_poi_type_from_query(edit_query: str) -> Optional[str]:
    return shared_extract_preferred_poi_type_from_query(edit_query)


def _extract_restaurant_type_from_query(edit_query: str) -> Optional[str]:
    return shared_extract_restaurant_type_from_query(edit_query)


def _extract_hotel_feature_from_query(edit_query: str) -> Optional[str]:
    return shared_extract_hotel_feature_from_query(edit_query)


def _extract_room_type_from_query(edit_query: str) -> Optional[int]:
    return shared_extract_room_type_from_query(edit_query)


def _extract_room_count_from_query(edit_query: str) -> Optional[int]:
    return shared_extract_room_count_from_query(edit_query)


def _extract_intercity_transport_modes_from_query(edit_query: str) -> List[str]:
    return shared_extract_intercity_transport_modes_from_query(edit_query)


def _extract_transport_time_windows_from_query(edit_query: str) -> List[Dict[str, str]]:
    return shared_extract_transport_time_windows_from_query(edit_query)


def _extract_activity_budget_requirement(edit_query: str) -> Optional[Dict[str, Any]]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        (r"每人每顿(?:餐饮|饭|用餐)预算(?:不超过|控制在|最多|上限)?\s*(\d+(?:\.\d+)?)\s*元", "meal", "avg_cost_per_meal_per_person"),
        (r"每顿(?:餐饮|饭|用餐)预算(?:不超过|控制在|最多|上限)?\s*(\d+(?:\.\d+)?)\s*元", "meal", "avg_cost_per_meal_per_person"),
        (r"每人每晚(?:住宿|酒店)预算(?:不超过|控制在|最多|上限)?\s*(\d+(?:\.\d+)?)\s*元", "accommodation", "avg_cost_per_person_per_night"),
        (r"每晚(?:住宿|酒店)预算(?:不超过|控制在|最多|上限)?\s*(\d+(?:\.\d+)?)\s*元", "accommodation", "avg_cost_per_person_per_night"),
    ]
    for pattern, activity_type, metric in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        try:
            return {
                "activity_type": activity_type,
                "metric": metric,
                "budget_cap": float(match.group(1)),
            }
        except ValueError:
            continue
    return None


def _extract_radius_anchor_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None

    patterns = [
        r"把(.+?)周边活动半径限制为",
        r"以(.+?)为中心(?:的)?周边活动半径",
        r"围绕(.+?)的活动半径",
        r"想把(.+?)附近的活动安排得更集中",
        r"希望(.+?)附近的活动更集中",
        r"围着(.+?)玩",
        r"以(.+?)为中心安排行程",
        r"希望以(.+?)为中心安排",
        r"围绕(.+?)安排行程",
    ]
    for pattern in patterns:
        m = re.search(pattern, edit_query)
        if not m:
            continue
        anchor = str(m.group(1)).strip().strip("，。；;,. ")
        if anchor:
            return anchor
    return None


def _extract_semantic_target_from_query(edit_query: str) -> Optional[str]:
    return shared_extract_semantic_target_from_query(edit_query)


def _extract_time_period_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    for token in ["上午", "下午", "晚上"]:
        if token in edit_query:
            return token
    return None


def _extract_ordered_pois_from_query(edit_query: str) -> List[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    patterns = [
        r"先去([^，。；;]+?)再去([^，。；;]+)",
        r"调整行程顺序，先去([^，。；;]+?)再去([^，。；;]+)",
        r"把([^和，。；;]+?)和([^，。；;]+?)安排得更紧凑",
        r"把([^和，。；;]+?)和([^，。；;]+?)安排在同一",
        r"把([^和，。；;]+?)和([^，。；;]+?)之间的交通方式改为步行",
    ]
    for pattern in patterns:
        m = re.search(pattern, edit_query)
        if not m:
            continue
        names = [str(m.group(1)).strip().strip("，。；;,. "), str(m.group(2)).strip().strip("，。；;,. ")]
        if all(names):
            return names
    return []


def _extract_budget_cap_source_poi(edit_query: str) -> Optional[str]:
    return _extract_ticket_price_cap_source_poi_from_query(edit_query) or _extract_poi_from_query_like(edit_query)


def _extract_poi_from_query_like(edit_query: str) -> Optional[str]:
    return (
        shared_extract_anchor_poi_from_query(edit_query)
        or shared_extract_hotel_name_from_query(edit_query)
        or shared_extract_restaurant_name_from_query(edit_query)
    )


def _extract_requested_day_count_from_query(edit_query: str) -> Optional[int]:
    return shared_extract_requested_day_count_from_query(edit_query)


def _extract_requested_day_delta_from_query(edit_query: str) -> Optional[int]:
    return shared_extract_requested_day_delta_from_query(edit_query)


def _extract_daily_poi_cap_from_query(edit_query: str) -> Optional[int]:
    return shared_extract_daily_poi_cap_from_query(edit_query)


def _origin_day_count_from_plan(origin_plan: Optional[Dict[str, Any]]) -> int:
    if not isinstance(origin_plan, dict):
        return 1
    itinerary = origin_plan.get("itinerary")
    if isinstance(itinerary, list) and itinerary:
        return len(itinerary)
    day_count = origin_plan.get("day_count") or origin_plan.get("days")
    try:
        value = int(day_count)
    except (TypeError, ValueError):
        value = 1
    return max(value, 1)


def _resolve_resulting_day_count(
    *,
    constraint_type: str,
    origin_plan: Optional[Dict[str, Any]],
    requested_day_count: Optional[int],
    requested_day_delta: Optional[int],
) -> Optional[int]:
    if constraint_type != "add_days":
        return requested_day_count
    if requested_day_delta is None:
        return requested_day_count
    return _origin_day_count_from_plan(origin_plan) + int(requested_day_delta)


def _extract_ticket_price_cap_source_poi(
    constraints_data: Dict[str, Any],
    edit_query: str,
    fallback_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    semantic = constraints_data.get("semantic", {})
    if isinstance(semantic, dict):
        replace_from = semantic.get("replace_from")
        if isinstance(replace_from, str) and replace_from.strip():
            return replace_from.strip()

    for item in _normalize_temporal_list(constraints_data.get("temporal", [])):
        must_visit = item.get("must_visit")
        if isinstance(must_visit, str) and must_visit.strip():
            return must_visit.strip()
        if isinstance(must_visit, list):
            for value in must_visit:
                if isinstance(value, str) and value.strip():
                    return value.strip()

    source_poi = _extract_ticket_price_cap_source_poi_from_query(edit_query)
    if source_poi:
        return source_poi

    if isinstance(fallback_params, dict):
        poi_name = fallback_params.get("poi_name")
        if isinstance(poi_name, str) and poi_name.strip():
            return poi_name.strip()

    return None


SAFE_CANONICALIZATION_TYPES = {"semantic_substitution", "cross_day_restructure"}


def _semantic_slots_need_llm_fallback(semantic_slots: Dict[str, Any], constraint_type: str) -> bool:
    ctype = str(constraint_type or "").strip()
    if ctype not in SEMANTIC_SLOT_FALLBACK_TYPES:
        return False
    if not isinstance(semantic_slots, dict):
        return True

    confidence = semantic_slots.get("parse_confidence")
    try:
        if confidence is not None and float(confidence) < 0.75:
            return True
    except (TypeError, ValueError):
        return True

    errors = semantic_slots.get("parse_errors", [])
    if isinstance(errors, list) and errors:
        return True

    required_fields = {
        "semantic_and_specific": ("specific_poi", "preferred_type"),
        "category_overlap": ("specific_poi", "preferred_type"),
        "semantic_substitution": ("source_poi", "semantic_target"),
        "type_filter": ("preferred_type",),
        "preference_constraint": ("preferred_type",),
        "required_restaurant_type": ("restaurant_type",),
        "required_hotel_feature": ("hotel_feature",),
        "resource_overlap": ("anchor_poi", "budget_cap"),
        "multi_city_split": ("city_split_targets",),
        "remove_poi": ("source_poi",),
        "cross_day_restructure": ("day_poi_pairs", "day_count"),
        "poi_local_radius_limit": ("anchor_poi", "distance_km"),
    }
    for key in required_fields.get(ctype, ()):
        value = semantic_slots.get(key)
        if value in (None, "", []):
            return True
    if ctype in {"semantic_and_specific", "category_overlap"}:
        specific_poi = str(semantic_slots.get("specific_poi") or "").strip()
        preferred_type = str(semantic_slots.get("preferred_type") or "").strip()
        if specific_poi.startswith(("我想", "想看", "想去")):
            return True
        if preferred_type.endswith("我") or preferred_type in {"其它", "其他", "类似"}:
            return True
    if ctype == "remove_poi":
        source_poi = str(semantic_slots.get("source_poi") or semantic_slots.get("anchor_poi") or "").strip()
        if source_poi.startswith(("掉", "删")) or source_poi in {"掉", "删", "删除"}:
            return True
    if ctype == "cross_day_restructure":
        for pair in semantic_slots.get("day_poi_pairs") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return True
            poi_name = str(pair[1] or "").strip()
            if poi_name.endswith(("？", "?", "。", "！", "!")):
                return True
    return False


def _parse_semantic_slots_with_llm(
    llm_client,
    sample_id: str,
    edit_query: str,
    constraint_type: str,
    template_str: str,
    deterministic_slots: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if llm_client is None:
        return None
    prompt = f"""你是一个信息抽取器。请只从下面这句中文旅游编辑需求中抽取结构化语义槽位，不能补造不存在的信息。

constraint_type: {constraint_type}
template: {template_str}
edit_query: {edit_query}

规则：
- 只能返回 JSON，不要解释。
- specific_poi/source_poi/anchor_poi/restaurant_name/hotel_name 必须是 query 原文中出现过的片段。
- preferred_type/semantic_target/restaurant_type/hotel_feature 必须是短语，不能包含“去/看看/安排/想去/类似的景点”等整句成分。
- 如果 query 只说“附近用餐/周边用餐/附近餐厅”，restaurant_name 必须为 null，并把 nearby_meal 设为 true。
- 如果 query 表达“X这样的历史古迹，还想看其他类似景点”，要拆成 specific_poi=X, preferred_type=历史古迹, additional_similar=true。
- 如果 query 表达“杭州和周边另一个城市”之类双城拆分，city_split_targets 返回两个目标槽位。

deterministic_parse:
{json.dumps(deterministic_slots, ensure_ascii=False)}

返回字段白名单：
{{
  "specific_poi": null,
  "preferred_type": null,
  "source_poi": null,
  "semantic_target": null,
  "anchor_poi": null,
  "day_poi_pairs": [],
  "day_count": null,
  "budget_cap": null,
  "distance_km": null,
  "restaurant_name": null,
  "restaurant_type": null,
  "hotel_name": null,
  "hotel_feature": null,
  "city_split_targets": [],
  "additional_similar": false,
  "nearby_meal": false
}}
"""
    messages = [
        {"role": "system", "content": "You extract structured travel-edit semantics and return JSON only."},
        {"role": "user", "content": prompt},
    ]
    debug_logger = create_llm_debug_logger("02", sample_id, "semantic_slot_parse")
    response = llm_client.call_with_retry(
        messages,
        max_retries=1,
        temperature=0.0,
        debug_logger=debug_logger,
        debug_context={
            "constraint_type": constraint_type,
            "semantic_slot_fallback": True,
        },
    )
    if response is None:
        return None
    parsed = parse_llm_response(response.content)
    return parsed if isinstance(parsed, dict) else None


def _parse_semantic_slots_with_fallback(
    *,
    llm_client,
    sample_id: str,
    edit_query: str,
    constraint_type: str,
    template_str: str,
    expected_slots: Optional[Dict[str, Any]] = None,
    anchor_hint: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    hints: Dict[str, Any] = {}
    if isinstance(expected_slots, dict) and expected_slots:
        hints["expected_slots"] = expected_slots
    if isinstance(anchor_hint, str) and anchor_hint.strip():
        hints["anchor_poi"] = anchor_hint.strip()

    semantic_slots = parse_query_semantic_slots(
        edit_query,
        str(constraint_type),
        template_str=template_str,
        hints=hints or None,
    )
    semantic_parse_source = str(semantic_slots.get("parse_source") or "regex")

    if _semantic_slots_need_llm_fallback(semantic_slots, str(constraint_type)):
        llm_slots = _parse_semantic_slots_with_llm(
            llm_client=llm_client,
            sample_id=sample_id,
            edit_query=edit_query,
            constraint_type=str(constraint_type),
            template_str=template_str,
            deterministic_slots=semantic_slots,
        )
        if isinstance(llm_slots, dict):
            semantic_slots = parse_query_semantic_slots(
                edit_query,
                str(constraint_type),
                template_str=template_str,
                hints={
                    "expected_slots": llm_slots,
                    "anchor_poi": (
                        semantic_slots.get("anchor_poi")
                        or llm_slots.get("anchor_poi")
                        or anchor_hint
                    ),
                },
            )
            semantic_slots = merge_semantic_slots(
                semantic_slots,
                {
                    "parse_source": "llm",
                    "parse_errors": semantic_slots.get("parse_errors", []),
                },
            )
            semantic_slots["parse_source"] = "llm"
            semantic_parse_source = "llm"

    semantic_slots.setdefault("parse_source", semantic_parse_source)
    return semantic_slots, semantic_parse_source


def _normalize_surface_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s，。；;,.！？!?、“”\"'：:（）()]+", "", text).strip().lower()


def _normalize_surface_candidate(text: str, constraint_type: str = "") -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""

    def _replace_compact_cn(match: re.Match[str]) -> str:
        value = _parse_compact_cn_numeral(match.group(1))
        if value is None:
            return match.group(0)
        return f"{value}{match.group(2)}"

    normalized = re.sub(
        r"([零〇一二两三四五六七八九十百千]+)(元|块|块钱|分钟|小时|天|个景点|个地方|个点)",
        _replace_compact_cn,
        normalized,
    )
    normalized = normalized.replace("块钱", "元").replace("块", "元")

    def _replace_compact_cn_without_unit(match: re.Match[str]) -> str:
        value = _parse_compact_cn_numeral(match.group(2))
        if value is None:
            return match.group(0)
        return f"{match.group(1)}{value}元"

    if constraint_type in {"budget_limit", "ticket_budget_limit", "budget_cap_preference", "budget_target_update"}:
        normalized = re.sub(
            r"((?:总)?预算[^，。；]{0,12}?)([零〇一二两三四五六七八九十百千]+)(?=(?:以内|以下|左右|就行|即可|也可以|别超过|不超过|上限|内))",
            _replace_compact_cn_without_unit,
            normalized,
        )
        normalized = re.sub(
            r"(门票(?:总价|预算)[^，。；]{0,12}?)([零〇一二两三四五六七八九十百千]+)(?=(?:以内|以下|左右|就行|即可|也可以|别超过|不超过|内))",
            _replace_compact_cn_without_unit,
            normalized,
        )
        normalized = re.sub(
            r"((?:总)?预算[^，。；]{0,12}?)(\d+)(?!\s*元)(?=(?:以内|以下|左右|就行|即可|也可以|别超过|不超过|上限|内))",
            r"\1\2元",
            normalized,
        )
        normalized = re.sub(
            r"(门票(?:总价|预算)[^，。；]{0,12}?)(\d+)(?!\s*元)(?=(?:以内|以下|左右|就行|即可|也可以|别超过|不超过|内))",
            r"\1\2元",
            normalized,
        )
    if constraint_type == "travel_time_constraint":
        normalized = re.sub(
            r"(交通时间[^，。；]{0,12}?)(\d+)(?!\s*(?:分钟|小时))(?=(?:以内|以下|左右|就行|即可|别超过|不超过|内))",
            r"\1\2分钟",
            normalized,
        )
    return normalized


def _clean_query_surface_response(content: Any, constraint_type: str = "") -> str:
    if not isinstance(content, str):
        return ""
    text = content.strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.strip().lstrip("-*0123456789. ").strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    text = lines[0]
    text = text.strip("“”\"' ")
    return _normalize_surface_candidate(text, constraint_type=str(constraint_type or "").strip())


def _prune_rendered_params_for_trace(template_str: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    kept: Dict[str, Any] = {}
    for key, value in params.items():
        placeholder = "{" + str(key) + "}"
        if key in {"start_city", "target_city", "day_count"} or placeholder in str(template_str):
            kept[key] = value
    return kept


def _render_surface_variants(
    template: Dict[str, Any],
    target_city: str,
    pois: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[str]:
    variants = template.get("surface_variants", [])
    if not isinstance(variants, list):
        return []

    rendered: List[str] = []
    seen = set()
    for variant in variants:
        if not isinstance(variant, str) or not variant.strip():
            continue
        rendered_variant = fill_template(variant, target_city, pois, params)
        unresolved = _find_unresolved_placeholders(rendered_variant)
        if unresolved:
            continue
        if rendered_variant not in seen:
            seen.add(rendered_variant)
            rendered.append(rendered_variant)
    return rendered


def assess_query_surface_quality(
    candidate_query: str,
    canonical_query: str,
    origin_query_text: str = "",
) -> Tuple[bool, str]:
    candidate = str(candidate_query or "").strip()
    canonical = str(canonical_query or "").strip()
    if not candidate:
        return False, "empty_surface"

    if QUERY_SURFACE_REGISTRY.get(candidate, 0) > 0:
        return False, "duplicate_surface_in_batch"

    normalized_candidate = _normalize_surface_text(candidate)
    normalized_canonical = _normalize_surface_text(canonical)
    if normalized_candidate == normalized_canonical:
        return False, "identical_to_canonical"

    similarity = SequenceMatcher(a=normalized_candidate, b=normalized_canonical).ratio()
    phrase_hits = sum(1 for phrase in SURFACE_STIFF_PHRASES if phrase in candidate)
    if similarity >= 0.92 and phrase_hits >= 1:
        return False, "too_close_to_canonical"
    if phrase_hits >= 2:
        return False, "engineering_tone"

    origin_text = str(origin_query_text or "").strip()
    if origin_text:
        for pattern in ORIGIN_LEAK_PATTERNS:
            for match in re.finditer(pattern, origin_text):
                leaked = str(match.group(0)).strip()
                if leaked and leaked in candidate and leaked not in canonical:
                    return False, "origin_context_leak"

    return True, ""


def register_query_surface(query_text: str) -> None:
    query = str(query_text or "").strip()
    if not query:
        return
    QUERY_SURFACE_REGISTRY[query] = QUERY_SURFACE_REGISTRY.get(query, 0) + 1


def rewrite_query_surface(
    *,
    llm_client,
    sample_id: str,
    canonical_query: str,
    constraint_type: str,
    evidence_trace: Dict[str, Any],
    soft_target_str: str,
    origin_query_text: str,
    surface_variants: List[str],
    rewrite_attempt: int = 0,
) -> Optional[str]:
    if llm_client is None:
        return None

    examples_block = "\n".join(f"- {item}" for item in surface_variants[:3]) if surface_variants else "(none)"
    evidence_block = format_evidence_for_prompt(constraint_type, evidence_trace)
    controlled_surface = str(constraint_type or "").strip() in CONTROLLED_SURFACE_TYPES
    extra_rules: List[str] = []
    if controlled_surface:
        extra_rules.append("- 必须保留阿拉伯数字，不要写成中文数字")
        if constraint_type in {"budget_limit", "ticket_budget_limit", "ticket_price_cap_or_substitute", "budget_cap_preference", "budget_target_update"}:
            extra_rules.append("- 预算金额必须写成“800元”这种形式，不要写“八百”或“800块”")
        if constraint_type == "travel_time_constraint":
            extra_rules.append("- 时间必须写成“50分钟”或“1小时”这种形式")
    if rewrite_attempt > 0 and controlled_surface:
        extra_rules.append("- 上一版没有通过，请更明确地保留数字和单位")
    extra_rules_block = "\n".join(extra_rules)
    prompt = f"""你是中文旅游用户表达改写助手。请把下面这条编辑需求改写得更像真实用户会说的话。

你只能改写表达方式，不能改变任何事实或约束：
- 城市、POI 名称、天数、预算、距离、时间、顺序必须保持不变
- 不能新增或删除要求
- 输出只要一句中文，不要解释，不要加引号，不要分点
- 只表达这条 edit request 本身，不要复述原始行程背景
- 不要重复人数、出发地、住宿、交通方式、酒店要求这类 origin 信息
{extra_rules_block}

当前 canonical query：
{canonical_query}

constraint_type: {constraint_type}
soft target: {soft_target_str}
证据锚点：
{evidence_block}

可参考但不要照抄的自然表达：
{examples_block}

请给出一条更自然的改写。"""

    debug_logger = create_llm_debug_logger("02", sample_id, "rewrite_query_surface")
    response = llm_client.call_with_retry(
        [
            {"role": "system", "content": "You rewrite Chinese travel edit requests while preserving every factual constraint."},
            {"role": "user", "content": prompt},
        ],
        max_retries=1,
        temperature=float(os.getenv("STEP2_SURFACE_TEMPERATURE", DEFAULT_SURFACE_REWRITE_TEMPERATURE)),
        max_tokens=160,
        debug_logger=debug_logger,
        debug_context={
            "constraint_type": constraint_type,
            "surface_rewrite": True,
        },
    )
    if response is None:
        return None
    rewritten = _clean_query_surface_response(response.content, str(constraint_type or "").strip())
    return rewritten or None


def select_query_surface(
    *,
    llm_client,
    sample_id: str,
    canonical_query: str,
    constraint_type: str,
    template: Dict[str, Any],
    target_city: str,
    pois: List[Dict[str, Any]],
    params: Dict[str, Any],
    evidence_trace: Dict[str, Any],
    soft_target_str: str,
    origin_query_text: str,
    constraints_data: Dict[str, Any],
    fallback_params: Dict[str, Any],
    evidence: Dict[str, Any],
    canonical_query_spec: Dict[str, Any],
    canonical_semantic_slots: Dict[str, Any],
    canonical_proof: FeasibilityResult,
    target_family: str,
    target_facet: str,
    target_mode: str,
    origin_profile: List[Dict[str, Any]],
) -> Dict[str, Any]:
    variants = _render_surface_variants(template, target_city, pois, params)
    last_reject_reason = "rewrite_disabled"
    for variant in variants:
        quality_pass, reject_reason = assess_query_surface_quality(
            variant,
            canonical_query,
            origin_query_text=origin_query_text,
        )
        if not quality_pass:
            last_reject_reason = reject_reason
            continue

        variant_semantic_slots, variant_parse_source = _parse_semantic_slots_with_fallback(
            llm_client=None,
            sample_id=sample_id,
            edit_query=variant,
            constraint_type=str(constraint_type or "").strip(),
            template_str=str(template.get("template", "")),
            expected_slots=canonical_semantic_slots,
            anchor_hint=(
                canonical_semantic_slots.get("anchor_poi")
                or canonical_query_spec.get("anchor_poi")
                or (params.get("poi") if isinstance(params, dict) else None)
            ),
        )
        variant_query_spec = extract_query_spec(
            variant,
            str(constraint_type or "").strip(),
            semantic_slots=variant_semantic_slots,
            origin_plan=fallback_params.get("_origin_plan"),
        )
        preservation_errors = validate_surface_query_spec_preservation(
            canonical_query_spec,
            variant,
            constraint_type=str(constraint_type or "").strip(),
            surface_semantic_slots=variant_semantic_slots,
            origin_plan=fallback_params.get("_origin_plan"),
        )
        if preservation_errors:
            last_reject_reason = "variant_spec_drift:" + "; ".join(preservation_errors)
            continue

        variant_proof = validate_conflict_trigger(
            constraints_data=constraints_data,
            evidence=evidence,
            constraint_type=constraint_type,
            fallback_params=fallback_params,
            edit_query=variant,
        )
        if variant_proof is None or not variant_proof.ok:
            last_reject_reason = (
                variant_proof.reason_code if variant_proof is not None else "variant_proof_missing"
            )
            continue

        soft_alignment_errors = validate_soft_query_alignment(
            edit_query=variant,
            origin_profile=origin_profile,
            target_family=target_family,
            target_facet=target_facet,
            target_mode=target_mode,
            template_soft_intensity=str(template.get("soft_intensity", "")).strip(),
            constraint_type=str(constraint_type or "").strip(),
            semantic_slots=variant_semantic_slots,
            query_spec=variant_query_spec,
        )
        if soft_alignment_errors:
            last_reject_reason = "variant_soft_drift:" + "; ".join(soft_alignment_errors)
            continue

        return {
            "edit_query": variant,
            "canonical_query_spec": canonical_query_spec,
            "canonical_semantic_slots": canonical_semantic_slots,
            "surface_query_spec": variant_query_spec,
            "surface_semantic_slots": variant_semantic_slots,
            "surface_semantic_parse_source": variant_parse_source,
            "surface_proof": variant_proof,
            "query_surface_source": "surface_variant",
            "surface_quality_pass": True,
            "surface_reject_reason": "",
        }

    if llm_client is None:
        return {
            "edit_query": canonical_query,
            "canonical_query_spec": canonical_query_spec,
            "canonical_semantic_slots": canonical_semantic_slots,
            "surface_query_spec": canonical_query_spec,
            "surface_semantic_slots": canonical_semantic_slots,
            "surface_semantic_parse_source": str(canonical_semantic_slots.get("parse_source") or "regex"),
            "surface_proof": canonical_proof,
            "query_surface_source": "template_fallback",
            "surface_quality_pass": False,
            "surface_reject_reason": last_reject_reason,
        }

    for rewrite_attempt in range(DEFAULT_SURFACE_REWRITE_ATTEMPTS):
        rewritten = rewrite_query_surface(
            llm_client=llm_client,
            sample_id=sample_id,
            canonical_query=canonical_query,
            constraint_type=constraint_type,
            evidence_trace=evidence_trace,
            soft_target_str=soft_target_str,
            origin_query_text=origin_query_text,
            surface_variants=variants,
            rewrite_attempt=rewrite_attempt,
        )
        if not rewritten:
            last_reject_reason = "rewrite_missing"
            continue

        quality_pass, reject_reason = assess_query_surface_quality(
            rewritten,
            canonical_query,
            origin_query_text=origin_query_text,
        )
        if not quality_pass:
            last_reject_reason = reject_reason
            continue

        preservation_errors = validate_surface_query_spec_preservation(
            canonical_query_spec,
            rewritten,
            constraint_type=str(constraint_type or "").strip(),
            surface_semantic_slots=canonical_semantic_slots,
            origin_plan=fallback_params.get("_origin_plan"),
        )
        if preservation_errors:
            last_reject_reason = "spec_drift:" + "; ".join(preservation_errors)
            continue

        rewritten_semantic_slots, rewritten_parse_source = _parse_semantic_slots_with_fallback(
            llm_client=llm_client,
            sample_id=sample_id,
            edit_query=rewritten,
            constraint_type=str(constraint_type or "").strip(),
            template_str=str(template.get("template", "")),
            expected_slots=canonical_semantic_slots,
            anchor_hint=(
                canonical_semantic_slots.get("anchor_poi")
                or canonical_query_spec.get("anchor_poi")
                or (params.get("poi") if isinstance(params, dict) else None)
            ),
        )
        rewritten_query_spec = extract_query_spec(
            rewritten,
            str(constraint_type or "").strip(),
            semantic_slots=rewritten_semantic_slots,
            origin_plan=fallback_params.get("_origin_plan"),
        )
        preservation_errors = validate_surface_query_spec_preservation(
            canonical_query_spec,
            rewritten,
            constraint_type=str(constraint_type or "").strip(),
            surface_semantic_slots=rewritten_semantic_slots,
            origin_plan=fallback_params.get("_origin_plan"),
        )
        if preservation_errors:
            last_reject_reason = "spec_drift:" + "; ".join(preservation_errors)
            continue

        rewritten_proof = validate_conflict_trigger(
            constraints_data=constraints_data,
            evidence=evidence,
            constraint_type=constraint_type,
            fallback_params=fallback_params,
            edit_query=rewritten,
        )
        if rewritten_proof is None or not rewritten_proof.ok:
            last_reject_reason = (
                rewritten_proof.reason_code if rewritten_proof is not None else "surface_proof_missing"
            )
            continue

        soft_alignment_errors = validate_soft_query_alignment(
            edit_query=rewritten,
            origin_profile=origin_profile,
            target_family=target_family,
            target_facet=target_facet,
            target_mode=target_mode,
            template_soft_intensity=str(template.get("soft_intensity", "")).strip(),
            constraint_type=str(constraint_type or "").strip(),
            semantic_slots=rewritten_semantic_slots,
            query_spec=rewritten_query_spec,
        )
        if soft_alignment_errors:
            last_reject_reason = "soft_drift:" + "; ".join(soft_alignment_errors)
            continue

        return {
            "edit_query": rewritten,
            "canonical_query_spec": canonical_query_spec,
            "canonical_semantic_slots": canonical_semantic_slots,
            "surface_query_spec": rewritten_query_spec,
            "surface_semantic_slots": rewritten_semantic_slots,
            "surface_semantic_parse_source": rewritten_parse_source,
            "surface_proof": rewritten_proof,
            "query_surface_source": "llm_rewrite",
            "surface_quality_pass": True,
            "surface_reject_reason": "",
        }

    return {
        "edit_query": canonical_query,
        "canonical_query_spec": canonical_query_spec,
        "canonical_semantic_slots": canonical_semantic_slots,
        "surface_query_spec": canonical_query_spec,
        "surface_semantic_slots": canonical_semantic_slots,
        "surface_semantic_parse_source": str(canonical_semantic_slots.get("parse_source") or "regex"),
        "surface_proof": canonical_proof,
        "query_surface_source": "template_fallback",
        "surface_quality_pass": False,
        "surface_reject_reason": last_reject_reason,
    }


def _extract_query_budget_value(edit_query: str) -> Optional[int]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query or "")
    if not match:
        return None
    return int(float(match.group(1)))


def _extract_query_distance_value(edit_query: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query or "")
    if not match:
        return None
    return float(match.group(1))


def _extract_query_duration_minutes(edit_query: str) -> Optional[int]:
    text = str(edit_query or "")
    patterns = [
        re.compile(
            r"(?:限制在|控制在|最好控制在|尽量控制在|不超过|最多|别超过|最好不超过|不要超过|压到|压在)\s*(\d+(?:\.\d+)?)\s*(小时|分钟)(?:左右|以内|之内|内)?"
        ),
        re.compile(
            r"(?:限制在|控制在|最好控制在|尽量控制在|不超过|最多|别超过|最好不超过|不要超过|压到|压在)\s*半小时(?:左右|以内|之内|内)?"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        if match.lastindex == 2:
            amount = float(match.group(1))
            unit = match.group(2)
            return int(amount * 60) if unit == "小时" else int(amount)
        return 30
    return None


def _extract_query_duration_text(edit_query: str) -> Optional[str]:
    text = str(edit_query or "")
    patterns = [
        re.compile(
            r"(?:限制在|控制在|最好控制在|尽量控制在|不超过|最多|别超过|最好不超过|不要超过|压到|压在)\s*(\d+(?:\.\d+)?)\s*(小时|分钟)(?:左右|以内|之内|内)?"
        ),
        re.compile(
            r"(?:限制在|控制在|最好控制在|尽量控制在|不超过|最多|别超过|最好不超过|不要超过|压到|压在)\s*(半小时)(?:左右|以内|之内|内)?"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        if match.lastindex == 2:
            amount = match.group(1)
            unit = match.group(2)
            return f"{amount}{unit}"
        return "30分钟"
    return None


def _extract_query_end_before(edit_query: str) -> Optional[str]:
    match = re.search(r"(\d{1,2}:\d{2})前", edit_query or "")
    if not match:
        return None
    return match.group(1)


def extract_query_spec(
    edit_query: str,
    constraint_type: str,
    semantic_slots: Optional[Dict[str, Any]] = None,
    origin_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"constraint_type": str(constraint_type or "").strip()}
    ctype = spec["constraint_type"]
    slots = semantic_slots if isinstance(semantic_slots, dict) else parse_query_semantic_slots(edit_query, ctype)
    day_pairs = slots.get("day_poi_pairs") or _extract_day_poi_pairs_from_query(edit_query)
    if day_pairs:
        spec["day_poi_pairs"] = day_pairs

    if ctype == "duration_limit":
        spec["anchor_poi"] = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        spec["duration_limit_min"] = slots.get("duration_limit_min") or _extract_query_duration_minutes(edit_query)
    elif ctype == "time_limit":
        spec["day"] = _coerce_day_value(next(iter(_extract_query_days(edit_query)), None))
        spec["end_before"] = _extract_query_end_before(edit_query)
    elif ctype == "poi_local_radius_limit":
        spec["anchor_poi"] = slots.get("anchor_poi") or _extract_radius_anchor_poi_from_query(edit_query)
        spec["distance_km"] = slots.get("distance_km") or _extract_query_distance_value(edit_query)
    elif ctype in {"max_distance_constraint", "walking_distance_limit", "spatial_overflow"}:
        spec["distance_km"] = slots.get("distance_km") or _extract_query_distance_value(edit_query)
        ordered_pois = _extract_ordered_pois_from_query(edit_query)
        if ordered_pois:
            spec["ordered_pois"] = ordered_pois
    elif ctype in {"budget_limit", "daily_budget_infeasible", "budget_cap_preference", "budget_target_update", "resource_overlap"}:
        spec["budget_cap"] = slots.get("budget_cap") or _extract_query_budget_value(edit_query)
        poi_name = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        if poi_name:
            spec["anchor_poi"] = poi_name
        if ctype == "resource_overlap":
            if slots.get("restaurant_name"):
                spec["restaurant_name"] = slots.get("restaurant_name")
            if slots.get("nearby_meal"):
                spec["nearby_meal"] = True
    elif ctype in {"travel_time_constraint", "access_time_infeasible"}:
        spec["anchor_poi"] = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        spec["duration_limit_min"] = slots.get("duration_limit_min") or _extract_query_duration_minutes(edit_query)
    elif ctype == "ticket_budget_limit":
        spec["ticket_budget_cap"] = slots.get("budget_cap") or _extract_query_budget_value(edit_query)
    elif ctype == "ticket_price_cap_or_substitute":
        spec["source_poi"] = slots.get("source_poi") or _extract_ticket_price_cap_source_poi({}, edit_query)
        spec["ticket_price_cap"] = slots.get("budget_cap") or _extract_query_budget_value(edit_query)
    elif ctype == "semantic_substitution":
        spec["source_poi"] = slots.get("source_poi") or _extract_replace_source_poi(edit_query)
        spec["semantic_target"] = slots.get("semantic_target") or _extract_semantic_target_from_query(edit_query)
    elif ctype in {"semantic_and_specific", "category_overlap"}:
        spec["specific_poi"] = slots.get("specific_poi") or _extract_specific_poi_from_query(edit_query)
        spec["preferred_type"] = slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
        if ctype == "category_overlap" and slots.get("additional_similar"):
            spec["additional_similar"] = True
    elif ctype in {"remove_poi", "substitute_poi", "famous_substitution", "type_replacement"}:
        spec["source_poi"] = slots.get("source_poi") or _extract_replace_source_poi(edit_query) or _extract_poi_from_query_like(edit_query)
    elif ctype in {"type_filter", "preference_constraint"}:
        spec["preferred_type"] = slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
    elif ctype == "required_restaurant_name":
        spec["anchor_poi"] = slots.get("restaurant_name") or _extract_poi_from_query_like(edit_query)
    elif ctype == "required_restaurant_type":
        restaurant_type = slots.get("restaurant_type") or _extract_restaurant_type_from_query(edit_query)
        spec["restaurant_type"] = restaurant_type
        spec["preferred_type"] = restaurant_type
    elif ctype == "required_hotel_name":
        spec["anchor_poi"] = slots.get("hotel_name") or _extract_poi_from_query_like(edit_query)
    elif ctype == "required_hotel_feature":
        spec["hotel_feature"] = slots.get("hotel_feature") or _extract_hotel_feature_from_query(edit_query)
    elif ctype == "required_room_type":
        spec["room_type"] = slots.get("hotel_room_type") or _extract_room_type_from_query(edit_query)
    elif ctype == "required_room_count":
        spec["room_count"] = slots.get("hotel_room_count") or _extract_room_count_from_query(edit_query)
    elif ctype == "required_intercity_transport_type":
        spec["intercity_transport_modes"] = slots.get("intercity_transport_modes") or _extract_intercity_transport_modes_from_query(edit_query)
    elif ctype == "transport_time_window":
        spec["transport_time_windows"] = slots.get("transport_time_windows") or _extract_transport_time_windows_from_query(edit_query)
    elif ctype == "activity_budget_limit":
        activity_budget = _extract_activity_budget_requirement(edit_query)
        if activity_budget:
            spec.update(activity_budget)
    elif ctype == "time_window":
        spec["anchor_poi"] = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        spec["time_period"] = slots.get("time_period") or _extract_time_period_from_query(edit_query)
        spec["day"] = _coerce_day_value(next(iter(_extract_query_days(edit_query)), None))
    elif ctype in {"order_constraint", "same_day_no_overlap_compact", "same_time_period_no_overlap"}:
        ordered_pois = _extract_ordered_pois_from_query(edit_query)
        if ordered_pois:
            spec["ordered_pois"] = ordered_pois
        spec["time_period"] = slots.get("time_period") or _extract_time_period_from_query(edit_query)
        spec["day"] = _coerce_day_value(next(iter(_extract_query_days(edit_query)), None))
    elif ctype == "multi_day_requirement":
        spec["day_count"] = slots.get("day_count") or _extract_requested_day_count_from_query(edit_query)
    elif ctype == "add_days":
        day_delta = slots.get("day_delta")
        if day_delta is None:
            day_delta = _extract_requested_day_delta_from_query(edit_query)
        day_count = _resolve_resulting_day_count(
            constraint_type=ctype,
            origin_plan=origin_plan,
            requested_day_count=slots.get("day_count") or _extract_requested_day_count_from_query(edit_query),
            requested_day_delta=day_delta,
        )
        if day_delta is not None:
            spec["day_delta"] = day_delta
        if day_count is not None:
            spec["day_count"] = day_count
        spec["daily_poi_cap"] = slots.get("daily_poi_cap") or _extract_daily_poi_cap_from_query(edit_query)
        spec["preferred_type"] = slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
    elif ctype in {"multi_day_budget_overflow", "extend_with_pacing"}:
        spec["day_count"] = slots.get("day_count") or _extract_requested_day_count_from_query(edit_query)
        spec["daily_poi_cap"] = slots.get("daily_poi_cap") or _extract_daily_poi_cap_from_query(edit_query)
    elif ctype == "multi_city_split":
        targets = slots.get("city_split_targets") or []
        if targets:
            spec["city_split_targets"] = targets

    return {key: value for key, value in spec.items() if value is not None and value != []}


def validate_surface_query_spec_preservation(
    expected_spec: Dict[str, Any],
    surface_query: str,
    *,
    constraint_type: str = "",
    surface_semantic_slots: Optional[Dict[str, Any]] = None,
    origin_plan: Optional[Dict[str, Any]] = None,
) -> List[str]:
    actual_spec = extract_query_spec(
        surface_query,
        constraint_type or expected_spec.get("constraint_type", ""),
        semantic_slots=surface_semantic_slots,
        origin_plan=origin_plan,
    )
    errors: List[str] = []
    for key, expected in expected_spec.items():
        if key == "constraint_type":
            continue
        actual = actual_spec.get(key)
        if key in {"anchor_poi", "source_poi", "specific_poi", "semantic_target", "preferred_type"}:
            actual = expected if isinstance(expected, str) and expected and expected in str(surface_query or "") else actual
        elif key in {"day_count", "daily_poi_cap"} and actual is None:
            actual = extract_query_spec(surface_query, "extend_with_pacing").get(key)
        elif key == "day_delta" and actual is None:
            actual = _extract_requested_day_delta_from_query(surface_query)
        elif key == "distance_km" and actual is None:
            actual = _extract_query_distance_value(surface_query)
        elif key in {"budget_cap", "ticket_budget_cap", "ticket_price_cap"} and actual is None:
            actual = _extract_query_budget_value(surface_query)
        elif key == "duration_limit_min" and actual is None:
            actual = _extract_query_duration_minutes(surface_query)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-6:
                errors.append(f"{key} changed: expected {expected}, got {actual}")
        elif isinstance(expected, list):
            if list(actual or []) != list(expected):
                errors.append(f"{key} changed: expected {expected}, got {actual}")
        else:
            if actual != expected:
                errors.append(f"{key} changed: expected {expected}, got {actual}")
    return errors


def validate_query_spec_alignment(
    query_spec: Dict[str, Any],
    constraints_data: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    ctype = str(query_spec.get("constraint_type", "")).strip()
    must_include_names = [
        poi.get("name")
        for poi in constraints_data.get("must_include", [])
        if isinstance(poi, dict) and isinstance(poi.get("name"), str) and poi.get("name")
    ]
    temporal = _normalize_temporal_list(constraints_data.get("temporal", []))
    semantic = constraints_data.get("semantic", {}) if isinstance(constraints_data.get("semantic", {}), dict) else {}
    forbidden = [
        name for name in constraints_data.get("forbidden", [])
        if isinstance(name, str) and name
    ]

    def expect_poi(expected_key: str, source: List[str], label: str) -> None:
        expected = query_spec.get(expected_key)
        if expected and expected not in source:
            errors.append(f"{label} mismatch: expected '{expected}'")

    if ctype == "duration_limit":
        expect_poi("anchor_poi", must_include_names, "duration_limit anchor_poi")
        actual = _parse_duration_text_to_minutes(_extract_first_threshold(constraints_data, ["duration_limit", "duration", "max_duration"]))
        expected = query_spec.get("duration_limit_min")
        if expected is not None and actual != expected:
            errors.append(f"duration_limit mismatch: expected {expected}min, got {actual}")
    elif ctype == "time_limit":
        day = _coerce_day_value(next((item.get("day") for item in temporal if item.get("day") is not None), None))
        expected_day = query_spec.get("day")
        if expected_day is not None and day != expected_day:
            errors.append(f"time_limit day mismatch: expected {expected_day}, got {day}")
        actual = _extract_first_threshold(constraints_data, ["end_before", "must_end_before", "time_limit"])
        expected = query_spec.get("end_before")
        if expected is not None and str(actual) != str(expected):
            errors.append(f"time_limit threshold mismatch: expected {expected}, got {actual}")
    elif ctype == "poi_local_radius_limit":
        expect_poi("anchor_poi", must_include_names, "poi_local_radius_limit anchor_poi")
        actual = _parse_numeric(_extract_first_threshold(constraints_data, ["max_walking_distance", "walk_dist", "max_distance"]))
        expected = query_spec.get("distance_km")
        if expected is not None and (actual is None or abs(float(actual) - float(expected)) > 1e-6):
            errors.append(f"poi_local_radius_limit threshold mismatch: expected {expected}, got {actual}")
    elif ctype in {"max_distance_constraint", "walking_distance_limit", "spatial_overflow"}:
        expected = query_spec.get("distance_km")
        actual = _parse_numeric(_extract_first_threshold(constraints_data, ["max_walking_distance", "walk_dist", "max_distance"]))
        if expected is not None and (actual is None or abs(float(actual) - float(expected)) > 1e-6):
            errors.append(f"{ctype} threshold mismatch: expected {expected}, got {actual}")
        for poi_name in query_spec.get("ordered_pois", []):
            if poi_name not in must_include_names:
                errors.append(f"{ctype} missing ordered poi '{poi_name}' in must_include")
    elif ctype in {"budget_limit", "daily_budget_infeasible", "budget_cap_preference", "budget_target_update", "resource_overlap"}:
        expected = query_spec.get("budget_cap")
        actual = _parse_numeric(_extract_first_threshold(constraints_data, ["budget_cap", "budget_target", "budget", "max_budget"]))
        if expected is not None and (actual is None or int(actual) != int(expected)):
            errors.append(f"{ctype} budget mismatch: expected {expected}, got {actual}")
        expect_poi("anchor_poi", must_include_names, f"{ctype} anchor_poi")
        if ctype == "resource_overlap":
            restaurant_name = query_spec.get("restaurant_name")
            if restaurant_name and restaurant_name not in must_include_names:
                errors.append(f"resource_overlap missing explicit restaurant '{restaurant_name}' in must_include")
    elif ctype in {"travel_time_constraint", "access_time_infeasible"}:
        expect_poi("anchor_poi", must_include_names, f"{ctype} anchor_poi")
        expected = query_spec.get("duration_limit_min")
        actual = _parse_duration_text_to_minutes(
            _extract_first_threshold(constraints_data, ["max_duration", "duration_limit", "duration"])
        )
        if expected is not None and actual != int(expected):
            errors.append(f"{ctype} duration mismatch: expected {expected}, got {actual}")
    elif ctype == "ticket_budget_limit":
        expected = query_spec.get("ticket_budget_cap")
        actual = _parse_numeric(_extract_first_threshold(constraints_data, ["ticket_budget_limit", "ticket_budget", "budget"]))
        if expected is not None and (actual is None or int(actual) != int(expected)):
            errors.append(f"ticket_budget_limit mismatch: expected {expected}, got {actual}")
    elif ctype == "ticket_price_cap_or_substitute":
        expected = query_spec.get("ticket_price_cap")
        actual = _parse_numeric(_extract_first_threshold(constraints_data, ["ticket_price_cap", "ticket_budget_limit", "ticket_budget", "budget"]))
        if expected is not None and (actual is None or int(actual) != int(expected)):
            errors.append(f"ticket_price_cap mismatch: expected {expected}, got {actual}")
        source_poi = query_spec.get("source_poi")
        actual_source = semantic.get("replace_from")
        if source_poi and source_poi != actual_source:
            errors.append(f"ticket_price_cap source_poi mismatch: expected '{source_poi}', got '{actual_source}'")
    elif ctype == "semantic_substitution":
        source_poi = query_spec.get("source_poi")
        if source_poi and source_poi not in forbidden:
            errors.append(f"semantic_substitution missing forbidden source '{source_poi}'")
        if source_poi and semantic.get("replace_from") != source_poi:
            errors.append(f"semantic_substitution replace_from mismatch: expected '{source_poi}', got '{semantic.get('replace_from')}'")
        semantic_target = query_spec.get("semantic_target")
        if semantic_target and semantic.get("semantic_target") != semantic_target:
            errors.append(f"semantic_substitution semantic_target mismatch: expected '{semantic_target}', got '{semantic.get('semantic_target')}'")
    elif ctype in {"semantic_and_specific", "category_overlap"}:
        expect_poi("specific_poi", must_include_names, f"{ctype} specific_poi")
        preferred_type = query_spec.get("preferred_type")
        if preferred_type and semantic.get("preferred_type") != preferred_type:
            errors.append(f"{ctype} preferred_type mismatch: expected '{preferred_type}', got '{semantic.get('preferred_type')}'")
        if ctype == "category_overlap" and query_spec.get("additional_similar") is True:
            if semantic.get("additional_similar") is not True:
                errors.append("category_overlap missing semantic.additional_similar=true")
    elif ctype in {"remove_poi", "substitute_poi", "famous_substitution", "type_replacement"}:
        source_poi = query_spec.get("source_poi")
        if source_poi and source_poi not in forbidden:
            errors.append(f"{ctype} missing forbidden source '{source_poi}'")
    elif ctype in {"type_filter", "preference_constraint"}:
        preferred_type = query_spec.get("preferred_type")
        if preferred_type and semantic.get("preferred_type") != preferred_type:
            errors.append(f"{ctype} preferred_type mismatch: expected '{preferred_type}', got '{semantic.get('preferred_type')}'")
    elif ctype == "required_restaurant_name":
        expect_poi("anchor_poi", must_include_names, "required_restaurant_name anchor_poi")
    elif ctype == "required_restaurant_type":
        preferred_type = query_spec.get("restaurant_type") or query_spec.get("preferred_type")
        if preferred_type and semantic.get("preferred_type") != preferred_type:
            errors.append(f"required_restaurant_type mismatch: expected '{preferred_type}', got '{semantic.get('preferred_type')}'")
    elif ctype == "required_hotel_name":
        expect_poi("anchor_poi", must_include_names, "required_hotel_name anchor_poi")
    elif ctype == "required_hotel_feature":
        hotel_feature = query_spec.get("hotel_feature")
        if hotel_feature and semantic.get("hotel_feature") != hotel_feature:
            errors.append(f"required_hotel_feature mismatch: expected '{hotel_feature}', got '{semantic.get('hotel_feature')}'")
    elif ctype == "required_room_type":
        room_type = query_spec.get("room_type")
        actual_room_type = _parse_numeric(_extract_first_threshold(constraints_data, ["room_type"]))
        if room_type is not None and (actual_room_type is None or int(actual_room_type) != int(room_type)):
            errors.append(f"required_room_type mismatch: expected {room_type}, got {actual_room_type}")
    elif ctype == "required_room_count":
        room_count = query_spec.get("room_count")
        actual_room_count = _parse_numeric(_extract_first_threshold(constraints_data, ["room_count", "rooms"]))
        if room_count is not None and (actual_room_count is None or int(actual_room_count) != int(room_count)):
            errors.append(f"required_room_count mismatch: expected {room_count}, got {actual_room_count}")
    elif ctype == "required_intercity_transport_type":
        expected_modes = [
            mode
            for mode in (query_spec.get("intercity_transport_modes") or [])
            if normalize_intercity_transport_mode(mode)
        ]
        transport = constraints_data.get("transport", {}) if isinstance(constraints_data.get("transport", {}), dict) else {}
        semantic = constraints_data.get("semantic", {}) if isinstance(constraints_data.get("semantic", {}), dict) else {}
        actual_modes = [
            normalize_intercity_transport_mode(mode)
            for mode in (transport.get("intercity_modes") or semantic.get("intercity_transport_modes") or [])
        ]
        actual_modes = [mode for mode in actual_modes if mode]
        for expected in expected_modes:
            normalized = normalize_intercity_transport_mode(expected)
            if normalized and normalized not in actual_modes:
                errors.append(f"required_intercity_transport_type missing mode '{normalized}'")
    elif ctype == "transport_time_window":
        expected_windows = query_spec.get("transport_time_windows") or []
        transport = constraints_data.get("transport", {}) if isinstance(constraints_data.get("transport", {}), dict) else {}
        temporal = constraints_data.get("temporal", []) if isinstance(constraints_data.get("temporal", []), list) else []
        actual_windows = transport.get("time_windows") if isinstance(transport.get("time_windows"), list) else []
        actual_windows = actual_windows or [
            item.get("transport_time_window")
            for item in temporal
            if isinstance(item, dict) and isinstance(item.get("transport_time_window"), dict)
        ]
        actual_keys = {
            (
                str(item.get("leg") or ""),
                str(item.get("field") or ""),
                str(item.get("operator") or ""),
                str(item.get("value") or ""),
            )
            for item in actual_windows
            if isinstance(item, dict)
        }
        for item in expected_windows:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("leg") or ""),
                str(item.get("field") or ""),
                str(item.get("operator") or ""),
                str(item.get("value") or ""),
            )
            if key not in actual_keys:
                errors.append(f"transport_time_window missing window {key}")
    elif ctype == "activity_budget_limit":
        activity_budget = query_spec.get("budget_cap")
        actual_budget = _parse_numeric(_extract_first_threshold(constraints_data, ["budget_cap", "budget", "max_budget"]))
        if activity_budget is not None and (actual_budget is None or int(actual_budget) != int(activity_budget)):
            errors.append(f"activity_budget_limit mismatch: expected {activity_budget}, got {actual_budget}")
        activity_type = query_spec.get("activity_type")
        if activity_type:
            resource = constraints_data.get("resource", {}) if isinstance(constraints_data.get("resource", {}), dict) else {}
            actual_activity_type = resource.get("activity_type")
            if actual_activity_type != activity_type:
                errors.append(f"activity_budget_limit activity_type mismatch: expected '{activity_type}', got '{actual_activity_type}'")
    elif ctype == "time_window":
        expect_poi("anchor_poi", must_include_names, "time_window anchor_poi")
        time_period = query_spec.get("time_period")
        actual_period = next((item.get("time_period") for item in temporal if item.get("time_period")), None)
        if time_period and actual_period != time_period:
            errors.append(f"time_window time_period mismatch: expected '{time_period}', got '{actual_period}'")
    elif ctype in {"order_constraint", "same_day_no_overlap_compact", "same_time_period_no_overlap"}:
        for poi_name in query_spec.get("ordered_pois", []):
            if poi_name not in must_include_names:
                errors.append(f"{ctype} missing ordered poi '{poi_name}' in must_include")
    elif ctype == "multi_day_requirement":
        expected_day_count = query_spec.get("day_count")
        actual_day_count = next((item.get("day_count") for item in temporal if item.get("day_count") is not None), None)
        if expected_day_count is not None and actual_day_count != expected_day_count:
            errors.append(f"multi_day_requirement day_count mismatch: expected {expected_day_count}, got {actual_day_count}")
    elif ctype in {"add_days", "multi_day_budget_overflow", "extend_with_pacing"}:
        expected_day_count = query_spec.get("day_count")
        actual_day_count = next((item.get("day_count") for item in temporal if item.get("day_count") is not None), None)
        if expected_day_count is not None and actual_day_count != expected_day_count:
            errors.append(f"{ctype} day_count mismatch: expected {expected_day_count}, got {actual_day_count}")
        if ctype == "add_days" and query_spec.get("day_delta") is None:
            errors.append("add_days missing day_delta in canonical query_spec")
        expected_cap = query_spec.get("daily_poi_cap")
        actual_cap = next((item.get("daily_poi_cap") for item in temporal if item.get("daily_poi_cap") is not None), None)
        if expected_cap is not None and actual_cap != expected_cap:
            errors.append(f"{ctype} daily_poi_cap mismatch: expected {expected_cap}, got {actual_cap}")
    elif ctype == "cross_day_restructure":
        actual_pairs = set()
        for item in temporal:
            day = _coerce_day_value(item.get("day"))
            must_visit = item.get("must_visit")
            if day is None:
                continue
            if isinstance(must_visit, str) and must_visit.strip():
                actual_pairs.add((day, must_visit.strip()))
            elif isinstance(must_visit, list):
                for value in must_visit:
                    if isinstance(value, str) and value.strip():
                        actual_pairs.add((day, value.strip()))
        for pair in query_spec.get("day_poi_pairs", []):
            if tuple(pair) not in actual_pairs:
                errors.append(f"cross_day_restructure missing day-poi binding: {tuple(pair)}")
    elif ctype == "multi_city_split":
        expected_targets = list(query_spec.get("city_split_targets", []) or [])
        actual_targets = list(semantic.get("city_split_targets", []) or [])
        if expected_targets and actual_targets[: len(expected_targets)] != expected_targets:
            errors.append(f"multi_city_split targets mismatch: expected {expected_targets}, got {actual_targets}")

    return errors


def _match_evidence_poi_name(
    poi_map: Dict[str, Dict[str, Any]],
    poi_name: Optional[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(poi_name, str) or not poi_name.strip():
        return None

    normalized = poi_name.strip()
    exact = poi_map.get(normalized)
    if exact is not None:
        return normalized, exact

    fuzzy_matches = [
        (name, info)
        for name, info in poi_map.items()
        if normalized in name or name in normalized
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    return None


def normalize_constraints_by_type(
    constraints_data: Dict[str, Any],
    constraint_type: str,
    template_str: str,
    edit_query: str,
    target_city: str,
    semantic_slots: Optional[Dict[str, Any]] = None,
    origin_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    对 LLM 产出的 constraints 做归一化，消除隐式注入并补足关键结构。
    """
    normalized = dict(constraints_data) if isinstance(constraints_data, dict) else {}
    slots = semantic_slots if isinstance(semantic_slots, dict) else parse_query_semantic_slots(edit_query, constraint_type, template_str=template_str)

    raw_must_include = normalized.get("must_include", [])
    must_include: List[Dict[str, Any]] = []
    seen_names = set()
    if isinstance(raw_must_include, list):
        for poi in raw_must_include:
            if not isinstance(poi, dict):
                continue
            name = str(poi.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            must_include.append({
                "name": name,
                "type": poi.get("type", "attraction"),
                "city": poi.get("city", target_city),
            })

    spatial = normalized.get("spatial", {})
    if not isinstance(spatial, dict):
        spatial = {}
    spatial["city"] = target_city

    temporal = _normalize_temporal_list(normalized.get("temporal", []))
    for item in temporal:
        day = _coerce_day_value(item.get("day"))
        if day is not None:
            item["day"] = day

    resource = normalized.get("resource", {})
    if not isinstance(resource, dict):
        resource = {}
    semantic = normalized.get("semantic", {})
    if not isinstance(semantic, dict):
        semantic = {}

    forbidden = normalized.get("forbidden", [])
    if not isinstance(forbidden, list):
        forbidden = []
    forbidden = [str(name).strip() for name in forbidden if isinstance(name, str) and str(name).strip()]

    # 模板未显式引用 POI 时，不应生成 must_include
    if not template_uses_poi_placeholders(template_str):
        must_include = []

    if constraint_type == "poi_local_radius_limit":
        anchor_poi = None
        if must_include and isinstance(must_include[0], dict):
            anchor_poi = must_include[0].get("name")
        if not isinstance(anchor_poi, str) or not anchor_poi.strip():
            anchor_poi = _extract_radius_anchor_poi_from_query(edit_query)

        if isinstance(anchor_poi, str) and anchor_poi.strip():
            anchor_name = anchor_poi.strip()
            if anchor_name not in seen_names:
                seen_names.add(anchor_name)
                must_include.append({"name": anchor_name, "type": "attraction", "city": target_city})

        threshold_raw = _extract_first_threshold(
            normalized,
            ["max_walking_distance", "walk_dist", "max_distance"],
        )
        if threshold_raw is None:
            query_match = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query)
            if query_match:
                threshold_raw = query_match.group(1)
        threshold = _parse_numeric(threshold_raw)
        if threshold is not None and threshold > 0:
            spatial["max_walking_distance"] = float(threshold)

        # 半径约束是空间结构约束，不应注入 temporal/resource
        temporal = []
        resource = {}

    if constraint_type in {"max_distance_constraint", "walking_distance_limit", "spatial_overflow"}:
        threshold_raw = _extract_first_threshold(
            normalized,
            ["max_walking_distance", "walk_dist", "max_distance"],
        )
        if threshold_raw is None:
            query_match = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query)
            if query_match:
                threshold_raw = query_match.group(1)
        threshold = _parse_numeric(threshold_raw)
        if threshold is not None and threshold > 0:
            spatial["max_walking_distance"] = float(threshold)

    if constraint_type == "duration_limit":
        poi_name = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        if poi_name and poi_name not in seen_names:
            seen_names.add(poi_name)
            must_include.append({"name": poi_name, "type": "attraction", "city": target_city})

        duration_raw = _extract_first_threshold(
            normalized,
            ["duration_limit", "duration", "max_duration"],
        )
        if duration_raw is None:
            duration_raw = _extract_query_duration_text(edit_query)
        if duration_raw is None:
            duration_minutes = slots.get("duration_limit_min") or _extract_query_duration_minutes(edit_query)
            if duration_minutes is not None:
                duration_raw = f"{int(duration_minutes)}分钟"
        if duration_raw:
            temporal = [{"duration_limit": duration_raw}]

    if constraint_type == "time_limit":
        query_day = _coerce_day_value(next(iter(_extract_query_days(edit_query)), None))
        end_before = _extract_query_end_before(edit_query)
        if end_before:
            temporal = [{
                **({"day": query_day} if query_day is not None else {}),
                "end_before": end_before,
            }]

    if constraint_type == "time_window":
        poi_name = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        time_period = _extract_time_period_from_query(edit_query)
        query_day = _coerce_day_value(next(iter(_extract_query_days(edit_query)), None))
        if poi_name and poi_name not in seen_names:
            seen_names.add(poi_name)
            must_include.append({"name": poi_name, "type": "attraction", "city": target_city})
        if poi_name and time_period:
            day = query_day
            if day is None:
                day = _coerce_day_value(
                    next(
                        (
                            item.get("day")
                            for item in temporal
                            if isinstance(item, dict) and item.get("day") is not None
                        ),
                        None,
                    )
                )
            temporal = [{
                "must_visit": poi_name,
                "time_period": time_period,
                **({"day": day} if day is not None else {}),
            }]

    if constraint_type in {"order_constraint", "same_day_no_overlap_compact", "same_time_period_no_overlap"}:
        poi_names = _extract_ordered_pois_from_query(edit_query)
        time_period = _extract_time_period_from_query(edit_query)
        query_day = _coerce_day_value(next(iter(_extract_query_days(edit_query)), None))
        day = query_day
        if day is None and constraint_type != "order_constraint":
            day = _coerce_day_value(
                next(
                    (
                        item.get("day")
                        for item in temporal
                        if isinstance(item, dict) and item.get("day") is not None
                    ),
                    None,
                )
            )
        for poi_name in poi_names:
            if poi_name not in seen_names:
                seen_names.add(poi_name)
                must_include.append({"name": poi_name, "type": "attraction", "city": target_city})
        if poi_names:
            payload: Dict[str, Any] = {"must_visit": poi_names}
            if day is not None:
                payload["day"] = day
            if time_period:
                payload["time_period"] = time_period
            if constraint_type == "order_constraint":
                payload["sequence"] = poi_names
            temporal = [payload]

    if constraint_type in {"remove_poi", "substitute_poi", "famous_substitution", "type_replacement"}:
        source_poi = slots.get("source_poi") or _extract_replace_source_poi(edit_query) or _extract_poi_from_query_like(edit_query)
        if source_poi:
            if source_poi not in forbidden:
                forbidden.append(source_poi)
            semantic["replace_from"] = source_poi

    if constraint_type in {"type_filter", "preference_constraint", "famous_substitution"}:
        preferred_type = slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
        if preferred_type:
            semantic["preferred_type"] = preferred_type

    if constraint_type == "required_restaurant_name":
        restaurant_name = slots.get("restaurant_name") or _extract_poi_from_query_like(edit_query)
        if restaurant_name and restaurant_name not in seen_names:
            seen_names.add(restaurant_name)
            must_include.append({"name": restaurant_name, "type": "restaurant", "city": target_city})

    if constraint_type == "required_restaurant_type":
        preferred_type = slots.get("restaurant_type") or _extract_restaurant_type_from_query(edit_query)
        if preferred_type:
            semantic["preferred_type"] = preferred_type
            semantic["required_restaurant_type"] = preferred_type

    if constraint_type == "required_hotel_name":
        hotel_name = slots.get("hotel_name") or _extract_poi_from_query_like(edit_query)
        if hotel_name and hotel_name not in seen_names:
            seen_names.add(hotel_name)
            must_include.append({"name": hotel_name, "type": "accommodation", "city": target_city})

    if constraint_type == "required_hotel_feature":
        hotel_feature = slots.get("hotel_feature") or _extract_hotel_feature_from_query(edit_query)
        if hotel_feature:
            semantic["hotel_feature"] = hotel_feature

    if constraint_type == "required_room_type":
        room_type = slots.get("hotel_room_type") or _extract_room_type_from_query(edit_query)
        if room_type is not None:
            resource["room_type"] = int(room_type)

    if constraint_type == "required_room_count":
        room_count = slots.get("hotel_room_count") or _extract_room_count_from_query(edit_query)
        if room_count is not None:
            resource["room_count"] = int(room_count)

    if constraint_type == "required_intercity_transport_type":
        modes = slots.get("intercity_transport_modes") or _extract_intercity_transport_modes_from_query(edit_query)
        modes = [normalize_intercity_transport_mode(mode) for mode in modes]
        modes = [mode for mode in modes if mode]
        if modes:
            transport = normalized.get("transport", {})
            if not isinstance(transport, dict):
                transport = {}
            transport["intercity_modes"] = sorted(set(modes))
            normalized["transport"] = transport
            semantic["intercity_transport_modes"] = sorted(set(modes))

    if constraint_type == "transport_time_window":
        windows = slots.get("transport_time_windows") or _extract_transport_time_windows_from_query(edit_query)
        windows = [dict(item) for item in windows if isinstance(item, dict)]
        if windows:
            transport = normalized.get("transport", {})
            if not isinstance(transport, dict):
                transport = {}
            transport["time_windows"] = windows
            normalized["transport"] = transport
            temporal = [{"transport_time_window": item} for item in windows]

    if constraint_type == "activity_budget_limit":
        activity_budget = _extract_activity_budget_requirement(edit_query)
        if activity_budget:
            resource["budget_cap"] = int(activity_budget["budget_cap"])
            resource["activity_type"] = activity_budget["activity_type"]
            resource["metric"] = activity_budget["metric"]

    if constraint_type in {"budget_limit", "daily_budget_infeasible", "budget_cap_preference", "budget_target_update"}:
        threshold = _extract_first_threshold(normalized, ["budget_cap", "budget_target", "budget", "max_budget"])
        if threshold is None:
            query_match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
            if query_match:
                threshold = query_match.group(1)
        budget = _parse_numeric(threshold)
        if budget is not None:
            resource["budget_cap"] = int(budget)
            resource["budget"] = int(budget)

    if constraint_type == "ticket_budget_limit":
        threshold = _extract_first_threshold(normalized, ["ticket_budget_limit", "ticket_budget", "budget"])
        if threshold is None:
            query_match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
            if query_match:
                threshold = query_match.group(1)
        budget = _parse_numeric(threshold)
        if budget is not None:
            resource["ticket_budget_limit"] = int(budget)
            resource["ticket_budget"] = int(budget)

    if constraint_type == "resource_overlap":
        poi_name = (
            slots.get("anchor_poi")
            or _extract_poi_from_query_like(edit_query)
        )
        if poi_name and poi_name not in seen_names:
            seen_names.add(poi_name)
            must_include.append({"name": poi_name, "type": "attraction", "city": target_city})
        restaurant_name = slots.get("restaurant_name")
        if restaurant_name and restaurant_name not in seen_names:
            seen_names.add(restaurant_name)
            must_include.append({"name": restaurant_name, "type": "restaurant", "city": target_city})
        budget = _parse_numeric(_extract_first_threshold(normalized, ["budget", "max_budget"]))
        if budget is None:
            query_match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
            if query_match:
                budget = float(query_match.group(1))
        if budget is not None:
            resource["max_budget"] = int(budget)
        if slots.get("nearby_meal"):
            resource["nearby_meal"] = True
        if poi_name and (slots.get("nearby_meal") or restaurant_name):
            resource["overlap_anchor_poi"] = poi_name
            resource["budget_scope"] = "anchor_bundle"
        commute_distance = _extract_query_distance_value(edit_query)
        if commute_distance is not None:
            spatial["max_distance"] = float(commute_distance)

    if constraint_type in {"travel_time_constraint", "access_time_infeasible"}:
        poi_name = slots.get("anchor_poi") or _extract_poi_from_query_like(edit_query)
        if poi_name and poi_name not in seen_names:
            seen_names.add(poi_name)
            must_include.append({"name": poi_name, "type": "attraction", "city": target_city})
        duration_minutes = slots.get("duration_limit_min") or _extract_query_duration_minutes(edit_query)
        if duration_minutes is not None:
            temporal = [{"max_duration": f"{int(duration_minutes)}分钟"}]

    if constraint_type == "walking_distance_limit":
        poi_names = _extract_ordered_pois_from_query(edit_query)
        for poi_name in poi_names:
            if poi_name not in seen_names:
                seen_names.add(poi_name)
                must_include.append({"name": poi_name, "type": "attraction", "city": target_city})

    if constraint_type == "multi_day_requirement":
        requested_days = _extract_requested_day_count_from_query(edit_query)
        if requested_days is not None:
            temporal = [{"day_count": requested_days}]

    if constraint_type == "add_days":
        requested_delta = slots.get("day_delta")
        if requested_delta is None:
            requested_delta = _extract_requested_day_delta_from_query(edit_query)
        resulting_days = _resolve_resulting_day_count(
            constraint_type=constraint_type,
            origin_plan=origin_plan,
            requested_day_count=slots.get("day_count") or _extract_requested_day_count_from_query(edit_query),
            requested_day_delta=requested_delta,
        )
        if resulting_days is not None:
            temporal = [{"day_count": resulting_days}]
            cap = _extract_daily_poi_cap_from_query(edit_query)
            if cap is not None:
                temporal[-1]["daily_poi_cap"] = cap
        preferred_type = slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
        if preferred_type:
            semantic["preferred_type"] = preferred_type
    if constraint_type in {"multi_day_budget_overflow", "extend_with_pacing"}:
        requested_days = _extract_requested_day_count_from_query(edit_query)
        if requested_days is not None:
            temporal = [{"day_count": requested_days}]
            cap = _extract_daily_poi_cap_from_query(edit_query)
            if cap is not None:
                temporal[-1]["daily_poi_cap"] = cap

    if constraint_type == "semantic_substitution":
        # semantic_substitution 使用独立 schema，不使用 must_include
        must_include = []
        source_poi = slots.get("source_poi") or _extract_replace_source_poi(edit_query)
        semantic_target = slots.get("semantic_target") or _extract_semantic_target_from_query(edit_query)
        if source_poi:
            if source_poi not in forbidden:
                forbidden.append(source_poi)
            semantic["replace_from"] = source_poi
        if semantic_target:
            semantic["semantic_target"] = semantic_target

    if constraint_type == "ticket_price_cap_or_substitute":
        source_poi = _extract_ticket_price_cap_source_poi(normalized, edit_query)
        if source_poi:
            semantic["replace_from"] = source_poi

        threshold_raw = _extract_first_threshold(
            normalized,
            ["ticket_price_cap", "ticket_budget_limit", "ticket_budget", "budget"],
        )
        if threshold_raw is None:
            query_match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
            if query_match:
                threshold_raw = query_match.group(1)
        threshold = _parse_numeric(threshold_raw)
        if threshold is not None and threshold >= 0:
            resource["ticket_price_cap"] = int(threshold)

    if constraint_type in {"semantic_and_specific", "category_overlap"}:
        specific_poi = slots.get("specific_poi") or _extract_specific_poi_from_query(edit_query)
        preferred_type = slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
        if specific_poi and specific_poi not in seen_names:
            seen_names.add(specific_poi)
            must_include.append({"name": specific_poi, "type": "attraction", "city": target_city})
        if preferred_type:
            semantic["preferred_type"] = preferred_type
        if constraint_type == "category_overlap" and slots.get("additional_similar"):
            semantic["additional_similar"] = True

    if constraint_type == "cross_day_restructure":
        # 对跨日模板写入 day->must_visit，避免所有 POI 被默认绑到同一天
        day_pairs = list(slots.get("day_poi_pairs") or _extract_day_poi_pairs_from_query(edit_query))
        day_to_pois: Dict[int, List[str]] = {}

        for item in temporal:
            item_day = _coerce_day_value(item.get("day"))
            must_visit = item.get("must_visit")
            if item_day is None or must_visit is None:
                continue
            if isinstance(must_visit, str):
                names = [must_visit.strip()] if must_visit.strip() else []
            elif isinstance(must_visit, list):
                names = [str(v).strip() for v in must_visit if isinstance(v, str) and str(v).strip()]
            else:
                names = []
            if names:
                day_to_pois.setdefault(item_day, [])
                for name in names:
                    if name not in day_to_pois[item_day]:
                        day_to_pois[item_day].append(name)

        for day, poi_name in day_pairs:
            day_to_pois.setdefault(day, [])
            if poi_name not in day_to_pois[day]:
                day_to_pois[day].append(poi_name)
            if poi_name not in seen_names:
                seen_names.add(poi_name)
                must_include.append({"name": poi_name, "type": "attraction", "city": target_city})

        if day_to_pois:
            temporal_no_must_visit = [item for item in temporal if "must_visit" not in item]
            temporal = temporal_no_must_visit + [
                {"day": day, "must_visit": names}
                for day, names in sorted(day_to_pois.items(), key=lambda kv: kv[0])
                if names
            ]

    if constraint_type == "multi_city_split":
        split_targets = list(slots.get("city_split_targets") or [])
        if split_targets:
            semantic["city_split_targets"] = split_targets
            semantic["split_signal"] = True

    if constraint_type == "thematic_split":
        temporal = []

    normalized["must_include"] = must_include
    normalized["spatial"] = spatial
    normalized["temporal"] = temporal
    normalized["resource"] = resource
    if constraint_type in {
        "semantic_substitution",
        "ticket_price_cap_or_substitute",
        "semantic_and_specific",
        "category_overlap",
        "substitute_poi",
        "famous_substitution",
        "type_replacement",
        "type_filter",
        "preference_constraint",
        "required_restaurant_type",
        "required_hotel_feature",
        "required_intercity_transport_type",
        "resource_overlap",
        "multi_city_split",
        "add_days",
    }:
        normalized["semantic"] = semantic
    if forbidden:
        normalized["forbidden"] = forbidden
    elif "forbidden" in normalized:
        normalized["forbidden"] = []
    return normalized


def build_origin_plan_evidence(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 origin_plan 提取可用于参数反推和一致性校验的证据。
    """
    evidence = build_origin_plan_evidence_for_plan(sample.get("origin_plan", {}))
    if not isinstance(evidence, dict):
        evidence = {}
    else:
        evidence = dict(evidence)

    origin_logical_constraints = sample.get("origin_logical_constraints")
    if not isinstance(origin_logical_constraints, list):
        meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
        origin_logical_constraints = meta.get("origin_logical_constraints")

    if _origin_has_budget_requirement(origin_logical_constraints):
        evidence["has_origin_budget_requirement"] = True
    origin_budget_total = _extract_origin_budget_total(origin_logical_constraints)
    if origin_budget_total is not None:
        evidence["origin_budget_total"] = origin_budget_total
        evidence["has_origin_budget_requirement"] = True

    return evidence


def _choose_numeric_threshold_below_baseline(
    baseline: float,
    options: List[Any],
    min_floor: float,
    unit: str,
) -> Tuple[str, float]:
    parsed_options = []
    for item in options:
        num = _parse_numeric(item)
        if num is not None:
            parsed_options.append((item, num))

    candidates = [x for x in parsed_options if x[1] < baseline]
    if candidates:
        selected = max(candidates, key=lambda x: x[1])
        return str(selected[0]), float(selected[1])

    inferred = max(min_floor, baseline * 0.8)
    if unit == "km":
        inferred = round(inferred, 1)
        return str(inferred), float(inferred)
    if unit == "money":
        inferred = int(max(min_floor, inferred))
        return str(inferred), float(inferred)
    return str(inferred), float(inferred)


def _choose_explicit_secondary_city(target_city: str) -> Optional[str]:
    normalized_target = str(target_city or "").strip()
    if not normalized_target:
        return None
    for city in sorted(_SUPPORTED_DATASET_CITIES):
        if city != normalized_target:
            return city
    return None


def apply_conflict_driven_param_adjustments(
    sample: Dict[str, Any],
    template: Dict[str, Any],
    params: Dict[str, Any],
    pois: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    基于 origin_plan 证据调整参数，尽量保证 query 对应 bucket 可触发。
    """
    adjusted = dict(params)
    adjusted_pois = list(pois)
    trace: Dict[str, Any] = {"constraint_type": template.get("constraint_type", "general")}

    constraint_type = str(template.get("constraint_type", "general"))
    template_params = template.get("params", {}) if isinstance(template.get("params", {}), dict) else {}
    poi_map = evidence.get("poi_duration_by_name", {})

    if constraint_type in {"duration_limit", "poi_local_radius_limit"}:
        # duration_limit 必须锚定到 origin_plan 已有 POI，才可判定 overflow
        current_name = adjusted_pois[0]["name"] if adjusted_pois else None
        if current_name not in poi_map and poi_map:
            anchor_name, anchor_info = max(
                poi_map.items(), key=lambda kv: kv[1].get("duration_min", 0)
            )
            if adjusted_pois:
                adjusted_pois[0]["name"] = anchor_name
                adjusted_pois[0]["city"] = sample["meta"]["target_city"]
                adjusted_pois[0]["type"] = "attraction"
            else:
                adjusted_pois = [{
                    "name": anchor_name,
                    "type": "attraction",
                    "city": sample["meta"]["target_city"],
                }]
            trace["poi_anchor_overridden"] = True
            trace["anchor_poi"] = anchor_name
            trace["anchor_duration_min"] = anchor_info.get("duration_min")

        selected_name = adjusted_pois[0]["name"] if adjusted_pois else None
        selected_info = poi_map.get(selected_name) if selected_name else None
        if selected_info:
            trace["anchor_poi"] = selected_name
            trace["anchor_duration_min"] = selected_info.get("duration_min", 0)

            if constraint_type == "duration_limit":
                baseline = selected_info.get("duration_min", 0)
                options = template_params.get("duration", [])
                option_minutes = []
                for opt in options if isinstance(options, list) else []:
                    parsed = _parse_duration_text_to_minutes(opt)
                    if parsed is not None:
                        option_minutes.append((str(opt), parsed))
                candidates = [x for x in option_minutes if x[1] < baseline]
                if candidates:
                    chosen_text, chosen_min = max(candidates, key=lambda x: x[1])
                else:
                    chosen_min = max(30, int(baseline) - 30)
                    chosen_text = _format_minutes_to_duration_text(chosen_min)
                adjusted["duration"] = chosen_text
                trace["chosen_duration_limit_min"] = chosen_min
                trace["chosen_duration_limit_text"] = chosen_text

            if constraint_type == "poi_local_radius_limit":
                baseline = float(evidence.get("max_transport_distance_km", 0.0))
                if baseline > 0:
                    options = template_params.get("walk_dist", [])
                    chosen_text, chosen_num = _choose_numeric_threshold_below_baseline(
                        baseline=baseline,
                        options=options if isinstance(options, list) else [],
                        min_floor=0.5,
                        unit="km",
                    )
                    adjusted["walk_dist"] = chosen_text
                    trace["baseline_max_distance_km"] = baseline
                    trace["chosen_walk_dist_km"] = chosen_num

    elif constraint_type == "time_limit":
        day_end = evidence.get("day_end_by_day", {})
        if day_end:
            attraction_days = {
                int(day)
                for day, count in evidence.get("day_attraction_count", {}).items()
                if int(count or 0) > 0 and int(day) in day_end
            }
            eligible_day_end = {day: end for day, end in day_end.items() if day in attraction_days} or day_end
            if _cn_to_day(adjusted.get("day")) not in eligible_day_end:
                day = max(eligible_day_end.items(), key=lambda x: x[1])[0]
                adjusted["day"] = _day_to_cn(day)
            day = _cn_to_day(adjusted.get("day"))
            if day in eligible_day_end:
                baseline_end = eligible_day_end[day]
                time_options = template_params.get("time", [])
                parsed_options = []
                for opt in time_options if isinstance(time_options, list) else []:
                    m = _parse_time_to_minutes(opt)
                    if m is not None:
                        parsed_options.append((str(opt), m))
                candidates = [x for x in parsed_options if x[1] < baseline_end]
                if candidates:
                    chosen_text, chosen_min = max(candidates, key=lambda x: x[1])
                else:
                    chosen_min = max(8 * 60, baseline_end - 60)
                    chosen_text = _format_minutes_to_clock(chosen_min)
                adjusted["time"] = chosen_text
                trace["anchor_day"] = day
                trace["anchor_day_end_min"] = baseline_end
                trace["chosen_end_before_min"] = chosen_min

    elif constraint_type in {"max_distance_constraint", "walking_distance_limit", "spatial_overflow"}:
        baseline = float(evidence.get("max_transport_distance_km", 0.0))
        if baseline > 0:
            options = template_params.get("walk_dist", [])
            chosen_text, chosen_num = _choose_numeric_threshold_below_baseline(
                baseline=baseline,
                options=options if isinstance(options, list) else [],
                min_floor=0.5,
                unit="km",
            )
            adjusted["walk_dist"] = chosen_text
            trace["baseline_max_distance_km"] = baseline
            trace["chosen_walk_dist_km"] = chosen_num

    elif constraint_type in {
        "budget_limit",
        "daily_budget_infeasible",
    }:
        if constraint_type == "budget_limit" and not bool(evidence.get("budget_baseline_reliable_for_total_budget_edit", True)):
            trace["budget_guard_reason"] = evidence.get("multi_day_budget_guard_reason") or "unreliable_total_budget_baseline"
            return adjusted, adjusted_pois, trace
        baseline = float(evidence.get("total_cost", 0.0))
        if baseline > 0:
            _, chosen_budget = _choose_budget_threshold_for_constraint(
                constraint_type,
                baseline_total_cost=baseline,
                options=template_params.get("budget", []),
                origin_budget_total=evidence.get("origin_budget_total") or _extract_origin_budget_hint(sample),
            )
            adjusted["budget"] = str(int(chosen_budget))
            trace["baseline_total_cost"] = baseline
            trace["chosen_budget"] = chosen_budget

    elif constraint_type == "resource_overlap":
        anchor_poi = adjusted_pois[0]["name"] if adjusted_pois else None
        baseline = _estimate_resource_overlap_budget_baseline(evidence, anchor_poi=anchor_poi)
        if baseline is not None:
            chosen_budget = _choose_adaptive_budget_cap(
                baseline,
                min_floor=50,
                rounding_step=20,
            )
            if chosen_budget is not None:
                adjusted["budget"] = str(int(chosen_budget))
                trace["baseline_anchor_bundle_cost"] = baseline
                trace["chosen_budget"] = chosen_budget
                if anchor_poi:
                    trace["anchor_poi"] = anchor_poi

    elif constraint_type == "activity_budget_limit":
        origin_plan = sample.get("origin_plan", {}) if isinstance(sample.get("origin_plan"), dict) else {}
        people = max(int(origin_plan.get("people_number", 1) or 1), 1)
        template_text = str(template.get("template", ""))
        if "餐" in template_text or "饭" in template_text:
            count = int(evidence.get("meal_count", 0) or 0)
            baseline = float(evidence.get("meal_cost", 0.0) or 0.0) / max(count, 1) / people if count > 0 else 0.0
            activity_type = "meal"
            metric = "avg_cost_per_meal_per_person"
        else:
            nights = max(int(evidence.get("day_count", 1) or 1) - 1, 1)
            count = int(evidence.get("accommodation_count", 0) or 0)
            baseline = float(evidence.get("accommodation_cost", 0.0) or 0.0) / people / nights if count > 0 else 0.0
            activity_type = "accommodation"
            metric = "avg_cost_per_person_per_night"
        if baseline > 0:
            options = template_params.get("budget", [])
            chosen_text, chosen_num = _choose_numeric_threshold_below_baseline(
                baseline=baseline,
                options=options if isinstance(options, list) else [],
                min_floor=20.0,
                unit="money",
            )
            adjusted["budget"] = str(int(chosen_num))
            trace["activity_type"] = activity_type
            trace["budget_metric"] = metric
            trace["baseline_activity_budget"] = baseline
            trace["chosen_budget"] = chosen_num

    elif constraint_type == "required_intercity_transport_type":
        existing_modes = set(str(mode) for mode in evidence.get("intercity_transport_modes", []) if str(mode))
        chosen_mode = ""
        if "train" in existing_modes and "airplane" not in existing_modes:
            chosen_mode = "飞机"
        elif "airplane" in existing_modes and "train" not in existing_modes:
            chosen_mode = "高铁"
        elif not existing_modes:
            chosen_mode = str(adjusted.get("transport_mode") or "")
        if chosen_mode:
            adjusted["transport_mode"] = chosen_mode
            trace["existing_intercity_transport_modes"] = sorted(existing_modes)
            trace["chosen_transport_mode"] = chosen_mode

    elif constraint_type == "transport_time_window":
        time_by_leg = evidence.get("intercity_time_by_leg", {})
        if isinstance(time_by_leg, dict) and time_by_leg:
            leg = "return" if "return" in time_by_leg else next(iter(time_by_leg.keys()))
            leg_info = time_by_leg.get(leg, {}) if isinstance(time_by_leg.get(leg, {}), dict) else {}
            field = "end_time" if leg == "return" else "start_time"
            action = "到达" if field == "end_time" else "出发"
            actual_min = _parse_time_to_minutes(leg_info.get(field))
            if actual_min is not None:
                chosen_min = max(0, actual_min - 30)
                adjusted["leg"] = "返程" if leg == "return" else "去程"
                adjusted["action"] = action
                adjusted["time"] = _format_minutes_to_clock(chosen_min)
                trace["transport_leg"] = leg
                trace["transport_time_field"] = field
                trace["baseline_transport_time_min"] = actual_min
                trace["chosen_transport_time_bound"] = adjusted["time"]

    elif constraint_type == "multi_day_budget_overflow":
        requested_day_count = None
        parsed_requested_day_count = _parse_numeric(adjusted.get("day_count"))
        if parsed_requested_day_count is not None:
            requested_day_count = int(parsed_requested_day_count)
        baseline = _project_multi_day_budget_baseline(
            evidence,
            requested_day_count=requested_day_count,
        )
        if baseline is not None and baseline > 0:
            chosen_budget = _choose_adaptive_budget_cap(
                baseline,
                min_floor=100,
                rounding_step=100,
            )
            if chosen_budget is not None:
                adjusted["budget"] = str(int(chosen_budget))
                trace["baseline_projected_multi_day_local_cost"] = baseline
                if evidence.get("multi_day_budget_guard_reason"):
                    trace["multi_day_budget_guard_reason"] = evidence.get("multi_day_budget_guard_reason")
                if requested_day_count is not None:
                    trace["requested_day_count"] = requested_day_count
                trace["chosen_budget"] = chosen_budget

    elif constraint_type in {"budget_cap_preference", "budget_target_update"}:
        baseline = float(evidence.get("total_cost", 0.0))
        origin_budget = evidence.get("origin_budget_total") or _extract_origin_budget_hint(sample)
        if origin_budget is not None:
            _, chosen_budget = _choose_budget_threshold_for_constraint(
                constraint_type,
                baseline_total_cost=baseline,
                options=template_params.get("budget", []),
                origin_budget_total=origin_budget,
            )
        else:
            chosen_budget = _choose_adaptive_budget_relaxation(
                origin_budget=origin_budget,
                baseline_total=baseline,
            )
        if chosen_budget is not None:
            adjusted["budget"] = str(int(chosen_budget))
            trace["baseline_total_cost"] = baseline
            if origin_budget is not None:
                trace["origin_budget_hint"] = origin_budget
            trace["chosen_budget"] = chosen_budget

    elif constraint_type == "extend_with_pacing":
        current_day_count = int(evidence.get("day_count", 0) or 0)
        max_day_cap = max((int(v) for v in evidence.get("day_attraction_count", {}).values()), default=0)
        requires_day_extension = template.get("requires_day_extension")
        chosen_day_count = _choose_stricter_int_option(
            template_params.get("day_count", []),
            baseline=current_day_count,
            direction="gt",
        )
        chosen_poi_cap = _choose_stricter_int_option(
            template_params.get("poi_count", []),
            baseline=max_day_cap,
            direction="lt",
        )
        if requires_day_extension is True:
            if chosen_day_count is not None:
                adjusted["day_count"] = chosen_day_count
                trace["current_day_count"] = current_day_count
                trace["chosen_day_count"] = chosen_day_count
        elif "day_count" in adjusted and "{day_count}" not in str(template.get("template", "")):
            adjusted.pop("day_count", None)
        if chosen_poi_cap is not None:
            adjusted["poi_count"] = chosen_poi_cap
            trace["current_max_daily_attractions"] = max_day_cap
            trace["chosen_daily_poi_cap"] = chosen_poi_cap

    elif constraint_type in {"ticket_budget_limit"}:
        baseline = float(evidence.get("ticket_cost", 0.0))
        if baseline > 0:
            chosen_budget = _choose_adaptive_budget_cap(
                baseline,
                min_floor=50,
                rounding_step=20,
            )
            if chosen_budget is not None:
                adjusted["budget"] = str(int(chosen_budget))
                trace["baseline_ticket_cost"] = baseline
                trace["chosen_ticket_budget"] = chosen_budget

    elif constraint_type == "travel_time_constraint":
        baseline = int(evidence.get("max_transport_duration_min", 0) or 0)
        chosen_duration = _choose_adaptive_duration_limit_minutes(baseline)
        if chosen_duration is not None:
            adjusted["duration"] = _format_minutes_to_duration_text(chosen_duration)
            trace["baseline_max_transport_duration_min"] = baseline
            trace["chosen_transport_duration_limit_min"] = chosen_duration
            trace["chosen_transport_duration_min"] = chosen_duration

    elif constraint_type == "multi_city_split":
        secondary_city = str(adjusted.get("nearby_city") or "").strip()
        if (
            not secondary_city
            or secondary_city == "附近的另一个城市"
            or secondary_city == sample.get("meta", {}).get("target_city")
        ):
            chosen_city = _choose_explicit_secondary_city(sample.get("meta", {}).get("target_city", ""))
            if chosen_city:
                adjusted["nearby_city"] = chosen_city
                trace["secondary_city"] = chosen_city

    elif constraint_type == "ticket_price_cap_or_substitute":
        price_map = evidence.get("poi_ticket_price_by_name", {})
        current_name = adjusted_pois[0]["name"] if adjusted_pois else None
        current_info = price_map.get(current_name) if isinstance(current_name, str) else None

        if not current_info or float(current_info.get("ticket_price", 0.0)) <= 0:
            priced_candidates = [
                (name, info)
                for name, info in price_map.items()
                if float(info.get("ticket_price", 0.0)) > 0
            ]
            if priced_candidates:
                anchor_name, anchor_info = max(
                    priced_candidates,
                    key=lambda kv: float(kv[1].get("ticket_price", 0.0)),
                )
                if adjusted_pois:
                    adjusted_pois[0]["name"] = anchor_name
                    adjusted_pois[0]["city"] = sample["meta"]["target_city"]
                    adjusted_pois[0]["type"] = "attraction"
                else:
                    adjusted_pois = [{
                        "name": anchor_name,
                        "type": "attraction",
                        "city": sample["meta"]["target_city"],
                    }]
                trace["poi_anchor_overridden"] = True
                current_name = anchor_name
                current_info = anchor_info

        if current_info and float(current_info.get("ticket_price", 0.0)) > 0:
            baseline = float(current_info["ticket_price"])
            options = template_params.get("budget", [])
            chosen_text, chosen_num = _choose_numeric_threshold_below_baseline(
                baseline=baseline,
                options=options if isinstance(options, list) else [],
                min_floor=1.0,
                unit="money",
            )
            adjusted["budget"] = str(int(chosen_num))
            trace["anchor_poi"] = current_name
            trace["baseline_ticket_price"] = baseline
            trace["chosen_ticket_price_cap"] = chosen_num

    return adjusted, adjusted_pois, trace


def format_evidence_for_prompt(constraint_type: str, evidence_trace: Dict[str, Any]) -> str:
    lines = [f"- constraint_type: {constraint_type}"]
    if "anchor_poi" in evidence_trace:
        lines.append(f"- anchor_poi: {evidence_trace['anchor_poi']}")
    if "anchor_duration_min" in evidence_trace:
        lines.append(f"- anchor_poi_duration_min: {evidence_trace['anchor_duration_min']}")
    if "chosen_duration_limit_text" in evidence_trace:
        lines.append(f"- chosen_duration_limit: {evidence_trace['chosen_duration_limit_text']}")
    if "anchor_day" in evidence_trace:
        lines.append(f"- anchor_day: {evidence_trace['anchor_day']}")
    if "anchor_day_end_min" in evidence_trace:
        lines.append(f"- anchor_day_end_min: {evidence_trace['anchor_day_end_min']}")
    if "chosen_end_before_min" in evidence_trace:
        lines.append(f"- chosen_end_before_min: {evidence_trace['chosen_end_before_min']}")
    if "baseline_max_distance_km" in evidence_trace:
        lines.append(f"- baseline_max_distance_km: {evidence_trace['baseline_max_distance_km']:.2f}")
    if "chosen_walk_dist_km" in evidence_trace:
        lines.append(f"- chosen_walk_dist_km: {evidence_trace['chosen_walk_dist_km']}")
    if "baseline_total_cost" in evidence_trace:
        lines.append(f"- baseline_total_cost: {evidence_trace['baseline_total_cost']:.2f}")
    if "chosen_budget" in evidence_trace:
        lines.append(f"- chosen_budget: {evidence_trace['chosen_budget']}")
    if "baseline_ticket_cost" in evidence_trace:
        lines.append(f"- baseline_ticket_cost: {evidence_trace['baseline_ticket_cost']:.2f}")
    if "chosen_ticket_budget" in evidence_trace:
        lines.append(f"- chosen_ticket_budget: {evidence_trace['chosen_ticket_budget']}")
    if "baseline_max_transport_duration_min" in evidence_trace:
        lines.append(f"- baseline_max_transport_duration_min: {evidence_trace['baseline_max_transport_duration_min']}")
    if "chosen_transport_duration_limit_min" in evidence_trace:
        lines.append(f"- chosen_transport_duration_limit_min: {evidence_trace['chosen_transport_duration_limit_min']}")
    if "baseline_ticket_price" in evidence_trace:
        lines.append(f"- baseline_ticket_price: {evidence_trace['baseline_ticket_price']:.2f}")
    if "chosen_ticket_price_cap" in evidence_trace:
        lines.append(f"- chosen_ticket_price_cap: {evidence_trace['chosen_ticket_price_cap']}")
    if len(lines) == 1:
        lines.append("- no_numeric_anchor_available")
    return "\n".join(lines)


def _extract_first_threshold(constraints_data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    temporal = constraints_data.get("temporal", [])
    if isinstance(temporal, list):
        for item in temporal:
            if not isinstance(item, dict):
                continue
            for key in keys:
                if key in item:
                    return item.get(key)
    spatial = constraints_data.get("spatial", {})
    if isinstance(spatial, dict):
        for key in keys:
            if key in spatial:
                return spatial.get(key)
    resource = constraints_data.get("resource", {})
    if isinstance(resource, dict):
        for key in keys:
            if key in resource:
                return resource.get(key)
    return None


def validate_conflict_trigger(
    constraints_data: Dict[str, Any],
    evidence: Dict[str, Any],
    constraint_type: str,
    fallback_params: Dict[str, Any],
    edit_query: str = "",
) -> FeasibilityResult:
    """
    deterministic 闸门：检查生成约束是否与 origin_plan 证据一致，并能触发预期冲突。
    """
    fallback = dict(fallback_params or {})
    fallback.setdefault("_constraint_type", str(constraint_type or "").strip())
    return validate_conflict_trigger_proof(
        origin_plan=fallback.get("_origin_plan", {}),
        constraints=constraints_data,
        constraint_type=constraint_type,
        edit_query=edit_query,
        fallback_params=fallback,
        evidence=evidence,
    )


def repair_constraints_with_anchor(
    constraints_data: Dict[str, Any],
    constraint_type: str,
    target_city: str,
    fallback_params: Dict[str, Any],
    pois: List[Dict[str, Any]],
    edit_query: str = "",
) -> Dict[str, Any]:
    repaired = dict(constraints_data) if isinstance(constraints_data, dict) else {}
    repaired.setdefault("must_include", [])
    repaired.setdefault("spatial", {})
    repaired.setdefault("temporal", [])
    repaired.setdefault("resource", {})

    if not isinstance(repaired["spatial"], dict):
        repaired["spatial"] = {}
    repaired["spatial"]["city"] = target_city

    if constraint_type == "semantic_substitution":
        source_poi = _extract_replace_source_poi(edit_query)
        semantic_target = _extract_semantic_target_from_query(edit_query)
        repaired["must_include"] = []
        semantic = repaired.get("semantic", {})
        if not isinstance(semantic, dict):
            semantic = {}
        if source_poi:
            forbidden = repaired.get("forbidden", [])
            if not isinstance(forbidden, list):
                forbidden = []
            if source_poi not in forbidden:
                forbidden.append(source_poi)
            repaired["forbidden"] = forbidden
            semantic["replace_from"] = source_poi
        if semantic_target:
            semantic["semantic_target"] = semantic_target
        repaired["semantic"] = semantic
    elif constraint_type == "cross_day_restructure":
        day_pairs = _extract_day_poi_pairs_from_query(edit_query)
        if day_pairs:
            existing = repaired.get("must_include", [])
            if not isinstance(existing, list):
                existing = []
            names = {
                poi.get("name")
                for poi in existing
                if isinstance(poi, dict) and isinstance(poi.get("name"), str)
            }
            day_to_pois: Dict[int, List[str]] = {}
            for day, poi_name in day_pairs:
                day_to_pois.setdefault(day, [])
                if poi_name not in day_to_pois[day]:
                    day_to_pois[day].append(poi_name)
                if poi_name not in names:
                    existing.append({"name": poi_name, "type": "attraction", "city": target_city})
                    names.add(poi_name)
            repaired["must_include"] = existing
            repaired["temporal"] = [
                {"day": day, "must_visit": poi_names}
                for day, poi_names in sorted(day_to_pois.items(), key=lambda kv: kv[0])
            ]
    return repaired


def parse_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    解析LLM返回的JSON

    Args:
        response_text: LLM返回的文本

    Returns:
        解析后的字典，如果失败返回None
    """
    # 尝试提取JSON
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 如果没有code block，尝试直接解析
        json_str = response_text.strip()

    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print(f"    Error parsing LLM response: {e}")
        return None


def _build_step2_recovery(sample: Dict[str, Any]) -> Dict[str, Any]:
    existing = sample.get("meta", {}).get("step2_recovery", {})
    if not isinstance(existing, dict):
        existing = {}
    return {
        "attempt_round": int(existing.get("attempt_round", 0)),
        "template_attempts": list(existing.get("template_attempts", [])) if isinstance(existing.get("template_attempts", []), list) else [],
        "used_templates": list(existing.get("used_templates", [])) if isinstance(existing.get("used_templates", []), list) else [],
        "used_origin_plan_files": list(existing.get("used_origin_plan_files", [])) if isinstance(existing.get("used_origin_plan_files", []), list) else [],
        "current_recovery_stage": str(existing.get("current_recovery_stage", "same_template")),
        "failure_reason_code": str(existing.get("failure_reason_code", "")),
        "failure_reason": str(existing.get("failure_reason", "")),
    }


def _generate_query_candidate(
    sample: Dict[str, Any],
    template: Dict[str, Any],
    llm_client,
    poi_db: POIDatabase,
    rng: random.Random,
    evidence: Dict[str, Any],
    target_bucket: Any,
    target_city: str,
    attempt_idx: int,
) -> Dict[str, Any]:
    template_str = template.get("template", "")
    constraint_type = template.get("constraint_type", "general")
    effective_template = dict(template)
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    soft_family = str(meta.get("soft_constraint_family", "")).strip()
    soft_facet = str(meta.get("soft_constraint_facet", "")).strip()
    soft_edit_mode = str(meta.get("soft_edit_mode", "")).strip()
    if str(meta.get("soft_control_status", "")).strip() == "bypassed_for_constraint_type_whitelist":
        soft_family = ""
        soft_facet = ""
        soft_edit_mode = ""
    origin_soft_profile = meta.get("origin_soft_profile", []) if isinstance(meta.get("origin_soft_profile", []), list) else []

    prefer_existing_poi = (
        template_uses_poi_placeholders(template_str)
        and template_refers_existing_poi_semantically(template_str, constraint_type)
        and effective_template.get("poi_source") not in {"existing_attractions", "restaurants", "accommodations"}
    )
    if prefer_existing_poi:
        effective_template["poi_source"] = "existing_attractions"

    pois = sample_pois_for_template(effective_template, target_city, sample, poi_db, rng)
    hard_anchor_existing_constraints = {"duration_limit", "poi_local_radius_limit", "ticket_price_cap_or_substitute"}
    if not pois and prefer_existing_poi:
        if constraint_type in hard_anchor_existing_constraints:
            return {
                "success": False,
                "reason_code": "missing_existing_poi_anchor",
                "reason": f"No existing attractions found for hard-anchor template (constraint_type={constraint_type})",
            }
        pois = sample_pois_for_template(template, target_city, sample, poi_db, rng)

    params = {
        "start_city": sample["meta"]["start_city"],
        "target_city": target_city,
        "day_count": sample["meta"]["day_count"],
    }
    template_params = template.get("params", {})
    import pandas as pd

    semantic_type_selection_trace: Dict[str, Any] = {}
    if isinstance(template_params, dict):
        for param_key, param_values in template_params.items():
            if isinstance(param_values, (list, tuple)):
                if len(param_values) > 0:
                    if (
                        param_key == "poi_type"
                        and str(constraint_type) in SEMANTIC_DIVERSITY_CONSTRAINT_TYPES
                    ):
                        selected_type, semantic_type_selection_trace = _choose_semantic_poi_type(
                            template=template,
                            param_values=param_values,
                            target_bucket=target_bucket,
                            target_city=target_city,
                            sample=sample,
                            poi_db=poi_db,
                            rng=rng,
                            soft_family=soft_family,
                            soft_facet=soft_facet,
                        )
                        params[param_key] = selected_type or rng.choice(list(param_values))
                    else:
                        params[param_key] = rng.choice(list(param_values))
            elif isinstance(param_values, pd.Series):
                if len(param_values) > 0:
                    params[param_key] = rng.choice(param_values.tolist())
            else:
                params[param_key] = param_values

    params, pois, evidence_trace = apply_conflict_driven_param_adjustments(
        sample=sample,
        template=template,
        params=params,
        pois=pois,
        evidence=evidence,
    )
    if semantic_type_selection_trace:
        evidence_trace["semantic_type_selection"] = semantic_type_selection_trace
    edit_soft_target_suppressed = should_suppress_extend_with_pacing_edit_soft(
        template=template,
        constraint_type=str(constraint_type),
        params=params,
    )
    original_soft_target = {
        "family": soft_family,
        "facet": soft_facet,
        "mode": soft_edit_mode,
        "intensity": str(template.get("soft_intensity", "")).strip(),
    }
    if edit_soft_target_suppressed:
        soft_family = ""
        soft_facet = ""
        soft_edit_mode = ""
    pois = _align_pois_with_type_requirements(
        template_str=template_str,
        constraint_type=str(constraint_type),
        params=params,
        target_city=target_city,
        poi_db=poi_db,
        fallback_pois=pois,
    )
    if pois and isinstance(pois[0], dict):
        params["poi_name"] = pois[0].get("name")

    canonical_query = fill_template(template_str, target_city, pois, params)
    unresolved_placeholders = _find_unresolved_placeholders(canonical_query)
    if unresolved_placeholders:
        return {
            "success": False,
            "reason_code": "template_param_missing",
            "reason": f"Template placeholders unresolved: {', '.join(unresolved_placeholders)}",
        }
    poi_list_str = format_pois(pois)
    evidence_str = format_evidence_for_prompt(constraint_type, evidence_trace)
    soft_target_str = (
        f"{soft_family}:{soft_facet}:{soft_edit_mode}"
        if soft_family and soft_facet and soft_edit_mode
        else "(none)"
    )
    origin_soft_profile_str = summarize_soft_profile(origin_soft_profile)

    prompt = f"""你是一个旅游规划助手。根据原始旅行计划和目标conflict bucket，生成一个编辑需求（edit query）。

原始计划：
- 出发城市: {sample["meta"]["start_city"]}
- 目标城市: {target_city}
- 天数: {sample["meta"]["day_count"]}天

目标冲突类型（用于数据集标注）: {target_bucket}

可用的景点（{target_city}）):
{poi_list_str}

参考模板类型: {constraint_type}

参考模板: {template_str}

目标 soft control: {soft_target_str}

origin soft profile: {origin_soft_profile_str}

生成的编辑需求:
{canonical_query}

原计划证据（必须与输出一致）:
{evidence_str}

请将上述编辑需求转换为结构化的约束条件，包括：
1. must_include: 必须包含的POI列表（仅当query明确提到具体POI时填写；否则返回空列表[]）
2. spatial: 空间约束（城市限制等）
3. temporal: 时间约束（如果有）
4. resource: 资源约束（如果有；酒店/餐饮预算、房型、房间数等都放在这里）
5. forbidden: 禁止出现的POI列表（仅替换/移除类需求需要）
6. semantic: 语义替换目标或类型/特征偏好（例如 replace_from、semantic_target、preferred_type、hotel_feature）

重要约束：
- 所有POI必须是{target_city}的真实景点，不要凭空捏造
- 不要提出更换城市的请求（保持target_city为目标城市）
- 时间、预算等约束要合理可执行
- 必须与“原计划证据”保持一致，不得捏造超限事实
- 若当前阈值不足以构成冲突，请给出更严格阈值以保证触发目标冲突
- 若是“把X改成Y/换成Y”类需求，X 不应放进 must_include，可放入 forbidden
- 若query包含“第N天去POI”，请在 temporal 中写明 day + must_visit
- 若目标 soft control 不为空，query 必须清晰表达对应 family/facet，且不能改成别的 soft 类别

返回JSON格式：
{{
  "must_include": [],
  "spatial": {{"city": "{target_city}"}},
  "temporal": [],
  "resource": {{}},
  "forbidden": [],
  "semantic": {{}}
}}
"""

    messages = [
        {"role": "system", "content": "You are a travel planning assistant. Generate edit queries in Chinese."},
        {"role": "user", "content": prompt},
    ]

    debug_logger = create_llm_debug_logger("02", sample["sample_id"], "generate_query")
    response = llm_client.call_with_retry(
        messages,
        max_retries=2,
        debug_logger=debug_logger,
        debug_context={
            "target_city": target_city,
            "target_bucket": target_bucket,
            "constraint_type": constraint_type,
            "attempt_idx": attempt_idx,
        },
    )
    if response is None:
        return {
            "success": False,
            "reason_code": "llm_response_missing",
            "reason": "LLM call failed",
        }

    constraints_data = parse_llm_response(response.content)
    if constraints_data is None or not isinstance(constraints_data, dict):
        return {
            "success": False,
            "reason_code": "constraint_parse_failed",
            "reason": "Failed to parse LLM constraints response",
        }

    semantic_slots, semantic_parse_source = _parse_semantic_slots_with_fallback(
        llm_client=llm_client,
        sample_id=sample["sample_id"],
        edit_query=canonical_query,
        constraint_type=str(constraint_type),
        template_str=template_str,
        anchor_hint=params.get("poi") if isinstance(params, dict) else None,
    )

    constraints_data = normalize_constraints_by_type(
        constraints_data=constraints_data,
        constraint_type=constraint_type,
        template_str=template_str,
        edit_query=canonical_query,
        target_city=target_city,
        semantic_slots=semantic_slots,
        origin_plan=sample.get("origin_plan"),
    )
    query_spec = extract_query_spec(
        canonical_query,
        str(constraint_type),
        semantic_slots=semantic_slots,
        origin_plan=sample.get("origin_plan"),
    )
    alignment_errors = validate_query_spec_alignment(query_spec, constraints_data)

    fallback_params = dict(params)
    fallback_params["_origin_plan"] = sample.get("origin_plan", {})
    fallback_params.update(query_spec)
    if "source_poi" in query_spec:
        fallback_params.setdefault("poi_name", query_spec.get("source_poi"))
    if "anchor_poi" in query_spec:
        fallback_params.setdefault("poi_name", query_spec.get("anchor_poi"))
    proof: Optional[FeasibilityResult] = None
    if not alignment_errors:
        proof = validate_conflict_trigger(
            constraints_data=constraints_data,
            evidence=evidence,
            constraint_type=constraint_type,
            fallback_params=fallback_params,
            edit_query=canonical_query,
        )
    repaired = False
    repair_reason = ""
    repair_mode = "none"
    if (alignment_errors or (proof is not None and not proof.ok)) and constraint_type in SAFE_CANONICALIZATION_TYPES:
        repaired = True
        repair_mode = "safe_canonicalization"
        repair_reason = "; ".join(alignment_errors) if alignment_errors else proof.reason
        constraints_data = repair_constraints_with_anchor(
            constraints_data=constraints_data,
            constraint_type=constraint_type,
            target_city=target_city,
            fallback_params=fallback_params,
            pois=pois,
            edit_query=canonical_query,
        )
        constraints_data = normalize_constraints_by_type(
            constraints_data=constraints_data,
            constraint_type=constraint_type,
            template_str=template_str,
            edit_query=canonical_query,
            target_city=target_city,
            semantic_slots=semantic_slots,
            origin_plan=sample.get("origin_plan"),
        )
        alignment_errors = validate_query_spec_alignment(query_spec, constraints_data)
        if not alignment_errors:
            proof = validate_conflict_trigger(
                constraints_data=constraints_data,
                evidence=evidence,
                constraint_type=constraint_type,
                fallback_params=fallback_params,
                edit_query=canonical_query,
            )
        else:
            proof = None

    if alignment_errors:
        return {
            "success": False,
            "reason_code": "query_constraint_alignment_failed",
            "reason": "; ".join(alignment_errors),
        }

    if proof is None or not proof.ok:
        return {
            "success": False,
            "reason_code": proof.reason_code if proof is not None else "proof_missing",
            "reason": proof.reason if proof is not None else "Proof was not executed",
        }

    soft_alignment_errors = validate_soft_query_alignment(
        edit_query=canonical_query,
        origin_profile=origin_soft_profile,
        target_family=soft_family,
        target_facet=soft_facet,
        target_mode=soft_edit_mode,
        template_soft_intensity=str(template.get("soft_intensity", "")).strip(),
        constraint_type=str(constraint_type),
        semantic_slots=semantic_slots,
        query_spec=query_spec,
    )
    if soft_alignment_errors:
        return {
            "success": False,
            "reason_code": "soft_alignment_failed",
            "reason": "; ".join(soft_alignment_errors),
        }

    template_semantic_spec = (
        template.get("template_semantic_spec")
        if isinstance(template.get("template_semantic_spec"), dict)
        else {}
    )
    if edit_soft_target_suppressed:
        template_semantic_spec = dict(template_semantic_spec)
        template_semantic_spec["soft_axes"] = []
    query_trace = {
        "constraint_type": constraint_type,
        "template": template_str,
        "template_id": template.get("template_id", ""),
        "template_semantic_spec_version": TEMPLATE_SEMANTIC_SPEC_VERSION,
        "template_semantic_spec": template_semantic_spec,
        "evidence": evidence_trace,
        "semantic_type_selection": evidence_trace.get("semantic_type_selection", {}),
        "rendered_params": _prune_rendered_params_for_trace(template_str, params),
        "canonical_query_spec": query_spec,
        "canonical_semantic_slots": semantic_slots,
        "canonical_semantic_parse_source": semantic_slots.get("parse_source", semantic_parse_source),
        "canonical_semantic_parse_confidence": semantic_slots.get("parse_confidence", 0.0),
        "canonical_semantic_parse_errors": semantic_slots.get("parse_errors", []),
        "structure_frozen": True,
        "canonical_alignment_pass": True,
        "canonical_alignment_errors": [],
        "canonical_gate_pass": proof.ok,
        "canonical_gate_reason": proof.reason,
        "canonical_proof_ok": proof.ok,
        "canonical_proof_reason_code": proof.reason_code,
        "canonical_proof_evidence": proof.evidence_used,
        "soft_target_family": soft_family,
        "soft_target_facet": soft_facet,
        "soft_target_mode": soft_edit_mode,
        "soft_target_intensity": "" if edit_soft_target_suppressed else str(template.get("soft_intensity", "")).strip(),
        "edit_soft_target_suppressed": edit_soft_target_suppressed,
        "edit_soft_target_suppressed_reason": EXTEND_PACING_SOFT_SUPPRESSION_REASON
        if edit_soft_target_suppressed
        else "",
        "suppressed_soft_target": original_soft_target if edit_soft_target_suppressed else {},
        "soft_alignment_pass": True,
        "soft_alignment_errors": [],
        "query_surface_source": "",
        "surface_quality_pass": False,
        "surface_reject_reason": "",
        "repaired": repaired,
        "repair_mode": repair_mode,
        "repair_reason": repair_reason,
    }
    canonical_constraint_ir = build_canonical_constraint_ir_from_structured_inputs(
        template_text=template_str,
        constraint_type=str(constraint_type),
        instantiated_slots={
            "query_spec": query_spec,
            "semantic_slots": semantic_slots,
            "constraints": constraints_data,
            "origin_plan": sample.get("origin_plan"),
            "origin_query_structured": sample.get("origin_query_structured"),
            "surface_query": canonical_query,
        },
        template_semantic_spec=template_semantic_spec,
        meta={
            "template_id": template.get("template_id", ""),
            "query_generation_trace": query_trace,
        },
    )
    surface_result = select_query_surface(
        llm_client=llm_client,
        sample_id=sample["sample_id"],
        canonical_query=canonical_query,
        constraint_type=str(constraint_type),
        template=template,
        target_city=target_city,
        pois=pois,
        params=params,
        evidence_trace=evidence_trace,
        soft_target_str=soft_target_str,
        origin_query_text=str(sample.get("origin_query_text", "") or ""),
        constraints_data=constraints_data,
        fallback_params=fallback_params,
        evidence=evidence,
        canonical_query_spec=query_spec,
        canonical_semantic_slots=semantic_slots,
        canonical_proof=proof,
        target_family=soft_family,
        target_facet=soft_facet,
        target_mode=soft_edit_mode,
        origin_profile=origin_soft_profile,
    )
    surface_proof = surface_result.get("surface_proof")
    query_trace.update(
        {
            "surface_query_spec": surface_result.get("surface_query_spec", query_spec),
            "surface_semantic_slots": surface_result.get("surface_semantic_slots", semantic_slots),
            "surface_semantic_parse_source": surface_result.get("surface_semantic_parse_source", ""),
            "query_surface_source": surface_result.get("query_surface_source", "template_fallback"),
            "surface_quality_pass": bool(surface_result.get("surface_quality_pass", False)),
            "surface_reject_reason": surface_result.get("surface_reject_reason", ""),
            "surface_proof_ok": bool(getattr(surface_proof, "ok", False)),
            "surface_proof_reason_code": getattr(surface_proof, "reason_code", ""),
            "surface_proof_reason": getattr(surface_proof, "reason", ""),
            "surface_proof_evidence": getattr(surface_proof, "evidence_used", {}),
        }
    )
    require_natural_surface = bool(template.get("require_natural_surface")) or str(
        os.getenv("STEP2_REQUIRE_NATURAL_SURFACE", "")
    ).strip().lower() in {"1", "true", "yes"}
    final_edit_query = str(surface_result.get("edit_query") or canonical_query).strip()
    if require_natural_surface and (
        not query_trace["surface_quality_pass"]
        or final_edit_query == canonical_query
        or query_trace["query_surface_source"] == "template_fallback"
    ):
        return {
            "success": False,
            "reason_code": "surface_naturalization_failed",
            "reason": str(query_trace.get("surface_reject_reason") or "no accepted natural surface"),
        }
    register_query_surface(final_edit_query)
    if str(constraint_type) in SEMANTIC_DIVERSITY_CONSTRAINT_TYPES:
        record_semantic_type_success(target_bucket, target_city, query_spec.get("preferred_type"))

    return {
        "success": True,
        "canonical_edit_query": canonical_query,
        "edit_query": final_edit_query,
        "constraints": constraints_data,
        "canonical_constraint_ir": canonical_constraint_ir,
        "query_generation_trace": query_trace,
    }


def generate_query_for_sample(sample: Dict[str, Any],
                               llm_client,
                               templates_config: Dict,
                               poi_db: POIDatabase,
                               rng: random.Random,
                               allowed_violations: set,
                               constraint_type_whitelist: Optional[set[str]] = None) -> Optional[Dict[str, Any]]:
    """
    为单个sample生成query

    Args:
        sample: sample字典
        llm_client: LLM客户端
        templates_config: 模板配置
        poi_db: POI数据库

    Returns:
        生成的edit_query和constraints，如果失败返回None
    """
    target_bucket = sample["meta"]["target_bucket"]
    target_city = sample["meta"]["target_city"]
    evidence = build_origin_plan_evidence(sample)
    template_config = select_template_for_bucket(target_bucket, templates_config)

    if not template_config:
        print(f"    ⚠ Warning: No template found for bucket {target_bucket}")
        return {
            "success": False,
            "status": "02_failed_exhausted",
            "failure_reason": f"No template found for bucket {target_bucket}",
            "failure_reason_code": "missing_template",
            "step2_recovery": _build_step2_recovery(sample),
        }

    if not is_bucket_eligible_for_edit(target_bucket, template_config, allowed_violations):
        return {
            "success": False,
            "status": "02_failed_exhausted",
            "failure_reason": f"Bucket {target_bucket} is not eligible_for_edit",
            "failure_reason_code": "ineligible_bucket",
            "step2_recovery": _build_step2_recovery(sample),
        }

    templates = list_templates_for_bucket(target_bucket, templates_config)
    sample_constraint_type_filter = resolve_sample_constraint_type_filter(sample, constraint_type_whitelist)
    templates = filter_templates_by_constraint_type(templates, sample_constraint_type_filter)
    templates = filter_templates_by_domain_scope(templates, sample)
    if sample_constraint_type_filter and not templates:
        return {
            "success": False,
            "status": "02_failed_exhausted",
            "failure_reason": (
                f"No templates configured for bucket {target_bucket} under constraint_type whitelist "
                f"{sorted(sample_constraint_type_filter)}"
            ),
            "failure_reason_code": "missing_constraint_type_template_entries",
            "step2_recovery": _build_step2_recovery(sample),
        }
    pre_filter_template_count = len(templates)
    filtered_templates = filter_templates_for_sample(templates, sample, evidence=evidence)
    if (
        not filtered_templates
        and sample_constraint_type_filter
        and str(sample.get("meta", {}).get("soft_control_status", "")).strip() == "controlled"
    ):
        filtered_templates = filter_templates_for_sample(
            templates,
            sample,
            evidence=evidence,
            ignore_soft_control=True,
        )
        if filtered_templates:
            meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
            meta["soft_control_status"] = "bypassed_for_constraint_type_whitelist"
            sample["meta"] = meta
    if (
        not filtered_templates
        and str(sample.get("meta", {}).get("soft_constraint_family", "")).strip() == "route_compactness"
        and str(sample.get("meta", {}).get("soft_constraint_facet", "")).strip() == "travel_time"
    ):
        filtered_templates = _retarget_or_fallback_soft_control(sample, templates, evidence, rng)
    templates = filtered_templates
    if not templates:
        meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
        soft_reason = ""
        if str(meta.get("soft_control_status", "")).strip() == "controlled":
            soft_reason = (
                " after soft-control filtering "
                f"({meta.get('soft_constraint_family', '')}:{meta.get('soft_constraint_facet', '')}:{meta.get('soft_edit_mode', '')})"
            )
        if pre_filter_template_count > 0:
            return {
                "success": False,
                "status": "02_failed_exhausted",
                "failure_reason": f"All templates for bucket {target_bucket} were filtered out{soft_reason}",
                "failure_reason_code": "no_applicable_templates_after_filtering",
                "step2_recovery": _build_step2_recovery(sample),
            }
        return {
            "success": False,
            "status": "02_failed_exhausted",
            "failure_reason": f"No templates configured for bucket {target_bucket}{soft_reason}",
            "failure_reason_code": "missing_soft_template_entries" if soft_reason else "missing_template_entries",
            "step2_recovery": _build_step2_recovery(sample),
        }

    recovery = _build_step2_recovery(sample)
    current_origin_file = sample.get("meta", {}).get("origin_plan_file", "")
    if current_origin_file and current_origin_file not in recovery["used_origin_plan_files"]:
        recovery["used_origin_plan_files"].append(current_origin_file)
    recovery["attempt_round"] = int(recovery.get("attempt_round", 0)) + 1
    recovery["current_recovery_stage"] = "same_template"

    ordered_templates = order_templates_for_bucket_rotation(templates, target_bucket, rng)

    template_str = ""
    constraint_type = ""
    for template in ordered_templates:
        if not isinstance(template, dict):
            recovery["template_attempts"].append(
                {
                    "template": "",
                    "constraint_type": "",
                    "attempt_index": 0,
                    "origin_plan_file": current_origin_file,
                    "result": "failed",
                    "reason_code": "invalid_template_entry",
                    "reason": f"Template entry is not a dict: {type(template).__name__}",
                }
            )
            continue
        template_str = template.get("template", "")
        constraint_type = str(template.get("constraint_type", "general"))
        template_id = str(template.get("template_id", template_str))
        recovery["used_templates"].append(template_str)
        record_template_attempt(target_bucket, template_id)

        for attempt_idx in range(1, DEFAULT_TEMPLATE_ATTEMPTS + 1):
            candidate = _generate_query_candidate(
                sample=sample,
                template=template,
                llm_client=llm_client,
                poi_db=poi_db,
                rng=rng,
                evidence=evidence,
                target_bucket=target_bucket,
                target_city=target_city,
                attempt_idx=attempt_idx,
            )
            attempt_record = {
                "template": template_str,
                "constraint_type": constraint_type,
                "attempt_index": attempt_idx,
                "origin_plan_file": current_origin_file,
                "result": "success" if candidate.get("success") else "failed",
                "reason_code": candidate.get("reason_code", ""),
                "reason": candidate.get("reason", ""),
            }
            recovery["template_attempts"].append(attempt_record)

            if candidate.get("success"):
                record_template_attempt(target_bucket, template_id, success=True, increment_attempted=False)
                trace = candidate["query_generation_trace"]
                trace["template_attempt_count"] = attempt_idx
                trace["origin_attempt_count"] = len(recovery["used_origin_plan_files"])
                trace["recovery_action"] = "accept"
                return {
                    "success": True,
                    "canonical_edit_query": candidate["canonical_edit_query"],
                    "edit_query": candidate.get("edit_query", candidate["canonical_edit_query"]),
                    "constraints": candidate["constraints"],
                    "canonical_constraint_ir": candidate.get("canonical_constraint_ir", {}),
                    "query_generation_trace": trace,
                    "step2_recovery": recovery,
                }

    used_origin_count = len(recovery["used_origin_plan_files"])
    recovery["current_recovery_stage"] = "switch_origin"
    last_reason = recovery["template_attempts"][-1]["reason"] if recovery["template_attempts"] else "No valid template attempt"
    last_reason_code = recovery["template_attempts"][-1]["reason_code"] if recovery["template_attempts"] else "no_valid_candidate"
    recovery["failure_reason"] = last_reason
    recovery["failure_reason_code"] = last_reason_code
    status = "02_failed_retryable_new_origin" if used_origin_count < DEFAULT_MAX_ORIGIN_SWITCHES else "02_failed_exhausted"
    suggested_action = "switch_origin" if status == "02_failed_retryable_new_origin" else "fail"

    return {
        "success": False,
        "status": status,
        "failure_reason": last_reason,
        "failure_reason_code": last_reason_code,
        "step2_recovery": recovery,
        "query_generation_trace": {
            "constraint_type": recovery["template_attempts"][-1]["constraint_type"] if recovery["template_attempts"] else "",
            "template": recovery["template_attempts"][-1]["template"] if recovery["template_attempts"] else "",
            "canonical_gate_pass": False,
            "canonical_gate_reason": last_reason,
            "canonical_proof_ok": False,
            "canonical_proof_reason_code": last_reason_code,
            "canonical_proof_evidence": {},
            "recovery_action": suggested_action,
            "template_attempt_count": len(recovery["template_attempts"]),
            "origin_attempt_count": used_origin_count,
        },
    }


def format_pois(poi_list: List[Dict]) -> str:
    """格式化POI列表用于prompt"""
    if not poi_list:
        return "(无)"

    lines = []
    for i, poi in enumerate(poi_list):
        lines.append(f"{i+1}. {poi['name']} ({poi.get('type', 'N/A')})")
        if "price" in poi and poi["price"] > 0:
            lines.append(f"   价格: ¥{poi['price']}")
    return "\n".join(lines)


def save_sample(sample: Dict[str, Any], output_dir: str):
    """保存sample到文件"""
    sample_file = Path(output_dir) / f"{sample['sample_id']}.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)


def clear_step2_outputs(sample: Dict[str, Any]):
    for key in [
        "canonical_edit_query",
        "canonical_constraint_ir",
        "constraints",
        "query_generation_trace",
        "conflict_facts",
        "conflict_labels",
        "conflict_set",
        "primary_conflict",
        "secondary_conflicts",
        "strategy_plan",
        "mock_edited_plan",
        "diff_summary",
        "checks",
    ]:
        sample.pop(key, None)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Generate edit queries and constraints")
    parser.add_argument("--input", required=True, help="Batch directory from step 1")
    parser.add_argument("--config", default="data_generation/config/templates.yaml",
                       help="Templates config file")
    parser.add_argument("--model", default="auto", choices=["auto", "siliconcloud", "dmxapi"],
                       help="LLM model to use (default: auto)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Step 2 (default: read STEP2_SEED env, else non-deterministic)",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="LLM temperature for Step 2 (default: 0.15, env fallback: STEP2_LLM_TEMPERATURE/LLM_TEMPERATURE)",
    )
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Max retries per sample (default: 5)")
    parser.add_argument(
        "--allow-violations",
        default="overflow,overlap,discontinuity,incompatibility",
        help="Comma-separated violation whitelist for Step 2 eligibility",
    )
    parser.add_argument(
        "--constraint-type-whitelist",
        default=None,
        help="Optional comma-separated constraint_type whitelist applied within each bucket",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Generate Query")
    print("=" * 60)

    allowed_violations = parse_allowed_violations(args.allow_violations)
    print(f"Allowed violations: {sorted(allowed_violations)}")
    constraint_type_whitelist = parse_constraint_type_whitelist(args.constraint_type_whitelist)
    if constraint_type_whitelist:
        print(f"Constraint type whitelist: {sorted(constraint_type_whitelist)}")

    seed = args.seed
    if seed is None:
        raw_seed = os.getenv("STEP2_SEED")
        if raw_seed is not None and str(raw_seed).strip():
            try:
                seed = int(str(raw_seed).strip())
            except (TypeError, ValueError):
                seed = None
    rng = random.Random(seed)
    print(f"Step2 RNG seed: {seed if seed is not None else 'None (non-deterministic)'}")

    llm_temperature = args.llm_temperature
    if llm_temperature is None:
        raw_temp = os.getenv("STEP2_LLM_TEMPERATURE") or os.getenv("LLM_TEMPERATURE")
        if raw_temp is not None and str(raw_temp).strip():
            try:
                llm_temperature = float(str(raw_temp).strip())
            except (TypeError, ValueError):
                llm_temperature = None
    if llm_temperature is None:
        llm_temperature = 0.15
    print(f"Step2 LLM temperature: {llm_temperature}")

    # 1. 加载配置
    print(f"\n1. Loading templates config from: {args.config}")
    try:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            templates_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading templates config: {e}")
        sys.exit(1)

    missing_validators = ensure_validator_coverage(templates_config)
    if missing_validators:
        print(f"Error: Missing feasibility validators for eligible constraint types: {missing_validators}")
        sys.exit(1)

    # 2. 初始化
    print("\n2. Initializing LLM client and POI database...")
    try:
        llm_client = load_client_from_env(provider=args.model, temperature=llm_temperature)
        poi_db = POIDatabase()
        print("  ✓ LLM client and POI database initialized")
    except Exception as e:
        print(f"  ✗ Error initializing: {e}")
        sys.exit(1)

    # 3. 加载samples
    print(f"\n3. Loading samples from: {args.input}")
    samples = load_samples(args.input)
    pending = filter_pending_samples(samples)

    if not pending:
        print("No samples need query generation. Exiting.")
        return

    # 4. 生成queries
    print(f"\n4. Generating queries for {len(pending)} samples...")
    success_count = 0
    fail_count = 0

    for i, sample in enumerate(pending):
        try:
            print(f"\n  [{i+1}/{len(pending)}] Processing {sample['sample_id']}...")

            # 生成query
            result = generate_query_for_sample(
                sample,
                llm_client,
                templates_config,
                poi_db,
                rng,
                allowed_violations=allowed_violations,
                constraint_type_whitelist=constraint_type_whitelist,
            )

            if result:
                sample["meta"]["step2_recovery"] = result.get("step2_recovery", {})
                if result.get("success"):
                    sample["canonical_edit_query"] = result["canonical_edit_query"]
                    sample["edit_query"] = result.get("edit_query", result["canonical_edit_query"])
                    sample["canonical_constraint_ir"] = result.get("canonical_constraint_ir", {})
                    sample["constraints"] = result["constraints"]
                    sample["query_generation_trace"] = result.get("query_generation_trace", {})
                    sample["meta"]["status"] = "02_completed"
                    sample["meta"]["last_step"] = "02_generate_query"
                    sample["meta"]["retry_count"] = 0
                    sample["meta"]["failure_reason_code"] = ""
                    sample["meta"]["failure_reason"] = ""
                    save_sample(sample, args.input)
                    print("    ✓ Canonical query generated successfully (canonical_proof_ok=True)")
                    success_count += 1
                else:
                    clear_step2_outputs(sample)
                    sample["query_generation_trace"] = result.get("query_generation_trace", {})
                    sample["meta"]["status"] = result.get("status", "02_failed_exhausted")
                    sample["meta"]["last_step"] = "02_generate_query"
                    sample["meta"]["failure_reason"] = result.get("failure_reason", "Failed to generate valid query")
                    sample["meta"]["failure_reason_code"] = result.get("failure_reason_code", "step2_generation_failed")
                    sample["meta"]["retry_count"] = sample["meta"].get("retry_count", 0) + 1
                    save_sample(sample, args.input)
                    print(f"    ⚠ Step 2 did not produce a valid query: {sample['meta']['failure_reason']}")
                    fail_count += 1
            else:
                clear_step2_outputs(sample)
                sample["meta"]["status"] = "02_failed"
                sample["meta"]["failure_reason"] = "Failed to generate or parse LLM response"
                sample["meta"]["failure_reason_code"] = "step2_generation_failed"
                sample["meta"]["retry_count"] = sample["meta"].get("retry_count", 0) + 1
                save_sample(sample, args.input)

                print(f"    ✗ Failed")
                fail_count += 1

        except Exception as e:
            print(f"    ✗ Error: {e}")
            clear_step2_outputs(sample)
            sample["meta"]["status"] = "02_failed"
            sample["meta"]["failure_reason"] = str(e)
            sample["meta"]["failure_reason_code"] = "step2_exception"
            sample["meta"]["retry_count"] = sample["meta"].get("retry_count", 0) + 1
            save_sample(sample, args.input)
            fail_count += 1

    # 5. 总结
    print("\n" + "=" * 60)
    print("✓ Step 2 Complete!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Success rate: {success_count}/{len(pending)} ({100*success_count/len(pending):.1f}%)")
    print(f"\nOutput location: {args.input}/")
    print(f"\nNext step:")
    print(f"  python scripts/03_freeze_edit_truth.py --input {args.input}")


if __name__ == "__main__":
    main()
