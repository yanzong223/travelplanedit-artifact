"""
Data Generation Validators
数据生成阶段专用的验证函数

Note: 这些validators专门用于数据生成质量控制，与src/utils/validation.py（用于评估edited plans）不同
"""

import sys
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from poi_database import POIDatabase
from conflict_resolver import (
    resolve_primary_conflict as resolve_primary_conflict_rule,
    normalize_conflict_label as normalize_conflict_label_rule,
)
from conflict_feasibility import (
    assess_edit_query_factual_validity,
    build_origin_plan_evidence,
    validate_conflict_trigger as validate_conflict_trigger_proof,
    _extract_activity_ticket_price,
)
from solver_feasibility import assess_solver_feasibility
from temporal_window_feasibility import assess_temporal_window_feasibility
from category_guard_validity import assess_category_guard_validity
from evaluation.benchmark.hard_truth import serialize_logical_constraints, validate_logical_constraint_schema
from evaluation.benchmark.constraint_merge import (
    MERGE_POLICY_VERSION,
    build_effective_logical_constraints,
)
from evaluation.benchmark.soft_truth import (
    build_edit_target_preference_tags,
    build_edit_target_preferences,
    serialize_preference_constraints,
)
from constraint_ir.adapters import build_edit_targets_from_ir
from constraint_ir.compiler import build_canonical_constraint_ir_from_structured_inputs
from constraint_ir.validator import (
    check_hard_soft_axis_dedup,
    check_ir_compiles,
    check_template_ir_consistency,
    validate_canonical_constraint_ir,
)
from soft_constraint_control import validate_soft_query_alignment
from query_surface_parsing import (
    extract_preferred_poi_type_from_query as shared_extract_preferred_poi_type_from_query,
    is_explicit_hard_travel_time_query,
    normalize_preferred_poi_type,
    resolve_query_semantic_slots,
)

# 使用geopy的geodesic计算距离（与Chinatravel项目一致）
try:
    from geopy.distance import geodesic
    GEODESIC_AVAILABLE = True
except ImportError:
    GEODESIC_AVAILABLE = False
    print("Warning: geopy not available, distance calculation will be limited")


SUPPORTED_CITY_NAMES = {
    "北京", "上海", "南京", "苏州", "杭州",
    "深圳", "成都", "武汉", "广州", "重庆",
}

TRANSIT_ACTIVITY_TYPES = {
    "train",
    "airplane",
    "flight",
    "bus",
    "taxi",
    "subway",
    "metro",
    "walk",
    "walking",
    "drive",
    "car",
    "boat",
    "ferry",
}

DIMENSION_COMPATIBILITY = {
    "temporal": {"sequence"},
    "sequence": {"temporal"},
}


def _extract_query_city_mentions(edit_query: str) -> List[str]:
    if not isinstance(edit_query, str) or not edit_query.strip():
        return []

    escaped = "|".join(sorted(SUPPORTED_CITY_NAMES, key=len, reverse=True))
    patterns = [
        rf"(?:在|去|到|前往|改到|换到|安排到|安排在)({escaped})",
        rf"({escaped})(?:玩|住|待|旅行|行程|游玩)",
    ]

    mentions: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, edit_query):
            city = next((group for group in match.groups() if group), None)
            if city and city not in mentions:
                mentions.append(city)
    return mentions


def _is_intercity_activity(activity_type: str) -> bool:
    """判断是否是跨城交通活动。"""
    return activity_type in {"train", "airplane"}


def _parse_time_to_minutes(time_str: Any) -> Optional[int]:
    """
    将 HH:MM 转为分钟数，支持 24:00。
    返回 None 表示格式非法。
    """
    if not isinstance(time_str, str):
        return None

    parts = time_str.split(":")
    if len(parts) != 2:
        return None

    try:
        hour, minute = map(int, parts)
    except ValueError:
        return None

    if hour == 24 and minute == 0:
        return 24 * 60
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour * 60 + minute
    return None


def _duration_minutes(start_time: Any, end_time: Any) -> Optional[int]:
    """
    计算活动时长（分钟），支持跨午夜。
    """
    start_minutes = _parse_time_to_minutes(start_time)
    end_minutes = _parse_time_to_minutes(end_time)
    if start_minutes is None or end_minutes is None:
        return None

    duration = end_minutes - start_minutes
    if duration < 0:
        duration += 24 * 60
    return duration


def _resolve_activity_from_citation(
    plan: Dict[str, Any],
    citation: Any,
) -> Optional[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]:
    if not isinstance(citation, str):
        return None
    match = re.fullmatch(r"\s*itinerary\[(\d+)\]\.activities\[(\d+)\]\s*", citation)
    if not match:
        return None

    day_idx = int(match.group(1))
    act_idx = int(match.group(2))
    itinerary = plan.get("itinerary", [])
    if not isinstance(itinerary, list) or day_idx < 0 or day_idx >= len(itinerary):
        return None

    day_plan = itinerary[day_idx]
    activities = day_plan.get("activities", [])
    if not isinstance(activities, list) or act_idx < 0 or act_idx >= len(activities):
        return None

    return day_idx, act_idx, day_plan, activities[act_idx]


def _extract_activity_endpoints(activity: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    提取活动的起终点。
    优先使用 start/end，其次使用 transports 链路，最后回退到 position。
    """
    start_point = activity.get("start")
    end_point = activity.get("end")
    position = activity.get("position")
    transports = activity.get("transports", [])

    if not start_point and transports and isinstance(transports, list):
        first = transports[0]
        if isinstance(first, dict):
            start_point = first.get("start")

    if not end_point and transports and isinstance(transports, list):
        last = transports[-1]
        if isinstance(last, dict):
            end_point = last.get("end")

    if not start_point:
        start_point = position
    if not end_point:
        end_point = position

    return start_point, end_point


def _extract_continuity_entry(activity: Dict[str, Any]) -> Optional[str]:
    """
    提取用于地点连续性比较的入口点。
    优先使用当前活动第一段 transport.start，再回退到活动 start/position。
    """
    transports = activity.get("transports", [])
    if isinstance(transports, list):
        for transport in transports:
            if not isinstance(transport, dict):
                continue
            transport_start = transport.get("start")
            if isinstance(transport_start, str) and transport_start.strip():
                return transport_start
            break
    start_point, _ = _extract_activity_endpoints(activity)
    return start_point


def _normalize_location_for_match(location: Optional[str]) -> str:
    """
    地点名轻量归一化，降低“同地不同写法”误判。
    """
    if not isinstance(location, str):
        return ""
    normalized = location.strip().lower()
    if not normalized:
        return ""

    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = re.sub(r"\s+", "", normalized)

    for suffix in (
        "-地铁站",
        "地铁站",
        "-地铁口",
        "地铁口",
        "南广场",
        "北广场",
        "东广场",
        "西广场",
    ):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    normalized = re.sub(r"[\-_/·•()]", "", normalized)
    return normalized


def _locations_equivalent(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_norm = _normalize_location_for_match(left)
    right_norm = _normalize_location_for_match(right)
    return bool(left_norm and right_norm and left_norm == right_norm)


def _collect_plan_pois(plan: Dict[str, Any]) -> set:
    """
    收集plan中出现过的地点名，兼容 chinatravel 的 position 字段语义。
    """
    plan_pois = set()
    itinerary = plan.get("itinerary", [])

    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        for activity in activities:
            for field in ["position", "start", "end"]:
                value = activity.get(field)
                if value:
                    plan_pois.add(value)

            transports = activity.get("transports", [])
            if isinstance(transports, list):
                for transport in transports:
                    if not isinstance(transport, dict):
                        continue
                    for field in ["start", "end"]:
                        value = transport.get(field)
                        if value:
                            plan_pois.add(value)

    return plan_pois


def _collect_visited_pois(plan: Dict[str, Any]) -> set:
    """
    收集语义上真正“访问”的 POI，尽量排除交通中转点。
    """
    visited_pois = set()
    itinerary = plan.get("itinerary", [])

    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        for activity in activities:
            if not isinstance(activity, dict):
                continue

            activity_type = str(activity.get("type") or "").strip().lower()
            position = activity.get("position")
            if isinstance(position, str) and position:
                visited_pois.add(position)
                continue

            if activity_type in TRANSIT_ACTIVITY_TYPES:
                continue

            end_point = activity.get("end")
            start_point = activity.get("start")
            if isinstance(end_point, str) and end_point:
                visited_pois.add(end_point)
            elif isinstance(start_point, str) and start_point:
                visited_pois.add(start_point)

    return visited_pois


def _collect_transit_points(plan: Dict[str, Any]) -> set:
    """
    收集交通链路中的起终点，单独用于判断 transit-only 命中。
    """
    transit_points = set()
    itinerary = plan.get("itinerary", [])

    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        for activity in activities:
            if not isinstance(activity, dict):
                continue

            activity_type = str(activity.get("type") or "").strip().lower()
            if activity_type in TRANSIT_ACTIVITY_TYPES:
                for field in ["start", "end"]:
                    value = activity.get(field)
                    if isinstance(value, str) and value:
                        transit_points.add(value)

            transports = activity.get("transports", [])
            if isinstance(transports, list):
                for transport in transports:
                    if not isinstance(transport, dict):
                        continue
                    for field in ["start", "end"]:
                        value = transport.get(field)
                        if isinstance(value, str) and value:
                            transit_points.add(value)

    return transit_points


def _collect_plan_pois_by_day(plan: Dict[str, Any]) -> Dict[int, set]:
    day_map: Dict[int, set] = {}
    itinerary = plan.get("itinerary", [])
    for day_plan in itinerary:
        day = day_plan.get("day")
        if not isinstance(day, int):
            continue
        day_map.setdefault(day, set())
        activities = day_plan.get("activities", [])
        for activity in activities:
            for field in ["position", "start", "end"]:
                value = activity.get(field)
                if value:
                    day_map[day].add(value)
            transports = activity.get("transports", [])
            if isinstance(transports, list):
                for transport in transports:
                    if not isinstance(transport, dict):
                        continue
                    for field in ["start", "end"]:
                        value = transport.get(field)
                        if value:
                            day_map[day].add(value)
    return day_map


def _normalize_temporal_constraints(temporal_raw: Any) -> List[Dict[str, Any]]:
    if isinstance(temporal_raw, dict):
        raw_items = [dict(temporal_raw)]
    elif isinstance(temporal_raw, list):
        raw_items = [dict(item) for item in temporal_raw if isinstance(item, dict)]
    else:
        raw_items = []

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        norm_item = dict(item)

        day_norm = _parse_day_token(norm_item.get("day"))
        if day_norm is not None:
            norm_item["day"] = day_norm
        norm_item["_day"] = day_norm

        must_visit = norm_item.get("must_visit")
        if isinstance(must_visit, str):
            must_visit_names = [must_visit] if must_visit else []
        elif isinstance(must_visit, list):
            must_visit_names = [name for name in must_visit if isinstance(name, str) and name]
        else:
            must_visit_names = []
        norm_item["_must_visit_names"] = must_visit_names

        end_before_min: Optional[int] = None
        for key in ("end_before", "must_end_before", "time_limit"):
            if key in norm_item:
                end_before_min = _parse_time_limit_to_minutes(norm_item.get(key))
                if end_before_min is not None:
                    break
        norm_item["_end_before_min"] = end_before_min

        max_duration_min: Optional[int] = None
        for key in ("max_duration", "duration_limit", "duration"):
            if key in norm_item:
                max_duration_min = _parse_duration_limit_to_minutes(norm_item.get(key))
                if max_duration_min is not None:
                    break
        norm_item["_max_duration_min"] = max_duration_min

        normalized.append(norm_item)

    return normalized


def _dimensions_compatible(primary_dim: Optional[str], target_dim: Optional[str]) -> bool:
    if not primary_dim or not target_dim:
        return False
    if primary_dim == target_dim:
        return True
    return target_dim in DIMENSION_COMPATIBILITY.get(primary_dim, set())


def _bucket_matches_target(
    candidate: Optional[Tuple[str, str, str]],
    target: Optional[Tuple[str, str, str]],
) -> bool:
    if candidate is None or target is None:
        return False
    return (
        candidate[0] == target[0]
        and candidate[2] == target[2]
        and _dimensions_compatible(candidate[1], target[1])
    )


def _any_location_equivalent(target_name: Optional[str], names: set) -> bool:
    if not target_name:
        return False
    return any(_locations_equivalent(target_name, name) for name in names)


def _activity_matches_anchor(activity: Dict[str, Any], anchor_poi: Optional[str]) -> bool:
    if not anchor_poi or not isinstance(activity, dict):
        return False

    for field in ["position", "start", "end"]:
        value = activity.get(field)
        if _locations_equivalent(value, anchor_poi):
            return True

    transports = activity.get("transports", [])
    if isinstance(transports, list):
        for transport in transports:
            if not isinstance(transport, dict):
                continue
            for field in ["start", "end"]:
                value = transport.get(field)
                if _locations_equivalent(value, anchor_poi):
                    return True

    return False


def _parse_time_limit_to_minutes(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return int(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    hhmm = _parse_time_to_minutes(text)
    if hhmm is not None:
        return hhmm

    if text.isdigit():
        return int(text)
    return None


def _parse_duration_limit_to_minutes(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return int(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)

    match = re.search(r"(\d+(?:\.\d+)?)\s*(小时|分钟)", text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    return int(amount * 60) if unit == "小时" else int(amount)


def _format_minutes_hhmm(minutes: Optional[int]) -> str:
    if minutes is None:
        return "unknown"
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _extract_query_days(edit_query: str) -> List[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    days = set()
    for match in re.finditer(r"第([一二三四五六七八九十\d]+)天", edit_query):
        day = _parse_day_token(match.group(1))
        if day is not None:
            days.add(day)
    return sorted(days)


def _normalize_key_fragment(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


ALLOWED_OPERATORS = {"LocalParamAdjust", "Substitute", "Reorder", "Prune"}
ALLOWED_SCOPES = {"local", "subgraph", "global"}
ALLOWED_SCOPE_ACTIONS = {"LocalRebuild", "GlobalReplan"}
ALLOWED_PRIMARY_STRATEGIES = ALLOWED_OPERATORS | ALLOWED_SCOPE_ACTIONS
ALLOWED_PRIMARY_SCOPES = {"local", "multi_day", "global"}
ALLOWED_FALLBACK_WHENS = {"operator_failed", "no_valid_anchor", "constraint_unsat"}
SCOPE_LEVEL = {"local": 1, "subgraph": 2, "global": 3}
ALLOWED_EDIT_VIOLATIONS = {"overflow", "overlap", "discontinuity", "incompatibility"}
ALLOWED_SECONDARY_VIOLATIONS = ALLOWED_EDIT_VIOLATIONS | {"infeasible", "incompatibility"}
CONFLICT_SCOPE_LEVEL = {"parameter": 0, "structural": 1, "compositional": 2}
BAD_CONFLICT_KEY_PATTERNS = {
    "scope_dimension_violation",
    "scope_resource_overflow",
    "structural_structural_overflow",
}
SOFT_PRIMARY_VIOLATIONS = {"infeasible", "incompatibility"}
PRIMARY_ALIGNMENT_OVERRIDE_BUCKETS = {
    "activity_budget_limit": {
        ("parameter", "resource", "overflow"),
    },
    "cross_day_restructure": {
        ("structural", "structural", "overlap"),
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "overlap"),
    },
    "semantic_and_specific": {
        ("structural", "temporal", "overlap"),
    },
    "category_overlap": {
        ("structural", "temporal", "overlap"),
    },
    "resource_overlap": {
        ("structural", "resource", "overflow"),
        ("structural", "resource", "overlap"),
        ("structural", "temporal", "overflow"),
    },
    "multi_city_split": {
        ("structural", "structural", "overlap"),
        ("compositional", "structural", "overlap"),
    },
    "time_window": {
        ("parameter", "structural", "discontinuity"),
        ("structural", "temporal", "overflow"),
    },
    "add_days": {
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "discontinuity"),
    },
    "ticket_price_cap_or_substitute": {
        ("structural", "resource", "overflow"),
        ("structural", "temporal", "overlap"),
    },
}
PRIMARY_INCOMPATIBILITY_ALLOWED_TYPES = {"remove_poi"}
PRIMARY_ACCEPTABLE_ALTERNATIVE_BUCKETS = {
    "cross_day_restructure": {
        ("compositional", "temporal", "overlap"),
        ("compositional", "temporal", "overflow"),
    },
    "add_days": {
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "discontinuity"),
    },
    "time_window": {
        ("parameter", "structural", "discontinuity"),
        ("structural", "temporal", "overflow"),
    },
    "resource_overlap": {
        ("structural", "resource", "overflow"),
        ("structural", "temporal", "overflow"),
    },
}


def normalize_conflict_label(label: Any) -> Optional[List[str]]:
    """标准化 conflict label 为 [scope, dimension, violation]。"""
    if not isinstance(label, (list, tuple)) or len(label) != 3:
        return None
    return [_normalize_key_fragment(part) for part in label]


def build_conflict_key(label: Any) -> Optional[str]:
    """从 conflict label 构建 conflict_key（scope_dimension_violation）。"""
    normalized = normalize_conflict_label(label)
    if not normalized:
        return None
    return "_".join(normalized)


def collect_strategy_target_from_plan(strategy_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 strategy_plan 收集主策略摘要，避免将 operator pool/fallback 压平成全集。
    """
    if not isinstance(strategy_plan, dict):
        return {}

    if strategy_plan.get("version") == "v2_primary_only":
        execution_plan = strategy_plan.get("execution_plan", [])
        exec0 = execution_plan[0] if isinstance(execution_plan, list) and execution_plan else {}
        strategy_target = strategy_plan.get("strategy_target", {})
        return {
            "primary_scope": strategy_plan.get("primary_scope"),
            "primary_strategy": strategy_plan.get("primary_strategy"),
            "primary_ops": exec0.get("operators", []) if isinstance(exec0, dict) else [],
            "primary_anchors": exec0.get("anchors", []) if isinstance(exec0, dict) else [],
            "target_conflicts": strategy_target.get("target_conflicts", []) if isinstance(strategy_target, dict) else [],
        }

    # 兼容旧版 v2_frozen
    phase_b = strategy_plan.get("phase_b_synthesis", {}) if isinstance(strategy_plan, dict) else {}
    execution_plan = phase_b.get("execution_plan", []) if isinstance(phase_b, dict) else []
    exec0 = execution_plan[0] if isinstance(execution_plan, list) and execution_plan else {}
    return {
        "primary_scope": phase_b.get("primary_scope") if isinstance(phase_b, dict) else None,
        "primary_strategy": (
            exec0.get("operators", [None])[0]
            if isinstance(exec0, dict) and isinstance(exec0.get("operators"), list) and exec0.get("operators")
            else None
        ),
        "primary_ops": exec0.get("operators", []) if isinstance(exec0, dict) else [],
        "primary_anchors": [],
        "target_conflicts": exec0.get("target_conflicts", []) if isinstance(exec0, dict) else [],
    }


def _normalize_bucket_tuple(label: Any) -> Optional[Tuple[str, str, str]]:
    normalized = normalize_conflict_label_rule(label)
    if not normalized:
        return None
    return tuple(normalized)


def _normalize_conflict_set_for_check(conflict_set: Optional[List[List[str]]]) -> List[Tuple[str, str, str]]:
    normalized: List[Tuple[str, str, str]] = []
    seen = set()
    for label in conflict_set or []:
        bucket = _normalize_bucket_tuple(label)
        if bucket and bucket not in seen:
            seen.add(bucket)
            normalized.append(bucket)
    return normalized


def check_primary_conflict_exists(
    primary_conflict: Optional[List[str]],
    conflict_set: Optional[List[List[str]]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    primary = _normalize_bucket_tuple(primary_conflict)
    normalized_set = _normalize_conflict_set_for_check(conflict_set)

    if primary is None:
        errors.append("primary_conflict missing or invalid")
        return False, errors
    if not normalized_set:
        errors.append("conflict_set missing or empty")
        return False, errors
    if primary not in normalized_set:
        errors.append(f"primary_conflict {list(primary)} not in conflict_set")
    return (len(errors) == 0, errors)


def check_primary_matches_target_bucket(
    primary_conflict: Optional[List[str]],
    conflict_set: Optional[List[List[str]]],
    target_bucket: Optional[List[str]],
    match_type_rule: Optional[str],
    purity_score_rule: Optional[float],
    constraint_type: Optional[str] = None,
    purity_threshold: float = 0.7,
) -> Tuple[bool, List[str]]:
    passed, errors, _ = evaluate_primary_target_alignment(
        primary_conflict=primary_conflict,
        conflict_set=conflict_set,
        target_bucket=target_bucket,
        match_type_rule=match_type_rule,
        purity_score_rule=purity_score_rule,
        constraint_type=constraint_type,
        purity_threshold=purity_threshold,
    )
    return passed, errors


def evaluate_primary_target_alignment(
    primary_conflict: Optional[List[str]],
    conflict_set: Optional[List[List[str]]],
    target_bucket: Optional[List[str]],
    match_type_rule: Optional[str],
    purity_score_rule: Optional[float],
    constraint_type: Optional[str] = None,
    purity_threshold: float = 0.7,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    normalized_primary = _normalize_bucket_tuple(primary_conflict)
    normalized_target = _normalize_bucket_tuple(target_bucket)
    normalized_set = set(_normalize_conflict_set_for_check(conflict_set))
    normalized_constraint_type = str(constraint_type or "").strip()
    acceptable_alternatives = PRIMARY_ACCEPTABLE_ALTERNATIVE_BUCKETS.get(normalized_constraint_type, set())
    details = {
        "target_bucket_present_in_conflict_set": False,
        "primary_matches_target_bucket": False,
        "primary_is_acceptable_alternative": False,
        "acceptable_alternative_buckets": [list(bucket) for bucket in sorted(acceptable_alternatives)],
    }

    if normalized_target is None:
        return True, errors, details

    raw_match = str(match_type_rule or "").strip().lower()
    if raw_match not in {"strong", "weak", "mismatch"}:
        errors.append(f"invalid match_type_rule: {match_type_rule}")
        return False, errors, details

    try:
        purity = float(purity_score_rule)
    except (TypeError, ValueError):
        purity = 0.0

    primary_is_target = _bucket_matches_target(normalized_primary, normalized_target)
    contains_target = any(
        _bucket_matches_target(candidate, normalized_target)
        for candidate in normalized_set
    )
    primary_is_acceptable_alternative = normalized_primary in acceptable_alternatives if normalized_primary is not None else False
    primary_soft_compatible = (
        normalized_primary is not None
        and normalized_primary[0] == normalized_target[0]
        and _dimensions_compatible(normalized_primary[1], normalized_target[1])
        and normalized_primary[2] in SOFT_PRIMARY_VIOLATIONS
        and normalized_target[2] in ALLOWED_EDIT_VIOLATIONS
    )
    override_buckets = PRIMARY_ALIGNMENT_OVERRIDE_BUCKETS.get(normalized_constraint_type, set())
    primary_allowed_override = normalized_primary in override_buckets if normalized_primary is not None else False
    details["target_bucket_present_in_conflict_set"] = contains_target
    details["primary_matches_target_bucket"] = primary_is_target
    details["primary_is_acceptable_alternative"] = primary_is_acceptable_alternative

    # 数据生成主任务池：目标 bucket 必须成为 primary（否则样本不合格）
    if normalized_target[2] in ALLOWED_EDIT_VIOLATIONS:
        if not contains_target and not primary_soft_compatible:
            errors.append(f"target_bucket {list(normalized_target)} not found in conflict_set")
            return (len(errors) == 0, errors, details)
        if primary_is_target:
            if raw_match != "strong":
                errors.append(
                    f"match_type_rule must be strong for edit pool target alignment, got: {raw_match}"
                )
            if purity < purity_threshold:
                errors.append(
                    f"purity_score_rule={purity:.3f} below threshold={purity_threshold:.3f} for edit pool"
                )
            return (len(errors) == 0, errors, details)
        if contains_target and primary_is_acceptable_alternative:
            if raw_match not in {"weak", "mismatch", "strong"}:
                errors.append(
                    f"match_type_rule invalid for acceptable alternative alignment, got: {raw_match}"
                )
            return (len(errors) == 0, errors, details)
        if primary_soft_compatible:
            # 允许 primary 以 infeasible/incompatibility 呈现同 scope+dimension 的边界情形。
            if raw_match not in {"weak", "mismatch", "strong"}:
                errors.append(
                    f"match_type_rule invalid for soft-compatible primary alignment, got: {raw_match}"
                )
            return (len(errors) == 0, errors, details)

        if contains_target and primary_allowed_override:
            if raw_match not in {"weak", "mismatch", "strong"}:
                errors.append(
                    f"match_type_rule invalid for override primary alignment, got: {raw_match}"
                )
            return (len(errors) == 0, errors, details)

        # 允许 MEC 导致的 scope 升级：同维度同 violation，但 scope 更高。
        allowed_scope_upgrade = False
        if normalized_primary is not None:
            primary_scope = CONFLICT_SCOPE_LEVEL.get(normalized_primary[0], -1)
            target_scope = CONFLICT_SCOPE_LEVEL.get(normalized_target[0], -1)
            same_dim_vio = (
                _dimensions_compatible(normalized_primary[1], normalized_target[1])
                and normalized_primary[2] == normalized_target[2]
            )
            allowed_scope_upgrade = (
                same_dim_vio
                and primary_scope > target_scope >= 0
            )

        if not allowed_scope_upgrade:
            errors.append(
                f"primary_conflict must equal target_bucket for edit pool: primary={primary_conflict}, target={target_bucket}"
            )
            return (len(errors) == 0, errors, details)

        if raw_match not in {"weak", "strong"}:
            errors.append(
                f"match_type_rule must be weak/strong for scope-upgrade alignment, got: {raw_match}"
            )
        return (len(errors) == 0, errors, details)

    if primary_is_target and purity >= purity_threshold:
        expected = "strong"
    elif contains_target or primary_is_target:
        expected = "weak"
    else:
        expected = "mismatch"

    if raw_match != expected:
        errors.append(
            f"match_type_rule={raw_match} inconsistent with primary/target relation (expected={expected})"
        )

    return (len(errors) == 0, errors, details)


def check_purity_above_threshold(
    purity_score_rule: Optional[float],
    match_type_rule: Optional[str],
    purity_threshold: float = 0.7,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        purity = float(purity_score_rule)
    except (TypeError, ValueError):
        errors.append(f"invalid purity_score_rule: {purity_score_rule}")
        return False, errors

    if purity < 0.0 or purity > 1.0:
        errors.append(f"purity_score_rule out of range [0,1]: {purity}")

    match_type = str(match_type_rule or "").strip().lower()
    if match_type == "strong" and purity < purity_threshold:
        errors.append(
            f"match_type_rule is strong but purity_score_rule={purity:.3f} < threshold={purity_threshold:.3f}"
        )

    return (len(errors) == 0, errors)


def check_no_ineligible_violation_in_pool(
    conflict_set: Optional[List[List[str]]],
    primary_conflict: Optional[List[str]] = None,
    target_bucket: Optional[List[str]] = None,
    constraint_type: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    violations: List[str] = []
    primary_bucket = _normalize_bucket_tuple(primary_conflict)
    target = _normalize_bucket_tuple(target_bucket)
    normalized_constraint_type = str(constraint_type or "").strip()
    for bucket in _normalize_conflict_set_for_check(conflict_set):
        violation = bucket[2]
        if bucket == primary_bucket and violation not in ALLOWED_EDIT_VIOLATIONS:
            if (
                normalized_constraint_type in PRIMARY_INCOMPATIBILITY_ALLOWED_TYPES
                and violation == "incompatibility"
            ):
                continue
            soft_compatible = (
                target is not None
                and bucket[0] == target[0]
                and _dimensions_compatible(bucket[1], target[1])
                and violation in SOFT_PRIMARY_VIOLATIONS
                and target[2] in ALLOWED_EDIT_VIOLATIONS
            )
            if soft_compatible:
                continue
            violations.append(
                f"primary_conflict has ineligible violation: {list(bucket)} "
                f"(allowed_primary={sorted(ALLOWED_EDIT_VIOLATIONS)})"
            )
            continue
        if violation not in ALLOWED_SECONDARY_VIOLATIONS:
            violations.append(
                f"ineligible violation in conflict_set: {list(bucket)} "
                f"(allowed_secondary={sorted(ALLOWED_SECONDARY_VIOLATIONS)})"
            )
    return (len(violations) == 0, violations)


def check_resolver_determinism(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    conflict_labels: Optional[List[List[str]]],
    conflict_set: Optional[List[List[str]]],
    target_bucket: Optional[List[str]],
    primary_conflict: Optional[List[str]],
    purity_score_rule: Optional[float],
    match_type_rule: Optional[str],
    constraint_type: Optional[str] = None,
    purity_threshold: float = 0.7,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    errors: List[str] = []
    resolver_input_labels = conflict_labels if conflict_labels else conflict_set

    recomputed = resolve_primary_conflict_rule(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query or "",
        conflict_labels_llm=resolver_input_labels or [],
        target_bucket=target_bucket,
        constraint_type=constraint_type or "",
        purity_threshold=purity_threshold,
    )

    expected_primary = _normalize_bucket_tuple(primary_conflict)
    actual_primary = _normalize_bucket_tuple(recomputed.get("primary_conflict"))
    if expected_primary != actual_primary:
        errors.append(
            f"primary_conflict not deterministic: saved={primary_conflict}, recomputed={recomputed.get('primary_conflict')}"
        )

    expected_match = str(match_type_rule or "").strip().lower()
    actual_match = str(recomputed.get("match_type_rule", "")).strip().lower()
    if expected_match != actual_match:
        errors.append(
            f"match_type_rule not deterministic: saved={expected_match}, recomputed={actual_match}"
        )

    try:
        expected_purity = float(purity_score_rule)
    except (TypeError, ValueError):
        expected_purity = -1.0
    try:
        actual_purity = float(recomputed.get("purity_score_rule"))
    except (TypeError, ValueError):
        actual_purity = -2.0

    if abs(expected_purity - actual_purity) > 1e-6:
        errors.append(
            f"purity_score_rule not deterministic: saved={expected_purity}, recomputed={actual_purity}"
        )

    return (len(errors) == 0, errors, recomputed)


def _trace_value(
    query_generation_trace: Optional[Dict[str, Any]],
    primary_key: str,
    legacy_key: Optional[str] = None,
) -> Any:
    if not isinstance(query_generation_trace, dict):
        return None
    if primary_key in query_generation_trace:
        return query_generation_trace.get(primary_key)
    if legacy_key is not None:
        return query_generation_trace.get(legacy_key)
    return None


def _trace_dict(
    query_generation_trace: Optional[Dict[str, Any]],
    primary_key: str,
    legacy_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    value = _trace_value(query_generation_trace, primary_key, legacy_key)
    return value if isinstance(value, dict) else None


def check_canonical_gate_trace(
    query_generation_trace: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(query_generation_trace, dict):
        return False, ["Missing query_generation_trace"]

    gate_pass = _trace_value(query_generation_trace, "canonical_gate_pass", "gate_pass")
    if gate_pass is not True:
        errors.append(f"query_generation_trace.canonical_gate_pass must be true, got: {gate_pass}")

    gate_reason = str(_trace_value(query_generation_trace, "canonical_gate_reason", "gate_reason") or "")
    if gate_reason == "no_strict_gate_for_constraint_type":
        errors.append("query_generation_trace.canonical_gate_reason must not be no_strict_gate_for_constraint_type")

    proof_ok = _trace_value(query_generation_trace, "canonical_proof_ok", "proof_ok")
    if proof_ok is not True:
        errors.append(f"query_generation_trace.canonical_proof_ok must be true, got: {proof_ok}")

    proof_reason_code = str(
        _trace_value(query_generation_trace, "canonical_proof_reason_code", "proof_reason_code") or ""
    )
    if not proof_reason_code:
        errors.append("query_generation_trace.canonical_proof_reason_code missing")

    return (len(errors) == 0, errors)


def check_query_factual_validity(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    constraint_type: Optional[str],
    query_generation_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback_params = _build_query_fallback_params(query_generation_trace)
    template_str: Optional[str] = None
    if isinstance(query_generation_trace, dict):
        raw_template = query_generation_trace.get("template")
        if isinstance(raw_template, str) and raw_template.strip():
            template_str = raw_template

    assessment = assess_edit_query_factual_validity(
        origin_plan=origin_plan,
        constraints=constraints,
        constraint_type=constraint_type,
        edit_query=edit_query or "",
        fallback_params=fallback_params,
        template_str=template_str,
        evidence=build_origin_plan_evidence(origin_plan),
    )
    is_valid = bool(assessment.get("is_valid"))
    errors = [] if is_valid else [str(assessment.get("reason") or "Unknown factual validity failure")]
    subtype = assessment.get("subtype")
    return {
        "pass": is_valid,
        "label": assessment.get("label"),
        "reason_code": assessment.get("reason_code"),
        "reason": assessment.get("reason"),
        "subtype": subtype,
        "invalidity_category": subtype,
        "plan_relation": assessment.get("plan_relation"),
        "errors": errors,
        "evidence_used": assessment.get("evidence_used", {}),
    }


def check_surface_proof_recomputed(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    constraint_type: Optional[str],
    query_generation_trace: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    if not constraint_type:
        return False, ["Missing constraint_type for proof recompute"], None

    fallback_params = _build_proof_fallback_params(query_generation_trace)
    recomputed = validate_conflict_trigger_proof(
        origin_plan=origin_plan,
        constraints=constraints,
        constraint_type=str(constraint_type),
        edit_query=edit_query or "",
        fallback_params=fallback_params,
        evidence=build_origin_plan_evidence(origin_plan),
    )

    if not recomputed.ok:
        errors.append(f"proof_mismatch: recomputed proof failed ({recomputed.reason})")

    expected_reason_code = ""
    if isinstance(query_generation_trace, dict):
        expected_reason_code = str(
            _trace_value(query_generation_trace, "surface_proof_reason_code", "proof_reason_code") or ""
        )
    if expected_reason_code and expected_reason_code != recomputed.reason_code:
        errors.append(
            f"proof_reason_code mismatch: saved={expected_reason_code}, recomputed={recomputed.reason_code}"
        )

    return (
        len(errors) == 0,
        errors,
        {
            "ok": recomputed.ok,
            "reason_code": recomputed.reason_code,
            "reason": recomputed.reason,
            "evidence_used": recomputed.evidence_used,
        },
    )


def check_hard_truth_rebuild(
    origin_plan: Dict[str, Any],
    origin_query_structured: Optional[Dict[str, Any]],
    edit_query: str,
    edit_target_constraints: Optional[List[Dict[str, Any]]],
    *,
    canonical_constraint_ir: Optional[Dict[str, Any]] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    constraint_type: Optional[str] = None,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    if not isinstance(edit_target_constraints, list) or not edit_target_constraints:
        return False, ["Missing edit_target_constraints"], None
    if not isinstance(canonical_constraint_ir, dict) or not canonical_constraint_ir:
        return False, ["Missing canonical_constraint_ir for hard truth rebuild"], None

    try:
        rebuilt = build_edit_targets_from_ir(canonical_constraint_ir)["edit_target_constraints"]
    except Exception as exc:
        return False, [f"hard truth rebuild failed: {exc}"], None

    def _freeze(items: List[Dict[str, Any]]) -> List[str]:
        return sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items)

    expected = _freeze(edit_target_constraints)
    actual = _freeze(rebuilt)
    if expected != actual:
        errors.append("reconstructed hard truth does not match edit_target_constraints")
        errors.append(f"expected={expected}")
        errors.append(f"actual={actual}")

    return (
        len(errors) == 0,
        errors,
        {
            "mode": "canonical_constraint_ir",
            "expected": edit_target_constraints,
            "actual": rebuilt,
        },
    )


def check_effective_constraint_merge(
    origin_logical_constraints: Optional[List[Dict[str, Any]]],
    edit_target_constraints: Optional[List[Dict[str, Any]]],
    effective_logical_constraints: Optional[List[Dict[str, Any]]],
    effective_constraint_merge_trace: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    if not isinstance(origin_logical_constraints, list) or not origin_logical_constraints:
        return False, ["Missing origin_logical_constraints"], None
    if not isinstance(edit_target_constraints, list) or not edit_target_constraints:
        return False, ["Missing edit_target_constraints"], None
    if not isinstance(effective_logical_constraints, list) or not effective_logical_constraints:
        return False, ["Missing effective_logical_constraints"], None

    rebuilt, trace = build_effective_logical_constraints(
        origin_logical_constraints,
        edit_target_constraints,
    )
    expected = sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in effective_logical_constraints)
    actual = sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in rebuilt)
    errors: List[str] = []
    if expected != actual:
        errors.append("effective_logical_constraints does not match origin/edit merge policy")
        errors.append(f"expected={expected}")
        errors.append(f"actual={actual}")

    saved_trace = effective_constraint_merge_trace if isinstance(effective_constraint_merge_trace, dict) else {}
    for key in (
        "merge_policy_version",
        "origin_constraint_count",
        "edit_constraint_count",
        "effective_constraint_count",
        "overridden_origin_constraint_ids",
    ):
        if saved_trace.get(key) != trace.get(key):
            errors.append(f"effective_constraint_merge_trace.{key} mismatch: saved={saved_trace.get(key)!r}, rebuilt={trace.get(key)!r}")

    return (
        len(errors) == 0,
        errors,
        {
            "merge_policy_version": MERGE_POLICY_VERSION,
            "expected": effective_logical_constraints,
            "actual": rebuilt,
            "trace": trace,
        },
    )


def check_constraint_ir_presence(
    canonical_constraint_ir: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    if isinstance(canonical_constraint_ir, dict) and canonical_constraint_ir:
        return True, []
    return False, ["Missing canonical_constraint_ir"]


def check_constraint_ir_schema(
    canonical_constraint_ir: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    return validate_canonical_constraint_ir(canonical_constraint_ir)


def check_template_ir_alignment(
    canonical_constraint_ir: Optional[Dict[str, Any]],
    *,
    constraint_type: Optional[str],
    template_str: Optional[str],
) -> Tuple[bool, List[str]]:
    return check_template_ir_consistency(
        canonical_constraint_ir,
        constraint_type=constraint_type,
        template_str=template_str,
    )


def check_ir_truth_compile(
    canonical_constraint_ir: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    return check_ir_compiles(canonical_constraint_ir)


def check_soft_tag_presence(
    query_generation_trace: Optional[Dict[str, Any]],
    edit_target_preference_tags: Optional[List[Dict[str, Any]]],
    edit_target_preferences: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[str]]:
    if _edit_soft_target_suppressed(query_generation_trace):
        return True, []
    family, facet, mode, _ = _soft_target_from_trace(
        query_generation_trace,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    if not any([family, facet, mode]):
        return True, []
    if isinstance(edit_target_preference_tags, list) and edit_target_preference_tags:
        return True, []
    return False, ["soft target trace present but edit_target_preference_tags missing or empty"]


def _edit_soft_target_suppressed(query_generation_trace: Optional[Dict[str, Any]]) -> bool:
    return isinstance(query_generation_trace, dict) and query_generation_trace.get("edit_soft_target_suppressed") is True


def _soft_target_from_trace(
    query_generation_trace: Optional[Dict[str, Any]],
    edit_target_preference_tags: Optional[List[Dict[str, Any]]] = None,
    edit_target_preferences: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str, str, str]:
    if not isinstance(query_generation_trace, dict):
        query_generation_trace = {}
    if _edit_soft_target_suppressed(query_generation_trace):
        return ("", "", "", "")

    family = str(query_generation_trace.get("soft_target_family", "") or "").strip()
    facet = str(query_generation_trace.get("soft_target_facet", "") or "").strip()
    mode = str(query_generation_trace.get("soft_target_mode", "") or "").strip()
    intensity = str(query_generation_trace.get("soft_target_intensity", "") or "").strip()

    if not intensity and isinstance(edit_target_preference_tags, list) and edit_target_preference_tags:
        for item in edit_target_preference_tags:
            if not isinstance(item, dict):
                continue
            intensity = str(item.get("intensity", "") or "").strip()
            if intensity:
                if not family:
                    family = str(item.get("family", "") or "").strip()
                if not facet:
                    facet = str(item.get("facet", "") or "").strip()
                if not mode:
                    mode = str(item.get("edit_mode", "") or "").strip()
                break

    if not intensity and isinstance(edit_target_preferences, list) and edit_target_preferences:
        for item in edit_target_preferences:
            if not isinstance(item, dict):
                continue
            params = item.get("params", {})
            if not isinstance(params, dict):
                continue
            intensity = str(params.get("intensity", "") or "").strip()
            if intensity:
                break

    return (family, facet, mode, intensity)


def check_soft_query_alignment(
    edit_query: str,
    query_generation_trace: Optional[Dict[str, Any]],
    origin_soft_profile: Optional[List[Dict[str, Any]]],
    edit_target_preference_tags: Optional[List[Dict[str, Any]]] = None,
    edit_target_preferences: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[str]]:
    family, facet, mode, intensity = _soft_target_from_trace(
        query_generation_trace,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    if not any([family, facet, mode]):
        return True, []
    constraint_type = str((query_generation_trace or {}).get("constraint_type", "") or "").strip()
    semantic_slots = _trace_dict(query_generation_trace, "surface_semantic_slots")
    if semantic_slots is None:
        semantic_slots = _trace_dict(query_generation_trace, "canonical_semantic_slots", "semantic_slots")
    query_spec = _trace_dict(query_generation_trace, "surface_query_spec")
    if query_spec is None:
        query_spec = _trace_dict(query_generation_trace, "canonical_query_spec", "query_spec")
    if (
        constraint_type == "travel_time_constraint"
        and family == "route_compactness"
        and facet == "travel_time"
        and is_explicit_hard_travel_time_query(
            edit_query,
            semantic_slots=semantic_slots if isinstance(semantic_slots, dict) else None,
            query_spec=query_spec if isinstance(query_spec, dict) else None,
        )
    ):
        return True, []
    errors = validate_soft_query_alignment(
        edit_query=edit_query,
        origin_profile=origin_soft_profile or [],
        target_family=family,
        target_facet=facet,
        target_mode=mode,
        template_soft_intensity=intensity,
        constraint_type=constraint_type,
        semantic_slots=semantic_slots if isinstance(semantic_slots, dict) else None,
        query_spec=query_spec if isinstance(query_spec, dict) else None,
    )
    return (len(errors) == 0, errors)


def check_soft_mapping_supported(
    edit_query: str,
    query_generation_trace: Optional[Dict[str, Any]],
    edit_target_preference_tags: Optional[List[Dict[str, Any]]] = None,
    edit_target_preferences: Optional[List[Dict[str, Any]]] = None,
    canonical_constraint_ir: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], Optional[List[Dict[str, Any]]]]:
    family, facet, mode, intensity = _soft_target_from_trace(
        query_generation_trace,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    if not any([family, facet, mode]):
        return True, [], []
    if not isinstance(canonical_constraint_ir, dict) or not canonical_constraint_ir:
        return False, ["Missing canonical_constraint_ir for soft preference mapping"], None

    try:
        preferences = build_edit_targets_from_ir(canonical_constraint_ir)["edit_target_preferences"]
    except Exception as exc:
        return False, [f"soft preference mapping failed: {exc}"], None

    if not preferences:
        return False, ["soft preference mapping produced no edit_target_preferences"], preferences
    return True, [], preferences


def check_soft_preference_rebuild_match(
    edit_query: str,
    query_generation_trace: Optional[Dict[str, Any]],
    edit_target_preference_tags: Optional[List[Dict[str, Any]]],
    edit_target_preferences: Optional[List[Dict[str, Any]]],
    canonical_constraint_ir: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str], Optional[List[Dict[str, Any]]]]:
    is_valid, errors, rebuilt = check_soft_mapping_supported(
        edit_query=edit_query,
        query_generation_trace=query_generation_trace,
        canonical_constraint_ir=canonical_constraint_ir,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    if not is_valid:
        return is_valid, errors, rebuilt

    if rebuilt == [] and not edit_target_preferences:
        return True, [], rebuilt
    if not isinstance(edit_target_preferences, list):
        return False, ["edit_target_preferences missing or invalid"], rebuilt

    expected = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in edit_target_preferences]
    actual = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in (rebuilt or [])]
    compare_expected = sorted(expected)
    compare_actual = sorted(actual)
    if compare_expected != compare_actual:
        return False, [
            "reconstructed soft preferences do not match edit_target_preferences",
            f"expected={compare_expected}",
            f"actual={compare_actual}",
        ], rebuilt
    return True, [], rebuilt


def check_surface_ir_equivalence(
    canonical_constraint_ir: Optional[Dict[str, Any]],
    query_generation_trace: Optional[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]],
    origin_plan: Optional[Dict[str, Any]],
    origin_query_structured: Optional[Dict[str, Any]],
    edit_query: str,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    if not isinstance(canonical_constraint_ir, dict) or not canonical_constraint_ir:
        return False, ["Missing canonical_constraint_ir for surface equivalence"], None
    trace = query_generation_trace if isinstance(query_generation_trace, dict) else {}
    surface_query_spec = trace.get("surface_query_spec") if isinstance(trace.get("surface_query_spec"), dict) else None
    surface_semantic_slots = trace.get("surface_semantic_slots") if isinstance(trace.get("surface_semantic_slots"), dict) else None
    if surface_query_spec is None and surface_semantic_slots is None:
        return True, [], {"skipped": True, "reason": "surface_structured_state_missing"}
    try:
        surface_ir = build_canonical_constraint_ir_from_structured_inputs(
            template_text=str(trace.get("template", "") or ""),
            constraint_type=str(trace.get("constraint_type", "") or ""),
            instantiated_slots={
                "query_spec": surface_query_spec or {},
                "semantic_slots": surface_semantic_slots or {},
                "constraints": constraints or {},
                "origin_plan": origin_plan or {},
                "origin_query_structured": origin_query_structured or {},
                "surface_query": edit_query,
            },
            template_semantic_spec=(
                trace.get("template_semantic_spec")
                if isinstance(trace.get("template_semantic_spec"), dict)
                else {}
            ),
            meta={
                "template_id": str(trace.get("template_id", "") or ""),
                "query_generation_trace": trace,
            },
        )
    except Exception as exc:
        return False, [f"surface IR equivalence failed: {exc}"], None

    canonical_bundle = build_edit_targets_from_ir(canonical_constraint_ir)
    surface_bundle = build_edit_targets_from_ir(surface_ir)
    expected = sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in canonical_bundle["edit_target_constraints"])
    actual = sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in surface_bundle["edit_target_constraints"])
    if expected != actual:
        return False, ["surface IR does not preserve frozen hard truth", f"expected={expected}", f"actual={actual}"], {
            "expected": canonical_bundle["edit_target_constraints"],
            "actual": surface_bundle["edit_target_constraints"],
        }
    return True, [], {"expected": canonical_bundle["edit_target_constraints"], "actual": surface_bundle["edit_target_constraints"]}


def check_canonical_alignment_trace(
    query_generation_trace: Optional[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(query_generation_trace, dict):
        return False, ["Missing query_generation_trace"]

    alignment_pass = _trace_value(query_generation_trace, "canonical_alignment_pass", "alignment_pass")
    if alignment_pass is not True:
        errors.append(f"query_generation_trace.canonical_alignment_pass must be true, got: {alignment_pass}")

    alignment_errors = _trace_value(query_generation_trace, "canonical_alignment_errors", "alignment_errors")
    if alignment_errors is None:
        alignment_errors = []
    if isinstance(alignment_errors, list):
        if alignment_errors:
            errors.append(f"query_generation_trace.canonical_alignment_errors must be empty, got: {alignment_errors}")
    else:
        errors.append("query_generation_trace.canonical_alignment_errors must be a list")

    query_spec = _trace_dict(query_generation_trace, "canonical_query_spec", "query_spec")
    if not isinstance(query_spec, dict) or not query_spec:
        errors.append("query_generation_trace.canonical_query_spec missing or invalid")

    return (len(errors) == 0, errors)


def _build_proof_fallback_params(query_generation_trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(query_generation_trace, dict):
        return {}
    fallback_params: Dict[str, Any] = {}
    rendered_params = query_generation_trace.get("rendered_params")
    if isinstance(rendered_params, dict):
        fallback_params.update(rendered_params)
    query_spec = _trace_dict(query_generation_trace, "surface_query_spec")
    if query_spec is None:
        query_spec = _trace_dict(query_generation_trace, "canonical_query_spec", "query_spec")
    if isinstance(query_spec, dict):
        for key, value in query_spec.items():
            fallback_params.setdefault(key, value)
        if "source_poi" in query_spec:
            fallback_params.setdefault("poi_name", query_spec.get("source_poi"))
        if "anchor_poi" in query_spec:
            fallback_params.setdefault("poi_name", query_spec.get("anchor_poi"))
    semantic_slots = _trace_dict(query_generation_trace, "surface_semantic_slots")
    if semantic_slots is None:
        semantic_slots = _trace_dict(query_generation_trace, "canonical_semantic_slots", "semantic_slots")
    if isinstance(semantic_slots, dict):
        for key, value in semantic_slots.items():
            if value not in (None, "", []):
                fallback_params.setdefault(key, value)
        if semantic_slots.get("source_poi"):
            fallback_params.setdefault("poi_name", semantic_slots.get("source_poi"))
        if semantic_slots.get("anchor_poi"):
            fallback_params.setdefault("poi_name", semantic_slots.get("anchor_poi"))
    return fallback_params


def _build_query_fallback_params(query_generation_trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _build_proof_fallback_params(query_generation_trace)


def calculate_distance(poi_db: POIDatabase, city: str, poi1: str, poi2: str) -> Optional[float]:
    """
    计算两个POI之间的距离（公里）
    使用geodesic方法（与Chinatravel项目一致）

    Args:
        poi_db: POI数据库
        city: 城市名称
        poi1: POI 1名称
        poi2: POI 2名称

    Returns:
        距离（公里），如果无法计算则返回None
    """
    try:
        # 获取POI坐标
        coord1 = poi_db.search_poi(city, poi1)
        coord2 = poi_db.search_poi(city, poi2)

        if not coord1 or not coord2:
            return None

        if GEODESIC_AVAILABLE:
            # 使用geodesic计算距离（与Chinatravel项目一致）
            distance = geodesic(coord1, coord2).kilometers
            return distance
        else:
            # Fallback: 使用简化的平面距离计算
            lat1, lon1 = coord1
            lat2, lon2 = coord2
            import math
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return 6371 * c
    except Exception as e:
        return None


def _extract_budget_limit_from_resource(resource: Dict[str, Any]) -> Optional[float]:
    if not isinstance(resource, dict):
        return None
    if str(resource.get("budget_scope") or "").strip() == "anchor_bundle":
        return None
    for key in ("max_cost", "max_budget", "budget_cap", "budget_target", "budget"):
        value = resource.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            numeric = re.search(r"(\d+(?:\.\d+)?)", value)
            if numeric:
                return float(numeric.group(1))
    return None


def _extract_ticket_budget_limit_from_resource(resource: Dict[str, Any]) -> Optional[float]:
    if not isinstance(resource, dict):
        return None
    for key in ("ticket_budget_limit", "ticket_budget"):
        value = resource.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            numeric = re.search(r"(\d+(?:\.\d+)?)", value)
            if numeric:
                return float(numeric.group(1))
    return None


def _extract_ticket_price_cap_from_resource(resource: Dict[str, Any]) -> Optional[float]:
    if not isinstance(resource, dict):
        return None
    value = resource.get("ticket_price_cap")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        numeric = re.search(r"(\d+(?:\.\d+)?)", value)
        if numeric:
            return float(numeric.group(1))
    return None


def _calculate_total_plan_cost(plan: Dict[str, Any]) -> float:
    total_cost = 0.0
    itinerary = plan.get("itinerary", [])
    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        for activity in activities:
            cost = activity.get("cost", 0)
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
    return total_cost


def _calculate_plan_ticket_cost(plan: Dict[str, Any]) -> float:
    ticket_cost = 0.0
    itinerary = plan.get("itinerary", [])
    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        for activity in activities:
            if activity.get("type") != "attraction":
                continue
            cost = activity.get("cost", 0)
            if isinstance(cost, (int, float)):
                ticket_cost += float(cost)
    return ticket_cost


def validate_poi_existence(constraints: Dict[str, Any], poi_db: POIDatabase) -> Tuple[bool, List[str]]:
    """
    验证constraints中的所有POI在数据库中存在

    Args:
        constraints: 约束字典，包含must_include等字段
        poi_db: POI数据库实例

    Returns:
        (is_valid, missing_pois) 元组
    """
    missing_pois = []

    # 检查must_include中的POI
    must_include = constraints.get("must_include", [])
    for poi in must_include:
        if isinstance(poi, dict):
            city = poi.get("city")
            name = poi.get("name")

            if not city or not name:
                missing_pois.append(f"Invalid POI entry: {poi}")
                continue

            # 检查景点
            if poi.get("type") == "attraction":
                if _looks_like_transport_hub(name):
                    missing_pois.append(f"Transport hub cannot be used as must_include attraction: {city} - {name}")
                    continue
                if not poi_db.attraction_exists(city, name):
                    missing_pois.append(f"Attraction not found: {city} - {name}")
            elif poi.get("type") == "restaurant":
                if poi_db.get_restaurant_by_name(city, name) is None:
                    missing_pois.append(f"Restaurant not found: {city} - {name}")
            elif poi.get("type") == "accommodation":
                if poi_db.get_accommodation_by_name(city, name) is None:
                    missing_pois.append(f"Accommodation not found: {city} - {name}")

    return (len(missing_pois) == 0, missing_pois)


def _looks_like_transport_hub(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    if any(token in text for token in ("轻轨", "地铁", "文化站")):
        return False
    if text.endswith("机场"):
        return True
    return text.endswith("站") and len(text) <= 6


def validate_city_consistency(
    constraints: Dict[str, Any],
    target_city: str,
    edit_query: str = "",
    start_city: Optional[str] = None,
    origin_query_text: str = "",
    origin_query_structured: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    验证城市一致性 - query不应该提出更换城市

    Args:
        constraints: 约束字典
        target_city: origin plan的目标城市

    Returns:
        (is_valid, errors) 元组
    """
    errors = []

    # 检查spatial约束中的城市
    spatial = constraints.get("spatial", {})
    if spatial:
        constraint_city = spatial.get("city")
        if constraint_city and constraint_city != target_city:
            errors.append(f"City constraint mismatch: query asks for {constraint_city}, but target_city is {target_city}")

    # 检查must_include中的POI城市
    must_include = constraints.get("must_include", [])
    for poi in must_include:
        if isinstance(poi, dict):
            poi_city = poi.get("city")
            if poi_city and poi_city != target_city:
                errors.append(f"POI city mismatch: {poi.get('name')} is in {poi_city}, but target_city is {target_city}")

    if isinstance(edit_query, str) and edit_query.strip():
        allowed_cities = {city for city in [target_city, start_city] if isinstance(city, str) and city}
        mentioned_cities = _extract_query_city_mentions(edit_query)
        unexpected_cities = [city for city in mentioned_cities if city not in allowed_cities]
        if unexpected_cities:
            errors.append(
                f"Edit query mentions unexpected cities: {unexpected_cities}, allowed={sorted(allowed_cities)}"
            )

    if isinstance(origin_query_text, str) and origin_query_text.strip():
        allowed_cities = {city for city in [target_city, start_city] if isinstance(city, str) and city}
        mentioned_cities = _extract_query_city_mentions(origin_query_text)
        unexpected_cities = [city for city in mentioned_cities if city not in allowed_cities]
        if unexpected_cities:
            errors.append(
                f"Origin query mentions unexpected cities: {unexpected_cities}, allowed={sorted(allowed_cities)}"
            )

    if isinstance(origin_query_structured, dict):
        structured_target = origin_query_structured.get("target_city")
        structured_start = origin_query_structured.get("start_city")
        if structured_target and target_city and structured_target != target_city:
            errors.append(
                f"Origin query target_city mismatch: {structured_target} != {target_city}"
            )
        if structured_start and start_city and structured_start != start_city:
            errors.append(
                f"Origin query start_city mismatch: {structured_start} != {start_city}"
            )

    return (len(errors) == 0, errors)


def _extract_replace_source_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    for pattern in [r"把(.+?)改成", r"把(.+?)换成", r"将(.+?)替换为"]:
        match = re.search(pattern, edit_query)
        if match:
            name = str(match.group(1)).strip()
            if name:
                return name
    return None


def _extract_specific_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"特别是([^，。；;]+)",
        r"我想去(.+?)这样的",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        name = str(match.group(1)).strip().strip("，。；;,. ")
        if name:
            return name
    return None


def _extract_preferred_poi_type_from_query(edit_query: str) -> Optional[str]:
    return shared_extract_preferred_poi_type_from_query(edit_query)


def _extract_semantic_target_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"(?:改成|换成|替换为)(.+?)类型",
        r"(?:改成|换成)(.+?)的其他景点",
        r"同类型的(.+?)(?:景点|地点)",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        target = str(match.group(1)).strip().strip("，。；;,. ")
        if target:
            return target
    return None


def _extract_day_poi_pairs_from_query(edit_query: str) -> List[Tuple[int, str]]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    pairs: List[Tuple[int, str]] = []
    pattern = re.compile(r"第([一二三四五六七八九十\d]+)天(?:去|游玩|参观|安排|打卡)([^，。；;]+)")
    for match in pattern.finditer(edit_query):
        day_token = match.group(1)
        poi_name = str(match.group(2)).strip()
        day = _parse_day_token(day_token)
        if day and poi_name:
            pairs.append((day, poi_name))
    return pairs


def _parse_day_token(day_token: Any) -> Optional[int]:
    if isinstance(day_token, int) and day_token > 0:
        return day_token
    if not isinstance(day_token, str):
        return None
    day_token = day_token.strip()
    if not day_token:
        return None
    if day_token.isdigit():
        return int(day_token)
    cn_map = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    if day_token in cn_map:
        return cn_map[day_token]
    m = re.search(r"第([一二三四五六七八九十\d]+)天", day_token)
    if m:
        return _parse_day_token(m.group(1))
    return None


def validate_query_constraint_alignment(
    edit_query: str,
    constraints: Dict[str, Any],
    constraint_type: Optional[str] = None,
    template_str: Optional[str] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    canonical_edit_query: str = "",
    edit_target_constraints: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[str]]:
    """
    检查 query 与 constraints 的结构语义是否一致，防止隐式约束注入。
    """
    errors: List[str] = []
    ctype = str(constraint_type or "").strip().lower()
    template_text = str(template_str or "")
    semantic_slots = resolve_query_semantic_slots(
        edit_query,
        ctype,
        query_generation_trace=query_generation_trace,
        template_str=template_str,
    )

    must_include = constraints.get("must_include", [])
    must_include_names = []
    if isinstance(must_include, list):
        for poi in must_include:
            if isinstance(poi, dict) and isinstance(poi.get("name"), str) and poi.get("name"):
                must_include_names.append(poi.get("name"))

    if (
        template_text
        and not any(token in template_text for token in ["{poi}", "{poi1}", "{poi2}"])
        and must_include_names
    ):
        errors.append("template has no poi placeholder but constraints.must_include is not empty")

    if ctype == "semantic_substitution":
        source_poi = semantic_slots.get("source_poi") or _extract_replace_source_poi_from_query(edit_query)
        semantic_target = semantic_slots.get("semantic_target") or _extract_semantic_target_from_query(edit_query)
        forbidden = constraints.get("forbidden", [])
        forbidden_set = {name for name in forbidden if isinstance(name, str) and name}
        semantic = constraints.get("semantic", {})

        if must_include_names:
            errors.append("semantic_substitution must_include must be empty")
        if source_poi and source_poi in set(must_include_names):
            errors.append(f"semantic_substitution source poi '{source_poi}' must not be in must_include")
        if source_poi and source_poi not in forbidden_set:
            errors.append(f"semantic_substitution source poi '{source_poi}' missing in forbidden")
        if not isinstance(semantic, dict):
            errors.append("semantic_substitution missing semantic schema")
        else:
            replace_from = semantic.get("replace_from")
            if source_poi and replace_from != source_poi:
                errors.append(f"semantic_substitution replace_from mismatch: {replace_from} != {source_poi}")
            target_value = semantic.get("semantic_target")
            if not isinstance(target_value, str) or not target_value.strip():
                errors.append("semantic_substitution missing semantic_target")
            elif semantic_target and target_value.strip() != semantic_target:
                errors.append(
                    f"semantic_substitution semantic_target mismatch: {target_value.strip()} != {semantic_target}"
                )

    if ctype in {"semantic_and_specific", "category_overlap"}:
        specific_poi = semantic_slots.get("specific_poi") or _extract_specific_poi_from_query(edit_query)
        preferred_type = semantic_slots.get("preferred_type") or _extract_preferred_poi_type_from_query(edit_query)
        semantic = constraints.get("semantic", {})

        if specific_poi and specific_poi not in set(must_include_names):
            errors.append(f"{ctype} missing specific poi in must_include: {specific_poi}")

        if preferred_type:
            if not isinstance(semantic, dict):
                errors.append(f"{ctype} missing semantic schema")
            else:
                current = semantic.get("preferred_type")
                expected_preferred_type = normalize_preferred_poi_type(preferred_type) or preferred_type
                actual_preferred_type = normalize_preferred_poi_type(current) if isinstance(current, str) else None
                if not isinstance(current, str) or not current.strip():
                    errors.append(f"{ctype} missing semantic.preferred_type")
                elif actual_preferred_type != expected_preferred_type:
                    errors.append(
                        f"{ctype} preferred_type mismatch: {current.strip()} != {preferred_type}"
                    )
        if ctype == "category_overlap" and semantic_slots.get("additional_similar") is True:
            if not isinstance(semantic, dict) or semantic.get("additional_similar") is not True:
                errors.append("category_overlap missing semantic.additional_similar=true")

    if ctype == "resource_overlap":
        anchor_poi = semantic_slots.get("anchor_poi")
        semantic = constraints.get("semantic", {})
        if anchor_poi and anchor_poi not in set(must_include_names):
            errors.append(f"resource_overlap missing anchor poi in must_include: {anchor_poi}")
        restaurant_name = semantic_slots.get("restaurant_name")
        if restaurant_name and restaurant_name not in set(must_include_names):
            errors.append(f"resource_overlap missing explicit restaurant in must_include: {restaurant_name}")
        resource = constraints.get("resource", {})
        has_generic_nearby_meal_signal = (
            semantic_slots.get("nearby_meal") is True
            or (isinstance(resource, dict) and resource.get("nearby_meal") is True)
            or ("餐厅" in str(edit_query or "") and any(token in str(edit_query or "") for token in ("附近", "周边")))
        )
        if has_generic_nearby_meal_signal:
            if not isinstance(resource, dict) or resource.get("nearby_meal") is not True:
                errors.append("resource_overlap missing resource.nearby_meal=true")
            if not restaurant_name:
                restaurant_items = [
                    poi.get("name")
                    for poi in must_include
                    if isinstance(poi, dict) and poi.get("type") == "restaurant" and poi.get("name")
                ]
                if restaurant_items:
                    errors.append(
                        f"generic resource_overlap nearby meal must not require explicit restaurants: {restaurant_items}"
                    )
                if isinstance(resource, dict) and resource.get("selected_meal_name"):
                    errors.append("generic resource_overlap nearby meal must not set resource.selected_meal_name")
                if isinstance(edit_target_constraints, list):
                    for item in edit_target_constraints:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "required_restaurant_name":
                            errors.append("generic resource_overlap nearby meal must not emit required_restaurant_name")
                        params = item.get("params") if isinstance(item.get("params"), dict) else {}
                        if params.get("selected_meal_name"):
                            errors.append("generic resource_overlap nearby meal must not set params.selected_meal_name")

        if isinstance(canonical_edit_query, str) and canonical_edit_query.strip():
            canonical_slots = resolve_query_semantic_slots(
                canonical_edit_query,
                ctype,
                query_generation_trace=None,
                template_str=template_str,
            )
            for key in ("anchor_poi", "budget_cap"):
                expected = canonical_slots.get(key)
                actual = semantic_slots.get(key)
                if expected not in (None, "", []) and actual not in (None, "", []) and expected != actual:
                    errors.append(f"canonical_edit_query {key} mismatch: {expected} != {actual}")

    if ctype == "multi_city_split":
        semantic = constraints.get("semantic", {})
        expected_targets = list(semantic_slots.get("city_split_targets") or [])
        actual_targets = list(semantic.get("city_split_targets", []) or []) if isinstance(semantic, dict) else []
        if expected_targets and actual_targets[: len(expected_targets)] != expected_targets:
            errors.append(f"multi_city_split semantic targets mismatch: {actual_targets} != {expected_targets}")

    query_days = _extract_query_days(edit_query)
    temporal = _normalize_temporal_constraints(constraints.get("temporal", []))
    if query_days:
        query_day_set = set(query_days)
        has_day_bound_time_limit = any(item.get("_end_before_min") is not None for item in temporal)
        if has_day_bound_time_limit:
            temporal_days = {item.get("_day") for item in temporal if item.get("_day") is not None}
            if not temporal_days:
                errors.append("temporal time_limit missing day binding for day-anchored query")
            elif temporal_days.isdisjoint(query_day_set):
                errors.append(
                    f"temporal day mismatch: query_days={sorted(query_day_set)}, constraint_days={sorted(temporal_days)}"
                )

    if ctype == "cross_day_restructure":
        day_pairs = _extract_day_poi_pairs_from_query(edit_query)
        if day_pairs:
            day_map: Dict[int, set] = {}
            for item in temporal:
                day = _parse_day_token(item.get("day"))
                must_visit = item.get("must_visit")
                if day is None or must_visit is None:
                    continue
                if isinstance(must_visit, str):
                    names = {must_visit} if must_visit else set()
                elif isinstance(must_visit, list):
                    names = {name for name in must_visit if isinstance(name, str) and name}
                else:
                    names = set()
                if names:
                    day_map.setdefault(day, set()).update(names)

            for day, poi_name in day_pairs:
                if poi_name not in day_map.get(day, set()):
                    errors.append(f"cross_day_restructure missing day-poi binding: day={day}, poi={poi_name}")

    return (len(errors) == 0, errors)


def validate_spatial_constraints(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    poi_db: POIDatabase,
    constraint_type: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    验证空间约束 - 检查距离/成本约束是否被违反
    使用plan中的实际cost和geodesic距离计算

    Args:
        origin_plan: 原始plan
        constraints: 约束条件
        poi_db: POI数据库

    Returns:
        (is_valid, violations) 元组
    """
    violations = []

    # 1. 检查max_walking_distance约束
    spatial = constraints.get("spatial", {})
    max_walking_distance = spatial.get("max_walking_distance")

    if max_walking_distance:
        target_city = origin_plan.get("target_city")
        itinerary = origin_plan.get("itinerary", [])

        # 检查所有相邻activities之间的距离
        for day_plan in itinerary:
            activities = day_plan.get("activities", [])

            for i in range(len(activities) - 1):
                act1 = activities[i]
                act2 = activities[i + 1]

                # 只检查同一城市内的POI（跳过train等）
                if act1.get("type") == "train" or act2.get("type") == "train":
                    continue

                poi1 = act1.get("end", act1.get("start"))
                poi2 = act2.get("start", act2.get("end"))

                if not poi1 or not poi2:
                    continue

                # 使用geodesic计算距离（与Chinatravel一致）
                distance = calculate_distance(poi_db, target_city, poi1, poi2)

                if distance is not None and distance > max_walking_distance:
                    violations.append(f"Walking distance constraint violated: {poi1} -> {poi2} is {distance:.2f}km, max is {max_walking_distance}km")

    # 2. 预算类约束由专门的 resource/budget 校验负责；这里仅保留空间距离校验，
    # 避免把 scoped budget 错记到 spatial_constraints。
    _ = str(constraint_type or "").strip()

    return (len(violations) == 0, violations)


def validate_budget_constraints(
    plan: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate total-budget and ticket-budget style resource limits."""
    violations: List[str] = []
    resource = constraints.get("resource", {})

    budget_limit = _extract_budget_limit_from_resource(resource)
    if budget_limit is not None:
        total_cost = _calculate_total_plan_cost(plan)
        if total_cost > budget_limit:
            violations.append(
                f"Total budget constraint violated: total cost ¥{total_cost:.2f} exceeds max ¥{budget_limit}"
            )

    ticket_budget_limit = _extract_ticket_budget_limit_from_resource(resource)
    if ticket_budget_limit is not None:
        ticket_cost = _calculate_plan_ticket_cost(plan)
        if ticket_cost > ticket_budget_limit:
            violations.append(
                f"Ticket budget constraint violated: total ticket cost ¥{ticket_cost:.2f} exceeds max ¥{ticket_budget_limit}"
            )

    ticket_price_cap = _extract_ticket_price_cap_from_resource(resource)
    if ticket_price_cap is not None:
        itinerary = plan.get("itinerary", [])
        for day_plan in itinerary:
            activities = day_plan.get("activities", [])
            for activity in activities:
                ticket_price = _extract_activity_ticket_price(activity)
                if ticket_price is not None and ticket_price > ticket_price_cap:
                    poi_name = activity.get("position") or activity.get("end") or activity.get("start") or "<unknown>"
                    violations.append(
                        f"Ticket price cap violated: {poi_name} ticket price ¥{ticket_price:.2f} exceeds cap ¥{ticket_price_cap}"
                    )

    return (len(violations) == 0, violations)


def validate_constraint_satisfaction(
    mock_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    poi_db: POIDatabase,
    constraint_type: Optional[str] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """
    验证mock_plan是否满足约束条件

    Args:
        mock_plan: Mock edited plan
        constraints: 约束条件
        poi_db: POI数据库

    Returns:
        (is_valid, violations) 元组
    """
    violations = []
    query_spec = _trace_dict(query_generation_trace, "surface_query_spec")
    if query_spec is None:
        query_spec = _trace_dict(query_generation_trace, "canonical_query_spec", "query_spec") or {}
    normalized_constraint_type = str(constraint_type or query_spec.get("constraint_type") or "").strip()
    visited_pois = _collect_visited_pois(mock_plan)
    transit_points = _collect_transit_points(mock_plan)

    # 1. 检查must_include是否都包含在plan中
    must_include = constraints.get("must_include", [])
    if must_include:
        plan_pois = _collect_plan_pois(mock_plan)

        # 检查must_include中的POI是否都在plan中
        for poi in must_include:
            if isinstance(poi, dict):
                name = poi.get("name")
                if name and name not in plan_pois:
                    violations.append(f"Required POI not in plan: {name}")

    # 2. 检查时间约束
    temporal = _normalize_temporal_constraints(constraints.get("temporal", []))
    if temporal:
        itinerary = mock_plan.get("itinerary", [])
        day_pois = _collect_plan_pois_by_day(mock_plan)
        anchor_poi = (
            query_spec.get("anchor_poi")
            or query_spec.get("source_poi")
            or next(
                (
                    poi.get("name")
                    for poi in must_include
                    if isinstance(poi, dict) and isinstance(poi.get("name"), str) and poi.get("name")
                ),
                None,
            )
        )

        for temp_constraint in temporal:
            bind_day = temp_constraint.get("_day")
            required_names = temp_constraint.get("_must_visit_names", [])
            if required_names and bind_day is not None:
                day_set = day_pois.get(bind_day, set())
                for poi_name in required_names:
                    if poi_name not in day_set:
                        violations.append(f"Day {bind_day}: required POI not found in that day: {poi_name}")

            constraint_minutes = temp_constraint.get("_end_before_min")
            if constraint_minutes is not None:
                for day_plan in itinerary:
                    day = day_plan.get("day")
                    if bind_day is not None and day != bind_day:
                        continue
                    activities = day_plan.get("activities", [])
                    if not activities:
                        continue
                    last_activity = activities[-1]
                    end_time = last_activity.get("end_time")
                    end_minutes = _parse_time_to_minutes(end_time)
                    if end_minutes is None:
                        continue

                    if end_minutes > constraint_minutes:
                        violations.append(
                            f"Day {day}: Ends at {end_time}, constraint: {_format_minutes_hhmm(constraint_minutes)}"
                        )

            max_duration = temp_constraint.get("_max_duration_min")
            if max_duration is not None:
                # duration_limit 优先只校验 query 锚定的 POI，避免误扫整天无关活动。
                anchor_violations: List[str] = []
                anchor_checked = False
                for day_plan in itinerary:
                    day = day_plan.get("day")
                    if bind_day is not None and day != bind_day:
                        continue
                    activities = day_plan.get("activities", [])
                    for activity in activities:
                        if (
                            normalized_constraint_type == "duration_limit"
                            and anchor_poi
                            and not _activity_matches_anchor(activity, anchor_poi)
                        ):
                            continue
                        start_time = activity.get("start_time")
                        end_time = activity.get("end_time")
                        if not (start_time and end_time):
                            continue
                        duration = _duration_minutes(start_time, end_time)
                        if normalized_constraint_type == "duration_limit" and anchor_poi:
                            anchor_checked = True
                            if duration is not None and duration > max_duration:
                                anchor_violations.append(
                                    f"Anchor POI '{anchor_poi}' duration {duration}min exceeds max {max_duration}min"
                                )
                        elif duration is not None and duration > max_duration:
                            violations.append(
                                f"Activity duration {duration}min exceeds max {max_duration}min"
                            )
                if anchor_violations:
                    violations.extend(anchor_violations)
                elif (
                    normalized_constraint_type == "duration_limit"
                    and anchor_poi
                    and not anchor_checked
                    and not _any_location_equivalent(anchor_poi, visited_pois)
                ):
                    violations.append(f"Anchor POI for duration_limit not found in visited plan: {anchor_poi}")

    # 3. 检查forbidden
    forbidden = constraints.get("forbidden", [])
    if forbidden:
        for poi_name in forbidden:
            if _any_location_equivalent(poi_name, visited_pois):
                violations.append(f"Forbidden POI found in plan: {poi_name}")
            elif (
                normalized_constraint_type == "ticket_price_cap_or_substitute"
                and _any_location_equivalent(poi_name, transit_points)
            ):
                continue

    # 4. 检查空间约束
    is_valid, spatial_violations = validate_spatial_constraints(
        mock_plan,
        constraints,
        poi_db,
        constraint_type=normalized_constraint_type,
    )
    if not is_valid:
        violations.extend(spatial_violations)

    # 5. 检查票务资源约束
    is_valid, budget_violations = validate_budget_constraints(mock_plan, constraints)
    if not is_valid:
        violations.extend(budget_violations)

    return (len(violations) == 0, violations)


def _validate_strategy_plan_schema_v2_frozen(
    conflict_labels: List[List[str]],
    strategy_plan: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    errors = []

    expected_keys = []
    for label in conflict_labels:
        key = build_conflict_key(label)
        if key:
            expected_keys.append(key)

    grounding = strategy_plan.get("phase_a_grounding")
    if not isinstance(grounding, list):
        errors.append("strategy_plan.phase_a_grounding must be a list")
        grounding = []

    grounded_keys = set()
    for idx, item in enumerate(grounding):
        if not isinstance(item, dict):
            errors.append(f"phase_a_grounding[{idx}] must be an object")
            continue

        conflict = item.get("conflict")
        conflict_key = item.get("conflict_key")
        derived_key = build_conflict_key(conflict)

        if not derived_key:
            errors.append(f"phase_a_grounding[{idx}].conflict must be [scope,dimension,violation]")
            continue

        if conflict_key != derived_key:
            errors.append(
                f"phase_a_grounding[{idx}].conflict_key mismatch: {conflict_key} != {derived_key}"
            )
        grounded_keys.add(derived_key)

    phase_b = strategy_plan.get("phase_b_synthesis")
    if not isinstance(phase_b, dict):
        errors.append("strategy_plan.phase_b_synthesis must be an object")
        phase_b = {}

    primary_scope = phase_b.get("primary_scope")
    if primary_scope not in ALLOWED_SCOPES:
        errors.append(f"phase_b_synthesis.primary_scope invalid: {primary_scope}")

    operator_priority = phase_b.get("operator_priority")
    if not isinstance(operator_priority, list) or not operator_priority:
        errors.append("phase_b_synthesis.operator_priority must be a non-empty list")
        operator_priority = []
    else:
        for op in operator_priority:
            if op not in ALLOWED_OPERATORS:
                errors.append(f"phase_b_synthesis.operator_priority contains invalid operator: {op}")

    execution_plan = phase_b.get("execution_plan")
    if not isinstance(execution_plan, list) or not execution_plan:
        errors.append("phase_b_synthesis.execution_plan must be a non-empty list")
        execution_plan = []
    else:
        for idx, item in enumerate(execution_plan):
            if not isinstance(item, dict):
                errors.append(f"execution_plan[{idx}] must be an object")
                continue

            scope = item.get("scope")
            if scope not in ALLOWED_SCOPES:
                errors.append(f"execution_plan[{idx}].scope invalid: {scope}")

            operators = item.get("operators")
            if not isinstance(operators, list) or not operators:
                errors.append(f"execution_plan[{idx}].operators must be a non-empty list")
            else:
                for op in operators:
                    if op not in ALLOWED_OPERATORS:
                        errors.append(f"execution_plan[{idx}] contains invalid operator: {op}")

            target_conflicts = item.get("target_conflicts")
            if not isinstance(target_conflicts, list) or not target_conflicts:
                errors.append(f"execution_plan[{idx}].target_conflicts must be a non-empty list")
            else:
                for key in target_conflicts:
                    if not isinstance(key, str):
                        errors.append(f"execution_plan[{idx} target_conflicts has non-string key: {key}")

    fallback_policy = phase_b.get("fallback_policy")
    if not isinstance(fallback_policy, list):
        errors.append("phase_b_synthesis.fallback_policy must be a list")
        fallback_policy = []
    else:
        prev_level = SCOPE_LEVEL.get(primary_scope, 0)
        for idx, item in enumerate(fallback_policy):
            if not isinstance(item, dict):
                errors.append(f"fallback_policy[{idx}] must be an object")
                continue
            upgrade_scope_to = item.get("upgrade_scope_to")
            action = item.get("action")
            if upgrade_scope_to not in ALLOWED_SCOPES:
                errors.append(f"fallback_policy[{idx}].upgrade_scope_to invalid: {upgrade_scope_to}")
            if action not in ALLOWED_SCOPE_ACTIONS:
                errors.append(f"fallback_policy[{idx}].action invalid: {action}")
            if upgrade_scope_to in SCOPE_LEVEL:
                level = SCOPE_LEVEL[upgrade_scope_to]
                if level < prev_level:
                    errors.append(
                        f"fallback_policy[{idx}] scope downgrade not allowed: {upgrade_scope_to}"
                    )
                prev_level = level

    # 关键约束：所有 conflict 必须先 grounding
    for key in expected_keys:
        if key not in grounded_keys:
            errors.append(f"missing grounding for conflict: {key}")

    return (len(errors) == 0, errors)


def _validate_strategy_plan_schema_v2_primary_only(
    conflict_labels: List[List[str]],
    strategy_plan: Dict[str, Any],
    origin_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    expected_keys = []
    for label in conflict_labels:
        key = build_conflict_key(label)
        if key:
            expected_keys.append(key)
    expected_set = {_normalize_key_fragment(key) for key in expected_keys}

    primary_scope = strategy_plan.get("primary_scope")
    if primary_scope not in ALLOWED_PRIMARY_SCOPES:
        errors.append(f"primary_scope invalid: {primary_scope}")

    primary_strategy = strategy_plan.get("primary_strategy")
    if primary_strategy not in ALLOWED_PRIMARY_STRATEGIES:
        errors.append(f"primary_strategy invalid: {primary_strategy}")

    strategy_target = strategy_plan.get("strategy_target")
    if not isinstance(strategy_target, dict):
        errors.append("strategy_target must be an object")
        strategy_target = {}

    anchor_citations = strategy_target.get("anchor_citations")
    if not isinstance(anchor_citations, list) or not anchor_citations:
        errors.append("strategy_target.anchor_citations must be a non-empty list")
    else:
        for idx, citation in enumerate(anchor_citations):
            if not isinstance(citation, str) or not re.fullmatch(r"itinerary\[\d+\]\.activities\[\d+\]", citation):
                errors.append(f"strategy_target.anchor_citations[{idx}] invalid: {citation}")
            elif origin_plan is not None and _resolve_activity_from_citation(origin_plan, citation) is None:
                errors.append(f"strategy_target.anchor_citations[{idx}] not found in origin_plan: {citation}")

    target_conflicts = strategy_target.get("target_conflicts")
    if not isinstance(target_conflicts, list) or not target_conflicts:
        errors.append("strategy_target.target_conflicts must be a non-empty list")
        target_conflicts = []
    else:
        for idx, key in enumerate(target_conflicts):
            if not isinstance(key, str):
                errors.append(f"strategy_target.target_conflicts[{idx}] must be string: {key}")
                continue
            normalized = _normalize_key_fragment(key)
            if normalized in BAD_CONFLICT_KEY_PATTERNS:
                errors.append(f"dirty_key:{key}")
            elif normalized not in expected_set:
                errors.append(f"unknown_key:{key}")

    expected_effect = strategy_target.get("expected_effect")
    if not isinstance(expected_effect, str) or not expected_effect.strip():
        errors.append("strategy_target.expected_effect must be a non-empty string")

    execution_plan = strategy_plan.get("execution_plan")
    if not isinstance(execution_plan, list) or not execution_plan:
        errors.append("execution_plan must be a non-empty list")
        execution_plan = []
    else:
        for idx, item in enumerate(execution_plan):
            if not isinstance(item, dict):
                errors.append(f"execution_plan[{idx}] must be an object")
                continue
            phase = item.get("phase")
            if not isinstance(phase, str) or not phase.strip():
                errors.append(f"execution_plan[{idx}].phase must be a non-empty string")

            operators = item.get("operators")
            if not isinstance(operators, list) or not operators:
                errors.append(f"execution_plan[{idx}].operators must be a non-empty list")
            else:
                for op in operators:
                    if op not in ALLOWED_PRIMARY_STRATEGIES:
                        errors.append(f"execution_plan[{idx}] contains invalid operator: {op}")

            anchors = item.get("anchors")
            if not isinstance(anchors, list):
                errors.append(f"execution_plan[{idx}].anchors must be a list")
            else:
                for anchor_idx, citation in enumerate(anchors):
                    if not isinstance(citation, str) or not re.fullmatch(r"itinerary\[\d+\]\.activities\[\d+\]", citation):
                        errors.append(f"execution_plan[{idx}].anchors[{anchor_idx}] invalid: {citation}")
                    elif origin_plan is not None and _resolve_activity_from_citation(origin_plan, citation) is None:
                        errors.append(f"execution_plan[{idx}].anchors[{anchor_idx}] not found in origin_plan: {citation}")

            rationale = item.get("rationale")
            if not isinstance(rationale, str):
                errors.append(f"execution_plan[{idx}].rationale must be a string")

    if execution_plan and isinstance(execution_plan[0], dict):
        operators0 = execution_plan[0].get("operators", [])
        if (
            primary_strategy in ALLOWED_PRIMARY_STRATEGIES
            and isinstance(operators0, list)
            and (not operators0 or operators0[0] != primary_strategy)
        ):
            errors.append("execution_plan[0].operators[0] must equal primary_strategy")

    fallback_chain = strategy_plan.get("fallback_chain")
    if fallback_chain is None:
        fallback_chain = []
    if not isinstance(fallback_chain, list):
        errors.append("fallback_chain must be a list")
    else:
        for idx, item in enumerate(fallback_chain):
            if not isinstance(item, dict):
                errors.append(f"fallback_chain[{idx}] must be an object")
                continue
            when = item.get("when")
            operator = item.get("operator")
            if when not in ALLOWED_FALLBACK_WHENS:
                errors.append(f"fallback_chain[{idx}].when invalid: {when}")
            if operator not in ALLOWED_PRIMARY_STRATEGIES:
                errors.append(f"fallback_chain[{idx}].operator invalid: {operator}")
            note = item.get("note")
            if not isinstance(note, str):
                errors.append(f"fallback_chain[{idx}].note must be a string")

    confidence = strategy_plan.get("confidence")
    if not isinstance(confidence, dict):
        errors.append("confidence must be an object")
    else:
        for key in ["primary_strategy", "anchors"]:
            value = confidence.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"confidence.{key} must be numeric")

    validity = strategy_plan.get("validity")
    if not isinstance(validity, dict):
        errors.append("validity must be an object")
    else:
        if not isinstance(validity.get("is_valid"), bool):
            errors.append("validity.is_valid must be bool")
        invalid_reasons = validity.get("invalid_reasons")
        if not isinstance(invalid_reasons, list):
            errors.append("validity.invalid_reasons must be a list")

    return (len(errors) == 0, errors)


def validate_strategy_plan_schema(conflict_labels: List[List[str]],
                                  strategy_plan: Dict[str, Any],
                                  origin_plan: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    """
    验证 strategy_plan 的结构与枚举合法性。
    兼容 v2_primary_only 和旧版 v2_frozen。
    """
    if not isinstance(strategy_plan, dict):
        return False, ["strategy_plan must be an object"]

    version = strategy_plan.get("version")
    if version == "v2_primary_only":
        return _validate_strategy_plan_schema_v2_primary_only(conflict_labels, strategy_plan, origin_plan=origin_plan)
    if version == "v2_frozen":
        return _validate_strategy_plan_schema_v2_frozen(conflict_labels, strategy_plan)
    return False, [f"unsupported strategy_plan.version: {version}"]


def validate_conflict_coverage(conflict_labels: List[List[str]],
                               strategy_plan: Dict[str, Any],
                               primary_conflict: Optional[List[str]] = None,
                               target_bucket: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    验证 strategy_plan 是否覆盖了所有 conflict：
    1) 必须 grounding
    2) 必须进入 execution_plan.target_conflicts
    """
    uncovered_conflicts = []
    expected_keys = []
    for label in conflict_labels:
        key = build_conflict_key(label)
        if key:
            expected_keys.append(key)

    version = strategy_plan.get("version") if isinstance(strategy_plan, dict) else None
    if version == "v2_primary_only":
        strategy_target = strategy_plan.get("strategy_target", {})
        target_conflicts = strategy_target.get("target_conflicts", []) if isinstance(strategy_target, dict) else []
        targeted_keys = {
            _normalize_key_fragment(key)
            for key in target_conflicts
            if isinstance(key, str)
        }
        required_keys: List[str] = []

        primary_key = build_conflict_key(primary_conflict) if primary_conflict else None
        if isinstance(primary_key, str):
            required_keys.append(primary_key)

        target_key = build_conflict_key(target_bucket) if target_bucket else None
        if (
            isinstance(target_key, str)
            and target_bucket
            and len(target_bucket) == 3
            and str(target_bucket[2]).strip().lower() in ALLOWED_EDIT_VIOLATIONS
        ):
            required_keys.append(target_key)

        if not required_keys:
            required_keys = expected_keys[:1] if expected_keys else []

        seen_required = set()
        for key in required_keys:
            normalized_key = _normalize_key_fragment(key)
            if normalized_key in seen_required:
                continue
            seen_required.add(normalized_key)
            if normalized_key not in targeted_keys:
                uncovered_conflicts.append(f"Required conflict {key} not targeted in strategy_target")
        return (len(uncovered_conflicts) == 0, uncovered_conflicts)

    grounding = strategy_plan.get("phase_a_grounding", [])
    grounded_keys = set()
    if isinstance(grounding, list):
        for item in grounding:
            if isinstance(item, dict):
                key = item.get("conflict_key")
                if isinstance(key, str):
                    grounded_keys.add(_normalize_key_fragment(key))

    phase_b = strategy_plan.get("phase_b_synthesis", {})
    execution_plan = phase_b.get("execution_plan", []) if isinstance(phase_b, dict) else []
    targeted_keys = set()
    if isinstance(execution_plan, list):
        for item in execution_plan:
            if not isinstance(item, dict):
                continue
            for key in item.get("target_conflicts", []):
                if isinstance(key, str):
                    targeted_keys.add(_normalize_key_fragment(key))

    for key in expected_keys:
        normalized_key = _normalize_key_fragment(key)
        if normalized_key not in grounded_keys:
            uncovered_conflicts.append(f"Conflict {key} missing in phase_a_grounding")
            continue
        if normalized_key not in targeted_keys:
            uncovered_conflicts.append(f"Conflict {key} not targeted in execution_plan")

    return (len(uncovered_conflicts) == 0, uncovered_conflicts)


def validate_strategy_diff_consistency(strategy_plan: Dict[str, Any], op_tags: List[str]) -> Tuple[bool, List[str]]:
    """
    验证 strategy_plan 与实际 diff(op_tags) 的一致性。
    """
    inconsistencies = []
    if not isinstance(strategy_plan, dict):
        return False, ["strategy_plan must be an object"]

    normalized_tags = [str(tag).upper() for tag in (op_tags or [])]
    version = strategy_plan.get("version")

    planned_operators = []
    execution_scopes = set()
    scope_actions = set()
    if version == "v2_primary_only":
        execution_plan = strategy_plan.get("execution_plan", [])
        if isinstance(execution_plan, list):
            for item in execution_plan:
                if not isinstance(item, dict):
                    continue
                for op in item.get("operators", []):
                    if isinstance(op, str):
                        planned_operators.append(op)
        primary_scope = strategy_plan.get("primary_scope")
        if isinstance(primary_scope, str):
            execution_scopes.add(primary_scope)
        fallback_chain = strategy_plan.get("fallback_chain", [])
        if isinstance(fallback_chain, list):
            for item in fallback_chain:
                if isinstance(item, dict):
                    operator = item.get("operator")
                    if isinstance(operator, str):
                        scope_actions.add(operator)
    else:
        phase_b = strategy_plan.get("phase_b_synthesis", {})
        execution_plan = phase_b.get("execution_plan", []) if isinstance(phase_b, dict) else []
        fallback_policy = phase_b.get("fallback_policy", []) if isinstance(phase_b, dict) else []

        if isinstance(execution_plan, list):
            for item in execution_plan:
                if not isinstance(item, dict):
                    continue
                scope = item.get("scope")
                if isinstance(scope, str):
                    execution_scopes.add(scope)
                for op in item.get("operators", []):
                    if isinstance(op, str):
                        planned_operators.append(op)

        if isinstance(fallback_policy, list):
            for item in fallback_policy:
                if isinstance(item, dict):
                    action = item.get("action")
                    if isinstance(action, str):
                        scope_actions.add(action)

    strategy_op_mapping = {
        "LocalParamAdjust": ["MODIFY", "UPDATE"],
        "Substitute": ["REPLACE", "SUBSTITUTE"],
        "Reorder": ["REORDER", "MOVE", "MODIFY-DAY", "MOVE-ACROSS-DAY"],
        "Prune": ["DELETE", "REMOVE", "DELETE-DAY"]
    }

    for op in planned_operators:
        expected_ops = strategy_op_mapping.get(op, [])
        if not expected_ops:
            continue
        has_match = any(any(expected in tag for expected in expected_ops) for tag in normalized_tags)
        if not has_match:
            inconsistencies.append(f"Operator {op} has no matching op_tags")

    if "ADD-DAY" in normalized_tags:
        has_global_semantics = ("global" in execution_scopes) or ("GlobalReplan" in scope_actions)
        if not has_global_semantics:
            inconsistencies.append("ADD-DAY found but strategy_plan lacks global scope/GlobalReplan")

    return (len(inconsistencies) == 0, inconsistencies)


def validate_structural_validity(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证plan结构的基本合法性

    Args:
        plan: Plan字典

    Returns:
        (is_valid, errors) 元组
    """
    errors = []

    # 基本字段检查
    required_fields = ["people_number", "start_city", "target_city", "itinerary"]
    for field in required_fields:
        if field not in plan:
            errors.append(f"Missing required field: {field}")

    itinerary = plan.get("itinerary", [])
    if not itinerary:
        errors.append("Empty itinerary")
        return (False, errors)

    # 检查每天的activities
    for day_plan in itinerary:
        day = day_plan.get("day")
        activities = day_plan.get("activities", [])

        if not activities:
            continue

        # 检查时间顺序（支持 24:00 和跨午夜）
        prev_end_time_text = None
        prev_end_abs_minutes = None
        day_offset = 0
        prev_end_location = None

        for activity in activities:
            start_time = activity.get("start_time")
            end_time = activity.get("end_time")
            activity_type = activity.get("type", "")
            start_location, end_location = _extract_activity_endpoints(activity)
            continuity_entry = _extract_continuity_entry(activity)

            if not start_time or not end_time:
                errors.append(f"Day {day}: Activity missing time")
                continue

            start_minutes = _parse_time_to_minutes(start_time)
            end_minutes = _parse_time_to_minutes(end_time)
            if start_minutes is None or end_minutes is None:
                errors.append(f"Day {day}: Invalid time format ({start_time} - {end_time})")
                continue

            start_abs = start_minutes + day_offset

            # 若当前活动看起来跨到次日早晨，允许自动滚动到下一天
            if (
                prev_end_abs_minutes is not None
                and start_abs < prev_end_abs_minutes
                and start_minutes <= 6 * 60
                and (prev_end_abs_minutes % (24 * 60)) >= 18 * 60
            ):
                day_offset += 24 * 60
                start_abs = start_minutes + day_offset

            end_abs = end_minutes + day_offset
            if end_abs < start_abs:
                end_abs += 24 * 60

            if end_abs <= start_abs:
                errors.append(f"Day {day}: End time before start time ({start_time} - {end_time})")

            # 检查时间连续性（仅检查非跨城交通活动）
            if prev_end_abs_minutes is not None and not _is_intercity_activity(activity_type):
                if start_abs < prev_end_abs_minutes:
                    errors.append(f"Day {day}: Activity overlaps previous ({prev_end_time_text} - {start_time})")

            prev_end_abs_minutes = end_abs
            prev_end_time_text = end_time

            # 检查地点连续性（跳过飞机和火车）
            if (
                prev_end_location
                and not _is_intercity_activity(activity_type)
            ):
                current_entry = continuity_entry or start_location
                is_continuous = _locations_equivalent(prev_end_location, current_entry)

                # 兼容旧数据：transport 入口缺失时回退比较 activity.start
                if not is_continuous and continuity_entry and start_location:
                    is_continuous = _locations_equivalent(prev_end_location, start_location)

                if current_entry and not is_continuous:
                    errors.append(
                        f"Day {day}: City discontinuity - previous end: {prev_end_location}, "
                        f"current start: {current_entry}"
                    )

            if end_location:
                prev_end_location = end_location

    return (len(errors) == 0, errors)


def detect_conflicts(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    poi_db: POIDatabase,
    constraint_type: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """
    检测origin_plan和constraints之间的冲突 (完整实现)

    Args:
        origin_plan: 原始plan
        constraints: 约束条件
        poi_db: POI数据库

    Returns:
        检测到的conflict列表，每个是 (scope, dimension, violation) 元组
    """
    conflicts = []

    # 1. 检查temporal overflow - 时间溢出
    temporal = _normalize_temporal_constraints(constraints.get("temporal", []))
    must_include = constraints.get("must_include", [])
    itinerary = origin_plan.get("itinerary", [])

    for temp_constraint in temporal:
        bind_day = temp_constraint.get("_day")
        constraint_minutes = temp_constraint.get("_end_before_min")
        if constraint_minutes is not None:
            for day_plan in itinerary:
                day = day_plan.get("day")
                if bind_day is not None and day != bind_day:
                    continue
                activities = day_plan.get("activities", [])
                if not activities:
                    continue
                last_activity = activities[-1]
                end_time = last_activity.get("end_time")
                end_minutes = _parse_time_to_minutes(end_time)
                if end_minutes is None:
                    continue
                if end_minutes > constraint_minutes:
                    conflicts.append(("parameter", "temporal", "overflow"))
                    break

        max_duration = temp_constraint.get("_max_duration_min")
        if max_duration is not None:
            for day_plan in itinerary:
                day = day_plan.get("day")
                if bind_day is not None and day != bind_day:
                    continue
                activities = day_plan.get("activities", [])
                hit = False
                for activity in activities:
                    if activity.get("type") == "train":
                        continue
                    start_time = activity.get("start_time")
                    end_time = activity.get("end_time")
                    if not (start_time and end_time):
                        continue
                    duration = _duration_minutes(start_time, end_time)
                    if duration is not None and duration > max_duration:
                        conflicts.append(("parameter", "temporal", "overflow"))
                        hit = True
                        break
                if hit:
                    break

    # 2. 检查spatial infeasibility - 空间不可行（使用plan中的cost和geodesic距离）
    is_valid, violations = validate_spatial_constraints(
        origin_plan,
        constraints,
        poi_db,
        constraint_type=constraint_type,
    )
    if not is_valid:
        for violation in violations:
            if "distance" in violation.lower():
                conflicts.append(("parameter", "spatial", "infeasible"))
            if "cost" in violation.lower():
                conflicts.append(("parameter", "resource", "overflow"))

    # 3. 检查structural incompatibility - 结构不兼容
    must_include_count = len(must_include)
    itinerary = origin_plan.get("itinerary", [])

    if must_include_count > 0:
        # 单日行程，要求多个POI
        if len(itinerary) == 1:
            day_plan = itinerary[0]
            activities = day_plan.get("activities", [])

            # 如果是往返行程，无法完成多个POI
            if len(activities) == 2:
                # 检查是否是往返
                start_city = origin_plan.get("start_city")
                target_city = origin_plan.get("target_city")

                if (activities[0].get("start") == start_city and
                    activities[1].get("end") == start_city):
                    if must_include_count > 1:
                        conflicts.append(("structural", "temporal", "incompatibility"))

    # 4. 检查compositional discontinuity - 组合不连续
    if must_include_count > 0:
        target_city = origin_plan.get("target_city")

        # 获取所有在target_city的POI（排除交通）
        city_pois = set()
        itinerary = origin_plan.get("itinerary", [])

        for day_plan in itinerary:
            activities = day_plan.get("activities", [])
            for activity in activities:
                # 只统计非交通活动
                if activity.get("type") == "train":
                    continue

                start = activity.get("start", "")
                end = activity.get("end", "")

                # 如果包含target_city，说明在当地有活动
                if target_city in start:
                    city_pois.add(start)
                if target_city in end:
                    city_pois.add(end)

        # 如果要求多个POI但实际没有足够的城市活动
        if must_include_count > len(city_pois):
            conflicts.append(("compositional", "structural", "discontinuity"))

    # 5. 检查resource overflow - 资源溢出
    resource = constraints.get("resource", {})

    budget_limit = _extract_budget_limit_from_resource(resource)
    if budget_limit is not None:
        origin_cost = _calculate_total_plan_cost(origin_plan)
        if origin_cost > budget_limit:
            conflicts.append(("parameter", "resource", "overflow"))

    ticket_budget_limit = _extract_ticket_budget_limit_from_resource(resource)
    if ticket_budget_limit is not None:
        origin_ticket_cost = _calculate_plan_ticket_cost(origin_plan)
        if origin_ticket_cost > ticket_budget_limit:
            conflicts.append(("parameter", "resource", "overflow"))

    ticket_price_cap = _extract_ticket_price_cap_from_resource(resource)
    if ticket_price_cap is not None:
        itinerary = origin_plan.get("itinerary", [])
        for day_plan in itinerary:
            activities = day_plan.get("activities", [])
            if any(
                (_extract_activity_ticket_price(activity) or 0.0) > ticket_price_cap
                for activity in activities
            ):
                conflicts.append(("parameter", "resource", "overflow"))
                break

    return conflicts


CORE_PRIMARY_CHECKS = [
    "poi_existence",
    "city_consistency",
    "constraint_ir_presence",
    "constraint_ir_schema",
    "template_ir_alignment",
    "ir_truth_compile",
    "surface_ir_equivalence",
    "hard_truth_rebuild",
    "effective_constraint_merge",
    "verifier_support",
    "soft_tag_presence",
    "soft_mapping_supported",
    "soft_preference_rebuild_match",
    "hard_soft_axis_dedup",
    "structural_validity",
    "primary_conflict_exists",
    "primary_target_alignment",
    "purity_rule",
    "eligible_violation_pool",
    "resolver_determinism",
]

STRATEGY_PRIMARY_CHECKS = [
    "strategy_plan_schema",
    "conflict_coverage",
]

FULL_PRIMARY_CHECKS = CORE_PRIMARY_CHECKS + STRATEGY_PRIMARY_CHECKS

OPTIONAL_EXECUTION_CHECKS = [
    "constraint_satisfaction",
    "structural_validity_mock",
    "strategy_diff_consistency",
]

PRIMARY_CHECKS = CORE_PRIMARY_CHECKS
REPORTED_NON_BLOCKING_CHECKS: List[str] = [
    "constraint_query_alignment",
    "query_factual_validity",
    "temporal_window_feasibility",
    "solver_feasibility",
    "category_guard_validity",
    "canonical_gate_trace",
    "canonical_alignment_trace",
    "surface_proof_recomputed",
    "soft_query_alignment",
]
ALL_REPORTED_CHECKS = FULL_PRIMARY_CHECKS + OPTIONAL_EXECUTION_CHECKS
VALIDATION_RESULT_KEYS = [
    "poi_existence",
    "city_consistency",
    "constraint_ir_presence",
    "constraint_ir_schema",
    "template_ir_alignment",
    "ir_truth_compile",
    "verifier_support",
    "surface_ir_equivalence",
    "hard_soft_axis_dedup",
    "effective_constraint_merge",
    "constraint_query_alignment",
    "query_factual_validity",
    "temporal_window_feasibility",
    "solver_feasibility",
    "category_guard_validity",
    "spatial_constraints",
    "budget_constraints",
    "constraint_satisfaction",
    "strategy_plan_schema",
    "conflict_coverage",
    "strategy_diff_consistency",
    "structural_validity",
    "structural_validity_mock",
    "conflicts_detected",
    "primary_conflict_exists",
    "primary_target_alignment",
    "purity_rule",
    "eligible_violation_pool",
    "resolver_determinism",
    "canonical_gate_trace",
    "canonical_alignment_trace",
    "surface_proof_recomputed",
    "hard_truth_rebuild",
    "soft_tag_presence",
    "soft_query_alignment",
    "soft_mapping_supported",
    "soft_preference_rebuild_match",
]


def _make_check_result(
    passed: bool,
    detail_key: Optional[str] = None,
    details: Optional[Any] = None,
    **extra: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"pass": passed}
    if detail_key is not None:
        result[detail_key] = details if details is not None else []
    result.update(extra)
    return result


def _make_skipped_result(reason: str, detail_key: str) -> Dict[str, Any]:
    return _make_check_result(
        True,
        detail_key,
        [],
        skipped=True,
        reason=reason,
    )


def _compute_all_pass(results: Dict[str, Any]) -> bool:
    return all(
        isinstance(results.get(check_name), dict)
        and results[check_name].get("pass", False)
        for check_name in CORE_PRIMARY_CHECKS
    )


def _compute_all_pass_for_checks(
    results: Dict[str, Any],
    required_checks: List[str],
) -> bool:
    return all(
        isinstance(results.get(check_name), dict)
        and results[check_name].get("pass", False)
        for check_name in required_checks
    )


def validate_all(origin_plan: Dict[str, Any], constraints: Dict[str, Any],
                 edit_query: str = "",
                 canonical_edit_query: str = "",
                 constraint_type: Optional[str] = None,
                 template_str: Optional[str] = None,
                 canonical_constraint_ir: Optional[Dict[str, Any]] = None,
                 query_generation_trace: Optional[Dict[str, Any]] = None,
                 origin_logical_constraints: Optional[List[Dict[str, Any]]] = None,
                 origin_preference_tags: Optional[List[Dict[str, Any]]] = None,
                 origin_preference_score_snapshot: Optional[List[Dict[str, Any]]] = None,
                 origin_query_text: str = "",
                 origin_query_structured: Optional[Dict[str, Any]] = None,
                 origin_soft_profile: Optional[List[Dict[str, Any]]] = None,
                 edit_target_constraints: Optional[List[Dict[str, Any]]] = None,
                 effective_logical_constraints: Optional[List[Dict[str, Any]]] = None,
                 effective_constraint_merge_trace: Optional[Dict[str, Any]] = None,
                 edit_target_preference_tags: Optional[List[Dict[str, Any]]] = None,
                 edit_target_preferences: Optional[List[Dict[str, Any]]] = None,
                 mock_plan: Optional[Dict[str, Any]] = None,
                 conflict_labels: Optional[List[List[str]]] = None,
                 conflict_set: Optional[List[List[str]]] = None,
                 primary_conflict: Optional[List[str]] = None,
                 purity_score_rule: Optional[float] = None,
                 match_type_rule: Optional[str] = None,
                 target_bucket: Optional[List[str]] = None,
                 strategy_plan: Optional[Dict[str, Any]] = None,
                 op_tags: Optional[List[str]] = None,
                 poi_db: Optional[POIDatabase] = None,
                 purity_threshold: float = 0.7,
                 require_strategy_checks: bool = False,
                 require_solver_feasibility: bool = False,
                 require_category_guard: bool = False,
                 solver_valid_profile: str = "off") -> Dict[str, Any]:
    """
    运行所有验证检查

    Args:
        origin_plan: 原始plan
        constraints: 约束条件
        edit_query: 编辑需求文本
        mock_plan: Mock edited plan (可选)
        conflict_labels: Conflict标签 (可选)
        conflict_set: 规则化冲突集合 (可选)
        primary_conflict: 规则主冲突 (可选)
        purity_score_rule: 规则纯度分数 (可选)
        match_type_rule: 规则匹配类型 (可选)
        target_bucket: 目标bucket (可选)
        strategy_plan: Strategy Plan v2（可选；未提供时跳过 strategy 相关校验）
        op_tags: 操作标签 (可选)
        poi_db: POI数据库 (可选)
        require_strategy_checks: 是否将 strategy 相关检查计入 all_pass
        require_solver_feasibility: 是否将 solver_feasibility 计入 all_pass
        require_category_guard: 是否将 category_guard_validity 计入 all_pass
        solver_valid_profile: off/normal/intentional_infeasible

    Returns:
        验证结果字典
    """
    results = {key: None for key in VALIDATION_RESULT_KEYS}

    effective_conflict_set = conflict_set or conflict_labels or []
    strategy_conflicts = conflict_labels or effective_conflict_set

    # POI存在性验证
    if poi_db:
        is_valid, errors = validate_poi_existence(constraints, poi_db)
        results["poi_existence"] = _make_check_result(is_valid, "errors", errors)

    # 城市一致性验证
    target_city = origin_plan.get("target_city")
    start_city = origin_plan.get("start_city")
    is_valid, errors = validate_city_consistency(
        constraints,
        target_city,
        edit_query=edit_query,
        start_city=start_city,
        origin_query_text=origin_query_text,
        origin_query_structured=origin_query_structured,
    )
    results["city_consistency"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors = check_constraint_ir_presence(canonical_constraint_ir)
    results["constraint_ir_presence"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors = check_constraint_ir_schema(canonical_constraint_ir)
    results["constraint_ir_schema"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors = check_template_ir_alignment(
        canonical_constraint_ir,
        constraint_type=constraint_type,
        template_str=template_str,
    )
    results["template_ir_alignment"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors, compile_details = check_ir_truth_compile(canonical_constraint_ir)
    results["ir_truth_compile"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=compile_details,
    )
    results["verifier_support"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=compile_details,
    )

    is_valid, errors, equivalence_details = check_surface_ir_equivalence(
        canonical_constraint_ir=canonical_constraint_ir,
        query_generation_trace=query_generation_trace,
        constraints=constraints,
        origin_plan=origin_plan,
        origin_query_structured=origin_query_structured,
        edit_query=edit_query,
    )
    results["surface_ir_equivalence"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=equivalence_details,
    )

    is_valid, errors, dedup_details = check_hard_soft_axis_dedup(canonical_constraint_ir)
    results["hard_soft_axis_dedup"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=dedup_details,
    )

    is_valid, errors = validate_query_constraint_alignment(
        edit_query=edit_query,
        constraints=constraints,
        constraint_type=constraint_type,
        template_str=template_str,
        query_generation_trace=query_generation_trace,
        canonical_edit_query=canonical_edit_query,
        edit_target_constraints=edit_target_constraints,
    )
    results["constraint_query_alignment"] = _make_check_result(
        is_valid,
        "errors",
        errors,
    )

    results["query_factual_validity"] = check_query_factual_validity(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query,
        constraint_type=constraint_type,
        query_generation_trace=query_generation_trace,
    )

    results["temporal_window_feasibility"] = assess_temporal_window_feasibility(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_target_constraints=edit_target_constraints,
        canonical_constraint_ir=canonical_constraint_ir,
        query_generation_trace=query_generation_trace,
    )

    results["solver_feasibility"] = assess_solver_feasibility(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_target_constraints=edit_target_constraints,
        origin_logical_constraints=origin_logical_constraints,
        effective_logical_constraints=effective_logical_constraints,
        canonical_constraint_ir=canonical_constraint_ir,
        query_generation_trace=query_generation_trace,
        conflict_set=effective_conflict_set,
        primary_conflict=primary_conflict,
        target_bucket=target_bucket,
        temporal_window_feasibility=results["temporal_window_feasibility"],
    )
    results["category_guard_validity"] = assess_category_guard_validity(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_target_constraints=edit_target_constraints,
        origin_logical_constraints=origin_logical_constraints,
        query_generation_trace=query_generation_trace,
        conflict_set=effective_conflict_set,
        primary_conflict=primary_conflict,
        target_bucket=target_bucket,
        solver_feasibility=results["solver_feasibility"],
        profile=solver_valid_profile,
    )

    is_valid, errors = check_canonical_gate_trace(query_generation_trace)
    results["canonical_gate_trace"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors = check_canonical_alignment_trace(query_generation_trace)
    results["canonical_alignment_trace"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors, recomputed = check_surface_proof_recomputed(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query,
        constraint_type=constraint_type,
        query_generation_trace=query_generation_trace,
    )
    results["surface_proof_recomputed"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        recomputed=recomputed,
    )

    is_valid, errors, rebuilt = check_hard_truth_rebuild(
        origin_plan=origin_plan,
        origin_query_structured=origin_query_structured,
        edit_query=edit_query,
        edit_target_constraints=edit_target_constraints,
        canonical_constraint_ir=canonical_constraint_ir,
        query_generation_trace=query_generation_trace,
        constraints=constraints,
        constraint_type=constraint_type,
    )
    results["hard_truth_rebuild"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=rebuilt,
    )

    is_valid, errors, rebuilt = check_effective_constraint_merge(
        origin_logical_constraints=origin_logical_constraints,
        edit_target_constraints=edit_target_constraints,
        effective_logical_constraints=effective_logical_constraints,
        effective_constraint_merge_trace=effective_constraint_merge_trace,
    )
    results["effective_constraint_merge"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=rebuilt,
    )

    is_valid, errors = check_soft_tag_presence(
        query_generation_trace=query_generation_trace,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    results["soft_tag_presence"] = _make_check_result(
        is_valid,
        "errors",
        errors,
    )

    is_valid, errors = check_soft_query_alignment(
        edit_query=edit_query,
        query_generation_trace=query_generation_trace,
        origin_soft_profile=origin_soft_profile,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    results["soft_query_alignment"] = _make_check_result(
        is_valid,
        "errors",
        errors,
    )

    is_valid, errors, rebuilt_preferences = check_soft_mapping_supported(
        edit_query=edit_query,
        query_generation_trace=query_generation_trace,
        canonical_constraint_ir=canonical_constraint_ir,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    results["soft_mapping_supported"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=rebuilt_preferences,
    )

    is_valid, errors, rebuilt_preferences = check_soft_preference_rebuild_match(
        edit_query=edit_query,
        query_generation_trace=query_generation_trace,
        canonical_constraint_ir=canonical_constraint_ir,
        edit_target_preference_tags=edit_target_preference_tags,
        edit_target_preferences=edit_target_preferences,
    )
    results["soft_preference_rebuild_match"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        rebuilt=rebuilt_preferences,
    )

    # 空间约束验证
    if poi_db:
        is_valid, violations = validate_spatial_constraints(
            origin_plan,
            constraints,
            poi_db,
            constraint_type=constraint_type,
        )
        results["spatial_constraints"] = _make_check_result(
            is_valid,
            "violations",
            violations,
        )

    is_valid, violations = validate_budget_constraints(
        origin_plan,
        constraints,
    )
    results["budget_constraints"] = _make_check_result(
        is_valid,
        "violations",
        violations,
    )

    # Conflict检测
    if poi_db:
        detected = detect_conflicts(
            origin_plan,
            constraints,
            poi_db,
            constraint_type=constraint_type,
        )
        results["conflicts_detected"] = detected

    # 结构合法性验证
    is_valid, errors = validate_structural_validity(origin_plan)
    results["structural_validity"] = _make_check_result(is_valid, "errors", errors)

    # Mock plan验证（execution sample 才要求）
    if mock_plan and poi_db:
        is_valid, violations = validate_constraint_satisfaction(
            mock_plan,
            constraints,
            poi_db,
            constraint_type=constraint_type,
            query_generation_trace=query_generation_trace,
        )
        results["constraint_satisfaction"] = _make_check_result(
            is_valid,
            "violations",
            violations,
        )

        is_valid, errors = validate_structural_validity(mock_plan)
        results["structural_validity_mock"] = _make_check_result(
            is_valid,
            "errors",
            errors,
        )
    else:
        skip_reason = (
            "mock_plan not provided; execution validation disabled for core sample"
        )
        results["constraint_satisfaction"] = _make_skipped_result(
            skip_reason,
            "violations",
        )
        results["structural_validity_mock"] = _make_skipped_result(
            skip_reason,
            "errors",
        )

    # Rule-based conflict checks
    is_valid, errors = check_primary_conflict_exists(primary_conflict, effective_conflict_set)
    results["primary_conflict_exists"] = _make_check_result(
        is_valid,
        "errors",
        errors,
    )

    is_valid, errors = check_primary_matches_target_bucket(
        primary_conflict=primary_conflict,
        conflict_set=effective_conflict_set,
        target_bucket=target_bucket,
        match_type_rule=match_type_rule,
        purity_score_rule=purity_score_rule,
        constraint_type=constraint_type,
        purity_threshold=purity_threshold,
    )
    _, _, alignment_details = evaluate_primary_target_alignment(
        primary_conflict=primary_conflict,
        conflict_set=effective_conflict_set,
        target_bucket=target_bucket,
        match_type_rule=match_type_rule,
        purity_score_rule=purity_score_rule,
        constraint_type=constraint_type,
        purity_threshold=purity_threshold,
    )
    results["primary_target_alignment"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        alignment_details=alignment_details,
    )

    is_valid, errors = check_purity_above_threshold(
        purity_score_rule=purity_score_rule,
        match_type_rule=match_type_rule,
        purity_threshold=purity_threshold,
    )
    results["purity_rule"] = _make_check_result(is_valid, "errors", errors)

    is_valid, errors = check_no_ineligible_violation_in_pool(
        effective_conflict_set,
        primary_conflict=primary_conflict,
        target_bucket=target_bucket,
        constraint_type=constraint_type,
    )
    results["eligible_violation_pool"] = _make_check_result(
        is_valid,
        "errors",
        errors,
    )

    is_valid, errors, recomputed = check_resolver_determinism(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query,
        conflict_labels=conflict_labels,
        conflict_set=effective_conflict_set,
        target_bucket=target_bucket,
        primary_conflict=primary_conflict,
        purity_score_rule=purity_score_rule,
        match_type_rule=match_type_rule,
        constraint_type=constraint_type,
        purity_threshold=purity_threshold,
    )
    results["resolver_determinism"] = _make_check_result(
        is_valid,
        "errors",
        errors,
        recomputed={
            "primary_conflict": recomputed.get("primary_conflict"),
            "match_type_rule": recomputed.get("match_type_rule"),
            "purity_score_rule": recomputed.get("purity_score_rule"),
        },
    )

    # Strategy plan schema验证（strategy 步骤已从主流程移除，未提供时跳过）
    if strategy_conflicts:
        if strategy_plan:
            is_valid, schema_errors = validate_strategy_plan_schema(
                strategy_conflicts,
                strategy_plan,
                origin_plan=origin_plan,
            )
            results["strategy_plan_schema"] = _make_check_result(
                is_valid,
                "errors",
                schema_errors,
            )
        else:
            results["strategy_plan_schema"] = _make_skipped_result(
                "strategy generation disabled; schema validation skipped",
                "errors",
            )

    # Conflict coverage验证（strategy 步骤已从主流程移除，未提供时跳过）
    if strategy_conflicts:
        if strategy_plan:
            is_valid, uncovered = validate_conflict_coverage(
                strategy_conflicts,
                strategy_plan,
                primary_conflict=primary_conflict,
                target_bucket=target_bucket,
            )
            results["conflict_coverage"] = _make_check_result(
                is_valid,
                "uncovered_conflicts",
                uncovered,
            )
        else:
            results["conflict_coverage"] = _make_skipped_result(
                "strategy generation disabled; conflict coverage validation skipped",
                "uncovered_conflicts",
            )

    # Strategy-diff一致性验证（未提供 strategy_plan 时跳过）
    if op_tags is not None:
        if strategy_plan:
            is_valid, inconsistencies = validate_strategy_diff_consistency(strategy_plan, op_tags)
            results["strategy_diff_consistency"] = _make_check_result(
                is_valid,
                "inconsistencies",
                inconsistencies,
            )
        else:
            results["strategy_diff_consistency"] = _make_skipped_result(
                "strategy generation disabled; diff consistency validation skipped",
                "inconsistencies",
            )
    else:
        results["strategy_diff_consistency"] = _make_skipped_result(
            "op_tags not provided; diff consistency validation disabled for core sample",
            "inconsistencies",
        )

    # 计算总体通过状态
    required_checks = list(FULL_PRIMARY_CHECKS if require_strategy_checks else CORE_PRIMARY_CHECKS)
    profile = str(solver_valid_profile or "off").strip().lower()
    if profile == "normal":
        require_solver_feasibility = True
        require_category_guard = True
    elif profile == "intentional_infeasible":
        require_category_guard = True
    if require_solver_feasibility and "solver_feasibility" not in required_checks:
        required_checks.append("solver_feasibility")
    if require_category_guard and "category_guard_validity" not in required_checks:
        required_checks.append("category_guard_validity")
    if profile == "normal" and "temporal_window_feasibility" not in required_checks:
        required_checks.append("temporal_window_feasibility")
    results["all_pass"] = _compute_all_pass_for_checks(results, required_checks)

    return results


if __name__ == "__main__":
    # 测试代码
    print("Data Generation Validators - Test")

    # 简单测试
    from poi_database import POIDatabase

    poi_db = POIDatabase()

    # 测试POI存在性
    constraints_valid = {
        "must_include": [
            {"name": "解放碑", "type": "attraction", "city": "重庆"}
        ]
    }

    is_valid, errors = validate_poi_existence(constraints_valid, poi_db)
    print(f"\nPOI Existence Test: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")

    # 测试距离计算（使用geodesic，与Chinatravel一致）
    distance = calculate_distance(poi_db, "重庆", "重庆北站", "解放碑")
    if distance:
        print(f"\nDistance (重庆北站 -> 解放碑): {distance:.2f} km (using geodesic)")
    else:
        print("\nDistance calculation failed")

    # 测试空间约束验证
    origin_plan = {
        "target_city": "重庆",
        "itinerary": [
            {
                "day": 1,
                "activities": [
                    {
                        "start": "重庆北站",
                        "end": "解放碑",
                        "type": "attraction",
                        "cost": 0
                    }
                ]
            }
        ]
    }

    constraints = {
        "spatial": {
            "max_walking_distance": 5  # 5公里内
        }
    }

    is_valid, violations = validate_spatial_constraints(
        origin_plan,
        constraints,
        poi_db,
        constraint_type=constraint_type,
    )
    print(f"\nSpatial Constraint Test: {'PASS' if is_valid else 'FAIL'}")
    if violations:
        print(f"  Violations: {violations}")
