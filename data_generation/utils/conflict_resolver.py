"""
Rule-based conflict resolver.

目标：在给定 (origin_plan, constraints, edit_query, conflict_labels_llm) 时，
输出可解释、可复现的唯一 primary_conflict。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


SCOPE_ORDER = {"parameter": 0, "structural": 1, "compositional": 2}
DIMENSION_ORDER = {
    "temporal": 6,
    "spatial": 5,
    "resource": 4,
    "structure": 3,
    "sequence": 2,
    "semantic": 1,
}
VIOLATION_ORDER = {"overlap": 3, "overflow": 2, "discontinuity": 1}
HARD_VIOLATIONS = {"overlap", "overflow"}
ALLOWED_EDIT_VIOLATIONS = {"overflow", "overlap", "discontinuity"}
ALLOWED_MATCH_TYPES = {"strong", "weak", "mismatch"}
ACCEPTABLE_PRIMARY_ALTERNATIVES = {
    "cross_day_restructure": {
        ("compositional", "temporal", "overlap"),
        ("compositional", "temporal", "overflow"),
    },
    "add_days": {
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "discontinuity"),
    },
    "time_window": {
        ("parameter", "structure", "discontinuity"),
        ("structural", "temporal", "overflow"),
    },
    "resource_overlap": {
        ("structural", "resource", "overflow"),
        ("structural", "temporal", "overflow"),
    },
}

DIMENSION_ALIASES = {
    "structural": "structure",
    "structure": "structure",
    "temporal": "temporal",
    "spatial": "spatial",
    "resource": "resource",
    "semantic": "semantic",
    "sequence": "sequence",
}

VIOLATION_ALIASES = {
    "overflow": "overflow",
    "overlap": "overlap",
    "discontinuity": "discontinuity",
    "infeasible": "infeasible",
    "incompatibility": "incompatibility",
    "infeasible / incompatibility": "incompatibility",
    "infeasible/incompatibility": "incompatibility",
}
VALID_DIMENSIONS = set(DIMENSION_ORDER) | {"unknown"}
VALID_CONFLICT_VIOLATIONS = {
    "overflow",
    "overlap",
    "discontinuity",
    "infeasible",
    "incompatibility",
}

CHINESE_NUM = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass
class ConstraintAtom:
    dim: str
    hardness: str = "hard"
    anchor: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    source: str = "constraints"


@dataclass
class DemandBlock:
    required_duration: int
    time_window: Optional[Tuple[int, int]] = None
    day: Optional[int] = None
    name: Optional[str] = None


def _scope_rank(scope: str) -> int:
    return SCOPE_ORDER.get(scope, 99)


def _to_minutes(value: Any) -> Optional[int]:
    if not isinstance(value, str) or ":" not in value:
        return None
    parts = value.split(":")
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


def _normalize_time_window(start: Optional[int], end: Optional[int]) -> Optional[Tuple[int, int]]:
    if start is None or end is None:
        return None
    if end <= start:
        end += 24 * 60
    return (start, end)


def _normalize_day_token(token: str) -> Optional[int]:
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token in CHINESE_NUM:
        return CHINESE_NUM[token]
    return None


def _extract_day_from_text(text: str) -> Optional[int]:
    if not isinstance(text, str) or not text:
        return None
    match = re.search(r"第([一二三四五六七八九十\d]+)天", text)
    if not match:
        return None
    return _normalize_day_token(match.group(1))


def _coerce_day_value(day_value: Any) -> Optional[int]:
    if isinstance(day_value, int) and day_value > 0:
        return day_value
    if isinstance(day_value, str):
        day_value = day_value.strip()
        if not day_value:
            return None
        if day_value.isdigit():
            return int(day_value)
        if "天" in day_value:
            return _extract_day_from_text(day_value)
        return _normalize_day_token(day_value)
    return None


def _iter_temporal_constraints(constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, dict):
        return [dict(temporal)]
    if isinstance(temporal, list):
        return [dict(item) for item in temporal if isinstance(item, dict)]
    return []


def _extract_day_poi_pairs_from_query(edit_query: str) -> List[Tuple[int, str]]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    pairs: List[Tuple[int, str]] = []
    pattern = re.compile(r"第([一二三四五六七八九十\d]+)天(?:去|游玩|参观|安排|打卡)([^，。；;]+)")
    for match in pattern.finditer(edit_query):
        day = _normalize_day_token(match.group(1))
        poi_name = str(match.group(2)).strip()
        if day and poi_name:
            pairs.append((day, poi_name))
    return pairs


def extract_day_poi_requirements(constraints: Dict[str, Any], edit_query: str) -> Dict[str, int]:
    """
    提取 POI -> day 的显式绑定。优先使用 constraints.temporal.must_visit，
    并用 query 中“第N天去POI”补充/覆盖。
    """
    poi_day: Dict[str, int] = {}

    for item in _iter_temporal_constraints(constraints):
        day = _coerce_day_value(item.get("day"))
        must_visit = item.get("must_visit")
        if day is None or must_visit is None:
            continue
        if isinstance(must_visit, str):
            names = [must_visit.strip()] if must_visit.strip() else []
        elif isinstance(must_visit, list):
            names = [str(v).strip() for v in must_visit if isinstance(v, str) and str(v).strip()]
        else:
            names = []
        for name in names:
            poi_day[name] = day

    for day, poi_name in _extract_day_poi_pairs_from_query(edit_query):
        poi_day[poi_name] = day

    return poi_day


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


def _extract_requested_day_count(text: str) -> Optional[int]:
    if not isinstance(text, str) or not text:
        return None
    match_digit = re.search(r"(\d+)\s*天", text)
    if match_digit:
        return int(match_digit.group(1))
    match_cn = re.search(r"([一二三四五六七八九十])\s*天", text)
    if match_cn:
        return _normalize_day_token(match_cn.group(1))
    return None


def _extract_daily_poi_cap(text: str) -> Optional[int]:
    if not isinstance(text, str) or not text:
        return None

    patterns_digit = [
        r"每天(?:只|最多|至多|不超过)?(?:安排|游玩|参观|去)?\s*(\d+)\s*个景点",
        r"每天(?:只|最多|至多|不超过)?(?:安排|游玩|参观|去)?\s*(\d+)\s*个(?:地点|POI)",
    ]
    for pattern in patterns_digit:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    patterns_cn = [
        r"每天(?:只|最多|至多|不超过)?(?:安排|游玩|参观|去)?\s*([一二三四五六七八九十])\s*个景点",
        r"每天(?:只|最多|至多|不超过)?(?:安排|游玩|参观|去)?\s*([一二三四五六七八九十])\s*个(?:地点|POI)",
    ]
    for pattern in patterns_cn:
        match = re.search(pattern, text)
        if match:
            return _normalize_day_token(match.group(1))

    return None


def _count_attractions_in_day(day_plan: Dict[str, Any]) -> int:
    activities = day_plan.get("activities", [])
    if not isinstance(activities, list):
        return 0
    count = 0
    for activity in activities:
        if isinstance(activity, dict) and activity.get("type") == "attraction":
            count += 1
    return count


def _extract_time_period_window(edit_query: str) -> Optional[Tuple[int, int]]:
    if "上午" in edit_query:
        return (9 * 60, 12 * 60)
    if "下午" in edit_query:
        return (12 * 60, 18 * 60)
    if "晚上" in edit_query:
        return (18 * 60, 22 * 60)
    return None


def _parse_duration_to_minutes(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        return int(float(value))
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    if not match:
        return None
    number = float(match.group(1))
    if "小时" in value:
        return int(number * 60)
    if "分钟" in value:
        return int(number)
    return int(number)


def _has_explicit_sequence_signal(edit_query: str, constraints: Dict[str, Any]) -> bool:
    query_signals = [
        "顺序",
        "依次",
        "先后",
        "同一天都去",
        "路线",
        "先去",
        "后去",
        "连着去",
        "->",
    ]
    if re.search(r"先.+再.+", edit_query):
        return True
    if any(token in edit_query for token in query_signals):
        return True

    sequence_keys = {"sequence", "order", "route", "visit_order", "same_day", "poi_order"}

    def _scan(obj: Any) -> bool:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).strip().lower() in sequence_keys:
                    return True
                if _scan(value):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if _scan(item):
                    return True
        return False

    return _scan(constraints)


def _normalize_output_label(label: Tuple[str, str, str]) -> List[str]:
    scope, dim, violation = label
    out_dim = "structural" if dim == "structure" else dim
    return [scope, out_dim, violation]


def normalize_conflict_label(label: Any) -> Optional[Tuple[str, str, str]]:
    """标准化 conflict label 到 (scope, dimension, violation)。"""
    if not isinstance(label, (list, tuple)) or len(label) != 3:
        return None

    scope = str(label[0]).strip().lower()
    dim_raw = str(label[1]).strip().lower()
    violation_raw = str(label[2]).strip().lower()

    dim = DIMENSION_ALIASES.get(dim_raw, dim_raw)
    violation = VIOLATION_ALIASES.get(violation_raw, violation_raw)
    if scope not in SCOPE_ORDER:
        return None
    if dim not in VALID_DIMENSIONS:
        return None
    if violation not in VALID_CONFLICT_VIOLATIONS:
        return None
    return (scope, dim, violation)


def _project_target_bucket(
    constraints: Dict[str, Any],
    edit_query: str,
    constraint_type: str,
    target_bucket: Optional[Tuple[str, str, str]],
    conflict_pool: List[Tuple[str, str, str]],
) -> Tuple[Optional[Tuple[str, str, str]], Optional[str]]:
    if (
        target_bucket is None
        or target_bucket in conflict_pool
        or target_bucket[2] not in ALLOWED_EDIT_VIOLATIONS
    ):
        return None, None

    normalized_constraint_type = str(constraint_type or "").strip()
    must_include = constraints.get("must_include", [])
    resource = constraints.get("resource", {})
    semantic = constraints.get("semantic", {})

    if normalized_constraint_type in {"category_overlap", "semantic_and_specific"}:
        has_specific_poi = any(
            isinstance(item, dict) and str(item.get("name", "")).strip()
            for item in must_include
        )
        has_semantic_preference = isinstance(semantic, dict) and bool(str(semantic.get("preferred_type", "")).strip())
        if has_specific_poi and has_semantic_preference and target_bucket == ("structural", "semantic", "overlap"):
            return target_bucket, "semantic_specific_overlap_projection"

    if normalized_constraint_type == "order_constraint":
        has_sequence = any(
            isinstance(item, dict) and item.get("sequence")
            for item in _iter_temporal_constraints(constraints)
        ) or _has_explicit_sequence_signal(edit_query, constraints)
        if has_sequence and target_bucket == ("structural", "temporal", "overlap"):
            return target_bucket, "order_constraint_projection"

    if normalized_constraint_type == "activity_budget_limit":
        if not isinstance(resource, dict):
            return None, None
        has_budget_cap = any(key in resource for key in ("budget_cap", "meal_budget_per_person_per_meal", "hotel_budget_per_night_per_person"))
        has_activity_metric = bool(str(resource.get("activity_type", "")).strip()) or bool(str(resource.get("metric", "")).strip())
        if has_budget_cap and has_activity_metric and target_bucket == ("structural", "resource", "overlap"):
            return target_bucket, "activity_budget_limit_projection"

    if normalized_constraint_type == "ticket_price_cap_or_substitute":
        if not isinstance(resource, dict):
            return None, None
        if not isinstance(semantic, dict):
            semantic = {}
        has_price_cap = any(key in resource for key in ("ticket_price_cap", "ticket_budget_limit", "ticket_budget"))
        has_replace_source = bool(str(semantic.get("replace_from", "")).strip()) or bool(constraints.get("forbidden"))
        if has_price_cap and has_replace_source and target_bucket == ("structural", "resource", "overlap"):
            return target_bucket, "ticket_price_cap_projection"

    if normalized_constraint_type == "resource_overlap":
        if not isinstance(resource, dict):
            return None, None
        has_budget_cap = any(key in resource for key in ("max_budget", "budget", "budget_cap"))
        has_anchor_bundle = bool(str(resource.get("overlap_anchor_poi", "")).strip())
        has_nearby_meal = resource.get("nearby_meal") is True
        has_specific_anchor = any(
            isinstance(item, dict) and str(item.get("name", "")).strip()
            for item in must_include
        )
        if (
            has_budget_cap
            and has_specific_anchor
            and (has_anchor_bundle or has_nearby_meal)
            and target_bucket == ("structural", "resource", "overlap")
        ):
            return target_bucket, "resource_overlap_projection"

    return None, None


def extract_constraint_atoms(constraints: Dict[str, Any], edit_query: str) -> List[ConstraintAtom]:
    atoms: List[ConstraintAtom] = []
    day_anchor = _extract_day_from_text(edit_query)
    explicit_sequence = _has_explicit_sequence_signal(edit_query, constraints)
    poi_day_requirements = extract_day_poi_requirements(constraints, edit_query)

    for item in _iter_temporal_constraints(constraints):
        item_day = _coerce_day_value(item.get("day"))
        anchor_day = item_day if item_day is not None else day_anchor
        has_temporal_threshold = any(
            key in item for key in (
                "end_before",
                "must_end_before",
                "start_after",
                "time_limit",
                "max_duration",
                "duration_limit",
                "duration",
            )
        )
        has_day_must_visit = (item_day is not None and item.get("must_visit") is not None)
        if has_temporal_threshold:
            atom_dim = "temporal"
        elif has_day_must_visit:
            # day+must_visit 更接近结构/顺序绑定，不应默认抬升为 temporal
            atom_dim = "sequence" if explicit_sequence else "structure"
        else:
            atom_dim = "temporal"
        atoms.append(
            ConstraintAtom(
                dim=atom_dim,
                hardness="hard",
                anchor=f"day:{anchor_day}" if anchor_day else None,
                params=item,
                source="constraints.temporal",
            )
        )

    spatial = constraints.get("spatial", {})
    if isinstance(spatial, dict):
        for key, value in spatial.items():
            if key == "city":
                continue
            atoms.append(
                ConstraintAtom(
                    dim="spatial",
                    hardness="hard",
                    params={key: value},
                    source="constraints.spatial",
                )
            )

    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        for key, value in resource.items():
            atoms.append(
                ConstraintAtom(
                    dim="resource",
                    hardness="hard",
                    params={key: value},
                    source="constraints.resource",
                )
            )

    must_include = constraints.get("must_include", [])
    if isinstance(must_include, list) and must_include:
        default_dim = "sequence" if explicit_sequence else "structure"
        for poi in must_include:
            if isinstance(poi, dict):
                poi_name = poi.get("name")
                poi_day = poi_day_requirements.get(poi_name) if isinstance(poi_name, str) else None
                anchor_day = poi_day if poi_day is not None else day_anchor
                atoms.append(
                    ConstraintAtom(
                        dim=default_dim,
                        hardness="hard",
                        anchor=f"day:{anchor_day}" if anchor_day else None,
                        params={"poi": poi_name},
                        source="constraints.must_include",
                    )
                )

    forbidden = constraints.get("forbidden", [])
    if isinstance(forbidden, list):
        for poi_name in forbidden:
            if isinstance(poi_name, str) and poi_name:
                atoms.append(
                    ConstraintAtom(
                        dim="semantic",
                        hardness="hard",
                        params={"forbidden_poi": poi_name},
                        source="constraints.forbidden",
                    )
                )

    # query-level heuristics
    if explicit_sequence:
        atoms.append(ConstraintAtom(dim="sequence", hardness="hard", source="query.heuristic.sequence"))
    if any(token in edit_query for token in ["每天", "天数", "扩展到", "增加", "多玩"]):
        atoms.append(ConstraintAtom(dim="structure", hardness="hard", source="query.heuristic.structure"))
    if any(token in edit_query for token in ["风格", "主题", "类似", "同类型", "偏"]):
        atoms.append(ConstraintAtom(dim="semantic", hardness="soft", source="query.heuristic.semantic"))

    return atoms


def extract_demand_blocks(constraints: Dict[str, Any], edit_query: str) -> List[DemandBlock]:
    day_anchor = _extract_day_from_text(edit_query)
    time_window = _extract_time_period_window(edit_query)
    blocks: List[DemandBlock] = []
    poi_day_requirements = extract_day_poi_requirements(constraints, edit_query)

    for item in _iter_temporal_constraints(constraints):
        end_before = _to_minutes(item.get("end_before"))
        start_after = _to_minutes(item.get("start_after"))
        if end_before is not None and start_after is not None:
            time_window = _normalize_time_window(start_after, end_before)
        item_day = _coerce_day_value(item.get("day"))
        if day_anchor is None and item_day is not None:
            day_anchor = item_day

    must_include = constraints.get("must_include", [])
    if isinstance(must_include, list):
        for poi in must_include:
            if isinstance(poi, dict):
                poi_name = poi.get("name")
                block_day = poi_day_requirements.get(poi_name) if isinstance(poi_name, str) else day_anchor
                blocks.append(
                    DemandBlock(
                        required_duration=90,
                        time_window=time_window,
                        day=block_day,
                        name=poi_name,
                    )
                )

    if not blocks and time_window is not None and ("安排" in edit_query or "参观" in edit_query):
        blocks.append(
            DemandBlock(
                required_duration=60,
                time_window=time_window,
                day=day_anchor,
                name=None,
            )
        )

    return blocks


def _collect_day_occupied_intervals(origin_plan: Dict[str, Any], day: int) -> List[Tuple[int, int]]:
    itinerary = origin_plan.get("itinerary", [])
    for day_plan in itinerary:
        if day_plan.get("day") != day:
            continue
        intervals: List[Tuple[int, int]] = []
        for activity in day_plan.get("activities", []):
            start = _to_minutes(activity.get("start_time"))
            end = _to_minutes(activity.get("end_time"))
            if start is None or end is None:
                continue
            if end <= start:
                end += 24 * 60
            intervals.append((start, end))
        return _merge_intervals(intervals)
    return []


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _window_available_minutes(occupied: List[Tuple[int, int]], window: Tuple[int, int]) -> int:
    start, end = window
    if end <= start:
        end += 24 * 60
    window_len = end - start
    occupied_in_window = 0
    for occ_start, occ_end in occupied:
        overlap_start = max(start, occ_start)
        overlap_end = min(end, occ_end)
        if overlap_end > overlap_start:
            occupied_in_window += overlap_end - overlap_start
    return max(0, window_len - occupied_in_window)


def _extract_plan_pois(origin_plan: Dict[str, Any]) -> Set[str]:
    poi_set: Set[str] = set()
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            for key in ("position", "start", "end"):
                value = activity.get(key)
                if isinstance(value, str) and value:
                    poi_set.add(value)
    return poi_set


def _extract_radius_anchor_poi(constraints: Dict[str, Any], edit_query: str) -> Optional[str]:
    must_include = constraints.get("must_include", [])
    if isinstance(must_include, list):
        for poi in must_include:
            if isinstance(poi, dict):
                name = poi.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()

    patterns = [
        r"把(.+?)周边活动半径限制为",
        r"以(.+?)为中心(?:的)?周边活动半径",
        r"围绕(.+?)的活动半径",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        anchor = str(match.group(1)).strip()
        if anchor:
            return anchor
    return None


def _extract_transport_distances(transports: Any) -> List[float]:
    values: List[float] = []
    if not isinstance(transports, list):
        return values
    for seg in transports:
        if not isinstance(seg, dict):
            continue
        dist = seg.get("distance")
        if isinstance(dist, (int, float)):
            values.append(float(dist))
    return values


def _collect_anchor_local_segment_distances(origin_plan: Dict[str, Any], anchor_poi: str) -> List[float]:
    distances: List[float] = []
    itinerary = origin_plan.get("itinerary", [])

    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        if not isinstance(activities, list):
            continue
        for idx, activity in enumerate(activities):
            if not isinstance(activity, dict):
                continue
            position = activity.get("position") or activity.get("end")
            if position != anchor_poi:
                continue

            # 到达 anchor 的路径（通常在当前 activity 的 transports 中）
            distances.extend(_extract_transport_distances(activity.get("transports", [])))

            # 离开 anchor 的路径（通常在下一条 activity 的 transports 中）
            if idx + 1 < len(activities):
                distances.extend(_extract_transport_distances(activities[idx + 1].get("transports", [])))

    if distances:
        return distances

    # 兜底：收集显式以 anchor 为起终点的 transport 片段
    for day_plan in itinerary:
        activities = day_plan.get("activities", [])
        if not isinstance(activities, list):
            continue
        for activity in activities:
            if not isinstance(activity, dict):
                continue
            transports = activity.get("transports", [])
            if not isinstance(transports, list):
                continue
            for seg in transports:
                if not isinstance(seg, dict):
                    continue
                if seg.get("start") == anchor_poi or seg.get("end") == anchor_poi:
                    dist = seg.get("distance")
                    if isinstance(dist, (int, float)):
                        distances.append(float(dist))
    return distances


def _max_plan_transport_segment_distance(origin_plan: Dict[str, Any]) -> float:
    max_dist = 0.0
    for day_plan in origin_plan.get("itinerary", []):
        activities = day_plan.get("activities", [])
        if not isinstance(activities, list):
            continue
        for activity in activities:
            if not isinstance(activity, dict):
                continue
            for dist in _extract_transport_distances(activity.get("transports", [])):
                if dist > max_dist:
                    max_dist = dist
    return max_dist


def _requires_anchor_radius_structural(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
) -> bool:
    if not isinstance(edit_query, str):
        edit_query = ""
    spatial = constraints.get("spatial", {})
    has_local_radius_limit = (
        isinstance(spatial, dict)
        and isinstance(spatial.get("local_radius_limit"), dict)
    )
    if not has_local_radius_limit and not any(token in edit_query for token in ["半径", "周边", "附近", "围绕"]):
        return False

    threshold = _extract_distance_limit_km(constraints, edit_query)
    if threshold is None or threshold <= 0:
        return False

    anchor_poi = _extract_radius_anchor_poi(constraints, edit_query)
    if not isinstance(anchor_poi, str) or not anchor_poi.strip():
        return False

    local_distances = _collect_anchor_local_segment_distances(origin_plan, anchor_poi.strip())
    if local_distances:
        return max(local_distances) > threshold

    # 无法可靠定位局部链路时，使用全局 segment 作为保守判据
    return _max_plan_transport_segment_distance(origin_plan) > threshold


def _requires_cross_day_rebuild(origin_plan: Dict[str, Any], constraints: Dict[str, Any], edit_query: str) -> bool:
    itinerary = origin_plan.get("itinerary", [])
    day_count = len(itinerary)
    requested_days = _extract_requested_day_count(edit_query)

    if requested_days is not None and requested_days > day_count:
        return True
    if any(token in edit_query for token in ["增加", "扩展到", "多玩几天", "拆分成"]):
        if requested_days and requested_days > day_count:
            return True
    if "第一天" in edit_query and "第二天" in edit_query and day_count < 2:
        return True

    for item in _iter_temporal_constraints(constraints):
        item_day = _coerce_day_value(item.get("day"))
        if item_day and item_day > day_count:
            return True
        for value in item.values():
            if isinstance(value, str):
                day_anchor = _extract_day_from_text(value)
                if day_anchor and day_anchor > day_count:
                    return True
    return False


def _requires_compositional_theme_split(constraints: Dict[str, Any], edit_query: str) -> bool:
    """
    识别“多日主题拆分/风格分化”类需求：
    - query 显式出现第一天+第二天且强调风格/主题差异
    - 或 constraints 中存在跨日 must_visit 且 query 有风格/主题差异信号
    """
    if not isinstance(edit_query, str):
        edit_query = ""

    has_multi_day_text = ("第一天" in edit_query and "第二天" in edit_query)
    theme_tokens = ["风格", "主题", "明显不同", "分别", "各自", "一类", "另一类"]
    has_theme_split_signal = any(token in edit_query for token in theme_tokens)

    if has_multi_day_text and has_theme_split_signal:
        return True

    days_with_must_visit: Set[int] = set()
    for item in _iter_temporal_constraints(constraints):
        day = _coerce_day_value(item.get("day"))
        if day is not None and item.get("must_visit") is not None:
            days_with_must_visit.add(day)

    return len(days_with_must_visit) >= 2 and has_theme_split_signal


def _requires_structural_scope(constraints: Dict[str, Any], edit_query: str) -> bool:
    """
    识别应升级为 structural 的全局/日级约束，不再仅依赖 cross-day rebuild。
    """
    if not isinstance(edit_query, str):
        edit_query = ""

    must_include = constraints.get("must_include", [])
    has_anchor_poi = isinstance(must_include, list) and len(must_include) > 0

    # 1) 全局预算上限
    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        if resource.get("overlap_anchor_poi"):
            return True
        budget_keys = {"budget", "max_budget", "budget_limit", "ticket_budget", "ticket_budget_limit"}
        has_budget_cap = any(key in resource for key in budget_keys)
        has_global_budget_text = any(token in edit_query for token in ["总预算", "总花费", "总价", "所有门票", "门票总价"])
        has_budget_value_in_query = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query) is not None
        if (has_budget_cap and (has_global_budget_text or not has_anchor_poi)) or (
            has_global_budget_text and has_budget_value_in_query
        ):
            return True

    # 2) 全局距离约束
    spatial = constraints.get("spatial", {})
    if isinstance(spatial, dict):
        distance_keys = {"max_walking_distance", "max_distance", "walk_dist"}
        has_distance_cap = any(key in spatial for key in distance_keys)
        has_global_distance_text = any(token in edit_query for token in ["所有景点", "景点之间", "都不能超过", "不想走太远"])
        has_distance_value_in_query = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query) is not None
        if (has_distance_cap and (has_global_distance_text or not has_anchor_poi)) or (
            has_global_distance_text and has_distance_value_in_query
        ):
            return True

    # 3) 日级结束时间上限（例如“确保第N天在XX前结束所有活动”）
    query_day = _extract_day_from_text(edit_query)
    for item in _iter_temporal_constraints(constraints):
        day = _coerce_day_value(item.get("day"))
        has_end_before = any(key in item for key in ("end_before", "must_end_before", "time_limit"))
        if has_end_before and (day is not None or query_day is not None):
            return True
    if ("前结束" in edit_query or "结束所有活动" in edit_query) and "前" in edit_query and query_day is not None:
        return True

    return False


def _requires_structural_operator(edit_query: str, constraints: Dict[str, Any]) -> bool:
    if _has_explicit_sequence_signal(edit_query, constraints):
        return True

    structural_tokens = [
        "删除",
        "移除",
        "替换",
        "换成",
        "改成",
        "新增",
        "插入",
        "加一个",
        "安排到行程中",
    ]
    if any(token in edit_query for token in structural_tokens):
        return True

    forbidden = constraints.get("forbidden", [])
    if isinstance(forbidden, list) and forbidden:
        return True
    return False


def _is_parameter_feasible(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    demand_blocks: List[DemandBlock],
) -> bool:
    if _requires_structural_operator(edit_query, constraints):
        return False

    # 局部半径约束且已超限时，必须删/换/重排活动，parameter 不可行。
    if _requires_anchor_radius_structural(origin_plan, constraints, edit_query):
        return False

    plan_pois = _extract_plan_pois(origin_plan)
    itinerary = origin_plan.get("itinerary", [])
    day_ids = [day_plan.get("day") for day_plan in itinerary if isinstance(day_plan.get("day"), int)]

    # 必须新增新POI时，parameter 不可行
    for block in demand_blocks:
        if block.name and block.name not in plan_pois:
            return False

    # 已存在POI且未指定明确时窗/日期时，不要求额外插入时间块
    for block in demand_blocks:
        if block.name and block.name in plan_pois and block.day is None and block.time_window is None:
            continue

        window = block.time_window or (8 * 60, 22 * 60)
        if block.day is not None:
            occupied = _collect_day_occupied_intervals(origin_plan, block.day)
            available = _window_available_minutes(occupied, window)
            if available < block.required_duration:
                return False
        else:
            can_fit_any = False
            for day in day_ids:
                occupied = _collect_day_occupied_intervals(origin_plan, day)
                available = _window_available_minutes(occupied, window)
                if available >= block.required_duration:
                    can_fit_any = True
                    break
            if not can_fit_any and day_ids:
                return False

    return True


def infer_min_scope(
    origin_plan: Dict[str, Any],
    constraint_atoms: List[ConstraintAtom],
    demand_blocks: List[DemandBlock],
    edit_query: str = "",
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """推断最小可行 scope：parameter / structural / compositional。"""
    constraints = constraints or {}

    # 1) compositional: 跨天重构 / 增天
    if _requires_cross_day_rebuild(origin_plan, constraints, edit_query):
        return "compositional"

    # 2) compositional: 多日主题拆分/风格分化
    if _requires_compositional_theme_split(constraints, edit_query):
        return "compositional"

    # 3) structural: 全局预算/距离、日级结束时限等结构性约束
    if _requires_structural_scope(constraints, edit_query):
        return "structural"

    # 3.5) structural: 局部 anchor 半径约束在原计划中已超限，需结构性编辑
    if _requires_anchor_radius_structural(origin_plan, constraints, edit_query):
        return "structural"

    # 4) parameter: 先尝试纯参数可行性，不可行再升级 scope
    if _is_parameter_feasible(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query,
        demand_blocks=demand_blocks,
    ):
        return "parameter"

    # 5) structural: 需要日内增删替换/重排
    return "structural"


def _extract_distance_limit_km(constraints: Dict[str, Any], edit_query: str) -> Optional[float]:
    spatial = constraints.get("spatial", {})
    if isinstance(spatial, dict):
        for key in ("max_walking_distance", "max_distance", "walk_dist"):
            value = spatial.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                m = re.search(r"(\d+(?:\.\d+)?)", value)
                if m:
                    return float(m.group(1))
    query_match = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query)
    if query_match:
        return float(query_match.group(1))
    return None


def _extract_budget_limit(constraints: Dict[str, Any], edit_query: str) -> Optional[float]:
    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        for key in ("max_budget", "budget", "ticket_budget"):
            value = resource.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                m = re.search(r"(\d+(?:\.\d+)?)", value)
                if m:
                    return float(m.group(1))
    query_match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
    if query_match:
        return float(query_match.group(1))
    return None


def _extract_end_before_limit(constraints: Dict[str, Any], edit_query: str) -> Optional[int]:
    for item in _iter_temporal_constraints(constraints):
        if "end_before" in item:
            parsed = _to_minutes(item.get("end_before"))
            if parsed is not None:
                return parsed
    query_match = re.search(r"(\d{1,2}:\d{2})前", edit_query)
    if query_match:
        return _to_minutes(query_match.group(1))
    return None


def _extract_duration_limit_minutes(constraints: Dict[str, Any], edit_query: str) -> Optional[int]:
    for item in _iter_temporal_constraints(constraints):
        for key in ("duration_limit", "max_duration", "duration"):
            if key in item:
                minutes = _parse_duration_to_minutes(item.get(key))
                if minutes is not None:
                    return minutes

    # 支持“限制在X小时以内 / 不超过X分钟 / 最多X小时”等表达
    query_match = re.search(
        r"(?:限制在|不超过|最多)\s*(\d+(?:\.\d+)?)\s*(小时|分钟)",
        edit_query
    )
    if not query_match:
        query_match = re.search(r"(\d+(?:\.\d+)?)\s*(小时|分钟)\s*以内", edit_query)
    if query_match:
        amount = float(query_match.group(1))
        unit = query_match.group(2)
        return int(amount * 60) if unit == "小时" else int(amount)
    return None


def _extract_target_poi_name(constraints: Dict[str, Any], edit_query: str) -> Optional[str]:
    source_poi = _extract_replace_source_poi_from_query(edit_query)
    if source_poi:
        return source_poi

    must_include = constraints.get("must_include", [])
    if isinstance(must_include, list):
        for poi in must_include:
            if isinstance(poi, dict):
                name = poi.get("name")
                if isinstance(name, str) and name:
                    return name

    match = re.search(r"把(.+?)的游玩时间限制在", edit_query)
    if match:
        return match.group(1).strip()
    return None


def _extract_poi_duration_minutes(origin_plan: Dict[str, Any]) -> Dict[str, int]:
    durations: Dict[str, int] = {}
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            poi_name = activity.get("position") or activity.get("end")
            if not isinstance(poi_name, str) or not poi_name:
                continue
            start = _to_minutes(activity.get("start_time"))
            end = _to_minutes(activity.get("end_time"))
            if start is None or end is None:
                continue
            if end <= start:
                end += 24 * 60
            duration = end - start
            if duration > durations.get(poi_name, 0):
                durations[poi_name] = duration
    return durations


def detect_violation_forms(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    constraint_atoms: List[ConstraintAtom],
    demand_blocks: List[DemandBlock],
    llm_violation_set: Optional[Set[str]] = None,
) -> Set[str]:
    """检测 violation forms，重点覆盖 overlap / overflow。"""
    forms: Set[str] = set()
    itinerary = origin_plan.get("itinerary", [])

    # overflow: 结束时间超限
    end_before_limit = _extract_end_before_limit(constraints, edit_query)
    if end_before_limit is not None:
        for day_plan in itinerary:
            activities = day_plan.get("activities", [])
            if not activities:
                continue
            last_end = _to_minutes(activities[-1].get("end_time"))
            if last_end is not None and last_end > end_before_limit:
                forms.add("overflow")
                break

    # overflow: 距离/资源阈值超限
    distance_limit = _extract_distance_limit_km(constraints, edit_query)
    if distance_limit is not None:
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                transports = activity.get("transports", [])
                if not isinstance(transports, list):
                    continue
                for segment in transports:
                    if not isinstance(segment, dict):
                        continue
                    distance = segment.get("distance")
                    if isinstance(distance, (int, float)) and distance > distance_limit:
                        forms.add("overflow")
                        break
                if "overflow" in forms:
                    break
            if "overflow" in forms:
                break

    budget_limit = _extract_budget_limit(constraints, edit_query)
    if budget_limit is not None:
        total_cost = 0.0
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                cost = activity.get("cost")
                if isinstance(cost, (int, float)):
                    total_cost += float(cost)
        if total_cost > budget_limit:
            forms.add("overflow")
            resource = constraints.get("resource", {})
            if isinstance(resource, dict) and resource.get("overlap_anchor_poi"):
                forms.add("overlap")

    # overflow: 结构性容量超限（目标天数/每日景点上限）
    requested_days = _extract_requested_day_count(edit_query)
    if requested_days is not None and requested_days > len(itinerary):
        forms.add("overflow")

    daily_poi_cap = _extract_daily_poi_cap(edit_query)
    if daily_poi_cap is not None and daily_poi_cap >= 0:
        for day_plan in itinerary:
            if _count_attractions_in_day(day_plan) > daily_poi_cap:
                forms.add("overflow")
                break
    if _requires_cross_day_rebuild(origin_plan, constraints, edit_query):
        forms.add("discontinuity")

    # overflow: 单POI时长阈值超限（duration_limit）
    duration_limit = _extract_duration_limit_minutes(constraints, edit_query)
    if duration_limit is not None:
        poi_durations = _extract_poi_duration_minutes(origin_plan)
        target_poi = _extract_target_poi_name(constraints, edit_query)
        if target_poi and target_poi in poi_durations:
            if poi_durations[target_poi] > duration_limit:
                forms.add("overflow")
        elif any(duration > duration_limit for duration in poi_durations.values()):
            forms.add("overflow")

    # overlap: 需求块无法在可用时间窗内容纳
    if demand_blocks:
        plan_pois = _extract_plan_pois(origin_plan)
        day_ids = [dp.get("day") for dp in itinerary if isinstance(dp.get("day"), int)]
        for block in demand_blocks:
            if block.name and block.name in plan_pois and block.day is None and block.time_window is None:
                continue
            window = block.time_window or (8 * 60, 22 * 60)
            if block.day is not None:
                occupied = _collect_day_occupied_intervals(origin_plan, block.day)
                if _window_available_minutes(occupied, window) < block.required_duration:
                    forms.add("overlap")
                    break
            else:
                can_fit_any = False
                for day in day_ids:
                    occupied = _collect_day_occupied_intervals(origin_plan, day)
                    if _window_available_minutes(occupied, window) >= block.required_duration:
                        can_fit_any = True
                        break
                if not can_fit_any and day_ids:
                    forms.add("overlap")
                    break

    # discontinuity: 软语义一致性冲突
    if any(atom.hardness == "soft" for atom in constraint_atoms):
        forms.add("discontinuity")
    elif llm_violation_set and "discontinuity" in llm_violation_set:
        forms.add("discontinuity")

    return forms


def detect_dimensions(constraint_atoms: List[ConstraintAtom], edit_query: str) -> Set[str]:
    dims = {atom.dim for atom in constraint_atoms if atom.dim in DIMENSION_ORDER}

    has_spatial_query_signal = bool(
        re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query)
        and any(token in edit_query for token in ["距离", "步行", "半径", "附近"])
    )
    has_resource_query_signal = bool(
        re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
        and any(token in edit_query for token in ["预算", "花费", "费用", "门票", "总价"])
    )

    if re.search(r"先.+再.+", edit_query) or "顺序" in edit_query:
        dims.add("sequence")
    if any(token in edit_query for token in ["时间", "时长", "小时", "分钟", "前结束"]):
        dims.add("temporal")
    if has_spatial_query_signal:
        dims.add("spatial")
    if has_resource_query_signal:
        dims.add("resource")
    if any(token in edit_query for token in ["每天", "天数", "扩展到", "增加", "天行程", "第一天", "第二天", "拆成"]):
        dims.add("structure")

    # 不再把未知维度强行回落到 temporal，避免伪造 temporal 冲突。
    if not dims:
        dims.add("unknown")
    return dims


def _label_key(label: Tuple[str, str, str]) -> Tuple[str, str, str]:
    return (label[0], label[1], label[2])


def _dim_rank(dim: str) -> int:
    return DIMENSION_ORDER.get(dim, 0)


def _violation_rank(violation: str) -> int:
    return VIOLATION_ORDER.get(violation, 0)


def resolve_primary_conflict(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    conflict_labels_llm: List[List[str]],
    target_bucket: Optional[Any] = None,
    purity_threshold: float = 0.7,
    constraint_type: str = "",
) -> Dict[str, Any]:
    """
    根据 rule-based 规则唯一化主冲突。
    """
    purity_threshold = max(0.0, min(1.0, float(purity_threshold)))

    normalized_llm_set: Set[Tuple[str, str, str]] = set()
    dropped_invalid_conflict_labels: List[Any] = []
    for label in conflict_labels_llm or []:
        normalized = normalize_conflict_label(label)
        if normalized:
            normalized_llm_set.add(normalized)
        else:
            dropped_invalid_conflict_labels.append(label)
    llm_violation_set = {label[2] for label in normalized_llm_set}

    atoms = extract_constraint_atoms(constraints, edit_query)
    demand_blocks = extract_demand_blocks(constraints, edit_query)
    inferred_scope = infer_min_scope(
        origin_plan=origin_plan,
        constraint_atoms=atoms,
        demand_blocks=demand_blocks,
        edit_query=edit_query,
        constraints=constraints,
    )
    dimensions = detect_dimensions(atoms, edit_query)
    violations = detect_violation_forms(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_query=edit_query,
        constraint_atoms=atoms,
        demand_blocks=demand_blocks,
        llm_violation_set=llm_violation_set,
    )

    derived_set = {(inferred_scope, dim, vio) for dim in dimensions for vio in violations} if violations else set()
    conflict_pool = sorted(normalized_llm_set | derived_set, key=_label_key)

    normalized_target = normalize_conflict_label(target_bucket) if target_bucket is not None else None
    acceptable_alternatives = ACCEPTABLE_PRIMARY_ALTERNATIVES.get(str(constraint_type or "").strip(), set())
    projected_target, target_projection_reason = _project_target_bucket(
        constraints=constraints,
        edit_query=edit_query,
        constraint_type=constraint_type,
        target_bucket=normalized_target,
        conflict_pool=conflict_pool,
    )
    if not conflict_pool and projected_target is not None:
        conflict_pool = [projected_target]

    if not conflict_pool:
        match_type = "mismatch"
        return {
            "conflict_set": [],
            "primary_conflict": None,
            "secondary_conflicts": [],
            "purity_score_rule": 0.0,
            "match_type_rule": match_type,
            "trace": {
                "reason": "empty_conflict_pool",
                "inferred_scope": inferred_scope,
                "dropped_invalid_conflict_labels": dropped_invalid_conflict_labels[:10],
                "target_projection_applied": False,
                "projected_target_bucket": None,
                "target_projection_reason": None,
            },
            "contains_target": False,
            "primary_is_target": False,
            "target_confirmed": False,
        }

    scope_for_filter = inferred_scope
    protected_hard_labels: List[Tuple[str, str, str]] = []

    k1 = [k for k in conflict_pool if k[0] == scope_for_filter]
    if not k1:
        k1 = list(conflict_pool)

    hard_subset = [k for k in k1 if k[2] in HARD_VIOLATIONS]
    k2 = hard_subset if hard_subset else k1

    # unknown 维度不允许直接成为 primary 候选
    k2_primary = [k for k in k2 if k[1] != "unknown"]
    unknown_filtered = len(k2) - len(k2_primary)
    if not k2_primary:
        non_unknown_pool = [k for k in conflict_pool if k[1] != "unknown"]
        if non_unknown_pool:
            k2_primary = non_unknown_pool
        else:
            return {
                "conflict_set": [_normalize_output_label(k) for k in conflict_pool],
                "primary_conflict": None,
                "secondary_conflicts": [],
                "purity_score_rule": 0.0,
                "match_type_rule": "mismatch",
                "trace": {
                    "reason": "unknown_only_conflict_pool",
                    "inferred_scope": inferred_scope,
                    "scope_for_filter": scope_for_filter,
                    "candidate_sizes": {
                        "k_all": len(conflict_pool),
                        "k_scope": len(k1),
                        "k_hard": len(k2),
                        "k_dim": 0,
                        "k_violation": 0,
                    },
                    "unknown_filtered": unknown_filtered,
                    "detected_dimensions": sorted(dimensions),
                    "detected_violations": sorted(violations),
                    "normalized_target": list(normalized_target) if normalized_target else None,
                    "dropped_invalid_conflict_labels": dropped_invalid_conflict_labels[:10],
                    "target_projection_applied": False,
                    "projected_target_bucket": None,
                    "target_projection_reason": None,
                },
                "contains_target": normalized_target in conflict_pool if normalized_target else False,
                "primary_is_target": False,
                "target_confirmed": False,
            }

    best_dim_rank = max(_dim_rank(k[1]) for k in k2_primary)
    k3 = [k for k in k2_primary if _dim_rank(k[1]) == best_dim_rank]

    best_v_rank = max(_violation_rank(k[2]) for k in k3)
    k4 = [k for k in k3 if _violation_rank(k[2]) == best_v_rank]

    if (
        normalized_target
        and normalized_target in conflict_pool
        and normalized_target[2] in ALLOWED_EDIT_VIOLATIONS
        and normalized_target[0] == scope_for_filter
    ):
        primary = normalized_target
        tie_break = "target_present_priority"
    elif (
        str(constraint_type or "").strip() == "resource_overlap"
        and projected_target is not None
        and normalized_target is not None
        and projected_target == normalized_target
        and normalized_target[0] == scope_for_filter
        and normalized_target[2] in ALLOWED_EDIT_VIOLATIONS
    ):
        primary = normalized_target
        tie_break = "projected_target_priority"
    elif normalized_target and normalized_target in k4:
        primary = normalized_target
        tie_break = "target_bucket"
    else:
        primary = sorted(k4, key=_label_key)[0]
        tie_break = "lexicographic"

    final_conflict_pool = list(conflict_pool)
    if projected_target is not None and projected_target not in final_conflict_pool:
        final_conflict_pool = sorted(final_conflict_pool + [projected_target], key=_label_key)

    secondary = [k for k in final_conflict_pool if k != primary]

    ambiguity = len(k4)
    if len(conflict_pool) <= 1:
        purity = 1.0
    else:
        ambiguity_penalty = (ambiguity - 1) / (len(conflict_pool) - 1)
        scope_focus = len(k1) / len(conflict_pool)
        purity = max(0.0, min(1.0, 0.6 * (1.0 - ambiguity_penalty) + 0.4 * scope_focus))
    purity = round(float(purity), 4)

    contains_target = normalized_target in final_conflict_pool if normalized_target else False
    primary_is_target = (primary == normalized_target) if normalized_target else False

    if normalized_target is None:
        match_type = "weak"
    elif primary_is_target and purity >= purity_threshold:
        match_type = "strong"
    elif contains_target or primary_is_target:
        match_type = "weak"
    else:
        match_type = "mismatch"

    if match_type not in ALLOWED_MATCH_TYPES:
        match_type = "mismatch"

    target_confirmed = bool(
        normalized_target is not None
        and contains_target
        and primary_is_target
        and match_type == "strong"
        and purity >= purity_threshold
    )

    return {
        "conflict_set": [_normalize_output_label(k) for k in final_conflict_pool],
        "primary_conflict": _normalize_output_label(primary),
        "secondary_conflicts": [_normalize_output_label(k) for k in secondary],
        "purity_score_rule": purity,
        "match_type_rule": match_type,
        "trace": {
            "inferred_scope": inferred_scope,
            "scope_for_filter": scope_for_filter,
            "candidate_sizes": {
                "k_all": len(conflict_pool),
                "k_scope": len(k1),
                "k_hard": len(k2),
                "k_primary_candidates": len(k2_primary),
                "k_dim": len(k3),
                "k_violation": len(k4),
            },
            "unknown_filtered": unknown_filtered,
            "protected_hard_labels": [_normalize_output_label(k) for k in protected_hard_labels],
            "constraint_atoms": [asdict(a) for a in atoms],
            "demand_blocks": [asdict(b) for b in demand_blocks],
            "detected_dimensions": sorted(dimensions),
            "detected_violations": sorted(violations),
            "tie_break": tie_break,
            "primary_selection_reason": tie_break,
            "normalized_target": list(normalized_target) if normalized_target else None,
            "target_bucket_present": normalized_target in final_conflict_pool if normalized_target else False,
            "acceptable_alternative_buckets": [_normalize_output_label(k) for k in sorted(acceptable_alternatives)],
            "dropped_invalid_conflict_labels": dropped_invalid_conflict_labels[:10],
            "target_projection_applied": projected_target is not None and projected_target not in conflict_pool,
            "projected_target_bucket": _normalize_output_label(projected_target) if projected_target is not None else None,
            "target_projection_reason": target_projection_reason,
        },
        "contains_target": contains_target,
        "primary_is_target": primary_is_target,
        "target_confirmed": target_confirmed,
    }
