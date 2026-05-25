"""
Deterministic conflict feasibility checks for data generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHINATRAVEL_PATH = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
UTILS_PATH = PROJECT_ROOT / "data_generation" / "utils"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(UTILS_PATH))
if str(CHINATRAVEL_PATH) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_PATH))

try:
    from data_generation.utils.query_surface_parsing import (
        extract_city_split_targets_from_query,
        extract_extend_with_pacing_spec,
        extract_intercity_transport_modes_from_query,
        extract_preferred_poi_type_from_query,
        extract_requested_day_count_from_query,
        extract_requested_day_delta_from_query,
        extract_transport_time_windows_from_query,
        normalize_intercity_transport_mode,
        normalize_preferred_poi_type,
    )
except ModuleNotFoundError:
    from query_surface_parsing import (
        extract_city_split_targets_from_query,
        extract_extend_with_pacing_spec,
        extract_intercity_transport_modes_from_query,
        extract_preferred_poi_type_from_query,
        extract_requested_day_count_from_query,
        extract_requested_day_delta_from_query,
        extract_transport_time_windows_from_query,
        normalize_intercity_transport_mode,
        normalize_preferred_poi_type,
    )

try:
    from data_generation.utils.commute_cap_feasibility import assess_commute_cap_insertion_feasibility
except ModuleNotFoundError:
    from commute_cap_feasibility import assess_commute_cap_insertion_feasibility


CHINESE_DAY_MAP = {
    "第一天": 1,
    "第二天": 2,
    "第三天": 3,
    "第四天": 4,
    "第五天": 5,
    "第六天": 6,
    "第七天": 7,
}

TIME_PERIOD_WINDOWS = {
    "上午": (9 * 60, 12 * 60),
    "下午": (12 * 60, 18 * 60),
    "晚上": (18 * 60, 22 * 60),
}
INTERCITY_TRANSPORT_ACTIVITY_TYPES = {
    "train",
    "airplane",
    "flight",
    "bus",
    "coach",
    "ferry",
    "ship",
}


@dataclass
class FeasibilityResult:
    ok: bool
    reason_code: str
    reason: str
    evidence_used: Dict[str, Any]
    repairable: bool
    suggested_action: str


FACTUAL_INVALID_REASON_CATEGORY = {
    "missing_validator": "unsupported_validation",
    "missing_constraint_type": "unsupported_validation",
    "missing_threshold": "missing_binding_or_threshold",
    "missing_day_binding": "missing_binding_or_threshold",
    "missing_time_period": "missing_binding_or_threshold",
    "missing_poi_anchor": "missing_binding_or_threshold",
    "missing_binding": "missing_binding_or_threshold",
    "missing_split_signal": "missing_binding_or_threshold",
    "missing_semantic_target": "missing_binding_or_threshold",
    "missing_preferred_type": "missing_binding_or_threshold",
    "missing_required_attraction_type": "ungrounded_reference",
    "missing_forbidden": "missing_binding_or_threshold",
    "must_include_not_empty": "invalid_constraint_state",
    "poi_not_found": "ungrounded_reference",
    "invalid_day": "ungrounded_reference",
    "missing_activity_time": "ungrounded_reference",
    "missing_activity_order": "ungrounded_reference",
    "period_not_grounded": "ungrounded_reference",
    "baseline_not_exceed_threshold": "already_satisfied_or_not_stricter",
    "already_satisfies_window": "already_satisfied_or_not_stricter",
    "already_satisfies_order": "already_satisfied_or_not_stricter",
    "already_compactable": "already_satisfied_or_not_stricter",
    "day_count_not_stricter": "already_satisfied_or_not_stricter",
    "window_has_capacity": "already_satisfied_or_not_stricter",
    "already_matches_day_split": "already_satisfied_or_not_stricter",
    "not_stricter_than_plan": "already_satisfied_or_not_stricter",
}

FACTUAL_ERROR_REASON_CODES = {
    "missing_validator",
    "missing_constraint_type",
    "missing_threshold",
    "missing_day_binding",
    "missing_time_period",
    "missing_poi_anchor",
    "missing_binding",
    "missing_split_signal",
    "missing_semantic_target",
    "missing_preferred_type",
    "missing_required_attraction_type",
    "missing_forbidden",
    "must_include_not_empty",
    "poi_not_found",
    "invalid_day",
    "missing_activity_time",
    "missing_activity_order",
    "period_not_grounded",
}

FACTUAL_VALID_NON_CONFLICTING_REASON_CODES = {
    "baseline_not_exceed_threshold",
    "already_satisfies_window",
    "already_satisfies_order",
    "already_compactable",
    "day_count_not_stricter",
    "window_has_capacity",
    "already_matches_day_split",
    "not_stricter_than_plan",
}


Validator = Callable[[Dict[str, Any], Dict[str, Any], str, Dict[str, Any], Optional[str]], FeasibilityResult]


def make_result(
    ok: bool,
    reason_code: str,
    reason: str,
    evidence_used: Optional[Dict[str, Any]] = None,
    repairable: bool = True,
    suggested_action: str = "retry_same_template",
) -> FeasibilityResult:
    return FeasibilityResult(
        ok=ok,
        reason_code=reason_code,
        reason=reason,
        evidence_used=evidence_used or {},
        repairable=repairable,
        suggested_action=suggested_action,
    )


def categorize_factual_invalid_reason(reason_code: str) -> str:
    normalized = str(reason_code or "").strip()
    if not normalized:
        return "unknown"
    return FACTUAL_INVALID_REASON_CATEGORY.get(normalized, "unknown")


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


def _parse_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    return None


def _normalize_day_value(day_value: Any) -> Optional[int]:
    if isinstance(day_value, int) and day_value > 0:
        return day_value
    if not isinstance(day_value, str):
        return None
    day_value = day_value.strip()
    if not day_value:
        return None
    if day_value in CHINESE_DAY_MAP:
        return CHINESE_DAY_MAP[day_value]
    match = re.search(r"第([一二三四五六七八九十\d]+)天", day_value)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    chinese_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return chinese_map.get(token)


def _extract_query_days(edit_query: str) -> List[int]:
    if not isinstance(edit_query, str):
        return []
    days: List[int] = []
    for token in re.findall(r"第([一二三四五六七八九十\d]+)天", edit_query):
        day = _normalize_day_value(f"第{token}天")
        if day is not None and day not in days:
            days.append(day)
    return days


def _extract_day_poi_pairs(edit_query: str) -> List[Tuple[int, str]]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    pairs: List[Tuple[int, str]] = []
    pattern = re.compile(r"第([一二三四五六七八九十\d]+)天(?:去|游玩|参观|安排|打卡)([^，。；;（(]+)")
    for match in pattern.finditer(edit_query):
        day = _normalize_day_value(f"第{match.group(1)}天")
        poi_name = str(match.group(2)).strip()
        if day is not None and poi_name:
            pairs.append((day, poi_name))
    return pairs


def _extract_time_period(edit_query: str, constraints: Dict[str, Any]) -> Optional[str]:
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, dict):
        temporal = [temporal]
    if isinstance(temporal, list):
        for item in temporal:
            if isinstance(item, dict):
                time_period = item.get("time_period")
                if isinstance(time_period, str) and time_period in TIME_PERIOD_WINDOWS:
                    return time_period
    if not isinstance(edit_query, str):
        return None
    for token in TIME_PERIOD_WINDOWS:
        if token in edit_query:
            return token
    return None


def _extract_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str):
        return None
    patterns = [
        r"把([^，。；;]+?)的时间安排在",
        r"把([^，。；;]+?)的游玩时间限制在",
        r"我想在(?:第[一二三四五六七八九十\d]+天)?去([^，。；;]+?)，但如果门票",
        r"把([^，。；;]+?)从行程中删除",
        r"把([^，。；;]+?)从行程里去掉",
        r"把([^，。；;]+?)换成",
        r"把([^，。；;]+?)改成",
        r"去掉([^，。；;]+?)，换成",
        r"安排([^，。；;]+?)和附近的餐厅",
        r"(?:想逛|想去|安排)([^，。；;]+?)顺便在附近(?:吃饭|用餐)",
        r"(?:想逛|想去|安排)([^，。；;]+?)并在附近(?:吃饭|用餐)",
        r"增加([^，。；;]+?)到行程中",
        r"我想在(?:第[一二三四五六七八九十\d]+天)?游玩([^，。；;]+?)，但总预算",
        r"我想在(?:第[一二三四五六七八九十\d]+天)?游玩([^，。；;]+?)，但不想走太远",
        r"把([^，。；;]+?)周边活动半径限制为",
        r"特别是([^，。；;]+)",
        r"([^，。；;]+?)就别去了吧",
        r"([^，。；;]+?)不去了吧",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if match:
            name = str(match.group(1)).strip()
            if name:
                return name
    return None


def _extract_two_pois(edit_query: str) -> List[str]:
    if not isinstance(edit_query, str):
        return []
    patterns = [
        r"先去([^，。；;]+?)再去([^，。；;]+)",
        r"把([^和，。；;]+?)和([^，。；;]+?)安排得更紧凑",
        r"把([^和，。；;]+?)和([^，。；;]+?)安排在同一",
        r"把([^和，。；;]+?)和([^，。；;]+?)之间的交通方式改为步行",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if match:
            names = [str(match.group(1)).strip(), str(match.group(2)).strip()]
            if all(names):
                return names
    return []


def _extract_preferred_type(edit_query: str, constraints: Dict[str, Any]) -> Optional[str]:
    semantic = constraints.get("semantic", {})
    if isinstance(semantic, dict):
        for key in ("preferred_type", "semantic_target"):
            value = semantic.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_preferred_poi_type(value) or value.strip()
    return normalize_preferred_poi_type(extract_preferred_poi_type_from_query(edit_query or ""))


@lru_cache(maxsize=32)
def _city_attraction_types(city: str) -> Tuple[str, ...]:
    if not city:
        return ()
    try:
        from chinatravel.environment.tools.attractions.apis import Attractions

        types = Attractions().get_type_list(city)
    except Exception:
        return ()
    normalized = []
    for item in types:
        value = normalize_preferred_poi_type(item) or str(item or "").strip()
        if value and value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _preferred_type_supported_by_city(preferred_type: Optional[str], city: Any) -> bool:
    preferred = normalize_preferred_poi_type(preferred_type) or str(preferred_type or "").strip()
    if not preferred:
        return False
    city_types = _city_attraction_types(str(city or "").strip())
    return not city_types or preferred in set(city_types)


def _extract_semantic_target(edit_query: str, constraints: Dict[str, Any]) -> Optional[str]:
    semantic = constraints.get("semantic", {})
    if isinstance(semantic, dict):
        value = semantic.get("semantic_target")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if not isinstance(edit_query, str):
        return None
    patterns = [
        r"(?:改成|换成|替换为)(.+?)类型",
        r"(?:改成|换成)(.+?)的其他景点",
        r"换成(.+?)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if match:
            value = str(match.group(1)).strip()
            if value:
                return value
    return None


def _extract_budget_threshold(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[float]:
    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        for key in ("max_budget", "budget", "budget_cap", "budget_target", "ticket_budget_limit", "ticket_budget", "ticket_price_cap"):
            value = resource.get(key)
            parsed = _parse_numeric(value)
            if parsed is not None:
                return parsed
    parsed = _parse_numeric(fallback_params.get("budget"))
    if parsed is not None:
        return parsed
    match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query or "")
    if match:
        return float(match.group(1))
    return None


def _extract_activity_budget_target(constraints: Dict[str, Any], edit_query: str) -> Tuple[Optional[str], Optional[str]]:
    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        activity_type = resource.get("activity_type")
        metric = resource.get("metric")
        if isinstance(activity_type, str) and activity_type.strip():
            return activity_type.strip(), str(metric).strip() if metric is not None else None

    if "住宿" in edit_query or "酒店" in edit_query:
        return "accommodation", "avg_cost_per_person_per_night"
    if "餐" in edit_query or "餐饮" in edit_query or "用餐" in edit_query or "吃" in edit_query:
        return "meal", "avg_cost_per_meal_per_person"
    return None, None


def _extract_room_count(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[int]:
    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        for key in ("room_count", "rooms"):
            parsed = _parse_numeric(resource.get(key))
            if parsed is not None:
                return int(parsed)
    parsed = _parse_numeric(fallback_params.get("room_count"))
    if parsed is not None:
        return int(parsed)
    match = re.search(r"(\d+)\s*间(?:房|客房)", edit_query or "")
    if match:
        return int(match.group(1))
    return None


def _extract_room_type(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[int]:
    resource = constraints.get("resource", {})
    if isinstance(resource, dict):
        parsed = _parse_numeric(resource.get("room_type"))
        if parsed is not None:
            return int(parsed)
    parsed = _parse_numeric(fallback_params.get("room_type"))
    if parsed is not None:
        return int(parsed)
    alias_map = {
        "大床房": 1,
        "单床房": 1,
        "单人间": 1,
        "双床房": 2,
        "标间": 2,
        "标准间": 2,
        "双人间": 2,
    }
    for label, value in alias_map.items():
        if label in (edit_query or ""):
            return value
    match = re.search(r"房型(?:改成|调整为|设置为|要)?\s*(\d+)", edit_query or "")
    if match:
        return int(match.group(1))
    return None


def _extract_restaurant_type(constraints: Dict[str, Any], edit_query: str) -> Optional[str]:
    semantic = constraints.get("semantic", {})
    if isinstance(semantic, dict):
        preferred = semantic.get("preferred_type") or semantic.get("required_restaurant_type")
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
    patterns = [
        r"我想吃([^，。；;]{2,24})",
        r"想吃([^，。；;]{2,24})",
        r"吃([^，。；;]{2,24})菜",
        r"([^，。；;]{2,24})餐厅",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if match:
            value = str(match.group(1)).strip()
            if value:
                return value
    return None


def _extract_intercity_transport_modes(
    constraints: Dict[str, Any],
    fallback_params: Dict[str, Any],
    edit_query: str,
) -> List[str]:
    raw_values: List[Any] = []
    transport = constraints.get("transport", {})
    if isinstance(transport, dict):
        raw = transport.get("intercity_modes")
        raw_values.extend(raw if isinstance(raw, list) else [raw])
    semantic = constraints.get("semantic", {})
    if isinstance(semantic, dict):
        raw = semantic.get("intercity_transport_modes")
        raw_values.extend(raw if isinstance(raw, list) else [raw])
    raw = fallback_params.get("intercity_transport_modes")
    raw_values.extend(raw if isinstance(raw, list) else [raw])
    raw_values.extend(extract_intercity_transport_modes_from_query(edit_query))

    modes: List[str] = []
    for value in raw_values:
        mode = normalize_intercity_transport_mode(value)
        if mode and mode not in modes:
            modes.append(mode)
    return modes


def _extract_transport_time_windows(
    constraints: Dict[str, Any],
    fallback_params: Dict[str, Any],
    edit_query: str,
) -> List[Dict[str, str]]:
    windows: List[Dict[str, str]] = []
    transport = constraints.get("transport", {})
    if isinstance(transport, dict) and isinstance(transport.get("time_windows"), list):
        windows.extend(item for item in transport["time_windows"] if isinstance(item, dict))
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, list):
        for item in temporal:
            if isinstance(item, dict) and isinstance(item.get("transport_time_window"), dict):
                windows.append(item["transport_time_window"])
    raw = fallback_params.get("transport_time_windows")
    if isinstance(raw, list):
        windows.extend(item for item in raw if isinstance(item, dict))
    windows.extend(extract_transport_time_windows_from_query(edit_query))

    normalized: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for item in windows:
        leg = str(item.get("leg") or "").strip()
        field = str(item.get("field") or "").strip()
        operator = str(item.get("operator") or "").strip() or "<="
        value = str(item.get("value") or "").strip()
        if leg not in {"outbound", "return"} or field not in {"start_time", "end_time"} or not value:
            continue
        key = (leg, field, operator, value)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"leg": leg, "field": field, "operator": operator, "value": value})
    return normalized


def _extract_hotel_feature(constraints: Dict[str, Any], edit_query: str) -> Optional[str]:
    semantic = constraints.get("semantic", {})
    if isinstance(semantic, dict):
        feature = semantic.get("hotel_feature")
        if isinstance(feature, str) and feature.strip():
            return feature.strip()
    patterns = [
        r"住([^，。；;]{2,24}?)酒店",
        r"入住([^，。；;]{2,24}?)酒店",
        r"选([^，。；;]{2,24}?)酒店",
        r"([^，。；;]{2,24}?)酒店",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if match:
            value = str(match.group(1)).strip()
            if value:
                return value
    return None


def _infer_restaurant_type(activity: Dict[str, Any], city: Optional[str]) -> Optional[str]:
    cuisine = activity.get("cuisine")
    if isinstance(cuisine, str) and cuisine.strip():
        return cuisine.strip()
    try:
        from chinatravel.symbol_verification.concept_func import restaurant_type as _restaurant_type

        inferred = _restaurant_type(activity, city or "")
        if isinstance(inferred, str) and inferred.strip() and inferred != "empty":
            return inferred.strip()
    except Exception:
        pass
    return None


def _infer_hotel_feature(activity: Dict[str, Any], city: Optional[str]) -> Optional[str]:
    feature = activity.get("featurehoteltype")
    if isinstance(feature, str) and feature.strip():
        return feature.strip()
    try:
        from chinatravel.symbol_verification.concept_func import accommodation_type as _accommodation_type

        inferred = _accommodation_type(activity, city or "")
        if isinstance(inferred, str) and inferred.strip():
            return inferred.strip()
    except Exception:
        pass
    return None


def _extract_distance_threshold(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[float]:
    spatial = constraints.get("spatial", {})
    if isinstance(spatial, dict):
        for key in ("max_walking_distance", "walk_dist", "max_distance"):
            value = spatial.get(key)
            parsed = _parse_numeric(value)
            if parsed is not None:
                return parsed
    parsed = _parse_numeric(fallback_params.get("walk_dist"))
    if parsed is not None:
        return parsed
    match = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query or "")
    if match:
        return float(match.group(1))
    return None


def _extract_duration_threshold(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[int]:
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, dict):
        temporal = [temporal]
    if isinstance(temporal, list):
        for item in temporal:
            if not isinstance(item, dict):
                continue
            for key in ("duration_limit", "duration", "max_duration"):
                parsed = _parse_duration_to_minutes(item.get(key))
                if parsed is not None:
                    return parsed
    parsed = _parse_duration_to_minutes(fallback_params.get("duration"))
    if parsed is not None:
        return parsed
    match = re.search(r"(?:限制在|不超过|最多)\s*(\d+(?:\.\d+)?)\s*(小时|分钟)", edit_query or "")
    if match:
        amount = float(match.group(1))
        unit = match.group(2)
        return int(amount * 60) if unit == "小时" else int(amount)
    return None


def _extract_end_before_threshold(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[int]:
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, dict):
        temporal = [temporal]
    if isinstance(temporal, list):
        for item in temporal:
            if not isinstance(item, dict):
                continue
            for key in ("end_before", "must_end_before", "time_limit"):
                parsed = _parse_time_to_minutes(item.get(key))
                if parsed is not None:
                    return parsed
    parsed = _parse_time_to_minutes(fallback_params.get("time"))
    if parsed is not None:
        return parsed
    match = re.search(r"(\d{1,2}:\d{2})前", edit_query or "")
    if match:
        return _parse_time_to_minutes(match.group(1))
    return None


def _get_plan_pois(origin_plan: Dict[str, Any]) -> Set[str]:
    poi_names: Set[str] = set()
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            for key in ("position", "start", "end"):
                value = activity.get(key)
                if isinstance(value, str) and value:
                    poi_names.add(value)
    return poi_names


def _get_existing_activity_types(origin_plan: Dict[str, Any]) -> Set[str]:
    kinds: Set[str] = set()
    for day_plan in origin_plan.get("itinerary", []):
        for activity in day_plan.get("activities", []):
            act_type = activity.get("type")
            if isinstance(act_type, str) and act_type:
                kinds.add(act_type)
    return kinds


def _extract_activity_ticket_price(activity: Dict[str, Any]) -> Optional[float]:
    price = activity.get("price")
    if isinstance(price, (int, float)):
        return float(price)
    cost = activity.get("cost")
    tickets = activity.get("tickets")
    if isinstance(cost, (int, float)) and isinstance(tickets, int) and tickets > 0:
        return float(cost) / tickets
    return None


def _extract_transport_distances(transports: Any) -> List[float]:
    distances: List[float] = []
    if not isinstance(transports, list):
        return distances
    for segment in transports:
        if not isinstance(segment, dict):
            continue
        distance = segment.get("distance")
        if isinstance(distance, (int, float)):
            distances.append(float(distance))
    return distances


def _extract_transport_durations(transports: Any) -> List[int]:
    durations: List[int] = []
    if not isinstance(transports, list):
        return durations
    for segment in transports:
        if not isinstance(segment, dict):
            continue
        start = _parse_time_to_minutes(segment.get("start_time"))
        end = _parse_time_to_minutes(segment.get("end_time"))
        if start is None or end is None:
            continue
        # These `transports` entries are local access legs attached to activities.
        # Equal start/end timestamps are noisy data and should not explode to 24h.
        if end == start:
            continue
        if end < start:
            end += 24 * 60
        duration = end - start
        if duration <= 0:
            continue
        # Local feeder legs longer than half a day are almost certainly malformed.
        if duration > 12 * 60:
            continue
        durations.append(duration)
    return durations


def _collect_anchor_local_segment_distances(origin_plan: Dict[str, Any], anchor_poi: str) -> List[float]:
    distances: List[float] = []
    for day_plan in origin_plan.get("itinerary", []):
        activities = day_plan.get("activities", [])
        if not isinstance(activities, list):
            continue
        for idx, activity in enumerate(activities):
            position = activity.get("position") or activity.get("end")
            if position != anchor_poi:
                continue
            distances.extend(_extract_transport_distances(activity.get("transports", [])))
            if idx + 1 < len(activities):
                distances.extend(_extract_transport_distances(activities[idx + 1].get("transports", [])))
    return distances


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _collect_day_occupied_intervals(origin_plan: Dict[str, Any], day: int) -> List[Tuple[int, int]]:
    for day_plan in origin_plan.get("itinerary", []):
        if day_plan.get("day") != day:
            continue
        intervals: List[Tuple[int, int]] = []
        for activity in day_plan.get("activities", []):
            start = _parse_time_to_minutes(activity.get("start_time"))
            end = _parse_time_to_minutes(activity.get("end_time"))
            if start is None or end is None:
                continue
            if end <= start:
                end += 24 * 60
            intervals.append((start, end))
        return _merge_intervals(intervals)
    return []


def _window_available_minutes(occupied: List[Tuple[int, int]], window: Tuple[int, int]) -> int:
    start, end = window
    occupied_minutes = 0
    for occ_start, occ_end in occupied:
        overlap_start = max(start, occ_start)
        overlap_end = min(end, occ_end)
        if overlap_end > overlap_start:
            occupied_minutes += overlap_end - overlap_start
    return max(0, (end - start) - occupied_minutes)


def _find_matching_poi_record(record_map: Dict[str, Dict[str, Any]], poi_name: Optional[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not isinstance(poi_name, str) or not poi_name.strip():
        return None
    name = poi_name.strip()
    exact = record_map.get(name)
    if exact is not None:
        return name, exact
    fuzzy_matches = [(record_name, record) for record_name, record in record_map.items() if name in record_name or record_name in name]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    return None


def _find_activity_type_in_plan(origin_plan: Any, poi_name: Optional[str]) -> Optional[str]:
    if not isinstance(origin_plan, dict) or not isinstance(poi_name, str) or not poi_name.strip():
        return None
    name = poi_name.strip()
    matches: List[str] = []
    for day_plan in origin_plan.get("itinerary", []):
        if not isinstance(day_plan, dict):
            continue
        for activity in day_plan.get("activities", []):
            if not isinstance(activity, dict):
                continue
            poi = activity.get("position") or activity.get("end")
            if not isinstance(poi, str) or not poi.strip():
                continue
            if name == poi or name in poi or poi in name:
                activity_type = str(activity.get("type") or "").strip().lower()
                if activity_type:
                    matches.append(activity_type)
    unique_matches = set(matches)
    return matches[0] if len(unique_matches) == 1 else None


def _flatten_poi_day_entries(entries: Any) -> List[Dict[str, Any]]:
    if isinstance(entries, dict):
        return [entries]
    if isinstance(entries, list):
        return [item for item in entries if isinstance(item, dict)]
    return []


def _is_entry_before(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[bool]:
    left_day = _normalize_day_value(left.get("day"))
    right_day = _normalize_day_value(right.get("day"))
    if left_day is None or right_day is None:
        return None
    if left_day != right_day:
        return left_day < right_day
    left_idx = left.get("activity_idx")
    right_idx = right.get("activity_idx")
    if not isinstance(left_idx, int) or not isinstance(right_idx, int):
        return None
    return left_idx < right_idx


def _entry_fully_in_period(entry: Dict[str, Any], window: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    start = _parse_time_to_minutes(entry.get("start_time"))
    end = _parse_time_to_minutes(entry.get("end_time"))
    if start is None or end is None:
        return None
    if end <= start:
        end += 24 * 60
    start_bound, end_bound = window
    if start >= start_bound and end <= end_bound:
        return start, end
    return None


def build_origin_plan_evidence(origin_plan: Dict[str, Any]) -> Dict[str, Any]:
    poi_duration_by_name: Dict[str, Dict[str, Any]] = {}
    poi_ticket_price_by_name: Dict[str, Dict[str, Any]] = {}
    poi_day_by_name: Dict[str, List[Dict[str, Any]]] = {}
    day_end_by_day: Dict[int, int] = {}
    day_attraction_count: Dict[int, int] = {}
    max_transport_distance_km = 0.0
    max_transport_duration_min = 0
    total_cost = 0.0
    ticket_cost = 0.0
    meal_cost = 0.0
    meal_count = 0
    accommodation_cost = 0.0
    accommodation_count = 0
    intercity_transport_cost = 0.0
    local_transport_cost = 0.0
    intercity_transport_modes: Set[str] = set()
    intercity_time_by_leg: Dict[str, Dict[str, Any]] = {}
    start_city = str(origin_plan.get("start_city") or "")
    target_city = str(origin_plan.get("target_city") or "")

    for day_plan in origin_plan.get("itinerary", []):
        day = day_plan.get("day")
        activities = day_plan.get("activities", [])
        day_latest = None
        attraction_count = 0

        for idx, activity in enumerate(activities):
            cost = activity.get("cost")
            if isinstance(cost, (int, float)):
                total_cost += float(cost)
                if str(activity.get("type") or "").lower() in INTERCITY_TRANSPORT_ACTIVITY_TYPES:
                    intercity_transport_cost += float(cost)
                if activity.get("type") == "attraction":
                    ticket_cost += float(cost)
                if activity.get("type") in {"breakfast", "lunch", "dinner"}:
                    meal_cost += float(cost)
                    meal_count += 1
                if activity.get("type") == "accommodation":
                    accommodation_cost += float(cost)
                    accommodation_count += 1

            if activity.get("type") == "attraction":
                attraction_count += 1

            activity_type = str(activity.get("type") or "").lower()
            if activity_type in INTERCITY_TRANSPORT_ACTIVITY_TYPES:
                normalized_mode = normalize_intercity_transport_mode(activity_type) or activity_type
                intercity_transport_modes.add(normalized_mode)
                start_loc = str(activity.get("start") or "")
                end_loc = str(activity.get("end") or activity.get("position") or "")
                leg = ""
                if start_city and start_city in start_loc:
                    leg = "outbound"
                elif start_city and start_city in end_loc:
                    leg = "return"
                elif target_city and target_city in end_loc:
                    leg = "outbound"
                elif target_city and target_city in start_loc:
                    leg = "return"
                elif not intercity_time_by_leg:
                    leg = "outbound"
                else:
                    leg = "return"
                intercity_time_by_leg.setdefault(
                    leg,
                    {
                        "start_time": activity.get("start_time"),
                        "end_time": activity.get("end_time"),
                        "mode": normalized_mode,
                        "citation": f"itinerary[{max(int(day or 1) - 1, 0)}].activities[{idx}]",
                    },
                )

            start = _parse_time_to_minutes(activity.get("start_time"))
            end = _parse_time_to_minutes(activity.get("end_time"))
            duration = None
            if start is not None and end is not None:
                if end <= start:
                    end += 24 * 60
                duration = end - start
                if day_latest is None or end > day_latest:
                    day_latest = end

            poi_name = activity.get("position") or activity.get("end")
            if isinstance(poi_name, str) and poi_name:
                if duration is not None:
                    prev = poi_duration_by_name.get(poi_name)
                    if not prev or duration > prev.get("duration_min", 0):
                        poi_duration_by_name[poi_name] = {
                            "day": day,
                            "activity_idx": idx,
                            "start_time": activity.get("start_time"),
                            "end_time": activity.get("end_time"),
                            "duration_min": duration,
                            "citation": f"itinerary[{max(int(day or 1) - 1, 0)}].activities[{idx}]",
                            "source_poi": poi_name,
                        }

                activity_type = str(activity.get("type") or "").strip().lower()
                ticket_price = _extract_activity_ticket_price(activity)
                if activity_type == "attraction" and ticket_price is not None and ticket_price >= 0:
                    prev_price = poi_ticket_price_by_name.get(poi_name)
                    if not prev_price or ticket_price > float(prev_price.get("ticket_price", -1)):
                        poi_ticket_price_by_name[poi_name] = {
                            "day": day,
                            "activity_idx": idx,
                            "activity_type": activity_type,
                            "ticket_price": ticket_price,
                            "cost": float(activity.get("cost", 0) or 0),
                            "tickets": activity.get("tickets"),
                            "citation": f"itinerary[{max(int(day or 1) - 1, 0)}].activities[{idx}]",
                            "source_poi": poi_name,
                        }

                poi_day_by_name.setdefault(poi_name, []).append({
                    "day": day,
                    "activity_idx": idx,
                    "start_time": activity.get("start_time"),
                    "end_time": activity.get("end_time"),
                    "citation": f"itinerary[{max(int(day or 1) - 1, 0)}].activities[{idx}]",
                    "source_poi": poi_name,
                })

            transports = activity.get("transports", [])
            if isinstance(transports, list):
                for transport in transports:
                    if not isinstance(transport, dict):
                        continue
                    transport_cost = transport.get("cost")
                    if isinstance(transport_cost, (int, float)):
                        total_cost += float(transport_cost)
                        local_transport_cost += float(transport_cost)
            for distance in _extract_transport_distances(transports):
                max_transport_distance_km = max(max_transport_distance_km, distance)
            for duration_min in _extract_transport_durations(transports):
                max_transport_duration_min = max(max_transport_duration_min, duration_min)

        if isinstance(day, int):
            day_attraction_count[day] = attraction_count
            if day_latest is not None:
                day_end_by_day[day] = day_latest

    itinerary = origin_plan.get("itinerary", [])
    attraction_days = sum(1 for count in day_attraction_count.values() if int(count or 0) > 0)
    local_core_cost = ticket_cost + meal_cost + accommodation_cost
    transferable_local_cost = max(local_core_cost, total_cost - intercity_transport_cost, 0.0)
    intercity_dominant = (
        total_cost > 0
        and intercity_transport_cost >= max(100.0, total_cost * 0.6)
    )
    has_meaningful_local_spend = (
        transferable_local_cost >= 400.0
        or meal_cost >= 150.0
        or accommodation_cost >= 120.0
        or ticket_cost >= 150.0
        or accommodation_count > 0
        or attraction_days >= 2
    )
    budget_baseline_reliable_for_multi_day = not (
        len(itinerary) == 1
        and intercity_dominant
        and not has_meaningful_local_spend
    )
    budget_baseline_reliable_for_total_budget_edit = not (
        len(itinerary) == 1
        and intercity_dominant
        and transferable_local_cost < 120.0
        and accommodation_count == 0
        and meal_cost < 80.0
        and ticket_cost < 80.0
    )
    guard_reason = ""
    if not budget_baseline_reliable_for_multi_day:
        guard_reason = "single_day_intercity_dominant_with_low_local_spend"

    return {
        "poi_duration_by_name": poi_duration_by_name,
        "poi_ticket_price_by_name": poi_ticket_price_by_name,
        "poi_day_by_name": poi_day_by_name,
        "day_end_by_day": day_end_by_day,
        "day_attraction_count": day_attraction_count,
        "total_cost": total_cost,
        "ticket_cost": ticket_cost,
        "meal_cost": meal_cost,
        "meal_count": meal_count,
        "accommodation_cost": accommodation_cost,
        "accommodation_count": accommodation_count,
        "intercity_transport_cost": intercity_transport_cost,
        "intercity_transport_modes": sorted(intercity_transport_modes),
        "intercity_time_by_leg": intercity_time_by_leg,
        "local_transport_cost": local_transport_cost,
        "local_core_cost": local_core_cost,
        "transferable_local_cost": transferable_local_cost,
        "budget_baseline_reliable_for_multi_day": budget_baseline_reliable_for_multi_day,
        "budget_baseline_reliable_for_total_budget_edit": budget_baseline_reliable_for_total_budget_edit,
        "multi_day_budget_guard_reason": guard_reason,
        "max_transport_distance_km": max_transport_distance_km,
        "max_transport_duration_min": max_transport_duration_min,
        "plan_poi_names": sorted(_get_plan_pois(origin_plan)),
        "existing_activity_types": sorted(_get_existing_activity_types(origin_plan)),
        "city": origin_plan.get("target_city"),
        "day_count": len(itinerary),
        "is_single_day": len(itinerary) == 1,
    }


def _extract_must_include_names(constraints: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    must_include = constraints.get("must_include", [])
    if not isinstance(must_include, list):
        return names
    for poi in must_include:
        if isinstance(poi, dict):
            name = poi.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def _extract_forbidden_names(constraints: Dict[str, Any]) -> List[str]:
    forbidden = constraints.get("forbidden", [])
    if not isinstance(forbidden, list):
        return []
    return [name.strip() for name in forbidden if isinstance(name, str) and name.strip()]


def _extract_temporal_items(constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
    temporal = constraints.get("temporal", [])
    if isinstance(temporal, dict):
        return [dict(temporal)]
    if isinstance(temporal, list):
        return [dict(item) for item in temporal if isinstance(item, dict)]
    return []


def _extract_first_must_visit(constraints: Dict[str, Any]) -> Optional[str]:
    for item in _extract_temporal_items(constraints):
        must_visit = item.get("must_visit")
        if isinstance(must_visit, str) and must_visit.strip():
            return must_visit.strip()
        if isinstance(must_visit, list):
            for value in must_visit:
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _extract_replace_from(constraints: Dict[str, Any], edit_query: str) -> Optional[str]:
    semantic = constraints.get("semantic", {})
    if isinstance(semantic, dict):
        replace_from = semantic.get("replace_from")
        if isinstance(replace_from, str) and replace_from.strip():
            return replace_from.strip()
    forbidden = _extract_forbidden_names(constraints)
    if forbidden:
        return forbidden[0]
    return _extract_poi_from_query(edit_query)


def _get_day_binding(constraints: Dict[str, Any], fallback_params: Dict[str, Any], edit_query: str) -> Optional[int]:
    for item in _extract_temporal_items(constraints):
        day = _normalize_day_value(item.get("day"))
        if day is not None:
            return day
    fallback_day = _normalize_day_value(fallback_params.get("day"))
    if fallback_day is not None:
        return fallback_day
    query_days = _extract_query_days(edit_query)
    return query_days[0] if query_days else None


def _validate_duration_limit(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    poi_name = (_extract_must_include_names(constraints) or [fallback_params.get("poi_name")])[0]
    match = _find_matching_poi_record(evidence["poi_duration_by_name"], poi_name)
    if not match:
        return make_result(False, "poi_not_found", f"duration_limit poi '{poi_name}' not found in origin_plan", repairable=True)
    matched_name, matched = match
    threshold = _extract_duration_threshold(constraints, fallback_params, edit_query)
    baseline = matched.get("duration_min")
    if threshold is None or baseline is None:
        return make_result(False, "missing_threshold", "duration_limit threshold missing or invalid", repairable=True)
    if baseline <= threshold:
        return make_result(
            False,
            "baseline_not_exceed_threshold",
            f"duration_limit not conflicting: baseline={baseline} <= threshold={threshold}",
            evidence_used={"poi": matched_name, "baseline": baseline, "threshold": threshold, "citation": matched.get("citation")},
        )
    return make_result(True, "ok", "ok", {"poi": matched_name, "baseline": baseline, "threshold": threshold, "citation": matched.get("citation")}, suggested_action="accept")


def _validate_time_limit(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    day = _get_day_binding(constraints, fallback_params, edit_query)
    if day is None:
        return make_result(False, "missing_day_binding", "time_limit missing day binding", repairable=True)
    threshold = _extract_end_before_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "time_limit threshold missing or invalid", repairable=True)
    baseline = evidence["day_end_by_day"].get(day)
    if baseline is None:
        return make_result(False, "invalid_day", f"time_limit anchor day invalid: {day}", repairable=True)
    attraction_count = int((evidence.get("day_attraction_count", {}) or {}).get(day, 0) or 0)
    if attraction_count <= 0:
        return make_result(
            False,
            "no_attraction_anchor_day",
            f"time_limit anchor day has no attractions: day={day}",
            {"day": day, "attraction_count": attraction_count},
        )
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"time_limit not conflicting: day_end={baseline} <= threshold={threshold}", {"day": day, "baseline": baseline, "threshold": threshold})
    return make_result(True, "ok", "ok", {"day": day, "baseline": baseline, "threshold": threshold}, suggested_action="accept")


def _validate_budget_threshold(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    threshold = _extract_budget_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "budget threshold missing or invalid", repairable=True)
    constraint_type = str(fallback_params.get("_constraint_type", "")).strip()

    def _is_cost_allocation_preference_query(query: str) -> bool:
        text = str(query or "")
        markers = (
            "住宿",
            "酒店",
            "餐饮",
            "餐费",
            "吃饭",
            "其他部分",
            "尽量压缩",
            "想多花一点",
            "分配",
            "分摊",
        )
        return any(marker in text for marker in markers)

    def _project_multi_day_local_budget() -> Optional[float]:
        requested_days = _parse_numeric(fallback_params.get("day_count"))
        if requested_days is None:
            requested_days = _parse_numeric(extract_requested_day_count_from_query(edit_query or ""))
        current_days = max(int(evidence.get("day_count", 0) or 0), 1)
        local_cost = _parse_numeric(evidence.get("transferable_local_cost"))
        if requested_days is None or local_cost is None or local_cost <= 0:
            return None
        if int(requested_days) <= current_days:
            return float(local_cost)
        per_day_local = float(local_cost) / float(current_days)
        return max(float(local_cost), per_day_local * float(requested_days))

    def _resource_overlap_bundle_budget() -> Optional[float]:
        resource = constraints.get("resource", {}) if isinstance(constraints.get("resource", {}), dict) else {}
        explicit_bundle = _parse_numeric(
            resource.get("origin_bundle_cost")
            or fallback_params.get("origin_bundle_cost")
            or fallback_params.get("baseline_anchor_bundle_cost")
        )
        if explicit_bundle is not None and explicit_bundle > 0:
            return float(explicit_bundle)
        poi_name = (
            _extract_poi_from_query(edit_query)
            or _extract_first_must_visit(constraints)
            or fallback_params.get("anchor_poi")
            or fallback_params.get("source_poi")
            or fallback_params.get("poi_name")
        )
        ticket_component = 0.0
        if isinstance(poi_name, str) and poi_name.strip():
            match = _find_matching_poi_record(evidence.get("poi_ticket_price_by_name", {}), poi_name)
            if match:
                _, matched = match
                ticket_component = float(matched.get("ticket_price", 0.0) or 0.0)
        meal_count = int(evidence.get("meal_count", 0) or 0)
        meal_component = 0.0
        if meal_count > 0:
            meal_component = float(evidence.get("meal_cost", 0.0) or 0.0) / float(meal_count)
        baseline_value = ticket_component + meal_component
        return baseline_value if baseline_value > 0 else None

    baseline_reason = "total_cost"
    if constraint_type == "multi_day_budget_overflow":
        projected_budget = _project_multi_day_local_budget()
        if projected_budget is None:
            return make_result(False, "missing_projected_budget_baseline", "multi_day_budget_overflow missing projected local budget baseline", repairable=True)
        baseline = projected_budget
        baseline_reason = "projected_transferable_local_cost"
    elif constraint_type == "resource_overlap":
        bundle_budget = _resource_overlap_bundle_budget()
        if bundle_budget is None:
            return make_result(False, "missing_bundle_budget_baseline", "resource_overlap missing anchor bundle budget baseline", repairable=True)
        baseline = bundle_budget
        baseline_reason = "anchor_bundle_budget"
    else:
        if (
            constraint_type in {"budget_limit", "daily_budget_infeasible"}
            and evidence.get("budget_baseline_reliable_for_total_budget_edit") is False
        ):
            return make_result(False, "budget_baseline_unreliable", "budget baseline is dominated by intercity transport and not suitable for total-budget edits", repairable=False, suggested_action="switch_origin")
        baseline = float(evidence["total_cost"])

    origin_budget_total = _parse_numeric(evidence.get("origin_budget_total"))
    if constraint_type in {"budget_limit", "daily_budget_infeasible"} and origin_budget_total is not None:
        if (
            origin_budget_total <= float(baseline) * 1.25
            and not _is_cost_allocation_preference_query(edit_query)
            and (threshold >= origin_budget_total or threshold < origin_budget_total * 0.75)
        ):
            return make_result(
                False,
                "budget_reduction_not_stricter",
                f"budget threshold outside origin-relative reduction band: threshold={threshold:.2f}, origin_budget={origin_budget_total:.2f}",
                {"baseline": baseline, "threshold": threshold, "origin_budget_total": origin_budget_total, "baseline_reason": baseline_reason},
            )
    if constraint_type in {"budget_cap_preference", "budget_target_update"} and origin_budget_total is not None:
        if threshold <= origin_budget_total:
            return make_result(
                False,
                "budget_update_not_above_origin_budget",
                f"budget update did not exceed origin budget: threshold={threshold:.2f} <= origin_budget={origin_budget_total:.2f}",
                {"baseline": baseline, "threshold": threshold, "origin_budget_total": origin_budget_total, "baseline_reason": baseline_reason},
            )
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"budget not conflicting: baseline={baseline:.2f} <= threshold={threshold:.2f}", {"baseline": baseline, "threshold": threshold, "baseline_reason": baseline_reason})
    return make_result(True, "ok", "ok", {"baseline": baseline, "threshold": threshold, "baseline_reason": baseline_reason}, suggested_action="accept")


def _validate_ticket_budget(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    threshold = _extract_budget_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "ticket budget threshold missing or invalid", repairable=True)
    baseline = float(evidence["ticket_cost"])
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"ticket budget not conflicting: baseline={baseline:.2f} <= threshold={threshold:.2f}", {"baseline": baseline, "threshold": threshold})
    return make_result(True, "ok", "ok", {"baseline": baseline, "threshold": threshold}, suggested_action="accept")


def _validate_ticket_price_cap(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    source_poi = _extract_replace_from(constraints, edit_query) or _extract_first_must_visit(constraints) or fallback_params.get("poi_name")
    match = _find_matching_poi_record(evidence["poi_ticket_price_by_name"], source_poi)
    if not match:
        activity_type = _find_activity_type_in_plan(fallback_params.get("_origin_plan"), source_poi)
        if activity_type and activity_type != "attraction":
            return make_result(
                False,
                "unsupported_source_type",
                f"ticket_price_cap_or_substitute source '{source_poi}' is {activity_type}, not attraction",
                {"poi": source_poi, "activity_type": activity_type},
                repairable=True,
            )
        return make_result(False, "poi_not_found", f"ticket_price_cap_or_substitute poi '{source_poi}' not found in origin_plan", repairable=True)
    matched_name, matched = match
    threshold = _extract_budget_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "ticket price cap threshold missing or invalid", repairable=True)
    baseline = float(matched.get("ticket_price", 0.0))
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"ticket price cap not conflicting: poi={matched_name}, baseline={baseline:.2f} <= threshold={threshold:.2f}", {"poi": matched_name, "baseline": baseline, "threshold": threshold, "citation": matched.get("citation")})
    return make_result(True, "ok", "ok", {"poi": matched_name, "baseline": baseline, "threshold": threshold, "citation": matched.get("citation")}, suggested_action="accept")


def _validate_distance_threshold(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    threshold = _extract_distance_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "distance threshold missing or invalid", repairable=True)
    baseline = float(evidence["max_transport_distance_km"])
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"distance not conflicting: baseline={baseline:.2f} <= threshold={threshold:.2f}", {"baseline": baseline, "threshold": threshold})
    return make_result(True, "ok", "ok", {"baseline": baseline, "threshold": threshold}, suggested_action="accept")


def _validate_anchor_distance_threshold(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    threshold = _extract_distance_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "poi_local_radius_limit threshold missing or invalid", repairable=True)
    anchor_poi = (_extract_must_include_names(constraints) or [fallback_params.get("poi_name")])[0]
    distances = _collect_anchor_local_segment_distances({"itinerary": fallback_params.get("_origin_itinerary", [])}, anchor_poi)
    if not distances:
        distances = _collect_anchor_local_segment_distances(fallback_params.get("_origin_plan", {}), anchor_poi)
    if not distances:
        return _validate_distance_threshold(evidence, constraints, edit_query, fallback_params, template_str)
    baseline = max(distances)
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"poi_local_radius_limit not conflicting: baseline={baseline:.2f} <= threshold={threshold:.2f}", {"poi": anchor_poi, "baseline": baseline, "threshold": threshold})
    return make_result(True, "ok", "ok", {"poi": anchor_poi, "baseline": baseline, "threshold": threshold}, suggested_action="accept")


def _validate_transport_duration(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    threshold = _extract_duration_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "transport duration threshold missing or invalid", repairable=True)
    commute_probe = assess_commute_cap_insertion_feasibility(
        origin_plan=fallback_params.get("_origin_plan", {}),
        constraints=constraints,
        edit_query=edit_query,
        fallback_params={**fallback_params, "duration": threshold},
    )
    if commute_probe.get("applicable") is True:
        evidence_used = {"threshold": threshold, "commute_cap_insertion": commute_probe}
        if commute_probe.get("pass") is True:
            return make_result(True, "ok", "ok", evidence_used, suggested_action="accept")
        reason_code = str(commute_probe.get("reason_code") or "commute_cap_insertion_infeasible")
        repairable = commute_probe.get("status") != "unknown_needs_probe"
        return make_result(
            False,
            reason_code,
            str(commute_probe.get("reason") or reason_code),
            evidence_used,
            repairable=repairable,
            suggested_action="retry_same_template" if repairable else "manual_review",
        )
    baseline = int(evidence["max_transport_duration_min"])
    if baseline <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"transport duration not conflicting: baseline={baseline} <= threshold={threshold}", {"baseline": baseline, "threshold": threshold})
    return make_result(True, "ok", "ok", {"baseline": baseline, "threshold": threshold}, suggested_action="accept")


def _validate_time_window(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    poi_candidates = _extract_must_include_names(constraints)
    if not poi_candidates:
        semantic = constraints.get("semantic", {})
        if isinstance(semantic, dict):
            replace_from = semantic.get("replace_from")
            if isinstance(replace_from, str) and replace_from.strip():
                poi_candidates = [replace_from.strip()]
    poi_name = poi_candidates[0] if poi_candidates else None
    if not poi_name:
        return make_result(False, "missing_poi_anchor", "time_window missing poi anchor in constraints", repairable=True, suggested_action="retry_same_template")
    time_period = _extract_time_period(edit_query, constraints)
    if time_period is None:
        return make_result(False, "missing_time_period", "time_window missing time period binding", repairable=True)
    window = TIME_PERIOD_WINDOWS[time_period]
    match = _find_matching_poi_record(evidence["poi_duration_by_name"], poi_name)
    if match:
        matched_name, record = match
        start = _parse_time_to_minutes(record.get("start_time"))
        end = _parse_time_to_minutes(record.get("end_time"))
        if start is None or end is None:
            return make_result(False, "missing_activity_time", f"time_window activity time missing for poi '{matched_name}'", repairable=True)
        if end <= start:
            end += 24 * 60
        start_bound, end_bound = window
        if start >= start_bound and end <= end_bound:
            return make_result(False, "already_satisfies_window", f"time_window not conflicting: poi '{matched_name}' already in {time_period}", {"poi": matched_name, "start_time": record.get("start_time"), "end_time": record.get("end_time"), "time_period": time_period, "citation": record.get("citation")})
        return make_result(True, "ok", "ok", {"poi": matched_name, "start_time": record.get("start_time"), "end_time": record.get("end_time"), "time_period": time_period, "citation": record.get("citation")}, suggested_action="accept")

    day = _get_day_binding(constraints, fallback_params, edit_query)
    if day is None:
        return make_result(False, "missing_day_binding", "time_window missing day binding for new poi", repairable=True)
    available = _window_available_minutes(_collect_day_occupied_intervals(fallback_params.get("_origin_plan", {}), day), window)
    if available < 90:
        return make_result(True, "ok", "ok", {"poi": poi_name, "day": day, "time_period": time_period, "available_minutes": available}, suggested_action="accept")
    return make_result(False, "window_has_capacity", f"time_window not conflicting: day {day} still has {available} minutes available in {time_period}", {"poi": poi_name, "day": day, "time_period": time_period, "available_minutes": available})


def _validate_order_constraint(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del constraints, fallback_params, template_str
    pois = _extract_two_pois(edit_query)
    if len(pois) != 2:
        return make_result(False, "missing_poi_anchor", "order_constraint missing poi pair", repairable=True)
    poi_day_map = evidence["poi_day_by_name"]
    left_match = _find_matching_poi_record(poi_day_map, pois[0])
    right_match = _find_matching_poi_record(poi_day_map, pois[1])
    if not left_match or not right_match:
        missing = [pois[idx] for idx, item in enumerate([left_match, right_match]) if item is None]
        return make_result(False, "poi_not_found", f"order_constraint poi not found in origin_plan: {missing}", repairable=True)

    left_name, left_entries_raw = left_match
    right_name, right_entries_raw = right_match
    left_entries = _flatten_poi_day_entries(left_entries_raw)
    right_entries = _flatten_poi_day_entries(right_entries_raw)
    if not left_entries or not right_entries:
        return make_result(False, "missing_activity_order", "order_constraint missing day/activity indices", repairable=True)

    before_exists = False
    comparable_pairs = 0
    for left_entry in left_entries:
        for right_entry in right_entries:
            is_before = _is_entry_before(left_entry, right_entry)
            if is_before is None:
                continue
            comparable_pairs += 1
            if is_before:
                before_exists = True
                break
        if before_exists:
            break

    if comparable_pairs == 0:
        return make_result(False, "missing_activity_order", "order_constraint missing comparable day/activity order", repairable=True)
    if before_exists:
        return make_result(
            False,
            "already_satisfies_order",
            f"order_constraint not conflicting: '{left_name}' already appears before '{right_name}'",
            {"poi_order": [left_name, right_name]},
        )
    return make_result(True, "ok", "ok", {"poi_order": [left_name, right_name]}, suggested_action="accept")


def _validate_same_day_compact(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    day = _get_day_binding(constraints, fallback_params, edit_query)
    pois = _extract_two_pois(edit_query)
    if day is None or len(pois) != 2:
        return make_result(False, "missing_binding", "same_day_no_overlap_compact missing day or poi bindings", repairable=True)
    poi_day_map = evidence["poi_day_by_name"]
    on_day = 0
    for poi in pois:
        match = _find_matching_poi_record(poi_day_map, poi)
        if match:
            _, entries = match
            if isinstance(entries, list):
                if any(item.get("day") == day for item in entries):
                    on_day += 1
    if on_day == 2:
        return make_result(False, "already_compactable", f"same_day_no_overlap_compact not conflicting: both pois already on day {day}", {"day": day, "pois": pois})
    return make_result(True, "ok", "ok", {"day": day, "pois": pois}, suggested_action="accept")


def _validate_same_period_no_overlap(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    time_period = _extract_time_period(edit_query, constraints)
    pois = _extract_two_pois(edit_query)
    if time_period is None or len(pois) != 2:
        return make_result(False, "missing_binding", "same_time_period_no_overlap missing poi or time period binding", repairable=True)
    window = TIME_PERIOD_WINDOWS[time_period]
    poi_day_map = evidence["poi_day_by_name"]
    left_match = _find_matching_poi_record(poi_day_map, pois[0])
    right_match = _find_matching_poi_record(poi_day_map, pois[1])
    if not left_match or not right_match:
        missing = [pois[idx] for idx, item in enumerate([left_match, right_match]) if item is None]
        return make_result(False, "poi_not_found", f"same_time_period_no_overlap poi not found in origin_plan: {missing}", repairable=True)

    left_name, left_entries_raw = left_match
    right_name, right_entries_raw = right_match
    left_entries = _flatten_poi_day_entries(left_entries_raw)
    right_entries = _flatten_poi_day_entries(right_entries_raw)
    if not left_entries or not right_entries:
        return make_result(False, "missing_activity_time", "same_time_period_no_overlap missing activity time evidence", repairable=True)

    has_period_grounding = False
    for left_entry in left_entries:
        left_interval = _entry_fully_in_period(left_entry, window)
        if left_interval is None:
            continue
        has_period_grounding = True
        for right_entry in right_entries:
            if _normalize_day_value(left_entry.get("day")) != _normalize_day_value(right_entry.get("day")):
                continue
            right_interval = _entry_fully_in_period(right_entry, window)
            if right_interval is None:
                continue
            has_period_grounding = True
            left_start, left_end = left_interval
            right_start, right_end = right_interval
            if left_end <= right_start or right_end <= left_start:
                return make_result(
                    False,
                    "already_satisfies_period_order",
                    f"same_time_period_no_overlap not conflicting: '{left_name}' and '{right_name}' already non-overlapping in {time_period}",
                    {"pois": [left_name, right_name], "time_period": time_period},
                )

    if not has_period_grounding:
        return make_result(
            False,
            "period_not_grounded",
            f"same_time_period_no_overlap not verifiable: no anchor activity falls inside {time_period}",
            {"pois": [left_name, right_name], "time_period": time_period},
            repairable=True,
        )

    return make_result(True, "ok", "ok", {"pois": [left_name, right_name], "time_period": time_period}, suggested_action="accept")


def _validate_remove_poi(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del constraints, template_str
    poi_name = (
        _extract_poi_from_query(edit_query)
        or str(fallback_params.get("source_poi") or "").strip()
        or str(fallback_params.get("poi_name") or "").strip()
    )
    if not poi_name:
        return make_result(False, "missing_poi_anchor", "remove_poi missing poi anchor", repairable=True)
    if poi_name not in evidence["plan_poi_names"]:
        return make_result(False, "poi_not_found", f"remove_poi target '{poi_name}' not found in origin_plan", repairable=True)
    return make_result(True, "ok", "ok", {"poi": poi_name}, suggested_action="accept")


def _validate_semantic_substitution(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del fallback_params, template_str
    source_poi = _extract_replace_from(constraints, edit_query)
    semantic_target = _extract_semantic_target(edit_query, constraints)
    if _extract_must_include_names(constraints):
        return make_result(False, "must_include_not_empty", "semantic_substitution must_include must be empty", repairable=True)
    if not source_poi or source_poi not in evidence["plan_poi_names"]:
        return make_result(False, "poi_not_found", f"semantic substitution source '{source_poi}' not found in origin_plan", repairable=True)
    forbidden = set(_extract_forbidden_names(constraints))
    if source_poi not in forbidden:
        return make_result(False, "missing_forbidden", f"semantic substitution source '{source_poi}' missing in forbidden", repairable=True)
    if not semantic_target:
        return make_result(False, "missing_semantic_target", "semantic_substitution missing semantic_target", repairable=True)
    return make_result(True, "ok", "ok", {"source_poi": source_poi, "semantic_target": semantic_target}, suggested_action="accept")


def _validate_substitute_semantic_family(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del fallback_params, template_str
    source_poi = _extract_replace_from(constraints, edit_query)
    if not source_poi or source_poi not in evidence["plan_poi_names"]:
        return make_result(False, "poi_not_found", f"semantic replacement source '{source_poi}' not found in origin_plan", repairable=True)
    target = _extract_semantic_target(edit_query, constraints) or _extract_preferred_type(edit_query, constraints)
    if not target:
        return make_result(False, "missing_semantic_target", "semantic replacement missing semantic target/type", repairable=True)
    return make_result(True, "ok", "ok", {"source_poi": source_poi, "target": target}, suggested_action="accept")


def _validate_type_filter_family(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del fallback_params, template_str
    preferred_type = _extract_preferred_type(edit_query, constraints)
    if not preferred_type:
        return make_result(False, "missing_preferred_type", "semantic preference missing preferred type", repairable=True)
    if not _preferred_type_supported_by_city(preferred_type, evidence.get("city")):
        return make_result(
            False,
            "missing_required_attraction_type",
            f"semantic preference type '{preferred_type}' not available in target city",
            {"preferred_type": preferred_type, "city": evidence.get("city")},
            repairable=True,
        )
    return make_result(True, "ok", "ok", {"preferred_type": preferred_type, "city": evidence.get("city")}, suggested_action="accept")


def _validate_semantic_and_specific(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del fallback_params, template_str
    must_include = _extract_must_include_names(constraints)
    if not must_include:
        return make_result(False, "missing_poi_anchor", "semantic_and_specific missing must_include poi", repairable=True)
    preferred_type = _extract_preferred_type(edit_query, constraints)
    if not preferred_type:
        return make_result(False, "missing_preferred_type", "semantic_and_specific missing preferred type", repairable=True)
    if not _preferred_type_supported_by_city(preferred_type, evidence.get("city")):
        return make_result(
            False,
            "missing_required_attraction_type",
            f"semantic preference type '{preferred_type}' not available in target city",
            {"must_include": must_include, "preferred_type": preferred_type, "city": evidence.get("city")},
            repairable=True,
        )
    return make_result(True, "ok", "ok", {"must_include": must_include, "preferred_type": preferred_type, "city": evidence.get("city")}, suggested_action="accept")


def _validate_cross_day_variety(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del fallback_params, template_str
    pairs = _extract_day_poi_pairs(edit_query)
    if len(pairs) < 2:
        return make_result(False, "missing_day_binding", "cross_day_variety_by_types missing day-poi pairs", repairable=True)
    temporal_map: Dict[int, Set[str]] = {}
    for item in _extract_temporal_items(constraints):
        day = _normalize_day_value(item.get("day"))
        must_visit = item.get("must_visit")
        if day is None or must_visit is None:
            continue
        if isinstance(must_visit, str):
            temporal_map.setdefault(day, set()).add(must_visit)
        elif isinstance(must_visit, list):
            for value in must_visit:
                if isinstance(value, str) and value.strip():
                    temporal_map.setdefault(day, set()).add(value.strip())
    for day, poi_name in pairs:
        if poi_name not in temporal_map.get(day, set()):
            return make_result(False, "missing_day_binding", f"cross_day_variety_by_types missing day-poi binding: day={day}, poi={poi_name}", repairable=True)
    mismatch = False
    for day, poi_name in pairs:
        match = _find_matching_poi_record(evidence["poi_day_by_name"], poi_name)
        if not match:
            mismatch = True
            break
        _, entries = match
        if not isinstance(entries, list) or not any(item.get("day") == day for item in entries):
            mismatch = True
            break
    if not mismatch:
        return make_result(False, "already_matches_day_split", "cross_day_variety_by_types not conflicting: current plan already matches requested day split", {"pairs": pairs})
    return make_result(True, "ok", "ok", {"pairs": pairs}, suggested_action="accept")


def _validate_multi_day_requirement(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del constraints, template_str
    requested_days = _parse_numeric(fallback_params.get("day_count"))
    if requested_days is None:
        requested_from_query = extract_requested_day_count_from_query(edit_query or "")
        if requested_from_query is not None:
            requested_days = float(requested_from_query)
        else:
            match = re.search(r"玩(\d+)天|扩展到(\d+)天|改成在.+玩(\d+)天", edit_query or "")
            if match:
                requested_days = float(next(group for group in match.groups() if group))
    if requested_days is None:
        if evidence.get("is_single_day"):
            return make_result(True, "ok", "ok", {"current_day_count": evidence["day_count"], "requested_multi_day": True}, suggested_action="accept")
        return make_result(False, "missing_day_count", "multi_day requirement missing requested day count", repairable=True)
    if int(requested_days) <= int(evidence["day_count"]):
        return make_result(False, "day_count_not_stricter", f"multi_day requirement not conflicting: current_day_count={evidence['day_count']} >= requested_day_count={int(requested_days)}", {"current_day_count": evidence["day_count"], "requested_day_count": int(requested_days)})
    return make_result(True, "ok", "ok", {"current_day_count": evidence["day_count"], "requested_day_count": int(requested_days)}, suggested_action="accept")


def _validate_add_days(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del constraints, template_str
    current_days = max(int(evidence.get("day_count", 0) or 0), 1)
    requested_delta = _parse_numeric(fallback_params.get("day_delta"))
    if requested_delta is None:
        parsed_delta = extract_requested_day_delta_from_query(edit_query or "")
        if parsed_delta is not None:
            requested_delta = float(parsed_delta)

    requested_days = _parse_numeric(fallback_params.get("day_count"))
    if requested_days is None and requested_delta is not None:
        requested_days = float(current_days) + float(requested_delta)

    if requested_days is None:
        return make_result(False, "missing_day_count", "add_days missing requested resulting day count", repairable=True)
    if requested_delta is None:
        return make_result(False, "missing_day_delta", "add_days missing requested day delta", repairable=True)
    if int(requested_delta) <= 0:
        return make_result(False, "non_positive_day_delta", f"add_days requires positive day delta, got {int(requested_delta)}", repairable=True)
    if int(requested_days) <= current_days:
        return make_result(False, "day_count_not_stricter", f"add_days not conflicting: current_day_count={current_days} >= requested_day_count={int(requested_days)}", {"current_day_count": current_days, "requested_day_count": int(requested_days), "requested_day_delta": int(requested_delta)})
    return make_result(True, "ok", "ok", {"current_day_count": current_days, "requested_day_count": int(requested_days), "requested_day_delta": int(requested_delta)}, suggested_action="accept")


def _validate_cross_day_restructure(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del fallback_params, template_str
    pairs = _extract_day_poi_pairs(edit_query)
    if not pairs:
        return make_result(False, "missing_day_binding", "cross_day_restructure missing day-poi pairs", repairable=True)
    temporal_map: Dict[int, Set[str]] = {}
    for item in _extract_temporal_items(constraints):
        day = _normalize_day_value(item.get("day"))
        must_visit = item.get("must_visit")
        if day is None or must_visit is None:
            continue
        if isinstance(must_visit, str):
            temporal_map.setdefault(day, set()).add(must_visit)
        elif isinstance(must_visit, list):
            for value in must_visit:
                if isinstance(value, str) and value.strip():
                    temporal_map.setdefault(day, set()).add(value.strip())
    for day, poi_name in pairs:
        if poi_name not in temporal_map.get(day, set()):
            return make_result(False, "missing_day_binding", f"cross_day_restructure missing day-poi binding: day={day}, poi={poi_name}", repairable=True)
    for day, poi_name in pairs:
        match = _find_matching_poi_record(evidence["poi_day_by_name"], poi_name)
        if not match:
            return make_result(True, "ok", "ok", {"pairs": pairs}, suggested_action="accept")
        _, entries = match
        if isinstance(entries, list) and not any(item.get("day") == day for item in entries):
            return make_result(True, "ok", "ok", {"pairs": pairs}, suggested_action="accept")
    return make_result(False, "already_matches_day_split", "cross_day_restructure not conflicting: current plan already matches requested day split", {"pairs": pairs})


def _validate_extend_with_pacing(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del constraints, template_str
    requested_days = _parse_numeric(fallback_params.get("day_count"))
    requested_cap = _parse_numeric(fallback_params.get("poi_count"))
    parsed_spec = extract_extend_with_pacing_spec(edit_query or "")
    if requested_days is None and parsed_spec.get("day_count") is not None:
        requested_days = float(parsed_spec["day_count"])
    if requested_cap is None and parsed_spec.get("daily_poi_cap") is not None:
        requested_cap = float(parsed_spec["daily_poi_cap"])
    day_stricter = requested_days is not None and int(requested_days) > int(evidence["day_count"])
    cap_stricter = requested_cap is not None and any(count > int(requested_cap) for count in evidence["day_attraction_count"].values())
    if not day_stricter and not cap_stricter:
        return make_result(False, "not_stricter_than_plan", f"extend_with_pacing not conflicting: requested_day_count={requested_days}, requested_poi_cap={requested_cap}, current_day_count={evidence['day_count']}, current_day_caps={evidence['day_attraction_count']}", {"requested_day_count": requested_days, "requested_poi_cap": requested_cap, "current_day_count": evidence["day_count"], "current_day_caps": evidence["day_attraction_count"]})
    return make_result(True, "ok", "ok", {"requested_day_count": requested_days, "requested_poi_cap": requested_cap, "current_day_count": evidence["day_count"], "current_day_caps": evidence["day_attraction_count"]}, suggested_action="accept")


def _validate_multi_city_split(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    if evidence.get("day_count", 0) < 2:
        return make_result(False, "day_count_not_stricter", "multi_city_split requires multi-day origin plan", repairable=False, suggested_action="switch_origin")
    semantic = constraints.get("semantic", {}) if isinstance(constraints.get("semantic", {}), dict) else {}
    split_targets = []
    if isinstance(semantic.get("city_split_targets"), list):
        split_targets = [str(item).strip() for item in semantic.get("city_split_targets", []) if str(item).strip()]
    if not split_targets and isinstance(fallback_params.get("city_split_targets"), list):
        split_targets = [str(item).strip() for item in fallback_params.get("city_split_targets", []) if str(item).strip()]
    if not split_targets:
        split_targets = extract_city_split_targets_from_query(edit_query or "")
    if len(split_targets) < 2:
        return make_result(False, "missing_split_signal", "multi_city_split missing split signal", repairable=True)
    return make_result(True, "ok", "ok", {"day_count": evidence.get("day_count"), "city_split_targets": split_targets}, suggested_action="accept")


def _validate_resource_overlap(evidence: Dict[str, Any], constraints: Dict[str, Any], edit_query: str, fallback_params: Dict[str, Any], template_str: Optional[str]) -> FeasibilityResult:
    del template_str
    poi_name = (
        _extract_poi_from_query(edit_query)
        or _extract_first_must_visit(constraints)
        or fallback_params.get("anchor_poi")
        or fallback_params.get("source_poi")
        or fallback_params.get("poi_name")
    )
    if not poi_name:
        return make_result(False, "missing_poi_anchor", "resource_overlap missing poi anchor", repairable=True)
    if poi_name not in evidence["plan_poi_names"]:
        return make_result(False, "poi_not_found", f"resource_overlap poi '{poi_name}' not found in origin_plan", repairable=True)
    budget_result = _validate_budget_threshold(evidence, constraints, edit_query, fallback_params, None)
    if not budget_result.ok:
        return budget_result
    enriched = dict(budget_result.evidence_used)
    enriched["poi"] = poi_name
    return make_result(True, "ok", "ok", enriched, suggested_action="accept")


def _validate_required_restaurant_name(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del fallback_params, template_str
    restaurant_names = [
        item.get("name")
        for item in constraints.get("must_include", [])
        if isinstance(item, dict)
        and str(item.get("type") or "").lower() in {"restaurant", "meal"}
        and isinstance(item.get("name"), str)
        and item.get("name", "").strip()
    ]
    poi_name = (restaurant_names or [_extract_poi_from_query(edit_query)])[0]
    if not poi_name:
        return make_result(False, "missing_poi_anchor", "required_restaurant_name missing restaurant anchor", repairable=True)
    if poi_name not in evidence["plan_poi_names"]:
        return make_result(True, "ok", "ok", {"poi": poi_name}, suggested_action="accept")
    return make_result(False, "already_satisfied_or_not_stricter", f"required_restaurant_name not conflicting: restaurant '{poi_name}' already exists", {"poi": poi_name})


def _validate_required_hotel_name(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del fallback_params, template_str
    hotel_names = [
        item.get("name")
        for item in constraints.get("must_include", [])
        if isinstance(item, dict)
        and str(item.get("type") or "").lower() in {"accommodation", "hotel"}
        and isinstance(item.get("name"), str)
        and item.get("name", "").strip()
    ]
    poi_name = (hotel_names or [_extract_poi_from_query(edit_query)])[0]
    if not poi_name:
        return make_result(False, "missing_poi_anchor", "required_hotel_name missing hotel anchor", repairable=True)
    if poi_name not in evidence["plan_poi_names"]:
        return make_result(True, "ok", "ok", {"poi": poi_name}, suggested_action="accept")
    return make_result(False, "already_satisfied_or_not_stricter", f"required_hotel_name not conflicting: hotel '{poi_name}' already exists", {"poi": poi_name})


def _validate_required_restaurant_type(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del template_str
    preferred_type = _extract_restaurant_type(constraints, edit_query)
    if not preferred_type:
        return make_result(False, "missing_preferred_type", "required_restaurant_type missing preferred type", repairable=True)
    target_city = fallback_params.get("_origin_plan", {}).get("target_city")
    for day_plan in fallback_params.get("_origin_plan", {}).get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if str(activity.get("type") or "").lower() not in {"breakfast", "lunch", "dinner"}:
                continue
            if _infer_restaurant_type(activity, target_city) == preferred_type:
                return make_result(False, "already_satisfied_or_not_stricter", f"required_restaurant_type not conflicting: already has restaurant type '{preferred_type}'", {"preferred_type": preferred_type})
    return make_result(True, "ok", "ok", {"preferred_type": preferred_type}, suggested_action="accept")


def _validate_required_hotel_feature(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del template_str
    hotel_feature = _extract_hotel_feature(constraints, edit_query)
    if not hotel_feature:
        return make_result(False, "missing_semantic_target", "required_hotel_feature missing hotel feature", repairable=True)
    target_city = fallback_params.get("_origin_plan", {}).get("target_city")
    for day_plan in fallback_params.get("_origin_plan", {}).get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if str(activity.get("type") or "").lower() != "accommodation":
                continue
            if _infer_hotel_feature(activity, target_city) == hotel_feature:
                return make_result(False, "already_satisfied_or_not_stricter", f"required_hotel_feature not conflicting: already has hotel feature '{hotel_feature}'", {"hotel_feature": hotel_feature})
    return make_result(True, "ok", "ok", {"hotel_feature": hotel_feature}, suggested_action="accept")


def _validate_required_room_type(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del evidence, template_str
    room_type = _extract_room_type(constraints, fallback_params, edit_query)
    if room_type is None:
        return make_result(False, "missing_binding_or_threshold", "required_room_type missing room type", repairable=True)
    if int(room_type) not in {1, 2}:
        return make_result(False, "unsupported_room_type", f"required_room_type unsupported by accommodation catalog: {room_type}", {"room_type": room_type}, repairable=True)
    target_city = fallback_params.get("_origin_plan", {}).get("target_city")
    for day_plan in fallback_params.get("_origin_plan", {}).get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if str(activity.get("type") or "").lower() != "accommodation":
                continue
            actual = activity.get("room_type")
            if actual is None:
                continue
            if int(actual) == int(room_type):
                return make_result(False, "already_satisfied_or_not_stricter", f"required_room_type not conflicting: already has room_type {room_type}", {"room_type": room_type, "city": target_city})
    return make_result(True, "ok", "ok", {"room_type": room_type}, suggested_action="accept")


def _validate_required_room_count(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del evidence, template_str
    room_count = _extract_room_count(constraints, fallback_params, edit_query)
    if room_count is None:
        return make_result(False, "missing_binding_or_threshold", "required_room_count missing room count", repairable=True)
    target_city = fallback_params.get("_origin_plan", {}).get("target_city")
    for day_plan in fallback_params.get("_origin_plan", {}).get("itinerary", []):
        for activity in day_plan.get("activities", []):
            if str(activity.get("type") or "").lower() != "accommodation":
                continue
            actual = activity.get("rooms", activity.get("room_count"))
            if actual is None:
                continue
            if int(actual) == int(room_count):
                return make_result(False, "already_satisfied_or_not_stricter", f"required_room_count not conflicting: already has room_count {room_count}", {"room_count": room_count, "city": target_city})
    return make_result(True, "ok", "ok", {"room_count": room_count}, suggested_action="accept")


def _validate_activity_budget_limit(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del template_str
    origin_plan = fallback_params.get("_origin_plan", {}) if isinstance(fallback_params.get("_origin_plan", {}), dict) else {}
    threshold = _extract_budget_threshold(constraints, fallback_params, edit_query)
    if threshold is None:
        return make_result(False, "missing_threshold", "activity_budget_limit missing budget threshold", repairable=True)
    activity_type, metric = _extract_activity_budget_target(constraints, edit_query)
    if not activity_type or not metric:
        return make_result(False, "missing_binding", "activity_budget_limit missing activity target or metric", repairable=True)
    if activity_type == "meal":
        baseline = float(evidence.get("meal_cost", 0.0))
        count = int(evidence.get("meal_count", 0) or 0)
        if count <= 0:
            return make_result(False, "missing_binding", "activity_budget_limit missing meal evidence", repairable=True)
        people = max(int(origin_plan.get("people_number", 1) or 1), 1)
        actual = baseline / max(count, 1) / people if metric == "avg_cost_per_meal_per_person" else baseline
    else:
        baseline = float(evidence.get("accommodation_cost", 0.0))
        count = int(evidence.get("accommodation_count", 0) or 0)
        if count <= 0:
            return make_result(False, "missing_binding", "activity_budget_limit missing accommodation evidence", repairable=True)
        people = max(int(origin_plan.get("people_number", 1) or 1), 1)
        days = max(int(evidence.get("day_count", 1) or 1) - 1, 1)
        actual = baseline / people / days if metric == "avg_cost_per_person_per_night" else baseline
    if actual <= threshold:
        return make_result(False, "baseline_not_exceed_threshold", f"activity_budget_limit not conflicting: baseline={actual:.2f} <= threshold={threshold:.2f}", {"activity_type": activity_type, "metric": metric, "baseline": actual, "threshold": threshold})
    return make_result(True, "ok", "ok", {"activity_type": activity_type, "metric": metric, "baseline": actual, "threshold": threshold}, suggested_action="accept")


def _validate_required_intercity_transport_type(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del template_str
    modes = _extract_intercity_transport_modes(constraints, fallback_params, edit_query)
    if not modes:
        return make_result(False, "missing_binding", "required_intercity_transport_type missing transport mode", repairable=True)
    existing_modes = set(str(item) for item in evidence.get("intercity_transport_modes", []) if str(item))
    if set(modes).issubset(existing_modes):
        return make_result(
            False,
            "already_satisfied_or_not_stricter",
            f"required_intercity_transport_type not conflicting: already has modes {sorted(existing_modes)}",
            {"required_modes": modes, "existing_modes": sorted(existing_modes)},
        )
    return make_result(
        True,
        "ok",
        "ok",
        {"required_modes": modes, "existing_modes": sorted(existing_modes)},
        suggested_action="accept",
    )


def _validate_transport_time_window(
    evidence: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_query: str,
    fallback_params: Dict[str, Any],
    template_str: Optional[str],
) -> FeasibilityResult:
    del template_str
    windows = _extract_transport_time_windows(constraints, fallback_params, edit_query)
    if not windows:
        return make_result(False, "missing_binding", "transport_time_window missing time window", repairable=True)
    time_by_leg = evidence.get("intercity_time_by_leg", {})
    if not isinstance(time_by_leg, dict) or not time_by_leg:
        return make_result(False, "missing_binding", "transport_time_window missing intercity transport evidence", repairable=True)

    checked: List[Dict[str, Any]] = []
    any_conflict = False
    for window in windows:
        leg_info = time_by_leg.get(window["leg"])
        if not isinstance(leg_info, dict):
            return make_result(False, "missing_binding", f"transport_time_window missing {window['leg']} evidence", repairable=True)
        actual_time = leg_info.get(window["field"])
        actual_min = _parse_time_to_minutes(actual_time)
        bound_min = _parse_time_to_minutes(window["value"])
        if actual_min is None or bound_min is None:
            return make_result(False, "missing_binding", "transport_time_window has unparsable time", repairable=True)
        operator = str(window["operator"])
        satisfied = actual_min >= bound_min if operator == ">=" else actual_min <= bound_min
        checked.append(
            {
                **window,
                "actual": actual_time,
                "citation": leg_info.get("citation"),
                "already_satisfied": satisfied,
            }
        )
        if not satisfied:
            any_conflict = True
    if not any_conflict:
        return make_result(
            False,
            "already_satisfied_or_not_stricter",
            "transport_time_window already satisfied by origin plan",
            {"windows": checked},
        )
    return make_result(True, "ok", "ok", {"windows": checked}, suggested_action="accept")


VALIDATORS: Dict[str, Validator] = {
    "duration_limit": _validate_duration_limit,
    "budget_cap_preference": _validate_budget_threshold,
    "budget_target_update": _validate_budget_threshold,
    "time_limit": _validate_time_limit,
    "time_window": _validate_time_window,
    "order_constraint": _validate_order_constraint,
    "same_day_no_overlap_compact": _validate_same_day_compact,
    "same_time_period_no_overlap": _validate_same_period_no_overlap,
    "remove_poi": _validate_remove_poi,
    "poi_local_radius_limit": _validate_anchor_distance_threshold,
    "max_distance_constraint": _validate_distance_threshold,
    "spatial_overflow": _validate_distance_threshold,
    "travel_time_constraint": _validate_transport_duration,
    "walking_distance_limit": _validate_distance_threshold,
    "access_time_infeasible": _validate_transport_duration,
    "budget_limit": _validate_budget_threshold,
    "ticket_budget_limit": _validate_ticket_budget,
    "ticket_price_cap_or_substitute": _validate_ticket_price_cap,
    "resource_overlap": _validate_resource_overlap,
    "required_restaurant_name": _validate_required_restaurant_name,
    "required_restaurant_type": _validate_required_restaurant_type,
    "required_hotel_name": _validate_required_hotel_name,
    "required_hotel_feature": _validate_required_hotel_feature,
    "required_room_type": _validate_required_room_type,
    "required_room_count": _validate_required_room_count,
    "activity_budget_limit": _validate_activity_budget_limit,
    "required_intercity_transport_type": _validate_required_intercity_transport_type,
    "transport_time_window": _validate_transport_time_window,
    "substitute_poi": _validate_substitute_semantic_family,
    "type_filter": _validate_type_filter_family,
    "famous_substitution": _validate_substitute_semantic_family,
    "preference_constraint": _validate_type_filter_family,
    "semantic_and_specific": _validate_semantic_and_specific,
    "category_overlap": _validate_semantic_and_specific,
    "semantic_substitution": _validate_semantic_substitution,
    "type_replacement": _validate_substitute_semantic_family,
    "multi_day_requirement": _validate_multi_day_requirement,
    "cross_day_restructure": _validate_cross_day_restructure,
    "add_days": _validate_add_days,
    "multi_city_split": _validate_multi_city_split,
    "extend_with_pacing": _validate_extend_with_pacing,
    "cross_day_variety_by_types": _validate_cross_day_variety,
    "multi_day_budget_overflow": _validate_budget_threshold,
}


def validate_conflict_trigger(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    constraint_type: str,
    edit_query: str = "",
    fallback_params: Optional[Dict[str, Any]] = None,
    template_str: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> FeasibilityResult:
    validator = VALIDATORS.get(str(constraint_type or "").strip())
    if validator is None:
        return make_result(
            False,
            "missing_validator",
            f"Missing feasibility validator for constraint_type={constraint_type}",
            repairable=False,
            suggested_action="fail",
        )

    evidence_payload = evidence or build_origin_plan_evidence(origin_plan)
    fallback = dict(fallback_params or {})
    fallback.setdefault("_constraint_type", str(constraint_type or "").strip())
    fallback.setdefault("_origin_plan", origin_plan)
    fallback.setdefault("_origin_itinerary", origin_plan.get("itinerary", []))
    return validator(evidence_payload, constraints or {}, edit_query or "", fallback, template_str)


def assess_edit_query_factual_validity(
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    constraint_type: Optional[str],
    edit_query: str = "",
    fallback_params: Optional[Dict[str, Any]] = None,
    template_str: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_constraint_type = str(constraint_type or "").strip()
    if not normalized_constraint_type:
        return {
            "is_valid": False,
            "label": "factually_invalid",
            "reason_code": "missing_constraint_type",
            "reason": "Missing constraint_type for factual validity assessment",
            "subtype": categorize_factual_invalid_reason("missing_constraint_type"),
            "plan_relation": "unknown",
            "evidence_used": {},
        }

    proof = validate_conflict_trigger(
        origin_plan=origin_plan,
        constraints=constraints,
        constraint_type=normalized_constraint_type,
        edit_query=edit_query,
        fallback_params=fallback_params,
        template_str=template_str,
        evidence=evidence,
    )
    reason_code = str(proof.reason_code or "").strip()
    if proof.ok:
        is_valid = True
        plan_relation = "requires_edit"
        invalidity_category = None
    elif reason_code in FACTUAL_VALID_NON_CONFLICTING_REASON_CODES:
        is_valid = False
        plan_relation = "already_satisfied_or_non_conflicting"
        invalidity_category = categorize_factual_invalid_reason(reason_code)
    elif reason_code in FACTUAL_ERROR_REASON_CODES:
        is_valid = False
        plan_relation = "unknown"
        invalidity_category = categorize_factual_invalid_reason(reason_code)
    else:
        is_valid = False
        plan_relation = "unknown"
        invalidity_category = "unknown"

    return {
        "is_valid": is_valid,
        "label": "factually_valid" if is_valid else "factually_invalid",
        "reason_code": reason_code,
        "reason": proof.reason,
        "subtype": invalidity_category,
        "plan_relation": plan_relation,
        "evidence_used": proof.evidence_used,
    }


def iter_eligible_constraint_types(templates_config: Dict[str, Any]) -> Set[str]:
    constraint_types: Set[str] = set()
    for _, value in templates_config.items():
        if not isinstance(value, dict) or not value.get("eligible_for_edit", False):
            continue
        templates = value.get("templates", [])
        if not isinstance(templates, list):
            continue
        for template in templates:
            if not isinstance(template, dict):
                continue
            constraint_type = template.get("constraint_type")
            if isinstance(constraint_type, str) and constraint_type.strip():
                constraint_types.add(constraint_type.strip())
    return constraint_types


def ensure_validator_coverage(templates_config: Dict[str, Any]) -> List[str]:
    eligible_types = iter_eligible_constraint_types(templates_config)
    missing = sorted(constraint_type for constraint_type in eligible_types if constraint_type not in VALIDATORS)
    return missing
