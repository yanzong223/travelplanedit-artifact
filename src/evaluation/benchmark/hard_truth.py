"""Canonical hard-truth builders for benchmark logical constraints."""

from __future__ import annotations

import ast
import re
import sys
from functools import lru_cache
from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHINATRAVEL_ROOT = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
UTILS_ROOT = PROJECT_ROOT / "data_generation" / "utils"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(UTILS_ROOT) not in sys.path:
    sys.path.insert(0, str(UTILS_ROOT))
if str(CHINATRAVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_ROOT))

from .models import LogicalConstraintObject

try:
    from data_generation.utils.query_surface_parsing import (
        extract_daily_poi_cap_from_query,
        extract_intercity_transport_modes_from_query,
        extract_preferred_poi_type_from_query,
        extract_transport_time_windows_from_query,
        normalize_intercity_transport_mode,
        parse_query_semantic_slots,
    )
except ModuleNotFoundError:
    from query_surface_parsing import (
        extract_daily_poi_cap_from_query,
        extract_intercity_transport_modes_from_query,
        extract_preferred_poi_type_from_query,
        extract_transport_time_windows_from_query,
        normalize_intercity_transport_mode,
        parse_query_semantic_slots,
    )

_SET_PATTERN = re.compile(r"result=\((\{.*?\})(==|<=)([A-Za-z_]+)\)", re.S)
_DAY_PATTERN = re.compile(r"day_count\(plan\)\s*([<>=!]+)\s*(\d+)")
_PEOPLE_PATTERN = re.compile(r"people_count\(plan\)\s*([<>=!]+)\s*(\d+)")
_BUDGET_PATTERN = re.compile(r"total_cost.*?<=\s*(\d+(?:\.\d+)?)", re.S)
_HOTEL_COST_PATTERN = re.compile(
    r"hotel_cost/people_count\(plan\)/\(day_count\(plan\)-1\)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)"
)
_FOOD_COST_PATTERN = re.compile(
    r"food_cost/food_count/people_count\(plan\)\s*([<>=!]+)\s*(\d+(?:\.\d+)?)"
)

_DAY_TOKEN_MAP = {
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
    "十": 10,
}
_COUNT_TOKEN_PATTERN = r"[一二三四五六七八九十两\d]+"

_TIME_PERIOD_MAP = {
    "上午": "morning",
    "早上": "morning",
    "清晨": "morning",
    "下午": "afternoon",
    "傍晚": "evening",
    "晚上": "evening",
    "晚间": "evening",
    "夜间": "night",
    "夜里": "night",
}


def _normalize_time_period(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = _TIME_PERIOD_MAP.get(text, text.lower())
    return normalized if normalized in {"morning", "afternoon", "evening", "night"} else None


def _parse_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value or "").strip()
    if not text or not text.isdigit():
        return None
    parsed = int(text)
    return parsed if parsed > 0 else None

_TRANSIT_HUB_KEYWORDS = (
    "火车站",
    "高铁站",
    "机场",
    "客运站",
    "汽车站",
    "地铁站",
    "轻轨站",
)
_TRANSIT_HUB_SUFFIXES = ("东站", "西站", "南站", "北站", "总站")
_TRANSIT_HUB_ALLOWLIST = {"李子坝轻轨站"}
_VIBE_SOFT_KEYWORDS = (
    "氛围",
    "感觉",
    "松弛",
    "放松",
    "轻松",
    "热闹",
    "安静",
    "安逸",
    "浪漫",
    "文艺",
    "夜生活",
    "夜店",
    "酒吧",
    "lively",
    "relaxed",
    "quiet",
    "nightlife",
)
_SUPPORTED_DATASET_CITIES = (
    "北京",
    "上海",
    "南京",
    "苏州",
    "杭州",
    "深圳",
    "成都",
    "武汉",
    "广州",
    "重庆",
)
_NEARBY_CITY_MAP = {
    "北京": "南京",
    "上海": "苏州",
    "南京": "苏州",
    "苏州": "上海",
    "杭州": "苏州",
    "深圳": "广州",
    "成都": "重庆",
    "武汉": "南京",
    "广州": "深圳",
    "重庆": "成都",
}


def _normalize_sequence_field(value: Any) -> List[Any]:
    """将输入值规范化为列表。

    支持 list、字符串（含 JSON-like 格式 '[]' / '()'）等类型，
    统一返回 List[Any]。空值或无法解析时返回空列表。
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("("):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, tuple):
                return list(parsed)
        return [text]
    return []


def _parse_set_literal(text: str) -> List[str]:
    """解析形如 "{'火车', '飞机'}" 的集合字面量字符串，返回去引号后的元素列表。"""
    return [item.strip().strip("'\"") for item in text.strip("{} ").split(",") if item.strip()]


def _make_logical(
    index: int,
    constraint_type: str,
    value: Any,
    *,
    source: str,
    operator: str = "==",
    target: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    raw_code: Optional[str] = None,
) -> LogicalConstraintObject:
    """统一工厂函数，创建一个 LogicalConstraintObject 实例。

    Args:
        index: 约束序号，用于生成唯一 id。
        constraint_type: 约束类型，如 'day_count'、'budget_total' 等。
        value: 约束的目标值（数字、字符串或列表）。
        source: 约束来源，'origin'（原始查询）或 'edit'（编辑指令）。
        operator: 比较运算符，如 '=='、'<='、'contains_all'。
        target: 约束作用的对象描述，如 {'activity_type': 'accommodation'}。
        params: 补充参数，如 {'metric': 'avg_cost_per_person_per_night'}。
        raw_code: 原始 Python 表达式代码（用于溯源）。

    Returns:
        构建好的 LogicalConstraintObject。
    """
    provenance: Dict[str, Any] = {"builder": "hard_truth", "source": source}
    if raw_code:
        provenance["raw_code"] = raw_code
    return LogicalConstraintObject(
        id=f"{source}_logic_{index}",
        source=source,
        type=constraint_type,
        scope={"level": f"{source}_query"},
        target=target or {},
        operator=operator,
        value=value,
        params=params or {},
        is_hard=True,
        provenance=provenance,
    )


def _normalize_day_token(day_token: str) -> Optional[int]:
    """将中文或阿拉伯数字的天数标记转换为整数。

    支持 '一'~'十' 的中文数字和纯阿拉伯数字，无法识别时返回 None。
    """
    token = str(day_token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if token in _DAY_TOKEN_MAP:
        return _DAY_TOKEN_MAP[token]
    if "十" in token:
        left, _, right = token.partition("十")
        tens = _DAY_TOKEN_MAP.get(left, 1 if not left else None)
        ones = _DAY_TOKEN_MAP.get(right, 0 if not right else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return None


def _extract_day_reference(text: str) -> Optional[int]:
    match = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*天", text or "")
    if not match:
        return None
    return _normalize_day_token(match.group(1))


def _parse_duration_to_minutes(amount_text: str, unit: str) -> Optional[int]:
    try:
        amount = float(amount_text)
    except (TypeError, ValueError):
        return None
    minutes = int(amount * 60) if unit == "小时" else int(amount)
    return minutes if minutes > 0 else None


def _collect_city_tokens(
    origin_plan: Optional[Dict[str, Any]],
    origin_query_structured: Optional[Dict[str, Any]],
) -> set[str]:
    tokens: set[str] = set()
    for payload in (origin_plan or {}, origin_query_structured or {}):
        for key in ("target_city", "start_city"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                tokens.add(value.strip())
    return tokens


def _resolve_target_city(
    origin_plan: Optional[Dict[str, Any]],
    origin_query_structured: Optional[Dict[str, Any]],
) -> Optional[str]:
    for payload in (origin_plan or {}, origin_query_structured or {}):
        value = payload.get("target_city")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_city_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in {"当前目标城市", "目标城市", "本次目标城市", "本地目标城市"}:
        return "__TARGET_CITY__"
    if text in {"另一个城市", "周边另一个城市", "附近另一个城市", "周边城市", "附近城市"}:
        return "__NEARBY_CITY__"
    for city in _SUPPORTED_DATASET_CITIES:
        if city == text:
            return city
    return text


def _resolve_city_split_targets(
    *,
    target_city: Optional[str],
    query_spec: Optional[Dict[str, Any]],
    semantic_slots: Optional[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]],
) -> List[str]:
    resolved: List[str] = []
    semantic = constraints.get("semantic") if isinstance(constraints, dict) and isinstance(constraints.get("semantic"), dict) else {}
    candidate_lists = (
        query_spec.get("city_split_targets") if isinstance(query_spec, dict) else None,
        semantic_slots.get("city_split_targets") if isinstance(semantic_slots, dict) else None,
        semantic.get("city_split_targets"),
    )
    for raw in candidate_lists:
        if not isinstance(raw, list):
            continue
        normalized = [_normalize_city_name(item) for item in raw]
        cleaned = [item for item in normalized if item]
        if cleaned:
            resolved = cleaned
            break
    if len(resolved) < 2:
        return []
    if "__NEARBY_CITY__" in resolved:
        # Enforce explicit secondary city instead of latent placeholders.
        return []

    final: List[str] = []
    for item in resolved:
        if item == "__TARGET_CITY__":
            item = str(target_city or "").strip()
        if item and item not in final:
            final.append(item)
    return final[:2] if len(final) >= 2 else []


def _freeze_constraint_key(constraint: LogicalConstraintObject) -> str:
    return json.dumps(
        {
            "type": constraint.type,
            "target": constraint.target,
            "operator": constraint.operator,
            "value": constraint.value,
            "params": constraint.params,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _append_unique_constraint(
    items: List[LogicalConstraintObject],
    constraint: LogicalConstraintObject,
    seen: set[str],
) -> None:
    key = _freeze_constraint_key(constraint)
    if key in seen:
        return
    seen.add(key)
    items.append(constraint)


def build_edit_logical_constraints_from_structured_inputs(
    *,
    constraint_type: str,
    query_spec: Optional[Dict[str, Any]] = None,
    semantic_slots: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    origin_plan: Optional[Dict[str, Any]] = None,
    origin_query_structured: Optional[Dict[str, Any]] = None,
) -> List[LogicalConstraintObject]:
    """Build edit-side hard truth from frozen structured step-2 artifacts.

    This builder is intentionally stricter than ``build_edit_logical_constraints``:
    it only trusts explicit structured inputs and never infers extra truth from
    free-form surface wording.
    """

    ctype = str(constraint_type or "").strip()
    if not ctype:
        raise ValueError("constraint_type is required for structured hard-truth building")

    query_spec = query_spec if isinstance(query_spec, dict) else {}
    semantic_slots = semantic_slots if isinstance(semantic_slots, dict) else {}
    constraints = constraints if isinstance(constraints, dict) else {}
    target_city = _resolve_target_city(origin_plan, origin_query_structured)

    extracted: List[LogicalConstraintObject] = []
    seen: set[str] = set()

    def add(
        logical_type: str,
        value: Any,
        *,
        operator: str = "==",
        target: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        _append_unique_constraint(
            extracted,
            _make_logical(
                len(extracted),
                logical_type,
                value,
                source="edit",
                operator=operator,
                target=target,
                params=params,
            ),
            seen,
        )

    def string_value(*candidates: Any) -> str:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    def int_value(*candidates: Any) -> Optional[int]:
        for candidate in candidates:
            if isinstance(candidate, bool):
                continue
            if isinstance(candidate, int):
                return int(candidate)
            if isinstance(candidate, float):
                return int(candidate)
            if isinstance(candidate, str) and candidate.strip().isdigit():
                return int(candidate.strip())
        return None

    def float_value(*candidates: Any) -> Optional[float]:
        for candidate in candidates:
            if isinstance(candidate, bool):
                continue
            if isinstance(candidate, (int, float)):
                return float(candidate)
            if isinstance(candidate, str):
                try:
                    return float(candidate.strip())
                except ValueError:
                    continue
        return None

    def number_from_text(*candidates: Any) -> Optional[float]:
        for candidate in candidates:
            numeric = float_value(candidate)
            if numeric is not None:
                return numeric
            if isinstance(candidate, str):
                match = re.search(r"-?\d+(?:\.\d+)?", candidate)
                if match:
                    return float(match.group(0))
        return None

    def intercity_modes_from_values(*candidates: Any) -> List[str]:
        modes: List[str] = []
        for candidate in candidates:
            values = candidate if isinstance(candidate, list) else [candidate]
            for value in values:
                normalized = normalize_intercity_transport_mode(value)
                if normalized and normalized not in modes:
                    modes.append(normalized)
        return modes

    def duration_minutes_value(*candidates: Any) -> Optional[int]:
        for candidate in candidates:
            numeric = int_value(candidate)
            if numeric is not None:
                return numeric
            if isinstance(candidate, str):
                text = candidate.strip()
                if not text:
                    continue
                match = re.search(r"(\d+(?:\.\d+)?)", text)
                if not match:
                    continue
                value = float(match.group(1))
                if "小时" in text or "hour" in text.lower():
                    return int(value * 60)
                return int(value)
        return None

    def iter_must_include() -> List[Dict[str, Any]]:
        raw = constraints.get("must_include")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def constraint_family(raw_type: str) -> str:
        if raw_type == "poi_local_radius_limit":
            return "anchor_commute_distance_cap"
        if raw_type == "walking_distance_limit":
            return "pairwise_walking_distance_cap"
        if raw_type == "travel_time_constraint":
            return "adjacent_travel_time_cap"
        if raw_type == "access_time_infeasible":
            return "poi_inbound_travel_time_cap"
        return raw_type

    def required_poi_names() -> List[str]:
        names: List[str] = []
        for item in iter_must_include():
            name = string_value(item.get("name"))
            poi_type = string_value(item.get("type"))
            if not name or poi_type not in {"", "attraction"}:
                continue
            if name not in names:
                names.append(name)
        return names

    def first_required_poi_name() -> str:
        names = required_poi_names()
        return names[0] if names else ""

    def forbidden_poi_names() -> List[str]:
        raw = constraints.get("forbidden")
        if not isinstance(raw, list):
            return []
        names: List[str] = []
        for item in raw:
            name = string_value(item)
            if name and name not in names:
                names.append(name)
        return names

    def ordered_poi_pair() -> List[str]:
        ordered = query_spec.get("ordered_pois")
        if isinstance(ordered, list):
            pair = [string_value(item) for item in ordered if string_value(item)]
            if len(pair) >= 2:
                return pair[:2]
        names = required_poi_names()
        if len(names) >= 2:
            return names[:2]
        return []

    family = constraint_family(ctype)

    if ctype in {"budget_limit", "daily_budget_infeasible", "budget_cap_preference", "budget_target_update", "multi_day_budget_overflow"}:
        budget = float_value(query_spec.get("budget_cap"), semantic_slots.get("budget_cap"))
        if budget is not None:
            add("budget_total", budget, operator="<=", params={"derived_from": "query_spec"})

    if ctype == "ticket_budget_limit":
        budget = float_value(
            query_spec.get("ticket_budget_cap"),
            query_spec.get("budget_cap"),
            semantic_slots.get("ticket_budget_cap"),
            semantic_slots.get("budget_cap"),
        )
        if budget is not None:
            add("ticket_budget_total", budget, operator="<=", params={"derived_from": "query_spec"})

    if ctype == "required_intercity_transport_type":
        modes = intercity_modes_from_values(
            query_spec.get("intercity_transport_modes"),
            semantic_slots.get("intercity_transport_modes"),
            constraints.get("transport", {}).get("intercity_modes") if isinstance(constraints.get("transport"), dict) else None,
            constraints.get("semantic", {}).get("intercity_transport_modes") if isinstance(constraints.get("semantic"), dict) else None,
        )
        if modes:
            add(
                "required_intercity_transport_type",
                modes,
                operator="contains_all",
                target={"activity_type": "intercity_transport"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "transport_time_window":
        windows = query_spec.get("transport_time_windows") or semantic_slots.get("transport_time_windows") or []
        if isinstance(windows, list):
            for window in windows:
                if not isinstance(window, dict):
                    continue
                leg = string_value(window.get("leg"))
                field = string_value(window.get("field"))
                operator = string_value(window.get("operator")) or "<="
                value = string_value(window.get("value"))
                if leg and field and value:
                    add(
                        "transport_time_window",
                        value,
                        operator=operator,
                        params={
                            "leg": leg,
                            "field": field,
                            "derived_from": "query_spec",
                        },
                    )

    if ctype in {"multi_day_requirement", "add_days", "multi_day_budget_overflow", "extend_with_pacing", "cross_day_restructure"}:
        day_count = int_value(query_spec.get("day_count"), semantic_slots.get("day_count"))
        if day_count is not None:
            add("day_count", day_count, operator="==", params={"derived_from": "query_spec"})

    if ctype == "add_days":
        preferred_type = string_value(
            query_spec.get("preferred_type"),
            semantic_slots.get("preferred_type"),
            constraints.get("semantic", {}).get("preferred_type") if isinstance(constraints.get("semantic"), dict) else None,
        )
        if preferred_type:
            add(
                "semantic_type_requirement",
                preferred_type,
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "query_spec"},
            )

    if ctype == "time_limit":
        day_value = int_value(query_spec.get("day"), semantic_slots.get("day"))
        end_before = string_value(
            query_spec.get("end_before"),
            semantic_slots.get("end_before"),
        )
        if day_value is not None and end_before:
            add(
                "day_end_time_limit",
                end_before,
                operator="<=",
                target={"day": day_value},
                params={"derived_from": "query_spec"},
            )

    if ctype == "duration_limit":
        poi_name = string_value(
            query_spec.get("anchor_poi"),
            semantic_slots.get("anchor_poi"),
            first_required_poi_name(),
        )
        duration = duration_minutes_value(
            query_spec.get("duration_limit_min"),
            semantic_slots.get("duration_limit_min"),
        )
        if poi_name and duration is not None:
            _raise_if_transport_hub_name(poi_name, context="activity_duration_limit")
            add(
                "activity_duration_limit",
                duration,
                operator="<=",
                target={"poi_name": poi_name},
                params={"unit": "minute", "derived_from": "query_spec"},
            )

    if ctype == "time_window":
        poi_name = string_value(
            query_spec.get("anchor_poi"),
            semantic_slots.get("anchor_poi"),
            first_required_poi_name(),
        )
        time_period = _normalize_time_period(string_value(
            query_spec.get("time_period"),
            semantic_slots.get("time_period"),
        ))
        day_value = int_value(query_spec.get("day"), semantic_slots.get("day"))
        if poi_name and time_period:
            _raise_if_transport_hub_name(poi_name, context="poi_time_window")
            add(
                "poi_time_window",
                time_period,
                operator="in",
                target={"poi_name": poi_name},
                params={"day": day_value, "derived_from": "query_spec"},
            )

    if ctype == "order_constraint":
        pair = ordered_poi_pair()
        if len(pair) >= 2:
            for poi_name in pair[:2]:
                _raise_if_transport_hub_name(poi_name, context="poi_order_constraint")
            add(
                "poi_order_constraint",
                pair[:2],
                operator="before",
                target={"first_poi": pair[0], "second_poi": pair[1]},
                params={"derived_from": "query_spec"},
            )

    if ctype == "same_day_no_overlap_compact":
        pair = ordered_poi_pair()
        day_value = int_value(query_spec.get("day"), semantic_slots.get("day"))
        if len(pair) >= 2 and day_value is not None:
            for poi_name in pair[:2]:
                _raise_if_transport_hub_name(poi_name, context="pair_same_day_no_overlap")
            add(
                "pair_same_day_no_overlap",
                day_value,
                operator="==",
                target={"first_poi": pair[0], "second_poi": pair[1]},
                params={"derived_from": "query_spec"},
            )

    if ctype == "same_time_period_no_overlap":
        pair = ordered_poi_pair()
        time_period = _normalize_time_period(string_value(query_spec.get("time_period"), semantic_slots.get("time_period")))
        if len(pair) >= 2 and time_period:
            for poi_name in pair[:2]:
                _raise_if_transport_hub_name(poi_name, context="pair_time_window_no_overlap")
            add(
                "pair_time_window_no_overlap",
                time_period,
                operator="in",
                target={"first_poi": pair[0], "second_poi": pair[1]},
                params={"derived_from": "query_spec"},
            )

    if ctype == "extend_with_pacing":
        daily_poi_cap = int_value(query_spec.get("daily_poi_cap"), semantic_slots.get("daily_poi_cap"))
        if daily_poi_cap is not None:
            add(
                "daily_poi_cap",
                daily_poi_cap,
                operator="<=",
                target={"day": "all"},
                params={"count_types": ["attraction"], "derived_from": "query_spec"},
            )

    if ctype in {"semantic_and_specific", "category_overlap"}:
        specific_poi = string_value(query_spec.get("specific_poi"), semantic_slots.get("specific_poi"))
        preferred_type = string_value(query_spec.get("preferred_type"), semantic_slots.get("preferred_type"))
        if specific_poi:
            _raise_if_transport_hub_name(specific_poi, context="required_attraction_name")
            add(
                "required_attraction_name",
                [specific_poi],
                operator="contains_all",
                target={"activity_type": "attraction"},
                params={"derived_from": "query_spec"},
            )
        if preferred_type:
            add(
                "semantic_type_requirement",
                preferred_type,
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "query_spec"},
            )

    if ctype in {"type_filter", "preference_constraint"}:
        preferred_type = string_value(query_spec.get("preferred_type"), semantic_slots.get("preferred_type"))
        if preferred_type:
            add(
                "semantic_type_requirement",
                preferred_type,
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "query_spec"},
            )

    if ctype == "cross_day_variety_by_types":
        for day_value, preferred_type in _semantic_type_by_day(
            query_spec=query_spec,
            semantic_slots=semantic_slots,
            constraints=constraints,
        ).items():
            add(
                "semantic_type_requirement",
                preferred_type,
                operator="includes_type",
                target={"activity_type": "attraction", "day": day_value},
                params={
                    "day": day_value,
                    "min_count": 1,
                    "strict_majority": True,
                    "derived_from": "constraints.semantic.preferred_type",
                },
            )

    if ctype == "multi_city_split":
        raw_split_targets = None
        for payload in (
            query_spec.get("city_split_targets"),
            semantic_slots.get("city_split_targets"),
            constraints.get("semantic", {}).get("city_split_targets") if isinstance(constraints.get("semantic"), dict) else None,
        ):
            if isinstance(payload, list) and payload:
                raw_split_targets = payload
                break
        split_targets = _resolve_city_split_targets(
            target_city=target_city,
            query_spec=query_spec,
            semantic_slots=semantic_slots,
            constraints=constraints,
        )
        if isinstance(raw_split_targets, list) and len(raw_split_targets) >= 2 and len(split_targets) < 2:
            raise ValueError("multi_city_split requires explicit secondary city; latent placeholders are not allowed")
        if len(split_targets) >= 2:
            add(
                "city_split_requirement",
                split_targets[:2],
                operator="contains_all",
                target={"target_city": split_targets[0], "secondary_city": split_targets[1]},
                params={
                    "supported_cities": list(_SUPPORTED_DATASET_CITIES),
                    "derived_from": "query_spec",
                },
            )

    if ctype in {"remove_poi", "substitute_poi", "famous_substitution", "type_replacement"}:
        source_poi = string_value(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_poi_names()[:1]),
        )
        if source_poi:
            _raise_if_transport_hub_name(source_poi, context="forbidden_attraction_name")
            add(
                "forbidden_attraction_name",
                [source_poi],
                operator="contains_none",
                target={"activity_type": "attraction"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "type_replacement":
        source_poi = string_value(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_poi_names()[:1]),
        )
        semantic_target = string_value(
            query_spec.get("semantic_target"),
            query_spec.get("preferred_type"),
            semantic_slots.get("semantic_target"),
            semantic_slots.get("preferred_type"),
            constraints.get("semantic", {}).get("semantic_target") if isinstance(constraints.get("semantic"), dict) else None,
            constraints.get("semantic", {}).get("preferred_type") if isinstance(constraints.get("semantic"), dict) else None,
        )
        if source_poi and semantic_target:
            add(
                "semantic_type_requirement",
                semantic_target,
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "query_spec", "replace_from": source_poi, "from_type_replacement": True},
            )

    if ctype == "substitute_poi":
        source_poi = string_value(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_poi_names()[:1]),
        )
        semantic_target = string_value(
            query_spec.get("semantic_target"),
            semantic_slots.get("semantic_target"),
            query_spec.get("preferred_type"),
            semantic_slots.get("preferred_type"),
            constraints.get("semantic", {}).get("semantic_target") if isinstance(constraints.get("semantic"), dict) else None,
        )
        candidate_names: List[str] = []
        if source_poi and not semantic_target:
            if not target_city:
                raise ValueError("target_city is required for substitute_poi semantic replacement inference")
            poi_record = _lookup_named_poi(target_city, source_poi)
            if poi_record is None or str(poi_record.get("category", "")) != "attraction":
                raise ValueError(f"substitute_poi target '{source_poi}' not found as attraction in database")
            semantic_target = str(poi_record.get("semantic_type", "") or "")
            if semantic_target:
                candidate_names = [
                    name
                    for name in _same_type_candidates(target_city, "attraction", semantic_target)
                    if name != source_poi
                ]
                if not candidate_names:
                    raise ValueError(f"substitute_poi has no same-type substitute candidate for '{source_poi}'")
        if source_poi and semantic_target:
            add(
                "semantic_type_requirement",
                semantic_target,
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={
                    "min_count": 1,
                    "derived_from": "query_spec",
                    "replace_from": source_poi,
                    "from_substitute_poi": True,
                    **({"candidate_names": candidate_names} if candidate_names else {}),
                },
            )

    if ctype == "semantic_substitution":
        source_poi = string_value(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_poi_names()[:1]),
        )
        semantic_target = string_value(
            query_spec.get("semantic_target"),
            semantic_slots.get("semantic_target"),
            constraints.get("semantic", {}).get("semantic_target") if isinstance(constraints.get("semantic"), dict) else None,
            semantic_slots.get("preferred_type"),
        )
        if source_poi:
            _raise_if_transport_hub_name(source_poi, context="semantic_substitution")
            add(
                "forbidden_attraction_name",
                [source_poi],
                operator="contains_none",
                target={"activity_type": "attraction"},
                params={"derived_from": "query_spec", "from_semantic_substitution": True},
            )
        if semantic_target:
            add(
                "semantic_type_requirement",
                semantic_target,
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={
                    "min_count": 1,
                    "derived_from": "query_spec",
                    **({"replace_from": source_poi} if source_poi else {}),
                    "from_semantic_substitution": True,
                },
            )

    if ctype == "famous_substitution":
        source_poi = string_value(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_poi_names()[:1]),
        )
        if source_poi:
            add(
                "fame_replacement_requirement",
                string_value(
                    query_spec.get("semantic_target"),
                    semantic_slots.get("semantic_target"),
                    constraints.get("semantic", {}).get("semantic_target") if isinstance(constraints.get("semantic"), dict) else None,
                    "更有名",
                ),
                operator="satisfies",
                target={"activity_type": "attraction"},
                params={"replace_from": source_poi, "derived_from": "query_spec"},
            )

    if ctype == "required_restaurant_type":
        restaurant_type = string_value(query_spec.get("restaurant_type"), semantic_slots.get("restaurant_type"))
        if restaurant_type:
            add(
                "required_restaurant_type",
                [restaurant_type],
                operator="contains_all",
                target={"activity_type": "meal"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "required_hotel_feature":
        hotel_feature = string_value(query_spec.get("hotel_feature"), semantic_slots.get("hotel_feature"))
        if hotel_feature:
            add(
                "required_hotel_feature",
                [hotel_feature],
                operator="contains_all",
                params={"derived_from": "query_spec"},
            )

    if ctype == "required_hotel_name":
        hotel_name = string_value(query_spec.get("anchor_poi"), semantic_slots.get("hotel_name"))
        if hotel_name:
            add(
                "required_hotel_name",
                [hotel_name],
                operator="contains_all",
                target={"activity_type": "accommodation"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "required_restaurant_name":
        restaurant_name = string_value(query_spec.get("anchor_poi"), semantic_slots.get("restaurant_name"))
        if restaurant_name:
            add(
                "required_restaurant_name",
                [restaurant_name],
                operator="contains_all",
                target={"activity_type": "meal"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "activity_budget_limit":
        activity_type = string_value(query_spec.get("activity_type"))
        metric = string_value(query_spec.get("metric"))
        budget = float_value(query_spec.get("budget_cap"), semantic_slots.get("budget_cap"))
        if activity_type and metric and budget is not None:
            add(
                "activity_budget_limit",
                budget,
                operator="<=",
                target={"activity_type": activity_type},
                params={"metric": metric, "derived_from": "query_spec"},
            )

    if ctype == "resource_overlap":
        anchor_poi = string_value(query_spec.get("anchor_poi"), semantic_slots.get("anchor_poi"))
        budget = float_value(query_spec.get("budget_cap"), semantic_slots.get("budget_cap"))
        nearby_meal = bool(query_spec.get("nearby_meal") or semantic_slots.get("nearby_meal"))
        nearby_params = {"derived_from": "query_spec"}
        for key in (
            "max_distance_km",
            "selected_meal_name",
            "origin_bundle_cost",
            "local_bundle_lower_bound",
            "anchor_meal_distance_km",
        ):
            value = query_spec.get(key, semantic_slots.get(key))
            if value is not None:
                nearby_params[key] = value
        if anchor_poi:
            add(
                "required_attraction_name",
                [anchor_poi],
                operator="contains_all",
                target={"activity_type": "attraction"},
                params={"derived_from": "query_spec"},
            )
        if anchor_poi and nearby_meal:
            add(
                "nearby_meal_requirement",
                True,
                operator="==",
                target={"anchor_poi": anchor_poi},
                params=nearby_params,
            )
        if anchor_poi and budget is not None:
            add(
                "anchor_bundle_budget_limit",
                budget,
                operator="<=",
                target={"anchor_poi": anchor_poi},
                params={
                    "budget_scope": "anchor_bundle",
                    "requires_nearby_meal": nearby_meal,
                    **nearby_params,
                },
            )

    if family == "anchor_commute_distance_cap":
        anchor_name = string_value(
            query_spec.get("anchor_poi"),
            semantic_slots.get("anchor_poi"),
        )
        if not anchor_name:
            attraction_names = required_poi_names()
            if len(attraction_names) == 1:
                anchor_name = attraction_names[0]
        distance = number_from_text(
            query_spec.get("distance_km"),
            semantic_slots.get("distance_km"),
            constraints.get("spatial", {}).get("max_walking_distance") if isinstance(constraints.get("spatial"), dict) else None,
            constraints.get("semantic", {}).get("max_distance_km") if isinstance(constraints.get("semantic"), dict) else None,
            constraints.get("semantic", {}).get("chosen_walk_dist_km") if isinstance(constraints.get("semantic"), dict) else None,
        )
        if anchor_name and distance is not None:
            add(
                "anchor_neighbor_commute_distance_cap",
                distance,
                operator="<=",
                target={"anchor_name": anchor_name},
                params={"unit": "km", "derived_from": "query_spec"},
            )

    if family == "pairwise_walking_distance_cap":
        ordered_pois = query_spec.get("ordered_pois")
        pair: List[str] = []
        if isinstance(ordered_pois, list):
            pair = [string_value(item) for item in ordered_pois if string_value(item)]
        if len(pair) < 2:
            attraction_names = required_poi_names()
            if len(attraction_names) >= 2:
                pair = attraction_names[:2]
        distance = number_from_text(
            query_spec.get("distance_km"),
            semantic_slots.get("distance_km"),
            constraints.get("spatial", {}).get("max_walking_distance") if isinstance(constraints.get("spatial"), dict) else None,
            constraints.get("semantic", {}).get("max_distance_km") if isinstance(constraints.get("semantic"), dict) else None,
            constraints.get("semantic", {}).get("chosen_walk_dist_km") if isinstance(constraints.get("semantic"), dict) else None,
        )
        if len(pair) >= 2 and distance is not None:
            for poi_name in pair[:2]:
                _raise_if_transport_hub_name(poi_name, context="pairwise_transport_mode_distance_cap")
            add(
                "pairwise_transport_mode_distance_cap",
                distance,
                operator="<=",
                target={"first_poi": pair[0], "second_poi": pair[1], "mode": "walk"},
                params={"unit": "km", "derived_from": "query_spec"},
            )

    if family == "adjacent_travel_time_cap":
        temporal_constraints = constraints.get("temporal")
        max_duration = None
        if isinstance(temporal_constraints, list):
            for item in temporal_constraints:
                if isinstance(item, dict) and item.get("max_duration") is not None:
                    max_duration = item.get("max_duration")
                    break
        duration = duration_minutes_value(
            query_spec.get("duration_limit_min"),
            semantic_slots.get("duration_limit_min"),
            max_duration,
        )
        if duration is not None:
            add(
                "adjacent_travel_time_cap",
                duration,
                operator="<=",
                target={"activity_types": ["attraction"]},
                params={"unit": "minute", "day": "all", "derived_from": "query_spec"},
            )

    if family == "poi_inbound_travel_time_cap":
        temporal_constraints = constraints.get("temporal")
        max_duration = None
        if isinstance(temporal_constraints, list):
            for item in temporal_constraints:
                if isinstance(item, dict) and item.get("max_duration") is not None:
                    max_duration = item.get("max_duration")
                    break
        duration = duration_minutes_value(
            query_spec.get("duration_limit_min"),
            semantic_slots.get("duration_limit_min"),
            max_duration,
        )
        poi_name = string_value(
            query_spec.get("source_poi"),
            query_spec.get("specific_poi"),
            query_spec.get("anchor_poi"),
            semantic_slots.get("source_poi"),
            semantic_slots.get("specific_poi"),
            semantic_slots.get("anchor_poi"),
            semantic_slots.get("restaurant_name"),
            semantic_slots.get("hotel_name"),
        )
        if not poi_name:
            attraction_names = required_poi_names()
            if len(attraction_names) == 1:
                poi_name = attraction_names[0]
        if poi_name and duration is not None:
            _raise_if_transport_hub_name(poi_name, context="poi_inbound_travel_time_cap")
            add(
                "poi_inbound_travel_time_cap",
                duration,
                operator="<=",
                target={"poi_name": poi_name},
                params={"unit": "minute", "derived_from": "query_spec"},
            )

    for day_value, poi_name in _normalized_day_poi_pairs(query_spec.get("day_poi_pairs"), semantic_slots.get("day_poi_pairs")):
            _raise_if_transport_hub_name(poi_name, context="poi_day_binding")
            add(
                "poi_day_binding",
                day_value,
                operator="==",
                target={"poi_name": poi_name},
                params={"derived_from": "query_spec"},
            )

    for poi in iter_must_include():
        poi_name = string_value(poi.get("name"))
        poi_type = string_value(poi.get("type"))
        if not poi_name:
            continue
        if poi_type == "restaurant":
            add(
                "required_restaurant_name",
                [poi_name],
                operator="contains_all",
                target={"activity_type": "meal"},
                params={"derived_from": "constraints.must_include"},
            )
        elif poi_type == "accommodation":
            add(
                "required_hotel_name",
                [poi_name],
                operator="contains_all",
                target={"activity_type": "accommodation"},
                params={"derived_from": "constraints.must_include"},
            )
        elif ctype not in {"semantic_and_specific", "category_overlap", "resource_overlap"}:
            add(
                "required_attraction_name",
                [poi_name],
                operator="contains_all",
                target={"activity_type": "attraction"},
                params={"derived_from": "constraints.must_include"},
            )

    if ctype == "ticket_price_cap_or_substitute":
        source_poi = string_value(query_spec.get("source_poi"))
        price_cap = float_value(query_spec.get("ticket_price_cap"))
        if source_poi and price_cap is not None:
            if not target_city:
                raise ValueError("target_city is required for ticket_price_cap_or_substitute extraction")
            poi_record = _lookup_named_poi(target_city, source_poi)
            if poi_record is None:
                raise ValueError(f"ticket_price_cap_or_substitute target '{source_poi}' not found in database")
            target_eligible = float(poi_record["price"]) <= price_cap
            candidate_names: List[str] = []
            if not target_eligible:
                candidate_names = _same_type_price_capped_candidates(
                    target_city,
                    str(poi_record["category"]),
                    poi_record["semantic_type"],
                    price_cap,
                )
                candidate_names = [name for name in candidate_names if name != source_poi]
            activity_type = "attraction"
            if poi_record["category"] == "restaurant":
                activity_type = "meal"
            elif poi_record["category"] == "accommodation":
                activity_type = "accommodation"
            add(
                "ticket_price_cap_or_substitute",
                price_cap,
                operator="<=",
                target={"poi_name": source_poi, "activity_type": activity_type},
                params={
                    "day": query_spec.get("day"),
                    "city": target_city,
                    "poi_category": poi_record["category"],
                    "semantic_type": poi_record["semantic_type"],
                    "target_price": float(poi_record["price"]),
                    "target_eligible": target_eligible,
                    "candidate_names": candidate_names,
                    "derived_from": "query_spec",
                },
            )

    return extracted


def validate_structured_edit_constraint_coverage(
    *,
    constraint_type: str,
    query_spec: Optional[Dict[str, Any]] = None,
    semantic_slots: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    built_constraints: Optional[List[LogicalConstraintObject]] = None,
) -> None:
    ctype = str(constraint_type or "").strip()
    query_spec = query_spec if isinstance(query_spec, dict) else {}
    semantic_slots = semantic_slots if isinstance(semantic_slots, dict) else {}
    constraints = constraints if isinstance(constraints, dict) else {}
    built_types = {
        str(item.type).strip()
        for item in (built_constraints or [])
        if isinstance(item, LogicalConstraintObject) and str(item.type).strip()
    }

    def has_string(*candidates: Any) -> bool:
        return any(isinstance(candidate, str) and candidate.strip() for candidate in candidates)

    def has_number(*candidates: Any) -> bool:
        for candidate in candidates:
            if isinstance(candidate, bool):
                continue
            if isinstance(candidate, (int, float)):
                return True
            if isinstance(candidate, str) and re.search(r"-?\d+(?:\.\d+)?", candidate):
                return True
        return False

    def must_include_names() -> List[str]:
        raw = constraints.get("must_include")
        if not isinstance(raw, list):
            return []
        names: List[str] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def forbidden_names() -> List[str]:
        raw = constraints.get("forbidden")
        if not isinstance(raw, list):
            return []
        names: List[str] = []
        for item in raw:
            name = str(item or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def require(logical_type: str, condition: bool) -> None:
        if condition and logical_type not in built_types:
            raise ValueError(f"structured coverage missing required logical type: {logical_type}")

    if ctype == "duration_limit":
        require(
            "activity_duration_limit",
            has_string(query_spec.get("anchor_poi"), semantic_slots.get("anchor_poi"), *(must_include_names()[:1]))
            and has_number(query_spec.get("duration_limit_min"), semantic_slots.get("duration_limit_min")),
        )
    elif ctype == "time_limit":
        require(
            "day_end_time_limit",
            has_number(query_spec.get("day"), semantic_slots.get("day"))
            and has_string(query_spec.get("end_before"), semantic_slots.get("end_before")),
        )
    elif ctype == "access_time_infeasible":
        require(
            "poi_inbound_travel_time_cap",
            has_string(query_spec.get("anchor_poi"), semantic_slots.get("anchor_poi"), *(must_include_names()[:1]))
            and has_number(query_spec.get("duration_limit_min"), semantic_slots.get("duration_limit_min")),
        )
    elif ctype == "time_window":
        require(
            "poi_time_window",
            has_string(query_spec.get("anchor_poi"), semantic_slots.get("anchor_poi"), *(must_include_names()[:1]))
            and has_string(query_spec.get("time_period"), semantic_slots.get("time_period")),
        )
    elif ctype == "required_intercity_transport_type":
        modes = query_spec.get("intercity_transport_modes") or semantic_slots.get("intercity_transport_modes")
        require(
            "required_intercity_transport_type",
            isinstance(modes, list) and any(normalize_intercity_transport_mode(item) for item in modes),
        )
    elif ctype == "transport_time_window":
        windows = query_spec.get("transport_time_windows") or semantic_slots.get("transport_time_windows")
        require(
            "transport_time_window",
            isinstance(windows, list)
            and any(
                isinstance(item, dict)
                and has_string(item.get("leg"))
                and has_string(item.get("field"))
                and has_string(item.get("value"))
                for item in windows
            ),
        )
    elif ctype == "order_constraint":
        ordered = query_spec.get("ordered_pois")
        require("poi_order_constraint", isinstance(ordered, list) and len([item for item in ordered if str(item or "").strip()]) >= 2)
    elif ctype == "same_day_no_overlap_compact":
        ordered = query_spec.get("ordered_pois")
        require(
            "pair_same_day_no_overlap",
            isinstance(ordered, list)
            and len([item for item in ordered if str(item or "").strip()]) >= 2
            and has_number(query_spec.get("day"), semantic_slots.get("day")),
        )
    elif ctype == "same_time_period_no_overlap":
        ordered = query_spec.get("ordered_pois")
        require(
            "pair_time_window_no_overlap",
            isinstance(ordered, list)
            and len([item for item in ordered if str(item or "").strip()]) >= 2
            and has_string(query_spec.get("time_period"), semantic_slots.get("time_period")),
        )
    elif ctype == "cross_day_restructure":
        day_pairs = _normalized_day_poi_pairs(query_spec.get("day_poi_pairs"), semantic_slots.get("day_poi_pairs"))
        require("poi_day_binding", bool(day_pairs))
        require("day_count", has_number(query_spec.get("day_count"), semantic_slots.get("day_count")))
    elif ctype == "cross_day_variety_by_types":
        day_pairs = _normalized_day_poi_pairs(query_spec.get("day_poi_pairs"), semantic_slots.get("day_poi_pairs"))
        require("poi_day_binding", bool(day_pairs))
        require(
            "semantic_type_requirement",
            bool(_semantic_type_by_day(query_spec=query_spec, semantic_slots=semantic_slots, constraints=constraints)),
        )
    elif ctype == "multi_city_split":
        require(
            "city_split_requirement",
            len(_resolve_city_split_targets(
                target_city=None,
                query_spec=query_spec,
                semantic_slots=semantic_slots,
                constraints=constraints,
            )) >= 2
            or len([
                item for item in (
                    query_spec.get("city_split_targets"),
                    semantic_slots.get("city_split_targets"),
                )
                if isinstance(item, list) and len(item) >= 2
            ]) > 0,
        )
    elif ctype in {"remove_poi", "substitute_poi", "famous_substitution", "type_replacement"}:
        has_source = has_string(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_names()[:1]),
        )
        require(
            "forbidden_attraction_name",
            has_source,
        )
        if ctype == "substitute_poi":
            require("semantic_type_requirement", has_source)
        elif ctype == "type_replacement":
            require("semantic_type_requirement", has_source)
        elif ctype == "famous_substitution":
            require("fame_replacement_requirement", has_source)
    elif ctype == "semantic_substitution":
        has_source = has_string(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_names()[:1]),
        )
        has_target = has_string(
            query_spec.get("semantic_target"),
            semantic_slots.get("semantic_target"),
            constraints.get("semantic", {}).get("semantic_target") if isinstance(constraints.get("semantic"), dict) else None,
            semantic_slots.get("preferred_type"),
        )
        require("forbidden_attraction_name", has_source)
        require("semantic_type_requirement", has_target)


def _is_transport_hub_name(name: Any) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    if text in _TRANSIT_HUB_ALLOWLIST:
        return False
    if any(token in text for token in _TRANSIT_HUB_KEYWORDS):
        return True
    if text.endswith("站") and len(text) <= 4:
        return True
    return text.endswith(_TRANSIT_HUB_SUFFIXES)


def _raise_if_transport_hub_name(name: Any, *, context: str) -> None:
    text = str(name or "").strip()
    if _is_transport_hub_name(text):
        raise ValueError(f"transport hub cannot be used as required POI in {context}: {text}")


def _strip_soft_annotations(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"（偏[^）]*）", "", value)
    value = re.sub(r"\(偏[^)]*\)", "", value)
    return value.strip()


def _normalized_day_poi_pairs(*sources: Any) -> List[tuple[int, str]]:
    pairs: List[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for raw in sources:
        if not isinstance(raw, list):
            continue
        for pair in raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            day_value = _parse_positive_int(pair[0])
            poi_name = _strip_soft_annotations(pair[1])
            if day_value is None or not poi_name:
                continue
            item = (day_value, poi_name)
            if item not in seen:
                seen.add(item)
                pairs.append(item)
    return pairs


def _soft_annotation_type_by_day(*sources: Any) -> Dict[int, Any]:
    result: Dict[int, Any] = {}
    for raw in sources:
        if not isinstance(raw, list):
            continue
        for pair in raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            day_value = _parse_positive_int(pair[0])
            if day_value is None or day_value in result:
                continue
            text = str(pair[1] or "").strip()
            if not text:
                continue
            match = re.search(r"[（(]偏([^）)]+)[）)]", text)
            if not match:
                continue
            result[day_value] = match.group(1).strip()
    return result


def _normalize_semantic_type_value(value: Any) -> Any:
    if isinstance(value, list):
        flattened: List[str] = []
        for item in value:
            normalized = _normalize_semantic_type_value(item)
            if isinstance(normalized, list):
                for token in normalized:
                    if token not in flattened:
                        flattened.append(token)
            elif isinstance(normalized, str) and normalized and normalized not in flattened:
                flattened.append(normalized)
        return flattened or None
    text = _strip_soft_annotations(value)
    if not text:
        return None
    if "/" not in text:
        return text
    parts = [part.strip() for part in text.split("/") if part.strip()]
    if not parts:
        return text
    merged: List[str] = [text]
    for part in parts:
        if part not in merged:
            merged.append(part)
    return merged


def _semantic_type_by_day(
    *,
    query_spec: Optional[Dict[str, Any]] = None,
    semantic_slots: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[int, Any]:
    spec = query_spec if isinstance(query_spec, dict) else {}
    slots = semantic_slots if isinstance(semantic_slots, dict) else {}
    constraint_payload = constraints if isinstance(constraints, dict) else {}
    day_pairs = _normalized_day_poi_pairs(spec.get("day_poi_pairs"), slots.get("day_poi_pairs"))
    poi_to_day = {poi_name: day_value for day_value, poi_name in day_pairs}
    result: Dict[int, Any] = {}

    def add(day_value: Any, type_value: Any) -> None:
        day = _parse_positive_int(day_value)
        normalized = _normalize_semantic_type_value(type_value)
        if day is None or normalized in (None, "", []) or day in result:
            return
        result[day] = normalized

    semantic = constraint_payload.get("semantic")
    preferred = semantic.get("preferred_type") if isinstance(semantic, dict) else None
    if isinstance(preferred, list):
        if preferred and all(not isinstance(item, dict) for item in preferred):
            for index, item in enumerate(preferred, start=1):
                if index <= len(day_pairs):
                    add(day_pairs[index - 1][0], item)
        for item in preferred:
            if not isinstance(item, dict):
                continue
            type_value = item.get("type") or item.get("preferred_type") or item.get("semantic_target")
            if item.get("day") is not None:
                add(item.get("day"), type_value)
                continue
            poi_name = _strip_soft_annotations(item.get("poi") or item.get("name") or item.get("must_visit"))
            if poi_name and poi_name in poi_to_day:
                add(poi_to_day[poi_name], type_value)
    elif isinstance(preferred, dict):
        for key, value in preferred.items():
            day = _parse_positive_int(key)
            if day is None and isinstance(key, str):
                match = re.fullmatch(r"day\s*(\d+)", key.strip(), flags=re.IGNORECASE)
                if match:
                    day = _parse_positive_int(match.group(1))
            if day is not None:
                add(day, value)
                continue
            poi_name = _strip_soft_annotations(key)
            if poi_name and poi_name in poi_to_day:
                add(poi_to_day[poi_name], value)

    for day_value, annotated_type in _soft_annotation_type_by_day(spec.get("day_poi_pairs"), slots.get("day_poi_pairs")).items():
        add(day_value, annotated_type)

    return result


def _contains_soft_only_cues(text: str) -> bool:
    raw = str(text or "")
    if "风格要明显不同" in raw:
        return True
    if "尽量别放其他不相关类型" in raw:
        return True
    if "但风格要相似" in raw:
        return True
    return "（偏" in raw or "(偏" in raw


@lru_cache(maxsize=1)
def _catalogs() -> Dict[str, Any]:
    from chinatravel.environment.tools.accommodations.apis import Accommodations
    from chinatravel.environment.tools.attractions.apis import Attractions
    from chinatravel.environment.tools.restaurants.apis import Restaurants

    return {
        "attraction": Attractions(),
        "restaurant": Restaurants(),
        "accommodation": Accommodations(),
    }


def _lookup_named_poi(city: str, poi_name: str) -> Optional[Dict[str, Any]]:
    catalogs = _catalogs()
    candidates = [
        ("attraction", "type"),
        ("restaurant", "cuisine"),
        ("accommodation", "featurehoteltype"),
    ]
    for category, semantic_key in candidates:
        frame = catalogs[category].select(city, "name", lambda x: x == poi_name)
        if len(frame) == 0:
            continue
        row = frame.iloc[0]
        return {
            "category": category,
            "semantic_type": row.get(semantic_key),
            "price": float(row.get("price", 0) or 0),
            "name": str(row.get("name", poi_name)),
        }
    return None


def _same_type_price_capped_candidates(city: str, category: str, semantic_type: Any, price_cap: float) -> List[str]:
    catalogs = _catalogs()
    if category == "attraction":
        semantic_key = "type"
    elif category == "restaurant":
        semantic_key = "cuisine"
    else:
        semantic_key = "featurehoteltype"
    frame = catalogs[category].select(city, semantic_key, lambda x: x == semantic_type)
    if len(frame) == 0:
        return []
    filtered = frame[frame["price"] <= price_cap]
    return [str(item) for item in filtered["name"].tolist() if str(item)]


def _same_type_candidates(city: str, category: str, semantic_type: Any) -> List[str]:
    catalogs = _catalogs()
    if category == "attraction":
        semantic_key = "type"
    elif category == "restaurant":
        semantic_key = "cuisine"
    else:
        semantic_key = "featurehoteltype"
    frame = catalogs[category].select(city, semantic_key, lambda x: x == semantic_type)
    if len(frame) == 0:
        return []
    return [str(item) for item in frame["name"].tolist() if str(item)]


def _extract_day_count(edit_query: str) -> Optional[int]:
    """从编辑指令中提取行程天数。

    匹配 'N天' 模式（如 '5天'），有多个匹配时取最后一个。
    """
    patterns = [
        re.compile(rf"(?:改成|调整为|改为|延长到|扩展到|增加到|改到|拉长到|延长为)\s*({_COUNT_TOKEN_PATTERN})\s*天"),
        re.compile(rf"(?:把)?(?:行程|路线|安排)(?:改成|调整为|改为|延长到|扩展到|增加到|改到)\s*(?:在[^，。；]{{0,20}})?(?:玩|住)?\s*({_COUNT_TOKEN_PATTERN})\s*(天|晚)"),
        re.compile(rf"(?:玩|安排|行程改成|行程调整为)\s*({_COUNT_TOKEN_PATTERN})\s*天"),
        re.compile(rf"住\s*({_COUNT_TOKEN_PATTERN})\s*晚"),
        re.compile(rf"(?<!第)({_COUNT_TOKEN_PATTERN})\s*天"),
    ]
    text = edit_query or ""
    last_value: Optional[int] = None
    for index, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            if index == len(patterns) - 1:
                trailing = text[match.end(): match.end() + 4]
                if any(token in trailing for token in ("风格", "感觉", "氛围")):
                    continue
            parsed = _normalize_day_token(match.group(1))
            if parsed is not None:
                unit = match.group(2) if len(match.groups()) >= 2 and match.group(2) in {"天", "晚"} else ("晚" if "晚" in match.group(0) else "天")
                last_value = parsed + 1 if unit == "晚" else parsed
    return last_value


def _extract_ticket_budget_limit(edit_query: str) -> Optional[float]:
    patterns = [
        r"所有门票总价(?:不能超过|不超过|控制在)?\s*(\d+(?:\.\d+)?)\s*元",
        r"门票(?:这部分)?(?:预算)?(?:别太高|别买得太贵)?(?:，|。|；)?(?:全部门票加起来|总价)?(?:最好|尽量)?(?:控制在|不超过|别超过)\s*(\d+(?:\.\d+)?)\s*元",
        r"门票(?:这部分)?[^，。；]{0,20}?(?:全部门票加起来|总价)(?:最好|尽量)?(?:控制在|不超过|别超过)\s*(\d+(?:\.\d+)?)\s*元",
        r"门票(?:这部分)?[^。；]{0,40}?总价(?:最好|尽量)?(?:控制在|不超过|别超过)\s*(\d+(?:\.\d+)?)\s*元",
        r"全部门票(?:加起来)?(?:尽量)?(?:控制在|不超过|别超过)\s*(\d+(?:\.\d+)?)\s*元",
        r"所有门票加起来(?:最好|尽量)?(?:控制在|不超过|别超过)\s*(\d+(?:\.\d+)?)\s*元",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if match:
            return float(match.group(1))
    return None


def _extract_budget_total(edit_query: str) -> Optional[float]:
    """从编辑指令中提取总预算上限。

    匹配如 '总预算不超过5000元'、'预算控制在3000元' 等中文表述。
    """
    patterns = [
        r"总预算(?:先|还是|仍然|继续)?(?:最好|尽量|希望)?(?:不能超过|不超过|控制在|设置为|改为|别超过)?\s*(\d+(?:\.\d+)?)\s*元(?:以内|内)?",
        r"预算(?:上限|目标)?(?:先|还是|仍然|继续)?(?:最好|尽量|希望)?(?:不能超过|不超过|控制在|设置为|改为|别超过)?\s*(\d+(?:\.\d+)?)\s*元(?:以内|内)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if match:
            return float(match.group(1))
    return None


def _extract_people_count_update(edit_query: str) -> Optional[int]:
    patterns = [
        re.compile(rf"同行(?:人数)?改为\s*({_COUNT_TOKEN_PATTERN})\s*人"),
        re.compile(rf"(?:人数|出行人数)(?:改成|调整为|改为)?\s*({_COUNT_TOKEN_PATTERN})\s*人"),
        re.compile(rf"(?:改成|调整为|改为)\s*({_COUNT_TOKEN_PATTERN})\s*人(?:同行|出行)?"),
        re.compile(rf"一共\s*({_COUNT_TOKEN_PATTERN})\s*个?人"),
        re.compile(rf"我们(?:一共)?\s*({_COUNT_TOKEN_PATTERN})\s*个?人"),
        re.compile(rf"同行改为\s*({_COUNT_TOKEN_PATTERN})\s*人"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if match:
            parsed = _normalize_day_token(match.group(1))
            if parsed is not None:
                return parsed
    return None


def _extract_duration_limit(edit_query: str) -> Optional[tuple[str, int]]:
    """从编辑指令中提取景点游玩时间限制。

    匹配如 '把故宫的游玩时间限制在2小时以内' 的表述。

    Returns:
        (景点名称, 分钟数) 元组，未匹配时返回 None。
    """
    patterns = [
        re.compile(
            r"(?:把|将)([^，。；]{2,40}?)(?:的)?(?:游玩时间|参观时间|停留时间)(?:限制在|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*(小时|分钟)(?:左右|以内|之内|内)?"
        ),
        re.compile(
            r"(?:把|将)([^，。；]{2,40}?)(?:的)?(?:游玩时间|参观时间|停留时间)(?:限制在|控制在|最好控制在|尽量控制在|别超过|不要超过|不超过)\s*半小时(?:左右|以内|之内|内)?"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        poi_name = match.group(1).strip()
        if not poi_name:
            continue
        if match.lastindex == 3:
            unit = match.group(3)
            minutes = _parse_duration_to_minutes(match.group(2), unit)
        else:
            minutes = 30
        if minutes is None or minutes <= 0:
            continue
        return poi_name, minutes
    return None


def _extract_day_end_time_limit(edit_query: str) -> Optional[dict[str, Any]]:
    patterns = [
        re.compile(r"(第[一二三四五六七八九十\d]+天).*?在\s*(\d{1,2}:\d{2})前结束(?:所有活动|行程)?"),
        re.compile(r"(第[一二三四五六七八九十\d]+天).*?(?:最晚|不要超过)\s*(\d{1,2}:\d{2})结束"),
        re.compile(r"在\s*(\d{1,2}:\d{2})前结束(?:所有活动|行程)"),
        re.compile(r"(?:最晚|不要超过)\s*(\d{1,2}:\d{2})结束(?:所有活动|行程)?"),
    ]
    text = edit_query or ""
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:
            day = _extract_day_reference(groups[0])
            return {"day": day, "value": groups[1]}
        return {"day": None, "value": groups[0]}
    return None


def _extract_poi_time_window(edit_query: str) -> Optional[dict[str, Any]]:
    patterns = [
        re.compile(
            r"(?:把|将)(第[一二三四五六七八九十\d]+天)?(?:的)?([^，。；]{2,40}?)(?:的)?(?:时间|游玩时间|安排)?安排在(上午|早上|下午|傍晚|晚上|夜间|夜里)"
        ),
        re.compile(
            r"(第[一二三四五六七八九十\d]+天)?([^，。；]{2,40}?)(?:可以|最好|尽量)?(上午|早上|下午|傍晚|晚上|夜间|夜里)去(?:吗)?"
        ),
        re.compile(
            r"(?:能不能把|把)([^，。；]{2,40}?)(?:改到|放到)(上午|早上|下午|傍晚|晚上|夜间|夜里)去"
        ),
    ]
    text = edit_query or ""
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 3:
            day_ref, poi_name, period_text = groups
        else:
            day_ref = None
            poi_name, period_text = groups
        poi_name = str(poi_name or "").strip()
        period = _TIME_PERIOD_MAP.get(period_text)
        if not poi_name or not period:
            continue
        return {
            "poi_name": poi_name,
            "day": _extract_day_reference(day_ref or ""),
            "value": period,
        }
    return None


def _extract_poi_clock_time_windows(edit_query: str) -> List[dict[str, Any]]:
    text = edit_query or ""
    patterns = [
        (
            re.compile(
                rf"(?:把|将)(第{_COUNT_TOKEN_PATTERN}天)?(?:的)?([^，。；]{{2,40}}?)(?:的)?(?:时间|游玩时间|参观时间|停留时间|安排)?(?:安排在|安排|改到|放到)?\s*(\d{{1,2}}:\d{{2}})\s*后(?:去|再去|游玩|参观)?"
            ),
            "after",
        ),
        (
            re.compile(
                rf"(?:把|将)(第{_COUNT_TOKEN_PATTERN}天)?(?:的)?([^，。；]{{2,40}}?)(?:的)?(?:时间|游玩时间|参观时间|停留时间|安排)?(?:安排在|安排|改到|放到)?\s*(\d{{1,2}}:\d{{2}})\s*前(?:去|结束|游玩|参观)?"
            ),
            "before",
        ),
        (
            re.compile(
                rf"(?:把|将)(第{_COUNT_TOKEN_PATTERN}天)?(?:的)?([^，。；]{{2,40}}?)(?:的)?(?:时间|游玩时间|参观时间|停留时间|安排)?(?:安排在|放到|改到)?\s*(\d{{1,2}}:\d{{2}})\s*(?:-|到|至|~)\s*(\d{{1,2}}:\d{{2}})(?:之间|之内|内)?"
            ),
            "between",
        ),
        (
            re.compile(
                rf"(第{_COUNT_TOKEN_PATTERN}天)?([^，。；]{{2,40}}?)(?:最好|尽量)?\s*(\d{{1,2}}:\d{{2}})\s*后去(?:吗)?"
            ),
            "after",
        ),
        (
            re.compile(
                rf"(第{_COUNT_TOKEN_PATTERN}天)?([^，。；]{{2,40}}?)(?:最好|尽量)?\s*(\d{{1,2}}:\d{{2}})\s*前去(?:吗)?"
            ),
            "before",
        ),
        (
            re.compile(
                rf"(第{_COUNT_TOKEN_PATTERN}天)?([^，。；]{{2,40}}?)(?:最好|尽量)?安排在\s*(\d{{1,2}}:\d{{2}})\s*(?:-|到|至|~)\s*(\d{{1,2}}:\d{{2}})"
            ),
            "between",
        ),
    ]
    extracted: List[dict[str, Any]] = []
    seen: set[tuple[str, Optional[int], str, str]] = set()

    def _is_valid_clock_poi_name(name: str) -> bool:
        if not name:
            return False
        if name.startswith("把"):
            return False
        if re.match(rf"^第{_COUNT_TOKEN_PATTERN}天", name):
            return False
        return not any(token in name for token in ("安排在", "改到", "放到"))

    for pattern, mode in patterns:
        for match in pattern.finditer(text):
            groups = match.groups()
            if mode == "between":
                day_ref, poi_name, start_clock, end_clock = groups
                normalized_poi = re.sub(r"^(?:把|将)\s*", "", _strip_soft_annotations(str(poi_name)).strip())
                if not _is_valid_clock_poi_name(normalized_poi):
                    continue
                day = _extract_day_reference(day_ref or "")
                key = (normalized_poi, day, mode, f"{start_clock}-{end_clock}")
                if key in seen:
                    continue
                seen.add(key)
                extracted.append(
                    {
                        "poi_name": normalized_poi,
                        "day": day,
                        "mode": mode,
                        "value": [start_clock, end_clock],
                    }
                )
                continue
            day_ref, poi_name, clock = groups
            normalized_poi = re.sub(r"^(?:把|将)\s*", "", _strip_soft_annotations(str(poi_name)).strip())
            if not _is_valid_clock_poi_name(normalized_poi):
                continue
            day = _extract_day_reference(day_ref or "")
            key = (normalized_poi, day, mode, clock)
            if key in seen:
                continue
            seen.add(key)
            extracted.append(
                {
                    "poi_name": normalized_poi,
                    "day": day,
                    "mode": mode,
                    "value": clock,
                }
            )
    return extracted


def _extract_daily_poi_cap(edit_query: str) -> Optional[int]:
    return extract_daily_poi_cap_from_query(edit_query or "")


def _extract_adjacent_travel_time_cap(edit_query: str) -> Optional[dict[str, Any]]:
    pattern = re.compile(
        r"(?:所有景点之间|景点之间|活动之间)(?:的)?交通时间(?:都)?(?:尽量|最好|尽量别|最好别|尽量控制|最好控制|尽量压到|最好压到)?"
        r"(?:不能超过|不超过|控制在|压到|压在|别超过|最好别超过|尽量不超过)?\s*(\d+(?:\.\d+)?)\s*(小时|分钟)(?:以内|内)?"
    )
    match = pattern.search(edit_query or "")
    if not match:
        return None
    minutes = _parse_duration_to_minutes(match.group(1), match.group(2))
    if minutes is None:
        return None
    activity_types = ["attraction"] if "景点" in (edit_query or "") else ["attraction", "restaurant"]
    return {"value": minutes, "activity_types": activity_types}


def _extract_activity_budget_limits(edit_query: str) -> List[dict[str, Any]]:
    text = edit_query or ""
    patterns = [
        (
            "accommodation",
            "avg_cost_per_person_per_night",
            [
                re.compile(r"(?:酒店|住宿|民宿)(?:人均|每人)?每晚(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
                re.compile(r"(?:人均|每人)(?:酒店|住宿|民宿)?每晚(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
                re.compile(r"每人每晚(?:住宿|酒店|民宿)(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
            ],
        ),
        (
            "meal",
            "avg_cost_per_meal_per_person",
            [
                re.compile(r"(?:餐食|餐饮|吃饭|餐厅|午餐|晚餐)(?:人均|每人)?每餐(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
                re.compile(r"(?:人均|每人)(?:餐食|餐饮|吃饭)?每餐(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
                re.compile(r"每人每(?:顿|餐)(?:餐食|餐饮|饭|用餐)(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
                re.compile(r"每顿饭(?:预算|花费|费用)?(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
            ],
        ),
    ]
    extracted: List[dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for activity_type, metric, pattern_group in patterns:
        for pattern in pattern_group:
            for match in pattern.finditer(text):
                value = float(match.group(1))
                key = (activity_type, metric, value)
                if key in seen:
                    continue
                seen.add(key)
                extracted.append(
                    {
                        "value": value,
                        "activity_type": activity_type,
                        "metric": metric,
                    }
                )
    return extracted


def _extract_activity_budget_limit(edit_query: str) -> Optional[dict[str, Any]]:
    extracted = _extract_activity_budget_limits(edit_query)
    return extracted[0] if extracted else None


def _extract_anchor_bundle_budget_limit(edit_query: str) -> Optional[dict[str, Any]]:
    text = edit_query or ""
    patterns = [
        re.compile(r"([^，。；]{2,40}?)\s*(?:\+|加上|和)(?:附近)?一餐(?:控制在|不超过|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
        re.compile(r"([^，。；]{2,40}?)附近一餐(?:控制在|不超过|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
        re.compile(r"([^，。；]{2,40}?)加上(?:附近)?一顿饭(?:控制在|不超过|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
        re.compile(r"([^，。；]{2,40}?)附近吃一顿饭和景点加起来(?:控制在|不超过|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        anchor_poi = _strip_soft_annotations(match.group(1))
        if not anchor_poi:
            continue
        return {
            "anchor_poi": anchor_poi,
            "value": float(match.group(2)),
            "requires_nearby_meal": True,
        }
    grouped_anchor = re.search(r"从([^，。；]{2,40}?)到(?:这家)?餐厅", text)
    grouped_budget = re.search(
        r"(?:这一组|这组|这一套|这套)活动的(?:总花费|总费用|总预算|预算|花费|费用)(?:不能超过|不超过|控制在|最好控制在|尽量控制在)\s*(\d+(?:\.\d+)?)\s*元",
        text,
    )
    if grouped_anchor and grouped_budget:
        anchor_poi = _strip_soft_annotations(grouped_anchor.group(1))
        if not anchor_poi:
            return None
        return {
            "anchor_poi": anchor_poi,
            "value": float(grouped_budget.group(1)),
            "requires_nearby_meal": True,
        }
    return None


def _extract_adjacent_travel_distance_cap(edit_query: str) -> Optional[dict[str, Any]]:
    text = edit_query or ""
    scoped_patterns = [
        re.compile(
            rf"我想在第({_COUNT_TOKEN_PATTERN})天游玩([^，。；]{{2,40}}?)，但不想走太远，景点间的距离控制在\s*(\d+(?:\.\d+)?)\s*公里内"
        ),
        re.compile(
            rf"第({_COUNT_TOKEN_PATTERN})天围绕([^，。；]{{2,40}}?)安排行程[^。；]*?景点(?:之间|间)的距离(?:都)?(?:不能超过|不超过|控制在)?\s*(\d+(?:\.\d+)?)\s*公里"
        ),
        re.compile(
            rf"围绕([^，。；]{{2,40}}?)安排行程[^。；]*?第({_COUNT_TOKEN_PATTERN})天[^。；]*?景点(?:之间|间)的距离(?:都)?(?:不能超过|不超过|控制在)?\s*(\d+(?:\.\d+)?)\s*公里"
        ),
    ]
    for pattern in scoped_patterns:
        scoped_match = pattern.search(text)
        if not scoped_match:
            continue
        if pattern is scoped_patterns[2]:
            poi_name = scoped_match.group(1).strip()
            day = _normalize_day_token(scoped_match.group(2))
            value = scoped_match.group(3)
        else:
            day = _normalize_day_token(scoped_match.group(1))
            poi_name = scoped_match.group(2).strip()
            value = scoped_match.group(3)
        if day is None or not poi_name:
            return None
        return {"value": float(value), "activity_types": ["attraction"], "day": day, "anchor_poi": poi_name}
    global_match = re.search(r"所有景点之间的距离都(?:不能超过|不超过|控制在)?\s*(\d+(?:\.\d+)?)\s*公里", text)
    if global_match:
        return {"value": float(global_match.group(1)), "activity_types": ["attraction"], "day": "all"}
    return None


def _extract_anchor_neighbor_commute_distance_cap(edit_query: str) -> Optional[dict[str, Any]]:
    patterns = [
        re.compile(r"把([^，。；]{2,40}?)周边活动半径限制为\s*(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"把([^，。；]{2,40}?)附近的活动安排得更集中[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"围绕([^，。；]{2,40}?)安排活动[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"围绕([^，。；]{2,40}?)安排周边活动[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"围绕([^，。；]{2,40}?)的周边活动安排得更集中[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"希望以([^，。；]{2,40}?)为中心安排活动[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"希望以([^，。；]{2,40}?)为核心[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"以([^，。；]{2,40}?)为中心安排活动[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"以([^，。；]{2,40}?)为中心安排行程[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"围绕([^，。；]{2,40}?)安排行程[^。；]*?(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"从([^，。；]{2,40}?)到这家餐厅的通勤距离(?:不要|不能)?超过\s*(\d+(?:\.\d+)?)\s*公里"),
        re.compile(r"从([^，。；]{2,40}?)到餐厅的通勤距离控制在\s*(\d+(?:\.\d+)?)\s*公里内"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        anchor_name = match.group(1).strip()
        if not anchor_name:
            continue
        try:
            limit_km = float(match.group(2))
        except ValueError:
            continue
        if limit_km <= 0:
            continue
        return {"anchor_name": anchor_name, "value": limit_km}
    return None


def _extract_theme_semantic_requirement(edit_query: str) -> Optional[str]:
    if _contains_soft_only_cues(edit_query or ""):
        return None
    preferred_type = parse_query_semantic_slots(
        edit_query or "",
        "type_filter",
    ).get("preferred_type") or extract_preferred_poi_type_from_query(edit_query or "")
    if not preferred_type:
        return None
    lowered = preferred_type.strip()
    if not lowered:
        return None
    if lowered in {"省", "省点", "省一点", "再省", "再省一点"} or "省" in lowered:
        return None
    if any(token in lowered.lower() for token in _VIBE_SOFT_KEYWORDS):
        return None
    if any(token in lowered for token in ("餐厅", "酒店", "住宿", "民宿", "宾馆")):
        return None
    return lowered


def _extract_pairwise_walking_distance_limit(edit_query: str) -> Optional[dict[str, Any]]:
    pattern = re.compile(r"把([^，。；]{2,40}?)和([^，。；]{2,40}?)之间的交通方式改为步行，不要超过\s*(\d+(?:\.\d+)?)\s*公里")
    match = pattern.search(edit_query or "")
    if not match:
        return None
    first_poi = match.group(1).strip()
    second_poi = match.group(2).strip()
    if not first_poi or not second_poi:
        return None
    return {"first_poi": first_poi, "second_poi": second_poi, "mode": "walk", "value": float(match.group(3))}


def _extract_access_time_inbound_cap(edit_query: str) -> Optional[dict[str, Any]]:
    patterns = [
        re.compile(r"把([^，。；]{2,40}?)安排到行程中，但从出发点到[^，。；]{0,20}?的交通时间不能超过\s*(\d+(?:\.\d+)?)\s*(小时|分钟)"),
        re.compile(r"行程里加上([^，。；]{2,40}?)(?:吧|呀|啊)?，不过从出发点到(?:那儿|那里|该地|[^，。；]{0,20}?)路上别超过\s*(\d+(?:\.\d+)?)\s*(小时|分钟)"),
        re.compile(r"行程里加上([^，。；]{2,40}?)(?:吧|呀|啊)?，不过从出发点到(?:那儿|那里|该地|[^，。；]{0,20}?)路上别超过半小时"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        poi_name = match.group(1).strip()
        minutes = 30 if match.lastindex == 1 else _parse_duration_to_minutes(match.group(2), match.group(3))
        if not poi_name or minutes is None:
            continue
        return {"poi_name": poi_name, "value": minutes}
    return None


def _extract_remove_poi_name(edit_query: str, *, blocked_names: Optional[set[str]] = None) -> Optional[str]:
    patterns = [
        re.compile(r"把([^，。；]{2,40}?)从行程中删除"),
        re.compile(r"去掉([^，。；]{2,40}?)(?=[，。；]|$)"),
        re.compile(r"把([^，。；]{2,40}?)从行程里去掉"),
        re.compile(r"([^，。；]{2,40}?)就别去了吧"),
        re.compile(r"([^，。；]{2,40}?)不去了吧"),
        re.compile(r"能不能不去([^，。；？?]{2,40}?)(?:了)?[？?]?$"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        candidate = _strip_soft_annotations(match.group(1))
        candidate = re.sub(r"(?:了)+$", "", candidate).strip()
        if not candidate:
            continue
        if blocked_names and candidate in blocked_names:
            continue
        return candidate
    return None


def _extract_specific_poi_name(edit_query: str, *, blocked_names: Optional[set[str]] = None) -> Optional[str]:
    for ctype in ("semantic_and_specific", "category_overlap"):
        parsed_specific = parse_query_semantic_slots(
            edit_query or "",
            ctype,
        ).get("specific_poi")
        if isinstance(parsed_specific, str) and parsed_specific.strip():
            candidate = _strip_soft_annotations(parsed_specific)
            if candidate and (blocked_names is None or candidate not in blocked_names):
                return candidate
    patterns = [
        re.compile(r"特别是([^，。；]{2,40}?)(?=[，。；]|$)"),
        re.compile(r"尤其是([^，。；]{2,40}?)(?=[，。；]|$)"),
        re.compile(r"我想去([^，。；]{2,40}?)这样(?:的)?[^，。；]{0,20}(?=[，。；]|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        candidate = _strip_soft_annotations(match.group(1))
        candidate = re.sub(r"[了呢吗吧呀啊？?]+$", "", candidate).strip()
        if not candidate:
            continue
        if blocked_names and candidate in blocked_names:
            continue
        return candidate
    return None


def _extract_poi_day_bindings(edit_query: str) -> List[dict[str, Any]]:
    pattern = re.compile(
        r"第([一二三四五六七八九十\d]+)天(?:去|安排|游玩|参观|打卡)([^，。；]{2,40}?)(?=，|。|；|$)"
    )
    bindings: List[dict[str, Any]] = []
    for match in pattern.finditer(edit_query or ""):
        day = _normalize_day_token(match.group(1))
        poi_name = _strip_soft_annotations(match.group(2))
        if day is None or not poi_name:
            continue
        bindings.append({"day": day, "poi_name": poi_name})
    return bindings


def _extract_poi_order_constraint(edit_query: str) -> Optional[dict[str, Any]]:
    pattern = re.compile(r"先去([^，。；]{2,40}?)再去([^，。；]{2,40}?)(?=[，。；]|$)")
    match = pattern.search(edit_query or "")
    if not match:
        return None
    first_poi = match.group(1).strip()
    second_poi = match.group(2).strip()
    if not first_poi or not second_poi:
        return None
    return {"first_poi": first_poi, "second_poi": second_poi}


def _extract_same_day_no_overlap_compact(edit_query: str) -> Optional[dict[str, Any]]:
    pattern = re.compile(
        r"我想在第([一二三四五六七八九十\d]+)天把([^，。；]{2,40}?)和([^，。；]{2,40}?)安排得更紧凑：同一天都去，但不要出现时间重叠"
    )
    match = pattern.search(edit_query or "")
    if not match:
        return None
    day = _normalize_day_token(match.group(1))
    first_poi = _strip_soft_annotations(match.group(2))
    second_poi = _strip_soft_annotations(match.group(3))
    if day is None or not first_poi or not second_poi:
        return None
    return {"day": day, "first_poi": first_poi, "second_poi": second_poi}


def _extract_same_time_period_no_overlap(edit_query: str) -> Optional[dict[str, Any]]:
    pattern = re.compile(
        r"把([^，。；]{2,40}?)和([^，。；]{2,40}?)安排在同一(上午|下午|晚上)，但要能顺序游玩，不要时间重叠"
    )
    match = pattern.search(edit_query or "")
    if not match:
        return None
    first_poi = _strip_soft_annotations(match.group(1))
    second_poi = _strip_soft_annotations(match.group(2))
    time_period = _TIME_PERIOD_MAP.get(match.group(3).strip())
    if not first_poi or not second_poi or time_period is None:
        return None
    return {"first_poi": first_poi, "second_poi": second_poi, "time_period": time_period}


def _extract_ticket_price_cap_or_substitute(edit_query: str) -> Optional[dict[str, Any]]:
    pattern = re.compile(
        r"我想在(?:第([一二三四五六七八九十\d]+)天)?去([^，。；]{2,40}?)，?但如果门票超过\s*(\d+(?:\.\d+)?)\s*元，就把它换成[^，。；]{0,50}?同类型景点"
    )
    match = pattern.search(edit_query or "")
    if not match:
        return None
    day_token, poi_name, budget_text = match.groups()
    poi_name = _strip_soft_annotations(poi_name)
    if not poi_name:
        return None
    try:
        budget = float(budget_text)
    except ValueError:
        return None
    return {
        "day": _normalize_day_token(day_token) if day_token else None,
        "poi_name": poi_name,
        "price_cap": budget,
    }


def _extract_substitute_poi(edit_query: str) -> Optional[str]:
    pattern = re.compile(r"把([^，。；]{2,40}?)换成[^，。；]{0,40}?类似景点")
    match = pattern.search(edit_query or "")
    if not match:
        return None
    poi_name = _strip_soft_annotations(match.group(1))
    return poi_name or None


def _extract_type_replacement(edit_query: str) -> Optional[dict[str, Any]]:
    patterns = [
        re.compile(r"去掉([^，。；]{2,40}?)，换成[^，。；]{0,40}?同类型的([^，。；]{2,40}?)(?=[，。；]|$)"),
        re.compile(r"([^，。；]{2,40}?)不去了，换(?:一个|成)?[^，。；]{0,20}?(历史文化景点|自然景观|历史景点|博物馆|自然风光|景点)(?:吧|吗|呢|啊)?(?=[，。；]|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        source_poi = _strip_soft_annotations(match.group(1))
        poi_type = match.group(2).strip()
        if source_poi and poi_type:
            return {"source_poi": source_poi, "semantic_type": poi_type}
    return None


def _extract_semantic_substitution(edit_query: str) -> Optional[dict[str, Any]]:
    patterns = [
        re.compile(r"把([^，。；]{2,40}?)改成([^，。；]{2,40}?)类型的其他景点"),
        re.compile(r"(?:把)?([^，。；]{2,40}?)换成其他类似风格的([^，。；]{2,40}?)(?:景点|地方)(?=[，。；？?]|$)"),
        re.compile(r"(?:把)?([^，。；]{2,40}?)能不能换成其他类似的([^，。；]{2,40}?)(?:景点|地方)(?=[，。；？?]|$)"),
        re.compile(r"(?:能不能把|把)?([^，。；]{2,40}?)换成[^，。；]{0,16}?(?:类似|同类型|差不多风格的)([^，。；]{2,40}?)(?:景点|地方|类型)(?=[，。；？?]|$)"),
        re.compile(r"换个跟([^，。；]{2,40}?)风格差不多的([^，。；]{2,40}?)(?:景点|地方)(?=[，。；？?]|$)"),
        re.compile(r"换个和([^，。；]{2,40}?)类似的([^，。；]{2,40}?)(?:景点|地方)(?=[，。；？?]|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        source_poi = _strip_soft_annotations(match.group(1))
        semantic_target = _strip_soft_annotations(match.group(2))
        if source_poi and semantic_target:
            return {"source_poi": source_poi, "semantic_target": semantic_target}
    return None


def _extract_multi_day_requirement(edit_query: str) -> Optional[dict[str, Any]]:
    exact_match = re.search(r"玩(\d+)\s*天", edit_query or "")
    if exact_match:
        return {"operator": "==", "value": int(exact_match.group(1))}
    if "多玩几天" in (edit_query or "") or "不要当天往返" in (edit_query or ""):
        return {"operator": ">=", "value": 2}
    return None


def _extract_transport_time_window(edit_query: str) -> List[dict[str, Any]]:
    """从编辑指令中提取交通出发/到达时间窗口约束。

    匹配去程和返程的时间约束，如 '去程最早8:00后出发'、'返程14:00前到达'。

    Returns:
        包含 leg（'outbound'/'return'）、field（'start_time'/'end_time'）、
        operator（'>='/'<='）和 value（时间字符串）的字典，未匹配时返回 None。
    """
    return extract_transport_time_windows_from_query(edit_query or "")


def _extract_named_requirement(
    edit_query: str,
    *,
    verbs: Iterable[str],
    suffixes: Iterable[str] = (),
    blocked_names: Optional[set[str]] = None,
) -> Optional[str]:
    """通用名称提取器：根据动词前缀和名词后缀从编辑指令中提取指定实体名称。

    Args:
        edit_query: 编辑指令文本。
        verbs: 触发词列表，如 ['住在', '入住']。
        suffixes: 名称后缀列表，如 ['酒店', '宾馆']，为空时不限制后缀。

    Returns:
        提取到的实体名称字符串，未匹配或命中停用词时返回 None。
    """
    verb_group = "|".join(re.escape(item) for item in verbs)
    suffix_group = "|".join(re.escape(item) for item in suffixes)
    if suffix_group:
        pattern = re.compile(
            rf"(?:{verb_group})([^，。；]{2,40}?(?:{suffix_group}))(?=[，。；]|$)"
        )
    else:
        pattern = re.compile(rf"(?:{verb_group})([^，。；]{2,40}?)(?=[，。；]|$)")
    match = pattern.search(edit_query or "")
    if not match:
        return None
    candidate = _strip_soft_annotations(match.group(1))
    if not candidate:
        return None
    for stop_token in ("总预算", "预算", "所有景点", "行程", "附近的另一个城市", "附近城市"):
        if candidate == stop_token:
            return None
    if blocked_names and candidate in blocked_names:
        return None
    return candidate


def _extract_required_hotel_name(edit_query: str, *, blocked_names: Optional[set[str]] = None) -> Optional[str]:
    """从编辑指令中提取指定入住的酒店名称（如 '入住XX酒店'）。"""
    for pattern in [
        re.compile(r"(?:住在|入住|住|订)([^，。；]{2,40}?)(?=[，。；]|$)"),
    ]:
        match = pattern.search(edit_query or "")
        if match:
            candidate = _strip_soft_annotations(match.group(1))
            if re.fullmatch(rf"{_COUNT_TOKEN_PATTERN}\s*晚", candidate):
                continue
            if candidate and (blocked_names is None or candidate not in blocked_names):
                return candidate
    return _extract_named_requirement(
        edit_query,
        verbs=["住在", "入住", "住", "酒店改成", "换成", "改成"],
        suffixes=["酒店", "宾馆", "民宿"],
        blocked_names=blocked_names,
    )


def _extract_required_restaurant_name(edit_query: str, *, blocked_names: Optional[set[str]] = None) -> Optional[str]:
    """从编辑指令中提取指定用餐的餐厅名称（如 '安排XX餐厅'）。"""
    for pattern in [
        re.compile(r"(?:安排|增加|加入|去|在)([^，。；]{2,40}?)(?=用餐|吃饭|就餐)"),
        re.compile(r"(?:安排|增加|加入)([^，。；]{2,40}?(?:餐厅|饭店|餐馆))用餐"),
        re.compile(r"(?:安排|增加|加入)([^，。；]{2,40}?(?:餐厅|饭店|餐馆))这顿饭"),
    ]:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        candidate = _strip_soft_annotations(match.group(1))
        if not candidate:
            continue
        if blocked_names and candidate in blocked_names:
            continue
        return candidate
    return _extract_named_requirement(
        edit_query,
        verbs=["安排", "增加", "加入", "去", "在"],
        suffixes=["餐厅", "饭店", "餐馆"],
        blocked_names=blocked_names,
    )


def _extract_required_restaurant_type(edit_query: str) -> Optional[str]:
    patterns = [
        r"我想吃([^，。；]{2,24})",
        r"想吃([^，。；]{2,24})",
        r"吃([^，。；]{2,24})菜",
        r"我想去([^，。；]{2,24})餐厅",
        r"([^，。；]{2,24})餐厅",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if not match:
            continue
        value = str(match.group(1)).strip()
        if value:
            return value
    return None


def _extract_required_hotel_feature(edit_query: str) -> Optional[str]:
    patterns = [
        r"住([^，。；]{2,24}?)酒店",
        r"入住([^，。；]{2,24}?)酒店",
        r"选([^，。；]{2,24}?)酒店",
        r"([^，。；]{2,24}?)酒店",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if not match:
            continue
        value = str(match.group(1)).strip()
        if value:
            return value
    return None


def _extract_required_room_type(edit_query: str) -> Optional[int]:
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
    patterns = [
        r"房型(?:改成|调整为|设置为|要)?\s*(\d+)",
        r"(\d+)\s*号房型",
        r"(\d+)\s*型房",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _extract_required_room_count(edit_query: str) -> Optional[int]:
    patterns = [
        r"(\d+)\s*间房",
        r"(\d+)\s*间客房",
        r"要\s*(\d+)\s*间房",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query or "")
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _extract_activity_budget_requirement(edit_query: str) -> Optional[Dict[str, Any]]:
    extracted = _extract_activity_budget_requirements(edit_query)
    return extracted[0] if extracted else None


def _extract_activity_budget_requirements(edit_query: str) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, float]] = set()
    for item in _extract_activity_budget_limits(edit_query):
        key = (str(item["activity_type"]), str(item["metric"]), float(item["value"]))
        if key in seen:
            continue
        seen.add(key)
        extracted.append(
            {
                "activity_type": str(item["activity_type"]),
                "metric": str(item["metric"]),
                "budget_cap": float(item["value"]),
            }
        )
    return extracted


def _extract_required_attraction_name(
    edit_query: str,
    *,
    blocked_names: Optional[set[str]] = None,
) -> Optional[str]:
    """从编辑指令中提取指定游玩的景点名称。

    匹配 '增加XX到行程中'、'去XX'、'参观XX' 等表述，
    排除包含餐厅/酒店后缀的误匹配。
    """
    patterns = [
        re.compile(r"(?:增加|加入|添加)([^，。；]{2,40}?)到行程中"),
        re.compile(r"把([^，。；]{2,40}?)加(?:进来|进去|进行程)"),
        re.compile(r"(?:去|游玩|参观|打卡)([^，。；]{2,40}?)(?=[，。；]|$)"),
        re.compile(r"(?:安排)([^，。；]{2,40}?)(?:和附近|到行程中|在第|\s|，|。|；|$)"),
    ]
    for pattern in patterns:
        match = pattern.search(edit_query or "")
        if not match:
            continue
        candidate = _strip_soft_annotations(match.group(1))
        candidate = re.sub(r"[了呢吗吧呀啊？?]+$", "", candidate).strip()
        if not candidate:
            continue
        if any(token in candidate for token in ("餐厅", "饭店", "餐馆", "酒店", "宾馆", "民宿")):
            continue
        if any(token in candidate for token in ("景点", "行程", "活动", "城市")):
            continue
        if any(token in candidate for token in ("附近的活动", "周边活动", "得更集中", "围绕")):
            continue
        if any(token in candidate for token in ("再去", "先去", "后去")):
            continue
        if any(token in candidate for token in ("这样的", "类似的", "同类型")):
            continue
        if any(token in candidate for token in ("上午", "早上", "下午", "傍晚", "晚上", "夜间", "夜里")):
            continue
        if any(token in candidate for token in ("去程", "返程", "回程", "出发", "到达")):
            continue
        if any(token in candidate for token in ("紧凑", "同一天都去", "不要时间重叠", "顺序游玩")):
            continue
        if any(token in candidate for token in ("时间限制", "游玩时间", "参观时间", "停留时间", "小时以内", "分钟以内")):
            continue
        if candidate.startswith("在"):
            continue
        if re.search(r"\d{1,2}:\d{2}", candidate):
            continue
        if re.fullmatch(r"\d+\s*个.*", candidate):
            continue
        if blocked_names and candidate in blocked_names:
            continue
        return candidate
    return None


def _parse_origin_hard_logic_item(index: int, logic: str) -> LogicalConstraintObject:
    """将单条原始 hard_logic_py 表达式解析为 LogicalConstraintObject。

    按优先级依次尝试匹配：天数、人数、预算、门票数、打车数、集合约束、
    房间数/房型、住宿/餐饮人均成本、交通时间窗口等。
    所有模式均未命中时，降级为 'python_expression' 类型保留原始代码。

    Args:
        index: 约束序号。
        logic: 原始 Python 表达式字符串。

    Returns:
        解析后的 LogicalConstraintObject。
    """
    if match := _DAY_PATTERN.search(logic):
        return _make_logical(index, "day_count", int(match.group(2)), source="origin", operator=match.group(1), raw_code=logic)

    if match := _PEOPLE_PATTERN.search(logic):
        return _make_logical(index, "people_count", int(match.group(2)), source="origin", operator=match.group(1), raw_code=logic)

    if match := _BUDGET_PATTERN.search(logic):
        return _make_logical(index, "budget_total", float(match.group(1)), source="origin", operator="<=", raw_code=logic)

    if "activity_tickets(activity)" in logic:
        count_match = re.search(r"activity_tickets\(activity\)!=\s*(\d+)", logic)
        value = int(count_match.group(1)) if count_match else None
        return _make_logical(
            index,
            "ticket_count_match",
            value,
            source="origin",
            params={"include_metro": "metro_tickets" in logic},
            raw_code=logic,
        )

    if "taxi_cars(activity_transports(activity))" in logic:
        count_match = re.search(r"taxi_cars\(activity_transports\(activity\)\)!=\s*(\d+)", logic)
        value = int(count_match.group(1)) if count_match else None
        return _make_logical(index, "taxi_car_count_match", value, source="origin", raw_code=logic)

    if match := _SET_PATTERN.search(logic):
        values = _parse_set_literal(match.group(1))
        relation = "equals_set" if match.group(2) == "==" else "contains_all"
        set_name = match.group(3)
        if set_name == "intercity_transport_set":
            return _make_logical(index, "required_intercity_transport_type", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "innercity_transport_set":
            return _make_logical(index, "required_innercity_transport_type", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "attraction_name_set":
            return _make_logical(index, "required_attraction_name", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "restaurant_name_set":
            return _make_logical(index, "required_restaurant_name", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "accommodation_name_set":
            return _make_logical(index, "required_hotel_name", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "attraction_type_set":
            return _make_logical(index, "required_attraction_type", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "restaurant_type_set":
            return _make_logical(index, "required_restaurant_type", values, source="origin", operator=relation, raw_code=logic)
        if set_name == "accommodation_type_set":
            return _make_logical(index, "required_hotel_feature", values, source="origin", operator=relation, raw_code=logic)

    if match := _HOTEL_COST_PATTERN.search(logic):
        return _make_logical(
            index,
            "activity_budget_limit",
            float(match.group(2)),
            source="origin",
            operator=match.group(1),
            target={"activity_type": "accommodation"},
            params={"metric": "avg_cost_per_person_per_night"},
            raw_code=logic,
        )

    if match := _FOOD_COST_PATTERN.search(logic):
        return _make_logical(
            index,
            "activity_budget_limit",
            float(match.group(2)),
            source="origin",
            operator=match.group(1),
            target={"activity_type": "meal"},
            params={"metric": "avg_cost_per_meal_per_person"},
            raw_code=logic,
        )

    if "go_start_time" in logic:
        value_match = re.search(r"result=\(go_start_time([<>=!]+)'([^']+)'\)", logic)
        if value_match:
            return _make_logical(
                index,
                "transport_time_window",
                value_match.group(2),
                source="origin",
                operator=value_match.group(1),
                params={"leg": "outbound", "field": "start_time"},
                raw_code=logic,
            )

    if "go_end_time" in logic:
        value_match = re.search(r"result=\(go_end_time([<>=!]+)'([^']+)'\)", logic)
        if value_match:
            return _make_logical(
                index,
                "transport_time_window",
                value_match.group(2),
                source="origin",
                operator=value_match.group(1),
                params={"leg": "outbound", "field": "end_time"},
                raw_code=logic,
            )

    if "return_start_time" in logic or "back_end_time" in logic:
        variable = "return_start_time" if "return_start_time" in logic else "back_end_time"
        field = "start_time" if variable == "return_start_time" else "end_time"
        value_match = re.search(rf"result=\({variable}([<>=!]+)'([^']+)'\)", logic)
        if value_match:
            return _make_logical(
                index,
                "transport_time_window",
                value_match.group(2),
                source="origin",
                operator=value_match.group(1),
                params={"leg": "return", "field": field},
                raw_code=logic,
            )

    return _make_logical(
        index,
        "python_expression",
        None,
        source="origin",
        params={"code": logic},
        raw_code=logic,
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _dedupe_constraints(items: Iterable[LogicalConstraintObject]) -> List[LogicalConstraintObject]:
    """对约束列表按 (type, target, operator, value, params) 组合键去重，保留首次出现的约束。"""
    seen = set()
    output: List[LogicalConstraintObject] = []
    for item in items:
        key = (
            item.type,
            _freeze_value(item.target),
            item.operator,
            _freeze_value(item.value),
            _freeze_value(item.params),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def build_origin_logical_constraints(origin_query_structured: Dict[str, Any]) -> List[LogicalConstraintObject]:
    """从原始结构化查询构建硬约束列表。

    解析 origin_query_structured 中的 'hard_logic_py' 字段（Python 表达式列表），
    逐条转换为 LogicalConstraintObject。若 hard_logic_py 中未包含天数或人数约束，
    则尝试从 'days' / 'people_number' 字段补充。

    Args:
        origin_query_structured: 原始查询的结构化字典，包含 hard_logic_py、days、people_number 等字段。

    Returns:
        经过 schema 验证的 LogicalConstraintObject 列表。
    """
    query = origin_query_structured or {}
    items = _normalize_sequence_field(query.get("hard_logic_py", []))
    constraints = [_parse_origin_hard_logic_item(index, str(logic)) for index, logic in enumerate(items)]

    existing_types = {item.type for item in constraints}
    if "day_count" not in existing_types and query.get("days") is not None:
        constraints.append(_make_logical(len(constraints), "day_count", int(query["days"]), source="origin"))
    if "people_count" not in existing_types and query.get("people_number") is not None:
        constraints.append(_make_logical(len(constraints), "people_count", int(query["people_number"]), source="origin"))

    validate_logical_constraint_schema(constraints)
    return constraints


def build_edit_logical_constraints(
    edit_query: str,
    origin_plan: Optional[Dict[str, Any]] = None,
    origin_query_structured: Optional[Dict[str, Any]] = None,
) -> List[LogicalConstraintObject]:
    """从自然语言编辑指令中提取硬约束列表。

    依次尝试从文本中提取：游玩时间限制、总预算、天数、酒店名称、
    餐厅名称、景点名称、交通时间窗口，并去重后返回。

    Args:
        edit_query: 用户的编辑指令文本（中文自然语言）。
        origin_plan: 原始旅行计划（当前未使用，保留接口）。
        origin_query_structured: 原始结构化查询（当前未使用，保留接口）。

    Returns:
        经过去重和 schema 验证的 LogicalConstraintObject 列表。

    Raises:
        ValueError: edit_query 为空时抛出。
    """
    text = str(edit_query or "").strip()
    if not text:
        raise ValueError("edit_query is required to build edit_target_constraints")
    soft_only_query = _contains_soft_only_cues(text)

    extracted: List[LogicalConstraintObject] = []
    blocked_names = _collect_city_tokens(origin_plan, origin_query_structured)
    target_city = _resolve_target_city(origin_plan, origin_query_structured)
    ticket_substitute_anchor_name: Optional[str] = None

    duration = _extract_duration_limit(text)
    if duration is not None:
        poi_name, minutes = duration
        extracted.append(
            _make_logical(
                len(extracted),
                "activity_duration_limit",
                minutes,
                source="edit",
                operator="<=",
                target={"poi_name": poi_name},
                params={"derived_from": "edit_query"},
            )
        )

    day_end_limit = _extract_day_end_time_limit(text)
    if day_end_limit is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "day_end_time_limit",
                day_end_limit["value"],
                source="edit",
                operator="<=",
                target={"day": day_end_limit["day"] if day_end_limit["day"] is not None else "all"},
                params={"derived_from": "edit_query"},
            )
        )

    poi_window = _extract_poi_time_window(text)
    if poi_window is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "poi_time_window",
                poi_window["value"],
                source="edit",
                operator="in",
                target={"poi_name": poi_window["poi_name"]},
                params={"day": poi_window["day"], "derived_from": "edit_query"},
            )
        )

    for poi_clock_window in _extract_poi_clock_time_windows(text):
        extracted.append(
            _make_logical(
                len(extracted),
                "poi_clock_time_window",
                poi_clock_window["value"],
                source="edit",
                operator="in",
                target={"poi_name": poi_clock_window["poi_name"]},
                params={
                    "day": poi_clock_window["day"],
                    "mode": poi_clock_window["mode"],
                    "derived_from": "edit_query",
                },
            )
        )

    budget_total = _extract_budget_total(text)
    if budget_total is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "budget_total",
                budget_total,
                source="edit",
                operator="<=",
                params={"derived_from": "edit_query"},
            )
        )

    for activity_budget_limit in _extract_activity_budget_limits(text):
        extracted.append(
            _make_logical(
                len(extracted),
                "activity_budget_limit",
                float(activity_budget_limit["value"]),
                source="edit",
                operator="<=",
                target={"activity_type": activity_budget_limit["activity_type"]},
                params={
                    "metric": activity_budget_limit["metric"],
                    "derived_from": "edit_query",
                },
            )
        )

    ticket_budget_total = _extract_ticket_budget_limit(text)
    if ticket_budget_total is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "ticket_budget_total",
                ticket_budget_total,
                source="edit",
                operator="<=",
                params={"derived_from": "edit_query"},
            )
        )

    people_count = _extract_people_count_update(text)
    if people_count is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "people_count",
                people_count,
                source="edit",
                operator="==",
                params={"derived_from": "edit_query"},
            )
        )

    intercity_modes = extract_intercity_transport_modes_from_query(text)
    if intercity_modes:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_intercity_transport_type",
                intercity_modes,
                source="edit",
                operator="contains_all",
                target={"activity_type": "intercity_transport"},
                params={"derived_from": "edit_query"},
            )
        )

    daily_poi_cap = _extract_daily_poi_cap(text)
    if daily_poi_cap is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "daily_poi_cap",
                daily_poi_cap,
                source="edit",
                operator="<=",
                target={"day": "all"},
                params={"count_types": ["attraction"], "derived_from": "edit_query"},
            )
        )

    multi_day_requirement = _extract_multi_day_requirement(text)
    if multi_day_requirement is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "day_count",
                multi_day_requirement["value"],
                source="edit",
                operator=str(multi_day_requirement["operator"]),
                params={"derived_from": "edit_query", "from_multi_day_requirement": True},
            )
        )

    day_count = _extract_day_count(text)
    if day_count is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "day_count",
                day_count,
                source="edit",
                operator="==",
                params={"derived_from": "edit_query"},
            )
        )

    adjacent_travel_time_cap = _extract_adjacent_travel_time_cap(text)
    if adjacent_travel_time_cap is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "adjacent_travel_time_cap",
                adjacent_travel_time_cap["value"],
                source="edit",
                operator="<=",
                target={"activity_types": adjacent_travel_time_cap["activity_types"]},
                params={"unit": "minute", "day": "all", "derived_from": "edit_query"},
            )
        )

    adjacent_travel_distance_cap = _extract_adjacent_travel_distance_cap(text)
    if adjacent_travel_distance_cap is not None:
        anchor_poi = adjacent_travel_distance_cap.get("anchor_poi")
        if anchor_poi:
            _raise_if_transport_hub_name(anchor_poi, context="adjacent_travel_distance_cap")
            extracted.append(
                _make_logical(
                    len(extracted),
                    "poi_day_binding",
                    int(adjacent_travel_distance_cap["day"]),
                    source="edit",
                    operator="==",
                    target={"poi_name": anchor_poi},
                    params={"derived_from": "edit_query"},
                )
            )
        extracted.append(
            _make_logical(
                len(extracted),
                "adjacent_travel_distance_cap",
                float(adjacent_travel_distance_cap["value"]),
                source="edit",
                operator="<=",
                target={"activity_types": adjacent_travel_distance_cap["activity_types"]},
                params={
                    "unit": "km",
                    "day": adjacent_travel_distance_cap.get("day", "all"),
                    "derived_from": "edit_query",
                },
            )
        )

    walking_distance_limit = _extract_pairwise_walking_distance_limit(text)
    if walking_distance_limit is not None:
        for poi_name in (walking_distance_limit["first_poi"], walking_distance_limit["second_poi"]):
            _raise_if_transport_hub_name(poi_name, context="pairwise_transport_mode_distance_cap")
        extracted.append(
            _make_logical(
                len(extracted),
                "pairwise_transport_mode_distance_cap",
                float(walking_distance_limit["value"]),
                source="edit",
                operator="<=",
                target={
                    "first_poi": walking_distance_limit["first_poi"],
                    "second_poi": walking_distance_limit["second_poi"],
                    "mode": walking_distance_limit["mode"],
                },
                params={"unit": "km", "derived_from": "edit_query"},
            )
        )

    access_time_limit = _extract_access_time_inbound_cap(text)
    if access_time_limit is not None:
        _raise_if_transport_hub_name(access_time_limit["poi_name"], context="poi_inbound_travel_time_cap")
        extracted.append(
            _make_logical(
                len(extracted),
                "poi_inbound_travel_time_cap",
                int(access_time_limit["value"]),
                source="edit",
                operator="<=",
                target={"poi_name": access_time_limit["poi_name"]},
                params={"unit": "minute", "derived_from": "edit_query"},
            )
        )

    ticket_substitute = _extract_ticket_price_cap_or_substitute(text)
    if ticket_substitute is not None:
        _raise_if_transport_hub_name(ticket_substitute["poi_name"], context="ticket_price_cap_or_substitute")
        if not target_city:
            raise ValueError("target_city is required for ticket_price_cap_or_substitute extraction")
        poi_record = _lookup_named_poi(target_city, ticket_substitute["poi_name"])
        if poi_record is None:
            raise ValueError(f"ticket_price_cap_or_substitute target '{ticket_substitute['poi_name']}' not found in database")
        target_eligible = float(poi_record["price"]) <= float(ticket_substitute["price_cap"])
        candidate_names: List[str] = []
        if not target_eligible:
            candidate_names = _same_type_price_capped_candidates(
                target_city,
                str(poi_record["category"]),
                poi_record["semantic_type"],
                float(ticket_substitute["price_cap"]),
            )
            candidate_names = [name for name in candidate_names if name != ticket_substitute["poi_name"]]
            if not candidate_names:
                raise ValueError(
                    f"ticket_price_cap_or_substitute has no same-type under-cap substitute for '{ticket_substitute['poi_name']}'"
                )
        ticket_substitute_anchor_name = ticket_substitute["poi_name"]
        blocked_names.add(ticket_substitute_anchor_name)
        activity_type = "attraction"
        if poi_record["category"] == "restaurant":
            activity_type = "meal"
        elif poi_record["category"] == "accommodation":
            activity_type = "accommodation"
        extracted.append(
            _make_logical(
                len(extracted),
                "ticket_price_cap_or_substitute",
                float(ticket_substitute["price_cap"]),
                source="edit",
                operator="<=",
                target={"poi_name": ticket_substitute["poi_name"], "activity_type": activity_type},
                params={
                    "day": ticket_substitute["day"],
                    "city": target_city,
                    "poi_category": poi_record["category"],
                    "semantic_type": poi_record["semantic_type"],
                    "target_price": float(poi_record["price"]),
                    "target_eligible": target_eligible,
                    "candidate_names": candidate_names,
                    "derived_from": "edit_query",
                },
            )
        )

    radius_limit = _extract_anchor_neighbor_commute_distance_cap(text)
    if radius_limit is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "anchor_neighbor_commute_distance_cap",
                radius_limit["value"],
                source="edit",
                operator="<=",
                target={"anchor_name": radius_limit["anchor_name"]},
                params={"unit": "km", "derived_from": "edit_query"},
            )
        )

    anchor_bundle_budget_limit = _extract_anchor_bundle_budget_limit(text)
    if anchor_bundle_budget_limit is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "anchor_bundle_budget_limit",
                float(anchor_bundle_budget_limit["value"]),
                source="edit",
                operator="<=",
                target={"anchor_poi": anchor_bundle_budget_limit["anchor_poi"]},
                params={
                    "budget_scope": "anchor_bundle",
                    "requires_nearby_meal": bool(anchor_bundle_budget_limit.get("requires_nearby_meal", True)),
                    "derived_from": "edit_query",
                },
            )
        )

    theme_requirement = _extract_theme_semantic_requirement(text)
    if theme_requirement:
        extracted.append(
            _make_logical(
                len(extracted),
                "semantic_type_requirement",
                str(theme_requirement),
                source="edit",
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "edit_query"},
            )
        )

    remove_poi_name = _extract_remove_poi_name(text, blocked_names=blocked_names)
    if remove_poi_name:
        extracted.append(
            _make_logical(
                len(extracted),
                "forbidden_attraction_name",
                [remove_poi_name],
                source="edit",
                operator="contains_none",
                target={"activity_type": "attraction"},
                params={"derived_from": "edit_query"},
            )
        )
        blocked_names.add(remove_poi_name)

    substitute_poi_name = _extract_substitute_poi(text)
    if substitute_poi_name:
        _raise_if_transport_hub_name(substitute_poi_name, context="substitute_poi")
        if not target_city:
            raise ValueError("target_city is required for substitute_poi extraction")
        poi_record = _lookup_named_poi(target_city, substitute_poi_name)
        if poi_record is None or str(poi_record["category"]) != "attraction":
            raise ValueError(f"substitute_poi target '{substitute_poi_name}' not found as attraction in database")
        candidates = [name for name in _same_type_candidates(target_city, "attraction", poi_record["semantic_type"]) if name != substitute_poi_name]
        if not candidates:
            raise ValueError(f"substitute_poi has no same-type substitute candidate for '{substitute_poi_name}'")
        extracted.append(
            _make_logical(
                len(extracted),
                "forbidden_attraction_name",
                [substitute_poi_name],
                source="edit",
                operator="contains_none",
                target={"activity_type": "attraction"},
                params={"derived_from": "edit_query", "from_substitute_poi": True},
            )
        )
        extracted.append(
            _make_logical(
                len(extracted),
                "semantic_type_requirement",
                str(poi_record["semantic_type"]),
                source="edit",
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "edit_query", "candidate_names": candidates},
            )
        )

    semantic_substitution = _extract_semantic_substitution(text)
    if semantic_substitution:
        _raise_if_transport_hub_name(semantic_substitution["source_poi"], context="semantic_substitution")
        extracted.append(
            _make_logical(
                len(extracted),
                "forbidden_attraction_name",
                [semantic_substitution["source_poi"]],
                source="edit",
                operator="contains_none",
                target={"activity_type": "attraction"},
                params={"derived_from": "edit_query", "from_semantic_substitution": True},
            )
        )
        extracted.append(
            _make_logical(
                len(extracted),
                "semantic_type_requirement",
                str(semantic_substitution["semantic_target"]),
                source="edit",
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={
                    "min_count": 1,
                    "derived_from": "edit_query",
                    "replace_from": semantic_substitution["source_poi"],
                    "from_semantic_substitution": True,
                },
            )
        )

    type_replacement = _extract_type_replacement(text)
    if type_replacement:
        _raise_if_transport_hub_name(type_replacement["source_poi"], context="type_replacement")
        extracted.append(
            _make_logical(
                len(extracted),
                "forbidden_attraction_name",
                [type_replacement["source_poi"]],
                source="edit",
                operator="contains_none",
                target={"activity_type": "attraction"},
                params={"derived_from": "edit_query", "from_type_replacement": True},
            )
        )
        extracted.append(
            _make_logical(
                len(extracted),
                "semantic_type_requirement",
                str(type_replacement["semantic_type"]),
                source="edit",
                operator="includes_type",
                target={"activity_type": "attraction"},
                params={"min_count": 1, "derived_from": "edit_query"},
            )
        )

    hotel_name = _extract_required_hotel_name(text, blocked_names=blocked_names)
    if hotel_name:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_hotel_name",
                [hotel_name],
                source="edit",
                operator="contains_all",
                target={"activity_type": "accommodation"},
                params={"derived_from": "edit_query"},
            )
        )

    restaurant_name = _extract_required_restaurant_name(text, blocked_names=blocked_names)
    if restaurant_name:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_restaurant_name",
                [restaurant_name],
                source="edit",
                operator="contains_all",
                target={"activity_type": "meal"},
                params={"derived_from": "edit_query"},
            )
        )

    restaurant_type = _extract_required_restaurant_type(text)
    if restaurant_type:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_restaurant_type",
                [restaurant_type],
                source="edit",
                operator="contains_all",
                target={"activity_type": "meal"},
                params={"derived_from": "edit_query"},
            )
        )

    hotel_feature = _extract_required_hotel_feature(text)
    if hotel_feature:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_hotel_feature",
                [hotel_feature],
                source="edit",
                operator="contains_all",
                target={"activity_type": "accommodation"},
                params={"derived_from": "edit_query"},
            )
        )

    room_type = _extract_required_room_type(text)
    if room_type is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_room_type",
                int(room_type),
                source="edit",
                operator="==",
                target={"activity_type": "accommodation"},
                params={"derived_from": "edit_query"},
            )
        )

    room_count = _extract_required_room_count(text)
    if room_count is not None:
        extracted.append(
            _make_logical(
                len(extracted),
                "required_room_count",
                int(room_count),
                source="edit",
                operator="==",
                target={"activity_type": "accommodation"},
                params={"derived_from": "edit_query"},
            )
        )

    for activity_budget in _extract_activity_budget_requirements(text):
        extracted.append(
            _make_logical(
                len(extracted),
                "activity_budget_limit",
                float(activity_budget["budget_cap"]),
                source="edit",
                operator="<=",
                target={"activity_type": activity_budget["activity_type"]},
                params={"metric": activity_budget["metric"], "derived_from": "edit_query"},
            )
        )

    order_constraint = _extract_poi_order_constraint(text)
    if order_constraint is not None:
        _raise_if_transport_hub_name(order_constraint["first_poi"], context="poi_order_constraint")
        _raise_if_transport_hub_name(order_constraint["second_poi"], context="poi_order_constraint")
        extracted.append(
            _make_logical(
                len(extracted),
                "poi_order_constraint",
                [order_constraint["first_poi"], order_constraint["second_poi"]],
                source="edit",
                operator="before",
                target={"first_poi": order_constraint["first_poi"], "second_poi": order_constraint["second_poi"]},
                params={"derived_from": "edit_query"},
            )
        )

    same_day_no_overlap = _extract_same_day_no_overlap_compact(text)
    if same_day_no_overlap is not None:
        for poi_name in (same_day_no_overlap["first_poi"], same_day_no_overlap["second_poi"]):
            _raise_if_transport_hub_name(poi_name, context="pair_same_day_no_overlap")
        extracted.append(
            _make_logical(
                len(extracted),
                "pair_same_day_no_overlap",
                int(same_day_no_overlap["day"]),
                source="edit",
                operator="==",
                target={"first_poi": same_day_no_overlap["first_poi"], "second_poi": same_day_no_overlap["second_poi"]},
                params={"derived_from": "edit_query"},
            )
        )

    same_period_no_overlap = _extract_same_time_period_no_overlap(text)
    if same_period_no_overlap is not None:
        for poi_name in (same_period_no_overlap["first_poi"], same_period_no_overlap["second_poi"]):
            _raise_if_transport_hub_name(poi_name, context="pair_time_window_no_overlap")
        extracted.append(
            _make_logical(
                len(extracted),
                "pair_time_window_no_overlap",
                str(same_period_no_overlap["time_period"]),
                source="edit",
                operator="in",
                target={"first_poi": same_period_no_overlap["first_poi"], "second_poi": same_period_no_overlap["second_poi"]},
                params={"derived_from": "edit_query"},
            )
        )

    if not soft_only_query:
        for binding in _extract_poi_day_bindings(text):
            if (
                ticket_substitute_anchor_name
                and binding["poi_name"] == ticket_substitute_anchor_name
            ):
                continue
            _raise_if_transport_hub_name(binding["poi_name"], context="poi_day_binding")
            extracted.append(
                _make_logical(
                    len(extracted),
                    "poi_day_binding",
                    binding["day"],
                    source="edit",
                    operator="==",
                    target={"poi_name": binding["poi_name"]},
                    params={"derived_from": "edit_query"},
                )
            )

    if not soft_only_query:
        specific_poi_name = _extract_specific_poi_name(text, blocked_names=blocked_names)
        if specific_poi_name:
            _raise_if_transport_hub_name(specific_poi_name, context="required_attraction_name")
            extracted.append(
                _make_logical(
                    len(extracted),
                    "required_attraction_name",
                    [specific_poi_name],
                    source="edit",
                    operator="contains_all",
                    target={"activity_type": "attraction"},
                    params={"derived_from": "edit_query", "semantic_specific_only": True},
                )
            )
            blocked_names.add(specific_poi_name)

    if not soft_only_query and duration is None:
        attraction_name = _extract_required_attraction_name(text, blocked_names=blocked_names)
        if attraction_name:
            _raise_if_transport_hub_name(attraction_name, context="required_attraction_name")
            extracted.append(
                _make_logical(
                    len(extracted),
                    "required_attraction_name",
                    [attraction_name],
                    source="edit",
                    operator="contains_all",
                    target={"activity_type": "attraction"},
                    params={"derived_from": "edit_query"},
                )
            )

    transport_windows = _extract_transport_time_window(text)
    for transport_window in transport_windows:
        extracted.append(
            _make_logical(
                len(extracted),
                "transport_time_window",
                transport_window["value"],
                source="edit",
                operator=transport_window["operator"],
                params={
                    "leg": transport_window["leg"],
                    "field": transport_window["field"],
                    "derived_from": "edit_query",
                },
            )
        )

    deduped = _dedupe_constraints(extracted)
    validate_logical_constraint_schema(deduped)
    return deduped


def serialize_logical_constraints(items: Iterable[LogicalConstraintObject]) -> List[Dict[str, Any]]:
    """将 LogicalConstraintObject 列表序列化为可 JSON 化的字典列表。"""
    return [item.to_dict() for item in items]


def validate_logical_constraint_schema(items: Iterable[LogicalConstraintObject]) -> None:
    """验证约束对象的 schema 完整性。

    检查每个 LogicalConstraintObject 序列化后是否包含所有必需字段
    （id、source、type、scope、target、operator、value、params、is_hard、provenance），
    并确保 id / type / source 不为空。

    Raises:
        ValueError: 缺少必需字段或关键标识字段为空时抛出。
    """
    required_keys = {"id", "source", "type", "scope", "target", "operator", "value", "params", "is_hard", "provenance"}
    for item in items:
        payload = item.to_dict()
        missing = sorted(required_keys - payload.keys())
        if missing:
            raise ValueError(f"logical constraint missing keys: {missing}")
        if not payload["id"] or not payload["type"] or not payload["source"]:
            raise ValueError(f"logical constraint missing required identity fields: {payload}")
