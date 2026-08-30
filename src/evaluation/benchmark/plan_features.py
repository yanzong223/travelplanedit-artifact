"""Shared plan feature extraction for preference scorers."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

try:  # pragma: no cover - optional ChinaTravel-backed helpers
    from chinatravel.symbol_verification.concept_func import (  # type: ignore
        accommodation_type as chinatravel_accommodation_type,
        attraction_type as chinatravel_attraction_type,
        restaurant_type as chinatravel_restaurant_type,
    )
except Exception:  # pragma: no cover - fallback when ChinaTravel import is unavailable
    chinatravel_accommodation_type = None
    chinatravel_attraction_type = None
    chinatravel_restaurant_type = None


TRANSPORT_ACTIVITY_TYPES = {"train", "airplane", "flight"}
MEAL_ACTIVITY_TYPES = {"breakfast", "lunch", "dinner", "restaurant"}
COUNTED_ACTIVITY_TYPES = {"attraction", "accommodation", *MEAL_ACTIVITY_TYPES}
THEME_BUCKETS = ("culture", "nature", "food", "photo", "family", "shopping")

THEME_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "culture": (
        "博物馆",
        "历史",
        "古迹",
        "文化",
        "纪念馆",
        "古镇",
        "古街",
        "museum",
        "heritage",
        "historic",
        "culture",
    ),
    "nature": (
        "自然",
        "风光",
        "山水",
        "公园",
        "山",
        "湖",
        "湿地",
        "植物园",
        "动物园",
        "nature",
        "scenic",
        "park",
    ),
    "food": (
        "美食",
        "小吃",
        "火锅",
        "菜",
        "cuisine",
        "food",
        "local delicacy",
    ),
    "photo": (
        "拍照",
        "出片",
        "打卡",
        "网红",
        "夜景",
        "观景台",
        "地标",
        "photo",
        "photogenic",
    ),
    "family": (
        "亲子",
        "儿童",
        "孩子",
        "family",
        "kid",
        "children",
        "游乐园",
        "海洋馆",
        "儿童乐园",
    ),
    "shopping": (
        "购物",
        "商场",
        "商业街",
        "shopping",
        "mall",
        "免税",
    ),
}

VIBE_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "relaxed": (
        "放松",
        "松弛",
        "漫步",
        "散步",
        "江边",
        "江景",
        "湖边",
        "公园",
        "温泉",
        "咖啡",
        "咖啡馆",
        "茶馆",
        "度假",
        "park",
        "lake",
        "cafe",
        "café",
        "tea",
        "spa",
        "resort",
    ),
    "lively": (
        "热闹",
        "繁华",
        "商圈",
        "商业街",
        "商场",
        "市集",
        "集市",
        "打卡",
        "夜景",
        "演出",
        "livehouse",
        "shopping",
        "mall",
        "market",
        "show",
        "entertainment",
    ),
    "quiet": (
        "安静",
        "清净",
        "静谧",
        "博物馆",
        "美术馆",
        "书店",
        "寺",
        "园林",
        "湖",
        "公园",
        "museum",
        "gallery",
        "library",
        "garden",
        "cafe",
        "café",
    ),
    "nightlife": (
        "夜景",
        "夜市",
        "夜游",
        "酒吧",
        "清吧",
        "livehouse",
        "club",
        "bar",
        "night market",
        "nightlife",
        "disco",
        "演出",
    ),
}


@dataclass(slots=True)
class EdgeFeature:
    day: int
    prev_name: str
    next_name: str
    travel_minutes: float
    distance_km: float
    gap_minutes: Optional[float]


def _itinerary(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return plan.get("itinerary", []) if isinstance(plan, dict) else []


TICKET_BUDGET_SEMANTICS_VERSION = "ticket_budget_total_v2_cost_else_price_times_people"


def _ticket_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _ticket_people_count(plan: Dict[str, Any]) -> float:
    people = _ticket_numeric(plan.get("people_number")) if isinstance(plan, dict) else None
    return people if people is not None and people > 0 else 1.0


def attraction_ticket_cost(
    activity: Dict[str, Any],
    *,
    people_number: float,
) -> Dict[str, Any]:
    """Resolve one attraction's whole-party ticket cost and consistency evidence.

    ``cost`` is authoritative when numeric because benchmark plans store it as an
    already-aggregated amount.  When it is absent or non-numeric, the fallback is
    the per-person ``price`` multiplied by the plan-level party size.
    """

    cost = _ticket_numeric(activity.get("cost"))
    price = _ticket_numeric(activity.get("price"))
    expected_from_price = price * people_number if price is not None else None
    if cost is not None:
        resolved = cost
        source = "cost"
    elif expected_from_price is not None:
        resolved = expected_from_price
        source = "price_x_people_number"
    else:
        resolved = 0.0
        source = "missing_price_and_cost"

    consistent: Optional[bool] = None
    if cost is not None and expected_from_price is not None:
        consistent = math.isclose(cost, expected_from_price, rel_tol=1e-6, abs_tol=0.01)

    return {
        "amount": float(resolved),
        "source": source,
        "cost": cost,
        "price": price,
        "people_number": people_number,
        "expected_price_times_people": expected_from_price,
        "price_cost_consistent": consistent,
    }


def summarize_ticket_budget(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize whole-party attraction tickets for the complete itinerary."""

    people_number = _ticket_people_count(plan)
    items: List[Dict[str, Any]] = []
    inconsistencies: List[Dict[str, Any]] = []
    total = 0.0
    for day_index, day in enumerate(_itinerary(plan)):
        if not isinstance(day, dict):
            continue
        activities = day.get("activities", [])
        if not isinstance(activities, list):
            continue
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict) or activity.get("type") != "attraction":
                continue
            resolved = attraction_ticket_cost(activity, people_number=people_number)
            row = {
                **resolved,
                "day": day.get("day", day_index + 1),
                "activity_index": activity_index,
                "poi_name": _activity_name(activity),
                "source_path": f"itinerary[{day_index}].activities[{activity_index}]",
            }
            items.append(row)
            total += float(resolved["amount"])
            if resolved["price_cost_consistent"] is False:
                inconsistencies.append(row)
    return {
        "semantics_version": TICKET_BUDGET_SEMANTICS_VERSION,
        "people_number": people_number,
        "total": total,
        "items": items,
        "item_count": len(items),
        "price_cost_inconsistencies": inconsistencies,
        "price_cost_inconsistency_count": len(inconsistencies),
    }


def ticket_budget_total(plan: Dict[str, Any]) -> float:
    """Return total attraction-ticket spend for all travellers and all days."""

    return float(summarize_ticket_budget(plan)["total"])


def _activity_name(activity: Dict[str, Any]) -> str:
    for key in ("position", "name", "end", "start"):
        value = str(activity.get(key, "") or "").strip()
        if value:
            return value
    return ""


def parse_time_to_minutes(value: Any) -> Optional[int]:
    if not isinstance(value, str) or ":" not in value:
        return None
    hour, minute = value.split(":")[:2]
    try:
        parsed = int(hour) * 60 + int(minute)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def duration_minutes(start: Any, end: Any) -> Optional[int]:
    start_minutes = parse_time_to_minutes(start)
    end_minutes = parse_time_to_minutes(end)
    if start_minutes is None or end_minutes is None or end_minutes < start_minutes:
        return None
    return end_minutes - start_minutes


def _transport_segments(activity: Dict[str, Any]) -> List[Dict[str, Any]]:
    transports = activity.get("transports", [])
    return transports if isinstance(transports, list) else []


def _sum_transport_minutes(activity: Dict[str, Any]) -> float:
    total = 0.0
    for segment in _transport_segments(activity):
        minutes = duration_minutes(segment.get("start_time"), segment.get("end_time"))
        if minutes is not None:
            total += float(minutes)
    return total


def _sum_transport_distance(activity: Dict[str, Any]) -> float:
    total = 0.0
    for segment in _transport_segments(activity):
        distance = segment.get("distance")
        if isinstance(distance, (int, float)):
            total += float(distance)
    return total


_COST_CATEGORIES = ("food", "hotel", "attraction", "transport", "other")


def _cost_category(activity_type: str) -> str:
    if activity_type in MEAL_ACTIVITY_TYPES:
        return "food"
    if activity_type == "accommodation":
        return "hotel"
    if activity_type == "attraction":
        return "attraction"
    if activity_type in TRANSPORT_ACTIVITY_TYPES:
        return "transport"
    return "other"


def _numeric_cost(value: Any) -> Optional[float]:
    # bool is technically numeric in Python, but it is never a valid price.
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _activity_label(activity: Dict[str, Any], activity_type: str, index: int) -> str:
    name = _activity_name(activity)
    if name:
        return name
    labels = {"train": "火车", "airplane": "飞机", "flight": "航班"}
    return labels.get(activity_type, f"活动 {index + 1}")


def _cost_formula(
    item: Dict[str, Any],
    *,
    cost: float,
    activity_type: str,
    people_count: int,
) -> tuple[float, float, str]:
    """Explain a pre-aggregated benchmark cost without changing its value."""

    candidates: list[tuple[str, Any]] = [
        ("tickets", item.get("tickets")),
        ("rooms", item.get("rooms")),
        ("cars", item.get("cars")),
    ]
    if activity_type in MEAL_ACTIVITY_TYPES:
        candidates.append(("people_number", people_count))
    price = _numeric_cost(item.get("price"))
    if price is not None:
        for field_name, raw_quantity in candidates:
            quantity = _numeric_cost(raw_quantity)
            if quantity is not None and quantity > 0 and math.isclose(price * quantity, cost, rel_tol=1e-9, abs_tol=1e-6):
                return price, quantity, f"price × {field_name}"
    # `cost` is the evaluator's authoritative, already-aggregated amount.
    return cost, 1.0, "cost (benchmark aggregate)"


def summarize_plan_cost(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return the benchmark cost aggregate plus participant-safe line items.

    This is the single public aggregation domain used by benchmark budget checks
    and the Human baseline. Missing prices contribute zero to the legacy benchmark
    aggregate, but are explicitly reported as unpriced so the Human UI never
    presents that subtotal as a complete estimate.
    """

    categories = {category: 0.0 for category in _COST_CATEGORIES}
    items: list[Dict[str, Any]] = []
    unpriced: list[Dict[str, Any]] = []
    people_count = max(int(plan.get("people_number", 1) or 1), 1) if isinstance(plan, dict) else 1

    def add_item(
        raw: Dict[str, Any],
        *,
        category: str,
        day_number: int,
        activity_index: int,
        transport_index: Optional[int],
        name: str,
        activity_type: str,
        source_path: str,
    ) -> None:
        cost = _numeric_cost(raw.get("cost")) if "cost" in raw else None
        base = {
            "day": day_number,
            "activity_index": activity_index,
            "transport_index": transport_index,
            "name": name,
            "activity_type": activity_type,
            "category": category,
            "source_path": source_path,
        }
        if cost is None:
            detail = {
                **base,
                "priced": False,
                "unit_price": None,
                "quantity": None,
                "formula": "missing cost",
                "subtotal": None,
            }
            items.append(detail)
            unpriced.append(detail)
            return
        unit_price, quantity, formula = _cost_formula(
            raw, cost=cost, activity_type=activity_type, people_count=people_count
        )
        categories[category] += cost
        items.append(
            {
                **base,
                "priced": True,
                "unit_price": unit_price,
                "quantity": quantity,
                "formula": formula,
                "subtotal": cost,
            }
        )

    for day_index, day in enumerate(_itinerary(plan)):
        if not isinstance(day, dict):
            continue
        try:
            day_number = int(day.get("day", day_index + 1) or day_index + 1)
        except (TypeError, ValueError):
            day_number = day_index + 1
        activities = day.get("activities", [])
        if not isinstance(activities, list):
            continue
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict):
                continue
            activity_type = str(activity.get("type", "") or "").strip()
            category = _cost_category(activity_type)
            activity_path = f"itinerary[{day_index}].activities[{activity_index}]"
            add_item(
                activity,
                category=category,
                day_number=day_number,
                activity_index=activity_index,
                transport_index=None,
                name=_activity_label(activity, activity_type, activity_index),
                activity_type=activity_type or "activity",
                source_path=f"{activity_path}.cost",
            )
            for transport_index, segment in enumerate(_transport_segments(activity)):
                if not isinstance(segment, dict):
                    continue
                mode = str(segment.get("mode", segment.get("type", "")) or "").strip() or "transport"
                start = str(segment.get("start", "") or "").strip()
                end = str(segment.get("end", "") or "").strip()
                name = f"{start} → {end}" if start or end else f"市内交通 {transport_index + 1}"
                add_item(
                    segment,
                    category="transport",
                    day_number=day_number,
                    activity_index=activity_index,
                    transport_index=transport_index,
                    name=name,
                    activity_type=mode,
                    source_path=f"{activity_path}.transports[{transport_index}].cost",
                )

    total = sum(categories.values())
    priced_count = len(items) - len(unpriced)
    return {
        "total": total,
        "categories": categories,
        "items": items,
        "unpriced_items": unpriced,
        "priced_count": priced_count,
        "unpriced_count": len(unpriced),
        "item_count": len(items),
        "coverage_ratio": priced_count / len(items) if items else 1.0,
        "status": "incomplete" if unpriced else "complete",
        "total_kind": "known_subtotal" if unpriced else "complete_total",
    }


def extract_cost_breakdown(plan: Dict[str, Any]) -> Dict[str, float]:
    summary = summarize_plan_cost(plan)
    return {**summary["categories"], "total": summary["total"]}


def extract_daily_activity_counts(plan: Dict[str, Any]) -> List[int]:
    counts: List[int] = []
    for day in _itinerary(plan):
        count = 0
        for activity in day.get("activities", []):
            if isinstance(activity, dict) and str(activity.get("type", "")).strip() in COUNTED_ACTIVITY_TYPES:
                count += 1
        counts.append(count)
    return counts


def extract_daily_attraction_count(plan: Dict[str, Any]) -> List[int]:
    counts: List[int] = []
    for day in _itinerary(plan):
        count = 0
        for activity in day.get("activities", []):
            if isinstance(activity, dict) and str(activity.get("type", "")).strip() == "attraction":
                count += 1
        counts.append(count)
    return counts


def extract_daily_travel_minutes(plan: Dict[str, Any], *, exclude_intercity: bool = True) -> List[float]:
    totals: List[float] = []
    for day in _itinerary(plan):
        total = 0.0
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            activity_type = str(activity.get("type", "")).strip()
            if exclude_intercity and activity_type in TRANSPORT_ACTIVITY_TYPES:
                continue
            if activity_type not in COUNTED_ACTIVITY_TYPES and exclude_intercity:
                continue
            total += _sum_transport_minutes(activity)
        totals.append(total)
    return totals


def extract_inter_activity_edges(plan: Dict[str, Any]) -> List[EdgeFeature]:
    edges: List[EdgeFeature] = []
    for day in _itinerary(plan):
        day_number = int(day.get("day", 0) or 0)
        activities = [activity for activity in day.get("activities", []) if isinstance(activity, dict)]
        for idx in range(1, len(activities)):
            prev_activity = activities[idx - 1]
            next_activity = activities[idx]
            travel_minutes = _sum_transport_minutes(next_activity)
            distance_km = _sum_transport_distance(next_activity)
            prev_end = parse_time_to_minutes(prev_activity.get("end_time"))
            next_start = parse_time_to_minutes(next_activity.get("start_time"))
            gap = None
            if prev_end is not None and next_start is not None:
                gap = float(next_start - prev_end - travel_minutes)
            edges.append(
                EdgeFeature(
                    day=day_number,
                    prev_name=_activity_name(prev_activity),
                    next_name=_activity_name(next_activity),
                    travel_minutes=travel_minutes,
                    distance_km=distance_km,
                    gap_minutes=gap,
                )
            )
    return edges


def _normalized_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_lower(value: Any) -> str:
    return _normalized_text(value).lower()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    haystack = _normalized_lower(text)
    return any(needle.lower() in haystack for needle in needles if needle)


def _average_numeric(values: Iterable[float]) -> Optional[float]:
    items = [float(value) for value in values if isinstance(value, (int, float))]
    if not items:
        return None
    return sum(items) / len(items)


def normalized_entropy(values: Iterable[float]) -> float:
    positive_values = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0]
    if len(positive_values) <= 1:
        return 0.0
    total = sum(positive_values)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for value in positive_values:
        probability = value / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(positive_values))


def _chinatravel_lookup(func: Any, activity: Dict[str, Any], city: str | None) -> str:
    if func is None or not city:
        return ""
    try:
        return _normalized_text(func(activity, city))
    except Exception:
        return ""


def _activity_theme_sources(activity: Dict[str, Any], target_city: str | None) -> List[str]:
    activity_type = _normalized_lower(activity.get("type"))
    sources: List[str] = []

    if activity_type == "attraction":
        sources.extend(
            value
            for value in (
                _normalized_text(activity.get("poiType")),
                _normalized_text(activity.get("category")),
                _chinatravel_lookup(chinatravel_attraction_type, activity, target_city),
            )
            if value
        )
    elif activity_type in MEAL_ACTIVITY_TYPES:
        sources.extend(
            value
            for value in (
                _normalized_text(activity.get("cuisine")),
                _normalized_text(activity.get("recommendedfood")),
                _chinatravel_lookup(chinatravel_restaurant_type, activity, target_city),
            )
            if value
        )
    elif activity_type == "accommodation":
        sources.extend(
            value
            for value in (
                _normalized_text(activity.get("featurehoteltype")),
                _normalized_text(activity.get("room_type")),
                _chinatravel_lookup(chinatravel_accommodation_type, activity, target_city),
            )
            if value
        )
    return sources


def _theme_buckets_for_text(text: str) -> Set[str]:
    labels: Set[str] = set()
    for bucket, keywords in THEME_KEYWORDS.items():
        if _contains_any(text, keywords):
            labels.add(bucket)
    return labels


def _theme_buckets_for_activity(activity: Dict[str, Any], target_city: str | None) -> Set[str]:
    labels: Set[str] = set()
    activity_type = _normalized_lower(activity.get("type"))
    if activity_type in MEAL_ACTIVITY_TYPES:
        labels.add("food")
    for source in _activity_theme_sources(activity, target_city):
        labels.update(_theme_buckets_for_text(source))
    return labels


def extract_theme_counter(plan: Dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    target_city = plan.get("target_city") if isinstance(plan, dict) else None
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            if str(activity.get("type", "")).strip() not in COUNTED_ACTIVITY_TYPES:
                continue
            labels = _theme_buckets_for_activity(activity, target_city)
            for label in labels:
                counter[label] += 1
    return counter


def extract_theme_distribution(plan: Dict[str, Any]) -> Dict[str, float]:
    counter = extract_theme_counter(plan)
    total = sum(counter.values())
    if total <= 0:
        return {bucket: 0.0 for bucket in THEME_BUCKETS}
    return {bucket: counter.get(bucket, 0) / total for bucket in THEME_BUCKETS}


def extract_attraction_type_distribution(plan: Dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    target_city = plan.get("target_city") if isinstance(plan, dict) else None
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            if str(activity.get("type", "")).strip() != "attraction":
                continue
            label = _normalized_text(
                activity.get("poiType")
                or activity.get("category")
                or _chinatravel_lookup(chinatravel_attraction_type, activity, target_city)
                or activity.get("name")
            )
            if label:
                counter[label] += 1
    return counter


def extract_food_experience_features(plan: Dict[str, Any]) -> Dict[str, Any]:
    target_city = plan.get("target_city") if isinstance(plan, dict) else None
    meal_count = 0
    cuisine_labels: Set[str] = set()
    recommended_food_labels: Set[str] = set()
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            if str(activity.get("type", "")).strip() not in MEAL_ACTIVITY_TYPES:
                continue
            meal_count += 1
            cuisine = _normalized_text(activity.get("cuisine") or _chinatravel_lookup(chinatravel_restaurant_type, activity, target_city))
            recommended_food = _normalized_text(activity.get("recommendedfood"))
            if cuisine:
                cuisine_labels.add(cuisine)
            if recommended_food:
                for item in re.split(r"[,，/、]", recommended_food):
                    cleaned = _normalized_text(item)
                    if cleaned:
                        recommended_food_labels.add(cleaned)
    return {
        "meal_count": meal_count,
        "distinct_cuisines": sorted(cuisine_labels),
        "distinct_cuisine_count": len(cuisine_labels),
        "distinct_recommended_food_count": len(recommended_food_labels),
        "recommended_food_hits": sorted(recommended_food_labels),
    }


def extract_walk_transport_stats(plan: Dict[str, Any]) -> Dict[str, float]:
    total_minutes = 0.0
    total_distance = 0.0
    segment_count = 0
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            for segment in _transport_segments(activity):
                mode = _normalized_lower(segment.get("mode"))
                if mode != "walk":
                    continue
                minutes = duration_minutes(segment.get("start_time"), segment.get("end_time"))
                if minutes is not None:
                    total_minutes += float(minutes)
                distance = segment.get("distance")
                if isinstance(distance, (int, float)):
                    total_distance += float(distance)
                segment_count += 1
    return {
        "segment_count": float(segment_count),
        "total_minutes": total_minutes,
        "total_distance": total_distance,
    }


GLOBAL_TRANSPORT_SEMANTICS_VERSION = "trip_innercity_transport_totals_v1"


def _normalized_transport_mode(segment: Dict[str, Any]) -> str:
    raw = _normalized_lower(segment.get("mode") or segment.get("type"))
    aliases = {
        "walking": "walk",
        "步行": "walk",
        "地铁": "metro",
        "subway": "metro",
        "出租车": "taxi",
        "打车": "taxi",
    }
    return aliases.get(raw, raw)


def summarize_innercity_transport_totals(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize whole-trip local transport duration, walk distance, and cost.

    Every nested ``activity.transports`` item is one local segment. Duration is
    derived from its start/end clock, walking distance includes only normalized
    walk segments, and cost uses the segment's numeric aggregate ``cost``.
    Missing metric fields are reported explicitly so generation can reject
    incomplete origins instead of treating missing values as zero evidence.
    """

    items: list[Dict[str, Any]] = []
    day_totals: Dict[int, Dict[str, float]] = {}
    missing = {"duration": 0, "distance": 0, "cost": 0, "mode": 0, "endpoints": 0}
    totals = {"duration_minutes": 0.0, "walking_distance_km": 0.0, "cost": 0.0}

    for day_index, day in enumerate(_itinerary(plan)):
        if not isinstance(day, dict):
            continue
        try:
            day_number = int(day.get("day", day_index + 1) or day_index + 1)
        except (TypeError, ValueError):
            day_number = day_index + 1
        per_day = day_totals.setdefault(
            day_number,
            {"duration_minutes": 0.0, "walking_distance_km": 0.0, "cost": 0.0, "segment_count": 0.0},
        )
        activities = day.get("activities", [])
        if not isinstance(activities, list):
            continue
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict):
                continue
            for transport_index, segment in enumerate(_transport_segments(activity)):
                if not isinstance(segment, dict):
                    continue
                mode = _normalized_transport_mode(segment)
                minutes = duration_minutes(segment.get("start_time"), segment.get("end_time"))
                distance = _numeric_cost(segment.get("distance"))
                cost = _numeric_cost(segment.get("cost"))
                start = str(segment.get("start", "") or "").strip()
                end = str(segment.get("end", "") or "").strip()
                if not mode:
                    missing["mode"] += 1
                if not start or not end:
                    missing["endpoints"] += 1
                if minutes is None:
                    missing["duration"] += 1
                else:
                    totals["duration_minutes"] += float(minutes)
                    per_day["duration_minutes"] += float(minutes)
                if distance is None:
                    missing["distance"] += 1
                elif mode == "walk":
                    totals["walking_distance_km"] += float(distance)
                    per_day["walking_distance_km"] += float(distance)
                if cost is None:
                    missing["cost"] += 1
                else:
                    totals["cost"] += float(cost)
                    per_day["cost"] += float(cost)
                per_day["segment_count"] += 1.0
                items.append(
                    {
                        "day": day_number,
                        "activity_index": activity_index,
                        "transport_index": transport_index,
                        "mode": mode,
                        "start": start,
                        "end": end,
                        "start_time": segment.get("start_time"),
                        "end_time": segment.get("end_time"),
                        "duration_minutes": float(minutes) if minutes is not None else None,
                        "distance_km": float(distance) if distance is not None else None,
                        "walking_distance_km": float(distance) if distance is not None and mode == "walk" else 0.0,
                        "cost": float(cost) if cost is not None else None,
                        "source_path": f"itinerary[{day_index}].activities[{activity_index}].transports[{transport_index}]",
                    }
                )

    return {
        "semantics_version": GLOBAL_TRANSPORT_SEMANTICS_VERSION,
        **totals,
        "segment_count": len(items),
        "contributing_day_count": sum(1 for values in day_totals.values() if values["segment_count"] > 0),
        "items": items,
        "day_totals": {str(day): values for day, values in sorted(day_totals.items())},
        "missing": missing,
        "complete": all(value == 0 for value in missing.values()),
    }


def extract_hotel_anchor_distances(plan: Dict[str, Any]) -> List[float]:
    hotel_names: Set[str] = set()
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            if _normalized_lower(activity.get("type")) != "accommodation":
                continue
            hotel_name = _activity_name(activity)
            if hotel_name:
                hotel_names.add(hotel_name)

    distances: List[float] = []
    for hotel_name in hotel_names:
        distances.extend(extract_anchor_distances(plan, hotel_name))
    return distances


def extract_route_smoothness_features(plan: Dict[str, Any]) -> Dict[str, Any]:
    edges = [edge for edge in extract_inter_activity_edges(plan) if edge.travel_minutes > 0 or edge.distance_km > 0]
    if not edges:
        return {
            "edge_count": 0,
            "travel_minutes": [],
            "distance_km": [],
            "long_jump_ratio": 0.0,
            "time_cv": 0.0,
            "distance_cv": 0.0,
        }

    travel_minutes = [float(edge.travel_minutes) for edge in edges if edge.travel_minutes > 0]
    distance_km = [float(edge.distance_km) for edge in edges if edge.distance_km > 0]
    avg_travel = sum(travel_minutes) / len(travel_minutes) if travel_minutes else 0.0
    avg_distance = sum(distance_km) / len(distance_km) if distance_km else 0.0

    long_jump_count = sum(1 for edge in edges if edge.travel_minutes >= 60 or edge.distance_km >= 8)
    time_cv = 0.0
    if avg_travel > 0 and len(travel_minutes) > 1:
        time_cv = math.sqrt(sum((value - avg_travel) ** 2 for value in travel_minutes) / len(travel_minutes)) / avg_travel
    distance_cv = 0.0
    if avg_distance > 0 and len(distance_km) > 1:
        distance_cv = math.sqrt(sum((value - avg_distance) ** 2 for value in distance_km) / len(distance_km)) / avg_distance

    return {
        "edge_count": len(edges),
        "travel_minutes": travel_minutes,
        "distance_km": distance_km,
        "long_jump_ratio": long_jump_count / len(edges),
        "time_cv": time_cv,
        "distance_cv": distance_cv,
    }


def _anchor_text_matches(name: str, anchor: str) -> bool:
    return bool(name) and bool(anchor) and (name == anchor or anchor in name or name in anchor)


def extract_anchor_distance_candidates(
    plan: Dict[str, Any],
    anchor: str,
    *,
    candidate_types: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    anchor = str(anchor or "").strip()
    if not anchor:
        return {"anchor": "", "anchor_hits": 0, "activity_names": []}
    allowed_types = set(candidate_types) if candidate_types is not None else set(COUNTED_ACTIVITY_TYPES)
    activity_names: List[str] = []
    anchor_hits = 0
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            activity_type = str(activity.get("type", "") or "").strip()
            if activity_type not in allowed_types:
                continue
            name = _activity_name(activity)
            if not name:
                continue
            if _anchor_text_matches(name, anchor):
                anchor_hits += 1
                continue
            activity_names.append(name)
    return {
        "anchor": anchor,
        "anchor_hits": anchor_hits,
        "activity_names": activity_names,
    }


def extract_anchor_distances(
    plan: Dict[str, Any],
    anchor: str,
    *,
    candidate_types: Optional[Set[str]] = None,
) -> List[float]:
    anchor = str(anchor or "").strip()
    if not anchor:
        return []
    allowed_types = set(candidate_types) if candidate_types is not None else None
    distances: List[float] = []
    for day in _itinerary(plan):
        activities = [activity for activity in day.get("activities", []) if isinstance(activity, dict)]
        for activity in activities:
            activity_type = str(activity.get("type", "") or "").strip()
            if allowed_types is not None and activity_type not in allowed_types:
                continue
            name = _activity_name(activity)
            if not name:
                continue
            if _anchor_text_matches(name, anchor):
                distances.append(0.0)
            for segment in _transport_segments(activity):
                start = str(segment.get("start", "") or "").strip()
                end = str(segment.get("end", "") or "").strip()
                distance = segment.get("distance")
                if not isinstance(distance, (int, float)):
                    continue
                if _anchor_text_matches(start, anchor) or _anchor_text_matches(end, anchor):
                    distances.append(float(distance))
    return distances


def _activity_text_blob(activity: Dict[str, Any], target_city: str | None) -> str:
    parts = [
        _normalized_text(activity.get("position")),
        _normalized_text(activity.get("name")),
        _normalized_text(activity.get("poiType")),
        _normalized_text(activity.get("category")),
        _normalized_text(activity.get("cuisine")),
        _normalized_text(activity.get("recommendedfood")),
        _normalized_text(activity.get("featurehoteltype")),
        _normalized_text(activity.get("room_type")),
        *_activity_theme_sources(activity, target_city),
    ]
    return " ".join(part for part in parts if part)


def extract_vibe_signals(plan: Dict[str, Any]) -> Dict[str, Any]:
    target_city = plan.get("target_city") if isinstance(plan, dict) else None
    counted_activity_count = 0
    evening_activity_count = 0
    late_night_activity_count = 0
    keyword_counts: Counter[str] = Counter()

    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            if str(activity.get("type", "") or "").strip() not in COUNTED_ACTIVITY_TYPES:
                continue
            counted_activity_count += 1
            text = _activity_text_blob(activity, target_city)
            for bucket, keywords in VIBE_KEYWORDS.items():
                if _contains_any(text, keywords):
                    keyword_counts[bucket] += 1

            start_minutes = parse_time_to_minutes(activity.get("start_time"))
            end_minutes = parse_time_to_minutes(activity.get("end_time"))
            reference_minutes = end_minutes if end_minutes is not None else start_minutes
            if reference_minutes is not None and reference_minutes >= 18 * 60:
                evening_activity_count += 1
            if reference_minutes is not None and reference_minutes >= 20 * 60:
                late_night_activity_count += 1

    gaps = extract_buffer_gaps(plan)
    end_minutes = extract_day_end_minutes(plan)
    theme_distribution = extract_theme_distribution(plan)
    walk_stats = extract_walk_transport_stats(plan)
    average_daily_activity_count = _average_numeric(extract_daily_activity_counts(plan))
    average_daily_travel_minutes = _average_numeric(extract_daily_travel_minutes(plan, exclude_intercity=True))
    average_day_end_minutes = _average_numeric(end_minutes)
    moderate_buffer_ratio = sum(1 for gap in gaps if 20 <= gap <= 90) / len(gaps) if gaps else 0.0
    late_day_ratio = sum(1 for item in end_minutes if item >= 21 * 60) / len(end_minutes) if end_minutes else 0.0
    denominator = float(counted_activity_count) if counted_activity_count > 0 else 1.0

    return {
        "counted_activity_count": counted_activity_count,
        "average_daily_activity_count": average_daily_activity_count,
        "average_daily_travel_minutes": average_daily_travel_minutes,
        "average_day_end_minutes": average_day_end_minutes,
        "evening_activity_ratio": evening_activity_count / denominator if counted_activity_count else 0.0,
        "late_night_activity_ratio": late_night_activity_count / denominator if counted_activity_count else 0.0,
        "late_day_ratio": late_day_ratio,
        "moderate_buffer_ratio": moderate_buffer_ratio,
        "keyword_counts": dict(keyword_counts),
        "keyword_ratios": {
            bucket: keyword_counts.get(bucket, 0) / denominator
            for bucket in VIBE_KEYWORDS
        },
        "theme_distribution": theme_distribution,
        "walk_total_minutes": float(walk_stats.get("total_minutes", 0.0) or 0.0),
        "walk_total_distance": float(walk_stats.get("total_distance", 0.0) or 0.0),
    }


def extract_day_end_minutes(plan: Dict[str, Any]) -> List[int]:
    end_minutes: List[int] = []
    for day in _itinerary(plan):
        activities = [activity for activity in day.get("activities", []) if isinstance(activity, dict)]
        if not activities:
            continue
        end = parse_time_to_minutes(activities[-1].get("end_time"))
        if end is not None:
            end_minutes.append(end)
    return end_minutes


def extract_buffer_gaps(plan: Dict[str, Any]) -> List[float]:
    gaps: List[float] = []
    for edge in extract_inter_activity_edges(plan):
        if edge.gap_minutes is not None:
            gaps.append(float(edge.gap_minutes))
    return gaps
