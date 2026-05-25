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


def extract_cost_breakdown(plan: Dict[str, Any]) -> Dict[str, float]:
    breakdown = {
        "food": 0.0,
        "hotel": 0.0,
        "attraction": 0.0,
        "transport": 0.0,
        "other": 0.0,
        "total": 0.0,
    }
    for day in _itinerary(plan):
        for activity in day.get("activities", []):
            if not isinstance(activity, dict):
                continue
            cost = float(activity.get("cost", 0) or 0.0)
            transport_cost = sum(float(segment.get("cost", 0) or 0.0) for segment in _transport_segments(activity))
            activity_type = str(activity.get("type", "") or "").strip()
            if activity_type in MEAL_ACTIVITY_TYPES:
                breakdown["food"] += cost
            elif activity_type == "accommodation":
                breakdown["hotel"] += cost
            elif activity_type == "attraction":
                breakdown["attraction"] += cost
            elif activity_type in TRANSPORT_ACTIVITY_TYPES:
                breakdown["transport"] += cost
            else:
                breakdown["other"] += cost
            breakdown["transport"] += transport_cost
            breakdown["total"] += cost + transport_cost
    return breakdown


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
