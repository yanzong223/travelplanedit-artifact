"""Constraint verification helpers for benchmark evaluation."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .compilers import compile_logical_constraint
from .models import LogicalConstraintObject, VerificationResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHINATRAVEL_ROOT = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
if str(CHINATRAVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_ROOT))

_MEAL_TYPES = {"breakfast", "breakfest", "lunch", "dinner"}
_INTERCITY_TYPES = {"train", "airplane"}
_TIME_WINDOWS = {
    "morning": (6 * 60, 12 * 60),
    "afternoon": (12 * 60, 18 * 60),
    "evening": (18 * 60, 22 * 60),
    "night": (22 * 60, 24 * 60),
}


@lru_cache(maxsize=1)
def _concept_funcs() -> dict[str, Callable[..., Any]]:
    from chinatravel.symbol_verification.concept_func import (
        accommodation_type,
        attraction_type,
        restaurant_type,
    )

    return {
        "accommodation_type": accommodation_type,
        "attraction_type": attraction_type,
        "restaurant_type": restaurant_type,
    }


@lru_cache(maxsize=1)
def _load_evaluator() -> Any:
    try:
        from chinatravel.symbol_verification.hard_constraint import evaluate_constraints_py
    except Exception:
        return None
    return evaluate_constraints_py


@lru_cache(maxsize=1)
def _world_env() -> Any:
    from chinatravel.environment.world_env import WorldEnv

    return WorldEnv()


def _itinerary(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return plan.get("itinerary", []) if isinstance(plan, dict) else []


def _activities(plan: dict[str, Any]) -> list[dict[str, Any]]:
    activities: list[dict[str, Any]] = []
    for day in _itinerary(plan):
        activities.extend(day.get("activities", []))
    return activities


def _transport_mode(transport: dict[str, Any]) -> str:
    return str(transport.get("mode", transport.get("type", "")))


def _compare(operator: str, left: Any, right: Any) -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<=":
        return left <= right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == ">":
        return left > right
    return False


def _clock_to_minutes(value: str) -> int | None:
    text = str(value or "")
    if ":" not in text:
        return None
    try:
        hour, minute = text.split(":")[:2]
        return int(hour) * 60 + int(minute)
    except ValueError:
        return None


def _normalize_clock_window_mode(constraint: LogicalConstraintObject) -> str | None:
    mode = str(constraint.params.get("mode", "") or "").strip().lower()
    if mode in {"after", "before", "between"}:
        return mode
    operator = str(constraint.operator or "").strip()
    if operator in {">=", ">"}:
        return "after"
    if operator in {"<=", "<"}:
        return "before"
    if operator in {"between", "in"} and isinstance(constraint.value, (list, tuple)) and len(constraint.value) == 2:
        return "between"
    return None


def _compare_clock(operator: str, left: str, right: str) -> bool:
    left_min = _clock_to_minutes(left)
    right_min = _clock_to_minutes(right)
    if left_min is None or right_min is None:
        return False
    return _compare(operator, left_min, right_min)


def _compare_set(operator: str, actual: set[str], expected: set[str]) -> bool:
    if operator == "equals_set":
        return actual == expected
    if operator == "contains_all":
        return expected <= actual
    return False


def _activity_duration_minutes(activity: dict[str, Any]) -> int | None:
    start = str(activity.get("start_time", ""))
    end = str(activity.get("end_time", ""))
    if ":" not in start or ":" not in end:
        return None
    sh, sm = start.split(":")[:2]
    eh, em = end.split(":")[:2]
    return (int(eh) * 60 + int(em)) - (int(sh) * 60 + int(sm))


def _iter_day_activities(plan: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    output: list[tuple[int, list[dict[str, Any]]]] = []
    for index, day in enumerate(_itinerary(plan), start=1):
        output.append((int(day.get("day", index) or index), day.get("activities", [])))
    return output


def _activity_matches_types(activity: dict[str, Any], activity_types: list[str]) -> bool:
    act_type = str(activity.get("type", ""))
    normalized = set(activity_types)
    if "restaurant" in normalized:
        normalized.remove("restaurant")
        normalized.update(_MEAL_TYPES)
    if "meal" in normalized:
        normalized.remove("meal")
        normalized.update(_MEAL_TYPES)
    return act_type in normalized


def _transport_distance_sum(activity: dict[str, Any]) -> float:
    transports = activity.get("transports", [])
    if not isinstance(transports, list):
        return 0.0
    return sum(float(item.get("distance", 0) or 0) for item in transports if isinstance(item, dict))


def _transport_duration_minutes(activity: dict[str, Any]) -> int | None:
    transports = activity.get("transports", [])
    if not isinstance(transports, list) or not transports:
        return None
    total = 0
    found = False
    for item in transports:
        if not isinstance(item, dict):
            continue
        start = _clock_to_minutes(str(item.get("start_time", "")))
        end = _clock_to_minutes(str(item.get("end_time", "")))
        if start is None or end is None:
            continue
        total += end - start
        found = True
    return total if found else None


def _lookup_distance_km(city: Any, start: str, end: str) -> float | None:
    if not city or not start or not end or start == end:
        return 0.0 if start and end and start == end else None
    try:
        route = _world_env().transportation.goto(
            city=city,
            start=start,
            end=end,
            start_time="00:00",
            transport_type="walk",
            verbose=False,
        )
    except Exception:
        return None
    if not isinstance(route, list) or not route:
        return None
    try:
        return float(route[0].get("distance", 0) or 0)
    except (TypeError, ValueError):
        return None


def _anchor_distance_for_activity(plan: dict[str, Any], anchor_name: str, activity: dict[str, Any]) -> float | None:
    position = str(activity.get("position", "") or "")
    if not position:
        return None
    if position == anchor_name:
        return 0.0
    looked_up = _lookup_distance_km(plan.get("target_city"), anchor_name, position)
    if looked_up is not None:
        return looked_up
    return _transport_distance_sum(activity)


def _transport_endpoint_matches(transport: dict[str, Any], anchor_name: str) -> bool:
    for key in ("start", "from", "origin", "end", "to", "destination"):
        if str(transport.get(key, "")) == anchor_name:
            return True
    return False


def _resolve_anchor_distance(
    plan: dict[str, Any],
    day_activities: list[dict[str, Any]],
    activity_index: int,
    anchor_name: str,
) -> float | None:
    activity = day_activities[activity_index]
    position = str(activity.get("position", ""))
    if position == anchor_name:
        return 0.0

    transports = activity.get("transports", [])
    if isinstance(transports, list):
        for transport in transports:
            if not isinstance(transport, dict) or not _transport_endpoint_matches(transport, anchor_name):
                continue
            try:
                return float(transport.get("distance", 0) or 0)
            except (TypeError, ValueError):
                continue

    if activity_index > 0 and str(day_activities[activity_index - 1].get("position", "")) == anchor_name:
        return _transport_distance_sum(activity)

    if activity_index + 1 < len(day_activities) and str(day_activities[activity_index + 1].get("position", "")) == anchor_name:
        return _transport_distance_sum(day_activities[activity_index + 1])

    return _anchor_distance_for_activity(plan, anchor_name, activity)


def _activity_counts_for_anchor_radius(activity: dict[str, Any]) -> bool:
    activity_type = str(activity.get("type", ""))
    if activity_type in _INTERCITY_TYPES:
        return False
    if str(activity.get("position", "")):
        return True
    return activity_type in (_MEAL_TYPES | {"accommodation", "attraction"})


def _matches_city_endpoint(value: Any, city: str) -> bool:
    return bool(city) and city in str(value or "")


def _choose_intercity_activity(
    plan: dict[str, Any],
    intercity: list[dict[str, Any]],
    leg: str,
    field: str,
) -> dict[str, Any] | None:
    if not intercity:
        return None

    start_city = str(plan.get("start_city", "") or "")
    target_city = str(plan.get("target_city", "") or "")

    if leg in {"return", "back"}:
        if field == "start_time":
            filtered = [activity for activity in intercity if _matches_city_endpoint(activity.get("start"), target_city)]
        else:
            filtered = [activity for activity in intercity if _matches_city_endpoint(activity.get("end"), start_city)]
        return filtered[-1] if filtered else intercity[-1]

    if field == "start_time":
        filtered = [activity for activity in intercity if _matches_city_endpoint(activity.get("start"), start_city)]
    else:
        filtered = [activity for activity in intercity if _matches_city_endpoint(activity.get("end"), target_city)]
    return filtered[0] if filtered else intercity[0]


def _total_cost(plan: dict[str, Any]) -> float:
    total = 0.0
    for activity in _activities(plan):
        total += float(activity.get("cost", 0) or 0)
        for transport in activity.get("transports", []):
            total += float(transport.get("cost", 0) or 0)
    return total


def _matched_activities(plan: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    activities = _activities(plan)
    poi_name = target.get("poi_name")
    activity_type = target.get("activity_type")
    if not poi_name and not activity_type:
        return activities
    matched: list[dict[str, Any]] = []
    for activity in activities:
        if poi_name and activity.get("position") != poi_name:
            continue
        if activity_type:
            if activity_type == "meal":
                if activity.get("type") not in _MEAL_TYPES:
                    continue
            elif activity.get("type") != activity_type:
                continue
        matched.append(activity)
    return matched


def _locate_poi_occurrences(plan: dict[str, Any], poi_name: str) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for day, day_activities in _iter_day_activities(plan):
        for index, activity in enumerate(day_activities):
            if activity.get("position") == poi_name:
                occurrences.append({"day": day, "index": index, "activity": activity})
    return occurrences


def _days_with_poi(plan: dict[str, Any], poi_name: str) -> list[int]:
    return [int(item["day"]) for item in _locate_poi_occurrences(plan, poi_name)]


def _meal_activities_on_days(plan: dict[str, Any], days: set[int]) -> list[dict[str, Any]]:
    meals: list[dict[str, Any]] = []
    for day, day_activities in _iter_day_activities(plan):
        if day not in days:
            continue
        meals.extend(
            activity
            for activity in day_activities
            if str(activity.get("type", "")) in _MEAL_TYPES
        )
    return meals


def _constraint_max_distance_km(constraint: LogicalConstraintObject, default: float = 3.0) -> float:
    for key in ("max_distance_km", "distance_km", "nearby_distance_km"):
        value = constraint.params.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return default


def _same_day_nearby_meal_candidates(
    plan: dict[str, Any],
    anchor_poi: str,
    *,
    max_distance_km: float,
    selected_meal_name: str = "",
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    selected = str(selected_meal_name or "").strip()
    city = plan.get("target_city")
    for occurrence in _locate_poi_occurrences(plan, anchor_poi):
        day = int(occurrence["day"])
        anchor_cost = _activity_cost_like(occurrence["activity"], prefer_price=True)
        for meal in _meal_activities_on_days(plan, {day}):
            meal_name = str(meal.get("position", "") or "")
            if selected and meal_name != selected:
                continue
            distance = _lookup_distance_km(city, anchor_poi, meal_name)
            if distance is None:
                distance = _anchor_distance_for_activity(plan, anchor_poi, meal)
            if distance is None:
                continue
            meal_cost = _activity_cost_like(meal)
            candidates.append(
                {
                    "day": day,
                    "anchor_poi": anchor_poi,
                    "meal_name": meal_name,
                    "distance_km": float(distance),
                    "within_distance": float(distance) <= float(max_distance_km) + 1e-6,
                    "anchor_cost": anchor_cost,
                    "meal_cost": meal_cost,
                    "bundle_cost": anchor_cost + meal_cost,
                }
            )
    return candidates


def _activity_cost_like(activity: dict[str, Any], *, prefer_price: bool = False) -> float:
    fields = ("price", "cost") if prefer_price else ("cost", "price")
    for field in fields:
        try:
            return float(activity.get(field, 0) or 0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _intervals_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_start = _clock_to_minutes(str(first.get("start_time", "")))
    first_end = _clock_to_minutes(str(first.get("end_time", "")))
    second_start = _clock_to_minutes(str(second.get("start_time", "")))
    second_end = _clock_to_minutes(str(second.get("end_time", "")))
    if None in {first_start, first_end, second_start, second_end}:
        return True
    return not (first_end <= second_start or second_end <= first_start)


def _direct_verify(plan: dict[str, Any], constraint: LogicalConstraintObject) -> tuple[bool, bool, str, dict[str, Any]]:
    operator = constraint.operator or "=="
    value = constraint.value
    activities = _activities(plan)

    if constraint.type == "day_count":
        actual = len(_itinerary(plan))
        return True, _compare(operator, actual, int(value)), "ok", {"actual": actual}

    if constraint.type == "people_count":
        actual = int(plan.get("people_number", 0) or 0)
        return True, _compare(operator, actual, int(value)), "ok", {"actual": actual}

    if constraint.type == "city_split_requirement":
        expected = {str(item) for item in value or [] if str(item)}
        supported = [str(item) for item in constraint.params.get("supported_cities", []) if str(item)]
        actual = {str(plan.get("target_city", "") or "").strip()}
        for activity in activities:
            if activity.get("type") not in _INTERCITY_TYPES:
                continue
            for endpoint in (activity.get("start"), activity.get("end")):
                endpoint_text = str(endpoint or "")
                for city in supported:
                    if city and city in endpoint_text:
                        actual.add(city)
        actual.discard("")
        return True, expected <= actual, "ok", {"actual": sorted(actual), "expected": sorted(expected)}

    if constraint.type == "budget_total":
        actual = _total_cost(plan)
        return True, _compare(operator, actual, float(value)), "ok", {"actual": actual}

    if constraint.type == "ticket_budget_total":
        actual = sum(
            _activity_cost_like(activity, prefer_price=True)
            for activity in activities
            if activity.get("type") == "attraction"
        )
        return True, _compare(operator, actual, float(value)), "ok", {"actual": actual}

    if constraint.type == "required_intercity_transport_type":
        actual = {activity.get("type", "") for activity in activities if activity.get("type") in _INTERCITY_TYPES}
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "required_innercity_transport_type":
        actual = {
            _transport_mode(transport)
            for activity in activities
            for transport in activity.get("transports", [])
            if _transport_mode(transport)
        }
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "required_attraction_name":
        actual = {activity.get("position", "") for activity in activities if activity.get("type") == "attraction"}
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "forbidden_attraction_name":
        actual = {activity.get("position", "") for activity in activities if activity.get("type") == "attraction"}
        forbidden = {str(item) for item in value or []}
        passed = not bool(actual & forbidden)
        return True, passed, "ok", {"actual": sorted(actual), "forbidden": sorted(forbidden)}

    if constraint.type == "required_restaurant_name":
        actual = {activity.get("position", "") for activity in activities if activity.get("type") in _MEAL_TYPES}
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "required_hotel_name":
        actual = {activity.get("position", "") for activity in activities if activity.get("type") == "accommodation"}
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "required_attraction_type":
        funcs = _concept_funcs()
        city = plan.get("target_city")
        actual = {
            str(funcs["attraction_type"](activity, city))
            for activity in activities
            if activity.get("type") == "attraction" and activity.get("position")
        }
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "required_restaurant_type":
        funcs = _concept_funcs()
        city = plan.get("target_city")
        actual = {
            str(funcs["restaurant_type"](activity, city))
            for activity in activities
            if activity.get("type") in _MEAL_TYPES and activity.get("position")
        }
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "required_hotel_feature":
        funcs = _concept_funcs()
        city = plan.get("target_city")
        actual = {
            str(funcs["accommodation_type"](activity, city))
            for activity in activities
            if activity.get("type") == "accommodation" and activity.get("position")
        }
        expected = {str(item) for item in value or []}
        return True, _compare_set(operator, actual, expected), "ok", {"actual": sorted(actual)}

    if constraint.type == "semantic_type_requirement":
        funcs = _concept_funcs()
        city = plan.get("target_city")
        semantic_types = [str(item) for item in value] if isinstance(value, (list, tuple, set)) else [str(value)]
        day_value = constraint.params.get("day", constraint.target.get("day"))
        scoped_activities = []
        for day, day_activities in _iter_day_activities(plan):
            if day_value is not None and day != int(day_value):
                continue
            scoped_activities.extend(day_activities)
        actual = [
            str(funcs["attraction_type"](activity, city))
            for activity in scoped_activities
            if activity.get("type") == "attraction"
            and activity.get("position")
        ]
        min_count = int(constraint.params.get("min_count", 1) or 1)
        matched = sum(1 for item in actual if item in semantic_types)
        strict_majority = bool(constraint.params.get("strict_majority"))
        passed = matched >= min_count
        if strict_majority:
            passed = passed and bool(actual) and matched * 2 > len(actual)
        return True, passed, "ok", {"actual": actual, "matched": matched, "required_type": semantic_types, "day": day_value}

    if constraint.type == "required_room_type":
        matched = [activity for activity in activities if activity.get("type") == "accommodation"]
        actual = [int(activity.get("room_type", 0) or 0) for activity in matched]
        passed = bool(matched) and all(item == int(value) for item in actual)
        return True, passed, "ok", {"actual": actual}

    if constraint.type == "required_room_count":
        matched = [activity for activity in activities if activity.get("type") == "accommodation"]
        actual = [int(activity.get("rooms", activity.get("room_count", 0)) or 0) for activity in matched]
        passed = bool(matched) and all(item == int(value) for item in actual)
        return True, passed, "ok", {"actual": actual}

    if constraint.type == "ticket_count_match":
        include_metro = bool(constraint.params.get("include_metro", True))
        actual = []
        for activity in activities:
            if activity.get("type") in {"attraction", "train", "airplane"}:
                actual.append(int(activity.get("tickets", 0) or 0))
            if include_metro:
                for transport in activity.get("transports", []):
                    if _transport_mode(transport) == "metro":
                        actual.append(int(transport.get("tickets", 0) or 0))
        passed = bool(actual) and all(item == int(value) for item in actual)
        return True, passed, "ok", {"actual": actual}

    if constraint.type == "taxi_car_count_match":
        actual = [
            int(transport.get("cars", 0) or 0)
            for activity in activities
            for transport in activity.get("transports", [])
            if _transport_mode(transport) == "taxi"
        ]
        # Match the original hard-truth semantics: if there is no taxi leg, the
        # universal quantification is vacuously true instead of failing.
        passed = all(item == int(value) for item in actual)
        return True, passed, "ok", {"actual": actual}

    if constraint.type == "poi_logic":
        actual = {
            activity.get("position", "")
            for activity in activities
            if activity.get("type") in _MEAL_TYPES | {"accommodation", "attraction"}
        }
        expected = {str(item) for item in value or []}
        relation = str(constraint.params.get("relation", "conjunction"))
        if relation == "conjunction":
            passed = expected <= actual
        elif relation == "disjunction":
            passed = bool(expected & actual)
        elif relation == "negation":
            passed = not bool(expected & actual)
        else:
            return False, False, "unsupported_poi_relation", {"relation": relation}
        return True, passed, "ok", {"actual": sorted(actual), "relation": relation}

    if constraint.type == "activity_duration_limit":
        matched = _matched_activities(plan, constraint.target)
        durations = [duration for activity in matched for duration in [_activity_duration_minutes(activity)] if duration is not None]
        if not durations:
            return True, False, "no_matching_activities", {"target": constraint.target}
        passed = all(_compare(operator, duration, int(value)) for duration in durations)
        return True, passed, "ok", {"actual": durations}

    if constraint.type == "activity_budget_limit":
        matched = _matched_activities(plan, constraint.target)
        if not matched:
            return True, False, "no_matching_activities", {"target": constraint.target}
        metric = str(constraint.params.get("metric", "activity_cost"))
        people = max(int(plan.get("people_number", 1) or 1), 1)
        days = len(_itinerary(plan))
        if metric == "avg_cost_per_person_per_night":
            actual = sum(float(item.get("cost", 0) or 0) for item in matched) / max(people * max(days - 1, 1), 1)
        elif metric == "avg_cost_per_meal_per_person":
            actual = sum(float(item.get("cost", 0) or 0) for item in matched) / max(len(matched) * people, 1)
        else:
            actual = sum(float(item.get("cost", 0) or 0) for item in matched)
        return True, _compare(operator, actual, float(value)), "ok", {"actual": actual, "metric": metric}

    if constraint.type == "day_end_time_limit":
        target_day = constraint.target.get("day")
        target_days = [int(target_day)] if target_day not in (None, "all") else [day for day, _ in _iter_day_activities(plan)]
        actual: dict[int, str] = {}
        for day, day_activities in _iter_day_activities(plan):
            if day not in target_days:
                continue
            end_times = [
                str(activity.get("end_time", ""))
                for activity in day_activities
                if isinstance(activity.get("end_time"), str) and activity.get("end_time")
            ]
            if not end_times:
                return True, False, "no_day_activities", {"target_day": target_day}
            latest = max(end_times, key=lambda item: _clock_to_minutes(item) or -1)
            actual[day] = latest
        passed = bool(actual) and all(_compare_clock(operator, item, str(value)) for item in actual.values())
        return True, passed, "ok", {"actual": actual}

    if constraint.type == "poi_time_window":
        poi_name = str(constraint.target.get("poi_name", ""))
        period = str(value)
        window = _TIME_WINDOWS.get(period)
        if not poi_name or window is None:
            return False, False, "unsupported_poi_time_window", {"constraint": constraint.to_dict()}
        target_day = constraint.params.get("day")
        matched: list[dict[str, Any]] = []
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            for activity in day_activities:
                if activity.get("position") == poi_name:
                    matched.append(activity)
        if not matched:
            return True, False, "no_matching_activities", {"target": constraint.target, "day": target_day}
        low, high = window
        actual = []
        passed = True
        for activity in matched:
            start_min = _clock_to_minutes(str(activity.get("start_time", "")))
            end_min = _clock_to_minutes(str(activity.get("end_time", "")))
            actual.append({"start": activity.get("start_time"), "end": activity.get("end_time")})
            if start_min is None or end_min is None or not (low <= start_min and end_min <= high):
                passed = False
        return True, passed, "ok", {"actual": actual, "window": period}

    if constraint.type == "poi_clock_time_window":
        poi_name = str(constraint.target.get("poi_name", ""))
        mode = _normalize_clock_window_mode(constraint)
        if not poi_name or mode not in {"after", "before", "between"}:
            return False, False, "unsupported_poi_clock_time_window", {"constraint": constraint.to_dict()}
        target_day = constraint.params.get("day", constraint.target.get("day"))
        matched: list[dict[str, Any]] = []
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            for activity in day_activities:
                if activity.get("position") == poi_name:
                    matched.append(activity)
        if not matched:
            return True, False, "no_matching_activities", {"target": constraint.target, "day": target_day}

        if mode == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False, False, "unsupported_poi_clock_time_window", {"constraint": constraint.to_dict()}
            low = _clock_to_minutes(str(value[0]))
            high = _clock_to_minutes(str(value[1]))
            if low is None or high is None:
                return False, False, "invalid_clock_window", {"value": value}
        else:
            threshold = _clock_to_minutes(str(value))
            if threshold is None:
                return False, False, "invalid_clock_window", {"value": value}

        actual: list[dict[str, Any]] = []
        passed = True
        for activity in matched:
            start = str(activity.get("start_time", ""))
            end = str(activity.get("end_time", ""))
            start_min = _clock_to_minutes(start)
            end_min = _clock_to_minutes(end)
            actual.append({"start": start, "end": end, "mode": mode})
            if start_min is None or end_min is None:
                passed = False
                continue
            if mode == "after":
                passed = passed and start_min >= threshold
            elif mode == "before":
                passed = passed and end_min <= threshold
            else:
                passed = passed and low <= start_min and end_min <= high
        return True, passed, "ok", {"actual": actual, "mode": mode}

    if constraint.type == "daily_poi_cap":
        count_types = [str(item) for item in constraint.params.get("count_types", ["attraction"])]
        target_day = constraint.target.get("day")
        actual: dict[int, int] = {}
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            actual[day] = sum(1 for activity in day_activities if _activity_matches_types(activity, count_types))
        passed = bool(actual) and all(_compare(operator, count, int(value)) for count in actual.values())
        return True, passed, "ok", {"actual": actual, "count_types": count_types}

    if constraint.type == "adjacent_travel_time_cap":
        activity_types = [str(item) for item in constraint.target.get("activity_types", constraint.params.get("activity_types", ["attraction"]))]
        target_day = constraint.params.get("day", "all")
        actual: list[dict[str, Any]] = []
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            filtered = [activity for activity in day_activities if _activity_matches_types(activity, activity_types)]
            for index in range(1, len(filtered)):
                prev_activity = filtered[index - 1]
                curr_activity = filtered[index]
                duration = _transport_duration_minutes(curr_activity)
                transports = curr_activity.get("transports", [])
                if duration is None:
                    if isinstance(transports, list) and not transports:
                        duration = 0
                    else:
                        return True, False, "missing_inbound_transport_duration", {"day": day}
                actual.append(
                    {
                        "day": day,
                        "from": prev_activity.get("position", prev_activity.get("type")),
                        "to": curr_activity.get("position", curr_activity.get("type")),
                        "minutes": duration,
                    }
                )
        if not actual:
            return True, False, "no_adjacent_activity_pairs", {"activity_types": activity_types}
        passed = all(_compare(operator, int(item["minutes"]), int(value)) for item in actual)
        return True, passed, "ok", {"actual": actual, "activity_types": activity_types}

    if constraint.type == "adjacent_travel_distance_cap":
        activity_types = [str(item) for item in constraint.target.get("activity_types", constraint.params.get("activity_types", ["attraction"]))]
        target_day = constraint.params.get("day", "all")
        actual: list[dict[str, Any]] = []
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            filtered = [activity for activity in day_activities if _activity_matches_types(activity, activity_types)]
            for index in range(1, len(filtered)):
                curr_activity = filtered[index]
                inbound = _transport_distance_sum(curr_activity)
                actual.append(
                    {
                        "day": day,
                        "to": curr_activity.get("position", curr_activity.get("type")),
                        "distance": inbound,
                    }
                )
        if not actual:
            return True, False, "no_adjacent_activity_pairs", {"activity_types": activity_types}
        passed = all(_compare(operator, float(item["distance"]), float(value)) for item in actual)
        return True, passed, "ok", {"actual": actual, "activity_types": activity_types}

    if constraint.type == "anchor_neighbor_commute_distance_cap":
        anchor_name = str(constraint.target.get("anchor_name", ""))
        if not anchor_name:
            return False, False, "unsupported_anchor_neighbor_commute_distance_cap", {"constraint": constraint.to_dict()}
        target_day = constraint.params.get("day", constraint.target.get("day"))
        actual: list[dict[str, Any]] = []
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            if not any(activity.get("position") == anchor_name for activity in day_activities):
                continue
            for index, activity in enumerate(day_activities):
                if not _activity_counts_for_anchor_radius(activity):
                    continue
                distance = _resolve_anchor_distance(plan, day_activities, index, anchor_name)
                if distance is None:
                    return True, False, "missing_anchor_distance", {
                        "anchor_name": anchor_name,
                        "day": day,
                        "activity": activity.get("position", activity.get("type")),
                    }
                actual.append(
                    {
                        "day": day,
                        "activity": activity.get("position", activity.get("type")),
                        "distance": distance,
                    }
                )
        if not actual:
            return True, False, "no_matching_anchor_activity", {"anchor_name": anchor_name}
        passed = all(_compare(operator, float(item["distance"]), float(value)) for item in actual)
        return True, passed, "ok", {"actual": actual, "anchor_name": anchor_name}

    if constraint.type == "poi_day_binding":
        poi_name = str(constraint.target.get("poi_name", ""))
        target_day = int(value)
        if not poi_name:
            return False, False, "unsupported_poi_day_binding", {"constraint": constraint.to_dict()}
        actual_days: list[int] = []
        for day, day_activities in _iter_day_activities(plan):
            for activity in day_activities:
                if activity.get("position") == poi_name:
                    actual_days.append(day)
        if not actual_days:
            return True, False, "no_matching_activities", {"target": constraint.target}
        passed = target_day in actual_days
        return True, passed, "ok", {"actual_days": actual_days, "target_day": target_day}

    if constraint.type == "poi_order_constraint":
        first_poi = str(constraint.target.get("first_poi", ""))
        second_poi = str(constraint.target.get("second_poi", ""))
        if not first_poi or not second_poi:
            return False, False, "unsupported_poi_order_constraint", {"constraint": constraint.to_dict()}
        first_occurrences = _locate_poi_occurrences(plan, first_poi)
        second_occurrences = _locate_poi_occurrences(plan, second_poi)
        if not first_occurrences or not second_occurrences:
            return True, False, "missing_ordered_poi", {"first_poi": first_poi, "second_poi": second_poi}
        first_ref = min(first_occurrences, key=lambda item: (item["day"], item["index"]))
        second_ref = min(second_occurrences, key=lambda item: (item["day"], item["index"]))
        passed = (first_ref["day"], first_ref["index"]) < (second_ref["day"], second_ref["index"])
        return True, passed, "ok", {"first": first_ref, "second": second_ref}

    if constraint.type == "pair_same_day_no_overlap":
        first_poi = str(constraint.target.get("first_poi", ""))
        second_poi = str(constraint.target.get("second_poi", ""))
        target_day = int(value)
        if not first_poi or not second_poi:
            return False, False, "unsupported_pair_same_day_no_overlap", {"constraint": constraint.to_dict()}
        first_occurrences = [item for item in _locate_poi_occurrences(plan, first_poi) if item["day"] == target_day]
        second_occurrences = [item for item in _locate_poi_occurrences(plan, second_poi) if item["day"] == target_day]
        if not first_occurrences or not second_occurrences:
            return True, False, "missing_poi_on_target_day", {"target_day": target_day, "first_poi": first_poi, "second_poi": second_poi}
        first_activity = first_occurrences[0]["activity"]
        second_activity = second_occurrences[0]["activity"]
        passed = not _intervals_overlap(first_activity, second_activity)
        return True, passed, "ok", {"target_day": target_day, "first_poi": first_poi, "second_poi": second_poi}

    if constraint.type == "pair_time_window_no_overlap":
        first_poi = str(constraint.target.get("first_poi", ""))
        second_poi = str(constraint.target.get("second_poi", ""))
        period = str(value)
        window = _TIME_WINDOWS.get(period)
        if not first_poi or not second_poi or window is None:
            return False, False, "unsupported_pair_time_window_no_overlap", {"constraint": constraint.to_dict()}
        first_occurrences = _locate_poi_occurrences(plan, first_poi)
        second_occurrences = _locate_poi_occurrences(plan, second_poi)
        if not first_occurrences or not second_occurrences:
            return True, False, "missing_pair_poi", {"first_poi": first_poi, "second_poi": second_poi}
        low, high = window
        for first in first_occurrences:
            for second in second_occurrences:
                if first["day"] != second["day"]:
                    continue
                first_start = _clock_to_minutes(str(first["activity"].get("start_time", "")))
                first_end = _clock_to_minutes(str(first["activity"].get("end_time", "")))
                second_start = _clock_to_minutes(str(second["activity"].get("start_time", "")))
                second_end = _clock_to_minutes(str(second["activity"].get("end_time", "")))
                if None in {first_start, first_end, second_start, second_end}:
                    continue
                in_window = low <= first_start and first_end <= high and low <= second_start and second_end <= high
                if in_window and not _intervals_overlap(first["activity"], second["activity"]):
                    return True, True, "ok", {"day": first["day"], "window": period}
        return True, False, "pair_not_satisfied_in_window", {"first_poi": first_poi, "second_poi": second_poi, "window": period}

    if constraint.type == "pairwise_transport_mode_distance_cap":
        first_poi = str(constraint.target.get("first_poi", ""))
        second_poi = str(constraint.target.get("second_poi", ""))
        expected_mode = str(constraint.target.get("mode", ""))
        if not first_poi or not second_poi or not expected_mode:
            return False, False, "unsupported_pairwise_transport_mode_distance_cap", {"constraint": constraint.to_dict()}
        matches: list[dict[str, Any]] = []
        for day, day_activities in _iter_day_activities(plan):
            for index in range(1, len(day_activities)):
                prev_activity = day_activities[index - 1]
                curr_activity = day_activities[index]
                names = {str(prev_activity.get("position", "")), str(curr_activity.get("position", ""))}
                if {first_poi, second_poi} != names:
                    continue
                transports = curr_activity.get("transports", [])
                modes = {_transport_mode(item) for item in transports if isinstance(item, dict)}
                distance = _transport_distance_sum(curr_activity)
                matches.append(
                    {
                        "day": day,
                        "from": prev_activity.get("position", prev_activity.get("type")),
                        "to": curr_activity.get("position", curr_activity.get("type")),
                        "modes": sorted(modes),
                        "distance": distance,
                        "passed": expected_mode in modes and _compare(operator, float(distance), float(value)),
                    }
                )
        if not matches:
            return True, False, "pairwise_transition_not_found", {"first_poi": first_poi, "second_poi": second_poi}
        passed = any(item["passed"] for item in matches)
        return True, passed, "ok", {"actual": matches, "first_poi": first_poi, "second_poi": second_poi}

    if constraint.type == "poi_inbound_travel_time_cap":
        poi_name = str(constraint.target.get("poi_name", ""))
        if not poi_name:
            return False, False, "unsupported_poi_inbound_travel_time_cap", {"constraint": constraint.to_dict()}
        actual: list[int] = []
        for occurrence in _locate_poi_occurrences(plan, poi_name):
            duration = _transport_duration_minutes(occurrence["activity"])
            if duration is not None:
                actual.append(duration)
        if not actual:
            return True, False, "missing_inbound_transport_duration", {"poi_name": poi_name}
        passed = all(_compare(operator, int(item), int(value)) for item in actual)
        return True, passed, "ok", {"actual": actual, "poi_name": poi_name}

    if constraint.type == "nearby_meal_requirement":
        anchor_poi = str(constraint.target.get("anchor_poi", "") or constraint.target.get("poi_name", ""))
        if not anchor_poi:
            return False, False, "unsupported_nearby_meal_requirement", {"constraint": constraint.to_dict()}
        anchor_days = set(_days_with_poi(plan, anchor_poi))
        if not anchor_days:
            return True, False, "anchor_poi_missing", {"anchor_poi": anchor_poi}
        max_distance = _constraint_max_distance_km(constraint)
        selected_meal = str(constraint.params.get("selected_meal_name") or "").strip()
        candidates = _same_day_nearby_meal_candidates(
            plan,
            anchor_poi,
            max_distance_km=max_distance,
            selected_meal_name=selected_meal,
        )
        passed = any(item["within_distance"] for item in candidates)
        return True, passed, "ok", {
            "anchor_poi": anchor_poi,
            "anchor_days": sorted(anchor_days),
            "max_distance_km": max_distance,
            "selected_meal_name": selected_meal or None,
            "candidate_meals": candidates,
        }

    if constraint.type == "anchor_bundle_budget_limit":
        anchor_poi = str(constraint.target.get("anchor_poi", "") or constraint.target.get("poi_name", ""))
        if not anchor_poi:
            return False, False, "unsupported_anchor_bundle_budget_limit", {"constraint": constraint.to_dict()}
        anchor_occurrences = _locate_poi_occurrences(plan, anchor_poi)
        if not anchor_occurrences:
            return True, False, "anchor_poi_missing", {"anchor_poi": anchor_poi}
        anchor_days = {int(item["day"]) for item in anchor_occurrences}
        max_distance = _constraint_max_distance_km(constraint)
        selected_meal = str(constraint.params.get("selected_meal_name") or "").strip()
        candidates = _same_day_nearby_meal_candidates(
            plan,
            anchor_poi,
            max_distance_km=max_distance,
            selected_meal_name=selected_meal,
        )
        nearby_candidates = [item for item in candidates if item["within_distance"]]
        if not nearby_candidates:
            return True, False, "bundle_meal_missing", {
                "anchor_poi": anchor_poi,
                "anchor_days": sorted(anchor_days),
                "max_distance_km": max_distance,
                "selected_meal_name": selected_meal or None,
                "candidate_meals": candidates,
            }

        passed = any(_compare(operator, item["bundle_cost"], float(value)) for item in nearby_candidates)
        return True, passed, "ok", {
            "anchor_poi": anchor_poi,
            "max_distance_km": max_distance,
            "selected_meal_name": selected_meal or None,
            "candidate_meals": candidates,
            "actual": {str(item["day"]): item["bundle_cost"] for item in nearby_candidates},
        }

    if constraint.type == "ticket_price_cap_or_substitute":
        poi_name = str(constraint.target.get("poi_name", ""))
        activity_type = str(constraint.target.get("activity_type", ""))
        target_eligible = bool(constraint.params.get("target_eligible", False))
        candidate_names = {
            str(item)
            for item in constraint.params.get("candidate_names", [])
            if str(item)
        }
        target_day = constraint.params.get("day")
        matched_names: set[str] = set()
        for day, day_activities in _iter_day_activities(plan):
            if target_day not in (None, "all") and day != int(target_day):
                continue
            for activity in day_activities:
                if activity_type == "meal":
                    if activity.get("type") not in _MEAL_TYPES:
                        continue
                elif activity_type and activity.get("type") != activity_type:
                    continue
                position = str(activity.get("position", ""))
                if position:
                    matched_names.add(position)
        if target_eligible:
            passed = poi_name in matched_names
            return True, passed, "ok", {"actual_names": sorted(matched_names), "target_eligible": True}
        if not candidate_names:
            return False, False, "missing_ticket_substitute_candidates", {"constraint": constraint.to_dict()}
        passed = bool(candidate_names & matched_names)
        return True, passed, "ok", {
            "actual_names": sorted(matched_names),
            "candidate_names": sorted(candidate_names),
            "target_eligible": False,
        }

    if constraint.type == "transport_time_window":
        intercity = [activity for activity in activities if activity.get("type") in _INTERCITY_TYPES]
        if not intercity:
            return True, False, "no_intercity_transport", {}
        leg = str(constraint.params.get("leg", "outbound"))
        field = str(constraint.params.get("field", "start_time"))
        chosen = _choose_intercity_activity(plan, intercity, leg, field)
        if chosen is None:
            return True, False, "no_intercity_transport", {}
        actual = str(chosen.get(field, ""))
        if not actual:
            return True, False, "missing_transport_time", {"field": field, "leg": leg}
        return True, _compare_clock(operator, actual, str(value)), "ok", {"actual": actual, "field": field, "leg": leg}

    return False, False, "unsupported_constraint_type", {"constraint": constraint.to_dict()}


def verify_constraint(
    plan: dict[str, Any],
    constraint: LogicalConstraintObject,
) -> VerificationResult:
    code = compile_logical_constraint(constraint)
    supported, passed, reason, details = _direct_verify(plan, constraint)
    if supported:
        return VerificationResult(
            constraint_id=constraint.id,
            constraint_type=constraint.type,
            supported=True,
            passed=passed,
            reason="ok" if passed else "constraint_failed",
            details={**details, "constraint": constraint.to_dict()},
            compiled_logic_py=code,
        )

    if code:
        try:
            evaluate_constraints_py = _load_evaluator()
            if evaluate_constraints_py is None:
                raise ModuleNotFoundError("chinatravel symbolic evaluator unavailable")
            result = evaluate_constraints_py([code], plan, verbose=False)
            passed = bool(result[0]) if result else False
            return VerificationResult(
                constraint_id=constraint.id,
                constraint_type=constraint.type,
                supported=True,
                passed=passed,
                reason="ok" if passed else "constraint_failed",
                details={"constraint": constraint.to_dict(), "verification_mode": "symbolic_fallback"},
                compiled_logic_py=code,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return VerificationResult(
                constraint_id=constraint.id,
                constraint_type=constraint.type,
                supported=False,
                passed=False,
                reason="verification_error",
                details={"error": str(exc), "constraint": constraint.to_dict(), **details},
                compiled_logic_py=code,
            )

    return VerificationResult(
        constraint_id=constraint.id,
        constraint_type=constraint.type,
        supported=False,
        passed=False,
        reason=reason,
        details=details,
        compiled_logic_py=code,
    )


def verify_constraints(
    plan: dict[str, Any],
    constraints: list[LogicalConstraintObject],
) -> list[VerificationResult]:
    return [verify_constraint(plan, constraint) for constraint in constraints]
