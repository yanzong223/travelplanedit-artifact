"""Validation helpers for canonical ChinaTravel plan payloads."""

from __future__ import annotations

import copy
from typing import Any


ALLOWED_ACTIVITY_TYPES = {
    "airplane",
    "train",
    "attraction",
    "breakfast",
    "lunch",
    "dinner",
    "accommodation",
}

ALLOWED_TRANSPORT_MODES = {"walk", "metro", "taxi"}
MEAL_ACTIVITY_TYPES = {"breakfast", "lunch", "dinner"}
INTERCITY_ACTIVITY_TYPES = {"train", "airplane"}
_ACTIVITY_TYPE_ALIASES = {
    "visit": "attraction",
    "sightseeing": "attraction",
    "poi": "attraction",
    "景点": "attraction",
    "游览": "attraction",
    "参观": "attraction",
    "hotel": "accommodation",
    "lodging": "accommodation",
    "住宿": "accommodation",
    "酒店": "accommodation",
    "flight": "airplane",
    "plane": "airplane",
    "飞机": "airplane",
    "航班": "airplane",
    "rail": "train",
    "火车": "train",
    "高铁": "train",
}
_TRANSPORT_TYPE_ALIASES = {
    "transport",
    "transportation",
    "transfer",
    "walk",
    "metro",
    "taxi",
    "交通",
    "步行",
    "地铁",
    "出租车",
    "打车",
}
_TRANSPORT_MODE_ALIASES = {
    "walk": "walk",
    "walking": "walk",
    "步行": "walk",
    "metro": "metro",
    "subway": "metro",
    "地铁": "metro",
    "taxi": "taxi",
    "cab": "taxi",
    "出租车": "taxi",
    "打车": "taxi",
    "transport": "taxi",
    "transportation": "taxi",
    "transfer": "taxi",
    "交通": "taxi",
}


def is_chinatravel_plan(plan: dict[str, Any] | None) -> bool:
    return isinstance(plan, dict) and isinstance(plan.get("itinerary"), list)


def _hour_from_time(value: Any) -> int | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    raw = value.split(":", 1)[0].replace("次日", "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def _infer_meal_type(activity: dict[str, Any]) -> str:
    for field in ("start_time", "end_time"):
        hour = _hour_from_time(activity.get(field))
        if hour is None:
            continue
        if hour < 11:
            return "breakfast"
        if hour < 16:
            return "lunch"
        return "dinner"
    return "lunch"


def _looks_like_transport_activity(activity: dict[str, Any], activity_type: str) -> bool:
    if activity_type in _TRANSPORT_TYPE_ALIASES:
        return True
    if activity.get("position"):
        return False
    has_route_shape = all(activity.get(field) for field in ("start", "end", "start_time", "end_time"))
    if not has_route_shape:
        return False
    return bool(activity.get("mode")) or activity_type in ALLOWED_TRANSPORT_MODES


def _normalize_transport_activity(activity: dict[str, Any], activity_type: str) -> dict[str, Any]:
    normalized = {
        key: copy.deepcopy(value)
        for key, value in activity.items()
        if key not in {"type", "position", "transports"} and value is not None
    }
    mode = str(activity.get("mode") or activity.get("transport_type") or activity_type or "").strip().lower()
    mode = _TRANSPORT_MODE_ALIASES.get(mode, mode)
    if mode not in ALLOWED_TRANSPORT_MODES:
        mode = "taxi"
    normalized["mode"] = mode
    return normalized


def _copy_first_present(activity: dict[str, Any], target: str, sources: tuple[str, ...]) -> None:
    if activity.get(target):
        return
    for source in sources:
        value = activity.get(source)
        if value:
            activity[target] = value
            return


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _preserve_position_coordinates(activity: dict[str, Any], position: Any) -> None:
    if not isinstance(position, dict):
        return
    for source_key, target_key in (("lat", "lat"), ("lon", "lon"), ("lng", "lon")):
        value = position.get(source_key)
        if value is not None and target_key not in activity:
            activity[target_key] = value


def _canonicalize_activity_position(activity: dict[str, Any]) -> None:
    position = activity.get("position")
    if _non_empty_string(position):
        activity["position"] = position.strip()
        return

    _preserve_position_coordinates(activity, position)
    for source in ("name", "place", "title", "location"):
        value = activity.get(source)
        if _non_empty_string(value):
            activity["position"] = value.strip()
            return


def _copy_intercity_aliases(activity: dict[str, Any]) -> None:
    aliases = (
        ("From", "start"),
        ("To", "end"),
        ("BeginTime", "start_time"),
        ("EndTime", "end_time"),
        ("Cost", "price"),
    )
    for source, target in aliases:
        if activity.get(target) is None and activity.get(source) is not None:
            activity[target] = activity[source]
    if activity.get("cost") is None and activity.get("Cost") is not None:
        activity["cost"] = activity["Cost"]


def normalize_loose_chinatravel_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    """Best-effort normalization for common non-canonical LLM output."""

    if not isinstance(plan, dict):
        return plan

    normalized = copy.deepcopy(plan)
    itinerary = normalized.get("itinerary")
    if not isinstance(itinerary, list):
        return normalized

    for day in itinerary:
        if not isinstance(day, dict):
            continue
        activities = day.get("activities")
        if not isinstance(activities, list):
            continue

        rebuilt_activities: list[dict[str, Any]] = []
        pending_transports: list[dict[str, Any]] = []

        for raw_activity in activities:
            if not isinstance(raw_activity, dict):
                continue

            activity = copy.deepcopy(raw_activity)
            raw_type = str(activity.get("type") or "").strip().lower()

            if _looks_like_transport_activity(activity, raw_type):
                pending_transports.append(_normalize_transport_activity(activity, raw_type))
                continue

            canonical_type = _ACTIVITY_TYPE_ALIASES.get(raw_type, raw_type)
            if canonical_type in {"restaurant", "meal", "food"}:
                canonical_type = _infer_meal_type(activity)
            if canonical_type:
                activity["type"] = canonical_type
            if canonical_type and canonical_type not in INTERCITY_ACTIVITY_TYPES:
                _canonicalize_activity_position(activity)
            elif canonical_type in INTERCITY_ACTIVITY_TYPES:
                _copy_intercity_aliases(activity)
                if canonical_type == "airplane":
                    _copy_first_present(activity, "FlightID", ("flight_id", "flight", "flight_number"))
                elif canonical_type == "train":
                    _copy_first_present(activity, "TrainID", ("train_id", "train", "train_number"))

            existing_transports = activity.get("transports")
            if not isinstance(existing_transports, list):
                existing_transports = []
            if pending_transports:
                activity["transports"] = pending_transports + existing_transports
                pending_transports = []
            else:
                activity["transports"] = existing_transports

            rebuilt_activities.append(activity)

        if pending_transports and rebuilt_activities:
            rebuilt_activities[-1].setdefault("transports", []).extend(pending_transports)

        day["activities"] = rebuilt_activities

    return normalized


def validate_chinatravel_plan(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]
    if "days" in plan:
        return ["plan must not contain legacy top-level 'days'"]
    if "itinerary" not in plan:
        return ["plan must contain top-level 'itinerary'"]
    if not isinstance(plan.get("itinerary"), list):
        return ["plan.itinerary must be a list"]

    issues: list[str] = []
    for day_index, day in enumerate(plan["itinerary"], start=1):
        if not isinstance(day, dict):
            issues.append(f"itinerary[{day_index - 1}] must be an object")
            continue
        if not isinstance(day.get("activities"), list):
            issues.append(f"itinerary[{day_index - 1}].activities must be a list")
            continue

        for activity_index, activity in enumerate(day["activities"]):
            prefix = f"itinerary[{day_index - 1}].activities[{activity_index}]"
            if not isinstance(activity, dict):
                issues.append(f"{prefix} must be an object")
                continue
            activity_type = activity.get("type")
            if activity_type not in ALLOWED_ACTIVITY_TYPES:
                issues.append(f"{prefix}.type must be one of {sorted(ALLOWED_ACTIVITY_TYPES)}")
                continue

            if activity_type in INTERCITY_ACTIVITY_TYPES:
                for field in ("start", "end", "start_time", "end_time"):
                    if not activity.get(field):
                        issues.append(f"{prefix}.{field} is required for {activity_type}")
                id_field = "FlightID" if activity_type == "airplane" else "TrainID"
                if not activity.get(id_field):
                    issues.append(f"{prefix}.{id_field} is required for {activity_type}")
            else:
                position = activity.get("position")
                if not isinstance(position, str) or not position.strip():
                    issues.append(
                        f"{prefix}.position must be a non-empty string POI name for {activity_type}"
                    )

            transports = activity.get("transports", [])
            if transports is None:
                continue
            if not isinstance(transports, list):
                issues.append(f"{prefix}.transports must be a list")
                continue
            for transport_index, transport in enumerate(transports):
                transport_prefix = f"{prefix}.transports[{transport_index}]"
                if not isinstance(transport, dict):
                    issues.append(f"{transport_prefix} must be an object")
                    continue
                mode = transport.get("mode", transport.get("type"))
                if mode not in ALLOWED_TRANSPORT_MODES:
                    issues.append(
                        f"{transport_prefix}.mode must be one of {sorted(ALLOWED_TRANSPORT_MODES)}"
                    )
                for field in ("start", "end", "start_time", "end_time"):
                    if not transport.get(field):
                        issues.append(f"{transport_prefix}.{field} is required")

    return issues


def require_chinatravel_plan(
    plan: dict[str, Any] | None,
    *,
    context: str,
) -> dict[str, Any]:
    issues = validate_chinatravel_plan(plan)
    if issues:
        raise ValueError(f"{context} is not a canonical ChinaTravel plan: {'; '.join(issues)}")
    assert isinstance(plan, dict)
    return plan
