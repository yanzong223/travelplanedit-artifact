"""Rule definitions for origin plan hygiene and quality gating."""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


NORMAL_ACTIVITY_TYPES = {"attraction"}
MEAL_ACTIVITY_TYPES = {"breakfast", "lunch", "dinner"}
LOW_SIGNAL_ACTIVITY_TYPES = {"accommodation", *MEAL_ACTIVITY_TYPES}
INTERCITY_TRANSPORT_TYPES = {"train", "airplane"}
POI_DUPLICATE_ACTIVITY_TYPES = {"attraction", "lunch", "dinner"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHINATRAVEL_DATABASE_ROOT = (
    PROJECT_ROOT / "Chinatravel" / "ChinaTravel" / "chinatravel" / "environment" / "database"
)
CHINATRAVEL_ROOT = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
DATA_ROOT = CHINATRAVEL_DATABASE_ROOT
CITY_EN_LIST = [
    "beijing",
    "shanghai",
    "nanjing",
    "suzhou",
    "hangzhou",
    "shenzhen",
    "chengdu",
    "wuhan",
    "guangzhou",
    "chongqing",
]
CITY_CN_LIST = [
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
]
CITY_NAMES = set(CITY_EN_LIST)
CITY_CN_TO_EN = dict(zip(CITY_CN_LIST, CITY_EN_LIST))
CITY_EN_TO_CN = dict(zip(CITY_EN_LIST, CITY_CN_LIST))


@dataclass(frozen=True)
class Thresholds:
    attraction_start_minute: int
    max_walk_distance_km: float
    max_walk_duration_min: int
    max_attractions_per_day: int
    day_boundary_prev_end_minute: int
    day_boundary_next_transfer_before: int
    max_idle_gap_min: int
    duplicate_poi_threshold: int


@dataclass(frozen=True)
class RuleConfig:
    name: str
    thresholds: Thresholds

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "thresholds": asdict(self.thresholds)}


HYGIENE_CONFIG = RuleConfig(
    name="hygiene",
    thresholds=Thresholds(
        attraction_start_minute=5 * 60,
        max_walk_distance_km=8.0,
        max_walk_duration_min=120,
        max_attractions_per_day=7,
        day_boundary_prev_end_minute=23 * 60,
        day_boundary_next_transfer_before=3 * 60,
        max_idle_gap_min=180,
        duplicate_poi_threshold=1,
    ),
)


QUALITY_GATE_CONFIG = RuleConfig(
    name="quality_gate",
    thresholds=Thresholds(
        attraction_start_minute=6 * 60,
        max_walk_distance_km=5.0,
        max_walk_duration_min=90,
        max_attractions_per_day=5,
        day_boundary_prev_end_minute=22 * 60,
        day_boundary_next_transfer_before=5 * 60,
        max_idle_gap_min=120,
        duplicate_poi_threshold=1,
    ),
)


@dataclass(frozen=True)
class Violation:
    code: str
    day: int
    activity_index: Optional[int]
    message: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "day": self.day,
            "activity_index": self.activity_index,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class PlanEvaluation:
    hygiene_pass: bool
    quality_pass: bool
    hygiene_violations: List[Violation]
    quality_violations: List[Violation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hygiene_pass": self.hygiene_pass,
            "quality_pass": self.quality_pass,
            "hygiene_violations": [item.to_dict() for item in self.hygiene_violations],
            "quality_violations": [item.to_dict() for item in self.quality_violations],
        }


def parse_time_to_minute(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    normalized = str(value).strip()
    day_offset = 0
    for prefix in ("次日", "翌日", "隔日", "第二天", "next day"):
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):].strip()
            day_offset = 24 * 60
            break
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", normalized)
    if match is None:
        raise ValueError(f"Unsupported time value: {value!r}")
    hour_text, minute_text = match.groups()
    hour = int(hour_text)
    minute = int(minute_text)
    if hour == 24:
        return day_offset + 24 * 60 + minute
    return day_offset + hour * 60 + minute


def duration_minutes(start: Optional[str], end: Optional[str]) -> Optional[int]:
    try:
        start_minute = parse_time_to_minute(start)
        end_minute = parse_time_to_minute(end)
    except ValueError:
        return None
    if start_minute is None or end_minute is None:
        return None
    if end_minute < start_minute:
        end_minute += 24 * 60
    return end_minute - start_minute


def _activity_label(activity: Dict[str, Any]) -> str:
    if activity.get("position"):
        return str(activity["position"])
    if activity.get("start") or activity.get("end"):
        return f'{activity.get("start", "?")} -> {activity.get("end", "?")}'
    return activity.get("type", "unknown")


def _append_if_missing(target: List[Violation], violation: Violation) -> None:
    key = (violation.code, violation.day, violation.activity_index, violation.message)
    existing = {
        (item.code, item.day, item.activity_index, item.message)
        for item in target
    }
    if key not in existing:
        target.append(violation)


def _safe_parse_time_to_minute(
    value: Optional[str],
    *,
    violations: List[Violation],
    day: int,
    activity_index: Optional[int],
    label: str,
    field_name: str,
) -> Optional[int]:
    if not value:
        return None
    try:
        return parse_time_to_minute(value)
    except ValueError:
        _append_if_missing(
            violations,
            Violation(
                code="invalid_time_value",
                day=day,
                activity_index=activity_index,
                message=f"{label} has unsupported {field_name}: {value}.",
                evidence={"field": field_name, "value": value},
            ),
        )
        return None


def _first_transport_start(
    activity: Dict[str, Any],
    *,
    violations: List[Violation],
    day: int,
    activity_index: int,
    label: str,
) -> Optional[int]:
    for transport in activity.get("transports", []):
        start_minute = _safe_parse_time_to_minute(
            transport.get("start_time"),
            violations=violations,
            day=day,
            activity_index=activity_index,
            label=label,
            field_name="transport.start_time",
        )
        if start_minute is not None:
            return start_minute
    return None


def _last_transport_end(
    activity: Dict[str, Any],
    *,
    violations: List[Violation],
    day: int,
    activity_index: int,
    label: str,
) -> Optional[int]:
    transports = activity.get("transports", [])
    for transport in reversed(transports):
        end_minute = _safe_parse_time_to_minute(
            transport.get("end_time"),
            violations=violations,
            day=day,
            activity_index=activity_index,
            label=label,
            field_name="transport.end_time",
        )
        if end_minute is not None:
            return end_minute
    return None


def _canonical_city_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    if text in CITY_CN_TO_EN:
        return CITY_CN_TO_EN[text]
    lowered = text.lower()
    return lowered if lowered in CITY_NAMES else None


def _city_candidates() -> List[str]:
    return list(CITY_EN_LIST)


def _normalize_poi_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return str(value).strip()


@lru_cache(maxsize=1)
def _load_poi_index() -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"ChinaTravel environment database not found: {DATA_ROOT}")

    index: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    if str(CHINATRAVEL_ROOT) not in sys.path:
        sys.path.insert(0, str(CHINATRAVEL_ROOT))

    from chinatravel.environment.tools.accommodations.apis import Accommodations
    from chinatravel.environment.tools.attractions.apis import Attractions
    from chinatravel.environment.tools.poi.apis import Poi
    from chinatravel.environment.tools.restaurants.apis import Restaurants

    poi_api = Poi()
    attractions_api = Attractions()
    restaurants_api = Restaurants()
    accommodations_api = Accommodations()

    for city in _city_candidates():
        city_cn = CITY_EN_TO_CN[city]
        city_index: Dict[str, Dict[str, Optional[str]]] = {}

        for name in poi_api.data[city_cn].keys():
            normalized = _normalize_poi_name(name)
            if not normalized:
                continue
            city_index[normalized] = {
                "city": city,
                "type": None,
                "opentime": None,
                "endtime": None,
            }

        for row in attractions_api.data[city_cn].to_dict("records"):
            name = _normalize_poi_name(row.get("name"))
            if not name:
                continue
            city_index[name] = {
                "city": city,
                "type": "attraction",
                "opentime": row.get("opentime"),
                "endtime": row.get("endtime"),
            }

        for row in restaurants_api.data[city_cn].to_dict("records"):
            name = _normalize_poi_name(row.get("name"))
            if not name:
                continue
            city_index[name] = {
                "city": city,
                "type": "restaurant",
                "opentime": row.get("opentime"),
                "endtime": row.get("endtime"),
            }

        for row in accommodations_api.data[city_cn].to_dict("records"):
            name = _normalize_poi_name(row.get("name"))
            if not name:
                continue
            city_index[name] = {
                "city": city,
                "type": "accommodation",
                "opentime": None,
                "endtime": None,
            }

        index[city] = city_index

    return index


def _lookup_poi(
    name: Optional[str],
    preferred_city: Optional[str] = None,
    *,
    allow_global_fallback: bool = True,
) -> Optional[Dict[str, Optional[str]]]:
    normalized_name = _normalize_poi_name(name)
    if not normalized_name:
        return None

    poi_index = _load_poi_index()
    canonical_city = _canonical_city_name(preferred_city)
    if canonical_city and normalized_name in poi_index.get(canonical_city, {}):
        return poi_index[canonical_city][normalized_name]
    if canonical_city and not allow_global_fallback:
        return None

    for city in _city_candidates():
        if normalized_name in poi_index.get(city, {}):
            return poi_index[city][normalized_name]
    return None


def _infer_day_city(
    activities: List[Dict[str, Any]],
    fallback_city: Optional[str],
) -> Optional[str]:
    counts: Dict[str, int] = {}
    for activity in activities:
        if activity.get("type") in INTERCITY_TRANSPORT_TYPES | MEAL_ACTIVITY_TYPES:
            continue
        poi_name = _activity_primary_poi(activity)
        if not poi_name:
            continue
        record = _lookup_poi(poi_name, fallback_city, allow_global_fallback=False)
        if not record or not record.get("city"):
            continue
        city = str(record["city"])
        counts[city] = counts.get(city, 0) + 1

    if counts:
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return _canonical_city_name(fallback_city)


def _activity_primary_poi(activity: Dict[str, Any]) -> Optional[str]:
    if activity.get("position"):
        return _normalize_poi_name(activity.get("position"))
    if activity.get("type") in INTERCITY_TRANSPORT_TYPES:
        return _normalize_poi_name(activity.get("start"))
    return None


def _activity_opening_record(
    activity: Dict[str, Any],
    preferred_city: Optional[str],
    *,
    allow_global_fallback: bool = True,
) -> Optional[Dict[str, Optional[str]]]:
    activity_type = str(activity.get("type", "unknown"))
    if activity_type in {"attraction", "breakfast", "lunch", "dinner", "accommodation"}:
        return _lookup_poi(
            activity.get("position"),
            preferred_city,
            allow_global_fallback=allow_global_fallback,
        )
    return None


def _record_city_for_endpoint(
    name: Optional[str],
    preferred_city: Optional[str],
) -> Optional[str]:
    record = _lookup_poi(name, preferred_city)
    return str(record["city"]) if record and record.get("city") else None


def _meal_requires_destination_checks(
    plan: Dict[str, Any],
    day_index: int,
    activities: List[Dict[str, Any]],
    activity_index: int,
    day_city: Optional[str],
) -> bool:
    """Return whether a meal falls within the target-city travel segment."""

    target_city = _canonical_city_name(plan.get("target_city"))
    if target_city is None:
        return True

    activity = activities[activity_index]
    if _lookup_poi(activity.get("position"), target_city, allow_global_fallback=False):
        return True

    start_city = _canonical_city_name(plan.get("start_city"))
    if start_city == target_city:
        return True

    has_intercity_today = any(
        item.get("type") in INTERCITY_TRANSPORT_TYPES for item in activities
    )
    if not has_intercity_today:
        return day_city == target_city

    in_target_segment = day_index > 0
    for previous in activities[:activity_index]:
        if previous.get("type") not in INTERCITY_TRANSPORT_TYPES:
            continue
        start_city_of_leg = _record_city_for_endpoint(
            previous.get("start"),
            plan.get("start_city"),
        )
        end_city_of_leg = _record_city_for_endpoint(
            previous.get("end"),
            plan.get("target_city"),
        )
        if end_city_of_leg == target_city:
            in_target_segment = True
        elif start_city_of_leg == target_city and end_city_of_leg != target_city:
            in_target_segment = False

    return in_target_segment


def _is_within_opening_hours(
    activity_start: Optional[int],
    activity_end: Optional[int],
    open_time: Optional[str],
    end_time: Optional[str],
) -> bool:
    try:
        open_minute = parse_time_to_minute(open_time)
        close_minute = parse_time_to_minute(end_time)
    except ValueError:
        return True
    if (
        activity_start is None
        or activity_end is None
        or open_minute is None
        or close_minute is None
    ):
        return True
    adjusted_close = close_minute
    adjusted_end = activity_end
    if adjusted_close < open_minute:
        adjusted_close += 24 * 60
    if adjusted_end < activity_start:
        adjusted_end += 24 * 60
    return activity_start >= open_minute and adjusted_end <= adjusted_close


def _check_with_config(plan: Dict[str, Any], config: RuleConfig) -> List[Violation]:
    violations: List[Violation] = []
    itinerary = plan.get("itinerary", [])
    target_city = plan.get("target_city")
    duplicate_counts: Dict[str, int] = {}

    if "people_number" not in plan:
        _append_if_missing(
            violations,
            Violation(
                code="missing_people_number",
                day=0,
                activity_index=None,
                message="Plan is missing people_number.",
                evidence={},
            ),
        )

    if not isinstance(itinerary, list) or not itinerary:
        _append_if_missing(
            violations,
            Violation(
                code="missing_itinerary",
                day=0,
                activity_index=None,
                message="Plan has no itinerary.",
                evidence={},
            ),
        )
        return violations

    for day_index, day_plan in enumerate(itinerary):
        day_number = int(day_plan.get("day", day_index + 1))
        activities = day_plan.get("activities", [])
        attraction_count = 0
        previous_end_minute: Optional[int] = None
        day_city = _infer_day_city(activities, target_city)
        has_cross_city_transport = False
        has_accommodation = any(
            str(activity.get("type")) == "accommodation" for activity in activities
        )

        for activity_index, activity in enumerate(activities):
            activity_type = str(activity.get("type", "unknown"))
            label = _activity_label(activity)
            activity_start = _safe_parse_time_to_minute(
                activity.get("start_time"),
                violations=violations,
                day=day_number,
                activity_index=activity_index,
                label=label,
                field_name="start_time",
            )
            activity_end = _safe_parse_time_to_minute(
                activity.get("end_time"),
                violations=violations,
                day=day_number,
                activity_index=activity_index,
                label=label,
                field_name="end_time",
            )
            preferred_city = day_city or target_city

            if activity_type in INTERCITY_TRANSPORT_TYPES:
                start_record = _lookup_poi(activity.get("start"), None)
                end_record = _lookup_poi(activity.get("end"), None)
                if (
                    start_record
                    and end_record
                    and start_record.get("city") != end_record.get("city")
                ):
                    has_cross_city_transport = True
                for field_name in ("start", "end"):
                    poi_name = activity.get(field_name)
                    endpoint_record = start_record if field_name == "start" else end_record
                    if poi_name and endpoint_record is None:
                        _append_if_missing(
                            violations,
                            Violation(
                                code="invalid_poi",
                                day=day_number,
                                activity_index=activity_index,
                                message=f"{label} references unknown POI: {poi_name}.",
                                evidence={"field": field_name, "poi_name": poi_name},
                            ),
                        )

            if activity_type not in INTERCITY_TRANSPORT_TYPES:
                check_meal_in_destination = (
                    activity_type not in MEAL_ACTIVITY_TYPES
                    or _meal_requires_destination_checks(
                        plan,
                        day_index,
                        activities,
                        activity_index,
                        day_city,
                    )
                )
                activity_record = (
                    _activity_opening_record(
                        activity,
                        preferred_city,
                        allow_global_fallback=False,
                    )
                    if check_meal_in_destination
                    else None
                )
                poi_name = activity.get("position")
                if poi_name and check_meal_in_destination and activity_record is None:
                    _append_if_missing(
                        violations,
                        Violation(
                            code="invalid_poi",
                            day=day_number,
                            activity_index=activity_index,
                            message=f"{label} references unknown POI: {poi_name}.",
                            evidence={"field": "position", "poi_name": poi_name},
                        ),
                    )

                if activity_record and activity_record.get("city") and day_city:
                    activity_city = str(activity_record["city"])
                    if (
                        activity_city != day_city
                        and not has_cross_city_transport
                        and activity_type not in MEAL_ACTIVITY_TYPES
                    ):
                        _append_if_missing(
                            violations,
                            Violation(
                                code="city_mismatch",
                                day=day_number,
                                activity_index=activity_index,
                                message=(
                                    f"{label} is in {activity_city}, but day {day_number} is "
                                    f"anchored in {day_city}."
                                ),
                                evidence={
                                    "activity_city": activity_city,
                                    "day_city": day_city,
                                },
                            ),
                        )

                if activity_record and not _is_within_opening_hours(
                    activity_start,
                    activity_end,
                    activity_record.get("opentime"),
                    activity_record.get("endtime"),
                ):
                    _append_if_missing(
                        violations,
                        Violation(
                            code="opening_hours_violation",
                            day=day_number,
                            activity_index=activity_index,
                            message=(
                                f"{label} is scheduled outside opening hours "
                                f"{activity_record.get('opentime')}-{activity_record.get('endtime')}."
                            ),
                            evidence={
                                "start_time": activity.get("start_time"),
                                "end_time": activity.get("end_time"),
                                "opentime": activity_record.get("opentime"),
                                "endtime": activity_record.get("endtime"),
                            },
                        ),
                    )

                if activity_type in POI_DUPLICATE_ACTIVITY_TYPES and poi_name:
                    normalized_poi = _normalize_poi_name(poi_name)
                    if normalized_poi:
                        duplicate_counts[normalized_poi] = (
                            duplicate_counts.get(normalized_poi, 0) + 1
                        )

            if activity_type in NORMAL_ACTIVITY_TYPES:
                attraction_count += 1
                if (
                    activity_start is not None
                    and activity_start < config.thresholds.attraction_start_minute
                ):
                    _append_if_missing(
                        violations,
                        Violation(
                            code="early_attraction",
                            day=day_number,
                            activity_index=activity_index,
                            message=(
                                f"{label} starts at {activity.get('start_time')}, earlier than "
                                f"{_minute_label(config.thresholds.attraction_start_minute)}."
                            ),
                            evidence={
                                "activity_type": activity_type,
                                "start_time": activity.get("start_time"),
                            },
                        ),
                    )

            if previous_end_minute is not None and activity_start is not None:
                gap = activity_start - previous_end_minute
                if gap < 0:
                    _append_if_missing(
                        violations,
                        Violation(
                            code="timeline_overlap",
                            day=day_number,
                            activity_index=activity_index,
                            message=(
                                f"{label} starts at {activity.get('start_time')} before the "
                                "previous activity ends."
                            ),
                            evidence={
                                "activity_start": activity.get("start_time"),
                                "previous_end_minute": previous_end_minute,
                            },
                        ),
                    )
                elif (
                    gap > config.thresholds.max_idle_gap_min
                    and activity_type not in LOW_SIGNAL_ACTIVITY_TYPES
                ):
                    _append_if_missing(
                        violations,
                        Violation(
                            code="large_idle_gap",
                            day=day_number,
                            activity_index=activity_index,
                            message=(
                                f"{label} has a {gap}-minute gap after the previous activity."
                            ),
                            evidence={
                                "gap_minutes": gap,
                                "activity_start": activity.get("start_time"),
                            },
                        ),
                    )

            first_transport_start = _first_transport_start(
                activity,
                violations=violations,
                day=day_number,
                activity_index=activity_index,
                label=label,
            )
            last_transport_end = _last_transport_end(
                activity,
                violations=violations,
                day=day_number,
                activity_index=activity_index,
                label=label,
            )
            if (
                previous_end_minute is not None
                and first_transport_start is not None
                and first_transport_start < previous_end_minute
            ):
                _append_if_missing(
                    violations,
                    Violation(
                        code="transport_before_previous_end",
                        day=day_number,
                        activity_index=activity_index,
                        message=(
                            f"{label} has transport starting at "
                            f"{activity.get('transports', [{}])[0].get('start_time')} before the "
                            "previous activity ends."
                        ),
                        evidence={
                            "transport_start_minute": first_transport_start,
                            "previous_end_minute": previous_end_minute,
                        },
                    ),
                )

            if (
                last_transport_end is not None
                and activity_start is not None
                and last_transport_end > activity_start
            ):
                _append_if_missing(
                    violations,
                    Violation(
                        code="transport_after_activity_start",
                        day=day_number,
                        activity_index=activity_index,
                        message=(
                            f"{label} starts at {activity.get('start_time')} before its transfer "
                            "chain finishes."
                        ),
                        evidence={
                            "activity_start": activity.get("start_time"),
                            "last_transport_end_minute": last_transport_end,
                        },
                    ),
                )

            if (
                last_transport_end is not None
                and activity_start is not None
                and activity_start - last_transport_end > config.thresholds.max_idle_gap_min
                and activity_type not in LOW_SIGNAL_ACTIVITY_TYPES
            ):
                _append_if_missing(
                    violations,
                    Violation(
                        code="post_transport_idle_gap",
                        day=day_number,
                        activity_index=activity_index,
                        message=(
                            f"{label} begins {activity_start - last_transport_end} minutes after "
                            "its last transfer ends."
                        ),
                        evidence={
                            "gap_minutes": activity_start - last_transport_end,
                            "activity_start": activity.get("start_time"),
                        },
                    ),
                )

            for transport in activity.get("transports", []):
                if str(transport.get("mode")) != "walk":
                    continue
                distance = transport.get("distance")
                walk_duration = duration_minutes(
                    transport.get("start_time"),
                    transport.get("end_time"),
                )
                if (
                    isinstance(distance, (int, float))
                    and distance > config.thresholds.max_walk_distance_km
                ):
                    _append_if_missing(
                        violations,
                        Violation(
                            code="long_walk_distance",
                            day=day_number,
                            activity_index=activity_index,
                            message=(
                                f"Walk to {label} is {distance:.2f}km, above "
                                f"{config.thresholds.max_walk_distance_km:.1f}km."
                            ),
                            evidence={
                                "distance_km": round(float(distance), 2),
                                "transport_start": transport.get("start"),
                                "transport_end": transport.get("end"),
                            },
                        ),
                    )
                if (
                    walk_duration is not None
                    and walk_duration > config.thresholds.max_walk_duration_min
                ):
                    _append_if_missing(
                        violations,
                        Violation(
                            code="long_walk_duration",
                            day=day_number,
                            activity_index=activity_index,
                            message=(
                                f"Walk to {label} lasts {walk_duration} minutes, above "
                                f"{config.thresholds.max_walk_duration_min} minutes."
                            ),
                            evidence={
                                "duration_min": walk_duration,
                                "transport_start_time": transport.get("start_time"),
                                "transport_end_time": transport.get("end_time"),
                            },
                        ),
                    )

            previous_end_minute = activity_end if activity_end is not None else previous_end_minute

        if attraction_count > config.thresholds.max_attractions_per_day:
            _append_if_missing(
                violations,
                Violation(
                    code="dense_day_attractions",
                    day=day_number,
                    activity_index=None,
                    message=(
                        f"Day {day_number} has {attraction_count} attractions, above "
                        f"{config.thresholds.max_attractions_per_day}."
                    ),
                    evidence={"attraction_count": attraction_count},
                ),
            )

        if day_index < len(itinerary) - 1 and not has_accommodation:
            _append_if_missing(
                violations,
                Violation(
                    code="missing_accommodation",
                    day=day_number,
                    activity_index=None,
                    message=f"Day {day_number} has no accommodation activity.",
                    evidence={},
                ),
            )

        if day_index == 0:
            continue

        previous_day = itinerary[day_index - 1]
        previous_activities = previous_day.get("activities", [])
        if not previous_activities or not activities:
            continue

        previous_label = _activity_label(previous_activities[-1])
        previous_last_end = _safe_parse_time_to_minute(
            previous_activities[-1].get("end_time"),
            violations=violations,
            day=int(previous_day.get("day", day_index)),
            activity_index=len(previous_activities) - 1,
            label=previous_label,
            field_name="end_time",
        )
        first_transfer_start = _first_transport_start(
            activities[0],
            violations=violations,
            day=day_number,
            activity_index=0,
            label=_activity_label(activities[0]),
        )
        if (
            previous_last_end is not None
            and first_transfer_start is not None
            and previous_last_end >= config.thresholds.day_boundary_prev_end_minute
            and first_transfer_start < config.thresholds.day_boundary_next_transfer_before
        ):
            _append_if_missing(
                violations,
                Violation(
                    code="day_boundary_transfer",
                    day=day_number,
                    activity_index=0,
                    message=(
                        f"Day {day_number} starts transfer at "
                        f"{activities[0].get('transports', [{}])[0].get('start_time')} after "
                        f"the previous day ends at {previous_activities[-1].get('end_time')}."
                    ),
                    evidence={
                        "previous_day_end": previous_activities[-1].get("end_time"),
                        "first_transfer_start": activities[0]
                        .get("transports", [{}])[0]
                        .get("start_time"),
                    },
                ),
            )

    for poi_name, count in sorted(duplicate_counts.items()):
        if count <= config.thresholds.duplicate_poi_threshold:
            continue
        _append_if_missing(
            violations,
            Violation(
                code="duplicate_poi",
                day=0,
                activity_index=None,
                message=(
                    f"{poi_name} appears {count} times, above allowed threshold "
                    f"{config.thresholds.duplicate_poi_threshold}."
                ),
                evidence={"poi_name": poi_name, "count": count},
            ),
        )

    return violations


def evaluate_plan(plan: Dict[str, Any]) -> PlanEvaluation:
    hygiene_violations = _check_with_config(plan, HYGIENE_CONFIG)
    quality_violations = _check_with_config(plan, QUALITY_GATE_CONFIG)
    return PlanEvaluation(
        hygiene_pass=not hygiene_violations,
        quality_pass=(not hygiene_violations) and (not quality_violations),
        hygiene_violations=hygiene_violations,
        quality_violations=quality_violations,
    )


def summarize_violation_codes(violations: Iterable[Violation]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for violation in violations:
        counts[violation.code] = counts.get(violation.code, 0) + 1
    return counts


def _minute_label(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"
