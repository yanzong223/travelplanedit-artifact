"""Deterministic local feasibility checks for POI time-window edits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHINATRAVEL_PATH = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
if str(CHINATRAVEL_PATH) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_PATH))


TIME_PERIOD_WINDOWS = {
    "morning": (9 * 60, 12 * 60),
    "afternoon": (12 * 60, 18 * 60),
    "evening": (18 * 60, 22 * 60),
}
TIME_PERIOD_ALIASES = {
    "上午": "morning",
    "早上": "morning",
    "morning": "morning",
    "下午": "afternoon",
    "afternoon": "afternoon",
    "晚上": "evening",
    "夜晚": "evening",
    "evening": "evening",
}
DEFAULT_VISIT_MINUTES = 90


@dataclass(frozen=True)
class PoiWindow:
    name: str
    open_minute: int
    close_minute: int
    duration_minutes: int
    duration_source: str
    evidence: Dict[str, Any]


class TemporalWindowCatalog:
    """Lazy ChinaTravel DB access used by the temporal-window checker."""

    def __init__(self) -> None:
        self._attractions = None
        self._transportation = None

    @property
    def attractions(self):
        if self._attractions is None:
            from chinatravel.environment.tools.attractions.apis import Attractions

            self._attractions = Attractions()
        return self._attractions

    @property
    def transportation(self):
        if self._transportation is None:
            from chinatravel.environment.tools.transportation.apis import Transportation

            self._transportation = Transportation()
        return self._transportation


def assess_temporal_window_feasibility(
    *,
    origin_plan: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    canonical_constraint_ir: Optional[Dict[str, Any]] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    catalog: Optional[Any] = None,
) -> Dict[str, Any]:
    """Check whether requested POI time-window edits have a local schedule.

    This is intentionally local and deterministic: it proves that the named
    attraction(s) can fit inside the requested broad time period with DB opening
    hours, visit durations, and taxi travel time. It is not a full itinerary
    solver.
    """

    del constraints, canonical_constraint_ir
    catalog = catalog or TemporalWindowCatalog()
    request = _extract_temporal_window_request(edit_target_constraints, query_generation_trace)
    if request is None:
        return _skipped("not_applicable", "no temporal-window edit constraint found")

    period = _normalize_period(request["period"])
    if period is None:
        return _unknown(
            "missing_time_period",
            "temporal-window feasibility missing supported time_period",
            repair_action="retry_same_template",
        )

    city = str(origin_plan.get("target_city") or "").strip()
    if not city:
        return _unknown("missing_target_city", "origin_plan.target_city is missing")

    pois = [str(item).strip() for item in request["pois"] if str(item).strip()]
    if len(pois) not in {1, 2}:
        return _unknown("missing_poi_anchor", "temporal-window feasibility requires one or two POIs")

    poi_infos: List[PoiWindow] = []
    for poi in pois:
        resolved = _build_poi_window(catalog, city, poi, origin_plan)
        if isinstance(resolved, dict):
            return resolved
        poi_infos.append(resolved)

    if len(poi_infos) == 1:
        return _check_single_poi(poi_infos[0], period, request.get("constraint_id"))
    return _check_pair(catalog, city, poi_infos[0], poi_infos[1], period, request.get("constraint_id"))


def _extract_temporal_window_request(
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]],
    query_generation_trace: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if isinstance(edit_target_constraints, Sequence) and not isinstance(edit_target_constraints, (str, bytes)):
        for item in edit_target_constraints:
            if not isinstance(item, dict):
                continue
            ctype = str(item.get("type") or "")
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            if ctype == "pair_time_window_no_overlap":
                return {
                    "constraint_id": item.get("id"),
                    "pois": [target.get("first_poi"), target.get("second_poi")],
                    "period": item.get("value"),
                }
            if ctype == "poi_time_window":
                return {
                    "constraint_id": item.get("id"),
                    "pois": [target.get("poi_name")],
                    "period": item.get("value"),
                }

    trace = query_generation_trace if isinstance(query_generation_trace, dict) else {}
    ctype = str(trace.get("constraint_type") or "")
    spec = trace.get("canonical_query_spec") if isinstance(trace.get("canonical_query_spec"), dict) else {}
    if ctype == "same_time_period_no_overlap":
        return {"pois": list(spec.get("ordered_pois") or []), "period": spec.get("time_period")}
    if ctype == "time_window":
        return {"pois": [spec.get("anchor_poi")], "period": spec.get("time_period")}
    return None


def _normalize_period(value: Any) -> Optional[str]:
    return TIME_PERIOD_ALIASES.get(str(value or "").strip().lower()) or TIME_PERIOD_ALIASES.get(str(value or "").strip())


def _parse_time(value: Any) -> Optional[int]:
    if not isinstance(value, str) or ":" not in value:
        return None
    left, right = value.split(":", 1)
    try:
        hour = int(left)
        minute = int(right)
    except ValueError:
        return None
    if hour == 24 and minute == 0:
        return 24 * 60
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour * 60 + minute
    return None


def _format_time(minutes: int) -> str:
    minutes = max(0, min(24 * 60, int(minutes)))
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _activity_name(activity: Dict[str, Any]) -> str:
    return str(activity.get("position") or activity.get("name") or "").strip()


def _activity_duration_minutes(activity: Dict[str, Any]) -> Optional[int]:
    start = _parse_time(activity.get("start_time"))
    end = _parse_time(activity.get("end_time"))
    if start is None or end is None:
        return None
    if end < start:
        end += 24 * 60
    duration = end - start
    return duration if duration > 0 else None


def _origin_duration(origin_plan: Dict[str, Any], poi_name: str) -> Optional[int]:
    for day_plan in origin_plan.get("itinerary") or []:
        for activity in day_plan.get("activities") or []:
            if isinstance(activity, dict) and _activity_name(activity) == poi_name:
                return _activity_duration_minutes(activity)
    return None


def _build_poi_window(
    catalog: Any,
    city: str,
    poi_name: str,
    origin_plan: Dict[str, Any],
) -> PoiWindow | Dict[str, Any]:
    try:
        df = catalog.attractions.data[city]
    except Exception as exc:
        return _unknown("attraction_catalog_unavailable", f"attraction catalog unavailable for {city}: {exc}")
    if "name" not in df.columns:
        return _unknown("attraction_catalog_invalid", f"attraction catalog missing name column for {city}")
    row = df[df["name"].astype(str) == poi_name]
    if len(row) == 0:
        return _infeasible("poi_not_found", f"attraction not found in DB: {city} - {poi_name}", "replace_poi")
    record = row.iloc[0]
    open_minute = _parse_time(_row_get(record, "opentime"))
    close_minute = _parse_time(_row_get(record, "endtime"))
    if open_minute is None or close_minute is None:
        return _unknown("opening_hours_missing", f"opening hours missing for {poi_name}")
    if close_minute <= open_minute:
        close_minute += 24 * 60

    duration = _origin_duration(origin_plan, poi_name)
    duration_source = "origin_plan"
    if duration is None:
        duration = _recommended_duration(record)
        duration_source = "db_recommendmintime" if duration is not None else "default_90min"
    if duration is None:
        duration = DEFAULT_VISIT_MINUTES

    return PoiWindow(
        name=poi_name,
        open_minute=open_minute,
        close_minute=close_minute,
        duration_minutes=int(duration),
        duration_source=duration_source,
        evidence={
            "opentime": _row_get(record, "opentime"),
            "endtime": _row_get(record, "endtime"),
            "recommendmintime": _row_get(record, "recommendmintime"),
            "recommendmaxtime": _row_get(record, "recommendmaxtime"),
        },
    )


def _row_get(row: Any, key: str) -> Any:
    try:
        return row.get(key)
    except AttributeError:
        return None


def _recommended_duration(row: Any) -> Optional[int]:
    for key in ("recommendmintime", "recommend_min_time"):
        value = _row_get(row, key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return max(1, int(round(parsed * 60)))
    return None


def _check_single_poi(poi: PoiWindow, period: str, constraint_id: Any) -> Dict[str, Any]:
    window_start, window_end = TIME_PERIOD_WINDOWS[period]
    start = max(window_start, poi.open_minute)
    end = start + poi.duration_minutes
    latest_end = min(window_end, poi.close_minute)
    if end <= latest_end:
        return _feasible(
            period=period,
            constraint_id=constraint_id,
            schedule=[
                _schedule_item(poi, start, end),
            ],
        )
    return _infeasible(
        "closed_in_requested_window" if poi.open_minute >= window_end or poi.close_minute <= window_start else "duration_exceeds_window",
        f"{poi.name} cannot fit in requested {period} window",
        "relax_time_window",
        constraint_id=constraint_id,
        lower_bounds={"required_minutes": poi.duration_minutes, "available_window_minutes": max(0, latest_end - start)},
    )


def _check_pair(
    catalog: Any,
    city: str,
    left: PoiWindow,
    right: PoiWindow,
    period: str,
    constraint_id: Any,
) -> Dict[str, Any]:
    attempts = []
    for first, second in ((left, right), (right, left)):
        transport = _transport_minutes(catalog, city, first.name, second.name)
        if isinstance(transport, dict):
            return transport
        schedule = _schedule_pair_order(first, second, period, transport)
        attempts.append({"order": [first.name, second.name], "transport_minutes": transport, "fits": schedule is not None})
        if schedule is not None:
            return _feasible(
                period=period,
                constraint_id=constraint_id,
                schedule=schedule,
                transport_minutes=transport,
                attempted_orders=attempts,
            )

    required = left.duration_minutes + right.duration_minutes + min(
        [item["transport_minutes"] for item in attempts if isinstance(item.get("transport_minutes"), int)] or [0]
    )
    available = TIME_PERIOD_WINDOWS[period][1] - TIME_PERIOD_WINDOWS[period][0]
    code = "pair_duration_exceeds_window" if required > available else "no_order_fits_window"
    return _infeasible(
        code,
        f"{left.name} and {right.name} cannot be scheduled sequentially in {period}",
        "relax_time_window",
        constraint_id=constraint_id,
        lower_bounds={"required_minutes": required, "available_window_minutes": available},
        attempted_orders=attempts,
    )


def _transport_minutes(catalog: Any, city: str, start: str, end: str) -> int | Dict[str, Any]:
    try:
        result = catalog.transportation.goto(city, start, end, "12:00", "taxi")
    except Exception as exc:
        return _unknown("transport_lookup_failed", f"taxi transport lookup failed for {start}->{end}: {exc}")
    if not isinstance(result, list) or not result:
        return _unknown("transport_lookup_failed", f"taxi transport lookup returned no route for {start}->{end}")
    leg = result[0]
    if not isinstance(leg, dict):
        return _unknown("transport_lookup_failed", f"taxi transport route malformed for {start}->{end}")
    start_min = _parse_time(leg.get("start_time"))
    end_min = _parse_time(leg.get("end_time"))
    if start_min is not None and end_min is not None:
        if end_min < start_min:
            end_min += 24 * 60
        return max(0, end_min - start_min)
    try:
        distance = float(leg.get("distance"))
    except (TypeError, ValueError):
        return _unknown("transport_lookup_failed", f"taxi transport route missing duration for {start}->{end}")
    return max(1, int(round(distance / 40.0 * 60)))


def _schedule_pair_order(
    first: PoiWindow,
    second: PoiWindow,
    period: str,
    transport_minutes: int,
) -> Optional[List[Dict[str, Any]]]:
    window_start, window_end = TIME_PERIOD_WINDOWS[period]
    first_start = max(window_start, first.open_minute)
    first_end = first_start + first.duration_minutes
    if first_end > min(window_end, first.close_minute):
        return None
    second_start = max(first_end + transport_minutes, second.open_minute, window_start)
    second_end = second_start + second.duration_minutes
    if second_end > min(window_end, second.close_minute):
        return None
    return [
        _schedule_item(first, first_start, first_end),
        {
            "transport_from_previous_minutes": transport_minutes,
            **_schedule_item(second, second_start, second_end),
        },
    ]


def _schedule_item(poi: PoiWindow, start: int, end: int) -> Dict[str, Any]:
    return {
        "poi": poi.name,
        "start_time": _format_time(start),
        "end_time": _format_time(end),
        "duration_minutes": poi.duration_minutes,
        "duration_source": poi.duration_source,
        "opening_hours": {
            "opentime": _format_time(poi.open_minute),
            "endtime": _format_time(min(poi.close_minute, 24 * 60)),
            **poi.evidence,
        },
    }


def _feasible(
    *,
    period: str,
    constraint_id: Any,
    schedule: List[Dict[str, Any]],
    transport_minutes: Optional[int] = None,
    attempted_orders: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = {
        "pass": True,
        "skipped": False,
        "status": "feasible",
        "reason_code": "ok",
        "errors": [],
        "blocking_reasons": [],
        "constructive_schedule": {
            "time_period": period,
            "window": [_format_time(TIME_PERIOD_WINDOWS[period][0]), _format_time(TIME_PERIOD_WINDOWS[period][1])],
            "activities": schedule,
        },
        "constraint_id": constraint_id,
    }
    if transport_minutes is not None:
        payload["constructive_schedule"]["transport_minutes"] = transport_minutes
    if attempted_orders is not None:
        payload["attempted_orders"] = attempted_orders
    return payload


def _skipped(reason_code: str, message: str) -> Dict[str, Any]:
    return {
        "pass": True,
        "skipped": True,
        "status": "skipped",
        "reason_code": reason_code,
        "reason": message,
        "errors": [],
        "blocking_reasons": [],
        "constructive_schedule": None,
    }


def _unknown(
    reason_code: str,
    message: str,
    repair_action: str = "manual_review",
    constraint_id: Any = None,
) -> Dict[str, Any]:
    return {
        "pass": False,
        "skipped": False,
        "status": "unknown",
        "reason_code": reason_code,
        "reason": message,
        "errors": [message],
        "blocking_reasons": [
            {
                "status": "unknown_needs_probe",
                "code": reason_code,
                "message": message,
                "constraint_id": constraint_id,
                "repair_action": repair_action,
            }
        ],
        "constructive_schedule": None,
    }


def _infeasible(
    reason_code: str,
    message: str,
    repair_action: str,
    constraint_id: Any = None,
    lower_bounds: Optional[Dict[str, Any]] = None,
    attempted_orders: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result = {
        "pass": False,
        "skipped": False,
        "status": "infeasible",
        "reason_code": reason_code,
        "reason": message,
        "errors": [message],
        "blocking_reasons": [
            {
                "status": "infeasible_by_capacity",
                "code": reason_code,
                "message": message,
                "constraint_id": constraint_id,
                "repair_action": repair_action,
            }
        ],
        "constructive_schedule": None,
    }
    if lower_bounds:
        result["lower_bounds"] = lower_bounds
    if attempted_orders is not None:
        result["attempted_orders"] = attempted_orders
    return result
