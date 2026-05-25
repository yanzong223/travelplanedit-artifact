"""Constructive feasibility checks for attraction commute-cap insertion edits."""

from __future__ import annotations

from itertools import permutations
from math import asin, ceil, cos, radians, sin, sqrt
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHINATRAVEL_PATH = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
if str(CHINATRAVEL_PATH) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_PATH))


ATTRACTION_TYPES = {"attraction"}


class _DefaultCatalog:
    def __init__(self) -> None:
        self._poi = None

    @property
    def poi(self):
        if self._poi is None:
            from chinatravel.environment.tools.poi.apis import Poi

            self._poi = Poi()
        return self._poi


def assess_commute_cap_insertion_feasibility(
    *,
    origin_plan: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    edit_query: str = "",
    fallback_params: Optional[Dict[str, Any]] = None,
    catalog: Optional[Any] = None,
) -> Dict[str, Any]:
    """Check whether a newly required attraction can satisfy adjacent commute cap.

    This is intentionally deterministic and conservative. It filters clearly
    impossible samples; it does not call routing APIs or claim exact solvability.
    """

    constraints = constraints if isinstance(constraints, dict) else {}
    trace = query_generation_trace if isinstance(query_generation_trace, dict) else {}
    fallback = fallback_params if isinstance(fallback_params, dict) else {}
    catalog = catalog or _DefaultCatalog()

    threshold = _extract_threshold(edit_target_constraints, constraints, trace, fallback, edit_query)
    required_names = _extract_required_attractions(edit_target_constraints, constraints, trace, fallback)
    if threshold is None or not required_names:
        return {
            "applicable": False,
            "skipped": True,
            "pass": True,
            "status": "not_applicable",
            "reason_code": "not_applicable",
            "reason": "requires required_attraction_name and adjacent_travel_time_cap",
        }

    target_city = str(origin_plan.get("target_city") or _extract_city(constraints, trace, fallback) or "").strip()
    if not target_city:
        return _unknown("missing_target_city", "target city missing", threshold, required_names)

    days = _attraction_days(origin_plan)
    if not days:
        return _unknown("missing_origin_attractions", "origin plan has no attraction sequence", threshold, required_names)

    all_results: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    for name in required_names:
        poi_coord = _resolve_coord(catalog, target_city, name)
        if poi_coord is None:
            blocking.append(_block("unknown_needs_probe", "new_poi_coord_missing", f"missing coordinate for {name}", name, threshold))
            all_results.append({"required_poi": name, "status": "unknown_needs_probe", "reason_code": "new_poi_coord_missing"})
            continue
        candidate = _assess_single_poi(catalog, target_city, days, name, poi_coord, threshold)
        all_results.append(candidate)
        if candidate.get("pass") is not True:
            code = str(candidate.get("reason_code") or "commute_cap_insertion_infeasible")
            status = "unknown_needs_probe" if code.endswith("_missing") or code.startswith("missing_") else "infeasible_by_spatial_bound"
            blocking.append(_block(status, code, str(candidate.get("reason") or code), name, threshold, candidate))

    pass_all = not blocking
    status = "feasible_by_constructive_probe" if pass_all else _status_from_blocks(blocking)
    reason_code = "ok" if pass_all else str(blocking[0].get("code") or "commute_cap_insertion_infeasible")
    return {
        "applicable": True,
        "skipped": False,
        "pass": pass_all,
        "status": status,
        "reason_code": reason_code,
        "reason": "ok" if pass_all else str(blocking[0].get("message") or reason_code),
        "threshold_minutes": threshold,
        "required_attractions": required_names,
        "target_city": target_city,
        "blocking_reasons": blocking,
        "results": all_results,
    }


def _assess_single_poi(
    catalog: Any,
    city: str,
    days: Sequence[Dict[str, Any]],
    poi_name: str,
    poi_coord: Tuple[float, float],
    threshold: int,
) -> Dict[str, Any]:
    best_failure: Optional[Dict[str, Any]] = None
    unresolved: List[str] = []
    for day in days:
        resolved: List[Dict[str, Any]] = []
        for item in day["attractions"]:
            coord = _resolve_coord(catalog, city, item["name"])
            if coord is None:
                unresolved.append(str(item["name"]))
                continue
            resolved.append({**item, "coord": coord})
        if len(resolved) != len(day["attractions"]):
            continue
        strict = _strict_insertions(day["day"], resolved, poi_name, poi_coord, threshold)
        for candidate in strict:
            if candidate["pass"]:
                return _success(poi_name, threshold, candidate)
            best_failure = _better_failure(best_failure, candidate)
        if 1 < len(resolved) <= 7:
            reorder = _reorder_probe(day["day"], resolved, poi_name, poi_coord, threshold)
            if reorder["pass"]:
                return _success(poi_name, threshold, reorder)
            best_failure = _better_failure(best_failure, reorder)

    if unresolved:
        return {
            "required_poi": poi_name,
            "pass": False,
            "status": "unknown_needs_probe",
            "reason_code": "origin_poi_coord_missing",
            "reason": f"missing coordinates for origin attractions: {', '.join(sorted(set(unresolved)))}",
            "unresolved_origin_pois": sorted(set(unresolved)),
            "threshold_minutes": threshold,
        }
    return {
        "required_poi": poi_name,
        "pass": False,
        "status": "infeasible_by_spatial_bound",
        "reason_code": "commute_cap_insertion_infeasible",
        "reason": f"no insertion or local reorder keeps adjacent attraction commute within {threshold} minutes",
        "threshold_minutes": threshold,
        "best_candidate": best_failure,
    }


def _strict_insertions(
    day: int,
    attractions: Sequence[Dict[str, Any]],
    poi_name: str,
    poi_coord: Tuple[float, float],
    threshold: int,
) -> List[Dict[str, Any]]:
    if not attractions:
        return [{"pass": True, "mode": "strict_insert", "day": day, "insert_index": 0, "edges": []}]
    candidates: List[Dict[str, Any]] = []
    for index in range(len(attractions) + 1):
        edges: List[Dict[str, Any]] = []
        if index > 0:
            prev = attractions[index - 1]
            edges.append(_edge(prev["name"], prev["coord"], poi_name, poi_coord))
        if index < len(attractions):
            nxt = attractions[index]
            edges.append(_edge(poi_name, poi_coord, nxt["name"], nxt["coord"]))
        candidates.append(
            {
                "pass": all(edge["estimated_minutes"] <= threshold for edge in edges),
                "mode": "strict_insert",
                "day": day,
                "insert_index": index,
                "insert_between": [
                    attractions[index - 1]["name"] if index > 0 else None,
                    attractions[index]["name"] if index < len(attractions) else None,
                ],
                "edges": edges,
                "max_estimated_minutes": max((edge["estimated_minutes"] for edge in edges), default=0),
            }
        )
    return candidates


def _reorder_probe(
    day: int,
    attractions: Sequence[Dict[str, Any]],
    poi_name: str,
    poi_coord: Tuple[float, float],
    threshold: int,
) -> Dict[str, Any]:
    nodes = [{"name": item["name"], "coord": item["coord"]} for item in attractions] + [{"name": poi_name, "coord": poi_coord}]
    best: Optional[Dict[str, Any]] = None
    for order in permutations(nodes):
        edges = [_edge(order[index]["name"], order[index]["coord"], order[index + 1]["name"], order[index + 1]["coord"]) for index in range(len(order) - 1)]
        candidate = {
            "pass": all(edge["estimated_minutes"] <= threshold for edge in edges),
            "mode": "local_reorder",
            "day": day,
            "order": [item["name"] for item in order],
            "edges": edges,
            "max_estimated_minutes": max((edge["estimated_minutes"] for edge in edges), default=0),
        }
        if candidate["pass"]:
            return candidate
        best = _better_failure(best, candidate)
    return best or {"pass": False, "mode": "local_reorder", "day": day, "edges": [], "max_estimated_minutes": 0}


def _success(poi_name: str, threshold: int, candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "required_poi": poi_name,
        "pass": True,
        "status": "feasible_by_constructive_probe",
        "reason_code": "ok",
        "reason": "ok",
        "threshold_minutes": threshold,
        "best_candidate": candidate,
    }


def _better_failure(left: Optional[Dict[str, Any]], right: Dict[str, Any]) -> Dict[str, Any]:
    if left is None:
        return right
    return right if int(right.get("max_estimated_minutes") or 10**9) < int(left.get("max_estimated_minutes") or 10**9) else left


def _edge(left_name: str, left: Tuple[float, float], right_name: str, right: Tuple[float, float]) -> Dict[str, Any]:
    distance = _haversine_km(left, right)
    if distance <= 1.2:
        minutes = ceil(distance / 4.5 * 60 + 3)
        mode = "walk"
    else:
        minutes = ceil(distance / 24.0 * 60 + 8)
        mode = "taxi"
    return {
        "from": left_name,
        "to": right_name,
        "distance_km": round(distance, 3),
        "estimated_minutes": int(minutes),
        "mode": mode,
    }


def _haversine_km(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    lat1, lon1 = radians(left[0]), radians(left[1])
    lat2, lon2 = radians(right[0]), radians(right[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0088 * 2 * asin(sqrt(a))


def _resolve_coord(catalog: Any, city: str, name: str) -> Optional[Tuple[float, float]]:
    if not city or not name:
        return None
    try:
        result = catalog.poi.search(city, name)
    except Exception:
        return None
    if isinstance(result, str) or not isinstance(result, (list, tuple)) or len(result) != 2:
        return None
    try:
        return float(result[0]), float(result[1])
    except (TypeError, ValueError):
        return None


def _attraction_days(origin_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    days: List[Dict[str, Any]] = []
    for day_index, day in enumerate(origin_plan.get("itinerary") or [], start=1):
        if not isinstance(day, dict):
            continue
        day_number = _as_int(day.get("day")) or day_index
        attractions: List[Dict[str, Any]] = []
        for activity_index, activity in enumerate(day.get("activities") or []):
            if not isinstance(activity, dict) or str(activity.get("type") or "") not in ATTRACTION_TYPES:
                continue
            name = str(activity.get("position") or activity.get("name") or activity.get("end") or "").strip()
            if name:
                attractions.append({"name": name, "activity_index": activity_index})
        if attractions:
            days.append({"day": day_number, "attractions": attractions})
    return days


def _extract_threshold(
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]],
    constraints: Dict[str, Any],
    trace: Dict[str, Any],
    fallback: Dict[str, Any],
    edit_query: str,
) -> Optional[int]:
    for item in edit_target_constraints or []:
        if isinstance(item, dict) and item.get("type") == "adjacent_travel_time_cap":
            value = _parse_minutes(item.get("value"))
            if value is not None:
                return value
    for item in constraints.get("temporal") or []:
        if isinstance(item, dict):
            for key in ("max_duration", "duration_limit", "duration", "max_transport_duration"):
                value = _parse_minutes(item.get(key))
                if value is not None:
                    return value
    for key in ("duration", "duration_limit_min", "max_duration", "commute_time_min"):
        value = _parse_minutes(fallback.get(key))
        if value is not None:
            return value
    for container_key in ("rendered_params", "canonical_query_spec", "surface_query_spec", "semantic_slots"):
        container = trace.get(container_key)
        if isinstance(container, dict):
            for key in ("duration", "duration_limit_min", "max_duration", "commute_time_min"):
                value = _parse_minutes(container.get(key))
                if value is not None:
                    return value
    return _parse_minutes(edit_query)


def _extract_required_attractions(
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]],
    constraints: Dict[str, Any],
    trace: Dict[str, Any],
    fallback: Dict[str, Any],
) -> List[str]:
    names: List[str] = []
    for item in edit_target_constraints or []:
        if not isinstance(item, dict) or item.get("type") != "required_attraction_name":
            continue
        value = item.get("value")
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            names.extend(str(part) for part in value if str(part or "").strip())
    for item in constraints.get("must_include") or []:
        if isinstance(item, dict) and str(item.get("type") or "attraction") == "attraction":
            names.append(str(item.get("name") or ""))
    for key in ("poi_name", "anchor_poi", "required_attraction_name"):
        value = fallback.get(key)
        if isinstance(value, str) and value.strip():
            names.append(value)
    for container_key in ("canonical_query_spec", "surface_query_spec", "semantic_slots"):
        container = trace.get(container_key)
        if isinstance(container, dict):
            value = container.get("anchor_poi") or container.get("poi_name")
            if isinstance(value, str) and value.strip():
                names.append(value)
    return _dedupe(names)


def _extract_city(constraints: Dict[str, Any], trace: Dict[str, Any], fallback: Dict[str, Any]) -> Optional[str]:
    spatial = constraints.get("spatial") if isinstance(constraints.get("spatial"), dict) else {}
    for value in (spatial.get("city"), fallback.get("target_city"), trace.get("target_city")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_minutes(value: Any) -> Optional[int]:
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
    return int(number)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _unknown(reason_code: str, reason: str, threshold: Optional[int], required_names: Sequence[str]) -> Dict[str, Any]:
    return {
        "applicable": True,
        "skipped": False,
        "pass": False,
        "status": "unknown_needs_probe",
        "reason_code": reason_code,
        "reason": reason,
        "threshold_minutes": threshold,
        "required_attractions": list(required_names),
        "blocking_reasons": [_block("unknown_needs_probe", reason_code, reason, "", threshold)],
    }


def _block(status: str, code: str, message: str, poi_name: str, threshold: Optional[int], detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "status": status,
        "code": code,
        "message": message,
        "constraint_type": "adjacent_travel_time_cap",
        "repair_action": "relax_spatial_threshold" if status == "infeasible_by_spatial_bound" else "manual_review",
        "required_poi": poi_name,
        "threshold_minutes": threshold,
    }
    if detail:
        payload["detail"] = detail
    return payload


def _status_from_blocks(blocking: Sequence[Dict[str, Any]]) -> str:
    if any(item.get("status") == "infeasible_by_spatial_bound" for item in blocking):
        return "infeasible_by_spatial_bound"
    return "unknown_needs_probe"
