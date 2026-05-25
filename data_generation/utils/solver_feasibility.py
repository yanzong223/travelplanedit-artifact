"""Static solver-feasibility checks for generated ChinaTravel edit samples.

The checks in this module are intentionally conservative: they only fail a
sample when a necessary condition is violated. Passing the static checks means
"no blocker was proven", not "a complete itinerary is guaranteed to exist".
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHINATRAVEL_PATH = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
if str(CHINATRAVEL_PATH) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_PATH))
UTILS_PATH = PROJECT_ROOT / "data_generation" / "utils"
if str(UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(UTILS_PATH))

from temporal_window_feasibility import assess_temporal_window_feasibility  # noqa: E402
from commute_cap_feasibility import assess_commute_cap_insertion_feasibility  # noqa: E402


SOLVER_FEASIBILITY_STATUSES = {
    "feasible_by_static_checks",
    "feasible_by_constructive_probe",
    "infeasible_by_lower_bound",
    "infeasible_by_inventory",
    "infeasible_by_capacity",
    "infeasible_by_spatial_bound",
    "unknown_needs_probe",
}

INTERCITY_TYPES = {"train", "airplane"}
MEAL_TYPES = {"breakfast", "lunch", "dinner", "meal"}


@dataclass(frozen=True)
class ConstraintView:
    constraint_id: str
    type: str
    operator: str
    value: Any
    target: Dict[str, Any]
    params: Dict[str, Any]
    source: str = ""


class ChinaTravelCatalog:
    """Lazy access to ChinaTravel static data tables."""

    def __init__(self) -> None:
        self._intercity = None
        self._accommodations = None
        self._attractions = None
        self._restaurants = None
        self._poi = None
        self._transportation = None

    @property
    def intercity(self):
        if self._intercity is None:
            from chinatravel.environment.tools.intercity_transport.apis import IntercityTransport

            self._intercity = IntercityTransport()
        return self._intercity

    @property
    def accommodations(self):
        if self._accommodations is None:
            from chinatravel.environment.tools.accommodations.apis import Accommodations

            self._accommodations = Accommodations()
        return self._accommodations

    @property
    def attractions(self):
        if self._attractions is None:
            from chinatravel.environment.tools.attractions.apis import Attractions

            self._attractions = Attractions()
        return self._attractions

    @property
    def restaurants(self):
        if self._restaurants is None:
            from chinatravel.environment.tools.restaurants.apis import Restaurants

            self._restaurants = Restaurants()
        return self._restaurants

    @property
    def poi(self):
        if self._poi is None:
            from chinatravel.environment.tools.poi.apis import Poi

            self._poi = Poi()
        return self._poi

    @property
    def transportation(self):
        if self._transportation is None:
            from chinatravel.environment.tools.transportation.apis import Transportation

            self._transportation = Transportation()
        return self._transportation


def assess_solver_feasibility(
    *,
    origin_plan: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    origin_logical_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    effective_logical_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    canonical_constraint_ir: Optional[Dict[str, Any]] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    conflict_set: Optional[Sequence[Sequence[str]]] = None,
    primary_conflict: Optional[Sequence[str]] = None,
    target_bucket: Optional[Sequence[str]] = None,
    temporal_window_feasibility: Optional[Dict[str, Any]] = None,
    catalog: Optional[ChinaTravelCatalog] = None,
) -> Dict[str, Any]:
    catalog = catalog or ChinaTravelCatalog()
    constraints = constraints if isinstance(constraints, dict) else {}
    edit_views = _constraint_views(edit_target_constraints)
    origin_views = _constraint_views(origin_logical_constraints)
    effective_views = _constraint_views(effective_logical_constraints)
    all_views = effective_views if effective_views else origin_views + edit_views

    blocking: List[Dict[str, Any]] = []
    lower_bounds: Dict[str, Any] = {}
    inventory = _check_inventory(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_constraints=all_views,
        catalog=catalog,
    )
    capacity = _check_temporal_capacity(
        origin_plan=origin_plan,
        edit_constraints=all_views,
    )
    spatial = _check_spatial_bounds(
        origin_plan=origin_plan,
        edit_constraints=all_views,
        catalog=catalog,
    )
    commute_cap = assess_commute_cap_insertion_feasibility(
        origin_plan=origin_plan,
        constraints=constraints,
        edit_target_constraints=all_views_to_dicts(edit_views),
        query_generation_trace=query_generation_trace,
        catalog=catalog,
    )
    temporal_window = (
        temporal_window_feasibility
        if isinstance(temporal_window_feasibility, dict)
        else assess_temporal_window_feasibility(
            origin_plan=origin_plan,
            constraints=constraints,
            edit_target_constraints=all_views_to_dicts(all_views),
            canonical_constraint_ir=canonical_constraint_ir,
            query_generation_trace=query_generation_trace,
            catalog=catalog,
        )
    )
    budget = _check_budget_lower_bound(
        origin_plan=origin_plan,
        edit_constraints=all_views,
        all_constraints=all_views,
        constraints=constraints,
        catalog=catalog,
    )

    for section_name, section in (
        ("budget", budget),
        ("inventory", inventory),
        ("temporal_capacity", capacity),
        ("temporal_window", {"blocking_reasons": _temporal_window_blocking_reasons(temporal_window), "lower_bounds": temporal_window}),
        ("spatial", spatial),
        ("commute_cap_insertion", {"blocking_reasons": _commute_cap_blocking_reasons(commute_cap), "lower_bounds": commute_cap}),
    ):
        lower_bounds[section_name] = section.get("lower_bounds", {})
        blocking.extend(section.get("blocking_reasons", []))

    category_issue = _check_category_alignment(
        conflict_set=conflict_set,
        primary_conflict=primary_conflict,
        target_bucket=target_bucket,
    )
    if category_issue:
        blocking.append(category_issue)

    status = _derive_status(blocking)
    return {
        "pass": status in {"feasible_by_static_checks", "feasible_by_constructive_probe"},
        "status": status,
        "blocking_reasons": blocking,
        "lower_bounds": lower_bounds,
        "repair_recommendation": _repair_recommendation(blocking),
        "constructive_probe": {
            "attempted": False,
            "status": "not_run",
            "reason": "static_audit_only",
        },
        "notes": [
            "Static checks are conservative necessary-condition checks.",
            "A pass means no static blocker was proven; exact solvability may still require RuleDFS probe.",
        ],
        "temporal_window_constructive_pass": (
            temporal_window.get("pass") is True and temporal_window.get("skipped") is not True
        ),
        "commute_cap_constructive_pass": (
            commute_cap.get("pass") is True and commute_cap.get("skipped") is not True
        ),
        "inputs_summary": {
            "edit_constraint_types": sorted({item.type for item in edit_views}),
            "origin_constraint_types": sorted({item.type for item in origin_views}),
            "effective_constraint_types": sorted({item.type for item in effective_views}),
            "constraint_type": _trace_constraint_type(query_generation_trace, canonical_constraint_ir),
        },
    }


def all_views_to_dicts(views: Sequence[ConstraintView]) -> List[Dict[str, Any]]:
    return [
        {
            "id": item.constraint_id,
            "source": item.source,
            "type": item.type,
            "operator": item.operator,
            "value": item.value,
            "target": item.target,
            "params": item.params,
        }
        for item in views
    ]


def _temporal_window_blocking_reasons(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict) or result.get("skipped") is True or result.get("pass") is True:
        return []
    blocking = result.get("blocking_reasons")
    if isinstance(blocking, list) and blocking:
        return [item for item in blocking if isinstance(item, dict)]
    status = "unknown_needs_probe" if result.get("status") == "unknown" else "infeasible_by_capacity"
    return [
        {
            "status": status,
            "code": result.get("reason_code") or "temporal_window_feasibility_failed",
            "message": result.get("reason") or "temporal-window feasibility failed",
            "repair_action": "relax_time_window",
        }
    ]


def _commute_cap_blocking_reasons(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict) or result.get("applicable") is not True or result.get("pass") is True:
        return []
    blocking = result.get("blocking_reasons")
    if isinstance(blocking, list) and blocking:
        return [item for item in blocking if isinstance(item, dict)]
    status = str(result.get("status") or "unknown_needs_probe")
    if status not in SOLVER_FEASIBILITY_STATUSES:
        status = "unknown_needs_probe"
    return [
        {
            "status": status,
            "code": result.get("reason_code") or "commute_cap_insertion_failed",
            "message": result.get("reason") or "commute-cap insertion feasibility failed",
            "repair_action": "relax_spatial_threshold" if status == "infeasible_by_spatial_bound" else "manual_review",
        }
    ]


def _constraint_views(items: Optional[Sequence[Dict[str, Any]]]) -> List[ConstraintView]:
    views: List[ConstraintView] = []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return views
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        views.append(
            ConstraintView(
                constraint_id=str(item.get("id", f"constraint_{index}")),
                type=str(item.get("type", item.get("constraint_type", ""))),
                operator=str(item.get("operator", item.get("predicate", "=="))),
                value=item.get("value"),
                target=dict(item.get("target", {})),
                params=dict(item.get("params", item.get("parameters", {}))),
                source=str(item.get("source", "")),
            )
        )
    return views


def _trace_constraint_type(
    query_generation_trace: Optional[Dict[str, Any]],
    canonical_constraint_ir: Optional[Dict[str, Any]],
) -> Optional[str]:
    if isinstance(query_generation_trace, dict) and query_generation_trace.get("constraint_type"):
        return str(query_generation_trace.get("constraint_type"))
    if isinstance(canonical_constraint_ir, dict):
        source = canonical_constraint_ir.get("source")
        if isinstance(source, dict) and source.get("constraint_type"):
            return str(source.get("constraint_type"))
    return None


def _derive_status(blocking: Sequence[Dict[str, Any]]) -> str:
    if not blocking:
        return "feasible_by_static_checks"
    priorities = [
        "infeasible_by_lower_bound",
        "infeasible_by_inventory",
        "infeasible_by_capacity",
        "infeasible_by_spatial_bound",
    ]
    reason_statuses = {str(item.get("status", "")) for item in blocking}
    for status in priorities:
        if status in reason_statuses:
            return status
    return "unknown_needs_probe"


def _repair_recommendation(blocking: Sequence[Dict[str, Any]]) -> str:
    actions = [str(item.get("repair_action", "")) for item in blocking]
    for action in ("raise_budget", "replace_poi", "increase_days", "relax_time_window", "relax_spatial_threshold"):
        if action in actions:
            return action
    if any(action == "reclassify_or_filter" for action in actions):
        return "manual_review"
    if blocking:
        return "manual_review"
    return "none"


def _check_budget_lower_bound(
    *,
    origin_plan: Dict[str, Any],
    edit_constraints: Sequence[ConstraintView],
    all_constraints: Sequence[ConstraintView],
    constraints: Dict[str, Any],
    catalog: ChinaTravelCatalog,
) -> Dict[str, Any]:
    anchor_bundle = _anchor_bundle_budget_request(edit_constraints, constraints)
    if anchor_bundle is not None:
        return _check_anchor_bundle_budget_lower_bound(
            origin_plan=origin_plan,
            edit_constraints=edit_constraints,
            constraints=constraints,
            catalog=catalog,
            anchor_poi=anchor_bundle[0],
            budget=anchor_bundle[1],
            max_distance_km=anchor_bundle[2],
            selected_meal_name=anchor_bundle[3],
        )

    budget = _first_numeric(
        [c.value for c in reversed(edit_constraints) if c.type == "budget_total"]
        + [_budget_from_structured_constraints(constraints)]
    )
    if budget is None:
        return {"blocking_reasons": [], "lower_bounds": {"applicable": False}}

    people = _resolve_people_count(origin_plan, all_constraints)
    days = _resolve_day_count(origin_plan, edit_constraints)
    start_city = str(origin_plan.get("start_city", "") or "")
    target_city = str(origin_plan.get("target_city", "") or "")
    required_modes = _required_intercity_modes(all_constraints)

    components: Dict[str, Any] = {
        "budget_threshold": budget,
        "people_number": people,
        "resolved_days": days,
        "intercity_modes_considered": sorted(required_modes) if required_modes else sorted(INTERCITY_TYPES),
    }
    unknown: List[str] = []

    intercity_min = _min_roundtrip_intercity_cost(
        catalog=catalog,
        start_city=start_city,
        target_city=target_city,
        people=people,
        required_modes=required_modes,
    )
    if intercity_min is None:
        unknown.append("intercity_min")
        intercity_min = 0.0
    accommodation_min = _min_required_accommodation_cost(
        catalog=catalog,
        target_city=target_city,
        days=days,
        people=people,
        constraints=all_constraints,
    )
    if accommodation_min is None:
        unknown.append("accommodation_min")
        accommodation_min = 0.0
    required_ticket_min = _min_required_ticket_cost(
        catalog=catalog,
        target_city=target_city,
        people=people,
        constraints=all_constraints,
    )
    if required_ticket_min is None:
        unknown.append("required_ticket_min")
        required_ticket_min = 0.0
    meal_bound = _min_required_meal_cost(
        catalog=catalog,
        target_city=target_city,
        days=days,
        people=people,
        constraints=all_constraints,
    )
    meal_min = meal_bound.get("cost")
    if meal_min is None:
        unknown.append("meal_min")
        meal_min = 0.0

    total = float(intercity_min) + float(accommodation_min) + float(required_ticket_min) + float(meal_min)
    components.update(
        {
            "intercity_min": round(float(intercity_min), 2),
            "accommodation_min": round(float(accommodation_min), 2),
            "required_ticket_min": round(float(required_ticket_min), 2),
            "meal_min": round(float(meal_min), 2),
            "meal_count_min": meal_bound.get("meal_count_min", 0),
            "meal_unit_min": meal_bound.get("meal_unit_min"),
            "meal_min_policy": meal_bound.get("policy"),
            "explicit_restaurant_requirements": meal_bound.get("explicit_requirements", {}),
            "local_transport_min_assumed": 0.0,
            "lower_bound_total": round(total, 2),
            "unknown_components": unknown,
        }
    )

    blocking: List[Dict[str, Any]] = []
    if not unknown and total > budget + 1e-6:
        blocking.append(
            {
                "status": "infeasible_by_lower_bound",
                "code": "budget_below_static_lower_bound",
                "message": f"budget {budget:.2f} is below static lower bound {total:.2f}",
                "constraint_type": "budget_total",
                "budget_threshold": budget,
                "lower_bound_total": round(total, 2),
                "components": components,
                "repair_action": "raise_budget",
                "suggested_min_budget": _round_budget(total * 1.05),
            }
        )
    elif unknown:
        components["needs_probe"] = True
    return {"blocking_reasons": blocking, "lower_bounds": components}


def _anchor_bundle_budget_request(
    edit_constraints: Sequence[ConstraintView],
    constraints: Dict[str, Any],
) -> Optional[Tuple[str, float, float, str]]:
    for constraint in edit_constraints:
        if constraint.type != "anchor_bundle_budget_limit":
            continue
        budget = _as_float(constraint.value)
        max_distance = _as_float(constraint.params.get("max_distance_km")) or 3.0
        selected_meal = str(constraint.params.get("selected_meal_name") or "").strip()
        anchor = str(
            constraint.target.get("anchor_poi")
            or constraint.target.get("poi_name")
            or constraint.params.get("anchor_poi")
            or ""
        ).strip()
        if budget is not None and anchor:
            return anchor, budget, max_distance, selected_meal

    resource = constraints.get("resource")
    if not isinstance(resource, dict):
        return None
    if str(resource.get("budget_scope") or "").strip() != "anchor_bundle":
        return None
    anchor = str(resource.get("overlap_anchor_poi") or "").strip()
    budget = _first_numeric(
        [
            resource.get("anchor_bundle_budget_limit"),
            resource.get("total_budget"),
            resource.get("budget"),
            resource.get("budget_cap"),
            resource.get("max_budget"),
        ]
    )
    if budget is None or not anchor:
        return None
    max_distance = _as_float(resource.get("max_distance_km")) or 3.0
    selected_meal = str(resource.get("selected_meal_name") or "").strip()
    return anchor, budget, max_distance, selected_meal


def _check_anchor_bundle_budget_lower_bound(
    *,
    origin_plan: Dict[str, Any],
    edit_constraints: Sequence[ConstraintView],
    constraints: Dict[str, Any],
    catalog: ChinaTravelCatalog,
    anchor_poi: str,
    budget: float,
    max_distance_km: float,
    selected_meal_name: str,
) -> Dict[str, Any]:
    people = _resolve_people_count(origin_plan, edit_constraints)
    target_city = str(origin_plan.get("target_city", "") or "")
    resource = constraints.get("resource") if isinstance(constraints.get("resource"), dict) else {}
    components: Dict[str, Any] = {
        "applicable": True,
        "budget_scope": "anchor_bundle",
        "budget_threshold": budget,
        "people_number": people,
        "target_city": target_city,
        "anchor_poi": anchor_poi,
        "requires_nearby_meal": bool(resource.get("nearby_meal", True)),
        "max_distance_km": max_distance_km,
    }
    if selected_meal_name:
        components["selected_meal_name"] = selected_meal_name
    unknown: List[str] = []
    blocking: List[Dict[str, Any]] = []

    ticket_min = _min_anchor_ticket_cost(
        catalog=catalog,
        target_city=target_city,
        anchor_poi=anchor_poi,
        people=people,
    )
    if ticket_min is None:
        unknown.append("anchor_ticket_min")
        ticket_min = 0.0
    meal_min = _min_nearby_meal_cost(
        catalog=catalog,
        target_city=target_city,
        anchor_poi=anchor_poi,
        people=people,
        max_distance_km=max_distance_km,
        selected_meal_name=selected_meal_name,
    )
    if meal_min is None:
        blocking.append(
            {
                "status": "infeasible_by_spatial_bound",
                "code": "missing_nearby_meal_within_distance",
                "message": f"no restaurant within {max_distance_km:.2f}km of {anchor_poi}",
                "constraint_type": "anchor_bundle_budget_limit",
                "budget_scope": "anchor_bundle",
                "anchor_poi": anchor_poi,
                "max_distance_km": max_distance_km,
                "repair_action": "relax_spatial_threshold",
            }
        )
        if selected_meal_name:
            blocking[-1]["selected_meal_name"] = selected_meal_name
        meal_min = 0.0

    total = float(ticket_min) + float(meal_min)
    components.update(
        {
            "anchor_ticket_min": round(float(ticket_min), 2),
            "nearby_meal_min": round(float(meal_min), 2),
            "lower_bound_total": round(total, 2),
            "unknown_components": unknown,
            "excluded_components": [
                "intercity_min",
                "accommodation_min",
                "required_trip_meals",
                "other_required_ticket_min",
            ],
        }
    )

    if not unknown and total > budget + 1e-6:
        blocking.append(
            {
                "status": "infeasible_by_lower_bound",
                "code": "anchor_bundle_budget_below_static_lower_bound",
                "message": f"anchor bundle budget {budget:.2f} is below static lower bound {total:.2f}",
                "constraint_type": "anchor_bundle_budget_limit",
                "budget_scope": "anchor_bundle",
                "budget_threshold": budget,
                "lower_bound_total": round(total, 2),
                "components": components,
                "repair_action": "raise_budget",
                "suggested_min_budget": _round_budget(total * 1.05),
            }
        )
    elif unknown:
        components["needs_probe"] = True
    return {"blocking_reasons": blocking, "lower_bounds": components}


def _min_anchor_ticket_cost(
    *,
    catalog: ChinaTravelCatalog,
    target_city: str,
    anchor_poi: str,
    people: int,
) -> Optional[float]:
    if not target_city or not anchor_poi:
        return None
    try:
        df = catalog.attractions.data[target_city]
    except Exception:
        return None
    if "name" not in df.columns or "price" not in df.columns:
        return None
    row = df[df["name"].astype(str) == anchor_poi]
    if len(row) == 0:
        return None
    return float(row["price"].min()) * int(people)


def _min_single_meal_cost(
    *,
    catalog: ChinaTravelCatalog,
    target_city: str,
    people: int,
) -> Optional[float]:
    if not target_city:
        return None
    try:
        df = catalog.restaurants.data[target_city]
    except Exception:
        return None
    if len(df) == 0 or "price" not in df.columns:
        return None
    prices = df["price"].dropna()
    if len(prices) == 0:
        return None
    return float(prices.min()) * int(people)


def _min_nearby_meal_cost(
    *,
    catalog: ChinaTravelCatalog,
    target_city: str,
    anchor_poi: str,
    people: int,
    max_distance_km: float,
    selected_meal_name: str = "",
) -> Optional[float]:
    if not target_city or not anchor_poi:
        return None
    try:
        df = catalog.restaurants.nearby(target_city, anchor_poi, topk=None, dist=float(max_distance_km))
    except Exception:
        return None
    if isinstance(df, str) or len(df) == 0 or "price" not in df.columns:
        return None
    prices = df["price"].dropna()
    if len(prices) == 0:
        return None
    return float(prices.min()) * int(people)


def _budget_from_structured_constraints(constraints: Dict[str, Any]) -> Optional[float]:
    resource = constraints.get("resource")
    if not isinstance(resource, dict):
        return None
    for key in ("total_budget", "budget", "budget_cap", "max_budget"):
        value = _as_float(resource.get(key))
        if value is not None:
            return value
    return None


def _first_numeric(values: Iterable[Any]) -> Optional[float]:
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    return None


def _as_int(value: Any) -> Optional[int]:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _resolve_people_count(origin_plan: Dict[str, Any], constraints: Sequence[ConstraintView]) -> int:
    for constraint in reversed(constraints):
        if constraint.type == "people_count" and constraint.operator == "==":
            value = _as_int(constraint.value)
            if value and value > 0:
                return value
    value = _as_int(origin_plan.get("people_number"))
    return value if value and value > 0 else 1


def _resolve_day_count(origin_plan: Dict[str, Any], edit_constraints: Sequence[ConstraintView]) -> int:
    for constraint in reversed(edit_constraints):
        if constraint.type == "day_count" and constraint.operator == "==":
            value = _as_int(constraint.value)
            if value and value > 0:
                return value
    itinerary = origin_plan.get("itinerary", [])
    return len(itinerary) if isinstance(itinerary, list) and itinerary else 1


def _required_intercity_modes(constraints: Sequence[ConstraintView]) -> set[str]:
    modes: set[str] = set()
    for constraint in constraints:
        if constraint.type != "required_intercity_transport_type":
            continue
        raw = constraint.value
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        for value in values:
            normalized = str(value or "").strip()
            if normalized in INTERCITY_TYPES:
                modes.add(normalized)
    return modes


def _min_roundtrip_intercity_cost(
    *,
    catalog: ChinaTravelCatalog,
    start_city: str,
    target_city: str,
    people: int,
    required_modes: set[str],
) -> Optional[float]:
    if not start_city or not target_city or start_city == target_city:
        return 0.0
    modes = sorted(required_modes) if required_modes else sorted(INTERCITY_TYPES)
    candidates: List[float] = []
    for mode in modes:
        try:
            outbound = catalog.intercity.select(start_city, target_city, mode)
            inbound = catalog.intercity.select(target_city, start_city, mode)
        except Exception:
            continue
        if not hasattr(outbound, "columns") or not hasattr(inbound, "columns"):
            continue
        if len(outbound) == 0 or len(inbound) == 0:
            continue
        if "Cost" not in outbound.columns or "Cost" not in inbound.columns:
            continue
        candidates.append((float(outbound["Cost"].min()) + float(inbound["Cost"].min())) * int(people))
    return min(candidates) if candidates else None


def _min_required_accommodation_cost(
    *,
    catalog: ChinaTravelCatalog,
    target_city: str,
    days: int,
    people: int,
    constraints: Sequence[ConstraintView],
) -> Optional[float]:
    nights = max(int(days) - 1, 0)
    if nights <= 0:
        return 0.0
    room_count = _required_room_count(constraints)
    room_type = _required_room_type(constraints)
    hotel_feature = _required_hotel_feature(constraints)
    hotel_names = _required_hotel_names(constraints)
    try:
        df = catalog.accommodations.data[target_city]
    except Exception:
        return None
    filtered = df
    if room_type is not None and "numbed" in filtered.columns:
        filtered = filtered[filtered["numbed"] == int(room_type)]
    if hotel_feature and "featurehoteltype" in filtered.columns:
        filtered = filtered[filtered["featurehoteltype"].astype(str).str.contains(str(hotel_feature), regex=False)]
    if hotel_names and "name" in filtered.columns:
        filtered = filtered[filtered["name"].astype(str).isin(hotel_names)]
    if len(filtered) == 0 or "price" not in filtered.columns:
        return None
    if room_count is None:
        capacity = int(room_type) if room_type and room_type > 0 else 2
        room_count = max(1, ceil(int(people) / capacity))
    return float(filtered["price"].min()) * int(room_count) * nights


def _required_room_count(constraints: Sequence[ConstraintView]) -> Optional[int]:
    for constraint in reversed(constraints):
        if constraint.type == "required_room_count":
            value = _as_int(constraint.value)
            if value:
                return value
        code = _constraint_code(constraint)
        match = re.search(r"room_count\(activity\)!=(\d+)", code)
        if match:
            return int(match.group(1))
    return None


def _required_room_type(constraints: Sequence[ConstraintView]) -> Optional[int]:
    for constraint in reversed(constraints):
        if constraint.type == "required_room_type":
            value = _as_int(constraint.value)
            if value:
                return value
        code = _constraint_code(constraint)
        match = re.search(r"room_type\(activity\)!=(\d+)", code)
        if match:
            return int(match.group(1))
    return None


def _constraint_code(constraint: ConstraintView) -> str:
    return str(constraint.params.get("code") or "")


def _required_hotel_feature(constraints: Sequence[ConstraintView]) -> Optional[str]:
    for constraint in reversed(constraints):
        if constraint.type == "required_hotel_feature":
            values = constraint.value if isinstance(constraint.value, list) else [constraint.value]
            for value in values:
                text = str(value or "").strip()
                if text:
                    return text
    return None


def _required_hotel_names(constraints: Sequence[ConstraintView]) -> set[str]:
    names: set[str] = set()
    for constraint in constraints:
        if constraint.type != "required_hotel_name":
            continue
        values = constraint.value if isinstance(constraint.value, list) else [constraint.value]
        names.update(str(value).strip() for value in values if str(value).strip())
    return names


def _min_required_ticket_cost(
    *,
    catalog: ChinaTravelCatalog,
    target_city: str,
    people: int,
    constraints: Sequence[ConstraintView],
) -> Optional[float]:
    names: set[str] = set()
    for constraint in constraints:
        if constraint.type == "required_attraction_name":
            values = constraint.value if isinstance(constraint.value, list) else [constraint.value]
            names.update(str(value).strip() for value in values if str(value).strip())
    if not names:
        return 0.0
    try:
        df = catalog.attractions.data[target_city]
    except Exception:
        return None
    if "name" not in df.columns or "price" not in df.columns:
        return None
    total = 0.0
    for name in names:
        row = df[df["name"].astype(str) == name]
        if len(row) == 0:
            return None
        total += float(row["price"].min()) * int(people)
    return total


def _min_required_meal_cost(
    *,
    catalog: ChinaTravelCatalog,
    target_city: str,
    days: int,
    people: int,
    constraints: Sequence[ConstraintView],
) -> Dict[str, Any]:
    """Return a conservative meal lower bound.

    ChinaTravel allows free hotel breakfast, so this lower bound only assumes
    one paid lunch/dinner-style meal per travel day, plus any explicit required
    restaurants or cuisines that force additional distinct meals.
    """

    base_count = max(int(days), 0)
    policy = "one_paid_meal_per_travel_day_plus_explicit_restaurant_requirements"
    if base_count <= 0:
        return {
            "cost": 0.0,
            "meal_count_min": 0,
            "meal_unit_min": 0.0,
            "policy": policy,
            "explicit_requirements": {},
        }
    try:
        df = catalog.restaurants.data[target_city]
    except Exception:
        return {
            "cost": None,
            "meal_count_min": base_count,
            "meal_unit_min": None,
            "policy": policy,
            "explicit_requirements": {},
        }
    if len(df) == 0 or "price" not in df.columns:
        return {
            "cost": None,
            "meal_count_min": base_count,
            "meal_unit_min": None,
            "policy": policy,
            "explicit_requirements": {},
        }

    prices = df["price"].dropna()
    if len(prices) == 0:
        return {
            "cost": None,
            "meal_count_min": base_count,
            "meal_unit_min": None,
            "policy": policy,
            "explicit_requirements": {},
        }
    generic_unit = float(prices.min())
    required_names = sorted(_required_restaurant_names(constraints))
    required_types = sorted(_required_restaurant_types(constraints))
    explicit_cost = 0.0
    explicit_count = 0
    satisfied_types: set[str] = set()

    for name in required_names:
        if "name" not in df.columns:
            return {
                "cost": None,
                "meal_count_min": base_count,
                "meal_unit_min": round(generic_unit, 2),
                "policy": policy,
                "explicit_requirements": {"restaurant_names": required_names, "restaurant_types": required_types},
            }
        row = df[df["name"].astype(str) == name]
        if len(row) == 0:
            return {
                "cost": None,
                "meal_count_min": base_count,
                "meal_unit_min": round(generic_unit, 2),
                "policy": policy,
                "explicit_requirements": {"restaurant_names": required_names, "restaurant_types": required_types},
            }
        explicit_cost += float(row["price"].min()) * int(people)
        explicit_count += 1
        if "cuisine" in row.columns:
            satisfied_types.update(str(item) for item in row["cuisine"].dropna().unique().tolist())

    remaining_types = [item for item in required_types if item not in satisfied_types]
    for cuisine in remaining_types:
        if "cuisine" not in df.columns:
            return {
                "cost": None,
                "meal_count_min": base_count,
                "meal_unit_min": round(generic_unit, 2),
                "policy": policy,
                "explicit_requirements": {"restaurant_names": required_names, "restaurant_types": required_types},
            }
        rows = df[df["cuisine"].astype(str) == cuisine]
        if len(rows) == 0:
            return {
                "cost": None,
                "meal_count_min": base_count,
                "meal_unit_min": round(generic_unit, 2),
                "policy": policy,
                "explicit_requirements": {"restaurant_names": required_names, "restaurant_types": required_types},
            }
        explicit_cost += float(rows["price"].min()) * int(people)
        explicit_count += 1

    generic_count = max(base_count - explicit_count, 0)
    total = explicit_cost + generic_count * generic_unit * int(people)
    return {
        "cost": total,
        "meal_count_min": explicit_count + generic_count,
        "meal_unit_min": round(generic_unit, 2),
        "policy": policy,
        "explicit_requirements": {
            "restaurant_names": required_names,
            "restaurant_types": required_types,
            "restaurant_types_satisfied_by_required_names": sorted(satisfied_types.intersection(required_types)),
        },
    }


def _required_restaurant_names(constraints: Sequence[ConstraintView]) -> set[str]:
    names: set[str] = set()
    for constraint in constraints:
        if constraint.type != "required_restaurant_name":
            continue
        names.update(_value_list(constraint.value))
    return names


def _required_restaurant_types(constraints: Sequence[ConstraintView]) -> set[str]:
    types: set[str] = set()
    for constraint in constraints:
        if constraint.type != "required_restaurant_type":
            continue
        types.update(_value_list(constraint.value))
    return types


def _round_budget(value: float) -> int:
    return int(ceil(float(value) / 10.0) * 10)


def _check_inventory(
    *,
    origin_plan: Dict[str, Any],
    constraints: Dict[str, Any],
    edit_constraints: Sequence[ConstraintView],
    catalog: ChinaTravelCatalog,
) -> Dict[str, Any]:
    spatial = constraints.get("spatial") if isinstance(constraints.get("spatial"), dict) else {}
    target_city = str(origin_plan.get("target_city", "") or spatial.get("city", "") or "")
    blocking: List[Dict[str, Any]] = []
    inventory: Dict[str, Any] = {"target_city": target_city}
    if not target_city:
        inventory["unknown_components"] = ["target_city"]
        return {"blocking_reasons": [], "lower_bounds": inventory}

    for constraint in edit_constraints:
        if constraint.type == "ticket_price_cap_or_substitute":
            target_eligible = bool(constraint.params.get("target_eligible", False))
            candidates = [str(item) for item in constraint.params.get("candidate_names", []) if str(item)]
            if not target_eligible and not candidates:
                blocking.append(_inventory_block("missing_ticket_substitute_candidates", constraint, "replace_poi"))
        elif constraint.type == "required_attraction_name":
            missing = _missing_names(catalog, "attraction", target_city, constraint.value)
            if missing:
                blocking.append(_inventory_block("missing_required_attraction", constraint, "replace_poi", missing))
        elif constraint.type == "required_hotel_name":
            missing = _missing_names(catalog, "hotel", target_city, constraint.value)
            if missing:
                blocking.append(_inventory_block("missing_required_hotel", constraint, "replace_poi", missing))
        elif constraint.type == "required_restaurant_name":
            missing = _missing_names(catalog, "restaurant", target_city, constraint.value)
            if missing:
                blocking.append(_inventory_block("missing_required_restaurant", constraint, "replace_poi", missing))
        elif constraint.type == "required_hotel_feature":
            missing = _missing_values(catalog, "hotel_feature", target_city, constraint.value)
            if missing:
                blocking.append(_inventory_block("missing_required_hotel_feature", constraint, "replace_poi", missing))
        elif constraint.type == "required_restaurant_type":
            missing = _missing_values(catalog, "restaurant_type", target_city, constraint.value)
            if missing:
                blocking.append(_inventory_block("missing_required_restaurant_type", constraint, "replace_poi", missing))
        elif constraint.type in {"required_attraction_type", "semantic_type_requirement"}:
            missing = _missing_values(catalog, "attraction_type", target_city, constraint.value)
            if missing:
                blocking.append(_inventory_block("missing_required_attraction_type", constraint, "replace_poi", missing))

    return {"blocking_reasons": blocking, "lower_bounds": inventory}


def _inventory_block(
    code: str,
    constraint: ConstraintView,
    repair_action: str,
    missing: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "status": "infeasible_by_inventory",
        "code": code,
        "message": code,
        "constraint_id": constraint.constraint_id,
        "constraint_type": constraint.type,
        "missing": list(missing or []),
        "repair_action": repair_action,
    }


def _value_list(value: Any) -> List[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _missing_names(catalog: ChinaTravelCatalog, kind: str, city: str, value: Any) -> List[str]:
    names = _value_list(value)
    if not names:
        return []
    try:
        if kind == "attraction":
            df = catalog.attractions.data[city]
        elif kind == "hotel":
            df = catalog.accommodations.data[city]
        else:
            df = catalog.restaurants.data[city]
    except Exception:
        return names
    if "name" not in df.columns:
        return names
    available = {str(item) for item in df["name"].tolist()}
    return [name for name in names if name not in available]


def _missing_values(catalog: ChinaTravelCatalog, kind: str, city: str, value: Any) -> List[str]:
    values = _value_list(value)
    if not values:
        return []
    try:
        if kind == "hotel_feature":
            df = catalog.accommodations.data[city]
            column = "featurehoteltype"
        elif kind == "restaurant_type":
            df = catalog.restaurants.data[city]
            column = "cuisine"
        else:
            df = catalog.attractions.data[city]
            column = "type"
    except Exception:
        return values
    if column not in df.columns:
        return values
    available = {str(item) for item in df[column].dropna().unique().tolist()}
    return [value for value in values if value not in available]


def _check_temporal_capacity(
    *,
    origin_plan: Dict[str, Any],
    edit_constraints: Sequence[ConstraintView],
) -> Dict[str, Any]:
    del origin_plan
    day_count = None
    daily_caps: Dict[Optional[int], int] = {}
    required_by_day: Dict[int, set[str]] = {}
    blocking: List[Dict[str, Any]] = []

    for constraint in edit_constraints:
        if constraint.type == "day_count" and constraint.operator == "==":
            day_count = _as_int(constraint.value)
        elif constraint.type == "daily_poi_cap":
            daily_caps[_as_int(constraint.target.get("day"))] = int(_as_int(constraint.value) or 0)
        elif constraint.type == "poi_day_binding":
            day = _as_int(constraint.value)
            name = str(constraint.target.get("poi_name", "") or "").strip()
            if day and name:
                required_by_day.setdefault(day, set()).add(name)

    if day_count is not None and required_by_day:
        max_required_day = max(required_by_day)
        if max_required_day > day_count:
            blocking.append(
                {
                    "status": "infeasible_by_capacity",
                    "code": "required_day_exceeds_day_count",
                    "message": f"required day {max_required_day} exceeds day_count {day_count}",
                    "repair_action": "increase_days",
                    "required_day": max_required_day,
                    "day_count": day_count,
                }
            )

    for day, names in required_by_day.items():
        cap = daily_caps.get(day, daily_caps.get(None))
        if cap is not None and cap >= 0 and len(names) > cap:
            blocking.append(
                {
                    "status": "infeasible_by_capacity",
                    "code": "required_poi_count_exceeds_daily_cap",
                    "message": f"day {day} requires {len(names)} POIs but cap is {cap}",
                    "repair_action": "increase_days",
                    "day": day,
                    "required_poi_count": len(names),
                    "daily_poi_cap": cap,
                }
            )

    return {
        "blocking_reasons": blocking,
        "lower_bounds": {
            "day_count": day_count,
            "required_poi_count_by_day": {str(day): len(names) for day, names in required_by_day.items()},
            "daily_poi_cap": {str(day): cap for day, cap in daily_caps.items()},
        },
    }


def _check_spatial_bounds(
    *,
    origin_plan: Dict[str, Any],
    edit_constraints: Sequence[ConstraintView],
    catalog: ChinaTravelCatalog,
) -> Dict[str, Any]:
    target_city = str(origin_plan.get("target_city", "") or "")
    blocking: List[Dict[str, Any]] = []
    bounds: Dict[str, Any] = {"target_city": target_city}
    if not target_city:
        return {"blocking_reasons": [], "lower_bounds": bounds}

    for constraint in edit_constraints:
        if constraint.type not in {"pairwise_transport_mode_distance_cap", "anchor_neighbor_commute_distance_cap"}:
            continue
        limit = _as_float(constraint.value)
        if limit is None:
            continue
        if constraint.type == "pairwise_transport_mode_distance_cap":
            first = str(constraint.target.get("first_poi", "") or "")
            second = str(constraint.target.get("second_poi", "") or "")
            if not first or not second:
                continue
            distance = _poi_distance_km(catalog, target_city, first, second)
            bounds[f"pair:{first}->{second}"] = {"distance": distance, "limit": limit}
            if distance is not None and distance > limit + 1e-6:
                blocking.append(
                    {
                        "status": "infeasible_by_spatial_bound",
                        "code": "pair_distance_exceeds_cap",
                        "message": f"{first} to {second} distance {distance:.2f}km exceeds cap {limit:.2f}km",
                        "repair_action": "relax_spatial_threshold",
                        "constraint_id": constraint.constraint_id,
                        "distance": round(distance, 3),
                        "limit": limit,
                    }
                )
        elif constraint.type == "anchor_neighbor_commute_distance_cap":
            anchor = str(constraint.target.get("anchor_name") or constraint.target.get("poi_name") or "")
            if not anchor:
                continue
            if _poi_distance_km(catalog, target_city, anchor, anchor) is None:
                blocking.append(_inventory_block("missing_spatial_anchor", constraint, "replace_poi", [anchor]))

    return {"blocking_reasons": blocking, "lower_bounds": bounds}


def _poi_distance_km(catalog: ChinaTravelCatalog, city: str, first: str, second: str) -> Optional[float]:
    try:
        from geopy.distance import geodesic

        left = catalog.poi.search(city, first)
        right = catalog.poi.search(city, second)
        if isinstance(left, str) or isinstance(right, str):
            return None
        return float(geodesic(tuple(left), tuple(right)).km)
    except Exception:
        return None


def _check_category_alignment(
    *,
    conflict_set: Optional[Sequence[Sequence[str]]],
    primary_conflict: Optional[Sequence[str]],
    target_bucket: Optional[Sequence[str]],
) -> Optional[Dict[str, Any]]:
    if not target_bucket:
        return None
    target = [str(item) for item in target_bucket]
    conflict_tuples = [[str(part) for part in item] for item in (conflict_set or [])]
    primary = [str(part) for part in primary_conflict] if primary_conflict else None
    if target in conflict_tuples or target == primary:
        return None
    return {
        "status": "unknown_needs_probe",
        "code": "category_target_mismatch",
        "message": f"target bucket {target} not found in conflict_set",
        "repair_action": "reclassify_or_filter",
        "target_bucket": target,
        "primary_conflict": primary,
        "conflict_set": conflict_tuples,
    }
