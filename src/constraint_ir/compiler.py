"""Compiler from template semantics and structured inputs to canonical constraint IR."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from evaluation.benchmark.compilers import compile_logical_constraint
from evaluation.benchmark.hard_truth import (
    _SUPPORTED_DATASET_CITIES,
    _append_unique_constraint,
    _lookup_named_poi,
    _make_logical,
    _normalize_time_period,
    _normalized_day_poi_pairs,
    _raise_if_transport_hub_name,
    _resolve_city_split_targets,
    _resolve_target_city,
    _same_type_candidates,
    _same_type_price_capped_candidates,
    _semantic_type_by_day,
)
from evaluation.benchmark.models import LogicalConstraintObject
from data_generation.utils.query_surface_parsing import normalize_intercity_transport_mode

from .adapters import logical_constraint_to_ir_node
from .models import CanonicalConstraintIR
from .template_semantics import resolve_template_semantic_spec


_HARD_AXIS_BY_CONSTRAINT_TYPE = {
    "adjacent_travel_time_cap": "travel_time",
    "poi_inbound_travel_time_cap": "travel_time",
    "activity_duration_limit": "pace",
    "daily_poi_cap": "pace",
    "budget_total": "budget",
    "ticket_budget_total": "budget",
    "activity_budget_limit": "budget",
    "anchor_bundle_budget_limit": "budget",
    "semantic_type_requirement": "semantic_type",
    "anchor_neighbor_commute_distance_cap": "anchor_proximity",
}

_SOFT_AXIS_BY_TARGET = {
    ("route_compactness", "travel_time"): "travel_time",
    ("theme_alignment", "culture"): "semantic_type",
    ("theme_alignment", "nature"): "semantic_type",
    ("theme_alignment", "food"): "semantic_type",
    ("theme_alignment", "photo"): "semantic_type",
    ("theme_alignment", "family"): "semantic_type",
    ("theme_alignment", "shopping"): "semantic_type",
    ("burden_reduction", "pace"): "pace",
    ("burden_reduction", "activity_load"): "pace",
    ("anchor_proximity", "poi_anchor"): "anchor_proximity",
}

_ALLOW_HARD_SOFT_COEXISTENCE_TYPES = {
    "travel_time_constraint",
    "poi_local_radius_limit",
    "extend_with_pacing",
    "type_filter",
}

_SOFT_FAMILY_ALIASES = {
    "theme_alignment": "theme_alignment",
    "theme_alignment": "theme_alignment",
}


def _soft_preference_hints(
    *,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    query_spec: Optional[Dict[str, Any]] = None,
    semantic_slots: Optional[Dict[str, Any]] = None,
    template_text: str = "",
    constraint_type: str = "",
    hard_constraints: Optional[List[LogicalConstraintObject]] = None,
) -> list[Dict[str, Any]]:
    trace = query_generation_trace if isinstance(query_generation_trace, dict) else {}
    if trace.get("edit_soft_target_suppressed") is True:
        return []
    family = str(trace.get("soft_target_family", "") or "").strip()
    facet = str(trace.get("soft_target_facet", "") or "").strip()
    edit_mode = str(trace.get("soft_target_mode", "") or "").strip()
    family = _SOFT_FAMILY_ALIASES.get(family, family)
    if not family or not facet or not edit_mode:
        return []

    hard_axes = {
        axis
        for item in (hard_constraints or [])
        for axis in [_HARD_AXIS_BY_CONSTRAINT_TYPE.get(str(item.type).strip())]
        if axis
    }
    soft_axis = _SOFT_AXIS_BY_TARGET.get((family, facet))
    if (
        constraint_type not in _ALLOW_HARD_SOFT_COEXISTENCE_TYPES
        and soft_axis
        and soft_axis in hard_axes
    ):
        return []

    semantic_slots = semantic_slots if isinstance(semantic_slots, dict) else {}
    query_spec = query_spec if isinstance(query_spec, dict) else {}
    anchor = (
        semantic_slots.get("anchor_poi")
        or query_spec.get("anchor_poi")
        or semantic_slots.get("district")
        or query_spec.get("district")
    )
    return [
        {
            "family": family,
            "facet": facet,
            "edit_mode": edit_mode,
            "anchor": anchor,
            "intensity": str(trace.get("soft_target_intensity", "") or "").strip() or None,
            "provenance": {
                "source": "canonical_constraint_ir",
                "constraint_type": constraint_type,
                "template_text": template_text,
            },
        }
    ]


def _build_structured_hard_constraints(
    *,
    constraint_type: str,
    query_spec: Optional[Dict[str, Any]] = None,
    semantic_slots: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    origin_plan: Optional[Dict[str, Any]] = None,
    origin_query_structured: Optional[Dict[str, Any]] = None,
) -> List[LogicalConstraintObject]:
    ctype = str(constraint_type or "").strip()
    if not ctype:
        raise ValueError("constraint_type is required for IR compilation")

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
        raw_modes: List[Any] = []
        for payload in (
            query_spec.get("intercity_transport_modes"),
            semantic_slots.get("intercity_transport_modes"),
            constraints.get("transport", {}).get("intercity_modes") if isinstance(constraints.get("transport"), dict) else None,
            constraints.get("semantic", {}).get("intercity_transport_modes") if isinstance(constraints.get("semantic"), dict) else None,
        ):
            if isinstance(payload, list):
                raw_modes.extend(payload)
            elif payload not in (None, ""):
                raw_modes.append(payload)
        modes: List[str] = []
        for raw_mode in raw_modes:
            mode = normalize_intercity_transport_mode(raw_mode)
            if mode and mode not in modes:
                modes.append(mode)
        if modes:
            add(
                "required_intercity_transport_type",
                modes,
                operator="contains_all",
                target={"activity_type": "intercity_transport"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "transport_time_window":
        raw_windows = query_spec.get("transport_time_windows") or semantic_slots.get("transport_time_windows") or []
        if isinstance(raw_windows, list):
            for window in raw_windows:
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
                        params={"leg": leg, "field": field, "derived_from": "query_spec"},
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
        end_before = string_value(query_spec.get("end_before"), semantic_slots.get("end_before"))
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
        time_period = _normalize_time_period(string_value(query_spec.get("time_period"), semantic_slots.get("time_period")))
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

    if ctype == "max_distance_constraint":
        distance = number_from_text(
            query_spec.get("distance_km"),
            semantic_slots.get("distance_km"),
            constraints.get("spatial", {}).get("max_distance_constraint", {}).get("max_distance_km")
            if isinstance(constraints.get("spatial", {}).get("max_distance_constraint"), dict)
            else None,
            constraints.get("spatial", {}).get("max_walking_distance") if isinstance(constraints.get("spatial"), dict) else None,
            constraints.get("semantic", {}).get("max_distance_km") if isinstance(constraints.get("semantic"), dict) else None,
            constraints.get("semantic", {}).get("chosen_walk_dist_km") if isinstance(constraints.get("semantic"), dict) else None,
        )
        if distance is not None:
            add(
                "adjacent_travel_distance_cap",
                distance,
                operator="<=",
                target={"activity_types": ["attraction"]},
                params={"unit": "km", "day": "all", "derived_from": "query_spec"},
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
                params={"day": day_value, "min_count": 1, "strict_majority": True, "derived_from": "constraints.semantic.preferred_type"},
            )

    if ctype == "required_room_type":
        room_type = int_value(
            query_spec.get("room_type"),
            semantic_slots.get("hotel_room_type"),
            constraints.get("resource", {}).get("room_type") if isinstance(constraints.get("resource"), dict) else None,
        )
        if room_type is not None:
            add(
                "required_room_type",
                room_type,
                operator="==",
                target={"activity_type": "accommodation"},
                params={"derived_from": "query_spec"},
            )

    if ctype == "required_room_count":
        room_count = int_value(
            query_spec.get("room_count"),
            semantic_slots.get("hotel_room_count"),
            constraints.get("resource", {}).get("room_count") if isinstance(constraints.get("resource"), dict) else None,
        )
        if room_count is not None:
            add(
                "required_room_count",
                room_count,
                operator="==",
                target={"activity_type": "accommodation"},
                params={"derived_from": "query_spec"},
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
                params={"supported_cities": list(_SUPPORTED_DATASET_CITIES), "derived_from": "query_spec"},
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

    if ctype in {"substitute_poi", "type_replacement"}:
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
        candidate_names: List[str] = []
        if ctype == "substitute_poi" and source_poi and not semantic_target:
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
                    **({"from_substitute_poi": True} if ctype == "substitute_poi" else {"from_type_replacement": True}),
                    **({"candidate_names": candidate_names} if candidate_names else {}),
                },
            )

    if ctype == "famous_substitution":
        source_poi = string_value(
            query_spec.get("source_poi"),
            semantic_slots.get("source_poi"),
            constraints.get("semantic", {}).get("replace_from") if isinstance(constraints.get("semantic"), dict) else None,
            *(forbidden_poi_names()[:1]),
        )
        replacement_criterion = string_value(
            query_spec.get("semantic_target"),
            semantic_slots.get("semantic_target"),
            constraints.get("semantic", {}).get("semantic_target") if isinstance(constraints.get("semantic"), dict) else None,
            "更有名",
        )
        if source_poi:
            add(
                "fame_replacement_requirement",
                replacement_criterion,
                operator="satisfies",
                target={"activity_type": "attraction"},
                params={"replace_from": source_poi, "derived_from": "query_spec"},
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
                params={"budget_scope": "anchor_bundle", "requires_nearby_meal": nearby_meal, **nearby_params},
            )

    if family == "anchor_commute_distance_cap":
        anchor_name = string_value(query_spec.get("anchor_poi"), semantic_slots.get("anchor_poi"))
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


def _build_root(
    *,
    constraint_type: str,
    template_semantic_spec: Dict[str, Any],
    logical_constraints: List[LogicalConstraintObject],
) -> Dict[str, Any]:
    root_type = str(template_semantic_spec.get("root_node_type", "atomic") or "atomic")
    child_ids = [item.id for item in logical_constraints]
    root: Dict[str, Any] = {
        "node_type": root_type,
        "constraint_type": constraint_type,
        "children": child_ids,
    }
    if constraint_type == "ticket_price_cap_or_substitute":
        condition_children = [item.id for item in logical_constraints if item.type == "ticket_price_cap_or_substitute"]
        root.update(
            {
                "condition": {
                    "predicate": "ticket_price_lte",
                },
                "condition_children": condition_children,
                "then_children": [],
                "else_children": condition_children,
            }
        )
    else:
        root.setdefault("condition_children", [])
        root.setdefault("then_children", [])
        root.setdefault("else_children", [])
    return root


def build_canonical_constraint_ir_from_structured_inputs(
    *,
    template_text: str,
    constraint_type: str,
    instantiated_slots: Dict[str, Any],
    template_semantic_spec: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    structured = instantiated_slots if isinstance(instantiated_slots, dict) else {}
    query_spec = structured.get("query_spec") if isinstance(structured.get("query_spec"), dict) else {}
    semantic_slots = structured.get("semantic_slots") if isinstance(structured.get("semantic_slots"), dict) else {}
    constraints = structured.get("constraints") if isinstance(structured.get("constraints"), dict) else {}
    origin_plan = structured.get("origin_plan") if isinstance(structured.get("origin_plan"), dict) else None
    origin_query_structured = (
        structured.get("origin_query_structured")
        if isinstance(structured.get("origin_query_structured"), dict)
        else None
    )

    logical_constraints = _build_structured_hard_constraints(
        constraint_type=constraint_type,
        query_spec=query_spec,
        semantic_slots=semantic_slots,
        constraints=constraints,
        origin_plan=origin_plan,
        origin_query_structured=origin_query_structured,
    )
    nodes = []
    for logical in logical_constraints:
        compiled = compile_logical_constraint(logical)
        node = logical_constraint_to_ir_node(logical, supported=compiled is not None)
        nodes.append(node)

    trace = meta.get("query_generation_trace") if isinstance(meta, dict) and isinstance(meta.get("query_generation_trace"), dict) else {}
    canonical_ir = CanonicalConstraintIR(
        ir_version="v1",
        source={
            "constraint_type": constraint_type,
            "template_id": str(meta.get("template_id", "") if isinstance(meta, dict) else ""),
            "template_text": template_text,
            "instantiated_slots": {
                "query_spec": query_spec,
                "semantic_slots": semantic_slots,
            },
            "template_semantic_spec": dict(template_semantic_spec),
        },
        root=_build_root(
            constraint_type=constraint_type,
            template_semantic_spec=template_semantic_spec,
            logical_constraints=logical_constraints,
        ),
        nodes=nodes,
        preference_hints=_soft_preference_hints(
            query_generation_trace=trace,
            query_spec=query_spec,
            semantic_slots=semantic_slots,
            template_text=template_text,
            constraint_type=constraint_type,
            hard_constraints=logical_constraints,
        ),
        provenance={
            "compiler": "constraint_ir.compiler",
            "schema_version": "v1",
            "surface_query": structured.get("surface_query") or structured.get("edit_query"),
            "source": "structured_inputs",
        },
    )
    return canonical_ir.to_dict()


def compile_constraint_ir(
    *,
    template_text: str,
    constraint_type: str,
    instantiated_slots: Dict[str, Any],
    template_semantic_spec: Dict[str, Any] | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    semantic_spec = resolve_template_semantic_spec(
        {"template_semantic_spec": template_semantic_spec or {}},
        constraint_type=constraint_type,
    )
    return build_canonical_constraint_ir_from_structured_inputs(
        template_text=template_text,
        constraint_type=constraint_type,
        instantiated_slots=instantiated_slots,
        template_semantic_spec=semantic_spec,
        meta=meta,
    )


def build_canonical_constraint_ir_from_legacy_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    trace = sample.get("query_generation_trace") if isinstance(sample.get("query_generation_trace"), dict) else {}
    constraint_type = str(trace.get("constraint_type", "") or "").strip()
    template_text = str(
        trace.get("template", "")
        or sample.get("canonical_edit_query", "")
        or sample.get("edit_query", "")
        or ""
    )
    semantic_spec = resolve_template_semantic_spec(trace, constraint_type=constraint_type)
    query_spec = trace.get("canonical_query_spec") if isinstance(trace.get("canonical_query_spec"), dict) else {}
    semantic_slots = trace.get("canonical_semantic_slots") if isinstance(trace.get("canonical_semantic_slots"), dict) else {}
    has_structured_state = constraint_type and (
        query_spec
        or semantic_slots
        or isinstance(sample.get("constraints"), dict)
    )
    if not has_structured_state:
        raise ValueError("legacy sample is missing canonical structured state for IR migration")
    return build_canonical_constraint_ir_from_structured_inputs(
        template_text=template_text,
        constraint_type=constraint_type,
        instantiated_slots={
            "query_spec": query_spec,
            "semantic_slots": semantic_slots,
            "constraints": sample.get("constraints", {}),
            "origin_plan": sample.get("origin_plan"),
            "origin_query_structured": sample.get("origin_query_structured"),
            "surface_query": sample.get("edit_query"),
        },
        template_semantic_spec=semantic_spec,
        meta={
            "template_id": str(trace.get("template_id", "") or ""),
            "query_generation_trace": trace,
        },
    )
