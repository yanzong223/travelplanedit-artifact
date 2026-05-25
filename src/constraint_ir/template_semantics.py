"""Template semantic registry for canonical constraint IR."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


TEMPLATE_SEMANTIC_SPEC_VERSION = "v1"

_EXPLICIT_TEMPLATE_FIELDS = (
    "required_entities",
    "forbidden_entities",
    "bound_slots",
    "relation_slots",
    "ir_outputs",
    "hard_axes",
    "soft_axes",
    "entity_slots_hard_policy",
    "bound_slots_hard_policy",
    "drop_behavior_when_bound_missing",
    "risk_level",
    "decision",
    "root_node_type",
)

_DEFAULT_TEMPLATE_SPECS: Dict[str, Dict[str, Any]] = {
    "duration_limit": {
        "required_entities": ["poi"],
        "forbidden_entities": [],
        "bound_slots": ["duration_limit_min"],
        "relation_slots": [],
        "ir_outputs": ["activity_duration_limit"],
        "hard_axes": ["pace"],
        "soft_axes": [],
        "entity_slots_hard_policy": "keep_if_explicit",
        "bound_slots_hard_policy": "require_explicit",
        "drop_behavior_when_bound_missing": "drop_bound_keep_entity",
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "budget_limit": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["budget_cap"],
        "relation_slots": [],
        "ir_outputs": ["budget_total"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "daily_budget_infeasible": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["budget_cap"],
        "relation_slots": [],
        "ir_outputs": ["budget_total"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "budget_cap_preference": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["budget_cap"],
        "relation_slots": ["preference_only"],
        "ir_outputs": ["budget_total"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "budget_target_update": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["budget_cap"],
        "relation_slots": ["preference_only"],
        "ir_outputs": ["budget_total"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "ticket_budget_limit": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["ticket_budget_cap"],
        "relation_slots": [],
        "ir_outputs": ["ticket_budget_total"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "activity_budget_limit": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["budget_cap"],
        "relation_slots": ["metric_scope"],
        "ir_outputs": ["activity_budget_limit"],
        "hard_axes": ["budget"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "time_limit": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["day", "end_before"],
        "relation_slots": ["day_scope"],
        "ir_outputs": ["day_end_time_limit"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "time_window": {
        "required_entities": ["poi"],
        "forbidden_entities": [],
        "bound_slots": ["time_period"],
        "relation_slots": ["day_scope"],
        "ir_outputs": ["poi_time_window"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "order_constraint": {
        "required_entities": ["poi1", "poi2"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["before_relation"],
        "ir_outputs": ["poi_order_constraint"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "travel_time_constraint": {
        "required_entities": ["poi"],
        "forbidden_entities": [],
        "bound_slots": ["duration_limit_min"],
        "relation_slots": ["adjacent_edge_scope"],
        "ir_outputs": ["required_attraction_name", "adjacent_travel_time_cap"],
        "hard_axes": ["travel_time"],
        "soft_axes": ["travel_time"],
        "entity_slots_hard_policy": "keep_if_explicit",
        "bound_slots_hard_policy": "require_explicit",
        "drop_behavior_when_bound_missing": "drop_bound_keep_entity",
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "access_time_infeasible": {
        "required_entities": ["poi"],
        "forbidden_entities": [],
        "bound_slots": ["duration_limit_min"],
        "relation_slots": ["poi_inbound_scope"],
        "ir_outputs": ["poi_inbound_travel_time_cap"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "poi_local_radius_limit": {
        "required_entities": ["anchor_poi"],
        "forbidden_entities": [],
        "bound_slots": ["distance_km"],
        "relation_slots": ["anchor_scope"],
        "ir_outputs": ["anchor_neighbor_commute_distance_cap"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "walking_distance_limit": {
        "required_entities": ["poi1", "poi2"],
        "forbidden_entities": [],
        "bound_slots": ["distance_km"],
        "relation_slots": ["pair_scope", "transport_mode"],
        "ir_outputs": ["pairwise_transport_mode_distance_cap"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "max_distance_constraint": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["distance_km"],
        "relation_slots": ["global_pair_scope"],
        "ir_outputs": ["max_distance_constraint"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "spatial_overflow": {
        "required_entities": ["poi"],
        "forbidden_entities": [],
        "bound_slots": ["day", "distance_km"],
        "relation_slots": ["day_scope", "anchor_scope"],
        "ir_outputs": ["required_attraction_name", "anchor_neighbor_commute_distance_cap"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "resource_overlap": {
        "required_entities": ["poi"],
        "forbidden_entities": [],
        "bound_slots": ["budget_cap"],
        "relation_slots": ["nearby_meal", "bundle_cost"],
        "ir_outputs": ["required_attraction_name", "nearby_meal_requirement", "anchor_bundle_budget_limit"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "type_filter": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["semantic_type", "min_count"],
        "ir_outputs": ["semantic_type_requirement"],
        "hard_axes": ["semantic_type"],
        "soft_axes": ["semantic_type"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "preference_constraint": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["semantic_target"],
        "ir_outputs": ["semantic_type_requirement"],
        "hard_axes": ["semantic_type"],
        "soft_axes": ["semantic_type"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "semantic_and_specific": {
        "required_entities": ["specific_poi"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["semantic_type", "coexist"],
        "ir_outputs": ["required_attraction_name", "semantic_type_requirement"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "category_overlap": {
        "required_entities": ["specific_poi"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["semantic_type", "coexist"],
        "ir_outputs": ["required_attraction_name", "semantic_type_requirement"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "multi_day_requirement": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["day_count"],
        "relation_slots": [],
        "ir_outputs": ["day_count"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "add_days": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["day_count"],
        "relation_slots": ["day_delta_source"],
        "ir_outputs": ["day_count"],
        "hard_axes": ["pace"],
        "soft_axes": ["pace"],
        "drop_behavior_when_bound_missing": "fail",
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "extend_with_pacing": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["day_count", "daily_poi_cap"],
        "relation_slots": ["pacing"],
        "ir_outputs": ["day_count", "daily_poi_cap"],
        "hard_axes": ["pace"],
        "soft_axes": ["pace"],
        "drop_behavior_when_bound_missing": "fail",
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "multi_day_budget_overflow": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["day_count", "budget_cap"],
        "relation_slots": [],
        "ir_outputs": ["day_count", "budget_total"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "cross_day_restructure": {
        "required_entities": ["poi1", "poi2"],
        "forbidden_entities": [],
        "bound_slots": ["day_count"],
        "relation_slots": ["day_poi_pairs"],
        "ir_outputs": ["day_count", "poi_day_binding"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "cross_day_variety_by_types": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["day_count"],
        "relation_slots": ["day_type_pairs"],
        "ir_outputs": ["day_count", "poi_day_binding", "semantic_type_requirement"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "same_day_no_overlap_compact": {
        "required_entities": ["poi1", "poi2"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["same_day", "no_overlap"],
        "ir_outputs": ["pair_same_day_no_overlap"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "same_time_period_no_overlap": {
        "required_entities": ["poi1", "poi2"],
        "forbidden_entities": [],
        "bound_slots": ["time_period"],
        "relation_slots": ["pair_time_window_no_overlap"],
        "ir_outputs": ["pair_time_window_no_overlap"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "semantic_substitution": {
        "required_entities": [],
        "forbidden_entities": ["source_poi"],
        "bound_slots": [],
        "relation_slots": ["semantic_type", "substitution"],
        "ir_outputs": ["forbidden_attraction_name", "semantic_type_requirement"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "type_replacement": {
        "required_entities": [],
        "forbidden_entities": ["source_poi"],
        "bound_slots": [],
        "relation_slots": ["semantic_type", "substitution"],
        "ir_outputs": ["forbidden_attraction_name", "semantic_type_requirement"],
        "hard_axes": ["semantic_type"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "substitute_poi": {
        "required_entities": ["target_poi"],
        "forbidden_entities": ["source_poi"],
        "bound_slots": [],
        "relation_slots": ["replacement_pair", "semantic_type"],
        "ir_outputs": ["forbidden_attraction_name", "semantic_type_requirement"],
        "hard_axes": ["semantic_type"],
        "risk_level": "medium",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "famous_substitution": {
        "required_entities": [],
        "forbidden_entities": ["source_poi"],
        "bound_slots": [],
        "relation_slots": ["replacement_criterion"],
        "ir_outputs": ["forbidden_attraction_name", "fame_replacement_requirement"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
    "remove_poi": {
        "required_entities": [],
        "forbidden_entities": ["source_poi"],
        "bound_slots": [],
        "relation_slots": [],
        "ir_outputs": ["forbidden_attraction_name"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_restaurant_type": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["restaurant_type"],
        "relation_slots": [],
        "ir_outputs": ["required_restaurant_type"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_restaurant_name": {
        "required_entities": ["restaurant_name"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": [],
        "ir_outputs": ["required_restaurant_name"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_hotel_feature": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["hotel_feature"],
        "relation_slots": [],
        "ir_outputs": ["required_hotel_feature"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_hotel_name": {
        "required_entities": ["hotel_name"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": [],
        "ir_outputs": ["required_hotel_name"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_room_count": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["room_count"],
        "relation_slots": [],
        "ir_outputs": ["required_room_count"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_room_type": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["room_type"],
        "relation_slots": [],
        "ir_outputs": ["required_room_type"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "required_intercity_transport_type": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["intercity_transport_modes"],
        "relation_slots": ["transport_mode"],
        "ir_outputs": ["required_intercity_transport_type"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "transport_time_window": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": ["transport_time_windows"],
        "relation_slots": ["leg", "field", "time_window"],
        "ir_outputs": ["transport_time_window"],
        "risk_level": "low",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "ticket_price_cap_or_substitute": {
        "required_entities": ["poi"],
        "forbidden_entities": ["source_poi"],
        "bound_slots": ["budget_cap"],
        "relation_slots": ["if_ticket_price_gt_then_substitute"],
        "ir_outputs": ["ticket_price_cap_or_substitute"],
        "hard_axes": ["budget"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "conditional",
    },
    "multi_city_split": {
        "required_entities": ["target_city"],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["second_city_explicit"],
        "ir_outputs": ["city_split_requirement"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "atomic",
    },
    "thematic_split": {
        "required_entities": [],
        "forbidden_entities": [],
        "bound_slots": [],
        "relation_slots": ["day_theme_pairs"],
        "ir_outputs": ["semantic_type_requirement"],
        "risk_level": "high",
        "decision": "keep",
        "root_node_type": "composite",
    },
}


def _as_list(values: Any) -> List[str]:
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    if values is None:
        return []
    text = str(values).strip()
    return [text] if text else []


def resolve_template_semantic_spec(
    template: Dict[str, Any] | None,
    *,
    constraint_type: str,
    bucket: Tuple[str, str, str] | None = None,
) -> Dict[str, Any]:
    template = template if isinstance(template, dict) else {}
    spec = dict(_DEFAULT_TEMPLATE_SPECS.get(str(constraint_type or "").strip(), {}))
    explicit = template.get("template_semantic_spec")
    if isinstance(explicit, dict):
        spec.update(explicit)
    for field_name in _EXPLICIT_TEMPLATE_FIELDS:
        if field_name in template and field_name != "template_semantic_spec":
            spec[field_name] = template[field_name]

    spec.setdefault("required_entities", [])
    spec.setdefault("forbidden_entities", [])
    spec.setdefault("bound_slots", [])
    spec.setdefault("relation_slots", [])
    spec.setdefault("ir_outputs", [])
    spec.setdefault("hard_axes", [])
    spec.setdefault("soft_axes", [])
    spec.setdefault("entity_slots_hard_policy", "none")
    spec.setdefault("bound_slots_hard_policy", "require_explicit")
    spec.setdefault("drop_behavior_when_bound_missing", "drop_bound_only")
    spec.setdefault("risk_level", "medium")
    spec.setdefault("decision", "keep")
    spec.setdefault("root_node_type", "atomic")
    spec["required_entities"] = _as_list(spec.get("required_entities"))
    spec["forbidden_entities"] = _as_list(spec.get("forbidden_entities"))
    spec["bound_slots"] = _as_list(spec.get("bound_slots"))
    spec["relation_slots"] = _as_list(spec.get("relation_slots"))
    spec["ir_outputs"] = _as_list(spec.get("ir_outputs"))
    spec["hard_axes"] = _as_list(spec.get("hard_axes"))
    spec["soft_axes"] = _as_list(spec.get("soft_axes"))
    spec["constraint_type"] = str(constraint_type or template.get("constraint_type", "") or "").strip()
    spec["template_semantic_spec_version"] = TEMPLATE_SEMANTIC_SPEC_VERSION
    if bucket is not None:
        spec["bucket"] = list(bucket)
    return spec


def annotate_template_semantics(
    template: Dict[str, Any] | None,
    *,
    constraint_type: str,
    bucket: Tuple[str, str, str] | None = None,
) -> Dict[str, Any]:
    normalized = dict(template or {})
    normalized["template_semantic_spec"] = resolve_template_semantic_spec(
        normalized,
        constraint_type=constraint_type,
        bucket=bucket,
    )
    normalized.setdefault(
        "template_semantic_spec_version",
        normalized["template_semantic_spec"]["template_semantic_spec_version"],
    )
    return normalized


def iter_template_semantic_specs(templates_config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not isinstance(templates_config, dict):
        return []
    output: List[Dict[str, Any]] = []
    for bucket_key, bucket_cfg in templates_config.items():
        if not isinstance(bucket_cfg, dict) or not bucket_cfg.get("eligible_for_edit", False):
            continue
        bucket_tuple = None
        if isinstance(bucket_key, tuple):
            bucket_tuple = bucket_key
        elif isinstance(bucket_key, str):
            stripped = bucket_key.strip().strip("()")
            parts = tuple(part.strip() for part in stripped.split(","))
            if len(parts) == 3:
                bucket_tuple = parts  # type: ignore[assignment]
        for template in bucket_cfg.get("templates", []):
            if not isinstance(template, dict):
                continue
            ctype = str(template.get("constraint_type", "")).strip()
            if not ctype:
                continue
            output.append(
                resolve_template_semantic_spec(
                    template,
                    constraint_type=ctype,
                    bucket=bucket_tuple,
                )
            )
    return output
