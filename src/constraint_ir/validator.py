"""Validation helpers for canonical constraint IR."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from evaluation.benchmark.compilers import compile_logical_constraint

from .adapters import build_edit_targets_from_ir, ir_to_logical_constraints
from .models import canonical_ir_from_dict


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

_ALLOW_HARD_SOFT_AXIS_OVERLAP_CONSTRAINT_TYPES = {
    "travel_time_constraint",
    "poi_local_radius_limit",
    "extend_with_pacing",
    "type_filter",
}


def validate_canonical_constraint_ir(canonical_ir: Dict[str, Any] | None) -> Tuple[bool, List[str]]:
    ir = canonical_ir_from_dict(canonical_ir)
    errors: List[str] = []
    if not ir.source:
        errors.append("canonical_constraint_ir.source missing")
    if not ir.root:
        errors.append("canonical_constraint_ir.root missing")
    root_type = str(ir.root.get("node_type", "") or "")
    if not isinstance(ir.root.get("children", []), list):
        errors.append("canonical_constraint_ir.root.children must be a list")
    if root_type == "conditional":
        if not isinstance(ir.root.get("condition", {}), dict) or not ir.root.get("condition"):
            errors.append("conditional canonical_constraint_ir.root.condition missing")
        for branch_field in ("condition_children", "then_children", "else_children"):
            if not isinstance(ir.root.get(branch_field, []), list):
                errors.append(f"conditional canonical_constraint_ir.root.{branch_field} must be a list")
    node_ids = set()
    for node in ir.nodes:
        if not node.id:
            errors.append("canonical_constraint_ir node missing id")
        elif node.id in node_ids:
            errors.append(f"duplicate canonical_constraint_ir node id: {node.id}")
        node_ids.add(node.id)
        if not node.constraint_type:
            errors.append(f"node {node.id or '<missing>'} missing constraint_type")
    for child_id in ir.root.get("children", []):
        if str(child_id) not in node_ids:
            errors.append(f"root child reference not found in nodes: {child_id}")
    if root_type == "conditional":
        for branch_field in ("condition_children", "then_children", "else_children"):
            for child_id in ir.root.get(branch_field, []):
                if str(child_id) not in node_ids:
                    errors.append(f"conditional branch child reference not found in nodes: {child_id}")
    return (len(errors) == 0, errors)


def check_template_ir_consistency(
    canonical_ir: Dict[str, Any] | None,
    *,
    constraint_type: str | None = None,
    template_str: str | None = None,
) -> Tuple[bool, List[str]]:
    ir = canonical_ir_from_dict(canonical_ir)
    errors: List[str] = []
    source = ir.source
    if constraint_type and str(source.get("constraint_type", "") or "").strip() != str(constraint_type).strip():
        errors.append(
            "canonical_constraint_ir.source.constraint_type does not match query_generation_trace.constraint_type"
        )
    if template_str and str(source.get("template_text", "") or "") != str(template_str):
        errors.append("canonical_constraint_ir.source.template_text does not match query_generation_trace.template")
    semantic_spec = source.get("template_semantic_spec", {})
    root_type = str(ir.root.get("node_type", "") or "")
    expected_root_type = str(semantic_spec.get("root_node_type", "") or "")
    if expected_root_type and root_type and expected_root_type != root_type:
        errors.append(
            f"canonical_constraint_ir.root.node_type mismatch: expected {expected_root_type}, got {root_type}"
        )
    return (len(errors) == 0, errors)


def check_ir_compiles(canonical_ir: Dict[str, Any] | None) -> Tuple[bool, List[str], Dict[str, Any]]:
    logical_constraints = ir_to_logical_constraints(canonical_ir)
    unsupported: List[str] = []
    for logical in logical_constraints:
        if compile_logical_constraint(logical) is None:
            unsupported.append(logical.type)
    return (
        len(unsupported) == 0,
        [f"unsupported verifier/compiler constraint types: {sorted(set(unsupported))}"] if unsupported else [],
        {
            "compiled_constraints": [item.to_dict() for item in logical_constraints],
            "unsupported_constraint_types": sorted(set(unsupported)),
        },
    )


def compile_ir_to_edit_targets(canonical_ir: Dict[str, Any] | None) -> Dict[str, List[Dict[str, Any]]]:
    return build_edit_targets_from_ir(canonical_ir)


def check_hard_soft_axis_dedup(canonical_ir: Dict[str, Any] | None) -> Tuple[bool, List[str], Dict[str, Any]]:
    ir = canonical_ir_from_dict(canonical_ir)
    constraint_type = str(ir.source.get("constraint_type", "") or "").strip()
    hard_axes = sorted(
        {
            axis
            for logical in ir_to_logical_constraints(canonical_ir)
            for axis in [_HARD_AXIS_BY_CONSTRAINT_TYPE.get(str(logical.type).strip())]
            if axis
        }
    )
    soft_axes = sorted(
        {
            axis
            for hint in ir.preference_hints
            for axis in [_SOFT_AXIS_BY_TARGET.get((str(hint.get("family", "")), str(hint.get("facet", ""))))]
            if axis
        }
    )
    overlap = sorted(set(hard_axes) & set(soft_axes))
    allowed_overlap = (
        constraint_type in _ALLOW_HARD_SOFT_AXIS_OVERLAP_CONSTRAINT_TYPES
        and bool(overlap)
    )
    return (
        len(overlap) == 0 or allowed_overlap,
        [f"hard/soft semantic axes overlap: {overlap}"] if overlap and not allowed_overlap else [],
        {
            "constraint_type": constraint_type,
            "hard_axes": hard_axes,
            "soft_axes": soft_axes,
            "overlap_axes": overlap,
            "allowed_overlap": allowed_overlap,
        },
    )
