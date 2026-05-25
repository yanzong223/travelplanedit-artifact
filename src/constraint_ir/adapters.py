"""Adapters between canonical constraint IR and legacy benchmark truth objects."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from evaluation.benchmark.models import LogicalConstraintObject, PreferenceTag
from evaluation.benchmark.soft_truth import build_edit_target_preferences, validate_preference_tags

from .models import IRNode, canonical_ir_from_dict


def logical_constraint_to_ir_node(
    constraint: LogicalConstraintObject | Dict[str, Any],
    *,
    supported: bool | None = None,
) -> IRNode:
    if isinstance(constraint, LogicalConstraintObject):
        payload = constraint.to_dict()
    else:
        payload = dict(constraint)
    return IRNode(
        node_type="atomic",
        id=str(payload.get("id", "")),
        constraint_type=str(payload.get("type", "")),
        predicate=str(payload.get("operator", payload.get("predicate", "==")) or "=="),
        value=payload.get("value"),
        target=dict(payload.get("target", {})),
        params=dict(payload.get("params", {})),
        scope=dict(payload.get("scope", {})),
        provenance=dict(payload.get("provenance", {})),
        supported=supported,
    )


def ir_node_to_logical_constraint(
    node: IRNode | Dict[str, Any],
    *,
    source: str = "edit",
) -> LogicalConstraintObject | None:
    if isinstance(node, dict):
        node = IRNode(
            node_type=str(node.get("node_type", "atomic") or "atomic"),
            id=str(node.get("id", "")),
            constraint_type=str(node.get("constraint_type", "")),
            predicate=str(node.get("predicate", node.get("operator", "==")) or "=="),
            value=node.get("value"),
            target=dict(node.get("target", {})),
            params=dict(node.get("params", {})),
            scope=dict(node.get("scope", {})),
            provenance=dict(node.get("provenance", {})),
            children=[str(item) for item in node.get("children", []) if isinstance(item, (str, int))],
            supported=node.get("supported"),
        )
    if node.node_type != "atomic" or not node.constraint_type:
        return None
    return LogicalConstraintObject(
        id=node.id or f"{source}_logic_ir_{node.constraint_type}",
        source=source,
        type=node.constraint_type,
        scope=dict(node.scope),
        target=dict(node.target),
        operator=node.predicate or "==",
        value=node.value,
        params=dict(node.params),
        is_hard=True,
        provenance=dict(node.provenance),
    )


def ir_to_logical_constraints(canonical_ir: Dict[str, Any] | None, *, source: str = "edit") -> List[LogicalConstraintObject]:
    ir = canonical_ir_from_dict(canonical_ir)
    logical_constraints: List[LogicalConstraintObject] = []
    for node in ir.nodes:
        logical = ir_node_to_logical_constraint(node, source=source)
        if logical is not None:
            logical_constraints.append(logical)
    return logical_constraints


def ir_to_preference_tags(canonical_ir: Dict[str, Any] | None) -> List[PreferenceTag]:
    ir = canonical_ir_from_dict(canonical_ir)
    return validate_preference_tags(
        (
            PreferenceTag(
                family=str(item.get("family", "")),
                facet=str(item.get("facet", "")),
                edit_mode=str(item.get("edit_mode")) if item.get("edit_mode") else None,
                anchor=item.get("anchor"),
                intensity=str(item.get("intensity", "")).strip() or None,
                provenance=dict(item.get("provenance", {})),
            )
            for item in ir.preference_hints
            if isinstance(item, dict)
        ),
        require_edit_mode=True,
    )


def ir_to_preference_constraints(canonical_ir: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in build_edit_target_preferences(ir_to_preference_tags(canonical_ir))]


def build_edit_targets_from_ir(canonical_ir: Dict[str, Any] | None) -> Dict[str, List[Dict[str, Any]]]:
    tags = ir_to_preference_tags(canonical_ir)
    preferences = build_edit_target_preferences(tags)
    return {
        "edit_target_constraints": [item.to_dict() for item in ir_to_logical_constraints(canonical_ir)],
        "edit_target_preference_tags": [item.to_dict() for item in tags],
        "edit_target_preferences": [item.to_dict() for item in preferences],
    }
