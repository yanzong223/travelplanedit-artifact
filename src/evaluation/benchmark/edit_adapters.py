"""Strict adapters for edit-side benchmark truth."""

from __future__ import annotations

from typing import Any, Dict, List

from .models import LogicalConstraintObject, PreferenceConstraintObject
from .soft_truth import build_edit_target_preferences, normalize_family_facet, validate_preference_tags
from constraint_ir.adapters import ir_to_logical_constraints, ir_to_preference_constraints


def _logical_from_dict(index: int, payload: Dict[str, Any], source: str) -> LogicalConstraintObject:
    return LogicalConstraintObject(
        id=str(payload.get("id", f"{source}_logic_{index}")),
        source=source,
        type=str(payload.get("type", "")),
        scope=dict(payload.get("scope", {})),
        target=dict(payload.get("target", {})),
        operator=str(payload.get("operator", payload.get("op", "=="))),
        value=payload.get("value"),
        params=dict(payload.get("params", payload.get("parameters", {}))),
        is_hard=bool(payload.get("is_hard", True)),
        provenance=dict(payload.get("provenance", {})),
    )


def _preference_from_dict(index: int, payload: Dict[str, Any], source: str) -> PreferenceConstraintObject:
    family, facet = normalize_family_facet(payload.get("family", "unknown_preference"), payload.get("facet", "unknown"))
    return PreferenceConstraintObject(
        id=str(payload.get("id", f"{source}_pref_{index}")),
        source=source,
        family=family or str(payload.get("family", "unknown_preference")),
        facet=facet or str(payload.get("facet", "unknown")),
        direction=str(payload.get("direction", "maximize")),
        anchor=payload.get("anchor"),
        edit_mode=str(payload.get("edit_mode", "addition")),
        params=dict(payload.get("params", {})),
        provenance=dict(payload.get("provenance", {})),
    )


def adapt_legacy_edit_record(record: dict[str, Any]) -> dict[str, Any]:
    explicit_constraints = record.get("edit_target_constraints")
    if explicit_constraints is None:
        explicit_constraints = []
    if not isinstance(explicit_constraints, list):
        raise ValueError("edit_target_constraints must be a list when provided")

    explicit_preferences = record.get("edit_target_preferences", [])
    if explicit_preferences is None:
        explicit_preferences = []
    if not isinstance(explicit_preferences, list):
        raise ValueError("edit_target_preferences must be a list when provided")

    raw_tags = record.get("edit_target_preference_tags")
    if raw_tags is not None and not isinstance(raw_tags, list):
        raise ValueError("edit_target_preference_tags must be a list when provided")

    logical_constraints: List[LogicalConstraintObject] = [
        _logical_from_dict(index, payload, "edit")
        for index, payload in enumerate(explicit_constraints)
        if isinstance(payload, dict)
    ]

    preference_constraints: List[PreferenceConstraintObject] = [
        _preference_from_dict(index, payload, "edit")
        for index, payload in enumerate(explicit_preferences)
        if isinstance(payload, dict)
    ]
    if not preference_constraints and raw_tags:
        preference_constraints = build_edit_target_preferences(
            validate_preference_tags(raw_tags, require_edit_mode=True)
        )

    return {
        "logical_constraints": logical_constraints,
        "preference_constraints": preference_constraints,
    }


def adapt_edit_record(record: dict[str, Any]) -> dict[str, Any]:
    canonical_constraint_ir = record.get("canonical_constraint_ir")
    if isinstance(canonical_constraint_ir, dict) and canonical_constraint_ir:
        logical_constraints = ir_to_logical_constraints(canonical_constraint_ir, source="edit")
        preference_constraints = [
            _preference_from_dict(index, payload, "edit")
            for index, payload in enumerate(ir_to_preference_constraints(canonical_constraint_ir))
            if isinstance(payload, dict)
        ]
        if not logical_constraints and not preference_constraints:
            raise ValueError(
                "canonical_constraint_ir contains no evaluable edit targets; "
                "repair the IR instead of treating the record as a target-free edit"
            )
        return {
            "logical_constraints": logical_constraints,
            "preference_constraints": preference_constraints,
        }
    raise ValueError("canonical_constraint_ir is required for adapt_edit_record")
