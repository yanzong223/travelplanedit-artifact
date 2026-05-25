"""Shared logical-constraint merge policy for edit benchmarks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

from .models import LogicalConstraintObject


MERGE_POLICY_VERSION = "effective_logical_constraints_v1"
GLOBAL_OVERRIDE_TYPES = {"budget_total", "day_count", "people_count"}


def logical_constraint_from_dict(
    index: int,
    payload: Dict[str, Any],
    source: str,
) -> LogicalConstraintObject:
    return LogicalConstraintObject(
        id=str(payload.get("id", f"{source}_logic_{index}")),
        source=str(payload.get("source", source)),
        type=str(payload.get("type", "")),
        scope=dict(payload.get("scope", {})),
        target=dict(payload.get("target", {})),
        operator=str(payload.get("operator", payload.get("op", "=="))),
        value=payload.get("value"),
        params=dict(payload.get("params", payload.get("parameters", {}))),
        is_hard=bool(payload.get("is_hard", True)),
        provenance=dict(payload.get("provenance", {})),
    )


def merge_logical_constraint_objects(
    origin_constraints: List[LogicalConstraintObject],
    edit_constraints: List[LogicalConstraintObject],
) -> Tuple[List[LogicalConstraintObject], List[str]]:
    retained_origin: List[LogicalConstraintObject] = []
    overridden_ids: List[str] = []

    for origin in origin_constraints:
        if any(is_overridden(origin, edit) for edit in edit_constraints):
            overridden_ids.append(origin.id)
            continue
        retained_origin.append(origin)

    return retained_origin + list(edit_constraints), overridden_ids


def build_effective_logical_constraints(
    origin_logical_constraints: Iterable[Dict[str, Any]] | None,
    edit_target_constraints: Iterable[Dict[str, Any]] | None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    origin_payloads = [item for item in (origin_logical_constraints or []) if isinstance(item, dict)]
    edit_payloads = [item for item in (edit_target_constraints or []) if isinstance(item, dict)]
    origin_objects = [
        logical_constraint_from_dict(index, item, "origin")
        for index, item in enumerate(origin_payloads)
    ]
    edit_objects = [
        logical_constraint_from_dict(index, item, "edit")
        for index, item in enumerate(edit_payloads)
    ]
    _, overridden_ids = merge_logical_constraint_objects(origin_objects, edit_objects)
    retained_payloads = [
        deepcopy(payload)
        for payload, obj in zip(origin_payloads, origin_objects)
        if obj.id not in set(overridden_ids)
    ]
    effective = retained_payloads + [deepcopy(item) for item in edit_payloads]
    trace = {
        "merge_policy_version": MERGE_POLICY_VERSION,
        "origin_constraint_count": len(origin_payloads),
        "edit_constraint_count": len(edit_payloads),
        "effective_constraint_count": len(effective),
        "overridden_origin_constraint_ids": overridden_ids,
    }
    return effective, trace


def is_overridden(origin: LogicalConstraintObject, edit: LogicalConstraintObject) -> bool:
    if origin.type != edit.type and origin.type not in GLOBAL_OVERRIDE_TYPES:
        return False
    if origin.type in GLOBAL_OVERRIDE_TYPES and edit.type == origin.type:
        return True
    if origin.type == edit.type and origin.target == edit.target:
        return True
    if origin.type == edit.type and _same_global_entity_target(origin.target, edit.target):
        return True
    return False


def _same_global_entity_target(origin_target: Dict[str, Any], edit_target: Dict[str, Any]) -> bool:
    if not origin_target or not edit_target:
        return False
    origin_name = str(origin_target.get("poi_name") or origin_target.get("entity") or "").strip()
    edit_name = str(edit_target.get("poi_name") or edit_target.get("entity") or "").strip()
    if origin_name and edit_name and origin_name == edit_name:
        return True
    origin_type = str(origin_target.get("activity_type") or "").strip()
    edit_type = str(edit_target.get("activity_type") or "").strip()
    return bool(origin_type and edit_type and origin_type == edit_type and origin_name == edit_name)
