"""Strict adapters for origin-side benchmark logical truth."""

from __future__ import annotations

from typing import Any, Dict, List

from .models import LogicalConstraintObject, PreferenceConstraintObject
from .soft_truth import (
    build_origin_preference_constraints,
    validate_preference_tags,
)


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


def adapt_origin_query(origin_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = origin_payload or {}
    explicit_constraints = payload.get("origin_logical_constraints")
    if not isinstance(explicit_constraints, list) or not explicit_constraints:
        raise ValueError("origin_logical_constraints is required; runtime fallback parsing is disabled")

    if "origin_preference_tags" not in payload:
        raise ValueError("origin_preference_tags is required; runtime fallback parsing is disabled")
    raw_tags = payload.get("origin_preference_tags")
    if raw_tags is None:
        raw_tags = []
    if not isinstance(raw_tags, list):
        raise ValueError("origin_preference_tags must be a list")

    logical_constraints: List[LogicalConstraintObject] = [
        _logical_from_dict(index, item, "origin")
        for index, item in enumerate(explicit_constraints)
        if isinstance(item, dict)
    ]
    if not logical_constraints:
        raise ValueError("origin_logical_constraints is present but empty after deserialization")

    origin_query = payload.get("origin_query_structured", {})
    if not isinstance(origin_query, dict):
        origin_query = {}
    preference_constraints: List[PreferenceConstraintObject] = build_origin_preference_constraints(
        validate_preference_tags(raw_tags, require_edit_mode=False)
    )

    return {
        "logical_constraints": logical_constraints,
        "preference_constraints": preference_constraints,
        "metadata": {
            "days": origin_query.get("days"),
            "people_number": origin_query.get("people_number"),
            "start_city": origin_query.get("start_city"),
            "target_city": origin_query.get("target_city"),
        },
    }
