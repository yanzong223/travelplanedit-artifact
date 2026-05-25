"""Level3 minimal-edit evaluation."""

from __future__ import annotations

from typing import Any

from .diffing import infer_edit_sequence, sequence_metrics
from .models import Level3EvaluationResult


def _constraint_support_counts(
    level1_result: dict[str, Any] | None,
    level2_result: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    level1_preservation = (
        level1_result.get("origin_logical_preservation", {})
        if isinstance(level1_result, dict)
        else {}
    )
    level2_logical = (
        level2_result.get("edit_logical_success", {})
        if isinstance(level2_result, dict)
        else {}
    )
    return (
        int(level1_preservation.get("supported_constraints", 0) or 0),
        int(level1_preservation.get("unsupported_constraints", 0) or 0),
        int(level2_logical.get("supported_constraints", 0) or 0),
        int(level2_logical.get("unsupported_constraints", 0) or 0),
    )


def evaluate_level3(
    origin_plan: dict[str, Any] | None,
    edited_plan: dict[str, Any] | None,
    level1_result: dict[str, Any] | None,
    level2_result: dict[str, Any] | None,
) -> dict[str, Any]:
    level1_supported, level1_unsupported, level2_supported, level2_unsupported = _constraint_support_counts(
        level1_result,
        level2_result,
    )
    if not isinstance(origin_plan, dict) or not isinstance(edited_plan, dict):
        return Level3EvaluationResult(
            eligible=False,
            reason="missing_origin_or_edited_plan",
            scope_level=0,
            scope_name="parameter",
            parameter_count=0,
            structural_count=0,
            compositional_count=0,
            atomic_counts={"change_time": 0, "insert": 0, "delete": 0, "replace": 0, "reorder": 0},
            edit_cost_tuple=[0, 0, 0],
            content_retention_rate=0.0,
            poi_seq_edit_distance=0,
            activity_change_ratio=0.0,
            level1_supported_constraints=level1_supported,
            level1_unsupported_constraints=level1_unsupported,
            level2_supported_constraints=level2_supported,
            level2_unsupported_constraints=level2_unsupported,
        ).to_dict()

    level1_pass = bool(level1_result and level1_result.get("pass"))
    level2_pass = bool(level2_result and level2_result.get("pass"))

    if not level1_pass or not level2_pass:
        if not level1_pass and not level2_pass:
            reason = "level1_and_level2_failed"
        elif not level1_pass:
            reason = "level1_failed"
        else:
            reason = "level2_failed"
        return Level3EvaluationResult(
            eligible=False,
            reason=reason,
            scope_level=0,
            scope_name="parameter",
            parameter_count=0,
            structural_count=0,
            compositional_count=0,
            atomic_counts={"change_time": 0, "insert": 0, "delete": 0, "replace": 0, "reorder": 0},
            edit_cost_tuple=[0, 0, 0],
            content_retention_rate=0.0,
            poi_seq_edit_distance=0,
            activity_change_ratio=0.0,
            level1_supported_constraints=level1_supported,
            level1_unsupported_constraints=level1_unsupported,
            level2_supported_constraints=level2_supported,
            level2_unsupported_constraints=level2_unsupported,
        ).to_dict()

    sequence = infer_edit_sequence(origin_plan, edited_plan)
    metrics = sequence_metrics(origin_plan, edited_plan, sequence)
    result = Level3EvaluationResult(
        eligible=True,
        reason="ok",
        scope_level=sequence.scope_level,
        scope_name=sequence.scope_name,
        parameter_count=metrics["parameter_count"],
        structural_count=metrics["structural_count"],
        compositional_count=metrics["compositional_count"],
        atomic_counts=metrics["atomic_counts"],
        edit_cost_tuple=[sequence.scope_level, metrics["structural_count"], metrics["parameter_count"]],
        content_retention_rate=metrics["content_retention_rate"],
        poi_seq_edit_distance=metrics["poi_seq_edit_distance"],
        activity_change_ratio=metrics["activity_change_ratio"],
        level1_supported_constraints=level1_supported,
        level1_unsupported_constraints=level1_unsupported,
        level2_supported_constraints=level2_supported,
        level2_unsupported_constraints=level2_unsupported,
        matched_pairs=[item.to_dict() for item in sequence.matched_pairs],
        unmatched_origin=sequence.unmatched_origin,
        unmatched_edited=sequence.unmatched_edited,
    )
    return result.to_dict()
