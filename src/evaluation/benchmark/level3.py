"""Level3 minimal-edit evaluation."""

from __future__ import annotations

import math
from typing import Any

from .diffing import infer_edit_sequence, sequence_metrics
from .models import Level3EvaluationResult


_CASCADE_COUNT_FIELDS = {
    "direct_target": ("four_way_direct_target_count",),
    "rollback_required_support": (
        "rollback_required_support_change_count",
        "hard_required_support_change_count",
    ),
    "scope_authorized_completion": ("scope_authorized_completion_change_count",),
    "verified_removable": (
        "verified_removable_change_count",
        "avoidable_extra_change_count",
    ),
    "unresolved": ("unresolved_change_count",),
}
_CASCADE_RATE_FIELDS = (
    "avoidable_lower_rate",
    "avoidable_upper_rate",
    "proof_coverage",
    "rollback_required_support_share",
    "hard_support_share",
    "authorized_completion_share",
    "unresolved_share",
)


def _empty_cascade_payload(
    *,
    status: str,
    reason: str,
    supported: bool | None,
    attribution_mode: Any = None,
    attribution_confidence: Any = None,
) -> dict[str, Any]:
    return {
        "algorithm": None,
        "metric_name": "verified_removable_change_rate",
        "metric_scope": "conditional_on_level1_and_level2_pass",
        "metric_interpretation": "evaluator_relative_not_global_irrelevance",
        "status": status,
        "reason": reason,
        "supported": supported,
        "target_satisfied": None,
        "counts": {key: None for key in _CASCADE_COUNT_FIELDS},
        "total_impact_count": None,
        **{field: None for field in _CASCADE_RATE_FIELDS},
        "verified_removable_change_count": None,
        "verified_removable_change_rate": None,
        "evaluator_relative_excess_change_rate_lower": None,
        "evaluator_relative_excess_change_rate_upper": None,
        "attribution_mode": attribution_mode,
        "attribution_confidence": attribution_confidence,
    }


def _validated_evaluated_cascade(cascade_result: dict[str, Any]) -> dict[str, Any]:
    missing = [
        sources[0]
        for sources in _CASCADE_COUNT_FIELDS.values()
        if not any(source in cascade_result for source in sources)
    ]
    missing.extend(
        field
        for field in (
            "four_way_total_impact_count",
            "avoidable_lower_rate",
            "avoidable_upper_rate",
            "proof_coverage",
            "authorized_completion_share",
            "unresolved_share",
        )
        if field not in cascade_result
    )
    if not any(
        field in cascade_result
        for field in ("rollback_required_support_share", "hard_support_share")
    ):
        missing.append("rollback_required_support_share")
    if missing:
        raise ValueError(f"cascade result lacks required fields: {', '.join(missing)}")
    counts: dict[str, int] = {}
    for name, sources in _CASCADE_COUNT_FIELDS.items():
        source = next(source for source in sources if source in cascade_result)
        value = cascade_result[source]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"cascade count {source} must be a non-negative integer")
        counts[name] = value
    total = cascade_result["four_way_total_impact_count"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("cascade total impact count must be a non-negative integer")
    if sum(counts.values()) != total:
        raise ValueError("cascade counts do not conserve total impact")
    expected = {
        "avoidable_lower_rate": counts["verified_removable"] / total if total else None,
        "avoidable_upper_rate": (
            (counts["verified_removable"] + counts["unresolved"]) / total
            if total else None
        ),
        "proof_coverage": (
            (
                counts["direct_target"]
                + counts["rollback_required_support"]
                + counts["verified_removable"]
            ) / total
            if total else None
        ),
        "rollback_required_support_share": (
            counts["rollback_required_support"] / total if total else None
        ),
        "authorized_completion_share": (
            counts["scope_authorized_completion"] / total if total else None
        ),
        "unresolved_share": counts["unresolved"] / total if total else None,
    }
    for field, expected_value in expected.items():
        source_field = field
        if field == "rollback_required_support_share" and field not in cascade_result:
            source_field = "hard_support_share"
        actual = cascade_result[source_field]
        if expected_value is None:
            if actual is not None:
                raise ValueError(f"cascade rate {field} must be null when total is zero")
        elif (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isclose(float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError(f"cascade rate {field} is inconsistent with counts")
    if not isinstance(cascade_result.get("target_satisfied"), bool):
        raise ValueError("evaluated cascade result requires boolean target_satisfied")
    return {
        "algorithm": cascade_result.get("algorithm"),
        "metric_name": "verified_removable_change_rate",
        "metric_scope": "conditional_on_level1_and_level2_pass",
        "metric_interpretation": "evaluator_relative_not_global_irrelevance",
        "status": "evaluated",
        "reason": str(cascade_result.get("reason") or "ok"),
        "supported": True,
        "target_satisfied": cascade_result["target_satisfied"],
        "counts": counts,
        "total_impact_count": total,
        "avoidable_lower_rate": cascade_result["avoidable_lower_rate"],
        "avoidable_upper_rate": cascade_result["avoidable_upper_rate"],
        "proof_coverage": cascade_result["proof_coverage"],
        "rollback_required_support_share": expected["rollback_required_support_share"],
        "hard_support_share": expected["rollback_required_support_share"],
        "authorized_completion_share": cascade_result["authorized_completion_share"],
        "unresolved_share": cascade_result["unresolved_share"],
        "verified_removable_change_count": counts["verified_removable"],
        "verified_removable_change_rate": cascade_result["avoidable_lower_rate"],
        "evaluator_relative_excess_change_rate_lower": cascade_result["avoidable_lower_rate"],
        "evaluator_relative_excess_change_rate_upper": cascade_result["avoidable_upper_rate"],
        "attribution_mode": cascade_result.get("attribution_mode"),
        "attribution_confidence": cascade_result.get("attribution_confidence"),
    }


def level3_cascade_payload(
    *, eligible: bool, reason: str, cascade_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical non-pass/fail Level3 cascade submetric payload.

    Missing tool evidence is represented by ``unresolved`` and by the gap
    between the avoidable lower/upper bounds.  It never changes eligibility.
    """
    if not eligible:
        return _empty_cascade_payload(
            status="not_evaluated",
            reason=f"level3_ineligible:{reason}",
            supported=None,
        )
    if not isinstance(cascade_result, dict):
        return _empty_cascade_payload(
            status="not_computed",
            reason="cascade_evidence_not_supplied",
            supported=None,
        )
    if not bool(cascade_result.get("supported")):
        return _empty_cascade_payload(
            status="not_computed",
            reason=str(cascade_result.get("reason") or "cascade_attribution_unsupported"),
            supported=False,
            attribution_mode=cascade_result.get("attribution_mode"),
            attribution_confidence=cascade_result.get("attribution_confidence"),
        )
    return _validated_evaluated_cascade(cascade_result)


def attach_level3_cascade(
    level3_result: dict[str, Any], cascade_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy with cascade formally nested under Level3."""
    enriched = dict(level3_result)
    enriched["cascade"] = level3_cascade_payload(
        eligible=bool(enriched.get("eligible")),
        reason=str(enriched.get("reason") or "unknown"),
        cascade_result=cascade_result,
    )
    return enriched


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
    *,
    cascade_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    level1_supported, level1_unsupported, level2_supported, level2_unsupported = _constraint_support_counts(
        level1_result,
        level2_result,
    )
    if not isinstance(origin_plan, dict) or not isinstance(edited_plan, dict):
        result = Level3EvaluationResult(
            eligible=False,
            reason="missing_origin_or_edited_plan",
            scope_level=0,
            scope_name="parameter",
            parameter_count=0,
            structural_count=0,
            compositional_count=0,
            atomic_counts={
                "change_time": 0,
                "change_transport": 0,
                "change_attribute": 0,
                "insert": 0,
                "delete": 0,
                "replace": 0,
                "reorder": 0,
            },
            edit_cost_tuple=[0, 0, 0],
            content_retention_rate=0.0,
            poi_seq_edit_distance=0,
            activity_change_ratio=0.0,
            level1_supported_constraints=level1_supported,
            level1_unsupported_constraints=level1_unsupported,
            level2_supported_constraints=level2_supported,
            level2_unsupported_constraints=level2_unsupported,
        ).to_dict()
        return attach_level3_cascade(result, None)

    level1_pass = bool(level1_result and level1_result.get("pass"))
    level2_pass = bool(level2_result and level2_result.get("pass"))

    if not level1_pass or not level2_pass:
        if not level1_pass and not level2_pass:
            reason = "level1_and_level2_failed"
        elif not level1_pass:
            reason = "level1_failed"
        else:
            reason = "level2_failed"
        result = Level3EvaluationResult(
            eligible=False,
            reason=reason,
            scope_level=0,
            scope_name="parameter",
            parameter_count=0,
            structural_count=0,
            compositional_count=0,
            atomic_counts={
                "change_time": 0,
                "change_transport": 0,
                "change_attribute": 0,
                "insert": 0,
                "delete": 0,
                "replace": 0,
                "reorder": 0,
            },
            edit_cost_tuple=[0, 0, 0],
            content_retention_rate=0.0,
            poi_seq_edit_distance=0,
            activity_change_ratio=0.0,
            level1_supported_constraints=level1_supported,
            level1_unsupported_constraints=level1_unsupported,
            level2_supported_constraints=level2_supported,
            level2_unsupported_constraints=level2_unsupported,
        ).to_dict()
        return attach_level3_cascade(result, None)

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
    return attach_level3_cascade(result.to_dict(), cascade_result)
