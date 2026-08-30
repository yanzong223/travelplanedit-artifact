"""Aggregate benchmark evaluation outputs into dataset-level reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _rate(passed: int, total: int) -> float:
    return passed / total if total else 0.0


def aggregate_level3_cascade(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the optional, eligible-only Level3 cascade submetrics."""
    statuses: dict[str, int] = defaultdict(int)
    evaluated: list[dict[str, Any]] = []
    for item in results:
        level3 = item.get("level3") if isinstance(item, dict) else None
        if not isinstance(level3, dict):
            continue
        cascade = level3.get("cascade") if isinstance(level3, dict) else None
        status = str(cascade.get("status") or "not_computed") if isinstance(cascade, dict) else "not_computed"
        statuses[status] += 1
        if status == "evaluated" and isinstance(cascade, dict):
            evaluated.append(cascade)

    count_names = (
        "direct_target", "rollback_required_support", "scope_authorized_completion",
        "verified_removable", "unresolved",
    )
    legacy_count_names = {
        "rollback_required_support": "hard_required_support",
        "verified_removable": "avoidable_extra",
    }
    count_sums = {
        name: sum(
            int(
                (row.get("counts") or {}).get(
                    name,
                    (row.get("counts") or {}).get(legacy_count_names.get(name, ""), 0),
                )
                or 0
            )
            for row in evaluated
        )
        for name in count_names
    }
    evaluated_n = len(evaluated)
    algorithms = {
        (
            str((row.get("algorithm") or {}).get("id") or ""),
            str((row.get("algorithm") or {}).get("version") or ""),
        )
        for row in evaluated
        if isinstance(row.get("algorithm"), dict)
    }

    def values(field: str) -> list[float]:
        return [
            float(row[field])
            for row in evaluated
            if isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
        ]

    def mean(field: str) -> float | None:
        field_values = values(field)
        return sum(field_values) / len(field_values) if field_values else None

    total_impact = sum(int(row.get("total_impact_count", 0) or 0) for row in evaluated)
    proven = (
        count_sums["direct_target"]
        + count_sums["rollback_required_support"]
        + count_sums["verified_removable"]
    )
    mean_fields = (
        "avoidable_lower_rate",
        "avoidable_upper_rate",
        "proof_coverage",
        "rollback_required_support_share",
        "hard_support_share",
        "authorized_completion_share",
        "unresolved_share",
    )
    return {
        "algorithms": [
            {"id": algorithm_id, "version": version}
            for algorithm_id, version in sorted(algorithms)
        ],
        "definition": (
            "Verified Removable-Change Rate is conditional on Level 1 and Level 2 "
            "passing. It is evaluator-relative and is not a third success gate."
        ),
        "metric_name": "verified_removable_change_rate",
        "records": sum(statuses.values()),
        "evaluated_records": evaluated_n,
        "not_evaluated_records": statuses.get("not_evaluated", 0),
        "not_computed_records": statuses.get("not_computed", 0),
        "status_counts": dict(sorted(statuses.items())),
        "mean_counts": {
            name: count_sums[name] / evaluated_n if evaluated_n else None
            for name in count_names
        },
        "pooled_counts": count_sums,
        "pooled_total_impact_count": total_impact,
        "mean_effective_n": {
            field: len(values(field))
            for field in mean_fields
        },
        "mean_avoidable_lower_rate": mean("avoidable_lower_rate"),
        "mean_avoidable_upper_rate": mean("avoidable_upper_rate"),
        "mean_proof_coverage": mean("proof_coverage"),
        "mean_hard_support_share": mean("hard_support_share"),
        "mean_rollback_required_support_share": (
            mean("rollback_required_support_share")
            if values("rollback_required_support_share")
            else mean("hard_support_share")
        ),
        "mean_authorized_completion_share": mean("authorized_completion_share"),
        "mean_unresolved_share": mean("unresolved_share"),
        "pooled_avoidable_lower_rate": (
            count_sums["verified_removable"] / total_impact if total_impact else None
        ),
        "pooled_avoidable_upper_rate": (
            (count_sums["verified_removable"] + count_sums["unresolved"]) / total_impact
            if total_impact else None
        ),
        "pooled_proof_coverage": proven / total_impact if total_impact else None,
        "mean_verified_removable_change_rate": mean("avoidable_lower_rate"),
        "pooled_verified_removable_change_rate": (
            count_sums["verified_removable"] / total_impact if total_impact else None
        ),
        "pooled_evaluator_relative_excess_change_rate_lower": (
            count_sums["verified_removable"] / total_impact if total_impact else None
        ),
        "pooled_evaluator_relative_excess_change_rate_upper": (
            (count_sums["verified_removable"] + count_sums["unresolved"]) / total_impact
            if total_impact
            else None
        ),
    }


def _finalize_support_breakdown(raw: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    finalized: dict[str, dict[str, float | int]] = {}
    for key, counts in raw.items():
        total = counts["total"]
        supported = counts["supported"]
        passed = counts["passed"]
        finalized[key] = {
            "total": total,
            "supported": supported,
            "unsupported": total - supported,
            "passed": passed,
            "failed": supported - passed,
            "pass_rate": _rate(passed, supported),
        }
    return finalized


def _finalize_check(
    total: int,
    passed: int,
    *,
    evaluable: int | None = None,
    rate_denominator: int | None = None,
) -> dict[str, float | int]:
    effective_evaluable = total if evaluable is None else evaluable
    denominator = effective_evaluable if rate_denominator is None else rate_denominator
    payload: dict[str, float | int] = {
        "total": total,
        "evaluable": effective_evaluable,
        "non_evaluable": total - effective_evaluable,
        "passed": passed,
        "failed": effective_evaluable - passed,
        "pass_rate": _rate(passed, denominator),
    }
    if evaluable is not None and denominator != effective_evaluable:
        payload["evaluable_pass_rate"] = _rate(passed, effective_evaluable)
    return payload


def _finalize_scope_distribution(raw: dict[str, int], total: int) -> dict[str, dict[str, float | int]]:
    finalized: dict[str, dict[str, float | int]] = {}
    for key, count in raw.items():
        finalized[key] = {
            "count": count,
            "rate": _rate(count, total),
        }
    return finalized


def _experiment_check(
    *,
    display_name: str,
    definition: str,
    denominator_definition: str,
    payload: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "definition": definition,
        "denominator_definition": denominator_definition,
        **payload,
    }


# ---------------------------------------------------------------------------
# Enhanced violation summary (Gap 1)
# ---------------------------------------------------------------------------


def _summarize_violations(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Return all violation codes with counts, not just the first one."""
    if not isinstance(items, list) or not items:
        return {"top_code": None, "all_codes": [], "code_counts": {}}
    counts: dict[str, int] = {}
    all_codes: list[str] = []
    for item in items:
        if isinstance(item, dict):
            code = item.get("code") or item.get("violation_code")
            if isinstance(code, str) and code:
                all_codes.append(code)
                counts[code] = counts.get(code, 0) + 1
    return {
        "top_code": all_codes[0] if all_codes else None,
        "all_codes": all_codes,
        "code_counts": counts,
    }


def _top_violation_code(items: list[dict[str, Any]] | None) -> str | None:
    """Backward-compatible wrapper: return only the first violation code."""
    return _summarize_violations(items)["top_code"]


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Run reason classification (Gap 7 — enriched with terminal_error_categories)
# ---------------------------------------------------------------------------


def _classify_run_reason(
    errors: list[str],
    terminal_error_categories: list[str] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Classify run errors with optional framework-level terminal categories.

    Returns (reason_code, source, diagnostics).
    """
    joined = " | ".join(errors)
    lowered = joined.lower()
    terminal_categories = list(terminal_error_categories or [])

    reason_code: str
    source: str

    if "quota" in lowered or "401" in lowered:
        reason_code, source = "api_quota", "runner"
    elif "json" in lowered and ("parse" in lowered or "unterminated" in lowered):
        reason_code, source = "json_parse_error", "runner"
    elif "validation" in lowered or "missing required" in lowered or "field required" in lowered:
        reason_code, source = "tool_schema_validation", "tool_contract"
    elif "canonical chinatravel plan" in lowered or "must be one of" in lowered:
        reason_code, source = "invalid_final_plan_schema", "schema_validation"
    elif "maximum steps" in lowered or "step budget" in lowered:
        reason_code, source = "step_budget_exceeded", "runner"
    elif "maximum tool calls" in lowered or "tool calls" in lowered:
        reason_code, source = "tool_budget_exceeded", "runner"
    elif "typeerror" in lowered or "nonetype" in lowered or "keyerror" in lowered:
        reason_code, source = "runtime_type_error", "runner"
    else:
        reason_code, source = "unknown_run_error", "runner"

    # Enrich with terminal categories for secondary classification signal
    step_budget_patterns = {"step_budget_exceeded", "tool_budget_exceeded"}
    if reason_code in step_budget_patterns and terminal_categories:
        if "ops_parse_error" in terminal_categories:
            reason_code = "step_budget_exceeded_parse_loop"
        elif "patch_execution_error" in terminal_categories:
            reason_code = "step_budget_exceeded_patch_loop"
        elif "invalid_edited_plan" in terminal_categories:
            reason_code = "step_budget_exceeded_invalid_plan"

    diagnostics: dict[str, Any] = {
        "error_pattern_matched": reason_code,
        "terminal_categories": terminal_categories,
    }
    return reason_code, source, diagnostics


# ---------------------------------------------------------------------------
# New attribution helpers (Gaps 3–8, 10)
# ---------------------------------------------------------------------------


def _build_origin_failure_comparison(
    feasibility: dict[str, Any],
    level1: dict[str, Any],
) -> dict[str, Any]:
    """Compare origin vs edited plan feasibility to classify failure provenance.

    Returns:
        failure_provenance: "pre_existing" | "edit_introduced" | "mixed" | "unknown"
        origin_passed: bool | None
        edited_passed: bool
        introduced_violation_codes: violation codes only in edited plan
    """
    origin = level1.get("origin_feasibility")
    if not isinstance(origin, dict):
        return {
            "failure_provenance": "unknown",
            "origin_passed": None,
            "edited_passed": False,
        }

    edited_passed = bool(feasibility.get("pass"))
    origin_passed = bool(origin.get("pass"))

    relative = feasibility.get("relative_non_regression")
    if isinstance(relative, dict):
        attributable = relative.get("new_or_worsened_violations")
        attributable = attributable if isinstance(attributable, list) else []
        introduced = sorted({
            str(item.get("code"))
            for item in attributable
            if isinstance(item, dict) and item.get("code")
        })
        classifications = sorted({
            str(item.get("classification"))
            for item in attributable
            if isinstance(item, dict) and item.get("classification")
        })
        return {
            "failure_provenance": "edit_introduced" if attributable else "pre_existing",
            "origin_passed": origin_passed,
            "edited_passed": edited_passed,
            "edited_absolute_passed": bool(feasibility.get("absolute_pass")),
            "scoring_mode": feasibility.get("scoring_mode"),
            "introduced_violation_codes": introduced,
            "relative_classifications": classifications,
            "inherited_violation_count": int(relative.get("inherited_count", 0) or 0),
            "resolved_violation_count": int(relative.get("resolved_count", 0) or 0),
        }

    def _collect_codes(section: dict[str, Any]) -> set[str]:
        codes: set[str] = set()
        for key in ("hygiene_violations", "quality_violations"):
            for item in section.get(key, []) or []:
                if isinstance(item, dict):
                    code = item.get("code")
                    if isinstance(code, str) and code:
                        codes.add(code)
        return codes

    edited_codes = _collect_codes(feasibility)
    origin_codes = _collect_codes(origin)
    introduced = edited_codes - origin_codes

    if not edited_passed and not origin_passed and introduced:
        provenance = "mixed"
    elif not edited_passed and origin_passed:
        provenance = "edit_introduced"
    elif not edited_passed and not origin_passed and not introduced:
        provenance = "pre_existing"
    else:
        provenance = "pre_existing"

    return {
        "failure_provenance": provenance,
        "origin_passed": origin_passed,
        "edited_passed": edited_passed,
        "introduced_violation_codes": sorted(introduced),
        "origin_violation_codes": sorted(origin_codes),
        "edited_violation_codes": sorted(edited_codes),
    }


def _build_preference_degradation_summary(
    preference_section: dict[str, Any],
) -> dict[str, Any]:
    """Quantify preference degradation: which preferences, by how much."""
    results = preference_section.get("per_preference_results", [])
    degraded: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for entry in results:
        if not isinstance(entry, dict):
            continue
        if not entry.get("supported"):
            unsupported.append({
                "family": entry.get("family"),
                "facet": entry.get("facet"),
                "reason": entry.get("reason", ""),
            })
        elif not entry.get("passed"):
            degraded.append({
                "family": entry.get("family"),
                "facet": entry.get("facet"),
                "canonical_facet": entry.get("canonical_facet", entry.get("facet")),
                "score_before": entry.get("score_before"),
                "score_after": entry.get("score_after"),
                "delta": entry.get("delta"),
                "degradation_magnitude": abs(entry.get("delta", 0.0)),
            })
        else:
            preserved.append({
                "family": entry.get("family"),
                "facet": entry.get("facet"),
                "canonical_facet": entry.get("canonical_facet", entry.get("facet")),
            })

    total_supported = len(degraded) + len(preserved)
    return {
        "total_supported_preferences": total_supported,
        "unsupported_count": len(unsupported),
        "degraded_count": len(degraded),
        "preserved_count": len(preserved),
        "degradation_rate": len(degraded) / total_supported if total_supported else 0.0,
        "degraded_preferences": degraded,
        "preserved_preferences": preserved,
        "max_degradation": max(
            (d["degradation_magnitude"] for d in degraded), default=0.0
        ),
    }


def _build_edit_intent_attribution(
    logical: dict[str, Any],
    preference: dict[str, Any],
) -> dict[str, Any]:
    """Connect failures to specific edit constraints that were targeted."""
    failed_logical: list[dict[str, Any]] = []
    for entry in logical.get("per_constraint_results", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("supported") and not entry.get("passed"):
            details = entry.get("details", {}) if isinstance(entry.get("details"), dict) else {}
            failed_logical.append({
                "constraint_id": entry.get("constraint_id"),
                "constraint_type": entry.get("constraint_type"),
                "reason": entry.get("reason"),
                "expected": details.get("expected"),
                "actual": details.get("actual"),
            })

    failed_preferences: list[dict[str, Any]] = []
    for entry in preference.get("per_preference_results", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("supported") and not entry.get("passed"):
            failed_preferences.append({
                "preference_id": entry.get("preference_id"),
                "family": entry.get("family"),
                "facet": entry.get("facet"),
                "reason": entry.get("reason"),
                "score_before": entry.get("score_before"),
                "score_after": entry.get("score_after"),
                "delta": entry.get("delta"),
            })

    return {
        "targeted_logical_constraints": len(logical.get("per_constraint_results", [])),
        "failed_logical_count": len(failed_logical),
        "failed_logical": failed_logical,
        "targeted_preferences": len(preference.get("per_preference_results", [])),
        "failed_preference_count": len(failed_preferences),
        "failed_preferences": failed_preferences,
    }


def _build_level2_diagnostic_detail(
    logical: dict[str, Any],
    preference: dict[str, Any],
) -> dict[str, Any]:
    """Build diagnostic detail for Level2 failures with expected vs actual."""
    logical_diagnostics: list[dict[str, Any]] = []
    for entry in logical.get("per_constraint_results", []):
        if not isinstance(entry, dict):
            continue
        details = entry.get("details", {}) if isinstance(entry.get("details"), dict) else {}
        logical_diagnostics.append({
            "constraint_id": entry.get("constraint_id"),
            "constraint_type": entry.get("constraint_type"),
            "supported": entry.get("supported"),
            "passed": entry.get("passed"),
            "reason": entry.get("reason"),
            "expected": details.get("expected"),
            "actual": details.get("actual"),
            "constraint_spec": details.get("constraint"),
        })

    preference_diagnostics: list[dict[str, Any]] = []
    for entry in preference.get("per_preference_results", []):
        if not isinstance(entry, dict):
            continue
        pref_diag: dict[str, Any] = {
            "preference_id": entry.get("preference_id"),
            "family": entry.get("family"),
            "facet": entry.get("facet"),
            "supported": entry.get("supported"),
            "passed": entry.get("passed"),
            "reason": entry.get("reason"),
            "score_before": entry.get("score_before"),
            "score_after": entry.get("score_after"),
            "delta": entry.get("delta"),
        }
        preference_diagnostics.append(pref_diag)

    return {
        "logical_diagnostics": logical_diagnostics,
        "preference_diagnostics": preference_diagnostics,
    }


def _build_failure_chain(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Trace how failures propagate across Level1, Level2, Level3."""
    chain: list[dict[str, Any]] = []
    run_context = result.get("run_context", {})
    if isinstance(run_context, dict) and run_context.get("errors"):
        chain.append({
            "stage": "run",
            "reason": " | ".join(
                _normalize_text(e) for e in run_context.get("errors", [])[:3]
                if _normalize_text(e)
            ),
            "blocked_downstream": True,
        })

    level1 = result.get("level1", {})
    if isinstance(level1, dict) and not level1.get("pass"):
        feasibility = level1.get("feasibility", {})
        preservation = level1.get("origin_logical_preservation", {})
        pref_pres = level1.get("origin_preference_preservation", {})
        chain.append({
            "stage": "level1",
            "feasibility_pass": bool(feasibility.get("pass")),
            "origin_preservation_pass": bool(preservation.get("pass")),
            "preference_preservation_pass": bool(pref_pres.get("pass")),
            "blocked_downstream": True,
        })

    level2 = result.get("level2", {})
    if isinstance(level2, dict) and not level2.get("pass"):
        logical = level2.get("edit_logical_success", {})
        preference = level2.get("edit_preference_success", {})
        chain.append({
            "stage": "level2",
            "logical_pass": bool(logical.get("pass")),
            "preference_pass": bool(preference.get("pass")),
            "blocked_downstream": True,
        })

    level3 = result.get("level3", {})
    if isinstance(level3, dict) and not level3.get("eligible"):
        chain.append({
            "stage": "level3",
            "eligible": False,
            "reason": level3.get("reason", ""),
            "blocked_downstream": False,
        })

    if not chain:
        chain.append({"stage": "all", "passed": True, "blocked_downstream": False})
    return chain


def _build_per_constraint_type_breakdown(result: dict[str, Any]) -> dict[str, Any]:
    """Extract per-constraint-type pass/fail at sample level."""
    level1 = result.get("level1", {})
    preservation = level1.get("origin_logical_preservation", {})
    level2 = result.get("level2", {})
    logical = level2.get("edit_logical_success", {})

    level1_breakdown: dict[str, dict[str, int]] = {}
    for entry in preservation.get("per_constraint_results", []):
        key = str(entry.get("constraint_type", "unknown"))
        if key not in level1_breakdown:
            level1_breakdown[key] = {"total": 0, "passed": 0, "supported": 0}
        level1_breakdown[key]["total"] += 1
        if entry.get("supported"):
            level1_breakdown[key]["supported"] += 1
        if entry.get("supported") and entry.get("passed"):
            level1_breakdown[key]["passed"] += 1

    level2_breakdown: dict[str, dict[str, int]] = {}
    for entry in logical.get("per_constraint_results", []):
        key = str(entry.get("constraint_type", "unknown"))
        if key not in level2_breakdown:
            level2_breakdown[key] = {"total": 0, "passed": 0, "supported": 0}
        level2_breakdown[key]["total"] += 1
        if entry.get("supported"):
            level2_breakdown[key]["supported"] += 1
        if entry.get("supported") and entry.get("passed"):
            level2_breakdown[key]["passed"] += 1

    return {
        "level1_origin_preservation": level1_breakdown,
        "level2_edit_logical": level2_breakdown,
    }


# ---------------------------------------------------------------------------
# Core failure attribution (Gaps 1–8, 10 integrated)
# ---------------------------------------------------------------------------


def _build_sample_failure_from_result(result: dict[str, Any]) -> dict[str, Any]:
    record_id = str(
        result.get("record_id")
        or result.get("sample_id")
        or result.get("plan_id")
        or "unknown_record"
    )
    run_context = result.get("run_context", {}) if isinstance(result.get("run_context"), dict) else {}
    run_errors = run_context.get("errors")
    if not isinstance(run_errors, list):
        run_errors = []
    run_errors = [_normalize_text(item) for item in run_errors if _normalize_text(item)]

    terminal_categories = run_context.get("terminal_error_categories", [])
    if not isinstance(terminal_categories, list):
        terminal_categories = []

    conflict_labels = result.get("conflict_labels")

    if run_errors:
        reason_code, source, run_diagnostics = _classify_run_reason(run_errors, terminal_categories)
        return {
            "record_id": record_id,
            "failure_stage": "run",
            "reason_code": reason_code,
            "reason_detail": " | ".join(run_errors),
            "severity": "error",
            "source": source,
            "reason_diagnostic": run_diagnostics,
            "failure_chain": _build_failure_chain(result),
            "conflict_labels": conflict_labels,
        }

    level1 = result.get("level1", {}) if isinstance(result.get("level1"), dict) else {}
    level2 = result.get("level2", {}) if isinstance(result.get("level2"), dict) else {}
    level3 = result.get("level3", {}) if isinstance(result.get("level3"), dict) else {}
    feasibility = level1.get("feasibility", {}) if isinstance(level1.get("feasibility"), dict) else {}
    preservation = (
        level1.get("origin_logical_preservation", {})
        if isinstance(level1.get("origin_logical_preservation"), dict)
        else {}
    )
    preference_preservation = (
        level1.get("origin_preference_preservation", {})
        if isinstance(level1.get("origin_preference_preservation"), dict)
        else {}
    )
    logical = (
        level2.get("edit_logical_success", {})
        if isinstance(level2.get("edit_logical_success"), dict)
        else {}
    )
    preference = (
        level2.get("edit_preference_success", {})
        if isinstance(level2.get("edit_preference_success"), dict)
        else {}
    )

    feasibility_reason = _normalize_text(feasibility.get("reason"))
    logical_reason = _normalize_text(logical.get("reason") or level2.get("reason"))
    level3_reason = _normalize_text(level3.get("reason"))

    # Early-return error cases (unchanged logic)
    if feasibility_reason == "infeasible_detection" or logical_reason == "infeasible_detection":
        detection = feasibility.get("infeasible_detection") if isinstance(feasibility, dict) else None
        if not isinstance(detection, dict):
            detection = logical.get("infeasible_detection") if isinstance(logical, dict) else {}
        return {
            "record_id": record_id,
            "failure_stage": "run",
            "reason_code": "infeasible_detection",
            "reason_detail": _normalize_text(detection.get("reason")) or "infeasible_detection",
            "severity": "failure",
            "source": "model",
            "infeasible_detection": detection,
            "failure_chain": _build_failure_chain(result),
            "conflict_labels": conflict_labels,
        }
    if feasibility_reason == "missing_edited_plan":
        return {
            "record_id": record_id,
            "failure_stage": "evaluation",
            "reason_code": "missing_edited_plan",
            "reason_detail": feasibility_reason,
            "severity": "error",
            "source": "benchmark",
            "failure_chain": _build_failure_chain(result),
            "conflict_labels": conflict_labels,
        }
    if logical_reason == "missing_ir":
        detail = logical.get("errors") if isinstance(logical.get("errors"), list) else []
        return {
            "record_id": record_id,
            "failure_stage": "evaluation",
            "reason_code": "missing_ir",
            "reason_detail": " | ".join(_normalize_text(item) for item in detail if _normalize_text(item)) or logical_reason,
            "severity": "error",
            "source": "benchmark",
            "failure_chain": _build_failure_chain(result),
            "conflict_labels": conflict_labels,
        }
    if feasibility_reason == "invalid_edited_plan" or logical_reason == "invalid_edited_plan":
        issues = feasibility.get("schema_issues") if isinstance(feasibility.get("schema_issues"), list) else []
        return {
            "record_id": record_id,
            "failure_stage": "evaluation",
            "reason_code": "invalid_edited_plan",
            "reason_detail": " | ".join(_normalize_text(item) for item in issues if _normalize_text(item)) or "invalid_edited_plan",
            "severity": "error",
            "source": "schema_validation",
            "failure_chain": _build_failure_chain(result),
            "conflict_labels": conflict_labels,
        }

    # --- Level1 failure (enriched) ---
    if level1 and not level1.get("pass"):
        feasibility_pass = bool(feasibility.get("pass"))
        preservation_pass = bool(preservation.get("pass"))
        preference_preservation_pass = bool(preference_preservation.get("pass"))
        if not feasibility_pass and not preservation_pass and not preference_preservation_pass:
            reason_code = "feasibility_and_hard_and_soft_preservation_failed"
        elif not feasibility_pass and not preservation_pass:
            reason_code = "feasibility_and_hard_preservation_failed"
        elif not feasibility_pass and not preference_preservation_pass:
            reason_code = "feasibility_and_soft_preservation_failed"
        elif not preservation_pass and not preference_preservation_pass:
            reason_code = "hard_and_soft_preservation_failed"
        elif not feasibility_pass:
            reason_code = "feasibility_failed"
        elif not preservation_pass:
            reason_code = "origin_preservation_failed"
        else:
            reason_code = "origin_preference_preservation_failed"

        relative = feasibility.get("relative_non_regression")
        relative_violations = (
            relative.get("new_or_worsened_violations")
            if isinstance(relative, dict)
            else None
        )
        if isinstance(relative_violations, list) and relative_violations:
            violation_summary = _summarize_violations(relative_violations)
        else:
            absolute_violations = list(feasibility.get("hygiene_violations") or [])
            absolute_violations.extend(feasibility.get("quality_violations") or [])
            violation_summary = _summarize_violations(absolute_violations)
        top_code = violation_summary["top_code"] or reason_code

        return {
            "record_id": record_id,
            "failure_stage": "level1",
            "reason_code": reason_code,
            "reason_detail": top_code,
            "severity": "failure",
            "source": "constraint_verifier",
            # New enriched fields
            "violation_summary": violation_summary,
            "origin_feasibility_comparison": _build_origin_failure_comparison(feasibility, level1),
            "preference_degradation": _build_preference_degradation_summary(preference_preservation),
            "failure_chain": _build_failure_chain(result),
            "per_constraint_type_breakdown": _build_per_constraint_type_breakdown(result),
            "conflict_labels": conflict_labels,
        }

    # --- Level2 failure (enriched) ---
    if level2 and not level2.get("pass"):
        logical_supported = int(logical.get("supported_constraints", 0) or 0)
        preference_supported = int(preference.get("supported_preferences", 0) or 0)
        logical_failed = logical_supported > 0 and not bool(logical.get("pass"))
        preference_failed = preference_supported > 0 and not bool(preference.get("pass"))
        if logical_supported == 0 and preference_supported == 0:
            reason_code = "no_supported_edit_targets"
        elif logical_failed and preference_failed:
            reason_code = "logical_and_preference_failed"
        elif logical_failed:
            reason_code = "logical_target_failed"
        else:
            reason_code = "preference_target_failed"

        return {
            "record_id": record_id,
            "failure_stage": "level2",
            "reason_code": reason_code,
            "reason_detail": reason_code,
            "severity": "failure",
            "source": "constraint_verifier",
            # New enriched fields
            "edit_intent_attribution": _build_edit_intent_attribution(logical, preference),
            "diagnostic_detail": _build_level2_diagnostic_detail(logical, preference),
            "failure_chain": _build_failure_chain(result),
            "per_constraint_type_breakdown": _build_per_constraint_type_breakdown(result),
            "conflict_labels": conflict_labels,
        }

    # --- Level3 non-eligible (enriched) ---
    if level3 and not bool(level3.get("eligible", True)):
        reason_code = level3_reason or "level3_not_eligible"
        return {
            "record_id": record_id,
            "failure_stage": "level3",
            "reason_code": reason_code,
            "reason_detail": level3_reason or reason_code,
            "severity": "non_evaluable",
            "source": "level3_gate",
            "failure_chain": _build_failure_chain(result),
            "conflict_labels": conflict_labels,
        }

    return {
        "record_id": record_id,
        "failure_stage": "none",
        "reason_code": "ok",
        "reason_detail": "",
        "severity": "none",
        "source": "benchmark",
        "failure_chain": _build_failure_chain(result),
        "conflict_labels": conflict_labels,
    }


def _build_sample_failure_from_evaluation_failure(item: dict[str, Any]) -> dict[str, Any]:
    detail = _normalize_text(item.get("error"))
    return {
        "record_id": _normalize_text(item.get("record_id") or item.get("file_path") or "unknown_record"),
        "failure_stage": "evaluation",
        "reason_code": "evaluation_exception",
        "reason_detail": detail,
        "severity": "error",
        "source": "benchmark",
    }


def _summarize_failures(sample_failures: list[dict[str, Any]], denominator: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], int] = defaultdict(int)
    for item in sample_failures:
        stage = _normalize_text(item.get("failure_stage"))
        reason_code = _normalize_text(item.get("reason_code"))
        if not stage or stage == "none":
            continue
        buckets[(stage, reason_code)] += 1
    rows = []
    for (stage, reason_code), count in sorted(buckets.items()):
        rows.append(
            {
                "failure_stage": stage,
                "reason_code": reason_code,
                "count": count,
                "rate": _rate(count, denominator),
            }
        )
    return rows


def _build_experiment_view(report: dict[str, Any]) -> dict[str, Any]:
    per_level = report.get("per_level_summary", {})
    level1 = per_level.get("level1", {}) if isinstance(per_level.get("level1"), dict) else {}
    level2 = per_level.get("level2", {}) if isinstance(per_level.get("level2"), dict) else {}
    level3 = per_level.get("level3", {}) if isinstance(per_level.get("level3"), dict) else {}
    oracle_scope = report.get("oracle_scope", {}) if isinstance(report.get("oracle_scope"), dict) else {}
    return {
        "level1": {
            "feasibility": _experiment_check(
                display_name="Level1 Feasibility",
                definition="Edited plan remains executable under benchmark feasibility checks.",
                denominator_definition="All records with Level1 evaluation.",
                payload=level1.get("feasibility", {}),
            ),
            "origin_logical_preservation": _experiment_check(
                display_name="Level1 Origin Logical Preservation",
                definition="Untargeted origin hard constraints are preserved after editing.",
                denominator_definition="All records with Level1 evaluation.",
                payload=level1.get("origin_logical_preservation", {}),
            ),
            "origin_preference_preservation": _experiment_check(
                display_name="Level1 Origin Preference Preservation",
                definition="Untargeted origin soft preferences are preserved within non-degradation tolerance.",
                denominator_definition="All records with Level1 evaluation.",
                payload=level1.get("origin_preference_preservation", {}),
            ),
            "origin_preference_preservation_supported": _experiment_check(
                display_name="Level1 Origin Preference Preservation (Supported Only)",
                definition="Untargeted origin soft preferences are preserved among records with supported preference evidence.",
                denominator_definition="Records with at least one supported origin preference.",
                payload=level1.get("origin_preference_preservation_supported", {}),
            ),
            "overall": _experiment_check(
                display_name="Level1 Overall",
                definition="Feasibility, origin logical preservation, and origin preference preservation all pass.",
                denominator_definition="All records with Level1 evaluation.",
                payload=level1.get("overall", {}),
            ),
        },
        "level2": {
            "edit_logical_success": _experiment_check(
                display_name="Level2 Logical Success",
                definition="Supported hard edit targets are satisfied.",
                denominator_definition="All Level2 records; logical-evaluable count is reported separately.",
                payload=level2.get("edit_logical_success", {}),
            ),
            "edit_preference_success": _experiment_check(
                display_name="Level2 Preference Success",
                definition="Supported preference edit targets are satisfied.",
                denominator_definition="All Level2 records; preference-evaluable count is reported separately.",
                payload=level2.get("edit_preference_success", {}),
            ),
            "overall": _experiment_check(
                display_name="Level2 Overall (Combined Edit Success)",
                definition="Logical and preference edit targets pass together when evaluable.",
                denominator_definition="All Level2 records; combined-evaluable count is reported separately.",
                payload=level2.get("overall", {}),
            ),
        },
        "level3": {
            "eligible": _experiment_check(
                display_name="Level3 Eligible Rate",
                definition="Records that pass Level1 and Level2 and therefore enter minimal-edit analysis.",
                denominator_definition="All records with Level3 evaluation.",
                payload=level3.get("eligible", {}),
            ),
            "averages": level3.get("averages", {}),
            "scope_distribution": level3.get("scope_distribution", {}),
            "cascade": level3.get("cascade", {}),
        },
        "diagnostic_oracle_scope": oracle_scope,
    }


def attach_report_views(report: dict[str, Any]) -> dict[str, Any]:
    report["experiment_view"] = _build_experiment_view(report)
    results = report.get("results", [])
    sample_failures = [
        _build_sample_failure_from_result(item)
        for item in results
        if isinstance(item, dict)
    ]
    evaluation_failures = report.get("evaluation_failures", [])
    for item in evaluation_failures:
        if isinstance(item, dict):
            sample_failures.append(_build_sample_failure_from_evaluation_failure(item))
    denominator = int(report.get("summary", {}).get("total_records", 0) or 0) + len(
        [item for item in evaluation_failures if isinstance(item, dict)]
    )
    report["sample_failures"] = sample_failures
    report["failure_summary"] = _summarize_failures(sample_failures, denominator)
    return report


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    level1_count = sum(1 for item in results if "level1" in item)
    level2_count = sum(1 for item in results if "level2" in item)
    level3_count = sum(1 for item in results if "level3" in item)

    feasibility_pass = sum(1 for item in results if item.get("level1", {}).get("feasibility", {}).get("pass"))
    absolute_feasibility_pass = sum(
        1
        for item in results
        if item.get("level1", {}).get("feasibility", {}).get(
            "absolute_pass",
            item.get("level1", {}).get("feasibility", {}).get("pass"),
        )
    )
    preservation_pass = sum(
        1
        for item in results
        if item.get("level1", {}).get("origin_logical_preservation", {}).get("pass")
    )
    preference_preservation_pass = sum(
        1
        for item in results
        if item.get("level1", {}).get("origin_preference_preservation", {}).get("pass")
    )
    preference_preservation_evaluable = 0
    preference_preservation_evaluable_pass = 0
    level1_invalid_or_missing_plan = 0
    level1_infeasible_detection = 0
    level1_pass = sum(1 for item in results if item.get("level1", {}).get("pass"))

    logical_evaluable = 0
    logical_pass = 0
    preference_evaluable = 0
    preference_pass = 0
    combined_evaluable = 0
    combined_pass = 0
    level1_logical_breakdown: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "supported": 0}
    )
    level1_preference_breakdown: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "supported": 0}
    )
    logical_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "supported": 0})
    preference_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "supported": 0})
    preference_canonical_breakdown: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "supported": 0}
    )
    preference_family_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "supported": 0})
    preference_score_before_sum: dict[str, float] = defaultdict(float)
    preference_score_before_count: dict[str, int] = defaultdict(int)
    preference_score_after_sum: dict[str, float] = defaultdict(float)
    preference_score_after_count: dict[str, int] = defaultdict(int)
    preference_delta_sum: dict[str, float] = defaultdict(float)
    preference_delta_count: dict[str, int] = defaultdict(int)
    level3_evaluable = 0
    scope_distribution: dict[str, int] = defaultdict(int)
    parameter_total = 0
    structural_total = 0
    compositional_total = 0
    retention_total = 0.0
    poi_distance_total = 0.0
    change_ratio_total = 0.0
    oracle_scope_records = 0
    oracle_scope_success = 0
    oracle_scope_local_success = 0
    oracle_scope_expanded = 0
    oracle_scope_frozen_pass = 0
    oracle_scope_depth_total = 0
    oracle_scope_added_total = 0
    oracle_scope_outside_change_total = 0
    oracle_scope_breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"records": 0, "success": 0, "local_success": 0})

    for item in results:
        level1 = item.get("level1", {})
        level2 = item.get("level2", {})
        level3 = item.get("level3", {})
        preservation = level1.get("origin_logical_preservation", {})
        preference_preservation = level1.get("origin_preference_preservation", {})
        logical = level2.get("edit_logical_success", {})
        preference = level2.get("edit_preference_success", {})
        run_metrics = item.get("run_context", {}).get("metrics", {})
        if isinstance(run_metrics, dict) and run_metrics.get("oracle_scope"):
            oracle_scope_records += 1
            success = bool(run_metrics.get("scope_status") in {"local_success", "expanded_success"})
            local_success = run_metrics.get("scope_status") == "local_success"
            expanded = bool(run_metrics.get("scope_expanded"))
            oracle_scope_success += int(success)
            oracle_scope_local_success += int(local_success)
            oracle_scope_expanded += int(expanded)
            oracle_scope_frozen_pass += int(bool(run_metrics.get("frozen_complement_pass")))
            oracle_scope_depth_total += int(run_metrics.get("expansion_steps", 0) or 0)
            oracle_scope_added_total += int(run_metrics.get("added_activity_count", 0) or 0)
            oracle_scope_outside_change_total += int(run_metrics.get("outside_final_scope_change_count", 0) or 0)
            day_key = "multi_day" if run_metrics.get("is_multi_day") else "single_day"
            cascade_value = run_metrics.get("cascade_required")
            cascade_key = "cascade_required" if cascade_value is True else "local" if cascade_value is False else "unlabeled_cascade"
            bucket = oracle_scope_breakdown[f"{day_key}:{cascade_key}"]
            bucket["records"] += 1
            bucket["success"] += int(success)
            bucket["local_success"] += int(local_success)

        feasibility_reason = str(level1.get("feasibility", {}).get("reason", ""))
        preservation_reason = str(preservation.get("reason", ""))
        if feasibility_reason in {"invalid_edited_plan", "missing_edited_plan"}:
            level1_invalid_or_missing_plan += 1
        if feasibility_reason == "infeasible_detection" or preservation_reason == "infeasible_detection":
            level1_infeasible_detection += 1

        supported_origin_preferences = preference_preservation.get("supported_preferences")
        if not isinstance(supported_origin_preferences, int):
            supported_origin_preferences = sum(
                1
                for entry in preference_preservation.get("per_preference_results", [])
                if entry.get("supported")
            )
        if supported_origin_preferences > 0:
            preference_preservation_evaluable += 1
            if preference_preservation.get("pass"):
                preference_preservation_evaluable_pass += 1

        for entry in preservation.get("per_constraint_results", []):
            key = str(entry.get("constraint_type", "unknown"))
            level1_logical_breakdown[key]["total"] += 1
            if entry.get("supported"):
                level1_logical_breakdown[key]["supported"] += 1
            if entry.get("supported") and entry.get("passed"):
                level1_logical_breakdown[key]["passed"] += 1
        for entry in preference_preservation.get("per_preference_results", []):
            key = f"{entry.get('family', 'unknown')}:{entry.get('canonical_facet', entry.get('facet', 'unknown'))}"
            level1_preference_breakdown[key]["total"] += 1
            if entry.get("supported"):
                level1_preference_breakdown[key]["supported"] += 1
            if entry.get("supported") and entry.get("passed"):
                level1_preference_breakdown[key]["passed"] += 1

        if logical.get("supported_constraints", 0) > 0:
            logical_evaluable += 1
            if logical.get("pass"):
                logical_pass += 1
        if preference.get("supported_preferences", 0) > 0:
            preference_evaluable += 1
            if preference.get("pass"):
                preference_pass += 1
        if logical.get("supported_constraints", 0) > 0 or preference.get("supported_preferences", 0) > 0:
            combined_evaluable += 1
            if level2.get("pass"):
                combined_pass += 1

        for entry in logical.get("per_constraint_results", []):
            key = str(entry.get("constraint_type", "unknown"))
            logical_breakdown[key]["total"] += 1
            if entry.get("supported"):
                logical_breakdown[key]["supported"] += 1
            if entry.get("supported") and entry.get("passed"):
                logical_breakdown[key]["passed"] += 1

        for entry in preference.get("per_preference_results", []):
            key = f"{entry.get('family', 'unknown')}:{entry.get('facet', 'unknown')}"
            family_key = str(entry.get("family", "unknown"))
            details = entry.get("details", {})
            if not isinstance(details, dict):
                details = {}
            canonical_facet = str(details.get("canonical_facet") or entry.get("facet", "unknown"))
            canonical_key = f"{entry.get('family', 'unknown')}:{canonical_facet}"
            preference_breakdown[key]["total"] += 1
            preference_canonical_breakdown[canonical_key]["total"] += 1
            preference_family_breakdown[family_key]["total"] += 1
            if entry.get("supported"):
                preference_breakdown[key]["supported"] += 1
                preference_canonical_breakdown[canonical_key]["supported"] += 1
                preference_family_breakdown[family_key]["supported"] += 1
            if entry.get("supported") and entry.get("passed"):
                preference_breakdown[key]["passed"] += 1
                preference_canonical_breakdown[canonical_key]["passed"] += 1
                preference_family_breakdown[family_key]["passed"] += 1
            if entry.get("supported") and isinstance(entry.get("score_before"), (int, float)):
                preference_score_before_sum[key] += float(entry["score_before"])
                preference_score_before_count[key] += 1
            if entry.get("supported") and isinstance(entry.get("score_after"), (int, float)):
                preference_score_after_sum[key] += float(entry["score_after"])
                preference_score_after_count[key] += 1
            if entry.get("supported") and isinstance(entry.get("delta"), (int, float)):
                preference_delta_sum[key] += float(entry["delta"])
                preference_delta_count[key] += 1

        if level3.get("eligible"):
            level3_evaluable += 1
            scope_distribution[str(level3.get("scope_name", "unknown"))] += 1
            parameter_total += int(level3.get("parameter_count", 0) or 0)
            structural_total += int(level3.get("structural_count", 0) or 0)
            compositional_total += int(level3.get("compositional_count", 0) or 0)
            retention_total += float(level3.get("content_retention_rate", 0.0) or 0.0)
            poi_distance_total += float(level3.get("poi_seq_edit_distance", 0.0) or 0.0)
            change_ratio_total += float(level3.get("activity_change_ratio", 0.0) or 0.0)

    level1_type_breakdown = _finalize_support_breakdown(level1_logical_breakdown)
    level1_preference_type_breakdown = _finalize_support_breakdown(level1_preference_breakdown)
    logical_type_breakdown = _finalize_support_breakdown(logical_breakdown)
    preference_type_breakdown = _finalize_support_breakdown(preference_breakdown)
    preference_canonical_type_breakdown = _finalize_support_breakdown(preference_canonical_breakdown)
    preference_family_support_breakdown = _finalize_support_breakdown(preference_family_breakdown)
    level3_scope_breakdown = _finalize_scope_distribution(scope_distribution, level3_evaluable)
    level3_cascade = aggregate_level3_cascade(results)
    avg_origin_score_by_facet = {
        key: preference_score_before_sum[key] / preference_score_before_count[key]
        for key in preference_score_before_count
        if preference_score_before_count[key]
    }
    avg_edited_score_by_facet = {
        key: preference_score_after_sum[key] / preference_score_after_count[key]
        for key in preference_score_after_count
        if preference_score_after_count[key]
    }
    avg_delta_by_facet = {
        key: preference_delta_sum[key] / preference_delta_count[key]
        for key in preference_delta_count
        if preference_delta_count[key]
    }
    unsupported_preference_count_by_facet = {
        key: metrics["unsupported"] for key, metrics in preference_type_breakdown.items()
    }

    report = {
        "summary": {
            "total_records": total,
            "level1_records": level1_count,
            "level2_records": level2_count,
            "level3_records": level3_count,
            "feasibility_pass_rate": _rate(feasibility_pass, level1_count),
            "relative_feasibility_pass_rate": _rate(feasibility_pass, level1_count),
            "absolute_feasibility_pass_rate": _rate(absolute_feasibility_pass, level1_count),
            "origin_logical_preservation_rate": _rate(preservation_pass, level1_count),
            "origin_preference_preservation_rate": _rate(preference_preservation_pass, level1_count),
            "origin_preference_preservation_evaluable_records": preference_preservation_evaluable,
            "origin_preference_preservation_evaluable_pass_rate": _rate(
                preference_preservation_evaluable_pass,
                preference_preservation_evaluable,
            ),
            "level1_invalid_or_missing_plan_records": level1_invalid_or_missing_plan,
            "level1_infeasible_detection_records": level1_infeasible_detection,
            "level1_pass_rate": _rate(level1_pass, level1_count),
            "edit_logical_success_rate": _rate(logical_pass, total),
            "edit_logical_evaluable_records": logical_evaluable,
            "edit_preference_success_rate": _rate(preference_pass, total),
            "edit_preference_evaluable_records": preference_evaluable,
            "combined_edit_success_rate": _rate(combined_pass, total),
            "combined_edit_evaluable_records": combined_evaluable,
            "level3_evaluable_records": level3_evaluable,
            "avg_parameter_count": parameter_total / level3_evaluable if level3_evaluable else 0.0,
            "avg_structural_count": structural_total / level3_evaluable if level3_evaluable else 0.0,
            "avg_compositional_count": compositional_total / level3_evaluable if level3_evaluable else 0.0,
            "avg_content_retention_rate": retention_total / level3_evaluable if level3_evaluable else 0.0,
            "avg_poi_seq_edit_distance": poi_distance_total / level3_evaluable if level3_evaluable else 0.0,
            "avg_activity_change_ratio": change_ratio_total / level3_evaluable if level3_evaluable else 0.0,
        },
        "per_level_summary": {
            "level1": {
                "records": level1_count,
                "feasibility": _finalize_check(level1_count, feasibility_pass),
                "absolute_feasibility": _finalize_check(
                    level1_count, absolute_feasibility_pass
                ),
                "origin_logical_preservation": _finalize_check(level1_count, preservation_pass),
                "origin_preference_preservation": _finalize_check(level1_count, preference_preservation_pass),
                "origin_preference_preservation_supported": _finalize_check(
                    level1_count,
                    preference_preservation_evaluable_pass,
                    evaluable=preference_preservation_evaluable,
                    rate_denominator=preference_preservation_evaluable,
                ),
                "overall": _finalize_check(level1_count, level1_pass),
            },
            "level2": {
                "records": level2_count,
                "edit_logical_success": _finalize_check(
                    level2_count,
                    logical_pass,
                    evaluable=logical_evaluable,
                    rate_denominator=total,
                ),
                "edit_preference_success": _finalize_check(
                    level2_count,
                    preference_pass,
                    evaluable=preference_evaluable,
                    rate_denominator=total,
                ),
                "overall": _finalize_check(
                    level2_count,
                    combined_pass,
                    evaluable=combined_evaluable,
                    rate_denominator=total,
                ),
            },
            "level3": {
                "records": level3_count,
                "eligible": _finalize_check(level3_count, level3_evaluable),
                "averages": {
                    "parameter_count": parameter_total / level3_evaluable if level3_evaluable else 0.0,
                    "structural_count": structural_total / level3_evaluable if level3_evaluable else 0.0,
                    "compositional_count": compositional_total / level3_evaluable if level3_evaluable else 0.0,
                    "content_retention_rate": retention_total / level3_evaluable if level3_evaluable else 0.0,
                    "poi_seq_edit_distance": poi_distance_total / level3_evaluable if level3_evaluable else 0.0,
                    "activity_change_ratio": change_ratio_total / level3_evaluable if level3_evaluable else 0.0,
                },
                "scope_distribution": level3_scope_breakdown,
                "cascade": level3_cascade,
            },
        },
        "per_type_breakdown": {
            "logical_constraints": logical_type_breakdown,
            "preferences": preference_type_breakdown,
            "preferences_canonical": preference_canonical_type_breakdown,
            "preferences_by_family": preference_family_support_breakdown,
        },
        "per_level_type_breakdown": {
            "level1": {
                "origin_logical_preservation": level1_type_breakdown,
                "origin_preferences": level1_preference_type_breakdown,
            },
            "level2": {
                "logical_constraints": logical_type_breakdown,
                "preferences": preference_type_breakdown,
                "preferences_canonical": preference_canonical_type_breakdown,
                "preferences_by_family": preference_family_support_breakdown,
            },
            "level3": {
                "scope_distribution": level3_scope_breakdown,
            },
        },
        "preference_support_rate_by_family": preference_family_support_breakdown,
        "preference_support_rate_by_facet": preference_type_breakdown,
        "preference_support_rate_by_canonical_facet": preference_canonical_type_breakdown,
        "avg_origin_score_by_facet": avg_origin_score_by_facet,
        "avg_edited_score_by_facet": avg_edited_score_by_facet,
        "avg_delta_by_facet": avg_delta_by_facet,
        "unsupported_preference_count_by_facet": unsupported_preference_count_by_facet,
        "level3": {
            "scope_distribution": {key: value["count"] for key, value in level3_scope_breakdown.items()},
            "cascade": level3_cascade,
        },
        "oracle_scope": {
            "diagnostic_only": True,
            "planner_mode": "oracle_patch_program",
            "records": oracle_scope_records,
            "local_success_rate": _rate(oracle_scope_local_success, oracle_scope_records),
            "eventual_success_rate": _rate(oracle_scope_success, oracle_scope_records),
            "expansion_rate": _rate(oracle_scope_expanded, oracle_scope_records),
            "frozen_complement_pass_rate": _rate(oracle_scope_frozen_pass, oracle_scope_records),
            "avg_expansion_depth": oracle_scope_depth_total / oracle_scope_records if oracle_scope_records else 0.0,
            "avg_added_activities": oracle_scope_added_total / oracle_scope_records if oracle_scope_records else 0.0,
            "avg_outside_final_scope_change_count": oracle_scope_outside_change_total / oracle_scope_records if oracle_scope_records else 0.0,
            "breakdown": {
                key: {**value, "eventual_success_rate": _rate(value["success"], value["records"]),
                      "local_success_rate": _rate(value["local_success"], value["records"])}
                for key, value in sorted(oracle_scope_breakdown.items())
            },
        },
        "results": results,
    }
    return attach_report_views(report)
