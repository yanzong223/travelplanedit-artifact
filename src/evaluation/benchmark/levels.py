"""Level runners for benchmark evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from data_clean.rules import evaluate_plan, summarize_violation_codes

from .compilers import compile_logical_constraints
from .edit_adapters import adapt_edit_record
from .level3 import evaluate_level3
from .models import (
    LogicalConstraintObject,
    PreferenceConstraintObject,
    constraints_to_dict,
    preferences_to_dict,
)
from .origin_adapters import adapt_origin_query
from .preference_scorers import evaluate_preference_target, score_preference_baseline
from .verifiers import verify_constraints
from utils.chinatravel_plan import validate_chinatravel_plan

SCORER_ID = "benchmark.preference_scorers"
SCORER_VERSION = "v2"


def _constraint_key(constraint: LogicalConstraintObject) -> tuple[Any, ...]:
    return (
        constraint.type,
        tuple(sorted((constraint.target or {}).items())),
        constraint.operator,
        tuple(constraint.value) if isinstance(constraint.value, list) else constraint.value,
    )


_VALUE_INSENSITIVE_TYPES = {
    "budget_total",
    "ticket_budget_total",
    "day_count",
    "people_count",
}

_TARGET_SCOPED_VALUE_INSENSITIVE_TYPES = {
    "activity_budget_limit",
    "activity_duration_limit",
    "semantic_type_requirement",
    "daily_poi_cap",
    "poi_time_window",
    "poi_day_binding",
    "transport_time_window",
    "ticket_price_cap_or_substitute",
    "required_attraction_name",
    "required_restaurant_name",
    "required_restaurant_type",
    "required_hotel_name",
    "required_hotel_feature",
    "required_attraction_type",
    "required_intercity_transport_type",
    "required_innercity_transport_type",
    "nearby_meal_requirement",
    "anchor_bundle_budget_limit",
}


def _normalized_target(constraint: LogicalConstraintObject) -> str:
    return json.dumps(constraint.target or {}, ensure_ascii=False, sort_keys=True)


def _is_targeted(origin: LogicalConstraintObject, edits: list[LogicalConstraintObject]) -> bool:
    origin_key = _constraint_key(origin)
    for edit in edits:
        if origin.type == edit.type and origin_key == _constraint_key(edit):
            return True
        if origin.type == edit.type and origin.type in _VALUE_INSENSITIVE_TYPES:
            return True
        if (
            origin.type == edit.type
            and origin.type in _TARGET_SCOPED_VALUE_INSENSITIVE_TYPES
            and _normalized_target(origin) == _normalized_target(edit)
        ):
            return True
    return False


def _normalize_anchor(anchor: Any) -> str:
    if anchor is None:
        return ""
    return str(anchor).strip().strip("'\"“”").casefold()


def _preference_match_key(preference: PreferenceConstraintObject) -> tuple[str, str, str]:
    canonical_facet = str(preference.params.get("canonical_facet") or preference.facet)
    return (
        str(preference.family),
        canonical_facet,
        _normalize_anchor(preference.anchor),
    )


def _is_targeted_preference(
    origin: PreferenceConstraintObject,
    edits: list[PreferenceConstraintObject],
) -> bool:
    origin_key = _preference_match_key(origin)
    return any(_preference_match_key(edit) == origin_key for edit in edits)


def _evaluate_origin_preference_preservation(
    origin_plan: dict[str, Any] | None,
    edited_plan: dict[str, Any],
    preserved_preferences: list[PreferenceConstraintObject],
) -> dict[str, Any]:
    per_results: list[dict[str, Any]] = []
    preserved_count = 0
    broken_count = 0
    supported_count = 0

    for preference in preserved_preferences:
        before_supported, score_before, before_details, before_reason = score_preference_baseline(
            origin_plan or {},
            preference,
        )
        after_supported, score_after, after_details, after_reason = score_preference_baseline(
            edited_plan,
            preference,
        )
        canonical_facet = str(preference.params.get("canonical_facet") or preference.facet)
        result: dict[str, Any] = {
            "preference_id": preference.id,
            "family": preference.family,
            "facet": preference.facet,
            "canonical_facet": canonical_facet,
            "anchor": preference.anchor,
            "supported_before": before_supported,
            "supported_after": after_supported,
            "supported": False,
            "score_before": score_before if before_supported else None,
            "score_after": score_after if after_supported else None,
            "delta": None,
            "passed": False,
            "reason": "",
            "details": {
                "before": before_details,
                "after": after_details,
            },
        }

        if not before_supported:
            result["reason"] = before_reason or "unsupported_before"
        elif not after_supported:
            result["reason"] = after_reason or "unsupported_after"
        else:
            supported_count += 1
            result["supported"] = True
            delta = (score_after or 0.0) - (score_before or 0.0)
            result["delta"] = delta
            passed = (score_after or 0.0) >= (score_before or 0.0)
            result["passed"] = passed
            result["reason"] = "ok" if passed else "score_degraded"
            if passed:
                preserved_count += 1
            else:
                broken_count += 1
        per_results.append(result)

    total = len(preserved_preferences)
    unsupported = total - supported_count
    passed = bool(supported_count) and broken_count == 0
    reason = "ok" if passed else "no_supported_origin_preferences" if supported_count == 0 else "origin_preference_degraded"
    return {
        "pass": passed,
        "reason": reason,
        "scorer_id": SCORER_ID,
        "scorer_version": SCORER_VERSION,
        "total_preferences": total,
        "supported_preferences": supported_count,
        "unsupported_preferences": unsupported,
        "preserved_count": preserved_count,
        "broken_count": broken_count,
        "per_preference_results": per_results,
    }


def evaluate_level1(
    origin_payload: dict[str, Any] | None,
    edited_plan: dict[str, Any] | None,
    edit_constraints: list[LogicalConstraintObject],
    edit_preferences: list[PreferenceConstraintObject] | None = None,
    *,
    origin_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(edited_plan, dict):
        return {
            "pass": False,
            "feasibility": {"pass": False, "reason": "missing_edited_plan"},
            "origin_logical_preservation": {"pass": False, "reason": "missing_edited_plan"},
            "origin_preference_preservation": {"pass": False, "reason": "missing_edited_plan"},
            "diagnostics": {},
        }

    feasibility_eval = evaluate_plan(edited_plan)
    feasibility = {
        "pass": feasibility_eval.hygiene_pass and feasibility_eval.quality_pass,
        "hygiene_pass": feasibility_eval.hygiene_pass,
        "quality_pass": feasibility_eval.quality_pass,
        "hygiene_violations": [item.to_dict() for item in feasibility_eval.hygiene_violations],
        "quality_violations": [item.to_dict() for item in feasibility_eval.quality_violations],
    }

    origin_feasibility = None
    if isinstance(origin_plan, dict):
        origin_feasibility_eval = evaluate_plan(origin_plan)
        origin_feasibility = {
            "pass": origin_feasibility_eval.hygiene_pass and origin_feasibility_eval.quality_pass,
            "hygiene_pass": origin_feasibility_eval.hygiene_pass,
            "quality_pass": origin_feasibility_eval.quality_pass,
            "hygiene_violations": [item.to_dict() for item in origin_feasibility_eval.hygiene_violations],
            "quality_violations": [item.to_dict() for item in origin_feasibility_eval.quality_violations],
        }

    origin_bundle = adapt_origin_query(origin_payload)
    preserved_constraints = [
        constraint
        for constraint in origin_bundle["logical_constraints"]
        if not _is_targeted(constraint, edit_constraints)
    ]
    edit_preferences = edit_preferences or []
    preserved_preferences = [
        preference
        for preference in origin_bundle["preference_constraints"]
        if not _is_targeted_preference(preference, edit_preferences)
    ]
    verification_results = verify_constraints(edited_plan, preserved_constraints)
    supported_results = [item for item in verification_results if item.supported]
    passed_results = [item for item in supported_results if item.passed]
    broken_results = [item for item in supported_results if not item.passed]
    preservation = {
        "pass": all(item.passed for item in supported_results) if supported_results else True,
        "total_constraints": len(preserved_constraints),
        "supported_constraints": len(supported_results),
        "unsupported_constraints": len(verification_results) - len(supported_results),
        "preserved_count": len(passed_results),
        "broken_count": len(broken_results),
        "per_constraint_results": [item.to_dict() for item in verification_results],
    }
    preference_preservation = _evaluate_origin_preference_preservation(
        origin_plan,
        edited_plan,
        preserved_preferences,
    )

    diagnostics = {
        "origin_preference_snapshot": preferences_to_dict(origin_bundle["preference_constraints"]),
        "origin_logical_snapshot": constraints_to_dict(origin_bundle["logical_constraints"]),
        "feasibility_violation_counts": {
            "hygiene": summarize_violation_codes(feasibility_eval.hygiene_violations),
            "quality": summarize_violation_codes(feasibility_eval.quality_violations),
        },
    }
    return {
        "pass": feasibility["pass"] and preservation["pass"] and preference_preservation["pass"],
        "feasibility": feasibility,
        "origin_logical_preservation": preservation,
        "origin_preference_preservation": preference_preservation,
        "origin_feasibility": origin_feasibility,
        "diagnostics": diagnostics,
    }


def evaluate_level2(
    record: dict[str, Any],
    origin_plan: dict[str, Any] | None,
    edited_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(edited_plan, dict):
        return {
            "pass": False,
            "edit_logical_success": {"pass": False, "reason": "missing_edited_plan"},
            "edit_preference_success": {"pass": False, "reason": "missing_edited_plan"},
        }

    try:
        edit_bundle = adapt_edit_record(record)
    except ValueError as exc:
        return {
            "pass": False,
            "edit_logical_success": {"pass": False, "reason": "missing_ir", "errors": [str(exc)]},
            "edit_preference_success": {"pass": False, "reason": "missing_ir", "errors": [str(exc)]},
        }
    logical_results = verify_constraints(edited_plan, edit_bundle["logical_constraints"])
    logical_supported = [item for item in logical_results if item.supported]
    total_logical = len(edit_bundle["logical_constraints"])
    unsupported_logical = len(logical_results) - len(logical_supported)
    logical = {
        "pass": bool(total_logical) and unsupported_logical == 0 and all(item.passed for item in logical_results),
        "total_constraints": total_logical,
        "supported_constraints": len(logical_supported),
        "unsupported_constraints": unsupported_logical,
        "successful_constraints": sum(1 for item in logical_supported if item.passed),
        "compiled_hard_logic_py": compile_logical_constraints(edit_bundle["logical_constraints"]),
        "per_constraint_results": [item.to_dict() for item in logical_results],
    }

    origin_plan = origin_plan or {}
    pref_results = [
        evaluate_preference_target(origin_plan, edited_plan, preference)
        for preference in edit_bundle["preference_constraints"]
    ]
    pref_supported = [item for item in pref_results if item.supported]
    preference = {
        "pass": all(item.passed for item in pref_supported) if pref_supported else False,
        "total_preferences": len(edit_bundle["preference_constraints"]),
        "supported_preferences": len(pref_supported),
        "unsupported_preferences": len(pref_results) - len(pref_supported),
        "successful_preferences": sum(1 for item in pref_supported if item.passed),
        "per_preference_results": [item.to_dict() for item in pref_results],
    }

    combined_supported = bool((total_logical > 0) or pref_supported)
    combined_pass = (
        (logical["pass"] if total_logical > 0 else True)
        and (preference["pass"] if pref_supported else True)
        and combined_supported
    )
    return {
        "pass": combined_pass,
        "edit_logical_success": logical,
        "edit_preference_success": preference,
    }


def evaluate_record(
    record: dict[str, Any],
    *,
    origin_query: dict[str, Any] | None = None,
    level: str = "all",
) -> dict[str, Any]:
    edited_plan = record.get("edited_plan")
    origin_plan = record.get("original_plan") or record.get("origin_plan")
    origin_query = origin_query or record.get("origin_query") or record.get("origin_query_structured")
    edit_truth_error: str | None = None
    try:
        edit_bundle = adapt_edit_record(record)
    except ValueError as exc:
        edit_bundle = {"logical_constraints": [], "preference_constraints": []}
        edit_truth_error = str(exc)
    origin_payload = {
        "origin_logical_constraints": record.get("origin_logical_constraints"),
        "origin_preference_tags": record.get("origin_preference_tags"),
        "origin_query_structured": origin_query,
        "origin_query_text": record.get("origin_query_text", ""),
        "meta": record.get("meta", {}),
    }
    level1_result = None
    level2_result = None

    result = {
        "record_id": record.get("episode_id") or record.get("sample_id") or record.get("plan_id") or "unknown_record",
        "plan_id": record.get("plan_id"),
    }
    run_errors = record.get("errors")
    if not isinstance(run_errors, list):
        run_errors = []
    metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
    result["run_context"] = {
        "success": record.get("success"),
        "status": record.get("status"),
        "error": record.get("error"),
        "errors": run_errors,
        "metrics": metrics,
        "terminal_error_categories": metrics.get("terminal_error_categories", []),
    }
    result["conflict_labels"] = record.get("conflict_labels") or record.get("primary_conflict")
    if edit_truth_error and level == "level2":
        result["level2"] = {
            "pass": False,
            "edit_logical_success": {"pass": False, "reason": "missing_ir", "errors": [edit_truth_error]},
            "edit_preference_success": {"pass": False, "reason": "missing_ir", "errors": [edit_truth_error]},
        }
        result.setdefault("diagnostics", {})["edit_truth_error"] = edit_truth_error
        return result
    plan_issues = validate_chinatravel_plan(edited_plan)
    infeasible_detection = metrics.get("infeasible_detection")
    if plan_issues and isinstance(infeasible_detection, dict):
        infeasible_reason = str(infeasible_detection.get("reason") or "infeasible_detection")
        result["level1"] = {
            "pass": False,
            "feasibility": {
                "pass": False,
                "reason": "infeasible_detection",
                "infeasible_detection": infeasible_detection,
                "hygiene_pass": False,
                "quality_pass": False,
                "hygiene_violations": [],
                "quality_violations": [],
            },
            "origin_logical_preservation": {
                "pass": False,
                "reason": "infeasible_detection",
                "infeasible_detection": infeasible_detection,
                "total_constraints": 0,
                "supported_constraints": 0,
                "unsupported_constraints": 0,
                "preserved_count": 0,
                "broken_count": 0,
                "per_constraint_results": [],
            },
            "origin_preference_preservation": {
                "pass": False,
                "reason": "infeasible_detection",
                "infeasible_detection": infeasible_detection,
                "scorer_id": SCORER_ID,
                "scorer_version": SCORER_VERSION,
                "total_preferences": 0,
                "supported_preferences": 0,
                "unsupported_preferences": 0,
                "preserved_count": 0,
                "broken_count": 0,
                "per_preference_results": [],
            },
            "diagnostics": {
                "infeasible_detection": infeasible_detection,
                "schema_issues": plan_issues,
            },
        }
        result["level2"] = {
            "pass": False,
            "reason": "infeasible_detection",
            "infeasible_detection": infeasible_detection,
            "edit_logical_success": {
                "pass": False,
                "reason": "infeasible_detection",
                "supported_constraints": 0,
                "total_constraints": len(edit_bundle["logical_constraints"]),
                "unsupported_constraints": 0,
                "successful_constraints": 0,
                "compiled_hard_logic_py": [],
                "per_constraint_results": [],
            },
            "edit_preference_success": {
                "pass": False,
                "reason": "infeasible_detection",
                "supported_preferences": 0,
                "total_preferences": len(edit_bundle["preference_constraints"]),
                "unsupported_preferences": 0,
                "successful_preferences": 0,
                "per_preference_results": [],
            },
        }
        if level in {"3", "all"}:
            result["level3"] = {
                "eligible": False,
                "reason": f"infeasible_detection: {infeasible_reason}",
                "scope_level": 0,
                "scope_name": "parameter",
                "parameter_count": 0,
                "structural_count": 0,
                "compositional_count": 0,
                "atomic_counts": {
                    "change_time": 0,
                    "insert": 0,
                    "delete": 0,
                    "replace": 0,
                    "reorder": 0,
                },
                "edit_cost_tuple": [0, 0, 0],
                "content_retention_rate": 0.0,
                "poi_seq_edit_distance": 0.0,
                "activity_change_ratio": 0.0,
                "matched_pairs": [],
                "unmatched_origin": [],
                "unmatched_edited": [],
            }
        return result
    if plan_issues:
        invalid_reason = "; ".join(plan_issues)
        result["level1"] = {
            "pass": False,
            "feasibility": {
                "pass": False,
                "reason": "invalid_edited_plan",
                "schema_issues": plan_issues,
                "hygiene_pass": False,
                "quality_pass": False,
                "hygiene_violations": [],
                "quality_violations": [],
            },
            "origin_logical_preservation": {
                "pass": False,
                "reason": "invalid_edited_plan",
                "schema_issues": plan_issues,
                "total_constraints": 0,
                "supported_constraints": 0,
                "unsupported_constraints": 0,
                "preserved_count": 0,
                "broken_count": 0,
                "per_constraint_results": [],
            },
            "origin_preference_preservation": {
                "pass": False,
                "reason": "invalid_edited_plan",
                "schema_issues": plan_issues,
                "scorer_id": SCORER_ID,
                "scorer_version": SCORER_VERSION,
                "total_preferences": 0,
                "supported_preferences": 0,
                "unsupported_preferences": 0,
                "preserved_count": 0,
                "broken_count": 0,
                "per_preference_results": [],
            },
            "diagnostics": {"schema_issues": plan_issues},
        }
        result["level2"] = {
            "pass": False,
            "reason": "invalid_edited_plan",
            "schema_issues": plan_issues,
            "edit_logical_success": {
                "pass": False,
                "reason": "invalid_edited_plan",
                "supported_constraints": 0,
                "total_constraints": len(edit_bundle["logical_constraints"]),
                "unsupported_constraints": 0,
                "successful_constraints": 0,
                "compiled_hard_logic_py": [],
                "per_constraint_results": [],
            },
            "edit_preference_success": {
                "pass": False,
                "reason": "invalid_edited_plan",
                "supported_preferences": 0,
                "total_preferences": len(edit_bundle["preference_constraints"]),
                "unsupported_preferences": 0,
                "successful_preferences": 0,
                "per_preference_results": [],
            },
        }
        if level in {"3", "all"}:
            result["level3"] = {
                "eligible": False,
                "reason": f"invalid_edited_plan: {invalid_reason}",
                "scope_level": 0,
                "scope_name": "parameter",
                "parameter_count": 0,
                "structural_count": 0,
                "compositional_count": 0,
                "atomic_counts": {
                    "change_time": 0,
                    "insert": 0,
                    "delete": 0,
                    "replace": 0,
                    "reorder": 0,
                },
                "edit_cost_tuple": [0, 0, 0],
                "content_retention_rate": 0.0,
                "poi_seq_edit_distance": 0.0,
                "activity_change_ratio": 0.0,
                "matched_pairs": [],
                "unmatched_origin": [],
                "unmatched_edited": [],
            }
        return result

    if level in {"1", "3", "all"}:
        level1_result = evaluate_level1(
            origin_payload,
            edited_plan,
            edit_bundle["logical_constraints"],
            edit_bundle["preference_constraints"],
            origin_plan=origin_plan,
        )
    if level in {"2", "3", "all"}:
        level2_result = evaluate_level2(record, origin_plan, edited_plan)
    if level in {"1", "all"} and level1_result is not None:
        result["level1"] = level1_result
    if level in {"2", "all"} and level2_result is not None:
        result["level2"] = level2_result
    if level in {"3", "all"}:
        result["level3"] = evaluate_level3(origin_plan, edited_plan, level1_result, level2_result)
    return result


@dataclass(slots=True)
class BenchmarkEvaluator:
    level: str = "all"

    def evaluate(self, record: dict[str, Any], *, origin_query: dict[str, Any] | None = None) -> dict[str, Any]:
        return evaluate_record(record, origin_query=origin_query, level=self.level)
