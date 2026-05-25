"""
Runtime helpers for Step 8 validation orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def build_validation_inputs(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the validate_all payload from a sample."""
    query_trace = sample.get("query_generation_trace", {})
    if not isinstance(query_trace, dict):
        query_trace = {}

    return {
        "origin_plan": sample.get("origin_plan", {}),
        "constraints": sample.get("constraints", {}),
        "canonical_constraint_ir": sample.get("canonical_constraint_ir"),
        "origin_logical_constraints": sample.get("origin_logical_constraints"),
        "origin_preference_tags": sample.get("origin_preference_tags"),
        "origin_preference_score_snapshot": sample.get("origin_preference_score_snapshot"),
        "origin_query_text": sample.get("origin_query_text"),
        "origin_query_structured": sample.get("origin_query_structured"),
        "origin_soft_profile": sample.get("meta", {}).get("origin_soft_profile"),
        "edit_target_constraints": sample.get("edit_target_constraints"),
        "effective_logical_constraints": sample.get("effective_logical_constraints"),
        "effective_constraint_merge_trace": sample.get("effective_constraint_merge_trace"),
        "edit_target_preference_tags": sample.get("edit_target_preference_tags"),
        "edit_target_preferences": sample.get("edit_target_preferences"),
        "edit_query": sample.get("edit_query", ""),
        "canonical_edit_query": sample.get("canonical_edit_query", ""),
        "constraint_type": query_trace.get("constraint_type"),
        "template_str": query_trace.get("template"),
        "query_generation_trace": query_trace,
        "mock_plan": sample.get("mock_edited_plan"),
        "conflict_labels": sample.get("conflict_labels"),
        "conflict_set": sample.get("conflict_set"),
        "primary_conflict": sample.get("primary_conflict"),
        "purity_score_rule": sample.get("purity_score_rule"),
        "match_type_rule": sample.get("match_type_rule"),
        "target_bucket": sample.get("meta", {}).get("target_bucket"),
        "strategy_plan": sample.get("strategy_plan"),
        "op_tags": sample.get("op_tags"),
    }


def collect_failed_checks(checks: Dict[str, Any]) -> List[str]:
    """Return failing check names from a validation result map."""
    failed_checks: List[str] = []
    for check_name, check_result in checks.items():
        if check_name == "all_pass" or not isinstance(check_result, dict):
            continue
        if not check_result.get("pass", False):
            failed_checks.append(check_name)
    return failed_checks


def collect_check_messages(check_result: Dict[str, Any]) -> List[str]:
    """Flatten the known error payload shapes for display."""
    if not isinstance(check_result, dict):
        return []

    messages: List[str] = []
    for key in (
        "errors",
        "violations",
        "inconsistencies",
        "uncovered_conflicts",
    ):
        value = check_result.get(key)
        if isinstance(value, list):
            messages.extend(str(item) for item in value)
    return messages


def apply_validation_outcome(
    sample: Dict[str, Any],
    checks: Dict[str, Any],
    *,
    strict: bool,
) -> Dict[str, Any]:
    """Persist validation result into sample metadata."""
    sample["checks"] = checks
    meta = sample.setdefault("meta", {})
    meta["last_step"] = "09_validate"
    meta["retry_count"] = 0

    failed_checks = collect_failed_checks(checks)
    if checks.get("all_pass", False):
        meta["status"] = "09_completed"
        meta.pop("failure_reason", None)
        return {
            "all_pass": True,
            "failed_checks": [],
            "status": meta["status"],
        }

    reason_prefix = "Validation failed" if strict else "Warnings"
    meta["status"] = "09_failed" if strict else "09_completed_with_warnings"
    meta["failure_reason"] = f"{reason_prefix}: {', '.join(failed_checks)}"
    return {
        "all_pass": False,
        "failed_checks": failed_checks,
        "status": meta["status"],
    }


def apply_validation_exception(sample: Dict[str, Any], exc: Exception) -> None:
    """Persist exception state into sample metadata."""
    meta = sample.setdefault("meta", {})
    meta["status"] = "09_failed"
    meta["failure_reason"] = f"Exception: {exc}"
    meta["last_step"] = "09_validate"
    meta["retry_count"] = meta.get("retry_count", 0) + 1


def compute_batch_check_stats(
    samples: List[Dict[str, Any]],
    primary_checks: Iterable[str],
    optional_checks: Iterable[str],
    reported_non_blocking_checks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Aggregate per-check pass counts for a validated batch."""
    primary = {check_name: 0 for check_name in primary_checks}
    optional = {
        check_name: {"passed": 0, "applicable": 0}
        for check_name in optional_checks
    }
    reported = {
        check_name: {"passed": 0, "applicable": 0}
        for check_name in (reported_non_blocking_checks or [])
    }

    for sample in samples:
        checks = sample.get("checks", {})
        for check_name in primary:
            if checks.get(check_name, {}).get("pass", False):
                primary[check_name] += 1
        for check_name in optional:
            result = checks.get(check_name, {})
            if not isinstance(result, dict) or result.get("skipped"):
                continue
            optional[check_name]["applicable"] += 1
            if result.get("pass", False):
                optional[check_name]["passed"] += 1
        for check_name in reported:
            result = checks.get(check_name, {})
            if not isinstance(result, dict) or result.get("skipped"):
                continue
            reported[check_name]["applicable"] += 1
            if result.get("pass", False):
                reported[check_name]["passed"] += 1

    all_pass_count = sum(
        1 for sample in samples if sample.get("checks", {}).get("all_pass", False)
    )
    return {
        "total": len(samples),
        "primary": primary,
        "optional": optional,
        "reported_non_blocking": reported,
        "all_pass_count": all_pass_count,
    }


def build_validation_metadata(
    samples: List[Dict[str, Any]],
    primary_checks: Iterable[str],
    optional_checks: Iterable[str],
    reported_non_blocking_checks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build the validation-related metadata payload for a batch."""
    stats = compute_batch_check_stats(
        samples,
        primary_checks,
        optional_checks,
        reported_non_blocking_checks,
    )
    total = stats["total"]

    validation_stats = {}
    for check_name, passed in stats["primary"].items():
        validation_stats[check_name] = {
            "passed": passed,
            "total": total,
            "pass_rate": passed / total if total > 0 else 0,
        }

    optional_validation_stats = {}
    for check_name, item in stats["optional"].items():
        applicable = item["applicable"]
        optional_validation_stats[check_name] = {
            "passed": item["passed"],
            "applicable": applicable,
            "pass_rate": item["passed"] / applicable if applicable > 0 else None,
        }

    reported_validation_stats = {}
    for check_name, item in stats["reported_non_blocking"].items():
        applicable = item["applicable"]
        reported_validation_stats[check_name] = {
            "passed": item["passed"],
            "applicable": applicable,
            "pass_rate": item["passed"] / applicable if applicable > 0 else None,
        }

    sample_statuses = [sample.get("meta", {}).get("status") for sample in samples]
    if any(status == "09_failed" for status in sample_statuses):
        batch_status = "09_failed"
    elif any(status == "09_completed_with_warnings" for status in sample_statuses):
        batch_status = "09_completed_with_warnings"
    else:
        batch_status = "09_completed"

    return {
        "validation": validation_stats,
        "optional_validation": optional_validation_stats,
        "reported_non_blocking_validation": reported_validation_stats,
        "validation_all_pass_count": stats["all_pass_count"],
        "validation_pass_rate": (
            stats["all_pass_count"] / total if total > 0 else 0
        ),
        "status": batch_status,
    }


def update_batch_metadata(
    batch_dir: str,
    samples: List[Dict[str, Any]],
    primary_checks: Iterable[str],
    optional_checks: Iterable[str],
    reported_non_blocking_checks: Optional[Iterable[str]] = None,
    *,
    timestamp: str,
) -> Path:
    """Update batch_metadata.json with validation summary."""
    metadata_file = Path(batch_dir) / "batch_metadata.json"

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        metadata = {
            "batch_id": Path(batch_dir).name,
            "sample_count": len(samples),
            "created_at": "",
        }

    metadata.update(
        build_validation_metadata(
            samples,
            primary_checks,
            optional_checks,
            reported_non_blocking_checks,
        )
    )
    metadata["completed_at"] = timestamp
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_file
