"""
Builders for benchmark export views.

This module creates two stable export surfaces from an internal batch sample:
- public benchmark view: task body + one frozen benchmark label
- internal audit view: task body + same label + audit/debug traces
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


BenchmarkLabel = Dict[str, str]


def _normalize_label_tuple(label: Any) -> Optional[Tuple[str, str, str]]:
    if not isinstance(label, (list, tuple)) or len(label) != 3:
        return None

    parts: List[str] = []
    for part in label:
        if not isinstance(part, str):
            return None
        normalized = part.strip()
        if not normalized:
            return None
        parts.append(normalized)

    return (parts[0], parts[1], parts[2])


def build_benchmark_label(primary_conflict: Any) -> BenchmarkLabel:
    """
    Build the unique benchmark label from a frozen primary_conflict field.

    Raises:
        ValueError: if primary_conflict is missing or malformed.
    """
    normalized = _normalize_label_tuple(primary_conflict)
    if normalized is None:
        raise ValueError("primary_conflict missing or invalid")

    scope, dimension, violation = normalized
    return {
        "schema_version": "v1",
        "source": "resolver_primary_conflict",
        "scope": scope,
        "dimension": dimension,
        "violation": violation,
        "key": f"{scope}_{dimension}_{violation}",
    }


def build_task_view(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "origin_plan": deepcopy(sample.get("origin_plan", {})),
        "origin_logical_constraints": deepcopy(sample.get("origin_logical_constraints", [])),
        "origin_preference_tags": deepcopy(sample.get("origin_preference_tags", [])),
        "origin_preference_constraints": deepcopy(sample.get("origin_preference_constraints", [])),
        "origin_preference_score_snapshot": deepcopy(sample.get("origin_preference_score_snapshot", [])),
        "origin_query_text": sample.get("origin_query_text"),
        "origin_query_structured": deepcopy(sample.get("origin_query_structured")),
        "edit_query": sample.get("edit_query", ""),
        "edit_target_constraints": deepcopy(sample.get("edit_target_constraints", [])),
        "effective_logical_constraints": deepcopy(sample.get("effective_logical_constraints", [])),
        "effective_constraint_merge_trace": deepcopy(sample.get("effective_constraint_merge_trace", {})),
        "edit_target_preference_tags": deepcopy(sample.get("edit_target_preference_tags", [])),
        "edit_target_preferences": deepcopy(sample.get("edit_target_preferences", [])),
        "constraints": deepcopy(sample.get("constraints", {})),
    }


def build_public_query_factual_validity(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    checks = sample.get("checks")
    if not isinstance(checks, dict):
        return None

    factual = checks.get("query_factual_validity")
    if not isinstance(factual, dict):
        return None

    public_payload: Dict[str, Any] = {}
    for key in [
        "label",
        "reason_code",
        "subtype",
        "invalidity_category",
        "plan_relation",
        "evidence_used",
    ]:
        if key in factual and factual.get(key) is not None:
            public_payload[key] = deepcopy(factual.get(key))

    return public_payload or None


def build_public_benchmark_view(sample: Dict[str, Any]) -> Dict[str, Any]:
    public_view = {
        "sample_id": sample.get("sample_id", ""),
        "task": build_task_view(sample),
        "benchmark_label": build_benchmark_label(sample.get("primary_conflict")),
    }
    factual_validity = build_public_query_factual_validity(sample)
    if factual_validity is not None:
        public_view["query_factual_validity"] = factual_validity
    return public_view


def build_internal_audit_view(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}

    return {
        "sample_id": sample.get("sample_id", ""),
        "task": build_task_view(sample),
        "benchmark_label": build_benchmark_label(sample.get("primary_conflict")),
        "internal_audit": {
            "sampling": {
                "batch_id": sample.get("batch_id"),
                "target_bucket": deepcopy(meta.get("target_bucket")),
                "target_city": meta.get("target_city"),
                "start_city": meta.get("start_city"),
                "origin_plan_file": meta.get("origin_plan_file"),
                "created_at": meta.get("created_at"),
                "status": meta.get("status"),
            },
            "query_generation": {
                "query_generation_trace": deepcopy(sample.get("query_generation_trace")),
                "retry_count": meta.get("retry_count"),
                "retry_count_step2": meta.get("retry_count_step2"),
                "last_failure_reason": meta.get("last_failure_reason"),
                "failure_reason": meta.get("failure_reason"),
            },
            "conflict_analysis": {
                "target_bucket": deepcopy(meta.get("target_bucket")),
                "conflict_facts": deepcopy(sample.get("conflict_facts")),
                "conflict_labels_llm": deepcopy(sample.get("conflict_labels")),
                "conflict_set_rule": deepcopy(sample.get("conflict_set")),
                "primary_conflict": deepcopy(sample.get("primary_conflict")),
                "secondary_conflicts": deepcopy(sample.get("secondary_conflicts")),
                "match_type_rule": sample.get("match_type_rule"),
                "purity_score_rule": sample.get("purity_score_rule"),
                "resolver_trace": deepcopy(sample.get("resolver_trace")),
                "target_confirmed": meta.get("target_confirmed"),
                "target_confidence": meta.get("target_confidence"),
                "target_explanation": meta.get("target_explanation"),
            },
            "strategy": {
                "strategy_target": deepcopy(sample.get("strategy_target")),
                "strategy_plan": deepcopy(sample.get("strategy_plan")),
                "all_conflicts_covered": meta.get("all_conflicts_covered"),
                "strategy_plan_version": meta.get("strategy_plan_version"),
            },
            "execution": {
                "mock_edited_plan": deepcopy(sample.get("mock_edited_plan")),
                "op_tags": deepcopy(sample.get("op_tags")),
                "diff_summary": deepcopy(sample.get("diff_summary")),
            },
            "validation": {
                "checks": deepcopy(sample.get("checks")),
                "failure_reason": meta.get("failure_reason"),
                "flagged_for_manual_review": meta.get("flagged_for_manual_review"),
                "status": meta.get("status"),
            },
        },
    }


def evaluate_export_eligibility(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Decide whether a sample is eligible for benchmark export.

    The decision is based only on saved frozen label stability, never by recomputing
    resolver outputs.
    """
    reasons: List[str] = []

    try:
        build_benchmark_label(sample.get("primary_conflict"))
    except ValueError as exc:
        reasons.append(str(exc))

    if not isinstance(sample.get("origin_logical_constraints"), list) or not sample.get("origin_logical_constraints"):
        reasons.append("origin_logical_constraints missing")
    if not isinstance(sample.get("edit_target_constraints"), list) or not sample.get("edit_target_constraints"):
        reasons.append("edit_target_constraints missing")
    if not isinstance(sample.get("effective_logical_constraints"), list) or not sample.get("effective_logical_constraints"):
        reasons.append("effective_logical_constraints missing")

    checks = sample.get("checks")
    if not isinstance(checks, dict):
        reasons.append("checks missing; export requires validation consistency checks")
        return (False, reasons)

    primary_exists = checks.get("primary_conflict_exists")
    if not isinstance(primary_exists, dict):
        reasons.append("checks.primary_conflict_exists missing")
    elif primary_exists.get("pass") is False:
        reasons.append("checks.primary_conflict_exists failed")

    resolver_determinism = checks.get("resolver_determinism")
    if not isinstance(resolver_determinism, dict):
        reasons.append("checks.resolver_determinism missing")
    elif resolver_determinism.get("pass") is False:
        reasons.append("checks.resolver_determinism failed")

    return (len(reasons) == 0, reasons)
