"""Runtime error-handling metadata for standalone edit frameworks."""

from __future__ import annotations

from typing import Any, Iterable


def framework_error_handling_metadata(
    framework: str,
    *,
    max_steps: int | None = None,
    max_tool_calls: int | None = None,
    max_reflections: int | None = None,
) -> dict[str, Any]:
    """Return explicit metadata for framework-level correction behavior."""

    has_tool_observe_loop = framework in {"react", "reflexion", "pter"}
    has_post_execution_correction_loop = framework == "reflexion"
    correction_budget: dict[str, Any] = {
        "max_steps": max_steps,
        "max_tool_calls": max_tool_calls,
        "max_reflections": max_reflections if framework == "reflexion" else 0,
    }
    return {
        "has_tool_observe_loop": has_tool_observe_loop,
        "has_post_execution_correction_loop": has_post_execution_correction_loop,
        "correction_budget": correction_budget,
    }


def classify_terminal_error_categories(
    errors: Iterable[Any] | None,
    *,
    metrics: dict[str, Any] | None = None,
    special_backend: str | None = None,
) -> list[str]:
    """Classify terminal runtime errors for run-summary diagnostics."""

    metrics = metrics or {}
    messages = [str(item) for item in (errors or []) if item is not None]
    lowered = "\n".join(messages).lower()
    categories: list[str] = []

    def add(category: str) -> None:
        if category not in categories:
            categories.append(category)

    if metrics.get("runner_exception_type"):
        add("runner_exception")
    if metrics.get("schema_validation_failures"):
        add("plan_schema_error")
    if metrics.get("parse_failures"):
        add("plan_parse_error")
    if metrics.get("rejected_fact_request_count"):
        add("fact_request_rejected")
    if "candidate_plan_guard_failed" in lowered:
        add("candidate_guard_reject")
    if "draft parse failed" in lowered or "could not parse json" in lowered:
        add("plan_parse_error")
    if "canonical chinatravel plan" in lowered or "schema validation failed" in lowered:
        add("plan_schema_error")
    if "no json array" in lowered or "could not parse json array" in lowered or "empty response content" in lowered:
        add("ops_parse_error")
    if "missing 'op'" in lowered or "missing 'path'" in lowered or "unsupported patch operation" in lowered:
        add("ops_schema_error")
    if "patch" in lowered or "path" in lowered or "out of range" in lowered or "invalid patch" in lowered:
        add("patch_execution_error")
    if "failed to extract json" in lowered or "legacy" in lowered or "edited_plan" in lowered:
        add("invalid_edited_plan")
    if "tool" in lowered:
        add("tool_error")
    if messages and not categories:
        add("unknown_runtime_failure")
    return categories
