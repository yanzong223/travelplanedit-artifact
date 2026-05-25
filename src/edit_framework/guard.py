"""Framework-level guard helpers.

The public artifact uses only the db_read_typed read-tool profile, so candidate
guards are disabled.
"""

from __future__ import annotations

from typing import Any, Dict


PAINPOINT_GUARD_TOOLS = {"diagnose_edit_requirements", "guard_candidate_plan"}


def painpoint_guard_enabled(tool_adapter: Any, guard_retries: int = 0) -> bool:
    enabled_tools = set(tool_adapter.tool_flags().get("enabled_semantic_tools") or [])
    return bool(PAINPOINT_GUARD_TOOLS & enabled_tools) or guard_retries > 0


def apply_candidate_guard(
    *,
    runtime: Any,
    tool_adapter: Any,
    framework_result: Dict[str, Any],
    original_plan: Dict[str, Any],
    edit_query: str,
    guard_retries: int = 0,
) -> Dict[str, Any]:
    return framework_result


def build_guard_retry_edit_query(edit_query: str, guard_report: Dict[str, Any]) -> str:
    hints = guard_report.get("compact_repair_hints") or []
    violations = []
    for key in (
        "must_fix_violations",
        "preservation_violations",
        "feasibility_violations",
        "fact_grounding_risks",
    ):
        for item in list(guard_report.get(key) or [])[:3]:
            if isinstance(item, dict):
                violations.append(item.get("message") or item.get("type") or item.get("code"))
            else:
                violations.append(str(item))
    feedback = {
        "guard_decision": guard_report.get("decision"),
        "must_repair": [str(item) for item in violations if item][:6],
        "repair_hints": [str(item) for item in hints][:5],
    }
    return (
        edit_query
        + "\n\n# Guard feedback for retry\n"
        + "上一版候选计划未通过提交前门禁。下一版必须修复以下问题，并仍然保持最小必要修改：\n"
        + str(feedback)
    )
