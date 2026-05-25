"""Utilities for detecting explicit infeasible edit conclusions."""

from __future__ import annotations

import json
import re
from typing import Any, Dict


INFEASIBLE_STATUS = "infeasible"


def parse_infeasible_response(response: Any) -> Dict[str, Any] | None:
    """Return a normalized infeasible payload when response declares infeasibility."""

    payload = _coerce_json_object(response)
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status", "")).strip().lower() != INFEASIBLE_STATUS:
        return None

    return {
        "status": INFEASIBLE_STATUS,
        "reason": _stringify(payload.get("reason")),
        "evidence": _string_list(payload.get("evidence")),
        "relaxation_suggestions": _string_list(payload.get("relaxation_suggestions")),
    }


def infeasible_result(
    *,
    detection: Dict[str, Any],
    conversation_log: list[Dict[str, Any]],
    metrics: Dict[str, Any],
    framework_type: str,
) -> Dict[str, Any]:
    """Build the legacy framework result shape for a detected infeasible edit."""

    return {
        "success": True,
        "edited_plan": None,
        "conversation_log": conversation_log,
        "metrics": {
            **metrics,
            "infeasible_detection": detection,
        },
        "framework_type": framework_type,
        "errors": [],
    }


def _coerce_json_object(response: Any) -> Dict[str, Any] | None:
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str) and content.strip():
            parsed = _parse_json_object_text(content)
            if parsed is not None:
                return parsed
        return response
    if isinstance(response, str):
        return _parse_json_object_text(response)
    return None


def _parse_json_object_text(text: str) -> Dict[str, Any] | None:
    content = text.strip()
    if not content:
        return None

    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content).strip()

    if content.startswith("{"):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    if start == -1:
        return None

    end = _matching_object_end(content, start)
    if end == -1:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _matching_object_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(text)):
        char = text[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(item) for item in value if _stringify(item)]
    text = _stringify(value)
    return [text] if text else []
