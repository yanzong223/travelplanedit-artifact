"""Structured tool-error helpers for semantic runtime tools."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from pydantic import ValidationError


def build_tool_error(
    *,
    tool_name: str,
    tool_args: Dict[str, Any] | None,
    error_code: str,
    message: str,
    missing_fields: Iterable[str] | None = None,
    invalid_fields: Iterable[str] | None = None,
    expected_shape: Any | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "tool_name": tool_name,
        "tool_args": dict(tool_args or {}),
    }
    missing = _dedupe_strings(missing_fields or [])
    invalid = _dedupe_strings(invalid_fields or [])
    if missing:
        payload["missing_fields"] = missing
    if invalid:
        payload["invalid_fields"] = invalid
    if expected_shape is not None:
        payload["expected_shape"] = expected_shape
    if details:
        payload["details"] = details
    return payload


def validation_error_to_tool_error(
    *,
    tool_name: str,
    tool_args: Dict[str, Any] | None,
    exc: ValidationError,
) -> Dict[str, Any]:
    missing_fields, invalid_fields = _extract_validation_fields(exc)
    missing_fields = _contextualize_validation_fields(tool_name, tool_args or {}, missing_fields)
    invalid_fields = _contextualize_validation_fields(tool_name, tool_args or {}, invalid_fields)
    error_code = _classify_validation_error(tool_name, missing_fields, invalid_fields)
    return build_tool_error(
        tool_name=tool_name,
        tool_args=tool_args,
        error_code=error_code,
        message=_validation_error_message(tool_name, error_code, missing_fields, invalid_fields),
        missing_fields=missing_fields,
        invalid_fields=invalid_fields,
        expected_shape=tool_expected_shape(tool_name),
        details={"validation_errors": exc.errors()},
    )


def key_error_to_tool_error(
    *,
    tool_name: str,
    tool_args: Dict[str, Any] | None,
    exc: KeyError,
) -> Dict[str, Any]:
    missing_key = str(exc).strip("'")
    missing_fields = [missing_key] if missing_key else []
    error_code = _classify_key_error(tool_name, missing_key)
    return build_tool_error(
        tool_name=tool_name,
        tool_args=tool_args,
        error_code=error_code,
        message=_key_error_message(tool_name, error_code, missing_key),
        missing_fields=missing_fields,
        expected_shape=tool_expected_shape(tool_name),
    )


def value_error_to_tool_error(
    *,
    tool_name: str,
    tool_args: Dict[str, Any] | None,
    exc: ValueError,
) -> Dict[str, Any]:
    error_code = "invalid_patch_args" if tool_name == "execute_plan_patch" else "invalid_tool_arguments_schema"
    return build_tool_error(
        tool_name=tool_name,
        tool_args=tool_args,
        error_code=error_code,
        message=str(exc),
        expected_shape=tool_expected_shape(tool_name),
    )


def unexpected_tool_error(
    *,
    tool_name: str,
    tool_args: Dict[str, Any] | None,
    exc: Exception,
) -> Dict[str, Any]:
    return build_tool_error(
        tool_name=tool_name,
        tool_args=tool_args,
        error_code="tool_execution_failed",
        message=str(exc),
    )


def tool_expected_shape(tool_name: str) -> Any | None:
    if tool_name == "retrieve_facts":
        return {
            "requests": [
                {
                    "request_id": "r1",
                    "intent": "lookup_entity",
                    "entity_type": "attraction",
                    "city": "上海",
                    "name": "外滩",
                }
            ]
        }
    if tool_name == "execute_plan_patch":
        return {
            "ops": [
                {
                    "op": "resize_activity",
                    "args": {
                        "day": 1,
                        "activity_id": "day1_activity_0",
                        "duration_minutes": 60,
                    },
                }
            ]
        }
    if tool_name in {"analyze_conflicts", "check_runtime_state"}:
        return {
            "active_constraints": [
                {
                    "id": "c_0",
                    "predicate": "must_schedule_entity_in_time_window",
                    "target": {"scope": "poi"},
                    "params": {"entity": "外滩", "time_window": "evening"},
                }
            ],
            "intent_anchors": [
                {
                    "id": "ia_0",
                    "type": "poi",
                    "semantic_key": "外滩",
                    "rule": "must_preserve",
                }
            ],
        }
    if tool_name == "construct_constraints":
        return {
            "edit_query": "把外滩安排到晚上",
            "origin_query_text": "去上海玩一天",
        }
    return None


def _extract_validation_fields(exc: ValidationError) -> tuple[List[str], List[str]]:
    missing_fields: List[str] = []
    invalid_fields: List[str] = []
    for entry in exc.errors():
        location = _format_loc(entry.get("loc", ()))
        if not location:
            location = "root"
        error_type = str(entry.get("type", ""))
        if error_type == "missing":
            missing_fields.append(location)
        else:
            invalid_fields.append(location)
    return _dedupe_strings(missing_fields), _dedupe_strings(invalid_fields)


def _format_loc(loc: Sequence[Any]) -> str:
    parts: List[str] = []
    for item in loc:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
            continue
        parts.append(str(item))
    return ".".join(part for part in parts if part)


def _classify_validation_error(
    tool_name: str,
    missing_fields: List[str],
    invalid_fields: List[str],
) -> str:
    all_fields = missing_fields + invalid_fields
    if tool_name == "retrieve_facts":
        if _has_field(all_fields, "request_id"):
            return "missing_request_id"
        if _has_field(all_fields, "intent"):
            return "missing_intent"
        return "invalid_retrieval_request_schema"
    if tool_name == "execute_plan_patch":
        if _has_field(all_fields, "day"):
            return "missing_day"
        return "invalid_patch_args"
    if tool_name in {"analyze_conflicts", "check_runtime_state"}:
        if _has_prefix(all_fields, "intent_anchors") or _has_field(all_fields, "semantic_key"):
            return "invalid_intent_anchor_shape"
        if _has_prefix(all_fields, "active_constraints"):
            return "invalid_active_constraint_shape"
    return "invalid_tool_arguments_schema"


def _classify_key_error(tool_name: str, missing_key: str) -> str:
    if tool_name == "execute_plan_patch" and missing_key == "day":
        return "missing_day"
    if tool_name == "execute_plan_patch":
        return "invalid_patch_args"
    return "invalid_tool_arguments_schema"


def _validation_error_message(
    tool_name: str,
    error_code: str,
    missing_fields: List[str],
    invalid_fields: List[str],
) -> str:
    if error_code == "missing_request_id":
        return "Each retrieve_facts request must include request_id. Re-call the tool with requests[i].request_id plus intent."
    if error_code == "missing_intent":
        return "Each retrieve_facts request must include intent. Re-call the tool with requests[i].intent plus request_id."
    if error_code == "missing_day":
        return "Each execute_plan_patch op args must include day for day-scoped mutations."
    if error_code == "invalid_intent_anchor_shape":
        return "intent_anchors entries must use the runtime anchor shape and include semantic_key."
    if error_code == "invalid_active_constraint_shape":
        return "active_constraints entries must include id, predicate, target, and params."
    if error_code == "invalid_patch_args":
        return "execute_plan_patch received invalid op args. Re-call the tool with a supported op and complete args."
    field_summary = []
    if missing_fields:
        field_summary.append("missing: " + ", ".join(missing_fields))
    if invalid_fields:
        field_summary.append("invalid: " + ", ".join(invalid_fields))
    suffix = f" ({'; '.join(field_summary)})" if field_summary else ""
    return f"{tool_name} arguments do not match the required schema{suffix}."


def _key_error_message(tool_name: str, error_code: str, missing_key: str) -> str:
    if error_code == "missing_day":
        return "execute_plan_patch is missing args.day for a day-scoped patch operation."
    if error_code == "invalid_patch_args":
        return f"execute_plan_patch is missing required arg '{missing_key}'."
    return f"{tool_name} is missing required arg '{missing_key}'."


def _has_field(fields: Sequence[str], field_name: str) -> bool:
    suffix = f".{field_name}"
    index_suffix = f"].{field_name}"
    return any(
        item == field_name or item.endswith(suffix) or item.endswith(index_suffix)
        for item in fields
    )


def _has_prefix(fields: Sequence[str], prefix: str) -> bool:
    return any(item == prefix or item.startswith(f"{prefix}.") or item.startswith(f"{prefix}[") for item in fields)


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _contextualize_validation_fields(
    tool_name: str,
    tool_args: Dict[str, Any],
    fields: List[str],
) -> List[str]:
    if tool_name not in {"analyze_conflicts", "check_runtime_state"}:
        return fields
    anchors = list(tool_args.get("intent_anchors") or [])
    if len(anchors) != 1:
        return fields
    contextualized: List[str] = []
    for field_name in fields:
        if field_name in {"semantic_key", "ref", "rule", "type", "id"}:
            contextualized.append(f"intent_anchors[0].{field_name}")
        else:
            contextualized.append(field_name)
    return _dedupe_strings(contextualized)
