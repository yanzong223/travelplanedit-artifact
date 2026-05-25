"""Session-aware world environment for standalone edit baselines."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from pandas import DataFrame
except Exception:  # pragma: no cover - pandas is available in runtime/test env
    DataFrame = None  # type: ignore[assignment]


@dataclass
class PagedResultEntry:
    cursor_id: str
    command: str
    tool_name: str
    whole_data: Any
    page_size: int
    current_page_index: int
    total_rows: int
    total_pages: int
    is_exhausted: bool = False


@dataclass
class QueryHistoryEntry:
    command: str
    tool_name: str
    success: bool
    cursor_id: str | None = None


@dataclass
class QuerySessionState:
    entries_by_cursor: dict[str, PagedResultEntry] = field(default_factory=dict)
    last_cursor_id: str | None = None
    history: list[QueryHistoryEntry] = field(default_factory=list)


@dataclass
class QueryExecutionResult:
    raw_result: Any
    tool_name: str
    success: bool
    cursor_id: str | None = None


class SessionWorldEnv:
    """Wrap a callable world env with cursor-based paging state."""

    def __init__(self, backend_env: Callable[[str], Any], page_size: int = 10) -> None:
        self.backend_env = backend_env
        self.page_size = max(1, int(page_size))
        self.session = QuerySessionState()
        self._cursor_seq = 0

    def __call__(self, command: str) -> Any:
        return self.execute_query(command).raw_result

    def execute_query(self, command: str) -> QueryExecutionResult:
        tool_name = _extract_tool_name(command)
        if tool_name == "next_page":
            cursor_id = _extract_next_page_cursor(command)
            result = self.next_page(cursor_id)
            self.session.history.append(
                QueryHistoryEntry(
                    command=command,
                    tool_name=tool_name,
                    success=bool(result.get("success", False)),
                    cursor_id=cursor_id,
                )
            )
            return QueryExecutionResult(
                raw_result=result,
                tool_name=tool_name,
                success=bool(result.get("success", False)),
                cursor_id=cursor_id,
            )

        result = self.backend_env(command)
        success, data, whole_data = _extract_env_payload(result)
        cursor_id: str | None = None
        if success and _is_pageable_payload(whole_data):
            cursor_id = self._register_paged_result(command, tool_name, whole_data)
            result = self._build_initial_paged_response(cursor_id, whole_data)
        self.session.history.append(
            QueryHistoryEntry(
                command=command,
                tool_name=tool_name,
                success=success,
                cursor_id=cursor_id,
            )
        )
        return QueryExecutionResult(
            raw_result=result,
            tool_name=tool_name,
            success=success,
            cursor_id=cursor_id,
        )

    def next_page(self, cursor_id: str) -> dict[str, Any]:
        entry = self.session.entries_by_cursor.get(cursor_id)
        if entry is None:
            return {
                "success": False,
                "data": f"Invalid cursor_id: {cursor_id}",
                "whole_data": None,
                "error_code": "invalid_cursor",
            }
        if entry.is_exhausted or entry.current_page_index >= entry.total_pages:
            entry.is_exhausted = True
            return {
                "success": False,
                "data": f"No more pages for cursor_id: {cursor_id}",
                "whole_data": entry.whole_data,
                "error_code": "no_more_pages",
                "cursor_id": cursor_id,
                "page": self._page_meta(entry, has_next=False),
            }

        entry.current_page_index += 1
        entry.is_exhausted = entry.current_page_index >= entry.total_pages
        page_slice = _slice_payload(
            entry.whole_data,
            (entry.current_page_index - 1) * entry.page_size,
            entry.current_page_index * entry.page_size,
        )
        return {
            "success": True,
            "data": page_slice,
            "whole_data": entry.whole_data,
            "cursor_id": cursor_id,
            "page": self._page_meta(entry, has_next=not entry.is_exhausted),
        }

    def reset(self) -> None:
        self.session = QuerySessionState()
        self._cursor_seq = 0
        backend_reset = getattr(self.backend_env, "reset", None)
        if callable(backend_reset):
            backend_reset()

    def _register_paged_result(self, command: str, tool_name: str, whole_data: Any) -> str:
        self._cursor_seq += 1
        cursor_id = f"cursor_{self._cursor_seq}"
        total_rows = _payload_length(whole_data)
        total_pages = max(1, math.ceil(total_rows / self.page_size))
        entry = PagedResultEntry(
            cursor_id=cursor_id,
            command=command,
            tool_name=tool_name,
            whole_data=whole_data,
            page_size=self.page_size,
            current_page_index=1,
            total_rows=total_rows,
            total_pages=total_pages,
            is_exhausted=total_pages <= 1,
        )
        self.session.entries_by_cursor[cursor_id] = entry
        self.session.last_cursor_id = cursor_id
        return cursor_id

    def _build_initial_paged_response(self, cursor_id: str, whole_data: Any) -> dict[str, Any]:
        entry = self.session.entries_by_cursor[cursor_id]
        page_slice = _slice_payload(whole_data, 0, entry.page_size)
        return {
            "success": True,
            "data": page_slice,
            "whole_data": whole_data,
            "cursor_id": cursor_id,
            "page": self._page_meta(entry, has_next=not entry.is_exhausted),
        }

    def _page_meta(self, entry: PagedResultEntry, *, has_next: bool) -> dict[str, Any]:
        return {
            "page": entry.current_page_index,
            "page_size": entry.page_size,
            "total": entry.total_rows,
            "has_next": has_next,
        }


def ensure_session_world_env(world_env: Any, page_size: int = 10) -> SessionWorldEnv:
    if isinstance(world_env, SessionWorldEnv):
        return world_env
    return SessionWorldEnv(world_env, page_size=page_size)


def _extract_tool_name(command: str) -> str:
    try:
        expr = ast.parse(command, mode="eval")
    except SyntaxError:
        return command.split("(", 1)[0].strip()
    call = expr.body
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
        return call.func.id
    return command.split("(", 1)[0].strip()


def _extract_next_page_cursor(command: str) -> str:
    expr = ast.parse(command, mode="eval")
    call = expr.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "next_page":
        raise ValueError(f"Unsupported next_page command: {command}")
    if not call.args:
        raise ValueError("next_page requires cursor_id")
    return str(ast.literal_eval(call.args[0]))


def _extract_env_payload(result: Any) -> tuple[bool, Any, Any]:
    if hasattr(result, "to_dict"):
        result = result.to_dict()

    if isinstance(result, dict):
        success = bool(result.get("success", False))
        data = result.get("data")
        whole_data = result.get("whole_data", data)
        return success, data, whole_data

    if hasattr(result, "__getitem__"):
        try:
            success = bool(result["success"])
            data = result["data"]
            whole_data = result["whole_data"]
            return success, data, whole_data
        except Exception:
            pass

    return False, result, result


def _is_pageable_payload(payload: Any) -> bool:
    if DataFrame is not None and isinstance(payload, DataFrame):
        return True
    return isinstance(payload, list)


def _payload_length(payload: Any) -> int:
    if DataFrame is not None and isinstance(payload, DataFrame):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _slice_payload(payload: Any, start: int, end: int) -> Any:
    if DataFrame is not None and isinstance(payload, DataFrame):
        return payload.iloc[start:end]
    if isinstance(payload, list):
        return payload[start:end]
    return payload
