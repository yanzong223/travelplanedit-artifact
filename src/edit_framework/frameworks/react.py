"""Standalone ReAct edit baseline."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

from agents.frameworks.react_framework import ReactFramework
from edit_framework.adapters.plan_adapter import require_chinatravel_plan, require_origin_plan
from edit_framework.base import EditFramework, EditInput, EditResult
from edit_framework.error_handling import (
    classify_terminal_error_categories,
    framework_error_handling_metadata,
)
from edit_framework.guard import apply_candidate_guard, build_guard_retry_edit_query
from edit_framework.infeasible_detection import parse_infeasible_response
from edit_framework.prompts.react import (
    REACT_SYSTEM_PROMPT,
    build_react_system_prompt,
    build_react_user_prompt,
)
from edit_framework.runtime_tools.types import ExposureMode
from edit_framework.tools.chinatravel_tools import ChinaTravelToolAdapter
from edit_framework.world_env import ensure_session_world_env
from utils.chinatravel_plan import normalize_loose_chinatravel_plan


class _StandaloneReactCore(ReactFramework):
    """React core wired to the standalone input and tool contracts."""

    def __init__(
        self,
        framework_id: str,
        llm_client: Any,
        world_env: Any,
        tool_adapter: Optional[ChinaTravelToolAdapter] = None,
        max_conversation_messages: int = 18,
        max_tool_rows_in_context: int = 3,
        max_tool_value_chars: int = 240,
        context_prompt: bool = False,
        database_prompt: bool = False,
        prompt_ablation: str = "original",
        annotation_scaffold_level: str = "none",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            framework_id=framework_id,
            llm_client=llm_client,
            world_env=world_env,
            **kwargs,
        )
        self.tool_adapter = tool_adapter or ChinaTravelToolAdapter(
            framework_name="react",
            exposure_mode=ExposureMode.HYBRID.value,
        )
        self.original_plan_for_tools: Dict[str, Any] | None = None
        self.current_edit_request: Dict[str, Any] = {}
        self.current_original_query: Dict[str, Any] = {}
        self.last_lifted_constraints: list[Dict[str, Any]] = []
        self.last_intent_anchors: list[Dict[str, Any]] = []
        self.ct_notepad: Dict[str, list[str]] = {}
        self.local_plan_mutation_count = 0
        self.infeasible_detection: Dict[str, Any] | None = None
        self.context_prompt = bool(context_prompt)
        self.database_prompt = bool(database_prompt)
        self.prompt_ablation = prompt_ablation
        self.annotation_scaffold_level = annotation_scaffold_level
        self.current_metadata: Dict[str, Any] = {}
        self.max_conversation_messages = max(4, max_conversation_messages)
        self.max_tool_rows_in_context = max(1, max_tool_rows_in_context)
        self.max_tool_value_chars = max(32, max_tool_value_chars)

    def reset(self) -> None:
        super().reset()
        self.original_plan_for_tools = None
        self.current_edit_request = {}
        self.current_original_query = {}
        self.last_lifted_constraints = []
        self.last_intent_anchors = []
        self.semantic_active_constraints = []
        self.semantic_intent_anchors = []
        self.semantic_grounded_conflicts = []
        self.semantic_retrieved_facts = []
        self.semantic_edit_proposals = []
        self.semantic_last_runtime_check = None
        self.semantic_last_decision = None
        self.ct_notepad = {}
        self.local_plan_mutation_count = 0
        self.infeasible_detection = None
        self.runtime_capability_state = None
        self.current_metadata = {}
        world_env_reset = getattr(self.world_env, "reset", None)
        if callable(world_env_reset):
            world_env_reset()

    async def edit_plan(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run query-only ReAct and always use the final LLM plan output."""

        import time

        start_time = time.time()
        try:
            self.reset()
            self.original_plan_for_tools = copy.deepcopy(original_plan)
            self.current_plan = copy.deepcopy(original_plan)
            self.current_edit_request = dict(edit_request)
            self.current_original_query = dict(original_query)
            self.current_metadata = dict(metadata or {})
            self.last_lifted_constraints = []
            self.last_intent_anchors = []
            self.ct_notepad = {}
            self.local_plan_mutation_count = 0
            messages = self._build_initial_messages(
                edit_request,
                original_plan,
                original_query,
                system_prompt,
                metadata=self.current_metadata,
            )
            tools = self._get_framework_tools()
            final_result = await self._execute_react_loop(
                messages,
                tools,
                edit_request,
                original_plan,
            )
            edited_plan = (
                self.current_plan
                if self.current_plan is not None and self.local_plan_mutation_count > 0
                else final_result
            )
            if self.infeasible_detection is not None:
                edited_plan = None
            self.metrics.execution_time_seconds = time.time() - start_time
            return {
                "success": True,
                "edited_plan": edited_plan,
                "conversation_log": self.get_execution_log(),
                "metrics": self.metrics.to_dict(),
                "framework_type": self.get_framework_type(),
                "errors": [],
            }
        except Exception as exc:
            self.metrics.execution_time_seconds = time.time() - start_time
            return {
                "success": False,
                "edited_plan": None,
                "conversation_log": self.get_execution_log(),
                "metrics": self.metrics.to_dict(),
                "framework_type": self.get_framework_type(),
                "errors": [str(exc)],
            }

    def _build_initial_messages(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        system_msg = {
            "role": "system",
            "content": system_prompt or build_react_system_prompt(self.tool_adapter.tool_flags(), prompt_ablation=self.prompt_ablation),
        }
        user_msg = {
            "role": "user",
            "content": build_react_user_prompt(
                origin_query_text=original_query.get("origin_query_text", ""),
                edit_query=edit_request.get("edit_query", ""),
                origin_plan=original_plan,
                metadata=metadata,
                context_prompt=self.context_prompt,
                database_prompt=self.database_prompt,
                annotation_scaffold_level=self.annotation_scaffold_level,
            ),
        }
        self.conversation_history = [system_msg, user_msg]
        return self.conversation_history.copy()

    def _get_query_tools(self) -> list[Dict[str, Any]]:
        return self.tool_adapter.read_only_tools()

    def _get_framework_tools(self) -> list[Dict[str, Any]]:
        """Expose query tools plus optional ChinaTravel-native local tools."""

        return self.tool_adapter.available_tools()

    async def _process_final_response(
        self,
        response: Any,
        messages: list[Dict[str, Any]],
        step_number: int,
    ) -> Dict[str, Any]:
        response_text = response if isinstance(response, str) else response.get("content", "")
        detection = parse_infeasible_response(response_text)
        if detection is not None:
            self.infeasible_detection = detection
            self._add_step(
                "infeasible_detection",
                detection.get("reason", ""),
                metadata={
                    "step_type": "infeasible_detection",
                    "infeasible_detection": detection,
                },
            )
            messages.append({"role": "assistant", "content": response_text})
            return {
                "next_action": "finish",
                "messages": messages,
                "final_result": None,
            }
        return await super()._process_final_response(response, messages, step_number)

    async def _execute_function_calling_step(
        self,
        messages: list[Dict[str, Any]],
        tools: list[Dict[str, Any]],
        step_number: int,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_messages = self._build_request_messages(messages)
        response = await self.llm_client.chat_completion(
            messages=request_messages,
            tools=tools,
            temperature=0.7,
            max_tokens=self.max_completion_tokens,
        )

        if self.include_thoughts:
            self._add_step(
                "thought",
                "Analyzing current situation and deciding next action...",
                metadata={"step_type": "reasoning"},
            )

        if isinstance(response, dict) and response.get("tool_calls"):
            return self._process_tool_calls(response, messages, step_number)
        return await self._process_final_response(response, messages, step_number)

    def _process_tool_calls(
        self,
        response: Dict[str, Any],
        messages: list[Dict[str, Any]],
        step_number: int,
    ) -> Dict[str, Any]:
        messages.append(self._assistant_message_from_response(response))

        self._add_step(
            "action",
            response.get("content", "Making tool calls..."),
            tool_calls=response["tool_calls"],
            metadata={"response_format": "function_calling"},
        )

        tool_results = []
        for tool_call in response["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments", "{}")
            tool_args, parse_error = self._parse_tool_arguments_safe(raw_args)
            if parse_error is not None:
                tool_result = {
                    "ok": False,
                    "error_code": "invalid_tool_arguments_json",
                    "message": parse_error,
                    "tool_name": tool_name,
                    "raw_arguments": raw_args,
                }
                self.metrics.tool_calls_made += 1
                self.metrics.failed_tool_calls += 1
                self.metrics.tool_argument_parse_error_count += 1
                compact_tool_result = self._compact_tool_result_for_context(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(compact_tool_result, ensure_ascii=False),
                    }
                )
                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "tool_args": {},
                        "result": compact_tool_result,
                    }
                )
                continue
            tool_result = self._execute_tool(tool_name, tool_args)
            compact_tool_result = self._compact_tool_result_for_context(tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(compact_tool_result, ensure_ascii=False),
                }
            )

            tool_results.append(
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": compact_tool_result,
                }
            )

        self._add_step(
            "observation",
            f"Executed {len(response['tool_calls'])} tool calls",
            tool_results=tool_results,
            metadata={"step_type": "tool_execution"},
        )

        return {
            "next_action": "continue",
            "messages": messages,
            "tool_results": tool_results,
        }

    def _build_env_command(self, tool_name: str, args: Dict[str, Any]) -> str:
        return self.tool_adapter.build_env_command(tool_name, args)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        local_result = self.tool_adapter.execute_local_tool(tool_name, args, self)
        if local_result is not None:
            self.metrics.tool_calls_made += 1
            if local_result.get("ok"):
                self.metrics.successful_tool_calls += 1
                if tool_name in {
                    "insert_activity_ct",
                    "replace_activity_ct",
                    "move_activity_ct",
                    "delete_activity_ct",
                    "reschedule_activity_ct",
                    "resize_activity_ct",
                    "reorder_day_ct",
                    "reroute_transport_ct",
                }:
                    self.local_plan_mutation_count += 1
            else:
                self.metrics.failed_tool_calls += 1
            return local_result
        return self._execute_query_tool(tool_name, args)

    def _execute_query_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            normalized_args = self.tool_adapter.normalize_query_args(tool_name, args)
            validation_error = self.tool_adapter.validate_query_args(tool_name, normalized_args)
            if validation_error is not None:
                self.metrics.tool_calls_made += 1
                self.metrics.failed_tool_calls += 1
                return validation_error
            command = self._build_env_command(tool_name, normalized_args)
            result = self.world_env(command)
            tool_result = self.tool_adapter.format_tool_result(
                tool_name,
                normalized_args,
                command,
                result,
            )
            self.metrics.tool_calls_made += 1
            if tool_result.get("ok"):
                self.metrics.successful_tool_calls += 1
            else:
                self.metrics.failed_tool_calls += 1
            return tool_result
        except Exception as exc:
            self.metrics.tool_calls_made += 1
            self.metrics.failed_tool_calls += 1
            return {
                "ok": False,
                "error_code": "tool_execution_failed",
                "message": str(exc),
                "tool_name": tool_name,
                "tool_args": args,
            }

    def _build_request_messages(
        self,
        messages: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        return self._trim_messages_preserving_tool_pairs(
            messages,
            max_messages=self.max_conversation_messages,
            head_messages=2,
        )

    def _compact_tool_result_for_context(
        self,
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_name = str(tool_result.get("tool_name") or "")
        compact = {
            "ok": bool(tool_result.get("ok")),
            "tool_name": tool_name or None,
        }
        if "message" in tool_result:
            compact["message"] = self._truncate_scalar(tool_result.get("message"))
        if "error_code" in tool_result:
            compact["error_code"] = tool_result.get("error_code")
        if "missing_fields" in tool_result:
            compact["missing_fields"] = self._truncate_jsonable(tool_result.get("missing_fields"))
        if "invalid_fields" in tool_result:
            compact["invalid_fields"] = self._truncate_jsonable(tool_result.get("invalid_fields"))
        if "expected_shape" in tool_result:
            compact["expected_shape"] = self._truncate_jsonable(tool_result.get("expected_shape"))
        semantic_payload = self._compact_semantic_tool_result(tool_name, tool_result)
        if semantic_payload:
            compact.update(semantic_payload)
            return compact
        if "page" in tool_result:
            compact["page"] = tool_result.get("page")
        if "cursor_id" in tool_result:
            compact["cursor_id"] = tool_result.get("cursor_id")
        if "rows" in tool_result and isinstance(tool_result["rows"], list):
            rows = tool_result["rows"]
            compact["rows"] = [
                self._truncate_jsonable(row) for row in rows[: self.max_tool_rows_in_context]
            ]
            compact["rows_truncated"] = len(rows) > self.max_tool_rows_in_context
        if "value" in tool_result:
            compact["value"] = self._truncate_jsonable(tool_result.get("value"))
        if "plan" in tool_result and isinstance(tool_result["plan"], dict):
            compact["plan_summary"] = self._summarize_plan_for_context(tool_result["plan"])
        if "delta" in tool_result and isinstance(tool_result["delta"], list):
            deltas = tool_result["delta"]
            compact["delta"] = [
                self._truncate_jsonable(item) for item in deltas[: self.max_tool_rows_in_context]
            ]
            compact["delta_truncated"] = len(deltas) > self.max_tool_rows_in_context
        if "issues" in tool_result and isinstance(tool_result["issues"], list):
            compact["issues"] = [
                self._truncate_jsonable(item)
                for item in tool_result["issues"][: self.max_tool_rows_in_context]
            ]
        if "pending_repairs" in tool_result:
            compact["pending_repairs"] = self._truncate_jsonable(tool_result["pending_repairs"])
        if "notepad" in tool_result and isinstance(tool_result["notepad"], dict):
            compact["notepad_summary"] = {
                section: len(notes) if isinstance(notes, list) else 0
                for section, notes in tool_result["notepad"].items()
            }
        return compact

    def _compact_semantic_tool_result(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Preserve semantic-tool outputs needed for downstream tool chaining."""
        if tool_name == "construct_constraints":
            return {
                "active_constraints": self._truncate_jsonable(tool_result.get("active_constraints", [])),
                "intent_anchors": self._truncate_jsonable(tool_result.get("intent_anchors", [])),
                "lifted_preferences": self._truncate_jsonable(tool_result.get("lifted_preferences", [])),
                "uncertainties": self._truncate_jsonable(tool_result.get("uncertainties", [])),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
        if tool_name == "analyze_conflicts":
            return {
                "violated_constraints": self._truncate_jsonable(tool_result.get("violated_constraints", [])),
                "grounded_conflicts": self._truncate_jsonable(tool_result.get("grounded_conflicts", [])),
                "unresolved_checks": self._truncate_jsonable(tool_result.get("unresolved_checks", [])),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
        if tool_name == "retrieve_facts":
            return {
                "results": self._truncate_jsonable(tool_result.get("results", [])),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
        if tool_name == "generate_edit_proposals":
            return {
                "edit_proposals": self._truncate_jsonable(tool_result.get("edit_proposals", [])),
                "issues": self._truncate_jsonable(tool_result.get("issues", [])),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
        if tool_name == "decide_next_step":
            return {
                "decision": self._truncate_jsonable(tool_result.get("decision", {})),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
        if tool_name == "execute_plan_patch":
            compact: Dict[str, Any] = {
                "applied_ops": self._truncate_jsonable(tool_result.get("applied_ops", [])),
            }
            if "plan" in tool_result and isinstance(tool_result["plan"], dict):
                compact["plan_summary"] = self._summarize_plan_for_context(tool_result["plan"])
            if "delta" in tool_result and isinstance(tool_result["delta"], list):
                deltas = tool_result["delta"]
                compact["delta"] = [
                    self._truncate_jsonable(item) for item in deltas[: self.max_tool_rows_in_context]
                ]
                compact["delta_truncated"] = len(deltas) > self.max_tool_rows_in_context
            if "issues" in tool_result:
                compact["issues"] = self._truncate_jsonable(tool_result.get("issues", []))
            if "pending_repairs" in tool_result:
                compact["pending_repairs"] = self._truncate_jsonable(tool_result.get("pending_repairs"))
            return compact
        if tool_name == "check_runtime_state":
            return {
                "structural_status": self._truncate_jsonable(tool_result.get("structural_status", {})),
                "constraint_checks": self._truncate_jsonable(tool_result.get("constraint_checks", [])),
                "risk_flags": self._truncate_jsonable(tool_result.get("risk_flags", [])),
                "next_step_hint": self._truncate_scalar(tool_result.get("next_step_hint")),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
        if tool_name == "diagnose_edit_requirements":
            return {
                "edit_targets": self._truncate_jsonable(tool_result.get("edit_targets", [])),
                "preserve_anchors": self._truncate_jsonable(tool_result.get("preserve_anchors", [])),
                "explicit_constraints": self._truncate_jsonable(tool_result.get("explicit_constraints", [])),
                "required_fact_slots": self._truncate_jsonable(tool_result.get("required_fact_slots", [])),
                "risk_flags": self._truncate_jsonable(tool_result.get("risk_flags", [])),
                "infeasible_signals": self._truncate_jsonable(tool_result.get("infeasible_signals", [])),
            }
        if tool_name == "guard_candidate_plan":
            return {
                "decision": self._truncate_scalar(tool_result.get("decision")),
                "must_fix_violations": self._truncate_jsonable(tool_result.get("must_fix_violations", [])),
                "preservation_violations": self._truncate_jsonable(tool_result.get("preservation_violations", [])),
                "feasibility_violations": self._truncate_jsonable(tool_result.get("feasibility_violations", [])),
                "fact_grounding_risks": self._truncate_jsonable(tool_result.get("fact_grounding_risks", [])),
                "compact_repair_hints": self._truncate_jsonable(tool_result.get("compact_repair_hints", [])),
            }
        return {}

    def _summarize_plan_for_context(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        itinerary = plan.get("itinerary", [])
        day_summaries = []
        for day_payload in itinerary[: self.max_tool_rows_in_context]:
            activities = day_payload.get("activities", [])
            day_summaries.append(
                {
                    "day": day_payload.get("day"),
                    "activity_count": len(activities),
                    "activity_ids": [
                        activity.get("id")
                        for activity in activities[: self.max_tool_rows_in_context]
                        if isinstance(activity, dict)
                    ],
                }
            )
        return {
            "day_count": len(itinerary),
            "days": day_summaries,
            "days_truncated": len(itinerary) > self.max_tool_rows_in_context,
        }

    def _truncate_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._truncate_jsonable(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._truncate_jsonable(item) for item in value[: self.max_tool_rows_in_context]]
        if isinstance(value, str):
            return self._truncate_scalar(value)
        return value

    def _truncate_scalar(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if len(value) <= self.max_tool_value_chars:
            return value
        return value[: self.max_tool_value_chars] + "...(truncated)"


class ReactEditFramework(EditFramework):
    """Standalone ReAct runtime over the legacy framework core."""

    framework_name = "react"

    def __init__(
        self,
        llm_client: Any,
        world_env: Any,
        *,
        tool_adapter: Optional[ChinaTravelToolAdapter] = None,
        max_steps: int = 30,
        max_tool_calls: int = 30,
        max_completion_tokens: Optional[int] = None,
        json_repair_max_tokens: Optional[int] = None,
        use_function_calling: bool = True,
        include_thoughts: bool = False,
        max_conversation_messages: int = 18,
        max_tool_rows_in_context: int = 3,
        max_tool_value_chars: int = 240,
        context_prompt: bool = False,
        database_prompt: bool = False,
        prompt_ablation: str = "original",
        guard_retries: int = 0,
        annotation_scaffold_level: str = "none",
    ) -> None:
        self.guard_retries = int(guard_retries)
        self.prompt_ablation = prompt_ablation
        self.annotation_scaffold_level = annotation_scaffold_level
        self.core = _StandaloneReactCore(
            framework_id="standalone_react",
            llm_client=llm_client,
            world_env=ensure_session_world_env(world_env),
            tool_adapter=tool_adapter,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_completion_tokens=max_completion_tokens,
            json_repair_max_tokens=json_repair_max_tokens,
            use_function_calling=use_function_calling,
            include_thoughts=include_thoughts,
            max_conversation_messages=max_conversation_messages,
            max_tool_rows_in_context=max_tool_rows_in_context,
            max_tool_value_chars=max_tool_value_chars,
            context_prompt=context_prompt,
            database_prompt=database_prompt,
            prompt_ablation=prompt_ablation,
            annotation_scaffold_level=annotation_scaffold_level,
        )

    async def run(self, edit_input: EditInput) -> EditResult:
        origin_plan = require_origin_plan(edit_input.origin_plan)
        self.core.original_plan_for_tools = origin_plan
        self.core.current_plan = copy.deepcopy(origin_plan)
        self.core.current_edit_request = {"edit_query": edit_input.edit_query}
        self.core.current_original_query = {"origin_query_text": edit_input.origin_query_text}
        self.core.current_metadata = dict(edit_input.metadata)
        self.core.last_lifted_constraints = []
        self.core.last_intent_anchors = []
        self.core.ct_notepad = {}
        self.core.local_plan_mutation_count = 0
        framework_result = await self.core.edit_plan(
            edit_request={"edit_query": edit_input.edit_query},
            original_plan=origin_plan,
            original_query={"origin_query_text": edit_input.origin_query_text},
            system_prompt=build_react_system_prompt(self.core.tool_adapter.tool_flags(), prompt_ablation=self.prompt_ablation),
            metadata=edit_input.metadata,
        )

        edited_plan = framework_result.get("edited_plan")
        if self.core.infeasible_detection is None:
            edited_plan = normalize_loose_chinatravel_plan(edited_plan)
        if self.core.infeasible_detection is None and framework_result.get("success", False):
            try:
                edited_plan = require_chinatravel_plan(edited_plan, context="edited_plan")
                framework_result["edited_plan"] = edited_plan
            except Exception as exc:
                framework_result["success"] = False
                framework_result.setdefault("errors", []).append(str(exc))
        if self.core.infeasible_detection is None:
            framework_result = apply_candidate_guard(
                runtime=self.core,
                tool_adapter=self.core.tool_adapter,
                framework_result=framework_result,
                original_plan=origin_plan,
                edit_query=edit_input.edit_query,
                guard_retries=self.guard_retries,
            )
            edited_plan = framework_result.get("edited_plan")
            if (
                not framework_result.get("success", False)
                and self.guard_retries > 0
                and isinstance(framework_result.get("metrics", {}).get("guard_report"), dict)
                and framework_result.get("metrics", {}).get("guard_decision") == "revise"
            ):
                first_guard_report = framework_result["metrics"]["guard_report"]
                retry_edit_query = build_guard_retry_edit_query(edit_input.edit_query, first_guard_report)
                self.core.infeasible_detection = None
                framework_result = await self.core.edit_plan(
                    edit_request={"edit_query": retry_edit_query},
                    original_plan=origin_plan,
                    original_query={"origin_query_text": edit_input.origin_query_text},
                    system_prompt=build_react_system_prompt(
                        self.core.tool_adapter.tool_flags(),
                        prompt_ablation=self.prompt_ablation,
                    ),
                    metadata=edit_input.metadata,
                )
                edited_plan = framework_result.get("edited_plan")
                if self.core.infeasible_detection is None:
                    edited_plan = normalize_loose_chinatravel_plan(edited_plan)
                if self.core.infeasible_detection is None and framework_result.get("success", False):
                    try:
                        edited_plan = require_chinatravel_plan(edited_plan, context="edited_plan")
                        framework_result["edited_plan"] = edited_plan
                    except Exception as exc:
                        framework_result["success"] = False
                        framework_result.setdefault("errors", []).append(str(exc))
                framework_result = apply_candidate_guard(
                    runtime=self.core,
                    tool_adapter=self.core.tool_adapter,
                    framework_result=framework_result,
                    original_plan=origin_plan,
                    edit_query=edit_input.edit_query,
                    guard_retries=self.guard_retries,
                )
                framework_result.setdefault("metrics", {})["guard_retry_attempts"] = 1
                framework_result.setdefault("metrics", {})["guard_previous_reports"] = [first_guard_report]
                edited_plan = framework_result.get("edited_plan")

        return EditResult(
            success=framework_result.get("success", False),
            framework=self.framework_name,
            exposure_mode=self.core.tool_adapter.exposure_mode.value,
            tool_profile=self.core.tool_adapter.tool_profile,
            db_read_enabled=self.core.tool_adapter.db_read_enabled,
            edited_plan=edited_plan,
            ops=[],
            trace=framework_result.get("conversation_log", []),
            metrics={
                **framework_result.get("metrics", {}),
                **framework_error_handling_metadata(
                    self.framework_name,
                    max_steps=self.core.max_steps,
                    max_tool_calls=self.core.max_tool_calls,
                ),
                "terminal_error_categories": classify_terminal_error_categories(
                    framework_result.get("errors", []),
                    metrics=framework_result.get("metrics", {}),
                )
                if not framework_result.get("success", False)
                else [],
                "tool_flags": self.core.tool_adapter.tool_flags(),
                "infeasible_detection": self.core.infeasible_detection,
                "prompt_modes": {
                    "context_prompt": self.core.context_prompt,
                    "database_prompt": self.core.database_prompt,
                    "annotation_scaffold_level": self.annotation_scaffold_level,
                },
            },
            errors=framework_result.get("errors", []),
        )
