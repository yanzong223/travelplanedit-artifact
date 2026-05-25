"""Standalone PTE-R edit baseline."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

from agents.frameworks.pter_framework import PTERFramework
from edit_framework.adapters.plan_adapter import require_chinatravel_plan, require_origin_plan
from edit_framework.base import EditFramework, EditInput, EditResult
from edit_framework.error_handling import (
    classify_terminal_error_categories,
    framework_error_handling_metadata,
)
from edit_framework.guard import apply_candidate_guard, build_guard_retry_edit_query
from edit_framework.infeasible_detection import infeasible_result, parse_infeasible_response
from edit_framework.prompts.pter import (
    PTER_SYSTEM_PROMPT,
    build_pter_system_prompt,
    build_pter_user_prompt,
)
from edit_framework.runtime_tools.types import ExposureMode
from edit_framework.tools.chinatravel_tools import ChinaTravelToolAdapter
from edit_framework.world_env import ensure_session_world_env


class _StandalonePTERCore(PTERFramework):
    """PTE-R core wired to the standalone input and tool contracts."""

    def __init__(
        self,
        framework_id: str,
        llm_client: Any,
        world_env: Any,
        tool_adapter: Optional[ChinaTravelToolAdapter] = None,
        prompt_ablation: str = "original",
        patch_repair_retries: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            framework_id=framework_id,
            llm_client=llm_client,
            world_env=world_env,
            **kwargs,
        )
        self.tool_adapter = tool_adapter or ChinaTravelToolAdapter(
            framework_name="pter",
            exposure_mode=ExposureMode.HYBRID.value,
        )
        self.generated_ops: list[Dict[str, Any]] = []
        self.current_plan: Dict[str, Any] | None = None
        self.original_plan_for_tools: Dict[str, Any] | None = None
        self.current_edit_request: Dict[str, Any] = {}
        self.current_original_query: Dict[str, Any] = {}
        self.last_lifted_constraints: list[Dict[str, Any]] = []
        self.last_intent_anchors: list[Dict[str, Any]] = []
        self.ct_notepad: Dict[str, list[str]] = {}
        self.prompt_ablation = prompt_ablation
        self.infeasible_detection: Dict[str, Any] | None = None
        self.patch_repair_retries = max(0, int(patch_repair_retries))
        self.pter_repair_metrics: Dict[str, Any] = self._new_repair_metrics()

    @staticmethod
    def _new_repair_metrics() -> Dict[str, Any]:
        return {
            "pter_ops_repair_attempts": 0,
            "pter_ops_repair_success": False,
            "pter_execution_repair_attempts": 0,
            "pter_execution_repair_success": False,
            "pter_repair_last_error": None,
        }

    def reset(self) -> None:
        super().reset()
        self.generated_ops = []
        self.current_plan = None
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
        self.infeasible_detection = None
        self.runtime_capability_state = None
        self.pter_repair_metrics = self._new_repair_metrics()
        world_env_reset = getattr(self.world_env, "reset", None)
        if callable(world_env_reset):
            world_env_reset()

    def _build_planning_messages(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        system_msg = {
            "role": "system",
            "content": system_prompt or build_pter_system_prompt(self.tool_adapter.tool_flags(), prompt_ablation=self.prompt_ablation),
        }
        user_msg = {
            "role": "user",
            "content": build_pter_user_prompt(
                origin_query_text=original_query.get("origin_query_text", ""),
                edit_query=edit_request.get("edit_query", ""),
                origin_plan=original_plan,
            ),
        }
        self.conversation_history = [system_msg, user_msg]
        return self.conversation_history.copy()

    def _get_framework_tools(self) -> list[Dict[str, Any]]:
        return self.tool_adapter.available_tools()

    def _build_env_command(self, tool_name: str, args: Dict[str, Any]) -> str:
        return self.tool_adapter.build_env_command(tool_name, args)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            local_result = self.tool_adapter.execute_local_tool(tool_name, args, self)
            if local_result is not None:
                self.metrics.tool_calls_made += 1
                if local_result.get("ok"):
                    self.metrics.successful_tool_calls += 1
                else:
                    self.metrics.failed_tool_calls += 1
                return local_result
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

    async def _generate_plan_ops_with_query(
        self,
        messages: list[Dict[str, Any]],
    ) -> tuple[list[Dict[str, Any]], int]:
        query_rounds = 0
        tools = self._get_framework_tools()

        for _ in range(self.max_query_rounds):
            query_rounds += 1
            request_messages = self._build_request_messages(messages)
            response = await self.llm_client.chat_completion(
                messages=request_messages,
                tools=tools,
                temperature=0.3,
                max_tokens=self.max_completion_tokens,
            )

            if isinstance(response, dict) and response.get("tool_calls"):
                tool_calls = response["tool_calls"]
                messages.append(self._assistant_message_from_response(response, tool_calls=tool_calls))
                self._add_step(
                    "query",
                    f"Round {query_rounds}: Executing {len(tool_calls)} tool calls",
                    tool_calls=tool_calls,
                    metadata={"round": query_rounds},
                )
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"].get("arguments", "{}")
                    tool_args, parse_error = self._parse_tool_arguments_safe(raw_args)
                    if parse_error is not None:
                        self.metrics.tool_calls_made += 1
                        self.metrics.failed_tool_calls += 1
                        self.metrics.tool_argument_parse_error_count += 1
                        tool_result = {
                            "ok": False,
                            "error_code": "invalid_tool_arguments_json",
                            "message": parse_error,
                            "tool_name": tool_name,
                            "raw_arguments": raw_args,
                        }
                    else:
                        tool_result = self._execute_tool(tool_name, tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": self._tool_result_content(tool_result),
                        }
                    )
                continue

            content = response if isinstance(response, str) else response.get("content", "")
            messages.append({"role": "assistant", "content": content})
            detection = parse_infeasible_response(content)
            if detection is not None:
                self.infeasible_detection = detection
                self._add_step(
                    "infeasible_detection",
                    detection.get("reason", ""),
                    metadata={
                        "query_rounds": query_rounds,
                        "infeasible_detection": detection,
                    },
                )
                return [], query_rounds

            ops = await self._extract_patch_ops_with_repair(
                content,
                messages,
                original_plan=self.current_plan or {},
                edit_query=str(self.current_edit_request.get("edit_query") or ""),
            )
            self.generated_ops = ops
            return ops, query_rounds

        content = await self._request_final_ops_without_tools(
            messages,
            reason=(
                f"你已经进行了 {self.max_query_rounds} 轮查询。"
                "不要再调用任何工具，直接输出最终 patch ops JSON 数组；"
                "若无法满足全部硬约束，输出 status=infeasible 的 JSON 对象。"
            ),
        )
        detection = parse_infeasible_response(content)
        if detection is not None:
            self.infeasible_detection = detection
            self._add_step(
                "infeasible_detection",
                detection.get("reason", ""),
                metadata={
                    "query_rounds": query_rounds,
                    "infeasible_detection": detection,
                    "forced_finalization": True,
                },
            )
            return [], query_rounds
        ops = await self._extract_patch_ops_with_repair(
            content,
            messages,
            original_plan=self.current_plan or {},
            edit_query=str(self.current_edit_request.get("edit_query") or ""),
        )
        self.generated_ops = ops
        return ops, query_rounds

    async def _extract_patch_ops_with_repair(
        self,
        content: str,
        messages: list[Dict[str, Any]],
        *,
        original_plan: Dict[str, Any],
        edit_query: str,
    ) -> list[Dict[str, Any]]:
        try:
            return self._extract_patch_ops(content)
        except Exception as exc:
            self.pter_repair_metrics["pter_repair_last_error"] = str(exc)
            if self.patch_repair_retries <= 0:
                raise

            self.pter_repair_metrics["pter_ops_repair_attempts"] += 1
            self._add_step(
                "ops_repair",
                "Repairing invalid PTE-R final ops output",
                metadata={
                    "error": str(exc),
                    "rejected_output": content,
                    "repair_attempt": self.pter_repair_metrics["pter_ops_repair_attempts"],
                },
            )
            repaired_content = await self._request_repaired_ops_without_tools(
                messages,
                original_plan=original_plan,
                edit_query=edit_query,
                rejected_output=content,
                error_message=str(exc),
                failure_phase="ops_schema",
            )
            try:
                repaired_ops = self._extract_patch_ops(repaired_content)
            except Exception as repair_exc:
                self.pter_repair_metrics["pter_repair_last_error"] = str(repair_exc)
                raise repair_exc from exc

            self.pter_repair_metrics["pter_ops_repair_success"] = True
            self._add_step(
                "ops_repair_result",
                f"Repaired invalid PTE-R ops output with {len(repaired_ops)} ops",
                metadata={"ops": repaired_ops},
            )
            return repaired_ops

    async def _request_repaired_ops_without_tools(
        self,
        messages: list[Dict[str, Any]],
        *,
        original_plan: Dict[str, Any],
        edit_query: str,
        rejected_output: Any,
        error_message: str,
        failure_phase: str,
    ) -> str:
        repair_messages = self._build_compact_patch_repair_messages(
            messages,
            original_plan=original_plan,
        )
        repair_messages.append(
            {
                "role": "user",
                "content": self._build_patch_repair_prompt(
                    original_plan=original_plan,
                    edit_query=edit_query,
                    rejected_output=rejected_output,
                    error_message=error_message,
                    failure_phase=failure_phase,
                ),
            }
        )
        response = await self.llm_client.chat_completion(
            messages=repair_messages,
            temperature=0.0,
            max_tokens=self.max_completion_tokens,
        )
        return response if isinstance(response, str) else response.get("content", "")

    def _build_compact_patch_repair_messages(
        self,
        messages: list[Dict[str, Any]],
        *,
        original_plan: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """Build a small repair context so long failed outputs do not exhaust context."""
        base_messages = [
            message
            for message in messages[:2]
            if message.get("role") in {"system", "user"}
        ]
        if len(base_messages) <= 1:
            return base_messages
        user_message = copy.deepcopy(base_messages[1])
        content = str(user_message.get("content") or "")
        marker = "# 原始计划"
        if marker in content:
            content = content.split(marker, 1)[0].rstrip()
        user_message["content"] = (
            content
            + "\n\n# 原始计划结构摘要\n"
            + json.dumps(
                self._summarize_plan_for_repair(original_plan),
                ensure_ascii=False,
                indent=2,
            )
        )
        return [copy.deepcopy(base_messages[0]), user_message]

    def _summarize_plan_for_repair(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        itinerary = plan.get("itinerary", [])
        days: list[Dict[str, Any]] = []
        if isinstance(itinerary, list):
            for day_index, day_payload in enumerate(itinerary):
                if not isinstance(day_payload, dict):
                    continue
                activities = day_payload.get("activities", [])
                activity_summaries: list[Dict[str, Any]] = []
                if isinstance(activities, list):
                    for activity_index, activity in enumerate(activities):
                        if not isinstance(activity, dict):
                            continue
                        activity_summaries.append(
                            {
                                "path": f"/itinerary/{day_index}/activities/{activity_index}",
                                "type": activity.get("type"),
                                "name": activity.get("position")
                                or activity.get("start")
                                or activity.get("end"),
                                "start_time": activity.get("start_time"),
                                "end_time": activity.get("end_time"),
                            }
                        )
                days.append(
                    {
                        "day": day_payload.get("day", day_index + 1),
                        "path": f"/itinerary/{day_index}",
                        "activities": activity_summaries,
                    }
                )
        return {
            "people_number": plan.get("people_number"),
            "start_city": plan.get("start_city"),
            "target_city": plan.get("target_city"),
            "itinerary": days,
        }

    def _build_patch_repair_prompt(
        self,
        *,
        original_plan: Dict[str, Any],
        edit_query: str,
        rejected_output: Any,
        error_message: str,
        failure_phase: str,
    ) -> str:
        try:
            rejected_json = json.dumps(rejected_output, ensure_ascii=False, indent=2)
        except TypeError:
            rejected_json = str(rejected_output)
        if len(rejected_json) > 12000:
            rejected_json = (
                rejected_json[:6000]
                + "\n...(truncated invalid output)...\n"
                + rejected_json[-3000:]
            )
        plan_summary_json = json.dumps(
            self._summarize_plan_for_repair(original_plan),
            ensure_ascii=False,
            indent=2,
        )
        return (
            "# PTE-R patch repair\n"
            "上一次 PTE-R 输出无法作为合法 patch ops 执行。不要调用工具；只输出修正后的 JSON 数组。\n\n"
            f"failure_phase: {failure_phase}\n"
            f"error: {error_message}\n\n"
            "# 原始 edit query\n"
            f"{edit_query}\n\n"
            "# 原始 plan 结构摘要\n"
            f"{plan_summary_json}\n\n"
            "# 上一次非法输出或 rejected ops\n"
            f"{rejected_json}\n\n"
            "# 必须遵守的 patch contract\n"
            "- 最终回答只能是 JSON array，不要解释文字，不要输出最终 plan。\n"
            "- 每个元素都必须是 object，并且必须包含 `op`。\n"
            "- 只允许 `op` 为 `replace`、`edit`、`delete`、`add_day`。\n"
            "- 除 `add_day` 外，每个 op 必须包含以 `/` 开头的 JSON Pointer `path`。\n"
            "- `replace/edit` 必须包含 `value`；`delete` 只删除 `path` 指向的字段或元素。\n"
            "- `add_day` 必须包含 `day` 和完整 `activities` 数组，且只能用于原计划中不存在的 day。\n"
            "- 不要输出裸 activity 或 transport 对象，例如不要直接输出 `{start,end,mode,...}`。\n"
            "- 如果要修改某个 activity 或 transport，必须输出它所在的精确 `path` 和完整 `value`。\n"
        )

    def _tool_result_content(self, tool_result: Dict[str, Any]) -> str:
        import json

        return json.dumps(
            self._compact_tool_result_for_context(tool_result),
            ensure_ascii=False,
        )

    async def edit_plan(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import time

        start_time = time.time()
        try:
            self.reset()
            self.original_plan_for_tools = copy.deepcopy(original_plan)
            self.current_plan = copy.deepcopy(original_plan)
            self.current_edit_request = dict(edit_request)
            self.current_original_query = dict(original_query)
            messages = self._build_planning_messages(
                edit_request, original_plan, original_query, system_prompt
            )
            ops, query_rounds = await self._generate_plan_ops_with_query(messages)
            self.metrics.execution_time_seconds = time.time() - start_time
            if self.infeasible_detection is not None:
                return infeasible_result(
                    detection=self.infeasible_detection,
                    conversation_log=self.get_execution_log(),
                    metrics={**self.metrics.to_dict(), **self.pter_repair_metrics},
                    framework_type=self.get_framework_type(),
                )

            self._add_step(
                "plan",
                f"Generated {len(ops)} atomic operations after {query_rounds} query rounds",
                metadata={
                    "ops": ops,
                    "ops_count": len(ops),
                    "query_rounds": query_rounds,
                    "planning_phase": "multi_round_llm_with_read_tools",
                },
            )
            execution_result = self._execute_plan_ops(original_plan, ops)
            self._add_step(
                "execute",
                f"Executed {len(ops)} operations",
                metadata={
                    "success": execution_result.success,
                    "deltas_count": len(execution_result.deltas),
                    "issues_count": len(execution_result.issues),
                    "execution_time_ms": execution_result.execution_time_ms,
                    "execution_phase": "pure_python_apply_ops",
                },
            )
            if not execution_result.success and self.patch_repair_retries > 0:
                error_messages = [
                    issue.message
                    for issue in execution_result.issues
                    if issue.severity.value == "error"
                ]
                self.pter_repair_metrics["pter_execution_repair_attempts"] += 1
                self.pter_repair_metrics["pter_repair_last_error"] = "\n".join(error_messages)
                self._add_step(
                    "execution_repair",
                    "Repairing PTE-R ops after patch execution failure",
                    metadata={
                        "error": self.pter_repair_metrics["pter_repair_last_error"],
                        "rejected_ops": ops,
                        "repair_attempt": self.pter_repair_metrics["pter_execution_repair_attempts"],
                    },
                )
                try:
                    repaired_content = await self._request_repaired_ops_without_tools(
                        messages,
                        original_plan=original_plan,
                        edit_query=str(edit_request.get("edit_query") or ""),
                        rejected_output=ops,
                        error_message=self.pter_repair_metrics["pter_repair_last_error"],
                        failure_phase="patch_execution",
                    )
                    repaired_ops = self._extract_patch_ops(repaired_content)
                    repaired_execution_result = self._execute_plan_ops(original_plan, repaired_ops)
                    self.generated_ops = repaired_ops
                    self._add_step(
                        "execution_repair_result",
                        f"Executed repaired PTE-R ops: success={repaired_execution_result.success}",
                        metadata={
                            "ops": repaired_ops,
                            "success": repaired_execution_result.success,
                            "deltas_count": len(repaired_execution_result.deltas),
                            "issues_count": len(repaired_execution_result.issues),
                        },
                    )
                    execution_result = repaired_execution_result
                    if execution_result.success:
                        self.pter_repair_metrics["pter_execution_repair_success"] = True
                    else:
                        retry_errors = [
                            issue.message
                            for issue in execution_result.issues
                            if issue.severity.value == "error"
                        ]
                        self.pter_repair_metrics["pter_repair_last_error"] = "\n".join(retry_errors)
                except Exception as repair_exc:
                    self.pter_repair_metrics["pter_repair_last_error"] = str(repair_exc)
                    error_messages.append(f"patch_repair_failed: {repair_exc}")
                    return {
                        "success": False,
                        "edited_plan": execution_result.plan,
                        "conversation_log": self.get_execution_log(),
                        "metrics": {**self.metrics.to_dict(), **self.pter_repair_metrics},
                        "framework_type": self.get_framework_type(),
                        "errors": error_messages,
                    }
            if execution_result.success:
                return {
                    "success": True,
                    "edited_plan": execution_result.plan,
                    "conversation_log": self.get_execution_log(),
                    "metrics": {**self.metrics.to_dict(), **self.pter_repair_metrics},
                    "framework_type": self.get_framework_type(),
                    "errors": [],
                }
            error_messages = [
                issue.message
                for issue in execution_result.issues
                if issue.severity.value == "error"
            ]
            return {
                "success": False,
                "edited_plan": execution_result.plan,
                "conversation_log": self.get_execution_log(),
                "metrics": {**self.metrics.to_dict(), **self.pter_repair_metrics},
                "framework_type": self.get_framework_type(),
                "errors": error_messages,
            }
        except Exception as exc:
            self.metrics.execution_time_seconds = time.time() - start_time
            return {
                "success": False,
                "edited_plan": None,
                "conversation_log": self.get_execution_log(),
                "metrics": {**self.metrics.to_dict(), **self.pter_repair_metrics},
                "framework_type": self.get_framework_type(),
                "errors": [str(exc)],
            }


class PTEREditFramework(EditFramework):
    """Standalone PTE-R runtime over the legacy framework core."""

    framework_name = "pter"

    def __init__(
        self,
        llm_client: Any,
        world_env: Any,
        *,
        tool_adapter: Optional[ChinaTravelToolAdapter] = None,
        max_steps: int = 30,
        max_tool_calls: int = 30,
        max_query_rounds: int = 20,
        max_conversation_messages: int = 18,
        max_tool_rows_in_context: int = 3,
        max_tool_value_chars: int = 240,
        max_completion_tokens: Optional[int] = None,
        patch_repair_retries: int = 1,
        prompt_ablation: str = "original",
        guard_retries: int = 0,
    ) -> None:
        self.guard_retries = int(guard_retries)
        self.prompt_ablation = prompt_ablation
        self.core = _StandalonePTERCore(
            framework_id="standalone_pter",
            llm_client=llm_client,
            world_env=ensure_session_world_env(world_env),
            tool_adapter=tool_adapter,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_query_rounds=max_query_rounds,
            max_conversation_messages=max_conversation_messages,
            max_tool_rows_in_context=max_tool_rows_in_context,
            max_tool_value_chars=max_tool_value_chars,
            max_completion_tokens=max_completion_tokens,
            patch_repair_retries=patch_repair_retries,
            prompt_ablation=prompt_ablation,
        )

    async def run(self, edit_input: EditInput) -> EditResult:
        origin_plan = require_origin_plan(edit_input.origin_plan)
        self.core.original_plan_for_tools = copy.deepcopy(origin_plan)
        self.core.current_plan = copy.deepcopy(origin_plan)
        self.core.current_edit_request = {"edit_query": edit_input.edit_query}
        self.core.current_original_query = {"origin_query_text": edit_input.origin_query_text}
        framework_result = await self.core.edit_plan(
            edit_request={"edit_query": edit_input.edit_query},
            original_plan=origin_plan,
            original_query={"origin_query_text": edit_input.origin_query_text},
            system_prompt=build_pter_system_prompt(self.core.tool_adapter.tool_flags(), prompt_ablation=self.prompt_ablation),
        )

        edited_plan = framework_result.get("edited_plan")
        if self.core.infeasible_detection is None and edited_plan is not None:
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
                    system_prompt=build_pter_system_prompt(
                        self.core.tool_adapter.tool_flags(),
                        prompt_ablation=self.prompt_ablation,
                    ),
                )
                edited_plan = framework_result.get("edited_plan")
                if self.core.infeasible_detection is None and edited_plan is not None:
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
            ops=list(self.core.generated_ops),
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
            },
            errors=framework_result.get("errors", []),
        )
