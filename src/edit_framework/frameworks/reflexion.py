"""Standalone Reflexion edit baseline."""

from __future__ import annotations
import copy
import json
import logging
import re
from typing import Any, Dict, Optional

from agents.frameworks.pter_framework import PTERFramework
from edit_framework.adapters.plan_adapter import (
    require_chinatravel_plan,
    require_origin_plan,
    validate_chinatravel_plan,
)
from edit_framework.base import EditFramework, EditInput, EditResult
from edit_framework.error_handling import (
    classify_terminal_error_categories,
    framework_error_handling_metadata,
)
from edit_framework.guard import apply_candidate_guard, build_guard_retry_edit_query
from edit_framework.prompts.reflexion import (
    REFLEXION_SYSTEM_PROMPT,
    build_reflection_generation_prompt,
    build_reflection_retry_prompt,
    build_reflexion_system_prompt,
    build_reflexion_user_prompt,
)
from edit_framework.runtime_tools.types import ExposureMode
from edit_framework.tools.chinatravel_tools import ChinaTravelToolAdapter
from edit_framework.world_env import ensure_session_world_env
from utils.chinatravel_plan import normalize_loose_chinatravel_plan

logger = logging.getLogger(__name__)


class _StandaloneReflexionCore(PTERFramework):
    """Reflexion core with read-only tools and full-plan validation retries."""

    def __init__(
        self,
        framework_id: str,
        llm_client: Any,
        world_env: Any,
        tool_adapter: Optional[ChinaTravelToolAdapter] = None,
        max_reflections: int = 2,
        reflection_strategy: str = "reflexion",
        context_prompt: bool = False,
        database_prompt: bool = False,
        prompt_ablation: str = "original",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            framework_id=framework_id,
            llm_client=llm_client,
            world_env=world_env,
            **kwargs,
        )
        self.tool_adapter = tool_adapter or ChinaTravelToolAdapter(
            framework_name="reflexion",
            exposure_mode=ExposureMode.HYBRID.value,
        )
        self.generated_ops: list[Dict[str, Any]] = []
        self.max_reflections = max_reflections
        self.current_plan: Dict[str, Any] | None = None
        self.original_plan_for_tools: Dict[str, Any] | None = None
        self.current_edit_request: Dict[str, Any] = {}
        self.current_original_query: Dict[str, Any] = {}
        self.last_lifted_constraints: list[Dict[str, Any]] = []
        self.last_intent_anchors: list[Dict[str, Any]] = []
        self.ct_notepad: Dict[str, list[str]] = {}
        self.prompt_ablation = prompt_ablation
        if reflection_strategy not in {"reflexion", "last_attempt"}:
            raise ValueError("reflection_strategy must be 'reflexion' or 'last_attempt'")
        self.reflection_strategy = reflection_strategy
        self.reflection_texts: list[str] = []
        self.retry_feedback_source = (
            "verbal_reflection_memory"
            if reflection_strategy == "reflexion"
            else "execution_feedback_direct"
        )
        self.context_prompt = bool(context_prompt)
        self.database_prompt = bool(database_prompt)
        self.current_metadata: Dict[str, Any] = {}

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
        self.runtime_capability_state = None
        self.reflection_texts = []
        self.current_metadata = {}
        world_env_reset = getattr(self.world_env, "reset", None)
        if callable(world_env_reset):
            world_env_reset()

    def _build_planning_messages(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        system_msg = {
            "role": "system",
            "content": system_prompt or build_reflexion_system_prompt(self.tool_adapter.tool_flags(), prompt_ablation=self.prompt_ablation),
        }
        user_msg = {
            "role": "user",
            "content": build_reflexion_user_prompt(
                origin_query_text=original_query.get("origin_query_text", ""),
                edit_query=edit_request.get("edit_query", ""),
                origin_plan=original_plan,
                metadata=metadata,
                context_prompt=self.context_prompt,
                database_prompt=self.database_prompt,
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

    async def _generate_edited_plan_with_query(
        self,
        messages: list[Dict[str, Any]],
    ) -> tuple[str, int]:
        """Generate a complete edited plan response after optional tool-assisted querying."""
        query_rounds = 0
        tools = self._get_framework_tools()

        for _iteration in range(self.max_query_rounds):
            query_rounds += 1
            request_messages = self._build_request_messages(messages)
            response = await self.llm_client.chat_completion(
                messages=request_messages,
                tools=tools,
                temperature=0.3,
                max_tokens=8000,
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
                            "content": self._compact_json_for_context(tool_result),
                        }
                    )
                continue

            content = response if isinstance(response, str) else response.get("content", "")
            logger.info("Reflexion round %s: LLM provided full-plan final response", query_rounds)
            return content, query_rounds

        logger.warning(
            "Reflexion hit maximum query rounds; forcing a final non-tool full-plan response",
            extra={"max_query_rounds": self.max_query_rounds},
        )
        content = await self._request_final_plan_without_tools(
            messages,
            reason=(
                f"你已经进行了 {self.max_query_rounds} 轮查询。"
                "不要再调用任何工具，直接输出最终完整 edited ChinaTravel plan JSON object。"
            ),
        )
        return content, query_rounds

    async def _request_final_plan_without_tools(
        self,
        messages: list[Dict[str, Any]],
        *,
        reason: str,
    ) -> str:
        """Force the model to stop querying and emit the final full edited plan."""
        request_messages = self._build_request_messages(messages)
        request_messages.append(
            {
                "role": "user",
                "content": (
                    reason
                    + "\n\n只输出一个 JSON object，不要输出解释、markdown、数组、diff 或操作列表。"
                    + "\n该 JSON object 必须是完整 ChinaTravel plan，包含 people_number/start_city/target_city/itinerary。"
                ),
            }
        )
        response = await self.llm_client.chat_completion(
            messages=request_messages,
            temperature=0.2,
            max_tokens=8000,
        )
        content = response if isinstance(response, str) else response.get("content", "")
        if content and content.strip():
            return content

        retry_messages = self._compact_final_plan_retry_messages(messages, reason=reason)
        retry_response = await self.llm_client.chat_completion(
            messages=retry_messages,
            temperature=0.0,
            max_tokens=8000,
        )
        return (
            retry_response
            if isinstance(retry_response, str)
            else retry_response.get("content", "")
        )

    def _compact_final_plan_retry_messages(
        self,
        messages: list[Dict[str, Any]],
        *,
        reason: str,
    ) -> list[Dict[str, Any]]:
        base_messages = [
            message for message in messages[:2] if message.get("role") in {"system", "user"}
        ]
        retry_instruction = (
            reason
            + "\n\n上一轮最终输出为空或仍尝试调用工具。现在禁止调用任何工具。"
            + "\n只输出一个完整 edited ChinaTravel plan JSON object，不要输出解释。"
        )
        return [*base_messages, {"role": "user", "content": retry_instruction}]

    def _compact_json_for_context(self, payload: Dict[str, Any]) -> str:
        return json.dumps(
            self._compact_tool_result_for_context(payload),
            ensure_ascii=False,
        )

    def _extract_full_plan_candidate(self, content: str) -> Dict[str, Any]:
        """Parse a full-plan JSON object without enforcing canonical validity yet."""
        if not content or not content.strip():
            raise ValueError("Empty response content")

        text = re.sub(r"```json\s*", "", content.strip())
        text = re.sub(r"```\s*", "", text).strip()
        json_patterns = [
            r'\{[\s\n]*"people_number"',
            r'\{[\s\n]*"itinerary"',
        ]
        best_start_idx = len(text)
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match and match.start() < best_start_idx:
                best_start_idx = match.start()
        start_idx = best_start_idx if best_start_idx < len(text) else text.find("{")
        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {text[:200]}...")

        brace_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        for index in range(start_idx, len(text)):
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
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = index + 1
                    break
        if end_idx == -1:
            raise ValueError(f"Could not find matching closing brace in: {text[start_idx:start_idx + 200]}...")

        json_str = text[start_idx:end_idx]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            cleaned_json = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
            cleaned_json = re.sub(r"/\*.*?\*/", "", cleaned_json, flags=re.DOTALL)
            cleaned_json = re.sub(r",(\s*[}\]])", r"\1", cleaned_json)
            try:
                parsed = json.loads(cleaned_json)
            except json.JSONDecodeError:
                try:
                    import json5

                    parsed = json5.loads(cleaned_json)
                except Exception as json5_exc:
                    raise ValueError(f"Could not parse JSON after cleanup: {exc.msg}") from json5_exc

        if not isinstance(parsed, dict):
            raise ValueError("Final response JSON must be an object representing edited_plan")
        normalized = normalize_loose_chinatravel_plan(parsed)
        if not isinstance(normalized, dict):
            raise ValueError("Final response JSON must be an object representing edited_plan")
        return normalized

    async def edit_plan(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import time

        start_time = time.time()
        attempt_count = 0
        reflection_rounds = 0
        execution_failures = 0
        plan_parse_failures = 0
        validation_failures = 0
        latest_errors: list[str] = []
        latest_plan: Optional[Dict[str, Any]] = None

        try:
            self.reset()
            self.original_plan_for_tools = copy.deepcopy(original_plan)
            self.current_plan = copy.deepcopy(original_plan)
            self.current_edit_request = dict(edit_request)
            self.current_original_query = dict(original_query)
            self.current_metadata = dict(metadata or {})
            messages = self._build_planning_messages(
                edit_request,
                original_plan,
                original_query,
                system_prompt,
                metadata=self.current_metadata,
            )

            while attempt_count < self.max_reflections + 1:
                attempt_count += 1
                self.generated_ops = []
                try:
                    plan_content, query_rounds = await self._generate_edited_plan_with_query(messages)
                    edited_plan = self._extract_full_plan_candidate(plan_content)
                    latest_plan = edited_plan
                    self.current_plan = copy.deepcopy(edited_plan)
                    validation_issue_messages = validate_chinatravel_plan(edited_plan)
                    parse_issue_messages: list[str] = []
                except Exception as exc:
                    query_rounds = 0
                    plan_parse_failures += 1
                    validation_issue_messages = []
                    parse_issue_messages = [str(exc)]

                if validation_issue_messages:
                    validation_failures += 1

                self._add_step(
                    "plan",
                    f"Attempt {attempt_count}: generated full edited plan after {query_rounds} query rounds",
                    metadata={
                        "attempt": attempt_count,
                        "success": not parse_issue_messages and not validation_issue_messages,
                        "output_form": "full_plan",
                        "query_rounds": query_rounds,
                        "parse_issue_count": len(parse_issue_messages),
                        "validation_issue_count": len(validation_issue_messages),
                    },
                )

                latest_errors = parse_issue_messages + validation_issue_messages
                if not latest_errors:
                    self.metrics.execution_time_seconds = time.time() - start_time
                    metrics = self.metrics.to_dict()
                    metrics.update(
                        {
                            "attempt_count": attempt_count,
                            "reflection_rounds": reflection_rounds,
                            "execution_failures": execution_failures,
                            "plan_parse_failures": plan_parse_failures,
                            "validation_failures": validation_failures,
                            "output_form": "full_plan",
                            "final_contract": "edited_plan",
                            **self._reflection_metrics(),
                        }
                    )
                    return {
                        "success": True,
                        "edited_plan": latest_plan,
                        "conversation_log": self.get_execution_log(),
                        "metrics": metrics,
                        "framework_type": self.get_framework_type(),
                        "errors": [],
                    }

                if attempt_count >= self.max_reflections + 1:
                    break

                reflection_rounds += 1
                reflection_text = None
                if self.reflection_strategy == "reflexion":
                    reflection_text = await self._generate_reflection_text(
                        edit_query=edit_request.get("edit_query", ""),
                        feedback_items=latest_errors,
                    )
                    self.reflection_texts.append(reflection_text)
                reflection_prompt = build_reflection_retry_prompt(
                    edit_query=edit_request.get("edit_query", ""),
                    feedback_items=latest_errors,
                    reflection_text=reflection_text,
                )
                messages.append({"role": "user", "content": reflection_prompt})
                self._add_step(
                    "reflection",
                    f"Reflection round {reflection_rounds}: retrying with {self.retry_feedback_source}",
                    metadata={
                        "attempt": attempt_count,
                        "reflection_round": reflection_rounds,
                        "feedback": list(latest_errors),
                        "reflection_strategy": self.reflection_strategy,
                        "reflection_text": reflection_text,
                        "retry_feedback_source": self.retry_feedback_source,
                    },
                )

            self.metrics.execution_time_seconds = time.time() - start_time
            metrics = self.metrics.to_dict()
            metrics.update(
                {
                    "attempt_count": attempt_count,
                    "reflection_rounds": reflection_rounds,
                    "execution_failures": execution_failures,
                    "plan_parse_failures": plan_parse_failures,
                    "validation_failures": validation_failures,
                    "output_form": "full_plan",
                    "final_contract": "edited_plan",
                    **self._reflection_metrics(),
                }
            )
            return {
                "success": False,
                "edited_plan": latest_plan,
                "conversation_log": self.get_execution_log(),
                "metrics": metrics,
                "framework_type": self.get_framework_type(),
                "errors": latest_errors or ["Reflexion exhausted retries without a valid plan"],
            }
        except Exception as exc:
            self.metrics.execution_time_seconds = time.time() - start_time
            metrics = self.metrics.to_dict()
            metrics.update(
                {
                    "attempt_count": attempt_count,
                    "reflection_rounds": reflection_rounds,
                    "execution_failures": execution_failures,
                    "plan_parse_failures": plan_parse_failures,
                    "validation_failures": validation_failures,
                    "output_form": "full_plan",
                    "final_contract": "edited_plan",
                    **self._reflection_metrics(),
                }
            )
            return {
                "success": False,
                "edited_plan": latest_plan,
                "conversation_log": self.get_execution_log(),
                "metrics": metrics,
                "framework_type": self.get_framework_type(),
                "errors": latest_errors + [str(exc)] if latest_errors else [str(exc)],
            }

    def get_framework_type(self) -> str:
        return "reflexion"

    def _reflection_metrics(self) -> Dict[str, Any]:
        return {
            "reflection_strategy": self.reflection_strategy,
            "reflection_texts": list(self.reflection_texts),
            "retry_feedback_source": self.retry_feedback_source,
            "reflection_memory_scope": "per_case",
            "has_explicit_reflection_memory": self.reflection_strategy == "reflexion",
            "has_last_attempt_retry": self.reflection_strategy == "last_attempt",
        }

    async def _generate_reflection_text(
        self,
        *,
        edit_query: str,
        feedback_items: list[str],
    ) -> str:
        reflection_prompt = build_reflection_generation_prompt(
            edit_query=edit_query,
            feedback_items=feedback_items,
        )
        response = await self.llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 Reflexion agent 的 verbal reflection generator。"
                        "只生成用于下一次尝试的自然语言记忆，不输出 JSON。"
                    ),
                },
                {"role": "user", "content": reflection_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = response if isinstance(response, str) else response.get("content", "")
        return content.strip() or "上一轮完整计划解析或校验失败；下一轮必须根据反馈做最小修复，并保留无关计划。"


class ReflexionEditFramework(EditFramework):
    """Standalone Reflexion runtime that emits complete edited plans."""

    framework_name = "reflexion"

    def __init__(
        self,
        llm_client: Any,
        world_env: Any,
        *,
        tool_adapter: Optional[ChinaTravelToolAdapter] = None,
        max_steps: int = 30,
        max_tool_calls: int = 30,
        max_reflections: int = 2,
        reflection_strategy: str = "reflexion",
        context_prompt: bool = False,
        database_prompt: bool = False,
        prompt_ablation: str = "original",
        guard_retries: int = 0,
    ) -> None:
        self.guard_retries = int(guard_retries)
        self.prompt_ablation = prompt_ablation
        self.core = _StandaloneReflexionCore(
            framework_id="standalone_reflexion",
            llm_client=llm_client,
            world_env=ensure_session_world_env(world_env),
            tool_adapter=tool_adapter,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_reflections=max_reflections,
            reflection_strategy=reflection_strategy,
            context_prompt=context_prompt,
            database_prompt=database_prompt,
            prompt_ablation=prompt_ablation,
        )

    async def run(self, edit_input: EditInput) -> EditResult:
        origin_plan = require_origin_plan(edit_input.origin_plan)
        self.core.original_plan_for_tools = copy.deepcopy(origin_plan)
        self.core.current_plan = copy.deepcopy(origin_plan)
        self.core.current_edit_request = {"edit_query": edit_input.edit_query}
        self.core.current_original_query = {"origin_query_text": edit_input.origin_query_text}
        self.core.current_metadata = dict(edit_input.metadata)
        framework_result = await self.core.edit_plan(
            edit_request={"edit_query": edit_input.edit_query},
            original_plan=origin_plan,
            original_query={"origin_query_text": edit_input.origin_query_text},
            system_prompt=build_reflexion_system_prompt(self.core.tool_adapter.tool_flags(), prompt_ablation=self.prompt_ablation),
            metadata=edit_input.metadata,
        )

        edited_plan = framework_result.get("edited_plan")
        if edited_plan is not None:
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
                system_prompt=build_reflexion_system_prompt(
                    self.core.tool_adapter.tool_flags(),
                    prompt_ablation=self.prompt_ablation,
                ),
                metadata=edit_input.metadata,
            )
            edited_plan = framework_result.get("edited_plan")
            if edited_plan is not None:
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
            ops=list(self.core.generated_ops),
            trace=framework_result.get("conversation_log", []),
            metrics={
                **framework_result.get("metrics", {}),
                **framework_error_handling_metadata(
                    self.framework_name,
                    max_steps=self.core.max_steps,
                    max_tool_calls=self.core.max_tool_calls,
                    max_reflections=self.core.max_reflections,
                ),
                "terminal_error_categories": classify_terminal_error_categories(
                    framework_result.get("errors", []),
                    metrics=framework_result.get("metrics", {}),
                )
                if not framework_result.get("success", False)
                else [],
                "tool_flags": self.core.tool_adapter.tool_flags(),
                "reflection_strategy": self.core.reflection_strategy,
                "reflection_texts": list(self.core.reflection_texts),
                "retry_feedback_source": self.core.retry_feedback_source,
                "reflection_memory_scope": "per_case",
                "prompt_modes": {
                    "context_prompt": self.core.context_prompt,
                    "database_prompt": self.core.database_prompt,
                },
            },
            errors=framework_result.get("errors", []),
        )
