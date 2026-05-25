"""
PTE-R Framework (Plan-Then-Execute with Read tools).

Implements plan-then-execute with read-only tool querying:
- Planning phase: Multi-round LLM calls with read-only tools
- Execution phase: Pure Python, applies operations sequentially
- Outputs ops array after querying (not plan JSON like ReAct)
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .plan_patch_executor import execute_plan_patch_ops, validate_patch_ops
from edit_framework.tools.chinatravel_tools import ChinaTravelToolAdapter
from .base_framework import BaseLLMFramework
from utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_PTER_MAX_COMPLETION_TOKENS = 12000


class PTERFramework(BaseLLMFramework):
    """
    PTE-R (Plan-Then-Execute with Read tools) Framework.

    This framework allows tool querying but maintains plan-then-execute structure:
    - Planning phase: Multi-round LLM calls with read-only tools
    - Final output: Operations array (not final plan like ReAct)
    - Execution phase: Pure Python, applies operations sequentially

    Key difference from ReAct:
    - ReAct: Query → Modify plan → Query → Modify plan → ...
    - PTE-R: Query → Query → ... → Output all ops at once → Execute

    Research focus: Study the value of tool querying with plan separation.
    """

    def __init__(
        self,
        framework_id: str,
        llm_client,
        world_env,
        *,
        max_query_rounds: int = 20,
        max_conversation_messages: int = 18,
        max_tool_rows_in_context: int = 3,
        max_tool_value_chars: int = 240,
        max_completion_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            framework_id=framework_id,
            llm_client=llm_client,
            world_env=world_env,
            **kwargs,
        )
        self.max_query_rounds = max_query_rounds
        self.max_conversation_messages = max(4, max_conversation_messages)
        self.max_tool_rows_in_context = max(1, max_tool_rows_in_context)
        self.max_tool_value_chars = max(32, max_tool_value_chars)
        self.max_completion_tokens = max_completion_tokens or getattr(
            llm_client,
            "pter_max_completion_tokens",
            DEFAULT_PTER_MAX_COMPLETION_TOKENS,
        )

    def get_framework_type(self) -> str:
        """Return the framework type identifier."""
        return "pter"

    def _get_framework_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for PTE-R.

        PTE-R uses read-only tools for querying POI information.
        """
        return self._get_read_only_tools()

    def _get_read_only_tools(self) -> List[Dict[str, Any]]:
        """Get read-only tool definitions (same as ReAct)."""
        return ChinaTravelToolAdapter().read_only_tools()

    async def edit_plan(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute plan editing using PTE-R framework.

        Args:
            edit_request: The editing request from user
            original_plan: Original travel plan to edit
            original_query: Original query that generated the plan
            system_prompt: Optional system prompt override
            **kwargs: Additional framework-specific parameters

        Returns:
            Dictionary with editing results and metrics (same format as ReAct)
        """
        import time
        start_time = time.time()

        try:
            # Reset framework state
            self.reset()

            # Build planning messages
            messages = self._build_planning_messages(
                edit_request, original_plan, original_query, system_prompt
            )

            # Phase 1: PLAN (multi-round LLM calls with read-only tools)
            ops, query_rounds = await self._generate_plan_ops_with_query(messages)

            # Log plan step
            self._add_step(
                "plan",
                f"Generated {len(ops)} atomic operations after {query_rounds} query rounds",
                metadata={
                    "ops": ops,
                    "ops_count": len(ops),
                    "query_rounds": query_rounds,
                    "planning_phase": "multi_round_llm_with_read_tools"
                }
            )

            # Phase 2: EXECUTE (pure Python, no LLM)
            execution_result = self._execute_plan_ops(original_plan, ops)

            # Log execute step
            self._add_step(
                "execute",
                f"Executed {len(ops)} operations",
                metadata={
                    "success": execution_result.success,
                    "deltas_count": len(execution_result.deltas),
                    "issues_count": len(execution_result.issues),
                    "execution_time_ms": execution_result.execution_time_ms,
                    "execution_phase": "pure_python_apply_ops"
                }
            )

            # Calculate metrics
            self.metrics.execution_time_seconds = time.time() - start_time

            # Return result (same format as ReAct)
            if execution_result.success:
                return {
                    "success": True,
                    "edited_plan": execution_result.plan,
                    "conversation_log": self.get_execution_log(),
                    "metrics": self.metrics.to_dict(),
                    "framework_type": self.get_framework_type(),
                    "errors": [],
                }
            else:
                # Execution failed - return partial plan with errors
                error_messages = [
                    issue.message for issue in execution_result.issues
                    if issue.severity.value == "error"
                ]
                return {
                    "success": False,
                    "edited_plan": execution_result.plan,  # Partial plan
                    "conversation_log": self.get_execution_log(),
                    "metrics": self.metrics.to_dict(),
                    "framework_type": self.get_framework_type(),
                    "errors": error_messages,
                }

        except Exception as e:
            self.metrics.execution_time_seconds = time.time() - start_time
            logger.error(f"PTE-R framework execution failed: {e}")
            return {
                "success": False,
                "edited_plan": None,
                "conversation_log": self.get_execution_log(),
                "metrics": self.metrics.to_dict(),
                "framework_type": self.get_framework_type(),
                "errors": [str(e)],
            }

    def _build_planning_messages(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the planning phase messages."""
        from llm.pter_prompts import (
            PTER_SYSTEM_PROMPT,
            build_pter_user_prompt,
        )

        system_msg = {
            "role": "system",
            "content": system_prompt or PTER_SYSTEM_PROMPT
        }

        user_msg = {
            "role": "user",
            "content": build_pter_user_prompt(
                original_query, edit_request, original_plan
            )
        }

        self.conversation_history = [system_msg, user_msg]
        return self.conversation_history.copy()

    async def _generate_plan_ops_with_query(
        self,
        messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Generate operations with multi-round querying.

        This is the PLAN phase for PTE-R:
        - Multiple LLM calls with read-only tools
        - Final output: operations array (not final plan)

        Args:
            messages: Initial conversation messages

        Returns:
            Tuple of (operations list, number of query rounds)
        """
        query_rounds = 0
        tools = self._get_framework_tools()

        for iteration in range(self.max_query_rounds):
            query_rounds += 1

            request_messages = self._build_request_messages(messages)

            # Call LLM with tools
            response = await self.llm_client.chat_completion(
                messages=request_messages,
                tools=tools,
                temperature=0.3,
                max_tokens=self.max_completion_tokens,
            )

            # Check if LLM wants to use tools
            if isinstance(response, dict) and response.get("tool_calls"):
                # LLM wants to query - execute tools
                tool_calls = response["tool_calls"]

                # Add assistant message with tool calls
                messages.append(self._assistant_message_from_response(response, tool_calls=tool_calls))

                # Log query step
                self._add_step(
                    "query",
                    f"Round {query_rounds}: Executing {len(tool_calls)} tool calls",
                    tool_calls=tool_calls,
                    metadata={"round": query_rounds}
                )

                # Execute each tool call
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
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(
                                self._compact_tool_result_for_context(tool_result),
                                ensure_ascii=False,
                            ),
                        })
                        continue

                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    compact_tool_result = self._compact_tool_result_for_context(tool_result)

                    # Add tool result message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(compact_tool_result, ensure_ascii=False)
                    })

                logger.info(f"PTE-R round {query_rounds}: Executed {len(tool_calls)} tool calls")

                # Continue to next iteration for more querying or final output

            else:
                # LLM provided final response - should be operations array
                content = response if isinstance(response, str) else response.get("content", "")

                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": content
                })

                # Log final output
                logger.info(f"PTE-R round {query_rounds}: LLM provided final response")

                ops = self._extract_patch_ops(content)

                logger.info(f"PTE-R generated {len(ops)} operations after {query_rounds} rounds")
                return ops, query_rounds

        logger.warning(
            "PTE-R hit maximum query rounds; forcing a final non-tool response",
            extra={"max_query_rounds": self.max_query_rounds},
        )
        content = await self._request_final_ops_without_tools(
            messages,
            reason=(
                f"你已经进行了 {self.max_query_rounds} 轮查询。"
                "不要再调用任何工具，直接输出最终 patch ops JSON 数组。"
            ),
        )
        ops = self._extract_patch_ops(content)
        logger.info(
            f"PTE-R generated {len(ops)} operations after forced finalization at round {query_rounds}"
        )
        return ops, query_rounds

    def _parse_tool_arguments_safe(self, raw_arguments: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """Best-effort tool-argument parsing that never crashes the round."""
        if isinstance(raw_arguments, dict):
            return dict(raw_arguments), None

        if not isinstance(raw_arguments, str):
            return {}, f"Tool arguments must be a JSON object string, got {type(raw_arguments).__name__}."

        text = raw_arguments.strip()
        if not text:
            return {}, None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed, None
            return {}, "Tool arguments JSON must be an object."
        except json.JSONDecodeError as first_error:
            # Some providers prepend or append text; salvage the object segment if present.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed, None
                    return {}, "Recovered tool arguments JSON is not an object."
                except json.JSONDecodeError:
                    repaired = self._repair_tool_argument_json_placeholders(candidate)
                    if repaired != candidate:
                        try:
                            parsed = json.loads(repaired)
                            if isinstance(parsed, dict):
                                return parsed, None
                            return {}, "Repaired tool arguments JSON is not an object."
                        except json.JSONDecodeError:
                            pass
            return {}, f"Malformed tool arguments JSON: {first_error.msg}"

    def _extract_patch_ops(self, content: str) -> List[Dict[str, Any]]:
        """Parse and validate final patch ops without repair retries.

        PTE-R prompts require an ops array, but smaller models sometimes return
        a single op object. That is unambiguous, so normalize it here instead
        of failing the whole run.
        """
        try:
            ops = self._extract_json_array(content)
        except ValueError as array_error:
            normalized = self._strip_json_fences(content)
            if normalized.startswith("["):
                raise array_error
            try:
                parsed_object = self._extract_json_object(normalized)
            except ValueError:
                raise array_error
            ops = [parsed_object]
        validate_patch_ops(ops)
        return ops

    async def _request_final_ops_without_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        reason: str,
    ) -> str:
        """Force the model to stop querying and emit the final patch ops array."""
        repair_messages = self._build_request_messages(messages)
        repair_messages.append(
            {
                "role": "user",
                "content": (
                    reason
                    + "\n\n只输出 JSON 数组，不要输出解释。"
                    + "\n允许的形式："
                    + '\n- {"op":"replace","path":"...","value":...}'
                    + '\n- {"op":"edit","path":"...","value":...}'
                    + '\n- {"op":"delete","path":"..."}'
                    + '\n- {"op":"add_day","day":2,"activities":[...]}'
                ),
            }
        )
        response = await self.llm_client.chat_completion(
            messages=repair_messages,
            temperature=0.2,
            max_tokens=self.max_completion_tokens,
        )
        content = response if isinstance(response, str) else response.get("content", "")
        if content and content.strip():
            return content

        retry_messages = self._compact_final_ops_retry_messages(
            messages,
            reason=reason,
        )
        retry_response = await self.llm_client.chat_completion(
            messages=retry_messages,
            temperature=0.0,
            max_tokens=self.max_completion_tokens,
        )
        return (
            retry_response
            if isinstance(retry_response, str)
            else retry_response.get("content", "")
        )

    def _compact_final_ops_retry_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        reason: str,
    ) -> List[Dict[str, Any]]:
        base_messages = [message for message in messages[:2] if message.get("role") in {"system", "user"}]
        retry_instruction = (
            reason
            + "\n\n上一轮最终输出为空或仍尝试调用工具。现在禁止调用任何工具。"
            + "\n只输出 JSON 数组，不要输出解释，不要输出工具调用标记。"
            + "\n允许的形式："
            + '\n- {"op":"replace","path":"...","value":...}'
            + '\n- {"op":"edit","path":"...","value":...}'
            + '\n- {"op":"delete","path":"..."}'
            + '\n- {"op":"add_day","day":2,"activities":[...]}'
        )
        return [*base_messages, {"role": "user", "content": retry_instruction}]

    def _build_request_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep the stable prompt plus only the most recent query/tool turns."""
        return self._trim_messages_preserving_tool_pairs(
            messages,
            max_messages=self.max_conversation_messages,
            head_messages=2,
        )

    def _compact_tool_result_for_context(
        self,
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reduce tool payload size before feeding it back into the next LLM turn."""
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
        if "value" in tool_result:
            compact["value"] = self._truncate_jsonable(tool_result.get("value"))
        if "page" in tool_result:
            compact["page"] = tool_result.get("page")
        if "cursor_id" in tool_result:
            compact["cursor_id"] = tool_result.get("cursor_id")
        rows = tool_result.get("rows")
        if isinstance(rows, list):
            compact["rows"] = [
                self._truncate_jsonable(row) for row in rows[: self.max_tool_rows_in_context]
            ]
            compact["rows_truncated"] = len(rows) > self.max_tool_rows_in_context
        return compact

    def _compact_semantic_tool_result(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            return {
                "applied_ops": self._truncate_jsonable(tool_result.get("applied_ops", [])),
                "delta": self._truncate_jsonable(tool_result.get("delta", [])),
                "issues": self._truncate_jsonable(tool_result.get("issues", [])),
                "pending_repairs": self._truncate_jsonable(tool_result.get("pending_repairs", {})),
                "details": self._truncate_jsonable(tool_result.get("details", {})),
            }
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

    def _extract_json_array(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON array from LLM response content.

        Args:
            content: LLM response content

        Returns:
            Parsed JSON array of operations
        """
        import re

        if not content or not content.strip():
            raise ValueError("Empty response content")

        original_content = content
        content = content.strip()

        # Try 1: Direct JSON parse (fastest path)
        if content.startswith('['):
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.debug(f"Direct array parse failed: {e}")

        # Try 2: Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()

        # Try 3: Look for JSON array after any explanatory text
        # Find the opening bracket
        start_idx = content.find('[')
        if start_idx == -1:
            raise ValueError(f"No JSON array found in response: {content[:200]}...")

        # Extract JSON array with proper bracket matching
        bracket_count = 0
        end_idx = -1
        in_string = False
        escape_next = False

        for i in range(start_idx, len(content)):
            char = content[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break

        if end_idx == -1:
            raise ValueError(f"Could not find matching closing bracket in: {content[start_idx:start_idx+200]}...")

        json_str = content[start_idx:end_idx]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try cleaning up common issues
            cleaned_json = json_str
            cleaned_json = re.sub(r'//.*?$', '', cleaned_json, flags=re.MULTILINE)
            cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)
            cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)

            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Last resort: try json5 if available
                try:
                    import json5
                    result = json5.loads(cleaned_json)
                    logger.warning("Successfully parsed JSON array using json5 (lenient parser)")
                    return result
                except ImportError:
                    pass
                except Exception:
                    pass

                raise ValueError(f"Could not parse JSON array after cleanup: {e.msg}")

    def _extract_json_object(self, content: str) -> Dict[str, Any]:
        """Extract one JSON object from direct, fenced, or explanatory output."""
        import re

        if not content or not content.strip():
            raise ValueError("Empty response content")

        content = self._strip_json_fences(content)

        if content.startswith("{"):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        start_idx = content.find("{")
        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {content[:200]}...")

        brace_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        for i in range(start_idx, len(content)):
            char = content[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
        if end_idx == -1:
            raise ValueError(
                f"Could not find matching closing brace in: {content[start_idx:start_idx+200]}..."
            )

        json_str = content[start_idx:end_idx]
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
                except Exception:
                    raise ValueError(f"Could not parse JSON object after cleanup: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Extracted JSON value is not an object.")
        return parsed

    @staticmethod
    def _strip_json_fences(content: str) -> str:
        import re

        content = "" if content is None else str(content)
        content = re.sub(r"```json\s*", "", content.strip())
        content = re.sub(r"```\s*", "", content).strip()
        return content

    def _execute_plan_ops(
        self,
        plan: Dict[str, Any],
        ops: List[Dict[str, Any]]
    ):
        """
        Execute a list of atomic operations.

        This is the EXECUTE phase - pure Python, no LLM.

        Args:
            plan: Original plan
            ops: List of operations to apply

        Returns:
            ExecutionResult from direct patch application
        """
        logger.info(f"Executing {len(ops)} operations using direct patch executor")
        result = execute_plan_patch_ops(plan, ops)

        logger.info(
            f"Execution completed: success={result.success}, "
            f"deltas={len(result.deltas)}, issues={len(result.issues)}"
        )

        return result

    def _execute_framework_step(
        self,
        messages: List[Dict[str, Any]],
        step_number: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single step of the framework logic.

        Note: PTE-R doesn't use this method as it has a custom planning loop.
        This is required by BaseLLMFramework but not actively used.
        """
        # Not used in PTE-R, but required by base class
        return {
            "next_action": "finish",
            "messages": messages,
        }
