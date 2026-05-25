"""
ReAct Framework Implementation.

Implements the ReAct (Reasoning and Acting) pattern where the LLM:
1. Thinks about the current situation
2. Takes action (usually calling tools)
3. Observes the results
4. Repeats until completion

This framework supports both modern OpenAI function calling and traditional tool calling approaches.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from .base_framework import BaseLLMFramework
from utils.logging import get_logger

logger = get_logger(__name__)


class ReactFramework(BaseLLMFramework):
    """
    ReAct (Reasoning and Acting) Framework implementation.

    This framework follows the classic Thought-Action-Observation cycle:
    - Thought: LLM reasons about the current state and what to do next
    - Action: LLM decides and executes actions (usually tool calls)
    - Observation: LLM observes the results of its actions
    - Repeat until the task is complete

    The framework supports both:
    1. Modern OpenAI-style function calling
    2. Traditional command-based tool interaction
    """

    def __init__(
        self,
        framework_id: str,
        llm_client,
        world_env,
        use_function_calling: bool = True,
        include_thoughts: bool = False,
        max_completion_tokens: Optional[int] = None,
        json_repair_max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize ReAct framework.

        Args:
            framework_id: Unique identifier for this framework instance
            llm_client: LLM client for making API calls
            world_env: Environment for tool execution
            use_function_calling: Whether to use OpenAI function calling (True) or text commands (False)
            include_thoughts: Whether to include explicit thought steps in the conversation
            **kwargs: Additional framework parameters
        """
        super().__init__(framework_id, llm_client, world_env, **kwargs)
        self.use_function_calling = use_function_calling
        self.include_thoughts = include_thoughts
        self.max_completion_tokens = max_completion_tokens or getattr(
            llm_client, "react_max_completion_tokens", 12000
        )
        self.json_repair_max_tokens = json_repair_max_tokens or getattr(
            llm_client, "react_json_repair_max_tokens", 8000
        )
        self.current_plan = None  # Maintain current plan state for incremental edits

    def get_framework_type(self) -> str:
        """Return the framework type identifier."""
        return "react"

    async def edit_plan(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute plan editing using the ReAct framework.

        Args:
            edit_request: The editing request from user
            original_plan: Original travel plan to edit
            original_query: Original query that generated the plan
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters

        Returns:
            Dictionary with editing results and metrics
        """
        import time
        start_time = time.time()

        try:
            # Reset framework state
            self.reset()

            # Initialize current plan for incremental edits
            import copy
            self.current_plan = copy.deepcopy(original_plan)

            # Build initial conversation
            messages = self._build_initial_messages(
                edit_request, original_plan, original_query, system_prompt
            )

            # Get framework tools (query + execution atoms)
            tools = self._get_framework_tools()

            # Execute ReAct loop
            final_result = await self._execute_react_loop(
                messages, tools, edit_request, original_plan
            )

            # Calculate execution time
            self.metrics.execution_time_seconds = time.time() - start_time

            # Return results
            # If execution atoms were used, current_plan is already updated
            # Otherwise, use the final_result from LLM
            edited_plan = self.current_plan if self.current_plan is not None else final_result

            return {
                "success": True,
                "edited_plan": edited_plan,
                "conversation_log": self.get_execution_log(),
                "metrics": self.metrics.to_dict(),
                "framework_type": self.get_framework_type(),
                "errors": [],
            }

        except Exception as e:
            self.metrics.execution_time_seconds = time.time() - start_time
            logger.error(f"ReAct framework execution failed: {e}")

            return {
                "success": False,
                "edited_plan": None,
                "conversation_log": self.get_execution_log(),
                "metrics": self.metrics.to_dict(),
                "framework_type": self.get_framework_type(),
                "errors": [str(e)],
            }

    def _build_initial_messages(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the initial conversation messages."""
        from llm.pure_llm_baseline_prompts import (
            PURE_LLM_EDIT_SYSTEM_PROMPT,
            build_user_prompt,
        )

        system_msg = {
            "role": "system",
            "content": system_prompt or PURE_LLM_EDIT_SYSTEM_PROMPT
        }

        # Add ReAct-specific instructions if needed
        if self.include_thoughts and self.use_function_calling:
            system_msg["content"] += "\n\n## ReAct Instructions\n\nPlease follow the ReAct pattern:\n1. Think about what you need to do\n2. Use tools to gather information\n3. Based on the results, continue thinking and acting\n4. When you have all information needed, provide the final edited plan as JSON"
        elif not self.use_function_calling:
            system_msg["content"] += "\n\n## Tool Usage Instructions\n\nWhen you need to use a tool, format your action as: `Action[tool_name(arg1=value1, arg2=value2)]`"

        user_msg = {
            "role": "user",
            "content": build_user_prompt(original_query, edit_request, original_plan)
        }

        self.conversation_history = [system_msg, user_msg]
        return self.conversation_history.copy()

    async def _execute_react_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the main ReAct reasoning loop."""
        step_number = 0

        while step_number < self.max_steps and self.metrics.tool_calls_made < self.max_tool_calls:
            step_number += 1

            # Execute one step of the framework
            step_result = await self._execute_framework_step(
                messages, step_number, {
                    "tools": tools,
                    "edit_request": edit_request,
                    "original_plan": original_plan,
                    "use_function_calling": self.use_function_calling,
                    "include_thoughts": self.include_thoughts,
                }
            )

            # Update messages with step results
            messages = step_result["messages"]

            # Check if we should continue
            if step_result["next_action"] == "finish":
                return step_result.get("final_result", {})
            elif step_result["next_action"] == "error":
                raise Exception(f"ReAct step failed: {step_result.get('error', 'Unknown error')}")

            # Continue to next step

        final_result = await self._attempt_budget_forced_finalize(messages)
        if final_result is not None:
            return final_result

        raise Exception(f"ReAct framework exceeded maximum steps ({self.max_steps}) or tool calls ({self.max_tool_calls})")

    async def _attempt_budget_forced_finalize(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        """Request one final no-tool completion before declaring budget exhaustion."""

        self.metrics.budget_finalize_attempted = True
        self._add_step(
            "budget_finalize",
            "Reached ReAct budget; requesting final JSON without further tool calls.",
            metadata={"step_type": "budget_finalize"},
        )

        finalize_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "不要再调用任何工具。"
                    "请基于目前已经拿到的信息，立即输出最终完整的 ChinaTravel 计划 JSON。"
                    "只输出 JSON，不要解释。"
                ),
            }
        ]
        request_builder = getattr(self, "_build_request_messages", None)
        request_messages = (
            request_builder(finalize_messages)
            if callable(request_builder)
            else finalize_messages
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=request_messages,
                temperature=0.2,
                max_tokens=self.max_completion_tokens,
            )
            step_result = await self._process_final_response(
                response,
                finalize_messages,
                len(self.steps) + 1,
            )
            if step_result.get("next_action") == "finish":
                self.metrics.budget_finalize_succeeded = True
                return step_result.get("final_result", {})
        except Exception as exc:
            logger.warning("Budget-forced finalize failed: %s", exc)

        return None

    async def _execute_framework_step(
        self,
        messages: List[Dict[str, Any]],
        step_number: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single ReAct step.

        Args:
            messages: Current conversation messages
            step_number: Current step number
            context: Execution context

        Returns:
            Step execution result dictionary
        """
        tools = context["tools"]
        use_function_calling = context["use_function_calling"]
        include_thoughts = context["include_thoughts"]

        if use_function_calling:
            return await self._execute_function_calling_step(messages, tools, step_number, context)
        else:
            return await self._execute_text_based_step(messages, tools, step_number, context)

    async def _execute_function_calling_step(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        step_number: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step using OpenAI function calling."""
        import asyncio

        # Call LLM with tools
        response = await self.llm_client.chat_completion(
            messages=messages,
            tools=tools,
            temperature=0.7,
            max_tokens=self.max_completion_tokens,
        )

        # Add thought step if enabled
        if self.include_thoughts:
            self._add_step(
                "thought",
                "Analyzing current situation and deciding next action...",
                metadata={"step_type": "reasoning"}
            )

        # Process response
        if isinstance(response, dict) and response.get("tool_calls"):
            # Handle tool calls
            return self._process_tool_calls(response, messages, step_number)
        else:
            # Final response
            return await self._process_final_response(response, messages, step_number)

    async def _execute_text_based_step(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        step_number: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a step using text-based commands."""
        import asyncio

        # Call LLM without tools
        response = await self.llm_client.chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=self.max_completion_tokens,
        )

        response_text = response if isinstance(response, str) else response.get("content", "")

        # Check if this is a tool call or final response
        if self._is_tool_call(response_text):
            return self._process_text_tool_call(response_text, messages, step_number)
        else:
            return await self._process_final_response(response, messages, step_number)

    def _process_tool_calls(
        self,
        response: Dict[str, Any],
        messages: List[Dict[str, Any]],
        step_number: int
    ) -> Dict[str, Any]:
        """Process tool calls from function calling response."""
        # Add assistant message with tool calls
        messages.append(self._assistant_message_from_response(response))

        # Log action step
        self._add_step(
            "action",
            response.get("content", "Making tool calls..."),
            tool_calls=response["tool_calls"],
            metadata={"response_format": "function_calling"}
        )

        # Execute tool calls
        tool_results = []
        for tool_call in response["tool_calls"]:
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
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
                tool_results.append({
                    "tool_name": tool_name,
                    "tool_args": {},
                    "result": tool_result
                })
                continue

            # Execute tool
            tool_result = self._execute_tool(tool_name, tool_args)

            # Add tool result message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

            tool_results.append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": tool_result
            })

        # Log observation step
        self._add_step(
            "observation",
            f"Executed {len(response['tool_calls'])} tool calls",
            tool_results=tool_results,
            metadata={"step_type": "tool_execution"}
        )

        return {
            "next_action": "continue",
            "messages": messages,
            "tool_results": tool_results,
        }

    def _parse_tool_arguments_safe(self, raw_arguments: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """Best-effort parsing for function-call arguments to avoid hard loop aborts."""
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

    def _process_text_tool_call(
        self,
        response_text: str,
        messages: List[Dict[str, Any]],
        step_number: int
    ) -> Dict[str, Any]:
        """Process text-based tool call."""
        # Parse the action from text
        action_match = self._parse_action_from_text(response_text)
        if not action_match:
            raise Exception(f"Could not parse action from: {response_text}")

        # Log action step
        self._add_step(
            "action",
            response_text,
            metadata={"response_format": "text_based"}
        )

        # Execute the action
        tool_name, tool_args = action_match
        tool_result = self._execute_tool(tool_name, tool_args)

        # Add observation
        observation_text = f"Result of {tool_name}: {tool_result.get('data', tool_result)}"
        self._add_step(
            "observation",
            observation_text,
            tool_results=[{"tool_name": tool_name, "tool_args": tool_args, "result": tool_result}],
            metadata={"step_type": "tool_execution"}
        )

        # Add assistant message and tool result to conversation
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": observation_text})

        return {
            "next_action": "continue",
            "messages": messages,
            "tool_results": [{"tool_name": tool_name, "tool_args": tool_args, "result": tool_result}],
        }

    async def _process_final_response(
        self,
        response,
        messages: List[Dict[str, Any]],
        step_number: int
    ) -> Dict[str, Any]:
        """Process the final response with the edited plan."""
        response_text = response if isinstance(response, str) else response.get("content", "")

        # Log final step
        self._add_step(
            "final_response",
            response_text,
            metadata={"step_type": "plan_completion"}
        )

        # Add final message to conversation
        messages.append({
            "role": "assistant",
            "content": response_text
        })

        # Extract JSON plan
        try:
            edited_plan = self._extract_json_plan(response_text)
            return {
                "next_action": "finish",
                "messages": messages,
                "final_result": edited_plan,
            }
        except Exception as e:
            repaired_plan = await self._attempt_json_repair(response_text, str(e))
            if repaired_plan is not None:
                return {
                    "next_action": "finish",
                    "messages": messages,
                    "final_result": repaired_plan,
                }

            meta = getattr(self.llm_client, "last_response_meta", {}) or {}
            logger.warning(
                "Failed to extract JSON from response: %s; finish_reason=%s; content_length=%s",
                e,
                meta.get("finish_reason"),
                meta.get("content_length"),
            )
            return {
                "next_action": "error",
                "messages": messages,
                "error": f"Failed to extract JSON: {str(e)}",
            }

    async def _attempt_json_repair(
        self,
        response_text: str,
        error_message: str,
    ) -> Optional[Dict[str, Any]]:
        """Use a small follow-up completion to repair malformed final JSON once."""
        if not response_text or not response_text.strip():
            return None

        self.metrics.hidden_repair_attempted = True
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "你是一个 JSON 修复器。"
                    "你会把旅行计划响应修复成完整、合法的 JSON。"
                    "只输出 JSON，不要解释，不要 Markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "下面是一段格式损坏或被截断的旅行计划 JSON。"
                    "请尽量保留原始内容，补全为一个合法的完整 JSON 对象。\n\n"
                    f"解析错误：{error_message}\n\n"
                    f"原始响应：\n{response_text}"
                ),
            },
        ]

        try:
            repaired_response = await self.llm_client.chat_completion(
                messages=repair_messages,
                temperature=0.1,
                max_tokens=self.json_repair_max_tokens,
                retry_count=1,
            )
            repaired_text = (
                repaired_response
                if isinstance(repaired_response, str)
                else repaired_response.get("content", "")
            )
            repaired_plan = self._extract_json_plan(repaired_text)
            self.metrics.hidden_repair_succeeded = True
            logger.warning("Recovered final JSON via repair call")
            return repaired_plan
        except Exception as repair_exc:
            logger.warning(f"JSON repair attempt failed: {repair_exc}")
            return None

    def _is_tool_call(self, text: str) -> bool:
        """Check if the text contains a tool call."""
        text = text.strip()
        return (
            text.startswith("Action[") or
            "attractions_select(" in text or
            "restaurants_select(" in text or
            "accommodations_select(" in text or
            "goto(" in text or
            "attractions_nearby(" in text or
            "restaurants_nearby(" in text
        )

    def _parse_action_from_text(self, text: str) -> Optional[tuple]:
        """Parse tool name and arguments from text-based action."""
        import re

        # Pattern 1: Action[tool_name(arg1=value1, arg2=value2)]
        action_pattern = r"Action\[(\w+)\((.*?)\)\]"
        match = re.search(action_pattern, text.strip())
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            return self._parse_tool_args(tool_name, args_str)

        # Pattern 2: direct tool_name(arg1=value1, arg2=value2)
        direct_pattern = r"(\w+)\((.*?)\)"
        match = re.search(direct_pattern, text.strip())
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            return self._parse_tool_args(tool_name, args_str)

        return None

    def _parse_tool_args(self, tool_name: str, args_str: str) -> tuple:
        """Parse tool arguments from argument string."""
        import shlex

        try:
            # Simple parsing - this could be made more robust
            args = {}
            if args_str.strip():
                # Split by commas but respect quotes
                parts = shlex.split(args_str)
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        # Try to convert to appropriate type
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit():
                            value = float(value)
                        elif value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        args[key] = value
            return tool_name, args
        except Exception as e:
            logger.warning(f"Failed to parse tool args '{args_str}': {e}")
            return tool_name, {}

    def _get_execution_atom_tools(self) -> List[Dict[str, Any]]:
        """
        Get execution atom tool definitions for ReAct.

        These tools allow ReAct to incrementally modify the plan using the same
        execution atoms as PTE-0/PTE-R, enabling fair comparison.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "insert_node",
                    "description": "在指定位置插入新节点（activity/transport/hotel）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {
                                "type": "object",
                                "description": "要插入的节点数据，包含 id, type, data 等字段。例如：{\"id\": \"activity_d1_a2_new\", \"type\": \"activity\", \"data\": {\"poi_name\": \"西湖\", \"duration_min\": 120}}"
                            },
                            "position": {
                                "type": "object",
                                "description": "位置信息，可以是 between 类型（{\"type\": \"between\", \"day\": 1, \"after_item_id\": \"...\"}）或 at_index 类型（{\"type\": \"at_index\", \"day\": 1, \"index\": 2}）"
                            }
                        },
                        "required": ["node", "position"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_node",
                    "description": "删除指定节点及其相关路线",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "要删除的节点 ID，例如：\"activity_d1_a2\""
                            }
                        },
                        "required": ["node_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_node",
                    "description": "移动节点到新位置或不同的天",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "要移动的节点 ID"
                            },
                            "new_position": {
                                "type": "object",
                                "description": "新位置信息（格式同 insert_node 的 position 参数）"
                            }
                        },
                        "required": ["node_id", "new_position"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "replace_node",
                    "description": "替换节点数据，保持位置不变",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_old_id": {
                                "type": "string",
                                "description": "要替换的旧节点 ID"
                            },
                            "node_new": {
                                "type": "object",
                                "description": "新节点数据（必须包含相同的 id）"
                            }
                        },
                        "required": ["node_old_id", "node_new"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reschedule_node",
                    "description": "修改节点的开始时间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "要重新调度的节点 ID"
                            },
                            "new_time": {
                                "type": "string",
                                "description": "新的开始时间，格式 HH:MM，例如：\"14:30\""
                            },
                            "policy": {
                                "type": "string",
                                "enum": ["shift_following", "no_propagation"],
                                "description": "时间传播策略：shift_following（将后续活动向后/向前移动，默认）或 no_propagation（只修改当前活动）"
                            }
                        },
                        "required": ["node_id", "new_time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "resize_node",
                    "description": "修改活动的持续时间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "要修改的节点 ID"
                            },
                            "new_duration": {
                                "type": "integer",
                                "description": "新的持续时间（分钟）"
                            },
                            "policy": {
                                "type": "string",
                                "enum": ["shift_following", "no_propagation"],
                                "description": "时间传播策略"
                            }
                        },
                        "required": ["node_id", "new_duration"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reorder_day",
                    "description": "重新排序一天中所有活动的顺序",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day": {
                                "type": "integer",
                                "description": "要排序的天数"
                            },
                            "new_order_item_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "新的活动 ID 顺序列表，例如：[\"activity_d1_a3\", \"activity_d1_a1\", \"activity_d1_a2\"]"
                            }
                        },
                        "required": ["day", "new_order_item_ids"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reroute_edge",
                    "description": "修改路线的交通方式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edge_id": {
                                "type": "string",
                                "description": "要修改的路线 ID，例如：\"route_activity_d1_a1_to_activity_d1_a2\""
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["walk", "metro", "taxi", "unknown"],
                                "description": "新的交通方式"
                            }
                        },
                        "required": ["edge_id", "mode"]
                    }
                }
            }
        ]

    def _get_framework_tools(self) -> List[Dict[str, Any]]:
        """
        Get the tool definitions specific to this framework.

        For the ReAct framework, this returns both query tools and execution atom tools,
        enabling fair comparison with PTE-0/PTE-R frameworks.
        """
        # Get query tools (original)
        query_tools = self._get_query_tools()

        # Get execution atom tools (new)
        execution_tools = self._get_execution_atom_tools()

        # Combine both
        return query_tools + execution_tools

    def _get_query_tools(self) -> List[Dict[str, Any]]:
        """
        Get query-only tool definitions (original ReAct tools).

        These are read-only tools for querying POI information.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "attractions_select",
                    "description": "查询城市的景点信息。可以按条件筛选，返回景点列表。注意：key 和 func_str 参数是必需的。如果要查询所有数据，使用 key='name' 和 func_str='lambda x: True'。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称，如'杭州'、'北京'等"
                            },
                            "key": {
                                "type": "string",
                                "description": "筛选字段名称，如'name'、'type'、'price'等。常用字段：name(名称), type(类型), price(价格)"
                            },
                            "func_str": {
                                "type": "string",
                                "description": "lambda函数字符串，必须返回布尔值。例如：'lambda x: True'(所有数据)，'lambda x: x == \"博物馆\"'(精确匹配)，'lambda x: \"西湖\" in x'(包含关键词)"
                            }
                        },
                        "required": ["city", "key", "func_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_select",
                    "description": "查询城市的餐厅信息。可以按条件筛选，返回餐厅列表。注意：key 和 func_str 参数是必需的。如果要查询所有数据，使用 key='name' 和 func_str='lambda x: True'。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "key": {
                                "type": "string",
                                "description": "筛选字段名称，如'name'、'cuisine'、'price'等"
                            },
                            "func_str": {
                                "type": "string",
                                "description": "lambda函数字符串，必须返回布尔值。例如：'lambda x: True'(所有数据)"
                            }
                        },
                        "required": ["city", "key", "func_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "accommodations_select",
                    "description": "查询城市的住宿信息。可以按条件筛选，返回住宿列表。注意:key 和 func_str 参数是必需的。如果要查询所有数据,使用 key='name' 和 func_str='lambda x: True'。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "key": {
                                "type": "string",
                                "description": "筛选字段名称,如'name'、'price'、'room_type'等"
                            },
                            "func_str": {
                                "type": "string",
                                "description": "lambda函数字符串,必须返回布尔值。例如:'lambda x: True'(所有数据)"
                            }
                        },
                        "required": ["city", "key", "func_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "attractions_nearby",
                    "description": "查询某个POI附近的景点。返回距离最近的景点列表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "point": {
                                "type": "string",
                                "description": "参考POI的名称"
                            },
                            "topk": {
                                "type": "integer",
                                "description": "返回前K个最近的景点，默认5"
                            },
                            "dist": {
                                "type": "number",
                                "description": "最大距离（公里），默认2"
                            }
                        },
                        "required": ["city", "point"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_nearby",
                    "description": "查询某个POI附近的餐厅。返回距离最近的餐厅列表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "point": {
                                "type": "string",
                                "description": "参考POI的名称"
                            },
                            "topk": {
                                "type": "integer",
                                "description": "返回前K个最近的餐厅，默认5"
                            },
                            "dist": {
                                "type": "number",
                                "description": "最大距离（公里），默认2"
                            }
                        },
                        "required": ["city", "point"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goto",
                    "description": "查询从起点到终点的交通路线。返回详细的交通方式、时间和费用信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "start": {
                                "type": "string",
                                "description": "起点POI名称"
                            },
                            "end": {
                                "type": "string",
                                "description": "终点POI名称"
                            },
                            "start_time": {
                                "type": "string",
                                "description": "出发时间，格式'HH:MM'"
                            },
                            "transport_type": {
                                "type": "string",
                                "enum": ["walk", "metro", "taxi"],
                                "description": "交通方式：'walk'(步行)、'metro'(地铁)、'taxi'(出租车)"
                            }
                        },
                        "required": ["city", "start", "end", "start_time", "transport_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "attractions_id_is_open",
                    "description": "检查指定ID的景点在某个时间是否开放。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "id": {
                                "type": "integer",
                                "description": "景点ID"
                            },
                            "time": {
                                "type": "string",
                                "description": "时间，格式'HH:MM'"
                            }
                        },
                        "required": ["city", "id", "time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_id_is_open",
                    "description": "检查指定ID的餐厅在某个时间是否营业。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            },
                            "id": {
                                "type": "integer",
                                "description": "餐厅ID"
                            },
                            "time": {
                                "type": "string",
                                "description": "时间，格式'HH:MM'"
                            }
                        },
                        "required": ["city", "id", "time"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call - supports both query tools and execution atom tools.

        This override extends the base implementation to handle execution atoms.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tool execution result as dict
        """
        # Check if this is an execution atom tool
        execution_atom_tools = [
            "insert_node", "delete_node", "move_node", "replace_node",
            "reschedule_node", "resize_node", "reorder_day", "reroute_edge"
        ]

        if tool_name in execution_atom_tools:
            # Use execution atom logic
            return self._execute_execution_atom(tool_name, args)
        else:
            # Use base class logic for query tools (WorldEnv tools)
            return super()._execute_tool(tool_name, args)

    def _execute_execution_atom(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an execution atom tool call.

        Execution atom tools are not exposed in the public artifact.

        Args:
            tool_name: Name of the execution atom tool
            args: Tool arguments

        Returns:
            Execution result with success status and data
        """
        return {
            "success": False,
            "error": f"Execution atom tool is not available in this artifact: {tool_name}",
            "data": f"Error: {tool_name} is not exposed by db_read_typed",
        }
