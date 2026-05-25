"""
Base Framework for LLM Agents.

Provides the abstract interface and common functionality for different LLM interaction frameworks.
All concrete framework implementations should inherit from BaseLLMFramework.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from utils.chinatravel_plan import normalize_loose_chinatravel_plan, require_chinatravel_plan
from utils.logging import get_logger

logger = get_logger(__name__)


class FrameworkStep:
    """Represents a single step in a framework execution."""

    def __init__(
        self,
        step_number: int,
        step_type: str,  # 'thought', 'action', 'observation', 'plan', etc.
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.step_number = step_number
        self.step_type = step_type
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class FrameworkMetrics:
    """Metrics collected during framework execution."""

    def __init__(self):
        self.total_steps = 0
        self.tool_calls_made = 0
        self.tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_call_count = 0
        self.failed_llm_call_count = 0
        self.llm_request_time_seconds = 0.0
        self.execution_time_seconds = 0.0
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0
        self.semantic_tool_failure_buckets: Dict[str, Dict[str, int]] = {}
        self.semantic_tool_failure_categories = {
            "schema_validation_failures": 0,
            "missing_required_field_failures": 0,
            "invalid_shape_failures": 0,
        }
        self.framework_type = "unknown"
        self.budget_finalize_attempted = False
        self.budget_finalize_succeeded = False
        self.tool_argument_parse_error_count = 0
        self.hidden_repair_attempted = False
        self.hidden_repair_succeeded = False
        self.llm_usage_tracker = None

    def _sync_llm_usage(self) -> None:
        if self.llm_usage_tracker is None:
            return
        usage = self.llm_usage_tracker.to_dict()
        self.prompt_tokens = usage["prompt_tokens"]
        self.completion_tokens = usage["completion_tokens"]
        self.tokens_used = usage["total_tokens"]
        self.llm_call_count = usage["llm_call_count"]
        self.failed_llm_call_count = usage["failed_llm_call_count"]
        self.llm_request_time_seconds = usage["llm_request_time_seconds"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        self._sync_llm_usage()
        return {
            "total_steps": self.total_steps,
            "tool_calls_made": self.tool_calls_made,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.tokens_used,
            "llm_call_count": self.llm_call_count,
            "failed_llm_call_count": self.failed_llm_call_count,
            "llm_request_time_seconds": self.llm_request_time_seconds,
            "execution_time_seconds": self.execution_time_seconds,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "semantic_tool_failure_buckets": self.semantic_tool_failure_buckets,
            "semantic_tool_failure_categories": self.semantic_tool_failure_categories,
            "framework_type": self.framework_type,
            "budget_finalize_attempted": self.budget_finalize_attempted,
            "budget_finalize_succeeded": self.budget_finalize_succeeded,
            "tool_argument_parse_error_count": self.tool_argument_parse_error_count,
            "hidden_repair_attempted": self.hidden_repair_attempted,
            "hidden_repair_succeeded": self.hidden_repair_succeeded,
            "success_rate": (
                self.successful_tool_calls / max(1, self.tool_calls_made)
            ),
        }

    def record_semantic_tool_failure(self, tool_name: str, tool_result: Dict[str, Any]) -> None:
        error_code = str(tool_result.get("error_code") or "unknown")
        bucket = self.semantic_tool_failure_buckets.setdefault(tool_name, {})
        bucket[error_code] = bucket.get(error_code, 0) + 1

        self.semantic_tool_failure_categories["schema_validation_failures"] += 1
        missing_fields = list(tool_result.get("missing_fields") or [])
        invalid_fields = list(tool_result.get("invalid_fields") or [])
        if missing_fields or error_code.startswith("missing_"):
            self.semantic_tool_failure_categories["missing_required_field_failures"] += 1
        if invalid_fields or error_code.startswith("invalid_"):
            self.semantic_tool_failure_categories["invalid_shape_failures"] += 1


class BaseLLMFramework(ABC):
    """
    Abstract base class for LLM interaction frameworks.

    A framework defines the pattern of interaction between the LLM and tools/environment.
    Examples include ReAct (Reason-Act-Observate), Plan-then-Execute, etc.
    """

    def __init__(
        self,
        framework_id: str,
        llm_client,
        world_env,
        max_steps: int = 30,
        max_tool_calls: int = 50,
        debug_logger=None,
    ):
        """
        Initialize the framework.

        Args:
            framework_id: Unique identifier for this framework instance
            llm_client: LLM client for making API calls
            world_env: Environment for tool execution
            max_steps: Maximum number of reasoning steps
            max_tool_calls: Maximum number of tool calls allowed
            debug_logger: Optional debug logger
        """
        self.framework_id = framework_id
        self.llm_client = llm_client
        self.world_env = world_env
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.debug_logger = debug_logger

        # Execution tracking
        self.steps: List[FrameworkStep] = []
        self.metrics = FrameworkMetrics()
        self._start_llm_usage_collection()
        self.conversation_history: List[Dict[str, Any]] = []

        # Framework identification
        self.metrics.framework_type = self.get_framework_type()

    def _repair_tool_argument_json_placeholders(self, text: str) -> str:
        """Replace structural ellipsis placeholders that some models emit in tool args."""
        repaired: list[str] = []
        i = 0
        in_string = False
        escape = False
        while i < len(text):
            char = text[i]
            if in_string:
                repaired.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                i += 1
                continue

            if char == '"':
                in_string = True
                repaired.append(char)
                i += 1
                continue
            if text.startswith("[...]", i) or text.startswith("[…]", i):
                repaired.append("[]")
                i += 5 if text.startswith("[...]", i) else 3
                continue
            if text.startswith("{...}", i) or text.startswith("{…}", i):
                repaired.append("{}")
                i += 5 if text.startswith("{...}", i) else 3
                continue
            repaired.append(char)
            i += 1
        return "".join(repaired)

    def _start_llm_usage_collection(self) -> None:
        """Attach a fresh LLM usage tracker to this framework run context."""
        try:
            from llm.client import begin_usage_collection

            self.metrics.llm_usage_tracker = begin_usage_collection()
        except Exception as exc:
            logger.debug("Unable to start LLM usage collection: %s", exc)

    @abstractmethod
    def get_framework_type(self) -> str:
        """Return the framework type identifier."""
        pass

    @abstractmethod
    async def edit_plan(
        self,
        edit_request: Dict[str, Any],
        original_plan: Dict[str, Any],
        original_query: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the plan editing using this framework.

        Args:
            edit_request: The editing request from user
            original_plan: Original travel plan to edit
            original_query: Original query that generated the plan
            **kwargs: Additional framework-specific parameters

        Returns:
            Dictionary containing:
            - success: bool indicating if editing succeeded
            - edited_plan: the modified plan (if successful)
            - conversation_log: detailed execution log
            - metrics: framework performance metrics
            - errors: list of error messages (if any)
        """
        pass

    @abstractmethod
    def _get_framework_tools(self) -> List[Dict[str, Any]]:
        """
        Get the tool definitions specific to this framework.

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        pass

    @abstractmethod
    def _execute_framework_step(
        self,
        messages: List[Dict[str, Any]],
        step_number: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single step of the framework logic.

        Args:
            messages: Current conversation messages
            step_number: Current step number
            context: Execution context containing tools, state, etc.

        Returns:
            Dictionary with step execution results including:
            - next_action: what to do next (continue, finish, error)
            - messages: updated conversation messages
            - tool_results: results from any tool calls
            - metadata: additional step information
        """
        pass

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool call via WorldEnv.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tool execution result as dict
        """
        try:
            tool_adapter = getattr(self, "tool_adapter", None)
            if tool_adapter is not None and hasattr(tool_adapter, "validate_query_args"):
                validation_error = tool_adapter.validate_query_args(tool_name, args)
                if validation_error is not None:
                    self.metrics.tool_calls_made += 1
                    self.metrics.failed_tool_calls += 1
                    return validation_error

            # Build command string for WorldEnv
            cmd = self._build_env_command(tool_name, args)

            logger.debug(f"WorldEnv command: {cmd}")

            # Execute in WorldEnv
            result = self.world_env(cmd)

            if tool_adapter is not None and hasattr(tool_adapter, "format_tool_result"):
                formatted = tool_adapter.format_tool_result(tool_name, args, cmd, result)
                self.metrics.tool_calls_made += 1
                if formatted.get("ok"):
                    self.metrics.successful_tool_calls += 1
                else:
                    self.metrics.failed_tool_calls += 1
                return formatted

            # Convert EnvOutput to dict
            # Handle both dict and EnvOutput-like objects
            if hasattr(result, "to_dict"):
                raw = result.to_dict()
                success = bool(raw.get("success", False))
                data = raw.get("whole_data", raw.get("data"))
            elif isinstance(result, dict):
                success = result.get("success", False)
                data = result.get("whole_data", result.get("data", str(result)))
            elif hasattr(result, "__getitem__"):
                success = bool(result["success"])
                data = result["whole_data"]
            else:
                success = False
                data = str(result)

            # Track metrics (after converting result to check success)
            self.metrics.tool_calls_made += 1
            if success:
                self.metrics.successful_tool_calls += 1
            else:
                self.metrics.failed_tool_calls += 1

            # Return serializable result (don't include raw EnvOutput object)
            return {
                "ok": success,
                "data": data,
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            self.metrics.tool_calls_made += 1
            self.metrics.failed_tool_calls += 1
            return {
                "ok": False,
                "error": str(e),
                "message": f"Error: {str(e)}",
            }

    def _build_env_command(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Build WorldEnv command string from tool name and args.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Command string for WorldEnv
        """
        # Escape string values
        def escape_value(v):
            if isinstance(v, str):
                return f"'{v}'"
            return str(v)

        # Define parameter orders for each tool (WorldEnv uses positional args)
        param_orders = {
            "attractions_select": ["city", "key", "func_str"],
            "restaurants_select": ["city", "key", "func_str"],
            "accommodations_select": ["city", "key", "func_str"],
            "attractions_nearby": ["city", "point", "topk", "dist"],
            "restaurants_nearby": ["city", "point", "topk", "dist"],
            "accommodations_nearby": ["city", "point", "topk", "dist"],
            "goto": ["city", "start", "end", "start_time", "transport_type"],
            "attractions_id_is_open": ["city", "id", "time"],
            "restaurants_id_is_open": ["city", "id", "time"],
        }

        # Get parameter order for this tool
        if tool_name in param_orders:
            param_order = param_orders[tool_name]
            # Build positional arguments in correct order
            arg_values = []
            for param in param_order:
                if param in args and args[param] is not None:
                    # func_str should not be escaped - it's Python code
                    if param == "func_str":
                        arg_values.append(args[param])
                    else:
                        arg_values.append(escape_value(args[param]))
            arg_str = ", ".join(arg_values)
        else:
            # Fallback to keyword arguments
            arg_strs = [f"{k}={escape_value(v)}" for k, v in args.items() if v is not None]
            arg_str = ", ".join(arg_strs)

        return f"{tool_name}({arg_str})"

    def _trim_messages_preserving_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_messages: int,
        head_messages: int = 2,
    ) -> List[Dict[str, Any]]:
        """Trim history without leaving `tool` messages orphaned.

        Some providers validate that each `tool` role message appears after an
        assistant message containing the corresponding `tool_calls`. When we
        shorten history, keep recent turns in whole units so an
        `assistant(tool_calls) -> tool...` block is either kept together or
        dropped together.
        """
        if len(messages) <= max_messages:
            return list(messages)

        head = list(messages[:head_messages])
        tail_budget = max(0, max_messages - len(head))
        if tail_budget == 0:
            return head

        body = messages[head_messages:]
        units: List[List[Dict[str, Any]]] = []
        index = 0

        while index < len(body):
            message = body[index]
            role = message.get("role")

            if role == "assistant" and message.get("tool_calls"):
                unit = [message]
                index += 1
                while index < len(body) and body[index].get("role") == "tool":
                    unit.append(body[index])
                    index += 1
                units.append(unit)
                continue

            if role == "tool":
                unit = [message]
                index += 1
                while index < len(body) and body[index].get("role") == "tool":
                    unit.append(body[index])
                    index += 1
                units.append(unit)
                continue

            units.append([message])
            index += 1

        kept_units: List[List[Dict[str, Any]]] = []
        used = 0
        for unit in reversed(units):
            unit_size = len(unit)
            if kept_units and used + unit_size > tail_budget:
                break
            if not kept_units and unit_size > tail_budget:
                continue
            kept_units.append(unit)
            used += unit_size
            if used >= tail_budget:
                break

        tail: List[Dict[str, Any]] = []
        for unit in reversed(kept_units):
            tail.extend(unit)
        return head + tail

    def _assistant_message_from_response(
        self,
        response: Dict[str, Any],
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a replayable assistant message from an adapter-normalized response."""
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": response.get("content", ""),
            "tool_calls": tool_calls if tool_calls is not None else response.get("tool_calls", []),
        }
        reasoning_content = response.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        return assistant_message

    def _extract_json_plan(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON plan from LLM response content.

        Args:
            content: LLM response content

        Returns:
            Parsed JSON plan
        """
        import re

        if not content or not content.strip():
            raise ValueError("Empty response content")

        original_content = content
        content = content.strip()

        # Try 1: Direct JSON parse (fastest path)
        if content.startswith('{'):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                logger.debug(f"Direct parse failed: {e}")
            else:
                parsed = normalize_loose_chinatravel_plan(parsed)
                return require_chinatravel_plan(parsed, context="final_response")

        # Try 2: Remove markdown code blocks if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()

        # Try 3: Look for JSON after any explanatory text
        json_patterns = [
            r'\{[\s\n]*"people_number"',  # ChinaTravel format
            r'\{[\s\n]*"itinerary"',  # Direct itinerary start
        ]

        best_start_idx = len(content)
        for pattern in json_patterns:
            match = re.search(pattern, content)
            if match and match.start() < best_start_idx:
                best_start_idx = match.start()

        if best_start_idx < len(content):
            start_idx = best_start_idx
        else:
            # Fallback: find any opening brace
            start_idx = content.find('{')
            if start_idx == -1:
                raise ValueError(f"No JSON object found in response: {content[:200]}...")

        # Extract JSON with proper string handling
        brace_count = 0
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
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

        if end_idx == -1:
            raise ValueError(f"Could not find matching closing brace in: {content[start_idx:start_idx+200]}...")

        json_str = content[start_idx:end_idx]

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try cleaning up common issues
            cleaned_json = json_str
            cleaned_json = re.sub(r'//.*?$', '', cleaned_json, flags=re.MULTILINE)
            cleaned_json = re.sub(r'/\*.*?\*/', '', cleaned_json, flags=re.DOTALL)
            cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)

            try:
                parsed = json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Last resort: try json5 if available
                try:
                    import json5
                    parsed = json5.loads(cleaned_json)
                    logger.warning("Successfully parsed JSON using json5 (lenient parser)")
                except ImportError:
                    pass
                except Exception:
                    pass

                raise ValueError(f"Could not parse JSON after cleanup: {e.msg}")

        parsed = normalize_loose_chinatravel_plan(parsed)
        return require_chinatravel_plan(parsed, context="final_response")

    def _add_step(
        self,
        step_type: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a step to the execution log.

        Args:
            step_type: Type of step (thought, action, observation, etc.)
            content: Step content
            tool_calls: Any tool calls made in this step
            tool_results: Results of tool calls
            metadata: Additional metadata
        """
        step = FrameworkStep(
            step_number=len(self.steps) + 1,
            step_type=step_type,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metadata=metadata or {},
        )
        self.steps.append(step)
        self.metrics.total_steps = len(self.steps)

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete execution log.

        Returns:
            List of step dictionaries with all execution details
        """
        return [
            {
                "step_number": step.step_number,
                "step_type": step.step_type,
                "content": step.content,
                "tool_calls": step.tool_calls,
                "tool_results": step.tool_results,
                "metadata": step.metadata,
                "timestamp": step.timestamp.isoformat(),
            }
            for step in self.steps
        ]

    def reset(self) -> None:
        """Reset the framework state for a new execution."""
        self.steps.clear()
        self.conversation_history.clear()
        self.metrics = FrameworkMetrics()
        self._start_llm_usage_collection()
        self.metrics.framework_type = self.get_framework_type()
