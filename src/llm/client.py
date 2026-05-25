"""
SiliconCloud LLM client for TPE system.

Provides OpenAI-compatible API integration with SiliconCloud for natural language processing.
"""

import asyncio
import copy
import contextvars
import html
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import httpx
import openai
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field, ValidationError

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from core.models.base import BaseTPEModel
from llm.model_router import ModelRoute, get_model_router
from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


SUPPORTED_MODEL_FORMATS = {"deepseek", "gemini", "openai", "qwen"}


class UnsupportedModelFormatError(ValueError):
    """Raised when a model family does not yet have an input/output adapter."""


class ModelFormatAdapter:
    """Provider-family adapter for request and response format differences."""

    family = "openai"

    def build_env_request_overrides(
        self,
        client: "SiliconCloudClient",
        model: str,
    ) -> Dict[str, Any]:
        return {}

    def should_preserve_reasoning_content(
        self,
        *,
        client: "SiliconCloudClient",
        model: Optional[str],
        route: Optional[ModelRoute],
    ) -> bool:
        return False

    def extract_textual_tool_calls(
        self,
        client: "SiliconCloudClient",
        content: str,
    ) -> Optional[List[Dict[str, Any]]]:
        return None

    def extract_reasoning_from_content(self, content: str) -> str:
        return ""

    def normalize_content(self, content: str) -> str:
        return content

    def build_assistant_tool_response(
        self,
        client: "SiliconCloudClient",
        *,
        content: str,
        tool_calls: List[Dict[str, Any]],
        reasoning_content: str = "",
    ) -> Dict[str, Any]:
        return client._assistant_tool_response(
            content=content,
            tool_calls=tool_calls,
        )


class OpenAIFormatAdapter(ModelFormatAdapter):
    """Standard OpenAI-compatible chat/tool-call format."""

    family = "openai"


class DeepSeekFormatAdapter(ModelFormatAdapter):
    """DeepSeek thinking-mode format, including reasoning_content replay."""

    family = "deepseek"

    def build_env_request_overrides(
        self,
        client: "SiliconCloudClient",
        model: str,
    ) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        extra_body = client._build_deepseek_extra_body(model)
        if extra_body:
            overrides["extra_body"] = extra_body
        reasoning_effort = client._build_deepseek_reasoning_effort(model)
        if reasoning_effort:
            overrides["reasoning_effort"] = reasoning_effort
        return overrides

    def should_preserve_reasoning_content(
        self,
        *,
        client: "SiliconCloudClient",
        model: Optional[str],
        route: Optional[ModelRoute],
    ) -> bool:
        if client.active_provider != "DMXAPI":
            return False
        if route is not None:
            return route.supports_thinking
        return bool(model and client._is_deepseek_reasoning_model(model))

    def extract_textual_tool_calls(
        self,
        client: "SiliconCloudClient",
        content: str,
    ) -> Optional[List[Dict[str, Any]]]:
        return client._extract_dsml_tool_calls(content)

    def build_assistant_tool_response(
        self,
        client: "SiliconCloudClient",
        *,
        content: str,
        tool_calls: List[Dict[str, Any]],
        reasoning_content: str = "",
    ) -> Dict[str, Any]:
        return client._assistant_tool_response(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )


class QwenFormatAdapter(ModelFormatAdapter):
    """Qwen OpenAI-compatible format with hybrid thinking controls."""

    family = "qwen"

    THINK_PATTERN = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL)

    def build_env_request_overrides(
        self,
        client: "SiliconCloudClient",
        model: str,
    ) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {}
        enable_thinking = client._get_provider_optional_bool_env("QWEN_ENABLE_THINKING")
        thinking_budget = client._get_provider_optional_int_env("QWEN_THINKING_BUDGET")
        preserve_thinking = client._get_provider_optional_bool_env("QWEN_PRESERVE_THINKING")
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            if thinking_budget > 0:
                extra_body["thinking_budget"] = thinking_budget
            else:
                logger.warning(
                    "Ignoring non-positive Qwen thinking_budget",
                    extra={"thinking_budget": thinking_budget},
                )
        if preserve_thinking is not None:
            extra_body["preserve_thinking"] = preserve_thinking
        return {"extra_body": extra_body} if extra_body else {}

    def should_preserve_reasoning_content(
        self,
        *,
        client: "SiliconCloudClient",
        model: Optional[str],
        route: Optional[ModelRoute],
    ) -> bool:
        env_override = client._get_provider_optional_bool_env("QWEN_PRESERVE_THINKING")
        if env_override is not None:
            return env_override
        if route is None:
            return False
        extra_body = route.request_overrides.get("extra_body")
        if isinstance(extra_body, dict):
            return bool(extra_body.get("preserve_thinking"))
        return False

    def extract_reasoning_from_content(self, content: str) -> str:
        match = self.THINK_PATTERN.match(content or "")
        return match.group(1).strip() if match else ""

    def normalize_content(self, content: str) -> str:
        return self.THINK_PATTERN.sub("", content or "", count=1).lstrip()

    def build_assistant_tool_response(
        self,
        client: "SiliconCloudClient",
        *,
        content: str,
        tool_calls: List[Dict[str, Any]],
        reasoning_content: str = "",
    ) -> Dict[str, Any]:
        return client._assistant_tool_response(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )


class GeminiFormatAdapter(ModelFormatAdapter):
    """Gemini OpenAI-compatible format with native thinking_config passthrough."""

    family = "gemini"

    def build_env_request_overrides(
        self,
        client: "SiliconCloudClient",
        model: str,
    ) -> Dict[str, Any]:
        thinking_level = client._get_provider_str_env(
            "GEMINI_THINKING_LEVEL",
            default="",
        )
        if not thinking_level:
            return {}
        return {"extra_body": {"thinking_config": {"thinking_level": thinking_level}}}


MODEL_FORMAT_ADAPTERS: Dict[str, ModelFormatAdapter] = {
    "deepseek": DeepSeekFormatAdapter(),
    "gemini": GeminiFormatAdapter(),
    "openai": OpenAIFormatAdapter(),
    "qwen": QwenFormatAdapter(),
}


@dataclass
class LLMUsageTracker:
    """Accumulates request time and provider token usage for one logical run."""

    llm_call_count: int = 0
    failed_llm_call_count: int = 0
    llm_request_time_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def record_success(self, meta: Dict[str, Any], elapsed_seconds: float) -> None:
        self.llm_call_count += 1
        self.llm_request_time_seconds += elapsed_seconds
        self.prompt_tokens += _coerce_token_count(meta.get("prompt_tokens"))
        self.completion_tokens += _coerce_token_count(meta.get("completion_tokens"))
        self.total_tokens += _coerce_token_count(meta.get("total_tokens"))

    def record_failure(self, elapsed_seconds: float) -> None:
        self.failed_llm_call_count += 1
        self.llm_request_time_seconds += elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_call_count": self.llm_call_count,
            "failed_llm_call_count": self.failed_llm_call_count,
            "llm_request_time_seconds": self.llm_request_time_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


_usage_tracker_var: contextvars.ContextVar[Optional[LLMUsageTracker]] = (
    contextvars.ContextVar("llm_usage_tracker", default=None)
)


def _coerce_token_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def begin_usage_collection() -> LLMUsageTracker:
    """Start collecting LLM usage in the current async context."""
    tracker = LLMUsageTracker()
    _usage_tracker_var.set(tracker)
    return tracker


def get_current_usage_tracker() -> Optional[LLMUsageTracker]:
    """Return the current context's usage tracker, if one is active."""
    return _usage_tracker_var.get()


class SiliconCloudClient:
    """SiliconCloud LLM client with OpenAI-compatible API."""

    # Keep a provider-level safety cap while allowing current DeepSeek V4
    # experiments to request the documented 64k thinking-mode output budget.
    DMXAPI_MAX_TOKENS = 65536

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize SiliconCloud client.

        Args:
            api_key: SiliconCloud API key (from env if not provided)
            base_url: SiliconCloud base URL (from env if not provided)
        """
        self.api_key = api_key or os.getenv("DMXAPI_API_KEY") or os.getenv(
            "SILICONCLOUD_API_KEY"
        )
        self.base_url = base_url or os.getenv("DMXAPI_BASE_URL") or os.getenv(
            "SILICONCLOUD_BASE_URL", "https://api.siliconflow.cn/v1"
        )

        if not self.api_key:
            raise ValueError("SiliconCloud API key not found in environment variables")

        # Clear proxy environment variables temporarily to avoid SOCKS proxy issues
        import httpx

        # Store original proxy values
        original_proxies = {
            "http_proxy": os.environ.get("http_proxy"),
            "https_proxy": os.environ.get("https_proxy"),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
            "all_proxy": os.environ.get("all_proxy"),
            "ALL_PROXY": os.environ.get("ALL_PROXY"),
        }

        # Clear proxy environment variables
        for proxy_var in original_proxies:
            if original_proxies[proxy_var]:
                os.environ.pop(proxy_var, None)

        try:
            # Initialize clients without proxy interference
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.async_client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url
            )
        finally:
            # Restore original proxy environment variables
            for proxy_var, value in original_proxies.items():
                if value is not None:
                    os.environ[proxy_var] = value

        # Default model. This may be a stable experiment alias; resolve it through
        # the model router before sending requests to the provider.
        configured_model = (
            os.getenv("DMXAPI_MODEL")
            or os.getenv("SILICONCLOUD_MODEL")
            or "deepseek-ai/DeepSeek-V3"
        )
        self.active_provider = self._detect_provider(api_key, base_url)
        self.model_router = get_model_router()
        self.default_model_route = self._resolve_model_route(configured_model)
        self.default_model_alias = (
            self.default_model_route.alias if self.default_model_route else configured_model
        )
        self.default_model = (
            self.default_model_route.model if self.default_model_route else configured_model
        )
        default_completion_tokens = (
            self.DMXAPI_MAX_TOKENS
            if self.active_provider == "DMXAPI"
            and self._is_deepseek_reasoning_model(self.default_model)
            else 12000
        )
        self.react_max_completion_tokens = self._get_provider_int_env(
            "REACT_MAX_COMPLETION_TOKENS",
            default=default_completion_tokens,
        )
        self.pter_max_completion_tokens = self._get_provider_int_env(
            "PTER_MAX_COMPLETION_TOKENS",
            default=self.react_max_completion_tokens,
        )
        self.react_json_repair_max_tokens = self._get_provider_int_env(
            "REACT_JSON_REPAIR_MAX_TOKENS",
            default=8000,
        )
        self.qwen_enable_thinking = self._get_provider_bool_env(
            "QWEN_ENABLE_THINKING",
            default=False,
        )
        self.qwen_thinking_budget = self._get_provider_optional_int_env(
            "QWEN_THINKING_BUDGET"
        )
        self.deepseek_thinking_type = self._get_provider_str_env(
            "DEEPSEEK_THINKING_TYPE",
            default="",
        )
        self.deepseek_reasoning_effort = self._get_provider_str_env(
            "DEEPSEEK_REASONING_EFFORT",
            default="",
        )
        self.last_response_meta: Dict[str, Any] = {}

    def _detect_provider(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> str:
        dmx_api_key = os.getenv("DMXAPI_API_KEY")
        dmx_base_url = os.getenv("DMXAPI_BASE_URL")
        silicon_api_key = os.getenv("SILICONCLOUD_API_KEY")
        silicon_base_url = os.getenv("SILICONCLOUD_BASE_URL")

        if api_key and dmx_api_key and api_key == dmx_api_key:
            return "DMXAPI"
        if base_url and dmx_base_url and base_url == dmx_base_url:
            return "DMXAPI"
        if api_key and silicon_api_key and api_key == silicon_api_key:
            return "SILICONCLOUD"
        if base_url and silicon_base_url and base_url == silicon_base_url:
            return "SILICONCLOUD"
        if dmx_api_key or dmx_base_url:
            return "DMXAPI"
        return "SILICONCLOUD"

    def _get_provider_int_env(self, key_suffix: str, *, default: int) -> int:
        candidates = []
        if self.active_provider == "DMXAPI":
            candidates.extend(
                [
                    f"DMXAPI_{key_suffix}",
                    f"SILICONCLOUD_{key_suffix}",
                ]
            )
        else:
            candidates.extend(
                [
                    f"SILICONCLOUD_{key_suffix}",
                    f"DMXAPI_{key_suffix}",
                ]
            )
        candidates.append(key_suffix)

        for env_name in candidates:
            raw_value = os.getenv(env_name)
            if raw_value is None or not raw_value.strip():
                continue
            try:
                return int(raw_value)
            except ValueError:
                logger.warning(
                    "Invalid integer env for LLM config",
                    extra={"env_name": env_name, "value": raw_value},
                )
        return default

    def _get_provider_optional_int_env(self, key_suffix: str) -> Optional[int]:
        candidates = []
        if self.active_provider == "DMXAPI":
            candidates.extend(
                [
                    f"DMXAPI_{key_suffix}",
                    f"SILICONCLOUD_{key_suffix}",
                ]
            )
        else:
            candidates.extend(
                [
                    f"SILICONCLOUD_{key_suffix}",
                    f"DMXAPI_{key_suffix}",
                ]
            )
        candidates.append(key_suffix)

        for env_name in candidates:
            raw_value = os.getenv(env_name)
            if raw_value is None or not raw_value.strip():
                continue
            try:
                return int(raw_value)
            except ValueError:
                logger.warning(
                    "Invalid integer env for LLM config",
                    extra={"env_name": env_name, "value": raw_value},
                )
        return None

    def _get_provider_str_env(self, key_suffix: str, *, default: str = "") -> str:
        candidates = []
        if self.active_provider == "DMXAPI":
            candidates.extend(
                [
                    f"DMXAPI_{key_suffix}",
                    f"SILICONCLOUD_{key_suffix}",
                ]
            )
        else:
            candidates.extend(
                [
                    f"SILICONCLOUD_{key_suffix}",
                    f"DMXAPI_{key_suffix}",
                ]
            )
        candidates.append(key_suffix)

        for env_name in candidates:
            raw_value = os.getenv(env_name)
            if raw_value is not None and raw_value.strip():
                return raw_value.strip()
        return default

    def _get_provider_bool_env(self, key_suffix: str, *, default: bool) -> bool:
        candidates = []
        if self.active_provider == "DMXAPI":
            candidates.extend(
                [
                    f"DMXAPI_{key_suffix}",
                    f"SILICONCLOUD_{key_suffix}",
                ]
            )
        else:
            candidates.extend(
                [
                    f"SILICONCLOUD_{key_suffix}",
                    f"DMXAPI_{key_suffix}",
                ]
            )
        candidates.append(key_suffix)

        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}

        for env_name in candidates:
            raw_value = os.getenv(env_name)
            if raw_value is None or not raw_value.strip():
                continue
            normalized = raw_value.strip().lower()
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
            logger.warning(
                "Invalid boolean env for LLM config",
                extra={"env_name": env_name, "value": raw_value},
            )
        return default

    def _get_provider_optional_bool_env(self, key_suffix: str) -> Optional[bool]:
        candidates = []
        if self.active_provider == "DMXAPI":
            candidates.extend(
                [
                    f"DMXAPI_{key_suffix}",
                    f"SILICONCLOUD_{key_suffix}",
                ]
            )
        else:
            candidates.extend(
                [
                    f"SILICONCLOUD_{key_suffix}",
                    f"DMXAPI_{key_suffix}",
                ]
            )
        candidates.append(key_suffix)

        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}

        for env_name in candidates:
            raw_value = os.getenv(env_name)
            if raw_value is None or not raw_value.strip():
                continue
            normalized = raw_value.strip().lower()
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
            logger.warning(
                "Invalid boolean env for LLM config",
                extra={"env_name": env_name, "value": raw_value},
            )
        return None

    def _extract_message_content(self, content: Any) -> str:
        """Normalize OpenAI-compatible content payloads into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue
                if isinstance(item.get("output_text"), str):
                    parts.append(item["output_text"])
            return "".join(parts)
        if isinstance(content, dict):
            for key in ("text", "content", "output_text"):
                value = content.get(key)
                if isinstance(value, str):
                    return value
        return str(content)

    def _normalize_max_tokens(self, max_tokens: Optional[int]) -> Optional[int]:
        """Clamp provider-specific max token values into a safe request range."""
        if max_tokens is None:
            return None
        if max_tokens < 1:
            logger.warning(
                "Ignoring non-positive max_tokens",
                extra={"provider": self.active_provider, "max_tokens": max_tokens},
            )
            return None
        if self.active_provider == "DMXAPI" and max_tokens > self.DMXAPI_MAX_TOKENS:
            logger.warning(
                "Clamping max_tokens to DMXAPI limit",
                extra={
                    "provider": self.active_provider,
                    "requested_max_tokens": max_tokens,
                    "clamped_max_tokens": self.DMXAPI_MAX_TOKENS,
                },
            )
            return self.DMXAPI_MAX_TOKENS
        return max_tokens

    def _is_deepseek_reasoning_model(self, model: str) -> bool:
        normalized = model.lower().replace("_", "-")
        return "deepseek-v4" in normalized or "deepseek-reasoner" in normalized

    def _is_deepseek_reasoning_request(self, model: Optional[str]) -> bool:
        adapter = self._model_format_adapter(model)
        route = self._resolve_model_route(model)
        return adapter.should_preserve_reasoning_content(
            client=self,
            model=model,
            route=route,
        )

    def _infer_model_family(self, model: Optional[str]) -> str:
        if not model:
            return "openai"
        route = self._resolve_model_route(model)
        if route is not None and route.family:
            return route.family.lower()
        normalized = (model or "").lower().replace("_", "-")
        if "deepseek" in normalized:
            return "deepseek"
        if "gemini" in normalized:
            return "gemini"
        if "qwen" in normalized:
            return "qwen"
        if normalized.startswith(("gpt-", "o1", "o3", "o4")) or "openai" in normalized:
            return "openai"
        return ""

    def _model_format_adapter(self, model: Optional[str]) -> ModelFormatAdapter:
        family = self._infer_model_family(model)
        adapter = MODEL_FORMAT_ADAPTERS.get(family)
        if adapter is None:
            route = self._resolve_model_route(model)
            route_label = route.alias if route is not None else model
            raise UnsupportedModelFormatError(
                "Unsupported model format "
                f"for model={route_label!r}, family={family or 'unknown'!r}. "
                "Add a model format adapter before running this model. "
                f"Supported families: {', '.join(sorted(SUPPORTED_MODEL_FORMATS))}."
            )
        return adapter

    def _resolve_model_route(self, model: Optional[str]) -> Optional[ModelRoute]:
        router = getattr(self, "model_router", None)
        if router is None:
            return None
        return router.resolve(model, provider=getattr(self, "active_provider", None))

    def _resolve_provider_model(self, model: str) -> str:
        route = self._resolve_model_route(model)
        return route.model if route is not None else model

    def _merge_extra_body(
        self,
        base: Optional[Dict[str, Any]],
        override: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not base and not override:
            return None
        merged: Dict[str, Any] = copy.deepcopy(base or {})
        for key, value in (override or {}).items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = self._merge_extra_body(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _build_deepseek_extra_body(self, model: str) -> Optional[Dict[str, Any]]:
        if self.active_provider != "DMXAPI":
            return None
        if not self._is_deepseek_reasoning_model(model):
            return None
        deepseek_thinking_type = getattr(self, "deepseek_thinking_type", "")
        if not deepseek_thinking_type:
            return None
        return {"thinking": {"type": deepseek_thinking_type}}

    def _build_deepseek_reasoning_effort(self, model: str) -> Optional[str]:
        if self.active_provider != "DMXAPI":
            return None
        if not self._is_deepseek_reasoning_model(model):
            return None
        deepseek_reasoning_effort = getattr(self, "deepseek_reasoning_effort", "")
        if not deepseek_reasoning_effort:
            return None
        return deepseek_reasoning_effort

    def _build_legacy_request_overrides(self, model: str) -> Dict[str, Any]:
        return self._model_format_adapter(model).build_env_request_overrides(self, model)

    def _build_request_overrides(self, model: str) -> Dict[str, Any]:
        env_overrides = self._build_legacy_request_overrides(model)
        route = self._resolve_model_route(model)
        if route is None:
            return env_overrides

        overrides: Dict[str, Any] = {}
        route_overrides = copy.deepcopy(route.request_overrides)
        env_extra_body = env_overrides.pop("extra_body", None)
        route_extra_body = route_overrides.pop("extra_body", None)
        extra_body = self._merge_extra_body(route_extra_body, env_extra_body)
        if extra_body:
            overrides["extra_body"] = extra_body
        overrides.update(route_overrides)
        overrides.update(env_overrides)
        return overrides

    def _extract_reasoning_content(self, message: Any) -> str:
        """Extract provider-specific hidden reasoning text when the SDK exposes it."""
        direct = getattr(message, "reasoning_content", None)
        if isinstance(direct, str):
            return direct
        if isinstance(message, dict) and isinstance(message.get("reasoning_content"), str):
            return message["reasoning_content"]
        model_extra = getattr(message, "model_extra", {}) or {}
        if isinstance(model_extra, dict) and isinstance(model_extra.get("reasoning_content"), str):
            return model_extra["reasoning_content"]
        return ""

    def _serialize_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls or []):
            if isinstance(tool_call, dict):
                serialized.append(copy.deepcopy(tool_call))
                continue
            function = getattr(tool_call, "function", None)
            serialized.append(
                {
                    "id": getattr(tool_call, "id", f"tool_call_{index}"),
                    "type": getattr(tool_call, "type", "function"),
                    "function": {
                        "name": getattr(function, "name", ""),
                        "arguments": getattr(function, "arguments", "{}"),
                    },
                }
            )
        return serialized

    def _assistant_tool_response(
        self,
        *,
        content: str,
        tool_calls: List[Dict[str, Any]],
        reasoning_content: str = "",
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        if reasoning_content:
            response["reasoning_content"] = reasoning_content
        return response

    def _extract_dsml_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse DMX/DeepSeek textual DSML tool calls into OpenAI-style calls."""
        if "DSML" not in content or "tool_calls" not in content:
            return None

        dsml_bar = r"[｜|]{1,2}"
        invoke_pattern = re.compile(
            rf"<{dsml_bar}DSML{dsml_bar}invoke\s+name=\"([^\"]+)\"\s*>(.*?)"
            rf"</{dsml_bar}DSML{dsml_bar}invoke>",
            re.DOTALL,
        )
        parameter_pattern = re.compile(
            rf"<{dsml_bar}DSML{dsml_bar}parameter"
            rf"\s+name=\"([^\"]+)\"([^>]*)>(.*?)"
            rf"</{dsml_bar}DSML{dsml_bar}parameter>",
            re.DOTALL,
        )
        calls: List[Dict[str, Any]] = []

        for call_index, invoke_match in enumerate(invoke_pattern.finditer(content)):
            tool_name = html.unescape(invoke_match.group(1))
            body = invoke_match.group(2)
            arguments: Dict[str, Any] = {}
            for parameter_match in parameter_pattern.finditer(body):
                arg_name = html.unescape(parameter_match.group(1))
                attrs = parameter_match.group(2) or ""
                raw_value = html.unescape(parameter_match.group(3).strip())
                if 'string="true"' in attrs:
                    value: Any = raw_value
                else:
                    try:
                        value = json.loads(raw_value)
                    except json.JSONDecodeError:
                        value = raw_value
                arguments[arg_name] = value

            calls.append(
                {
                    "id": f"dsml_call_{call_index}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )

        return calls or None

    def _normalize_tool_arguments_for_request(self, raw_arguments: Any) -> str:
        """Return a provider-safe JSON object string for historical tool calls.

        Some OpenAI-compatible providers reject the whole next request when a prior
        assistant tool call contains malformed ``function.arguments``. We still
        surface the malformed call through the paired tool result, but the replayed
        assistant message itself must be valid JSON for strict providers.
        """
        if isinstance(raw_arguments, dict):
            return json.dumps(raw_arguments, ensure_ascii=False)
        if raw_arguments is None:
            return "{}"
        if not isinstance(raw_arguments, str):
            return "{}"

        text = raw_arguments.strip()
        if not text:
            return "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return "{}"
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return "{}"
        if not isinstance(parsed, dict):
            return "{}"
        return json.dumps(parsed, ensure_ascii=False)

    def _sanitize_messages_for_request(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize message payloads before sending them to strict providers."""
        preserve_reasoning_content = self._is_deepseek_reasoning_request(model)
        sanitized: List[Dict[str, Any]] = []
        for message in messages:
            next_message = copy.deepcopy(message)
            tool_calls = next_message.get("tool_calls")
            reasoning_content = next_message.get("reasoning_content")
            if (
                not preserve_reasoning_content
                or next_message.get("role") != "assistant"
                or not isinstance(tool_calls, list)
                or not tool_calls
                or not isinstance(reasoning_content, str)
                or not reasoning_content
            ):
                next_message.pop("reasoning_content", None)
            if isinstance(tool_calls, list):
                normalized_calls = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        normalized_calls.append(tool_call)
                        continue
                    next_call = copy.deepcopy(tool_call)
                    function = next_call.get("function")
                    if isinstance(function, dict):
                        function["arguments"] = self._normalize_tool_arguments_for_request(
                            function.get("arguments")
                        )
                    normalized_calls.append(next_call)
                next_message["tool_calls"] = normalized_calls
            sanitized.append(next_message)
        return sanitized

    def _build_response_meta(
        self,
        response: Any,
        message: Any,
        *,
        model: str,
    ) -> Dict[str, Any]:
        """Capture response metadata for diagnostics and downstream recovery."""
        choice = response.choices[0] if getattr(response, "choices", None) else None
        usage = getattr(response, "usage", None)
        model_extra = getattr(message, "model_extra", {}) or {}
        tool_calls = getattr(message, "tool_calls", None)
        content_text = self._extract_message_content(getattr(message, "content", None))
        reasoning_content = self._extract_reasoning_content(message)
        meta = {
            "model": model,
            "finish_reason": getattr(choice, "finish_reason", None),
            "has_tool_calls": bool(tool_calls),
            "content_length": len(content_text),
            "reasoning_content_length": len(reasoning_content),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "message_extra_keys": sorted(model_extra.keys()),
        }
        self.last_response_meta = meta
        return meta

    def _record_successful_usage(
        self,
        meta: Dict[str, Any],
        elapsed_seconds: float,
    ) -> None:
        tracker = get_current_usage_tracker()
        if tracker is not None:
            tracker.record_success(meta, elapsed_seconds)

    def _record_failed_usage(self, elapsed_seconds: float) -> None:
        tracker = get_current_usage_tracker()
        if tracker is not None:
            tracker.record_failure(elapsed_seconds)

    def _raise_empty_response(self, meta: Dict[str, Any]) -> None:
        raise ValueError(
            "Empty response from LLM "
            f"(finish_reason={meta.get('finish_reason')}, "
            f"has_tool_calls={meta.get('has_tool_calls')}, "
            f"content_length={meta.get('content_length')}, "
            f"message_extra_keys={meta.get('message_extra_keys')})"
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_model: Optional[Type[T]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> Union[str, T, Dict[str, Any]]:
        """
        Get chat completion with optional structured output and tool calling.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (default from config)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_model: Pydantic model for structured output
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy ('auto', 'none', or specific tool)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Response as string, structured object, or dict with tool_calls if applicable
        """
        requested_model = model or self.default_model
        provider_model = self._resolve_provider_model(requested_model)
        adapter = self._model_format_adapter(requested_model)

        for attempt in range(retry_count + 1):
            attempt_start = time.perf_counter()
            usage_recorded = False
            try:
                # Only log retry attempts, not the first attempt
                if attempt > 0:
                    logger.warning(
                        f"LLM request retry {attempt}/{retry_count}",
                        extra={"model": provider_model}
                    )

                params = {
                    "model": provider_model,
                    "messages": self._sanitize_messages_for_request(
                        messages,
                        model=requested_model,
                    ),
                    "temperature": temperature,
                }
                params.update(self._build_request_overrides(requested_model))

                normalized_max_tokens = self._normalize_max_tokens(max_tokens)
                if normalized_max_tokens is not None:
                    params["max_tokens"] = normalized_max_tokens

                # Handle tool calling
                if tools:
                    params["tools"] = tools
                    if tool_choice:
                        params["tool_choice"] = tool_choice

                # Handle structured output (not compatible with tools)
                elif response_model:
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "schema": response_model.model_json_schema(),
                        },
                    }

                response = await self.async_client.chat.completions.create(**params)
                message = response.choices[0].message
                meta = self._build_response_meta(response, message, model=provider_model)
                self._record_successful_usage(
                    meta,
                    time.perf_counter() - attempt_start,
                )
                usage_recorded = True
                raw_content = self._extract_message_content(
                    getattr(message, "content", None)
                )
                reasoning_content = self._extract_reasoning_content(message)
                if not reasoning_content:
                    reasoning_content = adapter.extract_reasoning_from_content(
                        raw_content
                    )
                content = adapter.normalize_content(raw_content)

                # Handle tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    return adapter.build_assistant_tool_response(
                        self,
                        content=content,
                        tool_calls=self._serialize_tool_calls(message.tool_calls),
                        reasoning_content=reasoning_content,
                    )

                textual_tool_calls = adapter.extract_textual_tool_calls(self, content)
                if textual_tool_calls:
                    return adapter.build_assistant_tool_response(
                        self,
                        content="",
                        tool_calls=textual_tool_calls,
                        reasoning_content=reasoning_content,
                    )

                if not content:
                    self._raise_empty_response(meta)

                # Parse structured output if needed
                if response_model:
                    try:
                        json_content = json.loads(content)
                        return response_model.model_validate(json_content)
                    except (json.JSONDecodeError, ValidationError) as e:
                        logger.error(f"Failed to parse structured response: {e}")
                        raise ValueError(f"Invalid structured response: {e}")

                return content

            except Exception as e:
                if not usage_recorded:
                    self._record_failed_usage(time.perf_counter() - attempt_start)
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")

                if attempt < retry_count:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff
                else:
                    logger.error(
                        f"LLM request failed after {retry_count + 1} attempts: {e}"
                    )
                    raise

    def chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_model: Optional[Type[T]] = None,
    ) -> Union[str, T, Dict[str, Any]]:
        """
        Synchronous version of chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (default from config)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_model: Pydantic model for structured output

        Returns:
            Response as string, structured object, or dict
        """
        requested_model = model or self.default_model
        provider_model = self._resolve_provider_model(requested_model)
        adapter = self._model_format_adapter(requested_model)

        request_start = time.perf_counter()
        usage_recorded = False
        try:
            # Only log sync requests at debug level to reduce noise
            logger.debug(
                f"LLM sync request",
                extra={"model": provider_model, "message_count": len(messages)},
            )

            params = {
                "model": provider_model,
                "messages": self._sanitize_messages_for_request(
                    messages,
                    model=requested_model,
                ),
                "temperature": temperature,
            }
            params.update(self._build_request_overrides(requested_model))

            normalized_max_tokens = self._normalize_max_tokens(max_tokens)
            if normalized_max_tokens is not None:
                params["max_tokens"] = normalized_max_tokens

            if response_model:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                    },
                }

            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            meta = self._build_response_meta(response, message, model=provider_model)
            self._record_successful_usage(meta, time.perf_counter() - request_start)
            usage_recorded = True
            raw_content = self._extract_message_content(
                getattr(message, "content", None)
            )
            content = adapter.normalize_content(raw_content)

            if not content:
                self._raise_empty_response(meta)

            if response_model:
                try:
                    json_content = json.loads(content)
                    return response_model.model_validate(json_content)
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Failed to parse structured response: {e}")
                    raise ValueError(f"Invalid structured response: {e}")

            return content

        except Exception as e:
            if not usage_recorded:
                self._record_failed_usage(time.perf_counter() - request_start)
            logger.error(f"LLM sync request failed: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test connection to SiliconCloud API."""
        try:
            messages = [
                {"role": "user", "content": "Hello, can you respond with just 'OK'?"}
            ]
            response = await self.chat_completion(messages, max_tokens=10)
            return "OK" in str(response).upper()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


_llm_client: Optional[SiliconCloudClient] = None


def get_llm_client() -> SiliconCloudClient:
    """Return a singleton SiliconCloudClient. Raises if API key missing."""
    global _llm_client
    if _llm_client is None:
        _llm_client = SiliconCloudClient()
    return _llm_client


async def initialize_llm_client() -> bool:
    """Initialize and test connectivity to SiliconCloud API."""
    try:
        client = get_llm_client()
        return await client.test_connection()
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return False
