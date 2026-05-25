"""
LLM Client
封装LLM API调用，支持多种provider
"""

import os
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float


class LLMClient:
    """LLM客户端"""

    def __init__(self,
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 8000,
                 timeout: int = 60):
        """
        初始化LLM客户端

        Args:
            model: 模型名称
            api_key: API密钥（如果为None则从环境变量读取）
            base_url: API基础URL（如果为None则使用默认）
            temperature: 温度参数
            max_tokens: 最大输出tokens
            timeout: 请求超时时间（秒）
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 从环境变量读取API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and OPENAI_API_KEY not found in environment")

        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"

        # 尝试导入openai
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            self.use_openai = True
        except ImportError:
            print("Warning: openai package not found, falling back to requests")
            self.client = None
            self.use_openai = False

    def call(self,
             messages: List[Dict[str, str]],
             **kwargs) -> LLMResponse:
        """
        调用LLM

        Args:
            messages: 消息列表
            **kwargs: 额外的参数（会覆盖默认值）

        Returns:
            LLMResponse对象
        """
        # 合并参数
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        params.update(kwargs)

        start_time = time.time()

        if self.use_openai and self.client:
            # 使用openai包
            response = self.client.chat.completions.create(
                messages=messages,
                **params
            )

            content = response.choices[0].message.content
            usage = response.usage

            return LLMResponse(
                content=content,
                model=response.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency=time.time() - start_time
            )
        else:
            # 使用requests
            import json
            import requests

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "messages": messages,
                **params
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            return LLMResponse(
                content=content,
                model=result.get("model", self.model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency=time.time() - start_time
            )

    def call_with_retry(self,
                       messages: List[Dict[str, str]],
                       max_retries: int = 3,
                       retry_delay: float = 1.0,
                       debug_logger: Optional[Callable[[Dict[str, Any]], None]] = None,
                       debug_context: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Optional[LLMResponse]:
        """
        带重试的LLM调用

        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            **kwargs: 额外的参数

        Returns:
            LLMResponse对象，失败返回None
        """
        safe_context = debug_context or {}

        for attempt in range(max_retries):
            attempt_index = attempt + 1
            attempt_start = time.time()
            try:
                response = self.call(messages, **kwargs)

                if debug_logger:
                    try:
                        debug_logger({
                            **safe_context,
                            "attempt": attempt_index,
                            "max_retries": max_retries,
                            "success": True,
                            "error": "",
                            "request_messages": messages,
                            "response": {
                                "content": response.content,
                                "model": response.model,
                                "prompt_tokens": response.prompt_tokens,
                                "completion_tokens": response.completion_tokens,
                                "total_tokens": response.total_tokens,
                                "latency": response.latency
                            },
                            "attempt_latency": time.time() - attempt_start
                        })
                    except Exception as log_error:
                        print(f"Warning: Failed to write LLM debug log: {log_error}")

                return response
            except Exception as e:
                if debug_logger:
                    try:
                        debug_logger({
                            **safe_context,
                            "attempt": attempt_index,
                            "max_retries": max_retries,
                            "success": False,
                            "error": str(e),
                            "request_messages": messages,
                            "response": None,
                            "attempt_latency": time.time() - attempt_start
                        })
                    except Exception as log_error:
                        print(f"Warning: Failed to write LLM debug log: {log_error}")

                if attempt < max_retries - 1:
                    print(f"LLM call failed (attempt {attempt_index}/{max_retries}): {e}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    print(f"LLM call failed after {max_retries} attempts: {e}")
                    return None
        return None

    def call_with_json_schema(self,
                             messages: List[Dict[str, str]],
                             schema: Dict[str, Any],
                             **kwargs) -> LLMResponse:
        """
        带JSON schema的LLM调用（用于结构化输出）

        Args:
            messages: 消息列表
            schema: JSON schema
            **kwargs: 额外的参数

        Returns:
            LLMResponse对象
        """
        params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": schema
                }
            }
        }
        params.update(kwargs)

        return self.call(messages, **params)


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    从配置创建LLM客户端

    Args:
        config: 配置字典

    Returns:
        LLMClient实例
    """
    return LLMClient(
        model=config.get("llm_model", "gpt-4"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        temperature=config.get("llm_temperature", 0.7),
        max_tokens=config.get("llm_max_tokens", 8000),
        timeout=config.get("llm_timeout", 60)
    )


def load_client_from_env(
    env_path: str = ".env",
    provider: str = "auto",
    temperature: Optional[float] = None,
) -> LLMClient:
    """
    从.env文件加载LLM客户端

    Args:
        env_path: .env文件路径 (相对于项目根目录)
        provider: 提供商选择 ("auto", "siliconcloud", "dmxapi")
                 - "auto": 自动选择 (优先SILICONCLOUD，备选DMXAPI)
                 - "siliconcloud": 使用SILICONCLOUD
                 - "dmxapi": 使用DMXAPI
        temperature: 可选温度覆盖；若为None则读取环境变量 LLM_TEMPERATURE

    Returns:
        LLMClient实例

    Raises:
        ValueError: 如果必要的环境变量未设置
    """
    from dotenv import load_dotenv

    # 加载.env文件
    load_dotenv(env_path)

    # 选择provider
    if provider == "auto":
        # 自动选择：优先SILICONCLOUD
        if os.getenv("SILICONCLOUD_BASE_URL") and os.getenv("SILICONCLOUD_API_KEY") and os.getenv("SILICONCLOUD_MODEL"):
            provider = "siliconcloud"
        elif os.getenv("DMXAPI_BASE_URL") and os.getenv("DMXAPI_API_KEY") and os.getenv("DMXAPI_MODEL"):
            provider = "dmxapi"
        else:
            raise ValueError("No valid provider configuration found in .env file")

    # 从环境变量读取对应provider的配置
    if provider.lower() == "siliconcloud":
        base_url = os.getenv("SILICONCLOUD_BASE_URL")
        api_key = os.getenv("SILICONCLOUD_API_KEY")
        model = os.getenv("SILICONCLOUD_MODEL")

        if not all([base_url, api_key, model]):
            raise ValueError("Missing SILICONCLOUD environment variables (SILICONCLOUD_BASE_URL, SILICONCLOUD_API_KEY, SILICONCLOUD_MODEL)")

    elif provider.lower() == "dmxapi":
        base_url = os.getenv("DMXAPI_BASE_URL")
        api_key = os.getenv("DMXAPI_API_KEY")
        model = os.getenv("DMXAPI_MODEL")

        if not all([base_url, api_key, model]):
            raise ValueError("Missing DMXAPI environment variables (DMXAPI_BASE_URL, DMXAPI_API_KEY, DMXAPI_MODEL)")
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: auto, siliconcloud, dmxapi")

    resolved_temperature: Optional[float] = temperature
    if resolved_temperature is None:
        raw_temperature = os.getenv("LLM_TEMPERATURE")
        if raw_temperature is not None and str(raw_temperature).strip():
            try:
                resolved_temperature = float(raw_temperature)
            except (TypeError, ValueError):
                resolved_temperature = None

    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=resolved_temperature if resolved_temperature is not None else 0.7,
        max_tokens=8000  # 增加到 8000 以支持完整的 JSON 输出
    )


if __name__ == "__main__":
    # 测试代码
    client = LLMClient(model="gpt-4")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'Hello, World!'"}
    ]

    response = client.call_with_retry(messages, max_retries=3)

    if response:
        print(f"Response: {response.content}")
        print(f"Tokens: {response.total_tokens}")
        print(f"Latency: {response.latency:.2f}s")
    else:
        print("Failed to get response")
