"""
LLM Client Pool for parallel processing with multiple API keys.

支持从 pllm.yaml 配置文件加载多个 API key，实现并行请求处理。
同时提供与单个客户端兼容的接口。
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel

from llm.client import SiliconCloudClient
from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClientPool:
    """管理多个 LLM 客户端的池，支持并行请求处理"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化 LLM 客户端池

        Args:
            config_path: pllm.yaml 配置文件路径，默认为项目根目录下的 pllm.yaml
        """
        self.config_path = config_path or Path(__file__).parent.parent.parent / "pllm.yaml"
        self.clients: List[SiliconCloudClient] = []
        self.client_configs: List[Dict[str, Any]] = []
        self._semaphores: List[asyncio.Semaphore] = []
        self._current_index = 0
        self._lock = asyncio.Lock()
        
        # 加载配置并初始化客户端
        self._load_config()
        self._initialize_clients()
        
        # 设置默认模型（从第一个客户端获取）
        self.default_model = self.clients[0].default_model if self.clients else None
        
    def _load_config(self):
        """从 pllm.yaml 加载配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 获取使用的 LLM 提供商
        llm_provider = config.get("llm", {}).get("use", "siliconflow")
        
        if llm_provider not in config.get("llm", {}):
            raise ValueError(f"配置文件中未找到提供商 '{llm_provider}' 的配置")
        
        # 获取该提供商的所有 API 配置
        self.client_configs = config["llm"][llm_provider]
        
        if not self.client_configs:
            raise ValueError(f"提供商 '{llm_provider}' 没有配置任何 API key")
        
        logger.debug(f"从配置文件加载了 {len(self.client_configs)} 个 API 配置")
        
    def _initialize_clients(self):
        """初始化所有客户端和对应的信号量"""
        for i, config in enumerate(self.client_configs):
            try:
                api_key = config.get("api_key")
                api_base = config.get("api_base")
                rate_limit = config.get("rate_limit", 100)
                
                if not api_key:
                    logger.warning(f"配置 {i} 缺少 api_key，跳过")
                    continue
                
                # 创建客户端
                client = SiliconCloudClient(api_key=api_key, base_url=api_base)
                
                # 设置默认模型
                if "model" in config:
                    client.default_model = config["model"]
                
                self.clients.append(client)
                
                # 创建信号量以限制并发请求数（根据 rate_limit）
                # 假设每秒最多 rate_limit 个请求，我们设置并发限制为 rate_limit / 2
                max_concurrent = max(1, rate_limit // 2)
                self._semaphores.append(asyncio.Semaphore(max_concurrent))
                
                logger.debug(
                    f"初始化客户端 {i+1}/{len(self.client_configs)}: "
                    f"模型={config.get('model', 'default')}, "
                    f"rate_limit={rate_limit}, "
                    f"max_concurrent={max_concurrent}"
                )
                
            except Exception as e:
                logger.error(f"初始化客户端 {i} 失败: {e}")
                continue
        
        if not self.clients:
            raise RuntimeError("没有成功初始化任何 LLM 客户端")
        
        logger.info(f"成功初始化 {len(self.clients)} 个 LLM 客户端")  # 保留此重要信息
    
    def get_client_count(self) -> int:
        """获取可用客户端数量"""
        return len(self.clients)
    
    async def _get_next_client(self) -> tuple[SiliconCloudClient, asyncio.Semaphore]:
        """轮询获取下一个可用的客户端（负载均衡）"""
        async with self._lock:
            client = self.clients[self._current_index]
            semaphore = self._semaphores[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.clients)
            return client, semaphore
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_model: Optional[Type[T]] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> Union[str, T, Dict[str, Any]]:
        """
        使用客户端池进行聊天补全

        Args:
            messages: 消息列表
            model: 模型名称（可选，使用客户端的默认模型）
            temperature: 温度参数
            max_tokens: 最大 token 数
            tools: 工具列表（function calling）
            response_model: 结构化输出的 Pydantic 模型
            retry_count: 重试次数
            retry_delay: 重试延迟

        Returns:
            LLM 响应
        """
        # 获取一个客户端
        client, semaphore = await self._get_next_client()

        # 使用信号量控制并发
        async with semaphore:
            try:
                response = await client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    response_model=response_model,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                )
                return response
            except Exception as e:
                logger.error(f"客户端请求失败: {e}")
                raise
    
    async def test_connection(self) -> bool:
        """测试连接（测试第一个客户端）"""
        if not self.clients:
            return False
        try:
            return await self.clients[0].test_connection()
        except Exception:
            return False
    
    async def test_all_clients(self) -> Dict[int, bool]:
        """测试所有客户端的连接"""
        results = {}
        
        async def test_client(index: int, client: SiliconCloudClient):
            try:
                success = await client.test_connection()
                results[index] = success
                if not success:
                    logger.warning(f"客户端 {index+1} 连接测试失败")
            except Exception as e:
                results[index] = False
                logger.error(f"客户端 {index+1} 连接测试失败: {e}")
        
        # 并行测试所有客户端
        tasks = [test_client(i, client) for i, client in enumerate(self.clients)]
        await asyncio.gather(*tasks)
        
        return results


# 全局客户端池实例
_client_pool: Optional[LLMClientPool] = None
_use_pool: bool = False


def enable_client_pool(config_path: Optional[Path] = None):
    """启用客户端池模式"""
    global _use_pool, _client_pool
    _use_pool = True
    if _client_pool is None:
        _client_pool = LLMClientPool(config_path=config_path)


def disable_client_pool():
    """禁用客户端池模式"""
    global _use_pool
    _use_pool = False


def get_client_pool(config_path: Optional[Path] = None) -> LLMClientPool:
    """获取全局客户端池实例（单例模式）"""
    global _client_pool
    if _client_pool is None:
        _client_pool = LLMClientPool(config_path=config_path)
    return _client_pool


def is_pool_enabled() -> bool:
    """检查是否启用了客户端池"""
    return _use_pool and _client_pool is not None


async def initialize_client_pool(config_path: Optional[Path] = None) -> bool:
    """初始化并测试客户端池"""
    try:
        pool = get_client_pool(config_path=config_path)
        results = await pool.test_all_clients()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        logger.info(f"客户端池初始化完成: {success_count}/{total_count} 个客户端可用")
        
        return success_count > 0
    except Exception as e:
        logger.error(f"初始化客户端池失败: {e}")
        return False
