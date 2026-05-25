"""
Application configuration settings for TPE system.

Environment-based configuration management with validation.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass


def _get_env_value(*names: str, default: str = "") -> str:
    """Return the first non-empty environment value."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class Settings(BaseModel):
    """Application settings with environment variable support."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)
    api_title: str = "Travel Plan Editing System API"
    api_version: str = "1.0.0"

    # LLM Configuration (SiliconFlow)
    openai_api_base: str = Field(
        default_factory=lambda: _get_env_value(
            "DMXAPI_BASE_URL",
            "SILICONCLOUD_BASE_URL",
            default="https://api.siliconflow.cn/v1",
        ),
        description="OpenAI-compatible API base URL",
    )
    openai_api_key: str = Field(
        default_factory=lambda: _get_env_value(
            "DMXAPI_API_KEY",
            "SILICONCLOUD_API_KEY",
        ),
        description="OpenAI-compatible API key",
    )
    llm_model: str = Field(
        default_factory=lambda: _get_env_value(
            "DMXAPI_MODEL",
            "SILICONCLOUD_MODEL",
            default="gpt-4",
        ),
        description="LLM model name",
    )
    llm_timeout: int = Field(default=30, description="LLM request timeout")
    llm_max_retries: int = Field(default=3, description="LLM request max retries")

    # File Storage Configuration
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    # Note: session/audit/cache directories are deprecated and no longer used
    
    # TPE Dataset Configuration  
    tpe_dataset_dir: Path = Field(default=Path("./data/tpe_dataset"), description="TPE dataset directory")
    tpe_base_plans_dir: Path = Field(default=Path("./data/tpe_dataset/base_plans_qwen3-8b"), description="TPE base plans directory")
    tpe_episodes_dir: Path = Field(default=Path("./data/tpe_dataset/episodes/episodes"), description="TPE episodes directory")

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    log_file: Optional[Path] = Field(default=None, description="Log file path")

    # ChinaTravel Integration Configuration
    use_chinatravel_data: bool = Field(
        default=False, description="Enable ChinaTravel data providers"
    )
    chinatravel_data_path: Optional[Path] = Field(
        default=None, description="Path to ChinaTravel database files"
    )

    # Performance Configuration
    max_concurrent_sessions: int = Field(
        default=1000, description="Max concurrent sessions"
    )
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    solver_timeout: int = Field(default=30, description="Solver timeout in seconds")
    max_edit_candidates: int = Field(default=5, description="Max edit candidates")

    # Security Configuration
    api_key_required: bool = Field(default=False, description="API key required")
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")

    # Feature Flags
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    enable_tracing: bool = Field(default=True, description="Enable tracing")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def validate_data_dir(cls, v):
        """Ensure data directory exists and is writable."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f"Data directory {path} is not writable")
        return path

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is one of allowed values."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        level = v.upper()
        if level not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return level

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def validate_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    def create_directories(self) -> None:
        """Create all necessary directories based on configuration."""
        directories = [
            self.data_dir,
            self.tpe_dataset_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def siliconcloud_api_key(self) -> str:
        """Backward-compatible alias used by legacy health checks."""
        return self.openai_api_key

    @property
    def storage_path(self) -> Path:
        """Backward-compatible alias used by old health route."""
        return self.data_dir


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
