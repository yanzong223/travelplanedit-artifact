"""Standalone edit framework contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EditInput(BaseModel):
    """Minimal runtime input for standalone edit baselines."""

    case_id: str
    origin_plan: Dict[str, Any]
    origin_query_text: str
    edit_query: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EditResult(BaseModel):
    """Normalized runtime output for standalone edit baselines."""

    success: bool
    framework: str
    exposure_mode: Optional[str] = None
    tool_profile: Optional[str] = None
    db_read_enabled: Optional[bool] = None
    special_backend: Optional[str] = None
    edited_plan: Optional[Dict[str, Any]] = None
    ops: List[Dict[str, Any]] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class EditFramework(ABC):
    """Common async interface for standalone edit baselines."""

    framework_name: str

    @abstractmethod
    async def run(self, edit_input: EditInput) -> EditResult:
        """Execute one edit case."""
