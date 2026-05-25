"""
Travel plan data models for TPE system.

Implements the core travel plan structure and related entities.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseTPEModel
from .enums import PlanStatus

if TYPE_CHECKING:
    from .factor_graph import FactorGraph


class TravelPlan(BaseTPEModel):
    """Complete travel plan represented as a factor graph."""

    title: str = Field(description="Title of the travel plan")
    description: Optional[str] = Field(
        default=None, description="Description of the travel plan"
    )
    user_id: str = Field(description="User ID who owns this plan")
    graph: "FactorGraph" = Field(description="Factor graph representation")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    status: PlanStatus = Field(
        default=PlanStatus.DRAFT, description="Current status of the plan"
    )
    version: int = Field(default=1, description="Version number of the plan")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v):
        """Ensure version is positive."""
        if v < 1:
            raise ValueError("Version must be a positive integer")
        return v


# Update __all__ for proper import
__all__ = ["TravelPlan"]
