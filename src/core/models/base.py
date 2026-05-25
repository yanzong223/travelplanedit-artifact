"""
Base models and common types for TPE system.

Provides fundamental Pydantic models used throughout the system.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

# Type alias for time windows
TimeWindow = Tuple[str, str]  # e.g. ("10:30", "13:00")


class BaseTPEModel(BaseModel):
    """Base model for all TPE system models."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique identifier"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override to ensure datetime serialization."""
        data = super().model_dump(**kwargs)
        # Ensure datetime fields are serialized as ISO format
        for field_name, field_value in data.items():
            if isinstance(field_value, datetime):
                data[field_name] = field_value.isoformat()
        return data


class AuditEntry(BaseTPEModel):
    """Audit trail entry for all system actions."""

    agent_type: str = Field(description="Type of agent that performed the action")
    action: str = Field(description="Action performed")
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the action"
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Output data from the action"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Reasoning behind the action"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score"
    )

    # Context fields
    correlation_id: Optional[str] = Field(
        default=None, description="Correlation ID for request tracing"
    )
    request_id: Optional[str] = Field(default=None, description="Request ID")
    session_id: Optional[str] = Field(default=None, description="Session ID")


class SystemMetrics(BaseTPEModel):
    """Performance and operational metrics."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Metrics timestamp"
    )
    active_plans: int = Field(default=0, description="Number of active travel plans")
    concurrent_sessions: int = Field(
        default=0, description="Number of concurrent sessions"
    )
    avg_response_time: float = Field(
        default=0.0, description="Average response time in seconds"
    )
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Success rate")
    error_counts: Dict[str, int] = Field(
        default_factory=dict, description="Error counts by type"
    )


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    is_valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


class PaginationParams(BaseModel):
    """Pagination parameters for API requests."""

    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(
        default=50, ge=1, le=1000, description="Maximum number of items to return"
    )

    @property
    def slice_start(self) -> int:
        """Get the start index for slicing."""
        return self.offset

    @property
    def slice_end(self) -> int:
        """Get the end index for slicing."""
        return self.offset + self.limit


class PaginatedResult(BaseModel):
    """Paginated result wrapper."""

    items: List[Any] = Field(
        default_factory=list, description="Items in the current page"
    )
    total: int = Field(default=0, description="Total number of items")
    offset: int = Field(default=0, description="Current offset")
    limit: int = Field(default=50, description="Current limit")
    has_next: bool = Field(default=False, description="Whether there are more items")
    has_prev: bool = Field(
        default=False, description="Whether there are previous items"
    )

    @classmethod
    def create(
        cls,
        items: List[Any],
        total: int,
        pagination: PaginationParams,
    ) -> "PaginatedResult":
        """Create a paginated result."""
        return cls(
            items=items,
            total=total,
            offset=pagination.offset,
            limit=pagination.limit,
            has_next=pagination.offset + pagination.limit < total,
            has_prev=pagination.offset > 0,
        )


class Flaw(BaseModel):
    """Represents a flaw/issue in a travel plan that requires editing.

    A Flaw can be:
    - A violated constraint (conflict_variables in old design)
    - An unbound time interval that needs activities (temp_sparse)
    - Other types of issues (extensible)

    Flaws serve as anchors/seeds for scope selection in editing.
    """

    kind: Literal[
        "violated_constraint",  # Original conflict variable
        "unbound_interval",  # Time window lacking activities (temp_sparse)
        # Future extensions:
        # "category_sparse",
        # "resource_overuse",
    ] = Field(description="Type of flaw")

    # For violated_constraint: the problematic variable
    var_id: Optional[str] = Field(
        default=None, description="Variable ID for violated_constraint flaws"
    )

    # For unbound_interval: the time window
    time_window: Optional[TimeWindow] = Field(
        default=None, description="Time window (start, end) for unbound_interval flaws"
    )

    # Anchor slots: adjacent slots before/after the time window
    anchor_slots: List[str] = Field(
        default_factory=list,
        description="IDs of slots adjacent to the flaw (for context)",
    )

    # Location hint for retrieval
    location_hint: Optional[str] = Field(
        default=None, description="Location hint (e.g., POI name or coordinates)"
    )

    # For debugging and logging
    explanation: str = Field(
        default="", description="Human-readable explanation of the flaw"
    )

    # Additional context
    day_index: Optional[int] = Field(
        default=None, description="Day index in the plan (0-based)"
    )

    constraint_ids: List[str] = Field(
        default_factory=list, description="Related constraint IDs"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "kind": "violated_constraint",
                    "var_id": "activity_d1_a1",
                    "explanation": "Temporal constraint violation",
                },
                {
                    "kind": "unbound_interval",
                    "time_window": ("10:30", "13:00"),
                    "anchor_slots": ["activity_d1_a1", "activity_d1_a2"],
                    "location_hint": "水利博物馆",
                    "day_index": 0,
                    "explanation": "temp_sparse[(10:30, 13:00)]",
                },
            ]
        }
