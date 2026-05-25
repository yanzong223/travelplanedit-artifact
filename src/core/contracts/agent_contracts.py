"""
Agent data contracts for TPE system.

Defines Pydantic models for all agent inputs and outputs to ensure
type safety and runtime validation.

DESIGN PRINCIPLE:
- All agents use these standardized contracts as their native I/O format
- No intermediate conversions or adapters needed
- Type-safe at compile time with Pydantic validation at runtime
- Single source of truth for all agent interfaces
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from core.models.base import BaseTPEModel
from core.models.enums import ConstraintType, VariableType

# ============================================================================
# Base Contracts
# ============================================================================


class AgentInput(BaseTPEModel):
    """Base class for all agent inputs."""

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this request",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the request was made"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the request"
    )


class AgentOutput(BaseTPEModel):
    """Base class for all agent outputs."""

    request_id: str = Field(description="Reference to the original request")
    success: bool = Field(description="Whether the operation succeeded")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the response was generated"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of error messages if any"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of warnings if any"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the operation"
    )


# ============================================================================
# Constraint Extractor Contracts
# ============================================================================


class ExtractedConstraint(BaseTPEModel):
    """A single extracted constraint from natural language."""

    constraint_type: str = Field(
        description="Type of constraint (budget, time, preference, etc.)"
    )
    description: str = Field(
        description="Natural language description of the constraint"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Structured parameters for the constraint"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Priority of the constraint (1-10)"
    )
    is_hard_constraint: bool = Field(
        default=True,
        description="Whether this is a hard constraint (must be satisfied)",
    )

    @field_validator("constraint_type")
    @classmethod
    def validate_constraint_type(cls, v):
        """Validate constraint type."""
        valid_types = [
            "budget",
            "time",
            "preference",
            "location",
            "activity",
            "accommodation",
            "transportation",
            "duration",
            "schedule",
            "group_size",
            "other",
        ]
        if v.lower() not in valid_types:
            return "other"
        return v.lower()


class MissingInformation(BaseTPEModel):
    """Information that is missing or ambiguous in the request."""

    field: str = Field(description="What information is missing")
    reason: str = Field(description="Why this information is needed")
    suggestions: List[str] = Field(
        default_factory=list, description="Possible values or clarification questions"
    )


class ConstraintExtractorInput(AgentInput):
    """Input for constraint extraction."""

    natural_language_query: str = Field(
        description="User's edit request in natural language"
    )
    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional user context (preferences, history, etc.)",
    )


class ConstraintExtractorOutput(AgentOutput):
    """Output from constraint extraction."""

    constraints: List[ExtractedConstraint] = Field(
        default_factory=list, description="List of extracted constraints"
    )
    missing_information: List[MissingInformation] = Field(
        default_factory=list, description="List of missing or ambiguous information"
    )
    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extraction (0-1)",
    )
    inferred_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context inferred from plan: cities, dates, locations, etc."
    )


# ============================================================================
# Conflict Classifier Contracts
# ============================================================================


class ConflictType(str, Enum):
    """Types of conflicts identified."""

    LOGICAL_ISSUE = "LOGICAL_ISSUE"  # Logical conflict in the plan
    MISSING_INFO = "MISSING_INFO"  # Need external information retrieval


class RetrievalInstruction(BaseTPEModel):
    """Instruction for retrieving external information."""

    retrieval_type: str = Field(
        description="Type of retrieval (search_flights, search_hotels, search_activities)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the retrieval operation"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Priority of this retrieval (1-10)"
    )
    rationale: str = Field(default="", description="Why this retrieval is needed")

    @field_validator("retrieval_type")
    @classmethod
    def validate_retrieval_type(cls, v):
        """Validate retrieval type."""
        valid_types = [
            "search_flights",
            "search_hotels",
            "search_activities",
            "search_restaurants",
            "search_attractions",
            "search_transportation",
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid retrieval type: {v}")
        return v


class ConflictClassifierInput(AgentInput):
    """Input for conflict classification."""

    plan_data: Dict[str, Any] = Field(
        description="Serialized travel plan (FactorGraph)"
    )
    structured_constraints: List[Dict[str, Any]] = Field(
        description="Structured constraints from extractor"
    )
    conflict_variables: List[str] = Field(
        default_factory=list,
        description="Variables known to have conflicts (from verifier)",
    )


class ConflictClassifierOutput(AgentOutput):
    """Output from conflict classification."""

    conflict_type: ConflictType = Field(description="Type of conflict identified")
    conflict_reason: str = Field(description="Detailed explanation of the conflict")
    retrieval_instructions: List[RetrievalInstruction] = Field(
        default_factory=list, description="Instructions for retrieval if MISSING_INFO"
    )
    affected_entities: List[str] = Field(
        default_factory=list,
        description="Entities (variables) affected by the conflict",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the classification (0-1)",
    )

    @model_validator(mode="after")
    def validate_retrieval_instructions(self):
        """Ensure retrieval instructions are present for MISSING_INFO."""
        if self.conflict_type == ConflictType.MISSING_INFO:
            if not self.retrieval_instructions:
                self.warnings.append(
                    "MISSING_INFO conflict but no retrieval instructions provided"
                )
        return self


# ============================================================================
# Verifier Contracts
# ============================================================================


class ViolationDetail(BaseTPEModel):
    """Details about a constraint violation."""

    constraint_id: str = Field(description="ID of the violated constraint")
    variable_ids: List[str] = Field(description="Variables involved in the violation")
    violation_type: str = Field(description="Type of violation")
    description: str = Field(description="Description of the violation")
    severity: int = Field(
        default=5, ge=1, le=10, description="Severity of the violation (1-10)"
    )


class VerifierInput(AgentInput):
    """Input for constraint verification."""

    plan_data: Dict[str, Any] = Field(
        description="Serialized travel plan (FactorGraph)"
    )
    structured_constraints: List[Dict[str, Any]] = Field(
        description="All constraints to verify (original + new)"
    )


class VerifierOutput(AgentOutput):
    """Output from constraint verification."""

    has_conflicts: bool = Field(description="Whether any conflicts exist")
    conflict_variables: List[str] = Field(
        default_factory=list,
        description="Variable IDs involved in conflicts (for scope selector)",
    )
    satisfied_constraints: List[str] = Field(
        default_factory=list, description="Constraint IDs that are satisfied"
    )
    violated_constraints: List[str] = Field(
        default_factory=list, description="Constraint IDs that are violated"
    )
    violation_details: List[ViolationDetail] = Field(
        default_factory=list, description="Detailed information about each violation"
    )
    verification_summary: str = Field(
        default="", description="Summary of verification results"
    )


# ============================================================================
# Scope Selector Contracts
# ============================================================================


class SubgraphData(BaseTPEModel):
    """Data representing an affected subgraph."""

    variable_ids: List[str] = Field(description="Variable IDs in the subgraph")
    constraint_ids: List[str] = Field(description="Constraint IDs in the subgraph")
    edges: List[tuple[str, str]] = Field(
        default_factory=list,
        description="Edges in the subgraph (variable_id, constraint_id)",
    )
    depth: int = Field(
        default=1, ge=0, description="Propagation depth from seed variables"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional subgraph metadata"
    )


class ScopeSelectorInput(AgentInput):
    """Input for scope selection."""

    plan_data: Dict[str, Any] = Field(description="Serialized constraint factor graph")
    conflict_variables: List[str] = Field(
        description="Seed variables with conflicts (from verifier)"
    )
    max_depth: int = Field(
        default=2, ge=1, le=5, description="Maximum propagation depth"
    )


class ScopeSelectorOutput(AgentOutput):
    """Output from scope selection."""

    affected_subgraph: SubgraphData = Field(description="Selected subgraph for editing")
    scope_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Metrics about the selected scope"
    )
    excluded_variables: List[str] = Field(
        default_factory=list,
        description="Variables excluded from scope (too far from conflicts)",
    )


# ============================================================================
# Retrieval Agent Contracts
# ============================================================================


class FlightInfo(BaseTPEModel):
    """Information about a flight option."""

    flight_number: str = Field(description="Flight number")
    airline: str = Field(description="Airline name")
    departure_city: str = Field(description="Departure city")
    arrival_city: str = Field(description="Arrival city")
    departure_time: str = Field(description="Departure time")
    arrival_time: str = Field(description="Arrival time")
    price: float = Field(description="Price in local currency")
    duration_minutes: int = Field(description="Flight duration in minutes")
    available_seats: int = Field(default=0, description="Available seats")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HotelInfo(BaseTPEModel):
    """Information about a hotel option."""

    hotel_name: str = Field(description="Hotel name")
    city: str = Field(description="City location")
    address: str = Field(default="", description="Street address")
    price_per_night: float = Field(description="Price per night in local currency")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Star rating")
    amenities: List[str] = Field(
        default_factory=list, description="Available amenities"
    )
    available_rooms: int = Field(default=0, description="Available rooms")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActivityInfo(BaseTPEModel):
    """Information about an activity option."""

    activity_name: str = Field(description="Activity name")
    city: str = Field(description="City location")
    category: str = Field(description="Activity category")
    description: str = Field(default="", description="Activity description")
    duration_minutes: int = Field(description="Activity duration in minutes")
    price: float = Field(description="Price in local currency")
    available_slots: int = Field(default=0, description="Available time slots")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalInput(AgentInput):
    """Input for retrieval operations."""

    retrieval_instructions: List[RetrievalInstruction] = Field(
        description="List of retrieval instructions to execute"
    )
    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="User context for retrieval (preferences, constraints)",
    )


class RetrievalOutput(AgentOutput):
    """Output from retrieval operations."""

    flights: List[FlightInfo] = Field(
        default_factory=list, description="Retrieved flight options"
    )
    hotels: List[HotelInfo] = Field(
        default_factory=list, description="Retrieved hotel options"
    )
    activities: List[ActivityInfo] = Field(
        default_factory=list, description="Retrieved activity options"
    )
    retrieval_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Summary of retrieval results (counts by type)",
    )
    execution_time_ms: float = Field(
        default=0.0, description="Total retrieval execution time in milliseconds"
    )


# ============================================================================
# Local Editor Contracts
# ============================================================================


class OperationType(str, Enum):
    """Types of atomic edit operations."""

    INSERT = "Insert"
    REPLACE = "Replace"
    DELETE = "Delete"
    ADJUST_TIME = "AdjustTime"
    MODE_CHANGE = "ModeChange"
    UPDATE_CONSTRAINT = "UpdateConstraint"


class AtomicOperation(BaseTPEModel):
    """A single atomic edit operation."""

    operation_type: OperationType = Field(description="Type of operation")
    target_id: str = Field(description="ID of the variable or constraint to edit")
    changes: Dict[str, Any] = Field(
        default_factory=dict, description="Changes to apply"
    )
    reasoning: str = Field(description="Explanation for this operation")
    dependencies: List[int] = Field(
        default_factory=list, description="Indices of operations this depends on"
    )
    expected_outcome: str = Field(
        default="", description="Expected outcome after applying this operation"
    )


class ExecutionPlan(BaseTPEModel):
    """Plan for executing atomic operations."""

    total_operations: int = Field(description="Total number of operations")
    execution_order: List[int] = Field(
        description="Order in which to execute operations (indices)"
    )
    estimated_time_ms: float = Field(
        default=0.0, description="Estimated execution time in milliseconds"
    )
    rollback_strategy: str = Field(
        default="sequential", description="Strategy for rolling back on failure"
    )


class LocalEditorInput(AgentInput):
    """Input for local editor."""

    conflict_reason: str = Field(description="Reason for conflicts from classifier")
    retrieval_info: Dict[str, Any] = Field(
        default_factory=dict, description="Information retrieved from retrieval agent"
    )
    plan_data: Dict[str, Any] = Field(description="Serialized constraint factor graph")
    affected_subgraph: Optional[SubgraphData] = Field(
        default=None, description="Affected subgraph from scope selector (optional)"
    )
    user_constraints: List[Dict[str, Any]] = Field(
        default_factory=list, description="User's constraints to satisfy"
    )


class LocalEditorOutput(AgentOutput):
    """Output from local editor."""

    atomic_operations: List[AtomicOperation] = Field(
        default_factory=list, description="Sequence of atomic edit operations"
    )
    execution_plan: ExecutionPlan = Field(description="Plan for executing operations")
    edit_summary: str = Field(default="", description="Summary of proposed edits")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the edit plan (0-1)"
    )
    alternative_plans: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative edit plans if available"
    )
