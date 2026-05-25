"""
Enumerations for TPE system.

Defines all enum types used throughout the travel plan editing system.
"""

from enum import Enum


class VariableType(str, Enum):
    """Types of variable nodes in the factor graph."""

    FLIGHT = "flight"
    HOTEL = "hotel"
    ACTIVITY = "activity"
    TIME = "time"
    LOCATION = "location"
    TRANSPORT = "transport"


class ConstraintType(str, Enum):
    """Types of constraint nodes in the factor graph."""

    TEMPORAL = "temporal"  # Time-based constraints
    SPATIAL = "spatial"  # Location/distance constraints
    LOGICAL = "logical"  # Logical dependencies
    BUDGET = "budget"  # Cost constraints
    PREFERENCE = "preference"  # User preference constraints


class EdgeType(str, Enum):
    """Types of edges in the factor graph."""

    AFFECTS = "affects"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"


class EditType(str, Enum):
    """Types of edit operations."""

    ADD = "add"
    MODIFY = "modify"
    REMOVE = "remove"
    REORDER = "reorder"


class OperationType(str, Enum):
    """Specific operation types for structured edit candidates."""

    INSERT = "insert"  # Add new element
    REPLACE = "replace"  # Replace existing element
    ADJUST_TIME = "adjust_time"  # Modify timing
    MODE_CHANGE = "mode_change"  # Change transportation/accommodation mode


class AgentType(str, Enum):
    """Types of agents in the system."""

    CONSTRAINT_EXTRACTOR = "constraint_extractor"
    CONSTRAINT_VERIFIER = "constraint_verifier"
    CONFLICT_CLASSIFIER = "conflict_classifier"
    SCOPE_SELECTOR = "scope_selector"
    LOCAL_EDITOR = "local_editor"
    RETRIEVAL_AGENT = "retrieval_agent"
    ORCHESTRATOR = "orchestrator"


class WorkflowPhase(str, Enum):
    """Phases in the workflow orchestration."""

    INTENT_EXTRACTION = "intent_extraction"
    SCOPE_SELECTION = "scope_selection"
    CONSTRAINT_VERIFICATION = "constraint_verification"
    CONFLICT_CLASSIFICATION = "conflict_classification"
    DATA_RETRIEVAL = "data_retrieval"
    EDIT_GENERATION = "edit_generation"
    VALIDATION = "validation"
    CONVERGENCE = "convergence"


class TaskStatus(str, Enum):
    """Status of agent tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionStatus(str, Enum):
    """Status of workflow sessions."""

    ACTIVE = "active"
    CONVERGED = "converged"
    FAILED = "failed"
    ESCALATED = "escalated"


class PlanStatus(str, Enum):
    """Status of travel plans."""

    ACTIVE = "active"
    DRAFT = "draft"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Priority(str, Enum):
    """Priority levels for requests and tasks."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConflictType(str, Enum):
    """Types of conflicts identified by the conflict classifier."""

    MISSING_DATA = "missing_data"  # Missing information needed for validation
    LOGICAL_CONFLICT = "logical_conflict"  # Inconsistent constraints/requirements
    TEMPORAL_CONFLICT = "temporal_conflict"  # Time-based conflicts
    SPATIAL_CONFLICT = "spatial_conflict"  # Location/distance conflicts
    BUDGET_CONFLICT = "budget_conflict"  # Budget constraint violations


class FlawType(str, Enum):
    """Types of flaws/issues detected in a plan that need editing.

    Flaws represent both conflicts (violations) and gaps (missing elements).
    """

    # Constraint violations (conflicts)
    CONSTRAINT_VIOLATION = "constraint_violation"  # A hard constraint is violated
    TEMPORAL_CONFLICT = "temporal_conflict"  # Time overlap or ordering issue
    SPATIAL_CONFLICT = "spatial_conflict"  # Location/distance issue
    BUDGET_EXCEEDED = "budget_exceeded"  # Budget limit exceeded
    LOGICAL_INCONSISTENCY = "logical_inconsistency"  # Logical contradiction

    # Gaps and missing elements
    TEMPORAL_GAP = "temporal_gap"  # Time period without activities (temp_sparse)
    MISSING_ACTIVITY = "missing_activity"  # Required activity is missing
    MISSING_TRANSPORT = "missing_transport"  # Required transport is missing
    INCOMPLETE_ITINERARY = "incomplete_itinerary"  # Itinerary is incomplete

    # Suboptimal situations
    INEFFICIENT_ROUTE = "inefficient_route"  # Route can be optimized
    SUBOPTIMAL_TIMING = "suboptimal_timing"  # Timing can be improved

    # User requests
    USER_ADDITION = "user_addition"  # User wants to add something
    USER_MODIFICATION = "user_modification"  # User wants to modify something
    USER_REMOVAL = "user_removal"  # User wants to remove something


class CoreType(str, Enum):
    """Core types for explainability in conflict classification."""

    TIME_CONSTRAINT = "time_constraint"
    BUDGET_CONSTRAINT = "budget_constraint"
    LOGIC_CONSTRAINT = "logic_constraint"
    PREFERENCE_CONSTRAINT = "preference_constraint"
    EXTERNAL_DEPENDENCY = "external_dependency"


class LogLevel(str, Enum):
    """Log levels for the system."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditRetention(str, Enum):
    """Audit trail retention policies."""

    SESSION_END = "until_session_end"  # Retain until session reaches terminal state
    PERMANENT = "permanent"  # Retain indefinitely
    DAYS_30 = "30_days"  # Retain for 30 days
    DAYS_90 = "90_days"  # Retain for 90 days
