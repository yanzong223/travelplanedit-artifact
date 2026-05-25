"""
Data adapters for converting between legacy formats and new contracts.

Provides adapter functions to bridge the gap between existing agent
implementations and the new standardized contract interfaces.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.factor_graph import FactorGraph

from .agent_contracts import (  # Constraint Extractor; Conflict Classifier; Verifier; Scope Selector; Retrieval; Local Editor
    ActivityInfo,
    AtomicOperation,
    ConflictClassifierInput,
    ConflictClassifierOutput,
    ConflictType,
    ConstraintExtractorInput,
    ConstraintExtractorOutput,
    ExecutionPlan,
    ExtractedConstraint,
    FlightInfo,
    HotelInfo,
    LocalEditorInput,
    LocalEditorOutput,
    MissingInformation,
    OperationType,
    RetrievalInput,
    RetrievalInstruction,
    RetrievalOutput,
    ScopeSelectorInput,
    ScopeSelectorOutput,
    SubgraphData,
    VerifierInput,
    VerifierOutput,
    ViolationDetail,
)

# ============================================================================
# Constraint Extractor Adapters
# ============================================================================


def adapt_constraint_extractor_input(
    natural_language_query: str,
    user_context: Optional[Dict[str, Any]] = None,
) -> ConstraintExtractorInput:
    """
    Convert legacy constraint extractor input to contract format.

    Args:
        natural_language_query: User's edit request
        user_context: Additional user context

    Returns:
        ConstraintExtractorInput contract instance
    """
    return ConstraintExtractorInput(
        request_id=str(uuid.uuid4()),
        natural_language_query=natural_language_query,
        user_context=user_context or {},
    )


def adapt_constraint_extractor_output(
    legacy_output: Dict[str, Any],
    request_id: str,
) -> ConstraintExtractorOutput:
    """
    Convert legacy constraint extractor output to contract format.

    Args:
        legacy_output: Output from legacy agent
        request_id: Reference to original request

    Returns:
        ConstraintExtractorOutput contract instance
    """
    # Extract constraints from legacy format
    constraints = []
    for constraint_data in legacy_output.get("constraints", []):
        constraints.append(
            ExtractedConstraint(
                constraint_type=constraint_data.get("type", "other"),
                description=constraint_data.get("description", ""),
                parameters=constraint_data.get("parameters", {}),
                priority=constraint_data.get("priority", 5),
                is_hard_constraint=constraint_data.get("is_hard_constraint", True),
            )
        )

    # Extract missing information
    missing_info = []
    for info in legacy_output.get("metadata", {}).get("missing_information", []):
        missing_info.append(
            MissingInformation(
                field=info.get("field", "unknown"),
                reason=info.get("reason", ""),
                suggestions=info.get("suggestions", []),
            )
        )

    return ConstraintExtractorOutput(
        request_id=request_id,
        success=True,
        constraints=constraints,
        missing_information=missing_info,
        extraction_confidence=legacy_output.get("metadata", {}).get("confidence", 1.0),
        metadata=legacy_output.get("metadata", {}),
    )


# ============================================================================
# Conflict Classifier Adapters
# ============================================================================


def adapt_conflict_classifier_input(
    plan: FactorGraph,
    structured_constraints: List[Dict[str, Any]],
    conflict_variables: Optional[List[str]] = None,
) -> ConflictClassifierInput:
    """
    Convert legacy conflict classifier input to contract format.

    Args:
        plan: Travel plan factor graph
        structured_constraints: Structured constraints
        conflict_variables: Known conflict variables

    Returns:
        ConflictClassifierInput contract instance
    """
    return ConflictClassifierInput(
        request_id=str(uuid.uuid4()),
        plan_data=plan.model_dump(),
        structured_constraints=structured_constraints,
        conflict_variables=conflict_variables or [],
    )


def adapt_conflict_classifier_output(
    legacy_output: Dict[str, Any],
    request_id: str,
) -> ConflictClassifierOutput:
    """
    Convert legacy conflict classifier output to contract format.

    Args:
        legacy_output: Output from legacy agent
        request_id: Reference to original request

    Returns:
        ConflictClassifierOutput contract instance
    """
    # Parse conflict type
    conflict_type_str = legacy_output.get("conflict_type", "LOGICAL_ISSUE")
    try:
        conflict_type = ConflictType(conflict_type_str)
    except ValueError:
        conflict_type = ConflictType.LOGICAL_ISSUE

    # Parse retrieval instructions
    retrieval_instructions = []
    for instr in legacy_output.get("retrieval_instructions", []):
        retrieval_instructions.append(
            RetrievalInstruction(
                retrieval_type=instr.get("type", "search_activities"),
                parameters=instr.get("params", {}),
                priority=instr.get("priority", 5),
                rationale=instr.get("rationale", ""),
            )
        )

    return ConflictClassifierOutput(
        request_id=request_id,
        success=True,
        conflict_type=conflict_type,
        conflict_reason=legacy_output.get("conflict_reason", ""),
        retrieval_instructions=retrieval_instructions,
        affected_entities=legacy_output.get("affected_entities", []),
        confidence=legacy_output.get("confidence", 1.0),
        metadata=legacy_output.get("metadata", {}),
    )


# ============================================================================
# Verifier Adapters
# ============================================================================


def adapt_verifier_input(
    plan: FactorGraph,
    structured_constraints: List[Dict[str, Any]],
) -> VerifierInput:
    """
    Convert legacy verifier input to contract format.

    Args:
        plan: Travel plan factor graph
        structured_constraints: All constraints to verify

    Returns:
        VerifierInput contract instance
    """
    return VerifierInput(
        request_id=str(uuid.uuid4()),
        plan_data=plan.model_dump(),
        structured_constraints=structured_constraints,
    )


def adapt_verifier_output(
    legacy_output: Dict[str, Any],
    request_id: str,
) -> VerifierOutput:
    """
    Convert legacy verifier output to contract format.

    Args:
        legacy_output: Output from legacy agent
        request_id: Reference to original request

    Returns:
        VerifierOutput contract instance
    """
    # Parse violation details
    violation_details = []
    for detail in legacy_output.get("violation_details", {}).values():
        if isinstance(detail, list):
            for item in detail:
                violation_details.append(
                    ViolationDetail(
                        constraint_id=item.get("constraint_id", "unknown"),
                        variable_ids=item.get("variable_ids", []),
                        violation_type=item.get("type", "unknown"),
                        description=item.get("description", ""),
                        severity=item.get("severity", 5),
                    )
                )

    return VerifierOutput(
        request_id=request_id,
        success=True,
        has_conflicts=legacy_output.get("has_conflicts", False),
        conflict_variables=legacy_output.get("conflict_variables", []),
        satisfied_constraints=legacy_output.get("satisfied_constraints", []),
        violated_constraints=legacy_output.get("violated_constraints", []),
        violation_details=violation_details,
        verification_summary=legacy_output.get("summary", ""),
        metadata=legacy_output.get("metadata", {}),
    )


# ============================================================================
# Scope Selector Adapters
# ============================================================================


def adapt_scope_selector_input(
    plan: FactorGraph,
    conflict_variables: List[str],
    max_depth: int = 2,
) -> ScopeSelectorInput:
    """
    Convert legacy scope selector input to contract format.

    Args:
        plan: Constraint factor graph
        conflict_variables: Seed variables with conflicts
        max_depth: Maximum propagation depth

    Returns:
        ScopeSelectorInput contract instance
    """
    return ScopeSelectorInput(
        request_id=str(uuid.uuid4()),
        plan_data=plan.model_dump(),
        conflict_variables=conflict_variables,
        max_depth=max_depth,
    )


def adapt_scope_selector_output(
    legacy_output: Dict[str, Any],
    request_id: str,
) -> ScopeSelectorOutput:
    """
    Convert legacy scope selector output to contract format.

    Args:
        legacy_output: Output from legacy agent
        request_id: Reference to original request

    Returns:
        ScopeSelectorOutput contract instance
    """
    # Parse subgraph data
    subgraph_dict = legacy_output.get("affected_subgraph", {})
    subgraph = SubgraphData(
        variable_ids=subgraph_dict.get("variables", []),
        constraint_ids=subgraph_dict.get("constraints", []),
        edges=subgraph_dict.get("edges", []),
        depth=subgraph_dict.get("depth", 1),
        metadata=subgraph_dict.get("metadata", {}),
    )

    return ScopeSelectorOutput(
        request_id=request_id,
        success=True,
        affected_subgraph=subgraph,
        scope_metrics=legacy_output.get("scope_metrics", {}),
        excluded_variables=legacy_output.get("excluded_variables", []),
        metadata=legacy_output.get("metadata", {}),
    )


# ============================================================================
# Retrieval Agent Adapters
# ============================================================================


def adapt_retrieval_input(
    retrieval_instructions: List[Dict[str, Any]],
    user_context: Optional[Dict[str, Any]] = None,
) -> RetrievalInput:
    """
    Convert legacy retrieval input to contract format.

    Args:
        retrieval_instructions: List of retrieval instructions
        user_context: User context for retrieval

    Returns:
        RetrievalInput contract instance
    """
    # Convert instructions to contract format
    instructions = []
    for instr in retrieval_instructions:
        instructions.append(
            RetrievalInstruction(
                retrieval_type=instr.get("type", "search_activities"),
                parameters=instr.get("params", {}),
                priority=instr.get("priority", 5),
                rationale=instr.get("rationale", ""),
            )
        )

    return RetrievalInput(
        request_id=str(uuid.uuid4()),
        retrieval_instructions=instructions,
        user_context=user_context or {},
    )


def adapt_retrieval_output(
    legacy_output: Dict[str, Any],
    request_id: str,
) -> RetrievalOutput:
    """
    Convert legacy retrieval output to contract format.

    Args:
        legacy_output: Output from legacy agent
        request_id: Reference to original request

    Returns:
        RetrievalOutput contract instance
    """
    retrieved_info = legacy_output.get("retrieved_info", {})

    # Parse flights
    flights = []
    for flight_data in retrieved_info.get("flights", []):
        flights.append(FlightInfo(**flight_data))

    # Parse hotels
    hotels = []
    for hotel_data in retrieved_info.get("hotels", []):
        hotels.append(HotelInfo(**hotel_data))

    # Parse activities
    activities = []
    for activity_data in retrieved_info.get("activities", []):
        activities.append(ActivityInfo(**activity_data))

    # Create summary
    retrieval_summary = {
        "flights": len(flights),
        "hotels": len(hotels),
        "activities": len(activities),
    }

    return RetrievalOutput(
        request_id=request_id,
        success=True,
        flights=flights,
        hotels=hotels,
        activities=activities,
        retrieval_summary=retrieval_summary,
        execution_time_ms=legacy_output.get("metadata", {}).get(
            "execution_time_ms", 0.0
        ),
        metadata=legacy_output.get("metadata", {}),
    )


# ============================================================================
# Local Editor Adapters
# ============================================================================


def adapt_local_editor_input(
    conflict_reason: str,
    retrieval_info: Dict[str, Any],
    plan: FactorGraph,
    affected_subgraph: Optional[Dict[str, Any]] = None,
    user_constraints: Optional[List[Dict[str, Any]]] = None,
) -> LocalEditorInput:
    """
    Convert legacy local editor input to contract format.

    Args:
        conflict_reason: Reason for conflicts
        retrieval_info: Retrieved information
        plan: Constraint factor graph
        affected_subgraph: Affected subgraph data
        user_constraints: User's constraints

    Returns:
        LocalEditorInput contract instance
    """
    # Convert subgraph if provided
    subgraph = None
    if affected_subgraph:
        subgraph = SubgraphData(
            variable_ids=affected_subgraph.get("variables", []),
            constraint_ids=affected_subgraph.get("constraints", []),
            edges=affected_subgraph.get("edges", []),
            depth=affected_subgraph.get("depth", 1),
            metadata=affected_subgraph.get("metadata", {}),
        )

    return LocalEditorInput(
        request_id=str(uuid.uuid4()),
        conflict_reason=conflict_reason,
        retrieval_info=retrieval_info,
        plan_data=plan.model_dump(),
        affected_subgraph=subgraph,
        user_constraints=user_constraints or [],
    )


def adapt_local_editor_output(
    legacy_output: Dict[str, Any],
    request_id: str,
) -> LocalEditorOutput:
    """
    Convert legacy local editor output to contract format.

    Args:
        legacy_output: Output from legacy agent
        request_id: Reference to original request

    Returns:
        LocalEditorOutput contract instance
    """
    # Parse atomic operations
    atomic_operations = []
    for op_data in legacy_output.get("atomic_operations", []):
        atomic_operations.append(
            AtomicOperation(
                operation_type=OperationType(op_data.get("type", "Replace")),
                target_id=op_data.get("target", ""),
                changes=op_data.get("changes", {}),
                reasoning=op_data.get("reasoning", ""),
                dependencies=op_data.get("dependencies", []),
                expected_outcome=op_data.get("expected_outcome", ""),
            )
        )

    # Parse execution plan
    exec_plan_data = legacy_output.get("execution_plan", {})
    execution_plan = ExecutionPlan(
        total_operations=exec_plan_data.get("total_operations", len(atomic_operations)),
        execution_order=exec_plan_data.get(
            "execution_order", list(range(len(atomic_operations)))
        ),
        estimated_time_ms=exec_plan_data.get("estimated_time_ms", 0.0),
        rollback_strategy=exec_plan_data.get("rollback_strategy", "sequential"),
    )

    return LocalEditorOutput(
        request_id=request_id,
        success=True,
        atomic_operations=atomic_operations,
        execution_plan=execution_plan,
        edit_summary=legacy_output.get("summary", ""),
        confidence=legacy_output.get("confidence", 1.0),
        alternative_plans=legacy_output.get("alternative_plans", []),
        metadata=legacy_output.get("metadata", {}),
    )


# ============================================================================
# Utility Functions
# ============================================================================


def serialize_contract_output(output: Any) -> Dict[str, Any]:
    """
    Serialize a contract output to a plain dictionary.

    Args:
        output: Contract output instance

    Returns:
        Dictionary representation
    """
    return output.model_dump()


def deserialize_contract_input(
    input_class: type,
    data: Dict[str, Any],
) -> Any:
    """
    Deserialize a dictionary to a contract input.

    Args:
        input_class: Contract input class
        data: Dictionary data

    Returns:
        Contract input instance
    """
    return input_class.model_validate(data)
