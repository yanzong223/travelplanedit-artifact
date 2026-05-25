"""
Data contracts for agent interfaces in TPE system.

This module defines Pydantic models that standardize the input/output
contracts between agents and the orchestrator.
"""

from .adapters import (  # Constraint Extractor adapters; Conflict Classifier adapters; Verifier adapters; Scope Selector adapters; Retrieval adapters; Local Editor adapters; Utilities
    adapt_conflict_classifier_input,
    adapt_conflict_classifier_output,
    adapt_constraint_extractor_input,
    adapt_constraint_extractor_output,
    adapt_local_editor_input,
    adapt_local_editor_output,
    adapt_retrieval_input,
    adapt_retrieval_output,
    adapt_scope_selector_input,
    adapt_scope_selector_output,
    adapt_verifier_input,
    adapt_verifier_output,
    deserialize_contract_input,
    serialize_contract_output,
)
from .agent_contracts import (  # Base contracts; Constraint Extractor contracts; Conflict Classifier contracts; Verifier contracts; Scope Selector contracts; Retrieval Agent contracts; Local Editor contracts
    ActivityInfo,
    AgentInput,
    AgentOutput,
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

__all__ = [
    # Base
    "AgentInput",
    "AgentOutput",
    # Constraint Extractor
    "ConstraintExtractorInput",
    "ConstraintExtractorOutput",
    "ExtractedConstraint",
    "MissingInformation",
    "adapt_constraint_extractor_input",
    "adapt_constraint_extractor_output",
    # Conflict Classifier
    "ConflictClassifierInput",
    "ConflictClassifierOutput",
    "ConflictType",
    "RetrievalInstruction",
    "adapt_conflict_classifier_input",
    "adapt_conflict_classifier_output",
    # Verifier
    "VerifierInput",
    "VerifierOutput",
    "ViolationDetail",
    "adapt_verifier_input",
    "adapt_verifier_output",
    # Scope Selector
    "ScopeSelectorInput",
    "ScopeSelectorOutput",
    "SubgraphData",
    "adapt_scope_selector_input",
    "adapt_scope_selector_output",
    # Retrieval
    "RetrievalInput",
    "RetrievalOutput",
    "FlightInfo",
    "HotelInfo",
    "ActivityInfo",
    "adapt_retrieval_input",
    "adapt_retrieval_output",
    # Local Editor
    "LocalEditorInput",
    "LocalEditorOutput",
    "AtomicOperation",
    "ExecutionPlan",
    "OperationType",
    "adapt_local_editor_input",
    "adapt_local_editor_output",
    # Utilities
    "serialize_contract_output",
    "deserialize_contract_input",
]
