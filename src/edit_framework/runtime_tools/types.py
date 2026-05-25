"""Shared semantic runtime-tool types for the standalone edit framework.

These types belong to ``edit_framework`` rather than ``agent_loop`` because the
standalone runtime has its own semantic tool system and must not depend on the
agent-loop type surface.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RuntimeToolName(str, Enum):
    CONSTRAINT_CONSTRUCTOR = "ConstraintConstructor"
    CONFLICT_ANALYZER = "ConflictAnalyzer"
    FACT_RETRIEVER = "FactRetriever"
    EDIT_PROPOSAL_GENERATOR = "EditProposalGenerator"
    DECISION_ASSISTANT = "DecisionAssistant"
    PLAN_PATCH_EXECUTOR = "PlanPatchExecutor"
    RUNTIME_CHECKER = "RuntimeChecker"
    VERIFY_PLAN_CONSTRAINTS = "VerifyPlanConstraints"
    EDIT_REQUIREMENT_DIAGNOSTIC = "EditRequirementDiagnostic"
    CANDIDATE_PLAN_GUARD = "CandidatePlanGuard"


class ToolImplementationMode(str, Enum):
    FUNCTION = "function"
    PROMPT = "prompt"
    HYBRID = "hybrid"


class TruthRole(str, Enum):
    INTERPRETED = "interpreted"
    DERIVED = "derived"
    GROUNDED = "grounded"
    HYPOTHETICAL = "hypothetical"
    ADVISORY = "advisory"
    STATE_AUTHORITATIVE = "state_authoritative"
    CHECK_AUTHORITATIVE = "check_authoritative"


class ModelFacingExposure(str, Enum):
    NONE = "none"
    DIRECT_FUNCTION = "direct_function"
    FRAMEWORK_SPECIFIC = "framework_specific"


class ExposureMode(str, Enum):
    PRIMITIVE_ONLY = "primitive_only"
    RUNTIME_HIGH_LEVEL = "runtime_high_level"
    HYBRID = "hybrid"


class ActiveConstraint(BaseModel):
    id: str
    source: str = "runtime_tool"
    kind: str = "structural"
    hardness: str = "hard"
    target: Dict[str, Any] = Field(default_factory=lambda: {"scope": "global"})
    predicate: str
    params: Dict[str, Any] = Field(default_factory=dict)
    natural_language: Optional[str] = None


class IntentAnchor(BaseModel):
    id: str
    type: str = "poi"
    semantic_key: str
    ref: Optional[str] = None
    rule: str = "must_preserve"


class GroundedConflict(BaseModel):
    id: str
    constraint_id: str
    predicate: str
    status: Literal["violated", "unresolved", "at_risk"]
    reason: str
    anchor: Dict[str, Any] = Field(default_factory=dict)
    evidence_refs: List[str] = Field(default_factory=list)
    need_facts: bool = False
    missing_fact_slots: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class ConstraintConstructionResult(BaseModel):
    ok: bool = True
    active_constraints: List[ActiveConstraint] = Field(default_factory=list)
    intent_anchors: List[IntentAnchor] = Field(default_factory=list)
    lifted_preferences: List[Dict[str, Any]] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    uncertainties: List[str] = Field(default_factory=list)


class ConflictAnalysisResult(BaseModel):
    ok: bool = True
    violated_constraints: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_conflicts: List[GroundedConflict] = Field(default_factory=list)
    unresolved_checks: List[Dict[str, Any]] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class FactFilter(BaseModel):
    field: str
    op: Literal["eq", "le", "ge", "contains"]
    value: Any

    model_config = {"extra": "forbid"}


class FactRetrievalRequest(BaseModel):
    request_id: str
    intent: Literal[
        "lookup_entity",
        "lookup_nearby",
        "lookup_route",
        "lookup_intercity",
        "check_open_status",
        "paginate",
    ]
    entity_type: Optional[Literal["attraction", "restaurant", "accommodation", "poi", "intercity"]] = None
    city: Optional[str] = None
    name: Optional[str] = None
    anchor_name: Optional[str] = None
    filters: List[FactFilter] = Field(default_factory=list)
    start: Optional[str] = None
    end: Optional[str] = None
    start_time: Optional[str] = None
    transport_type: Optional[str] = None
    topk: Optional[int] = None
    dist: Optional[float] = None
    cursor_id: Optional[str] = None

    model_config = {"extra": "forbid"}


class FactRetrievalItemResult(BaseModel):
    request_id: str
    intent: str
    entity_type: Optional[str] = None
    ok: bool = True
    rows: Optional[List[Dict[str, Any]]] = None
    value: Optional[Any] = None
    page: Optional[Dict[str, Any]] = None
    cursor_id: Optional[str] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    backend_trace: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class FactRetrievalResult(BaseModel):
    ok: bool = True
    requests: List[FactRetrievalRequest] = Field(default_factory=list)
    results: List[FactRetrievalItemResult] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class PatchOp(BaseModel):
    op: Literal[
        "insert_activity",
        "replace_activity",
        "move_activity",
        "delete_activity",
        "reschedule_activity",
        "resize_activity",
        "reorder_day",
        "reroute_transport",
    ]
    args: Dict[str, Any] = Field(default_factory=dict)


class PatchExecutionResult(BaseModel):
    ok: bool = True
    plan: Dict[str, Any] = Field(default_factory=dict)
    delta: List[Dict[str, Any]] = Field(default_factory=list)
    issues: List[Any] = Field(default_factory=list)
    pending_repairs: Dict[str, Any] = Field(default_factory=dict)
    applied_ops: List[str] = Field(default_factory=list)


class ProposalEffect(BaseModel):
    constraint_id: str
    expected_status: Literal["improve", "resolve", "unknown"]
    note: Optional[str] = None


class EditProposal(BaseModel):
    proposal_id: str
    summary: str
    ops: List[PatchOp] = Field(default_factory=list)
    expected_constraint_effects: List[ProposalEffect] = Field(default_factory=list)
    confidence: float = 0.0


class ProposalGenerationResult(BaseModel):
    ok: bool = True
    edit_proposals: List[EditProposal] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class DecisionAdvice(BaseModel):
    decision_type: Literal[
        "select_proposal",
        "request_more_facts",
        "regenerate_proposals",
        "stop_no_safe_action",
    ]
    selected_proposal_id: Optional[str] = None
    reason: str
    confidence: float = 0.0


class DecisionAssistantResult(BaseModel):
    ok: bool = True
    decision: DecisionAdvice
    details: Dict[str, Any] = Field(default_factory=dict)


class ConstraintCheckResult(BaseModel):
    constraint_id: str
    predicate: str
    supported: bool
    status: Literal["pass", "fail", "unresolved"]
    params: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)


class RuntimeCheckReport(BaseModel):
    ok: bool = True
    structural_status: Dict[str, Any] = Field(
        default_factory=lambda: {"valid": True, "issues": []}
    )
    constraint_checks: List[ConstraintCheckResult] = Field(default_factory=list)
    risk_flags: List[Dict[str, Any]] = Field(default_factory=list)
    next_step_hint: Literal["accept_for_now", "retry", "need_more_facts"] = "accept_for_now"
    details: Dict[str, Any] = Field(default_factory=dict)


class PlanConstraintCheckResult(BaseModel):
    """Per-dimension result from verify_plan_constraints."""

    constraint_type: str = ""
    passed: bool = True
    violations: List[Dict[str, Any]] = Field(default_factory=list)


class PlanConstraintVerificationResult(BaseModel):
    """Output of verify_plan_constraints tool."""

    ok: bool = True
    error_code: str = ""
    message: str = ""
    checks: List[PlanConstraintCheckResult] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)


class EditRequirementDiagnosticResult(BaseModel):
    ok: bool = True
    edit_targets: List[Dict[str, Any]] = Field(default_factory=list)
    preserve_anchors: List[Dict[str, Any]] = Field(default_factory=list)
    explicit_constraints: List[Dict[str, Any]] = Field(default_factory=list)
    required_fact_slots: List[Dict[str, Any]] = Field(default_factory=list)
    risk_flags: List[Dict[str, Any]] = Field(default_factory=list)
    infeasible_signals: List[Dict[str, Any]] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class CandidatePlanGuardResult(BaseModel):
    ok: bool = True
    decision: Literal["accept", "revise", "infeasible"] = "accept"
    must_fix_violations: List[Dict[str, Any]] = Field(default_factory=list)
    preservation_violations: List[Dict[str, Any]] = Field(default_factory=list)
    feasibility_violations: List[Dict[str, Any]] = Field(default_factory=list)
    fact_grounding_risks: List[Dict[str, Any]] = Field(default_factory=list)
    compact_repair_hints: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


class RuntimeCapabilityState(BaseModel):
    working_plan: Dict[str, Any] = Field(default_factory=dict)
    active_constraints: List[ActiveConstraint] = Field(default_factory=list)
    intent_anchors: List[IntentAnchor] = Field(default_factory=list)
    lifted_preferences: List[Dict[str, Any]] = Field(default_factory=list)
    grounded_conflicts: List[GroundedConflict] = Field(default_factory=list)
    retrieved_facts: List[Dict[str, Any]] = Field(default_factory=list)
    edit_proposals: List[EditProposal] = Field(default_factory=list)
    last_decision: Optional[DecisionAdvice] = None
    last_runtime_check: Optional[RuntimeCheckReport] = None
