"""Structured benchmark evaluation data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class LogicalConstraintObject:
    id: str
    source: str
    type: str
    scope: Dict[str, Any] = field(default_factory=dict)
    target: Dict[str, Any] = field(default_factory=dict)
    operator: str = "=="
    value: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    is_hard: bool = True
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferenceConstraintObject:
    id: str
    source: str
    family: str
    facet: str
    direction: str
    anchor: Optional[str] = None
    edit_mode: str = "addition"
    params: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferenceTag:
    family: str
    facet: str
    edit_mode: Optional[str] = None
    anchor: Optional[str] = None
    intensity: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferenceScoreSnapshot:
    family: str
    facet: str
    supported: bool
    score: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)
    scorer_id: str = ""
    scorer_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationResult:
    constraint_id: str
    constraint_type: str
    supported: bool
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    compiled_logic_py: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreferenceScoreResult:
    preference_id: str
    family: str
    facet: str
    supported: bool
    passed: bool
    score_before: Optional[float]
    score_after: Optional[float]
    delta: Optional[float]
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatchedActivityPair:
    pair_id: str
    origin_ref: str
    edited_ref: str
    match_type: str
    retained: bool
    origin_day: int
    edited_day: int
    origin_index: int
    edited_index: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferredAtomicOp:
    op_type: str
    scope: str
    origin_refs: List[str] = field(default_factory=list)
    edited_refs: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferredEditSequence:
    scope_level: int
    scope_name: str
    atomic_ops: List[InferredAtomicOp] = field(default_factory=list)
    matched_pairs: List[MatchedActivityPair] = field(default_factory=list)
    unmatched_origin: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_edited: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_level": self.scope_level,
            "scope_name": self.scope_name,
            "atomic_ops": [item.to_dict() for item in self.atomic_ops],
            "matched_pairs": [item.to_dict() for item in self.matched_pairs],
            "unmatched_origin": self.unmatched_origin,
            "unmatched_edited": self.unmatched_edited,
        }


@dataclass(slots=True)
class Level3EvaluationResult:
    eligible: bool
    reason: str
    scope_level: int
    scope_name: str
    parameter_count: int
    structural_count: int
    compositional_count: int
    atomic_counts: Dict[str, int]
    edit_cost_tuple: List[int]
    content_retention_rate: float
    poi_seq_edit_distance: int
    activity_change_ratio: float
    level1_supported_constraints: int = 0
    level1_unsupported_constraints: int = 0
    level2_supported_constraints: int = 0
    level2_unsupported_constraints: int = 0
    matched_pairs: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_origin: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_edited: List[Dict[str, Any]] = field(default_factory=list)
    cascade: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def constraints_to_dict(items: List[LogicalConstraintObject]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]


def preferences_to_dict(items: List[PreferenceConstraintObject]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]


def preference_tags_to_dict(items: List[PreferenceTag]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]


def preference_snapshots_to_dict(items: List[PreferenceScoreSnapshot]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in items]
