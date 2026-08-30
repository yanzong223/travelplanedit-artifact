"""Canonical constraint IR data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class IRNode:
    node_type: str
    id: str
    constraint_type: str
    predicate: str = "=="
    value: Any = None
    target: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    scope: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    supported: bool | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CanonicalConstraintIR:
    ir_version: str
    source: Dict[str, Any] = field(default_factory=dict)
    root: Dict[str, Any] = field(default_factory=dict)
    nodes: List[IRNode] = field(default_factory=list)
    preference_hints: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["nodes"] = [node.to_dict() for node in self.nodes]
        return payload


def canonical_ir_from_dict(payload: Dict[str, Any] | None) -> CanonicalConstraintIR:
    data = payload if isinstance(payload, dict) else {}
    raw_nodes = data.get("nodes", [])
    nodes = []
    if isinstance(raw_nodes, list):
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            nodes.append(
                IRNode(
                    node_type=str(item.get("node_type", "atomic") or "atomic"),
                    id=str(item.get("id", "")),
                    constraint_type=str(item.get("constraint_type", "")),
                    predicate=str(item.get("predicate", item.get("operator", "==")) or "=="),
                    value=item.get("value"),
                    target=dict(item.get("target", {})),
                    params=dict(item.get("params", {})),
                    scope=dict(item.get("scope", {})),
                    provenance=dict(item.get("provenance", {})),
                    children=[
                        str(child)
                        for child in item.get("children", [])
                        if isinstance(child, (str, int))
                    ],
                    supported=item.get("supported"),
                )
            )
    return CanonicalConstraintIR(
        ir_version=str(data.get("ir_version", "v1") or "v1"),
        source=dict(data.get("source", {})),
        root=dict(data.get("root", {})),
        nodes=nodes,
        preference_hints=[
            dict(item) for item in data.get("preference_hints", []) if isinstance(item, dict)
        ],
        provenance=dict(data.get("provenance", {})),
    )


def canonical_ir_to_dict(payload: CanonicalConstraintIR | Dict[str, Any] | None) -> Dict[str, Any]:
    if isinstance(payload, CanonicalConstraintIR):
        return payload.to_dict()
    if isinstance(payload, dict):
        return canonical_ir_from_dict(payload).to_dict()
    return canonical_ir_from_dict({}).to_dict()
