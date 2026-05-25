"""
Factor graph data models for TPE system.

Implements the factor graph structure with variable and constraint nodes.
"""

import uuid
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, field_validator, model_validator

from .base import BaseTPEModel
from .enums import ConstraintType, EdgeType, VariableType


class VariableNode(BaseTPEModel):
    """Editable elements in the travel plan."""

    type: VariableType = Field(description="Type of variable")
    data: Dict[str, Any] = Field(default_factory=dict, description="Variable data")
    constraints: List[str] = Field(
        default_factory=list, description="Constraint IDs this variable participates in"
    )
    label: Optional[str] = Field(default=None, description="Human-readable label")
    immutable_fields: Set[str] = Field(
        default_factory=set, description="Fields that cannot be modified"
    )

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, v):
        """Ensure constraints are valid strings."""
        if not all(isinstance(c, str) for c in v):
            raise ValueError("All constraint IDs must be strings")
        return v

    @field_validator("immutable_fields", mode="before")
    @classmethod
    def validate_immutable_fields(cls, v):
        """Ensure immutable_fields is a set, handle string representation."""
        if isinstance(v, str):
            # Handle string representation of empty set
            if v == 'set()' or v == '':
                return set()
            # Handle string representation of set with items
            if v.startswith('{') and v.endswith('}'):
                try:
                    # Simple parsing for set-like strings
                    content = v[1:-1].strip()
                    if not content:
                        return set()
                    # Split by comma and clean up items
                    items = [item.strip().strip("'\"") for item in content.split(',')]
                    return set(items)
                except Exception:
                    pass
        if isinstance(v, (list, tuple)):
            return set(v)
        if isinstance(v, set):
            return v
        if v is None:
            return set()
        # Default to empty set for any other type
        return set()

    def is_activity(self) -> bool:
        """
        Check if this variable counts as an 'activity' for daily tracking.

        Policy: "activities" = POI/Meals/Lodging only;
        TRANSPORT counted separately.

        Note: Typical daily range is 6-8 activities for reference only,
        not a hard constraint or threshold.
        """
        return self.type in {
            VariableType.ACTIVITY,  # POI/Points of Interest
            VariableType.HOTEL,  # Lodging
            # Note: Meals represented as ACTIVITY with meal_type subcategory
            # TRANSPORT and FLIGHT excluded - counted separately
        }

    def is_transport(self) -> bool:
        """Check if this variable is transport-related (counted separately)."""
        return self.type in {VariableType.TRANSPORT, VariableType.FLIGHT}

    def get_field_value(self, field_path: str) -> Any:
        """Get a nested field value using dot notation."""
        current = self.data
        for part in field_path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def set_field_value(self, field_path: str, value: Any) -> None:
        """Set a nested field value using dot notation."""
        if field_path in self.immutable_fields:
            raise ValueError(
                f"Field '{field_path}' is immutable and cannot be modified"
            )

        current = self.data
        parts = field_path.split(".")

        # Navigate to the parent of the target field
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value


class ConstraintNode(BaseTPEModel):
    """Rules and relationships that must be satisfied."""

    type: ConstraintType = Field(description="Type of constraint")
    variables: List[str] = Field(description="Variable IDs this constraint applies to")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Constraint parameters"
    )
    weight: float = Field(default=1.0, description="Importance level for optimization")
    is_hard: bool = Field(
        default=False,
        description="Whether this is a hard constraint that must be satisfied",
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description"
    )

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, v):
        """Ensure variables are valid strings."""
        if not v:
            raise ValueError("Constraint must apply to at least one variable")
        if not all(isinstance(var, str) for var in v):
            raise ValueError("All variable IDs must be strings")
        return v

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v):
        """Ensure weight is non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v


class GraphEdge(BaseTPEModel):
    """Connections between nodes in the factor graph."""

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: EdgeType = Field(description="Type of edge relationship")
    weight: float = Field(default=1.0, description="Edge weight")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional edge metadata"
    )

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v):
        """Ensure weight is non-negative."""
        if v < 0:
            raise ValueError("Weight must be non-negative")
        return v


class FactorGraph(BaseTPEModel):
    """Factor graph with variable and constraint nodes."""

    variables: Dict[str, VariableNode] = Field(
        default_factory=dict, description="Variable nodes"
    )
    constraints: Dict[str, ConstraintNode] = Field(
        default_factory=dict, description="Constraint nodes"
    )
    edges: List[GraphEdge] = Field(
        default_factory=list, description="Edges between nodes"
    )

    @model_validator(mode="after")
    def validate_graph_connectivity(self):
        """Ensure graph references are consistent."""
        # Check that all referenced variables exist
        variable_ids = set(self.variables.keys())

        for constraint_id, constraint in self.constraints.items():
            for var_id in constraint.variables:
                if var_id not in variable_ids:
                    raise ValueError(
                        f"Constraint {constraint_id} references non-existent variable {var_id}"
                    )

        # Check that all edge nodes exist
        all_node_ids = variable_ids.union(set(self.constraints.keys()))
        for edge in self.edges:
            if edge.source not in all_node_ids:
                raise ValueError(
                    f"Edge references non-existent source node: {edge.source}"
                )
            if edge.target not in all_node_ids:
                raise ValueError(
                    f"Edge references non-existent target node: {edge.target}"
                )

        return self

    def add_variable(self, variable: VariableNode) -> None:
        """Add a variable node to the graph."""
        self.variables[variable.id] = variable

    def add_constraint(self, constraint: ConstraintNode) -> None:
        """Add a constraint node to the graph."""
        self.constraints[constraint.id] = constraint

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def remove_variable(self, variable_id: str) -> None:
        """Remove a variable node and related edges."""
        if variable_id in self.variables:
            del self.variables[variable_id]
            # Remove edges involving this variable
            self.edges = [
                e
                for e in self.edges
                if e.source != variable_id and e.target != variable_id
            ]
            # Remove constraints that only reference this variable
            constraints_to_remove = [
                cid
                for cid, constraint in self.constraints.items()
                if set(constraint.variables) == {variable_id}
            ]
            for cid in constraints_to_remove:
                del self.constraints[cid]

    def get_variables_by_type(self, var_type: VariableType) -> List[VariableNode]:
        """Get all variables of a specific type."""
        return [var for var in self.variables.values() if var.type == var_type]

    def get_constraints_by_type(
        self, constraint_type: ConstraintType
    ) -> List[ConstraintNode]:
        """Get all constraints of a specific type."""
        return [con for con in self.constraints.values() if con.type == constraint_type]

    def get_connected_variables(self, constraint_id: str) -> List[VariableNode]:
        """Get variables connected to a specific constraint."""
        if constraint_id not in self.constraints:
            return []
        constraint = self.constraints[constraint_id]
        return [
            self.variables[var_id]
            for var_id in constraint.variables
            if var_id in self.variables
        ]

    def get_connected_constraints(self, variable_id: str) -> List[ConstraintNode]:
        """Get constraints connected to a specific variable."""
        return [
            constraint
            for constraint in self.constraints.values()
            if variable_id in constraint.variables
        ]

    def is_connected(self) -> bool:
        """Check if the graph is fully connected."""
        if not self.variables:
            return True

        # Build adjacency list
        adjacency = {var_id: set() for var_id in self.variables.keys()}

        # Add edges through constraints
        for constraint in self.constraints.values():
            for i, var1 in enumerate(constraint.variables):
                for var2 in constraint.variables[i + 1 :]:
                    if var1 in adjacency and var2 in adjacency:
                        adjacency[var1].add(var2)
                        adjacency[var2].add(var1)

        # Add direct edges
        for edge in self.edges:
            if edge.source in adjacency and edge.target in adjacency:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)

        # BFS to check connectivity
        visited = set()
        queue = [next(iter(self.variables.keys()))]

        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(adjacency[current] - visited)

        return len(visited) == len(self.variables)

    def clone(self) -> "FactorGraph":
        """Create a deep clone of the factor graph."""
        return FactorGraph(
            variables={
                var_id: var.model_copy() for var_id, var in self.variables.items()
            },
            constraints={
                con_id: con.model_copy() for con_id, con in self.constraints.items()
            },
            edges=[edge.model_copy() for edge in self.edges],
        )
