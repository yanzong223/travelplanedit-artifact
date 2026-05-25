"""
Edit intent models for TPE system.

Defines structured edit intent detection and representation for travel plan editing.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import Field, field_validator

from .base import BaseTPEModel


class EditIntent(BaseTPEModel):
    """Structured representation of user's edit intention.

    Used by the orchestrator to decide between constraint-first vs operation-first workflows.
    """

    # Core classification
    explicit_operation: bool = Field(
        description="Whether this is an explicit operation request (e.g., 'move X to day 1')"
    )
    intent_type: Literal[
        "move_poi", "add_poi", "delete_poi", "reorder",
        "budget", "theme", "temporal", "spatial", "custom"
    ] = Field(description="Type of edit intent")

    # Extracted entities
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured entities extracted from natural language"
    )

    # Satisfaction metrics
    already_satisfied: bool = Field(
        default=False,
        description="Whether the current plan already satisfies this intent"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence score of intent detection"
    )

    # Specific fields for common operation types
    primary_target: Optional[str] = Field(
        default=None,
        description="Primary POI/attraction name targeted by the edit"
    )
    source_day: Optional[int] = Field(
        default=None, ge=1,
        description="Source day number (for move/reorder operations)"
    )
    target_day: Optional[int] = Field(
        default=None, ge=1,
        description="Target day number (for move/add operations)"
    )

    # Operation details
    operation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details specific to the operation type"
    )

    # Fallback information
    fallback_possible: bool = Field(
        default=True,
        description="Whether fallback to other operation types is possible"
    )
    alternative_intents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative intent interpretations if primary fails"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def is_move_operation(self) -> bool:
        """Check if this is a move operation."""
        return self.intent_type == "move_poi" and self.explicit_operation

    def is_add_operation(self) -> bool:
        """Check if this is an add operation."""
        return self.intent_type == "add_poi" and self.explicit_operation

    def is_delete_operation(self) -> bool:
        """Check if this is a delete operation."""
        return self.intent_type == "delete_poi" and self.explicit_operation

    def is_reorder_operation(self) -> bool:
        """Check if this is a reorder operation."""
        return self.intent_type == "reorder" and self.explicit_operation

    def is_constraint_driven(self) -> bool:
        """Check if this should use constraint-first workflow."""
        return not self.explicit_operation or self.intent_type in ["budget", "theme", "spatial"]

    def is_operation_driven(self) -> bool:
        """Check if this should use operation-first workflow."""
        return self.explicit_operation and self.intent_type in ["move_poi", "add_poi", "delete_poi", "reorder"]

    def get_target_variables(self) -> List[str]:
        """Get list of target variable names based on intent."""
        targets = []

        if self.primary_target:
            targets.append(self.primary_target)

        # Extract other target names from entities
        if "poi_names" in self.entities:
            targets.extend(self.entities["poi_names"])
        if "target_pois" in self.entities:
            targets.extend(self.entities["target_pois"])

        return list(set(targets))  # Remove duplicates

    def get_intent_lock_constraint(self) -> Optional[Dict[str, Any]]:
        """Generate a constraint to lock the primary intent during later corrections.

        This ensures that subsequent corrections don't undo the user's explicit request.
        """
        if not self.explicit_operation or not self.primary_target:
            return None

        lock_constraint = {
            "type": "intent_lock",
            "target": self.primary_target,
            "intent": self.intent_type,
            "description": f"Lock {self.primary_target} to preserve user's {self.intent_type} intent"
        }

        if self.intent_type == "move_poi" and self.target_day:
            lock_constraint["target_day"] = self.target_day

        return lock_constraint

    def get_operation_summary(self) -> str:
        """Get human-readable summary of the operation."""
        if not self.explicit_operation:
            return f"Abstract goal: {self.intent_type}"

        if self.intent_type == "move_poi":
            if self.primary_target and self.target_day:
                return f"Move {self.primary_target} to day {self.target_day}"
        elif self.intent_type == "add_poi":
            if self.primary_target and self.target_day:
                return f"Add {self.primary_target} to day {self.target_day}"
        elif self.intent_type == "delete_poi":
            if self.primary_target:
                return f"Delete {self.primary_target} from plan"
        elif self.intent_type == "reorder":
            return f"Reorder activities on day {self.target_day or 'specified day'}"

        return f"Operation: {self.intent_type}"


class IntentExtractionResult(BaseTPEModel):
    """Result of intent extraction process."""

    primary_intent: EditIntent = Field(description="Primary extracted intent")
    alternative_intents: List[EditIntent] = Field(
        default_factory=list,
        description="Alternative interpretations ranked by confidence"
    )
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the extraction process"
    )

    def get_best_intent(self) -> EditIntent:
        """Get the highest confidence intent."""
        if not self.alternative_intents:
            return self.primary_intent

        best_alternative = max(self.alternative_intents, key=lambda x: x.confidence)
        return best_alternative if best_alternative.confidence > self.primary_intent.confidence else self.primary_intent

    def has_multiple_candidates(self) -> bool:
        """Check if there are multiple plausible intent candidates."""
        high_confidence_alternatives = [
            intent for intent in self.alternative_intents
            if intent.confidence > 0.6  # Threshold for "high confidence"
        ]
        return len(high_confidence_alternatives) > 0