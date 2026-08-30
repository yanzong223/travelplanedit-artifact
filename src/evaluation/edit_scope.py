"""Public API for dependency-closed edit-scope attribution.

This module is the stable entry point for edit-scope analysis.  The historical
``cascade_analysis`` module remains the implementation engine so existing
callers and tests can keep using it, while this facade owns the algorithm name,
version, result contract, and accounting invariants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .cascade_analysis import (
    FullGateResult,
    FullGateValidator,
    analyze_cascade as _analyze_cascade,
)
from .route_evidence import MODES, RouteEvidenceCache


ALGORITHM_ID = "dependency-closed-counterfactual-attribution"
ALGORITHM_SHORT_NAME = "DCCA"
ALGORITHM_VERSION = "1.1.1"
MAX_ROUTE_REPAIR_BOUNDARIES = 6


class AttributionClass(StrEnum):
    """Mutually exclusive labels assigned to inferred changed units."""

    DIRECT_TARGET = "direct_target"
    ROLLBACK_REQUIRED_SUPPORT = "rollback_required_support"
    # Source compatibility only. This alias is deliberately absent from the
    # canonical class list because "hard required" overstates the quantifier.
    HARD_REQUIRED_SUPPORT = "rollback_required_support"
    SCOPE_AUTHORIZED_COMPLETION = "scope_authorized_completion"
    VERIFIED_REMOVABLE = "verified_removable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class EditScopeCounts:
    """Five-way attribution counts for one successful edited plan."""

    direct_target: int
    rollback_required_support: int
    scope_authorized_completion: int
    verified_removable: int
    unresolved: int
    total: int

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> "EditScopeCounts":
        rollback_required = result.get("rollback_required_support_change_count")
        if rollback_required is None:
            rollback_required = result["hard_required_support_change_count"]
        return cls(
            direct_target=int(result["four_way_direct_target_count"]),
            rollback_required_support=int(rollback_required),
            scope_authorized_completion=int(
                result["scope_authorized_completion_change_count"]
            ),
            verified_removable=int(result["verified_removable_change_count"]),
            unresolved=int(result["unresolved_change_count"]),
            total=int(result["four_way_total_impact_count"]),
        )

    @property
    def classified_total(self) -> int:
        return (
            self.direct_target
            + self.rollback_required_support
            + self.scope_authorized_completion
            + self.verified_removable
            + self.unresolved
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EditScopeRates:
    """Evaluator-relative quality rates derived from five-way counts."""

    removable_lower: float | None
    removable_upper: float | None
    unresolved: float | None
    proof_coverage: float | None

    @classmethod
    def from_counts(cls, counts: EditScopeCounts) -> "EditScopeRates":
        if counts.total == 0:
            return cls(None, None, None, None)
        return cls(
            removable_lower=counts.verified_removable / counts.total,
            removable_upper=(counts.verified_removable + counts.unresolved)
            / counts.total,
            unresolved=counts.unresolved / counts.total,
            # Scope-authorized completion is policy-authorized, not established
            # by a rollback proof, so it is intentionally outside the numerator.
            proof_coverage=(
                counts.direct_target
                + counts.rollback_required_support
                + counts.verified_removable
            )
            / counts.total,
        )

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def algorithm_manifest() -> dict[str, Any]:
    """Return machine-readable semantics needed to interpret a result."""

    return {
        "id": ALGORITHM_ID,
        "short_name": ALGORITHM_SHORT_NAME,
        "version": ALGORITHM_VERSION,
        "unit": "inferred_changed_activity_or_virtual_scope_unit",
        "eligibility": "level1_and_level2_pass",
        "classes": [item.value for item in AttributionClass],
        "decision_precedence": [
            AttributionClass.DIRECT_TARGET.value,
            AttributionClass.SCOPE_AUTHORIZED_COMPLETION.value,
            AttributionClass.VERIFIED_REMOVABLE.value,
            AttributionClass.ROLLBACK_REQUIRED_SUPPORT.value,
            AttributionClass.UNRESOLVED.value,
        ],
        "dependency_policy": "minimal_changed_predecessor_closure",
        "route_repair_policy": "joint_cartesian_product_across_boundaries",
        "route_modes": list(MODES),
        "max_route_repair_boundaries": MAX_ROUTE_REPAIR_BOUNDARIES,
        "rollback_family": "algorithm_defined_and_versioned",
        "rollback_required_support_quantifier": (
            "no_valid_counterfactual_found_in_complete_tested_rollback_family"
        ),
        "verified_removable_quantifier": "exists_saved_counterfactual_witness_passing_full_gate",
        "unresolved_rule": "required_counterfactual_evidence_is_incomplete",
        "determinism_scope": (
            "fixed_versioned_inputs_auxiliary_evidence_and_pure_full_gate_validator"
        ),
        "metric_semantics": "conditional_and_evaluator_relative",
    }


def _assert_optional_rate(
    result: dict[str, Any], key: str, expected: float | None
) -> None:
    actual = result.get(key)
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{key} mismatch: actual={actual!r}, expected={expected!r}")
        return
    if abs(float(actual) - expected) > 1e-12:
        raise ValueError(f"{key} mismatch: actual={actual!r}, expected={expected!r}")


def validate_edit_scope_result(result: dict[str, Any]) -> None:
    """Fail closed when a result violates the public attribution contract."""

    counts = EditScopeCounts.from_result(result)
    if min(counts.to_dict().values()) < 0:
        raise ValueError(f"edit-scope counts must be non-negative: {counts}")
    if counts.classified_total != counts.total:
        raise ValueError(
            "edit-scope attribution does not conserve changed units: "
            f"classified={counts.classified_total}, total={counts.total}"
        )

    rates = EditScopeRates.from_counts(counts)
    if (
        rates.removable_lower is not None
        and rates.removable_upper is not None
        and rates.removable_lower > rates.removable_upper
    ):
        raise ValueError("removable lower bound exceeds upper bound")
    _assert_optional_rate(
        result, "verified_removable_change_rate", rates.removable_lower
    )
    _assert_optional_rate(
        result, "evaluator_relative_excess_change_rate_lower", rates.removable_lower
    )
    _assert_optional_rate(
        result, "evaluator_relative_excess_change_rate_upper", rates.removable_upper
    )
    _assert_optional_rate(result, "unresolved_share", rates.unresolved)
    _assert_optional_rate(result, "proof_coverage", rates.proof_coverage)

    # Stored v1.0/v5 artifacts remain readable for count/rate comparisons, but
    # they predate saved witnesses and explicit completeness certificates.
    # Only v1.1+ results may claim the stronger program-checked contract.
    if "rollback_required_support_change_count" not in result:
        return

    verified_units = result.get(
        "verified_removable_units", result.get("avoidable_extra_units", [])
    )
    if not isinstance(verified_units, list):
        raise ValueError("verified_removable_units must be a list")
    if len(verified_units) != counts.verified_removable:
        raise ValueError("verified removable count does not match saved witnesses")
    for unit in verified_units:
        witness = unit.get("witness") if isinstance(unit, dict) else None
        if not isinstance(witness, dict) or witness.get("gate", {}).get("passed") is not True:
            raise ValueError("verified removable unit lacks a saved passing witness")
        candidate_plan = witness.get("candidate_plan")
        if witness.get("candidate_plan_sha256") != canonical_json_sha256(candidate_plan):
            raise ValueError("verified removable witness hash mismatch")
        outcomes = unit.get("tested_candidate_outcomes", [])
        if not any(
            isinstance(outcome, dict)
            and outcome.get("candidate_plan_sha256") == witness["candidate_plan_sha256"]
            and outcome.get("fully_valid") is True
            for outcome in outcomes
        ):
            raise ValueError("verified removable witness is not in the tested family")

    rollback_required_units = result.get(
        "rollback_required_support_units", result.get("required_support_units", [])
    )
    if not isinstance(rollback_required_units, list):
        raise ValueError("rollback_required_support_units must be a list")
    if len(rollback_required_units) != counts.rollback_required_support:
        raise ValueError("rollback-required count does not match unit evidence")
    if any(
        not isinstance(unit, dict)
        or unit.get("rollback_family_complete") is not True
        or unit.get("tested_candidate_count")
        != len(unit.get("tested_candidate_outcomes", []))
        or any(
            not isinstance(outcome, dict)
            or outcome.get("fully_valid") is not False
            for outcome in unit.get("tested_candidate_outcomes", [])
        )
        for unit in rollback_required_units
    ):
        raise ValueError("rollback-required unit lacks complete tested evidence")


def canonical_json_sha256(value: Any) -> str:
    """Return the deterministic digest used for outputs and saved witnesses."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_edit_scope_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical compact view used by reports and diagnostics."""

    validate_edit_scope_result(result)
    counts = EditScopeCounts.from_result(result)
    rates = EditScopeRates.from_counts(counts)
    return {
        "algorithm": result.get("algorithm", algorithm_manifest()),
        "counts": counts.to_dict(),
        "rates": rates.to_dict(),
    }


def analyze_edit_scope(
    origin_plan: dict[str, Any],
    edited_plan: dict[str, Any],
    edit_constraints: list[dict[str, Any]],
    *,
    route_evidence_cache: RouteEvidenceCache | None = None,
    full_gate_validator: FullGateValidator | None = None,
) -> dict[str, Any]:
    """Run DCCA for one plan pair and return a validated, versioned result.

    The metric is intended for outputs that pass both success gates.  When a
    caller omits the full-gate validator, the engine conservatively leaves
    proof-dependent units unresolved instead of treating them as removable.
    """

    result = _analyze_cascade(
        origin_plan,
        edited_plan,
        edit_constraints,
        route_evidence_cache=route_evidence_cache,
        full_gate_validator=full_gate_validator,
    )
    result["algorithm"] = algorithm_manifest()
    validate_edit_scope_result(result)
    return result


# Compatibility alias for callers migrating from ``cascade_analysis``.
analyze_cascade = analyze_edit_scope


__all__ = [
    "ALGORITHM_ID",
    "ALGORITHM_SHORT_NAME",
    "ALGORITHM_VERSION",
    "AttributionClass",
    "EditScopeCounts",
    "EditScopeRates",
    "FullGateResult",
    "FullGateValidator",
    "algorithm_manifest",
    "analyze_cascade",
    "analyze_edit_scope",
    "canonical_json_sha256",
    "summarize_edit_scope_result",
    "validate_edit_scope_result",
]
