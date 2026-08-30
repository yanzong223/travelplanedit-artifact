from __future__ import annotations

import pytest

from evaluation.edit_scope import (
    ALGORITHM_ID,
    FullGateResult,
    analyze_edit_scope,
    canonical_json_sha256,
    summarize_edit_scope_result,
    validate_edit_scope_result,
)
from evaluation.benchmark.level3 import level3_cascade_payload


def _plan(name: str) -> dict:
    return {
        "itinerary": [{
            "day": 1,
            "activities": [{
                "type": "attraction",
                "position": name,
                "start_time": "09:00",
                "end_time": "10:00",
                "transports": [],
            }],
        }]
    }


def _passing_gate(_plan: dict) -> FullGateResult:
    return FullGateResult(
        passed=True,
        components={"level1": True, "level2": True},
    )


def _two_activity_case() -> tuple[dict, dict, list[dict]]:
    origin = {
        "itinerary": [{
            "day": 1,
            "activities": [
                {
                    "type": "attraction", "position": "A",
                    "start_time": "09:00", "end_time": "10:00", "transports": [],
                },
                {
                    "type": "attraction", "position": "old",
                    "start_time": "11:00", "end_time": "12:00", "transports": [],
                },
            ],
        }],
    }
    edited = {
        "itinerary": [{
            "day": 1,
            "activities": [
                {
                    "type": "attraction", "position": "A",
                    "start_time": "08:00", "end_time": "09:00", "transports": [],
                },
                {
                    "type": "attraction", "position": "target",
                    "start_time": "11:00", "end_time": "12:00", "transports": [],
                },
            ],
        }],
    }
    constraints = [{
        "id": "target",
        "source": "edit",
        "type": "required_attraction_name",
        "target": {"activity_type": "attraction"},
        "operator": "contains_all",
        "value": ["target"],
    }]
    return origin, edited, constraints


def test_public_edit_scope_api_versions_and_validates_result():
    result = analyze_edit_scope(
        _plan("old"),
        _plan("target"),
        [{
            "id": "target",
            "source": "edit",
            "type": "required_attraction_name",
            "target": {"activity_type": "attraction"},
            "operator": "contains_all",
            "value": ["target"],
        }],
        full_gate_validator=_passing_gate,
    )

    assert result["algorithm"]["id"] == ALGORITHM_ID
    compact = summarize_edit_scope_result(result)
    assert compact["counts"]["direct_target"] == 1
    assert compact["counts"]["total"] == 1
    assert compact["rates"]["proof_coverage"] == 1.0
    nested = level3_cascade_payload(
        eligible=True,
        reason="ok",
        cascade_result=result,
    )
    assert nested["algorithm"]["id"] == ALGORITHM_ID


def test_public_contract_rejects_nonconserving_counts():
    result = analyze_edit_scope(
        _plan("old"),
        _plan("target"),
        [{
            "id": "target",
            "source": "edit",
            "type": "required_attraction_name",
            "operator": "contains_all",
            "value": ["target"],
        }],
        full_gate_validator=_passing_gate,
    )
    result["unresolved_change_count"] = 1

    with pytest.raises(ValueError, match="does not conserve"):
        validate_edit_scope_result(result)


def test_dcca_is_deterministic_for_fixed_inputs_and_gate():
    origin, edited, constraints = _two_activity_case()

    first = analyze_edit_scope(
        origin, edited, constraints, full_gate_validator=_passing_gate,
    )
    second = analyze_edit_scope(
        origin, edited, constraints, full_gate_validator=_passing_gate,
    )

    assert first == second
    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_verified_removable_saves_a_replayable_passing_witness():
    origin, edited, constraints = _two_activity_case()
    result = analyze_edit_scope(
        origin, edited, constraints, full_gate_validator=_passing_gate,
    )

    assert result["verified_removable_change_count"] == 1
    unit = result["verified_removable_units"][0]
    witness = unit["witness"]
    assert unit["witness_found"] is True
    assert witness["candidate_plan_sha256"] == canonical_json_sha256(
        witness["candidate_plan"]
    )
    assert any(
        outcome["candidate_plan_sha256"] == witness["candidate_plan_sha256"]
        and outcome["fully_valid"] is True
        for outcome in unit["tested_candidate_outcomes"]
    )
    assert _passing_gate(witness["candidate_plan"]).passed is True


def test_incomplete_gate_evidence_is_unresolved_not_rollback_required():
    origin, edited, constraints = _two_activity_case()
    result = analyze_edit_scope(origin, edited, constraints)

    assert result["rollback_required_support_change_count"] == 0
    assert result["verified_removable_change_count"] == 0
    assert result["unresolved_change_count"] == 1
    assert result["unresolved_units"][0]["evidence_complete"] is False
    assert "baseline_full_gate_unavailable" in result["unresolved_units"][0]["incomplete_reasons"]


def test_rollback_required_is_limited_to_a_complete_tested_family():
    origin, edited, constraints = _two_activity_case()

    def gate(plan: dict) -> FullGateResult:
        first_start = plan["itinerary"][0]["activities"][0]["start_time"]
        passed = first_start == "08:00"
        return FullGateResult(
            passed=passed,
            components={"level1": passed, "level2": True},
        )

    result = analyze_edit_scope(
        origin, edited, constraints, full_gate_validator=gate,
    )

    assert result["rollback_required_support_change_count"] == 1
    unit = result["rollback_required_support_units"][0]
    assert unit["rollback_family_complete"] is True
    assert unit["witness_found"] is False
    assert unit["rollback_family_version"] == "dcca-rollback-family-v1"
    assert all(
        outcome["fully_valid"] is False
        for outcome in unit["tested_candidate_outcomes"]
    )


def test_legacy_count_artifact_remains_readable_without_upgrading_its_claims():
    origin, edited, constraints = _two_activity_case()
    legacy = analyze_edit_scope(
        origin, edited, constraints, full_gate_validator=_passing_gate,
    )
    legacy.pop("rollback_required_support_change_count")
    legacy.pop("rollback_required_support_units")
    legacy.pop("verified_removable_units")
    legacy["algorithm"] = {"id": ALGORITHM_ID, "version": "1.0.0"}

    validate_edit_scope_result(legacy)
    assert summarize_edit_scope_result(legacy)["counts"]["verified_removable"] == 1
