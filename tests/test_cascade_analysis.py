import evaluation.cascade_analysis as cascade_analysis
from collections import Counter
import hashlib
from types import SimpleNamespace

from evaluation.cascade_analysis import (
    FullGateResult,
    _cross_city_gap_signature,
    _new_cross_city_gap_instances,
    _new_violation_instances,
    _raw_activities,
    _resolve_endpoint_city,
    _repair_from_cache,
    _route_cache_evidence_complete,
    _rollback_unit,
    _transport_continuity_issues,
    _verification_signature,
    analyze_cascade,
)
from evaluation.benchmark.diffing import infer_edit_sequence
from evaluation.cascade_dataset import (
    PAPER_MANIFEST_SCHEMA,
    input_fingerprint,
    load_paper_manifest,
    metadata_category,
    task_key,
)
from evaluation.route_evidence import RouteEvidenceCache, cache_payload
import json
import pytest


def _activity(name, start, end, kind="attraction"):
    return {"position": name, "type": kind, "start_time": start, "end_time": end, "transports": []}


def _plan(days):
    return {"itinerary": [{"day": index, "activities": activities} for index, activities in enumerate(days, 1)]}


def _passing_full_gate(_plan):
    return FullGateResult(
        passed=True,
        components={"level1": True, "level2": True},
        reason="ok",
    )


def test_target_replacement_separates_direct_change_from_neighbor_cascade():
    origin = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("old", "10:30", "11:30"),
        _activity("C", "12:00", "13:00"),
    ]])
    edited = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("target", "10:30", "11:30"),
        _activity("C", "12:30", "13:30"),
    ]])
    constraints = [{"id": "target", "source": "edit", "type": "required_attraction_name", "target": {"activity_type": "attraction"}, "operator": "contains_all", "value": ["target"]}]
    result = analyze_cascade(
        origin, edited, constraints,
        full_gate_validator=_passing_full_gate,
    )
    assert result["supported"] is True
    assert result["direct_target_change_count"] == 1
    assert result["cascade_change_count"] == 1
    assert result["cascade_amplification"] == 2.0
    assert result["cascade_radius"] == 1
    assert result["spillover_ratio"] == 0.5
    assert (
        result["four_way_direct_target_count"]
        + result["hard_required_support_change_count"]
        + result["scope_authorized_completion_change_count"]
        + result["avoidable_extra_change_count"]
        + result["unresolved_change_count"]
        == result["four_way_total_impact_count"]
    )


def test_cross_day_rewrite_is_visible_as_spillover():
    origin = _plan([
        [_activity("old", "09:00", "10:00")],
        [_activity("B", "09:00", "10:00")],
    ])
    edited = _plan([
        [_activity("target", "09:00", "10:00")],
        [_activity("B", "10:00", "11:00")],
    ])
    constraints = [{"type": "required_attraction_name", "value": "target", "target": {}}]
    result = analyze_cascade(
        origin, edited, constraints,
        full_gate_validator=_passing_full_gate,
    )
    assert result["cross_day_spillover_count"] == 1
    assert result["affected_day_count"] == 2
    assert result["disconnected_change_count"] == 1


def test_global_budget_uses_cost_reduction_not_all_changes_as_direct():
    origin = _plan([[_activity("A", "09:00", "10:00"), _activity("B", "11:00", "12:00")]])
    edited = _plan([[_activity("A", "10:00", "11:00"), _activity("B", "11:00", "12:00")]])
    origin["itinerary"][0]["activities"][0]["cost"] = 100
    edited["itinerary"][0]["activities"][0]["cost"] = 50
    result = analyze_cascade(origin, edited, [{"type": "budget_total", "value": 1000, "target": {}}])
    assert result["supported"] is True
    assert result["attribution_mode"] == "global_numeric_proxy"
    assert result["attribution_confidence"] == "medium"
    assert result["direct_target_change_count"] == 1


def test_semantic_constraint_without_id_uses_fine_tag_scorer(monkeypatch):
    monkeypatch.setattr(
        cascade_analysis,
        "_fine_semantic_type",
        lambda activity, city: "历史古迹" if activity.get("position") == "new" else "其它",
    )
    origin = _plan([[_activity("old", "09:00", "10:00")]])
    edited = _plan([[_activity("new", "09:00", "10:00")]])
    result = analyze_cascade(origin, edited, [{
        "type": "semantic_type_requirement", "value": "历史古迹",
        "target": {"activity_type": "attraction"},
    }])
    assert result["supported"] is True
    assert result["attribution_mode"] == "semantic_fine_tag"
    assert result["attribution_confidence"] == "medium"
    assert result["direct_target_change_count"] == 1


def test_global_time_constraint_attributes_only_time_changes():
    origin = _plan([[_activity("A", "09:00", "10:00"), _activity("B", "11:00", "12:00")]])
    edited = _plan([[_activity("A", "08:00", "09:00"), _activity("C", "11:00", "12:00")]])
    result = analyze_cascade(origin, edited, [{"type": "day_end_time_limit", "value": "18:00", "target": {}}])
    assert result["attribution_mode"] == "temporal_proxy"
    assert result["direct_target_change_count"] == 1
    assert result["cascade_change_count"] == 1


def test_exact_entity_view_remains_available_with_type_proxy():
    origin = _plan([[_activity("target", "09:00", "10:00"), _activity("B", "11:00", "12:00")]])
    edited = _plan([[_activity("target", "08:00", "09:00"), _activity("B", "12:00", "13:00")]])
    result = analyze_cascade(origin, edited, [
        {"type": "poi_time_window", "value": "morning", "target": {"poi_name": "target"}},
    ])
    assert result["attribution_mode"] == "exact_entity+temporal_proxy"
    assert result["exact_entity_direct_change_count"] == 1
    assert result["exact_entity_cascade_amplification"] == 2.0
    assert result["direct_target_change_count"] == 2


def test_day_count_uses_virtual_boundary_unit_not_whole_added_day():
    origin = _plan([[_activity("A", "09:00", "10:00")]])
    edited = _plan([[_activity("A", "09:00", "10:00")], [
        _activity("B", "09:00", "10:00"), _activity("C", "11:00", "12:00"),
    ]])
    result = analyze_cascade(origin, edited, [{"id": "days", "source": "edit", "type": "day_count", "operator": "==", "value": 2, "target": {}}])
    assert result["attribution_mode"] == "day_structure_proxy"
    assert result["virtual_direct_change_count"] == 1
    assert result["direct_target_change_count"] == 1
    assert result["cascade_change_count"] == 1
    assert result["three_way_total_impact_count"] == 3
    assert result["scope_authorized_completion_change_count"] == 0


def test_new_day_basic_roles_are_authorized_within_cap():
    origin = _plan([[_activity("A", "09:00", "10:00")]])
    edited = _plan([[_activity("A", "09:00", "10:00")], [
        _activity("hotel breakfast", "08:00", "08:30", kind="breakfast"),
        _activity("B", "09:00", "10:00"),
    ]])
    constraints = [
        {"id": "days", "source": "edit", "type": "day_count", "operator": "==", "value": 2, "target": {}},
        {"id": "cap", "source": "edit", "type": "daily_poi_cap", "operator": "<=", "value": 1, "target": {"day": "all"}, "params": {"count_types": ["attraction"]}},
    ]
    result = analyze_cascade(
        origin, edited, constraints,
        full_gate_validator=_passing_full_gate,
    )
    assert result["scope_authorized_completion_change_count"] == 2
    assert result["avoidable_extra_change_count"] == 0


def test_transport_endpoint_mismatch_is_unresolved_not_avoidable():
    origin = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("C", "12:00", "13:00"),
    ]])
    edited = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("B", "10:30", "11:30"),
        _activity("C", "12:00", "13:00"),
    ]])
    edited["itinerary"][0]["activities"][1]["transports"] = [{"start": "A", "end": "B", "start_time": "10:00", "end_time": "10:20"}]
    edited["itinerary"][0]["activities"][2]["transports"] = [{"start": "B", "end": "C", "start_time": "11:30", "end_time": "11:50"}]
    result = analyze_cascade(origin, edited, [{
        "id": "cap", "source": "edit", "type": "adjacent_travel_time_cap",
        "target": {"activity_types": ["attraction"]}, "operator": "<=", "value": 25,
    }])
    assert result["unresolved_change_count"] >= 1
    assert any(item.get("reason") == "rollback_requires_transport_repair" for item in result["unresolved_units"])


def _route_cache(duration):
    row = {"start": "A", "end": "C", "mode": "taxi", "start_time": "10:00",
           "end_time": "10:20" if duration <= 25 else "10:40", "duration": duration,
           "distance": 1.0, "cost": 1.0}
    entry = {"city": "X", "start": "A", "end": "C", "start_time": "10:00",
             "request": {"city": "X", "start": "A", "end": "C", "start_time": "10:00"},
             "modes": {mode: {"status": "ok", "rows": [{**row, "mode": mode}]} for mode in ("walk", "metro", "taxi")}}
    return RouteEvidenceCache(cache_payload([entry], tool_fingerprint={"test": True}))


def _empty_route_cache(statuses):
    request = {"city": "X", "start": "A", "end": "C", "start_time": "10:00"}
    entry = {
        **request,
        "request": request,
        "modes": {
            mode: {"status": status, "rows": []}
            for mode, status in zip(("walk", "metro", "taxi"), statuses)
        },
    }
    return RouteEvidenceCache(cache_payload([entry], tool_fingerprint={"test": True}))


def _two_boundary_route_cache():
    entries = []
    for start, end, start_time, end_time in (
        ("A", "B", "10:00", "10:20"),
        ("B", "C", "11:00", "11:20"),
    ):
        request = {
            "city": "X",
            "start": start,
            "end": end,
            "start_time": start_time,
        }
        entries.append({
            **request,
            "request": request,
            "modes": {
                mode: {
                    "status": "ok",
                    "rows": [{
                        "start": start,
                        "end": end,
                        "mode": mode,
                        "start_time": start_time,
                        "end_time": end_time,
                    }],
                }
                for mode in ("walk", "metro", "taxi")
            },
        })
    return RouteEvidenceCache(cache_payload(entries, tool_fingerprint={"test": True}))


def test_route_cache_jointly_repairs_multiple_boundaries():
    plan = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("B", "10:30", "11:00"),
        _activity("C", "11:30", "12:00"),
    ]])
    plan["target_city"] = "X"
    plan["itinerary"][0]["activities"][1]["transports"] = [{
        "start": "wrong-A", "end": "B", "start_time": "10:00", "end_time": "10:20",
    }]
    plan["itinerary"][0]["activities"][2]["transports"] = [{
        "start": "wrong-B", "end": "C", "start_time": "11:00", "end_time": "11:20",
    }]
    issues = sorted(_transport_continuity_issues(plan))
    cache = _two_boundary_route_cache()
    assert len(issues) == 2
    assert _route_cache_evidence_complete(plan, issues, cache) is True
    repaired = _repair_from_cache(plan, issues, cache)
    assert len(repaired) == 9
    assert all(not _transport_continuity_issues(candidate) for candidate, _ in repaired)
    assert all(evidence["combination_size"] == 2 for _, evidence in repaired)


def _route_rollback_case():
    origin = _plan([[_activity("A", "09:00", "10:00"), _activity("C", "12:00", "13:00")]])
    edited = _plan([[_activity("A", "09:00", "10:00"), _activity("B", "10:30", "11:30"), _activity("C", "12:00", "13:00")]])
    origin["target_city"] = edited["target_city"] = "X"
    edited["itinerary"][0]["activities"][1]["transports"] = [{"start": "A", "end": "B", "start_time": "10:00", "end_time": "10:20"}]
    edited["itinerary"][0]["activities"][2]["transports"] = [{"start": "B", "end": "C", "start_time": "11:30", "end_time": "11:50"}]
    constraints = [{"id": "cap", "source": "edit", "type": "adjacent_travel_time_cap",
                    "target": {"activity_types": ["attraction"]}, "operator": "<=", "value": 25}]
    return origin, edited, constraints


def test_cached_feasible_route_makes_rollback_avoidable():
    origin, edited, constraints = _route_rollback_case()
    result = analyze_cascade(
        origin, edited, constraints,
        route_evidence_cache=_route_cache(20),
        full_gate_validator=_passing_full_gate,
    )
    # C's inbound route is evaluated together with its changed predecessor;
    # reverting the minimal dependency closure restores the original A -> C
    # boundary instead of creating a false transport mismatch.
    assert result["unresolved_change_count"] == 0
    assert any(
        item["evidence_source"] == "dependency_closure"
        for item in result["avoidable_extra_units"]
    )
    assert any(item["evidence_source"] == "cache" for item in result["avoidable_extra_units"])


def test_all_cached_routes_over_cap_make_rollback_required_in_tested_family():
    origin, edited, constraints = _route_rollback_case()
    result = analyze_cascade(
        origin,
        edited,
        constraints,
        route_evidence_cache=_route_cache(40),
        full_gate_validator=_passing_full_gate,
    )
    assert result["unresolved_change_count"] == 0
    assert any(
        item["evidence_source"] == "cache"
        for item in result["rollback_required_support_units"]
    )


def test_ok_no_route_is_conclusive_but_query_error_remains_unresolved():
    origin, edited, constraints = _route_rollback_case()
    no_route = analyze_cascade(
        origin, edited, constraints,
        route_evidence_cache=_empty_route_cache(["ok_no_route"] * 3),
        full_gate_validator=_passing_full_gate,
    )
    assert no_route["hard_required_support_change_count"] >= 1
    assert no_route["unresolved_change_count"] == 0

    query_error = analyze_cascade(
        origin, edited, constraints,
        route_evidence_cache=_empty_route_cache(
            ["ok_no_route", "query_error", "ok_no_route"],
        ),
        full_gate_validator=_passing_full_gate,
    )
    assert query_error["unresolved_change_count"] > no_route["unresolved_change_count"]


def test_dependency_closure_without_complete_routes_stays_unresolved():
    origin = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("old", "10:30", "11:30"),
        _activity("C", "12:00", "13:00"),
    ]])
    edited = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("target", "10:30", "11:30"),
        _activity("C", "12:00", "13:00"),
    ]])
    origin["itinerary"][0]["activities"][1]["transports"] = [
        {"start": "A", "end": "old", "start_time": "10:00", "end_time": "10:20"}
    ]
    origin["itinerary"][0]["activities"][2]["transports"] = [
        {"start": "old", "end": "C", "start_time": "11:30", "end_time": "11:50"}
    ]
    edited["itinerary"][0]["activities"][1]["transports"] = [
        {"start": "A", "end": "target", "start_time": "10:00", "end_time": "10:20"}
    ]
    edited["itinerary"][0]["activities"][2]["transports"] = [
        {"start": "target", "end": "C", "start_time": "11:30", "end_time": "11:50"}
    ]
    constraints = [{
        "id": "target",
        "source": "edit",
        "type": "required_attraction_name",
        "target": {"activity_type": "attraction"},
        "operator": "contains_all",
        "value": ["target"],
    }]

    def target_gate(plan):
        present = any(
            activity.get("position") == "target"
            for day in plan.get("itinerary", [])
            for activity in day.get("activities", [])
        )
        return FullGateResult(
            passed=present,
            components={"level1": True, "level2": present},
            reason="ok" if present else "target_missing",
        )

    result = analyze_cascade(
        origin,
        edited,
        constraints,
        full_gate_validator=target_gate,
    )
    assert result["rollback_required_support_change_count"] == 0
    closure_units = [
        item for item in result["unresolved_units"]
        if item["evidence_source"] == "dependency_closure"
    ]
    assert len(closure_units) == 1
    assert closure_units[0]["evidence_complete"] is False
    assert "route_evidence_incomplete" in closure_units[0]["incomplete_reasons"]
    assert closure_units[0]["atomic_op_types"] == ["change_transport"]
    assert closure_units[0]["dependency_closure_members"] == [
        {"kind": "origin", "ref": "day1_act1"}
    ]
    assert closure_units[0]["route_requests"] == [
        {
            "city": "",
            "start": "target",
            "end": "C",
            "start_time": "11:30",
        }
    ]


def _cross_city_rollback_case(origin_endpoint="成都东站", edited_endpoint="上海站"):
    origin = _plan([[
        _activity(origin_endpoint, "09:00", "10:00"),
        _activity("上海博物馆", "11:00", "12:00"),
        _activity("old target", "13:00", "14:00"),
    ]])
    edited = _plan([[
        _activity(edited_endpoint, "09:00", "10:00"),
        _activity("上海博物馆", "11:00", "12:00"),
        _activity("target", "13:00", "14:00"),
    ]])
    origin["target_city"] = edited["target_city"] = "上海"
    constraints = [{
        "id": "target",
        "source": "edit",
        "type": "required_attraction_name",
        "target": {"activity_type": "attraction"},
        "operator": "contains_all",
        "value": ["target"],
    }]
    return origin, edited, constraints


def test_endpoint_city_resolution_is_exact_and_ambiguous_names_are_unknown():
    assert _resolve_endpoint_city("成都东站") == "成都"
    assert _resolve_endpoint_city("上海博物馆") == "上海"
    assert _resolve_endpoint_city("世界之窗") is None
    assert _resolve_endpoint_city("火星站") is None


def test_rollback_new_cross_city_gap_is_hard_without_route_cache_lookup():
    origin, edited, constraints = _cross_city_rollback_case()

    class RejectingRouteCache:
        def lookup(self, *_args):
            raise AssertionError("cross-city gaps must not query local routes")

    result = analyze_cascade(
        origin,
        edited,
        constraints,
        route_evidence_cache=RejectingRouteCache(),
        full_gate_validator=_passing_full_gate,
    )
    assert result["hard_required_support_change_count"] == 1
    assert result["unresolved_change_count"] == 0
    unit = result["required_support_units"][0]
    assert unit["reason"] == "rollback_introduces_cross_city_gap"
    assert unit["route_evidence_scope"] == []
    assert unit["new_cross_city_gap_instances"] == [{
        "start": "成都东站",
        "start_city": "成都",
        "end": "上海博物馆",
        "end_city": "上海",
        "count": 1,
    }]
    classified = sum([
        result["four_way_direct_target_count"],
        result["hard_required_support_change_count"],
        result["scope_authorized_completion_change_count"],
        result["avoidable_extra_change_count"],
        result["unresolved_change_count"],
    ])
    assert classified == result["four_way_total_impact_count"]


def test_cross_city_gap_already_in_baseline_does_not_block_a_positive_witness():
    origin, edited, constraints = _cross_city_rollback_case(
        origin_endpoint="成都东站",
        edited_endpoint="成都东站",
    )
    origin["itinerary"][0]["activities"][0]["start_time"] = "09:00"
    edited["itinerary"][0]["activities"][0]["start_time"] = "08:00"
    result = analyze_cascade(
        origin,
        edited,
        constraints,
        full_gate_validator=_passing_full_gate,
    )
    assert result["hard_required_support_change_count"] == 0
    assert result["verified_removable_change_count"] == 1
    unit = result["verified_removable_units"][0]
    assert unit["witness_found"] is True
    assert unit["witness"]["gate"]["passed"] is True


def test_unknown_endpoint_city_keeps_local_route_evidence_unresolved():
    origin, edited, constraints = _cross_city_rollback_case(
        origin_endpoint="火星站",
        edited_endpoint="上海站",
    )
    result = analyze_cascade(
        origin,
        edited,
        constraints,
        full_gate_validator=_passing_full_gate,
    )
    assert result["hard_required_support_change_count"] == 0
    assert result["unresolved_change_count"] == 1
    assert result["unresolved_units"][0]["reason"] == "rollback_requires_transport_repair"
    assert result["unresolved_units"][0]["new_cross_city_gap_instances"] == []


def test_explicit_intercity_activity_connecting_both_cities_is_not_a_gap():
    plan = _plan([[
        _activity("成都博物馆", "09:00", "10:00"),
        {
            "type": "train",
            "TrainID": "G1",
            "start": "成都东站",
            "end": "上海站",
            "start_time": "10:30",
            "end_time": "14:30",
            "transports": [],
        },
        _activity("上海博物馆", "15:00", "16:00"),
    ]])
    assert _cross_city_gap_signature(plan) == Counter()


def test_cross_city_gap_signature_is_stable_when_activity_indexes_shift():
    baseline = _plan([[
        _activity("成都东站", "09:00", "10:00"),
        _activity("上海博物馆", "11:00", "12:00"),
    ]])
    shifted = _plan([[
        _activity("未知地点", "08:00", "08:30"),
        _activity("成都东站", "09:00", "10:00"),
        _activity("上海博物馆", "11:00", "12:00"),
    ]])
    before = _cross_city_gap_signature(baseline)
    after = _cross_city_gap_signature(shifted)
    assert before == after
    assert _new_cross_city_gap_instances(after, before) == []


def test_route_cache_hash_is_deterministic_and_tampering_is_rejected(tmp_path):
    cache = _route_cache(20).payload
    reordered = cache_payload([
        next(iter(cache["entries"].values()))
    ], tool_fingerprint={"different_metadata": True})
    assert reordered["entries_sha256"] == cache["entries_sha256"]

    path = tmp_path / "routes.json"
    tampered = json.loads(json.dumps(cache))
    entry = next(iter(tampered["entries"].values()))
    entry["modes"]["taxi"]["rows"][0]["duration"] = 999
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        RouteEvidenceCache.load(path)


def test_task_fingerprint_ignores_run_paths_but_sample_id_disambiguates_duplicates():
    left = {"case_id": "old", "edit_query": "q", "origin_plan": _plan([[]]), "metadata": {
        "sample_id": "old", "sample_path": "/old/batches/wrong/sample_1.json",
        "primary_conflict": ["structural", "resource", "overlap"], "edit_target_constraints": [],
    }}
    right = json.loads(json.dumps(left))
    right["case_id"] = "new"
    right["metadata"].update({
        "sample_id": "new", "sample_path": "/new/batches/right/sample_1.json",
        "primary_conflict": ["parameter", "resource", "overflow"],
    })
    assert input_fingerprint(left) == input_fingerprint(right)
    assert task_key(left, "sample_1") == task_key(right, "sample_1")
    assert task_key(left, "sample_1") != task_key(right, "sample_2")


def test_paper_manifest_requires_explicit_canonical_eligibility(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {
        "schema_version": PAPER_MANIFEST_SCHEMA,
        "models": {"m": {"records": [{
            "task_key": "key", "category": "parameter_resource_overflow",
        }]}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical eligibility"):
        load_paper_manifest(path)


def test_paper_manifest_validates_task_set_hash_and_cross_model_cohort(tmp_path):
    keys = ["task-a"]
    payload = {
        "schema_version": PAPER_MANIFEST_SCHEMA,
        "task_count": 1,
        "task_set_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "models": {
            "m1": {"records": [{
                "task_key": "task-a",
                "category": "parameter_resource_overflow",
                "strict_eligible": True,
            }]},
            "m2": {"records": [{
                "task_key": "task-a",
                "category": "parameter_resource_overflow",
                "strict_eligible": False,
            }]},
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_paper_manifest(path)["task_count"] == 1

    tampered = json.loads(json.dumps(payload))
    tampered["task_set_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="task_set_sha256"):
        load_paper_manifest(path)

    mismatched = json.loads(json.dumps(payload))
    mismatched["models"]["m2"]["records"][0]["task_key"] = "task-b"
    path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="task sets differ"):
        load_paper_manifest(path)


def test_metadata_category_prefers_primary_conflict_over_historical_path():
    payload = {"metadata": {
        "primary_conflict": ["parameter", "resource", "overflow"],
        "sample_path": "/x/batches/structural_resource_overlap/sample_000003.json",
    }}
    assert metadata_category(payload) == "parameter_resource_overflow"


def test_semantic_failure_is_not_direct_success(monkeypatch):
    monkeypatch.setattr(cascade_analysis, "_fine_semantic_type", lambda activity, city: "公园")
    origin = _plan([[_activity("A", "09:00", "10:00")]])
    edited = _plan([[_activity("A", "09:00", "10:00"), _activity("park", "11:00", "12:00")]])
    result = analyze_cascade(origin, edited, [{
        "id": "semantic", "source": "edit", "type": "semantic_type_requirement",
        "target": {"activity_type": "attraction"}, "operator": "includes_type", "value": "自然风光",
    }])
    assert result["target_satisfied"] is False
    assert result["four_way_direct_target_count"] == 0


def test_target_day_context_does_not_make_neighbor_changes_direct():
    origin = _plan([[
        _activity("old", "09:00", "10:00"),
        _activity("neighbor", "11:00", "12:00"),
    ]])
    edited = _plan([[
        _activity("target", "09:00", "10:00"),
        _activity("neighbor", "12:00", "13:00"),
    ]])
    result = analyze_cascade(origin, edited, [
        {"type": "semantic_type_requirement", "value": "自然风光", "target": {"activity_type": "attraction", "day": 1}},
        {"type": "poi_day_binding", "value": 1, "target": {"poi_name": "target"}},
    ])
    assert result["attribution_mode"] == "exact_entity"
    assert result["direct_target_change_count"] == 1
    assert result["cascade_change_count"] == 1


def test_real_named_cross_day_move_triggers_day_allocation_proxy():
    origin = _plan([
        [_activity("target", "09:00", "10:00")],
        [_activity("B", "11:00", "12:00")],
    ])
    edited = _plan([
        [_activity("B", "11:00", "12:00")],
        [_activity("target", "09:00", "10:00")],
    ])
    result = analyze_cascade(origin, edited, [
        {"type": "poi_day_binding", "value": 2, "target": {"poi_name": "target"}},
    ])
    assert "day_allocation_proxy" in result["attribution_mode"]
    assert result["direct_target_change_count"] == 1
    # B also crosses days, but it is not the named binding target.
    assert result["cascade_change_count"] == 1


def test_forbidden_name_is_grounded_in_origin_deletion():
    origin = _plan([[_activity("forbidden", "09:00", "10:00"), _activity("B", "11:00", "12:00")]])
    edited = _plan([[_activity("B", "11:00", "12:00")]])
    result = analyze_cascade(origin, edited, [{"type": "forbidden_attraction_name", "value": ["forbidden"], "target": {}}])
    assert result["attribution_mode"] == "exact_entity"
    assert result["direct_target_change_count"] == 1


def test_full_level_gate_regression_dominates_avoidable_classification():
    origin = _plan([[
        _activity("old hotel", "09:00", "10:00", kind="accommodation"),
        _activity("old target", "11:00", "12:00"),
    ]])
    edited = _plan([[
        _activity("old hotel", "08:00", "10:00", kind="accommodation"),
        _activity("target", "11:00", "12:00"),
    ]])
    constraints = [{
        "id": "target", "source": "edit", "type": "required_attraction_name",
        "target": {"activity_type": "attraction"}, "operator": "contains_all",
        "value": ["target"],
    }]

    def full_gate(plan):
        activities = [
            item for day in plan["itinerary"] for item in day["activities"]
        ]
        names = {item.get("position") for item in activities}
        hotel_start = activities[0].get("start_time")
        passed = (
            ("old target" in names and hotel_start == "09:00")
            or ("target" in names and hotel_start == "08:00")
        )
        return FullGateResult(
            passed=passed,
            components={"level1.origin_logical_preservation": passed, "level2": True},
        )

    result = analyze_cascade(
        origin, edited, constraints, full_gate_validator=full_gate,
    )
    assert result["avoidable_extra_change_count"] == 0
    assert result["hard_required_support_change_count"] == 1
    unit = result["required_support_units"][0]
    assert unit["reason"] == "rollback_breaks_full_level_gate"
    assert "level1.origin_logical_preservation" in unit["full_gate_regressions"]


def test_full_level_gate_regression_dominates_day_count_authorization():
    origin = _plan([[_activity("A", "09:00", "10:00")]])
    edited = _plan([[_activity("A", "09:00", "10:00")], [
        _activity("breakfast", "08:00", "08:30", kind="breakfast"),
        _activity("B", "09:00", "10:00"),
    ]])
    constraints = [{
        "id": "days", "source": "edit", "type": "day_count",
        "operator": "==", "value": 2, "target": {},
    }]

    def full_gate(plan):
        day2_count = len(plan["itinerary"][1]["activities"])
        passed = day2_count == 2
        return FullGateResult(
            passed=passed,
            components={"level1.origin_logical_preservation": passed, "level2": True},
        )

    result = analyze_cascade(
        origin, edited, constraints, full_gate_validator=full_gate,
    )
    assert result["scope_authorized_completion_change_count"] == 0
    assert result["hard_required_support_change_count"] == 2


def test_violation_signature_detects_an_additional_same_code_instance(monkeypatch):
    monkeypatch.setattr(cascade_analysis, "verify_constraints", lambda plan, objects: [])

    def fake_evaluate(plan):
        count = int(plan["violation_count"])
        violations = [
            SimpleNamespace(
                code="large_idle_gap",
                day=1,
                activity_index=index,
                message=f"gap at activity {index}",
                evidence={"activity": index},
            )
            for index in range(count)
        ]
        return SimpleNamespace(hygiene_violations=violations, quality_violations=[])

    monkeypatch.setattr(cascade_analysis, "evaluate_plan", fake_evaluate)
    _, baseline = _verification_signature({"violation_count": 1}, [])
    _, candidate = _verification_signature({"violation_count": 2}, [])
    new_instances = _new_violation_instances(candidate, baseline)
    assert len(new_instances) == 1
    assert new_instances[0]["code"] == "large_idle_gap"


def test_candidate_exception_is_isolated_and_count_conservation_holds():
    origin = _plan([[_activity("A", "09:00", "10:00")]])
    edited = _plan([[_activity("A", "10:00", "11:00")]])
    calls = 0

    def flaky_gate(_plan):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("candidate failure")
        return _passing_full_gate(_plan)

    result = analyze_cascade(
        origin,
        edited,
        [{"id": "time", "source": "edit", "type": "day_end_time_limit",
          "operator": "<=", "value": "18:00", "target": {}}],
        full_gate_validator=flaky_gate,
    )
    classified = sum([
        result["four_way_direct_target_count"],
        result["hard_required_support_change_count"],
        result["scope_authorized_completion_change_count"],
        result["avoidable_extra_change_count"],
        result["unresolved_change_count"],
    ])
    assert classified == result["four_way_total_impact_count"]
    assert result["unresolved_change_count"] == 1


def test_same_day_reorder_rollback_restores_origin_position():
    origin = _plan([[
        _activity("A", "09:00", "10:00"),
        _activity("B", "11:00", "12:00"),
    ]])
    edited = _plan([[
        _activity("B", "09:00", "10:00"),
        _activity("A", "11:00", "12:00"),
    ]])
    sequence = infer_edit_sequence(origin, edited)
    origin_to_edited = {
        pair.origin_ref: pair.edited_ref for pair in sequence.matched_pairs
    }
    rolled_back = _rollback_unit(
        edited, "origin", "day1_act0",
        _raw_activities(origin), _raw_activities(edited), origin_to_edited,
        {"day1_act0", "day1_act1"},
    )
    assert [
        item["position"] for item in rolled_back["itinerary"][0]["activities"]
    ] == ["A", "B"]


def test_transport_field_change_is_included_in_rollback_attribution():
    origin = _plan([[_activity("A", "09:00", "10:00")]])
    edited = _plan([[_activity("A", "09:00", "10:00")]])
    origin["itinerary"][0]["activities"][0]["transports"] = [{
        "start": "hotel",
        "end": "A",
        "mode": "metro",
        "start_time": "08:30",
        "end_time": "08:50",
        "distance": 8,
        "cost": 6,
    }]
    edited["itinerary"][0]["activities"][0]["transports"] = [{
        "start": "hotel",
        "end": "A",
        "mode": "metro",
        "start_time": "08:30",
        "end_time": "08:50",
        "distance": 8,
        "cost": 4,
    }]
    constraints = [{
        "id": "transport_budget",
        "source": "edit",
        "type": "innercity_transport_cost_total",
        "operator": "<=",
        "value": 4,
        "target": {},
    }]

    def full_gate(plan):
        cost = plan["itinerary"][0]["activities"][0]["transports"][0]["cost"]
        passed = cost <= 4
        return FullGateResult(
            passed=passed,
            components={"level1": True, "level2": passed},
        )

    result = analyze_cascade(
        origin,
        edited,
        constraints,
        full_gate_validator=full_gate,
    )

    assert result["total_changed_activity_count"] == 1
    assert result["four_way_total_impact_count"] == 1
    assert result["hard_required_support_change_count"] == 1
    assert result["verified_removable_change_count"] == 0
    assert result["proof_coverage"] == 1.0
    assert result["required_support_units"][0]["atomic_op_types"] == [
        "change_transport"
    ]


def test_inserted_removable_unit_reports_insert_atomic_type():
    origin = _plan([[_activity("target", "09:00", "10:00")]])
    edited = _plan([[
        _activity("target", "09:00", "10:00"),
        _activity("extra", "11:00", "12:00"),
    ]])
    constraints = [{
        "id": "target",
        "source": "edit",
        "type": "required_attraction_name",
        "target": {"activity_type": "attraction"},
        "operator": "contains_all",
        "value": ["target"],
    }]

    result = analyze_cascade(
        origin,
        edited,
        constraints,
        full_gate_validator=_passing_full_gate,
    )

    assert result["verified_removable_change_count"] == 1
    assert result["verified_removable_units"][0]["kind"] == "inserted"
    assert result["verified_removable_units"][0]["atomic_op_types"] == ["insert"]


def test_multiple_semantic_requirements_allocate_direct_units_per_target(monkeypatch):
    monkeypatch.setattr(
        cascade_analysis,
        "_fine_semantic_type",
        lambda activity, city: {"new X": "X", "new Y": "Y"}.get(activity.get("position")),
    )
    monkeypatch.setattr(
        cascade_analysis,
        "_verification_signature",
        lambda plan, constraints: ({"semantic": True}, Counter()),
    )
    origin = _plan([[
        _activity("old X", "09:00", "10:00"),
        _activity("old Y", "11:00", "12:00"),
    ]])
    edited = _plan([[
        _activity("new X", "09:00", "10:00"),
        _activity("new Y", "11:00", "12:00"),
    ]])
    constraints = [
        {"type": "semantic_type_requirement", "value": "X", "target": {"activity_type": "attraction"}, "params": {"min_count": 1}},
        {"type": "semantic_type_requirement", "value": "Y", "target": {"activity_type": "attraction"}, "params": {"min_count": 1}},
    ]
    result = analyze_cascade(
        origin, edited, constraints, full_gate_validator=_passing_full_gate,
    )
    assert result["four_way_direct_target_count"] == 2
    assert result["hard_required_support_change_count"] == 0
