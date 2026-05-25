"""Category-level guards for solver-valid data generation."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple


NORMAL_PROFILE = "normal"
INTENTIONAL_INFEASIBLE_PROFILE = "intentional_infeasible"
OFF_PROFILE = "off"

RESOURCE_CONSTRAINT_TYPES = {
    "budget_limit",
    "daily_budget_infeasible",
    "multi_day_budget_overflow",
    "resource_overlap",
}

ALLOWED_EDIT_VIOLATIONS = {"overflow", "overlap", "discontinuity", "incompatibility"}
SOFT_PRIMARY_VIOLATIONS = {"infeasible", "incompatibility"}
CONFLICT_SCOPE_LEVEL = {"parameter": 0, "structural": 1, "compositional": 2}
DIMENSION_COMPATIBILITY = {
    "temporal": {"sequence"},
    "sequence": {"temporal"},
}
PRIMARY_ALIGNMENT_OVERRIDE_BUCKETS = {
    "activity_budget_limit": {
        ("parameter", "resource", "overflow"),
    },
    "resource_overlap": {
        ("structural", "resource", "overflow"),
        ("structural", "resource", "overlap"),
        ("structural", "temporal", "overflow"),
    },
    "ticket_price_cap_or_substitute": {
        ("structural", "resource", "overflow"),
        ("structural", "temporal", "overlap"),
    },
    "time_window": {
        ("parameter", "structural", "discontinuity"),
        ("structural", "temporal", "overflow"),
    },
    "add_days": {
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "discontinuity"),
    },
    "cross_day_restructure": {
        ("structural", "structural", "overlap"),
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "overlap"),
    },
    "semantic_and_specific": {
        ("structural", "temporal", "overlap"),
    },
    "category_overlap": {
        ("structural", "temporal", "overlap"),
    },
    "multi_city_split": {
        ("structural", "structural", "overlap"),
        ("compositional", "structural", "overlap"),
    },
}
PRIMARY_ACCEPTABLE_ALTERNATIVE_BUCKETS = {
    "cross_day_restructure": {
        ("compositional", "temporal", "overlap"),
        ("compositional", "temporal", "overflow"),
    },
    "add_days": {
        ("compositional", "temporal", "overflow"),
        ("compositional", "temporal", "discontinuity"),
    },
    "time_window": {
        ("parameter", "structural", "discontinuity"),
        ("structural", "temporal", "overflow"),
    },
    "resource_overlap": {
        ("structural", "resource", "overflow"),
        ("structural", "temporal", "overflow"),
    },
}


def assess_category_guard_validity(
    *,
    origin_plan: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    origin_logical_constraints: Optional[Sequence[Dict[str, Any]]] = None,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    conflict_set: Optional[Sequence[Sequence[str]]] = None,
    primary_conflict: Optional[Sequence[str]] = None,
    target_bucket: Optional[Sequence[str]] = None,
    solver_feasibility: Optional[Dict[str, Any]] = None,
    profile: str = NORMAL_PROFILE,
) -> Dict[str, Any]:
    """Assess whether a generated sample is valid for its intended category.

    These guards are deliberately stricter than static solver feasibility: they
    ensure the sample belongs to the intended category and that category-specific
    generation parameters do not produce solver-invalid normal samples.
    """

    del origin_plan, origin_logical_constraints
    normalized_profile = _normalize_profile(profile)
    if normalized_profile == OFF_PROFILE:
        return {
            "pass": True,
            "skipped": True,
            "profile": OFF_PROFILE,
            "errors": [],
            "warnings": [],
            "details": {"reason": "category guard disabled"},
        }

    target = _normalize_bucket(target_bucket)
    conflict_tuples = [_normalize_bucket(item) for item in (conflict_set or [])]
    conflict_tuples = [item for item in conflict_tuples if item is not None]
    primary = _normalize_bucket(primary_conflict)
    solver = solver_feasibility if isinstance(solver_feasibility, dict) else {}
    constraint_type = _trace_constraint_type(query_generation_trace)

    errors = []
    warnings = []
    details: Dict[str, Any] = {
        "profile": normalized_profile,
        "target_bucket": list(target) if target else None,
        "primary_conflict": list(primary) if primary else None,
        "conflict_set": [list(item) for item in conflict_tuples],
        "constraint_type": constraint_type,
        "solver_status": solver.get("status"),
    }
    semantic_type = _semantic_preferred_type(query_generation_trace, constraints)
    if semantic_type:
        details["semantic_preferred_type"] = semantic_type
    selection = _semantic_type_selection(query_generation_trace)
    if selection:
        details["semantic_type_selection"] = selection

    target_alignment = _assess_target_alignment(
        target=target,
        primary=primary,
        conflict_tuples=conflict_tuples,
        constraint_type=constraint_type,
    )
    details["target_alignment"] = target_alignment

    if target is None:
        errors.append("missing target_bucket")
    else:
        if not target_alignment["target_bucket_present_in_conflict_set"]:
            errors.append(f"target_bucket {list(target)} not found in conflict_set")
        if primary is not None and not target_alignment["primary_allowed"]:
            errors.append(f"primary_conflict {list(primary)} does not match target_bucket {list(target)}")

    blocking_codes = _solver_blocking_codes(solver)
    details["solver_blocking_codes"] = sorted(blocking_codes)

    if normalized_profile == INTENTIONAL_INFEASIBLE_PROFILE:
        if not _bucket_is_infeasible(target):
            warnings.append("intentional_infeasible profile used for a non-infeasible target bucket")
        if solver.get("pass") is True:
            warnings.append("solver feasibility did not prove this intentional infeasible sample blocked")
        return {
            "pass": not errors,
            "profile": normalized_profile,
            "errors": errors,
            "warnings": warnings,
            "details": details,
        }

    if _bucket_is_infeasible(target):
        errors.append("normal profile excludes intentional infeasible target buckets")

    if solver.get("pass") is False:
        status = str(solver.get("status") or "")
        if status == "infeasible_by_inventory":
            errors.append("normal profile inventory guard failed")
        elif status == "infeasible_by_capacity":
            errors.append("normal profile temporal/capacity guard failed")
        elif status == "infeasible_by_spatial_bound":
            errors.append("normal profile spatial bound guard failed")
        elif status == "infeasible_by_lower_bound":
            errors.append("normal profile budget/resource lower-bound guard failed")
        elif status == "unknown_needs_probe":
            errors.append("normal profile category guard requires manual review/probe")
        else:
            errors.append(f"normal profile solver feasibility failed: {status}")

    if target and target[1] == "semantic":
        semantic_blockers = {
            "missing_required_attraction",
            "missing_required_attraction_type",
            "missing_required_hotel",
            "missing_required_hotel_feature",
            "missing_required_restaurant",
            "missing_required_restaurant_type",
        }
        if semantic_blockers.intersection(blocking_codes):
            errors.append("semantic category guard failed inventory requirements")
        if constraint_type in {"type_filter", "preference_constraint", "semantic_and_specific", "category_overlap"} and not semantic_type:
            errors.append("semantic category guard missing preferred_type")

    if "required_day_exceeds_day_count" in blocking_codes:
        errors.append("category guard failed day binding: required day exceeds day_count")

    if _is_resource_budget_sample(target, constraint_type, edit_target_constraints):
        interval = _resource_budget_interval_details(solver, query_generation_trace)
        details["resource_budget_interval"] = interval
        if not interval.get("applicable"):
            errors.append("resource category guard missing budget interval evidence")
        elif not interval.get("valid"):
            errors.append(
                "resource category guard requires static_lower_bound < budget < conflict_proof_baseline"
            )

    return {
        "pass": not errors,
        "profile": normalized_profile,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def _normalize_profile(profile: str) -> str:
    text = str(profile or OFF_PROFILE).strip().lower()
    if text in {OFF_PROFILE, NORMAL_PROFILE, INTENTIONAL_INFEASIBLE_PROFILE}:
        return text
    return NORMAL_PROFILE


def _normalize_bucket(value: Optional[Sequence[str]]) -> Optional[Tuple[str, str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    return tuple(str(part).strip() for part in value)  # type: ignore[return-value]


def _bucket_is_infeasible(bucket: Optional[Tuple[str, str, str]]) -> bool:
    return bool(bucket and any(str(part).strip().lower() == "infeasible" for part in bucket))


def _dimensions_compatible(primary_dim: Optional[str], target_dim: Optional[str]) -> bool:
    if not primary_dim or not target_dim:
        return False
    if primary_dim == target_dim:
        return True
    return target_dim in DIMENSION_COMPATIBILITY.get(primary_dim, set())


def _bucket_matches_target(
    candidate: Optional[Tuple[str, str, str]],
    target: Optional[Tuple[str, str, str]],
) -> bool:
    if candidate is None or target is None:
        return False
    return (
        candidate[0] == target[0]
        and candidate[2] == target[2]
        and _dimensions_compatible(candidate[1], target[1])
    )


def _primary_soft_compatible(
    primary: Optional[Tuple[str, str, str]],
    target: Optional[Tuple[str, str, str]],
) -> bool:
    return bool(
        primary is not None
        and target is not None
        and primary[0] == target[0]
        and _dimensions_compatible(primary[1], target[1])
        and primary[2] in SOFT_PRIMARY_VIOLATIONS
        and target[2] in ALLOWED_EDIT_VIOLATIONS
    )


def _primary_scope_upgrade(
    primary: Optional[Tuple[str, str, str]],
    target: Optional[Tuple[str, str, str]],
) -> bool:
    if primary is None or target is None:
        return False
    primary_scope = CONFLICT_SCOPE_LEVEL.get(primary[0], -1)
    target_scope = CONFLICT_SCOPE_LEVEL.get(target[0], -1)
    return (
        _dimensions_compatible(primary[1], target[1])
        and primary[2] == target[2]
        and primary_scope > target_scope >= 0
    )


def _assess_target_alignment(
    *,
    target: Optional[Tuple[str, str, str]],
    primary: Optional[Tuple[str, str, str]],
    conflict_tuples: Sequence[Tuple[str, str, str]],
    constraint_type: str,
) -> Dict[str, Any]:
    acceptable = PRIMARY_ACCEPTABLE_ALTERNATIVE_BUCKETS.get(constraint_type, set())
    overrides = PRIMARY_ALIGNMENT_OVERRIDE_BUCKETS.get(constraint_type, set())
    contains_target = any(_bucket_matches_target(item, target) for item in conflict_tuples)
    primary_matches_target = _bucket_matches_target(primary, target)
    primary_is_acceptable = primary in acceptable if primary is not None else False
    primary_is_override = primary in overrides if primary is not None else False
    primary_soft = _primary_soft_compatible(primary, target)
    primary_scope_upgrade = _primary_scope_upgrade(primary, target)
    primary_allowed = (
        primary is None
        or primary_matches_target
        or (contains_target and primary_is_acceptable)
        or (contains_target and primary_is_override)
        or primary_soft
        or primary_scope_upgrade
    )
    return {
        "target_bucket_present_in_conflict_set": contains_target or primary_soft,
        "primary_matches_target_bucket": primary_matches_target,
        "primary_is_acceptable_alternative": primary_is_acceptable,
        "primary_is_override": primary_is_override,
        "primary_soft_compatible": primary_soft,
        "primary_scope_upgrade": primary_scope_upgrade,
        "primary_allowed": primary_allowed,
        "acceptable_alternative_buckets": [list(item) for item in sorted(acceptable)],
        "override_buckets": [list(item) for item in sorted(overrides)],
    }


def _trace_constraint_type(query_generation_trace: Optional[Dict[str, Any]]) -> str:
    if isinstance(query_generation_trace, dict):
        return str(query_generation_trace.get("constraint_type") or "").strip()
    return ""


def _semantic_type_selection(query_generation_trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(query_generation_trace, dict):
        return {}
    value = query_generation_trace.get("semantic_type_selection")
    if isinstance(value, dict):
        return value
    evidence = query_generation_trace.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("semantic_type_selection"), dict):
        return evidence["semantic_type_selection"]
    return {}


def _semantic_preferred_type(
    query_generation_trace: Optional[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]],
) -> str:
    if isinstance(query_generation_trace, dict):
        for key in ("surface_query_spec", "canonical_query_spec", "query_spec"):
            payload = query_generation_trace.get(key)
            if isinstance(payload, dict) and str(payload.get("preferred_type") or "").strip():
                return str(payload["preferred_type"]).strip()
        for key in ("surface_semantic_slots", "canonical_semantic_slots", "semantic_slots"):
            payload = query_generation_trace.get(key)
            if isinstance(payload, dict) and str(payload.get("preferred_type") or "").strip():
                return str(payload["preferred_type"]).strip()
    semantic = constraints.get("semantic") if isinstance(constraints, dict) else None
    if isinstance(semantic, dict) and str(semantic.get("preferred_type") or "").strip():
        return str(semantic["preferred_type"]).strip()
    return ""


def _solver_blocking_codes(solver_feasibility: Dict[str, Any]) -> set[str]:
    codes = set()
    for item in solver_feasibility.get("blocking_reasons", []) or []:
        if isinstance(item, dict) and item.get("code"):
            codes.add(str(item["code"]))
    return codes


def _is_resource_budget_sample(
    target: Optional[Tuple[str, str, str]],
    constraint_type: str,
    edit_target_constraints: Optional[Sequence[Dict[str, Any]]],
) -> bool:
    if constraint_type in RESOURCE_CONSTRAINT_TYPES:
        return True
    for item in edit_target_constraints or []:
        if isinstance(item, dict) and str(item.get("type") or "") == "budget_total":
            return True
    return False


def _resource_budget_interval_details(
    solver_feasibility: Dict[str, Any],
    query_generation_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    budget_bounds = ((solver_feasibility.get("lower_bounds") or {}).get("budget") or {})
    if not isinstance(budget_bounds, dict) or budget_bounds.get("applicable") is False:
        return {"applicable": False, "reason": "budget lower bound not applicable"}
    budget = _as_float(budget_bounds.get("budget_threshold"))
    lower = _as_float(budget_bounds.get("lower_bound_total"))
    baseline = _proof_baseline(query_generation_trace)
    if budget is None or lower is None or baseline is None:
        return {
            "applicable": False,
            "reason": "missing budget/lower_bound/conflict_baseline",
            "budget": budget,
            "static_lower_bound": lower,
            "conflict_proof_baseline": baseline,
        }
    return {
        "applicable": True,
        "budget": budget,
        "static_lower_bound": lower,
        "conflict_proof_baseline": baseline,
        "valid": lower < budget < baseline,
    }


def _proof_baseline(query_generation_trace: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(query_generation_trace, dict):
        return None
    for key in ("canonical_proof_evidence", "surface_proof_evidence"):
        payload = query_generation_trace.get(key)
        if isinstance(payload, dict):
            value = _as_float(payload.get("baseline"))
            if value is not None:
                return value
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    return None
