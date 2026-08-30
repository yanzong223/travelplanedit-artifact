"""Origin-relative feasibility checks for edited travel plans.

The benchmark must not punish an editor for a defect already present in the
provided origin plan, but it must reject defects introduced or worsened by the
edit.  Absolute ChinaTravel diagnostics are retained separately so the two
views remain auditable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from data_clean.rules import (
    HYGIENE_CONFIG,
    LOW_SIGNAL_ACTIVITY_TYPES,
    QUALITY_GATE_CONFIG,
    parse_time_to_minute,
)


SCORING_MODE = "origin_relative_non_regression_v1"

# These errors are never grandfathered.  They represent malformed plans or
# contradictory timelines rather than subjective plan-quality shortcomings.
NON_GRANDFATHERABLE_CODES = {
    "city_mismatch",
    "day_boundary_transfer",
    "invalid_poi",
    "invalid_time_value",
    "missing_itinerary",
    "missing_people_number",
    "timeline_overlap",
    "transport_after_activity_start",
    "transport_before_previous_end",
}

_NUMERIC_SEVERITY_FIELDS = {
    "dense_day_attractions": "attraction_count",
    "duplicate_poi": "count",
    "large_idle_gap": "gap_minutes",
    "long_walk_distance": "distance_km",
    "long_walk_duration": "duration_min",
    "post_transport_idle_gap": "gap_minutes",
    "pre_transport_idle_gap": "gap_minutes",
}


def _as_violation_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    return {
        "code": str(getattr(item, "code", "unknown")),
        "day": int(getattr(item, "day", 0) or 0),
        "activity_index": getattr(item, "activity_index", None),
        "message": str(getattr(item, "message", "")),
        "evidence": dict(getattr(item, "evidence", {}) or {}),
    }


def _day_activities(plan: dict[str, Any], day_number: int) -> list[dict[str, Any]]:
    itinerary = plan.get("itinerary")
    if not isinstance(itinerary, list):
        return []
    for position, day in enumerate(itinerary, start=1):
        if not isinstance(day, dict):
            continue
        try:
            candidate_day = int(day.get("day", position))
        except (TypeError, ValueError):
            candidate_day = position
        if candidate_day == day_number:
            activities = day.get("activities")
            return activities if isinstance(activities, list) else []
    return []


def _activity_anchor(
    plan: dict[str, Any], day_number: int, activity_index: Any,
) -> str:
    if not isinstance(activity_index, int):
        return "__day__"
    activities = _day_activities(plan, day_number)
    if activity_index < 0 or activity_index >= len(activities):
        return f"__index__:{activity_index}"
    activity = activities[activity_index]
    if not isinstance(activity, dict):
        return f"__index__:{activity_index}"
    activity_type = str(activity.get("type") or "unknown").strip().casefold()
    if activity_type in {"train", "airplane"}:
        service_id = (
            activity.get("TrainID")
            or activity.get("FlightID")
            or activity.get("train_id")
            or activity.get("flight_id")
            or ""
        )
        parts = (
            activity_type,
            activity.get("start") or "",
            activity.get("end") or "",
            service_id,
        )
    else:
        parts = (activity_type, activity.get("position") or "")
    return "|".join(str(part).strip().casefold() for part in parts)


def _plan_level_discriminator(violation: dict[str, Any]) -> str:
    evidence = violation.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    code = str(violation.get("code") or "")
    if code == "duplicate_poi":
        return str(evidence.get("poi_name") or "").strip().casefold()
    if code == "invalid_time_value":
        return str(evidence.get("field") or "").strip().casefold()
    return ""


def _severity(violation: dict[str, Any]) -> float | None:
    code = str(violation.get("code") or "")
    evidence = violation.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    if code == "early_attraction":
        try:
            minute = parse_time_to_minute(evidence.get("start_time"))
        except ValueError:
            return None
        return -float(minute) if minute is not None else None
    field = _NUMERIC_SEVERITY_FIELDS.get(code)
    value = evidence.get(field) if field else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _evidence_signature(violation: dict[str, Any]) -> str:
    evidence = violation.get("evidence")
    return json.dumps(
        evidence if isinstance(evidence, dict) else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _defect(
    plan: dict[str, Any], tier: str, violation: dict[str, Any], *, source: str,
) -> dict[str, Any]:
    day = int(violation.get("day", 0) or 0)
    activity_index = violation.get("activity_index")
    anchor = _activity_anchor(plan, day, activity_index)
    discriminator = (
        _plan_level_discriminator(violation)
        if not isinstance(activity_index, int)
        else ""
    )
    normalized = dict(violation)
    normalized["tier"] = tier
    normalized["source"] = source
    normalized["activity_anchor"] = anchor
    normalized["severity"] = _severity(violation)
    normalized["match_key"] = [
        tier,
        str(violation.get("code") or "unknown"),
        day,
        anchor,
        discriminator,
    ]
    normalized["evidence_signature"] = _evidence_signature(violation)
    return normalized


def _safe_minute(value: Any) -> int | None:
    try:
        return parse_time_to_minute(value)
    except ValueError:
        return None


def _low_signal_wait_violations(
    plan: dict[str, Any], *, threshold: int,
) -> list[dict[str, Any]]:
    """Expose long waits hidden by the legacy meal/accommodation exemption."""
    violations: list[dict[str, Any]] = []
    itinerary = plan.get("itinerary")
    if not isinstance(itinerary, list):
        return violations
    for day_position, day in enumerate(itinerary, start=1):
        if not isinstance(day, dict):
            continue
        try:
            day_number = int(day.get("day", day_position))
        except (TypeError, ValueError):
            day_number = day_position
        activities = day.get("activities")
        if not isinstance(activities, list):
            continue
        previous_end: int | None = None
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict):
                continue
            start = _safe_minute(activity.get("start_time"))
            end = _safe_minute(activity.get("end_time"))
            transports = activity.get("transports")
            transports = transports if isinstance(transports, list) else []
            if str(activity.get("type") or "") in LOW_SIGNAL_ACTIVITY_TYPES:
                valid_transports = [item for item in transports if isinstance(item, dict)]
                first_transport_start = next(
                    (
                        minute
                        for item in valid_transports
                        if (minute := _safe_minute(item.get("start_time"))) is not None
                    ),
                    None,
                )
                last_transport_end = next(
                    (
                        minute
                        for item in reversed(valid_transports)
                        if (minute := _safe_minute(item.get("end_time"))) is not None
                    ),
                    None,
                )
                label = str(activity.get("position") or activity.get("type") or "activity")
                if previous_end is not None and first_transport_start is not None:
                    gap = first_transport_start - previous_end
                    if gap > threshold:
                        violations.append({
                            "code": "pre_transport_idle_gap",
                            "day": day_number,
                            "activity_index": activity_index,
                            "message": (
                                f"{label} starts its transfer {gap} minutes after the "
                                "previous activity ends."
                            ),
                            "evidence": {"gap_minutes": gap},
                        })
                elif previous_end is not None and start is not None and not valid_transports:
                    gap = start - previous_end
                    if gap > threshold:
                        violations.append({
                            "code": "large_idle_gap",
                            "day": day_number,
                            "activity_index": activity_index,
                            "message": f"{label} has a {gap}-minute idle gap.",
                            "evidence": {"gap_minutes": gap},
                        })
                if last_transport_end is not None and start is not None:
                    gap = start - last_transport_end
                    if gap > threshold:
                        violations.append({
                            "code": "post_transport_idle_gap",
                            "day": day_number,
                            "activity_index": activity_index,
                            "message": (
                                f"{label} begins {gap} minutes after its last transfer ends."
                            ),
                            "evidence": {"gap_minutes": gap},
                        })
            if end is not None:
                previous_end = end
    return violations


def _collect_defects(
    plan: dict[str, Any], evaluation: Any,
) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []
    for tier, attribute in (
        ("hygiene", "hygiene_violations"),
        ("quality", "quality_violations"),
    ):
        for item in getattr(evaluation, attribute, []) or []:
            defects.append(
                _defect(plan, tier, _as_violation_dict(item), source="chinatravel_absolute")
            )
        threshold = (
            HYGIENE_CONFIG.thresholds.max_idle_gap_min
            if tier == "hygiene"
            else QUALITY_GATE_CONFIG.thresholds.max_idle_gap_min
        )
        for item in _low_signal_wait_violations(plan, threshold=threshold):
            defects.append(_defect(plan, tier, item, source="wait_non_regression"))
    return defects


def _compatible(origin: dict[str, Any], edited: dict[str, Any]) -> bool:
    origin_severity = origin.get("severity")
    edited_severity = edited.get("severity")
    if isinstance(origin_severity, (int, float)) and isinstance(edited_severity, (int, float)):
        return float(edited_severity) <= float(origin_severity)
    return origin.get("evidence_signature") == edited.get("evidence_signature")


def compare_origin_relative_feasibility(
    origin_plan: dict[str, Any],
    edited_plan: dict[str, Any],
    origin_evaluation: Any,
    edited_evaluation: Any,
) -> dict[str, Any]:
    """Classify edited defects as inherited, new/worsened, or resolved."""
    origin_defects = _collect_defects(origin_plan, origin_evaluation)
    edited_defects = _collect_defects(edited_plan, edited_evaluation)
    origin_groups: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, defect in enumerate(origin_defects):
        origin_groups[tuple(defect["match_key"])].append((index, defect))

    consumed: set[int] = set()
    inherited: list[dict[str, Any]] = []
    attributable: list[dict[str, Any]] = []
    for edited in edited_defects:
        candidates = [
            (index, origin)
            for index, origin in origin_groups.get(tuple(edited["match_key"]), [])
            if index not in consumed
        ]
        compatible = [item for item in candidates if _compatible(item[1], edited)]
        matched = compatible[0] if compatible else None
        code = str(edited.get("code") or "")
        if matched is not None:
            index, origin = matched
            consumed.add(index)
            comparison = dict(edited)
            comparison["origin_severity"] = origin.get("severity")
            if code in NON_GRANDFATHERABLE_CODES:
                comparison["classification"] = "non_grandfatherable"
                attributable.append(comparison)
            else:
                comparison["classification"] = "inherited_not_worsened"
                inherited.append(comparison)
            continue

        comparison = dict(edited)
        comparison["classification"] = "worsened" if candidates else "new"
        if candidates:
            comparison["origin_severity"] = candidates[0][1].get("severity")
        attributable.append(comparison)

    resolved = [
        {**defect, "classification": "resolved"}
        for index, defect in enumerate(origin_defects)
        if index not in consumed
    ]
    return {
        "pass": not attributable,
        "scoring_mode": SCORING_MODE,
        "policy": (
            "Inherited feasibility defects pass only when unchanged or improved; "
            "new, worsened, and non-grandfatherable defects fail."
        ),
        "new_or_worsened_violations": attributable,
        "inherited_violations": inherited,
        "resolved_violations": resolved,
        "origin_defect_count": len(origin_defects),
        "edited_defect_count": len(edited_defects),
        "new_or_worsened_count": len(attributable),
        "inherited_count": len(inherited),
        "resolved_count": len(resolved),
    }
