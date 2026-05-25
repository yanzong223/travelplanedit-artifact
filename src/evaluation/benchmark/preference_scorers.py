"""Preference scoring functions for Level2 benchmark evaluation."""

from __future__ import annotations

import statistics
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import PreferenceConstraintObject, PreferenceScoreResult
from .plan_features import (
    COUNTED_ACTIVITY_TYPES,
    extract_anchor_distance_candidates,
    extract_anchor_distances,
    extract_attraction_type_distribution,
    extract_buffer_gaps,
    extract_cost_breakdown,
    extract_daily_activity_counts,
    extract_daily_attraction_count,
    extract_daily_travel_minutes,
    extract_day_end_minutes,
    extract_food_experience_features,
    extract_hotel_anchor_distances,
    extract_inter_activity_edges,
    extract_route_smoothness_features,
    extract_theme_counter,
    extract_theme_distribution,
    extract_vibe_signals,
    extract_walk_transport_stats,
    normalized_entropy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHINATRAVEL_ROOT = PROJECT_ROOT / "Chinatravel" / "ChinaTravel"
if str(CHINATRAVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(CHINATRAVEL_ROOT))

def _average(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _inverse_cost_score(raw_value: float, *, scale: float) -> float:
    return 1.0 / (1.0 + max(raw_value, 0.0) / scale)


def _normalize_score(value: float, *, scale: float) -> float:
    if scale <= 0:
        return max(0.0, min(value, 1.0))
    return max(0.0, min(value / scale, 1.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


@lru_cache(maxsize=1)
def _world_env() -> Any:
    from chinatravel.environment.world_env import WorldEnv

    return WorldEnv()


def _compute_walk_relief_score(total_minutes: float, total_distance: float) -> float:
    minute_component = 1.0 / (1.0 + max(total_minutes, 0.0) / 90.0)
    distance_component = 1.0 / (1.0 + max(total_distance, 0.0) / 5.0)
    return (minute_component + distance_component) / 2.0


def _lookup_anchor_distances(
    plan: dict[str, Any],
    anchor: str,
    *,
    candidate_types: set[str] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    candidates = extract_anchor_distance_candidates(plan, anchor, candidate_types=candidate_types)
    anchor_hits = int(candidates.get("anchor_hits", 0) or 0)
    activity_names = list(candidates.get("activity_names", []) or [])
    if activity_names:
        env = _world_env()
        city = plan.get("target_city")
        looked_up: list[float] = []
        unresolved = 0
        for poi in activity_names:
            try:
                route = env.transportation.goto(
                    city=city,
                    start=anchor,
                    end=poi,
                    start_time="00:00",
                    transport_type="walk",
                    verbose=False,
                )
                if route and isinstance(route, list):
                    distance = route[0].get("distance")
                    if isinstance(distance, (int, float)):
                        looked_up.append(float(distance))
                        continue
            except Exception:
                pass
            unresolved += 1
        if looked_up:
            return [0.0] * anchor_hits + looked_up, {
                "anchor": anchor,
                "method": "point_to_point_lookup",
                "anchor_hits": anchor_hits,
                "candidate_count": len(activity_names),
                "resolved_count": len(looked_up),
                "unresolved_count": unresolved,
            }

    fallback = extract_anchor_distances(plan, anchor, candidate_types=candidate_types)
    if fallback:
        return fallback, {
            "anchor": anchor,
            "method": "transport_segments",
            "anchor_hits": anchor_hits,
            "candidate_count": len(activity_names),
        }
    if anchor_hits > 0:
        return [0.0] * anchor_hits, {
            "anchor": anchor,
            "method": "anchor_self_match",
            "anchor_hits": anchor_hits,
            "candidate_count": len(activity_names),
        }
    return [], {
        "anchor": anchor,
        "method": "unavailable",
        "anchor_hits": 0,
        "candidate_count": len(activity_names),
    }


def _score_distance_to_anchor(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    anchor = preference.anchor or preference.params.get("target_poi")
    if not anchor:
        return False, None, {}, "missing_anchor"
    distances, details = _lookup_anchor_distances(plan, anchor, candidate_types=set(COUNTED_ACTIVITY_TYPES))
    if distances:
        avg_distance = _average(distances)
        return True, _inverse_cost_score(avg_distance or 0.0, scale=3.0), {"raw_value": avg_distance, **details}, "ok"
    return True, 0.0, {"raw_value": 0.0, **details, "fallback": "distance_lookup_failed"}, "ok"


def _score_anchor_distance(
    plan: dict[str, Any],
    preference: PreferenceConstraintObject,
    *,
    preferred_types: set[str] | None = None,
    scale: float = 3.0,
) -> tuple[bool, float | None, dict[str, Any], str]:
    anchor = preference.anchor or preference.params.get("target_poi")
    if not anchor:
        return False, None, {}, "missing_anchor"

    candidate_types = set(preferred_types) if preferred_types is not None else set(COUNTED_ACTIVITY_TYPES)
    distances, details = _lookup_anchor_distances(plan, anchor, candidate_types=candidate_types)
    if distances:
        avg_distance = _average(distances)
        return True, _inverse_cost_score(avg_distance or 0.0, scale=scale), {"raw_value": avg_distance, **details}, "ok"
    return True, 0.0, {"raw_value": 0.0, **details, "fallback": "distance_lookup_failed"}, "ok"


def _score_theme_concentration(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    target_theme = str(
        preference.params.get("target_theme")
        or preference.params.get("canonical_facet")
        or preference.facet
        or ""
    ).strip()
    if not target_theme:
        return False, None, {}, "missing_target_theme"
    counter = extract_theme_counter(plan)
    total = sum(counter.values())
    if total <= 0:
        return True, 0.0, {"raw_value": 0.0, "target_theme": target_theme, "theme_counts": {}}, "ok"
    score = counter.get(target_theme, 0) / total
    return True, score, {"raw_value": score, "target_theme": target_theme, "theme_counts": dict(counter)}, "ok"


def _score_theme_evenness(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    distribution = extract_theme_distribution(plan)
    score = normalized_entropy(distribution.values())
    return True, score, {"raw_value": score, "theme_distribution": distribution}, "ok"


def _score_attraction_count(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    counts = extract_daily_attraction_count(plan)
    raw_value = _average([float(value) for value in counts])
    if raw_value is None:
        return True, 0.0, {"raw_value": 0.0, "fallback": "no_attractions"}, "ok"
    score = _normalize_score(raw_value, scale=4.0)
    return True, score, {"raw_value": raw_value, "counts": counts}, "ok"


def _score_category_diversity(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    distribution = extract_attraction_type_distribution(plan)
    score = normalized_entropy(distribution.values())
    return True, score, {"raw_value": score, "attraction_type_distribution": dict(distribution)}, "ok"


def _score_food_experience_count(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    features = extract_food_experience_features(plan)
    days = max(len(plan.get("itinerary", [])) if isinstance(plan, dict) else 0, 1)
    meal_count = int(features.get("meal_count", 0) or 0)
    distinct_cuisines = int(features.get("distinct_cuisine_count", 0) or 0)
    meal_count_norm = min(meal_count / max(days * 2, 1), 1.0)
    cuisine_diversity_norm = min(distinct_cuisines / 4.0, 1.0)
    score = 0.6 * meal_count_norm + 0.4 * cuisine_diversity_norm
    return True, score, {
        "raw_value": score,
        "meal_count": meal_count,
        "distinct_cuisines": distinct_cuisines,
        "distinct_recommended_food_count": int(features.get("distinct_recommended_food_count", 0) or 0),
        "recommended_food_hits": list(features.get("recommended_food_hits", [])),
    }, "ok"


def _score_vibe(plan: dict[str, Any], vibe: str) -> tuple[bool, float | None, dict[str, Any], str]:
    signals = extract_vibe_signals(plan)
    counted_activity_count = int(signals.get("counted_activity_count", 0) or 0)
    if counted_activity_count <= 0:
        return True, 0.0, {"raw_value": 0.0, "fallback": "no_counted_activities", **signals}, "ok"

    avg_daily_activity_count = float(signals.get("average_daily_activity_count") or 0.0)
    avg_daily_travel_minutes = float(signals.get("average_daily_travel_minutes") or 0.0)
    avg_day_end_minutes = float(signals.get("average_day_end_minutes") or 0.0)
    evening_ratio = _clamp01(float(signals.get("evening_activity_ratio", 0.0) or 0.0))
    late_night_ratio = _clamp01(float(signals.get("late_night_activity_ratio", 0.0) or 0.0))
    late_day_ratio = _clamp01(float(signals.get("late_day_ratio", 0.0) or 0.0))
    moderate_buffer_ratio = _clamp01(float(signals.get("moderate_buffer_ratio", 0.0) or 0.0))
    keyword_ratios = dict(signals.get("keyword_ratios", {}))
    theme_distribution = dict(signals.get("theme_distribution", {}))

    light_load_score = _inverse_cost_score(avg_daily_activity_count, scale=4.0)
    dense_load_score = _normalize_score(avg_daily_activity_count, scale=6.0)
    light_travel_score = _inverse_cost_score(avg_daily_travel_minutes, scale=90.0)
    walk_relief_score = _compute_walk_relief_score(
        float(signals.get("walk_total_minutes", 0.0) or 0.0),
        float(signals.get("walk_total_distance", 0.0) or 0.0),
    )
    early_end_score = _inverse_cost_score(max(avg_day_end_minutes - 19 * 60, 0.0), scale=180.0)
    late_end_score = _normalize_score(max(avg_day_end_minutes - 18 * 60, 0.0), scale=240.0)
    nature_share = _clamp01(float(theme_distribution.get("nature", 0.0) or 0.0))
    culture_share = _clamp01(float(theme_distribution.get("culture", 0.0) or 0.0))
    photo_share = _clamp01(float(theme_distribution.get("photo", 0.0) or 0.0))
    shopping_share = _clamp01(float(theme_distribution.get("shopping", 0.0) or 0.0))
    commercial_share = _clamp01(shopping_share + photo_share)
    relaxed_keyword_ratio = _clamp01(float(keyword_ratios.get("relaxed", 0.0) or 0.0))
    lively_keyword_ratio = _clamp01(float(keyword_ratios.get("lively", 0.0) or 0.0))
    quiet_keyword_ratio = _clamp01(float(keyword_ratios.get("quiet", 0.0) or 0.0))
    nightlife_keyword_ratio = _clamp01(float(keyword_ratios.get("nightlife", 0.0) or 0.0))

    if vibe == "relaxed":
        score = (
            0.20 * light_load_score
            + 0.18 * light_travel_score
            + 0.16 * walk_relief_score
            + 0.12 * moderate_buffer_ratio
            + 0.12 * early_end_score
            + 0.12 * relaxed_keyword_ratio
            + 0.10 * nature_share
        )
    elif vibe == "lively":
        score = (
            0.24 * dense_load_score
            + 0.18 * evening_ratio
            + 0.14 * late_end_score
            + 0.20 * max(lively_keyword_ratio, commercial_share)
            + 0.12 * photo_share
            + 0.12 * late_day_ratio
        )
    elif vibe == "quiet":
        quiet_theme_share = _clamp01(nature_share + culture_share)
        anti_nightlife = _clamp01(1.0 - max(nightlife_keyword_ratio, commercial_share, late_night_ratio))
        score = (
            0.18 * light_load_score
            + 0.18 * light_travel_score
            + 0.16 * walk_relief_score
            + 0.16 * quiet_keyword_ratio
            + 0.14 * quiet_theme_share
            + 0.18 * anti_nightlife
        )
    elif vibe == "nightlife":
        score = (
            0.28 * max(nightlife_keyword_ratio, lively_keyword_ratio * 0.7)
            + 0.22 * evening_ratio
            + 0.20 * late_night_ratio
            + 0.16 * late_day_ratio
            + 0.14 * max(late_end_score, commercial_share)
        )
    else:
        return False, None, {}, "unsupported_vibe"

    return True, _clamp01(score), {
        "raw_value": _clamp01(score),
        "target_vibe": vibe,
        "signals": signals,
        "components": {
            "light_load_score": light_load_score,
            "dense_load_score": dense_load_score,
            "light_travel_score": light_travel_score,
            "walk_relief_score": walk_relief_score,
            "moderate_buffer_ratio": moderate_buffer_ratio,
            "early_end_score": early_end_score,
            "late_end_score": late_end_score,
            "nature_share": nature_share,
            "culture_share": culture_share,
            "photo_share": photo_share,
            "shopping_share": shopping_share,
            "commercial_share": commercial_share,
            "relaxed_keyword_ratio": relaxed_keyword_ratio,
            "lively_keyword_ratio": lively_keyword_ratio,
            "quiet_keyword_ratio": quiet_keyword_ratio,
            "nightlife_keyword_ratio": nightlife_keyword_ratio,
            "evening_ratio": evening_ratio,
            "late_night_ratio": late_night_ratio,
            "late_day_ratio": late_day_ratio,
        },
    }, "ok"


def _score_route_smoothness(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    features = extract_route_smoothness_features(plan)
    edge_count = int(features.get("edge_count", 0) or 0)
    if edge_count <= 0:
        return True, 1.0, {"raw_value": 1.0, "edge_count": 0}, "ok"
    long_jump_ratio = min(max(float(features.get("long_jump_ratio", 0.0) or 0.0), 0.0), 1.0)
    time_cv = min(max(float(features.get("time_cv", 0.0) or 0.0), 0.0), 1.0)
    distance_cv = min(max(float(features.get("distance_cv", 0.0) or 0.0), 0.0), 1.0)
    penalty = 0.5 * long_jump_ratio + 0.25 * time_cv + 0.25 * distance_cv
    score = max(0.0, 1.0 - penalty)
    return True, score, {"raw_value": score, **features}, "ok"


def _score_walk_burden(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    stats = extract_walk_transport_stats(plan)
    total_minutes = float(stats["total_minutes"])
    total_distance = float(stats["total_distance"])
    if total_minutes <= 0 and total_distance <= 0:
        return True, 1.0, {"raw_value": 1.0, "segment_count": 0.0}, "ok"
    minute_component = 1.0 / (1.0 + total_minutes / 90.0)
    distance_component = 1.0 / (1.0 + total_distance / 5.0)
    score = (minute_component + distance_component) / 2.0
    return True, score, {"raw_value": score, **stats}, "ok"


def _score_experience_share(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    breakdown = extract_cost_breakdown(plan)
    total_cost = breakdown["total"]
    if total_cost <= 0:
        return False, None, {}, "non_positive_total_cost"
    ratio = breakdown["attraction"] / total_cost
    return True, ratio, {"raw_value": ratio, "experience_cost": breakdown["attraction"], "total_cost": total_cost}, "ok"


def _score_plan(plan: dict[str, Any], preference: PreferenceConstraintObject) -> tuple[bool, float | None, dict[str, Any], str]:
    facet = preference.facet
    itinerary = plan.get("itinerary", []) if isinstance(plan, dict) else []
    canonical_facet = str(preference.params.get("canonical_facet") or "")

    if preference.family == "vibe_alignment" and facet in {"relaxed", "lively", "quiet", "nightlife"}:
        facet = f"{facet}_vibe_maximize"
    elif preference.family == "vibe_alignment" and canonical_facet in {"relaxed", "lively", "quiet", "nightlife"}:
        if facet not in {"relaxed_vibe_maximize", "lively_vibe_maximize", "quiet_vibe_maximize", "nightlife_vibe_maximize"}:
            facet = f"{canonical_facet}_vibe_maximize"

    if facet == "daily_activity_load_minimize":
        counts = extract_daily_activity_counts(plan)
        raw_value = _average([float(value) for value in counts])
        if raw_value is None:
            return True, 1.0, {"raw_value": 0.0, "fallback": "no_activities"}, "ok"
        score = _inverse_cost_score(raw_value, scale=4.0)
        return True, score, {"raw_value": raw_value, "counts": counts}, "ok"

    if facet == "daily_travel_time_minimize":
        minutes = extract_daily_travel_minutes(plan, exclude_intercity=True)
        raw_value = _average(minutes)
        if raw_value is None:
            return True, 1.0, {"raw_value": 0.0, "fallback": "no_travel_segments"}, "ok"
        return True, _inverse_cost_score(raw_value, scale=90.0), {"raw_value": raw_value, "minutes": minutes}, "ok"

    if facet == "transport_time_minimize":
        edges = extract_inter_activity_edges(plan)
        raw_value = _average([edge.travel_minutes for edge in edges if edge.travel_minutes > 0])
        if raw_value is None:
            return True, 1.0, {"raw_value": 0.0, "fallback": "no_transport_segments"}, "ok"
        return True, _inverse_cost_score(raw_value, scale=60.0), {"raw_value": raw_value, "edge_count": len(edges)}, "ok"

    if facet == "inter_activity_distance_minimize":
        edges = extract_inter_activity_edges(plan)
        raw_value = _average([edge.distance_km for edge in edges if edge.distance_km > 0])
        if raw_value is None:
            return True, 1.0, {"raw_value": 0.0, "fallback": "no_distance_segments"}, "ok"
        return True, _inverse_cost_score(raw_value, scale=5.0), {"raw_value": raw_value, "edge_count": len(edges)}, "ok"

    if facet == "food_cost_ratio_maximize":
        breakdown = extract_cost_breakdown(plan)
        total_cost = breakdown["total"]
        if total_cost <= 0:
            return False, None, {}, "non_positive_total_cost"
        ratio = breakdown["food"] / total_cost
        return True, ratio, {"raw_value": ratio, "food_cost": breakdown["food"], "total_cost": total_cost}, "ok"

    if facet == "hotel_cost_ratio_maximize":
        breakdown = extract_cost_breakdown(plan)
        total_cost = breakdown["total"]
        if total_cost <= 0:
            return False, None, {}, "non_positive_total_cost"
        ratio = breakdown["hotel"] / total_cost
        return True, ratio, {"raw_value": ratio, "hotel_cost": breakdown["hotel"], "total_cost": total_cost}, "ok"

    if facet == "hotel_cost_minimize":
        breakdown = extract_cost_breakdown(plan)
        days = max(len(itinerary), 1)
        people = max(int(plan.get("people_number", 1) or 1), 1)
        denom = people * max(days - 1, 1)
        raw_value = breakdown["hotel"] / denom
        return True, _inverse_cost_score(raw_value, scale=300.0), {"raw_value": raw_value, "hotel_cost": breakdown["hotel"]}, "ok"

    if facet == "attraction_cost_ratio_minimize":
        breakdown = extract_cost_breakdown(plan)
        total_cost = breakdown["total"]
        if total_cost <= 0:
            return False, None, {}, "non_positive_total_cost"
        ratio = breakdown["attraction"] / total_cost
        score = max(0.0, 1.0 - ratio)
        return True, score, {"raw_value": ratio, "attraction_cost": breakdown["attraction"], "total_cost": total_cost}, "ok"

    if facet == "distance_to_poi_minimize":
        return _score_distance_to_anchor(plan, preference)

    if facet == "district_anchor_distance_minimize":
        return _score_anchor_distance(plan, preference, scale=4.0)

    if facet == "hotel_anchor_distance_minimize":
        if preference.anchor:
            return _score_anchor_distance(plan, preference, scale=3.5)
        hotel_distances = extract_hotel_anchor_distances(plan)
        if hotel_distances:
            avg_distance = _average(hotel_distances)
            return True, _inverse_cost_score(avg_distance or 0.0, scale=3.5), {
                "raw_value": avg_distance,
                "anchor_source": "plan_accommodations",
            }, "ok"
        return False, None, {}, "missing_anchor"

    if facet == "day_end_time_minimize":
        end_minutes = extract_day_end_minutes(plan)
        raw_value = _average([float(item) for item in end_minutes])
        if raw_value is None:
            return True, 1.0, {"raw_value": 0.0, "fallback": "no_day_end_times"}, "ok"
        return True, _inverse_cost_score(raw_value, scale=1200.0), {"raw_value": raw_value, "end_minutes": end_minutes}, "ok"

    if facet == "moderate_buffer_ratio_maximize":
        gaps = extract_buffer_gaps(plan)
        if not gaps:
            return True, 0.0, {"raw_value": 0.0, "moderate_count": 0, "gap_count": 0}, "ok"
        moderate_count = sum(1 for gap in gaps if 20 <= gap <= 90)
        ratio = moderate_count / len(gaps)
        return True, ratio, {"raw_value": ratio, "moderate_count": moderate_count, "gap_count": len(gaps)}, "ok"

    if facet == "excessive_gap_minimize":
        gaps = extract_buffer_gaps(plan)
        if not gaps:
            return True, 1.0, {"raw_value": 0.0, "excessive_count": 0, "gap_count": 0}, "ok"
        excessive_count = sum(1 for gap in gaps if gap > 120)
        ratio = excessive_count / len(gaps)
        score = max(0.0, 1.0 - ratio)
        return True, score, {"raw_value": ratio, "excessive_count": excessive_count, "gap_count": len(gaps)}, "ok"

    if facet == "theme_concentration_maximize":
        return _score_theme_concentration(plan, preference)

    if facet == "theme_evenness_maximize":
        return _score_theme_evenness(plan, preference)

    if facet in {"relaxed_vibe_maximize", "relaxed"}:
        return _score_vibe(plan, "relaxed")

    if facet in {"lively_vibe_maximize", "lively"}:
        return _score_vibe(plan, "lively")

    if facet in {"quiet_vibe_maximize", "quiet"}:
        return _score_vibe(plan, "quiet")

    if facet in {"nightlife_vibe_maximize", "nightlife"}:
        return _score_vibe(plan, "nightlife")

    if facet == "attraction_count_maximize":
        return _score_attraction_count(plan, preference)

    if facet == "category_evenness_maximize":
        return _score_category_diversity(plan, preference)

    if facet == "food_experience_richness_maximize":
        return _score_food_experience_count(plan, preference)

    if facet == "route_smoothness_maximize":
        return _score_route_smoothness(plan, preference)

    if facet == "walk_burden_minimize":
        return _score_walk_burden(plan, preference)

    if facet == "experience_cost_ratio_maximize":
        return _score_experience_share(plan, preference)

    return False, None, {}, "unsupported_preference_type"


def score_preference_baseline(
    plan: dict[str, Any],
    preference: PreferenceConstraintObject,
) -> tuple[bool, float | None, dict[str, Any], str]:
    return _score_plan(plan, preference)


def evaluate_preference_target(
    origin_plan: dict[str, Any],
    edited_plan: dict[str, Any],
    preference: PreferenceConstraintObject,
) -> PreferenceScoreResult:
    before_supported, score_before, before_details, before_reason = _score_plan(origin_plan, preference)
    after_supported, score_after, after_details, after_reason = _score_plan(edited_plan, preference)

    canonical_facet = str(preference.params.get("canonical_facet") or preference.facet)
    result_details = {
        "before": before_details,
        "after": after_details,
        "canonical_facet": canonical_facet,
        "scorer_facet": preference.facet,
    }

    if not after_supported:
        return PreferenceScoreResult(
            preference_id=preference.id,
            family=preference.family,
            facet=preference.facet,
            supported=False,
            passed=False,
            score_before=score_before,
            score_after=None,
            delta=None,
            reason=after_reason,
            details=result_details,
        )

    delta = None if score_before is None else score_after - score_before
    passed = delta is not None and delta >= 0.0
    reason = "ok" if passed else "score_degraded"

    return PreferenceScoreResult(
        preference_id=preference.id,
        family=preference.family,
        facet=preference.facet,
        supported=after_supported,
        passed=passed,
        score_before=score_before if before_supported else None,
        score_after=score_after,
        delta=delta,
        reason=reason if after_supported else after_reason,
        details=result_details,
    )
