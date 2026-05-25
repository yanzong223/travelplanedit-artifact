"""Soft-tag registry and dataset builders."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    PreferenceConstraintObject,
    PreferenceScoreSnapshot,
    PreferenceTag,
    preference_snapshots_to_dict,
    preference_tags_to_dict,
    preferences_to_dict,
)
from .preference_scorers import score_preference_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.utils.soft_constraint_control import (  # noqa: E402
    build_origin_soft_profile,
    detect_soft_constraints_from_text,
)


SOFT_FACET_REGISTRY: Dict[str, Tuple[str, ...]] = {
    "anchor_proximity": ("poi_anchor", "district_anchor", "hotel_anchor"),
    "route_compactness": ("travel_time", "route_smoothness", "idle_gap"),
    "burden_reduction": ("pace", "activity_load", "walk"),
    "cost_allocation_preference": ("food_share", "accommodation_cost", "experience_share"),
    "theme_alignment": ("culture", "nature", "food", "photo", "family", "shopping", "mixed_theme"),
    "vibe_alignment": ("relaxed", "lively", "quiet", "nightlife"),
    "experience_richness": ("attraction_count", "category_diversity", "food_experience_count"),
}

TAG_TO_SCORER: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("anchor_proximity", "poi_anchor"): ("distance_to_poi_minimize", "minimize"),
    ("anchor_proximity", "district_anchor"): ("district_anchor_distance_minimize", "minimize"),
    ("anchor_proximity", "hotel_anchor"): ("hotel_anchor_distance_minimize", "minimize"),
    ("route_compactness", "travel_time"): ("transport_time_minimize", "minimize"),
    ("route_compactness", "route_smoothness"): ("route_smoothness_maximize", "maximize"),
    ("route_compactness", "idle_gap"): ("excessive_gap_minimize", "minimize"),
    ("burden_reduction", "pace"): ("daily_activity_load_minimize", "minimize"),
    ("burden_reduction", "activity_load"): ("daily_activity_load_minimize", "minimize"),
    ("burden_reduction", "walk"): ("walk_burden_minimize", "minimize"),
    ("cost_allocation_preference", "food_share"): ("food_cost_ratio_maximize", "maximize"),
    ("cost_allocation_preference", "accommodation_cost"): ("hotel_cost_minimize", "minimize"),
    ("cost_allocation_preference", "experience_share"): ("experience_cost_ratio_maximize", "maximize"),
    ("theme_alignment", "culture"): ("theme_concentration_maximize", "maximize"),
    ("theme_alignment", "nature"): ("theme_concentration_maximize", "maximize"),
    ("theme_alignment", "food"): ("theme_concentration_maximize", "maximize"),
    ("theme_alignment", "photo"): ("theme_concentration_maximize", "maximize"),
    ("theme_alignment", "family"): ("theme_concentration_maximize", "maximize"),
    ("theme_alignment", "shopping"): ("theme_concentration_maximize", "maximize"),
    ("theme_alignment", "mixed_theme"): ("theme_evenness_maximize", "maximize"),
    ("vibe_alignment", "relaxed"): ("relaxed_vibe_maximize", "maximize"),
    ("vibe_alignment", "lively"): ("lively_vibe_maximize", "maximize"),
    ("vibe_alignment", "quiet"): ("quiet_vibe_maximize", "maximize"),
    ("vibe_alignment", "nightlife"): ("nightlife_vibe_maximize", "maximize"),
    ("experience_richness", "attraction_count"): ("attraction_count_maximize", "maximize"),
    ("experience_richness", "category_diversity"): ("category_evenness_maximize", "maximize"),
    ("experience_richness", "food_experience_count"): ("food_experience_richness_maximize", "maximize"),
}

SCORER_ID = "benchmark.preference_scorers"
SCORER_VERSION = "v2"

_SOFT_FAMILY_ALIASES: Dict[str, str] = {
    "theme_alignment": "theme_alignment",
    "theme_alignment": "theme_alignment",
}


def normalize_family_facet(family: Any, facet: Any) -> Tuple[str, str]:
    family_text = str(family or "").strip()
    facet_text = str(facet or "").strip()
    if not family_text or not facet_text:
        return ("", "")
    family_text = _SOFT_FAMILY_ALIASES.get(family_text, family_text)
    if family_text in SOFT_FACET_REGISTRY and facet_text in SOFT_FACET_REGISTRY[family_text]:
        return (family_text, facet_text)
    return (family_text, facet_text)


def _tag_sort_key(tag: PreferenceTag) -> Tuple[str, str, str, str]:
    return (
        tag.family,
        tag.facet,
        tag.anchor or "",
        tag.edit_mode or "",
    )


def _normalize_edit_mode(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text not in {"addition", "strengthen"}:
        raise ValueError(f"unsupported edit_mode: {value}")
    return text


def _dedupe_tags(tags: Iterable[PreferenceTag]) -> List[PreferenceTag]:
    deduped: Dict[Tuple[str, str, str, str], PreferenceTag] = {}
    for tag in tags:
        key = (tag.family, tag.facet, tag.anchor or "", tag.edit_mode or "")
        existing = deduped.get(key)
        if existing is None or (not existing.intensity and tag.intensity):
            deduped[key] = tag
    return sorted(deduped.values(), key=_tag_sort_key)


def _validate_tag(tag: PreferenceTag, *, require_edit_mode: bool) -> None:
    family, facet = normalize_family_facet(tag.family, tag.facet)
    tag.family = family
    tag.facet = facet
    allowed_facets = SOFT_FACET_REGISTRY.get(tag.family)
    if allowed_facets is None:
        raise ValueError(f"unsupported soft family: {tag.family}")
    if tag.facet not in allowed_facets:
        raise ValueError(f"unsupported facet for {tag.family}: {tag.facet}")
    if require_edit_mode and tag.edit_mode not in {"addition", "strengthen"}:
        raise ValueError(f"edit target tag missing edit_mode for {tag.family}:{tag.facet}")
    if tag.family == "anchor_proximity" and tag.facet in {"poi_anchor", "district_anchor"} and not tag.anchor:
        raise ValueError(f"anchor_proximity tag missing anchor for facet {tag.facet}")


def validate_preference_tags(
    tags: Iterable[PreferenceTag | Dict[str, Any]],
    *,
    require_edit_mode: bool = False,
) -> List[PreferenceTag]:
    validated: List[PreferenceTag] = []
    for raw in tags:
        if isinstance(raw, PreferenceTag):
            tag = raw
        elif isinstance(raw, dict):
            family, facet = normalize_family_facet(raw.get("family", ""), raw.get("facet", ""))
            tag = PreferenceTag(
                family=family,
                facet=facet,
                edit_mode=_normalize_edit_mode(raw.get("edit_mode")),
                anchor=raw.get("anchor"),
                intensity=str(raw.get("intensity", "")).strip() or None,
                provenance=dict(raw.get("provenance", {})),
            )
        else:
            raise ValueError(f"invalid preference tag payload: {raw!r}")
        _validate_tag(tag, require_edit_mode=require_edit_mode)
        validated.append(tag)
    return _dedupe_tags(validated)


def _profile_item_to_tag(item: Dict[str, Any]) -> PreferenceTag:
    family, facet = normalize_family_facet(item.get("family", ""), item.get("facet", ""))
    return PreferenceTag(
        family=family,
        facet=facet,
        edit_mode=None,
        anchor=item.get("anchor"),
        intensity=str(item.get("intensity", "")).strip() or None,
        provenance={
            "source": item.get("source", ""),
            "is_explicit": bool(item.get("is_explicit", False)),
            "evidence_span": item.get("evidence_span", ""),
        },
    )


def build_origin_preference_tags(
    origin_query_structured: Dict[str, Any] | None,
    origin_query_text: str = "",
) -> List[PreferenceTag]:
    profile = build_origin_soft_profile(origin_query_structured or {}, origin_query_text=origin_query_text)
    return validate_preference_tags((_profile_item_to_tag(item) for item in profile), require_edit_mode=False)


def _extract_matching_anchor(edit_query: str, family: str, facet: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query.strip():
        return None
    normalized_family, normalized_facet = normalize_family_facet(family, facet)
    detected = detect_soft_constraints_from_text(edit_query, source="edit_query")
    for item in detected:
        item_family, item_facet = normalize_family_facet(item.family, item.facet)
        if item_family == normalized_family and item_facet == normalized_facet and item.anchor:
            return item.anchor
    if normalized_family == "anchor_proximity":
        for pattern in (
            r"把([^，。；;]{2,32}?)附近的活动",
            r"(?:希望)?([^，。；;]{2,32}?)周边的活动安排",
            r"在([^，。；;]{2,32}?)附近",
            r"围绕([^，。；;]{2,32}?)",
            r"围绕([^，。；;]{2,32}?)安排行程",
        ):
            match = re.search(pattern, edit_query)
            if match:
                anchor = _normalize_anchor_text(str(match.group(1)))
                if anchor:
                    return anchor
    return None


def _normalize_anchor_text(raw_anchor: str) -> str:
    anchor = str(raw_anchor).strip("“”'\" ")
    anchor = re.sub(r"^(把|将|在)", "", anchor).strip("“”'\" ")
    return anchor


def build_edit_target_preference_tags(
    sample: Dict[str, Any] | None,
    edit_query: str,
    query_generation_trace: Dict[str, Any] | None,
) -> List[PreferenceTag]:
    sample = sample or {}
    query_generation_trace = query_generation_trace or {}
    if query_generation_trace.get("edit_soft_target_suppressed") is True:
        return []
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}

    raw_family = query_generation_trace.get("soft_target_family") or meta.get("soft_constraint_family") or ""
    raw_facet = query_generation_trace.get("soft_target_facet") or meta.get("soft_constraint_facet") or ""
    family, facet = normalize_family_facet(raw_family, raw_facet)
    edit_mode = _normalize_edit_mode(
        query_generation_trace.get("soft_target_mode") or meta.get("soft_edit_mode")
    )
    intensity = (
        str(query_generation_trace.get("soft_target_intensity") or "").strip()
        or str(meta.get("soft_intensity") or "").strip()
        or None
    )

    if not family and not facet and edit_mode is None:
        return []
    if not family or not facet or edit_mode is None:
        raise ValueError("soft target metadata is incomplete; family/facet/edit_mode must all be present")

    anchor_hint = ""
    if isinstance(query_generation_trace.get("anchor_poi"), str):
        anchor_hint = _normalize_anchor_text(str(query_generation_trace.get("anchor_poi")))
    else:
        query_spec = query_generation_trace.get("surface_query_spec")
        if not isinstance(query_spec, dict):
            query_spec = query_generation_trace.get("canonical_query_spec")
        if not isinstance(query_spec, dict):
            query_spec = query_generation_trace.get("query_spec")
        raw_anchor = query_spec.get("anchor_poi") if isinstance(query_spec, dict) else None
        if isinstance(raw_anchor, str):
            anchor_hint = _normalize_anchor_text(raw_anchor)
    anchor = anchor_hint or _extract_matching_anchor(edit_query, family, facet)
    tag = PreferenceTag(
        family=family,
        facet=facet,
        edit_mode=edit_mode,
        anchor=anchor,
        intensity=intensity,
        provenance={"source": "template_meta_query_trace"},
    )
    return validate_preference_tags([tag], require_edit_mode=True)


def _tag_to_preference_constraint(
    tag: PreferenceTag,
    *,
    source: str,
) -> Optional[PreferenceConstraintObject]:
    mapping = TAG_TO_SCORER.get((tag.family, tag.facet))
    if mapping is None:
        return None
    scorer_facet, direction = mapping
    params: Dict[str, Any] = {"canonical_facet": tag.facet}
    if tag.family == "theme_alignment" and tag.facet != "mixed_theme":
        params["target_theme"] = tag.facet
    if tag.family == "vibe_alignment":
        params["target_vibe"] = tag.facet
    if tag.intensity:
        params["intensity"] = tag.intensity
    return PreferenceConstraintObject(
        id=f"{source}_pref_{tag.family}_{tag.facet}",
        source=source,
        family=tag.family,
        facet=scorer_facet,
        direction=direction,
        anchor=tag.anchor,
        edit_mode=tag.edit_mode or "addition",
        params=params,
        provenance=dict(tag.provenance),
    )


def build_origin_preference_constraints(tags: Iterable[PreferenceTag]) -> List[PreferenceConstraintObject]:
    constraints = [
        item
        for tag in validate_preference_tags(tags, require_edit_mode=False)
        for item in [_tag_to_preference_constraint(tag, source="origin")]
        if item is not None
    ]
    return constraints


def build_edit_target_preferences(tags: Iterable[PreferenceTag]) -> List[PreferenceConstraintObject]:
    constraints = [
        item
        for tag in validate_preference_tags(tags, require_edit_mode=True)
        for item in [_tag_to_preference_constraint(tag, source="edit")]
        if item is not None
    ]
    return constraints


def compute_origin_preference_score_snapshot(
    origin_plan: Dict[str, Any] | None,
    origin_tags: Iterable[PreferenceTag],
) -> List[PreferenceScoreSnapshot]:
    snapshots: List[PreferenceScoreSnapshot] = []
    plan = origin_plan or {}
    for tag in validate_preference_tags(origin_tags, require_edit_mode=False):
        pref = _tag_to_preference_constraint(tag, source="origin")
        if pref is None:
            snapshots.append(
                PreferenceScoreSnapshot(
                    family=tag.family,
                    facet=tag.facet,
                    supported=False,
                    score=None,
                    details={"anchor": tag.anchor, "reason": "unsupported_facet"},
                    scorer_id=SCORER_ID,
                    scorer_version=SCORER_VERSION,
                )
            )
            continue
        supported, score, details, reason = score_preference_baseline(plan, pref)
        snapshot_details = dict(details)
        if tag.anchor:
            snapshot_details.setdefault("anchor", tag.anchor)
        if reason != "ok":
            snapshot_details.setdefault("reason", reason)
        snapshots.append(
            PreferenceScoreSnapshot(
                family=tag.family,
                facet=tag.facet,
                supported=supported,
                score=score if supported else None,
                details=snapshot_details,
                scorer_id=SCORER_ID,
                scorer_version=SCORER_VERSION,
            )
        )
    return snapshots


def serialize_preference_tags(tags: Iterable[PreferenceTag]) -> List[Dict[str, Any]]:
    return preference_tags_to_dict(validate_preference_tags(tags))


def serialize_preference_constraints(
    preferences: Iterable[PreferenceConstraintObject],
) -> List[Dict[str, Any]]:
    return preferences_to_dict(list(preferences))


def serialize_preference_snapshots(
    snapshots: Iterable[PreferenceScoreSnapshot],
) -> List[Dict[str, Any]]:
    return preference_snapshots_to_dict(list(snapshots))
