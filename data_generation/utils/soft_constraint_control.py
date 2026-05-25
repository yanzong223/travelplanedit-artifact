"""
Soft-constraint control utilities for data generation.

This module reuses the soft taxonomy heuristics from data_classify so Step 1/2
can sample, profile, and validate soft-control targets without changing the
main bucket -> template -> constraint_type pipeline.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
UTILS_DIR = ROOT_DIR / "data_generation" / "utils"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    from data_generation.utils.query_surface_parsing import (  # noqa: E402
        extract_extend_with_pacing_spec,
        is_explicit_hard_travel_time_query,
    )
except ModuleNotFoundError:
    from query_surface_parsing import extract_extend_with_pacing_spec, is_explicit_hard_travel_time_query  # noqa: E402
from data_classify.analyze_soft_constraints import (  # noqa: E402
    ConstraintMention,
    add_constraint,
    classify_anchor_facet,
    dedupe_constraints,
    detect_explicit_constraints,
    detect_intensity,
    detect_textual_constraints,
    merge_constraints,
)

SOFT_SCHEMA_VERSION = "v2"
INTENSITY_RANK = {"weak": 0, "medium": 1, "strong": 2}


def _choose_adaptive_duration_limit_minutes(baseline_minutes: Any) -> Optional[int]:
    baseline_match = re.search(r"(\d+(?:\.\d+)?)", str(baseline_minutes or ""))
    if baseline_match is None:
        return None
    baseline = float(baseline_match.group(1))
    if baseline <= 10:
        return None
    target = min(baseline * 0.8, baseline - max(5.0, baseline * 0.12))
    target = max(10.0, target)
    chosen = int((target + 4) // 5 * 5)
    if chosen >= baseline:
        chosen -= 5
    return chosen if chosen >= 10 else None


def _bucket_name(bucket: Any) -> str:
    if isinstance(bucket, (list, tuple)) and len(bucket) == 3:
        return "_".join(str(part).strip() for part in bucket)
    return str(bucket or "").strip()


def load_soft_control_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"soft-control config must be a mapping: {config_path}")
    return config


def build_origin_soft_profile(
    origin_query_structured: Dict[str, Any],
    origin_query_text: str = "",
) -> List[Dict[str, Any]]:
    query = dict(origin_query_structured) if isinstance(origin_query_structured, dict) else {}
    if not query.get("nature_language") and isinstance(origin_query_text, str) and origin_query_text.strip():
        query["nature_language"] = origin_query_text.strip()
    zh_text = str(query.get("nature_language", "") or "")
    en_text = str(query.get("nature_language_en", "") or "")

    explicit_constraints, _, _ = detect_explicit_constraints(query)
    zh_constraints = detect_soft_constraints_from_text(zh_text, source="origin_query_text")
    en_constraints = detect_soft_constraints_from_text(en_text, source="origin_query_en")
    merged = merge_constraints(explicit_constraints, zh_constraints, en_constraints)
    return [item.to_dict() for item in merged]


def detect_soft_constraints_from_text(
    text: str,
    *,
    source: str = "generated_text",
) -> List[ConstraintMention]:
    base = detect_textual_constraints(text or "", source=source)
    supplemental: List[ConstraintMention] = []
    plain_text = str(text or "")

    radius_patterns = [
        r"把([^，。；]{2,24}?)周边活动半径",
        r"以([^，。；]{2,24}?)为中心的周边活动半径",
        r"以([^，。；]{2,24}?)为中心安排(?:行程|活动)",
        r"(?:希望)?([^，。；]{2,24}?)周边的活动安排",
        r"围绕([^，。；]{2,24}?)安排活动",
        r"围绕([^，。；]{2,24}?)安排行程",
        r"围着([^，。；]{2,24}?)玩",
        r"以([^，。；]{2,24}?)为主",
    ]
    for pattern in radius_patterns:
        for match in re.finditer(pattern, plain_text):
            anchor = match.group(1).strip("“”'\" ")
            if not anchor:
                continue
            add_constraint(
                supplemental,
                family="anchor_proximity",
                facet=classify_anchor_facet(anchor),
                polarity="minimize",
                anchor=anchor,
                evidence_span=match.group(0),
                source=source,
                is_explicit=False,
                intensity_hint=detect_intensity(match.group(0)),
            )

    if (
        (
            "交通时间" in plain_text
            or "景点之间交通" in plain_text
            or "景点之间的交通" in plain_text
            or "路上" in plain_text
            or "通勤" in plain_text
        )
        and any(token in plain_text for token in ["不超过", "控制在", "少一点", "尽可能少", "别超过", "别太久", "别太远"])
    ) or any(
        token in plain_text
        for token in [
            "最省时间",
            "节省时间",
            "省时间",
            "少花时间在路上",
            "路上别花太多时间",
            "通勤快一点",
            "尽量快一点",
        ]
    ):
        add_constraint(
            supplemental,
            family="route_compactness",
            facet="travel_time",
            polarity="minimize",
            anchor=None,
            evidence_span=plain_text,
            source=source,
            is_explicit=False,
        )

    pacing_spec = extract_extend_with_pacing_spec(plain_text)
    if pacing_spec.get("daily_poi_cap") is not None or any(
        token in plain_text
        for token in [
            "宽松一点",
            "别太满",
            "不要排太满",
            "不用赶",
            "轻松点",
            "轻松一些",
            "悠闲",
            "休闲一点",
            "休闲一些",
            "主打一个悠闲",
            "慢一点",
            "松一点",
            "不要太赶",
        ]
    ):
        add_constraint(
            supplemental,
            family="burden_reduction",
            facet="pace",
            polarity="minimize",
            anchor=None,
            evidence_span=plain_text,
            source=source,
            is_explicit=False,
        )

    if any(
        token in plain_text
        for token in [
            "人比较少",
            "人少一点",
            "人少的地方",
            "避开人群",
            "清净一点",
            "安静一点",
        ]
    ):
        add_constraint(
            supplemental,
            family="burden_reduction",
            facet="activity_load",
            polarity="minimize",
            anchor=None,
            evidence_span=plain_text,
            source=source,
            is_explicit=False,
        )

    return dedupe_constraints(list(base) + supplemental)


def summarize_soft_profile(profile: Sequence[Dict[str, Any]], limit: int = 4) -> str:
    if not profile:
        return "(none)"
    parts = []
    for item in profile[:limit]:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family", "")).strip()
        facet = str(item.get("facet", "")).strip()
        intensity = str(item.get("intensity", "")).strip()
        if family and facet:
            parts.append(f"{family}:{facet}({intensity or 'weak'})")
    return ", ".join(parts) if parts else "(none)"


def find_profile_entry(
    profile: Sequence[Dict[str, Any]],
    family: str,
    facet: str,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_rank = -1
    for item in profile:
        if not isinstance(item, dict):
            continue
        if item.get("family") != family or item.get("facet") != facet:
            continue
        rank = INTENSITY_RANK.get(str(item.get("intensity", "weak")).strip(), 0)
        if rank > best_rank:
            best = item
            best_rank = rank
    return best


def edit_mode_is_compatible(
    profile: Sequence[Dict[str, Any]],
    family: str,
    facet: str,
    edit_mode: str,
) -> bool:
    has_origin = find_profile_entry(profile, family, facet) is not None
    if edit_mode == "addition":
        return not has_origin
    if edit_mode == "strengthen":
        return has_origin
    return False


def _iter_soft_candidates(config: Dict[str, Any], bucket: Any) -> List[Dict[str, Any]]:
    bucket_name = _bucket_name(bucket)
    families = config.get("families", {})
    if not isinstance(families, dict):
        return []

    candidates: List[Dict[str, Any]] = []
    for family_name, family_cfg in families.items():
        if not isinstance(family_cfg, dict):
            continue
        family_weight = float(family_cfg.get("weight", 1.0))
        family_buckets = family_cfg.get("allowed_buckets")
        facets = family_cfg.get("facets", {})
        if not isinstance(facets, dict):
            continue
        for facet_name, facet_cfg in facets.items():
            if not isinstance(facet_cfg, dict):
                continue
            facet_weight = float(facet_cfg.get("weight", 1.0))
            allowed_buckets = facet_cfg.get("allowed_buckets", family_buckets)
            if isinstance(allowed_buckets, list) and bucket_name not in allowed_buckets:
                continue
            edit_modes = facet_cfg.get("edit_modes", {})
            if not isinstance(edit_modes, dict):
                continue
            for edit_mode, mode_cfg in edit_modes.items():
                if not isinstance(mode_cfg, dict):
                    continue
                enabled = bool(mode_cfg.get("enabled", True))
                if not enabled:
                    continue
                weight = family_weight * facet_weight * float(mode_cfg.get("weight", 1.0))
                candidates.append(
                    {
                        "family": str(family_name).strip(),
                        "facet": str(facet_name).strip(),
                        "edit_mode": str(edit_mode).strip(),
                        "weight": max(weight, 0.0),
                        "allowed_buckets": list(allowed_buckets) if isinstance(allowed_buckets, list) else [],
                        "intensity": str(mode_cfg.get("intensity", "")).strip(),
                        "anchor_type": str(mode_cfg.get("anchor_type", "")).strip(),
                    }
                )
    return [item for item in candidates if item["weight"] > 0]


def choose_soft_control_target(
    profile: Sequence[Dict[str, Any]],
    bucket: Any,
    config: Dict[str, Any],
    rng: random.Random,
    *,
    preferred_family: str = "",
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    compatible = [
        candidate
        for candidate in _iter_soft_candidates(config, bucket)
        if (
            not preferred_family
            or candidate["family"] == preferred_family
        )
        if not (
            candidate["family"] == "route_compactness"
            and candidate["facet"] == "travel_time"
            and _choose_adaptive_duration_limit_minutes(
                (evidence or {}).get("max_transport_duration_min")
            ) is None
        )
        if edit_mode_is_compatible(
            profile,
            candidate["family"],
            candidate["facet"],
            candidate["edit_mode"],
        )
    ]
    if not compatible:
        return None

    weights = [item["weight"] for item in compatible]
    chosen = rng.choices(compatible, weights=weights, k=1)[0]
    return {
        "family": chosen["family"],
        "facet": chosen["facet"],
        "edit_mode": chosen["edit_mode"],
        "allowed_buckets": chosen["allowed_buckets"],
        "intensity": chosen["intensity"],
        "anchor_type": chosen["anchor_type"],
        "schema_version": SOFT_SCHEMA_VERSION,
    }


def template_matches_soft_control(template: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    family = str(meta.get("soft_constraint_family", "")).strip()
    facet = str(meta.get("soft_constraint_facet", "")).strip()
    edit_mode = str(meta.get("soft_edit_mode", "")).strip()
    if not family or not facet or not edit_mode:
        return True

    template_family = str(template.get("soft_family", "")).strip()
    template_facet = str(template.get("soft_facet", "")).strip()
    supported_modes = template.get("supported_edit_modes", [])
    if not isinstance(supported_modes, list):
        supported_modes = []

    return (
        template_family == family
        and template_facet == facet
        and edit_mode in supported_modes
    )


def validate_soft_query_alignment(
    *,
    edit_query: str,
    origin_profile: Sequence[Dict[str, Any]],
    target_family: str,
    target_facet: str,
    target_mode: str,
    template_soft_intensity: str = "",
    constraint_type: str = "",
    semantic_slots: Optional[Dict[str, Any]] = None,
    query_spec: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not target_family or not target_facet or not target_mode:
        return []
    if (
        str(constraint_type or "").strip() == "travel_time_constraint"
        and target_family == "route_compactness"
        and target_facet == "travel_time"
        and is_explicit_hard_travel_time_query(
            edit_query,
            semantic_slots=semantic_slots,
            query_spec=query_spec,
        )
    ):
        return []

    errors: List[str] = []
    query_profile = [item.to_dict() for item in detect_soft_constraints_from_text(edit_query, source="edit_query")]
    target_query_entry = find_profile_entry(query_profile, target_family, target_facet)
    target_origin_entry = find_profile_entry(origin_profile, target_family, target_facet)
    if (
        target_query_entry is None
        and target_family == "burden_reduction"
        and target_facet == "pace"
    ):
        pacing_spec = extract_extend_with_pacing_spec(edit_query)
        if pacing_spec.get("day_count") is not None or pacing_spec.get("daily_poi_cap") is not None:
            target_query_entry = {
                "family": target_family,
                "facet": target_facet,
                "intensity": detect_intensity(edit_query or ""),
            }

    if target_query_entry is None:
        return [f"soft target missing in query: {target_family}:{target_facet}"]

    if target_mode == "addition":
        if target_origin_entry is not None:
            errors.append(f"addition target already present in origin: {target_family}:{target_facet}")
        return errors

    if target_mode == "strengthen":
        if target_origin_entry is None:
            errors.append(f"strengthen target absent in origin: {target_family}:{target_facet}")
            return errors
        origin_rank = INTENSITY_RANK.get(str(target_origin_entry.get("intensity", "weak")), 0)
        query_rank = INTENSITY_RANK.get(str(target_query_entry.get("intensity", "weak")), 0)
        template_rank = INTENSITY_RANK.get(str(template_soft_intensity or "").strip(), query_rank)
        if max(query_rank, template_rank) <= origin_rank:
            errors.append(
                f"strengthen target not stronger than origin: {target_family}:{target_facet} "
                f"(origin={target_origin_entry.get('intensity', 'weak')}, query={target_query_entry.get('intensity', 'weak')})"
            )
    return errors


def build_soft_control_statistics(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    family_distribution: Dict[str, int] = {}
    facet_distribution: Dict[str, int] = {}
    edit_mode_distribution: Dict[str, int] = {}
    status_distribution: Dict[str, int] = {}
    schema_version_distribution: Dict[str, int] = {}
    anchor_type_distribution: Dict[str, int] = {}
    intent_group_distribution: Dict[str, int] = {}

    for sample in samples:
        meta = sample.get("meta", {}) if isinstance(sample, dict) else {}
        status = str(meta.get("soft_control_status", "unknown")).strip() or "unknown"
        status_distribution[status] = status_distribution.get(status, 0) + 1

        family = str(meta.get("soft_constraint_family", "")).strip()
        facet = str(meta.get("soft_constraint_facet", "")).strip()
        edit_mode = str(meta.get("soft_edit_mode", "")).strip()
        if family:
            family_distribution[family] = family_distribution.get(family, 0) + 1
        if family and facet:
            key = f"{family}:{facet}"
            facet_distribution[key] = facet_distribution.get(key, 0) + 1
        if edit_mode:
            edit_mode_distribution[edit_mode] = edit_mode_distribution.get(edit_mode, 0) + 1
        schema_version = str(meta.get("soft_schema_version", "")).strip()
        if schema_version:
            schema_version_distribution[schema_version] = schema_version_distribution.get(schema_version, 0) + 1
        anchor_type = str(meta.get("soft_anchor_type", "")).strip()
        if anchor_type:
            anchor_type_distribution[anchor_type] = anchor_type_distribution.get(anchor_type, 0) + 1
        intent_group = str(meta.get("intent_group", "")).strip()
        if intent_group:
            intent_group_distribution[intent_group] = intent_group_distribution.get(intent_group, 0) + 1

    return {
        "soft_family_distribution": family_distribution,
        "soft_facet_distribution": facet_distribution,
        "soft_edit_mode_distribution": edit_mode_distribution,
        "soft_control_status_distribution": status_distribution,
        "soft_schema_version_distribution": schema_version_distribution,
        "soft_anchor_type_distribution": anchor_type_distribution,
        "soft_intent_group_distribution": intent_group_distribution,
    }
