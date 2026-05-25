"""
Helpers for selecting a replacement origin plan for Step 2 retries.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _load_origin_plans(input_dir: str) -> List[Dict[str, Any]]:
    origin_plans: List[Dict[str, Any]] = []
    for plan_file in sorted(Path(input_dir).glob("*.json")):
        try:
            with open(plan_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_source_file"] = plan_file.name
        origin_plans.append(payload)
    return origin_plans


def _normalize_origin_plan(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "people_number": plan_data.get("people_number", 1),
        "start_city": plan_data.get("start_city", ""),
        "target_city": plan_data.get("target_city", ""),
        "itinerary": plan_data.get("itinerary", []),
    }


def _plan_meta(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    itinerary = plan_data.get("itinerary", [])
    return {
        "origin_plan": _normalize_origin_plan(plan_data),
        "origin_plan_file": plan_data.get("_source_file", ""),
        "target_city": plan_data.get("target_city", ""),
        "start_city": plan_data.get("start_city", ""),
        "day_count": len(itinerary),
        "is_single_day": len(itinerary) == 1,
    }


def select_replacement_origin_plan(
    input_dir: str,
    target_bucket: Tuple[str, str, str],
    target_city: str,
    day_count: int,
    is_single_day: bool,
    used_origin_plan_files: Set[str],
    preferred_start_city: Optional[str],
    is_bucket_applicable_fn,
    templates_index: Set[Tuple[str, str, str]],
    template_config: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []

    for plan_data in _load_origin_plans(input_dir):
        meta = _plan_meta(plan_data)
        source_file = meta["origin_plan_file"]
        if source_file in used_origin_plan_files:
            continue
        if meta["target_city"] != target_city:
            continue
        if meta["day_count"] != day_count:
            continue
        if meta["is_single_day"] != is_single_day:
            continue
        if not is_bucket_applicable_fn(target_bucket, meta, templates_index, template_config):
            continue
        if preferred_start_city and meta["start_city"] == preferred_start_city:
            candidates.append(meta)
        else:
            fallback_candidates.append(meta)

    pool = candidates or fallback_candidates
    if not pool:
        return None
    return random.choice(pool)
