"""Prompt-ablation helpers shared by standalone edit baselines."""

from __future__ import annotations

import json
from typing import Any, Dict


CONTEXT_METADATA_KEYS = (
    "origin_query_structured",
    "origin_logical_constraints",
    "origin_preference_constraints",
    "edit_target_constraints",
    "effective_logical_constraints",
    "effective_constraint_merge_trace",
    "edit_target_preferences",
    "origin_preference_tags",
    "edit_target_preference_tags",
    "batch_id",
    "sample_id",
)

DATABASE_METADATA_KEYS = (
    "retrieval_reference_result",
    "retrieval_hint_context",
    "database_prompt_context",
    "database_reference",
    "reference_information",
    "retrieved_database_context",
    "database_rows",
)


def build_prompt_ablation_sections(
    *,
    origin_plan: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    context_prompt: bool = False,
    database_prompt: bool = False,
    annotation_scaffold_level: str = "none",
) -> str:
    """Build optional prompt sections for context/database/annotation-scaffold ablations."""

    sections: list[str] = []
    metadata = metadata or {}
    if context_prompt:
        sections.append(_build_context_section(origin_plan=origin_plan, metadata=metadata))
    if database_prompt:
        sections.append(_build_database_section(metadata=metadata))
    if annotation_scaffold_level and annotation_scaffold_level != "none":
        from edit_framework.annotation_scaffold import build_annotation_scaffold_section

        scaffold = build_annotation_scaffold_section(
            metadata=metadata,
            level=annotation_scaffold_level,
        )
        if scaffold:
            sections.append(scaffold)
    return "\n\n".join(section for section in sections if section)


def _build_context_section(*, origin_plan: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    payload: Dict[str, Any] = {"plan_summary": _summarize_plan(origin_plan)}
    for key in CONTEXT_METADATA_KEYS:
        if key in metadata:
            payload[key] = metadata[key]
    return "# 结构化上下文提示\n\n" + _dump_json(payload)


def _build_database_section(*, metadata: Dict[str, Any]) -> str:
    payload = {key: metadata[key] for key in DATABASE_METADATA_KEYS if key in metadata}
    if not payload:
        payload = {
            "note": (
                "本样本未提供预检索数据库参考信息；如需真实 POI、餐厅、住宿或交通信息，"
                "仍必须通过可用工具查询。"
            )
        }
    return "# 数据库参考提示\n\n" + _dump_json(payload)


def _summarize_plan(origin_plan: Dict[str, Any]) -> Dict[str, Any]:
    itinerary = origin_plan.get("itinerary", [])
    day_summaries = []
    if isinstance(itinerary, list):
        for day_payload in itinerary:
            if not isinstance(day_payload, dict):
                continue
            activities = day_payload.get("activities", [])
            activity_summaries = []
            if isinstance(activities, list):
                for activity in activities:
                    if not isinstance(activity, dict):
                        continue
                    activity_summaries.append(
                        {
                            "id": activity.get("id"),
                            "type": activity.get("type"),
                            "name": activity.get("position")
                            or activity.get("start")
                            or activity.get("end"),
                            "start_time": activity.get("start_time"),
                            "end_time": activity.get("end_time"),
                        }
                    )
            day_summaries.append(
                {
                    "day": day_payload.get("day"),
                    "activity_count": len(activities) if isinstance(activities, list) else 0,
                    "activities": activity_summaries,
                }
            )
    return {
        "people_number": origin_plan.get("people_number"),
        "start_city": origin_plan.get("start_city"),
        "target_city": origin_plan.get("target_city"),
        "day_count": len(itinerary) if isinstance(itinerary, list) else 0,
        "days": day_summaries,
    }


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
