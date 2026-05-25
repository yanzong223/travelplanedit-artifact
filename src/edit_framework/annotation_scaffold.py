"""Annotation-scaffold diagnostic helpers.

Builds staged oracle-annotation prompt sections from benchmark ground-truth
metadata to diagnose where editing failures originate:

  target_grounding  -- can the model identify what to change?
  conflict_localization -- can the model locate where conflicts arise?
  preservation      -- can the model preserve inherited constraints?
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

VALID_LEVELS = frozenset({"target_grounding", "conflict_localization", "preservation"})

# Constraint types that target specific plan activities.
_ACTIVITY_TARGETING_TYPES = frozenset(
    {
        "required_attraction_name",
        "forbidden_attraction_name",
        "semantic_type_requirement",
        "required_restaurant_name",
        "required_hotel_name",
        "required_hotel_feature",
        "required_attraction_type",
        "required_restaurant_type",
        "required_room_type",
        "required_room_count",
        "poi_time_window",
        "poi_day_binding",
        "ticket_price_cap_or_substitute",
        "anchor_bundle_budget_limit",
        "nearby_meal_requirement",
        "activity_budget_limit",
        "activity_duration_limit",
    }
)

# Constraint types that are global / plan-level (not targeting specific activities).
_GLOBAL_TYPES = frozenset(
    {
        "day_count",
        "people_count",
        "budget_total",
        "ticket_count_match",
        "taxi_car_count_match",
        "ticket_budget_total",
        "day_end_time_limit",
        "daily_poi_cap",
    }
)

# Constraint-type → human-readable label (Chinese).
_TYPE_LABELS: Dict[str, str] = {
    "required_attraction_name": "必须包含景点",
    "forbidden_attraction_name": "禁止出现景点",
    "semantic_type_requirement": "活动类型要求",
    "required_restaurant_name": "必须包含餐厅",
    "required_hotel_name": "必须包含酒店",
    "required_hotel_feature": "酒店设施要求",
    "required_attraction_type": "景点类型要求",
    "required_restaurant_type": "餐厅类型要求",
    "required_room_type": "房型要求",
    "required_room_count": "房间数要求",
    "poi_time_window": "POI时间窗口",
    "poi_day_binding": "POI日期绑定",
    "day_count": "天数限制",
    "people_count": "人数限制",
    "budget_total": "总预算限制",
    "ticket_count_match": "门票数量匹配",
    "taxi_car_count_match": "出租车数量匹配",
    "ticket_budget_total": "门票预算限制",
    "day_end_time_limit": "每日结束时间限制",
    "daily_poi_cap": "每日POI上限",
    "activity_budget_limit": "活动级别预算",
    "activity_duration_limit": "活动时长限制",
    "ticket_price_cap_or_substitute": "门票价格上限/替代",
    "anchor_bundle_budget_limit": "锚点组合预算",
    "nearby_meal_requirement": "就近用餐要求",
    "transport_time_window": "交通时间窗口",
    "adjacent_travel_time_cap": "相邻旅行时间上限",
    "adjacent_travel_distance_cap": "相邻旅行距离上限",
    "max_distance_constraint": "最大距离限制",
    "poi_local_radius_limit": "POI本地半径限制",
    "required_intercity_transport_type": "城际交通类型要求",
    "required_innercity_transport_type": "市内交通类型要求",
    "python_expression": "Python表达式约束",
}

_PREFERENCE_FAMILY_LABELS: Dict[str, str] = {
    "theme_alignment": "主题一致性",
    "anchor_proximity": "锚点邻近性",
    "route_compactness": "路线紧凑性",
    "burden_reduction": "负担减轻",
    "cost_allocation_preference": "成本分配偏好",
    "vibe_alignment": "氛围一致性",
    "experience_richness": "体验丰富度",
}


def build_annotation_scaffold_section(
    metadata: Dict[str, Any],
    level: str,
) -> str:
    """Build a staged oracle-annotation prompt section.

    Args:
        metadata: Sample metadata containing ground-truth fields
            (edit_target_constraints, origin_logical_constraints, conflict_set, etc.).
        level: One of "target_grounding", "conflict_localization", "preservation".

    Returns:
        A Chinese-language prompt section with oracle annotations for the given level.
    """
    if level not in VALID_LEVELS:
        return ""

    sections: List[str] = []

    if level in ("target_grounding", "conflict_localization", "preservation"):
        sections.append(_build_target_grounding(metadata))

    if level in ("conflict_localization", "preservation"):
        conflict_section = _build_conflict_localization(metadata)
        if conflict_section:
            sections.append(conflict_section)

    if level == "preservation":
        preservation_section = _build_preservation(metadata)
        if preservation_section:
            sections.append(preservation_section)

    if not sections:
        return ""

    header = "# 标注脚手架（Oracle Annotation Scaffold）\n\n"
    header += (
        "以下是从基准标注中提取的 ground-truth 信息，帮助你理解编辑任务。"
        "你仍需要自己完成计划编辑，但这些信息可以帮助你定位关键目标和约束。\n"
    )
    return header + "\n\n".join(sections)


# ── target grounding ─────────────────────────────────────────────


def _build_target_grounding(metadata: Dict[str, Any]) -> str:
    edit_constraints: List[Dict[str, Any]] = metadata.get("edit_target_constraints") or []
    edit_prefs: List[Dict[str, Any]] = metadata.get("edit_target_preferences") or []

    lines: List[str] = ["## 编辑目标定位", ""]

    activity_lines = _describe_targeted_activities(edit_constraints)
    global_lines = _describe_global_constraints(edit_constraints)
    pref_lines = _describe_preference_targets(edit_prefs)

    if activity_lines:
        lines.append("### 编辑涉及的特定活动/POI")
        lines.extend(activity_lines)
        lines.append("")

    if global_lines:
        lines.append("### 编辑涉及的全局约束")
        lines.extend(global_lines)
        lines.append("")

    if pref_lines:
        lines.append("### 编辑涉及的软偏好目标")
        lines.extend(pref_lines)
        lines.append("")

    if not activity_lines and not global_lines and not pref_lines:
        lines.append("（未从标注中提取到明确的编辑目标约束）")
        lines.append("")

    return "\n".join(lines)


def _describe_targeted_activities(constraints: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for constraint in constraints:
        ctype = str(constraint.get("type") or "")
        if ctype not in _ACTIVITY_TARGETING_TYPES:
            continue
        label = _TYPE_LABELS.get(ctype, ctype)
        value = constraint.get("value")
        target = constraint.get("target", {})
        operator = constraint.get("operator", "")

        if isinstance(value, list):
            value_str = "、".join(str(v) for v in value)
        else:
            value_str = str(value) if value is not None else "（无具体值）"

        if ctype == "forbidden_attraction_name":
            lines.append(f"- {label}：移除 **{value_str}**")
        elif ctype == "semantic_type_requirement":
            activity_type = target.get("activity_type", "activity") if isinstance(target, dict) else "activity"
            op_label = "必须为" if operator == "includes_type" else "要求"
            lines.append(f"- {label}：{activity_type} 类型 {op_label} **{value_str}**")
        elif ctype == "required_attraction_name":
            lines.append(f"- {label}：**{value_str}**")
        else:
            lines.append(f"- {label}：{value_str}")
    return lines


def _describe_global_constraints(constraints: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for constraint in constraints:
        ctype = str(constraint.get("type") or "")
        if ctype not in _GLOBAL_TYPES and ctype in _ACTIVITY_TARGETING_TYPES:
            continue
        if ctype in _ACTIVITY_TARGETING_TYPES:
            continue
        label = _TYPE_LABELS.get(ctype, ctype)
        value = constraint.get("value")
        operator = constraint.get("operator", "")
        operator_label = _operator_label(operator)
        value_str = _format_value(value)
        if operator_label and value_str:
            lines.append(f"- {label}：{operator_label} {value_str}")
        elif value_str:
            lines.append(f"- {label}：{value_str}")
        else:
            lines.append(f"- {label}")
    return lines


def _describe_preference_targets(prefs: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for pref in prefs:
        family = str(pref.get("family") or pref.get("source_family") or "")
        facet = str(pref.get("facet") or "")
        direction = str(pref.get("direction") or "")
        anchor = pref.get("anchor")
        params = pref.get("params") or {}

        family_label = _PREFERENCE_FAMILY_LABELS.get(family, family)
        dir_label = {"maximize": "最大化", "minimize": "最小化", "maintain": "保持"}.get(
            direction, direction
        )
        parts = [f"- {family_label}"]
        if facet:
            parts.append(f"（{facet}）")
        if dir_label:
            parts.append(f"：{dir_label}")
        target_theme = params.get("target_theme") or params.get("canonical_facet")
        if target_theme:
            parts.append(f" **{target_theme}**")
        if anchor:
            parts.append(f" [锚点: {anchor}]")
        lines.append("".join(parts))
    return lines


# ── conflict localization ────────────────────────────────────────


def _build_conflict_localization(metadata: Dict[str, Any]) -> str:
    conflict_set: List[Any] = metadata.get("conflict_set") or []
    primary_conflict = metadata.get("primary_conflict")
    secondary_conflicts: List[Any] = metadata.get("secondary_conflicts") or []

    if not conflict_set and not primary_conflict:
        return ""

    lines: List[str] = ["## 冲突定位", ""]

    if primary_conflict:
        lines.append(f"- 主冲突类型：`{_format_conflict(primary_conflict)}`")
        lines.append(f"  {_explain_conflict(primary_conflict)}")

    if secondary_conflicts:
        lines.append("- 次要冲突：")
        for conflict in secondary_conflicts:
            lines.append(f"  - `{_format_conflict(conflict)}`：{_explain_conflict(conflict)}")
    elif conflict_set and not primary_conflict:
        lines.append("- 检测到的冲突：")
        for conflict in conflict_set:
            lines.append(f"  - `{_format_conflict(conflict)}`：{_explain_conflict(conflict)}")

    lines.append("")
    return "\n".join(lines)


def _format_conflict(conflict: Any) -> str:
    if isinstance(conflict, list):
        return "/".join(str(c) for c in conflict)
    return str(conflict)


_CONFLICT_EXPLANATIONS: Dict[str, str] = {
    "structural/semantic/overlap": "编辑目标与原计划在同一天存在语义重叠（如同类型活动过多）",
    "structural/semantic/discontinuity": "编辑目标与原计划存在语义不连续",
    "structural/structural/discontinuity": "编辑目标与原计划存在结构性不连续",
    "structural/semantic/incompatibility": "编辑目标与原计划存在语义不兼容",
    "structural/temporal/overlap": "编辑目标与原计划存在时间重叠",
    "structural/temporal/overflow": "编辑目标导致时间溢出（超出可用时间）",
    "structural/resource/overflow": "编辑目标导致资源溢出（如预算超支）",
    "compositional/resource/overflow": "编辑目标跨多天导致资源溢出",
    "compositional/spatial/discontinuity": "编辑目标跨多天导致空间不连续",
    "parameter/budget/incompatibility": "编辑目标与原预算约束不兼容",
    "parameter/temporal/incompatibility": "编辑目标与原时间约束不兼容",
}


def _explain_conflict(conflict: Any) -> str:
    key = _format_conflict(conflict)
    return _CONFLICT_EXPLANATIONS.get(key, "请根据冲突类型分析具体矛盾")


# ── preservation ──────────────────────────────────────────────────


def _build_preservation(metadata: Dict[str, Any]) -> str:
    origin_constraints: List[Dict[str, Any]] = metadata.get("origin_logical_constraints") or []
    edit_constraints: List[Dict[str, Any]] = metadata.get("edit_target_constraints") or []

    if not origin_constraints:
        return ""

    overridden_ids = _compute_overridden_ids(origin_constraints, edit_constraints)

    lines: List[str] = ["## 必须保留的原计划约束", ""]
    preserved_count = 0
    overridden_count = 0

    for constraint in origin_constraints:
        cid = str(constraint.get("id") or "")
        ctype = str(constraint.get("type") or "")
        label = _TYPE_LABELS.get(ctype, ctype)
        value = constraint.get("value")
        operator = constraint.get("operator", "")
        operator_label = _operator_label(operator)
        value_str = _format_value(value)

        desc_parts = [f"- {label}"]
        if operator_label and value_str:
            desc_parts.append(f"：{operator_label} {value_str}")
        elif value_str:
            desc_parts.append(f"：{value_str}")

        if cid in overridden_ids:
            desc_parts.append(" ⚡（已被编辑目标覆盖，无需保留）")
            overridden_count += 1
        else:
            preserved_count += 1

        lines.append("".join(desc_parts))

    lines.append("")
    lines.append(f"共 {len(origin_constraints)} 个原约束：**{preserved_count} 个必须保留**，{overridden_count} 个已被编辑覆盖。")
    lines.append("")
    return "\n".join(lines)


def _compute_overridden_ids(
    origin_constraints: List[Dict[str, Any]],
    edit_constraints: List[Dict[str, Any]],
) -> set[str]:
    """Determine which origin constraints are overridden by edit constraints.

    Replicates the merge policy from evaluation/benchmark/constraint_merge.py:
    - same type and same target → override
    - same GLOBAL_OVERRIDE_TYPE (budget_total, day_count, people_count) → override
    """
    overridden: set[str] = set()
    for oc in origin_constraints:
        oc_type = str(oc.get("type") or "")
        oc_target = oc.get("target", {})
        oc_id = str(oc.get("id") or "")

        for ec in edit_constraints:
            ec_type = str(ec.get("type") or "")
            ec_target = ec.get("target", {})

            if oc_type in _GLOBAL_TYPES and oc_type == ec_type:
                overridden.add(oc_id)
                break
            if oc_type == ec_type and oc_target == ec_target:
                overridden.add(oc_id)
                break

    return overridden


# ── helpers ───────────────────────────────────────────────────────


def _operator_label(operator: str) -> str:
    mapping = {
        "<=": "≤",
        ">=": "≥",
        "==": "=",
        "<": "<",
        ">": ">",
        "contains_all": "包含全部",
        "contains_any": "包含任一",
        "contains_none": "不包含",
        "includes_type": "类型为",
    }
    return mapping.get(operator, operator)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
