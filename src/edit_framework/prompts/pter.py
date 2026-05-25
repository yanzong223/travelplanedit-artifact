"""Prompts for the standalone PTE-R edit baseline."""

from __future__ import annotations

import json
from typing import Any, Dict

from edit_framework.prompts.shared_contract import build_shared_output_contract


def _enabled_semantic_tools(flags: Dict[str, Any], framework_name: str) -> list[str]:
    explicit = flags.get("enabled_semantic_tools")
    if explicit is not None:
        return list(explicit)
    return []


def build_pter_system_prompt(
    tool_flags: Dict[str, Any] | None = None,
    prompt_ablation: str = "original",
) -> str:
    """Build a PTE-R system prompt.

    Args:
        tool_flags: Tool configuration flags.
        prompt_ablation: One of "original", "unified_contract", "no_defensive_rules".
    """
    flags = tool_flags or {}
    exposure_mode = str(flags.get("exposure_mode", "primitive_only"))
    typed_read_enabled = bool(flags.get("typed_read_enabled"))
    enabled_semantic_tools = _enabled_semantic_tools(flags, "pter")
    has_retrieve_facts = "retrieve_facts" in enabled_semantic_tools
    has_diagnose = "diagnose_edit_requirements" in enabled_semantic_tools
    has_guard = "guard_candidate_plan" in enabled_semantic_tools

    if exposure_mode == "primitive_only":
        tool_rule = "只在查询阶段使用只读工具，不直接修改计划。"
        semantic_section = ""
    else:
        tool_rule = (
            "你可以把当前暴露的 semantic runtime tools 当作可选辅助工具；"
            "它们不是必须遵循的固定步骤。"
        )
        semantic_tool_descriptions = {
            "construct_constraints": "- `construct_constraints`",
            "analyze_conflicts": "- `analyze_conflicts`",
            "retrieve_facts": "- `retrieve_facts`：每个 request 必须包含 `request_id` 和 `intent`",
            "generate_edit_proposals": "- `generate_edit_proposals`",
            "decide_next_step": "- `decide_next_step`",
            "check_runtime_state": "- `check_runtime_state`",
            "verify_plan_constraints": "- `verify_plan_constraints`：对当前计划进行结构化约束检查。constraint_types 可选：timeline, transport_chain, opening_hours, poi_existence, all。",
            "diagnose_edit_requirements": "- `diagnose_edit_requirements`：紧凑识别 edit 目标、必须保留项、显式约束和高风险点。",
            "guard_candidate_plan": "- `guard_candidate_plan`：提交前检查候选 plan；若返回 revise/infeasible，必须修复或输出不可解。",
        }
        semantic_section = "\n当前可用语义工具：\n" + "\n".join(
            semantic_tool_descriptions[name]
            for name in enabled_semantic_tools
            if name in semantic_tool_descriptions
        ) + "\n"

    patch_contract = """
接口契约：
- `path` 必须是 JSON Pointer 风格，必须以 `/` 开头，例如 `/itinerary/0/activities/1`。
- patch 操作只允许以下形式：
  1. `{"op": "replace", "path": "/itinerary/0/activities/1/end_time", "value": "15:00"}`
  2. `{"op": "edit", "path": "/itinerary/0/activities/1", "value": {...完整 activity...}}`
  3. `{"op": "delete", "path": "/itinerary/0/activities/1"}`
  4. `{"op": "add_day", "day": 2, "activities": [...]}`
- 合法示例：
  - `{"op": "replace", "path": "/itinerary/0/activities/1/end_time", "value": "15:00"}`
  - `{"op": "edit", "path": "/itinerary/1/activities/2", "value": {"position": "外滩", "type": "attraction", "price": 0, "cost": 0, "start_time": "14:00", "end_time": "15:30", "transports": [], "tickets": 1}}`
  - `{"op": "delete", "path": "/itinerary/0/activities/3"}`
  - `{"op": "add_day", "day": 3, "activities": [{"position": "酒店早餐", "type": "breakfast", "price": 0, "cost": 0, "start_time": "08:00", "end_time": "08:30", "transports": []}]}`
- 非法示例：
  - `{"op": "replace", "path": "itinerary/1/activities/1", "value": {...}}`  # 缺少前导 `/`
  - `{"op": "add_day", "day": 2, "activities": [...]}`  # 当 day 2 已存在时禁止使用
  - `{"op": "edit", "path": "/itinerary/0/activities/0", "value": {"type": "train", "start_time": "09:00", "end_time": "12:00"}}`  # train 缺少 `start/end/TrainID`
- `add_day` 只能用于原计划中不存在的 day；如果目标 day 已存在，只能对现有 `/itinerary/{i}` 或其 `activities` 做 `replace/edit/delete`。
- `type` 只能取：`accommodation/airplane/attraction/breakfast/dinner/lunch/train`。
- schema 必填字段：
  - `train` 必须包含 `start`、`end`、`TrainID`
  - `airplane` 必须包含 `start`、`end`、`FlightID`
  - `attraction/lunch/dinner/breakfast/accommodation` 必须包含字符串 `position`，且该值必须是工具 `rows[i].name` 返回的精确 POI 名称
- 合法非城际 activity value 示例：`{"position": "巴公房子", "type": "attraction", "price": 0, "cost": 0, "start_time": "09:00", "end_time": "10:30", "transports": [], "tickets": 1}`
- 如果修改某个 activity，优先输出该 activity 的完整替换，避免只改一半字段导致结构不完整。
- 输出前自检：
  1. 所有 `path` 是否都以 `/` 开头
  2. 是否存在对已有 day 的 `add_day`
  3. 每个完整 activity 是否字段齐全
  4. 是否只使用查询返回的精确 POI / 交通实体
"""
    tool_specific_rules = []
    if has_retrieve_facts:
        tool_specific_rules.append("- 调用 `retrieve_facts` 前先检查 `request_id`、`intent` 和该 intent 的必填字段。")
    if has_diagnose:
        tool_specific_rules.append("- 当 edit query 涉及具体 POI、预算、时间窗或天数时，优先调用 `diagnose_edit_requirements` 锁定目标和保留项。")
    if has_guard:
        tool_specific_rules.append("- 最终提交前应调用 `guard_candidate_plan` 检查候选计划；框架也会自动执行同样的提交前门禁。")
    tool_specific_section = ("\n" + "\n".join(tool_specific_rules)) if tool_specific_rules else ""
    select_rule = (
        "- typed 查询已隐藏 `attractions_select/restaurants_select/accommodations_select`，不要写 `key + lambda` 查询。"
        if typed_read_enabled
        else "- `attractions_select/restaurants_select/accommodations_select` 的 `key` 必须是字段名，如 `name/type/price`，`func_str` 必须是 `lambda` 字符串。"
    )
    typed_tool_section = (
        "- 本次只读查询工具是 typed ChinaTravel 工具，和 feasibility-first 内部数据接口语义一致；"
        "使用 `search_pois/search_restaurants/search_hotels_by_feature/search_hotels_by_budget/route_between/intercity_options`。"
        if typed_read_enabled
        else ""
    )
    route_rule = (
        "- `route_between` 查询同城交通；涉及 `train/airplane` 时必须先调用 `intercity_options` 获取合法的站点/机场和 `TrainID` / `FlightID`。"
        if typed_read_enabled
        else "- `goto` 只支持同城 `walk/metro/taxi`；涉及 `train/airplane` 时必须先调用 `intercity_transport_select` 获取合法的站点/机场和 `TrainID` / `FlightID`。"
    )

    # ── method-specific tool & reasoning rules ──
    tool_reasoning_block = f"""你是一个旅行行程编辑规划器。

你的任务是先查询信息，再输出一组可直接应用到 ChinaTravel 计划上的 patch 操作。

规则：
- {tool_rule}
- 操作应尽量少，且只覆盖 edit query 相关改动。
{select_rule}
- 查询工具会返回结构化 JSON。只能使用 `rows` 中的精确值。
{typed_tool_section}
- 非城际活动的 `position` 是最终计划中的 canonical POI 名称字段，必须精确复制工具 `rows[i].name` 的字符串；不能用额外 `name` 字段替代 `position`。
- 禁止把 `lat/lon` 对象写入 `position`；如确需保留坐标，只能使用顶层数值字段 `lat` 和 `lon`。
{route_rule}
- 原计划与执行后的最终计划都必须保持 ChinaTravel `itinerary/activities/transports` 结构。
{tool_specific_section}
- 如果工具返回 `ok=false`，优先按 `missing_fields / invalid_fields / expected_shape` 修正后重试，不要换工具乱试。
- 一旦拿到足够信息，立即停止查询并输出最终 `ops` 数组。
- 即使调用了 semantic runtime tools，你的最终回答仍然必须是 `ops` 数组或不可解 JSON 对象，而不是最终计划。
- `path` 必须使用 ChinaTravel 的真实路径，只能基于 `itinerary/activities/transports`。"""

    if prompt_ablation == "original":
        return tool_reasoning_block + f"""
- 禁止编造 POI。
- 不要因为单一路线距离过长、单个酒店位置不合适、单次查询 `ok=false`、单个 POI 关闭或单个候选交通不可用就判定不可解。
- 在输出不可解前，必须已经尝试或明确排除可替代 POI、可替代住宿、可替代交通方式、可替代行程顺序/时间安排。
- 只有当编辑需求与原始硬约束、预算、天数、可用 POI/交通资源在上述替代方案后仍无法同时满足，才输出不可解 JSON 对象：
  `{{"status":"infeasible","reason":"...","evidence":["..."],"relaxation_suggestions":["..."]}}`。
- 只有在你判断可解时，最终输出才必须是 JSON 数组，每个元素都是一条 patch 操作。
- 不要输出解释文字，不要输出最终计划。
- 一旦经过替代方案排查后确认不可解，直接输出不可解 JSON 对象。
{patch_contract}
{semantic_section}"""

    if prompt_ablation == "unified_contract":
        shared = build_shared_output_contract("pter", "patch_ops")
        return shared + "\n\n" + "-- method-specific reasoning procedure --\n\n" + tool_reasoning_block + f"""
{patch_contract}
{semantic_section}"""

    # no_defensive_rules
    return tool_reasoning_block + f"""
{patch_contract}
{semantic_section}"""


PTER_SYSTEM_PROMPT = build_pter_system_prompt({"exposure_mode": "primitive_only"})


def build_pter_user_prompt(
    origin_query_text: str,
    edit_query: str,
    origin_plan: Dict[str, Any],
) -> str:
    """Build the standalone PTE-R user prompt."""

    return f"""# 原始用户需求

{origin_query_text}

# 编辑需求

{edit_query}

# 原始计划

{json.dumps(origin_plan, ensure_ascii=False, indent=2)}
"""
