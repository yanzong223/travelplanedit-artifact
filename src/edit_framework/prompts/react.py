"""Prompts for the standalone ReAct edit baseline."""

from __future__ import annotations

import json
from typing import Any, Dict

from edit_framework.prompts.ablation import build_prompt_ablation_sections
from edit_framework.prompts.shared_contract import build_shared_output_contract

def _enabled_semantic_tools(flags: Dict[str, Any], framework_name: str) -> list[str]:
    explicit = flags.get("enabled_semantic_tools")
    if explicit is not None:
        return list(explicit)
    return []


def build_react_system_prompt(
    tool_flags: Dict[str, Any] | None = None,
    prompt_ablation: str = "original",
) -> str:
    """Build a React system prompt that matches the enabled tool set.

    Args:
        tool_flags: Tool configuration flags.
        prompt_ablation: One of "original", "unified_contract", "no_defensive_rules".
    """
    flags = tool_flags or {}
    exposure_mode = str(flags.get("exposure_mode", "primitive_only"))
    enable_ct_atoms = bool(flags.get("enable_ct_atoms"))
    enable_ct_notepad = bool(flags.get("enable_ct_notepad"))
    enable_ct_verify = bool(flags.get("enable_ct_verify"))
    enable_ct_conflict_lift = bool(flags.get("enable_ct_conflict_lift"))
    typed_read_enabled = bool(flags.get("typed_read_enabled"))
    enabled_semantic_tools = set(_enabled_semantic_tools(flags, "react"))
    has_constraints = "construct_constraints" in enabled_semantic_tools
    has_conflicts = "analyze_conflicts" in enabled_semantic_tools
    has_retrieve_facts = "retrieve_facts" in enabled_semantic_tools
    has_proposals = "generate_edit_proposals" in enabled_semantic_tools
    has_decision = "decide_next_step" in enabled_semantic_tools
    has_execute_patch = "execute_plan_patch" in enabled_semantic_tools
    has_runtime_check = "check_runtime_state" in enabled_semantic_tools
    has_diagnose = "diagnose_edit_requirements" in enabled_semantic_tools
    has_guard = "guard_candidate_plan" in enabled_semantic_tools

    if exposure_mode == "runtime_high_level":
        local_tool_rule = (
            "优先通过当前暴露的 semantic runtime tools 工作：用可用的约束、冲突、事实、proposal、决策、patch、"
            "runtime check 工具推进编辑。"
            "不要把底层 mutation atom 当成一级工具。"
        )
    elif exposure_mode == "hybrid":
        local_tool_rule = (
            "你可以使用当前暴露的 semantic runtime tools 处理高层编辑决策；"
            "必要时也可以调用只读查询工具补事实。"
            "不要把底层 mutation atom 当成一级工具。"
        )
    else:
        local_tool_rule = (
            "当前 baseline 使用 primitive tools。"
            "若暴露了 ChinaTravel 本地编辑工具，你可以用它们对 working plan 做增量修改；"
            "若没有暴露本地编辑工具，你只能查询，再直接给出完整的修改后计划。"
        )
    notepad_rule = (
        "如果需要保留中间判断、候选 POI 或约束摘要，优先使用 `append_notepad_ct` 记录短笔记；"
        "需要回看时用 `read_notepad_ct`，不要在对话里反复复述长段上下文。 "
        if enable_ct_notepad
        else ""
    )
    verify_rule = (
        "当你不确定当前 working plan 是否仍满足显式约束时，可以调用 `check_runtime_state` 做 runtime feedback。 "
        if has_runtime_check or enable_ct_verify
        else ""
    )
    if has_constraints and has_conflicts:
        conflict_rule = (
            "当 edit query 含糊或涉及多重约束时，可以先调用 `construct_constraints` 再调用 `analyze_conflicts`。 "
        )
    elif has_constraints:
        conflict_rule = "当 edit query 含糊或涉及多重约束时，可以调用 `construct_constraints` 先结构化约束。 "
    elif has_conflicts:
        conflict_rule = "当你已经拿到约束后，可以调用 `analyze_conflicts` 在当前计划上定位冲突。 "
    else:
        conflict_rule = (
            "当 edit query 含糊、涉及多重约束或需要先结构化问题时，可以调用 `lift_conflicts_ct` 提升出约束与锚点。 "
            if enable_ct_conflict_lift
            else ""
        )
    optional_tool_lines = []
    if exposure_mode in {"runtime_high_level", "hybrid"}:
        semantic_tool_descriptions = {
            "construct_constraints": "- `construct_constraints`：把 edit query 结构化成 active constraints 与 anchors。",
            "analyze_conflicts": "- `analyze_conflicts`：基于当前 plan 和 constraints 输出 grounded conflicts。",
            "retrieve_facts": "- `retrieve_facts`：通过统一 retrieval 接口调用后端只读事实工具；每个 request 必须包含 `request_id` 和 `intent`。",
            "generate_edit_proposals": "- `generate_edit_proposals`：输出受 patch vocabulary 约束的 proposal。",
            "decide_next_step": "- `decide_next_step`：在 proposal / facts / checker feedback 上做当前轮决策支持。",
            "execute_plan_patch": "- `execute_plan_patch`：按语义 patch op 执行确定性修改；每个 `op.args` 都要先检查必填字段，尤其是 `day`。",
            "check_runtime_state": "- `check_runtime_state`：输出 runtime-local structural / constraint checks。",
            "verify_plan_constraints": "- `verify_plan_constraints`：对当前计划进行结构化约束检查。constraint_types 可选：timeline, transport_chain, opening_hours, poi_existence, all。返回每个维度的通过/失败状态及违规详情。",
            "diagnose_edit_requirements": "- `diagnose_edit_requirements`：紧凑识别 edit 目标、必须保留项、显式约束和高风险点。",
            "guard_candidate_plan": "- `guard_candidate_plan`：提交前检查候选 plan；若返回 revise/infeasible，必须修复或输出不可解。",
        }
        optional_tool_lines.extend(
            semantic_tool_descriptions[name]
            for name in _enabled_semantic_tools(flags, "react")
            if name in semantic_tool_descriptions
        )
    elif enable_ct_verify:
        optional_tool_lines.append(
            "- `verify_working_plan_ct`：可选；当你不确定当前 working plan 是否仍满足显式约束时使用。"
        )
    if exposure_mode not in {"runtime_high_level", "hybrid"} and enable_ct_conflict_lift:
        optional_tool_lines.append(
            "- `lift_conflicts_ct`：可选；当 edit query 含糊、包含多重约束，或你想先把问题结构化成约束与锚点时使用。"
        )
    optional_tools_section = ""
    if optional_tool_lines:
        optional_tools_section = "\n本次额外可选工具：\n" + "\n".join(optional_tool_lines) + "\n"
    tool_specific_rules = []
    if has_retrieve_facts:
        tool_specific_rules.append(
            "- 调用 `retrieve_facts` 前先检查：每个 request 都有 `request_id`、`intent`，并且补齐该 intent 的必填字段。"
        )
    if has_execute_patch:
        tool_specific_rules.append(
            "- 调用 `execute_plan_patch` 前先检查：每个 `op` 都有 `args`，并且 `args.day` 等必填字段齐全。"
        )
    if has_diagnose:
        tool_specific_rules.append("- 当 edit query 涉及具体 POI、预算、时间窗或天数时，优先调用 `diagnose_edit_requirements` 锁定目标和保留项。")
    if has_guard:
        tool_specific_rules.append("- 最终提交前应调用 `guard_candidate_plan` 检查候选计划；框架也会自动执行同样的提交前门禁。")
    tool_specific_section = ("\n" + "\n".join(tool_specific_rules)) if tool_specific_rules else ""
    typed_tool_section = ""
    if typed_read_enabled:
        typed_tool_section = """
- 本次只读查询工具是 typed ChinaTravel 工具，和 feasibility-first 内部数据接口语义一致。
- 用 `search_pois` 查询景点，`search_restaurants` 查询餐厅，`search_hotels_by_feature` / `search_hotels_by_budget` 查询酒店。
- 用 `route_between` 查询同城交通；用 `intercity_options` 查询 train/airplane。
- 不要写 `key + lambda` 查询；不要假设存在 row-wise lambda。
- typed rows 的 canonical 字段包括：`id/name/category/city/lat/lon/opentime/endtime/price/featurehoteltype/numbed/source_tool`。
"""
    primitive_query_rule = (
        "- typed 查询已隐藏 `attractions_select/restaurants_select/accommodations_select`，不要写 `key + lambda` 查询。"
        if typed_read_enabled
        else "- `attractions_select/restaurants_select/accommodations_select` 的 `key` 必须是字段名，如 `name/type/price`，`func_str` 必须是 `lambda` 字符串。"
    )
    route_query_rule = (
        "- `route_between` 查询同城交通；城际 `train/airplane` 必须通过 `intercity_options` 查询合法 `TrainID` / `FlightID`、起终点站或机场。"
        if typed_read_enabled
        else "- `goto` 只支持同城 `walk/metro/taxi`。城际 `train/airplane` 必须通过 `intercity_transport_select` 查询合法 `TrainID` / `FlightID`、起终点站或机场。"
    )

    # ── method-specific tool & reasoning rules (shared across all ablation modes) ──
    tool_reasoning_block = f"""{optional_tools_section}你是一个旅行行程编辑助手。

你的任务是根据原始用户需求和新的 edit query，对已有计划做最小必要修改。

规则：
- 只使用提供的工具查询真实 POI / 路线信息。
- {local_tool_rule}
- {notepad_rule}{verify_rule}{conflict_rule}
{primitive_query_rule}
- 查询工具会返回结构化 JSON。必须只使用其中 `rows` 里的精确字段值，不要根据字符串表格猜测。
{route_query_rule}
{tool_specific_section}
{typed_tool_section}
- 如果工具返回 `ok=false`，优先根据 `missing_fields / invalid_fields / expected_shape` 修正后重试，不要换工具乱试。
- 一旦查询信息足够，立即停止查询并输出最终 JSON。"""

    if prompt_ablation == "original":
        return f"""{tool_reasoning_block}
- 不允许编造新的景点、餐厅、住宿或路线。
- 最终计划中的 POI 名称必须是工具返回的精确名称，禁止使用"某景点附近餐厅/附近酒店/就近餐厅"等描述性占位名。
- 非城际活动的 `position` 是最终计划中的 canonical POI 名称字段，必须精确复制工具 `rows[i].name` 的字符串；不能用额外 `name` 字段替代 `position`。
- 禁止把 `lat/lon` 对象写入 `position`；如确需保留坐标，只能使用顶层数值字段 `lat` 和 `lon`。
- 合法非城际活动示例：`{{"type":"attraction","position":"巴公房子","start_time":"09:00","end_time":"10:30","transports":[]}}`。
- 最终计划必须是 ChinaTravel JSON：顶层只能包含 `people_number`、`start_city`、`target_city`、`itinerary`。禁止输出 `days/items/routes` 结构。
- 不能为了表达休息、中转、整理行李而创造新的 POI 或新的活动类型；如果需要回酒店休息，只能使用已查询到的真实住宿 POI，并保持类型为 `accommodation`。
- 尽量保持无关部分不变。
- 如果你判断编辑需求与原始硬约束、预算、天数、可用 POI/交通资源无法同时满足，必须直接输出不可解 JSON 对象：
  `{{"status":"infeasible","reason":"...","evidence":["..."],"relaxation_suggestions":["..."]}}`。
- 只有在你判断可解时，最终才必须返回完整的计划 JSON。
- 活动 `type` 只能使用这些合法值：`airplane`、`train`、`attraction`、`breakfast`、`lunch`、`dinner`、`accommodation`。禁止输出 `rest`、`hotel`、`restaurant`、`transport` 等其他类型。
- `train/airplane` 活动必须填写具体站点/机场名到 `start` / `end`，不能直接写城市名，并且必须包含合法 `TrainID` / `FlightID`。
- 非城际活动必须填写 `position`；所有活动必须填写合法 `type`。
- 若某次查询返回 `rows=[]` 或 `ok=false`，不能编造替代 POI，必须改用别的合法查询结果或保持原计划相关部分。
- 回复中不要包含解释，只输出 JSON。
"""

    if prompt_ablation == "unified_contract":
        shared = build_shared_output_contract("react", "full_plan")
        separator = "-- method-specific reasoning procedure --"
        return f"""{shared}

{separator}

{tool_reasoning_block}"""

    # no_defensive_rules
    return f"""{tool_reasoning_block}"""


REACT_SYSTEM_PROMPT = build_react_system_prompt()


def build_react_user_prompt(
    origin_query_text: str,
    edit_query: str,
    origin_plan: Dict[str, Any],
    *,
    metadata: Dict[str, Any] | None = None,
    context_prompt: bool = False,
    database_prompt: bool = False,
    annotation_scaffold_level: str = "none",
) -> str:
    """Build the standalone ReAct user prompt."""

    ablation_sections = build_prompt_ablation_sections(
        origin_plan=origin_plan,
        metadata=metadata,
        context_prompt=context_prompt,
        database_prompt=database_prompt,
        annotation_scaffold_level=annotation_scaffold_level,
    )
    optional_sections = f"\n\n{ablation_sections}" if ablation_sections else ""

    return f"""# 原始用户需求

{origin_query_text}

# 编辑需求

{edit_query}

# 原始计划

{json.dumps(origin_plan, ensure_ascii=False, indent=2)}
{optional_sections}
"""
