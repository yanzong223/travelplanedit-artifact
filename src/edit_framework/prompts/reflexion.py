"""Prompts for the standalone Reflexion edit baseline."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from edit_framework.prompts.ablation import build_prompt_ablation_sections
from edit_framework.prompts.shared_contract import build_shared_output_contract


def _enabled_semantic_tools(flags: Dict[str, Any], framework_name: str) -> list[str]:
    explicit = flags.get("enabled_semantic_tools")
    if explicit is not None:
        return list(explicit)
    return []


def build_reflexion_system_prompt(
    tool_flags: Dict[str, Any] | None = None,
    prompt_ablation: str = "original",
) -> str:
    """Build a Reflexion system prompt.

    Args:
        tool_flags: Tool configuration flags.
        prompt_ablation: One of "original", "unified_contract", "no_defensive_rules".
    """
    flags = tool_flags or {}
    exposure_mode = str(flags.get("exposure_mode", "primitive_only"))
    typed_read_enabled = bool(flags.get("typed_read_enabled"))
    enabled_semantic_tools = _enabled_semantic_tools(flags, "reflexion")
    has_retrieve_facts = "retrieve_facts" in enabled_semantic_tools
    has_execute_patch = "execute_plan_patch" in enabled_semantic_tools
    has_proposals = "generate_edit_proposals" in enabled_semantic_tools
    has_decision = "decide_next_step" in enabled_semantic_tools
    has_runtime_check = "check_runtime_state" in enabled_semantic_tools
    has_diagnose = "diagnose_edit_requirements" in enabled_semantic_tools
    has_guard = "guard_candidate_plan" in enabled_semantic_tools

    if exposure_mode == "primitive_only":
        tool_rule = "你只能使用只读查询工具；不要调用本地编辑工具。"
        semantic_lines: list[str] = []
    else:
        tool_rule = (
            "你可以把当前暴露的 semantic runtime tools 当作辅助工具来推进当前轮编辑。"
            "它们是可选工具，不是必须遵循的固定流程。"
        )
        semantic_tool_descriptions = {
            "construct_constraints": "- `construct_constraints`：把 edit query 结构化成 active constraints 与 anchors。",
            "analyze_conflicts": "- `analyze_conflicts`：基于当前 plan 与 constraints 输出 grounded conflicts。",
            "retrieve_facts": "- `retrieve_facts`：通过统一 retrieval 接口检索只读事实；每个 request 必须带 `request_id` 和 `intent`。",
            "generate_edit_proposals": "- `generate_edit_proposals`：生成候选编辑方案，供你构造最终完整计划时参考。",
            "decide_next_step": "- `decide_next_step`：在 proposal、facts 和 checker feedback 上做当前轮决策支持。",
            "execute_plan_patch": "- `execute_plan_patch`：如果你想验证候选修改，可调用它做确定性计划修改；每个 `op.args` 都要先检查必填字段，尤其是 `day`。",
            "check_runtime_state": "- `check_runtime_state`：输出 runtime-local structural / constraint feedback。",
            "verify_plan_constraints": "- `verify_plan_constraints`：对当前计划进行结构化约束检查。constraint_types 可选：timeline, transport_chain, opening_hours, poi_existence, all。返回每个维度的通过/失败状态及违规详情。",
            "diagnose_edit_requirements": "- `diagnose_edit_requirements`：紧凑识别 edit 目标、必须保留项、显式约束和高风险点。",
            "guard_candidate_plan": "- `guard_candidate_plan`：提交前检查候选 plan；若返回 revise/infeasible，必须修复或输出不可解。",
        }
        semantic_lines = [
            semantic_tool_descriptions[name]
            for name in enabled_semantic_tools
            if name in semantic_tool_descriptions
        ]

    semantic_section = ""
    if semantic_lines:
        semantic_section = "\n当前可用语义工具：\n" + "\n".join(semantic_lines) + "\n"

    # ── minimal plan format hint (used only when shared contract is absent) ──
    _minimal_plan_hint = """## 输出格式提示
- 最终输出必须是完整的 ChinaTravel plan JSON object。
- 顶层保留 `people_number/start_city/target_city/itinerary`。
- `itinerary` 为 day 列表，每个 day 含 `day` 和 `activities`。"""

    tool_specific_rules = []
    if has_retrieve_facts:
        tool_specific_rules.append("- 调用 `retrieve_facts` 前先检查 `request_id`、`intent` 和该 intent 的必填字段。")
    if has_execute_patch:
        tool_specific_rules.append("- 调用 `execute_plan_patch` 前先检查每个 `op.args`，尤其是 `day`。")
    if has_diagnose:
        tool_specific_rules.append("- 当 edit query 涉及具体 POI、预算、时间窗或天数时，优先调用 `diagnose_edit_requirements` 锁定目标和保留项。")
    if has_guard:
        tool_specific_rules.append("- 最终提交前应调用 `guard_candidate_plan` 检查候选计划；框架也会自动执行同样的提交前门禁。")
    tool_specific_section = ("\n" + "\n".join(tool_specific_rules)) if tool_specific_rules else ""
    semantic_runtime_mentions = [
        f"`{tool_name}`"
        for tool_name, enabled in (
            ("generate_edit_proposals", has_proposals),
            ("decide_next_step", has_decision),
            ("execute_plan_patch", has_execute_patch),
            ("check_runtime_state", has_runtime_check),
        )
        if enabled
    ]
    semantic_runtime_clause = " / ".join(semantic_runtime_mentions)
    if semantic_runtime_clause:
        semantic_runtime_clause = f"即使调用了 {semantic_runtime_clause} 或其他当前暴露的 semantic runtime tools"
    else:
        semantic_runtime_clause = "即使调用了当前暴露的 semantic runtime tools"
    route_rule = (
        "- `route_between` 查询同城交通；涉及 `train/airplane` 时必须先调用 `intercity_options` 获取合法信息。"
        if typed_read_enabled
        else "- `goto` 只支持同城 `walk/metro/taxi`；涉及 `train/airplane` 时必须先调用 `intercity_transport_select` 获取合法信息。"
    )
    typed_tool_section = (
        "- 本次只读查询工具是 typed ChinaTravel 工具，和 feasibility-first 内部数据接口语义一致；"
        "使用 `search_pois/search_restaurants/search_hotels_by_feature/search_hotels_by_budget/route_between/intercity_options`，"
        "不要写 `key + lambda` 查询。"
        if typed_read_enabled
        else ""
    )

    # ── method-specific reasoning procedure (shared across all ablation modes) ──
    reasoning_block = f"""你是一个旅行行程编辑规划器。

你的任务是先查询信息，再直接输出编辑后的完整 ChinaTravel 计划；如果校验反馈指出错误，你需要根据反馈修复完整计划并重试。

规则：
- {tool_rule}
- Reflexion-style retry 会把失败原因总结成 per-case reflection memory；下一轮必须遵守该 memory 中的修正点和保留约束。
- 最终输出必须是一个 JSON object，内容是完整的 edited ChinaTravel plan。
- 不要输出解释文字，不要输出操作数组、diff 或 markdown。
- 改动应尽量少，且只覆盖 edit query 相关内容。
- 查询工具会返回结构化 JSON。只能使用 `rows` 中的精确值。
{typed_tool_section}
- 非城际活动的 `position` 是最终计划中的 canonical POI 名称字段，必须精确复制工具 `rows[i].name` 的字符串；不能用额外 `name` 字段替代 `position`。
- 禁止把 `lat/lon` 对象写入 `position`；如确需保留坐标，只能使用顶层数值字段 `lat` 和 `lon`。
{route_rule}
- 原计划与最终计划都必须保持 ChinaTravel `itinerary/activities/transports` 结构。
{tool_specific_section}
- 如果工具返回 `ok=false`，优先按 `missing_fields / invalid_fields / expected_shape` 修正同一个工具调用。
- 如果上一轮校验失败，你会收到结构化反馈。必须优先修复反馈指出的问题，并尽量保持无关计划内容不变。
- 如果反馈是 JSON / canonical plan schema 错误，优先补齐或修正字段，不改变无关计划。
- 一旦信息足够，立即停止查询并输出完整 edited plan JSON object。
- {semantic_runtime_clause}，你的最终回答仍然必须是完整 edited plan JSON object。
{semantic_section}"""

    if prompt_ablation == "original":
        plan_contract = """
完整计划接口契约：
- 最终回答必须是一个 JSON object，表示编辑后的完整 ChinaTravel plan。
- 顶层必须保留 `people_number/start_city/target_city/itinerary` 等原计划字段。
- `itinerary` 必须是 day 列表；每个 day 包含 `day` 和 `activities`。
- 每个 activity 必须保持 canonical activity schema：
  - `type` 只能取 `accommodation/airplane/attraction/breakfast/dinner/lunch/train`
  - `train` 必须包含 `start/end/TrainID`
  - `airplane` 必须包含 `start/end/FlightID`
  - `attraction/lunch/dinner/breakfast/accommodation` 必须包含字符串 `position`，且该值必须是工具 `rows[i].name` 返回的精确 POI 名称
  - 常规活动应保留 `start_time/end_time/price/cost/transports` 等结构字段
- 合法非城际活动示例：`{"type":"attraction","position":"巴公房子","start_time":"09:00","end_time":"10:30","transports":[]}`
- 只修改 edit query 相关部分；无关 day、activity、transport 和用户原始约束应尽量原样保留。
- 输出前自检：
  1. 顶层是否是完整 plan object，而不是数组、diff 或操作列表
  2. 是否保留了所有无关行程内容
  3. 是否补齐了 train / airplane / attraction 等活动的必需字段
  4. 是否避免编造未由工具或原计划支持的 POI、车次或航班
"""
        anti_fab = "\n- 禁止编造 POI。"
        return reasoning_block + anti_fab + "\n" + plan_contract

    if prompt_ablation == "unified_contract":
        shared = build_shared_output_contract("reflexion", "full_plan")
        return shared + "\n\n" + "-- method-specific reasoning procedure --\n\n" + reasoning_block

    # no_defensive_rules
    return reasoning_block + "\n" + _minimal_plan_hint


REFLEXION_SYSTEM_PROMPT = build_reflexion_system_prompt({"exposure_mode": "primitive_only"})


def build_reflexion_user_prompt(
    origin_query_text: str,
    edit_query: str,
    origin_plan: Dict[str, Any],
    *,
    metadata: Dict[str, Any] | None = None,
    context_prompt: bool = False,
    database_prompt: bool = False,
) -> str:
    """Build the standalone Reflexion user prompt."""

    ablation_sections = build_prompt_ablation_sections(
        origin_plan=origin_plan,
        metadata=metadata,
        context_prompt=context_prompt,
        database_prompt=database_prompt,
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


def build_reflection_generation_prompt(
    *,
    edit_query: str,
    feedback_items: Iterable[str],
    attempted_plan_summary: str | None = None,
) -> str:
    """Build the explicit Reflexion memory-generation prompt."""

    feedback_text = "\n".join(f"- {item}" for item in feedback_items) or "- 未提供具体反馈"
    summary_section = (
        f"\n# 上一轮计划摘要（非完整计划）\n\n{attempted_plan_summary}\n"
        if attempted_plan_summary
        else ""
    )
    return f"""上一轮行程编辑尝试失败了。请先生成一段用于下一轮尝试的自然语言 reflection memory。

# 编辑需求

{edit_query}
{summary_section}

# 执行/校验反馈

{feedback_text}

请只输出一段简洁中文 reflection，不要输出 JSON。必须覆盖：
- 失败原因
- 下一轮需要具体修正什么
- 哪些原始约束、edit query 约束、无关计划部分必须保留
"""


def build_reflection_retry_prompt(
    *,
    edit_query: str,
    feedback_items: Iterable[str],
    reflection_text: str | None = None,
) -> str:
    """Build the retry prompt after one failed full-plan attempt."""

    feedback_text = "\n".join(f"- {item}" for item in feedback_items) or "- 未提供具体反馈"
    reflection_section = (
        f"\n# Reflection memory\n\n{reflection_text}\n"
        if reflection_text
        else ""
    )
    return f"""上一轮输出的完整 edited plan 在解析或校验时失败了，请基于以下反馈重新输出完整计划。

# 编辑需求

{edit_query}

# 失败反馈

{feedback_text}
{reflection_section}

修复规则：
- 不要把上一轮完整计划复制进 prompt；只根据原始计划、编辑需求、工具观察和上述错误摘要修复。
- 优先做最小修改，保持无关 day、activity、transport 和原始约束不变。
- 如果错误包含 JSON 解析问题，只输出一个合法 JSON object。
- 如果错误包含缺字段、非法 `type`、或 canonical plan schema 错误，只补齐或修正相关字段。
- 如果错误包含时间、交通或 POI 合法性问题，只修复相关活动和相邻 transport。

请只输出修复后的完整 edited plan JSON object，不要输出解释文字。
"""
