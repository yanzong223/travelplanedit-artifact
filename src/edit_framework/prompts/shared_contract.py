"""Shared output contract prepended to all baseline prompts.

Provides identical output-validity rules across frameworks so that
method-comparison is not confounded by prompt-contract differences.
"""

from __future__ import annotations


def build_shared_output_contract(
    framework_name: str = "react",
    output_mode: str = "full_plan",
) -> str:
    """Return a shared output-validity block.

    Args:
        framework_name: e.g. "react", "reflexion", "pter".
        output_mode: "full_plan" (react, reflexion) or "patch_ops" (pter).
    """

    if output_mode == "full_plan":
        output_format_rule = (
            "最终输出必须是完整的 ChinaTravel JSON plan object："
            "顶层只能包含 `people_number`、`start_city`、`target_city`、`itinerary`。"
            "禁止输出 `days/items/routes` 结构，禁止输出操作列表、diff 或 markdown。"
        )
    elif output_mode == "patch_ops":
        output_format_rule = (
            "最终输出必须是 JSON 数组，每个元素是一条 patch 操作。"
            "禁止输出最终计划、解释文字或 markdown。"
        )
    else:
        output_format_rule = ""

    return f"""# 输出合法性契约（所有方法共享）

以下规则对所有编辑方法均适用，违反即视为编辑失败：

## 数据真实性
- 不允许编造新的景点、餐厅、住宿或路线。
- 最终计划中的 POI 名称必须是工具返回的精确名称，禁止使用"某景点附近餐厅""附近酒店""就近餐厅"等描述性占位名。
- 非城际活动的 `position` 是最终计划中的 canonical POI 名称字段，必须精确复制工具 `rows[i].name` 的字符串；不能用额外 `name` 字段替代 `position`。
- 禁止把 `lat/lon` 对象写入 `position`；如确需保留坐标，只能使用顶层数值字段 `lat` 和 `lon`。
- 不能为了表达休息、中转、整理行李而创造新的 POI 或新的活动类型；如果需要回酒店休息，只能使用已查询到的真实住宿 POI，并保持类型为 `accommodation`。
- 若某次查询返回 `rows=[]` 或 `ok=false`，不能编造替代 POI，必须改用别的合法查询结果或保持原计划相关部分。

## 活动类型与字段
- 活动 `type` 只能使用这些合法值：`airplane`、`train`、`attraction`、`breakfast`、`lunch`、`dinner`、`accommodation`。
  禁止输出 `rest`、`hotel`、`restaurant`、`transport` 等其他类型。
- `train` 活动必须包含 `start`、`end`（具体站点名，不能写城市名）、`TrainID`。
- `airplane` 活动必须包含 `start`、`end`（具体机场名，不能写城市名）、`FlightID`。
- 非城际活动（`attraction/lunch/dinner/breakfast/accommodation`）必须填写 `position`。
- 合法非城际活动示例：`{{"type":"attraction","position":"巴公房子","start_time":"09:00","end_time":"10:30","transports":[]}}`。

## 输出格式
- {output_format_rule}
- 回复中不要包含解释，只输出 JSON。

## 不可解处理
- 如果你判断编辑需求与原始硬约束、预算、天数、可用 POI/交通资源无法同时满足，必须直接输出不可解 JSON 对象：
  `{{"status":"infeasible","reason":"...","evidence":["..."],"relaxation_suggestions":["..."]}}`。
- 只有在你判断可解时，才输出正常结果。

## 最小修改原则
- 尽量保持无关 day、activity、transport 和用户原始约束不变。
- 只修改 edit query 直接相关的部分。
"""
