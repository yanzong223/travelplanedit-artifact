"""
Diff Utilities for Plan Editing
计算和提取plan之间的差异

Adapted from src/utils/plan_comparator.py for data generation needs
"""

from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class ActivityDiff:
    """活动差异记录"""
    activity_id: str
    diff_type: str  # "added", "removed", "modified", "unchanged", "reordered"
    before_data: Optional[Dict[str, Any]] = None
    after_data: Optional[Dict[str, Any]] = None
    day_before: Optional[int] = None
    day_after: Optional[int] = None


@dataclass
class DayDiff:
    """天差异记录"""
    day_id: int
    diff_type: str  # "added", "removed", "modified", "unchanged"
    activities_diff: List[ActivityDiff]
    before_activities_count: int
    after_activities_count: int


@dataclass
class PlanDiff:
    """Plan差异总结"""
    origin_plan: Dict[str, Any]
    edited_plan: Dict[str, Any]
    days_diff: List[DayDiff]

    # 统计
    days_added: int
    days_removed: int
    days_modified: int
    activities_added: int
    activities_removed: int
    activities_modified: int
    activities_reordered: int


def calculate_plan_diff(origin_plan: Dict[str, Any], edited_plan: Dict[str, Any]) -> PlanDiff:
    """
    计算两个plan之间的差异

    Args:
        origin_plan: 原始plan
        edited_plan: 编辑后的plan

    Returns:
        PlanDiff对象
    """
    origin_itinerary = origin_plan.get("itinerary", [])
    edited_itinerary = edited_plan.get("itinerary", [])

    # 创建day索引
    origin_days = {day_plan.get("day"): day_plan for day_plan in origin_itinerary}
    edited_days = {day_plan.get("day"): day_plan for day_plan in edited_itinerary}

    all_day_numbers = set(origin_days.keys()) | set(edited_days.keys())

    days_diff = []
    days_added = 0
    days_removed = 0
    days_modified = 0
    activities_added = 0
    activities_removed = 0
    activities_modified = 0
    activities_reordered = 0

    for day_num in sorted(all_day_numbers):
        origin_day = origin_days.get(day_num)
        edited_day = edited_days.get(day_num)

        if origin_day and edited_day:
            # 天都存在，检查activities变化
            origin_acts = origin_day.get("activities", [])
            edited_acts = edited_day.get("activities", [])

            activities_diff = _compare_activities(origin_acts, edited_acts, day_num, day_num)

            # 检查是否有实质性变化
            has_changes = any(ad.diff_type != "unchanged" for ad in activities_diff)

            if has_changes:
                days_modified += 1
                day_diff = DayDiff(
                    day_id=day_num,
                    diff_type="modified",
                    activities_diff=activities_diff,
                    before_activities_count=len(origin_acts),
                    after_activities_count=len(edited_acts)
                )
            else:
                day_diff = DayDiff(
                    day_id=day_num,
                    diff_type="unchanged",
                    activities_diff=activities_diff,
                    before_activities_count=len(origin_acts),
                    after_activities_count=len(edited_acts)
                )

            days_diff.append(day_diff)

            # 统计activity变化
            for ad in activities_diff:
                if ad.diff_type == "added":
                    activities_added += 1
                elif ad.diff_type == "removed":
                    activities_removed += 1
                elif ad.diff_type == "modified":
                    activities_modified += 1
                elif ad.diff_type == "reordered":
                    activities_reordered += 1

        elif origin_day and not edited_day:
            # 天被删除
            days_removed += 1
            origin_acts = origin_day.get("activities", [])

            # 所有activities标记为removed
            activities_diff = []
            for i, act in enumerate(origin_acts):
                activities_diff.append(ActivityDiff(
                    activity_id=f"day{day_num}_act{i}",
                    diff_type="removed",
                    before_data=act,
                    day_before=day_num,
                    day_after=None
                ))
                activities_removed += 1

            day_diff = DayDiff(
                day_id=day_num,
                diff_type="removed",
                activities_diff=activities_diff,
                before_activities_count=len(origin_acts),
                after_activities_count=0
            )
            days_diff.append(day_diff)

        elif not origin_day and edited_day:
            # 天被添加
            days_added += 1
            edited_acts = edited_day.get("activities", [])

            # 所有activities标记为added
            activities_diff = []
            for i, act in enumerate(edited_acts):
                activities_diff.append(ActivityDiff(
                    activity_id=f"day{day_num}_act{i}",
                    diff_type="added",
                    after_data=act,
                    day_before=None,
                    day_after=day_num
                ))
                activities_added += 1

            day_diff = DayDiff(
                day_id=day_num,
                diff_type="added",
                activities_diff=activities_diff,
                before_activities_count=0,
                after_activities_count=len(edited_acts)
            )
            days_diff.append(day_diff)

    return PlanDiff(
        origin_plan=origin_plan,
        edited_plan=edited_plan,
        days_diff=days_diff,
        days_added=days_added,
        days_removed=days_removed,
        days_modified=days_modified,
        activities_added=activities_added,
        activities_removed=activities_removed,
        activities_modified=activities_modified,
        activities_reordered=activities_reordered
    )


def _compare_activities(origin_acts: List[Dict], edited_acts: List[Dict],
                         day_before: int, day_after: int) -> List[ActivityDiff]:
    """
    比较两个activities列表

    Args:
        origin_acts: 原始activities
        edited_acts: 编辑后的activities
        day_before: 前面的天数
        day_after: 后面的天数

    Returns:
        ActivityDiff列表
    """
    activities_diff = []

    # 创建activity指纹（基于start/end位置和时间）
    origin_fingerprints = []
    for i, act in enumerate(origin_acts):
        fp = _create_activity_fingerprint(act, i)
        origin_fingerprints.append(fp)

    edited_fingerprints = []
    for i, act in enumerate(edited_acts):
        fp = _create_activity_fingerprint(act, i)
        edited_fingerprints.append(fp)

    # 匹配activities
    matched_origin = set()
    matched_edited = set()

    # 尝试通过指纹匹配
    for i, orig_fp in enumerate(origin_fingerprints):
        for j, edit_fp in enumerate(edited_fingerprints):
            if j in matched_edited:
                continue

            if _activities_match(orig_fp, edit_fp):
                # 检查是否修改
                if _activities_modified(origin_acts[i], edited_acts[j]):
                    activities_diff.append(ActivityDiff(
                        activity_id=f"day{day_before}_act{i}",
                        diff_type="modified",
                        before_data=origin_acts[i],
                        after_data=edited_acts[j],
                        day_before=day_before,
                        day_after=day_after
                    ))
                else:
                    activities_diff.append(ActivityDiff(
                        activity_id=f"day{day_before}_act{i}",
                        diff_type="unchanged",
                        before_data=origin_acts[i],
                        after_data=edited_acts[j],
                        day_before=day_before,
                        day_after=day_after
                    ))

                matched_origin.add(i)
                matched_edited.add(j)
                break

    # 未匹配的origin activities -> removed
    for i in range(len(origin_acts)):
        if i not in matched_origin:
            activities_diff.append(ActivityDiff(
                activity_id=f"day{day_before}_act{i}",
                diff_type="removed",
                before_data=origin_acts[i],
                day_before=day_before,
                day_after=None
            ))

    # 未匹配的edited activities -> added
    for j in range(len(edited_acts)):
        if j not in matched_edited:
            activities_diff.append(ActivityDiff(
                activity_id=f"day{day_after}_act{j}",
                diff_type="added",
                after_data=edited_acts[j],
                day_before=None,
                day_after=day_after
            ))

    # 检查reorder - 如果所有activities都存在但顺序不同
    if (len(matched_origin) == len(origin_acts) and
        len(matched_edited) == len(edited_acts) and
        matched_origin != set(range(len(origin_acts)))):
        # 检查是否只是顺序变化
        for ad in activities_diff:
            if ad.diff_type == "unchanged":
                ad.diff_type = "reordered"

    return activities_diff


def _create_activity_fingerprint(activity: Dict, index: int) -> Dict:
    """创建activity的指纹用于匹配"""
    return {
        "index": index,
        "start": activity.get("start", ""),
        "end": activity.get("end", ""),
        "type": activity.get("type", ""),
        "start_time": activity.get("start_time", ""),
        "end_time": activity.get("end_time", "")
    }


def _activities_match(fp1: Dict, fp2: Dict) -> bool:
    """判断两个activity指纹是否匹配（同一个activity）"""
    # 比较start和end位置
    if fp1["start"] and fp2["start"] and fp1["start"] == fp2["start"]:
        return True
    if fp1["end"] and fp2["end"] and fp1["end"] == fp2["end"]:
        return True
    return False


def _activities_modified(act1: Dict, act2: Dict) -> bool:
    """判断两个activity是否有实质性差异"""
    # 检查关键字段
    for field in ["start_time", "end_time", "price", "cost", "TrainID"]:
        val1 = act1.get(field)
        val2 = act2.get(field)
        if val1 != val2:
            return True
    return False


def extract_op_tags(diff: PlanDiff) -> List[str]:
    """
    从PlanDiff提取操作标签

    Args:
        diff: PlanDiff对象

    Returns:
        操作标签列表
    """
    op_tags = []

    # 天级别的操作
    if diff.days_added > 0:
        op_tags.append("ADD-DAY")

    if diff.days_removed > 0:
        op_tags.append("DELETE-DAY")

    if diff.days_modified > 0:
        op_tags.append("MODIFY-DAY")

    # Activity级别的操作
    if diff.activities_added > 0:
        op_tags.append("ADD")

    if diff.activities_removed > 0:
        op_tags.append("DELETE")

    if diff.activities_modified > 0:
        op_tags.append("MODIFY")

    if diff.activities_reordered > 0:
        op_tags.append("REORDER")

    # 检查跨天移动
    for day_diff in diff.days_diff:
        for act_diff in day_diff.activities_diff:
            if act_diff.diff_type == "reordered":
                if (act_diff.day_before is not None and
                    act_diff.day_after is not None and
                    act_diff.day_before != act_diff.day_after):
                    op_tags.append("MOVE-ACROSS-DAY")
                    break

    # 检查transport变化
    if _has_transport_change(diff):
        op_tags.append("CHANGE-TRANSPORT")

    return list(set(op_tags))  # 去重


def _has_transport_change(diff: PlanDiff) -> bool:
    """检查是否有transport变化"""
    # 简化实现：检查是否有train/airplane类型的activities被添加或删除
    for day_diff in diff.days_diff:
        for act_diff in day_diff.activities_diff:
            if act_diff.diff_type == "added":
                if act_diff.after_data and act_diff.after_data.get("type") in ["train", "airplane"]:
                    return True
            elif act_diff.diff_type == "removed":
                if act_diff.before_data and act_diff.before_data.get("type") in ["train", "airplane"]:
                    return True
    return False


def compare_itineraries(origin: Dict[str, Any], edited: Dict[str, Any]) -> Dict[str, Any]:
    """
    比较两个itinerary，返回汇总信息

    Args:
        origin: 原始plan
        edited: 编辑后的plan

    Returns:
        比较结果字典
    """
    origin_itinerary = origin.get("itinerary", [])
    edited_itinerary = edited.get("itinerary", [])

    return {
        "origin_days": len(origin_itinerary),
        "edited_days": len(edited_itinerary),
        "day_difference": len(edited_itinerary) - len(origin_itinerary),
        "origin_total_activities": sum(len(d.get("activities", [])) for d in origin_itinerary),
        "edited_total_activities": sum(len(d.get("activities", [])) for d in edited_itinerary),
        "activity_difference": sum(len(d.get("activities", [])) for d in edited_itinerary) -
                               sum(len(d.get("activities", [])) for d in origin_itinerary)
    }


if __name__ == "__main__":
    # 测试代码
    print("Diff Utilities - Test")

    # 示例plans
    origin_plan = {
        "people_number": 1,
        "start_city": "成都",
        "target_city": "重庆",
        "itinerary": [
            {
                "day": 1,
                "activities": [
                    {
                        "start_time": "08:02",
                        "end_time": "10:17",
                        "start": "成都东站",
                        "end": "重庆北站",
                        "TrainID": "D2208",
                        "type": "train",
                        "cost": 158.49
                    }
                ]
            }
        ]
    }

    # 添加了一天的edited plan
    edited_plan = {
        "people_number": 1,
        "start_city": "成都",
        "target_city": "重庆",
        "itinerary": [
            {
                "day": 1,
                "activities": [
                    {
                        "start_time": "08:00",
                        "end_time": "10:00",
                        "start": "成都东站",
                        "end": "重庆北站",
                        "TrainID": "D2208",
                        "type": "train",
                        "cost": 158.49
                    },
                    {
                        "start_time": "10:30",
                        "end_time": "12:30",
                        "start": "重庆北站",
                        "end": "磁器口古镇",
                        "type": "attraction",
                        "cost": 0
                    }
                ]
            },
            {
                "day": 2,
                "activities": [
                    {
                        "start_time": "14:00",
                        "end_time": "16:00",
                        "start": "重庆北站",
                        "end": "成都东站",
                        "TrainID": "D5109",
                        "type": "train",
                        "cost": 158.49
                    }
                ]
            }
        ]
    }

    # 计算diff
    diff = calculate_plan_diff(origin_plan, edited_plan)

    print(f"\nPlan Diff Summary:")
    print(f"  Days added: {diff.days_added}")
    print(f"  Days removed: {diff.days_removed}")
    print(f"  Days modified: {diff.days_modified}")
    print(f"  Activities added: {diff.activities_added}")
    print(f"  Activities removed: {diff.activities_removed}")
    print(f"  Activities modified: {diff.activities_modified}")
    print(f"  Activities reordered: {diff.activities_reordered}")

    # 提取op_tags
    op_tags = extract_op_tags(diff)
    print(f"\nOperation Tags:")
    print(f"  {op_tags}")
