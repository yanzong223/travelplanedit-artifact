"""
Plan Parser
解析origin plan JSON文件，提取结构化信息
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class OriginPlanParser:
    """Origin plan解析器"""

    def __init__(self, plan_json: Dict[str, Any]):
        """
        初始化

        Args:
            plan_json: origin plan JSON字典
        """
        self.plan = plan_json

    def get_people_number(self) -> int:
        """获取人数"""
        return self.plan.get("people_number", 1)

    def get_start_city(self) -> str:
        """获取出发城市"""
        return self.plan.get("start_city", "")

    def get_target_city(self) -> str:
        """获取目标城市"""
        return self.plan.get("target_city", "")

    def get_itinerary(self) -> List[Dict]:
        """获取行程列表"""
        return self.plan.get("itinerary", [])

    def get_day_count(self) -> int:
        """获取行程天数"""
        return len(self.get_itinerary())

    def is_single_day(self) -> bool:
        """判断是否为单日行程"""
        return self.get_day_count() == 1

    def is_multi_day(self) -> bool:
        """判断是否为多日行程"""
        return self.get_day_count() > 1

    def get_day_activities(self, day: int) -> List[Dict]:
        """
        获取指定天的活动列表

        Args:
            day: 天数（从1开始）

        Returns:
            活动列表
        """
        itinerary = self.get_itinerary()
        for day_plan in itinerary:
            if day_plan.get("day") == day:
                return day_plan.get("activities", [])
        return []

    def get_all_cities(self) -> List[str]:
        """获取行程中涉及的所有城市"""
        cities = set()
        start_city = self.get_start_city()
        target_city = self.get_target_city()

        if start_city:
            cities.add(start_city)
        if target_city:
            cities.add(target_city)

        # 从activities中提取城市（如果有）
        itinerary = self.get_itinerary()
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                if "start" in activity:
                    cities.add(activity["start"])
                if "end" in activity:
                    cities.add(activity["end"])

        return list(cities)

    def get_activities_by_type(self, activity_type: str) -> List[Dict]:
        """
        按类型获取活动

        Args:
            activity_type: 活动类型（如"train", "attraction", "restaurant"等）

        Returns:
            活动列表
        """
        result = []
        itinerary = self.get_itinerary()
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                if activity.get("type") == activity_type:
                    result.append(activity)
        return result

    def get_total_cost(self) -> float:
        """计算总费用"""
        total = 0.0
        itinerary = self.get_itinerary()
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                cost = activity.get("cost", 0)
                if isinstance(cost, (int, float)):
                    total += cost
        return total

    def get_total_time_range(self) -> tuple:
        """
        获取整个行程的时间范围

        Returns:
            (start_time, end_time) 元组
        """
        itinerary = self.get_itinerary()
        if not itinerary:
            return (None, None)

        first_day = itinerary[0]
        last_day = itinerary[-1]

        first_activities = first_day.get("activities", [])
        last_activities = last_day.get("activities", [])

        if not first_activities or not last_activities:
            return (None, None)

        start_time = first_activities[0].get("start_time")
        end_time = last_activities[-1].get("end_time")

        return (start_time, end_time)

    def get_day_time_range(self, day: int) -> tuple:
        """
        获取指定天的时间范围

        Args:
            day: 天数（从1开始）

        Returns:
            (start_time, end_time) 元组
        """
        activities = self.get_day_activities(day)
        if not activities:
            return (None, None)

        start_time = activities[0].get("start_time")
        end_time = activities[-1].get("end_time")

        return (start_time, end_time)

    def has_attraction(self, attraction_name: str) -> bool:
        """
        检查行程中是否包含指定景点

        Args:
            attraction_name: 景点名称

        Returns:
            是否包含
        """
        itinerary = self.get_itinerary()
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                # 检查start或end字段
                for field in ["start", "end"]:
                    if field in activity and attraction_name in activity[field]:
                        return True
        return False

    def get_attractions(self) -> List[str]:
        """获取行程中的所有景点名称"""
        attractions = set()
        itinerary = self.get_itinerary()
        for day_plan in itinerary:
            for activity in day_plan.get("activities", []):
                # 假设景点类型不为train时，可能是景点
                if activity.get("type") != "train":
                    if "start" in activity:
                        attractions.add(activity["start"])
                    if "end" in activity:
                        attractions.add(activity["end"])
        return list(attractions)

    def to_summary(self) -> Dict[str, Any]:
        """
        生成plan的摘要信息

        Returns:
            摘要字典
        """
        return {
            "people_number": self.get_people_number(),
            "start_city": self.get_start_city(),
            "target_city": self.get_target_city(),
            "day_count": self.get_day_count(),
            "is_single_day": self.is_single_day(),
            "cities": self.get_all_cities(),
            "attractions": self.get_attractions(),
            "total_cost": self.get_total_cost(),
            "time_range": self.get_total_time_range(),
        }

    def get_plan_for_edit(self) -> Dict[str, Any]:
        """
        获取用于编辑的plan结构（去除metadata）

        Returns:
            清理后的plan字典
        """
        return {
            "people_number": self.get_people_number(),
            "start_city": self.get_start_city(),
            "target_city": self.get_target_city(),
            "itinerary": self.get_itinerary(),
        }


def load_origin_plan(file_path: str) -> OriginPlanParser:
    """
    从文件加载origin plan

    Args:
        file_path: JSON文件路径

    Returns:
        OriginPlanParser实例
    """
    with open(file_path, "r", encoding="utf-8") as f:
        plan_json = json.load(f)
    return OriginPlanParser(plan_json)


def parse_origin_plan(plan_json: Dict[str, Any]) -> OriginPlanParser:
    """
    从JSON字典解析origin plan

    Args:
        plan_json: plan JSON字典

    Returns:
        OriginPlanParser实例
    """
    return OriginPlanParser(plan_json)


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        # 从命令行读取文件路径
        file_path = sys.argv[1]
        parser = load_origin_plan(file_path)
    else:
        # 使用示例数据
        example_plan = {
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
                            "price": 158.49,
                            "cost": 158.49,
                            "TrainID": "D2208",
                            "type": "train"
                        },
                        {
                            "start_time": "10:55",
                            "end_time": "13:24",
                            "start": "重庆北站",
                            "end": "成都东站",
                            "price": 158.49,
                            "cost": 158.49,
                            "TrainID": "D5109",
                            "type": "train"
                        }
                    ]
                }
            ]
        }
        parser = parse_origin_plan(example_plan)

    print("=== Plan Summary ===")
    summary = parser.to_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Day 1 Activities ===")
    activities = parser.get_day_activities(1)
    for i, activity in enumerate(activities, 1):
        print(f"{i}. {activity.get('start_time')} - {activity.get('end_time')}: "
              f"{activity.get('start')} -> {activity.get('end')} "
              f"({activity.get('type')})")
