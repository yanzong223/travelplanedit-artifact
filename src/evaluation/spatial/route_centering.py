"""
Route Centeredness Evaluation Module

Provides universal route distance centeredness evaluation functions
for travel plan spatial analysis. This module implements algorithms
to measure how well activities are organized around a central point
based on route distances rather than direct spatial distances.

The main function compute_route_centeredness is completely generic
and can be used by any scenario that needs to evaluate spatial
organization around a specific POI or activity.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_route_centeredness(
    day_activities: List[Dict[str, Any]],
    center_index: int,
) -> Dict[str, Any]:
    """
    通用路径距离围绕度评估函数，可用于多个编辑场景。

    计算行程活动序列围绕某个中心点的组织程度，基于路径距离而非直线距离。
    越高的分数表示活动越围绕中心点组织。

    Args:
        day_activities: 某一天的活动列表，顺序为行程顺序。
                       每个活动应包含transports字段用于计算路径距离。
        center_index: 作为中心点的活动索引（如 must_visit_reorder 的目标 POI）。

    Returns:
        Dict[str, Any]: 包含以下字段：
            - route_src: float - [0,1] 空间围绕度（路径距离版）
            - distances_to_center: List[float] - 每个活动到中心活动的路径距离
            - max_route_distance: float - 用于归一化的尺度 (D_norm)
            - valid: bool - 是否成功计算（如只有一个活动则 SRC=1）

    算法说明:
        1. 对每个活动i，计算到中心活动k的路径距离
        2. 路径距离 = 沿活动序列的所有transports.distance之和
        3. 归一化: route_src = 1.0 - mean(distances[i] / max_distance)
        4. 异常情况处理: 空活动、索引越界、缺失transports等
    """
    try:
        # 输入验证
        if not day_activities or not isinstance(day_activities, list):
            logger.warning("Invalid day_activities: must be non-empty list")
            return {
                "route_src": None,
                "distances_to_center": [],
                "max_route_distance": 0.0,
                "valid": False,
                "error": "Invalid day_activities input"
            }

        if center_index < 0 or center_index >= len(day_activities):
            logger.warning(f"Invalid center_index: {center_index} for {len(day_activities)} activities")
            return {
                "route_src": None,
                "distances_to_center": [],
                "max_route_distance": 0.0,
                "valid": False,
                "error": f"center_index {center_index} out of range"
            }

        # 只有一个活动的情况 - 完美围绕
        if len(day_activities) == 1:
            logger.debug("Single activity day: perfect centeredness (SRC=1.0)")
            return {
                "route_src": 1.0,
                "distances_to_center": [0.0],
                "max_route_distance": 0.0,
                "valid": True
            }

        # 计算每个活动到中心活动的路径距离
        distances_to_center = []

        for i, activity in enumerate(day_activities):
            distance = _calculate_route_distance_to_center(day_activities, i, center_index)
            distances_to_center.append(distance)

        # 计算归一化尺度
        max_distance = max(distances_to_center) if distances_to_center else 0.0

        # 如果最大距离为0，所有活动都在同一位置 - 完美围绕
        if max_distance == 0.0:
            logger.debug("All distances zero: perfect centeredness (SRC=1.0)")
            return {
                "route_src": 1.0,
                "distances_to_center": distances_to_center,
                "max_route_distance": 0.0,
                "valid": True
            }

        # 计算围绕度分数 (SRC)
        # 排除中心活动本身（距离为0）避免影响平均值
        non_center_distances = [
            distances_to_center[i] for i in range(len(distances_to_center))
            if i != center_index
        ]

        if non_center_distances:
            mean_normalized_distance = sum(
                distance / max_distance for distance in non_center_distances
            ) / len(non_center_distances)
            route_src = 1.0 - mean_normalized_distance
        else:
            # 只有中心活动的情况
            route_src = 1.0

        # 确保结果在[0,1]范围内
        route_src = max(0.0, min(1.0, route_src))

        return {
            "route_src": route_src,
            "distances_to_center": distances_to_center,
            "max_route_distance": max_distance,
            "valid": True
        }

    except Exception as e:
        logger.error(f"Route centeredness calculation failed: {e}")
        return {
            "route_src": None,
            "distances_to_center": [],
            "max_route_distance": 0.0,
            "valid": False,
            "error": str(e)
        }


def _calculate_route_distance_to_center(
    day_activities: List[Dict[str, Any]],
    from_index: int,
    center_index: int
) -> float:
    """
    计算从活动from_index到中心活动center_index的路径距离。

    路径距离定义为沿活动序列的所有transports.distance之和：
    - 如果 from_index == center_index: 距离 = 0
    - 如果 from_index < center_index: 距离 = sum(从from_index+1到center_index的所有transports距离)
    - 如果 from_index > center_index: 距离 = sum(从center_index+1到from_index的所有transports距离)

    Args:
        day_activities: 活动列表
        from_index: 起始活动索引
        center_index: 中心活动索引

    Returns:
        float: 路径距离（公里）
    """
    if from_index == center_index:
        return 0.0

    if from_index < center_index:
        # 计算从from_index+1到center_index的路径距离
        start_idx = from_index
        end_idx = center_index
        direction = "forward"
    else:
        # 计算从center_index+1到from_index的路径距离
        start_idx = center_index
        end_idx = from_index
        direction = "backward"

    total_distance = 0.0

    # 沿活动序列累加transports距离
    for i in range(start_idx, end_idx):
        current_activity = day_activities[i]
        next_activity = day_activities[i + 1]

        # 从当前活动到下一个活动的transports距离
        segment_distance = _get_transport_distance_between_activities(
            current_activity, next_activity
        )
        total_distance += segment_distance

        logger.debug(f"Route segment {i}→{i+1} ({direction}): {segment_distance:.2f}km, cumulative: {total_distance:.2f}km")

    return total_distance


def _get_transport_distance_between_activities(
    from_activity: Dict[str, Any],
    to_activity: Dict[str, Any]
) -> float:
    """
    获取两个活动之间的transports距离。

    从from_activity的transports字段中提取到to_activity的路径距离。

    Args:
        from_activity: 起始活动
        to_activity: 目标活动

    Returns:
        float: 运输距离（公里），如果没有transports信息则返回0.0
    """
    try:
        # 从起始活动获取transports信息
        transports = from_activity.get("transports", [])

        if not transports:
            logger.debug(f"No transports found for activity {from_activity.get('type', 'unknown')}")
            return 0.0

        # 累加所有transport legs的距离
        total_distance = 0.0
        for leg in transports:
            if isinstance(leg, dict):
                distance = leg.get("distance", 0.0)
                # 确保distance是数值类型
                if isinstance(distance, (int, float)):
                    total_distance += float(distance)
                else:
                    # 尝试转换字符串类型的距离
                    try:
                        total_distance += float(distance)
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid distance value: {distance}, treating as 0")
                        total_distance += 0.0
            else:
                logger.debug(f"Invalid transport leg format: {type(leg)}")

        logger.debug(f"Transport distance sum: {total_distance:.2f}km from {len(transports)} legs")
        return total_distance

    except Exception as e:
        logger.warning(f"Error extracting transport distance: {e}")
        return 0.0


def validate_activity_sequence(day_activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证活动序列的完整性和有效性。

    Args:
        day_activities: 活动列表

    Returns:
        Dict[str, Any]: 验证结果，包含is_valid、errors、warnings等字段
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {
            "total_activities": len(day_activities),
            "activities_with_transports": 0,
            "activities_with_distances": 0,
            "total_transports": 0
        }
    }

    try:
        if not day_activities:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Empty activities list")
            return validation_result

        for i, activity in enumerate(day_activities):
            if not isinstance(activity, dict):
                validation_result["errors"].append(f"Activity {i} is not a dictionary")
                validation_result["is_valid"] = False
                continue

            transports = activity.get("transports", [])
            if transports:
                validation_result["statistics"]["activities_with_transports"] += 1
                validation_result["statistics"]["total_transports"] += len(transports)

                # 检查transports中的distance字段
                has_valid_distance = False
                for leg in transports:
                    if isinstance(leg, dict) and leg.get("distance") is not None:
                        try:
                            float(leg["distance"])
                            has_valid_distance = True
                            break
                        except (ValueError, TypeError):
                            pass

                if has_valid_distance:
                    validation_result["statistics"]["activities_with_distances"] += 1
                else:
                    validation_result["warnings"].append(
                        f"Activity {i} has transports but no valid distance information"
                    )

        # 如果没有找到任何有效的distance信息，发出警告
        if validation_result["statistics"]["activities_with_distances"] == 0:
            validation_result["warnings"].append(
                "No valid distance information found in any activity transports"
            )

    except Exception as e:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Validation failed: {str(e)}")

    return validation_result