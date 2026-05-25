# Spatial Evaluation Module

This module provides spatial analysis functions for travel plan evaluation, focusing on route centeredness and spatial organization metrics.

## 主要功能

### Route Centeredness (`compute_route_centeredness`)

衡量行程活动是否围绕某个中心点（POI）组织的通用评估函数。

#### 使用示例

```python
from evaluation.spatial import compute_route_centeredness

# 获取某一天的活动列表
day_activities = plan["itinerary"][0]["activities"]

# 假设我们要评估是否围绕第二个活动（索引1）组织
center_index = 1

# 计算围绕度
result = compute_route_centeredness(day_activities, center_index)

# 结果解读
if result["valid"]:
    print(f"Route SRC: {result['route_src']:.3f}")  # [0,1]，越接近1越好
    print(f"Max Distance: {result['max_route_distance']:.2f}km")
    print(f"Distance to center for each activity: {result['distances_to_center']}")
else:
    print(f"Calculation failed: {result.get('error', 'Unknown error')}")
```

#### 在评估场景中的应用

```python
# must_visit_reorder 场景示例
def evaluate_must_visit_reorder(edited_plan, target_poi_index):
    # 找到目标POI所在的位置
    day_activities = edited_plan["itinerary"][0]["activities"]

    # 计算围绕度
    centeredness = compute_route_centeredness(day_activities, target_poi_index)

    if centeredness["valid"]:
        return {
            "route_src": centeredness["route_src"],
            "spatial_organization": "excellent" if centeredness["route_src"] > 0.8 else
                                   "good" if centeredness["route_src"] > 0.6 else
                                   "poor" if centeredness["route_src"] > 0.4 else "very_poor"
        }
    else:
        return {"error": "Failed to calculate route centeredness"}

# time_rush_style 场景示例
def evaluate_time_efficiency(edited_plan):
    day_activities = edited_plan["itinerary"][0]["activities"]

    # 评估每个活动作为中心时的围绕度，找到最佳组织点
    best_src = 0.0
    best_center = 0

    for i, activity in enumerate(day_activities):
        if activity.get("type") == "attraction":  # 只考虑景点作为中心
            result = compute_route_centeredness(day_activities, i)
            if result["valid"] and result["route_src"] > best_src:
                best_src = result["route_src"]
                best_center = i

    return {
        "optimal_centeredness": best_src,
        "optimal_center_index": best_center,
        "efficiency_score": "high" if best_src > 0.7 else "medium" if best_src > 0.4 else "low"
    }
```

## 算法说明

### 路径距离定义

- **i == k**: 距离 = 0 (中心活动自身)
- **i < k**: 距离 = sum(从活动 i+1 到 k 的所有 transports.distance)
- **i > k**: 距离 = sum(从活动 k+1 到 i 的所有 transports.distance)

### 围绕度计算 (SRC)

```
D_norm = max(distances_to_center)
route_src = 1.0 - mean(distances[i] / D_norm for i != center_index)
```

### 结果解释

- **SRC ≈ 1.0**: 完美围绕，所有活动都紧密围绕中心点组织
- **SRC ≈ 0.8-0.6**: 良好组织，大部分活动围绕中心点
- **SRC ≈ 0.4-0.2**: 组织一般，活动分散，有一定围绕性
- **SRC ≈ 0.0**: 无围绕性，活动完全分散

## 数据格式要求

活动对象应包含以下字段：

```python
{
    "start_time": "09:00",
    "end_time": "10:30",
    "type": "attraction|restaurant|transport|...",
    "position": "POI Name",  # 或 start/end 字段
    "transports": [          # 关键字段：用于计算路径距离
        {
            "start": "From Location",
            "end": "To Location",
            "mode": "walk|metro|taxi|...",
            "distance": 2.5,    # 公里数，必需字段
            "cost": 4
        }
        # ... 更多 transport legs
    ]
}
```

## 错误处理

函数包含完整的错误处理机制：

- **空活动列表**: 返回 `valid: False`
- **无效中心索引**: 返回 `valid: False`
- **缺失 transports**: 距离按 0 处理
- **缺失 distance 字段**: 距离按 0 处理
- **单个活动**: 返回完美的 SRC = 1.0