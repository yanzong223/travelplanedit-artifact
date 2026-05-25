"""
POI Database Interface
封装Chinatravel tools，提供统一的POI查询接口
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Optional, Tuple

# 添加Chinatravel到路径
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CHINATRAVEL_PATH = os.path.join(BASE_DIR, "Chinatravel/ChinaTravel")
sys.path.insert(0, CHINATRAVEL_PATH)

from chinatravel.environment.tools.attractions.apis import Attractions
from chinatravel.environment.tools.accommodations.apis import Accommodations
from chinatravel.environment.tools.poi.apis import Poi
from chinatravel.environment.tools.intercity_transport.apis import IntercityTransport
from chinatravel.environment.tools.restaurants.apis import Restaurants


class POIDatabase:
    """POI数据库接口"""

    # 支持的城市列表
    SUPPORTED_CITIES = [
        "北京", "上海", "南京", "苏州", "杭州",
        "深圳", "成都", "武汉", "广州", "重庆"
    ]

    # 城市中英文映射
    CITY_MAPPING = {
        "北京": "beijing",
        "上海": "shanghai",
        "南京": "nanjing",
        "苏州": "suzhou",
        "杭州": "hangzhou",
        "深圳": "shenzhen",
        "成都": "chengdu",
        "武汉": "wuhan",
        "广州": "guangzhou",
        "重庆": "chongqing",
    }

    def __init__(self, poi_db_path: str = None):
        """
        初始化POI数据库

        Args:
            poi_db_path: POI数据库路径，如果为None则使用默认路径
        """
        self.attractions = Attractions()
        self.restaurants = Restaurants()
        self.accommodations = Accommodations()
        self.poi = Poi()
        self.transport = IntercityTransport()

    def get_supported_cities(self) -> List[str]:
        """获取支持的城市列表"""
        return self.SUPPORTED_CITIES

    def city_name_to_code(self, city: str) -> str:
        """城市名称转代码"""
        return self.CITY_MAPPING.get(city, city)

    # ========== Attractions 相关 ==========

    def get_attractions_by_city(self, city: str, type_filter: Optional[str] = None,
                                 top_k: Optional[int] = None) -> pd.DataFrame:
        """
        获取城市景点列表

        Args:
            city: 城市名称（中文）
            type_filter: 景点类型过滤（如"历史古迹", "自然风光"）
            top_k: 返回前k个景点

        Returns:
            景点DataFrame
        """
        if city not in self.SUPPORTED_CITIES:
            raise ValueError(f"城市 {city} 不在支持列表中")

        if type_filter:
            df = self.attractions.select(city, "type", lambda x: x == type_filter)
        else:
            df = self.attractions.data[city]

        if top_k:
            df = df.head(top_k)

        return df

    def get_attraction_by_name(self, city: str, name: str) -> Optional[pd.Series]:
        """
        按名称获取景点

        Args:
            city: 城市名称
            name: 景点名称

        Returns:
            景点信息Series，如果不存在返回None
        """
        result = self.attractions.select(city, "name", lambda x: x == name)
        if len(result) == 0:
            return None
        return result.iloc[0]

    def attraction_exists(self, city: str, name: str) -> bool:
        """检查景点是否存在"""
        return self.get_attraction_by_name(city, name) is not None

    def get_nearby_attractions(self, city: str, point: str,
                               top_k: int = 5, dist_km: float = 2) -> pd.DataFrame:
        """
        获取附近的景点

        Args:
            city: 城市名称
            point: 参考点名称
            top_k: 返回前k个
            dist_km: 距离限制（公里）

        Returns:
            附近的景点DataFrame
        """
        return self.attractions.nearby(city, point, topk=top_k, dist=dist_km)

    def get_attraction_types(self, city: str) -> List[str]:
        """获取城市的景点类型列表"""
        return list(self.attractions.get_type_list(city))

    def is_attraction_open(self, city: str, attraction_id: int, time: str) -> bool:
        """
        检查景点在指定时间是否开放

        Args:
            city: 城市名称
            attraction_id: 景点ID
            time: 时间字符串 "HH:MM"

        Returns:
            是否开放
        """
        return self.attractions.id_is_open(city, attraction_id, time)

    def sample_attractions(self, city: str, n: int = 1,
                          type_filter: Optional[str] = None,
                          random_state: Optional[int] = None) -> List[Dict]:
        """
        随机采样景点

        Args:
            city: 城市名称
            n: 采样数量
            type_filter: 类型过滤
            random_state: 采样随机种子（用于可复现）

        Returns:
            景点信息列表
        """
        df = self.get_attractions_by_city(city, type_filter=type_filter)
        sampled = df.sample(n=min(n, len(df)), replace=False, random_state=random_state)

        result = []
        for _, row in sampled.iterrows():
            result.append({
                "id": int(row["id"]),
                "name": row["name"],
                "type": row["type"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "opentime": row["opentime"],
                "endtime": row["endtime"],
                "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
                "recommend_min_time": float(row["recommendmintime"]) if "recommendmintime" in row else None,
                "recommend_max_time": float(row["recommendmaxtime"]) if "recommendmaxtime" in row else None,
            })
        return result

    # ========== Restaurants 相关 ==========

    def get_restaurants_by_city(
        self,
        city: str,
        cuisine_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取城市餐厅列表。"""
        if city not in self.SUPPORTED_CITIES:
            raise ValueError(f"城市 {city} 不在支持列表中")

        if cuisine_filter:
            df = self.restaurants.select(city, "cuisine", lambda x: x == cuisine_filter)
        else:
            df = self.restaurants.select(city, "name", lambda x: True)

        if top_k:
            df = df.head(top_k)
        return df

    def get_restaurant_by_name(self, city: str, name: str) -> Optional[pd.Series]:
        """按名称获取餐厅。"""
        result = self.restaurants.select(city, "name", lambda x: x == name)
        if len(result) == 0:
            return None
        return result.iloc[0]

    def get_nearby_restaurants(
        self,
        city: str,
        point: str,
        top_k: int = 5,
        dist_km: float = 2,
    ) -> pd.DataFrame:
        """获取参考点附近的餐厅。"""
        return self.restaurants.nearby(city, point, topk=top_k, dist=dist_km)

    def sample_restaurants(
        self,
        city: str,
        n: int = 1,
        cuisine_filter: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> List[Dict]:
        """随机采样餐厅。"""
        df = self.get_restaurants_by_city(city, cuisine_filter=cuisine_filter)
        sampled = df.sample(n=min(n, len(df)), replace=False, random_state=random_state)

        result = []
        for _, row in sampled.iterrows():
            result.append({
                "id": int(row["id"]),
                "name": row["name"],
                "type": "restaurant",
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
                "cuisine": row.get("cuisine"),
                "opentime": row["opentime"],
                "endtime": row["endtime"],
                "recommendedfood": row.get("recommendedfood"),
            })
        return result

    # ========== Accommodations 相关 ==========

    def get_accommodations_by_city(
        self,
        city: str,
        feature_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取城市住宿列表。"""
        if city not in self.SUPPORTED_CITIES:
            raise ValueError(f"城市 {city} 不在支持列表中")

        if feature_filter:
            df = self.accommodations.select(city, "featurehoteltype", lambda x: x == feature_filter)
        else:
            df = self.accommodations.select(city, "name", lambda x: True)

        if top_k:
            df = df.head(top_k)
        return df

    def get_accommodation_by_name(self, city: str, name: str) -> Optional[pd.Series]:
        """按名称获取住宿。"""
        result = self.accommodations.select(city, "name", lambda x: x == name)
        if len(result) == 0:
            return None
        return result.iloc[0]

    def accommodation_exists(self, city: str, name: str) -> bool:
        """检查住宿是否存在。"""
        return self.get_accommodation_by_name(city, name) is not None

    def sample_accommodations(
        self,
        city: str,
        n: int = 1,
        feature_filter: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> List[Dict]:
        """随机采样住宿。"""
        df = self.get_accommodations_by_city(city, feature_filter=feature_filter)
        sampled = df.sample(n=min(n, len(df)), replace=False, random_state=random_state)

        result = []
        for _, row in sampled.iterrows():
            result.append({
                "id": int(row["id"]),
                "name": row["name"],
                "type": "accommodation",
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
                "featurehoteltype": row.get("featurehoteltype"),
                "rooms": int(row["rooms"]) if "rooms" in row and pd.notna(row["rooms"]) else None,
                "room_type": int(row["room_type"]) if "room_type" in row and pd.notna(row["room_type"]) else None,
            })
        return result

    # ========== POI 相关 ==========

    def search_poi(self, city: str, name: str) -> Optional[Tuple[float, float]]:
        """
        搜索POI坐标

        Args:
            city: 城市名称
            name: POI名称

        Returns:
            (lat, lon)元组，如果不存在返回None
        """
        result = self.poi.search(city, name)
        if isinstance(result, str):  # 错误消息
            return None
        return result

    def poi_exists(self, city: str, name: str) -> bool:
        """检查POI是否存在"""
        return self.search_poi(city, name) is not None

    # ========== Intercity Transport 相关 ==========

    def get_trains(self, start_city: str, end_city: str,
                   earliest_leave_time: str = "00:00") -> pd.DataFrame:
        """
        获取两城市之间的火车班次

        Args:
            start_city: 出发城市
            end_city: 目的城市
            earliest_leave_time: 最早出发时间

        Returns:
            火车班次DataFrame
        """
        return self.transport.select(start_city, end_city, "train", earliest_leave_time)

    def get_flights(self, start_city: str, end_city: str,
                   earliest_leave_time: str = "00:00") -> pd.DataFrame:
        """
        获取两城市之间的航班班次

        Args:
            start_city: 出发城市
            end_city: 目的城市
            earliest_leave_time: 最早出发时间

        Returns:
            航班班次DataFrame
        """
        return self.transport.select(start_city, end_city, "airplane", earliest_leave_time)

    def has_transport(self, start_city: str, end_city: str,
                     transport_type: str = "train") -> bool:
        """检查两城市之间是否有交通连接"""
        df = self.transport.select(start_city, end_city, transport_type)
        return df is not None and len(df) > 0


# 便捷函数
def get_poi_db(poi_db_path: str = None) -> POIDatabase:
    """获取POI数据库实例"""
    return POIDatabase(poi_db_path)


if __name__ == "__main__":
    # 测试代码
    db = POIDatabase()

    print("支持的城市:", db.get_supported_cities())
    print()

    # 测试景点查询
    print("=== 重庆景点 ===")
    attractions = db.get_attractions_by_city("重庆", top_k=5)
    print(attractions[["name", "type", "price"]])
    print()

    # 测试景点类型
    print("=== 重庆景点类型 ===")
    print(db.get_attraction_types("重庆"))
    print()

    # 测试随机采样
    print("=== 随机采样3个重庆景点 ===")
    sampled = db.sample_attractions("重庆", n=3)
    for poi in sampled:
        print(f"- {poi['name']} ({poi['type']})")
    print()

    # 测试POI搜索
    print("=== 搜索POI ===")
    coord = db.search_poi("重庆", "重庆北站")
    print(f"重庆北站坐标: {coord}")
    print()

    # 测试交通查询
    print("=== 重庆到成都的火车 ===")
    trains = db.get_trains("重庆", "成都", earliest_leave_time="08:00")
    if trains is not None and len(trains) > 0:
        print(trains[["TrainID", "BeginTime", "EndTime"]].head(5))
    else:
        print("无火车班次")
