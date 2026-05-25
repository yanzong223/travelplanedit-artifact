"""Shared tool contracts for standalone edit baselines."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from pydantic import ValidationError

from edit_framework.runtime_tools.error_utils import (
    key_error_to_tool_error,
    unexpected_tool_error,
    validation_error_to_tool_error,
    value_error_to_tool_error,
)
from edit_framework.runtime_tools.types import ExposureMode
from edit_framework.tool_profiles import (
    TOOL_PROFILE_DB_READ_TYPED,
    resolve_tool_profile,
)

try:
    from pandas import DataFrame
except Exception:  # pragma: no cover - pandas is available in runtime/test env
    DataFrame = None  # type: ignore[assignment]


class ChinaTravelToolAdapter:
    """Stable tool contract over the ChinaTravel world env."""

    _KNOWN_SELECT_KEYS = {
        "name",
        "type",
        "price",
        "cuisine",
        "numbed",
        "featurehoteltype",
        "recommendedfood",
        "opentime",
        "endtime",
        "recommendmintime",
        "recommendmaxtime",
        "rating",
    }
    _SELECT_KEY_ALIASES = {
        "accommodations_select": {
            "room_type": "numbed",
            "roomtype": "numbed",
            "num_bed": "numbed",
            "num_beds": "numbed",
            "beds": "numbed",
            "bed_count": "numbed",
            "hotel_type": "featurehoteltype",
            "feature_hotel_type": "featurehoteltype",
        },
        "restaurants_select": {
            "recommended_food": "recommendedfood",
            "open_time": "opentime",
            "close_time": "endtime",
        },
        "attractions_select": {
            "open_time": "opentime",
            "close_time": "endtime",
            "ticket_price": "price",
            "recommend_min_time": "recommendmintime",
            "recommend_max_time": "recommendmaxtime",
        },
    }
    _PARAM_ORDERS = {
        "attractions_keys": ["city"],
        "attractions_select": ["city", "key", "func_str"],
        "attractions_types": ["city"],
        "restaurants_select": ["city", "key", "func_str"],
        "restaurants_keys": ["city"],
        "restaurants_cuisine": ["city"],
        "restaurants_with_recommended_food": ["city", "food"],
        "accommodations_keys": ["city"],
        "accommodations_select": ["city", "key", "func_str"],
        "attractions_nearby": ["city", "point", "topk", "dist"],
        "accommodations_nearby": ["city", "point", "topk", "dist"],
        "restaurants_nearby": ["city", "point", "topk", "dist"],
        "poi_lat_lon_search": ["city", "name"],
        "goto": ["city", "start", "end", "start_time", "transport_type"],
        "next_page": ["cursor_id"],
        "intercity_transport_select": [
            "start_city",
            "end_city",
            "intercity_type",
            "start_time",
        ],
        "attractions_id_is_open": ["city", "id", "time"],
        "restaurants_id_is_open": ["city", "id", "time"],
    }

    def __init__(
        self,
        *,
        framework_name: str = "react",
        exposure_mode: str | None = None,
        tool_profile: str | None = None,
        enable_ct_atoms: bool | None = None,
        enable_ct_verify: bool | None = None,
        enable_ct_conflict_lift: bool | None = None,
        enable_ct_notepad: bool | None = None,
        semantic_tool_allowlist: List[str] | None = None,
    ) -> None:
        self.framework_name = framework_name
        resolved = resolve_tool_profile(tool_profile=tool_profile)
        self.tool_profile = resolved.tool_profile
        self.db_read_enabled = resolved.db_read_enabled
        self.typed_read_enabled = resolved.tool_profile == TOOL_PROFILE_DB_READ_TYPED
        self.enable_ct_atoms = resolved.enable_ct_atoms
        self.enable_ct_verify = resolved.enable_ct_verify
        self.enable_ct_conflict_lift = resolved.enable_ct_conflict_lift
        self.enable_ct_notepad = resolved.enable_ct_notepad
        if exposure_mode is None:
            resolved_mode = ExposureMode.PRIMITIVE_ONLY
        else:
            resolved_mode = ExposureMode(exposure_mode)
        self.exposure_mode = resolved_mode
        self.semantic_tool_allowlist = None
        self.enabled_semantic_tools: list[str] = []

    def tool_flags(self) -> Dict[str, Any]:
        return {
            "framework_name": self.framework_name,
            "enable_ct_atoms": self.enable_ct_atoms,
            "enable_ct_verify": self.enable_ct_verify,
            "enable_ct_conflict_lift": self.enable_ct_conflict_lift,
            "enable_ct_notepad": self.enable_ct_notepad,
            "db_read_enabled": self.db_read_enabled,
            "typed_read_enabled": self.typed_read_enabled,
            "tool_profile": self.tool_profile,
            "exposure_mode": self.exposure_mode.value,
            "semantic_tool_allowlist": self.semantic_tool_allowlist,
            "enabled_semantic_tools": self.enabled_semantic_tools,
        }

    def local_tools(self) -> List[Dict[str, Any]]:
        return []

    def semantic_tools(self) -> List[Dict[str, Any]]:
        return []

    def available_tools(self) -> List[Dict[str, Any]]:
        return self.read_only_tools() if self.db_read_enabled else []

    def is_local_tool(self, tool_name: str) -> bool:
        return tool_name in self.typed_tool_names()

    def execute_local_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        runtime: Any,
    ) -> Dict[str, Any] | None:
        try:
            typed_result = self.execute_typed_tool(tool_name, args, runtime)
            if typed_result is not None:
                return typed_result
            return None
        except ValidationError as exc:
            return validation_error_to_tool_error(tool_name=tool_name, tool_args=args, exc=exc)
        except KeyError as exc:
            return key_error_to_tool_error(tool_name=tool_name, tool_args=args, exc=exc)
        except ValueError as exc:
            return value_error_to_tool_error(tool_name=tool_name, tool_args=args, exc=exc)
        except Exception as exc:
            return unexpected_tool_error(tool_name=tool_name, tool_args=args, exc=exc)

    def read_only_tools(self) -> List[Dict[str, Any]]:
        """Tool schema shared by ReAct and PTE-R query phases."""

        if self.typed_read_enabled:
            return self.typed_read_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": "attractions_keys",
                    "description": "查询城市景点表可用字段，返回 (key, type) 列表，适合在调用 attractions_select 前先了解可筛选字段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attractions_select",
                    "description": "查询城市景点。key 必须是字段名，例如 name/type/price；func_str 必须是 lambda 字符串，例如 lambda x: True 或 lambda x: x == '博物馆'。返回行包含字段：id, name, type（景点类型）, lat, lon, opentime（开始营业时间）, endtime（结束营业时间）, price（门票价格）, recommendmintime（建议最少游玩小时）, recommendmaxtime（建议最多游玩小时）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "key": {"type": "string"},
                            "func_str": {"type": "string"},
                        },
                        "required": ["city", "key", "func_str"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attractions_types",
                    "description": "查询城市景点的可用类型列表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "accommodations_keys",
                    "description": "查询城市住宿表可用字段，返回 (key, type) 列表，适合在调用 accommodations_select 前先了解可筛选字段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_select",
                    "description": "查询城市餐厅。key 必须是字段名，例如 name/cuisine/price；func_str 必须是 lambda 字符串。返回行包含字段：id, name, lat, lon, price（人均消费）, cuisine（菜系）, opentime（开始营业时间）, endtime（结束营业时间）, recommendedfood（推荐菜）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "key": {"type": "string"},
                            "func_str": {"type": "string"},
                        },
                        "required": ["city", "key", "func_str"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_keys",
                    "description": "查询城市餐厅表可用字段，返回 (key, type) 列表，适合在调用 restaurants_select 前先了解可筛选字段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_cuisine",
                    "description": "查询城市餐厅支持的菜系列表。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_with_recommended_food",
                    "description": "按推荐菜名称查询餐厅，例如想吃火锅或烤鸭时使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "food": {"type": "string"},
                        },
                        "required": ["city", "food"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "accommodations_select",
                    "description": "查询城市住宿。key 必须是字段名，例如 name/price/numbed/featurehoteltype；func_str 必须是 lambda 字符串。返回行包含字段：id, name, hotelname_en, featurehoteltype（酒店特色类型，如'舒适型''管家服务'等）, lat, lon, price（房价）, numbed（床位数）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "key": {"type": "string"},
                            "func_str": {"type": "string"},
                        },
                        "required": ["city", "key", "func_str"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attractions_nearby",
                    "description": "查询指定 POI 附近的景点。返回行包含字段：name, type, lat, lon, price, opentime, endtime, distance（与查询点的距离，km）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "point": {"type": "string"},
                            "topk": {"type": "integer"},
                            "dist": {"type": "number"},
                        },
                        "required": ["city", "point"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "accommodations_nearby",
                    "description": "查询指定 POI 附近的住宿。返回行包含字段：name, lat, lon, featurehoteltype, price, numbed, distance（与查询点的距离，km）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "point": {"type": "string"},
                            "topk": {"type": "integer"},
                            "dist": {"type": "number"},
                        },
                        "required": ["city", "point"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_nearby",
                    "description": "查询指定 POI 附近的餐厅。返回行包含字段：name, lat, lon, price, cuisine, opentime, endtime, distance（与查询点的距离，km）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "point": {"type": "string"},
                            "topk": {"type": "integer"},
                            "dist": {"type": "number"},
                        },
                        "required": ["city", "point"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "poi_lat_lon_search",
                    "description": "查询指定 POI 的经纬度坐标。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "name": {"type": "string"},
                        },
                        "required": ["city", "name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "goto",
                    "description": "查询同城两点之间的路线、时间和费用。只支持 walk/metro/taxi，返回结构化 JSON rows。返回行包含字段：mode（交通方式：walk/metro/taxi）, duration（耗时，分钟）, cost（费用）, distance（距离，km）, start_time, end_time。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                            "start_time": {"type": "string"},
                            "transport_type": {
                                "type": "string",
                                "enum": ["walk", "metro", "taxi"],
                            },
                        },
                        "required": [
                            "city",
                            "start",
                            "end",
                            "start_time",
                            "transport_type",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "next_page",
                    "description": "根据上一轮查询返回的 cursor_id 获取下一页结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cursor_id": {"type": "string"},
                        },
                        "required": ["cursor_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "intercity_transport_select",
                    "description": "查询两座城市之间的城际交通，必须用于 train/airplane。返回结构化 JSON rows，包含合法 TrainID 或 FlightID。返回行包含字段：TrainID 或 FlightID, start（出发站/机场）, end（到达站/机场）, start_time, end_time, price（票价）；同时保留原始字段 From/To/BeginTime/EndTime/Cost。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_city": {"type": "string"},
                            "end_city": {"type": "string"},
                            "intercity_type": {
                                "type": "string",
                                "enum": ["train", "airplane"],
                            },
                            "start_time": {"type": "string"},
                        },
                        "required": ["start_city", "end_city", "intercity_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attractions_id_is_open",
                    "description": "检查景点在指定时间是否开放，返回布尔值。attractions_select 返回行中的 opentime 和 endtime 字段表示景点的完整营业时间范围。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "id": {"type": "integer"},
                            "time": {"type": "string"},
                        },
                        "required": ["city", "id", "time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restaurants_id_is_open",
                    "description": "检查餐厅在指定时间是否营业，返回布尔值。restaurants_select 返回行中的 opentime 和 endtime 字段表示餐厅的完整营业时间范围。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "id": {"type": "integer"},
                            "time": {"type": "string"},
                        },
                        "required": ["city", "id", "time"],
                    },
                },
            },
        ]

    @staticmethod
    def typed_tool_names() -> set[str]:
        return {
            "search_pois",
            "search_restaurants",
            "search_hotels_by_feature",
            "search_hotels_by_budget",
            "route_between",
            "intercity_options",
            "check_open_status",
        }

    def typed_read_tools(self) -> List[Dict[str, Any]]:
        """LLM-friendly typed read tools aligned with feasibility-first data access."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "search_pois",
                    "description": (
                        "Typed ChinaTravel POI search for attractions. Use instead of key/lambda select. "
                        "Returns canonical rows with id, name, category='attraction', city, type, lat, lon, "
                        "opentime, endtime, price, recommendmintime, recommendmaxtime, source_tool. "
                        "Final plan activity.position must exactly copy rows[i].name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "name": {"type": "string", "description": "Exact attraction name when known."},
                            "poi_type": {"type": "string", "description": "Attraction type filter, e.g. 博物馆."},
                            "topk": {"type": "integer", "description": "Optional maximum rows returned."},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_restaurants",
                    "description": (
                        "Typed ChinaTravel restaurant search. Returns canonical rows with id, name, "
                        "category='restaurant', city, cuisine, lat, lon, opentime, endtime, price, "
                        "recommendedfood, source_tool. Final meal activity.position must exactly copy rows[i].name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "name": {"type": "string", "description": "Exact restaurant name when known."},
                            "cuisine": {"type": "string", "description": "Cuisine substring filter."},
                            "recommended_food": {"type": "string", "description": "Recommended dish substring filter."},
                            "topk": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_hotels_by_feature",
                    "description": (
                        "Typed ChinaTravel hotel search by featurehoteltype. Use this for required hotel features. "
                        "Returns canonical rows with id, name, category='accommodation', city, featurehoteltype, "
                        "lat, lon, price, numbed, source_tool. Final accommodation.position must exactly copy rows[i].name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "feature": {"type": "string", "description": "Feature substring, e.g. 健身室 or 管家服务."},
                            "topk": {"type": "integer"},
                        },
                        "required": ["city", "feature"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_hotels_by_budget",
                    "description": (
                        "Typed ChinaTravel hotel search by optional max_price. Returns canonical accommodation rows. "
                        "Use rows[i].name exactly as accommodation.position."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "max_price": {"type": "number"},
                            "topk": {"type": "integer"},
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "route_between",
                    "description": (
                        "Typed same-city route lookup aligned with feasibility-first route_between. "
                        "Requires concrete POI names, not city names. mode supports walk/metro/taxi; "
                        "if metro has no rows, backend falls back to taxi. Rows include mode, start, end, "
                        "start_time, end_time, duration, cost, distance."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                            "start_time": {"type": "string"},
                            "mode": {"type": "string", "enum": ["walk", "metro", "taxi"]},
                        },
                        "required": ["city", "start", "end", "start_time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "intercity_options",
                    "description": (
                        "Typed intercity transport lookup aligned with feasibility-first intercity_options. "
                        "mode must be train or airplane. Rows include TrainID or FlightID, start, end, "
                        "start_time, end_time, price, cost and original From/To/BeginTime/EndTime/Cost."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_city": {"type": "string"},
                            "end_city": {"type": "string"},
                            "mode": {"type": "string", "enum": ["train", "airplane"]},
                            "earliest_start_time": {"type": "string"},
                        },
                        "required": ["start_city", "end_city", "mode"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_open_status",
                    "description": (
                        "Typed open-status check for an attraction or restaurant row id. "
                        "Prefer using opentime/endtime from search rows for scheduling; call this when unsure."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "entity_type": {"type": "string", "enum": ["attraction", "restaurant"]},
                            "id": {"type": "integer"},
                            "time": {"type": "string"},
                        },
                        "required": ["city", "entity_type", "id", "time"],
                    },
                },
            },
        ]

    def execute_typed_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        runtime: Any,
    ) -> Dict[str, Any] | None:
        if tool_name not in self.typed_tool_names():
            return None
        if not self.typed_read_enabled:
            return {
                "ok": False,
                "error_code": "typed_read_tools_disabled",
                "message": "Typed read tools are only available with tool_profile='db_read_typed'.",
                "tool_name": tool_name,
                "tool_args": args,
            }

        if tool_name == "search_pois":
            city = str(args.get("city") or "").strip()
            name = str(args.get("name") or "").strip()
            poi_type = str(args.get("poi_type") or "").strip()
            if name:
                return self._execute_typed_select(
                    runtime,
                    "attractions_select",
                    {"city": city, "key": "name", "func_str": f"lambda x: x == {name!r}"},
                    category="attraction",
                    topk=args.get("topk"),
                )
            if poi_type:
                return self._execute_typed_select(
                    runtime,
                    "attractions_select",
                    {"city": city, "key": "type", "func_str": f"lambda x: x == {poi_type!r}"},
                    category="attraction",
                    topk=args.get("topk"),
                )
            return self._execute_typed_select(
                runtime,
                "attractions_select",
                {"city": city, "key": "name", "func_str": "lambda x: True"},
                category="attraction",
                topk=args.get("topk"),
            )

        if tool_name == "search_restaurants":
            city = str(args.get("city") or "").strip()
            name = str(args.get("name") or "").strip()
            cuisine = str(args.get("cuisine") or "").strip()
            recommended_food = str(args.get("recommended_food") or "").strip()
            if name:
                return self._execute_typed_select(
                    runtime,
                    "restaurants_select",
                    {"city": city, "key": "name", "func_str": f"lambda x: x == {name!r}"},
                    category="restaurant",
                    topk=args.get("topk"),
                )
            if cuisine:
                return self._execute_typed_select(
                    runtime,
                    "restaurants_select",
                    {"city": city, "key": "cuisine", "func_str": f"lambda x: {cuisine!r} in str(x)"},
                    category="restaurant",
                    topk=args.get("topk"),
                )
            if recommended_food:
                return self._execute_typed_select(
                    runtime,
                    "restaurants_select",
                    {
                        "city": city,
                        "key": "recommendedfood",
                        "func_str": f"lambda x: {recommended_food!r} in str(x)",
                    },
                    category="restaurant",
                    topk=args.get("topk"),
                )
            return self._execute_typed_select(
                runtime,
                "restaurants_select",
                {"city": city, "key": "name", "func_str": "lambda x: True"},
                category="restaurant",
                topk=args.get("topk"),
            )

        if tool_name == "search_hotels_by_feature":
            city = str(args.get("city") or "").strip()
            feature = str(args.get("feature") or "").strip()
            return self._execute_typed_select(
                runtime,
                "accommodations_select",
                {"city": city, "key": "featurehoteltype", "func_str": f"lambda x: {feature!r} in str(x)"},
                category="accommodation",
                topk=args.get("topk"),
            )

        if tool_name == "search_hotels_by_budget":
            city = str(args.get("city") or "").strip()
            max_price = args.get("max_price")
            if max_price is None:
                query_args = {"city": city, "key": "name", "func_str": "lambda x: True"}
            else:
                query_args = {
                    "city": city,
                    "key": "price",
                    "func_str": f"lambda x: float(x) <= {float(max_price)!r}",
                }
            return self._execute_typed_select(
                runtime,
                "accommodations_select",
                query_args,
                category="accommodation",
                topk=args.get("topk"),
            )

        if tool_name == "route_between":
            mode = str(args.get("mode") or "metro").strip().lower()
            start_time = self._normalize_typed_time(args.get("start_time"), default="09:00")
            result = self._execute_typed_route(
                runtime,
                {
                    "city": str(args.get("city") or "").strip(),
                    "start": str(args.get("start") or "").strip(),
                    "end": str(args.get("end") or "").strip(),
                    "start_time": start_time,
                    "transport_type": mode,
                },
            )
            if result.get("ok") and result.get("rows"):
                return result
            if mode != "taxi":
                return self._execute_typed_route(
                    runtime,
                    {
                        "city": str(args.get("city") or "").strip(),
                        "start": str(args.get("start") or "").strip(),
                        "end": str(args.get("end") or "").strip(),
                        "start_time": start_time,
                        "transport_type": "taxi",
                    },
                )
            return result

        if tool_name == "intercity_options":
            earliest_start_time = self._normalize_typed_time(args.get("earliest_start_time"), default="00:00")
            return self._execute_typed_intercity_options(
                runtime,
                {
                    "start_city": str(args.get("start_city") or "").strip(),
                    "end_city": str(args.get("end_city") or "").strip(),
                    "intercity_type": str(args.get("mode") or "").strip(),
                    "start_time": earliest_start_time,
                },
            )

        if tool_name == "check_open_status":
            entity_type = str(args.get("entity_type") or "").strip()
            primitive = {
                "attraction": "attractions_id_is_open",
                "restaurant": "restaurants_id_is_open",
            }.get(entity_type)
            if primitive is None:
                return {
                    "ok": False,
                    "error_code": "unsupported_arguments",
                    "message": "check_open_status entity_type must be attraction or restaurant.",
                    "tool_name": tool_name,
                    "tool_args": args,
                }
            return self._execute_typed_query(
                runtime,
                primitive,
                {
                    "city": str(args.get("city") or "").strip(),
                    "id": args.get("id"),
                    "time": self._normalize_typed_time(args.get("time"), default="09:00"),
                },
            )

        return {
            "ok": False,
            "error_code": "unsupported_tool",
            "message": f"Unsupported typed read tool: {tool_name}",
            "tool_name": tool_name,
            "tool_args": args,
        }

    def _execute_typed_select(
        self,
        runtime: Any,
        primitive_tool: str,
        args: Dict[str, Any],
        *,
        category: str,
        topk: Any = None,
    ) -> Dict[str, Any]:
        result = self._execute_typed_query(runtime, primitive_tool, args)
        rows = result.get("rows") if result.get("ok") else []
        if isinstance(rows, list):
            city = str(args.get("city") or "")
            canonical_rows = [
                self._canonical_typed_row(row, category=category, city=city, source_tool=primitive_tool)
                for row in rows
            ]
            limit = self._coerce_topk(topk)
            if limit is not None:
                canonical_rows = canonical_rows[:limit]
            result["rows"] = canonical_rows
            result["page"] = {
                "page": 1,
                "page_size": len(canonical_rows),
                "total": len(canonical_rows),
                "has_next": False,
            }
            if not canonical_rows:
                result["message"] = "no_results"
                result["empty_result_reason"] = "valid_query_no_rows"
        result["typed_interface"] = True
        return result

    def _execute_typed_route(self, runtime: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        result = self._execute_typed_query(runtime, "goto", args)
        rows = result.get("rows") if result.get("ok") else []
        if isinstance(rows, list):
            result["rows"] = [self._attach_route_duration(row) for row in rows]
        result["typed_interface"] = True
        return result

    def _execute_typed_intercity_options(self, runtime: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized_args = self.normalize_query_args("intercity_transport_select", args)
        validation_error = self.validate_query_args("intercity_transport_select", normalized_args)
        if validation_error is not None:
            validation_error["typed_interface"] = True
            return validation_error

        world_env = getattr(runtime, "world_env", None)
        backend_env = getattr(world_env, "backend_env", world_env)
        intercity_transport = getattr(backend_env, "intercitytransport", None)
        select = getattr(intercity_transport, "select", None)
        if callable(select):
            command = self.build_env_command("intercity_transport_select", normalized_args)
            try:
                whole_data = select(
                    normalized_args["start_city"],
                    normalized_args["end_city"],
                    normalized_args["intercity_type"],
                    normalized_args["start_time"],
                )
            except Exception as exc:
                message = f"Invalid command.\n{exc}"
                return {
                    "ok": False,
                    "error_code": self._infer_error_code(message),
                    "message": message,
                    "raw_command": command,
                    "tool_name": "intercity_transport_select",
                    "tool_args": normalized_args,
                    "typed_interface": True,
                    "source_tool": "intercity_transport_select",
                }

            page_size = max(1, int(getattr(world_env, "page_size", 10) or 10))
            page_data = self._slice_rows_payload(whole_data, 0, page_size)
            total_rows = self._payload_length(whole_data)
            result = {
                "success": True,
                "data": page_data,
                "whole_data": whole_data,
                "page": {
                    "page": 1,
                    "page_size": len(page_data) if isinstance(page_data, list) else min(total_rows, page_size),
                    "total": total_rows,
                    "has_next": total_rows > page_size,
                },
            }
            formatted = self.format_tool_result(
                "intercity_transport_select",
                normalized_args,
                command,
                result,
            )
            formatted["typed_interface"] = True
            formatted["source_tool"] = "intercity_transport_select"
            return formatted

        return self._execute_typed_query(runtime, "intercity_transport_select", normalized_args)

    def _execute_typed_query(self, runtime: Any, primitive_tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized_args = self.normalize_query_args(primitive_tool, args)
        validation_error = self.validate_query_args(primitive_tool, normalized_args)
        if validation_error is not None:
            validation_error["typed_interface"] = True
            return validation_error
        command = self.build_env_command(primitive_tool, normalized_args)
        result = runtime.world_env(command)
        formatted = self.format_tool_result(primitive_tool, normalized_args, command, result)
        formatted["typed_interface"] = True
        formatted["source_tool"] = primitive_tool
        return formatted

    def _canonical_typed_row(
        self,
        row: Dict[str, Any],
        *,
        category: str,
        city: str,
        source_tool: str,
    ) -> Dict[str, Any]:
        normalized = dict(row)
        normalized["category"] = category
        normalized["city"] = city
        normalized["source_tool"] = source_tool
        if "latitude" in normalized and "lat" not in normalized:
            normalized["lat"] = normalized["latitude"]
        if "longitude" in normalized and "lon" not in normalized:
            normalized["lon"] = normalized["longitude"]
        return normalized

    def _attach_route_duration(self, row: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(row)
        if "duration" not in normalized:
            duration = self._duration_minutes(normalized.get("start_time"), normalized.get("end_time"))
            if duration is not None:
                normalized["duration"] = duration
        return normalized

    def _duration_minutes(self, start_time: Any, end_time: Any) -> int | None:
        if not isinstance(start_time, str) or not isinstance(end_time, str):
            return None
        match_start = re.fullmatch(r"(\d{1,2}):(\d{2})", start_time.strip())
        match_end = re.fullmatch(r"(\d{1,2}):(\d{2})", end_time.strip())
        if not match_start or not match_end:
            return None
        start = int(match_start.group(1)) * 60 + int(match_start.group(2))
        end = int(match_end.group(1)) * 60 + int(match_end.group(2))
        if end < start:
            end += 24 * 60
        return end - start

    def _normalize_typed_time(self, value: Any, *, default: str) -> str:
        if not isinstance(value, str):
            return default
        text = value.strip()
        if not text:
            return default
        match = re.search(r"(\d{1,2}):(\d{2})", text)
        if not match:
            return default
        hour = int(match.group(1)) % 24
        minute = int(match.group(2))
        if minute >= 60:
            return default
        return f"{hour:02d}:{minute:02d}"

    def _coerce_topk(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None

    def _payload_length(self, payload: Any) -> int:
        if DataFrame is not None and isinstance(payload, DataFrame):
            return len(payload)
        if isinstance(payload, list):
            return len(payload)
        return 0

    def _slice_rows_payload(self, payload: Any, start: int, end: int) -> Any:
        if DataFrame is not None and isinstance(payload, DataFrame):
            return payload.iloc[start:end]
        if isinstance(payload, list):
            return payload[start:end]
        return payload

    def execution_atom_tools(self) -> List[Dict[str, Any]]:
        """Mutation tools used by the standalone ReAct baseline."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "insert_node",
                    "description": "在编辑视图中插入新节点。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {"type": "object"},
                            "position": {"type": "object"},
                        },
                        "required": ["node", "position"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_node",
                    "description": "删除编辑视图中的节点。",
                    "parameters": {
                        "type": "object",
                        "properties": {"node_id": {"type": "string"}},
                        "required": ["node_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_node",
                    "description": "移动编辑视图中的节点。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "new_position": {"type": "object"},
                        },
                        "required": ["node_id", "new_position"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "replace_node",
                    "description": "替换编辑视图中的节点。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_old_id": {"type": "string"},
                            "node_new": {"type": "object"},
                        },
                        "required": ["node_old_id", "node_new"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reschedule_node",
                    "description": "调整节点开始时间。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "new_time": {"type": "string"},
                            "policy": {
                                "type": "string",
                                "enum": ["shift_following", "no_propagation"],
                            },
                        },
                        "required": ["node_id", "new_time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "resize_node",
                    "description": "调整活动持续时间。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "new_duration": {"type": "integer"},
                            "policy": {
                                "type": "string",
                                "enum": ["shift_following", "no_propagation"],
                            },
                        },
                        "required": ["node_id", "new_duration"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reorder_day",
                    "description": "重排某一天的项目顺序。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "integer"},
                            "new_order_item_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["day", "new_order_item_ids"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reroute_edge",
                    "description": "修改路线交通方式。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edge_id": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["walk", "metro", "taxi", "unknown"],
                            },
                        },
                        "required": ["edge_id", "mode"],
                    },
                },
            },
        ]

    def react_tools(self) -> List[Dict[str, Any]]:
        """Combined tool set for the standalone ReAct baseline."""

        return self.available_tools()

    def build_env_command(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Map a tool call to the positional ChinaTravel world-env format."""

        args = self.normalize_query_args(tool_name, args)

        def _escape(value: Any) -> str:
            if isinstance(value, str):
                return f"'{value}'"
            return str(value)

        param_order = self._PARAM_ORDERS.get(tool_name)
        if param_order is None:
            arg_str = ", ".join(
                f"{key}={_escape(value)}"
                for key, value in args.items()
                if value is not None
            )
            return f"{tool_name}({arg_str})"

        arg_values = []
        for param in param_order:
            if param not in args or args[param] is None:
                continue
            if param == "func_str":
                arg_values.append(args[param])
            else:
                arg_values.append(_escape(args[param]))
        return f"{tool_name}({', '.join(arg_values)})"

    def normalize_query_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Repair common query-tool argument mistakes from the LLM."""

        normalized = dict(args)
        if tool_name == "intercity_transport_select":
            if "earliest_leave_time" in normalized and "start_time" not in normalized:
                normalized["start_time"] = normalized.pop("earliest_leave_time")
            intercity_type = str(normalized.get("intercity_type", "")).strip().lower()
            if intercity_type == "flight":
                normalized["intercity_type"] = "airplane"
            return normalized
        if tool_name == "next_page":
            cursor_id = str(normalized.get("cursor_id", "")).strip()
            if cursor_id:
                normalized["cursor_id"] = cursor_id
            return normalized
        if tool_name not in {
            "attractions_select",
            "restaurants_select",
            "accommodations_select",
        }:
            return normalized

        key = normalized.get("key")
        func_str = normalized.get("func_str")
        if isinstance(key, str):
            canonical_key = self._SELECT_KEY_ALIASES.get(tool_name, {}).get(key.strip().lower())
            if canonical_key:
                normalized["key"] = canonical_key
                key = canonical_key

        if (
            isinstance(key, str)
            and key not in self._KNOWN_SELECT_KEYS
            and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        ):
            normalized["key"] = "name"
            normalized["func_str"] = f"lambda x: x == {key!r}"
            return normalized

        if isinstance(func_str, str) and not func_str.strip().startswith("lambda"):
            normalized["func_str"] = f"lambda x: x == {func_str!r}"

        return normalized

    def validate_query_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any] | None:
        """Reject malformed tool calls before they hit WorldEnv."""

        if tool_name == "goto":
            transport_type = str(args.get("transport_type", "")).strip().lower()
            if transport_type and transport_type not in {"walk", "metro", "taxi"}:
                return {
                    "ok": False,
                    "error_code": "unsupported_arguments",
                    "message": "goto only supports transport_type in ['walk', 'metro', 'taxi']",
                    "tool_name": tool_name,
                    "tool_args": args,
                }
            city = str(args.get("city", "")).strip()
            start = str(args.get("start", "")).strip()
            end = str(args.get("end", "")).strip()
            if city and (start == city or end == city):
                return {
                    "ok": False,
                    "error_code": "invalid_poi_reference",
                    "message": "goto start/end must be concrete POI names, not city names",
                    "tool_name": tool_name,
                    "tool_args": args,
                }
        if tool_name == "intercity_transport_select":
            intercity_type = str(args.get("intercity_type", "")).strip().lower()
            if intercity_type not in {"train", "airplane"}:
                return {
                    "ok": False,
                    "error_code": "unsupported_arguments",
                    "message": "intercity_transport_select requires intercity_type in ['train', 'airplane']",
                    "tool_name": tool_name,
                    "tool_args": args,
                }
            start_city = str(args.get("start_city", "")).strip()
            end_city = str(args.get("end_city", "")).strip()
            if not start_city or not end_city:
                return {
                    "ok": False,
                    "error_code": "missing_required_fields",
                    "message": "intercity_transport_select requires both start_city and end_city",
                    "tool_name": tool_name,
                    "tool_args": args,
                }
        if tool_name == "next_page":
            cursor_id = str(args.get("cursor_id", "")).strip()
            if not cursor_id:
                return {
                    "ok": False,
                    "error_code": "missing_required_fields",
                    "message": "next_page requires cursor_id",
                    "tool_name": tool_name,
                    "tool_args": args,
                }
        return None

    def format_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        command: str,
        result: Any,
    ) -> Dict[str, Any]:
        success, data, whole_data = self._extract_env_payload(result)
        metadata = self._extract_env_metadata(result)
        if not success:
            message = str(data if data is not None else result)
            return {
                "ok": False,
                "error_code": metadata.get("error_code") or self._infer_error_code(message),
                "message": message,
                "raw_command": command,
                "tool_name": tool_name,
                "tool_args": args,
            }
        if self._is_error_string_payload(whole_data):
            message = str(whole_data)
            return {
                "ok": False,
                "error_code": self._infer_error_code(message),
                "message": message,
                "raw_command": command,
                "tool_name": tool_name,
                "tool_args": args,
            }

        payload = {
            "ok": True,
            "raw_command": command,
            "tool_name": tool_name,
            "tool_args": args,
        }
        if metadata.get("cursor_id"):
            payload["cursor_id"] = metadata["cursor_id"]

        if tool_name in {
            "attractions_keys",
            "accommodations_keys",
            "restaurants_keys",
        }:
            rows = self._normalize_key_type_rows(whole_data)
            payload["rows"] = rows
            payload["page"] = {
                "page": 1,
                "page_size": len(rows),
                "total": len(rows),
                "has_next": False,
            }
            if not rows:
                payload["message"] = "no_results"
                payload["empty_result_reason"] = "valid_query_no_rows"
            return payload

        if tool_name in {"attractions_types", "restaurants_cuisine"}:
            rows = self._normalize_scalar_rows(whole_data)
            payload["rows"] = rows
            payload["page"] = {
                "page": 1,
                "page_size": len(rows),
                "total": len(rows),
                "has_next": False,
            }
            if not rows:
                payload["message"] = "no_results"
                payload["empty_result_reason"] = "valid_query_no_rows"
            return payload

        if tool_name == "poi_lat_lon_search":
            payload["value"] = self._normalize_poi_search_value(whole_data)
            return payload

        if self._is_row_query_result(whole_data):
            row_source = data if metadata.get("page") else whole_data
            rows = self._normalize_rows(row_source, tool_name=tool_name)
            payload["rows"] = rows
            payload["page"] = metadata.get("page") or {
                "page": 1,
                "page_size": len(rows),
                "total": len(rows),
                "has_next": False,
            }
            if not rows:
                payload["message"] = "no_results"
                payload["empty_result_reason"] = "valid_query_no_rows"
            self._attach_field_hints(payload, tool_name)
            return payload

        if tool_name.endswith("_is_open"):
            payload["value"] = bool(whole_data)
            return payload

        if whole_data in (None, "No data.", []):
            payload["rows"] = []
            payload["page"] = {"page": 1, "page_size": 0, "total": 0, "has_next": False}
            payload["message"] = "no_results"
            payload["empty_result_reason"] = "valid_query_no_rows"
            return payload

        payload["value"] = self._to_jsonable(whole_data)
        return payload

    _FIELD_HINTS: dict[str, dict[str, str]] = {
        "attractions_select": {
            "opentime": "开始营业时间",
            "endtime": "结束营业时间",
            "price": "门票价格",
            "type": "景点类型",
            "recommendmintime": "建议最少游玩时间（小时）",
            "recommendmaxtime": "建议最多游玩时间（小时）",
        },
        "restaurants_select": {
            "opentime": "开始营业时间",
            "endtime": "结束营业时间",
            "price": "人均消费",
            "cuisine": "菜系",
            "recommendedfood": "推荐菜",
        },
        "accommodations_select": {
            "featurehoteltype": "酒店特色类型",
            "price": "房价",
            "numbed": "床位数",
        },
        "attractions_nearby": {
            "opentime": "开始营业时间",
            "endtime": "结束营业时间",
            "price": "门票价格",
            "type": "景点类型",
            "distance": "与查询点的距离（km）",
        },
        "restaurants_nearby": {
            "opentime": "开始营业时间",
            "endtime": "结束营业时间",
            "price": "人均消费",
            "cuisine": "菜系",
            "distance": "与查询点的距离（km）",
        },
    }

    def _attach_field_hints(self, payload: dict, tool_name: str) -> None:
        hints = self._FIELD_HINTS.get(tool_name)
        if hints:
            payload["field_hints"] = hints

    def _extract_env_payload(self, result: Any) -> tuple[bool, Any, Any]:
        if hasattr(result, "to_dict"):
            result = result.to_dict()

        if isinstance(result, dict):
            success = bool(result.get("success", False))
            data = result.get("data")
            whole_data = result.get("whole_data", data)
            return success, data, whole_data

        if hasattr(result, "__getitem__"):
            try:
                success = bool(result["success"])
                data = result["data"]
                whole_data = result["whole_data"]
                return success, data, whole_data
            except Exception:
                pass

        return False, result, result

    def _extract_env_metadata(self, result: Any) -> dict[str, Any]:
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        if not isinstance(result, dict):
            return {}
        metadata: dict[str, Any] = {}
        if "page" in result:
            metadata["page"] = result["page"]
        if "cursor_id" in result:
            metadata["cursor_id"] = result["cursor_id"]
        if "error_code" in result:
            metadata["error_code"] = result["error_code"]
        return metadata

    def _is_row_query_result(self, payload: Any) -> bool:
        if DataFrame is not None and isinstance(payload, DataFrame):
            return True
        return isinstance(payload, list)

    def _normalize_rows(self, payload: Any, *, tool_name: str | None = None) -> list[dict[str, Any]]:
        if DataFrame is not None and isinstance(payload, DataFrame):
            rows = payload.to_dict(orient="records")
        elif isinstance(payload, list):
            rows = [item if isinstance(item, dict) else {"value": item} for item in payload]
        else:
            rows = []
        if tool_name == "intercity_transport_select":
            rows = [self._normalize_intercity_transport_row(row) for row in rows]
        return [self._to_jsonable(row) for row in rows]

    def _normalize_intercity_transport_row(self, row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        normalized = dict(row)
        aliases = {
            "From": "start",
            "To": "end",
            "BeginTime": "start_time",
            "EndTime": "end_time",
            "Cost": "price",
        }
        for source, target in aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized[source]
        if "cost" not in normalized and "Cost" in normalized:
            normalized["cost"] = normalized["Cost"]
        return normalized

    def _normalize_key_type_rows(self, payload: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    rows.append(item)
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    rows.append({"key": item[0], "type": item[1]})
                    continue
                rows.append({"value": item})
        return [self._to_jsonable(row) for row in rows]

    def _normalize_scalar_rows(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, (list, tuple, set)):
            items = list(payload)
        elif hasattr(payload, "tolist"):
            converted = payload.tolist()
            items = converted if isinstance(converted, list) else [converted]
        else:
            items = []
        return [self._to_jsonable({"value": item}) for item in items]

    def _normalize_poi_search_value(self, payload: Any) -> Any:
        if isinstance(payload, tuple) and len(payload) == 2:
            return {
                "lat": self._to_jsonable(payload[0]),
                "lon": self._to_jsonable(payload[1]),
            }
        return self._to_jsonable(payload)

    def _is_error_string_payload(self, payload: Any) -> bool:
        if not isinstance(payload, str):
            return False
        stripped = payload.strip()
        if not stripped or stripped == "No data.":
            return False
        lowered = stripped.lower()
        return any(
            marker in lowered
            for marker in (
                "key not found",
                "no such point",
                "only support",
                "invalid command",
            )
        )

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, type):
            return value.__name__
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            if hasattr(value, "item"):
                try:
                    return value.item()
                except TypeError:
                    pass
            return str(value)

    def _infer_error_code(self, message: str) -> str:
        lowered = message.lower()
        if "only support" in lowered:
            return "unsupported_arguments"
        if "key not found" in lowered:
            return "invalid_field"
        if "no such point" in lowered:
            return "invalid_poi_reference"
        if "must be concrete poi names" in lowered:
            return "invalid_poi_reference"
        if "requires both" in lowered:
            return "missing_required_fields"
        if "invalid command" in lowered:
            return "invalid_command"
        return "tool_execution_failed"
