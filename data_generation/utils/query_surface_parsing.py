"""
Shared query-surface parsing helpers for data-generation validators.

This module intentionally stays pure and deterministic so Step 2 generation,
feasibility recompute, and post-hoc validation can reuse the same parsing
logic for natural language rewrites.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


CN_NUMERAL_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

INTERCITY_TRANSPORT_MODE_ALIASES = {
    "高铁": "train",
    "动车": "train",
    "火车": "train",
    "列车": "train",
    "铁路": "train",
    "飞机": "airplane",
    "航班": "airplane",
    "飞机航班": "airplane",
}

PREFERRED_POI_TYPE_ALIASES = {
    "博物馆": "博物馆/纪念馆",
    "纪念馆": "博物馆/纪念馆",
    "图书馆": "图书馆/纪念馆",
    "自然景点": "自然风光",
    "自然景观": "自然风光",
    "历史景点": "历史古迹",
    "历史文化景点": "历史古迹",
    "文化景点": "文化旅游区",
    "主题乐园": "游乐园/体育娱乐",
    "游乐园": "游乐园/体育娱乐",
    "艺术馆": "美术馆/艺术馆",
    "美术馆": "美术馆/艺术馆",
}


def normalize_intercity_transport_mode(value: Any) -> Optional[str]:
    text = _clean_fragment(value)
    if not text:
        return None
    if text in {"train", "airplane"}:
        return text
    if text == "flight":
        return "airplane"
    return INTERCITY_TRANSPORT_MODE_ALIASES.get(text)


def extract_intercity_transport_modes_from_query(edit_query: str) -> List[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    modes: List[str] = []
    for label, normalized in INTERCITY_TRANSPORT_MODE_ALIASES.items():
        if label not in edit_query:
            continue
        if normalized not in modes:
            modes.append(normalized)
    return modes


def normalize_preferred_poi_type(value: Any) -> Optional[str]:
    text = _clean_fragment(value)
    if not text:
        return None
    return PREFERRED_POI_TYPE_ALIASES.get(text, text)


def parse_compact_cn_numeral(token: Any) -> Optional[int]:
    text = str(token or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CN_NUMERAL_DIGITS:
        return CN_NUMERAL_DIGITS[text]
    digit_map = {"零": 0, "〇": 0, **CN_NUMERAL_DIGITS}
    unit_map = {"十": 10, "百": 100, "千": 1000}
    total = 0
    current = 0
    used_unit = False
    for char in text:
        if char in digit_map:
            current = digit_map[char]
            continue
        if char in unit_map:
            unit = unit_map[char]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            used_unit = True
            continue
        return None
    if used_unit:
        return total + current
    return None


def extract_requested_day_count_from_query(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    match = re.search(r"([零〇一二两三四五六七八九十\d]+)\s*天", edit_query)
    if match:
        return parse_compact_cn_numeral(match.group(1))
    return None


def extract_requested_day_delta_from_query(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    if not any(token in edit_query for token in ("增加", "加", "多", "扩展", "延长")):
        return None
    return extract_requested_day_count_from_query(edit_query)


def extract_daily_poi_cap_from_query(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"每天只安排([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)",
        r"每天最多安排([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)",
        r"每天安排([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)(?:就够了)?",
        r"每天只去([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)",
        r"每天最多去([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)",
        r"每天去([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)(?:就行|就够了)?",
        r"每天只玩([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)",
        r"每天玩([零〇一二两三四五六七八九十\d]+)个(?:景点|地方|点)(?:就行|就够了)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if match:
            return parse_compact_cn_numeral(match.group(1))
    return None


def extract_preferred_poi_type_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"(?:专门)?(?:看看|看|逛逛|安排)(?:[^\d零〇一二两三四五六七八九十，。；;]{1,12}?的)([^，。；;]{2,24}?)(?=[，。；;]|$)",
        r"(?:行程里|这次行程|这趟行程)?(?:希望|想)?多安排些([^，。；;]{2,24}?)(?:类)?的景点",
        r"(?:行程里|这次行程|这趟行程)?(?:希望|想)?多看看([^，。；;]{2,24}?)(?:类)?的景点",
        r"(?:我)?想去[^，。；;]{1,24}?这样(?:的|这种)([^，。；;]{2,24}?)(?:景点|地方|类型)?",
        r"(?:我)?想(?:看看|多看看|看点|多看点|去看看|逛逛|多逛逛)?([^，。；;]{2,24}?)类型的景点",
        r"(?:我)?想(?:看看|多看看|看点|多看点|去看看|逛逛|多逛逛)?([^，。；;]{2,24}?)(?:类)?的景点",
        r"(?:我)?想(?:看看|多看看|看点|多看点|去看看|逛逛|多逛逛)?([^，。；;]{2,24}?)(?:一点|一些)?的地方",
        r"(?:我)?想(?:看看|多看看|看点|多看点|去看看|逛逛|多逛逛)?([^，。；;]{2,24}?)景点",
        r"(?:我)?想(?:看看|多看看|看点|多看点|去看看|逛逛|多逛逛)?([^，。；;]{2,24}?)(?:类)?(?=[，。；;]|$)",
        r"(?:这次|整体|整个|行程(?:里|中)?|行程风格|路线|安排)(?:更)?偏(?:向)?([^，。；;]{2,24}?)(?:这种风格|风格|类型)",
        r"(?:这次|整体|整个|行程(?:里|中)?|路线|安排)(?:更)?想(?:主打|安排|多看|多看看)?([^，。；;]{2,24}?)(?:这类景点|这种类型|这种风格)",
        r"这样的([^，。；;]{2,24}?)(?:，|。|；|;|$)",
        r"我想去([^，。；;]{2,24}?)，但不要太",
        r"改成([^，。；;]{2,24}?)类型",
        r"同类型的([^，。；;]{2,24}?)(?:景点|地点)",
        r"更有名的([^，。；;]{2,24}?)(?:景点|地方|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        value = _sanitize_preferred_type_candidate(match.group(1))
        if value:
            return value
    return None


def extract_extend_with_pacing_spec(edit_query: str) -> Dict[str, int]:
    spec: Dict[str, int] = {}
    day_count = extract_requested_day_count_from_query(edit_query)
    if day_count is not None:
        spec["day_count"] = day_count
    daily_poi_cap = extract_daily_poi_cap_from_query(edit_query)
    if daily_poi_cap is not None:
        spec["daily_poi_cap"] = daily_poi_cap
    return spec


def _clean_fragment(value: Any) -> str:
    text = str(value or "").strip().strip("，。；;,.!?！？ ")
    text = re.sub(r"\s+", "", text)
    return text


def _sanitize_preferred_type_candidate(value: Any) -> Optional[str]:
    text = _clean_fragment(value)
    if not text:
        return None
    text = re.sub(r"(?:我还想.*|还想.*|我想.*)$", "", text).strip()
    text = re.sub(r"(?:一点|一些|一点点|这类景点|这种景点|这种类型|这种风格|这类地方)$", "", text)
    text = re.sub(r"^(?:点|些|这种|这类|这一类|更加|更偏|偏向|主打)", "", text)
    text = re.sub(r"(?:景点|地方|类型|风格)$", "", text)
    text = re.sub(r"我$", "", text).strip()
    text = text.strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\s*天|[零〇一二两三四五六七八九十]+\s*天", text):
        return None
    text = re.sub(r"^(?:多安排些|安排些|多看看|看看|想看|希望多安排些|希望多看看)", "", text).strip()
    text = re.sub(r"(?:类的?|风格的?)?$", "", text).strip()
    if _looks_like_dirty_semantic_phrase(text):
        return None
    return normalize_preferred_poi_type(text)


def _sanitize_named_entity_candidate(value: Any) -> Optional[str]:
    text = _clean_fragment(value)
    if not text:
        return None
    text = re.sub(r"（偏[^）]*）", "", text)
    text = re.sub(r"\(偏[^)]*\)", "", text)
    text = re.sub(
        r"^(?:我想看看|我想去看看|我想去|我想看|想去看看|想去|想看|能不能把|把|将|行程里去掉|行程里删掉|去掉|删掉|删除|不去|别去)+",
        "",
        text,
    ).strip()
    text = re.sub(r"(?:这样的|这种|这类).*$", "", text).strip()
    if not text:
        return None
    if text in {"掉", "删", "删除"}:
        return None
    if any(token in text for token in ("再去", "先去", "后去")):
        return None
    if any(token in text for token in ("景点", "地方", "类型", "风格", "附近", "周边")):
        return None
    if any(token in text for token in ("看看", "逛逛", "主打", "安排", "想去")):
        return None
    return text


def _looks_like_dirty_semantic_phrase(value: str) -> bool:
    text = _clean_fragment(value)
    if not text:
        return False
    dirty_tokens = (
        "想去",
        "去",
        "看看",
        "逛逛",
        "安排",
        "主打",
        "这样的",
        "这种",
        "这类",
        "类似的景点",
        "其他类似",
        "别太",
    )
    return any(token in text for token in dirty_tokens)


def _extract_query_budget_value(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*元", edit_query)
    if not match:
        return None
    return int(float(match.group(1)))


def _extract_query_distance_value(edit_query: str) -> Optional[float]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*公里", edit_query)
    if not match:
        return None
    return float(match.group(1))


def _extract_query_duration_minutes(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    text = str(edit_query or "")
    patterns = [
        re.compile(
            r"(?:限制在|控制在|最好控制在|尽量控制在|不超过|最多|别超过|最好不超过|不要超过|压到|压在)\s*(\d+(?:\.\d+)?)\s*(小时|分钟)(?:左右|以内|之内|内)?"
        ),
        re.compile(
            r"(?:限制在|控制在|最好控制在|尽量控制在|不超过|最多|别超过|最好不超过|不要超过|压到|压在)\s*半小时(?:左右|以内|之内|内)?"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        if match.lastindex == 2:
            amount = float(match.group(1))
            unit = match.group(2)
            return int(amount * 60) if unit == "小时" else int(amount)
        return 30
    return None


def _has_travel_time_scope_signal(edit_query: str) -> bool:
    text = str(edit_query or "")
    return any(
        token in text
        for token in (
            "交通时间",
            "景点之间交通",
            "景点之间的交通",
            "活动之间交通",
            "活动之间的交通",
            "路上",
            "通勤",
        )
    )


def _has_explicit_travel_time_threshold(edit_query: str) -> bool:
    text = str(edit_query or "")
    if not _has_travel_time_scope_signal(text):
        return False
    return re.search(r"(\d+(?:\.\d+)?)\s*(小时|分钟)", text) is not None


def is_explicit_hard_travel_time_query(
    edit_query: str,
    semantic_slots: Optional[Dict[str, Any]] = None,
    query_spec: Optional[Dict[str, Any]] = None,
) -> bool:
    slots = semantic_slots if isinstance(semantic_slots, dict) else {}
    spec = query_spec if isinstance(query_spec, dict) else {}
    duration_limit = (
        slots.get("duration_limit_min")
        or spec.get("duration_limit_min")
        or _extract_query_duration_minutes(edit_query)
    )
    if duration_limit is not None:
        return True

    text = str(edit_query or "")
    hard_limit_tokens = (
        "不超过",
        "别超过",
        "最好别超过",
        "最多",
        "控制在",
        "压到",
        "压在",
        "限制在",
        "以内",
        "内",
    )
    return _has_explicit_travel_time_threshold(text) and any(token in text for token in hard_limit_tokens)


def extract_day_poi_pairs_from_query(edit_query: str) -> List[Tuple[int, str]]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    pairs: List[Tuple[int, str]] = []
    pattern = re.compile(r"第([一二三四五六七八九十\d]+)天(?:去|游玩|参观|安排|打卡)([^，。；;]+)")
    for match in pattern.finditer(edit_query):
        day_token = match.group(1)
        poi_name = _sanitize_named_entity_candidate(match.group(2))
        day = parse_compact_cn_numeral(day_token)
        if day and poi_name:
            pairs.append((day, poi_name))
    return pairs


def extract_replace_source_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    for pattern in [r"把(.+?)改成", r"把(.+?)换成", r"将(.+?)替换为"]:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _sanitize_named_entity_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def extract_semantic_target_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"(?:改成|换成|替换为)(.+?)类型",
        r"(?:改成|换成)(.+?)的其他景点",
        r"同类型的(.+?)(?:景点|地点)",
        r"更有名的(.+?)(?:景点|地方)",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _sanitize_preferred_type_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def extract_specific_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"特别是([^，。；;]+)",
        r"尤其是([^，。；;]+)",
        r"(?:我)?想去([^，。；;]{1,24}?)这样(?:的|这种)",
        r"除了([^，。；;]{1,24}?)，我还想看看其他类似的",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _sanitize_named_entity_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def extract_semantic_specific_pair(edit_query: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(edit_query, str) or not edit_query:
        return None, None
    patterns = [
        r"(?:我)?想去([^，。；;]{1,24}?)这样(?:的|这种)([^，。；;]{1,24}?)(?:景点|地方|类型)?(?:逛逛|看看|还想|，|。|；|;|$)",
        r"([^，。；;]{1,24}?)这样(?:的|这种)([^，。；;]{1,24}?)(?:景点|地方|类型)?(?:逛逛|看看|还想|，|。|；|;|$)",
        r"除了([^，。；;]{1,24}?)，我还想看看其他类似的([^，。；;]{1,24}?)(?:景点|地方|旅游区|古迹|类型)?(?:。|，|；|;|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        specific_poi = _sanitize_named_entity_candidate(match.group(1))
        preferred_type = _sanitize_preferred_type_candidate(match.group(2))
        if specific_poi or preferred_type:
            return specific_poi, preferred_type
    return None, None


def extract_anchor_poi_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"保留([^，。；;]+?)",
        r"安排([^，。；;]+?)和附近的餐厅",
        r"安排([^，。；;]+?)和周边的餐厅",
        r"把([^，。；;]+?)和附近的餐厅",
        r"把([^，。；;]+?)和周边的餐厅",
        r"围绕([^，。；;]+?)安排活动",
        r"围绕([^，。；;]+?)安排行程",
        r"以([^，。；;]+?)为中心安排活动",
        r"以([^，。；;]+?)为中心安排行程",
        r"以([^，。；;]+?)为核心安排活动",
        r"以([^，。；;]+?)为核心安排行程",
        r"把([^，。；;]+?)的时间安排在",
        r"把([^，。；;]+?)的游玩时间限制在",
        r"增加([^，。；;]+?)到行程中",
        r"把([^，。；;]+?)加进去",
        r"想把([^，。；;]+?)加进去",
        r"把([^，。；;]+?)加上",
        r"想把([^，。；;]+?)加上",
        r"行程里加上([^，。；;]+?)(?:吧|呀|啊)?(?=，|。|；|;|$)",
        r"想在行程里加上([^，。；;]+?)(?:吧|呀|啊)?(?=，|。|；|;|$)",
        r"把([^，。；;]+?)安排到行程中",
        r"想把([^，。；;]+?)也?加进行程",
        r"把([^，。；;]+?)也?加进行程",
        r"想在行程里加上([^，。；;]+?)",
        r"把([^，。；;]+?)从行程中删除",
        r"我想去([^，。；;]+?)这样(?:的|这种)",
        r"(?:想坐|坐|乘坐|游玩|参观|去|打卡)([^，。；;]{2,30}?)(?=并在|并且|，|。|；|;|$)",
        r"住([^，。；;]+?)酒店",
        r"入住([^，。；;]+?)酒店",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _sanitize_named_entity_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def extract_restaurant_name_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"(?:安排在|安排|增加|加入|去|在)([^，。；;]{2,40}?)(?:用餐|吃一顿|就餐)",
        r"(?:安排|增加|加入)([^，。；;]{2,40}?(?:餐厅|饭店|餐馆))(?:用餐|就餐|这顿饭)?",
        r"(?:去|在)([^，。；;]{2,40}?(?:餐厅|饭店|餐馆))用餐",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _clean_fragment(match.group(1))
        if not candidate or "附近" in candidate or "周边" in candidate:
            continue
        return candidate
    return None


def extract_restaurant_type_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"我想吃([^，。；;]{2,24})",
        r"想吃([^，。；;]{2,24})",
        r"去吃([^，。；;]{2,24})",
        r"安排([^，。；;]{2,24})餐厅",
        r"([^，。；;]{2,24})餐厅",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _sanitize_preferred_type_candidate(match.group(1))
        if candidate and any(token in candidate for token in ("附近", "周边", "和")):
            continue
        if candidate:
            return candidate
    return None


def extract_hotel_name_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"(?:住在|入住|订|住)([^，。；;]{2,40}?(?:酒店|宾馆|民宿))",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _clean_fragment(match.group(1))
        if candidate:
            return candidate
    return None


def extract_hotel_feature_from_query(edit_query: str) -> Optional[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"住([^，。；;]{2,24}?)酒店",
        r"入住([^，。；;]{2,24}?)酒店",
        r"选([^，。；;]{2,24}?)酒店",
        r"([^，。；;]{2,24}?)酒店",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        candidate = _sanitize_preferred_type_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def extract_room_type_from_query(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    alias_map = {
        "大床房": 1,
        "单床房": 1,
        "单人间": 1,
        "双床房": 2,
        "标间": 2,
        "标准间": 2,
        "双人间": 2,
    }
    for label, value in alias_map.items():
        if label in edit_query:
            return value
    patterns = [
        r"房型(?:改成|调整为|设置为|要)?\s*(\d+)",
        r"(\d+)\s*号房型",
        r"(\d+)\s*型房",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def extract_room_count_from_query(edit_query: str) -> Optional[int]:
    if not isinstance(edit_query, str) or not edit_query:
        return None
    patterns = [
        r"(\d+)\s*间房",
        r"(\d+)\s*间客房",
        r"要\s*(\d+)\s*间房",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def extract_transport_time_windows_from_query(edit_query: str) -> List[Dict[str, str]]:
    if not isinstance(edit_query, str) or not edit_query:
        return []

    patterns = [
        (
            re.compile(r"去程[^。；;]*?(?:(最早|最晚)\s*)?(\d{1,2}:\d{2})(?:(前|后)\s*)?(出发|到达)?"),
            "outbound",
        ),
        (
            re.compile(r"(?:返程|回程)[^。；;]*?(?:(最早|最晚)\s*)?(\d{1,2}:\d{2})(?:(前|后)\s*)?(出发|到达)?"),
            "return",
        ),
    ]
    windows: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for pattern, leg in patterns:
        for match in pattern.finditer(edit_query):
            qualifier, clock, relation, action = match.groups()
            operator = "<=" if relation == "前" else ">="
            if qualifier == "最早":
                operator = ">="
            elif qualifier == "最晚":
                operator = "<="
            elif relation is None:
                operator = "<=" if action == "到达" else ">="
            field = "end_time" if action == "到达" else "start_time"
            key = (leg, field, operator, clock)
            if key in seen:
                continue
            seen.add(key)
            windows.append({"leg": leg, "field": field, "operator": operator, "value": clock})
    return windows


def extract_city_split_targets_from_query(edit_query: str) -> List[str]:
    if not isinstance(edit_query, str) or not edit_query:
        return []
    patterns = [
        r"拆成([^，。；;]{2,12})和周边另一个城市",
        r"拆成([^，。；;]{2,12})和附近另一个城市",
        r"([^，。；;]{2,12})和周边另一个城市",
        r"([^，。；;]{2,12})和附近另一个城市",
        r"([^，。；;]{2,12})和([^，。；;]{2,12})(?:两个城市|两座城市)",
        r"拆分成([^，。；;]{2,12})和([^，。；;]{2,12})两个城市",
    ]
    for pattern in patterns:
        match = re.search(pattern, edit_query)
        if not match:
            continue
        values = [_clean_fragment(group) for group in match.groups() if _clean_fragment(group)]
        if len(values) == 1:
            return [values[0], "周边另一个城市"]
        if len(values) >= 2:
            return values[:2]
    if any(token in edit_query for token in ("两个城市", "两座城市", "另一个城市", "拆分")):
        return ["当前目标城市", "另一个城市"]
    return []


def _confidence_from_slots(slots: Dict[str, Any], errors: List[str], constraint_type: str) -> float:
    score = 0.55
    key_map = {
        "semantic_and_specific": ("specific_poi", "preferred_type"),
        "category_overlap": ("specific_poi", "preferred_type"),
        "semantic_substitution": ("source_poi", "semantic_target"),
        "type_filter": ("preferred_type",),
        "preference_constraint": ("preferred_type",),
        "resource_overlap": ("anchor_poi", "budget_cap"),
        "multi_city_split": ("city_split_targets",),
        "required_restaurant_type": ("restaurant_type",),
        "required_hotel_feature": ("hotel_feature",),
        "required_intercity_transport_type": ("intercity_transport_modes",),
        "transport_time_window": ("transport_time_windows",),
    }
    for key in key_map.get(str(constraint_type or "").strip(), ()):
        value = slots.get(key)
        if value not in (None, "", [], False):
            score += 0.18
    if slots.get("day_poi_pairs"):
        score += 0.12
    if errors:
        score -= 0.15 * len(errors)
    return max(0.0, min(1.0, round(score, 3)))


def merge_semantic_slots(primary: Optional[Dict[str, Any]], fallback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for payload in (fallback, primary):
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if value in (None, "", []):
                continue
            merged[key] = value
    return merged


def _slot_value_looks_suspicious(key: str, value: Any) -> bool:
    if value in (None, "", []):
        return True
    if key in {"specific_poi", "source_poi", "anchor_poi", "restaurant_name", "hotel_name"}:
        text = str(value).strip()
        return (
            not text
            or text.startswith(("我想", "想看", "想去", "掉", "删"))
            or text in {"掉", "删", "删除"}
        )
    if key in {"preferred_type", "semantic_target", "restaurant_type", "hotel_feature"}:
        text = str(value).strip()
        return (not text) or _looks_like_dirty_semantic_phrase(text) or text.endswith("我")
    if key == "day_poi_pairs":
        for pair in value if isinstance(value, list) else []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return True
            poi_name = str(pair[1] or "").strip()
            if poi_name.endswith(("？", "?", "。", "！", "!")):
                return True
        return False
    return False


def parse_query_semantic_slots(
    edit_query: str,
    constraint_type: str,
    template_str: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del template_str
    ctype = str(constraint_type or "").strip()
    errors: List[str] = []

    specific_poi, paired_type = extract_semantic_specific_pair(edit_query)
    preferred_type = paired_type or extract_preferred_poi_type_from_query(edit_query)
    source_poi = extract_replace_source_poi_from_query(edit_query)
    semantic_target = extract_semantic_target_from_query(edit_query)
    anchor_poi = extract_anchor_poi_from_query(edit_query)
    restaurant_name = extract_restaurant_name_from_query(edit_query)
    restaurant_type = extract_restaurant_type_from_query(edit_query)
    hotel_name = extract_hotel_name_from_query(edit_query)
    hotel_feature = extract_hotel_feature_from_query(edit_query)
    city_split_targets = extract_city_split_targets_from_query(edit_query)
    intercity_transport_modes = extract_intercity_transport_modes_from_query(edit_query)
    transport_time_windows = extract_transport_time_windows_from_query(edit_query)

    if specific_poi is None:
        specific_poi = extract_specific_poi_from_query(edit_query)

    nearby_meal = any(token in str(edit_query or "") for token in ("周边用餐", "附近用餐", "附近的餐厅", "周边餐厅"))
    additional_similar = any(token in str(edit_query or "") for token in ("其他类似", "类似的景点", "同类景点", "类似景点"))

    slots: Dict[str, Any] = {
        "constraint_type": ctype,
        "specific_poi": specific_poi,
        "preferred_type": preferred_type,
        "source_poi": source_poi,
        "semantic_target": semantic_target,
        "anchor_poi": anchor_poi,
        "day_poi_pairs": extract_day_poi_pairs_from_query(edit_query),
        "budget_cap": _extract_query_budget_value(edit_query),
        "duration_limit_min": _extract_query_duration_minutes(edit_query),
        "distance_km": _extract_query_distance_value(edit_query),
        "restaurant_name": restaurant_name,
        "restaurant_type": restaurant_type,
        "hotel_name": hotel_name,
        "hotel_feature": hotel_feature,
        "hotel_room_type": extract_room_type_from_query(edit_query),
        "hotel_room_count": extract_room_count_from_query(edit_query),
        "city_split_targets": city_split_targets,
        "intercity_transport_modes": intercity_transport_modes,
        "transport_time_windows": transport_time_windows,
        "additional_similar": additional_similar,
        "nearby_meal": nearby_meal,
        "day_count": extract_requested_day_count_from_query(edit_query),
        "day_delta": extract_requested_day_delta_from_query(edit_query),
        "daily_poi_cap": extract_daily_poi_cap_from_query(edit_query),
        "parse_source": "regex",
        "parse_confidence": 0.0,
        "parse_errors": [],
    }
    if ctype in {"cross_day_restructure", "cross_day_variety_by_types"} and slots["day_poi_pairs"]:
        max_pair_day = max(day for day, _ in slots["day_poi_pairs"])
        if slots["day_count"] is None or slots["day_count"] < max_pair_day:
            slots["day_count"] = max_pair_day

    semantic_only_types = {
        "semantic_and_specific",
        "category_overlap",
        "semantic_substitution",
        "type_filter",
        "preference_constraint",
        "required_restaurant_type",
        "required_hotel_feature",
        "resource_overlap",
        "multi_city_split",
        "add_days",
    }
    if ctype not in semantic_only_types:
        for key in (
            "specific_poi",
            "preferred_type",
            "source_poi",
            "semantic_target",
            "restaurant_type",
            "city_split_targets",
            "additional_similar",
        ):
            if key == "city_split_targets":
                slots[key] = []
            elif key == "additional_similar":
                slots[key] = False
            else:
                slots[key] = None

    expected_slots = hints.get("expected_slots") if isinstance(hints, dict) and isinstance(hints.get("expected_slots"), dict) else None
    merged_hint = merge_semantic_slots(hints, expected_slots)
    for key in (
        "specific_poi",
        "preferred_type",
        "source_poi",
        "semantic_target",
        "anchor_poi",
        "restaurant_name",
        "restaurant_type",
        "hotel_name",
        "hotel_feature",
        "city_split_targets",
        "day_poi_pairs",
        "budget_cap",
        "duration_limit_min",
        "distance_km",
        "day_count",
        "day_delta",
        "daily_poi_cap",
        "additional_similar",
        "nearby_meal",
    ):
        if (
            merged_hint.get(key) not in (None, "", [])
            and _slot_value_looks_suspicious(key, slots.get(key))
        ):
            slots[key] = merged_hint.get(key)

    if slots.get("preferred_type") and _looks_like_dirty_semantic_phrase(str(slots["preferred_type"])):
        errors.append("dirty_preferred_type")
        slots["preferred_type"] = None

    if slots.get("restaurant_name") and any(token in str(slots["restaurant_name"]) for token in ("附近", "周边")):
        errors.append("implicit_restaurant_name")
        slots["restaurant_name"] = None

    if ctype in {"semantic_and_specific", "category_overlap"}:
        if not slots.get("specific_poi"):
            errors.append("missing_specific_poi")
        if not slots.get("preferred_type"):
            errors.append("missing_preferred_type")
    elif ctype == "semantic_substitution":
        if not slots.get("source_poi"):
            errors.append("missing_source_poi")
        if not slots.get("semantic_target"):
            errors.append("missing_semantic_target")
    elif ctype in {"type_filter", "preference_constraint"}:
        if not slots.get("preferred_type"):
            errors.append("missing_preferred_type")
    elif ctype == "resource_overlap":
        if slots.get("anchor_poi") is None:
            slots["anchor_poi"] = extract_anchor_poi_from_query(edit_query)
        if not slots.get("anchor_poi") and merged_hint.get("anchor_poi") not in (None, "", []):
            slots["anchor_poi"] = merged_hint.get("anchor_poi")
        if not slots.get("anchor_poi"):
            errors.append("missing_anchor_poi")
        if slots.get("budget_cap") is None:
            errors.append("missing_budget_cap")
    elif ctype == "multi_city_split":
        if len(slots.get("city_split_targets") or []) < 2:
            errors.append("missing_split_signal")
    elif ctype == "required_restaurant_type":
        if not slots.get("restaurant_type"):
            errors.append("missing_restaurant_type")
    elif ctype == "required_hotel_feature":
        if not slots.get("hotel_feature"):
            errors.append("missing_hotel_feature")

    slots["parse_errors"] = errors
    slots["parse_confidence"] = _confidence_from_slots(slots, errors, ctype)
    return slots


def resolve_query_semantic_slots(
    edit_query: str,
    constraint_type: str,
    query_generation_trace: Optional[Dict[str, Any]] = None,
    template_str: Optional[str] = None,
    hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    persisted = {}
    if isinstance(query_generation_trace, dict):
        raw = query_generation_trace.get("canonical_semantic_slots")
        if not isinstance(raw, dict):
            raw = query_generation_trace.get("semantic_slots")
        if isinstance(raw, dict):
            persisted = dict(raw)
    if persisted:
        return merge_semantic_slots(persisted, parse_query_semantic_slots(edit_query, constraint_type, template_str=template_str, hints=hints))
    return parse_query_semantic_slots(edit_query, constraint_type, template_str=template_str, hints=hints)
