"""Analyze ChinaTravel hf_queries and derive a soft-constraint taxonomy report."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ConstraintMention:
    family: str
    facet: str
    polarity: str
    anchor: Optional[str]
    intensity: str
    is_explicit: bool
    evidence_span: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "facet": self.facet,
            "polarity": self.polarity,
            "anchor": self.anchor,
            "intensity": self.intensity,
            "is_explicit": self.is_explicit,
            "evidence_span": self.evidence_span,
            "source": self.source,
        }


@dataclass
class QueryRecord:
    uid: str
    source_bucket: str
    days: Optional[int]
    people_number: Optional[int]
    nature_language: str
    nature_language_en: str
    hard_logic_py: List[str]
    has_explicit_preference: bool
    preference_en: List[str]
    preference_py: List[str]
    soft_constraints: List[ConstraintMention] = field(default_factory=list)
    hard_constraint_signals: List[str] = field(default_factory=list)
    travel_intent_signals: List[str] = field(default_factory=list)
    must_visit_mentions: List[str] = field(default_factory=list)
    seed_preference_family: Optional[str] = None

    def to_inventory_row(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "source_bucket": self.source_bucket,
            "days": self.days,
            "people_number": self.people_number,
            "nature_language": self.nature_language,
            "nature_language_en": self.nature_language_en,
            "hard_logic_py": self.hard_logic_py,
            "has_explicit_preference": self.has_explicit_preference,
            "preference_en": self.preference_en,
            "preference_py": self.preference_py,
            "seed_preference_family": self.seed_preference_family,
            "soft_constraints": [item.to_dict() for item in self.soft_constraints],
            "hard_constraint_signals": self.hard_constraint_signals,
            "travel_intent_signals": self.travel_intent_signals,
            "must_visit_mentions": self.must_visit_mentions,
        }


THEME_KEYWORDS = {
    "culture": [
        "人文",
        "历史",
        "博物馆",
        "古迹",
        "文化",
        "艺术",
        "遗址",
        "历史景点",
        "historical",
        "cultural",
        "museum",
        "heritage",
        "art",
    ],
    "nature": [
        "自然",
        "风景",
        "风光",
        "公园",
        "景色",
        "山",
        "湖",
        "自然景观",
        "nature",
        "scenic",
        "park",
        "landscape",
    ],
    "food": [
        "美食",
        "小吃",
        "火锅",
        "粤菜",
        "烤鸭",
        "豆汁",
        "川菜",
        "本地菜",
        "本地人的饮食习惯",
        "特色美食",
        "food",
        "cuisine",
        "local delicacies",
    ],
    "photo": [
        "拍照",
        "出片",
        "打卡",
        "网红",
        "会拍照",
        "photo",
        "photogenic",
        "instagrammable",
    ],
    "family": [
        "亲子",
        "孩子",
        "儿子",
        "女儿",
        "海洋馆",
        "动物园",
        "family-friendly",
        "kid-friendly",
        "children",
    ],
    "shopping": [
        "购物",
        "商场",
        "商业街",
        "shopping",
        "mall",
    ],
}

THEME_EXPLICIT_PATTERNS = {
    "culture": [
        r"(?:历史古迹|历史类|古迹类|文化类|人文类|博物馆或纪念馆|博物馆/纪念馆)",
        r"(?:文化|人文|历史|艺术)(?:主题|气息|体验|路线|景点|内容)",
    ],
    "nature": [
        r"(?:自然风光|自然景观|自然类|风景类|园林类|公园类|山水)",
        r"(?:自然|风景|园林|山水)(?:主题|气息|体验|路线|景点|内容)",
        r"(?:想看|希望看|喜欢看|偏好).{0,12}(?:园林|山水)",
    ],
    "food": [
        r"(?:美食|饮食)(?:主题|体验|路线|内容)",
        r"(?:当地特色美食|特色美食|美食体验|吃遍)",
        r"(?:想|希望|喜欢|偏好|主要目标是).{0,12}(?:美食体验|特色美食|当地美食)",
    ],
    "photo": [
        r"(?:拍照|打卡|出片|网红)(?:主题|体验|路线|景点|内容)?",
        r"(?:主要目标是|希望|想).{0,12}(?:拍照|打卡|出片)",
    ],
    "family": [
        r"(?:亲子|儿童友好|适合孩子|适合小朋友|带孩子).{0,12}(?:体验|景点|路线|活动)?",
    ],
    "shopping": [
        r"(?:购物|商场|商业街)(?:主题|体验|路线|内容)",
        r"(?:想|希望|喜欢|偏好|主要目标是).{0,12}(?:购物|逛商场|商业街)",
    ],
}

FAMILY_ORDER = [
    "burden_reduction",
    "route_compactness",
    "anchor_proximity",
    "cost_allocation_preference",
    "theme_alignment",
    "experience_richness",
]

FACETS_BY_FAMILY = {
    "burden_reduction": ["walk", "pace", "activity_load"],
    "route_compactness": ["travel_time", "route_smoothness", "idle_gap"],
    "anchor_proximity": ["poi_anchor", "district_anchor", "hotel_anchor"],
    "cost_allocation_preference": [
        "food_share",
        "accommodation_cost",
        "experience_share",
    ],
    "theme_alignment": [
        "culture",
        "nature",
        "food",
        "photo",
        "family",
        "shopping",
        "mixed_theme",
    ],
    "experience_richness": [
        "attraction_count",
        "category_diversity",
        "food_experience_count",
    ],
}

SEED_FAMILY_MAP = {
    "more attractions": ("experience_richness", "attraction_count", "maximize"),
    "less inner transports time": ("route_compactness", "travel_time", "minimize"),
    "minimize average transport time to restaurants": (
        "route_compactness",
        "travel_time",
        "minimize",
    ),
    "maximize food_cost_ratio": (
        "cost_allocation_preference",
        "food_share",
        "maximize",
    ),
    "minimize accommodation_cost": (
        "cost_allocation_preference",
        "accommodation_cost",
        "minimize",
    ),
    "minimize distance to poi": ("anchor_proximity", "poi_anchor", "minimize"),
}

METRIC_HOOKS = {
    "burden_reduction": {
        "candidate_metric": "total walking distance / walking duration / daily activity load",
        "direction": "lower is better",
        "existing_reusable_signals": [
            "long_walk_distance",
            "long_walk_duration",
            "dense_day_attractions",
        ],
        "reusable_code": [
            "data_clean/rules.py",
        ],
        "strengthen_rule": "same facet stays present and intensity moves weak->medium->strong, or the edit adds more explicit anti-fatigue wording.",
        "known_gaps": "No single aggregate load score exists yet; pace and fatigue are only partially covered by current signals.",
    },
    "route_compactness": {
        "candidate_metric": "average inner-city transport time / route smoothness penalty / large idle-gap count",
        "direction": "lower is better",
        "existing_reusable_signals": [
            "large_idle_gap",
            "post_transport_idle_gap",
            "day_boundary_transfer",
            "average_time_cost",
        ],
        "reusable_code": [
            "Chinatravel/ChinaTravel/chinatravel/evaluation/preference.py",
            "data_clean/rules.py",
        ],
        "strengthen_rule": "same facet stays present and edit adds a stronger request for smoother routing, shorter transport, or less backtracking.",
        "known_gaps": "Backtracking itself is not explicitly measured; current implementation mostly proxies with transport-time and idle-gap signals.",
    },
    "anchor_proximity": {
        "candidate_metric": "average distance to anchor POI / district centroid / accommodation-to-anchor distance",
        "direction": "lower is better",
        "existing_reusable_signals": ["near_poi", "average_dist_cost"],
        "reusable_code": [
            "Chinatravel/ChinaTravel/chinatravel/evaluation/preference.py",
        ],
        "strengthen_rule": "same anchor family stays present and edit either adds a concrete anchor or asks for tighter proximity.",
        "known_gaps": "District-level and hotel-anchor proximity need separate geospatial treatment; only POI-anchor distance is directly implemented now.",
    },
    "cost_allocation_preference": {
        "candidate_metric": "food_cost_ratio / accommodation_cost / experience spend share under the same hard budget",
        "direction": "depends on facet: food_share and experience_share higher is better; accommodation_cost lower is better",
        "existing_reusable_signals": [
            "food_cost_ratio",
            "accommodation_cost",
            "accommodation_cost_ratio",
        ],
        "reusable_code": [
            "Chinatravel/ChinaTravel/chinatravel/evaluation/preference.py",
        ],
        "strengthen_rule": "same facet stays present and edit increases spending preference intensity or more clearly asks to shift budget allocation.",
        "known_gaps": "Experience-share is not implemented as a first-class metric and will need a normalized denominator definition.",
    },
    "theme_alignment": {
        "candidate_metric": "POI-theme coverage or alignment score over itinerary activities",
        "direction": "higher is better",
        "existing_reusable_signals": ["theme_undercover", "theme preference heuristics"],
        "reusable_code": [
            "src/generation/scene_annotator.py",
        ],
        "strengthen_rule": "same theme facet stays present and edit becomes more explicit, more exclusive, or adds concrete thematic anchors.",
        "known_gaps": "No stable evaluator exists yet; theme scoring still needs POI tagging and calibrated coverage thresholds.",
    },
    "experience_richness": {
        "candidate_metric": "daily attraction count / category diversity ratio / food-experience count",
        "direction": "higher is better for attraction_count and diversity",
        "existing_reusable_signals": [
            "daily_attraction_count",
            "type_homogeneous",
        ],
        "reusable_code": [
            "Chinatravel/ChinaTravel/chinatravel/evaluation/preference.py",
            "src/generation/scene_annotator.py",
        ],
        "strengthen_rule": "same facet stays present and edit asks for more items, more variety, or fewer repeated categories.",
        "known_gaps": "Food-experience count and diversity need a stable activity taxonomy before evaluator implementation.",
    },
}

GENERATION_IMPLICATIONS = {
    "burden_reduction": "Can seed edits like 少走路 / 行程轻松点 / 别太赶, then validate against walk and daily-load signals.",
    "route_compactness": "Can generate reroute-style edits around smoother sequencing, less transfer time, and fewer idle gaps.",
    "anchor_proximity": "Can create add/strengthen edits around staying, dining, or playing near a named POI or district.",
    "cost_allocation_preference": "Can generate budget-reallocation edits without changing the hard budget cap.",
    "theme_alignment": "Can generate style-shift edits toward culture, nature, food, photo, family, or shopping themes.",
    "experience_richness": "Can generate edits that increase attractions, increase food experiences, or reduce homogeneous activity types.",
}


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent
    workspace_root = project_root.parent
    parser = argparse.ArgumentParser(
        description="Analyze ChinaTravel hf_queries soft constraints and generate taxonomy artifacts."
    )
    parser.add_argument(
        "--query-root",
        type=Path,
        default=workspace_root / "Chinatravel" / "ChinaTravel" / "chinatravel" / "data" / "hf_queries",
        help="Directory containing the hf_queries split folders.",
    )
    parser.add_argument(
        "--edit-root",
        type=Path,
        default=workspace_root / "data" / "tpe_dataset" / "base_plans_qwen3-8b",
        help="Directory containing ct_query.json and edit_queries.json pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=module_dir / "reports" / "soft_constraint_taxonomy",
        help="Directory to store generated inventory and reports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for audit sampling.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_pref_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def normalize_hard_logic(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def dedupe_constraints(items: Sequence[ConstraintMention]) -> List[ConstraintMention]:
    by_key: Dict[Tuple[str, str, str], ConstraintMention] = {}
    intensity_rank = {"weak": 0, "medium": 1, "strong": 2}

    def source_rank(source: str) -> int:
        if "origin_query_text" in source:
            return 2
        if source.endswith("_en") or source == "origin_query_en":
            return 0
        return 1

    for item in items:
        key = (item.family, item.facet, item.anchor or "")
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = item
            continue
        if intensity_rank[item.intensity] > intensity_rank[existing.intensity]:
            existing.intensity = item.intensity
        if item.is_explicit:
            existing.is_explicit = True
        if item.source not in existing.source.split("+"):
            existing.source = f"{existing.source}+{item.source}"
        if source_rank(item.source) > source_rank(existing.source) or (
            source_rank(item.source) == source_rank(existing.source)
            and len(item.evidence_span) > len(existing.evidence_span)
        ):
            existing.evidence_span = item.evidence_span
    return sorted(by_key.values(), key=lambda x: (FAMILY_ORDER.index(x.family), x.facet, x.anchor or ""))


def detect_intensity(text: str) -> str:
    lowered = text.lower()
    strong_patterns = [
        "尽可能",
        "尽量多",
        "尽量少",
        "越",
        "最大化",
        "最小化",
        "一定要更",
        "嘎嘎出片",
        "as much as possible",
        "the shortest possible",
        "maximize",
        "minimize",
        "must be more",
    ]
    medium_patterns = [
        "尽量",
        "更",
        "相对",
        "少一点",
        "多一点",
        "顺一点",
        "轻松一些",
        "more",
        "less",
        "better",
        "reduce",
        "increase",
    ]
    weak_patterns = [
        "希望",
        "想",
        "想要",
        "喜欢",
        "比较喜欢",
        "可以",
        "适合",
        "prefer",
        "hope",
        "would like",
        "want",
    ]
    if any(token in lowered for token in strong_patterns):
        return "strong"
    if any(token in lowered for token in medium_patterns):
        return "medium"
    if any(token in lowered for token in weak_patterns):
        return "weak"
    return "weak"


def extract_target_poi(preference_py_items: Sequence[str]) -> Optional[str]:
    for text in preference_py_items:
        match = re.search(r"target_poi\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1).strip()
    return None


def classify_anchor_facet(anchor: str, is_hotel_context: bool = False) -> str:
    if is_hotel_context or "住" in anchor or "酒店" in anchor or "宾馆" in anchor or "旅馆" in anchor:
        return "hotel_anchor"
    if any(suffix in anchor for suffix in ["区", "街", "路", "商圈", "片区", "外滩", "景区周边"]):
        return "district_anchor"
    return "poi_anchor"


def detect_travel_intent_signals(query: Dict[str, Any], zh_text: str, en_text: str) -> List[str]:
    signals = [
        f"destination:{query.get('target_city', '')}",
        f"days:{query.get('days', '')}",
        f"people:{query.get('people_number', '')}",
    ]
    lowered = f"{zh_text} {en_text}".lower()
    if any(token in lowered for token in ["旅行", "游玩", "trip", "travel", "itinerary", "规划"]):
        signals.append("trip_planning_request")
    if any(token in lowered for token in ["玩", "visit", "explore"]):
        signals.append("visit_or_explore")
    return [item for item in signals if not item.endswith(":")]


def detect_hard_constraint_signals(query: Dict[str, Any], zh_text: str, en_text: str) -> List[str]:
    signals: List[str] = []
    hard_logic = normalize_hard_logic(query.get("hard_logic_py"))
    combined = f"{zh_text} {en_text}".lower()
    if query.get("days") is not None:
        signals.append("fixed_trip_length")
    if query.get("people_number") is not None:
        signals.append("fixed_party_size")
    if query.get("limit_rooms") or query.get("limits_room_type"):
        signals.append("room_requirement")
    if any("total_cost<=" in item for item in hard_logic) or any(token in combined for token in ["预算", "budget", "元"]):
        signals.append("budget_cap")
    if any(token in combined for token in ["高铁", "火车", "飞机", "train", "plane", "flight"]):
        signals.append("transport_mode_constraint")
    if any(token in combined for token in ["19点前", "before 19", "晚上", "上午", "midnight"]):
        signals.append("time_window_constraint")
    if any(token in combined for token in ["双床房", "大床房", "family room", "亲子房"]):
        signals.append("room_type_constraint")
    return sorted(set(signals))


def detect_must_visit_mentions(zh_text: str, explicit_anchor: Optional[str]) -> List[str]:
    mentions: List[str] = []
    if explicit_anchor:
        mentions.append(explicit_anchor)
    patterns = [
        r"想吃([^，。；]{2,24})",
        r"想去([^，。；]{2,24})",
        r"重点参观([^，。；]{2,36})",
        r"必去([^，。；]{2,24})",
        r"打卡([^，。；]{2,24})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, zh_text):
            candidate = match.group(1).strip("、,.，。 ")
            if candidate and candidate not in mentions:
                mentions.append(candidate)
    return mentions[:6]


def add_constraint(
    buffer: List[ConstraintMention],
    *,
    family: str,
    facet: str,
    polarity: str,
    anchor: Optional[str],
    evidence_span: str,
    source: str,
    is_explicit: bool,
    intensity_hint: Optional[str] = None,
) -> None:
    buffer.append(
        ConstraintMention(
            family=family,
            facet=facet,
            polarity=polarity,
            anchor=anchor,
            intensity=intensity_hint or detect_intensity(evidence_span),
            is_explicit=is_explicit,
            evidence_span=evidence_span,
            source=source,
        )
    )


def detect_explicit_constraints(query: Dict[str, Any]) -> Tuple[List[ConstraintMention], Optional[str], Optional[str]]:
    items: List[ConstraintMention] = []
    pref_en = normalize_pref_value(query.get("preference_en") or query.get("preference"))
    pref_py = normalize_pref_value(query.get("preference_py"))
    explicit_anchor = extract_target_poi(pref_py)
    seed_family: Optional[str] = None
    matched_seed: Optional[str] = None
    for pref in pref_en:
        pref_norm = pref.strip("[]'\" ")
        mapping = SEED_FAMILY_MAP.get(pref_norm)
        if not mapping:
            continue
        family, facet, polarity = mapping
        matched_seed = pref_norm
        seed_family = family
        add_constraint(
            items,
            family=family,
            facet=facet,
            polarity=polarity,
            anchor=explicit_anchor if family == "anchor_proximity" else None,
            evidence_span=pref_norm,
            source="explicit_preference",
            is_explicit=True,
            intensity_hint="strong",
        )
    if not items and pref_py:
        first_line = pref_py[0].splitlines()[0].strip()
        for seed_key, mapping in SEED_FAMILY_MAP.items():
            if seed_key.replace(" ", "_") in first_line or seed_key.split()[0] in first_line:
                family, facet, polarity = mapping
                matched_seed = seed_key
                seed_family = family
                add_constraint(
                    items,
                    family=family,
                    facet=facet,
                    polarity=polarity,
                    anchor=explicit_anchor if family == "anchor_proximity" else None,
                    evidence_span=first_line,
                    source="explicit_preference",
                    is_explicit=True,
                    intensity_hint="strong",
                )
                break
    return dedupe_constraints(items), seed_family, explicit_anchor


def detect_textual_constraints(
    text: str,
    *,
    source: str,
    is_explicit: bool = False,
) -> List[ConstraintMention]:
    items: List[ConstraintMention] = []
    lowered = text.lower()

    burden_patterns = [
        ("walk", "minimize", r"(少走路|步行.*(少|短)|不要走太多路|walk less|less walking|minimize walking)"),
        ("pace", "minimize", r"(轻松一点|轻松一些|舒服.*步行|慢慢逛|不要太赶|不当特种兵|not too rushed|easy pace|relaxed)"),
        ("activity_load", "minimize", r"(不要太累|行程不要太满|每天轻松|activity load)"),
    ]
    for facet, polarity, pattern in burden_patterns:
        for match in re.finditer(pattern, lowered):
            add_constraint(
                items,
                family="burden_reduction",
                facet=facet,
                polarity=polarity,
                anchor=None,
                evidence_span=text[match.start() : match.end()],
                source=source,
                is_explicit=is_explicit,
            )

    route_patterns = [
        ("travel_time", "minimize", r"(交通便利|交通时间.*少|less inner transports time|minimize average transport time|少折腾|少换乘|less transport time)"),
        ("route_smoothness", "minimize", r"(顺路|更顺|不要来回折返|少折返|路线.*(自然|顺)|集中一点|东一榔头西一棒槌|backtrack|smooth route|optimize the path)"),
        ("idle_gap", "minimize", r"(衔接自然|不要空档|large idle gap|idle gap|空档)"),
    ]
    for facet, polarity, pattern in route_patterns:
        for match in re.finditer(pattern, lowered):
            add_constraint(
                items,
                family="route_compactness",
                facet=facet,
                polarity=polarity,
                anchor=None,
                evidence_span=text[match.start() : match.end()],
                source=source,
                is_explicit=is_explicit,
            )

    anchor_patterns = [
        r"在([^，。；]{2,24}?)(附近|周边|周围)",
        r"住在?([^，。；]{2,24}?)(附近|周边)",
        r"以([^，。；]{2,24}?)(为起点|为核心|为主心骨)",
    ]
    for pattern in anchor_patterns:
        for match in re.finditer(pattern, text):
            anchor = match.group(1).strip("“”'\" ")
            if not anchor:
                continue
            is_hotel_context = "住" in match.group(0)
            add_constraint(
                items,
                family="anchor_proximity",
                facet=classify_anchor_facet(anchor, is_hotel_context),
                polarity="minimize",
                anchor=anchor,
                evidence_span=match.group(0),
                source=source,
                is_explicit=is_explicit,
            )

    if any(
        token in lowered
        for token in [
            "food_cost_ratio",
            "餐饮上花费",
            "餐饮预算",
            "花在吃",
            "花费尽可能多",
            "maximize the proportion of dining expenses",
            "spend as much as possible on dining",
        ]
    ):
        add_constraint(
            items,
            family="cost_allocation_preference",
            facet="food_share",
            polarity="maximize",
            anchor=None,
            evidence_span=text,
            source=source,
            is_explicit=is_explicit,
        )
    if any(
        token in lowered
        for token in [
            "住宿尽可能便宜",
            "尽量减少住宿费用",
            "减少住宿费用",
            "降低住宿成本",
            "降低住宿费用",
            "住宿便宜一些",
            "酒店就省些钱",
            "save money on hotels",
            "minimize accommodation cost",
        ]
    ):
        add_constraint(
            items,
            family="cost_allocation_preference",
            facet="accommodation_cost",
            polarity="minimize",
            anchor=None,
            evidence_span=text,
            source=source,
            is_explicit=is_explicit,
        )
    if any(token in lowered for token in ["把预算花在体验", "experience spend", "体验上花费"]):
        add_constraint(
            items,
            family="cost_allocation_preference",
            facet="experience_share",
            polarity="maximize",
            anchor=None,
            evidence_span=text,
            source=source,
            is_explicit=is_explicit,
        )

    theme_hits: List[Tuple[str, str]] = []
    # Origin English text is usually a machine translation. Generic words such
    # as "park", "food", "art", and "scenic" cause false theme preferences for
    # named POIs and cuisine hard constraints, so textual theme extraction is
    # driven by explicit Chinese preference wording instead.
    if not source.endswith("_en") and source != "origin_query_en":
        for facet, patterns in THEME_EXPLICIT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    theme_hits.append((facet, match.group(0)))
                    break
    mixed_theme = "或者" in text or " or " in lowered
    if mixed_theme and len({facet for facet, _ in theme_hits}) >= 2:
        add_constraint(
            items,
            family="theme_alignment",
            facet="mixed_theme",
            polarity="prefer",
            anchor=None,
            evidence_span=text,
            source=source,
            is_explicit=is_explicit,
        )
    else:
        seen_theme_facets = set()
        for facet, keyword in theme_hits:
            if facet in seen_theme_facets:
                continue
            seen_theme_facets.add(facet)
            add_constraint(
                items,
                family="theme_alignment",
                facet=facet,
                polarity="prefer",
                anchor=None,
                evidence_span=keyword,
                source=source,
                is_explicit=is_explicit,
            )

    richness_patterns = [
        ("attraction_count", "maximize", r"(更多景点|景点更多|多安排(?:几个|一些|点)?景点|多加(?:几个|一些|点)?景点|尽可能多的景点|尽可能多地参观景点|参观尽可能多的景点|more attractions|visit as many attractions as possible)"),
        ("category_diversity", "maximize", r"(类型更丰富|不要总是|不要都|不要全是|diversity|丰富一点)"),
        ("food_experience_count", "maximize", r"(多安排.*美食体验|多一点美食体验|更多美食|more food experiences|多融入.*美食)"),
    ]
    for facet, polarity, pattern in richness_patterns:
        for match in re.finditer(pattern, lowered):
            add_constraint(
                items,
                family="experience_richness",
                facet=facet,
                polarity=polarity,
                anchor=None,
                evidence_span=text[match.start() : match.end()],
                source=source,
                is_explicit=is_explicit,
            )

    return dedupe_constraints(items)


def merge_constraints(*groups: Sequence[ConstraintMention]) -> List[ConstraintMention]:
    merged: List[ConstraintMention] = []
    for group in groups:
        merged.extend(group)
    return dedupe_constraints(merged)


def load_query_records(query_root: Path) -> List[QueryRecord]:
    records: List[QueryRecord] = []
    for bucket_dir in sorted(query_root.iterdir()):
        if not bucket_dir.is_dir():
            continue
        for file_path in sorted(bucket_dir.glob("*.json")):
            query = load_json(file_path)
            zh_text = str(query.get("nature_language", "") or "")
            en_text = str(query.get("nature_language_en", "") or "")
            explicit_constraints, seed_family, explicit_anchor = detect_explicit_constraints(query)
            zh_constraints = detect_textual_constraints(zh_text, source="nature_language")
            en_constraints = detect_textual_constraints(en_text, source="nature_language_en")
            constraints = merge_constraints(explicit_constraints, zh_constraints, en_constraints)
            record = QueryRecord(
                uid=str(query["uid"]),
                source_bucket=bucket_dir.name,
                days=query.get("days"),
                people_number=query.get("people_number"),
                nature_language=zh_text,
                nature_language_en=en_text,
                hard_logic_py=normalize_hard_logic(query.get("hard_logic_py")),
                has_explicit_preference=bool(
                    normalize_pref_value(query.get("preference_en") or query.get("preference"))
                    or normalize_pref_value(query.get("preference_py"))
                ),
                preference_en=normalize_pref_value(query.get("preference_en") or query.get("preference")),
                preference_py=normalize_pref_value(query.get("preference_py")),
                soft_constraints=constraints,
                hard_constraint_signals=detect_hard_constraint_signals(query, zh_text, en_text),
                travel_intent_signals=detect_travel_intent_signals(query, zh_text, en_text),
                must_visit_mentions=detect_must_visit_mentions(zh_text, explicit_anchor),
                seed_preference_family=seed_family,
            )
            records.append(record)
    return records


def unique_examples(records: Sequence[QueryRecord], family: str, limit: int = 5) -> List[Tuple[str, str, str]]:
    examples: List[Tuple[str, str, str]] = []
    seen_uids = set()
    for record in records:
        if record.uid in seen_uids:
            continue
        matched = [item for item in record.soft_constraints if item.family == family]
        if not matched:
            continue
        seen_uids.add(record.uid)
        examples.append(
            (
                record.uid,
                record.source_bucket,
                (record.nature_language or record.nature_language_en).strip(),
            )
        )
        if len(examples) >= limit:
            break
    return examples


def bucket_summary(records: Sequence[QueryRecord]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[QueryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_bucket].append(record)
    rows: List[Dict[str, Any]] = []
    for bucket, items in sorted(grouped.items()):
        rows.append(
            {
                "bucket": bucket,
                "query_count": len(items),
                "explicit_preference_queries": sum(1 for item in items if item.has_explicit_preference),
                "soft_constraint_queries": sum(1 for item in items if item.soft_constraints),
            }
        )
    return rows


def family_stats(records: Sequence[QueryRecord]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        query_ids = set()
        facet_counts: Counter[str] = Counter()
        mention_count = 0
        example_spans: Counter[str] = Counter()
        for record in records:
            matched = [item for item in record.soft_constraints if item.family == family]
            if not matched:
                continue
            query_ids.add(record.uid)
            mention_count += len(matched)
            facet_counts.update(item.facet for item in matched)
            example_spans.update(item.evidence_span for item in matched if item.evidence_span)
        stats[family] = {
            "query_count": len(query_ids),
            "mention_count": mention_count,
            "facet_counts": dict(facet_counts),
            "common_evidence_spans": [span for span, _ in example_spans.most_common(10)],
        }
    return stats


def structure_check(records: Sequence[QueryRecord]) -> Dict[str, Any]:
    bucket_rows = bucket_summary(records)
    total = len(records)
    return {
        "total_queries": total,
        "total_soft_constraint_queries": sum(1 for item in records if item.soft_constraints),
        "bucket_rows": bucket_rows,
    }


def seed_check(records: Sequence[QueryRecord]) -> Dict[str, Any]:
    expected = {
        "preference0_base50": "experience_richness",
        "preference1_base50": "route_compactness",
        "preference2_base50": "route_compactness",
        "preference3_base50": "cost_allocation_preference",
        "preference4_base50": "cost_allocation_preference",
        "preference5_base50": "anchor_proximity",
    }
    per_bucket = {}
    for bucket, family in expected.items():
        bucket_records = [item for item in records if item.source_bucket == bucket]
        matched = [item for item in bucket_records if item.seed_preference_family == family]
        per_bucket[bucket] = {
            "expected_family": family,
            "total": len(bucket_records),
            "matched": len(matched),
            "pass": len(matched) == len(bucket_records),
        }
    return per_bucket


def decomposition_ok(record: QueryRecord) -> bool:
    return bool(
        record.hard_constraint_signals
        or record.soft_constraints
        or record.travel_intent_signals
        or record.must_visit_mentions
    )


def coverage_audit(records: Sequence[QueryRecord], seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    human_records = [item for item in records if item.source_bucket == "human"]
    other_records = [
        item
        for item in records
        if item.source_bucket in {"easy", "medium", "multiday_wo_p"} and not item.has_explicit_preference
    ]
    human_sample = rng.sample(human_records, min(50, len(human_records)))
    other_sample = rng.sample(other_records, min(50, len(other_records)))
    combined = human_sample + other_sample
    unresolved = [item.uid for item in combined if not decomposition_ok(item)]
    return {
        "human_sample_size": len(human_sample),
        "other_sample_size": len(other_sample),
        "combined_size": len(combined),
        "resolved": len(combined) - len(unresolved),
        "unresolved_uids": unresolved,
        "sample_uids": [item.uid for item in combined],
    }


def proxy_agreement(records: Sequence[QueryRecord], seed: int) -> Dict[str, Any]:
    rng = random.Random(seed + 1)
    pool = [
        item
        for item in records
        if item.source_bucket in {"human", "easy", "medium", "multiday_wo_p"}
            and not item.has_explicit_preference
    ]
    sample = rng.sample(pool, min(60, len(pool)))
    agreed = 0
    disagreements: List[Dict[str, Any]] = []
    for record in sample:
        zh_labels = {(item.family, item.facet) for item in detect_textual_constraints(record.nature_language, source="nature_language")}
        en_labels = {(item.family, item.facet) for item in detect_textual_constraints(record.nature_language_en, source="nature_language_en")}
        if zh_labels == en_labels:
            agreed += 1
        else:
            disagreements.append(
                {
                    "uid": record.uid,
                    "bucket": record.source_bucket,
                    "zh_labels": sorted(list(zh_labels)),
                    "en_labels": sorted(list(en_labels)),
                    "query": record.nature_language,
                }
            )
    return {
        "sample_size": len(sample),
        "agreement": agreed,
        "agreement_rate": round(agreed / len(sample), 4) if sample else 0.0,
        "disagreements": disagreements[:10],
        "sample_uids": [item.uid for item in sample],
    }


def anchor_specificity(anchor: Optional[str]) -> int:
    if not anchor:
        return 0
    generic = {"景点", "餐厅", "酒店", "区域", "附近", "周边", "location", "restaurant", "hotel"}
    if anchor in generic:
        return 1
    return 2


def intensity_rank(value: str) -> int:
    return {"weak": 0, "medium": 1, "strong": 2}.get(value, 0)


def classify_edit_delta(
    origin_constraints: Sequence[ConstraintMention],
    edit_constraints: Sequence[ConstraintMention],
) -> List[Dict[str, Any]]:
    origin_map: Dict[Tuple[str, str], List[ConstraintMention]] = defaultdict(list)
    for item in origin_constraints:
        origin_map[(item.family, item.facet)].append(item)

    decisions: List[Dict[str, Any]] = []
    seen_keys = set()
    for edit_item in edit_constraints:
        key = (edit_item.family, edit_item.facet)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        origin_items = origin_map.get(key, [])
        if not origin_items:
            decisions.append(
                {
                    "family": edit_item.family,
                    "facet": edit_item.facet,
                    "decision": "addition",
                    "reason": "origin lacks this family+facet",
                }
            )
            continue
        strongest_origin = max(
            origin_items,
            key=lambda item: (intensity_rank(item.intensity), anchor_specificity(item.anchor)),
        )
        if (
            intensity_rank(edit_item.intensity) > intensity_rank(strongest_origin.intensity)
            or anchor_specificity(edit_item.anchor) > anchor_specificity(strongest_origin.anchor)
            or edit_item.polarity != strongest_origin.polarity
        ):
            decisions.append(
                {
                    "family": edit_item.family,
                    "facet": edit_item.facet,
                    "decision": "strengthen",
                    "reason": "same family+facet exists and edit increases intensity or anchor specificity",
                }
            )
    return decisions


def load_edit_pairs(edit_root: Path) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    if not edit_root.exists():
        return pairs
    for plan_dir in sorted(edit_root.iterdir()):
        if not plan_dir.is_dir():
            continue
        query_path = plan_dir / "ct_query.json"
        edit_path = plan_dir / "edit_queries.json"
        if not query_path.exists() or not edit_path.exists():
            continue
        ct_query = load_json(query_path)
        edit_data = load_json(edit_path)
        origin_text = str(ct_query.get("nature_language", "") or "")
        origin_constraints = merge_constraints(
            *(
                [
                    detect_explicit_constraints(ct_query)[0],
                    detect_textual_constraints(origin_text, source="nature_language"),
                    detect_textual_constraints(str(ct_query.get("nature_language_en", "") or ""), source="nature_language_en"),
                ]
            )
        )
        for scene_data in edit_data.get("scenes", {}).values():
            for query_item in scene_data.get("queries", []):
                edit_text = str(query_item.get("text", "") or "")
                edit_constraints = detect_textual_constraints(edit_text, source="edit_query")
                if not edit_constraints:
                    continue
                decisions = classify_edit_delta(origin_constraints, edit_constraints)
                if not decisions:
                    continue
                pairs.append(
                    {
                        "plan_id": edit_data.get("plan_id"),
                        "scene_type": scene_data.get("scene_type"),
                        "origin_query": origin_text,
                        "edit_query": edit_text,
                        "origin_constraints": [item.to_dict() for item in origin_constraints],
                        "edit_constraints": [item.to_dict() for item in edit_constraints],
                        "decisions": decisions,
                    }
                )
    return pairs


def interface_check(edit_pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sampled = list(edit_pairs[:20])
    decision_counts: Counter[str] = Counter()
    for pair in sampled:
        for decision in pair["decisions"]:
            decision_counts.update([decision["decision"]])
    return {
        "sample_size": len(sampled),
        "decision_counts": dict(decision_counts),
        "pairs": sampled,
    }


def build_summary_payload(
    records: Sequence[QueryRecord],
    family_summary: Dict[str, Dict[str, Any]],
    coverage: Dict[str, Any],
    agreement: Dict[str, Any],
    interface: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "total_queries": len(records),
        "explicit_preference_queries": sum(1 for item in records if item.has_explicit_preference),
        "soft_constraint_queries": sum(1 for item in records if item.soft_constraints),
        "bucket_summary": bucket_summary(records),
        "family_summary": family_summary,
        "coverage_audit": coverage,
        "proxy_agreement": agreement,
        "interface_check": {
            "sample_size": interface["sample_size"],
            "decision_counts": interface["decision_counts"],
        },
    }


def render_taxonomy_report(
    records: Sequence[QueryRecord],
    family_summary: Dict[str, Dict[str, Any]],
    structure: Dict[str, Any],
    seed_results: Dict[str, Any],
    coverage: Dict[str, Any],
    agreement: Dict[str, Any],
    interface: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Soft Constraint Taxonomy Analysis Report")
    lines.append("")
    lines.append("## 1. Query Overview")
    lines.append("")
    lines.append(f"- Total hf_queries: **{structure['total_queries']}**")
    lines.append(f"- Queries with at least one detected soft constraint: **{structure['total_soft_constraint_queries']}**")
    lines.append(f"- Queries with explicit preference fields: **{sum(1 for item in records if item.has_explicit_preference)}**")
    lines.append("")
    lines.append("| bucket | queries | explicit preference | detected soft constraints |")
    lines.append("|---|---:|---:|---:|")
    for row in structure["bucket_rows"]:
        lines.append(
            f"| {row['bucket']} | {row['query_count']} | {row['explicit_preference_queries']} | {row['soft_constraint_queries']} |"
        )
    lines.append("")
    lines.append("## 2. Soft Constraint Taxonomy")
    lines.append("")
    lines.append("| family | query count | mention count | top facets | metric hook |")
    lines.append("|---|---:|---:|---|---|")
    for family in FAMILY_ORDER:
        summary = family_summary[family]
        top_facets = ", ".join(
            f"{facet} ({count})"
            for facet, count in Counter(summary["facet_counts"]).most_common(3)
        ) or "-"
        lines.append(
            f"| {family} | {summary['query_count']} | {summary['mention_count']} | {top_facets} | {METRIC_HOOKS[family]['candidate_metric']} |"
        )
    lines.append("")
    for family in FAMILY_ORDER:
        summary = family_summary[family]
        hook = METRIC_HOOKS[family]
        lines.append(f"### {family}")
        lines.append("")
        lines.append(f"- Facets: {', '.join(FACETS_BY_FAMILY[family])}")
        lines.append(f"- Query count: {summary['query_count']}")
        lines.append(f"- Mention count: {summary['mention_count']}")
        lines.append(f"- Candidate metric: {hook['candidate_metric']}")
        lines.append(f"- Direction: {hook['direction']}")
        lines.append(f"- Reusable signals/code: {', '.join(hook['existing_reusable_signals'])}; {', '.join(hook['reusable_code'])}")
        lines.append(f"- What counts as strengthen: {hook['strengthen_rule']}")
        lines.append(f"- Known gaps: {hook['known_gaps']}")
        lines.append(f"- Dataset generation implication: {GENERATION_IMPLICATIONS[family]}")
        lines.append("- Example queries:")
        for uid, bucket, text in unique_examples(records, family, limit=5):
            lines.append(f"  - `{uid}` ({bucket}): {text}")
        lines.append("")
    lines.append("## 3. Seed Preference Alignment")
    lines.append("")
    lines.append("| bucket | expected family | matched | total | pass |")
    lines.append("|---|---|---:|---:|---|")
    for bucket, result in seed_results.items():
        lines.append(
            f"| {bucket} | {result['expected_family']} | {result['matched']} | {result['total']} | {result['pass']} |"
        )
    lines.append("")
    lines.append("## 4. Validation Snapshot")
    lines.append("")
    lines.append(f"- Coverage audit sample: {coverage['combined_size']} queries, resolved {coverage['resolved']}, unresolved {len(coverage['unresolved_uids'])}")
    lines.append(f"- Proxy zh/en agreement sample: {agreement['sample_size']} queries, agreement rate {agreement['agreement_rate']:.2%}")
    lines.append(f"- Origin/edit interface sample: {interface['sample_size']} pairs, decisions {interface['decision_counts']}")
    lines.append("")
    lines.append("## 5. Dataset Generation Implications")
    lines.append("")
    lines.append("- Addition rule: origin lacks the same `family + facet`, while edit introduces it.")
    lines.append("- Strengthen rule: origin already has the same `family + facet`, while edit makes it stronger through intensity, anchor specificity, or more explicit comparison wording.")
    lines.append("- Recommended next step: use this taxonomy to add family-aware edit templates before binding full evaluators.")
    lines.append("")
    return "\n".join(lines)


def find_record(records: Sequence[QueryRecord], predicate) -> Optional[QueryRecord]:
    for record in records:
        if predicate(record):
            return record
    return None


def render_decision_log(
    records: Sequence[QueryRecord],
    coverage: Dict[str, Any],
    agreement: Dict[str, Any],
    interface: Dict[str, Any],
) -> str:
    examples = {
        "hard_budget_not_soft": find_record(records, lambda r: "预算" in r.nature_language and not any(c.family == "cost_allocation_preference" for c in r.soft_constraints)),
        "hotel_requirement_not_soft": find_record(records, lambda r: ("亲子房" in r.nature_language or "双床房" in r.nature_language) and not any(c.family == "theme_alignment" and c.facet == "family" for c in r.soft_constraints)),
        "must_visit_poi_not_soft": find_record(records, lambda r: "重点参观" in r.nature_language and not any(c.family == "anchor_proximity" for c in r.soft_constraints)),
        "theme_kept_as_soft": find_record(records, lambda r: any(c.family == "theme_alignment" for c in r.soft_constraints)),
        "burden_example": find_record(records, lambda r: any(c.family == "burden_reduction" for c in r.soft_constraints)),
    }
    lines: List[str] = []
    lines.append("# Taxonomy Decision Log")
    lines.append("")
    lines.append("## Default Rules")
    lines.append("")
    lines.append("- Hard constraints from `hard_logic_py` remain outside the soft-constraint taxonomy.")
    lines.append("- Specific POI or restaurant requests remain `must_visit` unless they also express proximity, routing smoothness, or theme preference.")
    lines.append("- Theme preferences such as 人文 / 自然 / 美食 / 拍照 / 亲子 / 购物 remain soft constraints even when no evaluator is fully implemented yet.")
    lines.append("- Hotel room count / room type remains hard unless the query explicitly asks for lower accommodation cost or hotel proximity to an anchor.")
    lines.append("")
    lines.append("## Boundary Examples")
    lines.append("")
    if examples["hard_budget_not_soft"]:
        lines.append(f"- Hard budget only, not soft: `{examples['hard_budget_not_soft'].uid}` -> {examples['hard_budget_not_soft'].nature_language}")
    if examples["hotel_requirement_not_soft"]:
        lines.append(f"- Room requirement only, not family theme: `{examples['hotel_requirement_not_soft'].uid}` -> {examples['hotel_requirement_not_soft'].nature_language}")
    if examples["must_visit_poi_not_soft"]:
        lines.append(f"- Must-visit POI list stays outside soft taxonomy: `{examples['must_visit_poi_not_soft'].uid}` -> {examples['must_visit_poi_not_soft'].nature_language}")
    if examples["theme_kept_as_soft"]:
        lines.append(f"- Theme preference retained as soft: `{examples['theme_kept_as_soft'].uid}` -> {examples['theme_kept_as_soft'].nature_language}")
    if examples["burden_example"]:
        lines.append(f"- Burden reduction retained as soft: `{examples['burden_example'].uid}` -> {examples['burden_example'].nature_language}")
    lines.append("")
    lines.append("## Coverage Audit")
    lines.append("")
    lines.append(f"- Sample size: {coverage['combined_size']} (human={coverage['human_sample_size']}, other={coverage['other_sample_size']})")
    lines.append(f"- Resolved by current decomposition: {coverage['resolved']}")
    lines.append(f"- Unresolved uids: {coverage['unresolved_uids'] if coverage['unresolved_uids'] else 'none'}")
    lines.append(f"- Suggested manual review sample ids: {', '.join(coverage['sample_uids'][:20])}")
    lines.append("")
    lines.append("## Proxy Agreement")
    lines.append("")
    lines.append("- This is a zh/en heuristic proxy, not a substitute for human double review.")
    lines.append(f"- Sample size: {agreement['sample_size']}")
    lines.append(f"- Agreement rate: {agreement['agreement_rate']:.2%}")
    if agreement["disagreements"]:
        lines.append("- Representative disagreements:")
        for item in agreement["disagreements"][:5]:
            lines.append(f"  - `{item['uid']}` zh={item['zh_labels']} en={item['en_labels']} :: {item['query']}")
    lines.append("")
    lines.append("## Addition vs Strengthen Samples")
    lines.append("")
    for pair in interface["pairs"][:10]:
        decisions = ", ".join(f"{item['family']}:{item['facet']}->{item['decision']}" for item in pair["decisions"])
        lines.append(f"- `{pair['plan_id']}` [{pair['scene_type']}]")
        lines.append(f"  - Origin: {pair['origin_query']}")
        lines.append(f"  - Edit: {pair['edit_query']}")
        lines.append(f"  - Decisions: {decisions}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    query_root = args.query_root.resolve()
    edit_root = args.edit_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_query_records(query_root)
    family_summary = family_stats(records)
    structure = structure_check(records)
    seed_results = seed_check(records)
    coverage = coverage_audit(records, args.seed)
    agreement = proxy_agreement(records, args.seed)
    edit_pairs = load_edit_pairs(edit_root)
    interface = interface_check(edit_pairs)

    inventory_path = output_dir / "soft_constraint_inventory.jsonl"
    report_path = output_dir / "soft_constraint_taxonomy_report.md"
    decision_log_path = output_dir / "taxonomy_decision_log.md"
    summary_path = output_dir / "summary.json"

    write_jsonl(inventory_path, [record.to_inventory_row() for record in records])
    write_text(
        report_path,
        render_taxonomy_report(
            records,
            family_summary,
            structure,
            seed_results,
            coverage,
            agreement,
            interface,
        ),
    )
    write_text(
        decision_log_path,
        render_decision_log(records, coverage, agreement, interface),
    )
    write_json(
        summary_path,
        build_summary_payload(records, family_summary, coverage, agreement, interface),
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "inventory_rows": len(records),
                "total_queries": len(records),
                "soft_constraint_queries": sum(1 for item in records if item.soft_constraints),
                "interface_pairs_checked": interface["sample_size"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
