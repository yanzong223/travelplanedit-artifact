"""Implementation engine for target-aware edit-scope attribution.

New callers should use :mod:`evaluation.edit_scope`, which adds the stable
Dependency-Closed Counterfactual Attribution (DCCA) API, algorithm version,
and result-contract validation. This module retains the historical public and
private names needed by existing reports and regression tests.

The existing Level3 diff tells us how much changed.  This module adds the
missing attribution layer: which changed activities directly realize the edit
target, and how far the remaining changes spill through itinerary dependencies.
The implementation is deterministic and deliberately conservative. Named
entities use exact grounding; global numeric, temporal, semantic, and day-level
constraints use explicit type-aware proxies with reported confidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .benchmark.diffing import infer_edit_sequence
from .benchmark.models import LogicalConstraintObject
from .benchmark.verifiers import _concept_funcs, verify_constraints
from .route_evidence import MODES, RouteEvidenceCache
from data_clean.rules import evaluate_plan


_NAME_KEYS = {"poi_name", "anchor_name", "activity_name", "position", "name"}
_GLOBAL_TYPES = {"budget_total", "ticket_budget_total", "day_count", "people_count"}
_TRANSPORT_NUMERIC_TYPES = {
    "innercity_transport_cost_total",
    "innercity_transport_duration_total",
    "walking_distance_total",
}

ROLLBACK_FAMILY_VERSION = "dcca-rollback-family-v1"


def _canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible evidence without depending on dict insertion order."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Activity:
    ref: str
    day: int
    index: int
    kind: str
    name: str
    start_time: str
    end_time: str
    cost: float
    room_type: str


@dataclass(frozen=True, slots=True)
class FullGateResult:
    """Level1+Level2 validation result supplied by the owning evaluation flow."""

    passed: bool
    components: Mapping[str, bool]
    reason: str = ""


FullGateValidator = Callable[[dict[str, Any]], FullGateResult]
ViolationSignature = Counter[tuple[str, int | None, str, str]]
CrossCityGapSignature = Counter[tuple[str, str, str, str]]

_SUPPORTED_CITY_DIRECTORIES = {
    "beijing": "北京",
    "shanghai": "上海",
    "nanjing": "南京",
    "suzhou": "苏州",
    "hangzhou": "杭州",
    "shenzhen": "深圳",
    "chengdu": "成都",
    "wuhan": "武汉",
    "guangzhou": "广州",
    "chongqing": "重庆",
}


def _activities(plan: dict[str, Any]) -> list[Activity]:
    result: list[Activity] = []
    for day_index, day in enumerate(plan.get("itinerary", []), start=1):
        day_number = int(day.get("day", day_index) or day_index)
        for index, item in enumerate(day.get("activities", [])):
            name = str(item.get("position") or item.get("TrainID") or item.get("end") or "")
            raw_cost = item.get("cost", item.get("price", 0))
            try:
                cost = float(raw_cost or 0)
            except (TypeError, ValueError):
                cost = 0.0
            result.append(Activity(
                f"day{day_number}_act{index}", day_number, index,
                str(item.get("type", "")), name,
                str(item.get("start_time", "")), str(item.get("end_time", "")), cost,
                str(item.get("room_type", "")),
            ))
    return result


def _raw_activities(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for day_index, day in enumerate(plan.get("itinerary", []), start=1):
        day_number = int(day.get("day", day_index) or day_index)
        for index, item in enumerate(day.get("activities", [])):
            result[f"day{day_number}_act{index}"] = item
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_transport_metric(activity: dict[str, Any], metric: str) -> float:
    total = 0.0
    transports = activity.get("transports", [])
    if not isinstance(transports, list):
        return total
    for leg in transports:
        if not isinstance(leg, dict):
            continue
        if metric == "cost":
            total += _number(leg.get("cost")) or 0.0
        elif metric == "duration_minutes":
            try:
                start_hour, start_minute = str(leg.get("start_time", "")).split(":")[:2]
                end_hour, end_minute = str(leg.get("end_time", "")).split(":")[:2]
                start = int(start_hour) * 60 + int(start_minute)
                end = int(end_hour) * 60 + int(end_minute)
                total += max(end - start, 0)
            except (TypeError, ValueError):
                continue
        elif metric == "walking_distance_km":
            mode = str(leg.get("mode", "") or "").casefold()
            if mode in {"walk", "walking", "步行"}:
                total += _number(leg.get("distance")) or 0.0
    return total


def _numeric_edit_is_favorable(
    constraint_types: set[str],
    origin_activity: Activity,
    edited_activity: Activity,
    origin_raw: dict[str, Any],
    edited_raw: dict[str, Any],
) -> bool:
    if "ticket_budget_total" in constraint_types and origin_activity.kind == "attraction":
        if edited_activity.cost < origin_activity.cost:
            return True
    if constraint_types & {"budget_total", "activity_budget_limit", "anchor_bundle_budget_limit"}:
        origin_total = origin_activity.cost + _nested_transport_metric(origin_raw, "cost")
        edited_total = edited_activity.cost + _nested_transport_metric(edited_raw, "cost")
        if edited_total < origin_total:
            return True
    metric_by_type = {
        "innercity_transport_cost_total": "cost",
        "innercity_transport_duration_total": "duration_minutes",
        "walking_distance_total": "walking_distance_km",
    }
    for constraint_type, metric in metric_by_type.items():
        if constraint_type not in constraint_types:
            continue
        if _nested_transport_metric(edited_raw, metric) < _nested_transport_metric(origin_raw, metric):
            return True
    return False


def _fine_semantic_type(activity: dict[str, Any], city: Any) -> str | None:
    try:
        if str(activity.get("type", "")) == "attraction":
            return str(_concept_funcs()["attraction_type"](activity, city))
    except Exception:
        return None
    return None


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str) and value.strip():
        yield value.strip()
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)


def _target_spec(constraints: list[dict[str, Any]]) -> tuple[set[str], set[str], set[int]]:
    """Return explicit entity names and days, excluding broad/global values."""
    names: set[str] = set()
    forbidden_names: set[str] = set()
    days: set[int] = set()
    for constraint in constraints:
        ctype = str(constraint.get("type", ""))
        target = constraint.get("target") if isinstance(constraint.get("target"), dict) else {}
        for key, value in target.items():
            if key in _NAME_KEYS:
                names.update(_strings(value))
            elif key in {"day", "day_number", "target_day"}:
                try:
                    days.add(int(value))
                except (TypeError, ValueError):
                    pass
        # Required/forbidden named-entity constraints store the name in value.
        if ctype.startswith(("required_", "forbidden_")) and ctype.endswith("_name"):
            values = set(_strings(constraint.get("value")))
            if ctype.startswith("forbidden_"):
                forbidden_names.update(values)
            else:
                names.update(values)
        if ctype not in _GLOBAL_TYPES:
            params = constraint.get("params") if isinstance(constraint.get("params"), dict) else {}
            for key, value in params.items():
                if key in _NAME_KEYS:
                    names.update(_strings(value))
    return names, forbidden_names, days


def _kind_matches(actual: str, requested: str) -> bool:
    requested = requested.casefold()
    actual = actual.casefold()
    if requested in {"intercity_transport", "transport"}:
        return actual in {"train", "airplane"}
    if requested in {"meal", "restaurant"}:
        return actual in {"breakfast", "breakfest", "lunch", "dinner", "restaurant"}
    return actual == requested


def _constraint_kinds(constraints: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for constraint in constraints:
        target = constraint.get("target") if isinstance(constraint.get("target"), dict) else {}
        value = target.get("activity_type")
        if isinstance(value, str) and value:
            result.add(value)
    return result


def _confidence(modes: set[str]) -> str:
    if modes and modes <= {"exact_entity"}:
        return "high"
    if modes & {"semantic_type_proxy"}:
        return "low"
    return "medium"


def _dependency_graph(records: list[Activity]) -> dict[str, set[str]]:
    """Build observable itinerary dependencies.

    Consecutive activities share a temporal/transport boundary.  Repeated named
    entities (usually accommodation across days) form a cross-day continuity
    dependency.  Edges are undirected because the metric measures impact
    distance rather than causal direction.
    """
    graph = {item.ref: set() for item in records}
    by_day: dict[int, list[Activity]] = {}
    by_entity: dict[tuple[str, str], list[Activity]] = {}
    for item in records:
        by_day.setdefault(item.day, []).append(item)
        if item.name:
            by_entity.setdefault((item.kind, item.name.casefold()), []).append(item)
    for items in by_day.values():
        items.sort(key=lambda item: item.index)
        for left, right in zip(items, items[1:]):
            graph[left.ref].add(right.ref)
            graph[right.ref].add(left.ref)
    for items in by_entity.values():
        for left, right in zip(items, items[1:]):
            graph[left.ref].add(right.ref)
            graph[right.ref].add(left.ref)
    return graph


def _distances(graph: dict[str, set[str]], seeds: set[str]) -> dict[str, int]:
    distances = {seed: 0 for seed in seeds if seed in graph}
    queue = deque(distances)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _day_activities(plan: dict[str, Any], day_number: int) -> list[dict[str, Any]] | None:
    for index, day in enumerate(plan.get("itinerary", []), start=1):
        if int(day.get("day", index) or index) == day_number:
            return day.get("activities", [])
    return None


def _split_ref(ref: str) -> tuple[int, int]:
    day_text, act_text = ref.removeprefix("day").split("_act", 1)
    return int(day_text), int(act_text)


def _rollback_unit(
    edited_plan: dict[str, Any],
    unit_kind: str,
    unit_ref: str,
    origin_raw: dict[str, dict[str, Any]],
    edited_raw: dict[str, dict[str, Any]],
    origin_to_edited: dict[str, str],
    same_day_reorder_refs: set[str] | None = None,
) -> dict[str, Any] | None:
    plan = copy.deepcopy(edited_plan)
    if unit_kind == "inserted":
        day, index = _split_ref(unit_ref)
        activities = _day_activities(plan, day)
        if activities is None or index >= len(activities):
            return None
        activities.pop(index)
        return plan

    origin_activity = origin_raw.get(unit_ref)
    if origin_activity is None:
        return None
    origin_day, origin_index = _split_ref(unit_ref)
    edited_ref = origin_to_edited.get(unit_ref)
    if edited_ref:
        edited_day, edited_index = _split_ref(edited_ref)
        edited_activities = _day_activities(plan, edited_day)
        if edited_activities is None or edited_index >= len(edited_activities):
            return None
        if (
            edited_day == origin_day
            and unit_ref in (same_day_reorder_refs or set())
            and edited_index != origin_index
        ):
            edited_activities.pop(edited_index)
            edited_activities.insert(
                min(origin_index, len(edited_activities)),
                copy.deepcopy(origin_activity),
            )
        elif edited_day == origin_day:
            edited_activities[edited_index] = copy.deepcopy(origin_activity)
        else:
            edited_activities.pop(edited_index)
            origin_activities = _day_activities(plan, origin_day)
            if origin_activities is None:
                return None
            origin_activities.insert(min(origin_index, len(origin_activities)), copy.deepcopy(origin_activity))
        return plan

    origin_activities = _day_activities(plan, origin_day)
    if origin_activities is None:
        return None
    origin_activities.insert(min(origin_index, len(origin_activities)), copy.deepcopy(origin_activity))
    return plan


def _transport_dependency_closure_units(
    unit_kind: str,
    unit_ref: str,
    *,
    atomic_types_by_origin: Mapping[str, set[str]],
    changed_origin: set[str],
    inserted_edited: set[str],
    origin_to_edited: Mapping[str, str],
    edited_to_origin: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Return the minimal adjacent activity closure for an inbound route edit.

    Transport is stored on the destination activity.  Reverting that field
    alone can therefore create a false mismatch when its predecessor was
    inserted, deleted, or replaced.  The closure adds only the changed
    predecessor needed to reconstruct the original boundary.
    """
    if (
        unit_kind != "origin"
        or "change_transport" not in atomic_types_by_origin.get(unit_ref, set())
    ):
        return []
    edited_ref = origin_to_edited.get(unit_ref)
    if not edited_ref:
        return []
    origin_day, origin_index = _split_ref(unit_ref)
    edited_day, edited_index = _split_ref(edited_ref)
    if origin_index <= 0 or edited_index <= 0:
        return []

    origin_previous = f"day{origin_day}_act{origin_index - 1}"
    edited_previous = f"day{edited_day}_act{edited_index - 1}"
    members: list[tuple[str, str]] = []

    if edited_previous in inserted_edited:
        members.append(("inserted", edited_previous))
        mapped_previous = None
    else:
        mapped_previous = edited_to_origin.get(edited_previous)
        if mapped_previous in changed_origin:
            members.append(("origin", mapped_previous))

    if origin_previous in changed_origin and origin_previous != mapped_previous:
        members.append(("origin", origin_previous))

    return list(dict.fromkeys(members))


def _verification_signature(
    plan: dict[str, Any], constraints: list[dict[str, Any]],
) -> tuple[dict[str, bool], ViolationSignature]:
    objects = [LogicalConstraintObject(**item) for item in constraints]
    constraint_pass = {item.constraint_id: bool(item.passed) for item in verify_constraints(plan, objects)}
    feasibility = evaluate_plan(plan)
    violations: ViolationSignature = Counter()
    for item in feasibility.hygiene_violations + feasibility.quality_violations:
        evidence = getattr(item, "evidence", None)
        evidence_key = json.dumps(
            evidence if isinstance(evidence, dict) else {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        violations[(
            str(item.code),
            getattr(item, "day", None),
            str(getattr(item, "message", "") or ""),
            evidence_key,
        )] += 1
    return constraint_pass, violations


def _new_violation_instances(
    after: ViolationSignature,
    before: ViolationSignature,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for (code, day, message, evidence_key), count in sorted((after - before).items()):
        result.append({
            "code": code,
            "day": day,
            "message": message,
            "evidence": json.loads(evidence_key),
            "count": count,
        })
    return result


def _gate_regressions(
    baseline: FullGateResult | None,
    candidate: FullGateResult | None,
) -> list[str]:
    if baseline is None or candidate is None:
        return []
    return sorted(
        name
        for name, passed in baseline.components.items()
        if passed and not bool(candidate.components.get(name))
    )


def _activity_endpoint(activity: dict[str, Any], *, arrival: bool) -> str:
    if str(activity.get("type", "")) in {"train", "airplane"}:
        return str(activity.get("end" if arrival else "start", "") or "")
    return str(activity.get("position", "") or "")


def _endpoint_matches(left: str, right: str) -> bool:
    normalize = lambda value: "".join(str(value or "").casefold().split())
    left, right = normalize(left), normalize(right)
    return bool(left and right) and left == right


def _normalize_endpoint_name(value: Any) -> str:
    return "".join(str(value or "").casefold().split())


@lru_cache(maxsize=1)
def _endpoint_city_index() -> Mapping[str, frozenset[str]]:
    """Load exact endpoint-to-city evidence from the bundled ten-city snapshot.

    POI names cover local activities, railway stations, and airports. Train
    manifests add an explicit, independently auditable station mapping. A name
    observed in multiple cities remains ambiguous and is never guessed.
    """
    candidates: dict[str, set[str]] = {}

    def add(name: Any, city: str) -> None:
        normalized = _normalize_endpoint_name(name)
        if normalized and city in _SUPPORTED_CITY_DIRECTORIES.values():
            candidates.setdefault(normalized, set()).add(city)

    database_root = (
        PROJECT_ROOT
        / "Chinatravel/ChinaTravel/chinatravel/environment/database"
    )
    poi_root = database_root / "poi"
    for directory, city in _SUPPORTED_CITY_DIRECTORIES.items():
        path = poi_root / directory / "poi.json"
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    add(row.get("name"), city)

    train_root = database_root / "intercity_transport/train"
    for path in sorted(train_root.glob("from_*_to_*.json")):
        route = path.stem.removeprefix("from_")
        if "_to_" not in route:
            continue
        start_city, end_city = route.split("_to_", 1)
        if (
            start_city not in _SUPPORTED_CITY_DIRECTORIES.values()
            or end_city not in _SUPPORTED_CITY_DIRECTORIES.values()
        ):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    add(row.get("From"), start_city)
                    add(row.get("To"), end_city)

    return {
        name: frozenset(sorted(cities))
        for name, cities in sorted(candidates.items())
    }


def _resolve_endpoint_city(name: Any) -> str | None:
    cities = _endpoint_city_index().get(_normalize_endpoint_name(name), frozenset())
    return next(iter(cities)) if len(cities) == 1 else None


def _cross_city_gap_signature(plan: dict[str, Any]) -> CrossCityGapSignature:
    """Count stable, reliably grounded cross-city adjacency boundaries."""
    gaps: CrossCityGapSignature = Counter()
    for _, activities in _iter_plan_days(plan):
        for previous, current in zip(activities, activities[1:]):
            left = _activity_endpoint(previous, arrival=True)
            right = _activity_endpoint(current, arrival=False)
            left_city = _resolve_endpoint_city(left)
            right_city = _resolve_endpoint_city(right)
            if not left_city or not right_city or left_city == right_city:
                continue
            gaps[(
                _normalize_endpoint_name(left),
                left_city,
                _normalize_endpoint_name(right),
                right_city,
            )] += 1
    return gaps


def _new_cross_city_gap_instances(
    after: CrossCityGapSignature,
    before: CrossCityGapSignature,
) -> list[dict[str, Any]]:
    return [
        {
            "start": start,
            "start_city": start_city,
            "end": end,
            "end_city": end_city,
            "count": count,
        }
        for (start, start_city, end, end_city), count
        in sorted((after - before).items())
    ]


def _transport_continuity_issues(plan: dict[str, Any]) -> set[str]:
    issues: set[str] = set()
    for day_index, day in enumerate(plan.get("itinerary", []), start=1):
        activities = day.get("activities", [])
        for index in range(1, len(activities)):
            previous, current = activities[index - 1], activities[index]
            transports = current.get("transports", [])
            if not isinstance(transports, list) or not transports:
                expected_start = _activity_endpoint(previous, arrival=True)
                expected_end = _activity_endpoint(current, arrival=False)
                if expected_start and expected_end and not _endpoint_matches(expected_start, expected_end):
                    issues.add(
                        f"day{day.get('day', day_index)}_act{index}:missing_inbound:"
                        f"{expected_start}->{expected_end}"
                    )
                continue
            legs = [item for item in transports if isinstance(item, dict)]
            if not legs:
                continue
            expected_start = _activity_endpoint(previous, arrival=True)
            expected_end = _activity_endpoint(current, arrival=False)
            first_start = str(legs[0].get("start", "") or "")
            last_end = str(legs[-1].get("end", "") or "")
            if expected_start and first_start and not _endpoint_matches(expected_start, first_start):
                issues.add(f"day{day.get('day', day_index)}_act{index}:inbound_start")
            if expected_end and last_end and not _endpoint_matches(expected_end, last_end):
                issues.add(f"day{day.get('day', day_index)}_act{index}:inbound_end")
            for leg_index in range(1, len(legs)):
                before = str(legs[leg_index - 1].get("end", "") or "")
                after = str(legs[leg_index].get("start", "") or "")
                if before and after and not _endpoint_matches(before, after):
                    issues.add(f"day{day.get('day', day_index)}_act{index}:leg{leg_index}")
    return issues


def _authorized_new_day_unit(
    unit_kind: str,
    ref: str,
    origin_to_edited: dict[str, str],
    edited_by_ref: dict[str, Activity],
    added_days: set[int],
    edited_plan: dict[str, Any],
    constraints: list[dict[str, Any]],
) -> bool:
    edited_ref = ref if unit_kind == "inserted" else origin_to_edited.get(ref, "")
    item = edited_by_ref.get(edited_ref)
    if not item or item.day not in added_days:
        return False
    if item.kind not in {
        "breakfast", "breakfest", "lunch", "dinner", "accommodation",
        "attraction", "train", "airplane",
    }:
        return False
    day_activities = _day_activities(edited_plan, item.day) or []
    if item.kind == "attraction":
        caps = [int(c.get("value")) for c in constraints if c.get("type") == "daily_poi_cap"]
        cap = min(caps) if caps else 1
        if sum(str(a.get("type", "")) == "attraction" for a in day_activities) > cap:
            return False
        # A repeated attraction is not authorized merely because it is on a new day.
        occurrences = sum(
            str(a.get("position", "")) == item.name
            for day in edited_plan.get("itinerary", []) for a in day.get("activities", [])
        )
        if item.name and occurrences > 1:
            return False
    if item.kind in {"breakfast", "breakfest", "lunch", "dinner"}:
        if sum(str(a.get("type", "")) == item.kind for a in day_activities) > 1:
            return False
    if item.kind == "accommodation" and sum(str(a.get("type", "")) == "accommodation" for a in day_activities) > 1:
        return False
    if item.kind in {"train", "airplane"} and sum(str(a.get("type", "")) in {"train", "airplane"} for a in day_activities) > 1:
        return False
    return True


def _repair_inbound_from_origin(
    counterfactual: dict[str, Any], origin_plan: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Reuse an observed origin inbound route; never synthesize a route."""
    repaired = copy.deepcopy(counterfactual)
    changed = False
    origin_pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _, activities in _iter_plan_days(origin_plan):
        for index in range(1, len(activities)):
            key = (_activity_endpoint(activities[index - 1], arrival=True), _activity_endpoint(activities[index], arrival=False))
            transports = activities[index].get("transports", [])
            if isinstance(transports, list) and transports:
                origin_pairs[key] = copy.deepcopy(transports)
    for _, activities in _iter_plan_days(repaired):
        for index in range(1, len(activities)):
            key = (_activity_endpoint(activities[index - 1], arrival=True), _activity_endpoint(activities[index], arrival=False))
            if key in origin_pairs:
                current_issues = _transport_continuity_issues({"itinerary": [{"day": 1, "activities": [activities[index - 1], activities[index]]}]})
                if current_issues:
                    activities[index]["transports"] = copy.deepcopy(origin_pairs[key])
                    changed = True
    return repaired, changed


def _route_request(plan: dict[str, Any], issue: str) -> tuple[str, str, str, str] | None:
    try:
        prefix = issue.split(":", 1)[0]
        day_token, act_token = prefix.split("_act")
        day_number, activity_index = int(day_token.removeprefix("day")), int(act_token)
    except (ValueError, IndexError):
        return None
    activities = _day_activities(plan, day_number)
    if not activities or activity_index <= 0 or activity_index >= len(activities):
        return None
    previous, current = activities[activity_index - 1], activities[activity_index]
    return (str(plan.get("target_city", "") or ""),
            _activity_endpoint(previous, arrival=True),
            _activity_endpoint(current, arrival=False),
            str(previous.get("end_time", "") or ""))


def _repair_from_cache(
    counterfactual: dict[str, Any], issues: list[str], cache: RouteEvidenceCache,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not issues:
        return []

    # Multiple continuity errors can describe the same inbound boundary (for
    # example both inbound_start and inbound_end).  Repair each destination
    # activity once, then jointly enumerate route choices across boundaries.
    boundaries: dict[str, tuple[tuple[str, str, str, str], dict[str, Any]]] = {}
    for issue in issues:
        prefix = issue.split(":", 1)[0]
        if prefix in boundaries:
            continue
        request = _route_request(counterfactual, issue)
        entry = cache.lookup(*request) if request else None
        if request is None or not isinstance(entry, dict):
            return []
        boundaries[prefix] = (request, entry)

    # Current plans create at most three boundaries.  Keep a defensive limit
    # so malformed plans cannot trigger exponential enumeration.
    if len(boundaries) > 6:
        return []

    choices: list[list[tuple[str, list[dict[str, Any]], dict[str, Any]]]] = []
    prefixes = list(boundaries)
    for prefix in prefixes:
        _, entry = boundaries[prefix]
        boundary_choices = []
        for mode in MODES:
            evidence = entry.get("modes", {}).get(mode, {})
            rows = evidence.get("rows", [])
            if evidence.get("status") != "ok" or not isinstance(rows, list) or not rows:
                continue
            boundary_choices.append((mode, rows, entry))
        # Complete evidence with no route in any mode is conclusive, but it
        # yields no repair candidate for the Cartesian product.
        if not boundary_choices:
            return []
        choices.append(boundary_choices)

    options = []
    for combination in product(*choices):
        repaired = copy.deepcopy(counterfactual)
        evidence_rows = []
        applicable = True
        for prefix, (mode, rows, entry) in zip(prefixes, combination):
            day_token, act_token = prefix.split("_act")
            activities = _day_activities(
                repaired,
                int(day_token.removeprefix("day")),
            )
            if activities is None or int(act_token) >= len(activities):
                applicable = False
                break
            activities[int(act_token)]["transports"] = copy.deepcopy(rows)
            evidence_rows.append({
                "boundary": prefix,
                "request": entry.get("request"),
                "mode": mode,
            })
        if applicable:
            options.append((repaired, {
                "source": "cache",
                "repairs": evidence_rows,
                "combination_size": len(evidence_rows),
            }))
    return options


def _route_cache_evidence_complete(
    counterfactual: dict[str, Any],
    issues: list[str],
    cache: RouteEvidenceCache,
) -> bool:
    if not issues:
        return False
    prefixes: set[str] = set()
    for issue in issues:
        prefix = issue.split(":", 1)[0]
        if prefix in prefixes:
            continue
        prefixes.add(prefix)
        request = _route_request(counterfactual, issue)
        entry = cache.lookup(*request) if request else None
        if not isinstance(entry, dict):
            return False
        modes = entry.get("modes")
        if not isinstance(modes, dict) or not all(
            isinstance(modes.get(mode), dict)
            and modes[mode].get("status") in {"ok", "ok_no_route"}
            for mode in MODES
        ):
            return False
    return len(prefixes) <= 6


def _iter_plan_days(plan: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    return [
        (int(day.get("day", index) or index), day.get("activities", []))
        for index, day in enumerate(plan.get("itinerary", []), start=1)
    ]


def analyze_cascade(
    origin_plan: dict[str, Any],
    edited_plan: dict[str, Any],
    edit_constraints: list[dict[str, Any]],
    *,
    route_evidence_cache: RouteEvidenceCache | None = None,
    full_gate_validator: FullGateValidator | None = None,
) -> dict[str, Any]:
    """Compute target-aware cascade metrics for one origin/edited plan pair."""
    origin = _activities(origin_plan)
    edited = _activities(edited_plan)
    origin_by_ref = {item.ref: item for item in origin}
    edited_by_ref = {item.ref: item for item in edited}
    origin_raw = _raw_activities(origin_plan)
    edited_raw = _raw_activities(edited_plan)
    sequence = infer_edit_sequence(origin_plan, edited_plan)
    same_day_reorder_refs = {
        ref
        for op in sequence.atomic_ops
        if op.op_type == "reorder"
        and op.scope == "structural"
        and "day" in op.details
        for ref in op.origin_refs
    }

    changed_origin: set[str] = set()
    inserted_edited: set[str] = set()
    atomic_types_by_origin: dict[str, set[str]] = {}
    atomic_types_by_inserted: dict[str, set[str]] = {}
    for op in sequence.atomic_ops:
        changed_origin.update(op.origin_refs)
        for ref in op.origin_refs:
            atomic_types_by_origin.setdefault(ref, set()).add(op.op_type)
        if op.op_type == "insert":
            inserted_edited.update(op.edited_refs)
            for ref in op.edited_refs:
                atomic_types_by_inserted.setdefault(ref, set()).add(op.op_type)

    constraint_types = {str(item.get("type", "")) for item in edit_constraints}
    numeric_types = {
        "budget_total", "ticket_budget_total", "activity_budget_limit",
        "anchor_bundle_budget_limit", *_TRANSPORT_NUMERIC_TYPES,
    }
    temporal_types = {
        "poi_time_window", "activity_duration_limit", "adjacent_travel_time_cap",
        "pair_same_day_no_overlap", "pair_time_window_no_overlap", "day_end_time_limit",
    }
    semantic_types = {
        "semantic_type_requirement", "required_restaurant_type", "required_room_type",
        "required_intercity_transport_type",
    }
    day_types = {"day_count", "daily_poi_cap", "poi_day_binding"}

    names, forbidden_names, explicit_days = _target_spec(edit_constraints)
    folded_names = {name.casefold() for name in names}
    folded_forbidden = {name.casefold() for name in forbidden_names}
    target_edited = {
        item.ref
        for item in edited
        # A day is scope context, never sufficient evidence that every activity
        # on that day directly realizes the target.
        if item.name and item.name.casefold() in folded_names
    }
    named_origin = {
        item.ref for item in origin
        if item.name and item.name.casefold() in folded_names | folded_forbidden
    }

    # Map grounded edited targets back to their origin counterpart when one
    # exists; replacement pairs count as the direct target edit, not spillover.
    edited_to_origin = {pair.edited_ref: pair.origin_ref for pair in sequence.matched_pairs}
    origin_to_edited = {pair.origin_ref: pair.edited_ref for pair in sequence.matched_pairs}
    target_origin = {edited_to_origin[ref] for ref in target_edited if ref in edited_to_origin}
    target_origin.update(named_origin)
    direct_origin = changed_origin & target_origin
    direct_inserted = inserted_edited & target_edited
    modes: set[str] = set()
    virtual_direct = 0
    if direct_origin or direct_inserted:
        modes.add("exact_entity")

    # A forbidden entity is grounded in the origin, so its deletion/replacement
    # is directly required even though it must be absent from the edited plan.
    forbidden_origin = {
        item.ref for item in origin
        if item.name and item.name.casefold() in folded_forbidden
    }
    forbidden_direct = forbidden_origin & changed_origin
    if forbidden_direct:
        direct_origin.update(forbidden_direct)
        modes.add("exact_entity")

    # Four-way adjudication direct starts only from explicit named entities.
    policy_direct_origin = set(direct_origin)
    policy_direct_inserted = set(direct_inserted)
    policy_virtual_direct = 0

    # Preserve the original exact-entity decomposition as a parallel view even
    # when additional type-aware proxies later expand the necessary set.
    legacy_total_changed = len(changed_origin) + len(inserted_edited)
    exact_entity_direct_count = len(direct_origin) + len(direct_inserted)

    graph = _dependency_graph(origin)

    # Anchor constraints often leave the named anchor unchanged and modify an
    # adjacent commute/meal. Only changed distance-1 neighbors are direct.
    if names and not (direct_origin or direct_inserted):
        neighbor_direct = {
            ref for seed in target_origin for ref in graph.get(seed, set()) if ref in changed_origin
        }
        if neighbor_direct:
            direct_origin.update(neighbor_direct)
            modes.add("neighborhood_constraint_proxy")

    # Global numeric constraints have no activity ID. Attribute only favorable
    # reductions in the metric owned by the constraint; all other edits remain
    # candidates for counterfactual rollback.
    if constraint_types & numeric_types:
        requested_kinds = _constraint_kinds(edit_constraints)
        numeric_direct: set[str] = set()
        for origin_ref, edited_ref in origin_to_edited.items():
            left, right = origin_by_ref[origin_ref], edited_by_ref[edited_ref]
            kind_ok = not requested_kinds or any(_kind_matches(left.kind, kind) for kind in requested_kinds)
            if kind_ok and _numeric_edit_is_favorable(
                constraint_types,
                left,
                right,
                origin_raw.get(origin_ref, {}),
                edited_raw.get(edited_ref, {}),
            ):
                numeric_direct.add(origin_ref)
        for ref in changed_origin:
            if ref not in origin_to_edited:
                item = origin_by_ref.get(ref)
                if item and (not requested_kinds or any(_kind_matches(item.kind, kind) for kind in requested_kinds)):
                    numeric_direct.add(ref)
        if numeric_direct:
            direct_origin.update(numeric_direct)
            modes.add("global_numeric_proxy")
        elif changed_origin or inserted_edited:
            virtual_direct += 1
            modes.add("global_numeric_fallback")

    # Time constraints: only activities with changed temporal fields are in the
    # necessary-set proxy; replacements/additions remain collateral unless an
    # explicit entity target already grounds them.
    if constraint_types & temporal_types:
        time_direct: set[str] = set()
        for origin_ref, edited_ref in origin_to_edited.items():
            left, right = origin_by_ref[origin_ref], edited_by_ref[edited_ref]
            if (left.start_time, left.end_time) != (right.start_time, right.end_time):
                changed_origin.add(origin_ref)
                time_direct.add(origin_ref)
        if time_direct:
            direct_origin.update(time_direct)
            modes.add("temporal_proxy")
        elif (changed_origin or inserted_edited) and not modes:
            virtual_direct += 1
            modes.add("temporal_fallback")

    # Resolve semantic requirements with the same ChinaTravel concept scorer as
    # the benchmark verifier; retain coarse/fallback modes when no scorer applies.
    if constraint_types & semantic_types and not (direct_origin or direct_inserted):
        requested_kinds = _constraint_kinds(edit_constraints)
        semantic_requirements = [
            item for item in edit_constraints
            if item.get("type") == "semantic_type_requirement"
        ]
        required_semantic_types = {
            semantic_type
            for item in semantic_requirements
            for semantic_type in _strings(item.get("value"))
        }
        semantic_origin: set[str] = set()
        semantic_inserted: set[str] = set()
        semantic_units_by_type: dict[str, list[tuple[str, str]]] = {
            semantic_type: [] for semantic_type in required_semantic_types
        }
        fine_tag_used = False
        if required_semantic_types:
            city = edited_plan.get("target_city")
            for ref in changed_origin:
                edited_ref = origin_to_edited.get(ref)
                raw = edited_raw.get(edited_ref, {}) if edited_ref else {}
                fine_type = _fine_semantic_type(raw, city) if raw else None
                fine_tag_used = fine_tag_used or fine_type is not None
                if fine_type in required_semantic_types:
                    semantic_origin.add(ref)
                    semantic_units_by_type[fine_type].append(("origin", ref))
            for ref in inserted_edited:
                fine_type = _fine_semantic_type(edited_raw.get(ref, {}), city)
                fine_tag_used = fine_tag_used or fine_type is not None
                if fine_type in required_semantic_types:
                    semantic_inserted.add(ref)
                    semantic_units_by_type[fine_type].append(("inserted", ref))
        else:
            semantic_origin = {
                ref for ref in changed_origin if ref in origin_by_ref
                and any(_kind_matches(origin_by_ref[ref].kind, kind) for kind in requested_kinds)
            }
            semantic_inserted = {
                ref for ref in inserted_edited if ref in edited_by_ref
                and any(_kind_matches(edited_by_ref[ref].kind, kind) for kind in requested_kinds)
            }
        direct_origin.update(semantic_origin)
        direct_inserted.update(semantic_inserted)
        # A requested intercity-mode change directly targets each changed
        # intercity leg. Keep those logical replacements out of rollback
        # attribution; otherwise train-to-airplane edits can be mislabeled as
        # removable merely because another edited leg still satisfies the
        # plan-level mode constraint.
        if "required_intercity_transport_type" in constraint_types:
            policy_direct_origin.update(semantic_origin)
            policy_direct_inserted.update(semantic_inserted)
        if "required_room_type" in constraint_types:
            for origin_ref, edited_ref in origin_to_edited.items():
                left, right = origin_by_ref[origin_ref], edited_by_ref[edited_ref]
                if left.kind == "accommodation" and left.room_type != right.room_type:
                    changed_origin.add(origin_ref)
                    direct_origin.add(origin_ref)
        if semantic_origin or semantic_inserted or direct_origin:
            modes.add("semantic_fine_tag" if fine_tag_used and required_semantic_types else "semantic_type_proxy")
            for requirement in semantic_requirements:
                requested_types = list(_strings(requirement.get("value")))
                min_required = int(requirement.get("params", {}).get("min_count", 1) or 1)
                matching_units = sorted({
                    unit
                    for requested_type in requested_types
                    for unit in semantic_units_by_type.get(requested_type, [])
                })
                for unit_kind, ref in matching_units[:min_required]:
                    (policy_direct_origin if unit_kind == "origin" else policy_direct_inserted).add(ref)
        elif changed_origin or inserted_edited:
            virtual_direct += 1
            modes.add("semantic_fallback")

    # POI day binding is direct only when the named/matching activity actually
    # moves across days. Merely sharing a target day is not direct evidence.
    if "poi_day_binding" in constraint_types:
        moved = {
            origin_ref
            for origin_ref, edited_ref in origin_to_edited.items()
            if origin_by_ref[origin_ref].day != edited_by_ref[edited_ref].day
            and (
                not (folded_names | folded_forbidden)
                or origin_by_ref[origin_ref].name.casefold() in folded_names | folded_forbidden
                or edited_by_ref[edited_ref].name.casefold() in folded_names | folded_forbidden
            )
        }
        if moved:
            changed_origin.update(moved)
            direct_origin.update(moved)
            policy_direct_origin.update(moved)
            modes.add("day_allocation_proxy")

    # Day/global structural requests get one virtual required unit per changed
    # day boundary, not every activity on the affected day. POI binding alone
    # never receives a virtual day-boundary unit.
    structural_day_types = {"day_count", "daily_poi_cap"}
    if constraint_types & structural_day_types and not (direct_origin or direct_inserted):
        origin_days = {item.day for item in origin}
        edited_days = {item.day for item in edited}
        boundary_changes = len(origin_days ^ edited_days)
        if boundary_changes:
            virtual_direct += boundary_changes
            policy_virtual_direct += boundary_changes
            modes.add("day_structure_proxy")
        else:
            moved = {ref for ref in changed_origin if ref in origin_to_edited and origin_by_ref[ref].day != edited_by_ref[origin_to_edited[ref]].day}
            if moved:
                direct_origin.update(moved)
                modes.add("day_allocation_proxy")
            elif changed_origin or inserted_edited:
                virtual_direct += 1
                modes.add("day_structure_fallback")

    total_changed = len(changed_origin) + len(inserted_edited)
    virtual_direct = min(virtual_direct, max(total_changed - len(direct_origin) - len(direct_inserted), 0))
    direct_count = len(direct_origin) + len(direct_inserted) + virtual_direct
    cascade_count = max(total_changed - direct_count, 0)

    target_days = {edited_by_ref[ref].day for ref in target_edited if ref in edited_by_ref}
    target_days.update(origin_by_ref[ref].day for ref in direct_origin if ref in origin_by_ref)
    if not target_days and constraint_types & day_types:
        target_days.update({item.day for item in origin} ^ {item.day for item in edited})
    target_origin.update(direct_origin)
    distances = _distances(graph, target_origin)
    cascade_origin = changed_origin - direct_origin
    reachable_distances = [distances[ref] for ref in cascade_origin if ref in distances]
    disconnected = sum(1 for ref in cascade_origin if ref not in distances) + len(inserted_edited - direct_inserted)
    affected_days = {origin_by_ref[ref].day for ref in changed_origin if ref in origin_by_ref}
    affected_days.update(edited_by_ref[ref].day for ref in inserted_edited if ref in edited_by_ref)
    off_target = sum(
        1 for ref in changed_origin if ref in origin_by_ref and origin_by_ref[ref].day not in target_days
    ) + sum(
        1 for ref in inserted_edited if ref in edited_by_ref and edited_by_ref[ref].day not in target_days
    )

    supported = bool(modes) or total_changed == 0
    mode = "+".join(sorted(modes)) if modes else "no_observed_change"
    confidence = "not_applicable" if not modes else _confidence(modes)
    if modes & {"global_numeric_fallback", "temporal_fallback", "semantic_fallback", "day_structure_fallback"}:
        confidence = "low"

    rollback_required_units: list[dict[str, Any]] = []
    authorized_units: list[dict[str, Any]] = []
    avoidable_units: list[dict[str, Any]] = []
    unresolved_units: list[dict[str, Any]] = []
    target_satisfied = False
    baseline_gate: FullGateResult | None = None
    baseline_error: Exception | None = None
    origin_endpoints = _transport_continuity_issues(origin_plan)
    try:
        baseline_constraints, baseline_violations = _verification_signature(edited_plan, edit_constraints)
        baseline_endpoints = _transport_continuity_issues(edited_plan)
        baseline_cross_city_gaps = _cross_city_gap_signature(edited_plan)
        target_satisfied = all(baseline_constraints.values()) if baseline_constraints else True
    except Exception as exc:
        baseline_constraints = {}
        baseline_violations = Counter()
        baseline_endpoints = set()
        baseline_cross_city_gaps = Counter()
        baseline_error = exc
    if full_gate_validator is not None:
        try:
            baseline_gate = full_gate_validator(edited_plan)
        except Exception as exc:
            baseline_error = baseline_error or exc

    proven_direct_origin = policy_direct_origin if target_satisfied else set()
    proven_direct_inserted = policy_direct_inserted if target_satisfied else set()
    candidates = [("origin", ref) for ref in sorted(changed_origin - proven_direct_origin)]
    candidates.extend(("inserted", ref) for ref in sorted(inserted_edited - proven_direct_inserted))
    added_days = {item.day for item in edited} - {item.day for item in origin}

    def assess_candidate(plan: dict[str, Any]) -> dict[str, Any]:
        constraints, violations = _verification_signature(plan, edit_constraints)
        endpoint_issues = sorted(_transport_continuity_issues(plan) - baseline_endpoints)
        violation_instances = _new_violation_instances(violations, baseline_violations)
        cross_city_gaps = _cross_city_gap_signature(plan)
        new_cross_city_gaps = _new_cross_city_gap_instances(
            cross_city_gaps,
            baseline_cross_city_gaps,
        )
        gate: FullGateResult | None = None
        gate_error: str | None = None
        if full_gate_validator is not None:
            try:
                gate = full_gate_validator(plan)
            except Exception as exc:
                gate_error = f"{type(exc).__name__}:{exc}"
        regressions = sorted(
            key for key, passed in baseline_constraints.items()
            if passed and not constraints.get(key, False)
        )
        gate_regressions = _gate_regressions(baseline_gate, gate)
        if baseline_gate is not None and baseline_gate.passed and gate is not None and not gate.passed:
            gate_regressions = sorted(set(gate_regressions) | {"level3_eligibility"})
        return {
            "plan": plan,
            "regressed_constraints": regressions,
            "new_violation_instances": violation_instances,
            "new_endpoint_issues": endpoint_issues,
            "cross_city_gaps": cross_city_gaps,
            "new_cross_city_gaps": new_cross_city_gaps,
            "gate": gate,
            "gate_error": gate_error,
            "full_gate_regressions": gate_regressions,
        }

    def fully_valid(assessment: dict[str, Any]) -> bool:
        gate = assessment["gate"]
        return bool(
            not assessment["regressed_constraints"]
            and not assessment["new_violation_instances"]
            and not assessment["new_endpoint_issues"]
            and not assessment["new_cross_city_gaps"]
            and baseline_gate is not None
            and baseline_gate.passed
            and gate is not None
            and gate.passed
        )

    def gate_evidence_complete(assessments: list[dict[str, Any]]) -> bool:
        """Whether G was available for the baseline and every tested candidate."""

        return bool(
            baseline_gate is not None
            and baseline_gate.passed
            and all(
                item["gate"] is not None and item["gate_error"] is None
                for item in assessments
            )
        )

    def saved_witness(
        assessment: dict[str, Any],
        *,
        source: str,
        candidate_evidence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Persist the concrete q that establishes existential removability."""

        gate = assessment["gate"]
        if not fully_valid(assessment) or gate is None:
            raise AssertionError("a saved rollback witness must satisfy the full gate")
        candidate_plan = copy.deepcopy(assessment["plan"])
        return {
            "family_version": ROLLBACK_FAMILY_VERSION,
            "source": source,
            "candidate_plan": candidate_plan,
            "candidate_plan_sha256": _canonical_sha256(candidate_plan),
            "gate": {
                "passed": gate.passed,
                "components": dict(sorted(gate.components.items())),
                "reason": gate.reason,
            },
            "candidate_evidence": copy.deepcopy(candidate_evidence or {}),
        }

    def candidate_outcome(assessment: dict[str, Any]) -> dict[str, Any]:
        """Return a compact, deterministic audit record for one q in Q_A(u)."""

        gate = assessment["gate"]
        evidence = assessment.get("candidate_evidence") or {}
        return {
            "source": str(evidence.get("source") or "unknown"),
            "candidate_plan_sha256": _canonical_sha256(assessment["plan"]),
            "fully_valid": fully_valid(assessment),
            "gate_passed": gate.passed if gate is not None else None,
            "gate_error": assessment["gate_error"],
            "regressed_constraints": list(assessment["regressed_constraints"]),
            "new_violation_count": len(assessment["new_violation_instances"]),
            "new_endpoint_issue_count": len(assessment["new_endpoint_issues"]),
            "new_cross_city_gap_count": len(assessment["new_cross_city_gaps"]),
        }

    for unit_kind, ref in candidates:
        try:
            if baseline_error is not None:
                raise RuntimeError(
                    f"baseline_validation_unavailable:{type(baseline_error).__name__}"
                )
            counterfactual = _rollback_unit(
                edited_plan, unit_kind, ref, origin_raw, edited_raw, origin_to_edited,
                same_day_reorder_refs,
            )
            if counterfactual is None:
                unresolved_units.append({"kind": unit_kind, "ref": ref, "evidence_status": "unresolved", "reason": "rollback_unavailable"})
                continue
            assessment = assess_candidate(counterfactual)
            assessment["candidate_evidence"] = {"source": "single_rollback"}
            original_endpoint_issues = list(assessment["new_endpoint_issues"])
            candidate_has_cross_city_gap = bool(assessment["cross_city_gaps"])
            repair_applied = False
            evidence_source = "single_rollback"
            route_evidence_complete = False
            closure_members: list[tuple[str, str]] = []
            family_assessments: list[dict[str, Any]] = [assessment]
            if original_endpoint_issues and not candidate_has_cross_city_gap:
                repair_assessments: list[tuple[str, dict[str, Any]]] = []
                closure_members = _transport_dependency_closure_units(
                    unit_kind,
                    ref,
                    atomic_types_by_origin=atomic_types_by_origin,
                    changed_origin=changed_origin,
                    inserted_edited=inserted_edited,
                    origin_to_edited=origin_to_edited,
                    edited_to_origin=edited_to_origin,
                )
                if closure_members:
                    closure_plan = counterfactual
                    closure_complete = True
                    for closure_kind, closure_ref in closure_members:
                        rolled_back = _rollback_unit(
                            closure_plan,
                            closure_kind,
                            closure_ref,
                            origin_raw,
                            edited_raw,
                            origin_to_edited,
                            same_day_reorder_refs,
                        )
                        if rolled_back is None:
                            closure_complete = False
                            break
                        closure_plan = rolled_back
                    if closure_complete:
                        closure_plan, _ = _repair_inbound_from_origin(
                            closure_plan,
                            origin_plan,
                        )
                        closure_assessment = assess_candidate(closure_plan)
                        # The owning full gate already uses origin-relative
                        # feasibility.  Do not reject an otherwise valid closure
                        # merely because it restores a transport gap inherited
                        # verbatim from the origin plan.
                        closure_assessment["new_endpoint_issues"] = sorted(
                            set(closure_assessment["new_endpoint_issues"])
                            - origin_endpoints
                        )
                        closure_assessment["candidate_evidence"] = {
                            "source": "dependency_closure",
                            "members": [
                                {"kind": kind, "ref": member_ref}
                                for kind, member_ref in closure_members
                            ],
                        }
                        if (
                            fully_valid(closure_assessment)
                            or closure_assessment["regressed_constraints"]
                            or closure_assessment["full_gate_regressions"]
                            or closure_assessment["new_violation_instances"]
                        ):
                            repair_assessments.append(
                                ("dependency_closure", closure_assessment)
                            )
                repaired, repair_applied = _repair_inbound_from_origin(counterfactual, origin_plan)
                if repair_applied:
                    origin_assessment = assess_candidate(repaired)
                    origin_assessment["candidate_evidence"] = {
                        "source": "origin_inbound_repair"
                    }
                    repair_assessments.append(("origin", origin_assessment))
                cache_assessments: list[dict[str, Any]] = []
                if route_evidence_cache is not None:
                    route_evidence_complete = _route_cache_evidence_complete(
                        counterfactual, original_endpoint_issues, route_evidence_cache,
                    )
                    for cached_plan, cache_evidence in _repair_from_cache(
                        counterfactual, original_endpoint_issues, route_evidence_cache,
                    ):
                        cached_assessment = assess_candidate(cached_plan)
                        cached_assessment["candidate_evidence"] = cache_evidence
                        cache_assessments.append(cached_assessment)
                        repair_assessments.append(("cache", cached_assessment))

                # The malformed single rollback is not a member of Q_A(u) when
                # it creates a transport discontinuity. Its locally repaired
                # descendants are the tested rollback family.
                family_assessments = [item for _, item in repair_assessments]

                feasible_repair = next(
                    (
                        (source, item)
                        for source, item in repair_assessments
                        if fully_valid(item)
                    ),
                    None,
                )
                closure_proof = next(
                    (
                        (source, item)
                        for source, item in repair_assessments
                        if source == "dependency_closure"
                        and not item["new_endpoint_issues"]
                        and not item["new_cross_city_gaps"]
                        and (
                            item["regressed_constraints"]
                            or item["new_violation_instances"]
                            or item["full_gate_regressions"]
                        )
                    ),
                    None,
                )
                if feasible_repair is not None:
                    evidence_source, assessment = feasible_repair
                    repair_applied = True
                elif closure_proof is not None:
                    evidence_source, assessment = closure_proof
                    repair_applied = True
                elif route_evidence_complete:
                    conclusive = [
                        item for item in cache_assessments
                        if not item["new_endpoint_issues"]
                    ]
                    locally_feasible = [
                        item for item in conclusive
                        if not item["regressed_constraints"]
                        and not item["new_violation_instances"]
                    ]
                    if conclusive:
                        assessment = (locally_feasible or conclusive)[0]
                        assessment["regressed_constraints"] = sorted({
                            value
                            for item in conclusive
                            for value in item["regressed_constraints"]
                        })
                        assessment["new_violation_instances"] = [
                            instance
                            for item in conclusive
                            for instance in item["new_violation_instances"]
                        ]
                        assessment["full_gate_regressions"] = sorted({
                            value
                            for item in conclusive
                            for value in item["full_gate_regressions"]
                        })
                    else:
                        assessment["new_endpoint_issues"] = []
                    if (
                        not assessment["regressed_constraints"]
                        and not assessment["new_violation_instances"]
                        and not assessment["full_gate_regressions"]
                        and not locally_feasible
                    ):
                        assessment["regressed_constraints"] = ["cached_route_infeasible"]
                    evidence_source, repair_applied = "cache", True

            new_violation_instances = assessment["new_violation_instances"]
            new_violation_codes = sorted({
                item["code"] for item in new_violation_instances
            })
            new_cross_city_gaps = assessment["new_cross_city_gaps"]
            if new_cross_city_gaps:
                evidence_source = "offline_endpoint_city_index"
            selected_is_witness = fully_valid(assessment)
            witness = (
                saved_witness(
                    assessment,
                    source=evidence_source,
                    candidate_evidence=assessment.get("candidate_evidence"),
                )
                if selected_is_witness
                else None
            )
            gate_complete = gate_evidence_complete(family_assessments)
            incomplete_reasons: list[str] = []
            if baseline_gate is None:
                incomplete_reasons.append("baseline_full_gate_unavailable")
            elif not baseline_gate.passed:
                incomplete_reasons.append("baseline_full_gate_not_passed")
            if any(
                item["gate"] is None or item["gate_error"] is not None
                for item in family_assessments
            ):
                incomplete_reasons.append("candidate_full_gate_unavailable")
            if original_endpoint_issues and not candidate_has_cross_city_gap and not route_evidence_complete:
                incomplete_reasons.append("route_evidence_incomplete")
            if candidate_has_cross_city_gap and not new_cross_city_gaps:
                incomplete_reasons.append("cross_city_gap_preexists_baseline")

            # A negative attribution is complete only for the exact, versioned
            # rollback family Q_A(u). Positive witnesses remain sound even when
            # other candidates or route evidence are unavailable.
            if candidate_has_cross_city_gap:
                family_evidence_complete = bool(new_cross_city_gaps and gate_complete)
            elif original_endpoint_issues:
                family_evidence_complete = bool(route_evidence_complete and gate_complete)
            else:
                family_evidence_complete = gate_complete
            tested_candidate_outcomes = [
                candidate_outcome(item) for item in family_assessments
            ]
            if witness is not None and not any(
                item["candidate_plan_sha256"] == witness["candidate_plan_sha256"]
                and item["fully_valid"] is True
                for item in tested_candidate_outcomes
            ):
                raise AssertionError("saved witness is not a valid member of Q_A(u)")
            details = {
                "kind": unit_kind,
                "ref": ref,
                "atomic_op_types": sorted(
                    (
                        atomic_types_by_origin
                        if unit_kind == "origin"
                        else atomic_types_by_inserted
                    ).get(ref, set())
                ),
                "regressed_constraints": assessment["regressed_constraints"],
                "new_violation_codes": new_violation_codes,
                "new_violation_instances": new_violation_instances,
                "new_transport_continuity_issues": assessment["new_endpoint_issues"],
                "new_cross_city_gap_instances": new_cross_city_gaps,
                "cross_city_gap_instance_count": sum(
                    assessment["cross_city_gaps"].values()
                ),
                "repair_applied": repair_applied,
                "dependency_closure_members": [
                    {"kind": kind, "ref": member_ref}
                    for kind, member_ref in closure_members
                ],
                "evidence_source": evidence_source,
                "rollback_family_version": ROLLBACK_FAMILY_VERSION,
                "tested_candidate_count": len(family_assessments),
                "tested_candidate_outcomes": tested_candidate_outcomes,
                "rollback_family_complete": family_evidence_complete,
                "evidence_complete": family_evidence_complete,
                "gate_evidence_complete": gate_complete,
                "incomplete_reasons": sorted(set(incomplete_reasons)),
                "witness_found": witness is not None,
                "route_evidence_scope": (
                    [] if candidate_has_cross_city_gap else list(MODES)
                ),
                "route_evidence_complete": route_evidence_complete,
                "full_gate_baseline_passed": baseline_gate.passed if baseline_gate is not None else None,
                "full_gate_candidate_passed": (
                    assessment["gate"].passed
                    if assessment["gate"] is not None
                    else None
                ),
                "full_gate_regressions": assessment["full_gate_regressions"],
            }
            if witness is not None:
                details["witness"] = witness
            if assessment["gate_error"]:
                details["full_gate_error"] = assessment["gate_error"]
            # Precomputation must query the boundaries created by the original
            # single rollback.  A selected dependency-closure assessment may
            # already have cleared those endpoint issues while still requiring
            # route evidence to make the negative family complete.
            if original_endpoint_issues and not candidate_has_cross_city_gap:
                details["route_requests"] = [
                    {"city": req[0], "start": req[1], "end": req[2], "start_time": req[3]}
                    for issue in original_endpoint_issues
                    if (req := _route_request(counterfactual, issue)) is not None
                ]
            if witness is not None and "day_count" in constraint_types and _authorized_new_day_unit(
                unit_kind, ref, origin_to_edited, edited_by_ref, added_days,
                edited_plan, edit_constraints,
            ):
                details["evidence_status"] = "authorized"
                details["reason"] = "new_day_basic_role_within_scope"
                authorized_units.append(details)
            elif witness is not None:
                details["evidence_status"] = "verified"
                details["reason"] = "saved_counterfactual_passes_full_gate"
                avoidable_units.append(details)
            elif family_evidence_complete:
                details["evidence_status"] = "rollback_required_within_tested_family"
                details["reason"] = (
                    "rollback_introduces_cross_city_gap"
                    if new_cross_city_gaps
                    else (
                        "rollback_breaks_full_level_gate"
                        if assessment["full_gate_regressions"]
                        else "no_valid_counterfactual_in_complete_tested_family"
                    )
                )
                rollback_required_units.append(details)
            else:
                details["evidence_status"] = "unresolved"
                if candidate_has_cross_city_gap and not new_cross_city_gaps:
                    details["reason"] = "cross_city_gap_preexists_baseline"
                elif original_endpoint_issues:
                    details["reason"] = "rollback_requires_transport_repair"
                elif baseline_gate is None or any(
                    item["gate"] is None for item in family_assessments
                ):
                    details["reason"] = "full_gate_validation_unavailable"
                else:
                    details["reason"] = "rollback_evidence_incomplete"
                unresolved_units.append(details)
        except Exception as exc:
            unresolved_units.append({
                "kind": unit_kind,
                "ref": ref,
                "evidence_status": "unresolved",
                "reason": f"counterfactual_unavailable:{type(exc).__name__}",
            })

    actual_direct_count = len(policy_direct_origin) + len(policy_direct_inserted)
    four_way_direct_count = (actual_direct_count + policy_virtual_direct) if target_satisfied else 0
    rollback_required_count = len(rollback_required_units)
    authorized_count = len(authorized_units)
    avoidable_extra_count = len(avoidable_units)
    unresolved_count = len(unresolved_units) + (policy_virtual_direct if not target_satisfied else 0)
    four_way_total = total_changed + policy_virtual_direct
    classified_total = (
        four_way_direct_count + rollback_required_count + authorized_count
        + avoidable_extra_count + unresolved_count
    )
    if classified_total != four_way_total:
        raise AssertionError(
            "cascade attribution invariant violated: "
            f"classified={classified_total}, total={four_way_total}"
        )
    # Machine-check the two directional proof obligations before publishing.
    if any(
        not unit.get("witness_found")
        or not isinstance(unit.get("witness"), dict)
        or unit["witness"].get("gate", {}).get("passed") is not True
        for unit in avoidable_units
    ):
        raise AssertionError("verified removable unit lacks a saved valid witness")
    if any(
        not unit.get("rollback_family_complete")
        or any(
            outcome.get("fully_valid") is not False
            for outcome in unit.get("tested_candidate_outcomes", [])
        )
        for unit in rollback_required_units
    ):
        raise AssertionError("rollback-required unit lacks complete tested evidence")

    proof_covered = four_way_direct_count + rollback_required_count + avoidable_extra_count
    proof_coverage = proof_covered / four_way_total if four_way_total else None
    return {
        "supported": supported,
        "reason": "ok" if supported else "no_type_aware_attribution_rule",
        "coverage_reason": "type_aware_proxy_available" if supported else "no_type_aware_attribution_rule",
        "attribution_mode": mode,
        "attribution_confidence": confidence,
        "constraint_types": sorted(constraint_types),
        "target_names": sorted(names | forbidden_names),
        "target_days": sorted(target_days),
        "target_edited_refs": sorted(target_edited),
        "direct_target_change_count": direct_count,
        "virtual_direct_change_count": virtual_direct,
        "exact_entity_direct_change_count": exact_entity_direct_count,
        "exact_entity_cascade_amplification": (
            legacy_total_changed / exact_entity_direct_count if exact_entity_direct_count else None
        ),
        "exact_entity_spillover_ratio": (
            (legacy_total_changed - exact_entity_direct_count) / legacy_total_changed
            if exact_entity_direct_count and legacy_total_changed else None
        ),
        "cascade_change_count": cascade_count if supported else None,
        "target_satisfied": target_satisfied,
        "rollback_required_support_change_count": rollback_required_count,
        # v1 compatibility alias; reader-facing outputs use rollback-required.
        "hard_required_support_change_count": rollback_required_count,
        "scope_authorized_completion_change_count": authorized_count,
        "avoidable_extra_change_count": avoidable_extra_count,
        "verified_removable_change_count": avoidable_extra_count,
        "unresolved_change_count": unresolved_count,
        "proof_coverage": proof_coverage,
        "adjudication_coverage": proof_coverage,
        "avoidable_lower_rate": avoidable_extra_count / four_way_total if four_way_total else None,
        "avoidable_upper_rate": (avoidable_extra_count + unresolved_count) / four_way_total if four_way_total else None,
        "verified_removable_change_rate": avoidable_extra_count / four_way_total if four_way_total else None,
        "evaluator_relative_excess_change_rate_lower": avoidable_extra_count / four_way_total if four_way_total else None,
        "evaluator_relative_excess_change_rate_upper": (
            (avoidable_extra_count + unresolved_count) / four_way_total
            if four_way_total
            else None
        ),
        "rollback_required_support_share": rollback_required_count / four_way_total if four_way_total else None,
        "hard_support_share": rollback_required_count / four_way_total if four_way_total else None,
        "authorized_completion_share": authorized_count / four_way_total if four_way_total else None,
        "unresolved_share": unresolved_count / four_way_total if four_way_total else None,
        "four_way_direct_target_count": four_way_direct_count,
        "four_way_total_impact_count": four_way_total,
        # v1 three-way aliases retained for downstream consumers.
        "required_support_change_count": rollback_required_count,
        "three_way_direct_target_count": four_way_direct_count,
        "three_way_total_impact_count": four_way_total,
        "rollback_required_support_units": rollback_required_units,
        "required_support_units": rollback_required_units,
        "scope_authorized_completion_units": authorized_units,
        "verified_removable_units": avoidable_units,
        "avoidable_extra_units": avoidable_units,
        "unresolved_units": unresolved_units,
        "total_changed_activity_count": total_changed,
        "cascade_amplification": (total_changed / direct_count) if supported and direct_count else None,
        "spillover_ratio": (cascade_count / total_changed) if supported and total_changed else None,
        "affected_day_count": len(affected_days),
        "cross_day_spillover_count": off_target if supported else None,
        "cross_day_spillover_ratio": (off_target / total_changed) if supported and total_changed else None,
        "cascade_radius": max(reachable_distances, default=0) if supported else None,
        "disconnected_change_count": disconnected if supported else None,
        "distance_buckets": {
            "direct_d0": direct_count,
            "near_d1": sum(distance == 1 for distance in reachable_distances),
            "far_d2_plus": sum(distance >= 2 for distance in reachable_distances),
            "disconnected": disconnected,
        } if supported else None,
    }
