"""Plan diff inference for Level3 minimal-edit evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any
import unicodedata

from .models import InferredAtomicOp, InferredEditSequence, MatchedActivityPair

_MEAL_TYPES = {"breakfast", "breakfest", "lunch", "dinner"}
_INTERCITY_TYPES = {"train", "airplane"}
_PARAMETER_OP_TYPES = {"change_time", "change_transport", "change_attribute"}
_ACTIVITY_MATCH_KEYS = {
    "type", "position", "start", "end", "TrainID", "FlightID", "start_time", "end_time",
    "transports",
}
_NUMERIC_ATTRIBUTE_KEYS = {"cost", "price", "tickets", "room_type", "rooms"}
_TRANSPORT_ENDPOINT_KEYS = {"start", "end"}
_TRANSPORT_TIME_KEYS = {"start_time", "end_time"}
_TRANSPORT_COST_KEYS = {"cost", "price", "tickets", "cars"}
_TRANSPORT_MODE_ALIASES = {
    "subway": "metro",
    "underground": "metro",
    "walking": "walk",
    "on foot": "walk",
    "cab": "taxi",
    "taxicab": "taxi",
}
_MISSING = object()


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def _normalize_number(value: Any) -> str | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def _normalize_time(value: Any) -> int | str | None:
    text = _normalize_text(value)
    if not text:
        return None
    if ":" not in text:
        return text
    try:
        hour, minute = text.split(":", 2)[:2]
        return int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return text


def _normalize_value(value: Any, *, key: str = "") -> Any:
    if value is _MISSING:
        return ("missing",)
    if key in _NUMERIC_ATTRIBUTE_KEYS or key in _TRANSPORT_COST_KEYS or key == "distance":
        number = _normalize_number(value)
        if number is not None:
            return ("number", number)
    if key in _TRANSPORT_TIME_KEYS:
        return ("time", _normalize_time(value))
    if isinstance(value, dict):
        return tuple(
            (str(item_key), _normalize_value(item_value, key=str(item_key)))
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, str):
        return ("text", _normalize_text(value))
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        number = _normalize_number(value)
        return ("number", number) if number is not None else value
    return value


def _normalize_transport_mode(value: Any) -> str:
    mode = _normalize_text(value)
    return _TRANSPORT_MODE_ALIASES.get(mode, mode)


def _transport_semantic_legs(value: Any) -> tuple[Any, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        return (("invalid_transport_container", _normalize_value(value)),)
    legs: list[Any] = []
    for leg in value:
        if not isinstance(leg, dict):
            legs.append(("invalid_transport_leg", _normalize_value(leg)))
            continue
        semantic: list[tuple[str, Any]] = []
        for key in (
            "start", "end", "mode", "start_time", "end_time", "distance",
            "cost", "price", "tickets", "cars",
        ):
            if key not in leg:
                continue
            if key == "mode":
                normalized = ("text", _normalize_transport_mode(leg[key]))
            elif key in _TRANSPORT_ENDPOINT_KEYS:
                normalized = ("text", _normalize_text(leg[key]))
            else:
                normalized = _normalize_value(leg[key], key=key)
            semantic.append((key, normalized))
        legs.append(tuple(semantic))
    return tuple(legs)


def _transport_change_dimensions(before: Any, after: Any) -> list[str]:
    before_legs = before if isinstance(before, list) else []
    after_legs = after if isinstance(after, list) else []
    dimensions: set[str] = set()
    if not isinstance(before, list) or not isinstance(after, list) or len(before_legs) != len(after_legs):
        dimensions.add("topology")
    for left, right in zip(before_legs, after_legs):
        if not isinstance(left, dict) or not isinstance(right, dict):
            if _normalize_value(left) != _normalize_value(right):
                dimensions.add("other_semantic")
            continue
        if any(
            _normalize_text(left.get(key)) != _normalize_text(right.get(key))
            for key in _TRANSPORT_ENDPOINT_KEYS
        ):
            dimensions.add("endpoint")
        if _normalize_transport_mode(left.get("mode")) != _normalize_transport_mode(right.get("mode")):
            dimensions.add("mode")
        if any(
            _normalize_time(left.get(key)) != _normalize_time(right.get(key))
            for key in _TRANSPORT_TIME_KEYS
        ):
            dimensions.add("duration_or_timing")
        if any(
            _normalize_value(left.get(key, _MISSING), key=key)
            != _normalize_value(right.get(key, _MISSING), key=key)
            for key in _TRANSPORT_COST_KEYS
        ):
            dimensions.add("cost_or_capacity")
        if _normalize_value(left.get("distance", _MISSING), key="distance") != _normalize_value(
            right.get("distance", _MISSING), key="distance"
        ):
            dimensions.add("distance")
    if _transport_semantic_legs(before) != _transport_semantic_legs(after) and not dimensions:
        dimensions.add("other_semantic")
    return sorted(dimensions)


def _activity_payload_details(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_transport = before.get("transports", [])
    after_transport = after.get("transports", [])
    transport_raw_changed = before_transport != after_transport
    transport_semantic_changed = (
        _transport_semantic_legs(before_transport) != _transport_semantic_legs(after_transport)
    )

    attribute_fields = sorted((set(before) | set(after)) - _ACTIVITY_MATCH_KEYS)
    changed_attribute_fields: list[str] = []
    format_only_attribute_fields: list[str] = []
    for key in attribute_fields:
        raw_before = before.get(key, _MISSING)
        raw_after = after.get(key, _MISSING)
        if raw_before == raw_after:
            continue
        normalized_before = _normalize_value(raw_before, key=key)
        normalized_after = _normalize_value(raw_after, key=key)
        if normalized_before == normalized_after:
            format_only_attribute_fields.append(key)
        else:
            changed_attribute_fields.append(key)

    if not transport_raw_changed:
        transport_kind = "none"
    elif transport_semantic_changed:
        transport_kind = "semantic"
    else:
        transport_kind = "format_or_metadata_only"
    return {
        "transport_raw_changed": transport_raw_changed,
        "transport_semantic_changed": transport_semantic_changed,
        "transport_change_kind": transport_kind,
        "transport_change_dimensions": (
            _transport_change_dimensions(before_transport, after_transport)
            if transport_semantic_changed
            else []
        ),
        "attribute_raw_changed": bool(changed_attribute_fields or format_only_attribute_fields),
        "attribute_semantic_changed": bool(changed_attribute_fields),
        "changed_attribute_fields": changed_attribute_fields,
        "format_only_attribute_fields": format_only_attribute_fields,
    }


@dataclass(slots=True)
class _ActivityRecord:
    ref: str
    day: int
    index: int
    activity: dict[str, Any]
    type: str
    start_time: str
    end_time: str
    position: str
    start: str
    end: str
    train_id: str

    @property
    def identity_key(self) -> tuple[str, ...]:
        if self.type in _INTERCITY_TYPES:
            return ("intercity", self.type, self.start, self.end, self.train_id)
        if self.position:
            return ("poi", self.type, self.position)
        return ("generic", self.type, self.start, self.end)

    @property
    def exact_key(self) -> tuple[Any, ...]:
        return (self.day, self.identity_key, self.start_time, self.end_time)

    @property
    def token(self) -> str:
        if self.type in _INTERCITY_TYPES:
            core = self.train_id or f"{self.start}->{self.end}"
            return f"{self.type}:{core}"
        if self.position:
            return f"{self.type}:{self.position}"
        return f"{self.type}:{self.start}->{self.end}"

    def to_summary(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "day": self.day,
            "index": self.index,
            "type": self.type,
            "position": self.position or None,
            "start": self.start or None,
            "end": self.end or None,
            "start_time": self.start_time or None,
            "end_time": self.end_time or None,
            "token": self.token,
        }


def _flatten_plan(plan: dict[str, Any]) -> list[_ActivityRecord]:
    records: list[_ActivityRecord] = []
    for day_index, day in enumerate(plan.get("itinerary", []), start=1):
        day_number = int(day.get("day", day_index) or day_index)
        for activity_index, activity in enumerate(day.get("activities", [])):
            records.append(
                _ActivityRecord(
                    ref=f"day{day_number}_act{activity_index}",
                    day=day_number,
                    index=activity_index,
                    activity=activity,
                    type=str(activity.get("type", "")),
                    start_time=str(activity.get("start_time", "")),
                    end_time=str(activity.get("end_time", "")),
                    position=str(activity.get("position", "")),
                    start=str(activity.get("start", "")),
                    end=str(activity.get("end", "")),
                    train_id=str(activity.get("TrainID") or activity.get("FlightID") or ""),
                )
            )
    return records


def _time_to_minute(value: str) -> int | None:
    if ":" not in value:
        return None
    hour, minute = value.split(":")[:2]
    return int(hour) * 60 + int(minute)


def _time_overlap_ratio(origin: _ActivityRecord, edited: _ActivityRecord) -> float:
    origin_start = _time_to_minute(origin.start_time)
    origin_end = _time_to_minute(origin.end_time)
    edited_start = _time_to_minute(edited.start_time)
    edited_end = _time_to_minute(edited.end_time)
    if None in {origin_start, origin_end, edited_start, edited_end}:
        return 1.0 if origin.index == edited.index else 0.0
    overlap = min(origin_end, edited_end) - max(origin_start, edited_start)
    if overlap <= 0:
        return 0.0
    origin_duration = max(origin_end - origin_start, 1)
    edited_duration = max(edited_end - edited_start, 1)
    return overlap / max(origin_duration, edited_duration)


def _time_distance(origin: _ActivityRecord, edited: _ActivityRecord) -> int:
    origin_start = _time_to_minute(origin.start_time)
    edited_start = _time_to_minute(edited.start_time)
    if origin_start is None or edited_start is None:
        return abs(origin.index - edited.index)
    return abs(origin_start - edited_start)


def _same_time(origin: _ActivityRecord, edited: _ActivityRecord) -> bool:
    return origin.start_time == edited.start_time and origin.end_time == edited.end_time


def _bucket_by_identity(records: list[_ActivityRecord]) -> dict[tuple[str, ...], list[_ActivityRecord]]:
    buckets: dict[tuple[str, ...], list[_ActivityRecord]] = defaultdict(list)
    for record in records:
        buckets[record.identity_key].append(record)
    return buckets


def _pair_exact(
    origin_records: list[_ActivityRecord],
    edited_records: list[_ActivityRecord],
) -> tuple[list[MatchedActivityPair], set[str], set[str]]:
    edited_by_key: dict[tuple[Any, ...], list[_ActivityRecord]] = defaultdict(list)
    for record in edited_records:
        edited_by_key[record.exact_key].append(record)

    matched_pairs: list[MatchedActivityPair] = []
    matched_origin: set[str] = set()
    matched_edited: set[str] = set()
    for origin in origin_records:
        candidates = edited_by_key.get(origin.exact_key, [])
        while candidates and candidates[0].ref in matched_edited:
            candidates.pop(0)
        if not candidates:
            continue
        edited = candidates.pop(0)
        matched_origin.add(origin.ref)
        matched_edited.add(edited.ref)
        matched_pairs.append(
            MatchedActivityPair(
                pair_id=f"{origin.ref}__{edited.ref}",
                origin_ref=origin.ref,
                edited_ref=edited.ref,
                match_type="unchanged",
                retained=True,
                origin_day=origin.day,
                edited_day=edited.day,
                origin_index=origin.index,
                edited_index=edited.index,
                details={"token": origin.token, **_activity_payload_details(origin.activity, edited.activity)},
            )
        )
    return matched_pairs, matched_origin, matched_edited


def _pair_same_identity(
    origin_records: list[_ActivityRecord],
    edited_records: list[_ActivityRecord],
    matched_origin: set[str],
    matched_edited: set[str],
) -> tuple[list[MatchedActivityPair], set[str], set[str]]:
    origin_remaining = [item for item in origin_records if item.ref not in matched_origin]
    edited_remaining = [item for item in edited_records if item.ref not in matched_edited]
    edited_by_identity = _bucket_by_identity(edited_remaining)

    pairs: list[MatchedActivityPair] = []
    for origin in origin_remaining:
        candidates = [
            item
            for item in edited_by_identity.get(origin.identity_key, [])
            if item.ref not in matched_edited
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (abs(origin.day - item.day), _time_distance(origin, item), abs(origin.index - item.index)))
        edited = candidates[0]
        matched_origin.add(origin.ref)
        matched_edited.add(edited.ref)
        match_type = "change_time" if origin.day == edited.day else "moved_across_day"
        pairs.append(
            MatchedActivityPair(
                pair_id=f"{origin.ref}__{edited.ref}",
                origin_ref=origin.ref,
                edited_ref=edited.ref,
                match_type=match_type,
                retained=True,
                origin_day=origin.day,
                edited_day=edited.day,
                origin_index=origin.index,
                edited_index=edited.index,
                details={
                    "token": origin.token,
                    "time_changed": not _same_time(origin, edited),
                    "day_changed": origin.day != edited.day,
                    **_activity_payload_details(origin.activity, edited.activity),
                },
            )
        )
    return pairs, matched_origin, matched_edited


def _pair_replace(
    origin_records: list[_ActivityRecord],
    edited_records: list[_ActivityRecord],
    matched_origin: set[str],
    matched_edited: set[str],
) -> tuple[list[MatchedActivityPair], set[str], set[str]]:
    origin_remaining = [item for item in origin_records if item.ref not in matched_origin]
    edited_remaining = [item for item in edited_records if item.ref not in matched_edited]
    pairs: list[MatchedActivityPair] = []

    for origin in origin_remaining:
        candidates = []
        for edited in edited_remaining:
            if edited.ref in matched_edited:
                continue
            same_activity_family = (
                origin.type == edited.type
                or {origin.type, edited.type} <= _INTERCITY_TYPES
            )
            if origin.day != edited.day or not same_activity_family:
                continue
            if origin.identity_key == edited.identity_key:
                continue
            overlap = _time_overlap_ratio(origin, edited)
            if overlap <= 0.0 and origin.index != edited.index:
                continue
            candidates.append((edited, overlap))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[1], abs(origin.index - item[0].index)))
        edited, overlap = candidates[0]
        matched_origin.add(origin.ref)
        matched_edited.add(edited.ref)
        pairs.append(
            MatchedActivityPair(
                pair_id=f"{origin.ref}__{edited.ref}",
                origin_ref=origin.ref,
                edited_ref=edited.ref,
                match_type="replace",
                retained=False,
                origin_day=origin.day,
                edited_day=edited.day,
                origin_index=origin.index,
                edited_index=edited.index,
                details={
                    "origin_token": origin.token,
                    "edited_token": edited.token,
                    "time_overlap_ratio": round(overlap, 4),
                },
            )
        )
    return pairs, matched_origin, matched_edited


def _infer_reorder_ops(
    origin_records: list[_ActivityRecord],
    edited_records: list[_ActivityRecord],
    retained_pairs: list[MatchedActivityPair],
) -> tuple[list[InferredAtomicOp], set[str]]:
    reorder_ops: list[InferredAtomicOp] = []
    changed_refs: set[str] = set()
    origin_by_day: dict[int, list[_ActivityRecord]] = defaultdict(list)
    edited_by_day: dict[int, list[_ActivityRecord]] = defaultdict(list)
    retained_by_day: dict[int, list[MatchedActivityPair]] = defaultdict(list)

    for item in origin_records:
        origin_by_day[item.day].append(item)
    for item in edited_records:
        edited_by_day[item.day].append(item)
    for pair in retained_pairs:
        if pair.origin_day == pair.edited_day:
            retained_by_day[pair.origin_day].append(pair)

    for day, pairs in retained_by_day.items():
        origin_total = len(origin_by_day.get(day, []))
        edited_total = len(edited_by_day.get(day, []))
        if origin_total == 0 or edited_total == 0 or origin_total != edited_total or origin_total != len(pairs):
            continue
        origin_sequence = [pair.pair_id for pair in sorted(pairs, key=lambda item: item.origin_index)]
        edited_sequence = [pair.pair_id for pair in sorted(pairs, key=lambda item: item.edited_index)]
        if origin_sequence == edited_sequence:
            continue
        reorder_ops.append(
            InferredAtomicOp(
                op_type="reorder",
                scope="structural",
                origin_refs=[pair.origin_ref for pair in pairs],
                edited_refs=[pair.edited_ref for pair in pairs],
                details={"day": day},
            )
        )
        changed_refs.update(pair.origin_ref for pair in pairs)
    return reorder_ops, changed_refs


def _plan_tokens(plan: dict[str, Any]) -> list[str]:
    records = _flatten_plan(plan)
    return [record.token for record in sorted(records, key=lambda item: (item.day, item.index))]


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    rows = len(left) + 1
    cols = len(right) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
    return matrix[-1][-1]


def infer_edit_sequence(origin_plan: dict[str, Any], edited_plan: dict[str, Any]) -> InferredEditSequence:
    origin_records = _flatten_plan(origin_plan or {})
    edited_records = _flatten_plan(edited_plan or {})

    matched_pairs, matched_origin, matched_edited = _pair_exact(origin_records, edited_records)
    identity_pairs, matched_origin, matched_edited = _pair_same_identity(
        origin_records,
        edited_records,
        matched_origin,
        matched_edited,
    )
    matched_pairs.extend(identity_pairs)
    replace_pairs, matched_origin, matched_edited = _pair_replace(
        origin_records,
        edited_records,
        matched_origin,
        matched_edited,
    )
    matched_pairs.extend(replace_pairs)

    atomic_ops: list[InferredAtomicOp] = []
    retained_pairs = [item for item in matched_pairs if item.retained]
    reorder_ops, reorder_changed_refs = _infer_reorder_ops(origin_records, edited_records, retained_pairs)
    atomic_ops.extend(reorder_ops)

    changed_origin_refs: set[str] = set(reorder_changed_refs)
    compositional_refs: set[str] = set()

    for pair in retained_pairs:
        if pair.details.get("transport_semantic_changed"):
            atomic_ops.append(
                InferredAtomicOp(
                    op_type="change_transport",
                    scope="parameter",
                    origin_refs=[pair.origin_ref],
                    edited_refs=[pair.edited_ref],
                    details={
                        "day": pair.origin_day,
                        "change_kind": pair.details.get("transport_change_kind"),
                        "dimensions": pair.details.get("transport_change_dimensions", []),
                    },
                )
            )
            changed_origin_refs.add(pair.origin_ref)
        if pair.details.get("attribute_semantic_changed"):
            atomic_ops.append(
                InferredAtomicOp(
                    op_type="change_attribute",
                    scope="parameter",
                    origin_refs=[pair.origin_ref],
                    edited_refs=[pair.edited_ref],
                    details={
                        "day": pair.origin_day,
                        "fields": pair.details.get("changed_attribute_fields", []),
                    },
                )
            )
            changed_origin_refs.add(pair.origin_ref)

    for pair in identity_pairs:
        if pair.origin_day != pair.edited_day:
            atomic_ops.append(
                InferredAtomicOp(
                    op_type="reorder",
                    scope="compositional",
                    origin_refs=[pair.origin_ref],
                    edited_refs=[pair.edited_ref],
                    details={"origin_day": pair.origin_day, "edited_day": pair.edited_day},
                )
            )
            changed_origin_refs.add(pair.origin_ref)
            compositional_refs.add(pair.origin_ref)
        elif pair.details.get("time_changed"):
            atomic_ops.append(
                InferredAtomicOp(
                    op_type="change_time",
                    scope="parameter",
                    origin_refs=[pair.origin_ref],
                    edited_refs=[pair.edited_ref],
                    details={"day": pair.origin_day},
                )
            )
            changed_origin_refs.add(pair.origin_ref)

    for pair in replace_pairs:
        atomic_ops.append(
            InferredAtomicOp(
                op_type="replace",
                scope="structural",
                origin_refs=[pair.origin_ref],
                edited_refs=[pair.edited_ref],
                details=pair.details,
            )
        )
        changed_origin_refs.add(pair.origin_ref)

    origin_days = {item.day for item in origin_records}
    edited_days = {item.day for item in edited_records}
    day_count_changed = len(origin_days) != len(edited_days)

    unmatched_origin = [item for item in origin_records if item.ref not in matched_origin]
    unmatched_edited = [item for item in edited_records if item.ref not in matched_edited]

    for item in unmatched_origin:
        scope = "compositional" if day_count_changed or item.day not in edited_days else "structural"
        atomic_ops.append(
            InferredAtomicOp(
                op_type="delete",
                scope=scope,
                origin_refs=[item.ref],
                details={"day": item.day, "token": item.token},
            )
        )
        changed_origin_refs.add(item.ref)
        if scope == "compositional":
            compositional_refs.add(item.ref)

    for item in unmatched_edited:
        scope = "compositional" if day_count_changed or item.day not in origin_days else "structural"
        atomic_ops.append(
            InferredAtomicOp(
                op_type="insert",
                scope=scope,
                edited_refs=[item.ref],
                details={"day": item.day, "token": item.token},
            )
        )
        if scope == "compositional":
            compositional_refs.add(item.ref)

    scope_level = 0
    if day_count_changed or any(item.scope == "compositional" for item in atomic_ops):
        scope_level = 2
    elif any(item.op_type not in _PARAMETER_OP_TYPES for item in atomic_ops):
        scope_level = 1
    scope_name = {0: "parameter", 1: "structural", 2: "compositional"}[scope_level]

    return InferredEditSequence(
        scope_level=scope_level,
        scope_name=scope_name,
        atomic_ops=atomic_ops,
        matched_pairs=matched_pairs,
        unmatched_origin=[item.to_summary() for item in unmatched_origin],
        unmatched_edited=[item.to_summary() for item in unmatched_edited],
    )


def sequence_metrics(origin_plan: dict[str, Any], edited_plan: dict[str, Any], sequence: InferredEditSequence) -> dict[str, Any]:
    origin_records = _flatten_plan(origin_plan or {})
    origin_count = len(origin_records)
    retained_count = sum(1 for pair in sequence.matched_pairs if pair.retained)
    changed_origin_refs: set[str] = set()
    inserted_count = 0
    atomic_counter: Counter[str] = Counter()
    compositional_count = 0

    for item in sequence.atomic_ops:
        atomic_counter[item.op_type] += 1
        changed_origin_refs.update(item.origin_refs)
        inserted_count += len(item.edited_refs) if item.op_type == "insert" else 0
        if item.scope == "compositional":
            compositional_count += 1

    changed_activity_count = len(changed_origin_refs) + inserted_count
    origin_total = max(origin_count, 1)
    return {
        "parameter_count": sum(atomic_counter.get(name, 0) for name in _PARAMETER_OP_TYPES),
        "structural_count": atomic_counter.get("insert", 0)
        + atomic_counter.get("delete", 0)
        + atomic_counter.get("replace", 0)
        + atomic_counter.get("reorder", 0),
        "compositional_count": compositional_count,
        "atomic_counts": {
            "change_time": atomic_counter.get("change_time", 0),
            "change_transport": atomic_counter.get("change_transport", 0),
            "change_attribute": atomic_counter.get("change_attribute", 0),
            "insert": atomic_counter.get("insert", 0),
            "delete": atomic_counter.get("delete", 0),
            "replace": atomic_counter.get("replace", 0),
            "reorder": atomic_counter.get("reorder", 0),
        },
        "content_retention_rate": retained_count / origin_total,
        "poi_seq_edit_distance": _levenshtein_distance(_plan_tokens(origin_plan or {}), _plan_tokens(edited_plan or {})),
        "activity_change_ratio": changed_activity_count / origin_total,
    }
