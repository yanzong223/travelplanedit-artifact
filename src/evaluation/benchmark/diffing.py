"""Plan diff inference for Level3 minimal-edit evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .models import InferredAtomicOp, InferredEditSequence, MatchedActivityPair

_MEAL_TYPES = {"breakfast", "breakfest", "lunch", "dinner"}
_INTERCITY_TYPES = {"train", "airplane"}


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
                    train_id=str(activity.get("TrainID", "")),
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
                details={"token": origin.token},
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
            if origin.day != edited.day or origin.type != edited.type:
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
    elif any(item.op_type != "change_time" for item in atomic_ops):
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
        "parameter_count": atomic_counter.get("change_time", 0),
        "structural_count": atomic_counter.get("insert", 0)
        + atomic_counter.get("delete", 0)
        + atomic_counter.get("replace", 0)
        + atomic_counter.get("reorder", 0),
        "compositional_count": compositional_count,
        "atomic_counts": {
            "change_time": atomic_counter.get("change_time", 0),
            "insert": atomic_counter.get("insert", 0),
            "delete": atomic_counter.get("delete", 0),
            "replace": atomic_counter.get("replace", 0),
            "reorder": atomic_counter.get("reorder", 0),
        },
        "content_retention_rate": retained_count / origin_total,
        "poi_seq_edit_distance": _levenshtein_distance(_plan_tokens(origin_plan or {}), _plan_tokens(edited_plan or {})),
        "activity_change_ratio": changed_activity_count / origin_total,
    }
