"""
Helpers for aggregating core-pipeline generation cost by constraint type.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


STEP2_REASON_CODES = {
    "missing_template",
    "ineligible_bucket",
    "missing_template_entries",
    "missing_soft_template_entries",
    "missing_constraint_type_template_entries",
    "missing_existing_poi_anchor",
    "template_param_missing",
    "llm_response_missing",
    "constraint_parse_failed",
    "query_constraint_alignment_failed",
    "soft_alignment_failed",
    "soft_preference_build_failed",
    "step2_generation_failed",
    "step2_exception",
    "origin_plan_exhausted",
    "no_valid_candidate",
    "no_applicable_bucket_after_filtering",
}

OUT_OF_SCOPE_SUPPORT_STATUS = "out_of_scope"
NO_DATA_SUPPORT_STATUS = "no_data_sampling"
UNSUPPORTED_STEP3_STATUS = "unsupported_step3"
UNSUPPORTED_VALIDATION_STATUS = "unsupported_validation"
STABLE_SUPPORT_STATUS = "stable"
FRAGILE_SUPPORT_STATUS = "fragile"

OUT_OF_SCOPE_CONSTRAINT_TYPES = {
    "budget_cap_preference",
    "budget_target_update",
}

SUMMARY_FIELDS = [
    "constraint_type",
    "sample_count",
    "observed_sample_count",
    "archived_fail_count",
    "success_count",
    "fail_count",
    "final_success_rate",
    "avg_origin_plan_attempts",
    "avg_template_attempts",
    "avg_retry_count",
    "avg_failed_origin_plan_attempts",
    "avg_failed_template_attempts",
    "avg_failed_retry_count",
    "top_failure_stage",
    "top_failure_reason_code",
    "top_failed_check",
    "top_validation_reason",
    "harness_mismatch_count",
    "final_support_status",
]

BUCKET_FIELDS = [
    "constraint_type",
    "bucket_name",
    "sample_count",
    "success_count",
    "fail_count",
    "final_success_rate",
    "avg_origin_plan_attempts",
    "avg_template_attempts",
    "avg_retry_count",
    "avg_failed_origin_plan_attempts",
    "avg_failed_template_attempts",
    "avg_failed_retry_count",
    "top_failure_stage",
    "top_failure_reason_code",
    "top_failed_check",
    "top_validation_reason",
    "harness_mismatch_count",
]

TEMPLATE_FIELDS = [
    "constraint_type",
    "bucket_name",
    "template_id",
    "template",
    "sample_count",
    "success_count",
    "fail_count",
    "final_success_rate",
    "avg_origin_plan_attempts",
    "avg_template_attempts",
    "avg_retry_count",
    "avg_failed_origin_plan_attempts",
    "avg_failed_template_attempts",
    "avg_failed_retry_count",
    "top_failure_stage",
    "top_failure_reason_code",
    "top_failed_check",
    "top_validation_reason",
    "harness_mismatch_count",
]


def load_batch_samples(batch_dir: str, *, include_archived: bool = False) -> List[Dict[str, Any]]:
    batch_path = Path(batch_dir)
    samples: List[Dict[str, Any]] = []
    for sample_file in sorted(batch_path.glob("sample_*.json")):
        with sample_file.open("r", encoding="utf-8") as handle:
            samples.append(json.load(handle))
    if include_archived:
        for sample_file in sorted((batch_path / "_failed").glob("**/sample_*.json")):
            with sample_file.open("r", encoding="utf-8") as handle:
                sample = json.load(handle)
            meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
            if not meta.get("status"):
                parent_status = sample_file.parent.name
                if parent_status:
                    meta = dict(meta)
                    meta["status"] = parent_status
                    sample["meta"] = meta
            samples.append(sample)
    return samples


def bucket_name_from_value(bucket: Any) -> str:
    if isinstance(bucket, (list, tuple)) and len(bucket) == 3:
        return "_".join(str(part).strip() for part in bucket)
    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_origin_plan_attempts(sample: Dict[str, Any]) -> int:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    recovery = meta.get("step2_recovery", {}) if isinstance(meta.get("step2_recovery"), dict) else {}
    used_origin_plan_files = recovery.get("used_origin_plan_files", [])
    if isinstance(used_origin_plan_files, list):
        unique_files = []
        for item in used_origin_plan_files:
            if isinstance(item, str) and item and item not in unique_files:
                unique_files.append(item)
        if unique_files:
            return len(unique_files)
    trace = sample.get("query_generation_trace", {})
    if isinstance(trace, dict):
        return max(_safe_int(trace.get("origin_attempt_count"), 0), 0)
    return 0


def extract_template_attempts(sample: Dict[str, Any]) -> int:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    recovery = meta.get("step2_recovery", {}) if isinstance(meta.get("step2_recovery"), dict) else {}
    attempts = recovery.get("template_attempts", [])
    if isinstance(attempts, list) and attempts:
        return len(attempts)
    trace = sample.get("query_generation_trace", {})
    if isinstance(trace, dict):
        return max(_safe_int(trace.get("template_attempt_count"), 0), 0)
    return 0


def infer_failure_stage(sample: Dict[str, Any]) -> str:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    status = str(meta.get("status", "") or "").strip()
    checks = sample.get("checks", {}) if isinstance(sample.get("checks"), dict) else {}
    failure_reason_code = str(meta.get("failure_reason_code", "") or "").strip()

    if status in {"08_failed", "08_completed_with_warnings"} and not checks.get("all_pass", False):
        return "validation"
    if status == "03_failed":
        return "step3"
    if status == "04_failed" or (status == "04_completed" and not meta.get("target_confirmed", False)):
        return "step4"
    if status.startswith("02_") or failure_reason_code in STEP2_REASON_CODES:
        return "step2"
    return ""


def _failed_check_entries(checks: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
    entries: List[tuple[str, Dict[str, Any]]] = []
    for name, payload in checks.items():
        if isinstance(payload, dict) and payload.get("pass") is False:
            entries.append((str(name), payload))
    return entries


def _extract_top_failed_check(checks: Dict[str, Any]) -> str:
    failed_entries = _failed_check_entries(checks)
    return failed_entries[0][0] if failed_entries else ""


def _extract_top_validation_reason(checks: Dict[str, Any]) -> str:
    for _, payload in _failed_check_entries(checks):
        for key in ("reason_code", "reason", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def detect_harness_mismatch(sample: Dict[str, Any], expected_constraint_type: Optional[str]) -> bool:
    if not expected_constraint_type:
        return False
    trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
    actual_constraint_type = str(trace.get("constraint_type", "") or "").strip()
    return bool(actual_constraint_type) and actual_constraint_type != expected_constraint_type


def build_sample_record(
    sample: Dict[str, Any],
    *,
    batch_dir: Optional[str] = None,
    expected_constraint_type: Optional[str] = None,
) -> Dict[str, Any]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
    checks = sample.get("checks", {}) if isinstance(sample.get("checks"), dict) else {}
    is_archived = "/_failed/" in str(batch_dir or "")
    mismatch = detect_harness_mismatch(sample, expected_constraint_type)
    final_success = bool(checks.get("all_pass", False)) and not mismatch
    failure_reason_code = str(meta.get("failure_reason_code", "") or "").strip()
    failure_stage = infer_failure_stage(sample)
    top_failed_check = _extract_top_failed_check(checks)
    top_validation_reason = _extract_top_validation_reason(checks)
    if not failure_reason_code and top_validation_reason:
        failure_reason_code = top_validation_reason
    elif not failure_reason_code and top_failed_check:
        failure_reason_code = top_failed_check
    if mismatch:
        failure_reason_code = "harness_constraint_type_mismatch"
        failure_stage = "step2"

    return {
        "sample_id": str(sample.get("sample_id", "") or ""),
        "batch_dir": str(batch_dir or ""),
        "batch_name": Path(batch_dir).name if batch_dir else "",
        "expected_constraint_type": str(expected_constraint_type or "").strip(),
        "constraint_type": str(trace.get("constraint_type", "") or "").strip(),
        "bucket_name": bucket_name_from_value(meta.get("target_bucket")),
        "template_id": str(trace.get("template_id", "") or "").strip(),
        "template": str(trace.get("template", "") or "").strip(),
        "status": str(meta.get("status", "") or "").strip(),
        "origin_plan_file": str(meta.get("origin_plan_file", "") or "").strip(),
        "origin_plan_attempts": extract_origin_plan_attempts(sample),
        "template_attempts": extract_template_attempts(sample),
        "retry_count": max(_safe_int(meta.get("retry_count_step2"), 0), 0),
        "failure_stage": failure_stage,
        "failure_reason_code": failure_reason_code,
        "failure_reason": str(meta.get("failure_reason", "") or "").strip(),
        "top_failed_check": top_failed_check,
        "top_validation_reason": top_validation_reason,
        "target_confirmed": bool(meta.get("target_confirmed", False)),
        "checks_all_pass": bool(checks.get("all_pass", False)),
        "harness_mismatch": mismatch,
        "final_success": final_success,
        "archived": is_archived,
    }


def _avg(values: Iterable[float]) -> float:
    values_list = list(values)
    return (sum(values_list) / len(values_list)) if values_list else 0.0


def _top_value(values: Iterable[str]) -> str:
    normalized = [value for value in values if value]
    if not normalized:
        return ""
    return Counter(normalized).most_common(1)[0][0]


def classify_support_status(summary: Dict[str, Any]) -> str:
    if summary.get("constraint_type") in OUT_OF_SCOPE_CONSTRAINT_TYPES:
        return OUT_OF_SCOPE_SUPPORT_STATUS
    explicit_status = str(summary.get("explicit_support_status", "") or "").strip()
    if explicit_status:
        return explicit_status
    success_rate = _safe_float(summary.get("final_success_rate"), 0.0)
    success_count = _safe_int(summary.get("success_count"), 0)
    sample_count = _safe_int(summary.get("sample_count"), 0)
    observed_sample_count = _safe_int(summary.get("observed_sample_count"), sample_count)
    archived_fail_count = _safe_int(summary.get("archived_fail_count"), 0)
    avg_origin = _safe_float(summary.get("avg_origin_plan_attempts"), 0.0)
    avg_template = _safe_float(summary.get("avg_template_attempts"), 0.0)
    avg_retry = _safe_float(summary.get("avg_retry_count"), 0.0)
    dominant_failure_ratio = _safe_float(summary.get("top_failure_reason_ratio"), 0.0)
    top_failure_stage = str(summary.get("top_failure_stage", "") or "").strip()

    if observed_sample_count == 0:
        if archived_fail_count > 0 and top_failure_stage == "step3":
            return UNSUPPORTED_STEP3_STATUS
        return NO_DATA_SUPPORT_STATUS

    if (
        success_rate >= 0.8
        and avg_origin <= 2
        and avg_template <= 6
        and avg_retry <= 1
    ):
        return STABLE_SUPPORT_STATUS
    if success_rate < 0.3 or success_count == 0 or dominant_failure_ratio >= 0.7:
        if top_failure_stage == "step3":
            return UNSUPPORTED_STEP3_STATUS
        return UNSUPPORTED_VALIDATION_STATUS
    return FRAGILE_SUPPORT_STATUS


def _summarize_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    observed_records = [record for record in records if not record.get("archived", False)]
    archived_records = [record for record in records if record.get("archived", False)]
    failed_records = [record for record in observed_records if not record.get("final_success", False)]
    failure_pool = [record for record in records if not record.get("final_success", False)]
    failure_reason_counter = Counter(
        str(record.get("failure_reason_code", "") or "").strip()
        for record in failure_pool
        if str(record.get("failure_reason_code", "") or "").strip()
    )
    failed_check_counter = Counter(
        str(record.get("top_failed_check", "") or "").strip()
        for record in failure_pool
        if str(record.get("top_failed_check", "") or "").strip()
    )
    validation_reason_counter = Counter(
        str(record.get("top_validation_reason", "") or "").strip()
        for record in failure_pool
        if str(record.get("top_validation_reason", "") or "").strip()
    )
    top_failure_reason_code = failure_reason_counter.most_common(1)[0][0] if failure_reason_counter else ""
    top_failure_reason_ratio = (
        failure_reason_counter.most_common(1)[0][1] / len(failure_pool)
        if failure_reason_counter and failure_pool
        else 0.0
    )
    return {
        "sample_count": len(observed_records),
        "observed_sample_count": len(observed_records),
        "archived_fail_count": len(archived_records),
        "success_count": sum(1 for record in observed_records if record.get("final_success", False)),
        "fail_count": sum(1 for record in observed_records if not record.get("final_success", False)),
        "final_success_rate": _avg([1.0 if record.get("final_success", False) else 0.0 for record in observed_records]),
        "avg_origin_plan_attempts": _avg([_safe_float(record.get("origin_plan_attempts"), 0.0) for record in observed_records]),
        "avg_template_attempts": _avg([_safe_float(record.get("template_attempts"), 0.0) for record in observed_records]),
        "avg_retry_count": _avg([_safe_float(record.get("retry_count"), 0.0) for record in observed_records]),
        "avg_failed_origin_plan_attempts": _avg(
            [_safe_float(record.get("origin_plan_attempts"), 0.0) for record in failed_records]
        ),
        "avg_failed_template_attempts": _avg(
            [_safe_float(record.get("template_attempts"), 0.0) for record in failed_records]
        ),
        "avg_failed_retry_count": _avg([_safe_float(record.get("retry_count"), 0.0) for record in failed_records]),
        "top_failure_stage": _top_value(
            [str(record.get("failure_stage", "") or "").strip() for record in failure_pool]
        ),
        "top_failure_reason_code": top_failure_reason_code,
        "top_failed_check": failed_check_counter.most_common(1)[0][0] if failed_check_counter else "",
        "top_validation_reason": validation_reason_counter.most_common(1)[0][0] if validation_reason_counter else "",
        "top_failure_reason_ratio": top_failure_reason_ratio,
        "harness_mismatch_count": sum(1 for record in observed_records if record.get("harness_mismatch", False)),
    }


def aggregate_by_constraint_type(
    records: List[Dict[str, Any]],
    *,
    requested_constraint_types: Optional[Sequence[str]] = None,
    explicit_status_by_type: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get("expected_constraint_type") or record.get("constraint_type") or "").strip()
        if not key:
            key = "unknown"
        groups[key].append(record)
    for constraint_type in requested_constraint_types or []:
        groups.setdefault(str(constraint_type), [])

    rows: List[Dict[str, Any]] = []
    for constraint_type in sorted(groups):
        summary = _summarize_group(groups[constraint_type])
        row = {
            "constraint_type": constraint_type,
            "explicit_support_status": (explicit_status_by_type or {}).get(constraint_type, ""),
            **summary,
        }
        row["final_support_status"] = classify_support_status(row)
        row.pop("top_failure_reason_ratio", None)
        row.pop("explicit_support_status", None)
        rows.append(row)
    return rows


def aggregate_bucket_breakdown(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        constraint_type = str(record.get("expected_constraint_type") or record.get("constraint_type") or "").strip() or "unknown"
        bucket_name = str(record.get("bucket_name", "") or "").strip()
        groups[(constraint_type, bucket_name)].append(record)

    rows: List[Dict[str, Any]] = []
    for (constraint_type, bucket_name), grouped_records in sorted(groups.items()):
        row = {
            "constraint_type": constraint_type,
            "bucket_name": bucket_name,
            **_summarize_group(grouped_records),
        }
        row.pop("top_failure_reason_ratio", None)
        rows.append(row)
    return rows


def aggregate_template_breakdown(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        constraint_type = str(record.get("expected_constraint_type") or record.get("constraint_type") or "").strip() or "unknown"
        bucket_name = str(record.get("bucket_name", "") or "").strip()
        template_id = str(record.get("template_id", "") or "").strip()
        template = str(record.get("template", "") or "").strip()
        groups[(constraint_type, bucket_name, template_id, template)].append(record)

    rows: List[Dict[str, Any]] = []
    for (constraint_type, bucket_name, template_id, template), grouped_records in sorted(groups.items()):
        row = {
            "constraint_type": constraint_type,
            "bucket_name": bucket_name,
            "template_id": template_id,
            "template": template,
            **_summarize_group(grouped_records),
        }
        row.pop("top_failure_reason_ratio", None)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report_bundle(
    *,
    records: List[Dict[str, Any]],
    output_dir: str,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_runs = manifest_extra.get("batch_runs", []) if isinstance(manifest_extra, dict) else []
    requested_constraint_types = manifest_extra.get("constraint_types", []) if isinstance(manifest_extra, dict) else []
    explicit_status_by_type: Dict[str, str] = {}
    for item in batch_runs if isinstance(batch_runs, list) else []:
        if not isinstance(item, dict):
            continue
        constraint_type = str(item.get("constraint_type", "") or "").strip()
        explicit_status = str(item.get("explicit_support_status", "") or "").strip()
        if constraint_type and explicit_status:
            explicit_status_by_type[constraint_type] = explicit_status

    summary_rows = aggregate_by_constraint_type(
        records,
        requested_constraint_types=requested_constraint_types,
        explicit_status_by_type=explicit_status_by_type,
    )
    bucket_rows = aggregate_bucket_breakdown(records)
    template_rows = aggregate_template_breakdown(records)
    mismatches = [record for record in records if record.get("harness_mismatch", False)]

    summary_json_path = output_path / "summary_by_constraint_type.json"
    summary_csv_path = output_path / "summary_by_constraint_type.csv"
    bucket_csv_path = output_path / "bucket_breakdown.csv"
    template_csv_path = output_path / "template_breakdown.csv"
    sample_jsonl_path = output_path / "sample_records.jsonl"
    manifest_path = output_path / "manifest.json"

    summary_json_path.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "rows": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(summary_csv_path, summary_rows, SUMMARY_FIELDS)
    _write_csv(bucket_csv_path, bucket_rows, BUCKET_FIELDS)
    _write_csv(template_csv_path, template_rows, TEMPLATE_FIELDS)
    _write_jsonl(sample_jsonl_path, records)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "record_count": len(records),
        "constraint_type_count": len(summary_rows),
        "bucket_group_count": len(bucket_rows),
        "template_group_count": len(template_rows),
        "harness_mismatch_count": len(mismatches),
        "harness_mismatches": [
            {
                "sample_id": record.get("sample_id"),
                "expected_constraint_type": record.get("expected_constraint_type"),
                "constraint_type": record.get("constraint_type"),
                "batch_name": record.get("batch_name"),
            }
            for record in mismatches
        ],
        "outputs": {
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "bucket_csv": str(bucket_csv_path),
            "template_csv": str(template_csv_path),
            "sample_jsonl": str(sample_jsonl_path),
        },
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
