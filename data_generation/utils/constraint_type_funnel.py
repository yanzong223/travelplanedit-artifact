"""
Constraint-type funnel diagnostics for data-generation batches.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _bucket_key(bucket: Any) -> str:
    if isinstance(bucket, (list, tuple)):
        return "_".join(str(part).strip() for part in bucket if str(part).strip())
    return str(bucket or "").strip()


def _normalize_bucket_lookup_key(bucket: Any) -> str:
    if isinstance(bucket, tuple):
        return _bucket_key(bucket)
    text = str(bucket or "").strip()
    if text.startswith("(") and text.endswith(")"):
        parts = [part.strip() for part in text[1:-1].split(",") if part.strip()]
        if parts:
            return "_".join(parts)
    return _bucket_key(bucket)


def _load_template_index(templates_path: str) -> Dict[str, List[str]]:
    raw = yaml.safe_load(Path(templates_path).read_text(encoding="utf-8")) or {}
    index: Dict[str, List[str]] = {}
    for bucket_key, bucket_cfg in raw.items():
        templates = bucket_cfg.get("templates", []) if isinstance(bucket_cfg, dict) else []
        index[_normalize_bucket_lookup_key(bucket_key)] = [
            str(item.get("constraint_type", "")).strip()
            for item in templates
            if isinstance(item, dict) and str(item.get("constraint_type", "")).strip()
        ]
    return index


def _iter_sample_files(batch_dir: str) -> Iterable[Path]:
    batch_path = Path(batch_dir)
    yield from sorted(batch_path.glob("sample_*.json"))
    failed_root = batch_path / "_failed"
    if failed_root.exists():
        yield from sorted(failed_root.rglob("sample_*.json"))


def _load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for sample_file in _iter_sample_files(batch_dir):
        try:
            payload = json.loads(sample_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["_diagnostic_path"] = str(sample_file)
                samples.append(payload)
        except Exception:
            continue
    return samples


def _infer_constraint_type(sample: Dict[str, Any]) -> Optional[str]:
    trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace", {}), dict) else {}
    value = trace.get("constraint_type") or sample.get("constraint_type")
    text = str(value or "").strip()
    return text or None


def _eligible_types_for_sample(sample: Dict[str, Any], template_index: Dict[str, List[str]]) -> List[str]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
    bucket_value = (
        meta.get("target_bucket")
        or sample.get("target_bucket")
        or trace.get("target_bucket")
        or meta.get("sampled_bucket")
        or meta.get("bucket")
    )
    bucket_key = _bucket_key(bucket_value)
    return list(dict.fromkeys(template_index.get(bucket_key, [])))


def _normalize_bucket_tuple(bucket: Any) -> Optional[Tuple[str, str, str]]:
    if isinstance(bucket, (list, tuple)) and len(bucket) == 3:
        return tuple(str(part or "").strip() for part in bucket)
    return None


def _conflict_set_contains_target(sample: Dict[str, Any]) -> bool:
    target = _normalize_bucket_tuple((sample.get("meta", {}) or {}).get("target_bucket"))
    if target is None:
        return False
    conflict_set = sample.get("conflict_set")
    if not isinstance(conflict_set, list):
        return False
    return any(_normalize_bucket_tuple(item) == target for item in conflict_set)


def _primary_aligned_in_conflict_analysis(sample: Dict[str, Any]) -> bool:
    checks = sample.get("checks", {}) if isinstance(sample.get("checks"), dict) else {}
    payload = checks.get("primary_target_alignment")
    if isinstance(payload, dict) and payload.get("pass") is not None:
        return bool(payload.get("pass"))

    target = _normalize_bucket_tuple((sample.get("meta", {}) or {}).get("target_bucket"))
    primary = _normalize_bucket_tuple(sample.get("primary_conflict"))
    if target is None or primary is None:
        return False
    return primary == target


def _status_step(status: str) -> Optional[str]:
    text = str(status or "").strip()
    if not text:
        return None
    prefix = text.split("_", 1)[0]
    return prefix if prefix in {"01", "02", "03", "04", "05", "06", "07", "08", "09"} else None


def _first_failure(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    status = str(meta.get("status", "") or "")
    if "_failed" in status:
        return _status_step(status), str(meta.get("failure_reason_code", "") or status)

    checks = sample.get("checks", {}) if isinstance(sample.get("checks"), dict) else {}
    for name, payload in checks.items():
        if not isinstance(payload, dict):
            continue
        if payload.get("pass") is False:
            return "09", name
    return None, None


def build_constraint_type_funnel(
    batch_dir: str,
    templates_path: str,
) -> Dict[str, Any]:
    template_index = _load_template_index(templates_path)
    configured_types = sorted({ctype for values in template_index.values() for ctype in values})
    samples = _load_samples(batch_dir)

    metrics: Dict[str, Dict[str, int]] = {
        ctype: {
            "configured": 1,
            "sampled_step1": 0,
            "generated_step2": 0,
            "canonical_proof_pass_step2": 0,
            "frozen_truth_pass_step3": 0,
            "target_present_step5": 0,
            "primary_aligned_step5": 0,
            "validation_pass_step9": 0,
        }
        for ctype in configured_types
    }
    failure_steps: Dict[str, Counter[str]] = defaultdict(Counter)
    failure_reasons: Dict[str, Counter[str]] = defaultdict(Counter)
    only_failed_presence: Dict[str, bool] = {ctype: False for ctype in configured_types}
    generated_presence: Dict[str, bool] = {ctype: False for ctype in configured_types}

    for sample in samples:
        sample_path = str(sample.get("_diagnostic_path", "") or "")
        eligible_types = _eligible_types_for_sample(sample, template_index)
        for ctype in eligible_types:
            metrics.setdefault(ctype, {}).setdefault("sampled_step1", 0)
            metrics[ctype]["sampled_step1"] += 1

        ctype = _infer_constraint_type(sample)
        if not ctype:
            continue
        metrics.setdefault(
            ctype,
            {
                "configured": 0,
                "sampled_step1": 0,
                "generated_step2": 0,
                "canonical_proof_pass_step2": 0,
                "frozen_truth_pass_step3": 0,
                "target_present_step5": 0,
                "primary_aligned_step5": 0,
                "validation_pass_step9": 0,
            },
        )
        metrics[ctype]["generated_step2"] += 1
        generated_presence[ctype] = True
        if "/_failed/" in sample_path:
            only_failed_presence[ctype] = True
        else:
            only_failed_presence[ctype] = False

        trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace", {}), dict) else {}
        if trace.get("canonical_proof_ok") is True:
            metrics[ctype]["canonical_proof_pass_step2"] += 1

        status = str(sample.get("meta", {}).get("status", "") or "")
        step_number = _status_step(status)
        if step_number and int(step_number) >= 3 and not status.startswith("03_failed"):
            metrics[ctype]["frozen_truth_pass_step3"] += 1
        if _conflict_set_contains_target(sample):
            metrics[ctype]["target_present_step5"] += 1
        if _primary_aligned_in_conflict_analysis(sample):
            metrics[ctype]["primary_aligned_step5"] += 1
        if sample.get("checks", {}).get("all_pass") is True:
            metrics[ctype]["validation_pass_step9"] += 1

        failure_step, failure_reason = _first_failure(sample)
        if failure_step:
            failure_steps[ctype][failure_step] += 1
        if failure_reason:
            failure_reasons[ctype][failure_reason] += 1

    per_type: Dict[str, Dict[str, Any]] = {}
    for ctype in sorted(metrics):
        failure_step = failure_steps[ctype].most_common(1)[0][0] if failure_steps[ctype] else None
        failure_reason = failure_reasons[ctype].most_common(1)[0][0] if failure_reasons[ctype] else None
        per_type[ctype] = {
            **metrics[ctype],
            "never_generated": metrics[ctype]["generated_step2"] == 0,
            "only_in_failed_dir": bool(only_failed_presence.get(ctype)) and metrics[ctype]["generated_step2"] > 0 and metrics[ctype]["validation_pass_step9"] == 0,
            "top_failure_step": failure_step,
            "top_failure_reason": failure_reason,
        }

    return {
        "batch_dir": str(Path(batch_dir).resolve()),
        "templates_path": str(Path(templates_path).resolve()),
        "total_samples_considered": len(samples),
        "configured_constraint_types": configured_types,
        "per_type": per_type,
    }


def write_constraint_type_funnel(
    batch_dir: str,
    templates_path: str,
) -> Dict[str, Any]:
    report = build_constraint_type_funnel(batch_dir, templates_path)
    analysis_dir = Path(batch_dir) / "_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    json_path = analysis_dir / "constraint_type_funnel.json"
    csv_path = analysis_dir / "constraint_type_funnel.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "constraint_type",
        "configured",
        "sampled_step1",
        "generated_step2",
        "canonical_proof_pass_step2",
        "frozen_truth_pass_step3",
        "target_present_step5",
        "primary_aligned_step5",
        "validation_pass_step9",
        "never_generated",
        "only_in_failed_dir",
        "top_failure_step",
        "top_failure_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for ctype, payload in sorted(report["per_type"].items()):
            writer.writerow({"constraint_type": ctype, **payload})

    report["json_path"] = str(json_path)
    report["csv_path"] = str(csv_path)
    return report
