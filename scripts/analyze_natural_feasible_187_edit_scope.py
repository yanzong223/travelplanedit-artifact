#!/usr/bin/env python3
"""Analyze Actual Edit Scope in the natural-feasible-187 experiment matrix.

The analysis is deliberately eligible-only: Level 3 describes outputs that pass
both Plan Validity (Level 1) and Request Satisfaction (Level 2).  It does not
treat the request-side query class as a gold/reference edit scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

from evaluation.edit_scope import algorithm_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "human_baseline_data"
    / "batch006_four_types_natural_feasible_187_20260803_v1"
)
MODEL_LABELS = {
    "deepseek_v4_pro_guan": "DeepSeek V4 Pro",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "gpt-4o-mini": "GPT-4o-mini",
    "qwen3_32b": "Qwen3-32B",
}
METHOD_LABELS = {"pter": "PTER", "react": "ReAct", "reflexion": "Reflexion"}
QUERY_CLASS_LABELS = {
    "point": "Entity-targeted",
    "range": "Range-constrained",
    "set_predicate": "Predicate-defined",
    "global": "Plan-level",
}
SCOPE_ORDER = ("parameter", "structural", "compositional")
ATOMIC_ORDER = (
    "change_time", "change_transport", "change_attribute",
    "insert", "delete", "replace", "reorder",
)
METRIC_FIELDS = (
    "parameter_count",
    "structural_count",
    "compositional_count",
    "content_retention_rate",
    "poi_seq_edit_distance",
    "activity_change_ratio",
)
TRANSPORT_DIAGNOSTIC_FIELDS = (
    "transport_raw_rewrite_count",
    "transport_equivalent_format_or_metadata_count",
    "transport_semantic_change_count",
    "transport_mode_change_count",
    "transport_duration_or_timing_change_count",
    "transport_cost_or_capacity_change_count",
    "transport_endpoint_change_count",
    "transport_distance_change_count",
    "transport_topology_change_count",
    "transport_other_semantic_change_count",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _describe(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    return {
        "n": len(clean),
        "mean": fmean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "p25": _percentile(clean, 0.25),
        "p75": _percentile(clean, 0.75),
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
    }


def _paired_bootstrap(
    left: list[float], right: list[float], *, seed_text: str, draws: int = 5000
) -> dict[str, float | int | None]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have the same length")
    if not left:
        return {"n": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None}
    deltas = [a - b for a, b in zip(left, right, strict=True)]
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    bootstrap_means = [
        fmean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(draws)
    ]
    return {
        "n": len(deltas),
        "mean_delta": fmean(deltas),
        "ci95_low": _percentile(bootstrap_means, 0.025),
        "ci95_high": _percentile(bootstrap_means, 0.975),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    eligible_rows = [row for row in rows if row["eligible"]]
    eligible = len(eligible_rows)
    scope_counts = Counter(row["actual_scope"] for row in eligible_rows)
    atomic_counts = Counter()
    for row in eligible_rows:
        for field in ATOMIC_ORDER:
            atomic_counts[field] += int(row[field])
    cascade_rows = [row for row in eligible_rows if row["cascade_status"] == "evaluated"]
    cascade_statuses = Counter(row["cascade_status"] for row in eligible_rows)
    cascade_count_fields = (
        "cascade_direct_target_count",
        "cascade_rollback_required_support_count",
        "cascade_authorized_count",
        "cascade_verified_removable_count",
        "cascade_unresolved_count",
    )
    pooled_cascade_counts = {
        field: sum(int(row[field] or 0) for row in cascade_rows)
        for field in cascade_count_fields
    }
    pooled_cascade_total = sum(int(row["cascade_total_impact_count"] or 0) for row in cascade_rows)
    baseline_cascade_rows = [
        row
        for row in rows
        if row["baseline_cascade_status"] == "evaluated"
    ]
    baseline_pooled_cascade_counts = {
        field: sum(int(row[f"baseline_{field}"] or 0) for row in baseline_cascade_rows)
        for field in cascade_count_fields
    }
    baseline_pooled_cascade_total = sum(
        int(row["baseline_cascade_total_impact_count"] or 0)
        for row in baseline_cascade_rows
    )
    return {
        "records": total,
        "eligible_records": eligible,
        "eligible_rate": eligible / total if total else None,
        "scope_distribution": {
            scope: {
                "count": scope_counts[scope],
                "rate": scope_counts[scope] / eligible if eligible else None,
            }
            for scope in SCOPE_ORDER
        },
        "metrics": {field: _describe(row[field] for row in eligible_rows) for field in METRIC_FIELDS},
        "atomic_operation_totals": {field: atomic_counts[field] for field in ATOMIC_ORDER},
        "evaluator_revision_impact": {
            "eligibility_transitions": {
                "baseline_eligible": sum(bool(row["baseline_eligible"]) for row in rows),
                "current_eligible": eligible,
                "newly_eligible": sum(
                    row["eligible"] and not row["baseline_eligible"] for row in rows
                ),
                "no_longer_eligible": sum(
                    row["baseline_eligible"] and not row["eligible"] for row in rows
                ),
                "eligible_in_both": sum(
                    row["baseline_eligible"] and row["eligible"] for row in rows
                ),
            },
            "baseline_activity_change_ratio": _describe(
                row.get("baseline_activity_change_ratio") for row in eligible_rows
            ),
            "field_aware_activity_change_ratio": _describe(
                row.get("activity_change_ratio") for row in eligible_rows
            ),
            "mean_activity_change_ratio_delta": _describe(
                row.get("activity_change_ratio_delta") for row in eligible_rows
            ),
            "scope_label_changed_count": sum(
                row.get("baseline_eligible")
                and row.get("baseline_actual_scope") != row.get("actual_scope")
                for row in eligible_rows
            ),
            "baseline_zero_change_count": sum(
                row.get("baseline_activity_change_ratio") == 0 for row in eligible_rows
            ),
            "field_aware_zero_change_count": sum(
                row.get("activity_change_ratio") == 0 for row in eligible_rows
            ),
        },
        "transport_change_diagnostics": {
            field: {
                "total": sum(int(row[field]) for row in eligible_rows),
                "affected_outputs": sum(int(row[field]) > 0 for row in eligible_rows),
                "affected_output_rate": (
                    sum(int(row[field]) > 0 for row in eligible_rows) / eligible
                    if eligible
                    else None
                ),
            }
            for field in TRANSPORT_DIAGNOSTIC_FIELDS
        },
        "cascade_quality": {
            "definition": (
                "Eligible-only evaluator-relative rollback metric; it is not a third success gate."
            ),
            "status_counts": dict(sorted(cascade_statuses.items())),
            "evaluated_records": len(cascade_rows),
            "mean_verified_removable_change_rate": _describe(
                row["verified_removable_change_rate"] for row in cascade_rows
            ),
            "mean_evaluator_relative_excess_upper": _describe(
                row["evaluator_relative_excess_change_rate_upper"] for row in cascade_rows
            ),
            "mean_proof_coverage": _describe(row["proof_coverage"] for row in cascade_rows),
            "mean_unresolved_rate": _describe(row["unresolved_rate"] for row in cascade_rows),
            "pooled_counts": pooled_cascade_counts,
            "pooled_total_impact_count": pooled_cascade_total,
            "pooled_verified_removable_change_rate": (
                pooled_cascade_counts["cascade_verified_removable_count"] / pooled_cascade_total
                if pooled_cascade_total
                else None
            ),
            "pooled_evaluator_relative_excess_upper": (
                (
                    pooled_cascade_counts["cascade_verified_removable_count"]
                    + pooled_cascade_counts["cascade_unresolved_count"]
                ) / pooled_cascade_total
                if pooled_cascade_total
                else None
            ),
            "pooled_proof_coverage": (
                (
                    pooled_cascade_counts["cascade_direct_target_count"]
                    + pooled_cascade_counts[
                        "cascade_rollback_required_support_count"
                    ]
                    + pooled_cascade_counts["cascade_verified_removable_count"]
                ) / pooled_cascade_total
                if pooled_cascade_total
                else None
            ),
            "algorithm_version_migration": {
                "baseline_evaluated_records": len(baseline_cascade_rows),
                "current_evaluated_records": len(cascade_rows),
                "baseline_pooled_counts": baseline_pooled_cascade_counts,
                "current_pooled_counts": pooled_cascade_counts,
                "count_deltas_current_minus_baseline": {
                    field: pooled_cascade_counts[field]
                    - baseline_pooled_cascade_counts[field]
                    for field in cascade_count_fields
                },
                "baseline_pooled_total_impact_count": baseline_pooled_cascade_total,
                "current_pooled_total_impact_count": pooled_cascade_total,
            },
        },
        "tail_diagnostics": {
            "zero_activity_change_count": sum(
                row["activity_change_ratio"] == 0 for row in eligible_rows
            ),
            "activity_change_ge_50pct_count": sum(
                row["activity_change_ratio"] >= 0.5 for row in eligible_rows
            ),
            "activity_change_ge_100pct_count": sum(
                row["activity_change_ratio"] >= 1.0 for row in eligible_rows
            ),
            "seq_distance_ge_3_count": sum(
                row["poi_seq_edit_distance"] >= 3 for row in eligible_rows
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary(
    dimensions: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    row = {
        **dimensions,
        "records": summary["records"],
        "eligible_records": summary["eligible_records"],
        "eligible_rate": summary["eligible_rate"],
    }
    for scope in SCOPE_ORDER:
        row[f"scope_{scope}_count"] = summary["scope_distribution"][scope]["count"]
        row[f"scope_{scope}_rate"] = summary["scope_distribution"][scope]["rate"]
    for field in METRIC_FIELDS:
        for stat in ("mean", "median", "p25", "p75"):
            row[f"{field}_{stat}"] = summary["metrics"][field][stat]
    row.update(summary["atomic_operation_totals"])
    for field in TRANSPORT_DIAGNOSTIC_FIELDS:
        row[f"{field}_total"] = summary["transport_change_diagnostics"][field]["total"]
        row[f"{field}_affected_outputs"] = summary["transport_change_diagnostics"][field][
            "affected_outputs"
        ]
        row[f"{field}_affected_output_rate"] = summary["transport_change_diagnostics"][field][
            "affected_output_rate"
        ]
    cascade = summary["cascade_quality"]
    row["cascade_evaluated_records"] = cascade["evaluated_records"]
    row["verified_removable_change_rate_mean"] = cascade[
        "mean_verified_removable_change_rate"
    ]["mean"]
    row["evaluator_relative_excess_upper_mean"] = cascade[
        "mean_evaluator_relative_excess_upper"
    ]["mean"]
    row["proof_coverage_mean"] = cascade["mean_proof_coverage"]["mean"]
    row["unresolved_rate_mean"] = cascade["mean_unresolved_rate"]["mean"]
    row.update(summary["tail_diagnostics"])
    revision = summary["evaluator_revision_impact"]
    row["baseline_activity_change_ratio_mean"] = revision[
        "baseline_activity_change_ratio"
    ]["mean"]
    row["field_aware_activity_change_ratio_mean"] = revision[
        "field_aware_activity_change_ratio"
    ]["mean"]
    row["mean_activity_change_ratio_delta"] = revision[
        "mean_activity_change_ratio_delta"
    ]["mean"]
    row["scope_label_changed_count"] = revision["scope_label_changed_count"]
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--baseline-reports-dir", type=Path)
    parser.add_argument(
        "--allow-eligibility-change",
        action="store_true",
        help=(
            "Permit eligibility changes relative to the baseline reports. "
            "Use this for evaluator/ground-truth migrations, not field-only recomputes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _activity_map(plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    activities: dict[str, dict[str, Any]] = {}
    for day_index, day in enumerate((plan or {}).get("itinerary", []), start=1):
        day_number = int(day.get("day", day_index) or day_index)
        for activity_index, activity in enumerate(day.get("activities", [])):
            activities[f"day{day_number}_act{activity_index}"] = activity
    return activities


def _transport_change_diagnostics(level3: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for pair in level3.get("matched_pairs") or []:
        details = pair.get("details") if isinstance(pair.get("details"), dict) else {}
        if not details.get("transport_raw_changed"):
            continue
        counts["transport_raw_rewrite_count"] += 1
        if details.get("transport_change_kind") == "format_or_metadata_only":
            counts["transport_equivalent_format_or_metadata_count"] += 1
            continue
        if not details.get("transport_semantic_changed"):
            continue
        counts["transport_semantic_change_count"] += 1
        for dimension in details.get("transport_change_dimensions") or []:
            field = f"transport_{dimension}_change_count"
            if field in TRANSPORT_DIAGNOSTIC_FIELDS:
                counts[field] += 1
    return {field: counts[field] for field in TRANSPORT_DIAGNOSTIC_FIELDS}


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = dataset_root / "manifest.json"
    matrix_manifest_path = (
        dataset_root
        / "experiment_results"
        / "main_three_methods_four_models"
        / "manifest_natural_feasible_187.json"
    )
    reports_dir = (
        args.reports_dir.resolve()
        if args.reports_dir
        else dataset_root
        / "experiment_results"
        / "reports_natural_feasible_187_20260803_v1"
    )
    baseline_reports_dir = (
        args.baseline_reports_dir.resolve()
        if args.baseline_reports_dir
        else dataset_root
        / "experiment_results"
        / "reports_natural_feasible_187_20260803_v1"
    )
    dataset_manifest = _load(manifest_path)
    matrix_manifest = _load(matrix_manifest_path)
    task_meta = {item["task_id"]: item for item in dataset_manifest["tasks"]}
    expected_task_ids = set(task_meta)
    expected_task_count = int(dataset_manifest["task_count"])
    if expected_task_count != 187 or len(expected_task_ids) != expected_task_count:
        raise ValueError("dataset manifest is not the expected unique 187-task pool")

    record_rows: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    report_sources: list[dict[str, Any]] = []
    for method, method_block in matrix_manifest["methods"].items():
        for model, model_block in method_block["models"].items():
            matrix_records = list(model_block["records"])
            task_ids = [record["task_id"] for record in matrix_records]
            if len(task_ids) != expected_task_count or set(task_ids) != expected_task_ids:
                raise ValueError(f"task-set mismatch for {model}/{method}")
            report_path = reports_dir / f"{model}_{method}_all187_benchmark_report.json"
            report = _load(report_path)
            results = list(report["results"])
            baseline_report_path = (
                baseline_reports_dir / f"{model}_{method}_all187_benchmark_report.json"
            )
            baseline_results = list(_load(baseline_report_path)["results"])
            if len(results) != expected_task_count:
                raise ValueError(f"report length mismatch for {model}/{method}")
            if len(baseline_results) != expected_task_count:
                raise ValueError(f"baseline report length mismatch for {model}/{method}")
            report_sources.append(
                {
                    "model": model,
                    "method": method,
                    "path": str(report_path.relative_to(PROJECT_ROOT)),
                }
            )
            group_rows: list[dict[str, Any]] = []
            for matrix_record, result, baseline_result in zip(
                matrix_records, results, baseline_results, strict=True
            ):
                task_id = matrix_record["task_id"]
                meta = task_meta[task_id]
                level3 = result.get("level3") or {}
                baseline_level3 = baseline_result.get("level3") or {}
                if (
                    not args.allow_eligibility_change
                    and bool(level3.get("eligible"))
                    != bool(baseline_level3.get("eligible"))
                ):
                    raise ValueError(
                        f"eligibility changed during field-only recompute: {model}/{method}/{task_id}"
                    )
                atomic = level3.get("atomic_counts") or {}
                if level3.get("eligible"):
                    transport_diagnostics = _transport_change_diagnostics(level3)
                else:
                    transport_diagnostics = {
                        field: None for field in TRANSPORT_DIAGNOSTIC_FIELDS
                    }
                cascade = level3.get("cascade") if isinstance(level3.get("cascade"), dict) else {}
                cascade_counts = (
                    cascade.get("counts") if isinstance(cascade.get("counts"), dict) else {}
                )
                baseline_cascade = (
                    baseline_level3.get("cascade")
                    if isinstance(baseline_level3.get("cascade"), dict)
                    else {}
                )
                baseline_cascade_counts = (
                    baseline_cascade.get("counts")
                    if isinstance(baseline_cascade.get("counts"), dict)
                    else {}
                )
                row = {
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "task_id": task_id,
                    "sample_id": (meta.get("source") or {}).get("sample_id"),
                    "query_class": meta["query_class"],
                    "query_class_label": QUERY_CLASS_LABELS.get(
                        meta["query_class"], meta["query_class"]
                    ),
                    "query_type": meta.get("query_type"),
                    "eligible": bool(level3.get("eligible")),
                    "baseline_eligible": bool(baseline_level3.get("eligible")),
                    "eligibility_reason": level3.get("reason"),
                    "actual_scope": level3.get("scope_name") if level3.get("eligible") else None,
                    "baseline_actual_scope": (
                        baseline_level3.get("scope_name")
                        if level3.get("eligible") and baseline_level3.get("eligible")
                        else None
                    ),
                    "parameter_count": level3.get("parameter_count") if level3.get("eligible") else None,
                    "structural_count": level3.get("structural_count") if level3.get("eligible") else None,
                    "compositional_count": level3.get("compositional_count") if level3.get("eligible") else None,
                    "content_retention_rate": level3.get("content_retention_rate") if level3.get("eligible") else None,
                    "poi_seq_edit_distance": level3.get("poi_seq_edit_distance") if level3.get("eligible") else None,
                    "activity_change_ratio": level3.get("activity_change_ratio") if level3.get("eligible") else None,
                    "baseline_activity_change_ratio": (
                        baseline_level3.get("activity_change_ratio")
                        if level3.get("eligible") and baseline_level3.get("eligible")
                        else None
                    ),
                    "activity_change_ratio_delta": (
                        float(level3.get("activity_change_ratio", 0) or 0)
                        - float(baseline_level3.get("activity_change_ratio", 0) or 0)
                        if level3.get("eligible") and baseline_level3.get("eligible")
                        else None
                    ),
                    **{
                        field: int(atomic.get(field, 0)) if level3.get("eligible") else None
                        for field in ATOMIC_ORDER
                    },
                    **transport_diagnostics,
                    "cascade_status": cascade.get("status") if level3.get("eligible") else None,
                    "cascade_total_impact_count": cascade.get("total_impact_count"),
                    "cascade_direct_target_count": cascade_counts.get("direct_target"),
                    "cascade_rollback_required_support_count": cascade_counts.get(
                        "rollback_required_support",
                        cascade_counts.get("hard_required_support"),
                    ),
                    "cascade_authorized_count": cascade_counts.get("scope_authorized_completion"),
                    "cascade_verified_removable_count": cascade_counts.get(
                        "verified_removable",
                        cascade_counts.get("avoidable_extra"),
                    ),
                    "cascade_unresolved_count": cascade_counts.get("unresolved"),
                    "baseline_cascade_status": baseline_cascade.get("status"),
                    "baseline_cascade_total_impact_count": baseline_cascade.get(
                        "total_impact_count"
                    ),
                    "baseline_cascade_direct_target_count": baseline_cascade_counts.get(
                        "direct_target"
                    ),
                    "baseline_cascade_rollback_required_support_count": (
                        baseline_cascade_counts.get(
                            "rollback_required_support",
                            baseline_cascade_counts.get("hard_required_support"),
                        )
                    ),
                    "baseline_cascade_authorized_count": baseline_cascade_counts.get(
                        "scope_authorized_completion"
                    ),
                    "baseline_cascade_verified_removable_count": (
                        baseline_cascade_counts.get(
                            "verified_removable",
                            baseline_cascade_counts.get("avoidable_extra"),
                        )
                    ),
                    "baseline_cascade_unresolved_count": baseline_cascade_counts.get(
                        "unresolved"
                    ),
                    "verified_removable_change_rate": cascade.get(
                        "verified_removable_change_rate",
                        cascade.get("avoidable_lower_rate"),
                    ),
                    "evaluator_relative_excess_change_rate_upper": cascade.get(
                        "evaluator_relative_excess_change_rate_upper",
                        cascade.get("avoidable_upper_rate"),
                    ),
                    "proof_coverage": cascade.get("proof_coverage"),
                    "unresolved_rate": cascade.get("unresolved_share"),
                }
                record_rows.append(row)
                group_rows.append(row)
            reported_eligible = int(report["summary"]["level3_evaluable_records"])
            observed_eligible = sum(row["eligible"] for row in group_rows)
            if observed_eligible != reported_eligible:
                raise ValueError(
                    f"eligible mismatch for {model}/{method}: "
                    f"{observed_eligible} != {reported_eligible}"
                )
            by_group[(model, method)] = group_rows

    if len(by_group) != 12 or len(record_rows) != 12 * expected_task_count:
        raise ValueError("matrix must contain 12 complete model-method groups")

    group_summaries = []
    group_summary_rows = []
    class_summaries = []
    class_summary_rows = []
    for (model, method), rows in sorted(by_group.items()):
        summary = _summarize(rows)
        dimensions = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
        }
        group_summaries.append({**dimensions, **summary})
        group_summary_rows.append(_flatten_summary(dimensions, summary))
        for query_class in QUERY_CLASS_LABELS:
            subset = [row for row in rows if row["query_class"] == query_class]
            class_summary = _summarize(subset)
            class_dimensions = {
                **dimensions,
                "query_class": query_class,
                "query_class_label": QUERY_CLASS_LABELS[query_class],
            }
            class_summaries.append({**class_dimensions, **class_summary})
            class_summary_rows.append(_flatten_summary(class_dimensions, class_summary))

    pooled_summary = _summarize(record_rows)
    pooled_by_method = {
        method: _summarize([row for row in record_rows if row["method"] == method])
        for method in METHOD_LABELS
    }
    pooled_by_model = {
        model: _summarize([row for row in record_rows if row["model"] == model])
        for model in MODEL_LABELS
    }
    pooled_by_query_class = {
        query_class: _summarize(
            [row for row in record_rows if row["query_class"] == query_class]
        )
        for query_class in QUERY_CLASS_LABELS
    }

    eligible_by_group = {
        key: {row["task_id"] for row in rows if row["eligible"]}
        for key, rows in by_group.items()
    }
    matched_method_cohorts: dict[str, Any] = {}
    matched_method_rows: list[dict[str, Any]] = []
    matched_method_class_rows: list[dict[str, Any]] = []
    for model in MODEL_LABELS:
        common = set.intersection(
            *(eligible_by_group[(model, method)] for method in METHOD_LABELS)
        )
        method_payload = {}
        method_class_payload: dict[str, Any] = {}
        for method in METHOD_LABELS:
            subset = [
                row for row in by_group[(model, method)] if row["task_id"] in common
            ]
            summary = _summarize(subset)
            outside_common = [
                row
                for row in by_group[(model, method)]
                if row["eligible"] and row["task_id"] not in common
            ]
            method_payload[method] = {
                **summary,
                "outside_all_methods_common_eligible": _summarize(outside_common),
            }
            matched_method_rows.append(
                _flatten_summary(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "matched_task_count": len(common),
                    },
                    summary,
                )
            )
            method_class_payload[method] = {}
            for query_class in QUERY_CLASS_LABELS:
                class_subset = [
                    row for row in subset if row["query_class"] == query_class
                ]
                class_summary = _summarize(class_subset)
                method_class_payload[method][query_class] = class_summary
                matched_method_class_rows.append(
                    _flatten_summary(
                        {
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "query_class": query_class,
                            "query_class_label": QUERY_CLASS_LABELS[query_class],
                            "matched_task_count": len(class_subset),
                        },
                        class_summary,
                    )
                )
        scope_signatures = {}
        for task_id in common:
            signature = tuple(
                next(
                    row["actual_scope"]
                    for row in by_group[(model, method)]
                    if row["task_id"] == task_id
                )
                for method in METHOD_LABELS
            )
            scope_signatures[task_id] = signature
        unanimous = sum(len(set(signature)) == 1 for signature in scope_signatures.values())
        paired_differences: dict[str, Any] = {}
        for left_method, right_method in (
            ("pter", "react"),
            ("pter", "reflexion"),
            ("reflexion", "react"),
        ):
            comparison = f"{left_method}_minus_{right_method}"
            paired_differences[comparison] = {}
            left_by_task = {
                row["task_id"]: row
                for row in by_group[(model, left_method)]
                if row["task_id"] in common
            }
            right_by_task = {
                row["task_id"]: row
                for row in by_group[(model, right_method)]
                if row["task_id"] in common
            }
            for field in (
                "content_retention_rate",
                "activity_change_ratio",
                "poi_seq_edit_distance",
            ):
                ordered_tasks = sorted(common)
                paired_differences[comparison][field] = _paired_bootstrap(
                    [float(left_by_task[task_id][field]) for task_id in ordered_tasks],
                    [float(right_by_task[task_id][field]) for task_id in ordered_tasks],
                    seed_text=f"{model}:{comparison}:{field}",
                )
        matched_method_cohorts[model] = {
            "task_count": len(common),
            "task_ids": sorted(common),
            "methods": method_payload,
            "methods_by_query_class": method_class_payload,
            "actual_scope_agreement": {
                "unanimous_task_count": unanimous,
                "unanimous_rate": unanimous / len(common) if common else None,
                "method_order": list(METHOD_LABELS),
                "signature_counts": {
                    "|".join(signature): count
                    for signature, count in sorted(Counter(scope_signatures.values()).items())
                },
            },
            "paired_differences_with_bootstrap_ci95": paired_differences,
        }

    matched_model_cohorts: dict[str, Any] = {}
    for method in METHOD_LABELS:
        common = set.intersection(
            *(eligible_by_group[(model, method)] for model in MODEL_LABELS)
        )
        matched_model_cohorts[method] = {
            "task_count": len(common),
            "task_ids": sorted(common),
            "models": {
                model: _summarize(
                    [
                        row
                        for row in by_group[(model, method)]
                        if row["task_id"] in common
                    ]
                )
                for model in MODEL_LABELS
            },
        }

    all_groups_common = set.intersection(*eligible_by_group.values())
    eligible_rows = [row for row in record_rows if row["eligible"]]
    outliers = sorted(
        eligible_rows,
        key=lambda row: (
            float(row["activity_change_ratio"]),
            float(row["poi_seq_edit_distance"]),
            int(row["compositional_count"]),
        ),
        reverse=True,
    )[:50]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "edit_scope_records.csv", record_rows)
    _write_csv(output_dir / "edit_scope_group_summary.csv", group_summary_rows)
    _write_csv(output_dir / "edit_scope_by_query_class.csv", class_summary_rows)
    _write_csv(output_dir / "edit_scope_matched_methods.csv", matched_method_rows)
    _write_csv(
        output_dir / "edit_scope_matched_methods_by_query_class.csv",
        matched_method_class_rows,
    )
    _write_csv(output_dir / "edit_scope_outliers_top50.csv", outliers)

    payload = {
        "schema_version": "natural-feasible-187-edit-scope-analysis-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": {
            "dataset_id": dataset_manifest["dataset_id"],
            "task_count": expected_task_count,
            "model_count": len(MODEL_LABELS),
            "method_count": len(METHOD_LABELS),
            "output_count": len(record_rows),
            "level3_definition": (
                "Actual Edit Scope among outputs passing both Level 1 and Level 2"
            ),
            "interpretation_boundary": (
                "Query class is a request-target taxonomy, not a gold/reference edit scope; "
                "the analysis is descriptive and diagnostic, not scope-compliance scoring."
            ),
            "transport_comparison_rule": (
                "Mode, endpoints, timing/duration, distance, cost/price/tickets/cars and leg "
                "topology are semantic. Numeric formatting, safe mode aliases, dictionary key "
                "order, and unknown metadata alone are treated as equivalent rewrites."
            ),
            "global_transport_reference_scope": (
                "Use an allowed set of alternative route-leg closures. Do not force all legs or "
                "all affected days as one unique gold scope."
            ),
            "attribution_algorithm": algorithm_manifest(),
        },
        "sources": {
            "dataset_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "matrix_manifest": str(matrix_manifest_path.relative_to(PROJECT_ROOT)),
            "reports": report_sources,
            "baseline_reports_dir": str(baseline_reports_dir.relative_to(PROJECT_ROOT)),
        },
        "pooled_all_outputs": pooled_summary,
        "pooled_by_method": pooled_by_method,
        "pooled_by_model": pooled_by_model,
        "pooled_by_query_class": pooled_by_query_class,
        "group_summaries": group_summaries,
        "query_class_summaries": class_summaries,
        "matched_eligible_across_methods_within_model": matched_method_cohorts,
        "matched_eligible_across_models_within_method": matched_model_cohorts,
        "eligible_in_all_12_groups": {
            "task_count": len(all_groups_common),
            "task_ids": sorted(all_groups_common),
        },
        "supporting_files": {
            "record_level_csv": "edit_scope_records.csv",
            "group_summary_csv": "edit_scope_group_summary.csv",
            "query_class_csv": "edit_scope_by_query_class.csv",
            "matched_methods_csv": "edit_scope_matched_methods.csv",
            "matched_methods_by_query_class_csv": (
                "edit_scope_matched_methods_by_query_class.csv"
            ),
            "outlier_csv": "edit_scope_outliers_top50.csv",
        },
    }
    (output_dir / "edit_scope_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "records": len(record_rows),
                "eligible_outputs": pooled_summary["eligible_records"],
                "matched_within_model_task_counts": {
                    model: item["task_count"]
                    for model, item in matched_method_cohorts.items()
                },
                "eligible_in_all_12_groups": len(all_groups_common),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
