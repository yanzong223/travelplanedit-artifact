#!/usr/bin/env python3
"""Recompute one complete matrix into a separate report directory without mutation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluation.benchmark import (  # noqa: E402
    BenchmarkEvaluator,
    aggregate_results,
    attach_report_views,
)
from evaluation.edit_scope import (  # noqa: E402
    FullGateResult,
    algorithm_manifest,
    analyze_edit_scope,
    canonical_json_sha256,
)
from evaluation.route_evidence import RouteEvidenceCache  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    """Write deterministic, line-addressable proof records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        for value in values
    )
    path.write_text(payload, encoding="utf-8")


def build_edit_scope_proof_record(
    *,
    task_id: str,
    method: str,
    model: str,
    evaluation_path: Path,
    task_path: Path,
    cascade: dict[str, Any],
) -> dict[str, Any]:
    """Package the full DCCA result needed for post-hoc proof auditing."""

    return {
        "task_id": task_id,
        "method": method,
        "model": model,
        "evaluation_path": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "task_path": str(task_path.relative_to(PROJECT_ROOT)),
        "cascade_result_sha256": canonical_json_sha256(cascade),
        "cascade": cascade,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative_or_absolute(path: Path) -> str:
    """Keep repository artifacts portable while allowing temporary QA outputs."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


RUNTIME_RESULT_FIELDS = (
    "db_read_enabled",
    "edited_plan",
    "enabled_semantic_tools",
    "errors",
    "exposure_mode",
    "framework",
    "metrics",
    "model",
    "ops",
    "semantic_tool_allowlist",
    "special_backend",
    "success",
    "tool_profile",
    "trace",
)


def merge_authoritative_task_payload(
    task_payload: dict[str, Any], evaluation_payload: dict[str, Any]
) -> dict[str, Any]:
    """Combine frozen task truth with the immutable model-run result.

    Some early evaluation artifacts contain a stale or incomplete constraint IR
    even though the frozen 187-task release contains the repaired canonical IR.
    The task file is authoritative for inputs and evaluation labels; only
    runtime/output fields are copied from the evaluation artifact.  This keeps
    model outputs unchanged while ensuring every annotated target is evaluable.
    """

    merged = copy.deepcopy(task_payload)
    for field in RUNTIME_RESULT_FIELDS:
        if field in evaluation_payload:
            merged[field] = copy.deepcopy(evaluation_payload[field])
    return merged


def full_gate_validator(evaluation_record: dict[str, Any]):
    evaluator = BenchmarkEvaluator(level="all")

    def validate(plan: dict[str, Any]) -> FullGateResult:
        candidate = copy.deepcopy(evaluation_record)
        candidate["edited_plan"] = plan
        result = evaluator.evaluate(candidate)
        level1 = result.get("level1") if isinstance(result.get("level1"), dict) else {}
        level2 = result.get("level2") if isinstance(result.get("level2"), dict) else {}
        components = {
            "level1": bool(level1.get("pass")),
            "level1.feasibility": bool((level1.get("feasibility") or {}).get("pass")),
            "level1.origin_logical_preservation": bool(
                (level1.get("origin_logical_preservation") or {}).get("pass")
            ),
            "level1.origin_preference_preservation": bool(
                (level1.get("origin_preference_preservation") or {}).get("pass")
            ),
            "level2": bool(level2.get("pass")),
            "level2.edit_logical_success": bool(
                (level2.get("edit_logical_success") or {}).get("pass")
            ),
        }
        return FullGateResult(
            passed=components["level1"] and components["level2"],
            components=components,
            reason=str((result.get("level3") or {}).get("reason") or "unknown"),
        )

    return validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compute-cascade", action="store_true")
    parser.add_argument("--route-evidence-cache", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_json(manifest_path)
    dataset_root = manifest_path.parents[2]
    dataset_manifest_path = PROJECT_ROOT / str(manifest["dataset_manifest"])
    dataset_manifest = load_json(dataset_manifest_path)
    task_entries = {
        str(entry["task_id"]): entry for entry in dataset_manifest.get("tasks", [])
    }
    task_count = int(manifest["task_count"])
    if len(task_entries) != task_count:
        raise RuntimeError(
            f"Dataset manifest has {len(task_entries)} task entries, expected {task_count}"
        )
    route_cache = (
        RouteEvidenceCache.load(args.route_evidence_cache)
        if args.route_evidence_cache
        else None
    )
    outputs = []
    for method, method_block in manifest["methods"].items():
        for model, model_block in method_block["models"].items():
            print(
                json.dumps({"status": "group_started", "method": method, "model": model}),
                flush=True,
            )
            records = list(model_block["records"])
            if len(records) != task_count:
                raise RuntimeError(f"Incomplete group: {method}/{model}")
            evaluator = BenchmarkEvaluator(level="all")
            results = []
            proof_records: list[dict[str, Any]] = []
            for record in records:
                evaluation_path = dataset_root / str(record["evaluation_path"])
                if sha256_file(evaluation_path) != str(record["evaluation_sha256"]):
                    raise RuntimeError(f"Checksum mismatch: {evaluation_path}")
                evaluation_payload = load_json(evaluation_path)
                task_id = str(record["task_id"])
                task_entry = task_entries.get(task_id)
                if task_entry is None:
                    raise RuntimeError(f"Task missing from dataset manifest: {task_id}")
                task_path = dataset_root / str(task_entry["path"])
                if sha256_file(task_path) != str(task_entry["sha256"]):
                    raise RuntimeError(f"Task checksum mismatch: {task_path}")
                task_payload = load_json(task_path)
                payload = merge_authoritative_task_payload(
                    task_payload, evaluation_payload
                )
                origin_query = payload.get("origin_query") or payload.get(
                    "origin_query_structured"
                )
                result = evaluator.evaluate(payload, origin_query=origin_query)
                if args.compute_cascade and bool((result.get("level3") or {}).get("eligible")):
                    cascade = analyze_edit_scope(
                        payload.get("origin_plan") or {},
                        payload.get("edited_plan") or {},
                        payload.get("edit_target_constraints") or [],
                        route_evidence_cache=route_cache,
                        full_gate_validator=full_gate_validator(payload),
                    )
                    proof_records.append(
                        build_edit_scope_proof_record(
                            task_id=task_id,
                            method=method,
                            model=model,
                            evaluation_path=evaluation_path,
                            task_path=task_path,
                            cascade=cascade,
                        )
                    )
                    result = evaluator.evaluate(
                        payload,
                        origin_query=origin_query,
                        cascade_result=cascade,
                    )
                results.append(result)
            report = attach_report_views(aggregate_results(results))
            proof_output = (
                output_dir
                / f"{model}_{method}_all{task_count}_edit_scope_proofs.jsonl"
            )
            if args.compute_cascade:
                write_jsonl(proof_output, proof_records)
            report["offline_recompute_provenance"] = {
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
                "manifest_sha256": sha256_file(manifest_path),
                "method": method,
                "model": model,
                "record_count": len(records),
                "ground_truth_merge": {
                    "policy": "frozen_task_truth_plus_immutable_runtime_output_v1",
                    "dataset_manifest": str(
                        dataset_manifest_path.relative_to(PROJECT_ROOT)
                    ),
                    "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                    "task_payload_count": len(task_entries),
                    "runtime_fields": list(RUNTIME_RESULT_FIELDS),
                },
                "cascade_computed": bool(args.compute_cascade),
                "edit_scope_algorithm": (
                    algorithm_manifest() if args.compute_cascade else None
                ),
                "route_evidence_cache": (
                    str(args.route_evidence_cache.resolve().relative_to(PROJECT_ROOT))
                    if args.route_evidence_cache
                    else None
                ),
                "edit_scope_proof_artifact": (
                    {
                        "path": project_relative_or_absolute(proof_output),
                        "sha256": sha256_file(proof_output),
                        "record_count": len(proof_records),
                        "format": "jsonl",
                        "content": "full_validated_dcca_results_with_saved_witnesses",
                    }
                    if args.compute_cascade
                    else None
                ),
            }
            output = output_dir / f"{model}_{method}_all{task_count}_benchmark_report.json"
            write_json(output, report)
            outputs.append(str(output))
            print(
                json.dumps({
                    "status": "group_completed",
                    "method": method,
                    "model": model,
                    "eligible": report.get("summary", {}).get("level3_evaluable_records"),
                }),
                flush=True,
            )
    print(json.dumps({"reports": len(outputs), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
