#!/usr/bin/env python3
"""Precompute route evidence for unresolved rollbacks in a matrix cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from edit_framework.tools.chinatravel_tools import ChinaTravelToolAdapter  # noqa: E402
from evaluation.edit_scope import (  # noqa: E402
    ALGORITHM_VERSION,
    analyze_edit_scope,
)
from evaluation.route_evidence import (  # noqa: E402
    MODES,
    RouteEvidenceCache,
    cache_payload,
    route_key,
)
from recompute_matrix_reports_offline import (  # noqa: E402
    full_gate_validator,
    merge_authoritative_task_payload,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry_is_complete(entry: dict[str, Any] | None) -> bool:
    modes = entry.get("modes") if isinstance(entry, dict) else None
    return isinstance(modes, dict) and all(
        isinstance(modes.get(mode), dict)
        and modes[mode].get("status") in {"ok", "ok_no_route"}
        for mode in MODES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--seed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    reports_dir = args.reports_dir.resolve()
    seed_path = args.seed_cache.resolve()
    output_path = args.output.resolve()
    manifest = _load(manifest_path)
    dataset_root = manifest_path.parents[2]
    dataset_manifest_path = PROJECT_ROOT / str(manifest["dataset_manifest"])
    dataset_manifest = _load(dataset_manifest_path)
    task_entries = {
        str(entry["task_id"]): entry for entry in dataset_manifest.get("tasks", [])
    }
    seed_cache = RouteEvidenceCache.load(seed_path)

    requests: dict[str, dict[str, str]] = {}
    eligible_outputs = 0
    unresolved_units = 0
    for method, method_block in manifest["methods"].items():
        for model, model_block in method_block["models"].items():
            report = _load(
                reports_dir / f"{model}_{method}_all{manifest['task_count']}_benchmark_report.json"
            )
            for record, scored in zip(model_block["records"], report["results"]):
                if not bool((scored.get("level3") or {}).get("eligible")):
                    continue
                eligible_outputs += 1
                evaluation_payload = _load(dataset_root / record["evaluation_path"])
                task_entry = task_entries[str(record["task_id"])]
                task_payload = _load(dataset_root / str(task_entry["path"]))
                payload = merge_authoritative_task_payload(
                    task_payload, evaluation_payload
                )
                cascade = analyze_edit_scope(
                    payload.get("origin_plan") or {},
                    payload.get("edited_plan") or {},
                    payload.get("edit_target_constraints") or [],
                    route_evidence_cache=seed_cache,
                    full_gate_validator=full_gate_validator(payload),
                )
                for unit in cascade["unresolved_units"]:
                    unresolved_units += 1
                    for request in unit.get("route_requests", []):
                        if not isinstance(request, dict):
                            continue
                        normalized = {
                            key: str(request.get(key, "") or "")
                            for key in ("city", "start", "end", "start_time")
                        }
                        requests[route_key(**normalized)] = normalized

    entries = dict(seed_cache.payload["entries"])
    to_query = [
        request
        for key, request in sorted(requests.items())
        if not _entry_is_complete(entries.get(key))
    ]

    from chinatravel.environment.world_env import WorldEnv

    runtime = SimpleNamespace(world_env=WorldEnv())
    adapter = ChinaTravelToolAdapter(
        tool_profile="db_read_typed",
        exposure_mode="primitive_only",
    )
    status_counts = {"ok": 0, "ok_no_route": 0, "query_error": 0}
    for index, request in enumerate(to_query, start=1):
        modes = {}
        for mode in MODES:
            result = adapter._execute_typed_route(
                runtime,
                {**request, "transport_type": mode},
            )
            rows = result.get("rows", []) if result.get("ok") else []
            if not result.get("ok"):
                status = "query_error"
            elif rows:
                status = "ok"
            else:
                status = "ok_no_route"
            status_counts[status] += 1
            modes[mode] = {
                "status": status,
                "rows": rows,
                "error_code": result.get("error_code"),
                "message": result.get("message"),
            }
        entry = {**request, "request": request, "modes": modes}
        entries[route_key(**request)] = entry
        if index % 25 == 0 or index == len(to_query):
            print(json.dumps({
                "status": "route_progress",
                "queried": index,
                "total": len(to_query),
            }), flush=True)

    tool_fingerprint = {
        "schema": "matrix-cascade-route-precompute-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "ground_truth_merge_policy": (
            "frozen_task_truth_plus_immutable_runtime_output_v1"
        ),
        "dataset_manifest": str(dataset_manifest_path.relative_to(PROJECT_ROOT)),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "seed_cache": str(seed_path.relative_to(PROJECT_ROOT)),
        "seed_cache_sha256": _sha256(seed_path),
        "edit_scope_algorithm_version": ALGORITHM_VERSION,
        "edit_scope_api_sha256": _sha256(SRC_ROOT / "evaluation/edit_scope.py"),
        "cascade_code_sha256": _sha256(SRC_ROOT / "evaluation/cascade_analysis.py"),
        "route_tool_code_sha256": _sha256(
            SRC_ROOT / "edit_framework/tools/chinatravel_tools.py"
        ),
    }
    payload = cache_payload(list(entries.values()), tool_fingerprint=tool_fingerprint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "eligible_outputs": eligible_outputs,
        "unresolved_units_scanned": unresolved_units,
        "unique_route_requests": len(requests),
        "new_or_incomplete_requests_queried": len(to_query),
        "mode_status_counts": status_counts,
        "total_cache_entries": len(entries),
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
