"""Helpers for merging standalone framework outputs into eval-ready records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def discover_result_files(results_dir: str | Path) -> list[Path]:
    """Find standalone result files, skipping summary artifacts."""

    root = Path(results_dir)
    return sorted(
        path
        for path in root.glob("*.json")
        if path.name != "summary.json"
    )


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _build_sample_id_index(samples_root: Path) -> dict[str, Path]:
    """Map logical sample_id/case_id values to files for renamed batch layouts."""

    index: dict[str, Path] = {}
    for sample_file in sorted(samples_root.glob("*.json")):
        if sample_file.name in {"summary.json", "manifest.json", "regeneration_summary.json"}:
            continue
        try:
            sample_payload = load_json(sample_file)
        except Exception:
            continue
        for key in ("sample_id", "case_id", "episode_id"):
            value = sample_payload.get(key)
            if isinstance(value, str) and value and value not in index:
                index[value] = sample_file
    return index


def merge_sample_with_result(
    sample_payload: Dict[str, Any],
    result_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay runtime results onto the original sample record."""

    merged = dict(sample_payload)
    merged.update(
        {
            "framework": result_payload.get("framework"),
            "model": result_payload.get("model"),
            "exposure_mode": result_payload.get("exposure_mode"),
            "tool_profile": result_payload.get("tool_profile"),
            "db_read_enabled": result_payload.get("db_read_enabled"),
            "special_backend": result_payload.get("special_backend"),
            "success": result_payload.get("success"),
            "edited_plan": result_payload.get("edited_plan"),
            "ops": result_payload.get("ops", []),
            "trace": result_payload.get("trace", []),
            "metrics": result_payload.get("metrics", {}),
            "errors": result_payload.get("errors", []),
        }
    )
    return merged


def merge_results_directory(
    *,
    samples_dir: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Create eval-ready files by combining batch samples with result files."""

    samples_root = Path(samples_dir)
    results_root = Path(results_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    merged_count = 0
    missing_samples: list[str] = []
    result_files = discover_result_files(results_root)
    sample_id_index: dict[str, Path] | None = None

    for result_file in result_files:
        result_payload = load_json(result_file)
        case_id = (
            result_payload.get("case_id")
            or result_payload.get("sample_id")
            or result_file.stem
        )
        sample_file = samples_root / f"{case_id}.json"
        if not sample_file.exists():
            if sample_id_index is None:
                sample_id_index = _build_sample_id_index(samples_root)
            indexed_sample_file = sample_id_index.get(str(case_id))
            if indexed_sample_file is not None:
                sample_file = indexed_sample_file

        if not sample_file.exists():
            missing_samples.append(str(case_id))
            continue

        sample_payload = load_json(sample_file)
        merged_payload = merge_sample_with_result(sample_payload, result_payload)
        output_name = f"{case_id}.json"
        (output_root / output_name).write_text(
            json.dumps(merged_payload, ensure_ascii=False, indent=2)
        )
        merged_count += 1

    summary = {
        "samples_dir": str(samples_root),
        "results_dir": str(results_root),
        "output_dir": str(output_root),
        "merged_count": merged_count,
        "missing_samples": missing_samples,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary
