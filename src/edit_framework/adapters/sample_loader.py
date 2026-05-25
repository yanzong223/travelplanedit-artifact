"""Batch-sample loaders for standalone edit frameworks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from edit_framework.base import EditInput


def load_edit_input_from_sample(
    sample_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> EditInput:
    """Load an edit sample using only origin_query_text."""

    sample_file = Path(sample_path)
    sample = json.loads(sample_file.read_text())
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("sample_path", str(sample_file))
    merged_metadata.setdefault("sample_id", sample.get("sample_id", sample_file.stem))
    for key in (
        "origin_query_structured",
        "origin_logical_constraints",
        "origin_preference_constraints",
        "edit_target_constraints",
        "effective_logical_constraints",
        "effective_constraint_merge_trace",
        "edit_target_preferences",
        "origin_preference_tags",
        "edit_target_preference_tags",
        "retrieval_hint_context",
        "retrieval_reference_result",
        "database_prompt_context",
        "database_reference",
        "reference_information",
        "retrieved_database_context",
        "database_rows",
        "batch_id",
        "conflict_set",
        "primary_conflict",
        "secondary_conflicts",
        "query_generation_trace",
    ):
        if key in sample:
            merged_metadata.setdefault(key, sample.get(key))

    return EditInput(
        case_id=sample.get("sample_id", sample_file.stem),
        origin_plan=sample["origin_plan"],
        origin_query_text=sample.get("origin_query_text", ""),
        edit_query=sample["edit_query"],
        metadata=merged_metadata,
    )


def collect_sample_paths(
    *,
    sample_path: str | Path | None = None,
    batch_dir: str | Path | None = None,
    limit: Optional[int] = None,
) -> List[Path]:
    """Resolve one or many batch samples in a deterministic order."""

    if sample_path and batch_dir:
        raise ValueError("Provide either sample_path or batch_dir, not both")
    if not sample_path and not batch_dir:
        raise ValueError("Either sample_path or batch_dir is required")

    if sample_path:
        return [Path(sample_path)]

    root = Path(batch_dir)
    sample_files = sorted(root.glob("sample_*.json"))
    if not sample_files:
        sample_files = [
            path
            for path in sorted(root.glob("*.json"))
            if path.name not in {"summary.json", "manifest.json", "regeneration_summary.json"}
        ]
    if limit is not None:
        sample_files = sample_files[:limit]
    return sample_files
