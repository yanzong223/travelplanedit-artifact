#!/usr/bin/env python3
"""Step 3: Freeze edit-side hard/soft truth from structured step-2 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_GEN_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = Path(DATA_GEN_DIR).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constraint_ir.adapters import build_edit_targets_from_ir
from evaluation.benchmark.constraint_merge import build_effective_logical_constraints


def load_samples(batch_dir: str) -> List[Dict[str, Any]]:
    batch_path = Path(batch_dir)
    samples: List[Dict[str, Any]] = []
    for sample_file in sorted(batch_path.glob("sample_*.json")):
        try:
            with open(sample_file, "r", encoding="utf-8") as handle:
                samples.append(json.load(handle))
        except Exception as exc:
            print(f"Warning: Failed to load {sample_file}: {exc}")
    print(f"Loaded {len(samples)} samples from {batch_dir}")
    return samples


def save_sample(sample: Dict[str, Any], output_dir: str) -> None:
    sample_file = Path(output_dir) / f"{sample['sample_id']}.json"
    with open(sample_file, "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=2)


def _sync_existing_pass_views(sample: Dict[str, Any], batch_dir: str) -> List[str]:
    batch_path = Path(batch_dir)
    source_truth = {
        "canonical_constraint_ir": sample.get("canonical_constraint_ir", {})
        if isinstance(sample.get("canonical_constraint_ir"), dict)
        else {},
        "edit_target_constraints": sample.get("edit_target_constraints", [])
        if isinstance(sample.get("edit_target_constraints"), list)
        else [],
        "effective_logical_constraints": sample.get("effective_logical_constraints", [])
        if isinstance(sample.get("effective_logical_constraints"), list)
        else [],
        "effective_constraint_merge_trace": sample.get("effective_constraint_merge_trace", {})
        if isinstance(sample.get("effective_constraint_merge_trace"), dict)
        else {},
        "edit_target_preference_tags": sample.get("edit_target_preference_tags", [])
        if isinstance(sample.get("edit_target_preference_tags"), list)
        else [],
        "edit_target_preferences": sample.get("edit_target_preferences", [])
        if isinstance(sample.get("edit_target_preferences"), list)
        else [],
    }
    mirror_paths = [
        batch_path / "_validation_split" / "passed" / f"{sample['sample_id']}.json",
        batch_path / "_curated" / "passed" / f"{sample['sample_id']}.json",
    ]
    updated: List[str] = []
    for mirror_path in mirror_paths:
        if not mirror_path.exists():
            continue
        try:
            mirror_sample = json.loads(mirror_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mirror_truth = {
            "canonical_constraint_ir": mirror_sample.get("canonical_constraint_ir", {})
            if isinstance(mirror_sample.get("canonical_constraint_ir"), dict)
            else {},
            "edit_target_constraints": mirror_sample.get("edit_target_constraints", [])
            if isinstance(mirror_sample.get("edit_target_constraints"), list)
            else [],
            "edit_target_preference_tags": mirror_sample.get("edit_target_preference_tags", [])
            if isinstance(mirror_sample.get("edit_target_preference_tags"), list)
            else [],
            "edit_target_preferences": mirror_sample.get("edit_target_preferences", [])
            if isinstance(mirror_sample.get("edit_target_preferences"), list)
            else [],
        }
        if mirror_truth == source_truth:
            continue
        mirror_sample["canonical_constraint_ir"] = source_truth["canonical_constraint_ir"]
        mirror_sample["edit_target_constraints"] = source_truth["edit_target_constraints"]
        mirror_sample["effective_logical_constraints"] = source_truth["effective_logical_constraints"]
        mirror_sample["effective_constraint_merge_trace"] = source_truth["effective_constraint_merge_trace"]
        mirror_sample["edit_target_preference_tags"] = source_truth["edit_target_preference_tags"]
        mirror_sample["edit_target_preferences"] = source_truth["edit_target_preferences"]
        mirror_path.write_text(json.dumps(mirror_sample, ensure_ascii=False, indent=2), encoding="utf-8")
        updated.append(str(mirror_path))
    return updated


def filter_pending_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    for sample in samples:
        status = sample.get("meta", {}).get("status", "")
        if status != "02_completed":
            continue
        if not isinstance(sample.get("canonical_edit_query"), str) or not sample["canonical_edit_query"].strip():
            continue
        pending.append(sample)
    print(f"Found {len(pending)} samples needing frozen truth generation (from {len(samples)} total)")
    return pending


def build_frozen_truth_for_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    canonical_constraint_ir = (
        sample.get("canonical_constraint_ir")
        if isinstance(sample.get("canonical_constraint_ir"), dict)
        else None
    )
    if not canonical_constraint_ir:
        raise ValueError("missing_canonical_constraint_ir")
    return build_edit_targets_from_ir(canonical_constraint_ir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze edit-side truth from structured step-2 artifacts")
    parser.add_argument("--input", required=True, help="Batch directory from step 2")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: Freeze Edit Truth")
    print("=" * 60)

    samples = load_samples(args.input)
    pending = filter_pending_samples(samples)
    if not pending:
        print("No samples need frozen truth generation. Exiting.")
        return

    success_count = 0
    fail_count = 0
    synced_view_count = 0
    for index, sample in enumerate(pending, start=1):
        print(f"\n  [{index}/{len(pending)}] Processing {sample['sample_id']}...")
        try:
            frozen = build_frozen_truth_for_sample(sample)
            sample["edit_target_constraints"] = frozen["edit_target_constraints"]
            effective, merge_trace = build_effective_logical_constraints(
                sample.get("origin_logical_constraints", []),
                sample["edit_target_constraints"],
            )
            sample["effective_logical_constraints"] = effective
            sample["effective_constraint_merge_trace"] = merge_trace
            sample["edit_target_preference_tags"] = frozen["edit_target_preference_tags"]
            sample["edit_target_preferences"] = frozen["edit_target_preferences"]
            trace = sample.get("query_generation_trace", {}) if isinstance(sample.get("query_generation_trace"), dict) else {}
            trace["hard_truth_source"] = "canonical_constraint_ir"
            trace["soft_truth_source"] = "canonical_constraint_ir"
            trace["hard_truth_compile_mode"] = "partial" if any(
                item.get("type") in {"nearby_meal_requirement", "anchor_bundle_budget_limit"}
                for item in sample["edit_target_constraints"]
            ) else "full"
            sample["query_generation_trace"] = trace
            sample["meta"]["status"] = "03_completed"
            sample["meta"]["last_step"] = "03_freeze_edit_truth"
            sample["meta"]["failure_reason"] = ""
            sample["meta"]["failure_reason_code"] = ""
            save_sample(sample, args.input)
            synced_view_count += len(_sync_existing_pass_views(sample, args.input))
            success_count += 1
            print("    ✓ frozen edit truth generated")
        except Exception as exc:
            sample.pop("edit_target_constraints", None)
            sample.pop("effective_logical_constraints", None)
            sample.pop("effective_constraint_merge_trace", None)
            sample.pop("edit_target_preference_tags", None)
            sample.pop("edit_target_preferences", None)
            sample["meta"]["status"] = "03_failed"
            sample["meta"]["last_step"] = "03_freeze_edit_truth"
            sample["meta"]["failure_reason"] = str(exc)
            sample["meta"]["failure_reason_code"] = "freeze_edit_truth_failed"
            save_sample(sample, args.input)
            fail_count += 1
            print(f"    ✗ {exc}")

    print("\n" + "=" * 60)
    print("✓ Step 3 Complete!")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Synced passed views: {synced_view_count}")


if __name__ == "__main__":
    main()
