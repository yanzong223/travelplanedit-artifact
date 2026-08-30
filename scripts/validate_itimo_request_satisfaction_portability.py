#!/usr/bin/env python3
"""Lightweight portability check for request-satisfaction verification.

This script adapts iTIMO gold ADD, DELETE, and REPLACE annotations into the
structured edit representation used by the TravelPlanEdit-Bench verifier.  It
does not test ChinaTravel-specific plan validity and therefore must not be read
as full cross-dataset validation of the evaluator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.benchmark.models import LogicalConstraintObject
from evaluation.benchmark.verifiers import verify_constraints


def _activity(poi: list[Any]) -> dict[str, Any]:
    return {
        "position": str(poi[0]),
        "type": "attraction",
        "longitude": poi[2] if len(poi) > 2 else None,
        "latitude": poi[3] if len(poi) > 3 else None,
    }


def _plan(trajectory: list[list[Any]]) -> dict[str, Any]:
    return {
        "itinerary": [
            {
                "day": 1,
                "activities": [_activity(poi) for poi in trajectory],
            }
        ]
    }


def _apply_gold(
    trajectory: list[list[Any]], gold: dict[str, Any]
) -> tuple[str, dict[str, Any], list[list[Any]]]:
    updated = list(trajectory)
    if "insert_index" in gold:
        operation = "ADD"
        index = int(gold["insert_index"])
        selected_poi = gold["selected_poi"]
        updated.insert(max(0, min(index, len(updated))), selected_poi)
    elif "removed_index" in gold:
        operation = "DELETE"
        index = max(0, min(int(gold["removed_index"]), len(updated) - 1))
        selected_poi = updated[index]
        del updated[index]
    elif "replaced_index" in gold:
        operation = "REPLACE"
        index = max(0, min(int(gold["replaced_index"]), len(updated) - 1))
        selected_poi = gold["selected_poi"]
        updated[index] = selected_poi
    else:
        raise ValueError("unsupported iTIMO gold operation")
    return operation, {"index": index, "selected_poi": selected_poi}, updated


def _structured_requirements(
    operation: str,
    original: list[list[Any]],
    gold_details: dict[str, Any],
) -> tuple[list[LogicalConstraintObject], dict[str, Any]]:
    index = int(gold_details["index"])
    selected_name = str(gold_details["selected_poi"][0])
    requirements: list[LogicalConstraintObject] = []

    if operation in {"DELETE", "REPLACE"}:
        removed_name = str(original[index][0])
        requirements.append(
            LogicalConstraintObject(
                id="external_edit_forbidden_poi",
                source="edit",
                type="forbidden_attraction_name",
                target={"activity_type": "attraction"},
                operator="contains_none",
                value=[removed_name],
                provenance={"source_dataset": "iTIMO", "adapter": "entity_operation"},
            )
        )

    if operation in {"ADD", "REPLACE"}:
        requirements.append(
            LogicalConstraintObject(
                id="external_edit_required_poi",
                source="edit",
                type="required_attraction_name",
                target={"activity_type": "attraction"},
                operator="contains_all",
                value=[selected_name],
                provenance={"source_dataset": "iTIMO", "adapter": "entity_operation"},
            )
        )

    expected = list(original)
    if operation == "ADD":
        expected.insert(max(0, min(index, len(expected))), gold_details["selected_poi"])
    elif operation == "DELETE":
        del expected[max(0, min(index, len(expected) - 1))]
    else:
        expected[max(0, min(index, len(expected) - 1))] = gold_details["selected_poi"]

    sequence_requirement = {
        "type": "attraction_sequence",
        "operator": "equals",
        "value": [str(poi[0]) for poi in expected],
        "provenance": {"source_dataset": "iTIMO", "adapter": "sequence_operation"},
    }
    return requirements, sequence_requirement


def _verify_sequence_requirement(
    plan: dict[str, Any], requirement: dict[str, Any]
) -> bool:
    """Dataset adapter for the position-sensitive semantics of iTIMO labels."""
    actual = [
        str(activity.get("position", ""))
        for day in plan.get("itinerary", [])
        for activity in day.get("activities", [])
        if activity.get("type") == "attraction"
    ]
    return actual == requirement["value"]


def evaluate_file(path: Path) -> Counter[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for record in records.values():
        example_input = record["example_input"]
        gold = record["example_output"]
        original = example_input["need_to_modify itinerary"]
        operation, details, updated = _apply_gold(original, gold)
        requirements, sequence_requirement = _structured_requirements(operation, original, details)

        gold_results = verify_constraints(_plan(updated), requirements)
        identity_results = verify_constraints(_plan(original), requirements)
        sequence_gold_pass = _verify_sequence_requirement(_plan(updated), sequence_requirement)
        sequence_identity_pass = _verify_sequence_requirement(_plan(original), sequence_requirement)

        counts["samples"] += 1
        counts[f"operation_{operation}"] += 1
        counts["shared_requirements"] += len(requirements)
        counts["supported_shared_requirements"] += sum(result.supported for result in gold_results)
        counts["shared_gold_pass"] += int(bool(gold_results) and all(result.passed for result in gold_results))
        counts["shared_identity_rejected"] += int(
            bool(identity_results) and not all(result.passed for result in identity_results)
        )
        counts["adapter_requirements"] += 1
        counts["adapter_gold_pass"] += int(sequence_gold_pass)
        counts["adapter_identity_rejected"] += int(not sequence_identity_pass)
        counts["combined_gold_pass"] += int(
            sequence_gold_pass and bool(gold_results) and all(result.passed for result in gold_results)
        )
        counts["combined_identity_rejected"] += int(
            (not sequence_identity_pass)
            and bool(identity_results)
            and not all(result.passed for result in identity_results)
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to iTIMO's Benchmark/iTIMO_dataset directory.",
    )
    args = parser.parse_args()

    files = sorted(args.dataset_root.glob("*/*_test.json"))
    if not files:
        raise FileNotFoundError(f"No iTIMO test files found under {args.dataset_root}")

    total: Counter[str] = Counter()
    per_file: dict[str, dict[str, int]] = {}
    for path in files:
        counts = evaluate_file(path)
        total.update(counts)
        per_file[str(path.relative_to(args.dataset_root))] = dict(sorted(counts.items()))

    result = {
        "scope": "request_satisfaction_only",
        "source_dataset": "iTIMO",
        "files": len(files),
        "per_file": per_file,
        "total": dict(sorted(total.items())),
        "passes": {
            "all_shared_requirements_supported": (
                total["supported_shared_requirements"] == total["shared_requirements"]
            ),
            "all_adapter_requirements_pass_on_gold": total["adapter_gold_pass"] == total["samples"],
            "all_adapter_requirements_reject_identity": (
                total["adapter_identity_rejected"] == total["samples"]
            ),
        },
        "limitations": [
            "The adapter covers ADD, DELETE, and REPLACE annotations only.",
            "Position-sensitive operation semantics require an iTIMO sequence adapter.",
            "It does not port ChinaTravel-specific plan-validity checks.",
            "It does not reproduce iTIMO's popularity, category, or spatial hint metrics.",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result["passes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
