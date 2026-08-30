"""Reference-scope scoring with multiple admissible dependency closures.

This module is intentionally separate from Level 3.  Level 3 describes the
observed edit.  Precision/recall require human/adjudicated reference closures
and must not be fabricated from the request class.
"""

from __future__ import annotations

from typing import Any, Iterable

from .benchmark.models import InferredEditSequence


def inferred_changed_units(sequence: InferredEditSequence) -> set[str]:
    """Convert inferred atomic operations into stable field-level unit ids."""
    units: set[str] = set()
    for op in sequence.atomic_ops:
        if op.op_type == "change_time":
            units.update(f"{ref}.time" for ref in op.origin_refs)
        elif op.op_type == "change_transport":
            units.update(f"{ref}.transport" for ref in op.origin_refs)
        elif op.op_type == "change_attribute":
            fields = [str(field) for field in op.details.get("fields", [])]
            for ref in op.origin_refs:
                units.update(f"{ref}.{field}" for field in fields)
        elif op.op_type == "insert":
            units.update(f"{ref}.activity" for ref in op.edited_refs)
        else:
            units.update(f"{ref}.activity" for ref in op.origin_refs)
    return units


def _rate(numerator: int, denominator: int, *, empty_value: float) -> float:
    return numerator / denominator if denominator else empty_value


def score_scope_against_allowed_closures(
    actual_units: Iterable[str], annotation: dict[str, Any]
) -> dict[str, Any]:
    """Score one output against the best of several admissible closures.

    Each closure declares ``required_units`` and ``allowed_units``. Precision
    asks whether observed changes stay inside the closure. Recall asks whether
    the closure's required units were covered. Multiple closures represent
    alternative valid route/dependency solutions.
    """
    actual = {str(unit) for unit in actual_units}
    closures = annotation.get("allowed_closures")
    if not isinstance(closures, list) or not closures:
        raise ValueError("scope annotation requires at least one allowed closure")

    scores: list[dict[str, Any]] = []
    for index, closure in enumerate(closures):
        if not isinstance(closure, dict):
            raise ValueError("each allowed closure must be an object")
        allowed = {str(unit) for unit in closure.get("allowed_units", [])}
        required = {str(unit) for unit in closure.get("required_units", [])}
        if not required <= allowed:
            raise ValueError("required_units must be a subset of allowed_units")
        inside = actual & allowed
        covered = actual & required
        precision = _rate(len(inside), len(actual), empty_value=1.0 if not required else 0.0)
        recall = _rate(len(covered), len(required), empty_value=1.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(
            {
                "closure_id": str(closure.get("closure_id") or f"closure_{index}"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "outside_allowed_units": sorted(actual - allowed),
                "missing_required_units": sorted(required - actual),
            }
        )

    best = max(scores, key=lambda item: (item["f1"], item["precision"], item["recall"]))
    return {
        "annotation_id": annotation.get("annotation_id"),
        "actual_unit_count": len(actual),
        "closure_count": len(scores),
        "best_closure": best,
        "multi_solution_consistent": any(
            item["precision"] == 1.0 and item["recall"] == 1.0 for item in scores
        ),
        "closure_scores": scores,
    }
