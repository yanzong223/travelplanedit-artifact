"""Benchmark evaluation exports for ChinaTravel-format plan editing."""

from __future__ import annotations

from importlib import import_module


__all__ = [
    "BenchmarkEvaluator",
    "InferredAtomicOp",
    "InferredEditSequence",
    "Level3EvaluationResult",
    "LogicalConstraintObject",
    "MatchedActivityPair",
    "PreferenceConstraintObject",
    "PreferenceScoreResult",
    "VerificationResult",
    "adapt_edit_record",
    "adapt_origin_query",
    "aggregate_results",
    "evaluate_level3",
    "evaluate_level1",
    "evaluate_level2",
    "evaluate_record",
    "infer_edit_sequence",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    benchmark = import_module(".benchmark", __name__)
    return getattr(benchmark, name)
