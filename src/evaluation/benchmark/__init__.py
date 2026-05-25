"""Benchmark evaluation API for ChinaTravel-format travel plan editing."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "BenchmarkEvaluator": (".levels", "BenchmarkEvaluator"),
    "build_edit_target_preferences": (".soft_truth", "build_edit_target_preferences"),
    "build_edit_target_preference_tags": (".soft_truth", "build_edit_target_preference_tags"),
    "build_edit_logical_constraints": (".hard_truth", "build_edit_logical_constraints"),
    "build_edit_logical_constraints_from_structured_inputs": (
        ".hard_truth",
        "build_edit_logical_constraints_from_structured_inputs",
    ),
    "build_origin_logical_constraints": (".hard_truth", "build_origin_logical_constraints"),
    "build_effective_logical_constraints": (".constraint_merge", "build_effective_logical_constraints"),
    "build_origin_preference_constraints": (".soft_truth", "build_origin_preference_constraints"),
    "build_origin_preference_tags": (".soft_truth", "build_origin_preference_tags"),
    "compute_origin_preference_score_snapshot": (".soft_truth", "compute_origin_preference_score_snapshot"),
    "InferredAtomicOp": (".models", "InferredAtomicOp"),
    "InferredEditSequence": (".models", "InferredEditSequence"),
    "Level3EvaluationResult": (".models", "Level3EvaluationResult"),
    "LogicalConstraintObject": (".models", "LogicalConstraintObject"),
    "MatchedActivityPair": (".models", "MatchedActivityPair"),
    "PreferenceConstraintObject": (".models", "PreferenceConstraintObject"),
    "PreferenceScoreSnapshot": (".models", "PreferenceScoreSnapshot"),
    "PreferenceScoreResult": (".models", "PreferenceScoreResult"),
    "PreferenceTag": (".models", "PreferenceTag"),
    "SOFT_FACET_REGISTRY": (".soft_truth", "SOFT_FACET_REGISTRY"),
    "TAG_TO_SCORER": (".soft_truth", "TAG_TO_SCORER"),
    "VerificationResult": (".models", "VerificationResult"),
    "adapt_edit_record": (".edit_adapters", "adapt_edit_record"),
    "adapt_legacy_edit_record": (".edit_adapters", "adapt_legacy_edit_record"),
    "adapt_origin_query": (".origin_adapters", "adapt_origin_query"),
    "aggregate_results": (".reporting", "aggregate_results"),
    "attach_report_views": (".reporting", "attach_report_views"),
    "evaluate_level3": (".level3", "evaluate_level3"),
    "evaluate_level1": (".levels", "evaluate_level1"),
    "evaluate_level2": (".levels", "evaluate_level2"),
    "evaluate_record": (".levels", "evaluate_record"),
    "infer_edit_sequence": (".diffing", "infer_edit_sequence"),
    "serialize_logical_constraints": (".hard_truth", "serialize_logical_constraints"),
    "serialize_preference_constraints": (".soft_truth", "serialize_preference_constraints"),
    "serialize_preference_snapshots": (".soft_truth", "serialize_preference_snapshots"),
    "serialize_preference_tags": (".soft_truth", "serialize_preference_tags"),
    "validate_preference_tags": (".soft_truth", "validate_preference_tags"),
    "validate_logical_constraint_schema": (".hard_truth", "validate_logical_constraint_schema"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
