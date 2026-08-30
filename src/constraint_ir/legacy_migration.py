"""Legacy sample migration helpers for canonical constraint IR."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .compiler import build_canonical_constraint_ir_from_legacy_sample
from .validator import check_ir_compiles, validate_canonical_constraint_ir


def migrate_sample_to_canonical_ir(sample: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, List[str]]:
    trace = sample.get("query_generation_trace") if isinstance(sample.get("query_generation_trace"), dict) else {}
    if not str(trace.get("constraint_type", "") or "").strip():
        return None, ["missing_constraint_type"]

    try:
        canonical_ir = build_canonical_constraint_ir_from_legacy_sample(sample)
    except Exception as exc:
        return None, [str(exc)]
    is_valid, errors = validate_canonical_constraint_ir(canonical_ir)
    if not is_valid:
        return None, errors
    return canonical_ir, []


def build_migration_report(sample: Dict[str, Any], canonical_ir: Dict[str, Any] | None, errors: List[str]) -> Dict[str, Any]:
    compile_ok, compile_errors, compile_details = check_ir_compiles(canonical_ir)
    status = "migrated" if canonical_ir is not None and not errors else "failed_incomplete_semantics"
    if canonical_ir is not None and not errors and not compile_ok:
        status = "failed_unsupported_verifier"
    return {
        "sample_id": sample.get("sample_id"),
        "status": status,
        "errors": errors + compile_errors,
        "unsupported_constraint_types": compile_details.get("unsupported_constraint_types", []),
    }
