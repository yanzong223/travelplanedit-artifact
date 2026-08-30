"""Batch migration runner for canonical constraint IR."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .legacy_migration import build_migration_report, migrate_sample_to_canonical_ir
from .validator import compile_ir_to_edit_targets


def migrate_sample(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    canonical_ir, errors = migrate_sample_to_canonical_ir(sample)
    report = build_migration_report(sample, canonical_ir, errors)
    migrated = dict(sample)
    if canonical_ir is None:
        return migrated, report
    migrated["canonical_constraint_ir"] = canonical_ir
    migrated.update(compile_ir_to_edit_targets(canonical_ir))
    return migrated, report
