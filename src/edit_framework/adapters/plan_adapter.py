"""Canonical plan validation helpers for standalone edit baselines."""

from __future__ import annotations

from typing import Any

from utils.chinatravel_plan import (
    is_chinatravel_plan,
    require_chinatravel_plan,
    validate_chinatravel_plan,
)

__all__ = [
    "is_chinatravel_plan",
    "require_chinatravel_plan",
    "validate_chinatravel_plan",
]


def require_origin_plan(origin_plan: dict[str, Any] | None) -> dict[str, Any]:
    return require_chinatravel_plan(origin_plan, context="origin_plan")
