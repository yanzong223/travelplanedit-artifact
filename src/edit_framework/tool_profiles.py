"""Tool-profile helpers for standalone edit baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TOOL_PROFILE_DB_READ_TYPED = "db_read_typed"

SUPPORTED_TOOL_PROFILES = (
    TOOL_PROFILE_DB_READ_TYPED,
)


@dataclass(frozen=True)
class ToolProfileSpec:
    tool_profile: str
    enable_ct_atoms: bool
    enable_ct_verify: bool
    enable_ct_conflict_lift: bool
    enable_ct_notepad: bool
    db_read_enabled: bool = True


_TOOL_PROFILE_SPECS = {
    TOOL_PROFILE_DB_READ_TYPED: ToolProfileSpec(
        tool_profile=TOOL_PROFILE_DB_READ_TYPED,
        enable_ct_atoms=False,
        enable_ct_verify=False,
        enable_ct_conflict_lift=False,
        enable_ct_notepad=False,
    ),
}


def get_tool_profile_spec(tool_profile: str) -> ToolProfileSpec:
    try:
        return _TOOL_PROFILE_SPECS[tool_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool profile: {tool_profile}") from exc


def resolve_tool_profile(
    *,
    tool_profile: Optional[str] = None,
    enable_ct_atoms: Optional[bool] = None,
    enable_ct_verify: Optional[bool] = None,
    enable_ct_conflict_lift: Optional[bool] = None,
    enable_ct_notepad: Optional[bool] = None,
) -> ToolProfileSpec:
    return get_tool_profile_spec(tool_profile or TOOL_PROFILE_DB_READ_TYPED)
