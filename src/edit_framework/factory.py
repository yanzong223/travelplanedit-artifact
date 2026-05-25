"""Framework factory for standalone edit baselines."""

from __future__ import annotations

from typing import Any

from edit_framework.base import EditFramework
from edit_framework.frameworks import (
    PTEREditFramework,
    ReactEditFramework,
    ReflexionEditFramework,
)
from edit_framework.runtime_tools.types import ExposureMode
from edit_framework.tools.chinatravel_tools import ChinaTravelToolAdapter
from edit_framework.tool_profiles import TOOL_PROFILE_DB_READ_TYPED
from edit_framework.world_env import ensure_session_world_env


def create_edit_framework(
    framework: str,
    *,
    llm_client: Any,
    world_env: Any,
    exposure_mode: str | None = None,
    tool_profile: str | None = None,
    max_steps: int = 30,
    max_tool_calls: int = 30,
    max_reflections: int = 2,
    guard_retries: int = 0,
    reflection_strategy: str = "reflexion",
    context_prompt: bool = False,
    database_prompt: bool = False,
    prompt_ablation: str = "original",
    semantic_tool_allowlist: list[str] | None = None,
    annotation_scaffold_level: str = "none",
) -> EditFramework:
    """Instantiate one supported standalone edit baseline."""

    if framework not in {"react", "reflexion", "pter"}:
        raise ValueError(f"Unsupported framework in this artifact: {framework}")

    if framework == "pter" and tool_profile == TOOL_PROFILE_DB_READ_TYPED:
        if prompt_ablation == "original":
            prompt_ablation = "unified_contract"

    tool_adapter = ChinaTravelToolAdapter(
        framework_name=framework,
        exposure_mode=exposure_mode or ExposureMode.PRIMITIVE_ONLY.value,
        tool_profile=tool_profile,
        semantic_tool_allowlist=semantic_tool_allowlist,
    )
    world_env = ensure_session_world_env(world_env)
    if framework == "react":
        return ReactEditFramework(
            llm_client=llm_client,
            world_env=world_env,
            tool_adapter=tool_adapter,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            context_prompt=context_prompt,
            database_prompt=database_prompt,
            prompt_ablation=prompt_ablation,
            guard_retries=guard_retries,
            annotation_scaffold_level=annotation_scaffold_level,
        )
    if framework == "reflexion":
        return ReflexionEditFramework(
            llm_client=llm_client,
            world_env=world_env,
            tool_adapter=tool_adapter,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_reflections=max_reflections,
            reflection_strategy=reflection_strategy,
            context_prompt=context_prompt,
            database_prompt=database_prompt,
            prompt_ablation=prompt_ablation,
            guard_retries=guard_retries,
        )
    if framework == "pter":
        return PTEREditFramework(
            llm_client=llm_client,
            world_env=world_env,
            tool_adapter=tool_adapter,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            prompt_ablation=prompt_ablation,
            guard_retries=guard_retries,
        )
    raise ValueError(f"Unsupported framework: {framework}")
