"""Path helpers for standalone edit-framework runs and evaluations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def sanitize_path_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def build_run_timestamp(started_at: datetime | None = None) -> str:
    moment = started_at or datetime.now()
    return moment.strftime("%Y%m%d_%H%M%S")


def build_tool_config_label(
    *,
    tool_profile: str | None = None,
    exposure_mode: str | None = None,
    enable_ct_atoms: bool = False,
    enable_ct_verify: bool = False,
    enable_ct_conflict_lift: bool = False,
    enable_ct_notepad: bool | None = None,
) -> str:
    if tool_profile:
        return sanitize_path_token(tool_profile)
    parts = [exposure_mode or "default_tools"]
    enabled_flags = []
    if enable_ct_atoms:
        enabled_flags.append("atoms")
    if enable_ct_verify:
        enabled_flags.append("verify")
    if enable_ct_conflict_lift:
        enabled_flags.append("lift")
    if enable_ct_notepad is True:
        enabled_flags.append("notepad")
    elif enable_ct_notepad is False:
        enabled_flags.append("no_notepad")
    if enabled_flags:
        parts.extend(enabled_flags)
    return sanitize_path_token("_".join(parts))


def build_experiment_name(
    *,
    started_at: str,
    model_name: str,
    framework: str,
    tool_label: str,
    dataset_label: str | None = None,
) -> str:
    parts = [
        started_at,
        sanitize_path_token(model_name),
        sanitize_path_token(framework),
        sanitize_path_token(tool_label),
    ]
    if dataset_label:
        parts.append(sanitize_path_token(dataset_label))
    return "_".join(part for part in parts if part)


def resolve_model_experiment_dir(
    experiments_root: Path,
    *,
    started_at: str,
    model_name: str,
    framework: str,
    tool_label: str,
    dataset_label: str | None = None,
) -> Path:
    model_dir = Path(experiments_root) / sanitize_path_token(model_name)
    return model_dir / build_experiment_name(
        started_at=started_at,
        model_name=model_name,
        framework=framework,
        tool_label=tool_label,
        dataset_label=dataset_label,
    )


def resolve_runs_root(project_root: Path, output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return project_root / "results" / "edit_framework_runs"


def resolve_flat_experiment_dir(output_dir: str | Path | None) -> Path:
    if not output_dir:
        raise ValueError("flat output layout requires an explicit output_dir")
    return Path(output_dir)


def resolve_experiment_dir(
    project_root: Path,
    *,
    framework: str,
    model_name: str,
    tool_label: str | None = None,
    output_dir: str | Path | None = None,
    started_at: datetime | None = None,
    layout_mode: str = "nested",
) -> Path:
    if layout_mode == "flat":
        return resolve_flat_experiment_dir(output_dir)
    runs_root = resolve_runs_root(project_root, output_dir)
    framework_part = sanitize_path_token(framework)
    if tool_label:
        framework_part = f"{framework_part}_{sanitize_path_token(tool_label)}"
    return (
        runs_root
        / sanitize_path_token(model_name)
        / framework_part
        / build_run_timestamp(started_at)
    )


def resolve_run_dir(
    project_root: Path,
    *,
    framework: str,
    model_name: str,
    tool_label: str | None = None,
    output_dir: str | Path | None = None,
    started_at: datetime | None = None,
    layout_mode: str = "nested",
) -> Path:
    return resolve_experiment_dir(
        project_root,
        framework=framework,
        model_name=model_name,
        tool_label=tool_label,
        output_dir=output_dir,
        started_at=started_at,
        layout_mode=layout_mode,
    ) / "run"


def resolve_default_evaluation_dir(project_root: Path, run_dir: str | Path) -> Path:
    run_path = Path(run_dir).resolve()
    if run_path.name == "run":
        return run_path.parent / "evaluation"

    runs_root = (project_root / "results" / "edit_framework_runs").resolve()
    eval_root = project_root / "results" / "edit_framework_eval_ready"
    try:
        relative = run_path.relative_to(runs_root)
        return eval_root / relative
    except ValueError:
        return run_path.parent / f"{run_path.name}_eval_ready"
