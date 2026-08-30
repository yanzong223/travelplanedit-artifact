"""Canonical task identity and paper-manifest support for cascade analysis."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_MANIFEST_SCHEMA = "cascade-paper-manifest-v2"
_SUPPORTED_MANIFEST_SCHEMAS = {
    "cascade-paper-manifest-v1",
    PAPER_MANIFEST_SCHEMA,
}
_VOLATILE_METADATA_KEYS = {"batch_id", "primary_conflict", "sample_id", "sample_path"}


def input_fingerprint(input_payload: dict[str, Any]) -> str:
    """Hash task content while ignoring run-local identity/path metadata.

    The sample basename is deliberately *not* part of this hash.  ``task_key``
    adds it separately because a few batch006 cases have identical task content.
    """
    canonical = copy.deepcopy(input_payload)
    canonical.pop("case_id", None)
    metadata = canonical.get("metadata")
    if isinstance(metadata, dict):
        for key in _VOLATILE_METADATA_KEYS:
            metadata.pop(key, None)
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_key(input_payload: dict[str, Any], sample_id: str) -> str:
    return f"{input_fingerprint(input_payload)}:{Path(sample_id).stem}"


def metadata_category(input_payload: dict[str, Any]) -> str | None:
    """Return a semantic conflict category, never a timestamp directory name."""
    metadata = input_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    primary = metadata.get("primary_conflict")
    if isinstance(primary, list) and primary and all(isinstance(item, str) and item for item in primary):
        return "_".join(primary)
    sample_path = str(metadata.get("sample_path") or "")
    parts = Path(sample_path).parts
    if "batches" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("batches")
        if index + 1 < len(parts):
            candidate = parts[index + 1]
            if candidate.startswith(("parameter_", "structural_", "compositional_")):
                return candidate
    return None


def load_paper_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in _SUPPORTED_MANIFEST_SCHEMAS:
        raise ValueError("unsupported cascade paper manifest schema")
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("cascade paper manifest has no models")
    for model, spec in models.items():
        records = spec.get("records") if isinstance(spec, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"cascade paper manifest model {model!r} has no records")
        keys = [record.get("task_key") for record in records]
        if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError(f"cascade paper manifest model {model!r} has invalid/duplicate task keys")
        for record in records:
            if not isinstance(record.get("strict_eligible"), bool):
                raise ValueError(f"cascade paper manifest model {model!r} lacks canonical eligibility")
            if not str(record.get("category") or "").startswith(
                ("parameter_", "structural_", "compositional_")
            ):
                raise ValueError(f"cascade paper manifest model {model!r} has invalid category")
    task_count = payload.get("task_count")
    if not isinstance(task_count, int) or task_count < 0:
        raise ValueError("cascade paper manifest has invalid task_count")
    task_sets = {
        model: sorted(str(record["task_key"]) for record in spec["records"])
        for model, spec in models.items()
    }
    if any(len(keys) != task_count for keys in task_sets.values()):
        raise ValueError("cascade paper manifest task_count does not match model records")
    reference = next(iter(task_sets.values()))
    if any(keys != reference for keys in task_sets.values()):
        raise ValueError("cascade paper manifest model task sets differ")
    expected_sha = hashlib.sha256("\n".join(reference).encode("utf-8")).hexdigest()
    if payload.get("task_set_sha256") != expected_sha:
        raise ValueError("cascade paper manifest task_set_sha256 mismatch")
    payload["_manifest_dir"] = str(manifest_path.parent)
    return payload


def resolve_manifest_record_path(
    manifest: dict[str, Any],
    record: dict[str, Any],
    field: str,
) -> Path:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"cascade paper manifest record lacks {field}")
    path = Path(value)
    if path.is_absolute():
        return path
    if manifest.get("path_base") == "project_root":
        return (PROJECT_ROOT / path).resolve()
    manifest_dir = Path(str(manifest.get("_manifest_dir") or "."))
    return (manifest_dir / path).resolve()


def manifest_records(
    manifest: dict[str, Any], model: str, *, eligible_only: bool = False,
) -> Iterable[dict[str, Any]]:
    try:
        records = manifest["models"][model]["records"]
    except KeyError as exc:
        raise ValueError(f"model {model!r} is absent from cascade paper manifest") from exc
    for record in records:
        if not eligible_only or record["strict_eligible"]:
            yield record
