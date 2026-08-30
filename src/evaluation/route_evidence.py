"""Versioned, read-only route evidence for cascade counterfactuals."""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("walk", "metro", "taxi")
SCHEMA_VERSION = "chinatravel-route-evidence-v2"
_SUPPORTED_SCHEMA_VERSIONS = {
    "chinatravel-route-evidence-v1",
    SCHEMA_VERSION,
}

def _entries_sha256(entries: dict[str, Any]) -> str:
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

def route_key(city: str, start: str, end: str, start_time: str) -> str:
    values = ["".join(str(v or "").casefold().split()) for v in (city, start, end)]
    return "|".join([*values, str(start_time or "").strip()])

@dataclass(frozen=True)
class RouteEvidenceCache:
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RouteEvidenceCache":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS or not isinstance(payload.get("entries"), dict):
            raise ValueError("unsupported route evidence cache schema")
        if payload.get("entries_sha256") != _entries_sha256(payload["entries"]):
            raise ValueError("route evidence cache entries hash mismatch")
        return cls(payload)

    def lookup(self, city: str, start: str, end: str, start_time: str) -> dict[str, Any] | None:
        return self.payload["entries"].get(route_key(city, start, end, start_time))

def cache_payload(entries: list[dict[str, Any]], *, tool_fingerprint: dict[str, Any]) -> dict[str, Any]:
    indexed = {route_key(e["city"], e["start"], e["end"], e["start_time"]): e for e in entries}
    return {"schema_version": SCHEMA_VERSION, "tool_fingerprint": tool_fingerprint,
            "entries_sha256": _entries_sha256(indexed),
            "entries": dict(sorted(indexed.items()))}
