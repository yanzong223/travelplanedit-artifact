"""Model alias routing and provider-specific request metadata."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_ROUTER_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "model_router.yaml"
)


@dataclass(frozen=True)
class ModelRoute:
    """Resolved model route from a stable experiment alias to provider details."""

    alias: str
    model: str
    provider: str = ""
    family: str = ""
    group: str = ""
    open_weight: bool = False
    capabilities: dict[str, bool] = field(default_factory=dict)
    request_overrides: dict[str, Any] = field(default_factory=dict)
    response_format_policy: str = ""
    notes: str = ""

    @property
    def supports_tools(self) -> bool:
        return bool(self.capabilities.get("tools"))

    @property
    def supports_structured_output(self) -> bool:
        return bool(self.capabilities.get("structured_output"))

    @property
    def supports_thinking(self) -> bool:
        return bool(self.capabilities.get("thinking"))

    @property
    def supports_thinking_budget(self) -> bool:
        return bool(self.capabilities.get("thinking_budget"))


class ModelRouter:
    """Resolve experiment aliases without hard-coding provider IDs in code."""

    def __init__(
        self,
        routes: dict[str, ModelRoute],
        model_sets: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.routes = dict(routes)
        self.model_sets = dict(model_sets or {})
        self._by_model = {route.model: route for route in self.routes.values()}

    @classmethod
    def from_path(cls, path: Path = DEFAULT_MODEL_ROUTER_PATH) -> "ModelRouter":
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        routes: dict[str, ModelRoute] = {}
        for alias, raw_config in (payload.get("models") or {}).items():
            config = dict(raw_config or {})
            routes[str(alias)] = ModelRoute(
                alias=str(alias),
                model=str(config.get("model") or alias),
                provider=str(config.get("provider") or ""),
                family=str(config.get("family") or ""),
                group=str(config.get("group") or ""),
                open_weight=bool(config.get("open_weight", False)),
                capabilities=dict(config.get("capabilities") or {}),
                request_overrides=dict(config.get("request_overrides") or {}),
                response_format_policy=str(config.get("response_format_policy") or ""),
                notes=str(config.get("notes") or ""),
            )
        return cls(routes=routes, model_sets=dict(payload.get("model_sets") or {}))

    def resolve(self, model_or_alias: str | None, provider: str | None = None) -> ModelRoute | None:
        if not model_or_alias:
            return None
        route = self.routes.get(model_or_alias) or self._by_model.get(model_or_alias)
        if route is None:
            return None
        if provider and route.provider and route.provider.lower() != provider.lower():
            return None
        return route

    def resolve_model(self, model_or_alias: str, provider: str | None = None) -> str:
        route = self.resolve(model_or_alias, provider=provider)
        return route.model if route is not None else model_or_alias

    def request_overrides(self, model_or_alias: str, provider: str | None = None) -> dict[str, Any]:
        route = self.resolve(model_or_alias, provider=provider)
        if route is None:
            return {}
        return copy.deepcopy(route.request_overrides)

    def model_set(self, name: str) -> list[str]:
        config = self.model_sets.get(name)
        if not config:
            raise KeyError(f"Unknown model set: {name}")
        groups = config.get("groups") or {}
        aliases: list[str] = []
        for group_aliases in groups.values():
            aliases.extend(str(alias) for alias in group_aliases)
        return aliases


@lru_cache(maxsize=1)
def get_model_router() -> ModelRouter:
    return ModelRouter.from_path()


def get_model_set(name: str) -> list[str]:
    return get_model_router().model_set(name)
