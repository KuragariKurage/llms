from __future__ import annotations

from typing import Any

from llms.fetcher import fetch_models, flatten_models
from llms.query import Query, filter_models


class ModelNotFoundError(KeyError):
    pass


class Client:
    """Programmatic interface for querying LLM models."""

    def __init__(self, force_refresh: bool = False) -> None:
        self._data = fetch_models(force_refresh=force_refresh)

    def _all_models(self, provider: str | None = None) -> list[dict[str, Any]]:
        return flatten_models(self._data, provider_filter=provider)

    def get(self, model_id: str) -> dict[str, Any]:
        """Get a single model by full ID (e.g. 'anthropic/claude-sonnet-4-6')."""
        models = self._all_models()
        for model in models:
            if model["full_id"] == model_id:
                return model
        raise ModelNotFoundError(f"Model not found: {model_id}")

    def list(self, query: Query | None = None) -> list[dict[str, Any]]:
        """List models matching the given query filters."""
        if query is None:
            query = Query()
        models = self._all_models(provider=query.provider)
        return filter_models(models, query)

    def search(self, text: str, query: Query | None = None) -> list[dict[str, Any]]:
        """Search models by text with optional additional filters."""
        if query is None:
            query = Query()
        query = Query(
            provider=query.provider,
            text=text,
            caps=query.caps,
            min_context=query.min_context,
            max_input_cost=query.max_input_cost,
            sort=query.sort,
            limit=query.limit,
        )
        models = self._all_models(provider=query.provider)
        return filter_models(models, query)

    def providers(self) -> list[dict[str, str]]:
        """List all available providers."""
        result = []
        for provider_id, provider_data in self._data.items():
            if not isinstance(provider_data, dict):
                continue
            result.append(
                {
                    "id": provider_id,
                    "name": provider_data.get("name", provider_id),
                }
            )
        return sorted(result, key=lambda p: p["id"])
