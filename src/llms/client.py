from __future__ import annotations

from typing import Any

from llms.fetcher import fetch_models, flatten_models
from llms.query import Query, filter_models, search_models


class ModelNotFoundError(KeyError):
    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.suggestions: list[str] = suggestions or []


class Client:
    """Programmatic interface for querying LLM models."""

    def __init__(self, force_refresh: bool = False) -> None:
        self._data = fetch_models(force_refresh=force_refresh)

    def _all_models(self, provider: str | None = None) -> list[dict[str, Any]]:
        return flatten_models(self._data, provider_filter=provider)

    def find_similar_models(self, model_id: str) -> list[str]:
        """Find models with IDs that end with or contain the given model_id."""
        if "/" in model_id:
            return []

        models = self._all_models()
        suffix = "/" + model_id

        suffix_matches = sorted(
            m["full_id"] for m in models if m["full_id"].endswith(suffix)
        )
        if suffix_matches:
            return suffix_matches[:5]

        substring_matches = sorted(
            m["full_id"] for m in models if model_id in m["full_id"]
        )
        return substring_matches[:5]

    def get(self, model_id: str) -> dict[str, Any]:
        """Get a single model by full ID (e.g. 'anthropic/claude-sonnet-4-6')."""
        models = self._all_models()
        for model in models:
            if model["full_id"] == model_id:
                return model
        suggestions = self.find_similar_models(model_id)
        raise ModelNotFoundError(
            f"Model not found: {model_id}", suggestions=suggestions
        )

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
            cap_mode=query.cap_mode,
            min_context=query.min_context,
            max_input_cost=query.max_input_cost,
            sort=query.sort,
            limit=query.limit,
            all_providers=query.all_providers,
            include_deprecated=query.include_deprecated,
        )
        models = self._all_models(provider=query.provider)
        return search_models(models, query)

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
