from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# First-party model creators and major inference platforms.
# Aggregators/resellers (bedrock, azure, openrouter, kilo, vercel, etc.)
# are excluded by default — use --all to include them.
DEFAULT_PROVIDERS: frozenset[str] = frozenset(
    {
        # First-party model creators
        "alibaba",
        "anthropic",
        "bailing",
        "cohere",
        "deepseek",
        "google",
        "inception",
        "llama",
        "minimax",
        "mistral",
        "moonshotai",
        "nova",
        "nvidia",
        "openai",
        "perplexity",
        "stepfun",
        "upstage",
        "xai",
        "xiaomi",
        "zhipuai",
        # Major inference platforms
        "cerebras",
        "fireworks-ai",
        "groq",
        "huggingface",
        "togetherai",
    }
)


@dataclass
class Query:
    """Filter and sort criteria for model search."""

    provider: str | None = None
    text: str | None = None  # fuzzy text match against full_id and name
    caps: list[str] = field(default_factory=list)  # e.g. ["tool_call", "reasoning"]
    min_context: int | None = None  # minimum context window tokens
    max_input_cost: float | None = None  # max $/1M input tokens
    sort: str | None = None  # field path like "cost.input", "limit.context"
    limit: int | None = None
    all_providers: bool = False  # if True, don't filter by DEFAULT_PROVIDERS


def _parse_token_count(value: str) -> int:
    """Parse human-friendly token counts like '128k', '1m'."""
    v = value.strip().lower()
    if v.endswith("m"):
        return int(float(v[:-1]) * 1_000_000)
    if v.endswith("k"):
        return int(float(v[:-1]) * 1_000)
    return int(v)


def _get_nested(d: dict, path: str) -> Any:
    """Get nested dict value by dot path like 'cost.input'."""
    keys = path.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def filter_models(models: list[dict[str, Any]], query: Query) -> list[dict[str, Any]]:
    """Apply query filters, sorting, and limiting to a list of models."""
    result = models

    if query.provider:
        result = [m for m in result if m.get("provider_id") == query.provider]
    elif not query.all_providers:
        result = [m for m in result if m.get("provider_id") in DEFAULT_PROVIDERS]

    if query.text:
        text_lower = query.text.lower()
        result = [
            m
            for m in result
            if text_lower in m.get("full_id", "").lower()
            or text_lower in m.get("name", "").lower()
        ]

    for cap in query.caps:
        result = [m for m in result if m.get(cap)]

    if query.min_context is not None:
        result = [
            m
            for m in result
            if (m.get("limit") or {}).get("context") is not None
            and m["limit"]["context"] >= query.min_context
        ]

    if query.max_input_cost is not None:
        result = [
            m
            for m in result
            if (m.get("cost") or {}).get("input") is not None
            and m["cost"]["input"] <= query.max_input_cost
        ]

    if query.sort:
        result = sorted(
            result,
            key=lambda m: (
                _get_nested(m, query.sort) is None,
                _get_nested(m, query.sort) or 0,
            ),
        )

    if query.limit is not None:
        result = result[: query.limit]

    return result
