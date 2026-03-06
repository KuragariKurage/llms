from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SORTABLE_FIELDS: list[tuple[str, str]] = [
    ("cost.input", "Input cost per 1M tokens"),
    ("cost.output", "Output cost per 1M tokens"),
    ("cost.cache_read", "Cache read cost per 1M tokens"),
    ("cost.cache_write", "Cache write cost per 1M tokens"),
    ("limit.context", "Context window size (tokens)"),
    ("limit.output", "Max output tokens"),
    ("name", "Model name"),
    ("full_id", "Full model ID (provider/model)"),
    ("provider_id", "Provider ID"),
    ("release_date", "Release date"),
    ("knowledge", "Knowledge cutoff date"),
]

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
    cap_mode: str = "and"  # "and" (all caps must match) or "or" (any cap must match)
    min_context: int | None = None  # minimum context window tokens
    max_input_cost: float | None = None  # max $/1M input tokens
    sort: str | None = None  # field path like "cost.input", "limit.context"
    limit: int | None = None
    all_providers: bool = False  # if True, don't filter by DEFAULT_PROVIDERS
    include_deprecated: bool = False  # if True, include models with status="deprecated"


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


def _compute_match(text_lower: str, value: str) -> int:
    """Compute match score for a single field value.

    Returns:
        100 for exact match, 80 for starts-with, 50 for contains, 0 for no match.
    """
    v = value.lower()
    if v == text_lower:
        return 100
    if v.startswith(text_lower):
        return 80
    if text_lower in v:
        return 50
    return 0


def _apply_common_filters(
    result: list[dict[str, Any]], query: Query
) -> list[dict[str, Any]]:
    """Apply provider, deprecated, caps, context, cost filters."""
    if query.provider:
        result = [m for m in result if m.get("provider_id") == query.provider]
    elif not query.all_providers:
        result = [m for m in result if m.get("provider_id") in DEFAULT_PROVIDERS]

    if not query.include_deprecated:
        result = [m for m in result if m.get("status") != "deprecated"]

    if query.caps:
        if query.cap_mode == "or":
            result = [m for m in result if any(m.get(cap) for cap in query.caps)]
        else:
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

    return result


def _apply_sort_and_limit(
    result: list[dict[str, Any]], query: Query
) -> list[dict[str, Any]]:
    """Apply sorting and limiting."""
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


def filter_models(models: list[dict[str, Any]], query: Query) -> list[dict[str, Any]]:
    """Apply query filters, sorting, and limiting to a list of models."""
    result = _apply_common_filters(models, query)

    if query.text:
        text_lower = query.text.lower()
        result = [
            m
            for m in result
            if text_lower in m.get("full_id", "").lower()
            or text_lower in m.get("name", "").lower()
        ]

    return _apply_sort_and_limit(result, query)


def search_models(models: list[dict[str, Any]], query: Query) -> list[dict[str, Any]]:
    """Search models and annotate results with match metadata.

    Applies all the same filters as filter_models, but also adds
    _match_score (int) and _matched_fields (list[str]) to each result.
    Results are sorted by _match_score descending before applying query.sort.
    """
    result = _apply_common_filters(models, query)

    if query.text:
        text_lower = query.text.lower()
        match_fields = ["full_id", "name", "family", "id"]
        annotated = []
        for m in result:
            field_scores: dict[str, int] = {}
            for f in match_fields:
                val = m.get(f, "") or ""
                score = _compute_match(text_lower, val)
                if score > 0:
                    field_scores[f] = score
            if field_scores:
                best_score = max(field_scores.values())
                annotated.append(
                    {
                        **m,
                        "_match_score": best_score,
                        "_matched_fields": list(field_scores.keys()),
                    }
                )
        result = sorted(annotated, key=lambda m: m["_match_score"], reverse=True)
    else:
        result = [{**m, "_match_score": 0, "_matched_fields": []} for m in result]

    return _apply_sort_and_limit(result, query)
