from __future__ import annotations

from typing import Any

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def _format_cost(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.2f}"


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1_000_000:
        return f"{value // 1_000_000}M"
    if value >= 1_000:
        return f"{value // 1_000}K"
    return str(value)


def _format_modalities(model: dict[str, Any]) -> str:
    modalities = model.get("modalities", {})
    if not modalities:
        return ""
    inputs = modalities.get("input", [])
    outputs = modalities.get("output", [])
    if not inputs and not outputs:
        return ""
    return f"{','.join(inputs)}->{','.join(outputs)}"


def _format_capabilities(model: dict[str, Any]) -> str:
    caps = []
    if model.get("reasoning"):
        caps.append("reason")
    if model.get("tool_call"):
        caps.append("tools")
    if model.get("open_weights"):
        caps.append("open")
    return " ".join(caps)


def format_fzf_line(model: dict[str, Any]) -> str:
    return model.get("full_id", "")


def format_fzf_lines(models: list[dict[str, Any]]) -> str:
    return "\n".join(format_fzf_line(m) for m in models)


def format_preview(model: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append(f"{BOLD}{model.get('name', 'Unknown')}{RESET}")
    lines.append("")
    lines.append(f"Provider:    {model.get('provider_name', '-')}")
    lines.append(f"Model ID:    {model.get('full_id', '-')}")
    family = model.get("family")
    if family:
        lines.append(f"Family:      {family}")

    cost = model.get("cost", {})
    if cost:
        lines.append("")
        lines.append(f"{BOLD}-- Cost (per 1M tokens) --{RESET}")
        lines.append(f"Input:       {_format_cost(cost.get('input'))}")
        lines.append(f"Output:      {_format_cost(cost.get('output'))}")
        if cost.get("cache_read") is not None:
            lines.append(f"Cache Read:  {_format_cost(cost.get('cache_read'))}")
        if cost.get("cache_write") is not None:
            lines.append(f"Cache Write: {_format_cost(cost.get('cache_write'))}")

    limit = model.get("limit", {})
    if limit:
        lines.append("")
        lines.append(f"{BOLD}-- Limits --{RESET}")
        lines.append(f"Context:     {_format_tokens(limit.get('context'))}")
        lines.append(f"Max Output:  {_format_tokens(limit.get('output'))}")

    lines.append("")
    lines.append(f"{BOLD}-- Capabilities --{RESET}")
    cap_items = [
        ("Reasoning", model.get("reasoning")),
        ("Tool Call", model.get("tool_call")),
        ("Attachment", model.get("attachment")),
        ("Temperature", model.get("temperature")),
        ("Open Weights", model.get("open_weights")),
    ]
    cap_strs = [f"{name}: {'Y' if val else 'N'}" for name, val in cap_items]
    lines.append("  ".join(cap_strs))

    modalities = model.get("modalities", {})
    if modalities:
        lines.append("")
        lines.append(f"{BOLD}-- Modalities --{RESET}")
        inputs = modalities.get("input", [])
        outputs = modalities.get("output", [])
        if inputs:
            lines.append(f"Input:       {', '.join(inputs)}")
        if outputs:
            lines.append(f"Output:      {', '.join(outputs)}")

    knowledge = model.get("knowledge")
    release = model.get("release_date")
    if knowledge or release:
        lines.append("")
        if knowledge:
            lines.append(f"Knowledge:   {knowledge}")
        if release:
            lines.append(f"Released:    {release}")

    return "\n".join(lines)
