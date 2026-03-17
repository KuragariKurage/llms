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


def _compare_value(
    label: str, val_a: str, val_b: str, *, highlight_diff: bool = True
) -> str:
    """Format a single comparison row."""
    if highlight_diff and val_a != val_b:
        return f"{label:<14s} {YELLOW}{val_a:<24s}{RESET} {YELLOW}{val_b}{RESET}"
    return f"{label:<14s} {val_a:<24s} {val_b}"


def format_comparison(model_a: dict[str, Any], model_b: dict[str, Any]) -> str:
    """Format a side-by-side comparison of two models."""
    lines: list[str] = []

    name_a = model_a.get("name", "Unknown")
    name_b = model_b.get("name", "Unknown")
    lines.append(f"{'':<14s} {BOLD}{name_a:<24s}{RESET} {BOLD}{name_b}{RESET}")
    lines.append("")

    # Basic info
    lines.append(
        _compare_value(
            "Provider:",
            model_a.get("provider_name", "-"),
            model_b.get("provider_name", "-"),
        )
    )
    lines.append(
        _compare_value(
            "Model ID:",
            model_a.get("full_id", "-"),
            model_b.get("full_id", "-"),
            highlight_diff=False,
        )
    )
    lines.append(
        _compare_value(
            "Family:",
            model_a.get("family", "-") or "-",
            model_b.get("family", "-") or "-",
        )
    )

    # Cost
    cost_a = model_a.get("cost", {})
    cost_b = model_b.get("cost", {})
    if cost_a or cost_b:
        lines.append("")
        lines.append(f"{BOLD}-- Cost (per 1M tokens) --{RESET}")
        for field, label in [
            ("input", "Input:"),
            ("output", "Output:"),
            ("cache_read", "Cache Read:"),
            ("cache_write", "Cache Write:"),
        ]:
            va = cost_a.get(field)
            vb = cost_b.get(field)
            if va is not None or vb is not None:
                lines.append(_compare_value(label, _format_cost(va), _format_cost(vb)))

    # Limits
    limit_a = model_a.get("limit", {})
    limit_b = model_b.get("limit", {})
    if limit_a or limit_b:
        lines.append("")
        lines.append(f"{BOLD}-- Limits --{RESET}")
        lines.append(
            _compare_value(
                "Context:",
                _format_tokens(limit_a.get("context")),
                _format_tokens(limit_b.get("context")),
            )
        )
        lines.append(
            _compare_value(
                "Max Output:",
                _format_tokens(limit_a.get("output")),
                _format_tokens(limit_b.get("output")),
            )
        )

    # Capabilities
    lines.append("")
    lines.append(f"{BOLD}-- Capabilities --{RESET}")
    for cap_name, cap_key in [
        ("Reasoning:", "reasoning"),
        ("Tool Call:", "tool_call"),
        ("Attachment:", "attachment"),
        ("Temperature:", "temperature"),
        ("Open Weights:", "open_weights"),
    ]:
        va = "Y" if model_a.get(cap_key) else "N"
        vb = "Y" if model_b.get(cap_key) else "N"
        lines.append(_compare_value(cap_name, va, vb))

    # Modalities
    mod_a = model_a.get("modalities", {})
    mod_b = model_b.get("modalities", {})
    if mod_a or mod_b:
        lines.append("")
        lines.append(f"{BOLD}-- Modalities --{RESET}")
        in_a = ", ".join(mod_a.get("input", []))
        in_b = ", ".join(mod_b.get("input", []))
        out_a = ", ".join(mod_a.get("output", []))
        out_b = ", ".join(mod_b.get("output", []))
        lines.append(_compare_value("Input:", in_a or "-", in_b or "-"))
        lines.append(_compare_value("Output:", out_a or "-", out_b or "-"))

    # Metadata
    ka = model_a.get("knowledge", "")
    kb = model_b.get("knowledge", "")
    ra = model_a.get("release_date", "")
    rb = model_b.get("release_date", "")
    if ka or kb or ra or rb:
        lines.append("")
        if ka or kb:
            lines.append(_compare_value("Knowledge:", ka or "-", kb or "-"))
        if ra or rb:
            lines.append(_compare_value("Released:", ra or "-", rb or "-"))

    return "\n".join(lines)
