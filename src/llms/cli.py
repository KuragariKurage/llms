from __future__ import annotations

import argparse
import json
import sys

from llms.clipboard import copy_to_clipboard
from llms.fetcher import fetch_models, flatten_models
from llms.formatter import format_fzf_lines, format_preview
from llms.selector import FzfNotFoundError, get_preview_command, run_fzf


def _find_model(models: list[dict], full_id: str) -> dict | None:
    for model in models:
        if model["full_id"] == full_id:
            return model
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzzy search LLM models from models.dev"
    )
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    parser.add_argument("--preview", metavar="MODEL_ID", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Print model ID to stdout instead of copying",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output selected model as JSON"
    )
    parser.add_argument("-p", "--provider", help="Filter by provider")
    args = parser.parse_args()

    data = fetch_models(force_refresh=args.refresh)
    models = flatten_models(data, provider_filter=args.provider)

    if args.preview:
        model = _find_model(models, args.preview)
        if model:
            print(format_preview(model))
        else:
            print(f"Model not found: {args.preview}")
        return

    if not models:
        print("No models found.", file=sys.stderr)
        sys.exit(1)

    lines = format_fzf_lines(models)

    try:
        preview_cmd = get_preview_command()
        selected_id = run_fzf(lines, preview_cmd=preview_cmd)
    except FzfNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if not selected_id:
        sys.exit(0)

    if args.json:
        model = _find_model(models, selected_id)
        if model:
            print(json.dumps(model, indent=2, ensure_ascii=False))
        return

    if args.no_copy:
        print(selected_id)
        return

    if copy_to_clipboard(selected_id):
        print(f"Copied: {selected_id}")
    else:
        print(selected_id)


if __name__ == "__main__":
    main()
