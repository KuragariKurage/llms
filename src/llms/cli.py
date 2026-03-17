from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from llms.clipboard import copy_to_clipboard
from llms.formatter import format_comparison, format_fzf_lines, format_preview
from llms.selector import FzfNotFoundError, get_preview_command, run_fzf, run_fzf_multi


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="Output as JSON")
    group.add_argument("--jsonl", action="store_true", help="Output as JSON Lines")
    parser.add_argument("--refresh", action="store_true", help="Force cache refresh")


def _add_filter_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--provider", help="Filter by provider")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all_providers",
        help="Show models from all providers (default: major providers only)",
    )
    parser.add_argument(
        "--cap",
        action="append",
        dest="caps",
        metavar="CAPABILITY",
        help="Capability filter (repeatable). Combined with AND by default (see --cap-mode)",
    )
    parser.add_argument(
        "--cap-mode",
        dest="cap_mode",
        choices=["and", "or"],
        default="and",
        help="How to combine multiple --cap filters: 'and' (all must match) or 'or' (any must match). Default: and",
    )
    parser.add_argument(
        "--min-context",
        metavar="SIZE",
        help='Minimum context window (e.g. "128k", "1m")',
    )
    parser.add_argument(
        "--max-input-cost",
        type=float,
        metavar="COST",
        help="Maximum input cost per 1M tokens",
    )
    parser.add_argument(
        "--sort",
        metavar="FIELD",
        help='Sort field (e.g. "cost.input", "limit.context"). Use --sort-fields to see all options',
    )
    parser.add_argument("--limit", type=int, metavar="N", help="Max results")
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated models (excluded by default)",
    )


def _build_query(args: argparse.Namespace) -> Any:
    from llms.query import Query, _parse_token_count

    raw_min_context = getattr(args, "min_context", None)
    min_context = (
        _parse_token_count(raw_min_context) if raw_min_context is not None else None
    )

    return Query(
        provider=getattr(args, "provider", None),
        text=None,
        caps=getattr(args, "caps", None) or [],
        cap_mode=getattr(args, "cap_mode", "and"),
        min_context=min_context,
        max_input_cost=getattr(args, "max_input_cost", None),
        sort=getattr(args, "sort", None),
        limit=getattr(args, "limit", None),
        all_providers=getattr(args, "all_providers", False),
        include_deprecated=getattr(args, "include_deprecated", False),
    )


def _print_models(models: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
    elif args.jsonl:
        for model in models:
            print(json.dumps(model, ensure_ascii=False))
    else:
        for model in models:
            print(model["full_id"])


def _run_pick(args: argparse.Namespace) -> None:
    from llms.client import Client

    client = Client(force_refresh=getattr(args, "refresh", False))
    query = _build_query(args)
    models = client.list(query)

    if getattr(args, "preview", None):
        try:
            model = client.get(args.preview)
            print(format_preview(model))
        except Exception:
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

    if getattr(args, "json", False):
        model = client.get(selected_id)
        print(json.dumps(model, indent=2, ensure_ascii=False))
        return

    if getattr(args, "no_copy", False):
        print(selected_id)
        return

    if copy_to_clipboard(selected_id):
        print(f"Copied: {selected_id}")
    else:
        print(selected_id)


def _run_get(args: argparse.Namespace) -> None:
    from llms.client import Client, ModelNotFoundError

    client = Client(force_refresh=args.refresh)
    try:
        model = client.get(args.model_id)
    except ModelNotFoundError as e:
        print(f"Model not found: {args.model_id}", file=sys.stderr)
        if hasattr(e, "suggestions") and e.suggestions:
            print("Did you mean?", file=sys.stderr)
            for s in e.suggestions:
                print(f"  - {s}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(model, indent=2, ensure_ascii=False))
    elif args.jsonl:
        print(json.dumps(model, ensure_ascii=False))
    else:
        print(format_preview(model))


def _run_list(args: argparse.Namespace) -> None:
    if getattr(args, "sort_fields", False):
        from llms.query import SORTABLE_FIELDS

        print("Available sort fields:")
        for field, desc in SORTABLE_FIELDS:
            print(f"  {field:<20s} {desc}")
        return

    from llms.client import Client

    client = Client(force_refresh=args.refresh)
    query = _build_query(args)
    models = client.list(query)
    _print_models(models, args)


def _run_search(args: argparse.Namespace) -> None:
    from llms.client import Client
    from llms.query import Query

    client = Client(force_refresh=args.refresh)
    query = _build_query(args)
    query = Query(
        provider=query.provider,
        text=args.query,
        caps=query.caps,
        cap_mode=query.cap_mode,
        min_context=query.min_context,
        max_input_cost=query.max_input_cost,
        sort=query.sort,
        limit=query.limit,
        all_providers=query.all_providers,
        include_deprecated=query.include_deprecated,
    )
    models = client.search(args.query, query)
    _print_models(models, args)


def _run_compare(args: argparse.Namespace) -> None:
    from llms.client import Client

    client = Client(force_refresh=getattr(args, "refresh", False))
    query = _build_query(args)
    models = client.list(query)

    if not models:
        print("No models found.", file=sys.stderr)
        sys.exit(1)

    lines = format_fzf_lines(models)

    try:
        preview_cmd = get_preview_command()
        selected_ids = run_fzf_multi(lines, limit=2, preview_cmd=preview_cmd)
    except FzfNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if len(selected_ids) < 2:
        if selected_ids:
            print("Please select 2 models to compare.", file=sys.stderr)
        sys.exit(0)

    model_a = client.get(selected_ids[0])
    model_b = client.get(selected_ids[1])

    if getattr(args, "json", False):
        print(json.dumps([model_a, model_b], indent=2, ensure_ascii=False))
        return

    print(format_comparison(model_a, model_b))


def _run_providers(args: argparse.Namespace) -> None:
    from llms.client import Client

    client = Client(force_refresh=args.refresh)
    providers = client.providers()

    if args.json:
        print(json.dumps(providers, indent=2, ensure_ascii=False))
    else:
        for p in providers:
            print(p["id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzzy search LLM models from models.dev"
    )

    # Top-level backward-compat flags (used when no subcommand is given)
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
    parser.add_argument("--jsonl", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-p", "--provider", help="Filter by provider")
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all_providers",
        help="Show models from all providers",
    )

    subparsers = parser.add_subparsers(dest="command")

    # pick subcommand
    pick_parser = subparsers.add_parser("pick", help="Interactive fzf model picker")
    pick_parser.add_argument("-p", "--provider", help="Filter by provider")
    pick_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all_providers",
        help="Show models from all providers",
    )
    pick_parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Print model ID to stdout instead of copying",
    )
    pick_parser.add_argument("--json", action="store_true", help="Output model as JSON")
    pick_parser.add_argument("--jsonl", action="store_true", help=argparse.SUPPRESS)
    pick_parser.add_argument(
        "--refresh", action="store_true", help="Force cache refresh"
    )
    pick_parser.add_argument("--preview", metavar="MODEL_ID", help=argparse.SUPPRESS)

    # get subcommand
    get_parser = subparsers.add_parser("get", help="Get a model by ID")
    get_parser.add_argument("model_id", metavar="MODEL_ID")
    _add_output_flags(get_parser)

    # list subcommand
    list_parser = subparsers.add_parser(
        "list", help="List models with optional filters"
    )
    _add_output_flags(list_parser)
    _add_filter_flags(list_parser)
    list_parser.add_argument(
        "--sort-fields",
        action="store_true",
        help="Show available sort fields and exit",
    )

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Text search models")
    search_parser.add_argument("query", metavar="QUERY")
    _add_output_flags(search_parser)
    _add_filter_flags(search_parser)

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two models side by side (interactive fzf)"
    )
    compare_parser.add_argument("-p", "--provider", help="Filter by provider")
    compare_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all_providers",
        help="Show models from all providers",
    )
    compare_parser.add_argument(
        "--json", action="store_true", help="Output comparison as JSON"
    )
    compare_parser.add_argument(
        "--refresh", action="store_true", help="Force cache refresh"
    )

    # providers subcommand
    providers_parser = subparsers.add_parser("providers", help="List providers")
    providers_parser.add_argument("--json", action="store_true", help="Output as JSON")
    providers_parser.add_argument(
        "--refresh", action="store_true", help="Force cache refresh"
    )

    args = parser.parse_args()

    command = args.command

    if command is None:
        _run_pick(args)
    elif command == "pick":
        _run_pick(args)
    elif command == "get":
        _run_get(args)
    elif command == "list":
        _run_list(args)
    elif command == "search":
        _run_search(args)
    elif command == "compare":
        _run_compare(args)
    elif command == "providers":
        _run_providers(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
