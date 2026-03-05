# llms

Fuzzy search LLM models from [models.dev](https://models.dev/) in your terminal.

[日本語版 README はこちら](README.ja.md)

## Demo

```
$ llms
Model> claude-sonnet-4.6

  anthropic/claude-sonnet-4-6          | Claude Sonnet 4.6
  opencode/claude-sonnet-4-6           |
  venice/claude-sonnet-4-6             | Provider:    Anthropic
  ...                                  | Model ID:    anthropic/claude-sonnet-4-6
                                       | Cost:        $3.00 / $15.00
                                       | Context:     200K

Copied: anthropic/claude-sonnet-4-6
```

## Installation

### Prerequisites

- Python 3.13+
- [fzf](https://github.com/junegunn/fzf)
- [uv](https://github.com/astral-sh/uv)

### Quick Install

```bash
./install.sh
```

### Manual Install

```bash
uv tool install -e .
```

This makes the `llms` command available globally.

## Usage

```bash
llms                  # Fuzzy search → Enter to copy model ID to clipboard
llms --refresh        # Force refresh cache
llms --no-copy        # Print model ID to stdout (for piping)
llms --json           # Output selected model details as JSON
llms -p anthropic     # Filter by provider
```

### Key Bindings

| Key | Action |
|-----|--------|
| Type | Fuzzy search |
| Up/Down | Navigate models |
| Enter | Copy model ID to clipboard |
| Ctrl-C | Exit |

## Data Source

Model data is fetched from the [models.dev](https://models.dev/) `/api.json` endpoint.
Responses are cached locally for 1 hour at `~/Library/Caches/llms/`.

## Development

```bash
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
```
