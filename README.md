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

### Homebrew (macOS)

```bash
brew tap KuragariKurage/llms
brew install llms
```

This installs `llms` and fzf together.

### PyPI

```bash
# uv (recommended)
uv tool install llms-dev

# pipx
pipx install llms-dev
```

Prerequisites: Python 3.13+, [fzf](https://github.com/junegunn/fzf) (for interactive mode)

#### Install from GitHub (without cloning)

```bash
uv tool install git+https://github.com/KuragariKurage/llms.git
pipx install git+https://github.com/KuragariKurage/llms.git
```

This makes the `llms` command available globally.

## Usage

### Interactive Mode (fzf)

```bash
llms                  # Fuzzy search → Enter to copy model ID to clipboard
llms --refresh        # Force refresh cache
llms --no-copy        # Print model ID to stdout (for piping)
llms --json           # Output selected model details as JSON
llms -p anthropic     # Filter by provider
```

#### Key Bindings

| Key | Action |
|-----|--------|
| Type | Fuzzy search |
| Up/Down | Navigate models |
| Enter | Copy model ID to clipboard |
| Ctrl-C | Exit |

### Programmatic Mode (for AI agents & scripts)

Non-interactive subcommands that output structured data — no fzf required.

```bash
# Get a model by ID
llms get anthropic/claude-sonnet-4-6 --json

# List models with filters
llms list --json
llms list --cap tool_call --min-context 128k --sort cost.input --limit 5 --json
llms list --cap reasoning --max-input-cost 5.0 --json

# Text search
llms search claude --json --limit 5
llms search llama -p meta --json

# List providers
llms providers --json
```

#### Filter Flags (for `list` and `search`)

| Flag | Description | Example |
|------|-------------|---------|
| `-p`, `--provider` | Filter by provider | `-p anthropic` |
| `--cap` | Capability filter (repeatable) | `--cap tool_call --cap reasoning` |
| `--min-context` | Minimum context window | `--min-context 128k` |
| `--max-input-cost` | Max input cost ($/1M tokens) | `--max-input-cost 3.0` |
| `--sort` | Sort by field | `--sort cost.input` |
| `--limit` | Max number of results | `--limit 10` |

#### Output Formats

| Flag | Format |
|------|--------|
| `--json` | Pretty-printed JSON |
| `--jsonl` | One JSON object per line |
| *(none)* | One model ID per line |

### Python Library

```python
from llms.client import Client
from llms.query import Query

client = Client()

# Get by ID
model = client.get("anthropic/claude-sonnet-4-6")

# List with filters
models = client.list(Query(
    caps=["tool_call"],
    min_context=128_000,
    sort="cost.input",
    limit=5,
))

# Text search
results = client.search("claude")

# Providers
providers = client.providers()
```

## Data Source

Model data is fetched from the [models.dev](https://models.dev/) `/api.json` endpoint.
Responses are cached locally for 1 hour at `~/Library/Caches/llms/`.

## Development

```bash
uv sync
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
```
