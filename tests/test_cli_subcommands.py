from __future__ import annotations

import json
import sys

import pytest

from llms.cli import main

SAMPLE_RAW_DATA = {
    "anthropic": {
        "name": "Anthropic",
        "models": {
            "claude-sonnet-4-6": {
                "name": "Claude Sonnet 4.6",
                "family": "claude",
                "cost": {
                    "input": 3.0,
                    "output": 15.0,
                    "cache_read": 0.3,
                    "cache_write": 3.75,
                },
                "limit": {"context": 200000, "output": 64000},
                "reasoning": False,
                "tool_call": True,
                "attachment": True,
                "temperature": True,
                "open_weights": False,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "knowledge": "2025-03",
                "release_date": "2025-06",
            },
            "claude-opus-4-6": {
                "name": "Claude Opus 4.6",
                "family": "claude",
                "cost": {"input": 15.0, "output": 75.0},
                "limit": {"context": 200000, "output": 64000},
                "reasoning": True,
                "tool_call": True,
                "attachment": True,
                "temperature": True,
                "open_weights": False,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
            },
        },
    },
    "openai": {
        "name": "OpenAI",
        "models": {
            "gpt-4o": {
                "name": "GPT-4o",
                "family": "gpt-4",
                "cost": {"input": 2.5, "output": 10.0},
                "limit": {"context": 128000, "output": 16384},
                "reasoning": False,
                "tool_call": True,
                "attachment": True,
                "temperature": True,
                "open_weights": False,
                "modalities": {"input": ["text", "image", "audio"], "output": ["text"]},
            },
        },
    },
    "meta": {
        "name": "Meta",
        "models": {
            "llama-3.3-70b": {
                "name": "Llama 3.3 70B",
                "family": "llama",
                "cost": {"input": 0.6, "output": 0.6},
                "limit": {"context": 131072, "output": 131072},
                "reasoning": False,
                "tool_call": True,
                "attachment": False,
                "temperature": True,
                "open_weights": True,
                "modalities": {"input": ["text"], "output": ["text"]},
            },
        },
    },
}


@pytest.fixture(autouse=True)
def mock_fetch(mocker):
    """Patch fetch_models for all CLI tests to avoid network calls."""
    return mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)


def _run_cli(args: list[str], capsys) -> tuple[str, str, int | None]:
    """Helper: set sys.argv, call main(), return (stdout, stderr, exit_code)."""
    sys.argv = ["llms", *args]
    exit_code = None
    try:
        main()
    except SystemExit as e:
        exit_code = e.code
    captured = capsys.readouterr()
    return captured.out, captured.err, exit_code


class TestGetSubcommand:
    def test_getサブコマンドでJSON出力(self, capsys):
        # Arrange
        args = ["get", "anthropic/claude-sonnet-4-6", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        model = json.loads(out)
        assert model["full_id"] == "anthropic/claude-sonnet-4-6"
        assert model["name"] == "Claude Sonnet 4.6"

    def test_getで存在しないモデルはエラー(self, capsys):
        # Arrange
        args = ["get", "nonexistent/model-xyz", "--json"]

        # Act
        _, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code != 0


class TestListSubcommand:
    def test_listサブコマンドでモデル一覧(self, capsys):
        # Arrange
        args = ["list", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 4

    def test_listでcapフィルタ(self, capsys):
        # Arrange
        args = ["list", "--cap", "reasoning", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 1
        assert models[0]["full_id"] == "anthropic/claude-opus-4-6"

    def test_listでmin_contextフィルタ(self, capsys):
        # Arrange
        args = ["list", "--min-context", "200k", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2
        assert all(m["provider_id"] == "anthropic" for m in models)

    def test_listでmax_input_costフィルタ(self, capsys):
        # Arrange
        args = ["list", "--max-input-cost", "3.0", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        # sonnet(3.0), gpt-4o(2.5), llama(0.6) => 3 models
        assert len(models) == 3
        full_ids = {m["full_id"] for m in models}
        assert "anthropic/claude-opus-4-6" not in full_ids

    def test_listでsort(self, capsys):
        # Arrange
        args = ["list", "--sort", "cost.input", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        costs = [m["cost"]["input"] for m in models]
        assert costs == sorted(costs)

    def test_listでlimit(self, capsys):
        # Arrange
        args = ["list", "--limit", "2", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2


class TestSearchSubcommand:
    def test_searchサブコマンド(self, capsys):
        # Arrange
        args = ["search", "claude", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2
        assert all("claude" in m["full_id"] for m in models)


class TestProvidersSubcommand:
    def test_providersサブコマンド(self, capsys):
        # Arrange
        args = ["providers", "--json"]

        # Act
        out, _, exit_code = _run_cli(args, capsys)

        # Assert
        assert exit_code is None or exit_code == 0
        providers = json.loads(out)
        assert len(providers) == 3
        provider_ids = {p["id"] for p in providers}
        assert "anthropic" in provider_ids
        assert "openai" in provider_ids
        assert "meta" in provider_ids


class TestBackwardCompatibility:
    def test_サブコマンドなしは後方互換(self, mocker, capsys):
        # Arrange
        # No subcommand → should attempt fzf interactive mode (backward compat path)
        # Patch run_fzf at the cli import site to prevent actual fzf launch
        mocker.patch("llms.cli.get_preview_command", return_value="echo")
        mock_run_fzf = mocker.patch(
            "llms.cli.run_fzf",
            side_effect=Exception("fzf not available in test"),
        )

        sys.argv = ["llms"]
        with pytest.raises(Exception, match="fzf not available in test"):
            main()

        # Assert: run_fzf was called, confirming the backward-compat pick path ran
        assert mock_run_fzf.called
