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
                "modalities": {
                    "input": ["text", "image", "audio"],
                    "output": ["text"],
                },
            },
            "gpt-4-turbo": {
                "name": "GPT-4 Turbo",
                "family": "gpt-4",
                "status": "deprecated",
                "cost": {"input": 10.0, "output": 30.0},
                "limit": {"context": 128000, "output": 4096},
                "reasoning": False,
                "tool_call": True,
                "attachment": False,
                "temperature": True,
                "open_weights": False,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
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
        args = ["get", "anthropic/claude-sonnet-4-6", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        model = json.loads(out)
        assert model["full_id"] == "anthropic/claude-sonnet-4-6"
        assert model["name"] == "Claude Sonnet 4.6"

    def test_getで存在しないモデルはエラー(self, capsys):
        args = ["get", "nonexistent/model-xyz", "--json"]

        _, _, exit_code = _run_cli(args, capsys)

        assert exit_code != 0

    def test_getでプロバイダ省略時にDid_you_meanが表示される(self, capsys):
        args = ["get", "claude-sonnet-4-6", "--json"]

        _, err, exit_code = _run_cli(args, capsys)

        assert exit_code != 0
        assert "Did you mean?" in err


class TestListSubcommand:
    def test_listサブコマンドでモデル一覧(self, capsys):
        # deprecated is excluded by default, so 4 active models
        args = ["list", "--all", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 4

    def test_listでcapフィルタ(self, capsys):
        args = ["list", "--cap", "reasoning", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 1
        assert models[0]["full_id"] == "anthropic/claude-opus-4-6"

    def test_listでmin_contextフィルタ(self, capsys):
        args = ["list", "--min-context", "200k", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2
        assert all(m["provider_id"] == "anthropic" for m in models)

    def test_listでmax_input_costフィルタ(self, capsys):
        args = ["list", "--all", "--max-input-cost", "3.0", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        # sonnet(3.0), gpt-4o(2.5), llama(0.6) => 3 models (deprecated excluded)
        assert len(models) == 3
        full_ids = {m["full_id"] for m in models}
        assert "anthropic/claude-opus-4-6" not in full_ids

    def test_listでsort(self, capsys):
        args = ["list", "--sort", "cost.input", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        costs = [m["cost"]["input"] for m in models]
        assert costs == sorted(costs)

    def test_listでlimit(self, capsys):
        args = ["list", "--limit", "2", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2

    def test_listでcap_modeオプションがOR指定で動く(self, capsys):
        args = [
            "list",
            "--all",
            "--cap",
            "reasoning",
            "--cap",
            "open_weights",
            "--cap-mode",
            "or",
            "--json",
        ]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2
        full_ids = {m["full_id"] for m in models}
        assert "anthropic/claude-opus-4-6" in full_ids
        assert "meta/llama-3.3-70b" in full_ids

    def test_listで_sort_fieldsオプションでフィールド一覧が表示される(self, capsys):
        args = ["list", "--sort-fields"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        assert "cost.input" in out
        assert "limit.context" in out
        assert "Available sort fields:" in out

    def test_listで_sort_fieldsは他のフィルタより先に処理される(self, capsys):
        args = ["list", "--sort-fields"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        assert "anthropic/claude-sonnet-4-6" not in out
        assert "Available sort fields:" in out

    def test_listでdeprecatedモデルがデフォルトで除外される(self, capsys):
        args = ["list", "--all", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        full_ids = {m["full_id"] for m in models}
        assert "openai/gpt-4-turbo" not in full_ids

    def test_listでinclude_deprecatedフラグでdeprecatedモデルが含まれる(self, capsys):
        args = ["list", "--all", "--include-deprecated", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 5
        full_ids = {m["full_id"] for m in models}
        assert "openai/gpt-4-turbo" in full_ids


class TestSearchSubcommand:
    def test_searchサブコマンド(self, capsys):
        args = ["search", "claude", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) == 2
        assert all("claude" in m["full_id"] for m in models)

    def test_searchでJSON出力にマッチ情報が含まれる(self, capsys):
        args = ["search", "claude", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        models = json.loads(out)
        assert len(models) >= 1
        for model in models:
            assert "_match_score" in model
            assert "_matched_fields" in model
            assert isinstance(model["_match_score"], int)
            assert isinstance(model["_matched_fields"], list)


class TestProvidersSubcommand:
    def test_providersサブコマンド(self, capsys):
        args = ["providers", "--json"]

        out, _, exit_code = _run_cli(args, capsys)

        assert exit_code is None or exit_code == 0
        providers = json.loads(out)
        assert len(providers) == 3
        provider_ids = {p["id"] for p in providers}
        assert "anthropic" in provider_ids
        assert "openai" in provider_ids
        assert "meta" in provider_ids


class TestBackwardCompatibility:
    def test_サブコマンドなしは後方互換(self, mocker, capsys):
        mocker.patch("llms.cli.get_preview_command", return_value="echo")
        mock_run_fzf = mocker.patch(
            "llms.cli.run_fzf",
            side_effect=Exception("fzf not available in test"),
        )

        sys.argv = ["llms"]
        with pytest.raises(Exception, match="fzf not available in test"):
            main()

        assert mock_run_fzf.called
