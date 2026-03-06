from __future__ import annotations

import pytest

from llms.client import Client, ModelNotFoundError
from llms.query import Query

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


class TestClient:
    def test_getで単一モデルを取得できる(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act
        model = client.get("anthropic/claude-sonnet-4-6")

        # Assert
        assert model["full_id"] == "anthropic/claude-sonnet-4-6"
        assert model["name"] == "Claude Sonnet 4.6"
        assert model["provider_id"] == "anthropic"

    def test_get_で存在しないモデルはModelNotFoundError(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act & Assert
        with pytest.raises(ModelNotFoundError):
            client.get("nonexistent/model-id")

    def test_listで全モデルを取得できる(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act
        models = client.list(Query(all_providers=True))

        # Assert
        assert len(models) == 4
        full_ids = {m["full_id"] for m in models}
        assert "anthropic/claude-sonnet-4-6" in full_ids
        assert "anthropic/claude-opus-4-6" in full_ids
        assert "openai/gpt-4o" in full_ids
        assert "meta/llama-3.3-70b" in full_ids

    def test_listでQueryフィルタが適用される(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()
        query = Query(provider="anthropic")

        # Act
        models = client.list(query)

        # Assert
        assert len(models) == 2
        assert all(m["provider_id"] == "anthropic" for m in models)

    def test_searchでテキスト検索できる(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act
        models = client.search("claude")

        # Assert
        assert len(models) == 2
        assert all("claude" in m["full_id"] for m in models)

    def test_providersでプロバイダ一覧を取得できる(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act
        providers = client.providers()

        # Assert
        assert len(providers) == 3
        provider_ids = {p["id"] for p in providers}
        assert "anthropic" in provider_ids
        assert "openai" in provider_ids
        assert "meta" in provider_ids
        # Verify structure
        for p in providers:
            assert "id" in p
            assert "name" in p


class TestFuzzyMatch:
    def test_getでプロバイダ省略時にサジェストが表示される(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act & Assert
        with pytest.raises(ModelNotFoundError) as exc_info:
            client.get("claude-sonnet-4-6")

        error = exc_info.value
        assert hasattr(error, "suggestions")
        assert len(error.suggestions) > 0
        assert "anthropic/claude-sonnet-4-6" in error.suggestions

    def test_getでサジェストが空の場合はシンプルなエラー(self, mocker):
        # Arrange
        mocker.patch("llms.client.fetch_models", return_value=SAMPLE_RAW_DATA)
        client = Client()

        # Act & Assert
        with pytest.raises(ModelNotFoundError) as exc_info:
            client.get("completely-unknown-xyz")

        error = exc_info.value
        assert hasattr(error, "suggestions")
        assert error.suggestions == []

    def test_find_similar_modelsでsuffix_matchが優先される(self, mocker):
        # Arrange
        raw_data = {
            "anthropic": {
                "name": "Anthropic",
                "models": {
                    "claude-sonnet-4-6": {
                        "name": "Claude Sonnet 4.6",
                        "family": "claude",
                        "cost": {"input": 3.0, "output": 15.0},
                        "limit": {"context": 200000, "output": 64000},
                        "reasoning": False,
                        "tool_call": True,
                        "attachment": True,
                        "temperature": True,
                        "open_weights": False,
                        "modalities": {"input": ["text"], "output": ["text"]},
                    },
                },
            },
            "amazon-bedrock": {
                "name": "Amazon Bedrock",
                "models": {
                    "claude-sonnet-4-6": {
                        "name": "Claude Sonnet 4.6 (Bedrock)",
                        "family": "claude",
                        "cost": {"input": 3.0, "output": 15.0},
                        "limit": {"context": 200000, "output": 64000},
                        "reasoning": False,
                        "tool_call": True,
                        "attachment": True,
                        "temperature": True,
                        "open_weights": False,
                        "modalities": {"input": ["text"], "output": ["text"]},
                    },
                    "sonnet-extra": {
                        "name": "Sonnet Extra",
                        "family": "claude",
                        "cost": {"input": 3.0, "output": 15.0},
                        "limit": {"context": 200000, "output": 64000},
                        "reasoning": False,
                        "tool_call": True,
                        "attachment": True,
                        "temperature": True,
                        "open_weights": False,
                        "modalities": {"input": ["text"], "output": ["text"]},
                    },
                },
            },
        }
        mocker.patch("llms.client.fetch_models", return_value=raw_data)
        client = Client()

        # Act
        suggestions = client.find_similar_models("claude-sonnet-4-6")

        # Assert
        assert "anthropic/claude-sonnet-4-6" in suggestions
        assert "amazon-bedrock/claude-sonnet-4-6" in suggestions
        assert "amazon-bedrock/sonnet-extra" not in suggestions
        assert suggestions == sorted(suggestions)
