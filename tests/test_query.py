from __future__ import annotations

from llms.query import (
    DEFAULT_PROVIDERS,
    Query,
    _get_nested,
    _parse_token_count,
    filter_models,
)

SAMPLE_MODELS = [
    {
        "provider_id": "anthropic",
        "provider_name": "Anthropic",
        "full_id": "anthropic/claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
        "family": "claude",
        "cost": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75},
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
    {
        "provider_id": "anthropic",
        "provider_name": "Anthropic",
        "full_id": "anthropic/claude-opus-4-6",
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
    {
        "provider_id": "openai",
        "provider_name": "OpenAI",
        "full_id": "openai/gpt-4o",
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
    {
        "provider_id": "meta",
        "provider_name": "Meta",
        "full_id": "meta/llama-3.3-70b",
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
]


class TestParseTokenCount:
    def test_parse_token_count_with_k_suffix(self):
        # Arrange
        value = "128k"

        # Act
        result = _parse_token_count(value)

        # Assert
        assert result == 128000

    def test_parse_token_count_with_m_suffix(self):
        # Arrange
        value = "1m"

        # Act
        result = _parse_token_count(value)

        # Assert
        assert result == 1000000

    def test_parse_token_count_with_plain_number(self):
        # Arrange
        value = "500"

        # Act
        result = _parse_token_count(value)

        # Assert
        assert result == 500


class TestGetNested:
    def test_get_nested_simple_path(self):
        # Arrange
        d = {"cost": {"input": 3.0}}

        # Act
        result = _get_nested(d, "cost.input")

        # Assert
        assert result == 3.0

    def test_get_nested_missing_path(self):
        # Arrange
        d = {"cost": {"output": 15.0}}

        # Act
        result = _get_nested(d, "cost.input")

        # Assert
        assert result is None


class TestFilterModels:
    def test_providerでフィルタできる(self):
        # Arrange
        query = Query(provider="anthropic")

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 2
        assert all(m["provider_id"] == "anthropic" for m in result)

    def test_テキスト検索でfull_idにマッチする(self):
        # Arrange
        query = Query(text="gpt-4o")

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        assert result[0]["full_id"] == "openai/gpt-4o"

    def test_テキスト検索でnameにマッチする(self):
        # Arrange
        query = Query(text="Llama", all_providers=True)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        assert result[0]["full_id"] == "meta/llama-3.3-70b"

    def test_ケイパビリティでフィルタできる(self):
        # Arrange
        query = Query(caps=["reasoning"])

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        assert result[0]["full_id"] == "anthropic/claude-opus-4-6"

    def test_複数ケイパビリティはAND条件(self):
        # Arrange
        # tool_call=True AND open_weights=True => only llama
        query = Query(caps=["tool_call", "open_weights"], all_providers=True)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        assert result[0]["full_id"] == "meta/llama-3.3-70b"

    def test_最小コンテキストでフィルタできる(self):
        # Arrange
        # context >= 200000 => anthropic models only
        query = Query(min_context=200000)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 2
        assert all(m["provider_id"] == "anthropic" for m in result)

    def test_最大入力コストでフィルタできる(self):
        # Arrange
        # input cost <= 3.0 => sonnet(3.0), gpt-4o(2.5), llama(0.6)
        query = Query(max_input_cost=3.0, all_providers=True)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 3
        full_ids = {m["full_id"] for m in result}
        assert "anthropic/claude-opus-4-6" not in full_ids

    def test_ソートできる(self):
        # Arrange
        query = Query(sort="cost.input")

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        costs = [m["cost"]["input"] for m in result]
        assert costs == sorted(costs)

    def test_件数制限できる(self):
        # Arrange
        query = Query(limit=2)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 2

    def test_複合フィルタが機能する(self):
        # Arrange
        # provider=anthropic, max_input_cost=5.0, sort=cost.input, limit=1
        query = Query(
            provider="anthropic",
            max_input_cost=5.0,
            sort="cost.input",
            limit=1,
        )

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        # cheapest anthropic model under $5 is sonnet at $3.0
        assert result[0]["full_id"] == "anthropic/claude-sonnet-4-6"


class TestDefaultProviderFilter:
    def test_デフォルトでは非主要プロバイダが除外される(self):
        # Arrange
        models_with_aggregator = [
            *SAMPLE_MODELS,
            {
                "provider_id": "amazon-bedrock",
                "provider_name": "Amazon Bedrock",
                "full_id": "amazon-bedrock/claude-sonnet-4-6",
                "name": "Claude Sonnet 4.6",
                "cost": {"input": 3.0, "output": 15.0},
                "limit": {"context": 200000, "output": 64000},
            },
            {
                "provider_id": "qiniu-ai",
                "provider_name": "Qiniu AI",
                "full_id": "qiniu-ai/claude-sonnet-4-6",
                "name": "Claude Sonnet 4.6",
                "cost": {"input": 3.0, "output": 15.0},
                "limit": {"context": 200000, "output": 64000},
            },
        ]
        query = Query()

        # Act
        result = filter_models(models_with_aggregator, query)

        # Assert
        provider_ids = {m["provider_id"] for m in result}
        assert "amazon-bedrock" not in provider_ids
        assert "qiniu-ai" not in provider_ids
        assert "anthropic" in provider_ids
        assert "openai" in provider_ids

    def test_all_providersで全プロバイダが含まれる(self):
        # Arrange
        models_with_aggregator = [
            *SAMPLE_MODELS,
            {
                "provider_id": "amazon-bedrock",
                "provider_name": "Amazon Bedrock",
                "full_id": "amazon-bedrock/claude-sonnet-4-6",
                "name": "Claude Sonnet 4.6",
                "cost": {"input": 3.0, "output": 15.0},
                "limit": {"context": 200000, "output": 64000},
            },
        ]
        query = Query(all_providers=True)

        # Act
        result = filter_models(models_with_aggregator, query)

        # Assert
        assert len(result) == len(models_with_aggregator)

    def test_provider指定時はデフォルトフィルタをスキップする(self):
        # Arrange
        models_with_aggregator = [
            *SAMPLE_MODELS,
            {
                "provider_id": "amazon-bedrock",
                "provider_name": "Amazon Bedrock",
                "full_id": "amazon-bedrock/claude-sonnet-4-6",
                "name": "Claude Sonnet 4.6",
                "cost": {"input": 3.0, "output": 15.0},
                "limit": {"context": 200000, "output": 64000},
            },
        ]
        query = Query(provider="amazon-bedrock")

        # Act
        result = filter_models(models_with_aggregator, query)

        # Assert
        assert len(result) == 1
        assert result[0]["provider_id"] == "amazon-bedrock"

    def test_DEFAULT_PROVIDERSに主要プロバイダが含まれる(self):
        # Assert
        for provider in ["anthropic", "openai", "google", "deepseek", "mistral"]:
            assert provider in DEFAULT_PROVIDERS
