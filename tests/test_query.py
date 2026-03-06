from __future__ import annotations

from llms.query import (
    DEFAULT_PROVIDERS,
    SORTABLE_FIELDS,
    Query,
    _get_nested,
    _parse_token_count,
    filter_models,
    search_models,
)

SAMPLE_MODELS = [
    {
        "provider_id": "openai",
        "provider_name": "OpenAI",
        "full_id": "openai/gpt-4-turbo",
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

    def test_複数ケイパビリティをOR条件でフィルタできる(self):
        # Arrange
        query = Query(
            caps=["reasoning", "open_weights"], cap_mode="or", all_providers=True
        )

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 2
        full_ids = {m["full_id"] for m in result}
        assert "anthropic/claude-opus-4-6" in full_ids
        assert "meta/llama-3.3-70b" in full_ids

    def test_capモードのデフォルトはAND(self):
        # Assert
        query = Query()
        assert query.cap_mode == "and"

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
        assert result[0]["full_id"] == "anthropic/claude-sonnet-4-6"


class TestDeprecatedFilter:
    def test_deprecatedモデルがデフォルトで除外される(self):
        # Arrange
        query = Query(all_providers=True)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        full_ids = {m["full_id"] for m in result}
        assert "openai/gpt-4-turbo" not in full_ids

    def test_include_deprecatedでdeprecatedモデルが含まれる(self):
        # Arrange
        query = Query(all_providers=True, include_deprecated=True)

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        full_ids = {m["full_id"] for m in result}
        assert "openai/gpt-4-turbo" in full_ids

    def test_statusフィールドがないモデルは除外されない(self):
        # Arrange
        query = Query(provider="anthropic")

        # Act
        result = filter_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 2
        assert all(m["provider_id"] == "anthropic" for m in result)


class TestSearchModels:
    def test_search_modelsでマッチスコアが付与される(self):
        # Arrange
        query = Query(text="gpt-4o", all_providers=True)

        # Act
        result = search_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        assert "_match_score" in result[0]
        assert "_matched_fields" in result[0]
        assert isinstance(result[0]["_match_score"], int)
        assert isinstance(result[0]["_matched_fields"], list)

    def test_search_modelsでexact_matchが最高スコア(self):
        # Arrange
        query = Query(text="openai/gpt-4o", all_providers=True)

        # Act
        result = search_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 1
        assert result[0]["full_id"] == "openai/gpt-4o"
        assert result[0]["_match_score"] == 100

    def test_search_modelsで複数フィールドマッチ(self):
        # Arrange
        query = Query(text="claude", all_providers=True)

        # Act
        result = search_models(SAMPLE_MODELS, query)

        # Assert
        assert len(result) == 2
        for m in result:
            assert "_matched_fields" in m
            assert len(m["_matched_fields"]) >= 2

    def test_search_modelsでスコア順にソートされる(self):
        # Arrange
        models = [
            {
                "provider_id": "anthropic",
                "full_id": "anthropic/claude-sonnet",
                "name": "claude-sonnet",
                "family": "claude",
                "id": "claude-sonnet",
            },
            {
                "provider_id": "anthropic",
                "full_id": "anthropic/claude-sonnet-extra",
                "name": "Claude Sonnet Extra (contains claude-sonnet)",
                "family": "claude",
                "id": "claude-sonnet-extra",
            },
        ]
        query = Query(text="claude-sonnet", all_providers=True)

        # Act
        result = search_models(models, query)

        # Assert
        assert len(result) == 2
        assert result[0]["_match_score"] >= result[1]["_match_score"]
        assert result[0]["full_id"] == "anthropic/claude-sonnet"


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
        query = Query(all_providers=True, include_deprecated=True)

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


class TestSortableFields:
    def test_SORTABLE_FIELDSに主要フィールドが含まれる(self):
        # Arrange
        field_names = [name for name, _ in SORTABLE_FIELDS]

        # Assert
        assert "cost.input" in field_names
        assert "cost.output" in field_names
        assert "limit.context" in field_names
        assert "limit.output" in field_names
        assert "name" in field_names
        assert "release_date" in field_names

    def test_SORTABLE_FIELDSの各エントリはタプル形式(self):
        # Assert
        for entry in SORTABLE_FIELDS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            field_name, description = entry
            assert isinstance(field_name, str)
            assert isinstance(description, str)
            assert len(description) > 0
