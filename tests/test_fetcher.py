from llms.fetcher import flatten_models


class TestFlattenModels:
    def test_全プロバイダのモデルをフラットに展開する(self, sample_api_data):
        # Arrange & Act
        result = flatten_models(sample_api_data)

        # Assert
        assert len(result) == 3
        ids = [m["full_id"] for m in result]
        assert "anthropic/claude-opus-4-5-20251101" in ids
        assert "anthropic/claude-sonnet-4-6-20260301" in ids
        assert "openai/gpt-4o" in ids

    def test_プロバイダでフィルタできる(self, sample_api_data):
        # Arrange & Act
        result = flatten_models(sample_api_data, provider_filter="anthropic")

        # Assert
        assert len(result) == 2
        assert all(m["provider_id"] == "anthropic" for m in result)

    def test_存在しないプロバイダでフィルタすると空リスト(self, sample_api_data):
        # Arrange & Act
        result = flatten_models(sample_api_data, provider_filter="nonexistent")

        # Assert
        assert result == []

    def test_provider_nameが付与される(self, sample_api_data):
        # Arrange & Act
        result = flatten_models(sample_api_data)

        # Assert
        anthropic_model = next(m for m in result if m["provider_id"] == "anthropic")
        assert anthropic_model["provider_name"] == "Anthropic"

    def test_不正なデータはスキップされる(self):
        # Arrange
        data = {
            "valid": {
                "name": "Valid",
                "models": {
                    "m1": {"id": "m1", "name": "Model 1"},
                },
            },
            "invalid_string": "not a dict",
            "no_models": {"name": "No Models"},
        }

        # Act
        result = flatten_models(data)

        # Assert
        assert len(result) == 1
        assert result[0]["full_id"] == "valid/m1"
