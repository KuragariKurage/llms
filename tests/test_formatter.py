from llms.formatter import (
    _format_cost,
    _format_tokens,
    format_comparison,
    format_fzf_line,
    format_preview,
)


class TestFormatCost:
    def test_通常の値をドル表記にする(self):
        assert _format_cost(5.0) == "$5.00"
        assert _format_cost(0.5) == "$0.50"

    def test_Noneはハイフン(self):
        assert _format_cost(None) == "-"

    def test_ゼロも表示する(self):
        assert _format_cost(0) == "$0.00"


class TestFormatTokens:
    def test_100万以上はM表記(self):
        assert _format_tokens(1_000_000) == "1M"
        assert _format_tokens(2_000_000) == "2M"

    def test_1000以上はK表記(self):
        assert _format_tokens(128_000) == "128K"
        assert _format_tokens(200_000) == "200K"

    def test_1000未満はそのまま(self):
        assert _format_tokens(512) == "512"

    def test_Noneはハイフン(self):
        assert _format_tokens(None) == "-"


class TestFormatFzfLine:
    def test_full_idのみ返す(self, sample_api_data):
        # Arrange
        from llms.fetcher import flatten_models

        models = flatten_models(sample_api_data)
        model = next(
            m for m in models if m["full_id"] == "anthropic/claude-opus-4-5-20251101"
        )

        # Act
        line = format_fzf_line(model)

        # Assert
        assert line == "anthropic/claude-opus-4-5-20251101"


class TestFormatPreview:
    def test_プレビューにモデル情報が含まれる(self, sample_api_data):
        # Arrange
        from llms.fetcher import flatten_models

        models = flatten_models(sample_api_data)
        model = next(
            m for m in models if m["full_id"] == "anthropic/claude-opus-4-5-20251101"
        )

        # Act
        preview = format_preview(model)

        # Assert
        assert "Anthropic" in preview
        assert "anthropic/claude-opus-4-5-20251101" in preview
        assert "$5.00" in preview
        assert "$25.00" in preview
        assert "200K" in preview
        assert "claude-opus" in preview
        assert "2025-03-31" in preview


class TestFormatComparison:
    def test_比較出力に両モデルの情報が含まれる(self, sample_api_data):
        from llms.fetcher import flatten_models

        models = flatten_models(sample_api_data)
        model_a = next(
            m for m in models if m["full_id"] == "anthropic/claude-opus-4-5-20251101"
        )
        model_b = next(m for m in models if m["full_id"] == "openai/gpt-4o")

        result = format_comparison(model_a, model_b)

        assert "Claude Opus 4.5" in result
        assert "GPT-4o" in result
        assert "Anthropic" in result
        assert "OpenAI" in result
        assert "$5.00" in result
        assert "$2.50" in result
        assert "200K" in result
        assert "128K" in result

    def test_比較でCapabilitiesの差分が表示される(self, sample_api_data):
        from llms.fetcher import flatten_models

        models = flatten_models(sample_api_data)
        model_a = next(
            m for m in models if m["full_id"] == "anthropic/claude-opus-4-5-20251101"
        )
        model_b = next(m for m in models if m["full_id"] == "openai/gpt-4o")

        result = format_comparison(model_a, model_b)

        assert "Reasoning:" in result
        assert "Tool Call:" in result

    def test_同じモデル同士の比較も動作する(self, sample_api_data):
        from llms.fetcher import flatten_models

        models = flatten_models(sample_api_data)
        model = next(
            m for m in models if m["full_id"] == "anthropic/claude-opus-4-5-20251101"
        )

        result = format_comparison(model, model)

        assert "Claude Opus 4.5" in result
