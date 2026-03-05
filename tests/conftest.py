import pytest

SAMPLE_API_DATA = {
    "anthropic": {
        "name": "Anthropic",
        "api": "https://api.anthropic.com/v1",
        "models": {
            "claude-opus-4-5-20251101": {
                "id": "claude-opus-4-5-20251101",
                "name": "Claude Opus 4.5",
                "family": "claude-opus",
                "reasoning": True,
                "tool_call": True,
                "attachment": True,
                "temperature": True,
                "open_weights": False,
                "cost": {"input": 5.0, "output": 25.0},
                "limit": {"context": 200000, "output": 64000},
                "modalities": {
                    "input": ["text", "image", "pdf"],
                    "output": ["text"],
                },
                "knowledge": "2025-03-31",
                "release_date": "2025-11-01",
            },
            "claude-sonnet-4-6-20260301": {
                "id": "claude-sonnet-4-6-20260301",
                "name": "Claude Sonnet 4.6",
                "family": "claude-sonnet",
                "reasoning": False,
                "tool_call": True,
                "attachment": True,
                "temperature": True,
                "open_weights": False,
                "cost": {"input": 3.0, "output": 15.0},
                "limit": {"context": 200000, "output": 64000},
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
            },
        },
    },
    "openai": {
        "name": "OpenAI",
        "models": {
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "reasoning": False,
                "tool_call": True,
                "attachment": False,
                "temperature": True,
                "open_weights": False,
                "cost": {"input": 2.5, "output": 10.0},
                "limit": {"context": 128000, "output": 16384},
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
            },
        },
    },
}


@pytest.fixture
def sample_api_data():
    return SAMPLE_API_DATA
