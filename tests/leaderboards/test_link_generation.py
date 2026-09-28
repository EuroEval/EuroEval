"""Tests for model link generation."""

import pytest

from leaderboards.link_generation import generate_deepseek_url, generate_model_url


@pytest.mark.parametrize(
    "model_id",
    ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner", "deepseek/custom-model"],
)
def test_deepseek_provider_models_use_pricing_url(model_id: str) -> None:
    """Every model with the DeepSeek provider prefix uses its pricing URL."""
    expected_url = "https://api-docs.deepseek.com/quick_start/pricing/"

    assert generate_deepseek_url(model_id=model_id) == expected_url
    assert generate_model_url(model_id=model_id) == expected_url


@pytest.mark.parametrize(
    "model_id", ["deepseek-chat", "deepseek-ai/DeepSeek-V3", "openrouter/deepseek-chat"]
)
def test_non_provider_deepseek_models_have_no_deepseek_url(model_id: str) -> None:
    """Model names without the DeepSeek provider prefix are not API models."""
    assert generate_deepseek_url(model_id=model_id) is None
