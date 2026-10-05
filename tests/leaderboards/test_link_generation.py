"""Tests for model link generation."""

import pytest

from leaderboards import link_generation
from leaderboards.link_generation import (
    generate_deepseek_url,
    generate_model_url,
    generate_typesafe_url,
)


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
        "deepseek/deepseek-flash",
        "deepseek/custom-model",
    ],
)
def test_deepseek_provider_models_use_pricing_url(model_id: str) -> None:
    """Every model with the DeepSeek provider prefix uses its pricing URL."""
    expected_url = "https://api-docs.deepseek.com/quick_start/pricing/"

    assert generate_deepseek_url(model_id=model_id) == expected_url
    assert generate_model_url(model_id=model_id) == expected_url


def test_deepseek_url_overrides_stale_missing_url_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale missing-URL decision does not override a supported provider."""
    monkeypatch.setattr(link_generation, "_load_model_url_decision", lambda **_: False)

    assert (
        generate_model_url(model_id="deepseek/stale-cache-model")
        == "https://api-docs.deepseek.com/quick_start/pricing/"
    )


@pytest.mark.parametrize(
    "model_id", ["deepseek-chat", "deepseek-ai/DeepSeek-V3", "openrouter/deepseek-chat"]
)
def test_non_provider_deepseek_models_have_no_deepseek_url(model_id: str) -> None:
    """Model names without the DeepSeek provider prefix are not API models."""
    assert generate_deepseek_url(model_id=model_id) is None


@pytest.mark.parametrize("model_id", ["jev", "other/typesafe/jev"])
def test_non_provider_typesafe_models_have_no_typesafe_url(model_id: str) -> None:
    """Only IDs with a Typesafe provider prefix use the Typesafe link."""
    assert generate_typesafe_url(model_id=model_id) is None


@pytest.mark.parametrize("model_id", ["typesafe/jev", "typesafe/another-model"])
def test_typesafe_provider_models_use_docs_url(model_id: str) -> None:
    """Every Typesafe model links to the provider's model documentation."""
    expected_url = "https://docs.typesafe.ai/models"

    assert generate_typesafe_url(model_id=model_id) == expected_url
    assert generate_model_url(model_id=model_id) == expected_url
