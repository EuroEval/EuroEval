"""Unit tests for the `litellm` module."""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Choices

from euroeval.benchmark_modules.litellm import (
    LiteLLMModel,
    clean_model_id,
    get_api_model_release_date,
)
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.exceptions import InvalidModel
from euroeval.model_loading import load_model


class TestBPCGating:
    """Tests that BPC scoring is rejected for LiteLLM backend.

    BPC validation happens in load_model() before backend initialization.
    """

    def test_bpc_rejected_for_litellm(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """BPC scoring raises InvalidModel for LiteLLM backend."""
        bpc_config = dataclasses.replace(benchmark_config, use_bits_per_character=True)
        with pytest.raises(InvalidModel, match="vLLM backend"):
            load_model(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=bpc_config,
            )


class TestCreateModelOutput:
    """Tests for the _create_model_output method in LiteLLMModel."""

    def test_empty_choices_appends_empty_scores(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Test that responses with no choices get empty score lists.

        This is a regression test for a bug where evaluating on AngryTweets raised
        "Sequences and scores must have the same length. Got 1320 sequences and 1319
        scores" when the model returned no choices for some samples. The bug was in
        _create_model_output which appended an empty string to sequences but skipped
        appending to scores when choices were empty.

        The test mocks logprobs as a list of dicts matching ChoiceLogprobs schema
        to trigger the scores append for the valid response.
        """
        # Create a mock response with valid choices and logprobs (as list fallback)
        mock_valid_response = MagicMock()
        mock_choice = MagicMock(spec=Choices)
        mock_message = MagicMock()
        mock_message.content = "positive"
        mock_choice.message = mock_message
        # Mock logprobs as list of dicts matching ChoiceLogprobs schema
        # Each dict needs a 'content' field with list of token logprobs
        mock_choice.logprobs = [
            {
                "content": [
                    {
                        "token": "positive",
                        "logprob": -0.5,
                        "bytes": [112, 111, 115, 105, 116, 105, 118, 101],
                        "top_logprobs": [],
                    }
                ]
            }
        ]
        mock_valid_response.choices = [mock_choice]

        # Create a mock response with empty choices (model ran out of tokens)
        mock_empty_response = MagicMock()
        mock_empty_response.choices = []

        # Create the LiteLLMModel instance
        model = LiteLLMModel(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        # This should NOT raise InvalidBenchmark about length mismatch.
        # Without the fix, this raises:
        # "Sequences and scores must have the same length. Got 2 sequences and 1
        # scores."
        output = model._create_model_output(
            model_responses=[mock_valid_response, mock_empty_response],
            model_id="test-model",
        )

        # Verify the output is valid - sequences and scores are aligned
        assert len(output.sequences) == 2
        assert output.sequences[0] == "positive"
        assert output.sequences[1] == ""
        # Scores should be non-None since at least one sample has logprobs
        assert output.scores is not None
        assert len(output.scores) == 2
        # First sample has logprobs, second sample (empty choices) has empty list
        assert output.scores[0] is not None
        assert len(output.scores[0]) == 1
        assert output.scores[1] == []


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openai/gpt-4o", "2024-05-13"),
        ("openai/gpt-4o-2024-08-06", "2024-08-06"),
        ("openai/gpt-4-0613", "2023-06-13"),
        ("openai/gpt-3.5-turbo-0125", "2024-01-25"),
        ("openai/o1-pro", "2025-04-14"),
        ("openai/gpt-5-chat-latest", "2025-08-07"),
        ("openai/gpt-5.2-pro", "2025-12-11"),
        ("openai/gpt-5.4-pro", "2026-03-05"),
        ("openai/gpt-5.5-pro", "2026-04-23"),
        ("anthropic/claude-3-5-sonnet-20241022", "2024-10-22"),
        ("anthropic/claude-sonnet-4-6", "2026-02-17"),
        ("anthropic/claude-opus-4-7", "2026-04-16"),
        ("anthropic/claude-mythos-preview", "2026-04-07"),
        ("anthropic/claude-fable-5", "2026-06-09"),
        ("anthropic/claude-sonnet-5", "2026-06-30"),
        ("anthropic/claude-opus-5", "2026-07-24"),
        ("gemini/gemini-3.1-pro-preview", "2026-02-19"),
        ("gemini/gemini-3.1-flash-lite", "2026-03-03"),
        ("gemini/gemini-3.5-flash", "2026-05-19"),
        ("gemini/gemini-3.6-flash", "2026-07-21"),
        ("gemini/gemini-3.7-flash", "2026-08-13"),
        ("xai/grok-4-fast-reasoning", "2025-09-19"),
        ("xai/grok-4.20", "2026-03-10"),
        ("xai/grok-4.5", "2026-07-08"),
        ("xai/grok-4.6", "2026-08-12"),
        ("openai/gpt-5.6-luna", "2026-07-09"),
        ("openai/gpt-5.6", "2026-07-09"),
        ("provider/undated-model", None),
        ("provider/model-2024-99-99", None),
    ],
)
def test_get_api_model_release_date(model_id: str, expected: str | None) -> None:
    """API aliases and dated model IDs resolve to their release dates."""
    assert get_api_model_release_date(model_id) == expected


def test_huggingface_model_id_is_preserved_for_custom_api(
    benchmark_config: BenchmarkConfig,
) -> None:
    """Hugging Face model IDs remain valid for custom OpenAI-compatible APIs."""
    benchmark_config = dataclasses.replace(
        benchmark_config, api_base="https://router.huggingface.co/featherless-ai/v1"
    )
    model_id = "huggingface/mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    assert (
        clean_model_id(model_id=model_id, benchmark_config=benchmark_config) == model_id
    )


def test_litellm_model_config_includes_release_date(
    benchmark_config: BenchmarkConfig,
) -> None:
    """API release metadata is propagated into the model configuration."""
    config = LiteLLMModel.get_model_config(
        model_id="openai/gpt-4o-2024-08-06", benchmark_config=benchmark_config
    )
    assert config.release_date == "2024-08-06"


def test_manual_api_release_date_overrides_embedded_date() -> None:
    """Curated annotations take precedence over dates parsed from model IDs."""
    with patch.dict(
        "euroeval.benchmark_modules.litellm.MODEL_RELEASE_DATE_MAPPING",
        {r"provider/model-2024-01-01": "2024-02-03"},
        clear=True,
    ):
        assert get_api_model_release_date("provider/model-2024-01-01") == "2024-02-03"
