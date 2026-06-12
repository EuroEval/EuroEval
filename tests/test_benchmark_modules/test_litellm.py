"""Unit tests for the `litellm` module."""

from unittest.mock import MagicMock

import pytest
from litellm.types.utils import Choices

from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig


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

        # This should NOT raise InvalidBenchmark about length mismatch
        # Without the fix, this raises:
        # "Sequences and scores must have the same length. Got 2 sequences and 1 scores."
        output = model._create_model_output(
            model_responses=[mock_valid_response, mock_empty_response],
            model_id="test-model",
        )

        # Verify the output is valid
        assert len(output.sequences) == 2
        assert output.sequences[0] == "positive"
        assert output.sequences[1] == ""
        assert output.scores is not None
        assert len(output.scores) == 2
