"""Unit tests for the `litellm` module."""

import typing as t
from unittest.mock import patch

import ollama
import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.constants import REASONING_MAX_TOKENS
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import GenerativeType, InferenceBackend, ModelType
from euroeval.languages import DANISH


@pytest.fixture()
def litellm_model_config() -> ModelConfig:
    """A minimal ModelConfig for a generic OpenAI-compatible LiteLLM model.

    Returns:
        A ModelConfig with LiteLLM inference backend and generative model type.
    """
    return ModelConfig(
        model_id="openai/test-model",
        revision="main",
        param=None,
        task="task",
        languages=[DANISH],
        merge=False,
        inference_backend=InferenceBackend.LITELLM,
        model_type=ModelType.GENERATIVE,
        fresh=True,
        model_cache_dir="cache_dir",
        adapter_base_model_id=None,
    )


def _make_model_response(reasoning_content: str | None = None) -> ModelResponse:
    """Build a minimal ModelResponse, optionally with reasoning_content.

    Returns:
        A ModelResponse containing a single assistant choice.
    """
    msg = Message(
        content="Test response", role="assistant", reasoning_content=reasoning_content
    )
    choice = Choices(message=msg, finish_reason="stop", index=0)
    return ModelResponse(choices=[choice], model="test-model", id="test-id")


def _make_litellm_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
    reasoning_content: str | None = None,
) -> LiteLLMModel:
    """Create a LiteLLMModel with mocked Ollama and API calls.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.
        reasoning_content:
            If provided, the mocked test response will include this reasoning_content.

    Returns:
        A LiteLLMModel instance with mocked dependencies.
    """
    response = _make_model_response(reasoning_content=reasoning_content)

    async def mock_generate_async(
        *args, **kwargs
    ) -> tuple[list[tuple[int, ModelResponse]], list[t.Any]]:
        return [(0, response)], []

    with patch.object(
        ollama, "show", return_value=ollama.ShowResponse(model_info=None)
    ):
        with patch.object(
            LiteLLMModel, "_generate_async", side_effect=mock_generate_async
        ):
            return LiteLLMModel(
                model_config=model_config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
                log_metadata=False,
            )


@pytest.fixture()
def litellm_model(
    litellm_model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> LiteLLMModel:
    """A LiteLLMModel instance with no reasoning content in the test response.

    Returns:
        A LiteLLMModel with generative_type == INSTRUCTION_TUNED.
    """
    return _make_litellm_model(
        model_config=litellm_model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
        reasoning_content=None,
    )


@pytest.fixture()
def litellm_model_reasoning(
    litellm_model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> LiteLLMModel:
    """A LiteLLMModel instance with reasoning content in the test response.

    Returns:
        A LiteLLMModel with generative_type == REASONING.
    """
    return _make_litellm_model(
        model_config=litellm_model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
        reasoning_content="I am thinking deeply...",
    )


class TestReasoningContentDetection:
    """Tests for automatic reasoning model detection via response reasoning_content."""

    def test_reasoning_content_sets_buffer(
        self, litellm_model_reasoning: LiteLLMModel
    ) -> None:
        """buffer["test_response"] is set after model creation with reasoning."""
        assert litellm_model_reasoning.buffer.get("test_response") is not None

    def test_reasoning_content_updates_generative_type(
        self, litellm_model_reasoning: LiteLLMModel
    ) -> None:
        """generative_type property returns REASONING after detection."""
        assert litellm_model_reasoning.generative_type == GenerativeType.REASONING

    def test_reasoning_content_uses_reasoning_max_tokens(
        self, litellm_model_reasoning: LiteLLMModel, dataset_config: DatasetConfig
    ) -> None:
        """Returned kwargs use REASONING_MAX_TOKENS when reasoning_content is found."""
        generation_kwargs = litellm_model_reasoning.get_generation_kwargs(
            dataset_config=dataset_config
        )
        assert generation_kwargs["max_completion_tokens"] == REASONING_MAX_TOKENS

    def test_reasoning_content_removes_response_format(
        self, litellm_model_reasoning: LiteLLMModel, dataset_config: DatasetConfig
    ) -> None:
        """response_format is absent from kwargs when reasoning_content is found."""
        generation_kwargs = litellm_model_reasoning.get_generation_kwargs(
            dataset_config=dataset_config
        )
        assert "response_format" not in generation_kwargs

    def test_no_reasoning_content_keeps_instruction_tuned(
        self, litellm_model: LiteLLMModel
    ) -> None:
        """generative_type stays INSTRUCTION_TUNED when reasoning_content is absent."""
        assert litellm_model.generative_type == GenerativeType.INSTRUCTION_TUNED

    def test_empty_reasoning_content_keeps_instruction_tuned(
        self,
        litellm_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """generative_type stays INSTRUCTION_TUNED when reasoning_content is empty."""
        model = _make_litellm_model(
            model_config=litellm_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            reasoning_content="",
        )
        assert model.generative_type == GenerativeType.INSTRUCTION_TUNED
