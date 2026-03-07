"""Unit tests for the `litellm` module."""

from unittest.mock import patch

import litellm
import ollama
import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.constants import REASONING_MAX_TOKENS
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import GenerativeType, InferenceBackend, ModelType
from euroeval.languages import DANISH
from euroeval.tasks import SENT


@pytest.fixture()
def litellm_benchmark_config() -> BenchmarkConfig:
    """A minimal BenchmarkConfig for LiteLLM tests."""
    return BenchmarkConfig(
        languages=[DANISH],
        datasets=[],
        finetuning_batch_size=1,
        raise_errors=False,
        cache_dir=".euroeval_cache",
        api_key=None,
        force=False,
        progress_bar=False,
        save_results=True,
        device="cpu",
        verbose=False,
        trust_remote_code=True,
        clear_model_cache=False,
        evaluate_test_split=False,
        few_shot=True,
        num_iterations=1,
        api_base=None,
        api_version=None,
        gpu_memory_utilization=0.8,
        attention_backend=None,
        generative_type=None,
        debug=False,
        run_with_cli=False,
        requires_safetensors=False,
        download_only=False,
        max_context_length=None,
        vocabulary_size=None,
    )


@pytest.fixture()
def litellm_model_config() -> ModelConfig:
    """A minimal ModelConfig for a generic OpenAI-compatible LiteLLM model."""
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


@pytest.fixture()
def litellm_dataset_config() -> DatasetConfig:
    """A minimal DatasetConfig for LiteLLM tests."""
    return DatasetConfig(
        name="dataset",
        pretty_name="Dataset",
        source="dataset_id",
        task=SENT,
        languages=[DANISH],
    )


@pytest.fixture()
def litellm_model(
    litellm_model_config: ModelConfig,
    litellm_dataset_config: DatasetConfig,
    litellm_benchmark_config: BenchmarkConfig,
) -> LiteLLMModel:
    """A LiteLLMModel instance with Ollama calls mocked out."""
    with patch.object(
        ollama, "show", return_value=ollama.ShowResponse(model_info=None)
    ):
        return LiteLLMModel(
            model_config=litellm_model_config,
            dataset_config=litellm_dataset_config,
            benchmark_config=litellm_benchmark_config,
            log_metadata=False,
        )


def _make_model_response(reasoning_content: str | None = None) -> ModelResponse:
    """Build a minimal ModelResponse, optionally with reasoning_content."""
    msg = Message(
        content="Test response",
        role="assistant",
        reasoning_content=reasoning_content,
    )
    choice = Choices(message=msg, finish_reason="stop", index=0)
    return ModelResponse(choices=[choice], model="test-model", id="test-id")


class TestReasoningContentDetection:
    """Tests for automatic reasoning model detection via response reasoning_content."""

    def test_reasoning_content_sets_buffer(
        self,
        litellm_model: LiteLLMModel,
        litellm_dataset_config: DatasetConfig,
    ) -> None:
        """buffer["response_based_generative_type"] is set when reasoning_content is
        non-empty."""
        response = _make_model_response(reasoning_content="I am thinking deeply...")

        async def mock_generate_async(*args, **kwargs):
            return [(0, response)], []

        with patch.object(
            litellm_model, "_generate_async", side_effect=mock_generate_async
        ):
            litellm_model.get_generation_kwargs(dataset_config=litellm_dataset_config)

        assert (
            litellm_model.buffer.get("response_based_generative_type")
            == GenerativeType.REASONING
        )

    def test_reasoning_content_updates_generative_type(
        self,
        litellm_model: LiteLLMModel,
        litellm_dataset_config: DatasetConfig,
    ) -> None:
        """generative_type property returns REASONING after detection."""
        assert litellm_model.generative_type == GenerativeType.INSTRUCTION_TUNED

        response = _make_model_response(reasoning_content="I am thinking deeply...")

        async def mock_generate_async(*args, **kwargs):
            return [(0, response)], []

        with patch.object(
            litellm_model, "_generate_async", side_effect=mock_generate_async
        ):
            litellm_model.get_generation_kwargs(dataset_config=litellm_dataset_config)

        assert litellm_model.generative_type == GenerativeType.REASONING

    def test_reasoning_content_uses_reasoning_max_tokens(
        self,
        litellm_model: LiteLLMModel,
        litellm_dataset_config: DatasetConfig,
    ) -> None:
        """Returned kwargs use REASONING_MAX_TOKENS when reasoning_content is found."""
        response = _make_model_response(reasoning_content="I am thinking deeply...")

        async def mock_generate_async(*args, **kwargs):
            return [(0, response)], []

        with patch.object(
            litellm_model, "_generate_async", side_effect=mock_generate_async
        ):
            generation_kwargs = litellm_model.get_generation_kwargs(
                dataset_config=litellm_dataset_config
            )

        assert generation_kwargs["max_completion_tokens"] == REASONING_MAX_TOKENS

    def test_reasoning_content_removes_response_format(
        self,
        litellm_model: LiteLLMModel,
        litellm_dataset_config: DatasetConfig,
    ) -> None:
        """response_format is removed from kwargs when reasoning_content is found."""
        response = _make_model_response(reasoning_content="I am thinking deeply...")

        async def mock_generate_async(*args, **kwargs):
            return [(0, response)], []

        # Inject a response_format into the buffer so it ends up in the initial kwargs
        litellm_model.buffer["first_label_token_mapping"] = False

        with patch.object(
            litellm_model, "_generate_async", side_effect=mock_generate_async
        ):
            generation_kwargs = litellm_model.get_generation_kwargs(
                dataset_config=litellm_dataset_config
            )

        assert "response_format" not in generation_kwargs

    def test_no_reasoning_content_keeps_instruction_tuned(
        self,
        litellm_model: LiteLLMModel,
        litellm_dataset_config: DatasetConfig,
    ) -> None:
        """generative_type stays INSTRUCTION_TUNED when reasoning_content is absent."""
        response = _make_model_response(reasoning_content=None)

        async def mock_generate_async(*args, **kwargs):
            return [(0, response)], []

        with patch.object(
            litellm_model, "_generate_async", side_effect=mock_generate_async
        ):
            litellm_model.get_generation_kwargs(dataset_config=litellm_dataset_config)

        assert litellm_model.buffer.get("response_based_generative_type") is None
        assert litellm_model.generative_type == GenerativeType.INSTRUCTION_TUNED

    def test_empty_reasoning_content_keeps_instruction_tuned(
        self,
        litellm_model: LiteLLMModel,
        litellm_dataset_config: DatasetConfig,
    ) -> None:
        """generative_type stays INSTRUCTION_TUNED when reasoning_content is empty."""
        response = _make_model_response(reasoning_content="")

        async def mock_generate_async(*args, **kwargs):
            return [(0, response)], []

        with patch.object(
            litellm_model, "_generate_async", side_effect=mock_generate_async
        ):
            litellm_model.get_generation_kwargs(dataset_config=litellm_dataset_config)

        assert litellm_model.buffer.get("response_based_generative_type") is None
        assert litellm_model.generative_type == GenerativeType.INSTRUCTION_TUNED
