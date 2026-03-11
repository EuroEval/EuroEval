"""Tests for the `speed_benchmark` module."""

import collections.abc as c
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from tqdm.auto import tqdm

from euroeval.benchmark_modules.base import BenchmarkModule
from euroeval.benchmark_modules.hf import HuggingFaceEncoderModel
from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.benchmark_modules.vllm import VLLMModel
from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs import SPEED_CONFIG
from euroeval.exceptions import InvalidBenchmark
from euroeval.model_config import get_model_config
from euroeval.speed_benchmark import benchmark_speed, benchmark_speed_single_iteration


@pytest.fixture(scope="module")
def model(
    encoder_model_id: str, benchmark_config: BenchmarkConfig
) -> Generator[BenchmarkModule, None, None]:
    """Yields a model."""
    yield HuggingFaceEncoderModel(
        model_config=get_model_config(
            model_id=encoder_model_id, benchmark_config=benchmark_config
        ),
        dataset_config=SPEED_CONFIG,
        benchmark_config=benchmark_config,
    )


class TestBenchmarkSpeed:
    """Tests for the `benchmark_speed` function."""

    @pytest.fixture(scope="class")
    def itr(self) -> Generator[tqdm, None, None]:
        """Yields an iterator with a progress bar."""
        yield tqdm(range(2))

    @pytest.fixture(scope="class")
    def scores(
        self, model: BenchmarkModule, benchmark_config: BenchmarkConfig
    ) -> Generator[c.Sequence[dict[str, float]], None, None]:
        """Yields the benchmark speed scores."""
        yield benchmark_speed(model=model, benchmark_config=benchmark_config)

    def test_scores_is_list(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores is a list."""
        assert isinstance(scores, list)

    def test_scores_contain_dicts(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores contain dicts."""
        assert all(isinstance(x, dict) for x in scores)

    def test_scores_dicts_keys(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores dicts have the correct keys."""
        assert all(set(x.keys()) == {"test_speed", "test_speed_short"} for x in scores)

    def test_scores_dicts_values_dtypes(self, scores: list[dict[str, float]]) -> None:
        """Tests that the scores dicts have the correct values dtypes."""
        assert all(
            all(isinstance(value, float) for value in x.values()) for x in scores
        )


class TestBenchmarkSpeedSingleIteration:
    """Tests for the `benchmark_speed_single_iteration` function."""

    @pytest.fixture
    def mock_model(self) -> Generator[MagicMock, None, None]:
        """Yields a mock model."""
        mock = MagicMock(spec=BenchmarkModule)
        yield mock

    @pytest.fixture
    def mock_benchmark_config(self) -> BenchmarkConfig:
        """Yields a mock benchmark config.

        Returns:
            A mock benchmark config.
        """
        return BenchmarkConfig(
            languages=[],
            datasets=[],
            finetuning_batch_size=1,
            raise_errors=False,
            cache_dir="/tmp/test_cache",
            progress_bar=False,
            api_key=None,
            trust_remote_code=False,
            device=None,
            verbose=False,
            clear_model_cache=False,
            evaluate_test_split=False,
            few_shot=False,
            num_iterations=1,
            save_results=False,
            api_base=None,
            api_version=None,
            gpu_memory_utilization=0.8,
            attention_backend=None,
            generative_type=None,
            requires_safetensors=False,
            download_only=False,
            force=False,
            debug=False,
            run_with_cli=True,
            max_context_length=None,
            vocabulary_size=None,
        )

    def test_benchmark_speed_single_iteration_vllm_model(
        self, mock_model: MagicMock, mock_benchmark_config: BenchmarkConfig
    ) -> None:
        """Test speed benchmark with VLLM model type."""
        mock_model.__class__ = VLLMModel

        with patch(
            "euroeval.speed_benchmark.pyinfer.InferenceReport"
        ) as mock_inference_report:
            mock_report = MagicMock()
            mock_report.run.return_value = {"Infer(p/sec)": 10.0}
            mock_inference_report.return_value = mock_report

            with patch("euroeval.speed_benchmark.AutoTokenizer") as mock_tokenizer:
                mock_tokenizer.from_pretrained.return_value = MagicMock(
                    return_value={"input_ids": [[1, 2, 3, 4, 5]]}
                )

                scores = benchmark_speed_single_iteration(model=mock_model, itr_idx=0)

        assert "test_speed" in scores
        assert "test_speed_short" in scores
        assert all(isinstance(v, float) for v in scores.values())

    def test_benchmark_speed_single_iteration_litellm_model(
        self, mock_model: MagicMock, mock_benchmark_config: BenchmarkConfig
    ) -> None:
        """Test speed benchmark with LiteLLM model type."""
        mock_model.__class__ = LiteLLMModel

        with patch(
            "euroeval.speed_benchmark.pyinfer.InferenceReport"
        ) as mock_inference_report:
            mock_report = MagicMock()
            mock_report.run.return_value = {"Infer(p/sec)": 10.0}
            mock_inference_report.return_value = mock_report

            with patch("euroeval.speed_benchmark.AutoTokenizer") as mock_tokenizer:
                mock_tokenizer.from_pretrained.return_value = MagicMock(
                    return_value={"input_ids": [[1, 2, 3, 4, 5]]}
                )

                scores = benchmark_speed_single_iteration(model=mock_model, itr_idx=0)

        assert "test_speed" in scores
        assert "test_speed_short" in scores
        assert all(isinstance(v, float) for v in scores.values())

    def test_benchmark_speed_single_iteration_huggingface_encoder_model(
        self, mock_model: MagicMock, mock_benchmark_config: BenchmarkConfig
    ) -> None:
        """Test speed benchmark with HuggingFace encoder model type."""
        mock_model.__class__ = HuggingFaceEncoderModel
        mock_model.get_tokeniser.return_value = MagicMock()
        mock_model.get_pytorch_module.return_value = MagicMock(device="cpu")

        with patch(
            "euroeval.speed_benchmark.pyinfer.InferenceReport"
        ) as mock_inference_report:
            mock_report = MagicMock()
            mock_report.run.return_value = {"Infer(p/sec)": 10.0}
            mock_inference_report.return_value = mock_report

            with patch("euroeval.speed_benchmark.AutoTokenizer") as mock_tokenizer:
                mock_tokenizer.from_pretrained.return_value = MagicMock(
                    return_value={"input_ids": [[1, 2, 3, 4, 5]]}
                )

                scores = benchmark_speed_single_iteration(model=mock_model, itr_idx=0)

        assert "test_speed" in scores
        assert "test_speed_short" in scores
        assert all(isinstance(v, float) for v in scores.values())

    def test_benchmark_speed_single_iteration_invalid_model_raises_error(
        self, mock_model: MagicMock
    ) -> None:
        """Test ValueError raised for unsupported model types."""
        # Create a mock that is not any of the supported types
        mock_model.__class__.__name__ = "InvalidModel"

        with pytest.raises(ValueError, match="Model type.*not supported"):
            benchmark_speed_single_iteration(model=mock_model, itr_idx=0)

    def test_benchmark_speed_single_iteration_cuda_oom_error(
        self, mock_model: MagicMock, mock_benchmark_config: BenchmarkConfig
    ) -> None:
        """Test InvalidBenchmark raised when CUDA OOM occurs."""
        mock_model.__class__ = VLLMModel

        with patch(
            "euroeval.speed_benchmark.pyinfer.InferenceReport"
        ) as mock_inference_report:
            mock_inference_report.side_effect = RuntimeError("CUDA out of memory")

            with pytest.raises(InvalidBenchmark, match="Speed benchmark failed"):
                benchmark_speed_single_iteration(model=mock_model, itr_idx=0)
