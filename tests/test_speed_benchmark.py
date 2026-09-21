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

    def test_scores_shape(self, scores: list[dict[str, float]]) -> None:
        """The benchmark returns a list of complete, numeric score dictionaries."""
        assert isinstance(scores, list)
        assert all(
            isinstance(score, dict)
            and set(score) == {"test_speed", "test_speed_short"}
            and all(isinstance(value, float) for value in score.values())
            for score in scores
        )


class TestBenchmarkSpeedSingleIteration:
    """Tests for the `benchmark_speed_single_iteration` function."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Return a mock model."""
        return MagicMock(spec=BenchmarkModule)

    def test_benchmark_speed_single_iteration_cuda_oom_error(
        self, mock_model: MagicMock
    ) -> None:
        """Test InvalidBenchmark raised when CUDA OOM occurs."""
        mock_model.__class__ = VLLMModel

        with patch(
            "euroeval.speed_benchmark.pyinfer.InferenceReport"
        ) as mock_inference_report:
            mock_inference_report.side_effect = RuntimeError("CUDA out of memory")

            with pytest.raises(InvalidBenchmark, match="Speed benchmark failed"):
                benchmark_speed_single_iteration(model=mock_model, itr_idx=0)

    @pytest.mark.parametrize(
        "model_type",
        [HuggingFaceEncoderModel, LiteLLMModel, VLLMModel],
        ids=["HuggingFace encoder", "LiteLLM", "VLLM"],
    )
    def test_benchmark_speed_single_iteration_supported_models(
        self, mock_model: MagicMock, model_type: type[BenchmarkModule]
    ) -> None:
        """Supported model types return the standard speed scores."""
        mock_model.__class__ = model_type
        if model_type is HuggingFaceEncoderModel:
            mock_model.get_tokeniser.return_value = MagicMock()
            mock_model.get_pytorch_module.return_value = MagicMock(device="cpu")

        with (
            patch("euroeval.speed_benchmark.pyinfer.InferenceReport") as mock_report,
            patch("euroeval.speed_benchmark.AutoTokenizer") as mock_tokenizer,
        ):
            mock_report.return_value.run.return_value = {"Infer(p/sec)": 10.0}
            mock_tokenizer.from_pretrained.return_value = MagicMock(
                return_value={"input_ids": [[1, 2, 3, 4, 5]]}
            )
            scores = benchmark_speed_single_iteration(model=mock_model, itr_idx=0)

        assert set(scores) == {"test_speed", "test_speed_short"}
        assert all(isinstance(value, float) for value in scores.values())

    def test_benchmark_speed_single_iteration_invalid_model_raises_error(
        self, mock_model: MagicMock
    ) -> None:
        """Test ValueError raised for unsupported model types."""
        # Create a mock that is not any of the supported types
        mock_model.__class__.__name__ = "InvalidModel"

        with pytest.raises(ValueError, match="Model type.*not supported"):
            benchmark_speed_single_iteration(model=mock_model, itr_idx=0)


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
