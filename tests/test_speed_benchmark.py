"""Unit tests for the `speed_benchmark` module."""

from typing import Generator

import pytest
from tqdm.auto import tqdm

from euroeval.benchmark_modules.base import BenchmarkModule
from euroeval.benchmark_modules.hf import HuggingFaceEncoderModel
from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs import SPEED_CONFIG
from euroeval.model_config import get_model_config
from euroeval.speed_benchmark import benchmark_speed


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
    """Unit tests for the `benchmark_speed` function."""

    @pytest.fixture(scope="class")
    def itr(self) -> Generator[tqdm, None, None]:
        """Yields an iterator with a progress bar."""
        yield tqdm(range(2))

    @pytest.fixture(scope="class")
    def scores(
        self, model: BenchmarkModule, benchmark_config: BenchmarkConfig
    ) -> Generator[list[dict[str, float]], None, None]:
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
