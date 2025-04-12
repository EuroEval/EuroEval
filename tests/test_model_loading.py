"""Unit tests for the `model_loading` module."""

import pytest
import torch

from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs import get_dataset_config
from euroeval.exceptions import InvalidBenchmark
from euroeval.model_config import get_model_config
from euroeval.model_loading import load_model


def test_load_non_generative_model(
    encoder_model_id: str, benchmark_config: BenchmarkConfig
) -> None:
    """Test loading a non-generative model."""
    model_config = get_model_config(
        model_id=encoder_model_id, benchmark_config=benchmark_config
    )
    model = load_model(
        model_config=model_config,
        dataset_config=get_dataset_config("angry-tweets"),
        benchmark_config=benchmark_config,
    )
    assert model is not None


@pytest.mark.skipif(
    condition=not torch.cuda.is_available(), reason="CUDA is not available."
)
def test_load_generative_model(
    generative_model_id: str, benchmark_config: BenchmarkConfig
) -> None:
    """Test loading a generative model."""
    model_config = get_model_config(
        model_id=generative_model_id, benchmark_config=benchmark_config
    )
    model = load_model(
        model_config=model_config,
        dataset_config=get_dataset_config("angry-tweets"),
        benchmark_config=benchmark_config,
    )
    assert model is not None


def test_load_non_generative_model_with_generative_data(
    encoder_model_id: str, benchmark_config: BenchmarkConfig
) -> None:
    """Test loading a non-generative model with generative data."""
    model_config = get_model_config(
        model_id=encoder_model_id, benchmark_config=benchmark_config
    )
    with pytest.raises(InvalidBenchmark):
        load_model(
            model_config=model_config,
            dataset_config=get_dataset_config("nordjylland-news"),
            benchmark_config=benchmark_config,
        )
