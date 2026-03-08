"""Tests for the `model_loading` module."""

import os
import sys
from pathlib import Path
from shutil import rmtree

import pytest
import torch

from euroeval.data_models import BenchmarkConfig
from euroeval.dataset_configs import get_all_dataset_configs
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
        dataset_config=get_all_dataset_configs(
            custom_datasets_file=Path("custom_datasets.py"),
            dataset_ids=[],
            api_key=os.getenv("HF_TOKEN"),
            cache_dir=Path(".euroeval_cache"),
            trust_remote_code=True,
            run_with_cli=True,
        )["multi-wiki-qa-da"],
        benchmark_config=benchmark_config,
    )
    assert model is not None
    rmtree(path=Path(benchmark_config.cache_dir, "model_cache"), ignore_errors=True)


@pytest.mark.skipif(
    condition=sys.platform == "linux" and not torch.cuda.is_available(),
    reason="Running on Ubuntu but no CUDA available",
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
        dataset_config=get_all_dataset_configs(
            custom_datasets_file=Path("custom_datasets.py"),
            dataset_ids=[],
            api_key=os.getenv("HF_TOKEN"),
            cache_dir=Path(".euroeval_cache"),
            trust_remote_code=True,
            run_with_cli=True,
        )["multi-wiki-qa-da"],
        benchmark_config=benchmark_config,
    )
    assert model is not None
    rmtree(path=Path(benchmark_config.cache_dir, "model_cache"), ignore_errors=True)


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
            dataset_config=get_all_dataset_configs(
                custom_datasets_file=Path("custom_datasets.py"),
                dataset_ids=[],
                api_key=os.getenv("HF_TOKEN"),
                cache_dir=Path(".euroeval_cache"),
                trust_remote_code=True,
                run_with_cli=True,
            )["nordjylland-news"],
            benchmark_config=benchmark_config,
        )
    rmtree(path=Path(benchmark_config.cache_dir, "model_cache"), ignore_errors=True)
