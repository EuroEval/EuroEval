"""Unit tests for the `model_loading` module."""

import pytest
import torch
from scandeval.config import ModelConfig
from scandeval.enums import Framework
from scandeval.exceptions import InvalidBenchmark, InvalidModel
from scandeval.languages import DA
from scandeval.model_config import get_model_config
from scandeval.model_loading import load_model


def test_load_non_generative_model(model_id, dataset_config, benchmark_config):
    """Test loading a non-generative model."""
    model_config = get_model_config(
        model_id=model_id, benchmark_config=benchmark_config
    )
    tokenizer, model = load_model(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )
    assert tokenizer is not None
    assert model is not None


def test_load_generative_model(
    generative_model_id, generative_dataset_config, benchmark_config
):
    """Test loading a generative model."""
    model_config = get_model_config(
        model_id=generative_model_id, benchmark_config=benchmark_config
    )
    tokenizer, model = load_model(
        model_config=model_config,
        dataset_config=generative_dataset_config,
        benchmark_config=benchmark_config,
    )
    assert tokenizer is not None
    assert model is not None


def test_load_non_generative_model_with_generative_data(
    model_id, generative_dataset_config, benchmark_config
):
    """Test loading a non-generative model with generative data."""
    model_config = get_model_config(
        model_id=model_id, benchmark_config=benchmark_config
    )
    with pytest.raises(InvalidBenchmark):
        load_model(
            model_config=model_config,
            dataset_config=generative_dataset_config,
            benchmark_config=benchmark_config,
        )


def test_load_generative_model_with_non_generative_data(
    generative_model_id, dataset_config, benchmark_config
):
    """Test loading a generative model with non-generative data."""
    model_config = get_model_config(
        model_id=generative_model_id, benchmark_config=benchmark_config
    )
    tokenizer, model = load_model(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )
    assert tokenizer is not None
    assert model is not None


def test_load_non_existing_model(dataset_config, benchmark_config):
    """Test loading a non-existing model."""
    model_config = ModelConfig(
        model_id="non-existing-model",
        revision="revision",
        framework=Framework.PYTORCH,
        task="task",
        languages=[DA],
        model_type="fresh",
        model_cache_dir="cache_dir",
    )
    with pytest.raises(InvalidModel):
        load_model(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="No GPU available.")
def test_load_awq_model(awq_generative_model_id, dataset_config, benchmark_config):
    """Test loading an AWQ quantised model."""
    if benchmark_config.device.type != "cuda":
        pytest.skip("Skipping test because the device is not a GPU.")
    model_config = get_model_config(
        model_id=awq_generative_model_id, benchmark_config=benchmark_config
    )
    load_model(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="No GPU available.")
def test_load_gptq_model(gptq_generative_model_id, dataset_config, benchmark_config):
    """Test loading a GPTQ quantised model."""
    if benchmark_config.device.type != "cuda":
        pytest.skip("Skipping test because the device is not a GPU.")
    model_config = get_model_config(
        model_id=gptq_generative_model_id, benchmark_config=benchmark_config
    )
    load_model(
        model_config=model_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )
