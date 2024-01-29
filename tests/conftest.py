"""General fixtures used throughout test modules."""

import os
import sys
from typing import Generator

import pytest
import torch
from scandeval.config import (
    BenchmarkConfig,
    DatasetConfig,
    DatasetTask,
    Language,
    MetricConfig,
    ModelConfig,
)
from scandeval.dataset_configs import MMLU_DA_CONFIG
from scandeval.dataset_tasks import SPEED
from scandeval.enums import ModelType


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


@pytest.fixture(scope="session")
def auth() -> Generator[str | bool, None, None]:
    """Yields the authentication token to the Hugging Face Hub."""
    # Get the authentication token to the Hugging Face Hub
    auth = os.environ.get("HUGGINGFACE_HUB_TOKEN", True)

    # Ensure that the token does not contain quotes or whitespace
    if isinstance(auth, str):
        auth = auth.strip(" \"'")

    yield auth


@pytest.fixture(scope="session")
def device() -> Generator[torch.device, None, None]:
    """Yields the device to use for the tests."""
    device = torch.device("cuda" if os.getenv("USE_CUDA", False) else "cpu")
    yield device


@pytest.fixture(scope="session")
def benchmark_config(
    language, dataset_task, auth, device
) -> Generator[BenchmarkConfig, None, None]:
    """Yields a benchmark configuration used in tests."""
    yield BenchmarkConfig(
        model_languages=[language],
        dataset_languages=[language],
        dataset_tasks=[dataset_task],
        framework=None,
        batch_size=32,
        raise_errors=False,
        cache_dir=".scandeval_cache",
        evaluate_train=False,
        token=auth,
        openai_api_key=None,
        progress_bar=False,
        save_results=True,
        device=device,
        verbose=False,
        trust_remote_code=True,
        load_in_4bit=None,
        use_flash_attention=False,
        clear_model_cache=False,
        only_validation_split=False,
        few_shot=True,
    )


@pytest.fixture(scope="session")
def metric_config() -> Generator[MetricConfig, None, None]:
    """Yields a metric configuration used in tests."""
    yield MetricConfig(
        name="metric_name",
        pretty_name="Metric name",
        huggingface_id="metric_id",
        results_key="metric_key",
    )


@pytest.fixture(scope="session")
def dataset_task() -> Generator[DatasetTask, None, None]:
    """Yields a dataset task used in tests."""
    yield SPEED


@pytest.fixture(scope="session")
def language():
    """Yields a language used in tests."""
    yield Language(code="language_code", name="Language name")


@pytest.fixture(scope="session")
def model_id() -> Generator[str, None, None]:
    """Yields a model ID used in tests."""
    yield "jonfd/electra-small-nordic"


@pytest.fixture(scope="session")
def generative_model_id() -> Generator[str, None, None]:
    """Yields a generative model ID used in tests."""
    yield "AI-Sweden-Models/gpt-sw3-126m"


@pytest.fixture(scope="session")
def dataset_config(language, dataset_task) -> Generator[DatasetConfig, None, None]:
    """Yields a dataset configuration used in tests."""
    yield DatasetConfig(
        name="dataset_name",
        pretty_name="Dataset name",
        huggingface_id="dataset_id",
        task=dataset_task,
        languages=[language],
        prompt_template="{text}\n{label}",
        max_generated_tokens=1,
    )


@pytest.fixture(scope="session")
def model_config(language) -> Generator[ModelConfig, None, None]:
    """Yields a model configuration used in tests."""
    yield ModelConfig(
        model_id="model_id",
        revision="revision",
        framework="framework",
        task="task",
        languages=[language],
        model_type=ModelType.FRESH,
        model_cache_dir="cache_dir",
    )


@pytest.fixture(scope="session")
def generative_dataset_config() -> Generator[DatasetConfig, None, None]:
    """Yields a generative dataset configuration used in tests."""
    yield MMLU_DA_CONFIG
