"""General fixtures used throughout test modules."""

import os
import sys
from typing import Generator

import pytest
import torch

from scandeval.data_models import (
    BenchmarkConfig,
    DatasetConfig,
    MetricConfig,
    ModelConfig,
    Task,
)
from scandeval.dataset_configs import MMLU_CONFIG, SPEED_CONFIG, get_all_dataset_configs
from scandeval.enums import Framework, ModelType
from scandeval.languages import get_all_languages
from scandeval.tasks import get_all_tasks


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


ACTIVE_LANGUAGES = {
    language_code: language
    for language_code, language in get_all_languages().items()
    if any(
        language in cfg.languages
        for cfg in get_all_dataset_configs().values()
        if cfg != SPEED_CONFIG
    )
}


@pytest.fixture(scope="session")
def auth() -> Generator[str | bool, None, None]:
    """Yields the authentication token to the Hugging Face Hub."""
    # Get the authentication token to the Hugging Face Hub
    auth = os.environ.get("HUGGINGFACE_API_KEY", True)

    # Ensure that the token does not contain quotes or whitespace
    if isinstance(auth, str):
        auth = auth.strip(" \"'")

    yield auth


@pytest.fixture(scope="session")
def device() -> Generator[torch.device, None, None]:
    """Yields the device to use for the tests."""
    if os.getenv("USE_CUDA", "0") == "1":
        device = torch.device("cuda")
    elif os.getenv("USE_MPS", "0") == "1":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    yield device


@pytest.fixture(scope="session")
def benchmark_config(
    language, task, auth, device
) -> Generator[BenchmarkConfig, None, None]:
    """Yields a benchmark configuration used in tests."""
    yield BenchmarkConfig(
        model_languages=[language],
        dataset_languages=[language],
        tasks=[task],
        datasets=list(get_all_dataset_configs().keys()),
        framework=None,
        batch_size=32,
        raise_errors=False,
        cache_dir=".scandeval_cache",
        api_key=auth,
        force=False,
        progress_bar=False,
        save_results=True,
        device=device,
        verbose=False,
        trust_remote_code=True,
        load_in_4bit=None,
        use_flash_attention=False,
        clear_model_cache=False,
        evaluate_test_split=False,
        few_shot=True,
        num_iterations=10,
        debug=False,
        run_with_cli=True,
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


@pytest.fixture(
    scope="session",
    params=list(get_all_tasks().values()),
    ids=list(get_all_tasks().keys()),
)
def task(request) -> Generator[Task, None, None]:
    """Yields a dataset task used in tests."""
    yield request.param


@pytest.fixture(
    scope="session",
    params=list(ACTIVE_LANGUAGES.values()),
    ids=list(ACTIVE_LANGUAGES.keys()),
)
def language(request):
    """Yields a language used in tests."""
    yield request.param


@pytest.fixture(scope="session")
def encoder_model_id() -> Generator[str, None, None]:
    """Yields a model ID used in tests."""
    yield "jonfd/electra-small-nordic"


@pytest.fixture(scope="session")
def generative_model_id() -> Generator[str, None, None]:
    """Yields a generative model ID used in tests."""
    yield "mhenrichsen/danskgpt-tiny"


@pytest.fixture(scope="session")
def openai_model_id() -> Generator[str, None, None]:
    """Yields an OpenAI model ID used in tests."""
    yield "gpt-4o-mini"


@pytest.fixture(scope="session")
def anthropic_model_id() -> Generator[str, None, None]:
    """Yields an Anthropic model ID used in tests."""
    yield "claude-3-5-haiku-20241022"


@pytest.fixture(scope="session")
def awq_generative_model_id() -> Generator[str, None, None]:
    """Yields a generative model ID used in tests."""
    yield "casperhansen/opt-125m-awq"


@pytest.fixture(scope="session")
def gptq_generative_model_id() -> Generator[str, None, None]:
    """Yields a generative model ID used in tests."""
    yield "ybelkada/opt-125m-gptq-4bit"


@pytest.fixture(scope="session")
def dataset_config(language, task) -> Generator[DatasetConfig, None, None]:
    """Yields a dataset configuration used in tests."""
    yield DatasetConfig(
        name="dataset_name",
        pretty_name="Dataset name",
        huggingface_id="dataset_id",
        task=task,
        languages=[language],
        prompt_prefix="",
        prompt_template="{text}\n{label}",
        instruction_prompt="",
        num_few_shot_examples=0,
        max_generated_tokens=1,
    )


@pytest.fixture(scope="session")
def model_config(language) -> Generator[ModelConfig, None, None]:
    """Yields a model configuration used in tests."""
    yield ModelConfig(
        model_id="model_id",
        revision="revision",
        framework=Framework.PYTORCH,
        task="task",
        languages=[language],
        model_type=ModelType.FRESH,
        model_cache_dir="cache_dir",
        adapter_base_model_id=None,
    )


@pytest.fixture(scope="session")
def generative_dataset_config() -> Generator[DatasetConfig, None, None]:
    """Yields a generative dataset configuration used in tests."""
    yield MMLU_CONFIG


@pytest.fixture(scope="session")
def all_dataset_configs() -> Generator[list[DatasetConfig], None, None]:
    """Yields all dataset configurations used in tests."""
    if os.getenv("TEST_ALL_DATASETS", "0") == "1":
        yield list(get_all_dataset_configs().values())
    else:
        yield [
            dataset_config
            for dataset_config in get_all_dataset_configs().values()
            if not dataset_config.unofficial
        ]
