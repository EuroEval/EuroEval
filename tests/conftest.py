"""General fixtures used throughout test modules."""

import os
import sys
from typing import Generator

import pytest
import torch

from euroeval.data_models import BenchmarkConfig, ModelConfig, Task
from euroeval.dataset_configs import SPEED_CONFIG, get_all_dataset_configs
from euroeval.enums import InferenceBackend, ModelType
from euroeval.languages import DA, get_all_languages
from euroeval.metrics import HuggingFaceMetric
from euroeval.tasks import SENT, SPEED, get_all_tasks


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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    yield device


@pytest.fixture(scope="session")
def benchmark_config(
    auth: str, device: torch.device
) -> Generator[BenchmarkConfig, None, None]:
    """Yields a benchmark configuration used in tests."""
    yield BenchmarkConfig(
        model_languages=[DA],
        dataset_languages=[DA],
        tasks=[SENT],
        datasets=list(get_all_dataset_configs().keys()),
        batch_size=1,
        raise_errors=False,
        cache_dir=".euroeval_cache",
        api_key=auth,
        force=False,
        progress_bar=False,
        save_results=True,
        device=device,
        verbose=False,
        trust_remote_code=True,
        clear_model_cache=False,
        evaluate_test_split=False,
        few_shot=True,
        num_iterations=1,
        api_base=None,
        api_version=None,
        gpu_memory_utilization=0.9,
        debug=False,
        run_with_cli=True,
        only_allow_safetensors=False,
    )


@pytest.fixture(scope="session")
def metric() -> Generator[HuggingFaceMetric, None, None]:
    """Yields a metric configuration used in tests."""
    yield HuggingFaceMetric(
        name="metric_name",
        pretty_name="Metric name",
        huggingface_id="metric_id",
        results_key="metric_key",
    )


@pytest.fixture(
    scope="session",
    params=[task for task in get_all_tasks().values() if task != SPEED],
    ids=[name for name, task in get_all_tasks().items() if task != SPEED],
)
def task(request: pytest.FixtureRequest) -> Generator[Task, None, None]:
    """Yields a dataset task used in tests."""
    yield request.param


@pytest.fixture(
    scope="session",
    params=list(ACTIVE_LANGUAGES.values()),
    ids=list(ACTIVE_LANGUAGES.keys()),
)
def language(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Yields a language used in tests."""
    yield request.param


@pytest.fixture(scope="session")
def encoder_model_id() -> Generator[str, None, None]:
    """Yields a model ID used in tests."""
    yield "jonfd/electra-small-nordic"


@pytest.fixture(scope="session")
def generative_model_id() -> Generator[str, None, None]:
    """Yields a generative model ID used in tests."""
    yield "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="session")
def generative_adapter_model_id() -> Generator[str, None, None]:
    """Yields a generative adapter model ID used in tests."""
    yield "jekunz/smollm-135m-lora-fineweb-swedish"


@pytest.fixture(scope="session")
def openai_model_id() -> Generator[str, None, None]:
    """Yields an OpenAI model ID used in tests."""
    yield "gpt-4o-mini"


@pytest.fixture(scope="session")
def anthropic_model_id() -> Generator[str, None, None]:
    """Yields an Anthropic model ID used in tests."""
    yield "claude-3-5-haiku-20241022"


@pytest.fixture(scope="session")
def gemini_model_id() -> Generator[str, None, None]:
    """Yields a Gemini model ID used in tests."""
    yield "gemini/gemini-2.0-flash"


@pytest.fixture(scope="session")
def grok_model_id() -> Generator[str, None, None]:
    """Yields a Grok model ID used in tests."""
    yield "grok-2-1212"


@pytest.fixture(scope="session")
def ollama_model_id() -> Generator[str, None, None]:
    """Yields an Ollama model ID used in tests."""
    yield "ollama_chat/smollm2:135m"


@pytest.fixture(scope="session")
def model_config() -> Generator[ModelConfig, None, None]:
    """Yields a model configuration used in tests."""
    yield ModelConfig(
        model_id="model_id",
        revision="revision",
        task="task",
        languages=[DA],
        merge=False,
        inference_backend=InferenceBackend.TRANSFORMERS,
        model_type=ModelType.ENCODER,
        fresh=True,
        model_cache_dir="cache_dir",
        adapter_base_model_id=None,
    )
