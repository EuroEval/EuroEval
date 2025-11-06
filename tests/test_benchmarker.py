"""Tests for the `benchmarker` module."""

import logging
import os
import time
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from shutil import rmtree

import pytest
import torch
from requests.exceptions import RequestException

from euroeval.benchmarker import (
    Benchmarker,
    adjust_logging_level,
    clear_model_cache_fn,
    get_record,
    prepare_dataset_configs,
)
from euroeval.data_models import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    Language,
    ModelConfig,
    Task,
)
from euroeval.dataset_configs import get_dataset_config
from euroeval.exceptions import HuggingFaceHubDown


@pytest.fixture(scope="module")
def benchmarker() -> Generator[Benchmarker, None, None]:
    """A `Benchmarker` instance."""
    yield Benchmarker(progress_bar=False, save_results=False, num_iterations=1)


def test_benchmark_results_is_a_list(benchmarker: Benchmarker) -> None:
    """Test that the `benchmark_results` property is a list."""
    assert isinstance(benchmarker.benchmark_results, list)


@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_non_generative_model"])
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_benchmark_encoder(
    benchmarker: Benchmarker, task: Task, language: Language, encoder_model_id: str
) -> None:
    """Test that an encoder model can be benchmarked."""
    for _ in range(10):
        try:
            benchmark_result = benchmarker.benchmark(
                model=encoder_model_id, task=task.name, language=language.code
            )
            break
        except (HuggingFaceHubDown, RequestException, ConnectionError):
            time.sleep(5)
    else:
        pytest.skip(reason="Hugging Face Hub is down, so we skip this test.")
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


@pytest.mark.skipif(
    condition=not torch.cuda.is_available(), reason="CUDA is not available."
)
@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_generative_model"])
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_benchmark_generative(
    benchmarker: Benchmarker, task: Task, language: Language, generative_model_id: str
) -> None:
    """Test that a generative model can be benchmarked."""
    benchmark_result = benchmarker.benchmark(
        model=generative_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


@pytest.mark.skipif(
    condition=not torch.cuda.is_available(), reason="CUDA is not available."
)
@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_generative_model"])
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_benchmark_generative_adapter(
    benchmarker: Benchmarker,
    task: Task,
    language: Language,
    generative_adapter_model_id: str,
) -> None:
    """Test that a generative adapter model can be benchmarked."""
    benchmark_result = benchmarker.benchmark(
        model=generative_adapter_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


@pytest.mark.skipif(
    condition=os.getenv("OPENAI_API_KEY") is None,
    reason="OpenAI API key is not available.",
)
def test_benchmark_openai(
    benchmarker: Benchmarker, task: Task, language: Language, openai_model_id: str
) -> None:
    """Test that an OpenAI model can be benchmarked."""
    benchmark_result = benchmarker.benchmark(
        model=openai_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


@pytest.mark.skipif(
    condition=os.system("uv run ollama -v") != 0, reason="Ollama is not available."
)
def test_benchmark_ollama(
    benchmarker: Benchmarker, task: Task, language: Language, ollama_model_id: str
) -> None:
    """Test that an Ollama model can be benchmarked."""
    benchmark_result = benchmarker.benchmark(
        model=ollama_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


@pytest.mark.disable_socket
@pytest.mark.depends(on=["test_benchmark_encoder"])
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_benchmark_encoder_no_internet(
    task: Task, language: Language, encoder_model_id: str
) -> None:
    """Test that encoder models can be benchmarked without internet."""
    # We need a new benchmarker since we only check for internet once per instance
    benchmarker = Benchmarker(progress_bar=False, save_results=False, num_iterations=1)
    benchmark_result = benchmarker.benchmark(
        model=encoder_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


# Allow localhost since vllm uses it for some things
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.skipif(
    condition=not torch.cuda.is_available(), reason="CUDA is not available."
)
@pytest.mark.depends(on=["test_benchmark_generative"])
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_benchmark_generative_no_internet(
    task: Task, language: Language, generative_model_id: str
) -> None:
    """Test that generative models can be benchmarked without internet."""
    # We need a new benchmarker since we only check for internet once per instance
    benchmarker = Benchmarker(progress_bar=False, save_results=False, num_iterations=1)
    benchmark_result = benchmarker.benchmark(
        model=generative_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


# Allow localhost since vllm uses it for some things
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.skip(
    "Benchmarking adapter models without internet access are not implemented yet."
)
@pytest.mark.depends(on=["test_benchmark_generative_adapter"])
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_benchmark_generative_adapter_no_internet(
    task: Task, language: Language, generative_adapter_model_id: str
) -> None:
    """Test that generative adapter models can be benchmarked without internet."""
    # We need a new benchmarker since we only check for internet once per instance
    benchmarker = Benchmarker(progress_bar=False, save_results=False, num_iterations=1)
    benchmark_result = benchmarker.benchmark(
        model=generative_adapter_model_id, task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


@pytest.mark.parametrize(
    argnames=["few_shot", "evaluate_test_split", "benchmark_results", "expected"],
    argvalues=[
        (False, True, [], False),
        (
            False,
            True,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            True,
        ),
        (
            True,
            True,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="another-dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            False,
        ),
        (
            True,
            True,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=True,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            False,
        ),
        (
            True,
            True,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=True,
                    generative_type=None,
                    few_shot=True,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            True,
        ),
        (
            True,
            True,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            True,
        ),
        (
            False,
            False,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            False,
        ),
        (
            False,
            False,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=True,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                )
            ],
            True,
        ),
        (
            False,
            True,
            [
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                ),
                BenchmarkResult(
                    model="model_id@revision",
                    dataset="dataset",
                    generative=False,
                    generative_type=None,
                    few_shot=False,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    merge=False,
                    languages=["da"],
                    task="task",
                    results=dict(),
                ),
            ],
            True,
        ),
    ],
    ids=[
        "empty benchmark results",
        "model has been benchmarked",
        "model has not been benchmarked",
        "model few-shot has not been benchmarked",
        "model few-shot has been benchmarked",
        "model few-shot has been benchmarked, but not generative",
        "model validation split has not been benchmarked",
        "model validation split has been benchmarked",
        "model has been benchmarked twice",
    ],
)
def test_get_record(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
    few_shot: bool,
    evaluate_test_split: bool,
    benchmark_results: list[BenchmarkResult],
    expected: bool,
) -> None:
    """Test whether we can correctly check if a model has been benchmarked."""
    benchmark_config = replace(
        benchmark_config, few_shot=few_shot, evaluate_test_split=evaluate_test_split
    )
    benchmarked = (
        get_record(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            benchmark_results=benchmark_results,
        )
        is not None
    )
    assert benchmarked == expected


@pytest.mark.parametrize(
    argnames=["verbose", "expected_logging_level"],
    argvalues=[(False, logging.INFO), (True, logging.DEBUG)],
)
def test_adjust_logging_level(verbose: bool, expected_logging_level: int) -> None:
    """Test that the logging level is adjusted correctly."""
    logging_level = adjust_logging_level(verbose=verbose, ignore_testing=True)
    assert logging_level == expected_logging_level


class TestClearCacheFn:
    """Tests related to the `clear_cache_fn` function."""

    def test_clear_non_existing_cache(self) -> None:
        """Test that no errors are thrown when clearing a non-existing cache."""
        clear_model_cache_fn(cache_dir="does-not-exist")
        rmtree(path="does-not-exist", ignore_errors=True)

    def test_clear_existing_cache(self) -> None:
        """Test that a cache can be cleared."""
        cache_dir = Path(".test_euroeval_cache")
        model_cache_dir = cache_dir / "model_cache"
        example_model_dir = model_cache_dir / "example_model"
        dir_to_be_deleted = example_model_dir / "dir_to_be_deleted"

        dir_to_be_deleted.mkdir(parents=True, exist_ok=True)
        assert dir_to_be_deleted.exists()

        clear_model_cache_fn(cache_dir=cache_dir.as_posix())
        assert not dir_to_be_deleted.exists()
        assert example_model_dir.exists()

        rmtree(path=cache_dir, ignore_errors=True)


@pytest.mark.parametrize(
    argnames=["dataset_names", "dataset_configs"],
    argvalues=[
        ([], []),
        (["angry-tweets"], [get_dataset_config("angry-tweets")]),
        (
            ["angry-tweets", "dansk"],
            [get_dataset_config("angry-tweets"), get_dataset_config("dansk")],
        ),
    ],
)
def test_prepare_dataset_configs(
    dataset_names: list[str], dataset_configs: list[DatasetConfig]
) -> None:
    """Test that the `prepare_dataset_configs` function works as expected."""
    assert prepare_dataset_configs(dataset_names=dataset_names) == dataset_configs
