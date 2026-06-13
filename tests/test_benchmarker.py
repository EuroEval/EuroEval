"""Tests for the `benchmarker` module."""

import logging
import os
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from shutil import rmtree
from typing import Never

import pytest
import torch
from requests.exceptions import RequestException

from euroeval.benchmarker import (
    Benchmarker,
    adjust_logging_level,
    clear_model_cache_fn,
    get_record,
)
from euroeval.data_models import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    Language,
    ModelConfig,
    Task,
)
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import HuggingFaceHubDown


@pytest.fixture(scope="module")
def benchmarker() -> Generator[Benchmarker, None, None]:
    """A `Benchmarker` instance.

    Yields:
        A `Benchmarker` instance.
    """
    yield Benchmarker(progress_bar=False, save_results=False, num_iterations=1)


def test_benchmark_results_is_a_list(benchmarker: Benchmarker) -> None:
    """Test that the `benchmark_results` property is a list."""
    assert isinstance(benchmarker.benchmark_results, list)


@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_non_generative_model"])
def test_benchmark_encoder(
    benchmarker: Benchmarker, task: Task, language: Language, encoder_model_id: str
) -> None:
    """Test that an encoder model can be benchmarked."""
    benchmark_result = None
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
    condition=sys.platform == "linux" and not torch.cuda.is_available(),
    reason="Running on Ubuntu but no CUDA available",
)
@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_generative_model"])
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
    condition=sys.platform == "linux" and not torch.cuda.is_available(),
    reason="Running on Ubuntu but no CUDA available",
)
@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_generative_model"])
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
    condition=subprocess.run(
        ["uv", "run", "ollama", "-v"], capture_output=True
    ).returncode
    != 0,
    reason="Ollama is not available.",
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


@pytest.mark.depends(on=["test_benchmark_encoder"])
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


@pytest.mark.skipif(
    condition=sys.platform == "linux" and not torch.cuda.is_available(),
    reason="Running on Ubuntu but no CUDA available",
)
@pytest.mark.depends(on=["test_benchmark_generative"])
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


@pytest.mark.skipif(
    condition=sys.platform == "linux" and not torch.cuda.is_available(),
    reason="Running on Ubuntu but no CUDA available",
)
@pytest.mark.skip(
    "Benchmarking adapter models without internet access are not implemented yet."
)
@pytest.mark.depends(on=["test_benchmark_generative_adapter"])
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


def test_download_only_does_not_instantiate_model(
    task: Task,
    language: Language,
    encoder_model_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> None:
    """Test that download_only mode downloads models without instantiating them.

    This ensures that --download-only mode works on machines without CUDA/GPUs.
    """
    if task.name == "translation":
        pytest.skip("Translation tasks require two languages")

    snapshot_download_called = False
    load_model_called = False

    def mock_snapshot_download(*_args, **_kwargs) -> None:
        nonlocal snapshot_download_called
        snapshot_download_called = True

    def mock_load_model(*_args, **_kwargs) -> Never:
        nonlocal load_model_called
        load_model_called = True
        raise RuntimeError("load_model should not be called in download_only mode")

    monkeypatch.setattr(
        "euroeval.benchmarker.snapshot_download", mock_snapshot_download
    )
    monkeypatch.setattr("euroeval.model_loading.load_model", mock_load_model)

    def mock_load_raw_data(*_args, **_kwargs) -> None:
        pass

    monkeypatch.setattr("euroeval.benchmarker.load_raw_data", mock_load_raw_data)

    dataset_config = DatasetConfig(
        task=task,
        languages=[language],
        name="test_dataset",
        pretty_name="Test Dataset",
        source="test/source",
    )

    model_config = ModelConfig(
        model_id=encoder_model_id,
        revision="main",
        param=None,
        task="test",
        languages=[language],
        inference_backend=InferenceBackend.TRANSFORMERS,
        merge=False,
        model_type=ModelType.ENCODER,
        fresh=True,
        model_cache_dir=str(tmp_path / "models"),
        adapter_base_model_id=None,
    )
    benchmark_config = BenchmarkConfig(
        datasets=[dataset_config],
        languages=[language],
        finetuning_batch_size=1,
        raise_errors=True,
        cache_dir=str(tmp_path / "cache"),
        api_key=None,
        api_base=None,
        api_version=None,
        force=False,
        progress_bar=False,
        save_results=False,
        device=device,
        verbose=False,
        debug=False,
        trust_remote_code=False,
        clear_model_cache=False,
        evaluate_test_split=True,
        few_shot=False,
        num_iterations=1,
        gpu_memory_utilization=0.9,
        attention_backend=None,
        requires_safetensors=False,
        generative_type=None,
        run_with_cli=True,
        max_context_length=None,
        vocabulary_size=None,
        download_only=True,
    )

    benchmarker = Benchmarker(progress_bar=False, save_results=False, num_iterations=1)
    benchmarker._download(
        dataset_config=dataset_config,
        model_config=model_config,
        benchmark_config=benchmark_config,
    )

    assert snapshot_download_called, "snapshot_download should be called"
    assert not load_model_called, (
        "load_model should NOT be called in download_only mode"
    )


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
