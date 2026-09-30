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
from unittest.mock import MagicMock

import pytest
import torch
from requests.exceptions import RequestException

from euroeval.benchmarker import Benchmarker, adjust_logging_level, clear_model_cache_fn
from euroeval.data_models import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    Language,
    ModelConfig,
    Task,
)
from euroeval.enums import InferenceBackend, ModelType, ShotMode
from euroeval.exceptions import HuggingFaceHubDown
from euroeval.result_cache import get_record
from euroeval.tasks import CONTAMINATION_DETECTION, NER, SENT


class TestClearCacheFn:
    """Tests related to the `clear_cache_fn` function."""

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

    def test_clear_non_existing_cache(self) -> None:
        """Test that no errors are thrown when clearing a non-existing cache."""
        clear_model_cache_fn(cache_dir="does-not-exist")
        rmtree(path="does-not-exist", ignore_errors=True)


class TestCreateModelDatasetMapping:
    """Tests for `Benchmarker._create_model_dataset_mapping`."""

    def test_canary_is_preserved_alongside_supported_zero_shot_dataset(
        self, benchmarker: Benchmarker, model_config: ModelConfig
    ) -> None:
        """The virtual canary bypasses task-group filtering for Laya models."""
        zero_shot_model_config = replace(
            model_config, model_type=ModelType.ZERO_SHOT_CLASSIFIER
        )
        canary_dataset_config = DatasetConfig(
            name="contamination-canary",
            pretty_name="Contamination canary",
            task=CONTAMINATION_DETECTION,
            languages=[Language(code="da", name="Danish")],
        )
        sent_dataset_config = DatasetConfig(
            name="sent-dataset",
            pretty_name="Sentiment dataset",
            source="dataset_id",
            task=SENT,
            languages=[Language(code="da", name="Danish")],
        )

        mapping = benchmarker._create_model_dataset_mapping(
            model_configs=[zero_shot_model_config],
            dataset_configs=[canary_dataset_config, sent_dataset_config],
        )

        assert mapping[zero_shot_model_config] == [
            canary_dataset_config,
            sent_dataset_config,
        ]

    def test_standalone_canary_is_preserved_for_zero_shot_model(
        self, benchmarker: Benchmarker, model_config: ModelConfig
    ) -> None:
        """A Laya-only canary run still includes its virtual dataset."""
        zero_shot_model_config = replace(
            model_config, model_type=ModelType.ZERO_SHOT_CLASSIFIER
        )
        canary_dataset_config = DatasetConfig(
            name="contamination-canary",
            pretty_name="Contamination canary",
            task=CONTAMINATION_DETECTION,
            languages=[Language(code="da", name="Danish")],
        )

        mapping = benchmarker._create_model_dataset_mapping(
            model_configs=[zero_shot_model_config],
            dataset_configs=[canary_dataset_config],
        )

        assert mapping[zero_shot_model_config] == [canary_dataset_config]

    def test_unsupported_task_group_is_filtered_out_for_zero_shot_classifier(
        self, benchmarker: Benchmarker, model_config: ModelConfig
    ) -> None:
        """A NER dataset is filtered out up front for a zero-shot classifier model."""
        zero_shot_model_config = replace(
            model_config, model_type=ModelType.ZERO_SHOT_CLASSIFIER
        )
        ner_dataset_config = DatasetConfig(
            name="ner-dataset",
            pretty_name="NER dataset",
            source="dataset_id",
            task=NER,
            languages=[Language(code="da", name="Danish")],
        )
        sent_dataset_config = DatasetConfig(
            name="sent-dataset",
            pretty_name="Sentiment dataset",
            source="dataset_id",
            task=SENT,
            languages=[Language(code="da", name="Danish")],
        )

        mapping = benchmarker._create_model_dataset_mapping(
            model_configs=[zero_shot_model_config],
            dataset_configs=[ner_dataset_config, sent_dataset_config],
        )

        assert mapping[zero_shot_model_config] == [sent_dataset_config]


class TestDatasetArgumentConflicts:
    """Tests for the mutually exclusive `dataset` and `task` arguments."""

    def test_benchmark_with_dataset_and_language(
        self, benchmarker: Benchmarker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that a `language` argument does not conflict with a `dataset`."""
        monkeypatch.setattr(
            benchmarker, "_fetch_model_configs", lambda *args, **kwargs: []
        )
        monkeypatch.setattr(
            benchmarker, "_create_model_dataset_mapping", lambda *args, **kwargs: {}
        )
        benchmark_results = benchmarker.benchmark(
            model="dummy", dataset="dansk", language="da"
        )
        assert benchmark_results == []

    def test_benchmark_with_dataset_and_task_raises(
        self, benchmarker: Benchmarker
    ) -> None:
        """Test that specifying both `dataset` and `task` raises an error."""
        with pytest.raises(ValueError, match="Only one of `task` and `dataset"):
            benchmarker.benchmark(model="dummy", dataset="dansk", task="classification")

    @pytest.mark.parametrize(
        argnames=["language"],
        argvalues=[("all",), ("da",), (["da"],)],
        ids=["all", "str-code", "list-code"],
    )
    def test_init_with_dataset_and_language(self, language: str | list[str]) -> None:
        """Test that a language selection can be combined with a `dataset`."""
        benchmarker = Benchmarker(
            dataset="dansk", language=language, progress_bar=False, save_results=False
        )
        assert [dataset.name for dataset in benchmarker.benchmark_config.datasets] == [
            "dansk"
        ]

    def test_init_with_dataset_and_task_raises(self) -> None:
        """Test that specifying both `dataset` and `task` raises an error."""
        with pytest.raises(ValueError, match="Only one of `task` and `dataset"):
            Benchmarker(task="classification", dataset="dansk")


class TestDebugStartupVerbosity:
    """Tests for the --debug startup verbosity bug fix."""

    def test_call_debug_true_suppresses_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that call debug=True suppresses the --verbose hint."""
        logged_messages: list[str] = []
        monkeypatch.delenv("FULL_LOG", raising=False)

        def mock_log_once(message: str, level: int, prefix: str = "") -> None:
            logged_messages.append(message)

        monkeypatch.setattr("euroeval.benchmarker.log_once", mock_log_once)
        monkeypatch.setattr("euroeval.benchmarker.log", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "euroeval.benchmarker.adjust_logging_level", lambda *args, **kwargs: None
        )

        benchmarker = Benchmarker(
            progress_bar=False,
            save_results=False,
            num_iterations=1,
            debug=False,
            run_with_cli=True,
        )

        # Mock the methods that would do real work
        monkeypatch.setattr(
            benchmarker, "_fetch_model_configs", lambda *args, **kwargs: []
        )
        monkeypatch.setattr(
            benchmarker, "_create_model_dataset_mapping", lambda *args, **kwargs: {}
        )

        benchmarker.benchmark(model="test_model", debug=True)

        # Should use the short message (no --verbose hint) when debug=True
        assert any("Started EuroEval run." in msg for msg in logged_messages)
        assert not any(
            "Run with `--verbose` for more information" in msg
            for msg in logged_messages
        )

    def test_init_debug_true_suppresses_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that init debug=True suppresses the --verbose hint."""
        logged_messages: list[str] = []
        monkeypatch.delenv("FULL_LOG", raising=False)

        def mock_log_once(message: str, level: int, prefix: str = "") -> None:
            logged_messages.append(message)

        monkeypatch.setattr("euroeval.benchmarker.log_once", mock_log_once)
        monkeypatch.setattr("euroeval.benchmarker.log", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "euroeval.benchmarker.adjust_logging_level", lambda *args, **kwargs: None
        )

        benchmarker = Benchmarker(
            progress_bar=False,
            save_results=False,
            num_iterations=1,
            debug=True,
            run_with_cli=True,
        )

        # Mock the methods that would do real work
        monkeypatch.setattr(
            benchmarker, "_fetch_model_configs", lambda *args, **kwargs: []
        )
        monkeypatch.setattr(
            benchmarker, "_create_model_dataset_mapping", lambda *args, **kwargs: {}
        )

        benchmarker.benchmark(model="test_model")

        # Should use the short message (no --verbose hint) when debug=True
        assert any("Started EuroEval run." in msg for msg in logged_messages)
        assert not any(
            "Run with `--verbose` for more information" in msg
            for msg in logged_messages
        )

    def test_non_debug_shows_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the normal non-debug case still shows the --verbose hint."""
        logged_messages: list[str] = []
        monkeypatch.delenv("FULL_LOG", raising=False)

        def mock_log_once(message: str, level: int, prefix: str = "") -> None:
            logged_messages.append(message)

        monkeypatch.setattr("euroeval.benchmarker.log_once", mock_log_once)
        monkeypatch.setattr("euroeval.benchmarker.log", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "euroeval.benchmarker.adjust_logging_level", lambda *args, **kwargs: None
        )

        benchmarker = Benchmarker(
            progress_bar=False,
            save_results=False,
            num_iterations=1,
            debug=False,
            run_with_cli=True,
        )

        # Mock the methods that would do real work
        monkeypatch.setattr(
            benchmarker, "_fetch_model_configs", lambda *args, **kwargs: []
        )
        monkeypatch.setattr(
            benchmarker, "_create_model_dataset_mapping", lambda *args, **kwargs: {}
        )

        benchmarker.benchmark(model="test_model")

        # Should show the --verbose hint when debug=False
        assert any(
            "Run with `--verbose` for more information" in msg
            for msg in logged_messages
        )


class TestZeroShotClassifierModelReuse:
    """Tests that zero-shot classifier models are preloaded and reused."""

    def test_zero_shot_classifier_is_not_skipped_by_generative_type_check(
        self,
        benchmarker: Benchmarker,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A preloaded zero-shot model without a generative type is benchmarked."""
        zero_shot_model_config = replace(
            model_config, model_type=ModelType.ZERO_SHOT_CLASSIFIER
        )
        loaded_model = MagicMock()
        loaded_model.generative_type = None
        assert None not in dataset_config.allowed_generative_types

        monkeypatch.setattr(
            benchmarker,
            "_fetch_model_configs",
            MagicMock(return_value=[zero_shot_model_config]),
        )
        monkeypatch.setattr(
            benchmarker,
            "_create_model_dataset_mapping",
            MagicMock(return_value={zero_shot_model_config: [dataset_config]}),
        )
        monkeypatch.setattr(
            benchmarker,
            "_prepare_pending_benchmarks",
            MagicMock(
                return_value=(
                    loaded_model,
                    [(ShotMode.ZERO_SHOT, dataset_config)],
                    [],
                    None,
                )
            ),
        )
        monkeypatch.setattr(benchmarker, "_check_adapter_requirements", MagicMock())
        benchmark_single_mock = MagicMock()
        monkeypatch.setattr(benchmarker, "_benchmark_single", benchmark_single_mock)
        monkeypatch.setattr(
            benchmarker,
            "_handle_benchmark_result",
            MagicMock(return_value=(1, 0, 0, False)),
        )

        benchmarker.benchmark(model="test_model", dataset="sst5")

        benchmark_single_mock.assert_called_once()

    def test_zero_shot_classifier_model_is_loaded_once_across_datasets(
        self,
        benchmarker: Benchmarker,
        benchmark_config: BenchmarkConfig,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`load_model` is called once and reused across two datasets."""
        zero_shot_model_config = replace(
            model_config, model_type=ModelType.ZERO_SHOT_CLASSIFIER
        )
        second_dataset_config = DatasetConfig(
            name="dataset-2",
            pretty_name="Dataset 2",
            source="dataset_id",
            task=dataset_config.task,
            languages=dataset_config.languages,
        )

        loaded_model = MagicMock()
        loaded_model.generative_type = None
        loaded_model.num_params = 100
        loaded_model.model_max_length = 512
        loaded_model.vocab_size = 32_000
        loaded_model.prepare_datasets.return_value = MagicMock()

        load_model_mock = MagicMock(return_value=loaded_model)
        monkeypatch.setattr("euroeval.benchmarker.load_model", load_model_mock)
        monkeypatch.setattr("euroeval.benchmarker.enforce_reproducibility", MagicMock())
        monkeypatch.setattr("euroeval.benchmarker.initial_logging", MagicMock())
        monkeypatch.setattr("euroeval.benchmarker.load_data", MagicMock())
        monkeypatch.setattr("euroeval.benchmarker.generate", MagicMock(return_value={}))
        monkeypatch.setattr(
            "euroeval.benchmarker.log_scores", MagicMock(return_value={})
        )

        prepared_model, pending_benchmarks, _, load_error = (
            benchmarker._prepare_pending_benchmarks(
                model_config=zero_shot_model_config,
                datasets=[dataset_config, second_dataset_config],
                benchmark_config=benchmark_config,
                existing_results=[],
            )
        )

        assert load_error is None
        assert prepared_model is loaded_model
        assert load_model_mock.call_count == 1
        assert len(pending_benchmarks) == 2

        for _shot_mode, pending_dataset_config in pending_benchmarks:
            result = benchmarker._benchmark_single(
                model=prepared_model,
                model_config=zero_shot_model_config,
                dataset_config=pending_dataset_config,
                benchmark_config=benchmark_config,
                num_finished_benchmarks=0,
                num_total_benchmarks=len(pending_benchmarks),
            )
            assert isinstance(result, BenchmarkResult)

        assert load_model_mock.call_count == 1
        assert loaded_model.update_dataset_config.call_count == 2


@pytest.fixture(scope="module")
def benchmarker() -> Generator[Benchmarker, None, None]:
    """A `Benchmarker` instance.

    Yields:
        A `Benchmarker` instance.
    """
    yield Benchmarker(progress_bar=False, save_results=False, num_iterations=1)


@pytest.mark.parametrize(
    argnames=["verbose", "expected_logging_level"],
    argvalues=[(False, logging.INFO), (True, logging.DEBUG)],
)
def test_adjust_logging_level(verbose: bool, expected_logging_level: int) -> None:
    """Test that the logging level is adjusted correctly."""
    logging_level = adjust_logging_level(verbose=verbose, ignore_testing=True)
    assert logging_level == expected_logging_level


@pytest.mark.depends(on=["tests/test_model_loading.py::test_load_dummy_model"])
def test_benchmark_dummy(
    benchmarker: Benchmarker, task: Task, language: Language
) -> None:
    """Test that the dummy model can be benchmarked."""
    benchmark_result = benchmarker.benchmark(
        model="dummy", task=task.name, language=language.code
    )
    assert isinstance(benchmark_result, list)
    assert all(isinstance(result, BenchmarkResult) for result in benchmark_result)


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


@pytest.mark.skipif(
    condition=not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key is not available."
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


def test_benchmark_result_includes_model_release_date(
    monkeypatch: pytest.MonkeyPatch,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
) -> None:
    """A completed evaluation copies model release metadata into its result."""
    dated_config = replace(model_config, release_date="2024-02-03")
    model = MagicMock()
    model.num_params = 100
    model.model_max_length = 512
    model.vocab_size = 32_000
    model.generative_type = None
    model.prepare_datasets.return_value = MagicMock()

    monkeypatch.setattr("euroeval.benchmarker.enforce_reproducibility", MagicMock())
    monkeypatch.setattr("euroeval.benchmarker.initial_logging", MagicMock())
    monkeypatch.setattr("euroeval.benchmarker.load_data", MagicMock())
    monkeypatch.setattr(
        "euroeval.benchmarker.load_model", MagicMock(return_value=model)
    )
    monkeypatch.setattr("euroeval.benchmarker.finetune", MagicMock())
    monkeypatch.setattr("euroeval.benchmarker.log_scores", MagicMock(return_value={}))

    result = Benchmarker(progress_bar=False, save_results=False)._benchmark_single(
        model=model,
        model_config=dated_config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
        num_finished_benchmarks=0,
        num_total_benchmarks=1,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.release_date == "2024-02-03"


def test_benchmark_results_is_a_list(benchmarker: Benchmarker) -> None:
    """Test that the `benchmark_results` property is a list."""
    assert isinstance(benchmarker.benchmark_results, list)


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

    def mock_metric_download(*_args, **_kwargs) -> None:
        pass

    for metric in dataset_config.task.metrics:
        monkeypatch.setattr(metric, "download", mock_metric_download)

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
        num_parameters=None,
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


def test_encoder_canary_mixed_run_reuses_ordinary_result_metadata(
    benchmarker: Benchmarker,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mixed encoder run should reuse metadata from its ordinary result."""
    canary_dataset = DatasetConfig(
        task=CONTAMINATION_DETECTION,
        languages=benchmark_config.languages,
        name="contamination-canary",
    )
    ordinary_result = BenchmarkResult(
        dataset=dataset_config.name,
        task=dataset_config.task.name,
        languages=[language.code for language in dataset_config.languages],
        model="encoder-model",
        results={"raw": [], "total": {}},
        num_model_parameters=456,
        max_sequence_length=768,
        vocabulary_size=64_000,
        merge=False,
        generative=False,
        generative_type=None,
        few_shot=False,
        validation_split=True,
    )
    run_config = replace(
        benchmark_config,
        datasets=[dataset_config, canary_dataset],
        force=True,
        save_results=False,
    )
    encoder_config = replace(
        model_config, model_id="encoder-model", revision="main", fresh=False
    )

    monkeypatch.setattr(benchmarker, "results_path", tmp_path / "results.jsonl")
    monkeypatch.setattr(
        benchmarker, "_build_benchmark_config", lambda **kwargs: run_config
    )
    monkeypatch.setattr(
        benchmarker, "_fetch_model_configs", lambda *args, **kwargs: [encoder_config]
    )
    monkeypatch.setattr(
        benchmarker, "_benchmark_single", lambda **kwargs: ordinary_result
    )
    load_model_mock = MagicMock(side_effect=AssertionError("unexpected model load"))
    monkeypatch.setattr("euroeval.benchmarker.load_model", load_model_mock)

    results = benchmarker.benchmark(model=encoder_config.model_id)

    assert len(results) == 2
    canary_result = results[1]
    assert canary_result.task == CONTAMINATION_DETECTION.name
    assert canary_result.num_model_parameters == ordinary_result.num_model_parameters
    assert canary_result.max_sequence_length == ordinary_result.max_sequence_length
    assert canary_result.vocabulary_size == ordinary_result.vocabulary_size
    assert load_model_mock.call_count == 0


def test_encoder_canary_standalone_uses_supported_metadata_task(
    benchmarker: Benchmarker,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A standalone encoder canary must not load the virtual text-to-text task."""
    canary_dataset = DatasetConfig(
        task=CONTAMINATION_DETECTION,
        languages=benchmark_config.languages,
        name="contamination-canary",
    )
    run_config = replace(
        benchmark_config, datasets=[canary_dataset], force=True, save_results=False
    )
    encoder_config = replace(
        model_config, model_id="encoder-model", revision="main", fresh=False
    )
    metadata_model = MagicMock(
        num_params=123, model_max_length=512, vocab_size=32_000, generative_type=None
    )
    load_model_mock = MagicMock(return_value=metadata_model)

    monkeypatch.setattr(benchmarker, "results_path", tmp_path / "results.jsonl")
    monkeypatch.setattr(
        benchmarker, "_build_benchmark_config", lambda **kwargs: run_config
    )
    monkeypatch.setattr(
        benchmarker, "_fetch_model_configs", lambda *args, **kwargs: [encoder_config]
    )
    monkeypatch.setattr("euroeval.benchmarker.load_model", load_model_mock)

    results = benchmarker.benchmark(model=encoder_config.model_id)

    assert len(results) == 1
    result = results[0]
    assert result.task == CONTAMINATION_DETECTION.name
    assert result.contamination_canary_evidence is not None
    assert result.contamination_canary_evidence["status"] == "not_applicable"
    assert result.contamination_canary_evidence["reason"] == "encoder"
    assert result.num_model_parameters == metadata_model.num_params
    assert load_model_mock.call_args.kwargs["dataset_config"].task != (
        CONTAMINATION_DETECTION
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


def test_laya_download_only_resolves_root_checkpoint(
    benchmarker: Benchmarker,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Laya downloads resolve only the root checkpoint, not cache globbing."""
    model_cache_dir = tmp_path / "model-cache"
    (model_cache_dir / "other-variant").mkdir(parents=True)
    (model_cache_dir / "other-variant" / "model.safetensors").touch()
    zero_shot_config = replace(
        model_config,
        model_type=ModelType.ZERO_SHOT_CLASSIFIER,
        model_cache_dir=str(model_cache_dir),
        param=None,
    )
    resolver_mock = MagicMock()
    resolved_token = "resolved-token"
    monkeypatch.setattr(
        "euroeval.benchmark_modules.zero_shot_classifier._resolve_checkpoint_path",
        resolver_mock,
    )
    monkeypatch.setattr(
        "euroeval.benchmarker.get_hf_token", lambda **kwargs: resolved_token
    )

    benchmarker._download_model_only(
        model_config=zero_shot_config, benchmark_config=benchmark_config
    )

    resolver_mock.assert_called_once_with(
        model_id=zero_shot_config.model_id,
        cache_dir=str(model_cache_dir),
        token=resolved_token,
    )


def test_zero_shot_canary_standalone_reuses_loaded_model_metadata(
    benchmarker: Benchmarker,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A standalone Laya canary does not load the model a second time."""
    canary_dataset = DatasetConfig(
        task=CONTAMINATION_DETECTION,
        languages=benchmark_config.languages,
        name="contamination-canary",
    )
    run_config = replace(
        benchmark_config, datasets=[canary_dataset], force=True, save_results=False
    )
    zero_shot_config = replace(
        model_config,
        model_id="laya-model",
        revision="main",
        fresh=False,
        model_type=ModelType.ZERO_SHOT_CLASSIFIER,
    )
    metadata_model = MagicMock(
        num_params=123, model_max_length=512, vocab_size=32_000, generative_type=None
    )
    load_model_mock = MagicMock(return_value=metadata_model)

    monkeypatch.setattr(benchmarker, "results_path", tmp_path / "results.jsonl")
    monkeypatch.setattr(
        benchmarker, "_build_benchmark_config", lambda **kwargs: run_config
    )
    monkeypatch.setattr(
        benchmarker, "_fetch_model_configs", lambda *args, **kwargs: [zero_shot_config]
    )
    monkeypatch.setattr("euroeval.benchmarker.load_model", load_model_mock)

    results = benchmarker.benchmark(model=zero_shot_config.model_id)

    assert len(results) == 1
    assert results[0].task == CONTAMINATION_DETECTION.name
    assert load_model_mock.call_count == 1
    assert load_model_mock.call_args.kwargs["dataset_config"].task != (
        CONTAMINATION_DETECTION
    )
