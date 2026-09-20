"""Tests for automatic benchmark shot-mode selection."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from euroeval.benchmarker import Benchmarker, resolve_shot_modes
from euroeval.data_models import (
    BenchmarkConfig,
    BenchmarkResult,
    DatasetConfig,
    ModelConfig,
)
from euroeval.enums import GenerativeType, InferenceBackend, ModelType, ShotMode
from euroeval.exceptions import InvalidModel


def test_multi_model_progress_uses_full_workload(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    tmp_path: Path,
) -> None:
    """Progress totals include pending work from models not started yet."""
    models = [
        replace(model_config, model_id="first-model"),
        replace(model_config, model_id="second-model"),
    ]
    config = replace(
        benchmark_config, datasets=[dataset_config], few_shot=None, save_results=False
    )
    benchmarker = Benchmarker(progress_bar=False, save_results=False)
    benchmarker.results_path = tmp_path / "results.jsonl"
    monkeypatch.setattr(benchmarker, "_build_benchmark_config", lambda **_: config)
    monkeypatch.setattr(
        benchmarker,
        "_prepare_model_ids",
        lambda model_id: ["first-model", "second-model"],
    )
    monkeypatch.setattr(
        benchmarker, "_fetch_model_configs", lambda model_ids, benchmark_config: models
    )
    monkeypatch.setattr(
        benchmarker,
        "_create_model_dataset_mapping",
        lambda model_configs, dataset_configs: {
            model_config: [dataset_config] for model_config in model_configs
        },
    )
    monkeypatch.setattr(benchmarker, "_check_adapter_requirements", lambda *args: None)
    monkeypatch.setattr(
        benchmarker, "_update_benchmark_config_for_dataset", lambda *args: None
    )
    monkeypatch.setattr(
        benchmarker,
        "_prepare_shot_benchmarks",
        Mock(
            side_effect=[
                (None, [(ShotMode.ZERO_SHOT, dataset_config)], [], None),
                (None, [(ShotMode.FEW_SHOT, dataset_config)], [], None),
            ]
        ),
    )
    benchmark_calls = Mock(return_value=Mock())
    monkeypatch.setattr(benchmarker, "_benchmark_single", benchmark_calls)

    def handle_result(**kwargs: int) -> tuple[int, int, int, bool]:
        return (
            kwargs["num_finished"] + 1,
            kwargs["num_skipped"],
            kwargs["num_errored"],
            False,
        )

    monkeypatch.setattr(benchmarker, "_handle_benchmark_result", handle_result)

    benchmarker.benchmark(model=["first-model", "second-model"])

    assert [
        call.kwargs["num_total_benchmarks"] for call in benchmark_calls.call_args_list
    ] == [2, 2]


def test_auto_shot_mode_resolution(model_config: ModelConfig) -> None:
    """AUTO selects the agreed modes for each model category."""
    encoder = model_config
    local = replace(encoder, model_type=ModelType.GENERATIVE)
    api = replace(local, inference_backend=InferenceBackend.LITELLM)

    assert resolve_shot_modes(encoder, None) == [ShotMode.FEW_SHOT]
    assert resolve_shot_modes(local, None, GenerativeType.BASE) == [ShotMode.FEW_SHOT]
    assert resolve_shot_modes(local, None, GenerativeType.INSTRUCTION_TUNED) == [
        ShotMode.ZERO_SHOT,
        ShotMode.FEW_SHOT,
    ]
    assert resolve_shot_modes(api, None) == [ShotMode.ZERO_SHOT]


def test_explicit_shot_mode_overrides(model_config: ModelConfig) -> None:
    """Legacy boolean overrides remain single-mode selections."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)

    assert resolve_shot_modes(generative, True) == [ShotMode.FEW_SHOT]
    assert resolve_shot_modes(generative, False) == [ShotMode.ZERO_SHOT]
    assert resolve_shot_modes(generative, ShotMode.FEW_SHOT) == [ShotMode.FEW_SHOT]


def test_auto_dual_mode_loads_model_once(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
) -> None:
    """Both AUTO flows share one loaded model."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)
    config = replace(benchmark_config, few_shot=None)
    loaded_model = SimpleNamespace(generative_type=GenerativeType.INSTRUCTION_TUNED)
    load_model = Mock(return_value=loaded_model)
    monkeypatch.setattr("euroeval.benchmarker.load_model", load_model)

    loaded, pending, _, error = Benchmarker(
        progress_bar=False
    )._prepare_shot_benchmarks(
        model_config=generative,
        datasets=[dataset_config],
        benchmark_config=config,
        existing_results=[],
    )

    assert error is None
    assert loaded is loaded_model
    assert [mode for mode, _ in pending] == [ShotMode.ZERO_SHOT, ShotMode.FEW_SHOT]
    load_model.assert_called_once()


def test_auto_cached_base_model_uses_cached_metadata(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
) -> None:
    """AUTO does not return an invalid zero-shot cache for a base model."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)
    few_shot_result = BenchmarkResult(
        model="model_id@revision",
        dataset=dataset_config.name,
        generative=True,
        generative_type=GenerativeType.BASE.value,
        few_shot=True,
        validation_split=True,
        num_model_parameters=1,
        max_sequence_length=1,
        vocabulary_size=1,
        merge=False,
        languages=["da"],
        task=dataset_config.task.name,
        results={},
    )
    load_model = Mock()
    monkeypatch.setattr("euroeval.benchmarker.load_model", load_model)

    _, pending, cached, error = Benchmarker(
        progress_bar=False
    )._prepare_shot_benchmarks(
        model_config=generative,
        datasets=[dataset_config],
        benchmark_config=replace(benchmark_config, few_shot=None),
        existing_results=[few_shot_result],
    )

    assert error is None
    assert pending == []
    assert cached == [few_shot_result]
    load_model.assert_not_called()


def test_cached_shot_mode_survives_missing_mode_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
) -> None:
    """AUTO retains cached records when loading the missing mode fails."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)
    few_shot_result = BenchmarkResult(
        model="model_id@revision",
        dataset=dataset_config.name,
        generative=True,
        generative_type=GenerativeType.INSTRUCTION_TUNED.value,
        few_shot=True,
        validation_split=True,
        num_model_parameters=1,
        max_sequence_length=1,
        vocabulary_size=1,
        merge=False,
        languages=["da"],
        task=dataset_config.task.name,
        results={},
    )
    monkeypatch.setattr(
        "euroeval.benchmarker.load_model",
        Mock(side_effect=InvalidModel("model setup failed")),
    )

    _, pending, cached, error = Benchmarker(
        progress_bar=False
    )._prepare_shot_benchmarks(
        model_config=generative,
        datasets=[dataset_config],
        benchmark_config=replace(benchmark_config, few_shot=None),
        existing_results=[few_shot_result],
    )

    assert isinstance(error, InvalidModel)
    assert [mode for mode, _ in pending] == [ShotMode.ZERO_SHOT]
    assert cached == [few_shot_result]


def test_cached_shot_modes_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
) -> None:
    """A cached result in one mode does not hide the other mode."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)
    few_shot_result = BenchmarkResult(
        model="model_id@revision",
        dataset=dataset_config.name,
        generative=True,
        generative_type=GenerativeType.INSTRUCTION_TUNED.value,
        few_shot=True,
        validation_split=True,
        num_model_parameters=1,
        max_sequence_length=1,
        vocabulary_size=1,
        merge=False,
        languages=["da"],
        task=dataset_config.task.name,
        results={},
    )
    config = replace(benchmark_config, few_shot=None)
    monkeypatch.setattr(
        "euroeval.benchmarker.load_model",
        Mock(
            return_value=SimpleNamespace(
                generative_type=GenerativeType.INSTRUCTION_TUNED
            )
        ),
    )

    _, pending, cached, error = Benchmarker(
        progress_bar=False
    )._prepare_shot_benchmarks(
        model_config=generative,
        datasets=[dataset_config],
        benchmark_config=config,
        existing_results=[few_shot_result],
    )

    assert error is None
    assert [mode for mode, _ in pending] == [ShotMode.ZERO_SHOT]
    assert cached == [few_shot_result]


def test_load_error_counts_concrete_remaining_work() -> None:
    """A mode-level model failure counts the other concrete work items."""
    benchmarker = Benchmarker(progress_bar=False)
    dataset_config = Mock()
    model_config = Mock()

    finished, skipped, errored, should_break = benchmarker._handle_benchmark_result(
        result_or_error=InvalidModel("model setup failed"),
        dataset_config=dataset_config,
        benchmark_config=Mock(raise_errors=False),
        num_finished=0,
        num_skipped=0,
        num_errored=0,
        model_config=model_config,
        model_mapping={model_config: [dataset_config]},
        current_results=[],
        remaining_work=1,
    )

    assert (finished, skipped, errored, should_break) == (0, 0, 2, True)


def test_model_cache_is_cleared_when_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    tmp_path: Path,
) -> None:
    """Per-model cleanup also runs when model setup raises."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)
    config = replace(
        benchmark_config,
        datasets=[dataset_config],
        few_shot=None,
        clear_model_cache=True,
        raise_errors=False,
        save_results=False,
    )
    benchmarker = Benchmarker(progress_bar=False, save_results=False)
    benchmarker.results_path = tmp_path / "results.jsonl"
    monkeypatch.setattr(benchmarker, "_build_benchmark_config", lambda **_: config)
    monkeypatch.setattr(benchmarker, "_prepare_model_ids", lambda model_id: ["model"])
    monkeypatch.setattr(
        benchmarker,
        "_fetch_model_configs",
        lambda model_ids, benchmark_config: [generative],
    )
    monkeypatch.setattr(
        benchmarker,
        "_create_model_dataset_mapping",
        lambda model_configs, dataset_configs: {generative: [dataset_config]},
    )
    monkeypatch.setattr(
        "euroeval.benchmarker.load_model",
        Mock(side_effect=InvalidModel("model setup failed")),
    )
    clear_cache = Mock()
    monkeypatch.setattr("euroeval.benchmarker.clear_model_cache_fn", clear_cache)

    assert benchmarker.benchmark(model="model") == []
    assert clear_cache.call_count == 2


def test_zero_shot_tasks_are_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
) -> None:
    """A task that requires zero-shot evaluation has one AUTO work item."""
    generative = replace(model_config, model_type=ModelType.GENERATIVE)
    original_task = dataset_config.task
    zero_shot_task = replace(original_task, requires_zero_shot=True)
    dataset_config.task = zero_shot_task
    config = replace(benchmark_config, few_shot=None)
    monkeypatch.setattr(
        "euroeval.benchmarker.load_model",
        Mock(
            return_value=SimpleNamespace(
                generative_type=GenerativeType.INSTRUCTION_TUNED
            )
        ),
    )

    _, pending, _, error = Benchmarker(progress_bar=False)._prepare_shot_benchmarks(
        model_config=generative,
        datasets=[dataset_config],
        benchmark_config=config,
        existing_results=[],
    )
    dataset_config.task = original_task

    assert error is None
    assert [mode for mode, _ in pending] == [ShotMode.ZERO_SHOT]
