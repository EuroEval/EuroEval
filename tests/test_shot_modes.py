"""Tests for automatic benchmark shot-mode selection."""

from dataclasses import replace
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
