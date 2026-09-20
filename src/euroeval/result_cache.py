"""Matching and filtering of concrete benchmark-result cache entries."""

import collections.abc as c
import typing as t

from .data_models import BenchmarkResult
from .enums import ShotMode
from .shot_modes import ShotModeRequest, ShotWork, coerce_shot_mode
from .string_utils import split_model_id

if t.TYPE_CHECKING:
    from .data_models import BenchmarkConfig, DatasetConfig, ModelConfig


def partition_shot_work(
    model_config: "ModelConfig",
    work: c.Sequence[ShotWork],
    benchmark_config: "BenchmarkConfig",
    benchmark_results: c.Sequence[BenchmarkResult],
) -> tuple[list[ShotWork], list[BenchmarkResult]]:
    """Separate concrete shot work into pending and cached portions.

    Args:
        model_config:
            The model configuration being evaluated.
        work:
            Concrete mode/dataset pairs in execution order.
        benchmark_config:
            The general benchmark configuration.
        benchmark_results:
            Results already present in the local cache.

    Returns:
        A tuple containing pending work and unique cached records. When ``force`` is
        enabled, all work is pending and cached records are omitted.
    """
    pending: list[ShotWork] = []
    cached: list[BenchmarkResult] = []
    for mode, dataset_config in work:
        record = get_record(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            benchmark_results=benchmark_results,
            shot_mode=mode,
        )
        if benchmark_config.force or record is None:
            pending.append((mode, dataset_config))
        elif record not in cached:
            cached.append(record)
    return pending, cached


def get_record(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    benchmark_results: c.Sequence[BenchmarkResult],
    shot_mode: ShotModeRequest = None,
) -> BenchmarkResult | None:
    """Find a cached result for a model, dataset, split, and shot mode.

    Args:
        model_config:
            The model configuration being evaluated.
        dataset_config:
            The dataset configuration being evaluated.
        benchmark_config:
            The general benchmark configuration.
        benchmark_results:
            Results already present in the local cache.
        shot_mode:
            The concrete mode to match. ``None`` inherits the mode from
            ``benchmark_config``. ``AUTO`` matches either stored generative mode while
            planning is still provisional.

    Returns:
        The matching result, or None if no result exists.
    """
    requested_mode = coerce_shot_mode(
        benchmark_config.few_shot if shot_mode is None else shot_mode
    )
    for record in benchmark_results:
        model_id_components = split_model_id(model_id=record.model)
        same_model = (
            model_id_components.model_id == model_config.model_id
            and model_id_components.revision == model_config.revision
            and model_id_components.param == model_config.param
        )
        same_dataset = record.dataset == dataset_config.name
        same_split = record.validation_split != benchmark_config.evaluate_test_split
        same_shot_mode = (
            not record.generative
            or requested_mode is ShotMode.AUTO
            or (requested_mode is ShotMode.FEW_SHOT and record.few_shot is True)
            or (
                requested_mode is ShotMode.ZERO_SHOT
                and record.few_shot in (False, None)
            )
        )
        if same_model and same_dataset and same_split and same_shot_mode:
            return record
    return None
