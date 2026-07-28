"""Shared utility functions for task group metric computation."""

import typing as t

import numpy as np

from ..types import Predictions
from ..utils import raise_if_model_output_contains_nan_values

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers.trainer_utils import EvalPrediction

    from ..data_models import BenchmarkConfig, DatasetConfig
    from ..types import Labels


def compute_simple_metrics(
    predictions: "Predictions",
    references: "Labels",
    dataset: "Dataset",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> dict[str, float]:
    """Compute metrics using the standard metric loop.

    Iterates over all metrics defined in the dataset configuration, calls each
    metric with the provided predictions and references, and collects non-None
    scores.

    Args:
        predictions:
            The model predictions.
        references:
            The ground truth references.
        dataset:
            The dataset used for evaluation. This is only used in case any
            additional metadata is needed to compute the metrics.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.

    Returns:
        A dictionary with the names of the metrics as keys and the metric values
        as values. Only includes metrics that returned a non-None score.
    """
    results: dict[str, float] = dict()
    for metric in dataset_config.task.metrics:
        score: float | None = metric(
            predictions=predictions,  # ty: ignore[invalid-argument-type]
            references=references,  # ty: ignore[invalid-argument-type]
            dataset=dataset,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if score is not None:
            results[metric.name] = score

    return results


def normalise_model_outputs(
    model_outputs_and_labels: "tuple[Predictions, Labels] | EvalPrediction",
) -> tuple["Predictions", "Labels", "Predictions"]:
    """Normalise model outputs and compute predictions.

    Performs shared preprocessing: unpacks outputs, handles tuple-wrapped
    predictions, checks for NaN values, and converts probability distributions
    to class predictions via argmax when needed.

    Args:
        model_outputs_and_labels:
            The first sequence contains the model outputs and the second sequence
            contains the true labels.

    Returns:
        A tuple containing:
        - model_outputs: The squeezed model outputs (first element if tuple).
        - labels: The true labels.
        - predictions: Either the raw outputs (if integer type) or the argmax
          across the last axis (if float type).
    """
    model_outputs, labels = model_outputs_and_labels

    # If the model outputs is a pair, then the first element corresponds to the model
    # predictions
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        model_outputs = model_outputs[0]

    raise_if_model_output_contains_nan_values(model_output=model_outputs)

    model_output_dtype = np.asarray(model_outputs).dtype
    if model_output_dtype in [np.float16, np.float32, np.float64]:
        predictions = np.asarray(model_outputs).argmax(axis=-1)
    else:
        predictions = model_outputs

    return model_outputs, labels, predictions
