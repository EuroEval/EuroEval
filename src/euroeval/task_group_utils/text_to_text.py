"""Utility functions related to the text-to-text task group."""

import logging
import typing as t

import numpy as np

from ..constants import METRIC_ATTRIBUTES_TAKING_UP_MEMORY
from ..exceptions import InvalidBenchmark
from ..metrics import HuggingFaceMetric
from ..utils import raise_if_model_output_contains_nan_values

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers.trainer_utils import EvalPrediction

    from ..data_models import BenchmarkConfig, DatasetConfig, GenerativeModelOutput
    from ..types import Labels, Predictions


logger = logging.getLogger("euroeval")


def compute_metrics(
    model_outputs_and_labels: "tuple[Predictions, Labels] | EvalPrediction",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    dataset: "Dataset",
) -> dict[str, float]:
    """Compute the metrics needed for evaluation.

    Args:
        model_outputs_and_labels:
            The first sequence contains the model outputs and the second sequence
            contains the true labels.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
        dataset:
            The dataset used for evaluation. This is only used in case any additional
            metadata is used to compute the metrics.

    Returns:
        A dictionary with the names of the metrics as keys and the metric values as
        values.
    """
    model_outputs, labels = model_outputs_and_labels

    # If the model outputs is a pair, then the first element corresponds to the model
    # predictions
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        model_outputs = model_outputs[0]

    assert not isinstance(model_outputs, tuple)
    raise_if_model_output_contains_nan_values(model_output=model_outputs)

    model_output_dtype = np.asarray(model_outputs).dtype
    output_is_prob = model_output_dtype in [np.float16, np.float32, np.float64]
    if output_is_prob:
        predictions = np.asarray(model_outputs).argmax(axis=-1)
    else:
        predictions = model_outputs

    results: dict[str, float] = dict()
    for metric in dataset_config.task.metrics:
        # Some metrics can be computed on hardware accelerators. In this case we
        # start by setting the device to the same device as the model
        if (
            isinstance(metric, HuggingFaceMetric)
            and metric.compute_kwargs.get("device", None) == "auto"
        ):
            metric.compute_kwargs["device"] = benchmark_config.device.type

        while True:
            try:
                score: float | None = metric(
                    predictions=predictions, references=labels, dataset=dataset
                )
                break
            except Exception as e:
                oom_error = [
                    "CUDA out of memory",
                    "CUDA error",
                    "MPS backend out of memory",
                ]
                if not any(error in str(e) for error in oom_error):
                    raise InvalidBenchmark(str(e))

                if (
                    isinstance(metric, HuggingFaceMetric)
                    and metric.compute_kwargs.get("device", "cpu") != "cpu"
                ):
                    metric.compute_kwargs["device"] = "cpu"
                    logger.debug(
                        "Out of memory error occurred during the computation of "
                        f"the metric {metric.pretty_name}. Moving the computation to "
                        "the CPU."
                    )
                else:
                    raise InvalidBenchmark(str(e))
            finally:
                for attribute in METRIC_ATTRIBUTES_TAKING_UP_MEMORY:
                    if hasattr(metric, attribute):
                        logger.debug(
                            f"Deleting the {attribute!r} attribute of the metric "
                            f"{metric.pretty_name} to free up memory."
                        )
                        delattr(metric, attribute)

        # The metric returns None if we are running on multi-GPU and the current
        # process is not the main process
        if score is not None:
            results[metric.name] = score

    return results


def extract_labels_from_generation(
    input_batch: dict[str, list], model_output: "GenerativeModelOutput"
) -> list[t.Any]:
    """Extract the predicted labels from the generated output.

    Args:
        input_batch:
            The input batch, where the keys are the feature names and the values
            are lists with the feature values.
        model_output:
            The raw generated output of the model.

    Returns:
        The predicted labels.
    """
    return model_output.sequences
