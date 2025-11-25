"""Utility functions related to the text-to-text task group."""

import collections.abc as c
import logging
import typing as t
from copy import deepcopy

import numpy as np
from typeguard import check_type

from ..constants import METRIC_ATTRIBUTES_TAKING_UP_MEMORY
from ..exceptions import InvalidBenchmark
from ..logging_utils import log
from ..metrics import HuggingFaceMetric
from ..utils import (
    extract_json_dict_from_string,
    raise_if_model_output_contains_nan_values,
)

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers.trainer_utils import EvalPrediction

    from ..data_models import BenchmarkConfig, DatasetConfig, GenerativeModelOutput
    from ..types import Labels, Predictions

logger = logging.getLogger("euroeval")

def check_full_type(variable: object, expected_type: t.Type) -> bool:
    """Check if a variable is of the expected type."""
    try:
        check_type(variable, expected_type)
        return True
    except TypeError:
        return False

def compute_metrics(
    model_outputs_and_labels: "tuple[Predictions, Labels] | EvalPrediction",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    dataset: "Dataset",
    json_format: bool = False,
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
        json_format:
            Whether the model outputs and labels are in JSON format.

    Returns:
        A dictionary with the names of the metrics as keys and the metric values as
        values.

    Raises:
        InvalidBenchmark:
            If the metric computation fails.
    """
    model_outputs, raw_labels = model_outputs_and_labels

    # If the model outputs is a pair, then the first element corresponds to the model
    # predictions
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        model_outputs = model_outputs[0]

    assert not isinstance(model_outputs, tuple)
    raise_if_model_output_contains_nan_values(model_output=model_outputs)

    if json_format:
        # Parse the model outputs and create predictions
        if check_full_type(model_outputs, list[dict[str, list[str]]]):
            predictions: list[dict[str, list[str]]] = deepcopy(model_outputs)  # type: ignore[arg-type]
        else:
            predictions = []
            for raw_prediction in model_outputs:
                if not isinstance(raw_prediction, str):
                    logger.warning(
                        "The prediction is not a string. Please ensure that the model "
                        "outputs are parsed correctly."
                    )
                    raw_prediction = str(raw_prediction)
                prediction = extract_json_dict_from_string(s=raw_prediction)
                if not check_full_type(prediction, dict[str, list[str]]):
                    logger.warning(
                        "The prediction string was not converted to a dictionary. "
                        "Please ensure that the model outputs are parsed correctly."
                    )
                    prediction = {"object_1": [f"Invalid prediction: {raw_prediction}"]}
                predictions.append(prediction)  # type: ignore[arg-type]
        # Parse the labels
        if check_full_type(raw_labels, list[dict[str, list[str]]]):
            labels: list[dict[str, list[str]]] = deepcopy(raw_labels)  # type: ignore[arg-type]
        else:
            for raw_label in raw_labels:
                if not isinstance(raw_label, str):
                    raise InvalidBenchmark(
                        "The label is not a string. Please ensure that the labels are "
                        "parsed correctly."
                    )
                label = extract_json_dict_from_string(s=raw_label)
                if not check_full_type(label, dict[str, list[str]]):
                    raise InvalidBenchmark(
                        "The label string was not converted to a dictionary. "
                        "Please ensure that the labels are parsed correctly."
                    )
                labels.append(extract_json_dict_from_string(s=raw_label))  # type: ignore[arg-type]
        
        #TODO: Figure out how to get the results using the structured generation metrics
        # Do we use dataset_config.task.metrics?
        # Should some functions from structured_generation.py be moved to this file?
        results = compare_all_json_predictions_and_labels(predictions=predictions,
                                                          labels=labels)

    else:
        labels = raw_labels
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

            for _ in range(num_attempts := 5):
                try:
                    score: float | None = metric(
                        predictions=predictions,
                        references=labels,
                        dataset=dataset,
                        dataset_config=dataset_config,
                        benchmark_config=benchmark_config,
                    )
                    break
                except Exception as e:
                    oom_error = [
                        "CUDA out of memory",
                        "CUDA error",
                        "MPS backend out of memory",
                    ]
                    if not any(error in str(e) for error in oom_error):
                        raise InvalidBenchmark(str(e)) from e

                    if (
                        isinstance(metric, HuggingFaceMetric)
                        and metric.compute_kwargs.get("device", "cpu") != "cpu"
                    ):
                        metric.compute_kwargs["device"] = "cpu"
                        log(
                            "Out of memory error occurred during the computation of "
                            f"the metric {metric.pretty_name}. Moving the computation to "
                            "the CPU.",
                            level=logging.DEBUG,
                        )
                    else:
                        raise InvalidBenchmark(str(e)) from e
                finally:
                    for attribute in METRIC_ATTRIBUTES_TAKING_UP_MEMORY:
                        if hasattr(metric, attribute):
                            log(
                                f"Deleting the {attribute!r} attribute of the metric "
                                f"{metric.pretty_name} to free up memory.",
                                level=logging.DEBUG,
                            )
                            delattr(metric, attribute)
            else:
                raise InvalidBenchmark(
                    f"Could not compute the metric {metric.pretty_name} after "
                    f"{num_attempts} attempts due to out of memory errors."
                )

            # The metric returns None if we are running on multi-GPU and the current
            # process is not the main process
            if score is not None:
                results[metric.name] = score

    return results


def extract_labels_from_generation(
    input_batch: dict[str, list], model_output: "GenerativeModelOutput"
) -> c.Sequence[t.Any]:
    """Extract the predicted labels from the generated output.

    Args:
        predictions: The model predictions as a list of dictionaries.
        labels: The true labels as a list of dictionaries.

    Returns:
        Results for a metric.
    """
    return model_output.sequences
