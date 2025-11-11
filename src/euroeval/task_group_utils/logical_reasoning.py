"""Utility functions related to the logical-reasoning task group."""

import itertools
import logging
import typing as t
from copy import deepcopy
from typeguard import check_type

import numpy as np

from ..exceptions import InvalidBenchmark
from ..utils import (
    extract_json_dict_from_string,
    raise_if_model_output_contains_nan_values,
)

if t.TYPE_CHECKING:
    from transformers.trainer_utils import EvalPrediction

    from ..types import Labels, Predictions


logger = logging.getLogger("euroeval")

def check_full_type(variable, expected_type) -> bool:
    try:
        check_type(variable, expected_type)
        return True
    except TypeError:
        return False

def compute_metrics(
    model_outputs_and_labels: "tuple[Predictions, Labels] | EvalPrediction",
) -> dict[str, float]:
    """Compute the metrics needed for evaluation.

    Args:
        model_outputs_and_labels:
            The first array contains the model response and the second
            array contains the true labels. Both are JSON dictionaries.
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
    model_outputs, raw_labels = model_outputs_and_labels

    if len(raw_labels) != len(model_outputs):
        raise InvalidBenchmark(
            "The number of model outputs does not match the number of labels."
        )

    # If the model outputs is a pair, then the first element corresponds to the model
    # predictions
    if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
        model_outputs = model_outputs[0]

    # Parse the model outputs

    if check_full_type(model_outputs, list[dict[str, list[str]]]):
        predictions: list[dict[str, list[str]]] = deepcopy(model_outputs)
    else:
        for raw_prediction in model_outputs:
            if not isinstance(raw_prediction, str):
                raise InvalidBenchmark(
                    "The model output is not a string. Please ensure that the model "
                    "outputs are parsed correctly."
                )
            predictions.append(extract_json_dict_from_string(s=raw_prediction))

    raise_if_model_output_contains_nan_values(model_output=predictions)

    # Parse the labels
    if check_full_type(raw_labels, list[dict[str, list[str]]]):
        labels: list[dict[str, list[str]]] = deepcopy(raw_labels)
    else:
        for raw_label in raw_labels:
            if not isinstance(raw_label, str):
                raise InvalidBenchmark(
                    "The label is not a string. Please ensure that the labels are "
                    "parsed correctly."
                )
            labels.append(extract_json_dict_from_string(s=raw_label))

    # Compute the metrics
    puzzle_scores, cell_scores, best_permuted_cell_scores = (
        compare_all_predictions_and_labels(predictions=predictions, labels=labels)
    )

    return dict(
        puzzle_level_accuracy=puzzle_scores,
        cell_wise_accuracy=cell_scores,
        best_permuted_cell_wise_accuracy=best_permuted_cell_scores,
    )


def compare_all_predictions_and_labels(
    predictions: list[dict[str, list[str]]], labels: list[dict[str, list[str]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare all predictions and labels and compute the metrics.

    Args:
        predictions: The model predictions as a list of dictionaries.
        labels: The true labels as a list of dictionaries.

    Returns:
        A tuple containing the puzzle scores, cell scores and best permuted cell scores.
    """
    n_puzzles = len(labels)

    # Initialize scores
    puzzle_scores: np.ndarray = np.zeros(n_puzzles)
    cell_scores: np.ndarray = np.zeros(n_puzzles)
    best_permuted_cell_scores: np.ndarray = np.zeros(n_puzzles)

    # Compute the metrics
    for i, (prediction, label) in enumerate(zip(predictions, labels)):
        if not isinstance(prediction, dict):
            raise InvalidBenchmark(
                "The model output is not a dictionary. Please ensure that the "
                "model outputs are parsed correctly."
            )
        if not isinstance(label, dict):
            raise InvalidBenchmark(
                "The label is not a dictionary. Please ensure that the labels are "
                "parsed correctly."
            )
        puzzle_score, cell_score, best_permuted_cell_score = (
            compare_prediction_and_label(prediction, label)
        )
        puzzle_scores[i] = puzzle_score
        cell_scores[i] = cell_score
        best_permuted_cell_scores[i] = best_permuted_cell_score

    # Raise error if the metrics are invalid
    if (
        puzzle_scores is None
        or cell_scores is None
        or best_permuted_cell_scores is None
    ):
        raise InvalidBenchmark("The metrics are invalid.")

    return puzzle_scores, cell_scores, best_permuted_cell_scores


def compare_prediction_and_label(
    prediction: dict[str, list[str]], label: dict[str, list[str]]
) -> tuple[int, float, float]:
    """Compare a prediction and a label and compute the metrics.

    Args:
        prediction:
            The model predictions as a dictionary.
        label:
            The true labels as a dictionary.

    Returns:
        A tuple containing the puzzle score, cell score and best permuted cell score.
    """
    n_keys = len(label)
    n_elements_per_key = len(label[0])

    # Convert each row to a set of values so the order within the row does not matter
    prediction_sets = {
        obj: set(row_attributes)
        for obj, row_attributes in zip(prediction.keys(), prediction.values())
    }
    label_sets = {
        obj: set(row_attributes)
        for obj, row_attributes in zip(label.keys(), label.values())
    }

    # Compare the full prediction to the full label
    if prediction_sets == label_sets:
        puzzle_score = 1
        cell_score = 1.0
        best_permuted_cell_score = 1.0
        return puzzle_score, cell_score, best_permuted_cell_score

    # Compare all cells
    cell_score = compute_cell_score(
        prediction=prediction_sets,
        label=label_sets,
        n_keys=n_keys,
        n_elements_per_key=n_elements_per_key,
    )

    # Check if the puzzle is solved after stripping whitespace in cells
    if cell_score == 1:
        puzzle_score = 1
        best_permuted_cell_score = 1.0
    else:
        puzzle_score = 0

        # Evaluate every permutation of the objects in the response
        best_permuted_cell_score = compute_best_permuted_cell_score(
            prediction=prediction_sets,
            label=label_sets,
            n_keys=n_keys,
            n_elements_per_key=n_elements_per_key,
        )
    return puzzle_score, cell_score, best_permuted_cell_score


def compute_cell_score(
    prediction: dict[str, set],
    label: dict[str, set],
    n_keys: int,
    n_elements_per_key: int,
) -> float:
    """Compute the cell score.

    Args:
        prediction: The prediction as a dictionary.
        label: The label as a dictionary.
        n_keys: Number of keys in the label.
        n_elements_per_key: Number of elements per key in the label.

    Returns:
        The cell score as a float.

    # TODO: Make sure the order of attributes in each object does not matter
    """
    # Sort the prediction and label by object keys to ensure consistent order
    prediction = dict(sorted(prediction.items()))
    label = dict(sorted(label.items()))

    # Compare each cell
    cell_score: float = 0.0
    n_correct_attributes: int = 0
    for attributes_pred, attributes_label in zip(prediction.values(), label.values()):
        # strip whitespace
        attributes_pred = {attr.strip() for attr in attributes_pred}
        attributes_label = {attr.strip() for attr in attributes_label}

        # Count the number of correct attributes
        n_correct_attributes += len(attributes_pred.intersection(attributes_label))

    # Normalise the cell score
    cell_score = float(n_correct_attributes) / float(n_keys * n_elements_per_key)

    return cell_score


def compute_best_permuted_cell_score(
    prediction: dict[str, set],
    label: dict[str, set],
    n_keys: int,
    n_elements_per_key: int,
) -> float:
    """Compute the best permuted cell score.

    Args:
        prediction: The prediction as a dictionary.
        label: The label as a dictionary.
        n_keys: Number of keys in the label.
        n_elements_per_key: Number of elements per key in the label.

    Returns:
        The best permuted cell score as a float.
    """
    best_permuted_cell_score = 0.0
    objects = list(prediction.keys())

    # Create all permutations of the objects where each object appears exactly once

    object_permutations = list(itertools.permutations(objects))

    # Evaluate each permutation
    for object_permutation in object_permutations:
        # Create a new prediction with the objects permuted
        prediction_permuted = {
            obj: prediction[perm_obj]
            for obj, perm_obj in zip(objects, object_permutation)
        }

        # Compare the permuted prediction to the label
        permuted_cell_score = compute_cell_score(
            prediction=prediction_permuted,
            label=label,
            n_keys=n_keys,
            n_elements_per_key=n_elements_per_key,
        )

        # Update the best permuted cell score
        if permuted_cell_score > best_permuted_cell_score:
            best_permuted_cell_score = permuted_cell_score

    return best_permuted_cell_score
