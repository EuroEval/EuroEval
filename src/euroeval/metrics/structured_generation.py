"""Logic reasoning metrics for puzzles."""

import collections.abc as c
import itertools
import logging
import typing as t
from copy import deepcopy

import numpy as np
from typeguard import check_type

from ..exceptions import InvalidBenchmark
from ..utils import extract_json_dict_from_string
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class StructuredGenerationMetric(Metric):
    """Base class for structured generation metrics."""

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Compute the metric.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is not used.
            dataset_config:
                The dataset configuration. This is not used.
            benchmark_config:
                The benchmark configuration. This is not used.

        Returns:
            The calculated metric, or None if the score should be ignored.
        """  # noqa: D214, D405, D410, D411
        if not predictions or not references:
            return None
        elif len(predictions) != len(references):
            raise InvalidBenchmark(
                f"The number of predictions ({len(predictions):,}) does not match the "
                f"number of references ({len(references):,})."
            )

        raw_labels = deepcopy(references)
        raw_predictions = deepcopy(predictions)

        # Parse the model outputs and create predictions
        if self._check_full_type(raw_predictions, list[dict[str, list[str]]]):
            formatted_predictions: list[dict[str, list[str]]] = deepcopy(
                raw_predictions
            )  # type: ignore[arg-type]
        else:
            formatted_predictions = []
            for raw_prediction in raw_predictions:
                if not isinstance(raw_prediction, str):
                    logger.warning(
                        "The prediction is not a string. Please ensure that the model "
                        "outputs are parsed correctly."
                    )
                    raw_prediction = str(raw_prediction)
                formatted_prediction = extract_json_dict_from_string(s=raw_prediction)
                if not self._check_full_type(
                    formatted_prediction, dict[str, list[str]]
                ):
                    logger.warning(
                        "The prediction string was not converted to a dictionary. "
                        "Please ensure that the model outputs are parsed correctly."
                    )
                    formatted_prediction = {
                        "object_1": [f"Invalid prediction: {raw_prediction}"]
                    }
                formatted_predictions.append(formatted_prediction)  # type: ignore[arg-type]
        # Parse the labels
        if self._check_full_type(raw_labels, list[dict[str, list[str]]]):
            labels: list[dict[str, list[str]]] = deepcopy(raw_labels)  # type: ignore[arg-type]
        else:
            for raw_label in raw_labels:
                if not isinstance(raw_label, str):
                    raise InvalidBenchmark(
                        "The label is not a string. Please ensure that the labels are "
                        "parsed correctly."
                    )
                label = extract_json_dict_from_string(s=raw_label)
                if not self._check_full_type(label, dict[str, list[str]]):
                    raise InvalidBenchmark(
                        "The label string was not converted to a dictionary. "
                        "Please ensure that the labels are parsed correctly."
                    )
                labels.append(extract_json_dict_from_string(s=raw_label))  # type: ignore[arg-type]

        results = self._compare_all_json_predictions_and_labels(
            predictions=formatted_predictions, labels=labels
        )
        mean_result = float(np.mean(results))
        return mean_result

    @staticmethod
    def _check_full_type(variable: object, expected_type: t.Type) -> bool:
        """Check if a variable is of the expected type."""
        try:
            check_type(variable, expected_type)
            return True
        except TypeError:
            return False

    def _compare_all_json_predictions_and_labels(
        self,
        predictions: list[dict[str, list[str]]],
        labels: list[dict[str, list[str]]],
    ) -> np.ndarray:
        """Compare all JSON predictions and labels.

        Args:
            predictions:
                The model predictions.
            labels:
                The ground truth labels.

        Returns:
            An array with comparison results.
        """
        n_puzzles = len(labels)

        # Initialize scores
        results: np.ndarray = np.zeros(n_puzzles)

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
            results[i] = self._compare_prediction_and_label(prediction, label)

        # Raise error if the metrics are invalid
        if results is None:
            raise InvalidBenchmark("The metric is invalid.")

        return results

    def _compare_prediction_and_label(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Compare a prediction and a label and compute a metric.

        This method must be implemented by subclasses.

        Args:
            prediction:
                The model predictions as a dictionary.
            label:
                The true labels as a dictionary.

        Returns:
            The metric result.
        """
        raise NotImplementedError(
            "Subclasses must implement _compare_prediction_and_label"
        )

    @staticmethod
    def _prepare_data(
        prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> tuple[dict[str, set], dict[str, set], int, int]:
        """Prepare prediction and label data for comparison.

        Args:
            prediction:
                The model predictions as a dictionary.
            label:
                The true labels as a dictionary.

        Returns:
            A tuple containing:
                - prediction_sets: The prediction as a dictionary of sets.
                - label_sets: The label as a dictionary of sets.
                - n_keys: Number of keys in the label.
                - n_elements_per_key: Number of elements per key in the label.
        """
        n_keys = len(label)
        # Get the first item to determine the number of elements per key
        first_key = next(iter(label))
        n_elements_per_key = len(label[first_key])

        # Convert each row to a set of values so the order within the row doesn't matter
        prediction_sets = {
            obj: set(row_attributes) for obj, row_attributes in prediction.items()
        }
        label_sets = {obj: set(row_attributes) for obj, row_attributes in label.items()}

        return prediction_sets, label_sets, n_keys, n_elements_per_key


class PuzzleLevelAccuracyMetric(StructuredGenerationMetric):
    """Puzzle-level accuracy metric."""

    def __init__(self) -> None:
        """Initialise the puzzle-level accuracy metric."""
        super().__init__(
            name="puzzle_level_accuracy",
            pretty_name="Puzzle-level Accuracy",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        )

    def _compare_prediction_and_label(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Compare a prediction and a label and compute the puzzle score.

        Args:
            prediction:
                The model predictions as a dictionary.
            label:
                The true labels as a dictionary.

        Returns:
            The puzzle score.
        """
        prediction_sets, label_sets, _, _ = self._prepare_data(prediction, label)
        return float(
            self._compute_puzzle_score(prediction=prediction_sets, label=label_sets)
        )

    def _compute_puzzle_score(prediction: dict[str, set], label: dict[str, set]) -> int:
        """Compute the puzzle score.

        Args:
            prediction: The prediction as a dictionary.
            label: The label as a dictionary.

        Returns:
            The puzzle score as an integer (1 if correct, 0 otherwise).
        """
        # Sort the prediction and label by object keys to ensure consistent order
        prediction = dict(sorted(prediction.items()))
        label = dict(sorted(label.items()))

        if prediction == label:
            return 1

        # Check if all rows are correct
        for attributes_pred, attributes_label in zip(
            prediction.values(), label.values()
        ):
            # strip whitespace
            attributes_pred = {attr.strip() for attr in attributes_pred}
            attributes_label = {attr.strip() for attr in attributes_label}
            if attributes_pred != attributes_label:
                return 0

        return 1


class CellWiseAccuracyMetric(StructuredGenerationMetric):
    """Cell-wise accuracy metric."""

    def __init__(self) -> None:
        """Initialise the cell-wise accuracy metric."""
        super().__init__(
            name="cell_wise_accuracy",
            pretty_name="Cell-wise Accuracy",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.2f}"),
        )

    def _compare_prediction_and_label(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Compare a prediction and a label and compute the cell score.

        Args:
            prediction:
                The model predictions as a dictionary.
            label:
                The true labels as a dictionary.

        Returns:
            The cell score.
        """
        prediction_sets, label_sets, n_keys, n_elements_per_key = self._prepare_data(
            prediction, label
        )
        return self._compute_cell_score(
            prediction=prediction_sets,
            label=label_sets,
            n_keys=n_keys,
            n_elements_per_key=n_elements_per_key,
        )

    def _compute_cell_score(
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
        """
        # Sort the prediction and label by object keys to ensure consistent order
        prediction = dict(sorted(prediction.items()))
        label = dict(sorted(label.items()))

        if prediction == label:
            return 1.0

        # Compare each cell
        cell_score: float = 0.0
        n_correct_attributes: int = 0
        for attributes_pred, attributes_label in zip(
            prediction.values(), label.values()
        ):
            # strip whitespace
            attributes_pred = {attr.strip() for attr in attributes_pred}
            attributes_label = {attr.strip() for attr in attributes_label}

            # Count the number of correct attributes
            n_correct_attributes += len(attributes_pred.intersection(attributes_label))

        # Normalise the cell score
        cell_score = float(n_correct_attributes) / float(n_keys * n_elements_per_key)

        return cell_score


class BestPermutedCellWiseAccuracyMetric(StructuredGenerationMetric):
    """Best permuted cell-wise accuracy metric."""

    def __init__(self) -> None:
        """Initialise the best permuted cell-wise accuracy metric."""
        super().__init__(
            name="best_permuted_cell_wise_accuracy",
            pretty_name="Best Permuted Cell-wise Accuracy",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.2f}"),
        )
        # Use CellWiseAccuracyMetric's _compute_cell_score method
        self._cell_score_computer = CellWiseAccuracyMetric()

    def _compare_prediction_and_label(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Compare a prediction and a label and compute the best permuted score.

        Args:
            prediction:
                The model predictions as a dictionary.
            label:
                The true labels as a dictionary.

        Returns:
            The best permuted cell score.
        """
        prediction_sets, label_sets, n_keys, n_elements_per_key = self._prepare_data(
            prediction, label
        )
        return self._compute_best_permuted_cell_score(
            prediction=prediction_sets,
            label=label_sets,
            n_keys=n_keys,
            n_elements_per_key=n_elements_per_key,
        )

    def _compute_best_permuted_cell_score(
        self,
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
            permuted_cell_score = self._cell_score_computer._compute_cell_score(
                prediction=prediction_permuted,
                label=label,
                n_keys=n_keys,
                n_elements_per_key=n_elements_per_key,
            )

            # No need to continue if we have a perfect score
            if permuted_cell_score == 1.0:
                return 1.0

            # Update the best permuted cell score
            if permuted_cell_score > best_permuted_cell_score:
                best_permuted_cell_score = permuted_cell_score

        return best_permuted_cell_score


puzzle_level_accuracy_metric = PuzzleLevelAccuracyMetric()
cell_wise_accuracy_metric = CellWiseAccuracyMetric()
best_permuted_cell_wise_accuracy_metric = BestPermutedCellWiseAccuracyMetric()
