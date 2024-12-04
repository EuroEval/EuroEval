"""Types used throughout the project."""

import typing as t

import numpy as np

if t.TYPE_CHECKING:
    from .data_models import GenerativeModelOutput


ScoreDict = dict[str, dict[str, float] | list[dict[str, float]]]
Predictions = np.ndarray | list[str] | list[list[str]]
Labels = np.ndarray | list[str] | list[list[str]]


class ComputeMetricsFunction(t.Protocol):
    """A function used to compute the metrics."""

    def __call__(
        self,
        model_outputs_and_labels: tuple[
            np.ndarray | list[str] | list[list[str]],
            np.ndarray | list[str] | list[list[str]],
        ],
        id2label: dict[int, str],
    ) -> dict[str, float]:
        """Compute the metrics.

        Args:
            model_outputs_and_labels:
                The model outputs and labels.
            id2label:
                The mapping from label IDs to labels.

        Returns:
            The computed metrics.
        """
        ...


class ExtractLabelsFunction(t.Protocol):
    """A function used to extract the labels from the generated output."""

    def __call__(
        self, input_batch: dict[str, list], model_output: "GenerativeModelOutput"
    ) -> list[str]:
        """Extract the labels from the generated output.

        Args:
            input_batch:
                The input batch.
            model_output:
                The model output.

        Returns:
            The extracted labels.
        """
        ...


def is_list_of_int(x: t.Any) -> t.TypeGuard[list[int]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of integers.
    """
    return isinstance(x, list) and all(isinstance(i, int) for i in x)


def is_list_of_list_of_int(x: t.Any) -> t.TypeGuard[list[list[int]]]:
    """Check if an object is a list of list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of list of integers.
    """
    return (
        isinstance(x, list)
        and all(isinstance(i, list) for i in x)
        and all(isinstance(j, int) for i in x for j in i)
    )


def is_list_of_str(x: t.Any) -> t.TypeGuard[list[str]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of strings.
    """
    return isinstance(x, list) and all(isinstance(i, str) for i in x)
