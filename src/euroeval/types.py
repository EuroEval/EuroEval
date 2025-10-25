"""Types used throughout the project."""

import collections.abc as c
import typing as t

from transformers.trainer_utils import EvalPrediction

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from numpy.typing import NDArray

    from .data_models import BenchmarkConfig, GenerativeModelOutput

ScoreDict: t.TypeAlias = dict[str, dict[str, float] | c.Sequence[dict[str, float]]]
Predictions: t.TypeAlias = "NDArray | c.Sequence[str] | c.Sequence[c.Sequence[str]]"
Labels: t.TypeAlias = "NDArray | c.Sequence[str] | c.Sequence[c.Sequence[str]]"


class ComputeMetricsFunction(t.Protocol):
    """A function used to compute the metrics."""

    def __call__(
        self,
        model_outputs_and_labels: EvalPrediction
        | tuple[
            "NDArray | c.Sequence[str] | c.Sequence[c.Sequence[str]]",
            "NDArray | c.Sequence[str] | c.Sequence[c.Sequence[str]]",
        ],
        dataset: "Dataset",
        benchmark_config: "BenchmarkConfig",
    ) -> dict[str, float]:
        """Compute the metrics.

        Args:
            model_outputs_and_labels:
                The model outputs and labels.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.

        Returns:
            The computed metrics.
        """
        ...


class ExtractLabelsFunction(t.Protocol):
    """A function used to extract the labels from the generated output."""

    def __call__(
        self, input_batch: dict[str, list], model_output: "GenerativeModelOutput"
    ) -> c.Sequence[str]:
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


def is_list_of_int(x: object) -> t.TypeGuard[c.Sequence[int]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of integers.
    """
    return isinstance(x, list) and all(isinstance(i, int) for i in x)


def is_list_of_list_of_int(x: object) -> t.TypeGuard[c.Sequence[c.Sequence[int]]]:
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


def is_list_of_str(x: object) -> t.TypeGuard[c.Sequence[str]]:
    """Check if an object is a list of integers.

    Args:
        x:
            The object to check.

    Returns:
        Whether the object is a list of strings.
    """
    return isinstance(x, list) and all(isinstance(i, str) for i in x)
