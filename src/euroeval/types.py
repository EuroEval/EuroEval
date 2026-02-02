"""Types used throughout the project."""

import collections.abc as c
import typing as t

from transformers import PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction

try:
    from transformers.tokenization_mistral_common import MistralCommonTokenizer
except ImportError:
    from transformers.tokenization_mistral_common import (
        MistralCommonBackend as MistralCommonTokenizer,
    )


if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from numpy.typing import NDArray
    from pydantic import BaseModel

    from .data_models import BenchmarkConfig, GenerativeModelOutput

ScoreDict: t.TypeAlias = dict[str, dict[str, float] | c.Sequence[dict[str, float]]]
Predictions: t.TypeAlias = "NDArray | c.Sequence[str] | c.Sequence[c.Sequence[str]]"
Labels: t.TypeAlias = "NDArray | c.Sequence[str] | c.Sequence[c.Sequence[str]]"
Tokeniser: t.TypeAlias = PreTrainedTokenizer | MistralCommonTokenizer


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


class ScoringFunction(t.Protocol):
    """A function used to compute a score from a single model output."""

    def __call__(self, output: "BaseModel") -> float:
        """Compute a score from a model output.

        Args:
            output:
                A model output (Pydantic model) from the judge.

        Returns:
            A float score computed from the output.
        """
        ...


class BatchScoringFunction(t.Protocol):
    """A function used to compute batch scores from model outputs."""

    def __call__(
        self, outputs: list["BaseModel"], dataset: "Dataset | None" = None
    ) -> float:
        """Compute a batch score from model outputs.

        Args:
            outputs:
                List of model outputs (Pydantic models) from the judge.
            dataset:
                Optional dataset used for evaluation. Can be used for additional
                context when computing the score.

        Returns:
            A float score computed from the batch of outputs.
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
