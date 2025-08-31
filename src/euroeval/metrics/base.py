"""The abstract base class for all metrics."""

import abc
import collections.abc as c
import logging
import typing as t

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class Metric(abc.ABC):
    """Abstract base class for all metrics."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        """Initialise the metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            postprocessing_fn:
                A function to apply to the metric scores after they are computed,
                taking the score to the postprocessed score along with its string
                representation. Defaults to x -> (100 * x, f"{x:.2%}").
        """
        self.name = name
        self.pretty_name = pretty_name
        self.postprocessing_fn = (
            postprocessing_fn
            if postprocessing_fn is not None
            else lambda x: (100 * x, f"{x:.2%}")
        )

    @abc.abstractmethod
    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the metric score.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The calculated metric score, or None if the score should be ignored.
        """
        ...

    def __hash__(self) -> int:
        """Return a hash of the metric configuration."""
        return hash(self.name)
