"""Logic reasoning metrics for puzzles."""

import collections.abc as c
import logging
import typing as t

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class PuzzleMetric(Metric):
    """Puzzle metric."""

    def __init__(self, name: str, pretty_name: str) -> None:
        """Initialise the puzzle metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name=name,
            pretty_name=pretty_name,
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.0f}"),
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Not used with the puzzle metric, but required for consistency.

        TODO: How are predictions used?
        """
        raise NotImplementedError


puzzle_level_accuracy_metric = PuzzleMetric(name="puzzle_level_accuracy",
                                     pretty_name="Puzzle Level Accuracy")

cell_wise_accuracy_metric = PuzzleMetric(name="cell_wise_accuracy",
                                  pretty_name="Cell Wise Accuracy")

best_permuted_cell_wise_accuracy_metric = PuzzleMetric(
    name="best_permuted_cell_wise_accuracy",
    pretty_name="Best Permuted Cell Wise Accuracy"
)
