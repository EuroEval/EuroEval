"""Logic reasoning metrics for puzzles."""

import collections.abc as c
import logging
import typing as t

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class PuzzleLevelAccuracyMetric(Metric):
    """Puzzle-level accuracy metric."""

    def __init__(self) -> None:
        """Initialise the puzzle-level accuracy metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name="puzzle_level_accuracy",
            pretty_name="Puzzle-level Accuracy",
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
        """Compute the puzzle-level accuracy."""
        # TODO: Implement puzzle-level accuracy computation as in
        # task_group_utils/logical_reasoning.py (move it)
        raise NotImplementedError


class CellWiseAccuracyMetric(Metric):
    """Cell-wise accuracy metric."""

    def __init__(self) -> None:
        """Initialise the cell-wise accuracy metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name="cell_wise_accuracy",
            pretty_name="Cell-wise Accuracy",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.2f}"),
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Compute the cell-wise accuracy."""
        # TODO: Implement cell-wise accuracy computation as in
        # task_group_utils/logical_reasoning.py (move it)
        raise NotImplementedError


class BestPermutedCellWiseAccuracyMetric(Metric):
    """Best permuted cell-wise accuracy metric."""

    def __init__(self) -> None:
        """Initialise the best permuted cell-wise accuracy metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name="best_permuted_cell_wise_accuracy",
            pretty_name="Best Permuted Cell-wise Accuracy",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.2f}"),
        )
        self.cell_accuracy_metric = CellWiseAccuracyMetric()

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Compute the best permuted cell-wise accuracy."""
        # TODO: Implement best permuted cell-wise accuracy computation as in
        # task_group_utils/logical_reasoning.py (move it)
        raise NotImplementedError


puzzle_level_accuracy_metric = PuzzleLevelAccuracyMetric()
cell_wise_accuracy_metric = CellWiseAccuracyMetric() 
best_permuted_cell_wise_accuracy_metric = BestPermutedCellWiseAccuracyMetric()