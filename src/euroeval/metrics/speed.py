"""Inference speed metric."""

import collections.abc as c
import logging
import typing as t

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

logger: logging.Logger = logging.getLogger("euroeval")


class SpeedMetric(Metric):
    """Speed metric."""

    def __init__(self, name: str, pretty_name: str) -> None:
        """Initialise the speed metric.

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
        """Not used with the speed metric, but required for consistency."""
        raise NotImplementedError


speed_metric = SpeedMetric(name="speed", pretty_name="Tokens per second")

speed_short_metric = SpeedMetric(
    name="speed_short", pretty_name="Tokens per second on short documents"
)
