"""Bits-per-character metric for EuroEval."""

import collections.abc as c
import logging
import math
import typing as t

from ..logging_utils import log
from .base import Metric

__all__ = ["BitsPerCharacterMetric", "bpc_metric"]

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


class BitsPerCharacterMetric(Metric):
    """Bits-per-character (BPC) metric.

    Returns the average BPC score across all examples. Lower is better.
    Predictions should be a list of BPC scores (positive floats).
    References are ignored as BPC is self-contained.
    """

    def __init__(self) -> None:
        """Initialise the BPC metric."""
        super().__init__(
            name="bits_per_character",
            pretty_name="Bits per character",
            postprocessing_fn=lambda x: (x, f"{x:.4f}"),
        )

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the average BPC score.

        Args:
            predictions:
                List of BPC scores (positive floats).
            references:
                Ignored (BPC is self-contained).
            dataset:
                The dataset used for evaluation (not used).
            dataset_config:
                The dataset configuration (not used).
            benchmark_config:
                The benchmark configuration (not used).

        Returns:
            The average BPC score, or None if there are no finite scores to average.
        """
        if not predictions:
            return None

        # Exclude non-finite scores (e.g. an example whose answer could not be located
        # in the scored tokens yields an infinite BPC) so a single bad example cannot
        # drag the whole dataset's mean to infinity.
        bpc_scores = [float(score) for score in predictions]
        finite_scores = [score for score in bpc_scores if math.isfinite(score)]
        num_dropped = len(bpc_scores) - len(finite_scores)
        if num_dropped > 0:
            log(
                f"Dropped {num_dropped} of {len(bpc_scores)} non-finite "
                "bits-per-character score(s) before averaging.",
                level=logging.WARNING,
            )
        if not finite_scores:
            return None
        return sum(finite_scores) / len(finite_scores)


bpc_metric = BitsPerCharacterMetric()
