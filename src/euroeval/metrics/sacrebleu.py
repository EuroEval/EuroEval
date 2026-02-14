"""Metrics from the SacreBLEU package."""

import collections.abc as c
import typing as t

from sacrebleu.metrics import CHRF

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


class ChrF(Metric):
    """The ChrF metric."""

    def __init__(self, word_order: int = 0, beta: int = 2) -> None:
        """Initialise the ChrF++ metric.

        Args:
            word_order (optional):
                The word order for the ChrF++ metric. Defaults to 0, which is the
                original chrF metric. If set to 2, it is the chrF++ metric.
            beta (optional):
                The beta parameter for the ChrF++ metric. Defaults to 2, which is the
                original chrF (and chrF++) metric.
        """
        super().__init__(
            name=f"chr_f{beta}" + "p" * word_order,
            pretty_name=f"ChrF{beta}" + "+" * word_order,
            postprocessing_fn=lambda x: (x, f"{x:.2f}%"),
        )
        self.word_order = word_order
        self.beta = beta
        self.metric = CHRF(char_order=6, word_order=self.word_order, beta=self.beta)

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the ChrF score.

        Args:
            predictions:
                The predictions of the model.
            references:
                The references for the predictions.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The ChrF score.
        """
        return self.metric.corpus_score(
            hypotheses=predictions, references=[[ref] for ref in references]
        ).score


chrf_metric = ChrF()
chrf4_metric = ChrF(beta=4)
chrfpp_metric = ChrF(word_order=2)
chrf4pp_metric = ChrF(word_order=2, beta=4)
