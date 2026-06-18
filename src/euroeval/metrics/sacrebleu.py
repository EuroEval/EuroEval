"""Metrics from the SacreBLEU package."""

import collections.abc as c
import typing as t

from sacrebleu.metrics import CHRF

from .base import Metric
from .language_detection import language_detector

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig
    from .language_detection import LanguageDetector


class ChrF(Metric):
    """The ChrF metric."""

    def __init__(
        self,
        word_order: int = 0,
        beta: int = 2,
        language_detector: "LanguageDetector | None" = None,
    ) -> None:
        """Initialise the ChrF metric.

        Args:
            word_order (optional):
                The word order for the ChrF metric. Defaults to 0, which is the
                original chrF metric. If set to 2, it is the chrF++ metric.
            beta (optional):
                The beta parameter for the ChrF metric. Defaults to 2, which is the
                original chrF (and chrF++) metric.
            language_detector (optional):
                A LanguageDetector instance. If provided, each per-sentence score is
                multiplied by a binary language penalty (1.0 if the prediction is in
                the correct language, 0.0 otherwise) before averaging. Defaults to
                None, which disables language penalization.
        """
        super().__init__(
            name=f"chr_f{beta}" + "p" * word_order,
            pretty_name=f"ChrF{beta}" + "+" * word_order,
            postprocessing_fn=lambda x: (x, f"{x:.2f}%"),
        )
        self.word_order = word_order
        self.beta = beta
        self.language_detector = language_detector
        self.metric = CHRF(char_order=6, word_order=self.word_order, beta=self.beta)

    def download(self, cache_dir: str) -> "ChrF":
        """Download the language detection model if needed.

        Args:
            cache_dir:
                The directory where the metric will be downloaded to.

        Returns:
            The metric object itself.
        """
        if self.language_detector is not None:
            self.language_detector.download()
        return self

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
            The ChrF score, penalized per-sentence by language correctness.
        """
        scores = [
            self.metric.sentence_score(
                hypothesis=prediction, references=[reference]
            ).score
            for prediction, reference in zip(predictions, references)
        ]

        if not scores:
            return 1.0

        if self.language_detector is not None:
            penalties = self.language_detector(
                predictions=predictions, dataset_config=dataset_config
            )
            scores = [s * p for s, p in zip(scores, penalties)]

        return sum(scores) / len(scores)


chrf2_metric = ChrF(language_detector=language_detector)
chrf3_metric = ChrF(beta=3, language_detector=language_detector)
chrf4_metric = ChrF(beta=4, language_detector=language_detector)
chrf2pp_metric = ChrF(word_order=2, language_detector=language_detector)
chrf3pp_metric = ChrF(word_order=2, beta=3, language_detector=language_detector)
chrf4pp_metric = ChrF(word_order=2, beta=4, language_detector=language_detector)
