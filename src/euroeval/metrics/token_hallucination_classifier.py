"""Hallucination metric."""

import collections.abc as c
import typing as t

from lettucedetect import HallucinationDetector
from tqdm.auto import tqdm

from ..exceptions import InvalidBenchmark
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import DatasetConfig


class HallucinationRateMetric(Metric):
    """Hallucination metric."""

    def __init__(self) -> None:
        """Initialise the hallucination metric."""
        super().__init__(
            name="hallucination_rate",
            pretty_name="Hallucination rate",
            postprocessing_fn=None,
        )
        self.detector: HallucinationDetector | None = None

    def __call__(
        self,
        predictions: c.Iterable[dict[str, str]],
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        **kwargs,
    ) -> float | None:
        """Compute the hallucination rate for a set of predictions.

        Args:
            predictions:
                The model predictions. Each prediction must provide a
                ``"prediction_text"`` field containing the model's answer text.
            dataset:
                The dataset used for evaluation.
            dataset_config:
                The dataset configuration.
            **kwargs:
                For API consistency, this metric accepts other arguments like
                `references` and `benchmark_config`, but they are ignored.

        Returns:
            The hallucination rate.

        Raises:
            InvalidBenchmark:
                If the number of predictions does not match the number of dataset
                examples.
        """
        if self.detector is None:
            model_id = (
                f"alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-"
                f"{dataset_config.main_language.code}"
            )
            self.detector = HallucinationDetector(
                method="transformer", model_path=model_id, device="cpu"
            )
        assert self.detector is not None, "The detector should not be None but it is!"

        predicted_texts = [p["prediction_text"] for p in predictions]
        if len(predicted_texts) == 0:
            return 0.0

        if len(predicted_texts) != len(dataset):
            raise InvalidBenchmark(
                f"Number of predictions ({len(predicted_texts)}) does not match the "
                f"number of dataset examples ({len(dataset)})."
            )

        hallucinated_samples = 0
        for context, question, predicted_text in zip(
            dataset["context"],
            dataset["question"],
            tqdm(iterable=predicted_texts, desc="Detect hallucinations", leave=False),
        ):
            predict_answer = self.detector.predict(
                context=[context], question=question, answer=predicted_text
            )
            hallucinated_samples += max(token["pred"] for token in predict_answer)

        return hallucinated_samples / len(predicted_texts)


hallucination_rate_metric = HallucinationRateMetric()
