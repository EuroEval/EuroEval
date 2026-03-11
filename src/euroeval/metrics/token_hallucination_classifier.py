"""Hallucination metric."""

import collections.abc as c
import typing as t

from lettucedetect import HallucinationDetector

from ..exceptions import InvalidBenchmark
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import DatasetConfig


def detect_hallucinations(
    dataset: "Dataset", predictions: c.Iterable[dict[str, str]], model: str
) -> float:
    """Detect hallucinations using a transformer-based token-level classifier.

    Loads a HallucinationDetector with the given model path and computes the
    hallucination rate over all provided predictions.

    Args:
        dataset:
            Hallucination dataset, generated with, e.g., lettucedetect.
        predictions:
            Sequence of prediction objects, each containing a `prediction_text` field
            with the model's answer text.
        model:
            The Hugging Face model ID of the hallucination detection model.

    Returns:
        A hallucination rate (hallucinated_tokens/total_tokens).

    Raises:
        InvalidBenchmark:
            If the number of predictions does not match the number of dataset examples,
            or if there are no tokens found in predicted answers.
    """
    detector = HallucinationDetector(
        method="transformer", model_path=model, device="cpu"
    )

    predicted_texts = [p["prediction_text"] for p in predictions]

    if len(predicted_texts) != len(dataset):
        raise InvalidBenchmark(
            f"Number of predictions ({len(predicted_texts)}) does not match the "
            f"number of dataset examples ({len(dataset)})."
        )

    hallucinated_tokens = 0
    total_tokens = 0

    for context, question, predicted_text in zip(
        dataset["context"], dataset["question"], predicted_texts
    ):
        predict_answer = detector.predict(
            context=[context], question=question, answer=predicted_text
        )

        for token in predict_answer:
            hallucinated_tokens += token["pred"]
            total_tokens += 1

    if total_tokens == 0:
        raise InvalidBenchmark(
            "Failed to run hallucination detection task "
            "(there were no tokens found in predictions)."
        )

    hallucination_rate = hallucinated_tokens / total_tokens

    return hallucination_rate


class TokenHallucinationMetric(Metric):
    """Hallucination metric."""

    def __init__(self) -> None:
        """Initialise the token hallucination metric."""
        super().__init__(
            name="hallucination_rate",
            pretty_name="Token-level hallucination rate",
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.2f}"),
        )

    def __call__(
        self,
        predictions: c.Iterable[dict[str, str]],
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        **kwargs,
    ) -> float | None:
        """Compute the token-level hallucination rate for a set of predictions.

        This method wraps `detect_hallucinations` to run a token-level
        hallucination detector over the provided predictions and dataset contexts,
        and returns the rate of tokens classified as hallucinated.

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
            The hallucination rate (hallucinated_tokens/total_tokens).
        """
        hallucination_rate = detect_hallucinations(
            dataset=dataset,
            predictions=predictions,
            model=(
                f"alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-"
                f"{dataset_config.main_language.code}"
            ),
        )
        return hallucination_rate


hallucination_metric = TokenHallucinationMetric()
