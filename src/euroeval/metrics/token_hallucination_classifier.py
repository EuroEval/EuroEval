"""Hallucination metric."""

import collections.abc as c
import logging
import typing as t

from datasets import Dataset
from lettucedetect import HallucinationDetector

from .base import Metric

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import DatasetConfig


def detect_hallucinations(
    dataset: Dataset,
    predictions: c.Sequence,
    model: str = "alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-en",
    device: str = "cpu",
) -> float | None:
    """Load tinylettuce model and detect hallucinations.

    Args:
        dataset: Hallucination dataset, generated with e.g. lettuce.
        model: Path to model.
        device: Device to run on ('cpu' or 'cuda').

    Returns:
        A hallucination rate (hallucinated_tokens/total_tokens).
    """
    detector = HallucinationDetector(
        method="transformer", model_path=model, device=device
    )

    predicted_texts = [p["prediction_text"] for p in predictions]

    hallucinated_tokens = 0
    total_tokens = 0

    for context, question, predicted_text in zip(
        dataset["context"], dataset["question"], predicted_texts
    ):
        try:
            predict_answer = detector.predict(
                context=[context], question=question, answer=predicted_text
            )

            for token in predict_answer:
                hallucinated_tokens += token["pred"]
                total_tokens += 1

        except Exception:
            logger.exception("Error during hallucination detection. Skipping.")
            continue

    if total_tokens == 0:
        logger.warning(
            "Failed to run hallucination detection task "
            "(there were no tokens found in predictions), returning None."
        )
        return None

    hallucination_rate = hallucinated_tokens / total_tokens

    logger.debug(
        f"Hallucination rate (hallucinated_tokens/total_tokens): "
        f"{hallucination_rate:.2f}"
    )
    return hallucination_rate


class TokenHallucinationMetric(Metric):
    """Hallucination metric."""

    def __init__(self, name: str, pretty_name: str) -> None:
        """Initialise the token hallucination metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
        """
        super().__init__(
            name=name,
            pretty_name=pretty_name,
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.2f}"),
        )

    def __call__(
        self,
        predictions: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        **kwargs,
    ) -> float | None:
        """Compute the token-level hallucination rate for a set of predictions.

        This method wraps :func:`detect_hallucinations` to run a token-level
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
        """
        hallucination_rate = detect_hallucinations(
            dataset=dataset,
            predictions=predictions,
            model="alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-"
            + dataset_config.main_language.code,
            device="cpu",
        )
        return hallucination_rate


hallucination_metric = TokenHallucinationMetric(
    name="hallucination_token", pretty_name="Hallucination rate"
)
