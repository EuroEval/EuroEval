"""Hallucination metric."""

import collections.abc as c
import logging
import typing as t

from datasets import Dataset
from lettucedetect import HallucinationDetector

from ..enums import Device
from ..exceptions import InvalidBenchmark
from .base import Metric

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..data_models import BenchmarkConfig, DatasetConfig
def detect_hallucinations(
    dataset: Dataset,
    predictions: c.Iterable[dict[str, str]],
    model: str,
    device: Device = Device.CPU,
) -> float:
    """Load tinylettuce model and detect hallucinations.

    Args:
        dataset: 
            Hallucination dataset, generated with e.g. lettuce. Each example must
            provide a ``"context"`` field containing the full RAG prompt.
        predictions: Iterable of prediction objects, each containing a
            ``"prediction_text"`` field with the model's answer text.
        model: 
            Path to hallucination detection model.
        device: 
            Device to run on ('cpu' or 'cuda').

    Returns:
        A hallucination rate (hallucinated_tokens/total_tokens).

    Raises:
        InvalidBenchmark:
            If there are no tokens found in predicted answers of the
            hallucination detection model.
    """
    detector = HallucinationDetector(
        method="transformer", model_path=model, device=device
    )

    transformer_detector = detector.detector
    tokenizer = transformer_detector.tokenizer
    max_length = transformer_detector.max_length

    predicted_texts = [p["prediction_text"] for p in predictions]

    hallucinated_tokens = 0
    total_tokens = 0
    skipped_samples = 0

    for prompt, predicted_text in zip(dataset["context"], predicted_texts):
        if _answer_too_long(
            answer=predicted_text, tokenizer=tokenizer, max_length=max_length
        ):
            skipped_samples += 1
            continue

        predict_answer = detector.predict_prompt(
            prompt=prompt, answer=predicted_text
        )

        for token in predict_answer:
            hallucinated_tokens += token["pred"]
            total_tokens += 1

    if skipped_samples > 0:
        logger.warning(
            f"Skipped {skipped_samples} sample(s) during hallucination detection "
            f"because the predicted answer alone exceeded the detector's maximum "
            f"context length of {max_length} tokens."
        )

    if total_tokens == 0:
        raise InvalidBenchmark(
            "Failed to run hallucination detection task "
            "(there were no tokens found in predictions)."
        )

    hallucination_rate = hallucinated_tokens / total_tokens

    return hallucination_rate


def _answer_too_long(
    answer: str, tokenizer: "PreTrainedTokenizerBase", max_length: int
) -> bool:
    """Check whether an answer alone exceeds the detector's token budget.

    The hallucination detector tokenises the prompt and answer together with
    ``truncation="only_first"``, which only truncates the prompt. If the answer
    alone leaves no room for the prompt (e.g. for reasoning models that emit long
    answers), the tokeniser raises a truncation error. Such samples are skipped.

    Args:
        answer:
            The predicted answer text to check.
        tokenizer:
            The detector's tokeniser, used to count tokens.
        max_length:
            The detector's maximum input sequence length.

    Returns:
        Whether the answer is too long to be evaluated alongside a prompt.
    """
    answer_token_count = len(
        tokenizer(answer, add_special_tokens=False)["input_ids"]
    )
    # Reserve room for special tokens ([CLS], two [SEP]) and at least one prompt
    # token, matching the detector's ``truncation="only_first"`` requirement.
    return answer_token_count >= max_length - 4


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
        predictions: c.Iterable[dict[str, str]],
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
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
            benchmark_config:
                The benchmark configuration, used to determine the compute device.
            **kwargs:
                For API consistency, this metric accepts other arguments like
                `references`, but they are ignored.

        Returns:
            The hallucination rate (hallucinated_tokens/total_tokens).
        """
        hallucination_rate = detect_hallucinations(
            dataset=dataset,
            predictions=predictions,
            model=(
                f"EuroEval/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-ragtruth-"
                f"{dataset_config.main_language.code}"
            ),
            device=Device(benchmark_config.device.type),
        )
        return hallucination_rate


hallucination_metric = TokenHallucinationMetric(
    name="hallucination_rate", pretty_name="Hallucination rate"
)
