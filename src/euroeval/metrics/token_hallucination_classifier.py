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
            Path to hallucination detection model. Defaults to english.
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
    truncated_samples = 0

    for prompt, predicted_text in zip(dataset["context"], predicted_texts):
        answer, was_truncated = _truncate_answer(
            answer=predicted_text, tokenizer=tokenizer, max_length=max_length
        )
        if was_truncated:
            truncated_samples += 1

        predict_answer = detector.predict_prompt(prompt=prompt, answer=answer)

        for token in predict_answer:
            hallucinated_tokens += token["pred"]
            total_tokens += 1

    if truncated_samples > 0:
        logger.warning(
            f"Truncated the predicted answer of {truncated_samples} sample(s) "
            f"during hallucination detection because it exceeded the detector's "
            f"maximum context length of {max_length} tokens."
        )

    if total_tokens == 0:
        raise InvalidBenchmark(
            "Failed to run hallucination detection task "
            "(there were no tokens found in predictions)."
        )

    hallucination_rate = hallucinated_tokens / total_tokens

    return hallucination_rate


def _truncate_answer(
    answer: str, tokenizer: "PreTrainedTokenizerBase", max_length: int
) -> tuple[str, bool]:
    """Truncate an answer so the detector can fit it alongside the prompt.

    The hallucination detector tokenises the prompt and answer together with
    ``truncation="only_first"``, which only truncates the prompt. If the answer
    alone leaves no room for the prompt (e.g. for reasoning models that emit long
    answers), the tokeniser raises a truncation error. To avoid this we cap the
    answer to at most half of ``max_length`` tokens, leaving room for the prompt.

    Args:
        answer:
            The predicted answer text to potentially truncate.
        tokenizer:
            The detector's tokeniser, used to count and truncate tokens.
        max_length:
            The detector's maximum input sequence length.

    Returns:
        A tuple of the (possibly truncated) answer and whether it was truncated.
    """
    max_answer_tokens = max(1, (max_length - 3) // 2)
    token_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_answer_tokens:
        return answer, False
    truncated_ids = token_ids[:max_answer_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True), True


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
                f"alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-ragtruth-"
                f"{dataset_config.main_language.code}"
            ),
            device=Device(benchmark_config.device.type),
        )
        return hallucination_rate


hallucination_metric = TokenHallucinationMetric(
    name="hallucination_rate", pretty_name="Hallucination rate"
)
