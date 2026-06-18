"""Hallucination metric."""

import collections.abc as c
import logging
import typing as t

from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from lettucedetect import HallucinationDetector

from ..enums import Device
from ..exceptions import InvalidBenchmark
from .base import Metric

logger = logging.getLogger(__name__)

# The underlying mmBERT-small detector models support up to 8192 positions, but
# lettucedetect's ``TransformerDetector`` defaults to a 4096-token budget. Use the
# model's full capacity so fewer samples are skipped for exceeding the budget.
DEFAULT_MAX_INPUT_LENGTH = 8192

if t.TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..data_models import BenchmarkConfig, DatasetConfig


def _hallucination_model_id(dataset_config: "DatasetConfig") -> str:
    """Build the hallucination detection model ID for a dataset.

    Args:
        dataset_config:
            The dataset configuration, whose main language determines the
            language-specific hallucination detection model.

    Returns:
        The Hugging Face Hub repository ID of the hallucination detection model.
    """
    return (
        "EuroEval/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-ragtruth-"
        f"{dataset_config.main_language.code}"
    )


def _hallucination_model_ids() -> set[str]:
    """Collect the model IDs of every dataset using the hallucination metric.

    This enables pre-downloading all language-specific hallucination detection
    models up front (e.g. for offline benchmarking), since the per-dataset language
    is not known inside the metric's ``download`` method.

    Returns:
        The set of Hugging Face Hub repository IDs of all hallucination detection
        models referenced by built-in dataset configurations.
    """
    # Imported here rather than at module level to avoid a circular import, since
    # the dataset configurations import this metric module via the task registry.
    from .. import dataset_configs  # noqa: PLC0415
    from ..data_models import DatasetConfig  # noqa: PLC0415

    model_ids: set[str] = set()
    for obj in vars(dataset_configs).values():
        if not isinstance(obj, DatasetConfig):
            continue
        if any(
            isinstance(metric, TokenHallucinationMetric) for metric in obj.task.metrics
        ):
            model_ids.add(_hallucination_model_id(dataset_config=obj))
    return model_ids


def detect_hallucinations(
    dataset: Dataset,
    predictions: c.Iterable[dict[str, t.Any]],
    model: str,
    device: Device,
    cache_dir: str,
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH,
) -> float:
    """Load model and detect hallucinations.

    Args:
        dataset:
            Hallucination dataset, generated with e.g. lettuce. Each example must
            provide an ``"id"`` field and a ``"context"`` field containing the full
            RAG prompt.
        predictions: Iterable of prediction objects, each containing an ``"id"``
            field matching a dataset example and a ``"prediction_text"`` field with
            the model's answer text.
        model:
            Path to hallucination detection model.
        device:
            Device to run on.
        cache_dir:
            The directory where the detection model is cached. Loading from the same
            directory that ``download`` populates is what enables offline runs.
        max_input_length:
            The maximum combined prompt and answer token budget for the detector.
            Defaults to ``DEFAULT_MAX_INPUT_LENGTH``.

    Returns:
        A hallucination rate (hallucinated_tokens/total_tokens).

    Raises:
        InvalidBenchmark:
            If there are no tokens found in predicted answers of the
            hallucination detection model.
    """
    if not HfApi().repo_exists(repo_id=model):
        raise InvalidBenchmark(
            f"The hallucination detection model {model!r} does not exist on the "
            "Hugging Face Hub."
        )

    detector = HallucinationDetector(
        method="transformer", model_path=model, device=device, cache_dir=cache_dir
    )

    transformer_detector = detector.detector
    # ``HallucinationDetector`` does not forward ``max_length`` to the underlying
    # transformer detector, so override it directly to use the configured budget.
    transformer_detector.max_length = max_input_length
    tokenizer = transformer_detector.tokenizer
    max_length = transformer_detector.max_length

    id_to_context = dict(zip(dataset["id"], dataset["context"]))

    hallucinated_tokens = 0
    total_tokens = 0
    skipped_samples = 0

    for prediction in predictions:
        prompt = id_to_context[prediction["id"]]
        predicted_text = prediction["prediction_text"]

        if _answer_too_long(
            answer=predicted_text, tokenizer=tokenizer, max_length=max_length
        ):
            skipped_samples += 1
            continue

        predict_answer = detector.predict_prompt(prompt=prompt, answer=predicted_text)

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
    answer_token_count = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
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
            postprocessing_fn=lambda raw_score: (raw_score, f"{raw_score:,.4f}"),
        )

    def download(self, cache_dir: str) -> "TokenHallucinationMetric":
        """Pre-download all hallucination detection models.

        The hallucination detection model is language-specific, but the dataset
        language is not available in this method. To support offline benchmarking
        (where only ``download`` is run and inference happens later without network
        access), all hallucination detection models referenced by built-in dataset
        configurations are fetched.

        Args:
            cache_dir:
                The directory where the models will be downloaded to.

        Returns:
            The metric object itself.
        """
        api = HfApi()
        for model_id in _hallucination_model_ids():
            if not api.repo_exists(repo_id=model_id):
                logger.warning(
                    f"The hallucination detection model {model_id!r} does not exist "
                    "on the Hugging Face Hub, skipping download."
                )
                continue
            snapshot_download(repo_id=model_id, repo_type="model", cache_dir=cache_dir)
        return self

    def __call__(
        self,
        predictions: c.Iterable[dict[str, t.Any]],
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Compute the token-level hallucination rate for a set of predictions.

        This method wraps `detect_hallucinations` to run a token-level
        hallucination detector over the provided predictions and dataset contexts,
        and returns the rate of tokens classified as hallucinated.

        Args:
            predictions:
                The model predictions. Each prediction must provide a
                ``"prediction_text"`` field containing the model's answer text.
            references:
                The ground truth references. Unused by this metric, but accepted
                for API consistency with the base ``Metric`` interface.
            dataset:
                The dataset used for evaluation.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration, used to determine the compute device.

        Returns:
            The hallucination rate (hallucinated_tokens/total_tokens).
        """
        hallucination_rate = detect_hallucinations(
            dataset=dataset,
            predictions=predictions,
            model=_hallucination_model_id(dataset_config=dataset_config),
            device=Device(benchmark_config.device.type),
            cache_dir=benchmark_config.cache_dir,
        )
        return hallucination_rate


hallucination_metric = TokenHallucinationMetric(
    name="hallucination_rate", pretty_name="Hallucination rate"
)
