"""Hallucination metric."""

from __future__ import annotations

import collections.abc as c
import logging
import typing as t
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from lettucedetect import HallucinationDetector

from ..constants import MAX_CONTEXT_LENGTH
from ..enums import Device
from ..exceptions import InvalidBenchmark
from ..logging_utils import no_terminal_output
from .base import Metric

logger = logging.getLogger(__name__)


if t.TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from ..data_models import BenchmarkConfig, DatasetConfig


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

    def download(
        self, cache_dir: str, dataset_config: "DatasetConfig" | None = None
    ) -> "TokenHallucinationMetric":
        """Pre-download hallucination detection models.

        The hallucination detection model is language-specific. When a dataset
        configuration is provided, only the model(s) for the relevant language(s)
        are downloaded. Otherwise, all hallucination detection models referenced
        by built-in dataset configurations are fetched for offline benchmarking.

        Args:
            cache_dir:
                The directory where the models will be downloaded to.
            dataset_config (optional):
                The dataset configuration, used to filter which hallucination
                detection models to download based on language. When None, all
                models are downloaded. Defaults to None.

        Returns:
            The metric object itself.
        """
        for model_id in _hallucination_model_ids(
            cache_dir=cache_dir, dataset_config=dataset_config
        ):
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


def _hallucination_model_ids(
    cache_dir: str, dataset_config: "DatasetConfig" | None = None
) -> set[str]:
    """Collect the model IDs of datasets using the hallucination metric.

    When a dataset configuration is provided, returns only the model ID(s) for
    the relevant language(s). Otherwise, scans all built-in dataset configurations
    and returns all hallucination detection model IDs for offline benchmarking.

    Args:
        cache_dir:
            The directory to store the dataset configuration cache in.
        dataset_config (optional):
            The dataset configuration to filter by language. When provided,
            extracts language(s) from ``main_language`` and returns only the
            corresponding model ID(s). Defaults to None.

    Returns:
        The set of Hugging Face Hub repository IDs of hallucination detection
        models. If ``dataset_config`` is provided, contains only the model for
        the relevant language (the target language for translation tasks).
        Otherwise, contains all models referenced by built-in dataset
        configurations.
    """
    if dataset_config is not None:
        # Extract language(s) from the provided dataset configuration
        main_language = dataset_config.main_language
        if isinstance(main_language, tuple):
            # Translation task: use target language (second element) since that's
            # what the model outputs and what we check for hallucinations
            return {
                "EuroEval/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-ragtruth-"
                f"{main_language[1].code}"
            }
        else:
            # Single language
            return {
                "EuroEval/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-ragtruth-"
                f"{main_language.code}"
            }

    # Imported here rather than at module level to avoid a circular import, since
    # the dataset configurations import this metric module via the task registry.
    from ..dataset_configs import get_all_dataset_configs  # noqa: PLC0415

    dataset_configs = get_all_dataset_configs(
        custom_datasets_file=Path("custom_datasets.py"),
        dataset_ids=[],
        api_key=None,
        cache_dir=Path(cache_dir),
        trust_remote_code=False,
        run_with_cli=False,
    )
    model_ids: set[str] = set()
    for dataset_config in dataset_configs.values():
        if any(
            isinstance(metric, TokenHallucinationMetric)
            for metric in dataset_config.task.metrics
        ):
            model_ids.add(_hallucination_model_id(dataset_config=dataset_config))
    return model_ids


def detect_hallucinations(
    dataset: Dataset,
    predictions: c.Iterable[dict[str, t.Any]],
    model: str,
    device: Device,
    cache_dir: str,
) -> float:
    """Load model and detect hallucinations.

    Args:
        dataset:
            Hallucination dataset, generated with e.g. lettucedetect. Each example must
            provide an ``"id"`` field and a ``"context"`` field containing the full
            RAG prompt.
        predictions:
            Iterable of prediction objects, each containing an ``"id"``
            field matching a dataset example and a ``"prediction_text"`` field with
            the model's answer text.
        model:
            Path to hallucination detection model.
        device:
            Device to run on.
        cache_dir:
            The directory where the detection model is cached. Loading from the same
            directory that ``download`` populates is what enables offline runs.

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

    # Suppress the verbose "Loading weights" progress bars from transformers
    with no_terminal_output():
        detector = HallucinationDetector(
            method="transformer", model_path=model, device=device, cache_dir=cache_dir
        )

        transformer_detector = detector.detector
        # ``HallucinationDetector`` does not forward ``max_length`` to the underlying
        # transformer detector, so override it directly to use the configured budget.
        transformer_detector.max_length = MAX_CONTEXT_LENGTH
        tokenizer = transformer_detector.tokenizer

    id_to_context = dict(zip(dataset["id"], dataset["context"]))

    hallucinated_tokens = 0
    total_tokens = 0
    skipped_samples = 0

    for prediction in predictions:
        prompt = id_to_context[prediction["id"]]
        predicted_text = prediction["prediction_text"]

        if _answer_too_long(
            answer=predicted_text, tokenizer=tokenizer, max_length=MAX_CONTEXT_LENGTH
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
            f"context length of {MAX_CONTEXT_LENGTH} tokens."
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


hallucination_metric = TokenHallucinationMetric(
    name="hallucination_rate", pretty_name="Hallucination rate"
)
