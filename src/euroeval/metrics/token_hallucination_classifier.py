"""Hallucination metric."""

from __future__ import annotations

import collections.abc as c
import dataclasses
import logging
import typing as t
import weakref
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


@dataclasses.dataclass(frozen=True)
class _HallucinationDetection:
    """Token-level detector output grouped by evaluated sample."""

    token_predictions: tuple[tuple[int, ...], ...]

    @property
    def hallucinated_tokens(self) -> int:
        """Number of hallucinated tokens."""
        return sum(sum(tokens) for tokens in self.token_predictions)

    @property
    def total_tokens(self) -> int:
        """Number of detected tokens."""
        return sum(len(tokens) for tokens in self.token_predictions)


_DetectionCacheKey = tuple[str, Device, str, tuple[tuple[str, str], ...]]


class _HallucinationDetectionCache:
    """Cache detector output shared by the hallucination metrics."""

    def __init__(self) -> None:
        self._cache: weakref.WeakKeyDictionary[
            Dataset, tuple[_DetectionCacheKey, _HallucinationDetection]
        ] = weakref.WeakKeyDictionary()

    def get_or_compute(
        self,
        dataset: Dataset,
        predictions: c.Sequence[dict[str, t.Any]],
        model: str,
        device: Device,
        cache_dir: str,
        compute: c.Callable[[], _HallucinationDetection],
    ) -> _HallucinationDetection:
        """Return cached detector output or compute it once for these inputs."""
        key: _DetectionCacheKey = (
            model,
            device,
            cache_dir,
            tuple(
                (str(prediction["id"]), str(prediction["prediction_text"]))
                for prediction in predictions
            ),
        )
        cached = self._cache.get(dataset)
        if cached is not None and cached[0] == key:
            return cached[1]

        detection = compute()
        self._cache[dataset] = (key, detection)
        return detection


class _HallucinationMetric(Metric):
    """Base class for metrics sharing one hallucination detector run."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        detection_cache: _HallucinationDetectionCache | None = None,
    ) -> None:
        """Initialise a hallucination metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            detection_cache (optional):
                Cache shared by metrics evaluating the same predictions. Defaults to
                the module-level cache.
        """
        super().__init__(name=name, pretty_name=pretty_name, postprocessing_fn=None)
        self._detection_cache = detection_cache or _hallucination_detection_cache

    def __call__(
        self,
        predictions: c.Iterable[dict[str, t.Any]],
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Compute a hallucination metric for a set of predictions.

        Args:
            predictions:
                The model predictions. Each prediction must provide ``"id"`` and
                ``"prediction_text"`` fields.
            references:
                The ground truth references. Unused by these metrics.
            dataset:
                The dataset used for evaluation.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration, used to determine the compute device.

        Returns:
            The hallucination score.
        """
        prediction_list = list(predictions)
        main_language = dataset_config.main_language
        language_code: str = (
            main_language[1].code
            if isinstance(main_language, tuple)
            else main_language.code
        )
        model = _hallucination_model_id(language_code=language_code)
        device = Device(benchmark_config.device.type)
        detection = self._detection_cache.get_or_compute(
            dataset=dataset,
            predictions=prediction_list,
            model=model,
            device=device,
            cache_dir=benchmark_config.cache_dir,
            compute=lambda: _detect_hallucinations(
                dataset=dataset,
                predictions=prediction_list,
                model=model,
                device=device,
                cache_dir=benchmark_config.cache_dir,
            ),
        )
        return self._score_detection(detection=detection)

    def _score_detection(self, detection: _HallucinationDetection) -> float:
        """Convert detector output into this metric's score."""
        raise NotImplementedError

    def download(
        self, cache_dir: str, dataset_config: "DatasetConfig" | None = None
    ) -> "_HallucinationMetric":
        """Pre-download hallucination detection models.

        The hallucination detection model is language-specific. When a dataset
        configuration is provided, only the model for the relevant language is
        downloaded. Otherwise, all models referenced by built-in configurations are
        fetched for offline benchmarking.

        Args:
            cache_dir:
                The directory where the models will be downloaded to.
            dataset_config (optional):
                The dataset configuration, used to filter models by language.
                Defaults to None.

        Returns:
            The metric object itself.
        """
        for model_id in _hallucination_model_ids(
            cache_dir=cache_dir, dataset_config=dataset_config
        ):
            snapshot_download(repo_id=model_id, repo_type="model", cache_dir=cache_dir)
        return self


class SampleHallucinationMetric(_HallucinationMetric):
    """Metric reporting the proportion of evaluated samples with hallucinations."""

    def _score_detection(self, detection: _HallucinationDetection) -> float:
        """Return the sample-level hallucination rate."""
        evaluated_samples = [tokens for tokens in detection.token_predictions if tokens]
        return sum(any(tokens) for tokens in evaluated_samples) / len(evaluated_samples)


class TokenHallucinationMetric(_HallucinationMetric):
    """Metric reporting the proportion of hallucinated tokens."""

    def _score_detection(self, detection: _HallucinationDetection) -> float:
        """Return the token-level hallucination rate."""
        return detection.hallucinated_tokens / detection.total_tokens


_hallucination_detection_cache = _HallucinationDetectionCache()


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
    # Extract language(s) from the provided dataset configuration if available
    if dataset_config is not None:
        main_language = dataset_config.main_language
        language_code: str = (
            main_language[1].code
            if isinstance(main_language, tuple)
            else main_language.code
        )
        return {_hallucination_model_id(language_code=language_code)}

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
            isinstance(metric, _HallucinationMetric)
            for metric in dataset_config.task.metrics
        ):
            main_language = dataset_config.main_language
            language_code: str = (
                main_language[1].code
                if isinstance(main_language, tuple)
                else main_language.code
            )
            model_ids.add(_hallucination_model_id(language_code=language_code))
    return model_ids


def _hallucination_model_id(language_code: str) -> str:
    """Build the hallucination detection model ID for a dataset.

    Args:
        language_code:
            The language code of the dataset.

    Returns:
        The Hugging Face Hub repository ID of the hallucination detection model.
    """
    return (
        "EuroEval/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-"
        f"ragtruth-{language_code}"
    )


def detect_hallucinations(
    dataset: Dataset,
    predictions: c.Iterable[dict[str, t.Any]],
    model: str,
    device: Device,
    cache_dir: str,
) -> float:
    """Load the detector and return its token-level hallucination rate.

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

    """
    detection = _detect_hallucinations(
        dataset=dataset,
        predictions=list(predictions),
        model=model,
        device=device,
        cache_dir=cache_dir,
    )
    return detection.hallucinated_tokens / detection.total_tokens


def _detect_hallucinations(
    dataset: Dataset,
    predictions: c.Sequence[dict[str, t.Any]],
    model: str,
    device: Device,
    cache_dir: str,
) -> _HallucinationDetection:
    """Run the detector and retain token predictions for multiple aggregations.

    Args:
        dataset:
            Hallucination dataset containing ``"id"`` and ``"context"`` fields.
        predictions:
            Prediction objects containing matching ``"id"`` and
            ``"prediction_text"`` fields.
        model:
            Path to the hallucination detection model.
        device:
            Device to run on.
        cache_dir:
            Directory containing the cached detection model.

    Returns:
        Detector token predictions grouped by evaluated sample.

    Raises:
        InvalidBenchmark:
            If the model does not exist or no prediction tokens are available.
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

    token_predictions: list[tuple[int, ...]] = []
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
        sample_tokens = tuple(int(token["pred"]) for token in predict_answer)
        token_predictions.append(sample_tokens)
        total_tokens += len(sample_tokens)

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

    return _HallucinationDetection(token_predictions=tuple(token_predictions))


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


sample_hallucination_metric = SampleHallucinationMetric(
    name="sample_hallucination_rate", pretty_name="Sample hallucination rate"
)

hallucination_metric = TokenHallucinationMetric(
    name="hallucination_rate", pretty_name="Token hallucination rate"
)
