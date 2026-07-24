"""Tests for the `token_hallucination_classifier` metrics module."""

import collections.abc as c
import typing as t
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from euroeval.enums import Device

from euroeval.exceptions import InvalidBenchmark
from euroeval.metrics.token_hallucination_classifier import (
    TokenHallucinationMetric,
    hallucination_metric,
)

DETECTOR_PATH = "euroeval.metrics.token_hallucination_classifier.HallucinationDetector"
HF_API_PATH = "euroeval.metrics.token_hallucination_classifier.HfApi"


@pytest.fixture(autouse=True)
def _mock_hf_api() -> t.Generator[None, None, None]:
    """Patch ``HfApi`` so the hallucination model is always reported as existing.

    Yields:
        None.
    """
    with patch(HF_API_PATH) as mock_api:
        mock_api.return_value.repo_exists.return_value = True
        yield


class DummyLanguage:
    """Dummy language for testing."""

    code = "da"


class DummyDatasetConfig:
    """Dummy dataset config for testing."""

    main_language = DummyLanguage()


class DummyDevice:
    """Dummy device for testing."""

    type = "cpu"


class DummyBenchmarkConfig:
    """Dummy benchmark config for testing."""

    device = DummyDevice()
    cache_dir = ".euroeval_cache"


@pytest.fixture
def metric() -> t.Generator[TokenHallucinationMetric, None, None]:
    """Yield a fresh TokenHallucinationMetric instance for each test.

    Yields:
        A new TokenHallucinationMetric instance.
    """
    yield TokenHallucinationMetric(
        name="hallucination_rate", pretty_name="Hallucination rate"
    )


@pytest.fixture
def dataset_config() -> t.Generator[DummyDatasetConfig, None, None]:
    """Yield a dummy dataset config.

    Yields:
        A DummyDatasetConfig instance.
    """
    yield DummyDatasetConfig()


@pytest.fixture
def benchmark_config() -> t.Generator[DummyBenchmarkConfig, None, None]:
    """Yield a dummy benchmark config.

    Yields:
        A DummyBenchmarkConfig instance.
    """
    yield DummyBenchmarkConfig()


@pytest.fixture
def make_dataset() -> t.Callable[[list[str]], Dataset]:
    """Return a factory that builds datasets from contexts.

    Returns:
        A function that creates datasets from lists of contexts.
    """

    def _make(contexts: list[str]) -> Dataset:
        return Dataset.from_list(
            [{"id": str(idx), "context": ctx} for idx, ctx in enumerate(contexts)]
        )

    return _make


def _make_detector(
    predict_return: list[dict[str, int]] | None = None,
    predict_side_effect: c.Callable[..., list[dict[str, int]]] | None = None,
    max_length: int = 512,
) -> MagicMock:
    """Build a mocked ``HallucinationDetector`` instance.

    Args:
        predict_return:
            The value returned by ``predict_prompt`` for every call.
        predict_side_effect:
            A callable used as the ``predict_prompt`` side effect, taking
            precedence over ``predict_return``.
        max_length:
            The detector's maximum input sequence length.

    Returns:
        A configured mock detector instance.
    """
    detector = MagicMock()
    detector.detector.tokenizer.return_value = {"input_ids": [1, 2, 3]}
    detector.detector.max_length = max_length
    if predict_side_effect is not None:
        detector.predict_prompt.side_effect = predict_side_effect
    else:
        detector.predict_prompt.return_value = predict_return
    return detector


def test_metric_initialization(metric: TokenHallucinationMetric) -> None:
    """Test that the metric is initialised with correct attributes."""
    assert metric.name == "hallucination_rate"
    assert metric.pretty_name == "Hallucination rate"
    assert callable(metric.postprocessing_fn)
    score, score_str = metric.postprocessing_fn(0.25)
    assert score == pytest.approx(25.0)
    assert score_str == "25.00%"


def test_module_level_instance() -> None:
    """Test that the module-level instance has the correct type."""
    assert isinstance(hallucination_metric, TokenHallucinationMetric)


def test_no_hallucinations_returns_zero(
    metric: TokenHallucinationMetric,
    dataset_config: DummyDatasetConfig,
    benchmark_config: DummyBenchmarkConfig,
    make_dataset: t.Callable[[list[str]], Dataset],
) -> None:
    """Return 0.0 when the detector finds no hallucinated tokens."""
    detector = _make_detector(predict_return=[{"pred": 0}])
    dataset = make_dataset(["ctx1", "ctx2"])
    predictions = [
        {"id": "0", "prediction_text": "answer1", "no_answer_probability": 0.0},
        {"id": "1", "prediction_text": "answer2", "no_answer_probability": 0.0},
    ]
    with patch(DETECTOR_PATH, return_value=detector):
        result = metric(
            predictions=predictions,
            references=[],
            dataset=dataset,
            dataset_config=dataset_config,  # ty: ignore[invalid-argument-type]
            benchmark_config=benchmark_config,  # ty: ignore[invalid-argument-type]
        )
    assert result == pytest.approx(0.0)


def test_all_hallucinations_returns_one(
    metric: TokenHallucinationMetric,
    dataset_config: DummyDatasetConfig,
    benchmark_config: DummyBenchmarkConfig,
    make_dataset: t.Callable[[list[str]], Dataset],
) -> None:
    """Return 1.0 when every token is flagged as hallucinated."""
    detector = _make_detector(predict_return=[{"pred": 1}])
    dataset = make_dataset(["ctx1", "ctx2"])
    predictions = [
        {"id": "0", "prediction_text": "hallucinated1", "no_answer_probability": 0.0},
        {"id": "1", "prediction_text": "hallucinated2", "no_answer_probability": 0.0},
    ]
    with patch(DETECTOR_PATH, return_value=detector):
        result = metric(
            predictions=predictions,
            references=[],
            dataset=dataset,
            dataset_config=dataset_config,  # ty: ignore[invalid-argument-type]
            benchmark_config=benchmark_config,  # ty: ignore[invalid-argument-type]
        )
    assert result == pytest.approx(1.0)


def test_partial_hallucinations_returns_correct_rate(
    metric: TokenHallucinationMetric,
    dataset_config: DummyDatasetConfig,
    benchmark_config: DummyBenchmarkConfig,
    make_dataset: t.Callable[[list[str]], Dataset],
) -> None:
    """Return the correct fraction when only some tokens are hallucinated."""
    call_count = 0

    def predict_side_effect(prompt: str, answer: str) -> list[dict[str, int]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [{"pred": 1}]
        return [{"pred": 0}]

    detector = _make_detector(predict_side_effect=predict_side_effect)
    dataset = make_dataset(["ctx1", "ctx2"])
    predictions = [
        {"id": "0", "prediction_text": "hallucinated", "no_answer_probability": 0.0},
        {"id": "1", "prediction_text": "correct", "no_answer_probability": 0.0},
    ]
    with patch(DETECTOR_PATH, return_value=detector):
        result = metric(
            predictions=predictions,
            references=[],
            dataset=dataset,
            dataset_config=dataset_config,  # ty: ignore[invalid-argument-type]
            benchmark_config=benchmark_config,  # ty: ignore[invalid-argument-type]
        )
    assert result == pytest.approx(0.5)


def test_no_tokens_raises_invalid_benchmark(
    metric: TokenHallucinationMetric,
    dataset_config: DummyDatasetConfig,
    benchmark_config: DummyBenchmarkConfig,
    make_dataset: t.Callable[[list[str]], Dataset],
) -> None:
    """Raise InvalidBenchmark when no tokens are found in the predictions."""
    detector = _make_detector(predict_return=[])
    dataset = make_dataset(["ctx1"])
    predictions = [
        {"id": "0", "prediction_text": "answer", "no_answer_probability": 0.0}
    ]
    with patch(DETECTOR_PATH, return_value=detector):
        with pytest.raises(InvalidBenchmark):
            metric(
                predictions=predictions,
                references=[],
                dataset=dataset,
                dataset_config=dataset_config,  # ty: ignore[invalid-argument-type]
                benchmark_config=benchmark_config,  # ty: ignore[invalid-argument-type]
            )


def test_detector_uses_correct_model_id(
    metric: TokenHallucinationMetric,
    dataset_config: DummyDatasetConfig,
    benchmark_config: DummyBenchmarkConfig,
    make_dataset: t.Callable[[list[str]], Dataset],
) -> None:
    """Verify the model ID is built from dataset_config.main_language.code."""
    detector = _make_detector(predict_return=[{"pred": 0}])
    dataset = make_dataset(["ctx1"])
    with patch(DETECTOR_PATH, return_value=detector) as mock_cls:
        metric(
            predictions=[
                {"id": "0", "prediction_text": "answer", "no_answer_probability": 0.0}
            ],
            references=[],
            dataset=dataset,
            dataset_config=dataset_config,  # ty: ignore[invalid-argument-type]
            benchmark_config=benchmark_config,  # ty: ignore[invalid-argument-type]
        )

    expected_model_id = (
        "alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-with-ragtruth-da"
    )
    mock_cls.assert_called_once_with(
        method="transformer",
        model_path=expected_model_id,
        device=Device.CPU,
        cache_dir=".euroeval_cache",
    )
