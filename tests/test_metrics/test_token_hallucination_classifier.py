"""Tests for the `token_hallucination_classifier` metrics module."""

import typing as t

import pytest
from datasets import Dataset
from pytest_mock import MockerFixture

from euroeval.exceptions import InvalidBenchmark
from euroeval.metrics.token_hallucination_classifier import (
    HallucinationRateMetric,
    hallucination_rate_metric,
)


class DummyLanguage:
    """Dummy language for testing."""

    code = "da"


class DummyDatasetConfig:
    """Dummy dataset config for testing."""

    main_language = DummyLanguage()


@pytest.fixture
def metric() -> t.Generator[HallucinationRateMetric, None, None]:
    """Yield a fresh HallucinationRateMetric instance for each test.

    Yields:
        A new HallucinationRateMetric instance.
    """
    yield HallucinationRateMetric()


@pytest.fixture
def dataset_config() -> t.Generator[DummyDatasetConfig, None, None]:
    """Yield a dummy dataset config.

    Yields:
        A DummyDatasetConfig instance.
    """
    yield DummyDatasetConfig()


@pytest.fixture
def make_dataset() -> t.Callable[[list[str], list[str]], Dataset]:
    """Return a factory that builds datasets from contexts and questions.

    Returns:
        A function that creates datasets from lists of contexts and questions.
    """

    def _make(contexts: list[str], questions: list[str]) -> Dataset:
        return Dataset.from_list(
            [{"context": c, "question": q} for c, q in zip(contexts, questions)]
        )

    return _make


def test_metric_initialization(metric: HallucinationRateMetric) -> None:
    """Test that the metric is initialised with correct attributes."""
    assert metric.name == "hallucination_rate"
    assert metric.pretty_name == "Hallucination rate"
    # The base class substitutes None with a default lambda, verify it behaves correctly
    assert callable(metric.postprocessing_fn)
    score, score_str = metric.postprocessing_fn(0.25)
    assert score == pytest.approx(25.0)
    assert score_str == "25.00%"
    assert metric.detector is None


def test_module_level_instance() -> None:
    """Test that the module-level instance has the correct type."""
    assert isinstance(hallucination_rate_metric, HallucinationRateMetric)


def test_empty_predictions_returns_zero(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Return 0.0 when no predictions are provided."""
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector"
    )
    dataset = make_dataset([], [])
    result = metric(
        predictions=[],
        dataset=dataset,
        dataset_config=dataset_config,  # type: ignore[arg-type]
    )
    assert result == 0.0


def test_mismatched_lengths_raise_invalid_benchmark(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Raise InvalidBenchmark when prediction count differs from dataset length."""
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector"
    )
    dataset = make_dataset(["ctx1", "ctx2"], ["q1", "q2"])
    with pytest.raises(InvalidBenchmark):
        metric(
            predictions=[{"prediction_text": "answer1"}],
            dataset=dataset,
            dataset_config=dataset_config,  # type: ignore[arg-type]
        )


def test_no_hallucinations_returns_zero(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Return 0.0 when the detector finds no hallucinated tokens."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = [{"pred": 0}, {"pred": 0}]
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    dataset = make_dataset(["ctx1", "ctx2"], ["q1", "q2"])
    predictions = [
        {"prediction_text": "answer1"},
        {"prediction_text": "answer2"},
    ]
    result = metric(
        predictions=predictions,
        dataset=dataset,
        dataset_config=dataset_config,  # type: ignore[arg-type]
    )
    assert result == pytest.approx(0.0)


def test_all_hallucinations_returns_one(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Return 1.0 when every sample is flagged as hallucinated."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = [{"pred": 1}, {"pred": 1}]
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    dataset = make_dataset(["ctx1", "ctx2"], ["q1", "q2"])
    predictions = [
        {"prediction_text": "hallucinated1"},
        {"prediction_text": "hallucinated2"},
    ]
    result = metric(
        predictions=predictions,
        dataset=dataset,
        dataset_config=dataset_config,  # type: ignore[arg-type]
    )
    assert result == pytest.approx(1.0)


def test_partial_hallucinations_returns_correct_rate(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Return the correct fraction when only some samples are hallucinated."""
    call_count = 0

    def predict_side_effect(
        context: list[str], question: str, answer: str
    ) -> list[dict[str, int]]:
        nonlocal call_count
        call_count += 1
        # First call: hallucinated, second: not hallucinated
        if call_count == 1:
            return [{"pred": 1}]
        return [{"pred": 0}]

    mock_detector = mocker.MagicMock()
    mock_detector.predict.side_effect = predict_side_effect
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    dataset = make_dataset(["ctx1", "ctx2"], ["q1", "q2"])
    predictions = [
        {"prediction_text": "hallucinated"},
        {"prediction_text": "correct"},
    ]
    result = metric(
        predictions=predictions,
        dataset=dataset,
        dataset_config=dataset_config,  # type: ignore[arg-type]
    )
    assert result == pytest.approx(0.5)


def test_detector_uses_correct_model_id(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Verify the model ID is built from dataset_config.main_language.code."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = [{"pred": 0}]
    mock_cls = mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    dataset = make_dataset(["ctx1"], ["q1"])
    metric(
        predictions=[{"prediction_text": "answer"}],
        dataset=dataset,
        dataset_config=dataset_config,  # type: ignore[arg-type]
    )

    expected_model_id = (
        "alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-da"
    )
    mock_cls.assert_called_once_with(
        method="transformer", model_path=expected_model_id, device="cpu"
    )


def test_detector_is_reused_across_calls(
    metric: HallucinationRateMetric,
    dataset_config: DummyDatasetConfig,
    make_dataset: t.Callable[[list[str], list[str]], Dataset],
    mocker: MockerFixture,
) -> None:
    """Ensure the detector is only created once and reused on subsequent calls."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = [{"pred": 0}]
    mock_cls = mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    dataset = make_dataset(["ctx1"], ["q1"])
    predictions = [{"prediction_text": "answer"}]
    call_kwargs = dict(
        predictions=predictions,
        dataset=dataset,
        dataset_config=dataset_config,  # type: ignore[arg-type]
    )

    metric(**call_kwargs)
    metric(**call_kwargs)

    # Constructor should only be called once
    assert mock_cls.call_count == 1
