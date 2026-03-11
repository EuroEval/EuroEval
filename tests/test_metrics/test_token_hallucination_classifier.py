"""Tests for the `token_hallucination_classifier` metrics module."""

import pytest
from datasets import Dataset

from euroeval.data_models import DatasetConfig
from euroeval.exceptions import InvalidBenchmark
from euroeval.languages import DANISH
from euroeval.metrics.token_hallucination_classifier import (
    TokenHallucinationMetric,
    detect_hallucinations,
    hallucination_metric,
)
from euroeval.tasks import HALLU


@pytest.fixture
def dataset_config() -> DatasetConfig:
    """Return a DatasetConfig for the hallucination task."""
    return DatasetConfig(
        name="multi-wiki-hallucination-qa-da",
        pretty_name="MultiWikiHalluQA-da",
        source="EuroEval/multi-wiki-qa-da-mini",
        task=HALLU,
        languages=[DANISH],
    )


@pytest.fixture
def sample_dataset() -> Dataset:
    """Return a small hallucination detection dataset."""
    return Dataset.from_list(
        [
            {
                "context": "The sky is blue.",
                "question": "What colour is the sky?",
                "answers": {"answer_start": [4], "text": ["sky is blue"]},
            },
            {
                "context": "Water boils at 100 degrees Celsius.",
                "question": "At what temperature does water boil?",
                "answers": {"answer_start": [0], "text": ["100 degrees Celsius"]},
            },
        ]
    )


@pytest.fixture
def sample_predictions() -> list[dict[str, str]]:
    """Return sample predictions for the hallucination task."""
    return [
        {"prediction_text": "blue"},
        {"prediction_text": "100 degrees"},
    ]


def test_metric_initialization() -> None:
    """Test that the metric is initialized with correct parameters."""
    metric = TokenHallucinationMetric()
    assert metric.name == "hallucination_rate"
    assert metric.pretty_name == "Token-level hallucination rate"
    assert metric.postprocessing_fn is not None


def test_postprocessing_fn() -> None:
    """Test that the postprocessing function works correctly."""
    score: float = 0.25

    (processed_score, score_str) = hallucination_metric.postprocessing_fn(score)  # type: ignore[misc]

    assert processed_score == 0.25
    assert score_str == "0.25"


def test_detect_hallucinations_no_hallucinations(
    mocker, sample_dataset: Dataset, sample_predictions: list[dict[str, str]]
) -> None:
    """Test detect_hallucinations returns 0.0 when no tokens are hallucinated."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = [{"pred": 0}, {"pred": 0}]
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    rate = detect_hallucinations(
        dataset=sample_dataset,
        predictions=sample_predictions,
        model="test-model",
    )

    assert rate == 0.0


def test_detect_hallucinations_all_hallucinated(
    mocker, sample_dataset: Dataset, sample_predictions: list[dict[str, str]]
) -> None:
    """Test detect_hallucinations returns 1.0 when all tokens are hallucinated."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = [{"pred": 1}, {"pred": 1}]
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    rate = detect_hallucinations(
        dataset=sample_dataset,
        predictions=sample_predictions,
        model="test-model",
    )

    assert rate == 1.0


def test_detect_hallucinations_partial_hallucination(
    mocker, sample_dataset: Dataset, sample_predictions: list[dict[str, str]]
) -> None:
    """Test detect_hallucinations computes the correct partial hallucination rate."""
    mock_detector = mocker.MagicMock()
    # First call: 1 hallucinated out of 2 tokens; second call: 0 out of 2 tokens
    mock_detector.predict.side_effect = [
        [{"pred": 1}, {"pred": 0}],
        [{"pred": 0}, {"pred": 0}],
    ]
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    rate = detect_hallucinations(
        dataset=sample_dataset,
        predictions=sample_predictions,
        model="test-model",
    )

    # 1 hallucinated out of 4 total tokens
    assert rate == pytest.approx(0.25)


def test_detect_hallucinations_no_tokens_raises(
    mocker, sample_dataset: Dataset, sample_predictions: list[dict[str, str]]
) -> None:
    """Test that detect_hallucinations raises InvalidBenchmark when no tokens found."""
    mock_detector = mocker.MagicMock()
    mock_detector.predict.return_value = []  # No tokens in predictions
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector",
        return_value=mock_detector,
    )

    with pytest.raises(InvalidBenchmark):
        detect_hallucinations(
            dataset=sample_dataset,
            predictions=sample_predictions,
            model="test-model",
        )


def test_detect_hallucinations_mismatched_lengths_raises(
    mocker, sample_dataset: Dataset
) -> None:
    """Test that a mismatch between predictions and dataset raises InvalidBenchmark."""
    mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.HallucinationDetector"
    )

    # Only 1 prediction for a dataset of 2 examples
    mismatched_predictions = [{"prediction_text": "blue"}]

    with pytest.raises(InvalidBenchmark):
        detect_hallucinations(
            dataset=sample_dataset,
            predictions=mismatched_predictions,
            model="test-model",
        )


def test_token_hallucination_metric_call(
    mocker,
    sample_dataset: Dataset,
    sample_predictions: list[dict[str, str]],
    dataset_config: DatasetConfig,
) -> None:
    """Test that calling TokenHallucinationMetric delegates to detect_hallucinations."""
    mock_detect = mocker.patch(
        "euroeval.metrics.token_hallucination_classifier.detect_hallucinations",
        return_value=0.1,
    )

    metric = TokenHallucinationMetric()
    result = metric(
        predictions=sample_predictions,
        dataset=sample_dataset,
        dataset_config=dataset_config,
    )

    assert result == 0.1
    mock_detect.assert_called_once_with(
        dataset=sample_dataset,
        predictions=sample_predictions,
        model="alexandrainst/mmBERT-small-multi-wiki-qa-synthetic-hallucinations-da",
    )
