"""Tests for the `pipeline` metrics module."""

import collections.abc as c
import logging

import pytest
from _pytest.logging import LogCaptureFixture
from datasets import Dataset

from euroeval.metrics.pipeline import european_values_preprocessing_fn

NUM_QUESTIONS = 53


@pytest.fixture(scope="module")
def make_ev_dataset() -> c.Generator[c.Callable[[list[dict]], Dataset], None, None]:
    """Create a European Values dataset with a given idx_to_choice per question.

    Yields:
        A factory function that builds a Dataset with ``num_questions`` rows and
        the ``idx_to_choice`` values supplied via ``choices_per_question``.
    """

    def _make(choices_per_question: list[dict]) -> Dataset:
        assert len(choices_per_question) == NUM_QUESTIONS
        records = [{"idx_to_choice": c} for c in choices_per_question]
        return Dataset.from_list(records)

    yield _make


def test_valid_predictions(make_ev_dataset: c.Callable[[list[dict]], Dataset]) -> None:
    """All valid predictions should be processed without error."""
    three_choices = {"0": 1, "1": 2, "2": 3}
    dataset = make_ev_dataset([three_choices] * NUM_QUESTIONS)
    predictions = [1] * NUM_QUESTIONS
    result = european_values_preprocessing_fn(predictions=predictions, dataset=dataset)
    assert len(result) == NUM_QUESTIONS


def test_invalid_prediction_does_not_raise(
    make_ev_dataset: c.Callable[[list[dict]], Dataset],
) -> None:
    """An out-of-range prediction should be handled gracefully (no exception)."""
    three_choices = {"0": 1, "1": 2, "2": 3}
    dataset = make_ev_dataset([three_choices] * NUM_QUESTIONS)

    # Prediction 8 is not a valid index for a question with choices {0, 1, 2}
    predictions = [0] * (NUM_QUESTIONS - 1) + [8]
    result = european_values_preprocessing_fn(predictions=predictions, dataset=dataset)
    assert len(result) == NUM_QUESTIONS


def test_invalid_prediction_uses_first_valid_index(
    make_ev_dataset: c.Callable[[list[dict]], Dataset],
) -> None:
    """An invalid prediction should default to the first (minimum) valid index."""
    # All 53 questions have choices {0: 10, 1: 20, 2: 30}
    three_choices = {"0": 10, "1": 20, "2": 30}
    dataset = make_ev_dataset([three_choices] * NUM_QUESTIONS)

    # Use a valid prediction for all questions except the first one
    predictions = [8] + [0] * (NUM_QUESTIONS - 1)

    result = european_values_preprocessing_fn(predictions=predictions, dataset=dataset)

    # The invalid prediction 8 should default to index 0 → choice value 10.
    # Question 0 is in question_choices with target choice 1; since 10 != 1, the
    # binary mapping gives 0.
    assert result[0] == 0


def test_invalid_prediction_logs_warning(
    make_ev_dataset: c.Callable[[list[dict]], Dataset], caplog: LogCaptureFixture
) -> None:
    """A warning should be logged when an invalid prediction is encountered."""
    three_choices = {"0": 1, "1": 2, "2": 3}
    dataset = make_ev_dataset([three_choices] * NUM_QUESTIONS)
    # Use a unique value (99) to avoid being deduplicated by log_once's cache
    predictions = [99] + [0] * (NUM_QUESTIONS - 1)

    with caplog.at_level(logging.WARNING, logger="euroeval"):
        european_values_preprocessing_fn(predictions=predictions, dataset=dataset)

    assert any("not a valid index" in record.message for record in caplog.records), (
        "Expected a warning about the invalid prediction index"
    )
