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


def test_invalid_prediction_defaults_and_logs_warning(
    make_ev_dataset: c.Callable[[list[dict]], Dataset], caplog: LogCaptureFixture
) -> None:
    """Invalid predictions use the first valid index and emit a warning."""
    three_choices = {"0": 10, "1": 20, "2": 30}
    dataset = make_ev_dataset([three_choices] * NUM_QUESTIONS)
    predictions = [99] + [0] * (NUM_QUESTIONS - 1)

    with caplog.at_level(logging.WARNING, logger="euroeval"):
        result = european_values_preprocessing_fn(
            predictions=predictions, dataset=dataset
        )

    assert len(result) == NUM_QUESTIONS
    assert result[0] == 0
    assert any("not a valid index" in record.message for record in caplog.records), (
        "Expected a warning about the invalid prediction index"
    )


def test_valid_predictions(make_ev_dataset: c.Callable[[list[dict]], Dataset]) -> None:
    """All valid predictions should be processed without error."""
    three_choices = {"0": 1, "1": 2, "2": 3}
    dataset = make_ev_dataset([three_choices] * NUM_QUESTIONS)
    predictions = [1] * NUM_QUESTIONS
    result = european_values_preprocessing_fn(predictions=predictions, dataset=dataset)
    assert len(result) == NUM_QUESTIONS
