"""Tests for task-level metric wiring."""

from euroeval.metrics.sacrebleu import ChrF
from euroeval.tasks import SUMM, TRANSLATION


def test_translation_and_summarisation_use_chrf() -> None:
    """Ensure translation and summarisation are scored with ChrF metrics."""
    for task in [SUMM, TRANSLATION]:
        for metric in task.metrics:
            assert isinstance(metric, ChrF)
