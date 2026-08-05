"""Tests for task-level metric wiring."""

from euroeval.metrics.sacrebleu import ChrF
from euroeval.tasks import SUMM, TRANSLATION


def test_translation_and_summarisation_use_language_penalised_chrf() -> None:
    """Ensure translation keeps the same language check as summarisation."""
    for task in [SUMM, TRANSLATION]:
        for metric in task.metrics:
            assert isinstance(metric, ChrF)
            assert metric.language_detector is not None
