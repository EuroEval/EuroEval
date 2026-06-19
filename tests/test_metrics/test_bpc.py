"""Tests for the bits-per-character metric."""

import math

from euroeval.metrics.bpc import bpc_metric


class TestBitsPerCharacterMetric:
    """Tests for `BitsPerCharacterMetric`.

    The metric only uses its `predictions` argument; the references, dataset and
    configs are ignored, so `None` is passed for them.
    """

    def _call(self, predictions: list[float]) -> float | None:
        """Call the metric with only predictions set.

        Args:
            predictions: The per-example BPC scores.

        Returns:
            The averaged BPC score, or None.
        """
        return bpc_metric(
            predictions=predictions,
            references=[],
            dataset=None,  # ty: ignore[invalid-argument-type]
            dataset_config=None,  # ty: ignore[invalid-argument-type]
            benchmark_config=None,  # ty: ignore[invalid-argument-type]
        )

    def test_averages_finite_scores(self) -> None:
        """The mean of finite scores is returned."""
        assert self._call([1.0, 2.0, 3.0]) == 2.0

    def test_empty_predictions_returns_none(self) -> None:
        """No predictions yields None."""
        assert self._call([]) is None

    def test_infinite_score_is_excluded_from_mean(self) -> None:
        """A single infinite score does not poison the average."""
        result = self._call([1.0, float("inf"), 3.0])
        assert result == 2.0

    def test_nan_score_is_excluded_from_mean(self) -> None:
        """NaN scores are excluded rather than propagating to the mean."""
        result = self._call([2.0, float("nan"), 4.0])
        assert result == 3.0

    def test_all_non_finite_returns_none(self) -> None:
        """If every score is non-finite there is nothing to average."""
        assert self._call([float("inf"), float("nan")]) is None

    def test_finite_only_average_is_unaffected(self) -> None:
        """Without any non-finite scores the mean equals the plain average."""
        scores = [0.5, 1.5, 2.5, 3.5]
        result = self._call(scores)
        assert result is not None
        assert math.isclose(result, sum(scores) / len(scores))
