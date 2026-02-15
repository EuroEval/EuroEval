"""Tests for the `scores` module."""

from typing import Generator

import numpy as np
import pytest

from euroeval.metrics import Metric
from euroeval.scores import aggregate_scores, log_scores
from euroeval.types import ScoreDict


@pytest.fixture(scope="module")
def scores(metric: Metric) -> Generator[list[dict[str, float]], None, None]:
    """Yield a dictionary of scores.

    Yields:
        A dictionary of scores.
    """
    yield [
        {f"test_{metric.name}": 0.50},
        {f"test_{metric.name}": 0.55},
        {f"test_{metric.name}": 0.60},
    ]


class TestAggregateScores:
    """Tests for the `aggregate_scores` function."""

    def test_scores(self, scores: list[dict[str, float]], metric: Metric) -> None:
        """Test that `aggregate_scores` works when scores are provided."""
        # Aggregate scores using the `agg_scores` function
        agg_scores = aggregate_scores(scores=scores, metric=metric)

        # Manually compute the mean and standard error of the test scores
        test_scores = [dct[f"test_{metric.name}"] for dct in scores]
        test_mean = np.mean(test_scores)
        test_se = 1.96 * np.std(test_scores, ddof=1) / np.sqrt(len(test_scores))

        # Assert that `aggregate_scores` computed the same
        assert agg_scores == (test_mean, test_se)

    def test_no_scores(self, metric: Metric) -> None:
        """Test that `aggregate_scores` works when no scores are provided."""
        agg_scores = aggregate_scores(scores=list(), metric=metric)
        assert np.isnan(agg_scores).all()

    def test_single_score(self, metric: Metric) -> None:
        """Test that `aggregate_scores` handles a single score correctly."""
        single_scores = [{f"test_{metric.name}": 0.75}]
        agg_scores = aggregate_scores(scores=single_scores, metric=metric)
        # With a single score, mean should be the value and SE should be NaN
        assert agg_scores[0] == 0.75
        assert np.isnan(agg_scores[1])

    def test_scores_with_metric_name_without_prefix(self, metric: Metric) -> None:
        """Test aggregate_scores with metric name without test_ prefix."""
        scores_no_prefix = [
            {metric.name: 0.40},
            {metric.name: 0.50},
            {metric.name: 0.60},
        ]
        agg_scores = aggregate_scores(scores=scores_no_prefix, metric=metric)
        assert agg_scores[0] == 0.50
        assert agg_scores[1] > 0  # Should have non-zero SE


class TestLogScores:
    """Tests for the `log_scores` function."""

    @pytest.fixture(scope="class")
    def logged_scores(
        self, metric: Metric, scores: list[dict[str, float]]
    ) -> Generator[ScoreDict, None, None]:
        """Yields the logged scores.

        Yields:
            The logged scores.
        """
        yield log_scores(
            dataset_name="dataset",
            metrics=[metric],
            scores=scores,
            model_id="model_id",
            model_revision="main",
            model_param=None,
        )

    def test_is_correct_type(self, logged_scores: ScoreDict) -> None:
        """Test that `log_scores` returns a dictionary."""
        assert isinstance(logged_scores, dict)

    def test_has_correct_keys(self, logged_scores: ScoreDict) -> None:
        """Test that `log_scores` returns a dictionary with the correct keys."""
        assert sorted(logged_scores.keys()) == ["raw", "total"]

    def test_raw_scores_are_identical_to_input(
        self, logged_scores: ScoreDict, scores: list[dict[str, float]]
    ) -> None:
        """Test that `log_scores` returns the same raw scores as the input."""
        assert logged_scores["raw"] == scores

    def test_total_scores_is_dict(self, logged_scores: ScoreDict) -> None:
        """Test that `log_scores` returns a dictionary for the total scores."""
        assert isinstance(logged_scores["total"], dict)

    def test_total_scores_keys(self, logged_scores: ScoreDict, metric: Metric) -> None:
        """Test that `log_scores` returns a dictionary with the correct keys."""
        total_dict = logged_scores["total"]
        assert isinstance(total_dict, dict)
        assert sorted(total_dict.keys()) == [
            f"test_{metric.name}",
            f"test_{metric.name}_se",
        ]

    def test_total_scores_values_are_floats(self, logged_scores: ScoreDict) -> None:
        """Test that `log_scores` returns a dictionary with float values."""
        total_dict = logged_scores["total"]
        assert isinstance(total_dict, dict)
        for val in total_dict.values():
            assert isinstance(val, float)

    def test_log_scores_with_revision(
        self, metric: Metric, scores: list[dict[str, float]]
    ) -> None:
        """Test log_scores with non-main revision."""
        result = log_scores(
            dataset_name="dataset",
            metrics=[metric],
            scores=scores,
            model_id="model_id",
            model_revision="v1.0",
            model_param=None,
        )
        assert isinstance(result, dict)

    def test_log_scores_with_param(
        self, metric: Metric, scores: list[dict[str, float]]
    ) -> None:
        """Test log_scores with model parameter."""
        result = log_scores(
            dataset_name="dataset",
            metrics=[metric],
            scores=scores,
            model_id="model_id",
            model_revision="main",
            model_param="param_value",
        )
        assert isinstance(result, dict)

    def test_log_scores_with_revision_and_param(
        self, metric: Metric, scores: list[dict[str, float]]
    ) -> None:
        """Test log_scores with both revision and parameter."""
        result = log_scores(
            dataset_name="dataset",
            metrics=[metric],
            scores=scores,
            model_id="model_id",
            model_revision="v2.0",
            model_param="param_value",
        )
        assert isinstance(result, dict)
        assert "raw" in result
        assert "total" in result
