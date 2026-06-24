"""Tests for the logic puzzle metrics."""

import pytest

from euroeval.metrics.logic_puzzles import (
    BestPermutedCellWiseAccuracyMetric,
    CellWiseAccuracyMetric,
    PuzzleLevelAccuracyMetric,
)


class TestPuzzleLevelAccuracyMetric:
    """Tests for `PuzzleLevelAccuracyMetric`."""

    def _call(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Call the metric with a single prediction/label pair.

        Args:
            prediction: The model prediction as a dictionary.
            label: The ground truth label as a dictionary.

        Returns:
            The puzzle-level accuracy score (0.0 or 1.0).
        """
        metric = PuzzleLevelAccuracyMetric()
        result = metric(
            predictions=[prediction],
            references=[label],
            dataset=None,  # ty: ignore[invalid-argument-type]
            dataset_config=None,  # ty: ignore[invalid-argument-type]
            benchmark_config=None,  # ty: ignore[invalid-argument-type]
        )
        assert result is not None
        return result

    def test_perfect_match(self) -> None:
        """Identical prediction and label yields 1.0."""
        pred = {
            "object_1": ["Alice", "Red"],
            "object_2": ["Bob", "Blue"],
            "object_3": ["Charlie", "Green"],
        }
        assert self._call(prediction=pred, label=pred) == 1.0

    def test_completely_wrong(self) -> None:
        """All attributes wrong yields 0.0."""
        pred = {
            "object_1": ["Bob", "Blue"],
            "object_2": ["Charlie", "Green"],
            "object_3": ["Alice", "Red"],
        }
        label = {
            "object_1": ["Alice", "Red"],
            "object_2": ["Bob", "Blue"],
            "object_3": ["Charlie", "Green"],
        }
        assert self._call(prediction=pred, label=label) == 0.0

    def test_missing_keys_yields_zero(self) -> None:
        """Prediction with fewer keys than label yields 0.0.

        This tests the critical bug fix: previously zip() would truncate
        and this would incorrectly return 1.0.
        """
        pred = {"object_1": ["Alice"], "object_2": ["Bob"]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"], "object_3": ["Charlie"]}
        assert self._call(prediction=pred, label=label) == 0.0

    def test_extra_keys_yields_zero(self) -> None:
        """Prediction with more keys than label yields 0.0."""
        pred = {
            "object_1": ["Alice"],
            "object_2": ["Bob"],
            "object_3": ["Charlie"],
            "object_4": ["David"],
        }
        label = {"object_1": ["Alice"], "object_2": ["Bob"], "object_3": ["Charlie"]}
        assert self._call(prediction=pred, label=label) == 0.0

    def test_whitespace_normalization(self) -> None:
        """Leading/trailing whitespace is normalized before comparison."""
        pred = {"object_1": ["  Alice  "], "object_2": [" Bob "]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"]}
        assert self._call(prediction=pred, label=label) == 1.0

    def test_order_independent(self) -> None:
        """Order of attributes within a key doesn't matter (sets)."""
        pred = {"object_1": ["Red", "Alice"], "object_2": ["Bob", "Blue"]}
        label = {"object_1": ["Alice", "Red"], "object_2": ["Blue", "Bob"]}
        assert self._call(prediction=pred, label=label) == 1.0

    def test_key_order_independent(self) -> None:
        """Order of keys doesn't matter (sorted before comparison)."""
        pred = {"object_2": ["Bob"], "object_1": ["Alice"]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"]}
        assert self._call(prediction=pred, label=label) == 1.0


class TestCellWiseAccuracyMetric:
    """Tests for `CellWiseAccuracyMetric`."""

    def _call(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Call the metric with a single prediction/label pair.

        Args:
            prediction: The model prediction as a dictionary.
            label: The ground truth label as a dictionary.

        Returns:
            The cell-wise accuracy score.
        """
        metric = CellWiseAccuracyMetric()
        result = metric(
            predictions=[prediction],
            references=[label],
            dataset=None,  # ty: ignore[invalid-argument-type]
            dataset_config=None,  # ty: ignore[invalid-argument-type]
            benchmark_config=None,  # ty: ignore[invalid-argument-type]
        )
        assert result is not None
        return result

    def test_perfect_match(self) -> None:
        """Identical prediction and label yields 1.0."""
        pred = {
            "object_1": ["Alice", "Red"],
            "object_2": ["Bob", "Blue"],
            "object_3": ["Charlie", "Green"],
        }
        assert self._call(prediction=pred, label=pred) == 1.0

    def test_completely_wrong(self) -> None:
        """All attributes wrong yields 0.0."""
        pred = {
            "object_1": ["Bob", "Blue"],
            "object_2": ["Charlie", "Green"],
            "object_3": ["Alice", "Red"],
        }
        label = {
            "object_1": ["Alice", "Red"],
            "object_2": ["Bob", "Blue"],
            "object_3": ["Charlie", "Green"],
        }
        # 0 correct attributes out of 6 total
        assert self._call(prediction=pred, label=label) == 0.0

    def test_partial_match(self) -> None:
        """Some correct attributes yields proportional score."""
        pred = {
            "object_1": ["Alice", "Blue"],  # 1 correct (Alice)
            "object_2": ["Bob", "Blue"],  # 2 correct (Bob, Blue)
            "object_3": ["Charlie", "Green"],  # 2 correct (Charlie, Green)
        }
        label = {
            "object_1": ["Alice", "Red"],
            "object_2": ["Bob", "Blue"],
            "object_3": ["Charlie", "Green"],
        }
        # 5 correct out of 6 total (2 attrs * 3 objects)
        assert self._call(prediction=pred, label=label) == pytest.approx(5 / 6)

    def test_missing_keys_yields_zero_denominator(self) -> None:
        """Prediction with fewer keys yields score based on actual denominator.

        The fix ensures we explicitly reject mismatched key counts.
        """
        pred = {"object_1": ["Alice"], "object_2": ["Bob"]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"], "object_3": ["Charlie"]}
        # With the fix, this should return 0.0 because key counts don't match
        assert self._call(prediction=pred, label=label) == 0.0

    def test_extra_keys_yields_zero(self) -> None:
        """Prediction with more keys than label yields 0.0."""
        pred = {
            "object_1": ["Alice"],
            "object_2": ["Bob"],
            "object_3": ["Charlie"],
            "object_4": ["David"],
        }
        label = {"object_1": ["Alice"], "object_2": ["Bob"], "object_3": ["Charlie"]}
        assert self._call(prediction=pred, label=label) == 0.0

    def test_whitespace_normalization(self) -> None:
        """Leading/trailing whitespace is normalized before comparison."""
        pred = {"object_1": ["  Alice  "], "object_2": [" Bob "]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"]}
        assert self._call(prediction=pred, label=label) == 1.0

    def test_partial_attribute_match(self) -> None:
        """Partial attribute overlap is counted correctly."""
        pred = {"object_1": ["Alice", "Red", "Tall"]}
        label = {"object_1": ["Alice", "Red", "Short"]}
        # 2 correct out of 3 (using label's denominator: 1 key * 3 attrs)
        assert self._call(prediction=pred, label=label) == pytest.approx(2 / 3)


class TestBestPermutedCellWiseAccuracyMetric:
    """Tests for `BestPermutedCellWiseAccuracyMetric`."""

    def _call(
        self, prediction: dict[str, list[str]], label: dict[str, list[str]]
    ) -> float:
        """Call the metric with a single prediction/label pair.

        Args:
            prediction: The model prediction as a dictionary.
            label: The ground truth label as a dictionary.

        Returns:
            The best permuted cell-wise accuracy score.
        """
        metric = BestPermutedCellWiseAccuracyMetric()
        result = metric(
            predictions=[prediction],
            references=[label],
            dataset=None,  # ty: ignore[invalid-argument-type]
            dataset_config=None,  # ty: ignore[invalid-argument-type]
            benchmark_config=None,  # ty: ignore[invalid-argument-type]
        )
        assert result is not None
        return result

    def test_perfect_match(self) -> None:
        """Identical prediction and label yields 1.0."""
        pred = {"object_1": ["Alice", "Red"], "object_2": ["Bob", "Blue"]}
        assert self._call(prediction=pred, label=pred) == 1.0

    def test_permuted_keys_match(self) -> None:
        """Correct attributes under wrong keys can match with permutation."""
        # Prediction has Alice/Red for object_2 instead of object_1
        pred = {"object_1": ["Bob", "Blue"], "object_2": ["Alice", "Red"]}
        label = {"object_1": ["Alice", "Red"], "object_2": ["Bob", "Blue"]}
        # With best permutation, should find perfect match
        assert self._call(prediction=pred, label=label) == 1.0

    def test_missing_keys_yields_zero(self) -> None:
        """Prediction with fewer keys yields 0.0."""
        pred = {"object_1": ["Alice"]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"]}
        assert self._call(prediction=pred, label=label) == 0.0

    def test_extra_keys_yields_zero(self) -> None:
        """Prediction with more keys yields 0.0."""
        pred = {"object_1": ["Alice"], "object_2": ["Bob"], "object_3": ["Charlie"]}
        label = {"object_1": ["Alice"], "object_2": ["Bob"]}
        assert self._call(prediction=pred, label=label) == 0.0

    def test_partial_permuted_match(self) -> None:
        """Partial match with permutation is scored correctly."""
        pred = {"object_1": ["Alice", "Blue"], "object_2": ["Bob", "Red"]}
        label = {"object_1": ["Alice", "Red"], "object_2": ["Bob", "Blue"]}
        # Best permutation finds 2 correct attributes (Alice, Bob)
        # but colors are swapped, so 2/4 = 0.5
        assert self._call(prediction=pred, label=label) == pytest.approx(0.5)
