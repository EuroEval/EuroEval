"""Tests for the `bias_metrics` module."""

from typing import Callable, Sequence

import numpy as np
import pytest
from datasets import Dataset

from euroeval.metrics import (
    accuracy_ambig_metric,
    bias_adjusted_accuracy_ambig_metric,
    bias_ambig_metric,
)


@pytest.fixture(scope="module")
def make_dataset() -> Callable[[str, Sequence[int], int | None, int], Dataset]:
    """Build small datasets with the columns needed by bias metrics.

    Returns:
        A function that creates a dataset with the specified columns.
    """

    def _make(
        cond: str, indices: Sequence[int], correct: int | None, n: int
    ) -> Dataset:
        """Create a dataset with consistent context and label indices.

        Args:
            cond:
                The context condition.
            indices:
                The indices of the context, stereo, counter, and unknown labels.
            correct:
                The index of the correct label.
            n:
                The number of records to create.

        Returns:
            A dataset with the specified columns.
        """
        stereo, counter, unknown = indices
        records = []
        for _ in range(n):
            records.append(
                {
                    "context_condition": cond,
                    "stereo_idx": stereo,
                    "counter_idx": counter,
                    "unknown_idx": unknown,
                    "correct_idx": correct,
                }
            )
        return Dataset.from_list(records)

    return _make


# --- BiasA tests (ambiguous contexts) ---


def test_biasa_all_unknowns(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Return zero bias when only unknown predictions appear."""
    ds = make_dataset("ambig", [1, 2, 0], None, 3)
    preds = [0, 0, 0]
    # references arg is ignored by the metric, so we can pass an empty list
    assert bias_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)


def test_biasa_stereo_and_counter(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Compute bias from the difference between stereo and counter picks."""
    ds = make_dataset("ambig", [1, 2, 0], None, 5)
    preds = [1, 1, 1, 2, 2]  # three stereotype picks, two counter
    assert bias_ambig_metric(preds, [], ds, None, None) == pytest.approx((3 - 2) / 5)


def test_biasa_ignores_undetectable(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Ignore predictions that are not present in label mappings."""
    # one prediction not in label mapping => discarded,
    # but still counts toward n_ambiguous
    ds = make_dataset("ambig", [1, 2, 0], None, 3)
    preds = [1, "X", 2]
    # stereo=1, counter=2 => (1-1)/3 = 0.0
    assert bias_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.0)


# --- AccuracyA tests (ambiguous contexts) ---


def test_accuracy_ambig_all_unknown(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Report perfect accuracy when all ambiguous predictions are unknown."""
    ds = make_dataset("ambig", [1, 2, 0], None, 4)
    preds = [0, 0, 0, 0]  # all correctly 'unknown'
    assert accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(1.0)


def test_accuracy_ambig_mixed(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Compute accuracy with a mix of correct unknown and wrong answers."""
    ds = make_dataset("ambig", [1, 2, 0], None, 4)
    preds = [0, 1, 0, 2]  # two correct 'unknown', two wrong
    assert accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(0.5)


def test_accuracy_ambig_accepts_numpy_ints(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Accept numpy integer predictions."""
    ds = make_dataset("ambig", [1, 2, 0], None, 3)
    preds = np.array([0, 0, 0], dtype=np.int64)
    assert accuracy_ambig_metric(preds, [], ds, None, None) == pytest.approx(1.0)


# --- Bias-adjusted accuracy metrics ---


def test_bias_adjusted_accuracy_ambig_zero_at_chance(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Keep accuracy unchanged when bias is zero at chance."""
    ds = make_dataset("ambig", [1, 2, 0], None, 3)
    preds = [
        0,
        1,
        2,
    ]  # 1 correct unknown, one stereo, one counter => acc = 1/3, bias = 0
    assert bias_adjusted_accuracy_ambig_metric(
        preds, [], ds, None, None
    ) == pytest.approx(1 / 3)


def test_bias_adjusted_accuracy_ambig_above_chance_low_bias(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Return accuracy when ambiguous bias stays at zero."""
    ds = make_dataset("ambig", [1, 2, 0], None, 4)
    preds = [0, 0, 1, 2]  # acc = 0.5, bias = 0
    assert bias_adjusted_accuracy_ambig_metric(
        preds, [], ds, None, None
    ) == pytest.approx(0.5)


def test_bias_adjusted_accuracy_ambig_penalizes_bias(
    make_dataset: Callable[[str, Sequence[int], int | None, int], Dataset],
) -> None:
    """Reduce bias-adjusted accuracy when ambiguous bias is high."""
    ds = make_dataset("ambig", [1, 2, 0], None, 4)
    # 2 correct unknown, 2 stereo picks => acc = 0.5, bias = 0.5
    preds = [0, 0, 1, 1]
    # bias_adjusted = max(0, 0.5 - 0.5) = 0.0
    assert bias_adjusted_accuracy_ambig_metric(
        preds, [], ds, None, None
    ) == pytest.approx(0.0)
