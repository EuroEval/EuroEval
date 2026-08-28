"""Tests for date utilities."""

import datetime

import pytest

from euroeval.date_utils import normalise_release_date


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-02-03", "2024-02-03"),
        (datetime.date(2024, 2, 3), "2024-02-03"),
        (datetime.datetime(2024, 2, 3, 12, 30), "2024-02-03"),
        (None, None),
        ("", None),
        ("not-a-date", None),
        ("2025-02-30", None),
        ("20240203", None),
        ("2024-W05-6", None),
        (20240203, None),
    ],
)
def test_normalise_release_date(value: object, expected: str | None) -> None:
    """Release dates are normalised consistently across all consumers."""
    assert normalise_release_date(value) == expected
