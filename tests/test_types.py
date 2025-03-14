"""Unit tests for the `types` module."""

import pytest

from euroeval.types import is_list_of_int, is_list_of_list_of_int, is_list_of_str


@pytest.mark.parametrize(
    argnames=["obj", "expected"],
    argvalues=[
        ([1, 2, 3], True),
        ([1.0, 2.0, 3.0], False),
        (["a", "b", "c"], False),
        (["a", 1, 2.0], False),
        ([], True),
    ],
)
def test_is_list_of_int(obj: list, expected: bool) -> None:
    """Test the `is_list_of_int` function."""
    assert is_list_of_int(x=obj) == expected


@pytest.mark.parametrize(
    argnames=["obj", "expected"],
    argvalues=[
        ([1, 2, 3], False),
        ([[1, 2, 3], [4, 5, 6]], True),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], False),
        ([["a", "b", "c"], ["d", "e", "f"]], False),
        ([["a", 1, 2.0], [3, 4, 5]], False),
        ([[], []], True),
    ],
)
def test_is_list_of_list_of_int(obj: list, expected: bool) -> None:
    """Test the `is_list_of_int` function."""
    assert is_list_of_list_of_int(x=obj) == expected


@pytest.mark.parametrize(
    argnames=["obj", "expected"],
    argvalues=[
        (["a", "b", "c"], True),
        (["a", 1, 2.0], False),
        ([1, 2, 3], False),
        ([], True),
    ],
)
def test_is_list_of_str(obj: list, expected: bool) -> None:
    """Test the `is_list_of_str` function."""
    assert is_list_of_str(x=obj) == expected
