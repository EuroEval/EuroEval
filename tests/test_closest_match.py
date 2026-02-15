"""Tests for the `closest_match` module."""

import pytest

from euroeval.closest_match import get_closest_match


def test_get_closest_match_exact() -> None:
    """Test that exact matches are found correctly."""
    string = "apple"
    options = ["apple", "banana", "orange"]
    match, distance = get_closest_match(string, options, case_sensitive=True)
    assert match == "apple"
    assert distance == 0


def test_get_closest_match_case_insensitive() -> None:
    """Test case-insensitive matching."""
    string = "APPLE"
    options = ["apple", "banana", "orange"]
    match, distance = get_closest_match(string, options, case_sensitive=False)
    assert match == "apple"
    assert distance == 0


def test_get_closest_match_case_sensitive() -> None:
    """Test case-sensitive matching."""
    string = "APPLE"
    options = ["apple", "banana", "orange"]
    match, distance = get_closest_match(string, options, case_sensitive=True)
    # Should match "apple" but with distance > 0 due to case difference
    assert match == "apple"
    assert distance == 5  # All 5 characters are different case


def test_get_closest_match_single_char_diff() -> None:
    """Test matching with single character difference."""
    string = "appl"
    options = ["apple", "apply", "apples"]
    match, distance = get_closest_match(string, options, case_sensitive=True)
    # Both "apple" and "apply" are distance 1, but min returns first
    assert match in ["apple", "apply"]
    assert distance == 1


def test_get_closest_match_custom_weights() -> None:
    """Test custom insertion, deletion, and substitution weights."""
    string = "cat"
    options = ["bat", "ca", "cats"]

    # With default weights
    match1, distance1 = get_closest_match(string, options, case_sensitive=True)
    assert match1 == "bat"  # 1 substitution
    assert distance1 == 1

    # With higher substitution weight
    match2, distance2 = get_closest_match(
        string, options, case_sensitive=True, substitution_weight=10
    )
    # Now "ca" (1 deletion, weight 1) should be closer than
    # "bat" (1 substitution, weight 10)
    assert match2 == "ca"
    assert distance2 == 1


def test_get_closest_match_empty_string() -> None:
    """Test matching with an empty string."""
    string = ""
    options = ["a", "ab", "abc"]
    match, distance = get_closest_match(string, options, case_sensitive=True)
    # Empty string closest to "a" (distance 1 - one insertion)
    assert match == "a"
    assert distance == 1


def test_get_closest_match_multiple_words() -> None:
    """Test matching with longer strings."""
    string = "hello world"
    options = ["hello world", "hello word", "helo world"]
    match, distance = get_closest_match(string, options, case_sensitive=True)
    assert match == "hello world"
    assert distance == 0


@pytest.mark.parametrize(
    "string,options,case_sensitive,expected_match,expected_distance",
    [
        ("test", ["test", "best", "rest"], True, "test", 0),
        ("test", ["TEST", "BEST", "REST"], False, "TEST", 0),
        ("abc", ["xyz", "def", "ghi"], True, "xyz", 3),
        ("color", ["colour", "colors", "colored"], True, "colour", 1),
    ],
    ids=["exact_match", "case_insensitive", "no_similar", "british_spelling"],
)
def test_get_closest_match_parametrized(
    string: str,
    options: list[str],
    case_sensitive: bool,
    expected_match: str,
    expected_distance: int,
) -> None:
    """Parametrized test for various matching scenarios."""
    match, distance = get_closest_match(string, options, case_sensitive)
    assert match == expected_match
    assert distance == expected_distance
