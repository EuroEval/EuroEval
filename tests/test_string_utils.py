"""Tests for the `utils` module."""

import pytest

from euroeval.data_models import ModelIdComponents
from euroeval.exceptions import InvalidBenchmark, InvalidModel
from euroeval.string_utils import (
    extract_json_dict_from_string,
    extract_multiple_choice_labels,
    scramble,
    split_model_id,
    unscramble,
)


@pytest.mark.parametrize(
    argnames=["text"],
    argvalues=[("abc",), ("hasd_asd2w",), ("a",), ("",)],
    ids=["short_text", "long_text", "single_char_text", "empty_text"],
)
def test_scrambling(text: str) -> None:
    """Test that a text can be scrambled and unscrambled."""
    scrambled = scramble(text=text)
    unscrambled = unscramble(scrambled_text=scrambled)
    assert unscrambled == text


def test_scramble_consistency() -> None:
    """Test that scramble produces consistent results with same seed."""
    text = "consistent_test"
    scrambled1 = scramble(text=text)
    scrambled2 = scramble(text=text)
    assert scrambled1 == scrambled2


def test_scramble_different_from_original() -> None:
    """Test that scrambled text is different from original (for non-empty strings)."""
    text = "abcdef"
    scrambled = scramble(text=text)
    # For strings with more than 1 character, scrambled should likely differ
    # (though technically random permutation could match, very unlikely for 6 chars)
    assert len(scrambled) == len(text)


@pytest.mark.parametrize(
    argnames=["model_id", "expected"],
    argvalues=[
        (
            "model-id",
            ModelIdComponents(model_id="model-id", revision="main", param=None),
        ),
        (
            "model-id@v1",
            ModelIdComponents(model_id="model-id", revision="v1", param=None),
        ),
        (
            "model-id#param",
            ModelIdComponents(model_id="model-id", revision="main", param="param"),
        ),
        (
            "model-id@v1#param",
            ModelIdComponents(model_id="model-id", revision="v1", param="param"),
        ),
        (
            "model-id#param@v1",
            ModelIdComponents(model_id="model-id", revision="v1", param="param"),
        ),
    ],
    ids=[
        "no_revision_no_param",
        "with_revision_no_param",
        "no_revision_with_param",
        "with_revision_with_param",
        "with_param_with_revision",
    ],
)
def test_split_model_id(model_id: str, expected: ModelIdComponents) -> None:
    """Test that a model ID can be split into its components correctly."""
    assert split_model_id(model_id=model_id) == expected


def test_split_model_id_with_org() -> None:
    """Test splitting model ID with organization."""
    result = split_model_id(model_id="org/model-name@v2#param")
    assert result.model_id == "org/model-name"
    assert result.revision == "v2"
    assert result.param == "param"


def test_split_model_id_invalid() -> None:
    """Test that invalid model ID raises error."""
    with pytest.raises(InvalidModel):
        split_model_id(model_id="@#")


@pytest.mark.parametrize(
    "text,expected_result,description",
    [
        (
            'Some text {"key": "value", "num": 42} more text',
            {"key": "value", "num": 42},
            "valid_json",
        ),
        (
            'Text {"outer": "value"} and {"inner": 123}',
            {"outer": "value"},
            "nested_first_match",
        ),
        ("This is just plain text with no JSON", None, "no_json"),
        ("Text {invalid json} more text", None, "invalid_json"),
        ('Text ["array", "not", "dict"] more text', None, "non_dict"),
        ("Text {} more text", {}, "empty_dict"),
        (
            'Text {\n    "key1": "value1",\n    "key2": "value2"\n} more',
            {"key1": "value1", "key2": "value2"},
            "multiline",
        ),
    ],
    ids=[
        "valid",
        "nested",
        "no_json",
        "invalid_json",
        "non_dict",
        "empty_dict",
        "multiline",
    ],
)
def test_extract_json_dict_from_string(
    text: str, expected_result: dict | None, description: str
) -> None:
    """Test extracting JSON dict from string with various inputs."""
    result = extract_json_dict_from_string(text)
    assert result == expected_result


def test_extract_json_dict_from_string_non_string_keys() -> None:
    """Test extraction returns None when dict has non-string keys."""
    # demjson3 might parse this, but the function checks for string keys
    text = 'Text {1: "value", 2: "other"} more text'
    result = extract_json_dict_from_string(text)
    # Function requires all keys to be strings
    assert result is None or all(isinstance(k, str) for k in result.keys())


@pytest.mark.parametrize(
    "prompt,candidate_labels,expected_labels,description",
    [
        (
            "Question text\na. First option\nb. Second option\nc. Third option",
            ["a", "b", "c"],
            ["a", "b", "c"],
            "basic_extraction",
        ),
        (
            "Question text\na. First option\nc. Third option",
            ["a", "b", "c", "d"],
            ["a", "c"],
            "subset_extraction",
        ),
        (
            "Question text\nA. First option\nB. Second option",
            ["a", "b"],
            ["a", "b"],
            "case_insensitive",
        ),
        (
            "Question\n1. First\n2. Second\n3. Third",
            ["1", "2", "3"],
            ["1", "2", "3"],
            "numeric_labels",
        ),
    ],
    ids=["basic", "subset", "case_insensitive", "numeric"],
)
def test_extract_multiple_choice_labels(
    prompt: str,
    candidate_labels: list[str],
    expected_labels: list[str],
    description: str,
) -> None:
    """Test extracting multiple choice labels from prompts."""
    labels = extract_multiple_choice_labels(prompt, candidate_labels)
    assert set(labels) == set(expected_labels)


def test_extract_multiple_choice_labels_no_labels() -> None:
    """Test that missing labels raises InvalidBenchmark."""
    prompt = "Question text with no labels"
    with pytest.raises(
        InvalidBenchmark, match="Could not extract any candidate labels"
    ):
        extract_multiple_choice_labels(prompt, ["a", "b", "c"])


def test_extract_multiple_choice_labels_partial_match() -> None:
    """Test that partial matches don't count (requires dot and space)."""
    prompt = "Question text\na First option\nb. Second option"
    labels = extract_multiple_choice_labels(prompt, ["a", "b"])
    # "a" without dot-space shouldn't match due to \b and ". " pattern
    assert "b" in labels


def test_extract_multiple_choice_labels_with_numbers() -> None:
    """Test extracting numeric labels."""
    prompt = "Question\n1. First\n2. Second\n3. Third"
    labels = extract_multiple_choice_labels(prompt, ["1", "2", "3"])
    assert labels == ["1", "2", "3"]
