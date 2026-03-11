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


def test_split_model_id_invalid_raises_error() -> None:
    """Test InvalidModel raised for invalid model ID."""
    with pytest.raises(InvalidModel):
        split_model_id(model_id="")


@pytest.mark.parametrize(
    argnames=["text"],
    argvalues=[("abc",), ("hasd_asd2w",), ("a",), ("",)],
    ids=["short_text", "long_text", "single_char_text", "empty_text"],
)
def test_scramble_and_unscramble_roundtrip(text: str) -> None:
    """Test that scrambling and unscrambling restores original string."""
    scrambled = scramble(text=text)
    unscrambled = unscramble(scrambled_text=scrambled)
    assert unscrambled == text


def test_scramble_deterministic_with_seed() -> None:
    """Test that scramble produces same result with same seed."""
    text = "test text for scrambling"
    scrambled_first = scramble(text=text)
    scrambled_second = scramble(text=text)
    assert scrambled_first == scrambled_second


@pytest.mark.parametrize(
    argnames=["s", "expected"],
    argvalues=[
        ('{"key": "value"}', {"key": "value"}),
        ('prefix {"key": "value"} suffix', {"key": "value"}),
        ('line1\n{   "key": 123   }\nline3', {"key": 123}),
    ],
    ids=["plain_json", "with_prefix_suffix", "with_whitespace"],
)
def test_extract_json_dict_from_string_with_prefix_suffix(
    s: str, expected: dict
) -> None:
    """Test JSON extraction from string with surrounding text."""
    assert extract_json_dict_from_string(s=s) == expected


def test_extract_json_dict_from_string_no_json_returns_none() -> None:
    """Test None returned when no JSON found."""
    assert extract_json_dict_from_string(s="no json here") is None


def test_extract_json_dict_from_string_invalid_json_returns_none() -> None:
    """Test None returned when JSON is invalid."""
    assert extract_json_dict_from_string(s='{"key": }') is None


def test_extract_json_dict_from_string_non_dict_returns_none() -> None:
    """Test None returned when JSON is not a dictionary."""
    assert extract_json_dict_from_string(s='["array"]') is None


def test_extract_json_dict_from_string_non_string_keys_returns_none() -> None:
    """Test None returned when JSON has non-string keys."""
    result = extract_json_dict_from_string(s="{123: 'value'}")
    assert result is None


@pytest.mark.parametrize(
    argnames=["prompt", "candidate_labels", "expected"],
    argvalues=[
        (
            "What is the capital of France? a. Paris b. London c. Berlin",
            ["a", "b", "c"],
            ["a", "b", "c"],
        ),
        ("Options: A. Yes B. No", ["a", "b", "c"], ["a", "b"]),
    ],
    ids=["basic_labels", "uppercase_labels"],
)
def test_extract_multiple_choice_labels_with_labels(
    prompt: str, candidate_labels: list[str], expected: list[str]
) -> None:
    """Test label extraction with provided candidate labels."""
    assert (
        extract_multiple_choice_labels(prompt=prompt, candidate_labels=candidate_labels)
        == expected
    )


def test_extract_multiple_choice_labels_without_labels() -> None:
    """Test label extraction falls back to alphabet."""
    prompt = "Choose one: a. option1 b. option2 c. option3"
    result = extract_multiple_choice_labels(prompt=prompt, candidate_labels=[])
    assert result == ["a", "b", "c"]


def test_extract_multiple_choice_labels_case_insensitive() -> None:
    """Test case-insensitive label extraction."""
    prompt = "Answers: A. true B. false C. maybe"
    result = extract_multiple_choice_labels(prompt=prompt, candidate_labels=[])
    assert result == ["a", "b", "c"]


def test_extract_multiple_choice_labels_no_match_raises_error() -> None:
    """Test InvalidBenchmark raised when no labels found."""
    prompt = "No labels here, just text"
    with pytest.raises(InvalidBenchmark):
        extract_multiple_choice_labels(prompt=prompt, candidate_labels=[])
