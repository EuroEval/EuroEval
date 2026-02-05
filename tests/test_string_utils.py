"""Tests for the `utils` module."""

import pytest

from euroeval.data_models import ModelIdComponents
from euroeval.string_utils import scramble, split_model_id, unscramble


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
