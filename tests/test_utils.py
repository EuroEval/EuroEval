"""Tests for the `utils` module."""

import random

import numpy as np
import pytest
import torch

from euroeval.data_models import ModelIdComponents
from euroeval.utils import enforce_reproducibility, scramble, split_model_id, unscramble


class TestEnforceReproducibility:
    """Tests for the `enforce_reproducibility` function."""

    def test_random_arrays_not_equal(self) -> None:
        """Test that two random arrays are not equal."""
        first_random_number = random.random()
        second_random_number = random.random()
        assert first_random_number != second_random_number

    def test_random_arrays_equal(self) -> None:
        """Test that two random arrays are equal after enforcing reproducibility."""
        enforce_reproducibility()
        first_random_number = random.random()
        enforce_reproducibility()
        second_random_number = random.random()
        assert first_random_number == second_random_number

    def test_numpy_arrays_not_equal(self) -> None:
        """Test that two random numpy arrays are not equal."""
        first_random_numbers = np.random.rand(10)
        second_random_numbers = np.random.rand(10)
        assert not np.array_equal(first_random_numbers, second_random_numbers)

    def test_numpy_arrays_equal(self) -> None:
        """Test that two random arrays are equal after enforcing reproducibility."""
        enforce_reproducibility()
        first_random_numbers = np.random.rand(10)
        enforce_reproducibility()
        second_random_numbers = np.random.rand(10)
        assert np.array_equal(first_random_numbers, second_random_numbers)

    def test_pytorch_tensors_not_equal(self) -> None:
        """Test that two random pytorch tensors are not equal."""
        first_random_numbers = torch.rand(10)
        second_random_numbers = torch.rand(10)
        assert not torch.equal(first_random_numbers, second_random_numbers)

    def test_pytorch_tensors_equal(self) -> None:
        """Test that two random tensors are equal after enforcing reproducibility."""
        enforce_reproducibility()
        first_random_numbers = torch.rand(10)
        enforce_reproducibility()
        second_random_numbers = torch.rand(10)
        assert torch.equal(first_random_numbers, second_random_numbers)


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
