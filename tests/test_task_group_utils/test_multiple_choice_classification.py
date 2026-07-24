"""Tests for the `task_group_utils.multiple_choice_classification` module."""

from unittest.mock import MagicMock

import pytest

from euroeval.exceptions import InvalidBenchmark
from euroeval.task_group_utils import multiple_choice_classification


class TestPrepareExamples:
    """Tests for `prepare_examples`."""

    @pytest.fixture
    def mock_tokeniser(self) -> MagicMock:
        """Create a mock tokeniser that returns nested input_ids.

        Returns:
            A MagicMock tokeniser that returns a dict with nested input_ids.
        """
        tokeniser = MagicMock()
        # When called, return a dict with flat lists that will be regrouped
        tokeniser.return_value = {
            "input_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],  # 4 examples
            "attention_mask": [[1, 1], [1, 1], [1, 1], [1, 1]],
        }
        return tokeniser

    @pytest.fixture
    def valid_batch(self) -> dict:
        """Create a valid batch with 2 MC questions, each with 2 choices.

        Returns:
            A batch dict with 'text' and 'label' keys.
        """
        return {
            "text": [
                "What is 2+2?\nChoices:\na. 3\nb. 4",
                "What is 3+3?\nChoices:\na. 5\nb. 6",
            ],
            "label": ["b", "b"],  # Both answers are "b"
        }

    def test_valid_input_produces_nested_output(
        self, mock_tokeniser: MagicMock, valid_batch: dict
    ) -> None:
        """Valid MC input produces correctly nested tokeniser output."""
        result = multiple_choice_classification.prepare_examples(
            examples=valid_batch,  # ty: ignore[invalid-argument-type]
            tokeniser=mock_tokeniser,
            num_choices=2,
        )

        # Check tokeniser was called
        mock_tokeniser.assert_called_once()

        # Check output is nested: 2 questions, each with 2 choices
        assert len(result["input_ids"]) == 2  # 2 questions
        assert len(result["input_ids"][0]) == 2  # Each has 2 choices
        assert len(result["input_ids"][1]) == 2

        # Check labels are integers
        assert result["label"] == [1, 1]  # "b" -> index 1

    def test_empty_choices_raises(self, mock_tokeniser: MagicMock) -> None:
        """A document with no choices raises `InvalidBenchmark`."""
        batch = {"text": ["Just a question, no choices."], "label": ["a"]}

        with pytest.raises(InvalidBenchmark, match="No choices found"):
            multiple_choice_classification.prepare_examples(
                examples=batch,  # ty: ignore[invalid-argument-type]
                tokeniser=mock_tokeniser,
                num_choices=2,
            )

    def test_mismatched_choice_count_raises(
        self, mock_tokeniser: MagicMock, valid_batch: dict
    ) -> None:
        """A question with wrong number of choices raises `InvalidBenchmark`."""
        # First question has 2 choices, second has 3
        batch = {
            "text": [
                "What is 2+2?\nChoices:\na. 3\nb. 4",
                "What is 3+3?\nChoices:\na. 5\nb. 6\nc. 7",
            ],
            "label": ["b", "c"],
        }

        with pytest.raises(
            InvalidBenchmark,
            match="Multiple-choice example has 3 choices, but this dataset requires 2",
        ):
            multiple_choice_classification.prepare_examples(
                examples=batch,  # ty: ignore[invalid-argument-type]
                tokeniser=mock_tokeniser,
                num_choices=2,
            )

    def test_invalid_gold_label_raises(
        self, mock_tokeniser: MagicMock, valid_batch: dict
    ) -> None:
        """A gold label outside the valid range raises `InvalidBenchmark`."""
        # Second question has gold label "d" but only 2 choices (a, b)
        batch = {
            "text": [
                "What is 2+2?\nChoices:\na. 3\nb. 4",
                "What is 3+3?\nChoices:\na. 5\nb. 6",
            ],
            "label": ["b", "d"],  # "d" is invalid for 2-choice question
        }

        with pytest.raises(
            InvalidBenchmark, match="Gold label 'd' is not a valid choice"
        ):
            multiple_choice_classification.prepare_examples(
                examples=batch,  # ty: ignore[invalid-argument-type]
                tokeniser=mock_tokeniser,
                num_choices=2,
            )

    def test_uppercase_gold_label_accepted(
        self, mock_tokeniser: MagicMock, valid_batch: dict
    ) -> None:
        """Uppercase gold labels are normalised to lowercase."""
        batch = {
            "text": ["What is 2+2?\nChoices:\na. 3\nb. 4"],
            "label": ["B"],  # Uppercase
        }

        result = multiple_choice_classification.prepare_examples(
            examples=batch,  # ty: ignore[invalid-argument-type]
            tokeniser=mock_tokeniser,
            num_choices=2,
        )

        assert result["label"] == [1]  # "B" -> "b" -> index 1

    def test_dynamic_choice_count_from_first_document(
        self, mock_tokeniser: MagicMock
    ) -> None:
        """When num_choices=0, count is inferred from first document."""
        batch = {"text": ["What is 2+2?\nChoices:\na. 3\nb. 4\nc. 5"], "label": ["c"]}

        result = multiple_choice_classification.prepare_examples(
            examples=batch,  # ty: ignore[invalid-argument-type]
            tokeniser=mock_tokeniser,
            num_choices=0,
        )

        # Should work with 3 choices inferred
        mock_tokeniser.assert_called_once()
        assert result["label"] == [2]  # "c" -> index 2
