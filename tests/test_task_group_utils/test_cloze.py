"""Tests for the `task_group_utils.cloze` module.

Only tests for `letter_to_choice_text` remain - all CF-specific functions
were removed when BPC scoring was simplified to treat MCQ like text-to-text.
"""

import pytest

from euroeval.exceptions import InvalidBenchmark
from euroeval.task_group_utils import cloze


class TestLetterToChoiceText:
    """Tests for `letter_to_choice_text`."""

    choices = ["apple", "banana", "cherry", "date"]

    def test_lowercase_letter(self) -> None:
        """The letter indexes into the choices list in order."""
        assert cloze.letter_to_choice_text("a", self.choices) == "apple"
        assert cloze.letter_to_choice_text("c", self.choices) == "cherry"

    def test_uppercase_letter_is_lowered(self) -> None:
        """Upper-case letters are accepted and normalised."""
        assert cloze.letter_to_choice_text("D", self.choices) == "date"

    def test_out_of_range_raises(self) -> None:
        """Letters beyond the provided choices raise `InvalidBenchmark`."""
        with pytest.raises(InvalidBenchmark):
            cloze.letter_to_choice_text("e", self.choices)

    def test_invalid_letter_raises(self) -> None:
        """Non-letters raise `InvalidBenchmark`."""
        with pytest.raises(InvalidBenchmark):
            cloze.letter_to_choice_text("!", self.choices)
