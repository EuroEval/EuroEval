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


class TestParseBareQuestionAndChoices:
    """Tests for `parse_bare_question_and_choices`."""

    def test_splits_question_and_choices(self) -> None:
        """The bare question and choice texts are recovered from a formatted prompt."""
        text = (
            "Hvad er hovedstaden i Danmark?\n"
            "Svarmuligheder:\n"
            "a. København\n"
            "b. Aarhus\n"
            "c. Odense\n"
            "d. Aalborg"
        )
        question, choices = cloze.parse_bare_question_and_choices(text)
        assert question == "Hvad er hovedstaden i Danmark?"
        assert choices == ["København", "Aarhus", "Odense", "Aalborg"]

    def test_strips_trailing_choices_label(self) -> None:
        """A trailing choices-label line ending in a colon is removed."""
        text = "Question: foo?\nChoices:\na. x\nb. y"
        question, choices = cloze.parse_bare_question_and_choices(text)
        assert question == "Question: foo?"
        assert choices == ["x", "y"]

    def test_no_choices_returns_text_unchanged(self) -> None:
        """Text without enumerated options yields the original text and no choices."""
        question, choices = cloze.parse_bare_question_and_choices("Just a sentence.")
        assert question == "Just a sentence."
        assert choices == []

    def test_multiline_question(self) -> None:
        """A question spanning multiple lines is preserved in full."""
        text = "Line one.\nLine two?\nChoices:\na. first\nb. second"
        question, choices = cloze.parse_bare_question_and_choices(text)
        assert question == "Line one.\nLine two?"
        assert choices == ["first", "second"]

    def test_question_line_starting_with_enumerator_is_kept(self) -> None:
        """A question line that itself starts with ``N.`` is not mistaken for a choice.

        Only the final contiguous block of enumerated lines counts as choices, so an
        enumerated line earlier in the question is preserved as part of the question.
        """
        text = "1. What is true?\nChoices:\na. first\nb. second"
        question, choices = cloze.parse_bare_question_and_choices(text)
        assert question == "1. What is true?"
        assert choices == ["first", "second"]
