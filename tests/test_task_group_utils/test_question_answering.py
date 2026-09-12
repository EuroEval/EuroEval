"""Tests for the `task_group_utils.question_answering` module."""

import numpy as np

from euroeval.task_group_utils.question_answering import (
    _answer_text_from_offsets,
    find_valid_answers,
)


class TestAnswerTextFromOffsets:
    """Tests for reconstructing answer text from token offset mappings."""

    def test_overflow_feature_starting_mid_word_is_not_expanded(self) -> None:
        """A stride that starts mid-word must not pull in the missing prefix."""
        context = "xxAmsterdam"
        # Feature starts at 'm' (index 3) inside "Amsterdam"; previous token is
        # not in this context, and the character before is not whitespace.
        offsets = [(-1, -1), (3, 11)]
        assert _answer_text_from_offsets(offsets, 1, 1, context) == "msterdam"

    def test_robbert_word_initial_offset_is_repaired(self) -> None:
        """A word-initial start offset of +1 still yields the full word.

        This is the RobBERT-family ByteLevel hole from
        https://github.com/EuroEval/EuroEval/issues/2171.
        """
        context = "Amsterdam is the capital"
        # First context token claims (1, 9) instead of (0, 9); previous token is
        # a question/special token marked as not in the context.
        offsets = [(-1, -1), (1, 9), (10, 12), (13, 16), (17, 24)]
        assert _answer_text_from_offsets(offsets, 1, 1, context) == "Amsterdam"

    def test_standard_offsets_are_unchanged(self) -> None:
        """Contiguous BERT-style offsets slice the context as reported."""
        context = "The capital of France is Paris."
        #                    0123456789012345678901234567890
        offsets = [(0, 3), (4, 11), (12, 14), (15, 21), (22, 24), (25, 30), (30, 31)]
        assert _answer_text_from_offsets(offsets, 5, 5, context) == "Paris"

    def test_subword_continuation_is_not_expanded(self) -> None:
        """A continuation subword whose previous token is in-context is left as-is."""
        context = "playing"
        offsets = [(0, 4), (4, 7)]
        assert _answer_text_from_offsets(offsets, 1, 1, context) == "ing"

    def test_word_initial_offset_after_space_is_repaired(self) -> None:
        """The same +1 hole after a previous word is filled."""
        context = "in Amsterdam today"
        # "Amsterdam" is [3:12]; the token reports (4, 12).
        offsets = [(-1, -1), (0, 2), (4, 12), (13, 18)]
        assert _answer_text_from_offsets(offsets, 2, 2, context) == "Amsterdam"


class TestFindValidAnswersRobbertOffsets:
    """`find_valid_answers` must emit the repaired span, not the truncated one."""

    def test_best_answer_keeps_the_leading_letter(self) -> None:
        """The predicted span is the full gold word despite a +1 start offset."""
        context = "Amsterdam is the capital"
        offset_mapping = [(-1, -1), (1, 9), (10, 12), (13, 16), (17, 24)]
        start_logits = np.array([-10.0, 5.0, 0.0, 0.0, 0.0])
        end_logits = np.array([-10.0, 5.0, 0.0, 0.0, 0.0])
        answers = find_valid_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            offset_mapping=offset_mapping,
            context=context,
            max_answer_length=30,
            num_best_logits=5,
            min_null_score=-100.0,
        )
        texts = {answer["text"] for answer in answers}
        assert "Amsterdam" in texts
        assert "msterdam" not in texts
