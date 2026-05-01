"""Tests for the `task_group_utils.cloze` module."""

import math

import pytest

from euroeval.data_models import GenerativeModelOutput
from euroeval.enums import CFNormalization
from euroeval.exceptions import InvalidBenchmark
from euroeval.task_group_utils import cloze


class TestNormalizeCfScore:
    """Tests for `normalize_cf_score`."""

    token_lps = [-1.0, -2.0, -3.0]

    def test_none(self) -> None:
        """Raw sum is returned when no normalization is applied."""
        score = cloze.normalize_cf_score(
            token_logprobs=self.token_lps,
            answer_text="abcd",
            method=CFNormalization.NONE,
        )
        assert math.isclose(score, -6.0)

    def test_token(self) -> None:
        """Sum is divided by number of tokens."""
        score = cloze.normalize_cf_score(
            token_logprobs=self.token_lps,
            answer_text="abcd",
            method=CFNormalization.TOKEN,
        )
        assert math.isclose(score, -6.0 / 3)

    def test_character(self) -> None:
        """Sum is divided by number of characters."""
        score = cloze.normalize_cf_score(
            token_logprobs=self.token_lps,
            answer_text="abcd",
            method=CFNormalization.CHARACTER,
        )
        assert math.isclose(score, -6.0 / 4)

    def test_empty_tokens_does_not_divide_by_zero(self) -> None:
        """Empty logprob sequences return 0 without raising."""
        score = cloze.normalize_cf_score(
            token_logprobs=[], answer_text="abcd", method=CFNormalization.TOKEN
        )
        assert score == 0.0

    def test_empty_answer_does_not_divide_by_zero(self) -> None:
        """Empty answer strings still produce a finite character-normalized score."""
        score = cloze.normalize_cf_score(
            token_logprobs=self.token_lps,
            answer_text="",
            method=CFNormalization.CHARACTER,
        )
        assert math.isclose(score, -6.0)


class TestExtractLabelsFromCf:
    """Tests for `extract_labels_from_cf`."""

    def test_argmax_selects_highest_scoring_choice(self) -> None:
        """The highest-scoring candidate per sample becomes the predicted letter."""
        output = GenerativeModelOutput(
            sequences=["", ""],
            cf_scores=[[-1.0, -0.5, -2.0, -1.5], [-3.0, -2.5, -2.0, -2.7]],
        )
        labels = cloze.extract_labels_from_cf(
            input_batch={"prompt": ["", ""]},
            model_output=output,
            dataset_config=None,  # pyrefly: ignore[bad-argument-type]
            model_config=None,  # pyrefly: ignore[bad-argument-type]
            first_label_token_mapping=False,
        )
        assert list(labels) == ["b", "c"]

    def test_missing_cf_scores_raises(self) -> None:
        """Extractor complains loudly if CF was not actually executed."""
        output = GenerativeModelOutput(sequences=[""], cf_scores=None)
        with pytest.raises(InvalidBenchmark):
            cloze.extract_labels_from_cf(
                input_batch={"prompt": [""]},
                model_output=output,
                dataset_config=None,  # pyrefly: ignore[bad-argument-type]
                model_config=None,  # pyrefly: ignore[bad-argument-type]
                first_label_token_mapping=False,
            )


class TestParseMcqText:
    """Tests for `parse_mcq_text`."""

    def test_standard_mcq_format(self) -> None:
        """Parses EuroEval-formatted MCQ text with a 'Choices:' header."""
        text = (
            "Tekst: De aarde is de derde planeet.\n"
            "Vraag: Welke planeet is de derde?\n"
            "Keuzes:\n"
            "a. Mercurius\n"
            "b. Venus\n"
            "c. Aarde\n"
            "d. Mars"
        )
        bare, choices = cloze.parse_mcq_text(text)
        assert "Mercurius" not in bare
        assert "Keuzes:" not in bare
        assert choices == ["Mercurius", "Venus", "Aarde", "Mars"]

    def test_no_choices_header(self) -> None:
        """Parses MCQ text without a 'Choices:' header line."""
        text = "Question?\na. alpha\nb. beta\nc. gamma\nd. delta"
        bare, choices = cloze.parse_mcq_text(text)
        assert bare == "Question?"
        assert choices == ["alpha", "beta", "gamma", "delta"]

    def test_no_choices_raises(self) -> None:
        """Input without enumerated choices raises `InvalidBenchmark`."""
        with pytest.raises(InvalidBenchmark):
            cloze.parse_mcq_text("Just a question with no choices.")


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


class TestRenderCfFewShot:
    """Tests for `render_cf_few_shot`."""

    def test_substitutes_full_answer_text(self) -> None:
        """The {label} slot is filled with the full answer text, not a letter."""
        rendered = cloze.render_cf_few_shot(
            bare_input="Wat is de hoofdstad van NL?",
            answer_text="Amsterdam",
            prompt_template="Vraag: {text}\nAntwoord: {label}",
        )
        assert rendered == "Vraag: Wat is de hoofdstad van NL?\nAntwoord: Amsterdam"

    def test_strips_newlines_in_inputs(self) -> None:
        """Embedded newlines in the question and answer are flattened to spaces.

        This matches the behaviour of `merge_input_and_choices` so that bare-input
        and rendered few-shot are tokenised the same way.
        """
        rendered = cloze.render_cf_few_shot(
            bare_input="Line 1\nLine 2",
            answer_text="Ans 1\nAns 2",
            prompt_template="Q: {text}\nA: {label}",
        )
        assert rendered == "Q: Line 1 Line 2\nA: Ans 1 Ans 2"


class TestBuildCfPrompt:
    """Tests for `build_cf_prompt`."""

    template = "Vraag: {text}\nAntwoord: {label}"

    def test_zero_shot_no_prefix(self) -> None:
        """With no prefix and no few-shot, the prompt is just the rendered template."""
        prompt = cloze.build_cf_prompt(
            bare_input="Wat is 2+2?",
            few_shot_rendered=[],
            prompt_template=self.template,
            prompt_prefix="",
        )
        assert prompt == "Vraag: Wat is 2+2?\nAntwoord: "

    def test_with_prefix_strips_labels_str(self) -> None:
        """Prefix appears first; `{labels_str}` is replaced with an empty string.

        CF does not enumerate label letters, so the prefix's `{labels_str}` slot
        (which would otherwise list "a/b/c/d") must collapse to nothing.
        """
        prompt = cloze.build_cf_prompt(
            bare_input="Wat is 2+2?",
            few_shot_rendered=[],
            prompt_template=self.template,
            prompt_prefix="De volgende vragen ({labels_str}).",
        )
        assert prompt == "De volgende vragen ().\n\nVraag: Wat is 2+2?\nAntwoord: "

    def test_with_few_shot_uses_blank_line_separator(self) -> None:
        """Few-shot blocks are joined with blank lines and precede the bare prompt."""
        prompt = cloze.build_cf_prompt(
            bare_input="Q3?",
            few_shot_rendered=["Vraag: Q1?\nAntwoord: A1", "Vraag: Q2?\nAntwoord: A2"],
            prompt_template=self.template,
            prompt_prefix="",
        )
        assert prompt == (
            "Vraag: Q1?\nAntwoord: A1\n\n"
            "Vraag: Q2?\nAntwoord: A2\n\n"
            "Vraag: Q3?\nAntwoord: "
        )

    def test_ends_with_label_marker(self) -> None:
        """The prompt ends with whatever the template renders for an empty label.

        This is the contract that `score_completions` relies on: it appends the
        candidate text directly to the returned prompt.
        """
        prompt = cloze.build_cf_prompt(
            bare_input="Q?",
            few_shot_rendered=[],
            prompt_template="Q: {text}\nA: {label}",
            prompt_prefix="",
        )
        assert prompt.endswith("A: ")

    def test_strips_newlines_in_bare_input(self) -> None:
        """Embedded newlines in the bare question are flattened, not preserved."""
        prompt = cloze.build_cf_prompt(
            bare_input="Line 1\nLine 2",
            few_shot_rendered=[],
            prompt_template=self.template,
            prompt_prefix="",
        )
        assert "Line 1\nLine 2" not in prompt
        assert "Line 1 Line 2" in prompt
