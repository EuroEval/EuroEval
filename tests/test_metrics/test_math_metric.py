"""Tests for the mathematical exact-match metric."""

import time

import pytest

from euroeval.metrics.math import (
    MathAccuracy,
    _answer_candidates,
    _answer_matches,
    _boxed_candidates,
    _equivalent,
    _symbolically_equivalent,
)


class TestBailOut:
    """Test that anything untranslatable is passed back to the caller."""

    @pytest.mark.parametrize(
        "text",
        [
            r"\begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}",
            r"\int_0^1 x \, dx",
            r"\text{Yes}",
            r"\vec{v}",
            "the same as before",
        ],
    )
    def test_structure_beyond_the_rewrite_is_unknown(self, text: str) -> None:
        """Untranslatable input is reported as unknown instead of guessed."""
        assert _symbolically_equivalent(text, "1") is None

    def test_unknown_falls_back_to_text(self) -> None:
        """A comparison this module abstains from is still answered as text."""
        assert _answer_matches(r"\boxed{\text{Ja}}", "Ja")
        assert not _answer_matches(r"\boxed{\text{Ja}}", "Nej")


class TestLimits:
    """Test that a degenerate answer cannot make comparison run away."""

    def test_a_long_decimal_is_answered_without_stalling(self) -> None:
        """A long number is compared rather than refused, and in no time at all."""
        text = "9" * 600 + "." + "9" * 600
        start = time.perf_counter()
        assert _symbolically_equivalent(text, "1") is False
        assert time.perf_counter() - start < 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "9" * 5000,
            r"9^{9^{9}}",
            r"2^{100000}",
            "2**-1000000000",
            "2**-10000000000",
            "2**(0-1000000000)",
            "2^{" * 1000 + "2" + "}" * 1000,
            "(" + "1," * 400 + "1)",
            "",
            "   ",
        ],
    )
    def test_pathological_input_is_refused_quickly(self, text: str) -> None:
        """Pathological candidates are unknown, and unknown within milliseconds."""
        start = time.perf_counter()
        assert _symbolically_equivalent(text, "1") is None
        assert time.perf_counter() - start < 1.0


class TestMathHelpers:
    """Test extraction and comparison behaviour."""

    def test_boxed_reference_and_fraction(self) -> None:
        """Boxed references work, and a LaTeX fraction equals its value."""
        assert _equivalent("42", r"\boxed{42}")
        assert _equivalent(r"\frac{1}{2}", "0.5")

    def test_fbox_and_fallback(self) -> None:
        """Fbox and prose/number fallback are supported."""
        assert _equivalent(r"\fbox{12}", "12")
        assert _answer_matches("The answer is 12 because it follows.", "12")

    def test_last_balanced_box_wins(self) -> None:
        """Nested braces are balanced and the final box is selected."""
        assert _boxed_candidates(r"\boxed{1+\{2\}} then \boxed{42}")[-1] == "42"

    def test_numeric_equality_is_exact(self) -> None:
        """Plain numeric values require exact equality."""
        assert not _equivalent("1", "1.00000000001")
        assert not _equivalent("99999999999", "100000000000")

    def test_unicode_minus_and_percent_scales(self) -> None:
        """Unicode minus and single-scale percentages compare."""
        assert _equivalent("−2", "-2")
        assert _equivalent(r"50\%", "0.5")
        assert not _equivalent(r"50\%", "50")
        assert _equivalent(r"50 \text{percent}", "0.5")

    def test_wrappers_and_grouping(self) -> None:
        """Currency, delimiters, Markdown and comma grouping are normalised."""
        assert _equivalent("**$1,234$**", "1234")
        assert _equivalent(r"1\,234", "1234")
        assert not _equivalent("1 234", "1234")
        assert not _equivalent("10\n20", "1020")


class TestPercentages:
    """Test that a percentage is its value divided by 100, exactly."""

    def test_a_long_percentage_is_not_rounded(self) -> None:
        """A percentage longer than the decimal context keeps its neighbours apart."""
        assert not _answer_matches("9" * 30 + "%", "9" * 29 + "8%")

    def test_a_percent_sign_mid_expression_is_not_modulo(self) -> None:
        """`50% + 10%` is not a remainder, and dividing by zero is not an answer."""
        assert not _answer_matches("50% + 10%", "0")
        assert not _answer_matches("1 % 0", "1")

    def test_a_percentage_matches_its_value(self) -> None:
        """`50%` is 0.5, and is not 50."""
        assert _answer_matches(r"50\%", "0.5")
        assert _answer_matches("50%", "0.5")
        assert not _answer_matches(r"50\%", "50")


class TestSymbolicEquivalence:
    """Test values that only a mathematical comparison can equate."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (r"\frac{1}{2}", "0.5"),
            (r"\dfrac{2}{4}", "0.5"),
            (r"-\frac{1}{4}", "-0.25"),
            (r"2\pi", "6.283185307179586"),
            (r"\sqrt{4}", "2"),
            (r"\sqrt{8}", r"2\sqrt{2}"),
            (r"\sqrt[3]{27}", "3"),
            (r"2^{10}", "1024"),
            (r"3 \cdot 4", "12"),
            (r"12 \div 4", "3"),
            (r"0.1 + 0.2", "0.3"),
            (r"1\,234", "1234"),
            (r"50\%", "0.5"),
            (r"\frac{1}{2} + \frac{1}{4}", "0.75"),
        ],
    )
    def test_expressions_equal_their_values(self, left: str, right: str) -> None:
        """An expression is compared as the value it denotes."""
        assert _symbolically_equivalent(left, right)
        assert _answer_matches(r"\boxed{" + left + "}", right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("0.5", r"\frac{1}{3}"),
            (r"\frac{1}{2}", "0.51"),
            ("1,000", "1001"),
            ("99999999999", "100000000000"),
            ("123456789012", "123456789013"),
            (r"2\pi", "6.28"),
            ("1", "2"),
        ],
    )
    def test_nearby_values_stay_different(self, left: str, right: str) -> None:
        """Exact values are never approximated, however close they look."""
        assert not _symbolically_equivalent(left, right)


class TestWordsAreStillWords:
    """Test that reading an answer as mathematics never overrides prose."""

    def test_a_sequence_is_not_an_arithmetic_operand(self) -> None:
        """Structured answers are compared as themselves, never added or powered."""
        assert not _answer_matches("(1,2)^2", "(1,4)")
        assert not _answer_matches("(1,2)-(1/2)", "1")
        assert _answer_matches("(1, 2)", "(1, 2)")

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [("Ja", "ja"), ("CO2", "co2"), ("T-shirt", "t-shirt"), (r"\boxed{Ja}", "JA")],
    )
    def test_case_folded_words_match(self, prediction: str, reference: str) -> None:
        """Words that also parse as names or subtractions match on text.

        A symbolic comparison would answer these -- `CO2` is a name and `T-shirt` is a
        difference of names -- and would get them wrong, so a value carrying names is
        referred back to text.
        """
        assert _answer_matches(prediction, reference)


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        (r"The answer is \boxed{x = 5}", "5", True),
        (r"The answer is \boxed{x=5}", "5", True),
        (r"The answer is x = 5", "5", True),
        (r"The answer is \boxed{x = 5}", "6", False),
        (r"The answer is \boxed{x = y}", "5", False),
    ],
)
def test_a_named_equation_scores_the_value_it_names(
    prediction: str, reference: str, expected: bool
) -> None:
    """An answer boxed as `x = 5` is the number 5, as Inspect AI also reads it."""
    assert _answer_matches(prediction, reference) is expected


def test_box_drawing_unicode_is_boxed() -> None:
    """Inspect-style box-drawing output is extracted as a boxed answer."""
    assert _answer_matches("\n│\n42\n│", "42")


def test_box_openers() -> None:
    """Supported boxed-answer spellings are extracted."""
    assert _answer_matches(r"\fbox{42}", "42")
    assert _answer_matches("boxed{42}", "42")
    assert _answer_matches(r"\beginboxed{42}", "42")


def test_boxed_expression_does_not_fall_back_to_later_number() -> None:
    """A boxed expression remains authoritative over a later number."""
    assert not _answer_matches(r"\boxed{3+3}, i.e. 9", "9")
    assert not _answer_matches(r"\boxed{2^3}, not 9", "9")
    assert not _answer_matches(r"\boxed{6/2}, i.e. 9", "9")


def test_candidate_precedence_and_markers() -> None:
    """The first candidate wins, and answer markers require a connector."""
    assert not _answer_matches(r"\boxed{7} ... the answer is 42", "42")
    assert _answer_matches(r"\boxed{42} junk 7", "42")
    assert _answer_matches("The answer is 42.", "42")
    assert not any(
        candidate.startswith("to the problem")
        for candidate in _answer_candidates("the answer to the problem")
    )


def test_metric_mean_and_empty_input() -> None:
    """The metric returns a mean and None when there are no paired items."""
    assert _score([r"\boxed{1}", "wrong"], ["1", "2"]) == 0.5
    assert _score([], []) is None


def _score(predictions: list[str], references: list[str]) -> float | None:
    """Call the metric with unused interface arguments set to None.

    Returns:
        The metric score.
    """
    return MathAccuracy(name="math_accuracy", pretty_name="Math Accuracy")(
        predictions=predictions,
        references=references,
        dataset=None,  # ty: ignore[invalid-argument-type]
        dataset_config=None,  # ty: ignore[invalid-argument-type]
        benchmark_config=None,  # ty: ignore[invalid-argument-type]
    )
