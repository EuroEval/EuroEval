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


class TestIdentities:
    """Test that an answer meets an equivalent form of itself."""

    def test_a_different_expression_does_not_match(self) -> None:
        """Comparing values must not drift into comparing anything that looks alike."""
        assert not _answer_matches("x + y", "y + z")
        assert not _answer_matches(r"\sqrt{x^2}", "x")
        assert not _answer_matches("(x + 1)^2", "x^2 + 2x + 1")

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [("x + y", "y + x"), (r"\boxed{x + y}", "y + x"), ("2*x", "x*2")],
    )
    def test_a_rearranged_expression_matches(
        self, prediction: str, reference: str
    ) -> None:
        """Order carries no meaning in a sum or a product."""
        assert _answer_matches(prediction, reference)


class TestKnownDivergences:
    """Test the answers given here that Inspect AI's ``math`` scorer does not give.

    Each is a choice, recorded so that it cannot change by accident.
    """

    def test_a_boxed_word_meets_the_bare_word(self) -> None:
        """A boxed answer saying the right thing is right, box and all."""
        assert _answer_matches(r"\boxed{\text{Yes}}", "Yes")

    def test_a_cancelled_ratio_meets_its_simplified_form(self) -> None:
        """SymPy cancels as it builds; Inspect AI builds unevaluated and abstains."""
        assert _answer_matches(r"\boxed{\frac{x^2 - 1}{x - 1}}", "x + 1")


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


class TestWords:
    """Test that words are scored as Inspect AI's ``math`` scorer scores them.

    An answer which is not prose is read as a name, and names are case-sensitive, so
    ``CO2`` does not meet ``co2`` and ``Ja`` does not meet ``ja``. An answer of two or
    more words is prose, and prose is compared case-insensitively. This follows
    Inspect AI rather than intuition, which would fold case everywhere: a benchmark
    whose targets differ from its predictions only in case scores low here, and scores
    just as low under Inspect AI.
    """

    def test_a_sequence_is_not_an_arithmetic_operand(self) -> None:
        """Structured answers are compared as themselves, never added or powered."""
        assert not _answer_matches("(1,2)^2", "(1,4)")
        assert not _answer_matches("(1,2)-(1/2)", "1")
        assert _answer_matches("(1, 2)", "(1, 2)")

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            ("Ja", "ja"),
            ("ja", "JA"),
            ("CO2", "co2"),
            ("T-shirt", "t-shirt"),
            (r"\boxed{Ja}", "JA"),
        ],
    )
    def test_a_single_word_is_a_name_and_names_are_case_sensitive(
        self, prediction: str, reference: str
    ) -> None:
        """One word reaches SymPy as a name, where capitalisation counts."""
        assert not _answer_matches(prediction, reference)

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [("Ja er svaret", "ja er svaret"), ("Svaret er 42", "42")],
    )
    def test_prose_is_compared_as_text(self, prediction: str, reference: str) -> None:
        """Several words are not a product of names, so they are compared as text."""
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
