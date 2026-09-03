"""Tests for the mathematical exact-match metric."""

from euroeval.metrics.math import (
    MathAccuracy,
    _answer_candidates,
    _answer_matches,
    _boxed_candidates,
    _equivalent,
)


class TestMathHelpers:
    """Test extraction and comparison behaviour."""

    def test_boxed_reference_and_unsupported_fraction(self) -> None:
        """Boxed references work, but LaTeX fractions are not parsed."""
        assert _equivalent("42", r"\boxed{42}")
        assert not _equivalent(r"\frac{1}{2}", "0.5")

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


def test_box_openers() -> None:
    """Supported boxed-answer spellings are extracted."""
    assert _answer_matches(r"\fbox{42}", "42")
    assert _answer_matches("boxed{42}", "42")
    assert _answer_matches(r"\beginboxed{42}", "42")


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
