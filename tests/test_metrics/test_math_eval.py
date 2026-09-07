"""Tests for the symbolic comparison of mathematical answers."""

import time

import pytest

from euroeval.metrics.math import _answer_matches
from euroeval.metrics.math_eval import is_symbolically_equivalent


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
        assert is_symbolically_equivalent(text, "1") is None

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
        assert is_symbolically_equivalent(text, "1") is False
        assert time.perf_counter() - start < 1.0

    @pytest.mark.parametrize(
        "text",
        ["9" * 5000, r"9^{9^{9}}", r"2^{100000}", "(" + "1," * 400 + "1)", "", "   "],
    )
    def test_pathological_input_is_refused_quickly(self, text: str) -> None:
        """Pathological candidates are unknown, and unknown within milliseconds."""
        start = time.perf_counter()
        assert is_symbolically_equivalent(text, "1") is None
        assert time.perf_counter() - start < 1.0


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
        assert is_symbolically_equivalent(left, right)
        assert _answer_matches(r"\boxed{%s}" % left, right)

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
        assert not is_symbolically_equivalent(left, right)
