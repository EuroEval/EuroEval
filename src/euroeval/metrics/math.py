r"""Scoring for mathematical answers.

The candidate ladder prefers the last boxed answer, then a connected answer
marker, delimited mathematics, the whole short text, the last line, and the
last number. The first candidate wins; opaque prose may fall back to later
numeric candidates. Plain numbers use exact equality after percent values are
divided by 100, expressions are compared as values by ``math_eval``, and
anything left over uses normalised case-folded text equality.

Diverges from Inspect AI: LaTeX is not parsed by a grammar, only rewritten, so
answers built from structure this rewrite does not cover -- matrices, integrals,
piecewise braces -- are compared as text.
"""

from __future__ import annotations

import collections.abc as c
import decimal
import re
import typing as t

from .base import Metric
from .math_eval import is_symbolically_equivalent

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

_NUMBER = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?\s*%?"
)
_PERCENT_SUFFIX = re.compile(
    r"\s*(?:\\?%|\\text\s*\{\s*(?:percent(?:age)?|pct)\s*\}|"
    r"\s+(?:percent(?:age)?|pct))\s*$",
    re.IGNORECASE,
)
_MARKER = re.compile(
    r"(?:final\s+answer|answer|result)\s*(?:is\b|[:=])\s*", re.IGNORECASE
)
_ASSIGNMENT = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<value>.+)$", re.IGNORECASE
)
_BOX = re.compile(
    r"(?:\\(?:beginboxed|boxed|fbox)|(?<![A-Za-z\\])(?:boxed|fbox|oxed))\s*\{"
)
_DELIMITERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
_WORD = re.compile(r"[A-Za-z]{2,}")


class MathAccuracy(Metric):
    r"""Score answers with the Inspect-style candidate ladder.

    The first candidate wins. Only an unmatched opaque prose candidate may
    fall back to later candidates that are plain numbers. Plain numbers are
    compared exactly after percent values are divided by 100, expressions are
    compared as values by ``math_eval``, and anything else uses normalised
    case-folded text equality.

    Diverges from Inspect AI: LaTeX is rewritten rather than parsed by a
    grammar, so structure beyond fractions, roots, powers and products --
    matrices, integrals, piecewise braces -- is compared as text.
    """

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the mean per-item mathematical exact-match score.

        Args:
            predictions: Model answers whose first matching candidate wins.
            references: Expected answers.
            dataset: Unused benchmark dataset.
            dataset_config: Unused dataset configuration.
            benchmark_config: Unused benchmark configuration.

        Returns:
            The mean score, or None when there are no paired inputs.
        """
        scores = [
            float(_answer_matches(prediction=str(prediction), reference=str(reference)))
            for prediction, reference in zip(predictions, references)
        ]
        return sum(scores) / len(scores) if scores else None


def _answer_matches(prediction: str, reference: str) -> bool:
    """Compare the first candidate, falling back only from opaque prose.

    Args:
        prediction: Model completion.
        reference: Expected answer.

    Returns:
        Whether the winning candidate matches a reference candidate.
    """
    reference_candidates = _reference_candidates(reference)
    prediction_candidates = _answer_candidates(prediction)
    if not prediction_candidates:
        return False
    primary = prediction_candidates[0]
    if any(_equivalent(primary, target) for target in reference_candidates):
        return True
    if _number_value(primary) is not None or not _looks_like_prose(primary):
        return False
    return any(
        any(_equivalent(candidate, target) for target in reference_candidates)
        for candidate in prediction_candidates[1:]
        if _number_value(candidate) is not None
    )


def _answer_candidates(text: str) -> list[str]:
    """Build the Inspect-style answer candidate ladder, without expression parsing.

    Returns:
        Candidate answer strings in preference order.
    """
    text = _replace_unicode(text)
    candidates: list[str] = []
    boxes = _boxed_candidates(text)
    if boxes:
        _append_candidate(candidates, boxes[-1])
    markers = list(_MARKER.finditer(text))
    if markers:
        suffix = text[markers[-1].end() :].splitlines()
        _append_candidate(candidates, suffix[0] if suffix else None)
    _append_candidate(candidates, _last_delimited_math(text))
    if len(text) <= 4096:
        _append_candidate(candidates, text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    _append_candidate(candidates, lines[-1] if lines else None)
    numbers = _NUMBER.findall(text)
    _append_candidate(candidates, numbers[-1] if numbers else None)
    return candidates


def _append_candidate(candidates: list[str], candidate: str | None) -> None:
    """Add a non-empty, normalised candidate once."""
    if candidate:
        candidate = _strip_delimiters(candidate)
        if candidate and candidate not in candidates:
            candidates.append(candidate)


def _strip_delimiters(text: str) -> str:
    """Remove display, Markdown, TeX spacing, and punctuation wrappers.

    Returns:
        The unwrapped text.
    """
    text = text.strip()
    while text.startswith("**") and text.endswith("**") and len(text) >= 4:
        text = text[2:-2].strip()
    for opening, closing in _DELIMITERS:
        if (
            text.startswith(opening)
            and text.endswith(closing)
            and len(text) > len(opening) + len(closing)
        ):
            text = text[len(opening) : -len(closing)].strip()
            break
    text = re.sub(r"^(?:\\[,;:]|\\quad|\\qquad|\\;|\\,)\s*", "", text)
    text = re.sub(r"(?:\\[,;:]|\\quad|\\qquad|\\;|\\,)\s*$", "", text)
    text = text.rstrip(" .,;:")
    # A boxed equation naming the answer, such as `x = 5`, is the value it names, as
    # Inspect AI's symbolic comparison also reduces it to that
    assignment = _ASSIGNMENT.match(text)
    if assignment is not None and _number_value(assignment.group("value")) is not None:
        text = assignment.group("value").strip()
    return text


def _number_value(text: str) -> tuple[decimal.Decimal, bool] | None:
    """Parse a plain number and return its exact value and percent status.

    Returns:
        An exact numeric value and percent flag, or None for non-numeric text.
    """
    normalised = _normalize_text(text)
    percent_match = _PERCENT_SUFFIX.search(normalised)
    percent = percent_match is not None
    if percent_match is not None:
        normalised = normalised[: percent_match.start()].strip()
    normalised = normalised.replace(r"\,", ",")
    if "," in normalised and not re.fullmatch(
        r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?", normalised
    ):
        return None
    normalised = normalised.replace(",", "")
    if not re.fullmatch(
        r"[-+]?(?:(?:\d{1,3}(?:\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        normalised,
    ):
        return None
    try:
        return decimal.Decimal(normalised), percent
    except decimal.InvalidOperation:
        return None


def _normalize_text(text: str) -> str:
    """Apply the shared normalisation used for both predictions and references.

    Returns:
        The case-folded normalised text.
    """
    text = _replace_unicode(_strip_delimiters(text))
    text = re.sub(r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"^[£€$]\s*", "", text)
    text = text.replace(r"\ ", " ").replace(r"\%", "%")
    text = re.sub(r"\s+", " ", text).strip(" .,;:")
    return text.casefold()


def _replace_unicode(text: str) -> str:
    """Replace mathematical Unicode symbols using Inspect AI's table.

    Returns:
        Text with mathematical Unicode symbols replaced.
    """
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    replacements = {
        "\u23a7": r"\boxed{",
        "\u23ab": "}",
        "\n\u2502": r"\boxed{",
        "\u2502": "}",
        "\n\u2503": r"\boxed{",
        "\u2503": "}",
        "\n\uf8f0": r"\boxed{",
        "\uf8fb": "}",
        "√": r"\sqrt",
        "×": r"\cdot",
        "÷": "/",
        "\u202f": " ",
        "−": "-",
        "–": "-",
        "π": r"\pi",
        "°": r"^\circ",
        "∞": r"\infty",
        "≤": r"\le",
        "≥": r"\ge",
        "≠": r"\ne",
        "∪": r"\cup",
        "∩": r"\cap",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    # A line-oriented box has an opening glyph on the first line and a closing
    # glyph on the last; the table's overlapping newline and single-glyph forms
    # can otherwise turn the latter into a second opener.
    text = re.sub(r"\\boxed\{([^{}]*)\\boxed\{", r"\\boxed{\1}", text)
    return text


def _boxed_candidates(text: str) -> list[str]:
    """Return brace-balanced contents of boxes, in source order."""
    matches: list[str] = []
    position = 0
    while match := _BOX.search(text, position):
        content = _balanced_content(text, match.end() - 1)
        if content is not None:
            matches.append(content[0])
            position = content[1]
        else:
            position = match.end()
    return matches


def _balanced_content(text: str, opening: int) -> tuple[str, int] | None:
    """Return content and end position for a balanced opening brace."""
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None


def _last_delimited_math(text: str) -> str | None:
    """Return the delimited mathematical span ending furthest to the right."""
    spans: list[tuple[int, str]] = []
    for opening, closing in _DELIMITERS:
        end = text.rfind(closing)
        start = text.rfind(opening, 0, end)
        if start >= 0 and end > start:
            spans.append((end, text[start + len(opening) : end]))
    return max(spans, default=(0, None))[1]


def _equivalent(left: str, right: str) -> bool:
    r"""Compare plain numbers exactly, then values, then normalised text.

    Plain numbers are compared without any mathematics, so every value is exact and
    ``==`` is intentional. Values that look like mathematics are compared as values by
    :mod:`euroeval.metrics.math_eval`, which is where ``\\frac{1}{2}`` meets ``0.5``; a
    string it cannot translate is compared as text instead.

    Returns:
        Whether the values are equivalent.
    """
    left_boxes = _boxed_candidates(left)
    right_boxes = _boxed_candidates(right)
    if left_boxes:
        left = left_boxes[-1]
    if right_boxes:
        right = right_boxes[-1]
    left_number = _number_value(left)
    right_number = _number_value(right)
    if left_number is not None and right_number is not None:
        left_value = left_number[0] / 100 if left_number[1] else left_number[0]
        right_value = right_number[0] / 100 if right_number[1] else right_number[0]
        return left_value == right_value
    if _has_math_syntax(left) or _has_math_syntax(right):
        symbolic = is_symbolically_equivalent(left, right)
        if symbolic is not None:
            return symbolic
    return _normalize_text(left) == _normalize_text(right)


def _has_math_syntax(text: str) -> bool:
    """Return whether a candidate is worth evaluating as mathematics."""
    return bool(re.search(r"\\[A-Za-z]|\d|[-+*/^=]", text))


def _looks_like_prose(candidate: str) -> bool:
    """Return whether a candidate is opaque prose rather than an expression."""
    if "\\" in candidate or any(char in candidate for char in "=+-*/^<>[]{}()"):
        return False
    return len(_WORD.findall(candidate)) >= 2


def _reference_candidates(text: str) -> list[str]:
    """Extract the boxed reference followed by its complete text.

    Returns:
        Candidate reference strings in preference order.
    """
    text = _replace_unicode(text)
    boxes = _boxed_candidates(text)
    values = [_strip_delimiters(boxes[-1])] if boxes else []
    values.append(text)
    return _unique(values)


def _unique(values: list[str]) -> list[str]:
    """Return values with duplicates removed while preserving order."""
    return list(dict.fromkeys(value for value in values if value))


math_accuracy_metric = MathAccuracy(name="math_accuracy", pretty_name="Math Accuracy")
