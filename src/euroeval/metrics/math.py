r"""Exact-match scoring for mathematical answers.

The candidate ladder prefers the last boxed answer, then a connected answer
marker, delimited mathematics, the whole short text, the last line, and the
last number. The first candidate wins; opaque prose may fall back to later
numeric candidates. Numbers use exact equality after percent values are divided
by 100, while other values use normalised case-folded text equality.

Diverges from Inspect AI: LaTeX expressions are not parsed symbolically, so
``\\frac{1}{2}`` does not equal ``0.5`` and ``0.1 + 0.2`` is not evaluated.
"""

from __future__ import annotations

import collections.abc as c
import decimal
import re
import typing as t

from .base import Metric

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
_BOX = re.compile(
    r"(?:\\(?:beginboxed|boxed|fbox)|(?<![A-Za-z\\])(?:boxed|fbox|oxed))\s*\{"
)
_DELIMITERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))


class MathAccuracy(Metric):
    r"""Score answers with the Inspect-style candidate ladder.

    The first candidate wins. Only an unmatched opaque prose candidate may
    fall back to later candidates that are plain numbers. Numeric comparison is
    exact after percent values are divided by 100; all other comparison is
    normalised case-folded text equality.

    Diverges from Inspect AI: LaTeX is not parsed symbolically, so
    ``\\frac{1}{2}`` does not equal ``0.5`` and ``0.1 + 0.2`` is not evaluated.
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
    if _number_value(primary) is not None:
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
    return text.rstrip(" .,;:")


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


def _replace_unicode(text: str) -> str:
    """Replace common mathematical Unicode symbols with textual equivalents.

    Returns:
        Text with supported Unicode symbols replaced.
    """
    return (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("\u202f", " ")
        .replace("√", r"\sqrt")
        .replace("×", r"\cdot")
    )


def _equivalent(left: str, right: str) -> bool:
    """Compare exact plain numbers, otherwise normalised text.

    Plain numbers are parsed without symbolic mathematics, so every value is
    exact and ``==`` is intentional: Inspect AI's floating-point tolerance is
    deliberately absent. Percent values are scaled by dividing by 100.

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
    return _normalize_text(left) == _normalize_text(right)


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
