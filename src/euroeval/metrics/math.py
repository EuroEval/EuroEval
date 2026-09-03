"""Exact-match scoring for mathematical answers.

This intentionally implements numeric and normalised-text equality only; it does
not depend on SymPy, latex2sympy2, or an ANTLR parser.
"""

from __future__ import annotations

import collections.abc as c
import re
import typing as t

from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d[\d,\s]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?%?")
_MARKER = re.compile(r"(?:final\s+answer|answer|result)\s*[:=]?", re.IGNORECASE)
_BOX = re.compile(r"(?:\\(?:boxed|fbox)|(?<![A-Za-z])boxed)\s*\{")
_DELIMITERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))


class MathAccuracy(Metric):
    """Score mathematical answers using numeric or normalised-text equality."""

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the mean per-item mathematical exact-match score.

        Returns:
            The mean score, or None when there are no paired inputs.
        """
        scores = [
            float(_answer_matches(prediction=str(prediction), reference=str(reference)))
            for prediction, reference in zip(predictions, references)
        ]
        return sum(scores) / len(scores) if scores else None


def _answer_matches(prediction: str, reference: str) -> bool:
    """Return whether an answer candidate matches the reference."""
    reference_candidates = _reference_candidates(reference)
    prediction_candidates = _answer_candidates(prediction)
    for candidate in prediction_candidates:
        if any(_equivalent(candidate, target) for target in reference_candidates):
            return True
    numeric_predictions = [
        candidate
        for candidate in prediction_candidates
        if _number_value(candidate) is not None
    ]
    return any(
        _equivalent(candidate, target)
        for candidate in numeric_predictions
        for target in reference_candidates
    )


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
    if len(text) <= 120:
        _append_candidate(candidates, text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    _append_candidate(candidates, lines[-1] if lines else None)
    numbers = _NUMBER.findall(text)
    _append_candidate(candidates, numbers[-1] if numbers else None)
    return candidates


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


def _append_candidate(candidates: list[str], candidate: str | None) -> None:
    """Add a non-empty, normalised candidate once."""
    if candidate:
        candidate = _strip_delimiters(candidate)
        if candidate and candidate not in candidates:
            candidates.append(candidate)


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


def _number_value(text: str) -> tuple[float, bool] | None:
    """Parse a plain number and return its value and whether it is a percent.

    Returns:
        A numeric value and percent flag, or None for non-numeric text.
    """
    normalised = _normalize_text(text)
    percent = normalised.endswith("%") or normalised.endswith(" percent")
    if percent:
        normalised = re.sub(r"(?:%|\s+percent)$", "", normalised).strip()
    normalised = normalised.replace(r"\,", "").replace(",", "")
    normalised = re.sub(r"(?<=\d)\s+(?=\d)", "", normalised)
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", normalised):
        return None
    try:
        return float(normalised), percent
    except ValueError:
        return None


def _equivalent(left: str, right: str) -> bool:
    """Compare numbers with Inspect AI's tolerance, otherwise normalised text.

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
        left_values = (
            (left_number[0] / 100, left_number[0])
            if left_number[1]
            else (left_number[0],)
        )
        right_values = (
            (right_number[0] / 100, right_number[0])
            if right_number[1]
            else (right_number[0],)
        )
        return any(
            abs(a - b) < 1e-10 or abs(a - b) / max(abs(a), abs(b), 1e-10) < 1e-10
            for a in left_values
            for b in right_values
        )
    return _normalize_text(left) == _normalize_text(right)


def _unique(values: list[str]) -> list[str]:
    """Return values with duplicates removed while preserving order."""
    return list(dict.fromkeys(value for value in values if value))


math_accuracy_metric = MathAccuracy(name="math_accuracy", pretty_name="Math Accuracy")
