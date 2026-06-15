"""Utilities for bits-per-character (BPC) evaluation of multiple-choice tasks."""

import collections.abc as c

from ..exceptions import InvalidBenchmark
from ..string_utils import CHOICE_LETTERS


def letter_to_choice_text(letter: str, raw_choices: c.Sequence[str]) -> str:
    """Return the full choice text corresponding to a letter label.

    For BPC scoring on multiple-choice tasks, we convert the label letter
    ("a", "b", ...) to the full answer text that the model should generate.

    Args:
        letter:
            A single lowercase letter ("a", "b", ...).
        raw_choices:
            The ordered list of raw choice strings for the example.

    Returns:
        The raw choice string at the index encoded by `letter`.

    Raises:
        InvalidBenchmark:
            If the letter does not correspond to a choice in `raw_choices`.
    """
    letter = letter.strip().lower()
    idx = CHOICE_LETTERS.find(letter)
    if idx == -1 or idx >= len(raw_choices):
        raise InvalidBenchmark(
            f"Could not map label letter {letter!r} to a choice; "
            f"available choices: {list(raw_choices)!r}."
        )
    return raw_choices[idx]
