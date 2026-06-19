"""Utilities for bits-per-character (BPC) evaluation of multiple-choice tasks."""

import collections.abc as c
import re

from ..exceptions import InvalidBenchmark
from ..string_utils import CHOICE_LETTERS

_CHOICE_LINE_REGEX = re.compile(r"^\s*([a-zA-Z])\.\s+(.+?)\s*$")


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


def parse_bare_question_and_choices(text: str) -> tuple[str, list[str]]:
    """Recover the bare question and the choice texts from a formatted MCQ prompt.

    Multiple-choice datasets currently store the question together with its enumerated
    answer options in a single ``text`` field, formatted as::

        <question>
        <choices label>:
        a. <choice 0>
        b. <choice 1>
        ...

    Cloze (BPC) scoring needs the bare question and the individual choice texts
    separately, so this splits the formatted prompt back into those parts.

    Args:
        text:
            The formatted multiple-choice prompt.

    Returns:
        A pair ``(bare_question, choices)`` where ``bare_question`` is the prompt with
        the choices label and enumerated options removed, and ``choices`` is the ordered
        list of choice texts. If no enumerated options are found, ``choices`` is empty
        and ``bare_question`` is the original text unchanged.
    """
    lines = text.split("\n")
    first_choice_idx: int | None = None
    choices: list[str] = []
    for idx, line in enumerate(lines):
        match = _CHOICE_LINE_REGEX.match(line)
        if match is None:
            # Only the contiguous trailing block of enumerated lines counts as choices;
            # stop once a non-choice line appears after the block has started.
            if first_choice_idx is not None:
                break
            continue
        if first_choice_idx is None:
            first_choice_idx = idx
        choices.append(match.group(2).strip())

    if first_choice_idx is None:
        return text, []

    # Everything before the first option is the question, minus a trailing choices-label
    # line (e.g. "Choices:"), mirroring how the prompt was assembled as
    # ``<question>\n<choices label>:\n<options>``.
    head_lines = lines[:first_choice_idx]
    while head_lines and (
        head_lines[-1].strip() == "" or head_lines[-1].rstrip().endswith(":")
    ):
        head_lines.pop()
    bare_question = "\n".join(head_lines).strip()
    return bare_question, choices
