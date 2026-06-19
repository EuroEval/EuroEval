"""Utilities for bits-per-character (BPC) evaluation of multiple-choice tasks."""

import collections.abc as c
import re

from ..exceptions import InvalidBenchmark
from ..string_utils import CHOICE_LETTERS

_CHOICE_LINE_REGEX = re.compile(r"^\s*([a-zA-Z0-9]+)\.\s+(.+?)\s*$")


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
    candidate_idxs = [
        idx for idx, line in enumerate(lines) if _CHOICE_LINE_REGEX.match(line)
    ]
    if not candidate_idxs:
        return text, []

    # Only the final contiguous block of enumerated lines counts as choices: the
    # question itself can contain lines that start with e.g. "1." or "a.", so we walk
    # backwards from the end and stop at the first gap.
    block_idxs: list[int] = []
    for idx in reversed(candidate_idxs):
        if not block_idxs or idx == block_idxs[-1] - 1:
            block_idxs.append(idx)
        else:
            break
    block_idxs.reverse()
    first_choice_idx = block_idxs[0]

    choices: list[str] = []
    for idx in block_idxs:
        match = _CHOICE_LINE_REGEX.match(lines[idx])
        assert match is not None  # guaranteed by candidate_idxs construction
        choices.append(match.group(2).strip())

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
