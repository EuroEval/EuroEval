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
        list of choice texts. If no sequential "a", "b", "c", ... options are found,
        ``choices`` is empty and ``bare_question`` is the original text unchanged.
    """
    lines = text.split("\n")
    candidates = [
        (idx, match.group(1).lower(), match.group(2).strip())
        for idx, line in enumerate(lines)
        if (match := _CHOICE_LINE_REGEX.match(line))
    ]
    alpha_candidates = [
        candidate
        for candidate in candidates
        if len(candidate[1]) == 1 and candidate[1].isalpha()
    ]

    # Option bodies may themselves contain numbered lists (including blank lines).
    # Treat only a sequential alphabetic run as the outer option boundaries; numeric
    # enumerators between those boundaries remain part of the option text.
    selected: list[tuple[int, str, str]] = []
    for start_idx, (_, marker, _) in enumerate(alpha_candidates):
        if marker != CHOICE_LETTERS[0]:
            continue
        run = [alpha_candidates[start_idx]]
        valid_run = True
        for candidate in alpha_candidates[start_idx + 1 :]:
            expected_idx = len(run)
            if (
                expected_idx >= len(CHOICE_LETTERS)
                or candidate[1] != CHOICE_LETTERS[expected_idx]
            ):
                valid_run = False
                break
            run.append(candidate)
        if valid_run and len(run) > len(selected):
            selected = run

    if len(selected) < 2:
        return text, []

    first_choice_idx = selected[0][0]
    choices = []
    for choice_idx, (_, _, first_line) in enumerate(selected):
        start = selected[choice_idx][0]
        end = (
            selected[choice_idx + 1][0]
            if choice_idx + 1 < len(selected)
            else len(lines)
        )
        body_lines = [first_line, *lines[start + 1 : end]]
        choices.append("\n".join(body_lines).strip())

    # Everything before the first option is the question, minus a trailing choices-label
    # line (e.g. "Choices:"), mirroring how the prompt was assembled as
    # ``<question>\n<choices label>:\n<options>``.
    head_lines = lines[:first_choice_idx]
    while head_lines and not head_lines[-1].strip():
        head_lines.pop()
    if head_lines and head_lines[-1].rstrip().endswith(":"):
        head_lines.pop()
    while head_lines and not head_lines[-1].strip():
        head_lines.pop()
    bare_question = "\n".join(head_lines).strip()
    return bare_question, choices
