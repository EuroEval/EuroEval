"""Utilities for Cloze Formulation (CF) evaluation of multiple-choice tasks."""

import collections.abc as c
import re
import typing as t

import numpy as np

from ..enums import CFNormalization
from ..exceptions import InvalidBenchmark

if t.TYPE_CHECKING:
    from ..data_models import DatasetConfig, GenerativeModelOutput, ModelConfig


_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_CHOICE_LINE_RE = re.compile(r"^[a-z0-9]+\. ")


def build_cf_prompt(
    bare_input: str,
    few_shot_rendered: c.Sequence[str],
    prompt_template: str,
    prompt_prefix: str,
) -> str:
    r"""Assemble the final CF prompt that the model will score candidates against.

    The prompt ends with whatever trailing marker the template renders for an
    empty ``{label}`` (typically ``"Antwoord: "`` or ``"Answer: "``), so that
    candidate texts can be appended directly for scoring.

    Args:
        bare_input:
            The bare question text (no enumerated choices).
        few_shot_rendered:
            Pre-rendered few-shot sections, each of the form
            ``"Vraag: <q>\nAntwoord: <full-answer-text>"``.
        prompt_template:
            The dataset's few-shot prompt template with ``{text}`` and ``{label}``
            placeholders.
        prompt_prefix:
            The dataset's prompt prefix (may be empty). ``{labels_str}``, if
            present, is substituted with an empty string because CF does not
            enumerate the label letters.

    Returns:
        The full prompt string.
    """
    tail = prompt_template.format(text=bare_input.replace("\n", " ").strip(), label="")
    rendered_prefix = prompt_prefix.format(labels_str="") if prompt_prefix else ""
    if rendered_prefix:
        rendered_prefix += "\n\n"
    few_shot_block = "\n\n".join(few_shot_rendered)
    if few_shot_block:
        few_shot_block += "\n\n"
    return rendered_prefix + few_shot_block + tail


def render_cf_few_shot(bare_input: str, answer_text: str, prompt_template: str) -> str:
    """Render a single few-shot example for CF with the full answer text.

    Args:
        bare_input:
            The bare question of the few-shot example.
        answer_text:
            The full text of the correct answer choice.
        prompt_template:
            The dataset's few-shot prompt template.

    Returns:
        The rendered few-shot section.
    """
    return prompt_template.format(
        text=bare_input.replace("\n", " ").strip(),
        label=answer_text.replace("\n", " ").strip(),
    )


def parse_mcq_text(text: str) -> tuple[str, list[str]]:
    """Recover the bare question and choice texts from a formatted MCQ prompt.

    EuroEval-prepared MCQ datasets store ``text`` as a single string of the form::

        <passage/question>
        Choices:
        a. <option a>
        b. <option b>
        ...

    Some datasets instead embed the raw choices in separate columns that the
    preprocessing step merges into ``text``. This helper reverses that merge so
    CF evaluation can score each raw choice on its own.

    Args:
        text:
            The formatted MCQ text as stored in the dataset's ``"text"`` column.

    Returns:
        A pair ``(bare_input, raw_choices)`` where ``bare_input`` is everything
        before the enumerated choices (typically the passage and question) and
        ``raw_choices`` are the option texts with the letter prefix stripped.

    Raises:
        InvalidBenchmark:
            If no enumerated choices can be located in ``text``.
    """
    sections = text.split("\n")
    candidate_choice_idxs = [
        idx
        for idx, section in enumerate(sections)
        if _CHOICE_LINE_RE.match(section) is not None
    ]

    # Take the final contiguous block of at least two choice lines, to skip
    # question lines that coincidentally start with "a. " etc.
    choice_idxs: list[int] = list()
    for idx in reversed(candidate_choice_idxs):
        if len(choice_idxs) < 2 or (
            len(choice_idxs) >= 2 and idx == choice_idxs[-1] - 1
        ):
            choice_idxs.append(idx)
    choice_idxs = list(reversed(choice_idxs))

    if len(choice_idxs) < 2:
        raise InvalidBenchmark(
            "CF evaluation needs at least two enumerated answer choices in the "
            "MCQ prompt, but none were found. The prompt may not be formatted as "
            f"expected:\n{text!r}"
        )

    choices = [sections[idx] for idx in choice_idxs]
    raw_choices = [_CHOICE_LINE_RE.sub("", choice).strip() for choice in choices]

    # Drop the 'Choices:' header line immediately above the first choice, if any.
    first_choice_idx = choice_idxs[0]
    if first_choice_idx >= 2 and sections[first_choice_idx - 1].strip().endswith(":"):
        bare_end = first_choice_idx - 1
    else:
        bare_end = first_choice_idx
    bare_input = "\n".join(sections[:bare_end]).strip()
    return bare_input, raw_choices


def normalize_cf_score(
    token_logprobs: c.Sequence[float], answer_text: str, method: CFNormalization
) -> float:
    """Compute the normalized CF score for a single answer candidate.

    Args:
        token_logprobs:
            Per-token logprobs of the candidate continuation.
        answer_text:
            The raw candidate text, used for character-length normalization.
        method:
            The length-normalization method to apply.

    Returns:
        The normalized score. Higher is better.

    Raises:
        InvalidBenchmark:
            If the normalization method is not supported.
    """
    total = float(sum(token_logprobs))
    match method:
        case CFNormalization.NONE:
            return total
        case CFNormalization.TOKEN:
            return total / max(1, len(token_logprobs))
        case CFNormalization.CHARACTER:
            return total / max(1, len(answer_text))
        case _:
            raise InvalidBenchmark(f"Unsupported CF normalization: {method!r}.")


def extract_labels_from_cf(
    input_batch: dict[str, list],
    model_output: "GenerativeModelOutput",
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
    first_label_token_mapping: dict[str, str] | bool,
) -> c.Sequence[str]:
    """Pick the highest-scoring choice letter for each sample.

    The signature mirrors
    :func:`euroeval.task_group_utils.sequence_classification.extract_labels_from_generation`
    so that the backend's :pyattr:`extract_labels_from_generation` property can
    drop this in for MCQ samples evaluated with CF.

    Args:
        input_batch:
            The input batch (unused, kept for signature parity).
        model_output:
            The generative model output with ``cf_scores`` populated (shape
            ``(batch_size, num_choices)``).
        dataset_config:
            The dataset configuration (unused, kept for signature parity).
        model_config:
            The model configuration (unused, kept for signature parity).
        first_label_token_mapping:
            Unused, kept for signature parity with the MCF extractor.

    Returns:
        The predicted letter label (``"a"``, ``"b"``, ...) for each sample.

    Raises:
        InvalidBenchmark:
            If ``cf_scores`` is missing from ``model_output``.
    """
    del input_batch, dataset_config, model_config, first_label_token_mapping
    if model_output.cf_scores is None:
        raise InvalidBenchmark(
            "CF evaluation expected `cf_scores` on the model output, but none was "
            "produced. This is likely a bug."
        )
    return [_LETTERS[int(np.argmax(row))] for row in model_output.cf_scores]


def letter_to_choice_text(letter: str, raw_choices: c.Sequence[str]) -> str:
    """Return the full choice text corresponding to a letter label.

    Args:
        letter:
            A single lowercase letter (``"a"``, ``"b"``, ...).
        raw_choices:
            The ordered list of raw choice strings for the example.

    Returns:
        The raw choice string at the index encoded by ``letter``.

    Raises:
        InvalidBenchmark:
            If the letter does not correspond to a choice in ``raw_choices``.
    """
    letter = letter.strip().lower()
    idx = _LETTERS.find(letter)
    if idx == -1 or idx >= len(raw_choices):
        raise InvalidBenchmark(
            f"Could not map label letter {letter!r} to a choice; "
            f"available choices: {list(raw_choices)!r}."
        )
    return raw_choices[idx]
