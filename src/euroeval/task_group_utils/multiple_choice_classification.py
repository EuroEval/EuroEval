"""Utility functions related to the multiple-choice classification task group."""

import typing as t

from ..exceptions import InvalidBenchmark
from ..string_utils import CHOICE_LETTERS
from .cloze import parse_bare_question_and_choices

if t.TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding


def prepare_examples(
    examples: "BatchEncoding", tokeniser: "PreTrainedTokenizer", num_choices: int = 0
) -> dict:
    """Tokenise choices while preserving each question's actual choice count.

    Args:
        examples: The input batch containing text and gold labels.
        tokeniser: The tokenizer used to encode question-choice pairs.
        num_choices: Retained for API compatibility; choice counts are inferred per row.

    Returns:
        Tokenized features grouped by question, with one label per question.

    Raises:
        InvalidBenchmark: If a question has no choices, too many choices, or an
            invalid gold label.
    """
    del num_choices
    all_texts: list[str] = []
    all_choices: list[str] = []
    counts: list[int] = []
    all_labels: list[int] = []
    for doc, gold_letter in zip(examples["text"], examples["label"]):
        context_and_question, choices = parse_bare_question_and_choices(doc)
        count = len(choices)
        if count == 0:
            raise InvalidBenchmark("No choices found in the document.")
        if count > len(CHOICE_LETTERS):
            raise InvalidBenchmark(
                f"Multiple-choice example has {count} choices, exceeding the maximum "
                f"of {len(CHOICE_LETTERS)} supported choices."
            )

        gold = gold_letter.lower()
        if gold not in CHOICE_LETTERS[:count]:
            raise InvalidBenchmark(f"Gold label {gold_letter!r} is not a valid choice.")

        all_texts.extend([context_and_question] * count)
        all_choices.extend(choices)
        counts.append(count)
        all_labels.append(CHOICE_LETTERS.index(gold))

    tokenized = tokeniser(text=all_texts, text_pair=all_choices, truncation=True)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    new_examples = {
        key: [values[start:end] for start, end in zip(offsets, offsets[1:])]
        for key, values in tokenized.items()
    }
    new_examples["label"] = all_labels
    return new_examples
