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
    """Prepare the features.

    Args:
        examples:
            The examples to prepare.
        tokeniser:
            The tokeniser to use to prepare the examples.
        num_choices:
            The number of choices each question is expected to have, taken from the
            dataset configuration. If 0 (datasets with dynamic labels that do not
            declare a fixed count), the count is inferred from the first document.

    Returns:
        The prepared examples.

    Raises:
        InvalidBenchmark:
            If a document has no choices, a number of choices that differs from the
            count required by the dataset, or a gold label that is not one of the
            available choices.
    """
    # `datasets.map` hands us a batch of documents at a time, so we iterate over every
    # document and concatenate the rows they produce. Each document is a multiple-choice
    # question, with one label per example. We keep each example as
    # [{input_ids: [[1], [2]]}, {input_ids: [[3], [4]]}].
    all_texts: list[str] = []
    all_choices: list[str] = []
    all_labels: list[int] = []
    expected_num_choices = num_choices
    for doc, gold_letter in zip(examples["text"], examples["label"]):
        # Recover the bare question and the individual choice texts from the formatted
        # prompt.
        context_and_question, choices = parse_bare_question_and_choices(doc)
        len_choices = len(choices)
        if len_choices == 0:
            raise InvalidBenchmark("No choices found in the document.")

        if expected_num_choices == 0:
            expected_num_choices = len_choices
        elif len_choices != expected_num_choices:
            raise InvalidBenchmark(
                f"A question has {len_choices} choices but the dataset requires "
                f"{expected_num_choices}; a fixed count per dataset is required."
            )

        gold = gold_letter.lower()
        if gold not in CHOICE_LETTERS[:expected_num_choices]:
            raise InvalidBenchmark(f"Gold label {gold_letter!r} is not a valid choice.")

        all_texts.extend([context_and_question] * len_choices)
        all_choices.extend(choices)
        all_labels.append(CHOICE_LETTERS.index(gold))

    new_examples = tokeniser(text=all_texts, text_pair=all_choices, truncation=True)
    # regroup the flat (question, choice) rows back into one nested row per question.
    if expected_num_choices > 0:
        new_examples = {
            k: [
                v[i : i + expected_num_choices]
                for i in range(0, len(v), expected_num_choices)
            ]
            for k, v in new_examples.items()
        }
    new_examples["label"] = all_labels
    return new_examples
