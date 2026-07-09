"""Utility functions related to the multiple-choice classification task group."""

import typing as t

from ..string_utils import CHOICE_LETTERS
from ..exceptions import InvalidBenchmark
from .cloze import parse_bare_question_and_choices

if t.TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding



def prepare_examples(
    examples: "BatchEncoding", tokeniser: "PreTrainedTokenizer"
) -> "BatchEncoding":
    """Prepare the features.

    Args:
        examples:
            The examples to prepare.
        tokeniser:
            The tokeniser to use to prepare the examples.

    Returns:
        The prepared examples.
    """
    # `datasets.map` hands us a batch of documents at a time, so we iterate over every
    # document and concatenate the rows they produce. Each document is a multiple-choice
    # question, we keep each example as [{input_ids: [[1], [2]]}, {input_ids: [[3], [4]]}]
    # with one label per example
    all_texts: list[str] = []
    all_choices: list[str] = []
    all_labels: list[int] = []
    num_choices = None
    for doc, gold_letter in zip(examples["text"], examples["label"]):
        # Recover the bare question and the individual choice texts from the formatted
        # prompt.
        context_and_question, choices = parse_bare_question_and_choices(doc)
        len_choices = len(choices)
        assert len_choices > 0, "No choices found in the document."
        
        if num_choices is None:
            num_choices = len_choices
        elif len_choices != num_choices:
            raise InvalidBenchmark(
                f"Questions have differing choices counts {num_choices} vs"
                f" {len_choices}; a fixed count per dataset is required"
            )

        all_texts.extend([context_and_question] * len_choices)
        all_choices.extend(choices)
        all_labels.append(CHOICE_LETTERS.index(gold_letter.lower()))

    new_examples = tokeniser(
        text=all_texts,
        text_pair=all_choices,
        truncation=True,
    )
    # regroup the flat (question, choice) rows back into one nested row per question.
    new_examples = {k: [v[i : i + num_choices] for i in range(0,len(v),num_choices)] for k, v in new_examples.items()}
    new_examples["label"] = all_labels
    return new_examples
