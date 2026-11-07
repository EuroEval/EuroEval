"""Shared typed-choice classification behaviour for remote adapters."""

import collections.abc as c
import math

from ..data_models import GenerativeModelOutput
from ..exceptions import InvalidBenchmark
from ..task_group_utils.cloze import parse_bare_question_and_choices


def classify_typed_choices(
    inputs: dict,
    labels: list[str],
    multiple_choice: bool,
    classify: c.Callable[[str, dict[str, str | None]], dict[str, float]],
) -> GenerativeModelOutput:
    """Classify text batches and produce EuroEval's ranked log-score output.

    Args:
        inputs:
            A batch containing a ``text`` sequence.
        labels:
            Candidate labels in dataset order.
        multiple_choice:
            Whether inputs may contain parsed choice options.
        classify:
            Adapter callback returning probabilities for the supplied criteria.

    Returns:
        Predicted labels and per-example log-probability scores.

    Raises:
        InvalidBenchmark:
            If the input does not contain text.
    """
    if "text" not in inputs:
        raise InvalidBenchmark("The inputs must contain a 'text' key.")
    generic_criteria = {label: None for label in labels}
    sequences: list[str] = []
    scores: list[list[list[tuple[str, float]]]] = []
    for text in inputs["text"]:
        state = text
        criteria: dict[str, str | None] = generic_criteria
        if multiple_choice:
            question, options = parse_bare_question_and_choices(text=text)
            if len(options) == len(labels) and len(set(options)) == len(options):
                state = question
                criteria = dict(zip(labels, options, strict=True))
        probabilities = classify(state, criteria)
        label_scores = sorted(
            ((label, math.log(max(probabilities[label], 1e-12))) for label in labels),
            key=lambda item: item[1],
            reverse=True,
        )
        sequences.append(label_scores[0][0])
        scores.append([label_scores])
    return GenerativeModelOutput(sequences=sequences, scores=scores)
