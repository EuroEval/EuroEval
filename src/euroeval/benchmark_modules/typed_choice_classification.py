"""Shared typed-choice classification behaviour for remote adapters."""

import abc
import collections.abc as c
import math
import typing as t

from ..data_models import DatasetConfig, GenerativeModelOutput
from ..enums import TaskGroup
from ..exceptions import InvalidBenchmark
from ..task_group_utils.cloze import parse_bare_question_and_choices
from .zero_shot_classifier import ZeroShotClassifierModel


class TypedChoiceClassificationModel(ZeroShotClassifierModel, abc.ABC):
    """Share batch classification and dataset updates for typed-choice adapters."""

    @abc.abstractmethod
    def _classify_one(
        self, state: str, criteria: dict[str, str | None]
    ) -> dict[str, float]:
        """Classify one state using the adapter's typed endpoint.

        Args:
            state:
                The text state to classify.
            criteria:
                Candidate labels and optional descriptions.

        Returns:
            A probability for each candidate label.
        """
        raise NotImplementedError

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Classify inputs and expose probabilities as log-score output.

        Args:
            inputs:
                A batch containing the prepared text values.

        Returns:
            Predicted labels and log-probability scores.
        """
        return classify_typed_choices(
            inputs=inputs,
            labels=[
                self.dataset_config.prompt_label_mapping[label]
                for label in self.dataset_config.id2label.values()
            ],
            multiple_choice=self.dataset_config.task.task_group
            == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
            classify=self._classify_one,
        )

    def update_dataset_config(self, dataset_config: DatasetConfig) -> t.Self:
        """Update the active dataset and its classification instructions.

        Args:
            dataset_config:
                The new dataset configuration.

        Returns:
            This model instance.
        """
        self.dataset_config = dataset_config
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()
        return self


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
