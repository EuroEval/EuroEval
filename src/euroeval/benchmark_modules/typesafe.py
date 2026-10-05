"""Direct client for Typesafe's hosted System One classifier."""

import collections.abc as c
import math
import os
import typing as t
from functools import cached_property

import requests

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
)
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import InvalidBenchmark, InvalidModel, NeedsAdditionalArgument
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..task_group_utils.cloze import parse_bare_question_and_choices
from ..types import ExtractLabelsFunction
from .base import BenchmarkModule, _extract_labels_from_generation_helper
from .zero_shot_classifier import ZeroShotClassifierModel

_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
_MODEL_ID = "jev-latest"
_LOGPROB_FLOOR = 1e-12

if t.TYPE_CHECKING:
    from transformers.trainer import Trainer


class TypesafeSystemOneModel(ZeroShotClassifierModel):
    """Hosted zero-shot classifier using Typesafe System One."""

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
    ) -> None:
        """Initialise the hosted classifier without loading local model weights.

        Raises:
            NeedsAdditionalArgument:
                If TYPESAFE_API_KEY is not set.
            InvalidBenchmark:
                If the dataset task is unsupported.
        """
        self.api_key = os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise NeedsAdditionalArgument(
                cli_argument="TYPESAFE_API_KEY environment variable",
                script_argument=(
                    'os.environ["TYPESAFE_API_KEY"] = "<your-typesafe-api-key>"'
                ),
                run_with_cli=benchmark_config.run_with_cli,
            )
        if dataset_config.task.task_group not in {
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        }:
            raise InvalidBenchmark(
                "Typesafe System One supports sequence and multiple-choice "
                "classification tasks only."
            )
        BenchmarkModule.__init__(
            self,
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )
        self.buffer["first_label_token_mapping"] = True
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()

    @property
    def data_collator(self) -> c.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """Raise because the hosted model does not support finetuning."""
        raise NotImplementedError("Typesafe System One does not support finetuning.")

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The standard EuroEval classification-label extractor."""
        return _extract_labels_from_generation_helper(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            first_label_token_mapping=True,
        )

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Evaluate each input and expose returned probabilities as log scores.

        Returns:
            Predicted labels and log-probability scores.

        Raises:
            InvalidBenchmark:
                If the input does not contain text or the task is unsupported.
        """
        if "text" not in inputs:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")
        texts = list(inputs["text"])
        labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        if (
            self.dataset_config.task.task_group
            == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            label_probs: list[dict[str, float]] = []
            for text in texts:
                question, options = parse_bare_question_and_choices(text=text)
                criteria: dict[str, str | None]
                if len(options) != len(labels) or len(set(options)) != len(options):
                    question, criteria = text, {label: None for label in labels}
                else:
                    criteria = dict(zip(labels, options, strict=True))
                label_probs.append(
                    self._classify_one(state=question, criteria=criteria)
                )
        else:
            criteria = {label: None for label in labels}
            label_probs = [
                self._classify_one(state=text, criteria=criteria) for text in texts
            ]

        sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for probabilities in label_probs:
            label_scores = sorted(
                (
                    (label, math.log(max(probabilities[label], _LOGPROB_FLOOR)))
                    for label in labels
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            sequences.append(label_scores[0][0])
            scores.append([label_scores])
        return GenerativeModelOutput(sequences=sequences, scores=scores)

    def _classify_one(
        self, state: str, criteria: dict[str, str | None]
    ) -> dict[str, float]:
        """Send one typed choice question to the System One endpoint.

        Returns:
            A probability for every candidate label.

        Raises:
            InvalidBenchmark:
                If the response does not contain probabilities for every label.
        """
        response = requests.post(
            _SYSTEM_ONE_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "state": state,
                "model": _MODEL_ID,
                "questions": {
                    "name": {
                        "type": "choice",
                        "instructions": self.buffer["instructions"],
                        "criteria": criteria,
                    }
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        try:
            probabilities = response.json()["answers"]["name"]["probabilities"]
        except (KeyError, TypeError) as error:
            raise InvalidBenchmark(
                "Typesafe System One returned a response without choice probabilities."
            ) from error
        if not isinstance(probabilities, dict):
            raise InvalidBenchmark(
                "Typesafe System One returned malformed choice probabilities."
            )
        missing_labels = [label for label in criteria if label not in probabilities]
        if missing_labels:
            raise InvalidBenchmark(
                "Typesafe System One did not return probabilities for labels "
                f"{missing_labels!r}."
            )
        parsed: dict[str, float] = {}
        for label in criteria:
            try:
                probability = float(probabilities[label])
            except (TypeError, ValueError) as error:
                raise InvalidBenchmark(
                    "Typesafe System One returned a non-numeric choice probability."
                ) from error
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise InvalidBenchmark(
                    "Typesafe System One returned a probability outside [0, 1]."
                )
            parsed[label] = probability
        return parsed

    @property
    def generative_type(self) -> GenerativeType | None:
        """None, because System One is a typed classifier, not a generator."""
        return None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Create configuration for the exact hosted model identifier.

        Returns:
            The dedicated Typesafe model configuration.

        Raises:
            InvalidModel:
                If a parameter or revision suffix is supplied.
        """
        components = split_model_id(model_id=model_id)
        if model_id != _MODEL_ID:
            raise InvalidModel(
                f"Typesafe System One supports only the exact model ID {_MODEL_ID!r}; "
                "parameter and revision suffixes are not supported."
            )
        return ModelConfig(
            model_id=components.model_id,
            revision=components.revision,
            param=None,
            task="text-classification",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.TYPESAFE,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=components.model_id
            ),
            adapter_base_model_id=None,
        )

    @classmethod
    def model_exists(cls, model_id: str, benchmark_config: BenchmarkConfig) -> bool:
        """Recognise the documented model ID without making a billable request.

        Returns:
            Whether the ID selects Jev System One.
        """
        return split_model_id(model_id=model_id).model_id == _MODEL_ID

    @cached_property
    def model_max_length(self) -> int:
        """The unknown context limit; Typesafe publishes no verified value."""
        return -1

    @cached_property
    def num_params(self) -> int:
        """The unknown parameter count; Typesafe publishes no verified value."""
        return -1

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """Raise because the hosted model does not support finetuning."""
        raise NotImplementedError("Typesafe System One does not support finetuning.")

    def update_dataset_config(self, dataset_config: DatasetConfig) -> t.Self:
        """Update task-specific instructions when moving to another dataset.

        Returns:
            This model instance.
        """
        self.dataset_config = dataset_config
        self._validate_labels()
        self.buffer["instructions"] = self._build_instructions()
        return self

    @cached_property
    def vocab_size(self) -> int:
        """The unknown vocabulary size; Typesafe publishes no verified value."""
        return -1
