"""A built-in dummy model, used for debugging."""

import math
import random
import re
import typing as t
from functools import cached_property

from ..data_models import (
    BenchmarkConfig,
    DatasetConfig,
    GenerativeModelOutput,
    ModelConfig,
    Task,
)
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import InvalidBenchmark, NeedsEnvironmentVariable, NeedsExtraInstalled
from ..generation_utils import raise_if_wrong_params
from ..model_cache import create_model_cache_dir
from ..task_group_utils.token_classification import serialise_ner_tags
from ..tokenisation_utils import get_first_label_token_mapping
from ..types import ExtractLabelsFunction
from .base import (
    BenchmarkModule,
    _extract_labels_from_generation_helper,
    _prepare_dataset_helper,
)

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.trainer import Trainer


DUMMY_MODEL_ID = "dummy"


class DummyModel(BenchmarkModule):
    """A built-in model that predicts an even distribution over labels.

    This model does not download or run any real model, and requires no
    inference framework (HF `transformers`, vLLM, an API, etc.) at all. It
    exists to let a user debug a dataset/task pipeline in isolation from any
    real inference backend, and can also serve as a naive random baseline.
    """

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    allowed_params = {re.compile(r".*"): []}

    # Checked before any other backend, so that benchmarking "dummy" never
    # triggers a real HF Hub lookup for a repo literally named "dummy".
    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
            log_metadata:
                Whether to log the model metadata.
        """
        raise_if_wrong_params(
            model_config=model_config, allowed_params=self.allowed_params
        )

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokeniser=None,
            generative_type=self.generative_type,
            log_metadata=self.log_metadata,
        )

    @property
    def data_collator(self) -> t.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for dummy models."
        )

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        return _extract_labels_from_generation_helper(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            first_label_token_mapping=self.buffer["first_label_token_mapping"],
        )

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Predicts an even probability distribution across the candidate labels
        for classification-style tasks, an explicit "no entities" prediction
        for token classification, and a generic placeholder answer for every
        other (free-text-answer) task group.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.

        Raises:
            InvalidBenchmark:
                If the inputs do not contain either 'messages' or 'text' keys.
        """
        # Recomputed on every call rather than once in __init__, since
        # update_dataset_config can swap dataset_config between calls.
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokeniser=None,
            generative_type=self.generative_type,
            log_metadata=self.log_metadata,
        )

        if "messages" in inputs:
            num_samples = len(inputs["messages"])
        elif "text" in inputs:
            num_samples = len(inputs["text"])
        else:
            raise InvalidBenchmark(
                "The inputs must contain either 'messages' or 'text' keys."
            )

        if self.dataset_config.task.task_group == TaskGroup.TOKEN_CLASSIFICATION:
            # Reuses the same serialiser as the few-shot demonstrations and
            # gold BPC answers, so the "no entities" shape can't drift out of
            # sync with it.
            empty_prediction = serialise_ner_tags(
                tokens=[],
                labels=[],
                prompt_label_mapping=self.dataset_config.prompt_label_mapping,
            )
            return GenerativeModelOutput(sequences=[empty_prediction] * num_samples)
        elif self.dataset_config.task.task_group in (
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        ):
            # For sequence classification and multiple choice classification,
            # we sample uniformly at random from the available candidate labels.
            candidate_labels = [
                self.dataset_config.prompt_label_mapping[label]
                for label in self.dataset_config.id2label.values()
            ]
            if not candidate_labels:
                raise InvalidBenchmark(
                    "No candidate labels found for this dataset. "
                    "Set DatasetConfig.labels/prompt_label_mapping for classification "
                    "tasks before using the dummy backend."
                )
            uniform_logprob = math.log(1 / len(candidate_labels))
            scores = [
                [[(label, uniform_logprob) for label in candidate_labels]]
                for _ in range(num_samples)
            ]
            sequences = [random.choice(candidate_labels) for _ in range(num_samples)]
            return GenerativeModelOutput(sequences=sequences, scores=scores)

        # A generic non-empty placeholder answer for non-classification tasks.
        # Some metrics (e.g. in the hallucination task) can't process an empty answer.
        return GenerativeModelOutput(sequences=["answer"] * num_samples)

    @property
    def generative_type(self) -> GenerativeType | None:
        """The generative type of the model.

        Always instruction-tuned: this must not be `REASONING`, since that
        would make `get_first_label_token_mapping` disable logprob output
        entirely, while `generate` always populates scores for classification
        tasks regardless.

        Returns:
            The generative type of the model.
        """
        return GenerativeType.INSTRUCTION_TUNED

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        return ModelConfig(
            model_id=DUMMY_MODEL_ID,
            revision="main",
            param=None,
            task="text-generation",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.DUMMY,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=DUMMY_MODEL_ID
            ),
            adapter_base_model_id=None,
        )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists.
        """
        return model_id == DUMMY_MODEL_ID

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        return -1

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        return -1

    def prepare_dataset(
        self, dataset: "DatasetDict", task: Task, itr_idx: int
    ) -> "DatasetDict":
        """Prepare the dataset for the model.

        Args:
            dataset:
                The dataset to prepare.
            task:
                The task to prepare the dataset for.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The prepared dataset.
        """
        return _prepare_dataset_helper(
            dataset=dataset,
            task=task,
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            benchmark_config=self.benchmark_config,
            generative_type=self.generative_type,
            itr_idx=itr_idx,
            always_populate_text_field=False,
            tokeniser=None,
        )

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for dummy models."
        )

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        return -1
