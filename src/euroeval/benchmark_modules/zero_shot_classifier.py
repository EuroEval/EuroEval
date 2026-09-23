"""A benchmark module wrapping non-generative zero-shot "decision" models."""

import math
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
from ..exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..tokenisation_utils import get_first_label_token_mapping
from ..types import ExtractLabelsFunction
from ..zero_shot_adapters import ADAPTERS, ZeroShotClassifierAdapter, get_adapter
from .base import (
    BenchmarkModule,
    _extract_labels_from_generation_helper,
    _prepare_dataset_helper,
)

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.trainer import Trainer


class ZeroShotClassifierModel(BenchmarkModule):
    """A benchmark module wrapping a non-generative zero-shot "decision" model.

    These models (e.g. Laya) are encoders with trained heads that answer typed
    questions with calibrated probabilities. They are loaded through their own
    package, via one of the adapters in `euroeval.zero_shot_adapters.ADAPTERS`, and
    are evaluated zero-shot only: no finetuning, no few-shot demonstrations.
    """

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE

    # Checked before the generic encoder/generative backends, since these models
    # would otherwise be misidentified (e.g. an encoder without a root config.json).
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

        Raises:
            InvalidModel:
                If no adapter matches the model.
        """
        adapter_cls = get_adapter(
            model_id=model_config.model_id, benchmark_config=benchmark_config
        )
        if adapter_cls is None:
            raise InvalidModel(
                f"No zero-shot classifier adapter matches the model "
                f"{model_config.model_id!r}."
            )
        self.adapter: ZeroShotClassifierAdapter = adapter_cls(
            model_config=model_config, benchmark_config=benchmark_config
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
            "The `data_collator` property has not been implemented for zero-shot "
            "classifier models, as they are not finetuned."
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

    def _build_instructions(self) -> str:
        """Build the classification instructions from the dataset's templates.

        Returns:
            The instructions describing the classification task, with the
            `{text}` placeholder (which is filled in per-sample by the `texts`
            argument to `adapter.classify`) removed.
        """
        return self.dataset_config.instruction_prompt.format(
            text="", labels_str=self.dataset_config.get_labels_str()
        ).strip()

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.

        Raises:
            InvalidBenchmark:
                If the inputs do not contain a 'text' key, or if the dataset's task
                group is not supported.
        """
        if self.dataset_config.task.task_group not in (
            TaskGroup.SEQUENCE_CLASSIFICATION,
            TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
        ):
            raise InvalidBenchmark(
                "Zero-shot classifier models only support sequence classification "
                "and multiple-choice classification tasks, but the task group of "
                f"the dataset {self.dataset_config.name!r} is "
                f"{self.dataset_config.task.task_group!r}."
            )

        if "text" not in inputs:
            raise InvalidBenchmark("The inputs must contain a 'text' key.")

        texts = list(inputs["text"])

        candidate_labels = [
            self.dataset_config.prompt_label_mapping[label]
            for label in self.dataset_config.id2label.values()
        ]
        if not candidate_labels:
            raise InvalidBenchmark(
                "No candidate labels found for this dataset. Set "
                "DatasetConfig.labels/prompt_label_mapping for classification tasks "
                "before using a zero-shot classifier model."
            )

        instructions = self._build_instructions()
        label_probs = self.adapter.classify(
            texts=texts, candidate_labels=candidate_labels, instructions=instructions
        )

        sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for probs in label_probs:
            sample_scores: list[tuple[str, float]] = []
            best_label = candidate_labels[0]
            best_prob = -math.inf
            for label in candidate_labels:
                prob = max(probs.get(label, 0.0), 1e-12)
                sample_scores.append((label, math.log(prob)))
                if prob > best_prob:
                    best_prob = prob
                    best_label = label
            sequences.append(best_label)
            scores.append([sample_scores])

        return GenerativeModelOutput(sequences=sequences, scores=scores)

    @property
    def generative_type(self) -> GenerativeType | None:
        """The generative type of the model.

        Zero-shot classifier models are not generative, so this is always None.

        Returns:
            The generative type of the model.
        """
        return None

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

        Raises:
            InvalidModel:
                If no adapter matches the model, or the given parameter is not
                allowed for the matching adapter.
        """
        model_id_components = split_model_id(model_id=model_id)
        adapter_cls = get_adapter(
            model_id=model_id_components.model_id, benchmark_config=benchmark_config
        )
        if adapter_cls is None:
            raise InvalidModel(
                f"No zero-shot classifier adapter matches the model {model_id!r}."
            )

        param = model_id_components.param
        if param is not None and param not in adapter_cls.allowed_params:
            msg = f"Invalid parameter {param!r} for model {model_id!r}."
            if adapter_cls.allowed_params:
                msg += (
                    f" Allowed parameters are: {', '.join(adapter_cls.allowed_params)}."
                )
            else:
                msg += " No parameters are allowed."
            raise InvalidModel(msg)

        return ModelConfig(
            model_id=model_id_components.model_id,
            revision=model_id_components.revision,
            param=param,
            task="text-classification",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.ZERO_SHOT_CLASSIFIER,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir,
                model_id=model_id_components.model_id,
            ),
            adapter_base_model_id=None,
        )

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists.
        """
        model_id_components = split_model_id(model_id=model_id)
        needs_extras: list[str] = list()
        for adapter_cls in ADAPTERS:
            matches_or_err = adapter_cls.matches(
                model_id=model_id_components.model_id, benchmark_config=benchmark_config
            )
            if isinstance(matches_or_err, NeedsExtraInstalled):
                needs_extras.append(matches_or_err.extra)
            elif matches_or_err is True:
                return True
        if needs_extras:
            return NeedsExtraInstalled(extra=needs_extras[0])
        return False

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        return self.adapter.max_length

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        if self.benchmark_config.num_parameters is not None:
            return self.benchmark_config.num_parameters
        adapter_cls = type(self.adapter)
        return adapter_cls.num_params(
            model_id=self.model_config.model_id, param=self.model_config.param
        )

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
            "The `trainer_class` property has not been implemented for zero-shot "
            "classifier models, as they are not finetuned."
        )

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        return -1
