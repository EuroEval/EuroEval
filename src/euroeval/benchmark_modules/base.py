"""Abstract benchmark module class that the model classes inherit from."""

import collections.abc as c
import logging
import re
import typing as t
from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import cast

from datasets import Dataset, DatasetDict
from torch import nn

from ..constants import MERGE_TAGS
from ..data_models import ModelConfig
from ..enums import TaskGroup
from ..exceptions import InvalidBenchmark, NeedsEnvironmentVariable, NeedsExtraInstalled
from ..generation_utils import apply_prompt, extract_few_shot_examples
from ..languages import get_all_languages
from ..logging_utils import get_pbar, log_once
from ..model_cache import create_model_cache_dir
from ..string_utils import split_model_id
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)

if t.TYPE_CHECKING:
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.trainer import Trainer

    from ..data_models import (
        BenchmarkConfig,
        DatasetConfig,
        GenerativeModelOutput,
        HFModelInfo,
        Task,
    )
    from ..enums import BatchingPreference, GenerativeType, InferenceBackend, ModelType
    from ..types import ComputeMetricsFunction, ExtractLabelsFunction


class BenchmarkModule(ABC):
    """Abstract class for a benchmark module.

    Attributes:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.
        buffer:
            A buffer to store temporary data.
    """

    fresh_model: bool
    batching_preference: "BatchingPreference"
    high_priority: bool
    allowed_params: dict[re.Pattern[str], c.Sequence[str]] = {re.compile(r".*"): []}
    _model: nn.Module

    def __init__(
        self,
        model_config: "ModelConfig",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
        log_metadata: bool = True,
    ) -> None:
        """Initialise the benchmark module.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
            log_metadata:
                Whether to log the metadata of the model.
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.benchmark_config = benchmark_config
        self.log_metadata = log_metadata
        self.buffer: dict[str, t.Any] = dict()
        if self.log_metadata:
            self._log_metadata()

    def _log_metadata(self) -> None:
        """Log the metadata of the model."""
        model_id = self.model_config.model_id
        logging_msg: str = "    ↳ "
        if self.num_params < 0:
            logging_msg += f"The model {model_id} has an unknown number of parameters, "
        else:
            logging_msg += f"The model {model_id} has {self.num_params:,} parameters, "
        if self.vocab_size < 0:
            logging_msg += "an unknown vocabulary size, "
        else:
            logging_msg += f"a vocabulary size of {self.vocab_size:,}, "
        if self.model_max_length < 0:
            logging_msg += "and an unknown maximum sequence length."
        else:
            logging_msg += f"and a maximum context length of {self.model_max_length:,}."
        log_once(message=logging_msg, level=logging.INFO)

    @property
    def compute_metrics(self) -> "ComputeMetricsFunction":
        """The function used to compute the metrics.

        Returns:
            The function used to compute the metrics.
        """
        match self.dataset_config.task.task_group:
            case TaskGroup.SEQUENCE_CLASSIFICATION:
                return cast(
                    "ComputeMetricsFunction",
                    partial(
                        sequence_classification.compute_metrics,
                        dataset_config=self.dataset_config,
                        benchmark_config=self.benchmark_config,
                    ),
                )
            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                return cast(
                    "ComputeMetricsFunction",
                    partial(
                        sequence_classification.compute_metrics,
                        dataset_config=self.dataset_config,
                        benchmark_config=self.benchmark_config,
                    ),
                )
            case TaskGroup.TEXT_TO_TEXT:
                return cast(
                    "ComputeMetricsFunction",
                    partial(
                        text_to_text.compute_metrics,
                        dataset_config=self.dataset_config,
                        benchmark_config=self.benchmark_config,
                    ),
                )
            case TaskGroup.TOKEN_CLASSIFICATION:
                return cast(
                    "ComputeMetricsFunction",
                    partial(
                        token_classification.compute_metrics,
                        has_misc_tags=self.buffer.get("has_misc_tags", True),
                        dataset_config=self.dataset_config,
                        benchmark_config=self.benchmark_config,
                    ),
                )
            case TaskGroup.QUESTION_ANSWERING:
                return cast(
                    "ComputeMetricsFunction",
                    partial(
                        question_answering.compute_metrics,
                        dataset_config=self.dataset_config,
                        benchmark_config=self.benchmark_config,
                    ),
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    @abstractmethod
    def data_collator(self) -> c.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        ...

    @property
    @abstractmethod
    def extract_labels_from_generation(self) -> "ExtractLabelsFunction":
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        ...

    def generate(self, inputs: dict) -> "GenerativeModelOutput":
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        raise NotImplementedError(
            "The `generate` method has not been implemented for "
            f"{self.__class__.__name__}."
        )

    @property
    @abstractmethod
    def generative_type(self) -> "GenerativeType | None":
        """The generative type of the model.

        Returns:
            The generative type of the model, or None if the model is not generative.
        """
        ...

    @classmethod
    @abstractmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: "BenchmarkConfig"
    ) -> "ModelConfig":
        """Fetch the model configuration.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The model configuration.
        """
        ...

    def get_pytorch_module(self) -> "nn.Module":
        """Get the underlying PyTorch module.

        Returns:
            The PyTorch module.
        """
        if hasattr(self, "_model"):
            return self._model
        raise NotImplementedError(
            "The `get_pytorch_module` method has not been implemented for "
            f"{self.__class__.__name__}."
        )

    def get_tokeniser(self) -> "PreTrainedTokenizer":
        """Get the underlying tokeniser.

        Returns:
            The tokeniser.
        """
        if hasattr(self, "_tokeniser"):
            return self._tokeniser
        raise NotImplementedError(
            "The `get_tokeniser` method has not been implemented for "
            f"{self.__class__.__name__}."
        )

    @classmethod
    @abstractmethod
    def model_exists(
        cls, model_id: str, benchmark_config: "BenchmarkConfig"
    ) -> bool | NeedsExtraInstalled | NeedsEnvironmentVariable:
        """Check if a model exists.

        Args:
            model_id:
                The model ID.
            benchmark_config:
                The benchmark configuration.

        Returns:
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        ...

    @cached_property
    @abstractmethod
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        ...

    @cached_property
    @abstractmethod
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        ...

    def prepare_datasets(
        self, datasets: list[DatasetDict], task: "Task"
    ) -> c.Sequence[DatasetDict]:
        """Prepare the datasets for the model.

        This includes things like tokenisation.

        Args:
            datasets:
                The datasets to prepare.
            task:
                The task to prepare the datasets for.

        Returns:
            The prepared datasets.

        Raises:
            InvalidBenchmark:
                If the dataset does not have a 'train' split for token classification
                tasks.
        """
        for idx, dataset in enumerate(
            get_pbar(
                iterable=datasets,
                desc="Preparing datasets",
                disable=not self.benchmark_config.progress_bar,
            )
        ):
            prepared_dataset = self.prepare_dataset(
                dataset=dataset, task=task, itr_idx=idx
            )
            if self.dataset_config.task.task_group == TaskGroup.TOKEN_CLASSIFICATION:
                if "train" not in dataset:
                    raise InvalidBenchmark(
                        "The dataset does not have a 'train' split, which is required "
                        "for token classification tasks."
                    )
                labels_in_train: set[str] = {
                    tag for tag_list in dataset["train"]["labels"] for tag in tag_list
                }
                self.buffer["has_misc_tags"] = (
                    "B-MISC" in labels_in_train or "I-MISC" in labels_in_train
                )

            datasets_dict: dict[str, Dataset] = dict()
            for split_name, split in prepared_dataset.items():
                datasets_dict[str(split_name)] = split
            for split_name, split in dataset.items():
                datasets_dict[f"original_{split_name}"] = split

            datasets[idx] = DatasetDict(datasets_dict)
        return datasets

    @abstractmethod
    def prepare_dataset(
        self, dataset: DatasetDict, task: "Task", itr_idx: int
    ) -> DatasetDict:
        """Prepare the dataset for the model.

        This includes things like tokenisation.

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
        ...

    @property
    @abstractmethod
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        ...

    def update_dataset_config(self, dataset_config: "DatasetConfig") -> t.Self:
        """Update the dataset config registered in the benchmark module.

        Args:
            dataset_config:
                The new dataset config.

        Returns:
            The benchmark module.
        """
        self.dataset_config = dataset_config
        return self

    @cached_property
    @abstractmethod
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        ...


def _build_model_config_helper(
    model_id: str,
    revision: str,
    param: str | None,
    task: str,
    model_info: "HFModelInfo",
    benchmark_config: "BenchmarkConfig",
    inference_backend: "InferenceBackend",
    model_type: "ModelType",
    adapter_base_model_id: str | None,
    generation_config: "GenerationConfig | None" = None,
) -> "ModelConfig":
    """Helper function to build a ModelConfig from shared components.

    Args:
        model_id:
            The model ID.
        revision:
            The model revision.
        param:
            The model parameter, or None if not applicable.
        task:
            The task that the model was trained on.
        model_info:
            The model information from Hugging Face Hub.
        benchmark_config:
            The benchmark configuration.
        inference_backend:
            The inference backend to use.
        model_type:
            The model type.
        adapter_base_model_id:
            The base model ID if this is an adapter model.
        generation_config:
            The generation configuration, or None if not applicable.

    Returns:
        The constructed ModelConfig.
    """
    language_mapping = get_all_languages()
    language_codes = list(language_mapping.keys())

    return ModelConfig(
        model_id=model_id,
        revision=revision,
        param=param,
        task=task,
        languages=[
            language_mapping[tag] for tag in model_info.tags if tag in language_codes
        ],
        merge=any(tag in model_info.tags for tag in MERGE_TAGS),
        inference_backend=inference_backend,
        model_type=model_type,
        fresh=False,
        model_cache_dir=create_model_cache_dir(
            cache_dir=benchmark_config.cache_dir, model_id=model_id
        ),
        adapter_base_model_id=adapter_base_model_id,
        generation_config=generation_config,
    )


def _extract_labels_from_generation_helper(
    dataset_config: "DatasetConfig",
    model_config: "ModelConfig",
    first_label_token_mapping: dict[str, str] | bool,
) -> "ExtractLabelsFunction":
    """Helper function to extract labels from generated output.

    Args:
        dataset_config:
            The dataset configuration.
        model_config:
            The model configuration.
        first_label_token_mapping:
            Mapping from labels to their first token IDs.

    Returns:
        The function used to extract labels from the generated output.

    Raises:
        NotImplementedError:
            If the task group is not supported.
    """
    match dataset_config.task.task_group:
        case (
            TaskGroup.SEQUENCE_CLASSIFICATION | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            return partial(
                sequence_classification.extract_labels_from_generation,
                dataset_config=dataset_config,
                model_config=model_config,
                first_label_token_mapping=first_label_token_mapping,
            )
        case TaskGroup.TEXT_TO_TEXT:
            return text_to_text.extract_labels_from_generation
        case TaskGroup.TOKEN_CLASSIFICATION:
            return partial(
                token_classification.extract_labels_from_generation,
                dataset_config=dataset_config,
            )
        case TaskGroup.QUESTION_ANSWERING:
            return question_answering.extract_labels_from_generation
        case _:
            raise NotImplementedError(
                f"Unsupported task group: {dataset_config.task.task_group}."
            )


def _lookup_model_info(
    model_id: str, benchmark_config: "BenchmarkConfig"
) -> tuple[str, str, "HFModelInfo | None"]:
    """Helper function to look up model information from Hugging Face Hub.

    This performs the shared lookup logic: split the model ID and fetch
    repository information.

    Args:
        model_id:
            The model ID.
        benchmark_config:
            The benchmark configuration.

    Returns:
        A tuple of (model_id, revision, model_info), where model_info may be None
        if the model was not found.

    Note:
        Uses a late import of get_model_repo_info to avoid circular imports.
    """
    # Late import to avoid circular dependency with hf.py
    from .hf import get_model_repo_info  # noqa: PLC0415

    model_id_components = split_model_id(model_id=model_id)
    model_info = get_model_repo_info(
        model_id=model_id_components.model_id,
        revision=model_id_components.revision,
        api_key=benchmark_config.api_key,
        cache_dir=benchmark_config.cache_dir,
        trust_remote_code=benchmark_config.trust_remote_code,
        requires_safetensors=benchmark_config.requires_safetensors,
        run_with_cli=benchmark_config.run_with_cli,
    )
    return (model_id_components.model_id, model_id_components.revision, model_info)


def _prepare_dataset_helper(
    dataset: DatasetDict,
    task: "Task",
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    generative_type: "GenerativeType | None",
    itr_idx: int,
    always_populate_text_field: bool,
    tokeniser: "PreTrainedTokenizer | None",
) -> DatasetDict:
    """Helper function to prepare a dataset for a generative model.

    Args:
        dataset:
            The dataset to prepare.
        task:
            The task to prepare the dataset for.
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration.
        generative_type:
            The generative type of the model.
        itr_idx:
            The index of the dataset in the iterator.
        always_populate_text_field:
            Whether to always populate the text field.
        tokeniser:
            The tokeniser to use, or None if not applicable.

    Returns:
        The prepared dataset.
    """
    if task.task_group == TaskGroup.QUESTION_ANSWERING:
        dataset = dataset.map(
            lambda examples: dict(
                label=[
                    dict(
                        id=id,
                        answers=dict(
                            answer_start=answer_dct["answer_start"],
                            text=[
                                answer_text.lower()
                                for answer_text in answer_dct["text"]
                            ],
                        ),
                    )
                    for id, answer_dct in zip(examples["id"], examples["answers"])
                ]
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    if benchmark_config.few_shot:
        few_shot_examples = extract_few_shot_examples(
            dataset=dataset,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            itr_idx=itr_idx,
        )
    else:
        few_shot_examples = list()

    mapped_dataset = dataset["test"].map(
        partial(
            apply_prompt,
            few_shot_examples=few_shot_examples,
            model_config=model_config,
            dataset_config=dataset_config,
            generative_type=generative_type,
            always_populate_text_field=always_populate_text_field,
            tokeniser=tokeniser,
            use_bits_per_character=benchmark_config.use_bits_per_character,
        ),
        batched=True,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    assert isinstance(mapped_dataset, Dataset), (
        "Mapped dataset is not a Dataset instance."
    )
    dataset["test"] = mapped_dataset

    return dataset
