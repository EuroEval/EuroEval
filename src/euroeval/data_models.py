"""Data models used in EuroEval."""

import collections.abc as c
import importlib.metadata
import json
import logging
import re
import typing as t
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pydantic
import torch
from transformers.generation.configuration_utils import GenerationConfig

from .constants import ATTENTION_BACKENDS, MAX_NUMBER_OF_LOGGING_LANGUAGES
from .enums import Device, GenerativeType, ModelType, TaskGroup
from .exceptions import InvalidBenchmark
from .languages import (
    ENGLISH,
    EUROPEAN_PORTUGUESE,
    NORWEGIAN,
    NORWEGIAN_BOKMÅL,
    NORWEGIAN_NYNORSK,
    PORTUGUESE,
    Language,
)
from .logging_utils import log_once
from .metrics.base import Metric
from .types import ScoreDict

if t.TYPE_CHECKING:
    from .enums import InferenceBackend


def get_package_version(package_name: str) -> str | None:
    """Get the version of a package.

    Args:
        package_name:
            The name of the package.

    Returns:
        The version of the package, or None if the package is not installed.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass
class PromptConfig:
    """Configuration for task-specific prompting across languages.

    Defines the prompt templates needed for evaluating a specific task in a given
    language.

    Attributes:
        default_prompt_prefix:
            The default prefix to use in the few-shot prompt.
        default_prompt_template:
            The default template for the prompt to use when benchmarking the dataset
            using few-shot evaluation.
        default_instruction_prompt:
            The default prompt to use when benchmarking the dataset using
            instruction-based evaluation.
        default_prompt_label_mapping:
            The default mapping from the labels to another phrase which is used as a
            substitute for the label in few-shot evaluation. If set to "auto", the
            mapping will be set to a 1:1 mapping between the labels and themselves.
    """

    default_prompt_prefix: str
    default_prompt_template: str
    default_instruction_prompt: str
    default_prompt_label_mapping: dict[str, str] | t.Literal["auto"]


@dataclass
class Task:
    """A dataset task.

    Attributes:
        name:
            The name of the task.
        task_group:
            The task group of the task.
        template_dict:
            The template dictionary for the task, from language (or language tuples) to
            prompt template.
        metrics:
            The metrics used to evaluate the task.
        default_num_few_shot_examples:
            The default number of examples to use when benchmarking the task using
            few-shot evaluation. For a classification task, these will be drawn evenly
            from each label.
        default_max_generated_tokens:
            The default maximum number of tokens to generate when benchmarking the task
            using few-shot evaluation.
        default_labels (optional):
            The default labels for datasets using this task. Can be None if the labels
            should be set manually in the dataset configs. Defaults to an empty tuple.
        requires_zero_shot (optional):
            Whether to only allow zero-shot evaluation for this task. If True, the
            task will not be evaluated using few-shot examples.
        uses_structured_output (optional):
            Whether the task uses structured output. If True, the task will return
            structured output (e.g., BIO tags for NER). Defaults to False.
        uses_logprobs (optional):
            Whether the task uses log probabilities. If True, the task will return
            log probabilities for the generated tokens. Defaults to False.
        requires_logprobs (optional):
            Whether the task requires log probabilities. Implies `uses_logprobs`.
        default_allowed_model_types (optional):
            A list of model types that are allowed to be evaluated on this task.
            Defaults to all model types being allowed.
        default_allowed_generative_types (optional):
            A list of generative model types that are allowed to be evaluated on this
            task. If None, all generative model types are allowed. Only relevant if
            `allowed_model_types` includes generative models.
        default_allow_invalid_model_outputs (optional):
            Whether to allow invalid model outputs. This is only relevant for generative
            models on classification tasks, where the model may generate an output
            which is not one of the allowed labels. If True, the model output will be
            mapped to the closest valid label. If False, the model output will be
            considered incorrect and the evaluation will be aborted. Defaults to True.
    """

    model_config = pydantic.ConfigDict(
        protected_namespaces=(), arbitrary_types_allowed=True
    )

    name: str
    task_group: TaskGroup
    template_dict: (
        dict[Language, PromptConfig] | dict[tuple[Language, Language], PromptConfig]
    )
    metrics: c.Sequence[Metric]
    default_num_few_shot_examples: int
    default_max_generated_tokens: int
    default_labels: c.Sequence[str] | None = tuple()
    requires_zero_shot: bool = False
    uses_structured_output: bool = False
    uses_logprobs: bool = False
    requires_logprobs: bool = False
    default_allowed_model_types: c.Sequence[ModelType] = field(
        default_factory=lambda: [ModelType.ENCODER, ModelType.GENERATIVE]
    )
    default_allowed_generative_types: c.Sequence[GenerativeType] = field(
        default_factory=lambda: [
            GenerativeType.BASE,
            GenerativeType.INSTRUCTION_TUNED,
            GenerativeType.REASONING,
        ]
    )
    default_allow_invalid_model_outputs: bool = True

    def __post_init__(self) -> None:
        """Post-initialisation checks."""
        self.uses_logprobs = self.uses_logprobs or self.requires_logprobs

    def __hash__(self) -> int:
        """Return a hash of the task."""
        return hash(self.name)


class DatasetConfig:
    """Configuration for a dataset."""

    def __init__(
        self,
        task: Task,
        languages: c.Sequence[Language],
        name: str | None = None,
        pretty_name: str | None = None,
        source: str | dict[str, str] | None = None,
        prompt_prefix: str | None = None,
        prompt_template: str | None = None,
        instruction_prompt: str | None = None,
        num_few_shot_examples: int | None = None,
        max_generated_tokens: int | None = None,
        labels: c.Sequence[str] | None = None,
        prompt_label_mapping: dict[str, str] | t.Literal["auto"] | None = None,
        allowed_model_types: c.Sequence[ModelType] | None = None,
        allowed_generative_types: c.Sequence[GenerativeType] | None = None,
        allow_invalid_model_outputs: bool | None = None,
        train_split: str | None = "train",
        val_split: str | None = "val",
        test_split: str = "test",
        bootstrap_samples: bool = True,
        unofficial: bool = False,
        _prompt_prefix: str | None = None,
        _prompt_template: str | None = None,
        _instruction_prompt: str | None = None,
        _num_few_shot_examples: int | None = None,
        _max_generated_tokens: int | None = None,
        _labels: c.Sequence[str] | None = None,
        _prompt_label_mapping: dict[str, str] | t.Literal["auto"] | None = None,
        _allowed_model_types: c.Sequence[ModelType] | None = None,
        _allowed_generative_types: c.Sequence[GenerativeType] | None = None,
        _allow_invalid_model_outputs: bool | None = None,
        _logging_string: str | None = None,
    ) -> None:
        """Initialise a DatasetConfig object.

        Args:
            task:
                The task of the dataset.
            languages:
                The ISO 639-1 language codes of the entries in the dataset.
            name (optional):
                The name of the dataset. Must be lower case with no spaces. Can be None
                if and only if the dataset config resides directly in the Hugging Face
                dataset repo. Defaults to None.
            pretty_name (optional):
                A longer prettier name for the dataset, which allows cases and spaces.
                Used for logging. Can be None if and only if the dataset config resides
                directly in the Hugging Face dataset repo. Defaults to None.
            source (optional):
                The source of the dataset, which can be a Hugging Face ID or a
                dictionary with keys "train", "val" and "test" mapping to local CSV file
                paths. Can be None if and only if the dataset config resides directly in
                the Hugging Face dataset repo. Defaults to None.
            prompt_prefix (optional):
                The prefix to use in the few-shot prompt. Defaults to the template for
                the task and language.
            prompt_template (optional):
                The template for the prompt to use when benchmarking the dataset using
                few-shot evaluation. Defaults to the template for the task and language.
            instruction_prompt (optional):
                The prompt to use when benchmarking the dataset using instruction-based
                evaluation. Defaults to the template for the task and language.
            num_few_shot_examples (optional):
                The number of examples to use when benchmarking the dataset using
                few-shot evaluation. For a classification task, these will be drawn
                evenly from each label. Defaults to the template for the task and
                language.
            max_generated_tokens (optional):
                The maximum number of tokens to generate when benchmarking the dataset
                using few-shot evaluation. Defaults to the template for the task and
                language.
            labels (optional):
                The labels in the dataset. Defaults to the template for the task and
                language.
            prompt_label_mapping (optional):
                A mapping from the labels to another phrase which is used as a
                substitute for the label in few-shot evaluation. If "auto" then the
                mapping will be set to a 1:1 mapping between the labels and themselves.
                If None then the mapping will be set to the default mapping for the task
                and language. Defaults to None.
            allowed_model_types (optional):
                A list of model types that are allowed to be evaluated on this dataset.
                Defaults to the one for the task.
            allowed_generative_types (optional):
                A list of generative model types that are allowed to be evaluated on
                this dataset. If None, all generative model types are allowed. Only
                relevant if `allowed_model_types` includes generative models. Defaults
                to the one for the task.
            allow_invalid_model_outputs (optional):
                Whether to allow invalid model outputs. This is only relevant for
                generative models on classification tasks, where the model may generate
                an output which is not one of the allowed labels. If True, the model
                output will be mapped to the closest valid label. If False, the model
                output will be considered incorrect and the evaluation will be aborted.
                Defaults to the one for the task.
            train_split (optional):
                The name of the split to use as the training set. Can be None if there
                is no training split in the dataset. Defaults to "train".
            val_split (optional):
                The name of the split to use as the validation set. Can be None if there
                is no validation split in the dataset. Defaults to "val".
            test_split (optional):
                The name of the split to use as the test set. Defaults to "test".
            bootstrap_samples (optional):
                Whether to bootstrap the dataset samples. Defaults to True.
            unofficial (optional):
                Whether the dataset is unofficial. Defaults to False.
            _prompt_prefix (optional):
                This argument is deprecated. Please use `prompt_prefix` instead.
            _prompt_template (optional):
                This argument is deprecated. Please use `prompt_template` instead.
            _instruction_prompt (optional):
                This argument is deprecated. Please use `instruction_prompt` instead.
            _num_few_shot_examples (optional):
                This argument is deprecated. Please use `num_few_shot_examples` instead.
            _max_generated_tokens (optional):
                This argument is deprecated. Please use `max_generated_tokens` instead.
            _labels (optional):
                This argument is deprecated. Please use `labels` instead.
            _prompt_label_mapping (optional):
                This argument is deprecated. Please use `prompt_label_mapping` instead.
            _allowed_model_types (optional):
                This argument is deprecated. Please use `allowed_model_types` instead.
            _allowed_generative_types (optional):
                This argument is deprecated. Please use `allowed_generative_types`
                instead.
            _allow_invalid_model_outputs (optional):
                This argument is deprecated. Please use `allow_invalid_model_outputs`
                instead.
            _logging_string (optional):
                This argument is deprecated. Please use `logging_string` instead.
        """
        # Deprecation warnings
        if _prompt_prefix is not None:
            log_once(
                "The `_prompt_prefix` argument is deprecated. Please use "
                "`prompt_prefix` instead.",
                level=logging.WARNING,
            )
            prompt_prefix = _prompt_prefix
        if _prompt_template is not None:
            log_once(
                "The `_prompt_template` argument is deprecated. Please use "
                "`prompt_template` instead.",
                level=logging.WARNING,
            )
            prompt_template = _prompt_template
        if _instruction_prompt is not None:
            log_once(
                "The `_instruction_prompt` argument is deprecated. Please use "
                "`instruction_prompt` instead.",
                level=logging.WARNING,
            )
            instruction_prompt = _instruction_prompt
        if _num_few_shot_examples is not None:
            log_once(
                "The `_num_few_shot_examples` argument is deprecated. Please use "
                "`num_few_shot_examples` instead.",
                level=logging.WARNING,
            )
            num_few_shot_examples = _num_few_shot_examples
        if _max_generated_tokens is not None:
            log_once(
                "The `_max_generated_tokens` argument is deprecated. Please use "
                "`max_generated_tokens` instead.",
                level=logging.WARNING,
            )
            max_generated_tokens = _max_generated_tokens
        if _labels is not None:
            log_once(
                "The `_labels` argument is deprecated. Please use `labels` instead.",
                level=logging.WARNING,
            )
            labels = _labels
        if _prompt_label_mapping is not None:
            log_once(
                "The `_prompt_label_mapping` argument is deprecated. Please use "
                "`prompt_label_mapping` instead.",
                level=logging.WARNING,
            )
            prompt_label_mapping = _prompt_label_mapping
        if _allowed_model_types is not None:
            log_once(
                "The `_allowed_model_types` argument is deprecated. Please use "
                "`allowed_model_types` instead.",
                level=logging.WARNING,
            )
            allowed_model_types = _allowed_model_types
        if _allowed_generative_types is not None:
            log_once(
                "The `_allowed_generative_types` argument is deprecated. Please use "
                "`allowed_generative_types` instead.",
                level=logging.WARNING,
            )
            allowed_generative_types = _allowed_generative_types
        if _allow_invalid_model_outputs is not None:
            log_once(
                "The `_allow_invalid_model_outputs` argument is deprecated. Please use "
                "`allow_invalid_model_outputs` instead.",
                level=logging.WARNING,
            )
            allow_invalid_model_outputs = _allow_invalid_model_outputs
        if _logging_string is not None:
            log_once(
                "The `_logging_string` argument is deprecated and is not used anymore. "
                "Using it will have no effect.",
                level=logging.WARNING,
            )

        self._name = name
        self._pretty_name = pretty_name
        self._source = source
        self.task = task
        self.languages = languages

        template = self.task.template_dict.get(  # pyrefly: ignore[no-matching-overload]
            self.main_language
        )
        self.prompt_prefix = (
            prompt_prefix
            if prompt_prefix is not None
            else template.default_prompt_prefix
            if template is not None
            else ""
        )
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else template.default_prompt_template
            if template is not None
            else ""
        )
        self.instruction_prompt = (
            instruction_prompt
            if instruction_prompt is not None
            else template.default_instruction_prompt
            if template is not None
            else ""
        )
        self.num_few_shot_examples = (
            num_few_shot_examples
            if num_few_shot_examples is not None
            else self.task.default_num_few_shot_examples
        )
        self.max_generated_tokens = (
            max_generated_tokens
            if max_generated_tokens is not None
            else self.task.default_max_generated_tokens
        )
        self.labels = (
            labels if labels is not None else self.task.default_labels or list()
        )
        if prompt_label_mapping is None:
            prompt_label_mapping = (
                template.default_prompt_label_mapping
                if template is not None
                else dict()
            )
        self.prompt_label_mapping = (
            {label: label for label in self.labels}
            if prompt_label_mapping == "auto"
            else prompt_label_mapping
        )
        self.allowed_model_types = (
            allowed_model_types
            if allowed_model_types is not None
            else self.task.default_allowed_model_types
        )
        self.allowed_generative_types = (
            allowed_generative_types
            if allowed_generative_types is not None
            else self.task.default_allowed_generative_types
        )
        self.allow_invalid_model_outputs = (
            allow_invalid_model_outputs
            if allow_invalid_model_outputs is not None
            else self.task.default_allow_invalid_model_outputs
        )
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.bootstrap_samples = bootstrap_samples
        self.unofficial = unofficial

    @property
    def name(self) -> str:
        """The name of the dataset.

        Returns:
            The name of the dataset.

        Raises:
            ValueError:
                If the name of the dataset is not set.
        """
        if self._name is None:
            raise ValueError("The name of the dataset is not set!")
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the dataset.

        Args:
            value:
                The new name of the dataset.
        """
        self._name = value

    @property
    def pretty_name(self) -> str:
        """The pretty name of the dataset.

        Returns:
            The pretty name of the dataset.

        Raises:
            ValueError:
                If the pretty name of the dataset is not set.
        """
        if self._pretty_name is None:
            raise ValueError("The pretty name of the dataset is not set!")
        return self._pretty_name

    @pretty_name.setter
    def pretty_name(self, value: str) -> None:
        """Set the pretty name of the dataset.

        Args:
            value:
                The new pretty name of the dataset.
        """
        self._pretty_name = value

    @property
    def source(self) -> str | dict[str, str]:
        """The source of the dataset.

        Returns:
            The source of the dataset.

        Raises:
            ValueError:
                If the source of the dataset is not set.
        """
        if self._source is None:
            raise ValueError("The source of the dataset is not set!")
        return self._source

    @source.setter
    def source(self, value: str | dict[str, str]) -> None:
        """Set the source of the dataset.

        Args:
            value:
                The new source of the dataset.
        """
        self._source = value

    @property
    def logging_string(self) -> str:
        """The string used to describe evaluation on the dataset in logging.

        Returns:
            The logging string.
        """
        truncated_str = (
            "truncated version of the "
            if isinstance(self.source, str) and self.source.endswith("-mini")
            else ""
        )

        logging_languages = list(deepcopy(self.languages))
        if len(self.languages) > 1:
            if (
                NORWEGIAN_BOKMÅL in self.languages
                and NORWEGIAN_NYNORSK in self.languages
                and NORWEGIAN in self.languages
            ):
                logging_languages.remove(NORWEGIAN_BOKMÅL)
                logging_languages.remove(NORWEGIAN_NYNORSK)
            elif (
                NORWEGIAN_BOKMÅL in self.languages
                or NORWEGIAN_NYNORSK in self.languages
            ) and NORWEGIAN in self.languages:
                logging_languages.remove(NORWEGIAN)
            if PORTUGUESE in self.languages and EUROPEAN_PORTUGUESE in self.languages:
                logging_languages.remove(EUROPEAN_PORTUGUESE)

        if len(logging_languages) > MAX_NUMBER_OF_LOGGING_LANGUAGES:
            languages_str = ""
        elif len(logging_languages) > 1:
            languages_str = (
                ", ".join([lang.name for lang in logging_languages[:-1]])
                + f" and {logging_languages[-1].name}"
                + " "
            )
        else:
            languages_str = logging_languages[0].name + " "

        task_str = self.task.name.replace("-", " ")
        dataset_name_str = (
            self.pretty_name or self.name.replace("-", " ").replace("_", " ").title()
        )
        return (
            f"the {truncated_str}{languages_str}{task_str} dataset {dataset_name_str}"
        )

    @property
    def main_language(self) -> Language | tuple[Language, Language]:
        """Get the main language of the dataset.

        Returns:
            The main language or languages of the dataset.

        Raises:
            InvalidBenchmark:
                If the dataset has no languages.
        """
        # Importing here to avoid circular imports
        from .tasks import TRANSLATION

        # Special case for datasets with multiple languages
        if self.task == TRANSLATION:
            return (self.languages[0], self.languages[1])

        match len(self.languages):
            case 0:
                raise InvalidBenchmark(
                    f"Dataset {self.name!r} must have at least one language."
                )
            case 1:
                return self.languages[0]
            case _:
                if ENGLISH in self.languages:
                    return ENGLISH
                elif NORWEGIAN in self.languages:
                    return NORWEGIAN
                elif PORTUGUESE in self.languages:
                    return PORTUGUESE
                else:
                    return self.languages[0]

    @property
    def id2label(self) -> "HashableDict":
        """The mapping from ID to label."""
        return HashableDict({idx: label for idx, label in enumerate(self.labels)})

    @property
    def label2id(self) -> "HashableDict":
        """The mapping from label to ID."""
        return HashableDict({label: i for i, label in enumerate(self.labels)})

    @property
    def num_labels(self) -> int:
        """The number of labels in the dataset."""
        return len(self.labels)

    def __hash__(self) -> int:
        """Return a hash of the dataset configuration."""
        return hash(self.name)

    def get_labels_str(self, labels: c.Sequence[str] | None = None) -> str:
        """Converts a set of labels to a natural string, in the specified language.

        If the task is NER, we separate using 'and' and use the mapped labels instead of
        the BIO NER labels.

        Args:
            labels (optional):
                The labels to convert to a natural string. If None, uses all the labels
                in the dataset. Defaults to None.

        Returns:
            The natural string representation of the labels in specified language.
        """
        if self.task.task_group == TaskGroup.TOKEN_CLASSIFICATION:
            sep_word = self.main_language.and_separator
        else:
            sep_word = self.main_language.or_separator

        if labels is None:
            labels = list()
            for english_label in self.labels:
                if english_label not in self.prompt_label_mapping:
                    continue
                label = self.prompt_label_mapping[english_label]
                if label not in labels:
                    labels.append(label)

        # Convert labels to single-quoted labels - and remove duplicates
        quoted_labels = [f"'{label}'" for label in labels]

        if not quoted_labels:
            return ""
        elif len(quoted_labels) == 1:
            return quoted_labels[0]
        elif len(quoted_labels) == 2:
            return f"{quoted_labels[0]} {sep_word} {quoted_labels[1]}"
        else:
            return f"{', '.join(quoted_labels[:-1])} {sep_word} {quoted_labels[-1]}"


@dataclass
class BenchmarkConfig:
    """General benchmarking configuration, across datasets and models.

    Attributes:
        datasets:
            The datasets to benchmark on.
        finetuning_batch_size:
            The batch size to use for finetuning.
        raise_errors:
            Whether to raise errors instead of skipping them.
        cache_dir:
            Directory to store cached models and datasets.
        api_key:
            The API key to use for a given inference API.
        api_base:
            The base URL for a given inference API. Only relevant if `model` refers to a
            model on an inference API.
        api_version:
            The version of the API to use. Only relevant if `model` refers to a model on
            an inference API.
        progress_bar:
            Whether to show a progress bar.
        save_results:
            Whether to save the benchmark results to 'euroeval_benchmark_results.json'.
        device:
            The device to use for benchmarking.
        trust_remote_code:
            Whether to trust remote code when loading models from the Hugging Face Hub.
        clear_model_cache:
            Whether to clear the model cache after benchmarking each model.
        evaluate_test_split:
            Whether to evaluate on the test split.
        few_shot:
            Whether to only evaluate the model using few-shot evaluation. Only relevant
            if the model is generative.
        num_iterations:
            The number of iterations each model should be evaluated for.
        gpu_memory_utilization:
            The GPU memory utilization to use for vLLM. A larger value will result in
            faster evaluation, but at the risk of running out of GPU memory. Only reduce
            this if you are running out of GPU memory. Only relevant if the model is
            generative.
        attention_backend:
            The attention backend to use for vLLM. Defaults to FLASHINFER. Only
            relevant if the model is generative.
        requires_safetensors:
            Whether to only allow models that use the safetensors format.
        generative_type:
            The type of generative model to benchmark. Only relevant if the model is
            generative.
        download_only:
            Whether to only download the models, metrics and datasets without
            evaluating.
        force:
            Whether to force the benchmark to run even if the results are already
            cached.
        verbose:
            Whether to print verbose output.
        debug:
            Whether to run the benchmark in debug mode.
        run_with_cli:
            Whether the benchmark is being run with the CLI.
    """

    datasets: c.Sequence[DatasetConfig]
    languages: c.Sequence[Language]
    finetuning_batch_size: int
    raise_errors: bool
    cache_dir: str
    api_key: str | None
    api_base: str | None
    api_version: str | None
    progress_bar: bool
    save_results: bool
    device: torch.device
    trust_remote_code: bool
    clear_model_cache: bool
    evaluate_test_split: bool
    few_shot: bool
    num_iterations: int
    gpu_memory_utilization: float
    attention_backend: t.Literal[
        *ATTENTION_BACKENDS  # pyrefly: ignore[invalid-literal]
    ]
    requires_safetensors: bool
    generative_type: GenerativeType | None
    download_only: bool
    force: bool
    verbose: bool
    debug: bool
    run_with_cli: bool

    @property
    def tasks(self) -> c.Sequence[Task]:
        """Get the tasks in the benchmark configuration."""
        return list({dataset_config.task for dataset_config in self.datasets})

    def __post_init__(self) -> None:
        """Post-initialisation checks."""
        # Set dummy API key if it has not been set and we're benchmarking a model on an
        # inference API
        if self.api_key is None and self.api_base is not None:
            self.api_key = "dummy"


class BenchmarkConfigParams(pydantic.BaseModel):
    """The parameters for the benchmark configuration."""

    model_config = pydantic.ConfigDict(
        protected_namespaces=(), arbitrary_types_allowed=True
    )

    task: str | Task | c.Sequence[str | Task] | None
    dataset: str | DatasetConfig | c.Sequence[str | DatasetConfig] | None
    progress_bar: bool
    save_results: bool
    language: str | c.Sequence[str]
    device: Device | None
    finetuning_batch_size: int
    raise_errors: bool
    cache_dir: str
    api_key: str | None
    api_base: str | None
    api_version: str | None
    trust_remote_code: bool
    clear_model_cache: bool
    evaluate_test_split: bool
    few_shot: bool
    num_iterations: int
    requires_safetensors: bool
    download_only: bool
    gpu_memory_utilization: float
    attention_backend: t.Literal[
        *ATTENTION_BACKENDS  # pyrefly: ignore[invalid-literal]
    ]
    generative_type: GenerativeType | None
    custom_datasets_file: Path
    force: bool
    verbose: bool
    debug: bool
    run_with_cli: bool


class BenchmarkResult(pydantic.BaseModel):
    """A benchmark result."""

    dataset: str
    task: str
    languages: c.Sequence[str]
    model: str
    results: ScoreDict
    num_model_parameters: int
    max_sequence_length: int
    vocabulary_size: int
    merge: bool
    generative: bool
    generative_type: str | None
    few_shot: bool | None
    validation_split: bool | None
    euroeval_version: str | None = get_package_version("euroeval")
    transformers_version: str | None = get_package_version("transformers")
    torch_version: str | None = get_package_version("torch")
    vllm_version: str | None = get_package_version("vllm")
    xgrammar_version: str | None = get_package_version("xgrammar")

    @classmethod
    def from_dict(cls, config: dict) -> "BenchmarkResult":
        """Create a benchmark result from a dictionary.

        Args:
            config:
                The configuration dictionary.

        Returns:
            The benchmark result.
        """
        # To be backwards compatible, we accept old results which changed the model
        # name with parameters rather than adding them as explicit parameters
        val_matches = re.search(r"\(.*val.*\)$", config["model"])
        few_shot_matches = re.search(r"\(.*few-shot.*\)$", config["model"])
        zero_shot_matches = re.search(r"\(.*zero-shot.*\)$", config["model"])
        config["model"] = re.sub(
            r"\(.*(few-shot|val).*\)$", "", config["model"]
        ).strip()

        if "merge" not in config:
            config["merge"] = False
        if "generative" not in config:
            config["generative"] = (
                few_shot_matches is not None or zero_shot_matches is not None
            )
        if "generative_type" not in config:
            config["generative_type"] = None
        if "few_shot" not in config:
            config["few_shot"] = zero_shot_matches is None
        if "validation_split" not in config:
            config["validation_split"] = val_matches is not None

        # Backwards compatibility
        if "dataset_languages" in config:
            config["languages"] = config.pop("dataset_languages")

        return cls(**config)

    def append_to_results(self, results_path: Path) -> None:
        """Append the benchmark result to the results file.

        Args:
            results_path:
                The path to the results file.
        """
        json_str = json.dumps(self.model_dump(), ensure_ascii=False)
        with results_path.open("a") as f:
            f.write("\n" + json_str)


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        model_id:
            The ID of the model.
        revision:
            The revision of the model.
        param:
            The parameter of the model, or None if the model has no parameters.
        task:
            The task that the model was trained on.
        languages:
            The languages of the model.
        inference_backend:
            The backend used to perform inference with the model.
        merge:
            Whether the model is a merged model.
        model_type:
            The type of the model (e.g., encoder, base decoder, instruction tuned).
        fresh:
            Whether the model is freshly initialised.
        model_cache_dir:
            The directory to cache the model in.
        adapter_base_model_id:
            The model ID of the base model if the model is an adapter model. Can be None
            if the model is not an adapter model.
        generation_config (optional):
            The generation configuration for generative models, if specified in the
            model repository. Defaults to no generation configuration.
    """

    model_id: str
    revision: str
    param: str | None
    task: str
    languages: c.Sequence[Language]
    inference_backend: "InferenceBackend"
    merge: bool
    model_type: ModelType
    fresh: bool
    model_cache_dir: str
    adapter_base_model_id: str | None
    generation_config: GenerationConfig | None = None

    def __hash__(self) -> int:
        """Return a hash of the model configuration."""
        return hash(self.model_id)


@dataclass
class PreparedModelInputs:
    """The inputs to a model.

    Attributes:
        texts:
            The texts to input to the model. Can be None if the input IDs and attention
            mask are provided instead.
        input_ids:
            The input IDs of the texts. Can be None if the texts are provided instead.
        attention_mask:
            The attention mask of the texts. Can be None if the texts are provided
            instead.
    """

    texts: c.Sequence[str] | None = None
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class GenerativeModelOutput:
    """The output of a generative model.

    Attributes:
        sequences:
            The generated sequences.
        predicted_labels (optional):
            The predicted labels from the `sequences` and sometimes also `scores`. Can
            be None if the labels have not been predicted yet. Defaults to None.
        scores (optional):
            The scores of the sequences. This is an array of shape (batch_size,
            num_tokens, num_logprobs, 2), where the last dimension contains the
            token and its logprob. Can be None if the scores are not available. Defaults
            to None.
        metadatas (optional):
            All the metadata fields for the samples, including ground truth labels (if
            applicable). Defaults to an empty list.
    """

    sequences: c.Sequence[str]
    predicted_labels: c.Sequence | None = None
    scores: c.Sequence[c.Sequence[c.Sequence[tuple[str, float]]]] | None = None
    metadatas: list["HashableDict | None"] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialisation."""
        if not self.metadatas:
            self.metadatas = [None] * len(self.sequences)


@dataclass
class SingleGenerativeModelOutput:
    """A single output of a generative model.

    Attributes:
        sequence:
            The generated sequence.
        predicted_label (optional):
            The predicted label from the `sequence` and sometimes also `scores`. Can be
            None if the label has not been predicted yet. Defaults to None.
        scores (optional):
            The scores of the sequence. This is an array of shape (num_tokens,
            num_logprobs, 2), where the last dimension contains the token and its
            logprob. Can be None if the scores are not available. Defaults to None.
        metadata (optional):
            The metadata fields for the sample, including ground truth labels (if
            applicable). Can be None if the metadata is not available. Defaults to None.
    """

    sequence: str
    predicted_label: str | None = None
    scores: c.Sequence[c.Sequence[tuple[str, float]]] | None = None
    metadata: "HashableDict | None" = None


@dataclass
class HFModelInfo:
    """Information about a Hugging Face model.

    Attributes:
        pipeline_tag:
            The pipeline tag of the model.
        tags:
            The other tags of the model.
        adapter_base_model_id:
            The model ID of the base model if the model is an adapter model. Can be None
            if the model is not an adapter model.
    """

    pipeline_tag: str
    tags: c.Sequence[str]
    adapter_base_model_id: str | None


@dataclass
class ModelIdComponents:
    """A model ID split into its components.

    Attributes:
        model_id:
            The main model ID without revision or parameters.
        revision:
            The revision of the model, if any.
        param:
            The parameter of the model, if any.
    """

    model_id: str
    revision: str
    param: str | None


class HashableDict(dict):
    """A hashable dictionary."""

    def __hash__(self) -> int:  # type: ignore[override]
        """Return the hash of the dictionary."""
        return hash(frozenset(self.items()))
