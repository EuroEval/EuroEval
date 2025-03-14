"""Data models used in EuroEval."""

import collections.abc as c
import importlib.metadata
import json
import pathlib
import re
import typing as t
from dataclasses import dataclass, field

import pydantic
import torch

from .enums import Device, InferenceBackend, ModelType, TaskGroup
from .types import ScoreDict


@dataclass
class MetricConfig:
    """Configuration for a metric.

    Attributes:
        name:
            The name of the metric.
        pretty_name:
            A longer prettier name for the metric, which allows cases and spaces. Used
            for logging.
        huggingface_id:
            The Hugging Face ID of the metric.
        results_key:
            The name of the key used to extract the metric scores from the results
            dictionary.
        compute_kwargs:
            Keyword arguments to pass to the metric's compute function. Defaults to
            an empty dictionary.
        postprocessing_fn:
            A function to apply to the metric scores after they are computed, taking
            the score to the postprocessed score along with its string representation.
            Defaults to x -> (100 * x, f"{x:.2%}").
    """

    name: str
    pretty_name: str
    huggingface_id: str
    results_key: str
    compute_kwargs: dict[str, t.Any] = field(default_factory=dict)
    postprocessing_fn: c.Callable[[float], tuple[float, str]] = field(
        default_factory=lambda: lambda raw_score: (100 * raw_score, f"{raw_score:.2%}")
    )

    def __hash__(self) -> int:
        """Return a hash of the metric configuration."""
        return hash(self.name)


@dataclass
class Task:
    """A dataset task.

    Attributes:
        name:
            The name of the task.
        task_group:
            The task group of the task.
        metrics:
            The metrics used to evaluate the task.
    """

    name: str
    task_group: TaskGroup
    metrics: list[MetricConfig]

    def __hash__(self) -> int:
        """Return a hash of the task."""
        return hash(self.name)


@dataclass
class Language:
    """A benchmarkable language.

    Attributes:
        code:
            The ISO 639-1 language code of the language.
        name:
            The name of the language.
    """

    code: str
    name: str

    def __hash__(self) -> int:
        """Return a hash of the language."""
        return hash(self.code)


@dataclass
class BenchmarkConfig:
    """General benchmarking configuration, across datasets and models.

    Attributes:
        model_languages:
            The languages of the models to benchmark.
        dataset_languages:
            The languages of the datasets in the benchmark.
        tasks:
            The tasks benchmark the model(s) on.
        datasets:
            The datasets to benchmark on.
        batch_size:
            The batch size to use.
        raise_errors:
            Whether to raise errors instead of skipping them.
        cache_dir:
            Directory to store cached models and datasets.
        api_key:
            The API key to use for a given inference API.
        force:
            Whether to force the benchmark to run even if the results are already
            cached.
        progress_bar:
            Whether to show a progress bar.
        save_results:
            Whether to save the benchmark results to 'euroeval_benchmark_results.json'.
        device:
            The device to use for benchmarking.
        verbose:
            Whether to print verbose output.
        trust_remote_code:
            Whether to trust remote code when loading models from the Hugging Face Hub.
        use_flash_attention:
            Whether to use Flash Attention. If None then this will be used for
            generative models.
        clear_model_cache:
            Whether to clear the model cache after benchmarking each model.
        evaluate_test_split:
            Whether to evaluate on the test split.
        few_shot:
            Whether to only evaluate the model using few-shot evaluation. Only relevant
            if the model is generative.
        num_iterations:
            The number of iterations each model should be evaluated for.
        api_base:
            The base URL for a given inference API. Only relevant if `model` refers to a
            model on an inference API.
        api_version:
            The version of the API to use. Only relevant if `model` refers to a model on
            an inference API.
        debug:
            Whether to run the benchmark in debug mode.
        run_with_cli:
            Whether the benchmark is being run with the CLI.
        only_allow_safetensors:
            Whether to only allow models that use the safetensors format.
    """

    model_languages: list[Language]
    dataset_languages: list[Language]
    tasks: list[Task]
    datasets: list[str]
    batch_size: int
    raise_errors: bool
    cache_dir: str
    api_key: str | None
    force: bool
    progress_bar: bool
    save_results: bool
    device: torch.device
    verbose: bool
    trust_remote_code: bool
    use_flash_attention: bool | None
    clear_model_cache: bool
    evaluate_test_split: bool
    few_shot: bool
    num_iterations: int
    api_base: str | None
    api_version: str | None
    debug: bool
    run_with_cli: bool
    only_allow_safetensors: bool


class BenchmarkConfigParams(pydantic.BaseModel):
    """The parameters for the benchmark configuration."""

    model_config = pydantic.ConfigDict(protected_namespaces=())

    progress_bar: bool
    save_results: bool
    task: str | list[str] | None
    dataset: str | list[str] | None
    language: str | list[str]
    model_language: str | list[str] | None
    dataset_language: str | list[str] | None
    device: Device | None
    batch_size: int
    raise_errors: bool
    cache_dir: str
    api_key: str | None
    force: bool
    verbose: bool
    trust_remote_code: bool
    use_flash_attention: bool | None
    clear_model_cache: bool
    evaluate_test_split: bool
    few_shot: bool
    num_iterations: int
    api_base: str | None
    api_version: str | None
    debug: bool
    run_with_cli: bool
    only_allow_safetensors: bool


class BenchmarkResult(pydantic.BaseModel):
    """A benchmark result."""

    dataset: str
    task: str
    dataset_languages: list[str]
    model: str
    results: ScoreDict
    num_model_parameters: int
    max_sequence_length: int
    vocabulary_size: int
    merge: bool
    generative: bool
    generative_type: str | None
    few_shot: bool
    validation_split: bool
    euroeval_version: str = importlib.metadata.version("euroeval")

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

        return cls(**config)

    def append_to_results(self, results_path: pathlib.Path) -> None:
        """Append the benchmark result to the results file.

        Args:
            results_path:
                The path to the results file.
        """
        json_str = json.dumps(self.model_dump())
        with results_path.open("a") as f:
            f.write("\n" + json_str)


@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        name:
            The name of the dataset. Must be lower case with no spaces.
        pretty_name:
            A longer prettier name for the dataset, which allows cases and spaces. Used
            for logging.
        huggingface_id:
            The Hugging Face ID of the dataset.
        task:
            The task of the dataset.
        languages:
            The ISO 639-1 language codes of the entries in the dataset.
        id2label:
            The mapping from ID to label.
        label2id:
            The mapping from label to ID.
        num_labels:
            The number of labels in the dataset.
        prompt_template:
            The template for the prompt to use when benchmarking the dataset using
            few-shot evaluation.
        max_generated_tokens:
            The maximum number of tokens to generate when benchmarking the dataset
            using few-shot evaluation.
        prompt_prefix:
            The prefix to use in the few-shot prompt.
        num_few_shot_examples:
            The number of examples to use when benchmarking the dataset using few-shot
            evaluation. For a classification task, these will be drawn evenly from
            each label.
        instruction_prompt:
            The prompt to use when benchmarking the dataset using instruction-based
            evaluation.
        labels (optional):
            The labels in the dataset. Defaults to an empty list.
        prompt_label_mapping (optional):
            A mapping from the labels to another phrase which is used as a substitute
            for the label in few-shot evaluation. Defaults to an empty dictionary.
        unofficial (optional):
            Whether the dataset is unofficial. Defaults to False.
    """

    name: str
    pretty_name: str
    huggingface_id: str
    task: Task
    languages: list[Language]
    prompt_template: str
    max_generated_tokens: int
    prompt_prefix: str
    num_few_shot_examples: int
    instruction_prompt: str
    labels: list[str] = field(default_factory=list)
    prompt_label_mapping: dict[str, str] = field(default_factory=dict)
    unofficial: bool = False

    @property
    def id2label(self) -> dict[int, str]:
        """The mapping from ID to label."""
        return {idx: label for idx, label in enumerate(self.labels)}

    @property
    def label2id(self) -> dict[str, int]:
        """The mapping from label to ID."""
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def num_labels(self) -> int:
        """The number of labels in the dataset."""
        return len(self.labels)

    def __hash__(self) -> int:
        """Return a hash of the dataset configuration."""
        return hash(self.name)


@dataclass
class ModelConfig:
    """Configuration for a model.

    Attributes:
        model_id:
            The ID of the model.
        revision:
            The revision of the model.
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
    """

    model_id: str
    revision: str
    task: str
    languages: list[Language]
    inference_backend: InferenceBackend
    merge: bool
    model_type: ModelType
    fresh: bool
    model_cache_dir: str
    adapter_base_model_id: str | None

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

    texts: list[str] | None = None
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class GenerativeModelOutput:
    """The output of a generative model.

    Attributes:
        sequences:
            The generated sequences.
        scores:
            The scores of the sequences. This is an array of shape (batch_size,
            num_tokens, num_logprobs, 2), where the last dimension contains the
            token and its logprob. Can be None if the scores are not available.
    """

    sequences: list[str]
    scores: list[list[list[tuple[str, float]]]] | None = None


@dataclass
class SingleGenerativeModelOutput:
    """A single output of a generative model.

    Attributes:
        sequence:
            The generated sequence.
        scores:
            The scores of the sequence. This is an array of shape (num_tokens,
            num_logprobs, 2), where the last dimension contains the token and its
            logprob. Can be None if the scores are not available.
    """

    sequence: str
    scores: list[list[tuple[str, float]]] | None = None


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
    tags: list[str]
    adapter_base_model_id: str | None
