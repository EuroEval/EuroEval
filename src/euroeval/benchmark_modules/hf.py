"""Encoder models from the Hugging Face Hub."""

import collections.abc as c
import importlib
import logging
import re
import typing as t
from functools import cached_property, partial
from json import JSONDecodeError
from pathlib import Path
from time import sleep

import torch
from datasets import DatasetDict
from huggingface_hub import HfApi
from huggingface_hub import whoami as hf_whoami
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalTokenNotFoundError,
    RepositoryNotFoundError,
)
from huggingface_hub.hf_api import ModelInfo as HfApiModelInfo
from peft import PeftConfig
from requests.exceptions import RequestException
from torch import nn
from transformers import PretrainedConfig
from transformers.data.data_collator import (
    DataCollatorForMultipleChoice,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
)
from transformers.modelcard import TASK_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from urllib3.exceptions import RequestError

from ..caching_utils import cache_arguments
from ..constants import (
    DUMMY_FILL_VALUE,
    GENERATIVE_PIPELINE_TAGS,
    LOCAL_MODELS_REQUIRED_FILES,
    MAX_CONTEXT_LENGTH,
)
from ..data_models import HashableDict, HFModelInfo, ModelConfig
from ..enums import (
    BatchingPreference,
    DataType,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import (
    InvalidBenchmark,
    InvalidModel,
    NeedsAdditionalArgument,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..generation_utils import raise_if_wrong_params
from ..logging_utils import block_terminal_output, log, log_once
from ..model_cache import create_model_cache_dir
from ..safetensors_utils import get_num_params_from_safetensors_metadata
from ..string_utils import split_model_id
from ..task_group_utils import (
    multiple_choice_classification,
    question_answering,
    token_classification,
)
from ..tokenisation_utils import get_bos_token, get_eos_token
from ..types import Tokeniser
from ..utils import get_hf_token, internet_connection_available
from .base import BenchmarkModule, _build_model_config_helper, _lookup_model_info

try:
    from transformers.tokenization_mistral_common import MistralCommonTokenizer
except ImportError:
    from transformers.tokenization_mistral_common import MistralCommonBackend as MCB

    MistralCommonTokenizer = MCB

if t.TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_base import BatchEncoding

    from ..data_models import BenchmarkConfig, DatasetConfig, Task
    from ..types import ExtractLabelsFunction


class HuggingFaceEncoderModel(BenchmarkModule):
    """An encoder model from the Hugging Face Hub."""

    fresh_model = False
    batching_preference = BatchingPreference.NO_PREFERENCE
    high_priority = True
    allowed_params = {re.compile(r".*"): ["slow-tokenizer"]}

    def __init__(
        self,
        model_config: "ModelConfig",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
        log_metadata: bool = True,
        dtype_override: "DataType | None" = None,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
            log_metadata (optional):
                Whether to log the model metadata. Defaults to True.
            dtype_override (optional):
                An explicit data type to load the model weights in, taking precedence
                over the hardware-derived default. Used by the finetuning NaN-retry to
                force a full fp32 reload. Defaults to None.
        """
        raise_if_wrong_params(
            model_config=model_config, allowed_params=self.allowed_params
        )

        # This is already set when calling `super().__init__`, but we need it to get
        # the correct value from `self.model_max_length`, so we set it here as well.
        self.benchmark_config = benchmark_config

        model, tokeniser = load_model_and_tokeniser(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            dtype_override=dtype_override,
        )
        self._model: "PreTrainedModel" = model
        self._tokeniser: "PreTrainedTokenizer | MistralCommonTokenizer" = tokeniser

        self._model, self._tokeniser = align_model_and_tokeniser(
            model=self._model,
            tokeniser=self._tokeniser,
            model_max_length=self.model_max_length,
            raise_errors=benchmark_config.raise_errors,
            is_multiple_choice=(
                dataset_config.task.task_group
                == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ),
        )

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )

    def _numericalise_labels(self, examples: dict) -> dict:
        """Numericalise labels in examples.

        Args:
            examples:
                The examples to numericalise.

        Returns:
            The numericalised examples.

        Raises:
            InvalidBenchmark:
                If a label is not found in the label2id dictionary.
        """
        if "label" in examples:
            label2id: dict[str, int] | None = self._model.config.label2id  # ty: ignore[invalid-assignment]
            if label2id is not None:
                new_labels: list[int] = []
                for lbl in examples["label"]:
                    lbl_str = str(lbl).lower()
                    if lbl_str not in label2id:
                        raise InvalidBenchmark(
                            f"One of the labels in the dataset, "
                            f"{lbl_str}, does not occur in the "
                            f"label2id dictionary {label2id}."
                        )
                    new_labels.append(label2id[lbl_str])
                examples["label"] = new_labels
        return examples

    def _tokenise(self, examples: dict) -> "BatchEncoding":
        """Tokenise examples.

        Args:
            examples:
                The examples to tokenise.

        Returns:
            The tokenised examples.
        """
        return self._tokeniser(text=examples["text"], truncation=True, padding=True)

    @property
    def data_collator(self) -> c.Callable[[list[dict[str, t.Any]]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        assert isinstance(self._tokeniser, PreTrainedTokenizerBase), (
            "The data collator property is only supported for models with a "
            "Hugging Face tokeniser."
        )
        match self.dataset_config.task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.TEXT_TO_TEXT
                | TaskGroup.QUESTION_ANSWERING
            ):
                return DataCollatorWithPadding(
                    tokenizer=self._tokeniser, padding="longest"
                )
            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                return DataCollatorForMultipleChoice(
                    tokenizer=self._tokeniser, padding="longest"
                )
            case TaskGroup.TOKEN_CLASSIFICATION:
                return DataCollatorForTokenClassification(
                    tokenizer=self._tokeniser, label_pad_token_id=-100
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    def extract_labels_from_generation(self) -> "ExtractLabelsFunction":
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        raise NotImplementedError(
            "The `extract_labels_from_generation` property has not been implemented "
            "for Hugging Face Encoder models."
        )

    @property
    def generative_type(self) -> GenerativeType | None:
        """Generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        return None

    @classmethod
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

        Raises:
            InvalidModel:
                If the model could not be found.
        """
        resolved_model_id, revision, model_info = _lookup_model_info(
            model_id=model_id, benchmark_config=benchmark_config
        )
        if model_info is None:
            raise InvalidModel(f"The model {model_id!r} could not be found.")

        model_id_components = split_model_id(model_id=model_id)

        return _build_model_config_helper(
            model_id=resolved_model_id,
            revision=revision,
            param=model_id_components.param,
            task=model_info.pipeline_tag,
            model_info=model_info,
            benchmark_config=benchmark_config,
            inference_backend=InferenceBackend.TRANSFORMERS,
            model_type=ModelType.ENCODER,
            adapter_base_model_id=None,
        )

    @classmethod
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
        _, _, model_info = _lookup_model_info(
            model_id=model_id, benchmark_config=benchmark_config
        )
        return (
            model_info is not None
            and model_info.pipeline_tag not in GENERATIVE_PIPELINE_TAGS
        )

    @cached_property
    def model_max_length(self) -> int:
        """The maximum context length of the model.

        Returns:
            The maximum context length of the model.
        """
        if self.benchmark_config.max_context_length is not None:
            return self.benchmark_config.max_context_length
        all_max_lengths: list[int] = list()

        if hasattr(
            self._tokeniser, "model_max_length"
        ) and self._tokeniser.model_max_length < int(1e30):
            all_max_lengths.append(self._tokeniser.model_max_length)

        if hasattr(self._tokeniser, "max_model_input_sizes"):
            all_max_lengths.extend(
                [
                    size
                    for size in self._tokeniser.max_model_input_sizes.values()
                    if size is not None
                ]
            )

        candidate_config_max_lengths = [
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
            "n_positions",
        ]
        for candidate_config_max_length in candidate_config_max_lengths:
            if (
                hasattr(self._model.config, candidate_config_max_length)
                and (value := getattr(self._model.config, candidate_config_max_length))
                is not None
            ):
                all_max_lengths.append(value)

        # To avoid models having artificially low max lengths, we remove any max lengths
        # that are less than 128
        all_max_lengths = [
            max_length for max_length in all_max_lengths if max_length >= 128
        ]

        if len(list(all_max_lengths)) > 0:
            model_max_length = min(list(all_max_lengths))
        else:
            model_max_length = -1

        return model_max_length

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        num_params_or_none = get_num_params_from_safetensors_metadata(
            model_id=(
                self.model_config.adapter_base_model_id or self.model_config.model_id
            ),
            revision=self.model_config.revision,
            api_key=self.benchmark_config.api_key,
        )
        if num_params_or_none is not None:
            return num_params_or_none

        num_params = -1
        if (
            hasattr(self._model.config, "num_params")
            and self._model.config.num_params is not None
        ):
            num_params = int(self._model.config.num_params)  # ty: ignore[invalid-argument-type]
        elif hasattr(self._model, "parameters"):
            num_params = sum(p.numel() for p in self._model.parameters())
        else:
            log_once(
                "The number of parameters could not be determined for the model "
                f"{self.model_config.model_id}, neither from the safetensors metadata "
                "nor from the model configuration.",
                level=logging.WARNING,
            )
        return num_params

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
        match task.task_group:
            case TaskGroup.SEQUENCE_CLASSIFICATION:
                return self._prepare_sequence_classification(dataset=dataset)
            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                return self._prepare_multiple_choice(dataset=dataset)
            case TaskGroup.TEXT_TO_TEXT:
                return self._prepare_text_to_text(dataset=dataset)
            case TaskGroup.TOKEN_CLASSIFICATION:
                return self._prepare_token_classification(dataset=dataset)
            case TaskGroup.QUESTION_ANSWERING:
                return self._prepare_question_answering(dataset=dataset)
            case _:
                raise NotImplementedError(f"Unsupported task group: {task.task_group}.")

    def _prepare_multiple_choice(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for multiple choice classification.

        Args:
            dataset:
                The dataset to prepare.

        Returns:
            The prepared dataset.
        """
        return DatasetDict(
            {
                split_name: split.map(
                    partial(
                        multiple_choice_classification.prepare_examples,
                        tokeniser=self._tokeniser,
                        num_choices=self.dataset_config.num_labels,
                    ),
                    batched=True,
                    batch_size=10,
                    remove_columns=split.column_names,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )
                for split_name, split in dataset.items()
            }
        )

    def _prepare_question_answering(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for question answering.

        Args:
            dataset:
                The dataset to prepare.

        Returns:
            The prepared dataset.
        """
        data_dict = dict()
        test_columns = dataset["test"].column_names if "test" in dataset else []

        for split_name in ["train", "val", "test"]:
            if split_name not in dataset:
                continue

            split = dataset[split_name]
            if split_name == "test":
                prep_func = question_answering.prepare_test_examples
            else:
                prep_func = question_answering.prepare_train_examples

            data_dict[split_name] = split.map(
                partial(prep_func, tokeniser=self._tokeniser),
                batched=True,
                batch_size=10,
                remove_columns=test_columns,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        result: DatasetDict = DatasetDict(data_dict)

        # Restore columns hidden by Trainer (id, offset_mapping) for post-processing
        for split_name, split in result.items():
            result[split_name].set_format(
                type=split.format["type"], columns=list(split.features.keys())
            )

        return result

    def _prepare_sequence_classification(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for sequence classification.

        Args:
            dataset:
                The dataset to prepare.

        Returns:
            The prepared dataset.
        """
        return dataset.map(
            self._numericalise_labels, batched=True, load_from_cache_file=False
        ).map(self._tokenise, batched=True, load_from_cache_file=False)

    def _prepare_text_to_text(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for text-to-text tasks.

        Args:
            dataset:
                The dataset to prepare.

        Returns:
            The prepared dataset.
        """
        return dataset.map(
            self._tokenise,
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    def _prepare_token_classification(self, dataset: DatasetDict) -> DatasetDict:
        """Prepare dataset for token classification.

        Args:
            dataset:
                The dataset to prepare.

        Returns:
            The prepared dataset.
        """
        return dataset.map(
            partial(
                token_classification.tokenize_and_align_labels,
                tokeniser=self._tokeniser,
                label2id=self._model.config.label2id,  # ty: ignore[invalid-argument-type]
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        match self.dataset_config.task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.TEXT_TO_TEXT
                | TaskGroup.TOKEN_CLASSIFICATION
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                return Trainer
            case TaskGroup.QUESTION_ANSWERING:
                return question_answering.QuestionAnsweringTrainer
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        if self.benchmark_config.vocabulary_size is not None:
            return self.benchmark_config.vocabulary_size
        vocab_size = -1
        if (
            hasattr(self._model.config, "vocab_size")
            and self._model.config.vocab_size is not None
        ):
            vocab_size = int(self._model.config.vocab_size)  # ty: ignore[invalid-argument-type]
        elif (
            hasattr(self._tokeniser, "vocab_size")
            and self._tokeniser.vocab_size is not None
        ):
            vocab_size = self._tokeniser.vocab_size
        return vocab_size


def align_model_and_tokeniser(
    model: "PreTrainedModel",
    tokeniser: Tokeniser,
    model_max_length: int,
    raise_errors: bool = False,
    is_multiple_choice: bool = False,
) -> tuple["PreTrainedModel", Tokeniser]:
    """Aligns the model and the tokeniser.

    Args:
        model:
            The model to fix.
        tokeniser:
            The tokeniser to fix.
        model_max_length:
            The maximum length of the model.
        raise_errors (optional):
            Whether to raise errors instead of trying to fix them silently.
            Defaults to False.
        is_multiple_choice (optional):
            Whether the model is being evaluated on a multiple-choice task, in which
            case it expects a 3-D dummy input when probing the maximum length.
            Defaults to False.

    Returns:
        The fixed model and tokeniser.
    """
    model_max_length = min(model_max_length, MAX_CONTEXT_LENGTH)
    tokeniser.model_max_length = model_max_length if model_max_length > 0 else 512

    # Test on CPU to avoid GPU memory issues
    model_device = model.device
    model.to(torch.device("cpu"))  # ty: ignore[invalid-argument-type]

    initial_max_length = tokeniser.model_max_length
    valid_max_length = _find_valid_model_max_length(
        model=model,
        tokeniser=tokeniser,
        initial_max_length=initial_max_length,
        is_multiple_choice=is_multiple_choice,
    )
    tokeniser.model_max_length = valid_max_length

    model.to(model_device)  # ty: ignore[invalid-argument-type]

    _adjust_vocab_size(model=model, tokeniser=tokeniser, raise_errors=raise_errors)
    _adjust_vocab_size(model=model, tokeniser=tokeniser, raise_errors=raise_errors)

    _set_bos_token(tokeniser=tokeniser)

    return model, tokeniser


@cache_arguments()
def _adjust_vocab_size(
    model: "PreTrainedModel", tokeniser: Tokeniser, raise_errors: bool
) -> None:
    """Adjust model vocab size if tokeniser is larger.

    Args:
        model:
            The model to potentially resize.
        tokeniser:
            The tokeniser.
        raise_errors:
            Whether to raise errors instead of auto-adjusting.

    Raises:
        InvalidModel:
            If vocab size mismatch and raise_errors is True.
    """
    if not hasattr(model.config, "vocab_size"):
        return

    if model.config.vocab_size >= len(tokeniser):
        return

    if raise_errors:
        raise InvalidModel(
            "The vocab size of the tokeniser is larger than the vocab size of "
            "the model. As the --raise-errors option was specified, the "
            "embeddings of the model will not be automatically adjusted."
        )

    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(new_num_tokens=tokeniser.vocab_size + 1)


@cache_arguments()
def _find_valid_model_max_length(
    model: "PreTrainedModel",
    tokeniser: Tokeniser,
    initial_max_length: int,
    is_multiple_choice: bool,
) -> int:
    """Find the maximum valid sequence length for the model.

    Args:
        model:
            The model to test.
        tokeniser:
            The tokeniser.
        initial_max_length:
            The initial maximum length to test.
        is_multiple_choice:
            Whether this is a multiple-choice model.

    Returns:
        The valid maximum length.

    Raises:
        ValueError:
            If an unexpected error occurs during inference.
    """
    for max_length in range(initial_max_length, 0, -1):
        tokeniser.model_max_length = max_length
        dummy_inputs = torch.full(
            size=(1, 2, max_length) if is_multiple_choice else (1, max_length),
            fill_value=DUMMY_FILL_VALUE,
            dtype=torch.long,
            device=model.device,
        )
        with torch.inference_mode():
            try:
                model(dummy_inputs, attention_mask=torch.ones_like(dummy_inputs))
                return max_length
            except IndexError:
                continue
            except ValueError as e:
                if "cpu tensor" in str(e):
                    return max_length
                raise
    return 1


def _set_bos_token(tokeniser: Tokeniser) -> None:
    """Set BOS token from EOS token if BOS is not set.

    Args:
        tokeniser:
            The tokeniser to update.
    """
    if tokeniser.bos_token is None and tokeniser.eos_token is not None:
        tokeniser.bos_token = tokeniser.eos_token
        tokeniser.bos_token_id = tokeniser.eos_token_id


def load_model_and_tokeniser(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    dtype_override: "DataType | None" = None,
) -> tuple["PreTrainedModel", Tokeniser]:
    """Load the model and tokeniser.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration
        dtype_override (optional):
            An explicit data type to load the model weights in, taking precedence
            over the hardware-derived default. Used by the finetuning NaN-retry to
            force a full fp32 reload. Defaults to None.

    Returns:
        A pair (model, tokeniser), with the loaded model and tokeniser

    Raises:
        InvalidBenchmark:
            If the model could not be loaded for this particular dataset.
    """
    block_terminal_output()

    model_id = model_config.model_id
    task_group = dataset_config.task.task_group

    id2label = dataset_config.id2label
    config = load_hf_model_config(
        model_id=model_id,
        num_labels=len(id2label),
        id2label=HashableDict(id2label),
        label2id=HashableDict({label: idx for idx, label in id2label.items()}),
        revision=model_config.revision,
        model_cache_dir=model_config.model_cache_dir,
        api_key=benchmark_config.api_key,
        trust_remote_code=benchmark_config.trust_remote_code,
        run_with_cli=benchmark_config.run_with_cli,
    )

    # If the model is a DeBERTaV2 model then ensure `pooler_hidden_size` matches
    if config.model_type == "deberta-v2":
        config.pooler_hidden_size = config.hidden_size

    model_kwargs: dict[str, t.Any] = dict(
        config=config,
        ignore_mismatched_sizes=False,
        revision=model_config.revision,
        token=get_hf_token(api_key=benchmark_config.api_key),
        cache_dir=model_config.model_cache_dir,
        trust_remote_code=benchmark_config.trust_remote_code,
        dtype=get_dtype(
            device=benchmark_config.device,
            dtype_is_set=config.to_dict().get("dtype") is not None,
            bf16_available=(
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            dtype_override=dtype_override,
        ),
    )

    model_cls_or_none: t.Type[PreTrainedModel] | None = t.cast(
        "t.Type[PreTrainedModel] | None",
        get_class_by_name(
            class_name=task_group_to_class_name(task_group=task_group),
            module_name="transformers",
        ),
    )

    if not model_cls_or_none:
        raise InvalidBenchmark(
            f"The task group {task_group.value!r} does not correspond to a "
            "Hugging Face AutoModel type (such as "
            "`AutoModelForSequenceClassification`)."
        )

    model_or_tuple = _load_model_from_pretrained(
        model_cls=model_cls_or_none,
        model_id=model_config.model_id,
        model_kwargs=model_kwargs,
        task_group=task_group,
    )

    if isinstance(model_or_tuple, tuple):
        model = t.cast(PreTrainedModel, model_or_tuple[0])
    else:
        model = t.cast(PreTrainedModel, model_or_tuple)

    assert model is not None, "The model should not be None."
    model = t.cast("PreTrainedModel", model)  # ty: ignore[redundant-cast]

    model.eval()
    model.to(benchmark_config.device)  # ty: ignore[invalid-argument-type]

    if (
        isinstance(model, PreTrainedModel)
        and task_group == TaskGroup.QUESTION_ANSWERING
    ):
        model = setup_model_for_question_answering(model=model)

    tokeniser = load_tokeniser(
        model=model,
        model_id=model_id,
        trust_remote_code=benchmark_config.trust_remote_code,
        model_config=model_config,
    )

    return model, tokeniser


def _load_model_from_pretrained(
    model_cls: t.Type[PreTrainedModel],
    model_id: str,
    model_kwargs: dict[str, t.Any],
    task_group: TaskGroup,
) -> PreTrainedModel | tuple[PreTrainedModel, ...]:
    """Load a model from pretrained with error handling.

    Args:
        model_cls:
            The model class to load.
        model_id:
            The model ID.
        model_kwargs:
            Keyword arguments for loading the model.
        task_group:
            The task group for error messages.

    Returns:
        The loaded model or tuple of models.

    Raises:
        InvalidModel:
            If the model could not be loaded.
        InvalidBenchmark:
            If the model architecture doesn't support the task.
    """
    for _ in range(num_attempts := 5):
        try:
            return model_cls.from_pretrained(
                pretrained_model_name_or_path=model_id, **model_kwargs
            )
        except (KeyError, RuntimeError) as e:
            if not model_kwargs.get("ignore_mismatched_sizes", False):
                log(
                    f"{type(e).__name__} occurred during the loading "
                    f"of the {model_id!r} model. Retrying with "
                    "`ignore_mismatched_sizes` set to True.",
                    level=logging.DEBUG,
                )
                model_kwargs["ignore_mismatched_sizes"] = True
                continue
            raise InvalidModel(str(e)) from e
        except (TimeoutError, RequestError):
            log(
                f"Couldn't load the model {model_id!r}. Retrying.",
                level=logging.WARNING,
            )
            sleep(5)
            continue
        except (OSError, ValueError) as e:
            error_str = str(e)
            if "checkpoint seems to be incorrect" in error_str:
                raise InvalidModel(
                    f"The model {model_id!r} has an incorrect checkpoint."
                ) from e
            if "trust_remote_code" in error_str:
                raise InvalidModel(
                    f"Loading the model {model_id!r} needs to trust remote code. "
                    "If you trust the suppliers of this model, then you can enable "
                    "this by setting the `--trust-remote-code` flag."
                ) from e
            if (
                "Unrecognized configuration class" in error_str
                and "AutoModelFor" in error_str
            ):
                raise InvalidBenchmark(
                    f"The model {model_id!r} does not support the "
                    f"task group {task_group.value!r} as its architecture is not "
                    f"compatible with the required HuggingFace model class. "
                    f"Error: {e}"
                ) from e
            raise InvalidModel(
                f"The model {model_id!r} could not be loaded. The error was {e!r}."
            ) from e
    raise InvalidModel(
        f"Could not load the model {model_id!r} after {num_attempts} attempts."
    )


def get_class_by_name(
    class_name: str | c.Sequence[str], module_name: str
) -> t.Type | None:
    """Get a class by its name.

    Args:
        class_name:
            The name of the class, written in kebab-case. The corresponding class name
            must be the same, but written in PascalCase, and lying in a module with the
            same name, but written in snake_case. If a list of strings is passed, the
            first class that is found is returned.
        module_name:
            The name of the module where the class is located.

    Returns:
        The class. If the class is not found, None is returned.
    """
    if isinstance(class_name, str):
        class_name = [class_name]

    error_messages = list()
    for name in class_name:
        try:
            module = importlib.import_module(name=module_name)
            class_: t.Type = getattr(module, name)
            return class_
        except (ModuleNotFoundError, AttributeError) as e:
            error_messages.append(str(e))

    if error_messages:
        errors = "\n- " + "\n- ".join(error_messages)
        log(
            f"Could not find the class with the name(s) {', '.join(class_name)}. The "
            f"following error messages were raised: {errors}",
            level=logging.DEBUG,
        )

    return None


@cache_arguments()
def get_dtype(
    device: torch.device,
    dtype_is_set: bool,
    bf16_available: bool,
    dtype_override: "DataType | None" = None,
) -> str | torch.dtype:
    """Get the torch dtype, used for loading the model.

    Args:
        device:
            The device to use.
        dtype_is_set:
            Whether the data type is set in the model configuration.
        bf16_available:
            Whether bfloat16 is available.
        dtype_override (optional):
            An explicit data type to load the model weights in, taking precedence
            over both the model configuration and the hardware-derived default. Used
            by the finetuning NaN-retry, which reloads the model in full fp32 after
            detecting NaN values under mixed precision; without honouring the
            override the model would be reloaded in the same (NaN-producing) dtype.
            Defaults to None.

    Returns:
        The dtype.
    """
    if dtype_override is not None:
        return {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }[dtype_override]

    using_cuda = device == torch.device("cuda")
    if using_cuda and dtype_is_set:
        return "auto"
    elif using_cuda and bf16_available:
        return torch.bfloat16
    elif using_cuda:
        return torch.float16
    return torch.float32


@cache_arguments("model_id", "revision", "num_labels", "id2label", "label2id")
def load_hf_model_config(
    model_id: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    revision: str,
    model_cache_dir: str | None,
    api_key: str | None,
    trust_remote_code: bool,
    run_with_cli: bool,
) -> "PretrainedConfig":
    """Load the Hugging Face model configuration.

    Args:
        model_id:
            The Hugging Face model ID.
        num_labels:
            The number of labels in the dataset.
        id2label:
            The mapping from label IDs to labels.
        label2id:
            The mapping from labels to label IDs.
        revision:
            The revision of the model.
        model_cache_dir:
            The directory to cache the model in.
        api_key:
            The Hugging Face API key.
        trust_remote_code:
            Whether to trust remote code.
        run_with_cli:
            Whether the script is being run with the CLI.

    Returns:
        The Hugging Face model configuration.

    Raises:
        InvalidModel:
            If the model configuration could not be loaded.
    """
    for _ in range(num_attempts := 5):
        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_id,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                revision=revision,
                token=get_hf_token(api_key=api_key),
                trust_remote_code=trust_remote_code,
                cache_dir=model_cache_dir,
                local_files_only=not internet_connection_available(),
            )
            break
        except Exception as e:
            result = _handle_model_config_error(
                error=e, model_id=model_id, run_with_cli=run_with_cli
            )
            if result == "retry":
                continue
            if isinstance(result, PretrainedConfig):
                return result
            if result is None:
                raise
            # result == "continue" - fall through to retry
    else:
        raise InvalidModel(
            f"Couldn't load model config for {model_id!r} after {num_attempts} "
            "attempts."
        )

    _set_pad_token_id(config=config)
    return config


@cache_arguments("model_id", "run_with_cli")
def _handle_model_config_error(
    error: Exception, model_id: str, run_with_cli: bool
) -> t.Literal["retry", "continue"] | PretrainedConfig | None:
    """Handle an error during model config loading.

    Args:
        error:
            The exception that was raised.
        model_id:
            The model ID.
        run_with_cli:
            Whether running with CLI.

    Returns:
        "retry" to retry loading, "continue" to skip, a PretrainedConfig to return
        immediately, or None to raise an exception.

    Raises:
        InvalidModel:
            If the model config could not be loaded.
        NeedsAdditionalArgument:
            If trust_remote_code is required.
    """
    e = error
    if isinstance(e, KeyError):
        raise InvalidModel(
            f"The model config for the model {model_id!r} could not be "
            f"loaded, as the key {e.args[0]!r} was not found in the config."
        ) from e

    if isinstance(e, (OSError, GatedRepoError)):
        if isinstance(e, GatedRepoError) or "gated repo" in str(e).lower():
            raise InvalidModel(
                f"The model {model_id!r} is a gated repository. Please ensure "
                "that you are logged in with `hf auth login` or have provided a "
                "valid Hugging Face access token with the `HUGGINGFACE_API_KEY` "
                "or `HF_TOKEN` environment variable or the `--api-key` argument. "
                "Also check that your account has access to this model."
            ) from e
        if not internet_connection_available():
            log(
                f"Couldn't load model config for {model_id!r} offline. "
                f"The error was {e!r}. Returning minimal config.",
                level=logging.WARNING,
            )
            return PretrainedConfig()
        raise InvalidModel(
            f"Couldn't load model config for {model_id!r}. The error was "
            f"{e!r}. Skipping"
        ) from e

    if isinstance(e, (TimeoutError, RequestError)):
        log(
            f"Couldn't load model config for {model_id!r}. Retrying.",
            level=logging.WARNING,
        )
        sleep(5)
        return "retry"

    if isinstance(e, ValueError):
        if "awaiting a review from the repo authors" in str(e):
            raise InvalidModel(
                f"The model {model_id!r} is awaiting a review from the repository "
                "authors. Please try again later."
            ) from e
        if "trust_remote_code" in str(e):
            raise NeedsAdditionalArgument(
                cli_argument="--trust-remote-code",
                script_argument="trust_remote_code=True",
                run_with_cli=run_with_cli,
            ) from e
        raise InvalidModel(
            f"The config for the model {model_id!r} could not be loaded. The "
            f"error was {e!r}."
        ) from e

    return None


def _set_pad_token_id(config: PretrainedConfig) -> None:
    """Set the PAD token ID from EOS token ID if not set.

    Args:
        config:
            The model configuration to update.
    """
    if (
        hasattr(config, "eos_token_id")
        and config.eos_token_id is not None
        and (not hasattr(config, "pad_token_id") or config.pad_token_id is None)
    ):
        if isinstance(config.eos_token_id, list):
            config.pad_token_id = config.eos_token_id[0]
        else:
            config.pad_token_id = config.eos_token_id


def load_tokeniser(
    model: "PreTrainedModel | None",
    model_id: str,
    trust_remote_code: bool,
    model_config: "ModelConfig",
) -> Tokeniser:
    """Load the tokeniser.

    Args:
        model:
            The model, which is used to determine whether to add a prefix space to
            the tokens. Can be None.
        model_id:
            The model identifier. Used for logging.
        trust_remote_code:
            Whether to trust remote code.
        model_config:
            The model configuration.

    Returns:
        The loaded tokeniser.

    Raises:
        InvalidModel:
            If the tokeniser could not be loaded.
    """
    loading_kwargs: dict[str, bool | str] = dict(
        use_fast=False if model_config.param == "slow-tokenizer" else True,
        trust_remote_code=trust_remote_code,
        padding_side="right",
        truncation_side="right",
        cache_dir=model_config.model_cache_dir,
    )

    # If the model is a subclass of a certain model types then we have to add a prefix
    # space to the tokens, by the way the model is constructed.
    if model is not None:
        prefix_models = ["Roberta", "GPT", "Deberta"]
        add_prefix = any(
            model_type in type(model).__name__ for model_type in prefix_models
        )
        if add_prefix:
            loading_kwargs["add_prefix_space"] = True

    num_retries = 5
    for attempt in range(num_retries):
        try:
            tokeniser: Tokeniser = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_id, **loading_kwargs
            )  # ty: ignore[invalid-assignment]
            break
        except TypeError as e:
            # XLM-RoBERTa variant models like 'EMBEDDIA/litlat-bert' raise TypeError
            # when loading fast tokenizers. Fall back to slow tokenizer.
            if loading_kwargs.get("use_fast", True):
                log(
                    f"TypeError occurred during the loading of the tokeniser for "
                    f"{model_id!r}. Retrying with use_fast=False.",
                    level=logging.DEBUG,
                )
                loading_kwargs["use_fast"] = False
                continue
            else:
                raise InvalidModel(
                    f"Could not load tokeniser for model {model_id!r}."
                ) from e
        except (JSONDecodeError, OSError) as e:
            raise InvalidModel(
                f"Could not load tokeniser for model {model_id!r}."
            ) from e
        except (TimeoutError, RequestError):
            log(
                f"Couldn't load tokeniser for {model_id!r}. Retrying.",
                level=logging.WARNING,
            )
            sleep(5)
            continue
    else:
        raise InvalidModel(
            f"Could not load tokeniser for model {model_id!r} after {num_retries} "
            "attempts."
        )

    tokeniser.bos_token, tokeniser.bos_token_id = get_bos_token(tokeniser=tokeniser)
    tokeniser.eos_token, tokeniser.eos_token_id = get_eos_token(tokeniser=tokeniser)

    return tokeniser


def setup_model_for_question_answering(model: "PreTrainedModel") -> "PreTrainedModel":
    """Setup a model for question answering.

    Args:
        model:
            The model to setup.

    Returns:
        The setup model.

    Raises:
        InvalidModel:
            If the model does not have token type embeddings.
    """
    children = get_children_of_module(name="model", module=model)
    assert isinstance(children, dict)

    if children:
        attribute_list = list()
        done = False
        while not done:
            for key, value in children.items():
                attribute_list.append(key)
                if isinstance(value, dict):
                    children = value
                else:
                    done = True
                break

        token_type_embeddings = model
        for attribute in attribute_list:
            token_type_embeddings = getattr(token_type_embeddings, attribute)

        token_type_embedding_tensor = token_type_embeddings.weight.data
        assert isinstance(token_type_embedding_tensor, torch.Tensor)

        # If the token type embeddings has shape (1, ...) then set the shape to
        # (2, ...) by randomly initializing the second token type embedding
        if token_type_embedding_tensor.shape[0] == 1:
            if not hasattr(token_type_embeddings.weight, "data"):
                raise InvalidModel(
                    "The token type embeddings of the model do not have a `data` "
                    "attribute, which is needed to modify the embeddings."
                )
            token_type_embeddings.weight.data = torch.cat(
                tensors=(
                    token_type_embedding_tensor,
                    torch.rand_like(token_type_embedding_tensor),
                ),
                dim=0,
            )
            token_type_embeddings.num_embeddings = 2  # ty: ignore[invalid-assignment]

        model.config.type_vocab_size = 2

    return model


def get_children_of_module(
    name: str, module: nn.Module
) -> nn.Module | dict[str, t.Any] | None:
    """Get the children of a module.

    Args:
        name:
            The name of the module.
        module:
            The module to get the children of.

    Returns:
        The children of the module, or None if the module has no children.
    """
    if len(list(module.children())) == 0:
        if name == "token_type_embeddings":
            return module
        else:
            return None
    else:
        submodules = dict()
        for subname, submodule in module.named_children():
            children = get_children_of_module(name=subname, module=submodule)
            if children:
                submodules[subname] = children
        return submodules


def task_group_to_class_name(task_group: TaskGroup) -> str:
    """Convert a task group to a class name.

    Args:
        task_group:
            The task group.

    Returns:
        The class name.
    """
    pascal_case = task_group.title().replace("_", "")

    # Bridge task-group names that don't map 1:1 onto a Hugging Face AutoModel class:
    # the multiple-choice group is internally named "..._classification" but the HF
    # class is `AutoModelForMultipleChoice`, and `Speed` reuses the sequence
    # classification head.
    special_case_mapping = dict(
        MultipleChoiceClassification="MultipleChoice", Speed="SequenceClassification"
    )
    pascal_case = special_case_mapping.get(pascal_case, pascal_case)
    return f"AutoModelFor{pascal_case}"


def get_model_repo_info(
    model_id: str,
    revision: str,
    api_key: str | None,
    cache_dir: str,
    trust_remote_code: bool,
    requires_safetensors: bool,
    run_with_cli: bool,
) -> "HFModelInfo | None":
    """Get the information about the model from the HF Hub or a local directory.

    Args:
        model_id:
            The model ID.
        revision:
            The revision of the model.
        api_key:
            The Hugging Face API key.
        cache_dir:
            The directory to cache the model in.
        trust_remote_code:
            Whether to trust remote code.
        requires_safetensors:
            Whether the model requires safetensors.
        run_with_cli:
            Whether the script is being run with the CLI.

    Returns:
        The information about the model, or None if the model could not be found.
    """
    token = get_hf_token(api_key=api_key)
    hf_api = HfApi(token=token)

    # Try to get model info from local directory first
    model_info: HfApiModelInfo | None = None
    if Path(model_id).is_dir():
        model_info = _get_local_model_info(model_id=model_id)
        if model_info is None:
            return None
    elif not internet_connection_available():
        model_info = HfApiModelInfo(id=model_id, tags=None, pipeline_tag=None)

    # Fetch from HF Hub if not found locally
    if model_info is None:
        model_info = _fetch_model_info_from_hub(
            hf_api=hf_api, model_id=model_id, revision=revision, token=token
        )
        if model_info is None:
            return None

    # Handle adapter models - get base model tags
    tags = model_info.tags or list()
    base_model_id: str | None = None
    has_adapter_config = model_info.siblings is not None and any(
        sibling.rfilename == "adapter_config.json" for sibling in model_info.siblings
    )
    if has_adapter_config:
        tags, base_model_id = _get_tags_for_adapter_model(
            model_id=model_id,
            revision=revision,
            model_info=model_info,
            hf_api=hf_api,
            token=token,
        )

    # Infer pipeline tag if not specified
    pipeline_tag = model_info.pipeline_tag
    if pipeline_tag is None:
        pipeline_tag = _infer_pipeline_tag(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            api_key=api_key,
            trust_remote_code=trust_remote_code,
            run_with_cli=run_with_cli,
            base_model_id=base_model_id,
        )

    # Check safetensors requirement
    if requires_safetensors and not _check_safetensors_available(
        hf_api=hf_api,
        model_id=model_id,
        revision=revision,
        base_model_id=base_model_id,
        run_with_cli=run_with_cli,
    ):
        return None

    return HFModelInfo(
        pipeline_tag=pipeline_tag, tags=tags, adapter_base_model_id=base_model_id
    )


def _check_safetensors_available(
    hf_api: HfApi,
    model_id: str,
    revision: str,
    base_model_id: str | None,
    run_with_cli: bool,
) -> bool:
    """Check if safetensors weights are available.

    Args:
        hf_api:
            The Hugging Face API client.
        model_id:
            The model ID.
        revision:
            The revision.
        base_model_id:
            Base model ID if this is an adapter.
        run_with_cli:
            Whether running with CLI.

    Returns:
        True if safetensors are available, False otherwise.
    """
    repo_files = hf_api.list_repo_files(repo_id=model_id, revision=revision)
    has_safetensors = any(f.endswith(".safetensors") for f in repo_files)
    if not has_safetensors:
        msg = f"Model {model_id} does not have safetensors weights available. "
        if run_with_cli:
            msg += "Skipping since the `--only-allow-safetensors` flag is set."
        else:
            msg += (
                "Skipping since the `requires_safetensors` argument is set to `True`."
            )
        log(msg, level=logging.WARNING)
        return False

    if base_model_id is not None:
        base_repo_files = hf_api.list_repo_files(repo_id=base_model_id)
        base_has_safetensors = any(f.endswith(".safetensors") for f in base_repo_files)
        if not base_has_safetensors:
            msg = (
                f"Base model {base_model_id} does not have safetensors "
                "weights available."
            )
            if run_with_cli:
                msg += " Skipping since the `--only-allow-safetensors` flag is set."
            else:
                msg += (
                    " Skipping since the `requires_safetensors` argument is set "
                    "to `True`."
                )
            logging.warning(msg)
            return False
    return True


def _fetch_model_info_from_hub(
    hf_api: HfApi, model_id: str, revision: str, token: str | None
) -> HfApiModelInfo | None:
    """Fetch model info from HF Hub with retry logic.

    Args:
        hf_api:
            The Hugging Face API client.
        model_id:
            The model ID.
        revision:
            The revision to fetch.
        token:
            The API token.

    Returns:
        Model info object, or None if not found or access denied.
    """
    num_attempts = 3
    errors: list[Exception] = list()
    for _ in range(num_attempts):
        try:
            return hf_api.model_info(repo_id=model_id, revision=revision, token=token)
        except (GatedRepoError, LocalTokenNotFoundError) as e:
            try:
                hf_whoami(token=token)
                log(
                    f"Could not access the model {model_id} with the revision "
                    f"{revision}. The error was {str(e)!r}.",
                    level=logging.DEBUG,
                )
                return None
            except LocalTokenNotFoundError:
                log(
                    f"Could not access the model {model_id} with the revision "
                    f"{revision}. The error was {str(e)!r}. Please set the "
                    "`HUGGINGFACE_API_KEY` or `HF_TOKEN` environment variable or "
                    "use the `--api-key` argument.",
                    level=logging.DEBUG,
                )
                return None
        except (RepositoryNotFoundError, HFValidationError, HfHubHTTPError):
            return None
        except (OSError, RequestException) as e:
            if internet_connection_available():
                errors.append(e)
                continue
            log(
                "Could not access the Hugging Face Hub. Please check your internet "
                "connection.",
                level=logging.DEBUG,
            )
            return None
    else:
        log(
            f"Could not access model info for the model {model_id!r} from the "
            f"Hugging Face Hub, after {num_attempts} attempts. The errors "
            f"encountered were {errors!r}.",
            level=logging.DEBUG,
        )
        return None


@cache_arguments("model_id")
def _get_local_model_info(model_id: str) -> HfApiModelInfo | None:
    """Get model info for a local model directory.

    Args:
        model_id:
            Path to the local model directory.

    Returns:
        Model info object, or None if required files are missing.
    """
    if Path(model_id, "config.json").exists():
        log_once(
            f"The local model directory {model_id!r} has a 'config.json' file, so "
            "we're skipping looking up model information from the Hugging Face "
            "Hub.",
            level=logging.DEBUG,
        )
        return HfApiModelInfo(id=model_id, tags=None, pipeline_tag=None)
    elif Path(model_id, "adapter_config.json").exists():
        log_once(
            f"The local model directory {model_id!r} has an 'adapter_config.json' "
            "file, so we're skipping looking up model information from the Hugging "
            "Face Hub.",
            level=logging.DEBUG,
        )
        return HfApiModelInfo(
            id=model_id,
            tags=None,
            pipeline_tag=None,
            siblings=[dict(rfilename="adapter_config.json")],
        )
    else:
        log_once(
            f"The local model directory {model_id} does not contain any of the "
            f"required files: {LOCAL_MODELS_REQUIRED_FILES}. Skipping this "
            f"model.",
            level=logging.WARNING,
        )
        return None


def _get_tags_for_adapter_model(
    model_id: str,
    revision: str,
    model_info: HfApiModelInfo,
    hf_api: HfApi,
    token: str | None,
) -> tuple[list[str], str | None]:
    """Get tags for an adapter model including base model tags.

    Args:
        model_id:
            The adapter model ID.
        revision:
            The revision.
        model_info:
            The model info for the adapter.
        hf_api:
            The Hugging Face API client.
        token:
            The API token.

    Returns:
        Tuple of (tags, base_model_id).
    """
    adapter_config = PeftConfig.from_pretrained(
        pretrained_model_name_or_path=model_id, revision=revision
    )
    base_model_id = adapter_config.base_model_name_or_path
    log_once(
        f"Model {model_id!r} identified as an adapter model, with base model "
        f"{base_model_id!r}.",
        level=logging.DEBUG,
    )
    tags = model_info.tags or list()
    if base_model_id is not None:
        base_model_info = hf_api.model_info(repo_id=base_model_id, token=token)
        tags += base_model_info.tags or list()
        tags = list(set(tags))
    return tags, base_model_id


def _infer_pipeline_tag(
    model_id: str,
    revision: str,
    cache_dir: str,
    api_key: str | None,
    trust_remote_code: bool,
    run_with_cli: bool,
    base_model_id: str | None,
) -> str:
    """Infer pipeline tag from model architecture.

    Args:
        model_id:
            The model ID.
        revision:
            The revision.
        cache_dir:
            Cache directory.
        api_key:
            API key.
        trust_remote_code:
            Whether to trust remote code.
        run_with_cli:
            Whether running with CLI.
        base_model_id:
            Base model ID if this is an adapter.

    Returns:
        The inferred pipeline tag.
    """
    hf_config = load_hf_model_config(
        model_id=base_model_id or model_id,
        num_labels=0,
        id2label=HashableDict(),
        label2id=HashableDict(),
        revision=revision,
        model_cache_dir=create_model_cache_dir(cache_dir=cache_dir, model_id=model_id),
        api_key=api_key,
        trust_remote_code=trust_remote_code,
        run_with_cli=run_with_cli,
    )
    class_names = hf_config.architectures
    generative_class_names = [
        class_name
        for tag in GENERATIVE_PIPELINE_TAGS
        for class_name in TASK_MAPPING.get(tag, dict()).values()
    ]
    if class_names is not None and (
        any(class_name in generative_class_names for class_name in class_names)
        or any("ForCausalLM" in class_name for class_name in class_names)
    ):
        return "text-generation"
    return "fill-mask"
