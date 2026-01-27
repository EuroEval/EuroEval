"""Encoder models from the Hugging Face Hub."""

import collections.abc as c
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
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import ModelInfo as HfApiModelInfo
from peft import PeftConfig
from requests.exceptions import RequestException
from torch import nn
from transformers.data.data_collator import (
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
    MERGE_TAGS,
)
from ..data_models import HashableDict, HFModelInfo, ModelConfig
from ..enums import (
    BatchingPreference,
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
from ..languages import get_all_languages
from ..logging_utils import block_terminal_output, log, log_once
from ..task_group_utils import (
    multiple_choice_classification,
    question_answering,
    token_classification,
)
from ..tokenisation_utils import get_bos_token, get_eos_token
from ..types import Tokeniser
from ..utils import (
    create_model_cache_dir,
    get_class_by_name,
    get_hf_token,
    internet_connection_available,
    split_model_id,
)
from .base import BenchmarkModule

try:
    from transformers.tokenization_mistral_common import MistralCommonTokenizer
except ImportError:
    from transformers.tokenization_mistral_common import (
        MistralCommonBackend as MistralCommonTokenizer,
    )

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

        model, tokeniser = load_model_and_tokeniser(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        self._model: "PreTrainedModel" = model
        self._tokeniser: "PreTrainedTokenizer | MistralCommonTokenizer" = tokeniser

        self._model, self._tokeniser = align_model_and_tokeniser(
            model=self._model,
            tokeniser=self._tokeniser,
            model_max_length=self.model_max_length,
            raise_errors=benchmark_config.raise_errors,
        )

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        # No need to try to use the API if we have no internet.
        if not internet_connection_available():
            repo_info = None
        else:
            token = get_hf_token(api_key=self.benchmark_config.api_key)
            hf_api = HfApi(token=token)
            try:
                repo_info = hf_api.model_info(
                    repo_id=self.model_config.adapter_base_model_id
                    or self.model_config.model_id,
                    revision=self.model_config.revision,
                )
            except (
                RepositoryNotFoundError,
                RevisionNotFoundError,
                RequestException,
                HFValidationError,
            ):
                repo_info = None

        if (
            repo_info is not None
            and hasattr(repo_info, "safetensors")
            and repo_info.safetensors is not None
            and "total" in repo_info.safetensors
        ):
            num_params_candidates: list[int] = [repo_info.safetensors["total"]]
            if "parameters" in repo_info.safetensors and isinstance(
                repo_info.safetensors["parameters"], dict
            ):
                num_params_candidates.extend(
                    int(v)
                    for v in repo_info.safetensors["parameters"].values()
                    if isinstance(v, int) or (isinstance(v, str) and v.isdigit())
                )
            num_params = max(num_params_candidates)
        elif (
            hasattr(self._model.config, "num_params")
            and self._model.config.num_params is not None
        ):
            num_params = self._model.config.num_params
        elif hasattr(self._model, "parameters"):
            num_params = sum(p.numel() for p in self._model.parameters())
        else:
            log(
                "The number of parameters could not be determined for the model, since "
                "the model is not stored in the safetensors format. If this is your "
                "own model, then you can use this Hugging Face Space to convert your "
                "model to the safetensors format: "
                "https://huggingface.co/spaces/safetensors/convert.",
                level=logging.WARNING,
            )
            num_params = -1
        return num_params

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        if (
            hasattr(self._model.config, "vocab_size")
            and self._model.config.vocab_size is not None
        ):
            vocab_size = self._model.config.vocab_size
        elif (
            hasattr(self._tokeniser, "vocab_size")
            and self._tokeniser.vocab_size is not None
        ):
            vocab_size = self._tokeniser.vocab_size
        else:
            vocab_size = -1
        return vocab_size

    @cached_property
    def model_max_length(self) -> int:
        """The maximum context length of the model.

        Returns:
            The maximum context length of the model.
        """
        all_max_lengths: list[int] = list()

        # Add the registered max length of the tokeniser
        if hasattr(
            self._tokeniser, "model_max_length"
        ) and self._tokeniser.model_max_length < int(1e30):
            all_max_lengths.append(self._tokeniser.model_max_length)

        # Add the max length derived from the model's input sizes
        if hasattr(self._tokeniser, "max_model_input_sizes"):
            all_max_lengths.extend(
                [
                    size
                    for size in self._tokeniser.max_model_input_sizes.values()
                    if size is not None
                ]
            )

        # Add max length candidates from the model's configuration
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
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                return DataCollatorWithPadding(self._tokeniser, padding="longest")
            case TaskGroup.TOKEN_CLASSIFICATION:
                return DataCollatorForTokenClassification(
                    tokenizer=self._tokeniser, label_pad_token_id=-100
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    def generative_type(self) -> GenerativeType | None:
        """Get the generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        return None

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
            ):
                return Trainer
            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                return (
                    multiple_choice_classification.MultipleChoiceClassificationTrainer
                )
            case TaskGroup.QUESTION_ANSWERING:
                return question_answering.QuestionAnsweringTrainer
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

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

        def numericalise_labels(examples: dict) -> dict:
            if "label" in examples:
                try:
                    examples["label"] = [
                        self._model.config.label2id[lbl.lower()]
                        if self._model.config.label2id is not None
                        else lbl
                        for lbl in examples["label"]
                    ]
                except KeyError as e:
                    raise InvalidBenchmark(
                        f"One of the labels in the dataset, "
                        f"{examples['label'].lower()}, does not occur in the "
                        f"label2id dictionary {self._model.config.label2id}."
                    ) from e
            return examples

        def tokenise(examples: dict) -> "BatchEncoding":
            return self._tokeniser(text=examples["text"], truncation=True, padding=True)

        match task.task_group:
            case TaskGroup.SEQUENCE_CLASSIFICATION:
                dataset = dataset.map(
                    numericalise_labels, batched=True, load_from_cache_file=False
                ).map(tokenise, batched=True, load_from_cache_file=False)

            case TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
                dataset = DatasetDict(  # type: ignore[no-matching-overload]
                    {
                        split_name: split.map(
                            partial(
                                multiple_choice_classification.prepare_examples,
                                tokeniser=self._tokeniser,
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

            case TaskGroup.TEXT_TO_TEXT:
                dataset = dataset.map(
                    tokenise,
                    batched=True,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )

            case TaskGroup.TOKEN_CLASSIFICATION:
                dataset = dataset.map(
                    partial(
                        token_classification.tokenize_and_align_labels,
                        tokeniser=self._tokeniser,
                        label2id=self._model.config.label2id,
                    ),
                    batched=True,
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )

            case TaskGroup.QUESTION_ANSWERING:
                data_dict = dict()
                if "train" in dataset:
                    data_dict["train"] = dataset["train"].map(
                        partial(
                            question_answering.prepare_train_examples,
                            tokeniser=self._tokeniser,
                        ),
                        batched=True,
                        batch_size=10,
                        remove_columns=dataset["test"].column_names,
                        load_from_cache_file=False,
                        keep_in_memory=True,
                    )
                if "val" in dataset:
                    data_dict["val"] = dataset["val"].map(
                        partial(
                            question_answering.prepare_train_examples,
                            tokeniser=self._tokeniser,
                        ),
                        batched=True,
                        batch_size=10,
                        remove_columns=dataset["test"].column_names,
                        load_from_cache_file=False,
                        keep_in_memory=True,
                    )
                if "test" in dataset:
                    data_dict["test"] = dataset["test"].map(
                        partial(
                            question_answering.prepare_test_examples,
                            tokeniser=self._tokeniser,
                        ),
                        batched=True,
                        batch_size=10,
                        remove_columns=dataset["test"].column_names,
                        load_from_cache_file=False,
                        keep_in_memory=True,
                    )
                dataset = DatasetDict(data_dict)  # type: ignore[no-matching-overload]

                # The Trainer hides the columns that are not used by the model (here
                # `id` and `offset_mapping` which we will need for our post-processing),
                # so we put them back
                for split_name, split in dataset.items():
                    dataset[split_name].set_format(
                        type=split.format["type"], columns=list(split.features.keys())
                    )

            case _:
                raise NotImplementedError(f"Unsupported task group: {task.task_group}.")

        return dataset

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
        return (
            model_info is not None
            and model_info.pipeline_tag not in GENERATIVE_PIPELINE_TAGS
        )

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
        """
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
        if model_info is None:
            raise InvalidModel(f"The model {model_id!r} could not be found.")

        language_mapping = get_all_languages()
        language_codes = list(language_mapping.keys())

        model_config = ModelConfig(
            model_id=model_id_components.model_id,
            revision=model_id_components.revision,
            param=model_id_components.param,
            task=model_info.pipeline_tag,
            languages=[
                language_mapping[tag]
                for tag in model_info.tags
                if tag in language_codes
            ],
            merge=any(tag in model_info.tags for tag in MERGE_TAGS),
            inference_backend=InferenceBackend.TRANSFORMERS,
            model_type=ModelType.ENCODER,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=None,
        )

        return model_config


def load_model_and_tokeniser(
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> tuple["PreTrainedModel", Tokeniser]:
    """Load the model and tokeniser.

    Args:
        model_config:
            The model configuration.
        dataset_config:
            The dataset configuration.
        benchmark_config:
            The benchmark configuration

    Returns:
        A pair (model, tokeniser), with the loaded model and tokeniser
    """
    config: "PretrainedConfig"
    block_terminal_output()

    model_id = model_config.model_id
    task_group = dataset_config.task.task_group
    ignore_mismatched_sizes = False

    # Special case where there is a mismatch between the labels during training and
    # testing
    if dataset_config.task.task_group == TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION:
        id2label = {0: "0", 1: "1"}
    else:
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

    model_kwargs = dict(
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
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
        ),
    )

    model: "PreTrainedModel | None" = None
    for _ in range(num_attempts := 5):
        # Get the model class associated with the task group
        model_cls_or_none: t.Type["PreTrainedModel"] | None = get_class_by_name(
            class_name=task_group_to_class_name(task_group=task_group),
            module_name="transformers",
        )

        # If the model class could not be found then raise an error
        if not model_cls_or_none:
            raise InvalidBenchmark(
                f"The task group {task_group.value!r} does not correspond to a "
                "Hugging Face AutoModel type (such as "
                "`AutoModelForSequenceClassification`)."
            )

        # If the model is a DeBERTaV2 model then we ensure that
        # `pooler_hidden_size` is the same size as `hidden_size`
        if config.model_type == "deberta-v2":
            config.pooler_hidden_size = config.hidden_size

        try:
            model_or_tuple = model_cls_or_none.from_pretrained(
                model_config.model_id, **model_kwargs
            )
            break
        except (KeyError, RuntimeError) as e:
            if not model_kwargs["ignore_mismatched_sizes"]:
                log(
                    f"{type(e).__name__} occurred during the loading "
                    f"of the {model_id!r} model. Retrying with "
                    "`ignore_mismatched_sizes` set to True.",
                    level=logging.DEBUG,
                )
                model_kwargs["ignore_mismatched_sizes"] = True
                continue
            else:
                raise InvalidModel(str(e)) from e
        except (TimeoutError, RequestError):
            log(
                f"Couldn't load the model {model_id!r}. Retrying.",
                level=logging.WARNING,
            )
            sleep(5)
            continue
        except (OSError, ValueError) as e:
            if "checkpoint seems to be incorrect" in str(e):
                raise InvalidModel(
                    f"The model {model_id!r} has an incorrect checkpoint."
                ) from e
            if "trust_remote_code" in str(e):
                raise InvalidModel(
                    f"Loading the model {model_id!r} needs to trust remote code. "
                    "If you trust the suppliers of this model, then you can enable "
                    "this by setting the `--trust-remote-code` flag."
                ) from e
            raise InvalidModel(
                f"The model {model_id!r} could not be loaded. The error was {e!r}."
            ) from e
    else:
        raise InvalidModel(
            f"Could not load the model {model_id!r} after {num_attempts} attempts."
        )

    if isinstance(model_or_tuple, tuple):
        model = model_or_tuple[0]
    else:
        model = model_or_tuple

    assert model is not None, "The model should not be None."

    model.eval()
    model.to(benchmark_config.device)  # type: ignore[arg-type]

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


@cache_arguments("model_id", "revision")
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

    Returns:
        The information about the model, or None if the model could not be found.
    """
    token = get_hf_token(api_key=api_key)
    hf_api = HfApi(token=token)

    # Get information on the model.
    # The first case is when the model is a local model, in which case we create a dummy
    # model info object.
    model_info: HfApiModelInfo | None = None
    if Path(model_id).is_dir():
        if all(
            (Path(model_id) / required_file).exists()
            for required_file in LOCAL_MODELS_REQUIRED_FILES
        ):
            log_once(
                f"The local model directory {model_id!r} has all the required model "
                f"files ({LOCAL_MODELS_REQUIRED_FILES}), so we're skipping looking up "
                "model information from the Hugging Face Hub.",
                level=logging.DEBUG,
            )
            model_info = HfApiModelInfo(id=model_id, tags=None, pipeline_tag=None)
        else:
            log_once(
                f"The local model directory {model_id} does not contain all the "
                f"required files: {LOCAL_MODELS_REQUIRED_FILES}. Skipping this "
                f"model.",
                level=logging.WARNING,
            )
            return None

    # If we have not internet, and the model_id is not a directory for a local model
    # we also just create a dummy model info object.
    elif not internet_connection_available():
        model_info = HfApiModelInfo(id=model_id, tags=None, pipeline_tag=None)

    # If the model does not exist locally, then we get the model info from the Hugging
    # Face Hub, if possible
    if model_info is None:
        num_attempts = 3
        errors: list[Exception] = list()
        for _ in range(num_attempts):
            try:
                model_info = hf_api.model_info(
                    repo_id=model_id, revision=revision, token=token
                )
                break
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
                        "`HUGGINGFACE_API_KEY` environment variable or use the "
                        "`--api-key` argument.",
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

    # Get all the Hugging Face repository tags for the model. If the model is an adapter
    # model, then we also get the tags for the base model
    tags = model_info.tags or list()
    base_model_id: str | None = None
    has_adapter_config = model_info.siblings is not None and any(
        sibling.rfilename == "adapter_config.json" for sibling in model_info.siblings
    )
    if has_adapter_config:
        adapter_config = PeftConfig.from_pretrained(model_id, revision=revision)
        base_model_id = adapter_config.base_model_name_or_path
        log_once(
            f"Model {model_id!r} identified as an adapter model, with base model "
            f"{base_model_id!r}.",
            level=logging.DEBUG,
        )
        if base_model_id is not None:
            base_model_info = hf_api.model_info(repo_id=base_model_id, token=token)
            tags += base_model_info.tags or list()
            tags = list(set(tags))

    # Get the pipeline tag for the model. If it is not specified, then we determine it
    # by checking the model's architecture as written in the model's Hugging Face config
    pipeline_tag = model_info.pipeline_tag
    if pipeline_tag is None:
        hf_config = load_hf_model_config(
            model_id=base_model_id or model_id,
            num_labels=0,
            id2label=HashableDict(),
            label2id=HashableDict(),
            revision=revision,
            model_cache_dir=create_model_cache_dir(
                cache_dir=cache_dir, model_id=model_id
            ),
            api_key=api_key,
            trust_remote_code=trust_remote_code,
            run_with_cli=run_with_cli,
        )
        class_names = hf_config.architectures
        generative_class_names = [
            class_name
            for tag in GENERATIVE_PIPELINE_TAGS
            for class_name in TASK_MAPPING.get(tag, dict()).values()  # type: ignore[attr-defined]
        ]
        if class_names is not None and (
            any(class_name in generative_class_names for class_name in class_names)
            or any("ForCausalLM" in class_name for class_name in class_names)
        ):
            pipeline_tag = "text-generation"
        else:
            pipeline_tag = "fill-mask"

    if requires_safetensors:
        repo_files = hf_api.list_repo_files(repo_id=model_id, revision=revision)
        has_safetensors = any(f.endswith(".safetensors") for f in repo_files)
        if not has_safetensors:
            msg = f"Model {model_id} does not have safetensors weights available. "
            if run_with_cli:
                msg += "Skipping since the `--only-allow-safetensors` flag is set."
            else:
                msg += (
                    "Skipping since the `requires_safetensors` argument is set "
                    "to `True`."
                )
            log(msg, level=logging.WARNING)
            return None

        # Also check base model if we are evaluating an adapter
        if base_model_id is not None:
            base_repo_files = hf_api.list_repo_files(repo_id=base_model_id)
            base_has_safetensors = any(
                f.endswith(".safetensors") for f in base_repo_files
            )
            if not base_has_safetensors:
                msg = (
                    f"Base model {base_model_id} does not have safetensors weights "
                    "available."
                )
                if run_with_cli:
                    msg += " Skipping since the `--only-allow-safetensors` flag is set."
                else:
                    msg += (
                        " Skipping since the `requires_safetensors` argument is set "
                        "to `True`."
                    )
                logging.warning(msg)
                return None

    return HFModelInfo(
        pipeline_tag=pipeline_tag, tags=tags, adapter_base_model_id=base_model_id
    )


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
    """
    loading_kwargs: dict[str, bool | str] = dict(
        use_fast=False if model_config.param == "slow-tokenizer" else True,
        verbose=False,
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
    for _ in range(num_retries):
        try:
            tokeniser = AutoTokenizer.from_pretrained(model_id, **loading_kwargs)
            break
        except (JSONDecodeError, OSError, TypeError) as e:
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

    # Ensure that BOS, EOS and PAD tokens are set
    tokeniser.bos_token, tokeniser.bos_token_id = get_bos_token(tokeniser=tokeniser)
    tokeniser.eos_token, tokeniser.eos_token_id = get_eos_token(tokeniser=tokeniser)

    return tokeniser


@cache_arguments()
def get_dtype(
    device: torch.device, dtype_is_set: bool, bf16_available: bool
) -> str | torch.dtype:
    """Get the torch dtype, used for loading the model.

    Args:
        device:
            The device to use.
        dtype_is_set:
            Whether the data type is set in the model configuration.
        bf16_available:
            Whether bfloat16 is available.

    Returns:
        The dtype.
    """
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
    """
    for _ in range(num_attempts := 5):
        try:
            config = AutoConfig.from_pretrained(
                model_id,
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
        except KeyError as e:
            key = e.args[0]
            raise InvalidModel(
                f"The model config for the model {model_id!r} could not be "
                f"loaded, as the key {key!r} was not found in the config."
            ) from e
        except (OSError, GatedRepoError) as e:
            if isinstance(e, GatedRepoError) or "gated repo" in str(e).lower():
                raise InvalidModel(
                    f"The model {model_id!r} is a gated repository. Please ensure "
                    "that you are logged in with `hf auth login` or have provided a "
                    "valid Hugging Face access token with the `HUGGINGFACE_API_KEY` "
                    "environment variable or the `--api-key` argument. Also check that "
                    "your account has access to this model."
                ) from e
            raise InvalidModel(
                f"Couldn't load model config for {model_id!r}. The error was "
                f"{e!r}. Skipping"
            ) from e
        except (TimeoutError, RequestError):
            log(
                f"Couldn't load model config for {model_id!r}. Retrying.",
                level=logging.WARNING,
            )
            sleep(5)
            continue
        except ValueError as e:
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
    else:
        raise InvalidModel(
            f"Couldn't load model config for {model_id!r} after {num_attempts} "
            "attempts."
        )

    # Ensure that the PAD token ID is set
    if config.eos_token_id is not None and config.pad_token_id is None:
        if isinstance(config.eos_token_id, list):
            config.pad_token_id = config.eos_token_id[0]
        else:
            config.pad_token_id = config.eos_token_id

    return config


def setup_model_for_question_answering(model: "PreTrainedModel") -> "PreTrainedModel":
    """Setup a model for question answering.

    Args:
        model:
            The model to setup.

    Returns:
        The setup model.
    """
    # Get the models' token type embedding children, if they exist
    children = get_children_of_module(name="model", module=model)
    assert isinstance(children, dict)

    # If the model has token type embeddings then get them
    if children:
        # Get the list of attributes that are token type embeddings
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

        # Get the token type embeddings
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
                (
                    token_type_embedding_tensor,
                    torch.rand_like(token_type_embedding_tensor),
                ),
                dim=0,
            )
            token_type_embeddings.num_embeddings = 2  # type: ignore[assignment]

        # Set the model config to use the new type vocab size
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


def align_model_and_tokeniser(
    model: "PreTrainedModel",
    tokeniser: Tokeniser,
    model_max_length: int,
    raise_errors: bool = False,
) -> tuple["PreTrainedModel", Tokeniser]:
    """Aligns the model and the tokeniser.

    Args:
        model:
            The model to fix.
        tokeniser:
            The tokeniser to fix.
        model_max_length:
            The maximum length of the model.
        raise_errors:
            Whether to raise errors instead of trying to fix them silently.

    Returns:
        The fixed model and tokeniser.
    """
    model_max_length = min(model_max_length, MAX_CONTEXT_LENGTH)

    if model_max_length > 0:
        tokeniser.model_max_length = model_max_length
    else:
        tokeniser.model_max_length = 512

    # Move the model to the CPU, since otherwise we can't catch the IndexErrors when
    # finding the maximum sequence length of the model
    model_device = model.device
    model.to(torch.device("cpu"))  # type: ignore[arg-type]

    # Manually check that this model max length is valid for the model, and adjust
    # otherwise
    initial_max_length = tokeniser.model_max_length
    for max_length in range(initial_max_length, 0, -1):
        tokeniser.model_max_length = max_length
        dummy_inputs = torch.full(
            size=(1, max_length),
            fill_value=DUMMY_FILL_VALUE,
            dtype=torch.long,
            device=model.device,
        )
        with torch.inference_mode():
            try:
                model(dummy_inputs, attention_mask=torch.ones_like(dummy_inputs))
                break

            # This happens if `max_length` is too large
            except IndexError:
                continue

            except ValueError as e:
                # This happens when the model is using Triton, such as with ModernBERT,
                # which doesn't work with CPU tensors at all
                if "cpu tensor" in str(e):
                    break
                else:
                    raise e

    # Move the model back to the original device
    model.to(model_device)  # type: ignore[arg-type]

    # If there is a mismatch between the vocab size according to the tokeniser and
    # the vocab size according to the model, we raise an error
    if hasattr(model.config, "vocab_size"):
        if model.config.vocab_size < len(tokeniser):
            if raise_errors:
                raise InvalidModel(
                    "The vocab size of the tokeniser is larger than the vocab size of "
                    "the model. As the --raise-errors option was specified, the "
                    "embeddings of the model will not be automatically adjusted."
                )
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(new_num_tokens=tokeniser.vocab_size + 1)

    if tokeniser.bos_token is None and tokeniser.eos_token is not None:
        tokeniser.bos_token = tokeniser.eos_token
        tokeniser.bos_token_id = tokeniser.eos_token_id

    return model, tokeniser


@cache_arguments()
def task_group_to_class_name(task_group: TaskGroup) -> str:
    """Convert a task group to a class name.

    Args:
        task_group:
            The task group.

    Returns:
        The class name.
    """
    pascal_case = task_group.title().replace("_", "")
    special_case_mapping = dict(
        MultipleChoiceClassification="SequenceClassification",
        Speed="SequenceClassification",
    )
    pascal_case = special_case_mapping.get(pascal_case, pascal_case)
    return f"AutoModelFor{pascal_case}"
