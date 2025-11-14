"""Generative models using the vLLM inference framework."""

import collections.abc as c
import contextlib
import importlib.util
import json
import logging
import re
import shutil
import typing as t
from functools import partial
from pathlib import Path
from time import sleep

import torch
from huggingface_hub import snapshot_download
from pydantic import conlist, create_model
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_mistral_common import MistralCommonTokenizer
from urllib3.exceptions import RequestError

from ..constants import (
    CUSTOM_STOP_TOKENS,
    GENERATIVE_PIPELINE_TAGS,
    MAX_CONTEXT_LENGTH,
    MAX_VLLM_LOGPROBS,
    MERGE_TAGS,
    REASONING_MAX_TOKENS,
    REASONING_TOKENS,
    VLLM_BF16_MIN_CUDA_COMPUTE_CAPABILITY,
)
from ..data_models import GenerativeModelOutput, HashableDict, ModelConfig
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
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
    NeedsSystemDependency,
)
from ..generation_utils import (
    apply_prompt,
    extract_few_shot_examples,
    raise_if_wrong_params,
)
from ..languages import get_all_languages
from ..logging_utils import get_pbar, log, log_once, no_terminal_output
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..tokenisation_utils import (
    apply_chat_template,
    get_bos_token,
    get_end_of_chat_token_ids,
    get_eos_token,
    get_first_label_token_mapping,
    get_pad_token,
    has_chat_template,
    should_prompts_be_stripped,
)
from ..types import ExtractLabelsFunction
from ..utils import (
    clear_memory,
    create_model_cache_dir,
    flash_attention_backend,
    get_hf_token,
    get_min_cuda_compute_capability,
    internet_connection_available,
    resolve_model_path,
    split_model_id,
)
from .hf import HuggingFaceEncoderModel, get_model_repo_info, load_hf_model_config

if t.TYPE_CHECKING or importlib.util.find_spec("vllm") is not None:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.trainer import Trainer

    from ..data_models import BenchmarkConfig, DatasetConfig, Task


MODELS_REQUIRING_FLASH_ATTENTION: list[re.Pattern] = [
    re.compile(r".*gpt-oss.*", flags=re.IGNORECASE)
]


class VLLMModel(HuggingFaceEncoderModel):
    """A generative model using the vLLM inference framework."""

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    high_priority = True
    allowed_params = {
        re.compile(r".*"): ["thinking", "no-thinking", "slow-tokenizer"],
        re.compile(r".*gpt-oss.*", flags=re.IGNORECASE): ["low", "medium", "high"],
    }

    def __init__(
        self,
        model_config: "ModelConfig",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
        log_metadata: bool = True,
    ) -> None:
        """Initialise the vLLM model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
            log_metadata:
                Whether to log the model and dataset metadata.
        """
        if importlib.util.find_spec("vllm") is None:
            raise NeedsExtraInstalled(extra="generative")

        if shutil.which("nvcc") is None:
            raise NeedsSystemDependency(
                dependency="nvcc",
                instructions=(
                    "Please install the CUDA Toolkit from "
                    "https://developer.nvidia.com/cuda-downloads or ensure that NVCC "
                    "is available in your PATH."
                ),
            )

        raise_if_wrong_params(
            model_config=model_config, allowed_params=self.allowed_params
        )

        with (
            no_terminal_output(disable=benchmark_config.verbose),
            flash_attention_backend(
                disabled=all(
                    not re.search(pattern=pattern, string=model_config.model_id)
                    for pattern in MODELS_REQUIRING_FLASH_ATTENTION
                )
            ),
        ):
            model, tokeniser = load_model_and_tokeniser(
                model_config=model_config, benchmark_config=benchmark_config
            )
        self._model: "LLM" = model
        self._tokeniser: "PreTrainedTokenizer" = tokeniser

        # We specify `HuggingFaceEncoderModel` here instead of `VLLMModel`, as we want
        # to call the `__init__` method of the `BenchmarkModule` class.
        super(HuggingFaceEncoderModel, self).__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )

        self.end_of_reasoning_token = get_end_of_reasoning_token(
            model=self._model, tokeniser=self._tokeniser, model_config=model_config
        )
        self.end_of_chat_token_ids = get_end_of_chat_token_ids(
            tokeniser=self._tokeniser, generative_type=self.generative_type
        )
        self.custom_stop_tokens = get_custom_stop_tokens(
            model=self._model,
            tokeniser=self._tokeniser,
            model_id=model_config.model_id,
            generative_type=self.generative_type,
        )

        self.buffer |= dict(
            first_label_token_mapping=get_first_label_token_mapping(
                dataset_config=self.dataset_config,
                model_config=self.model_config,
                tokeniser=self._tokeniser,
                generative_type=self.generative_type,
                log_metadata=self.log_metadata,
            )
        )
        if self.model_config.adapter_base_model_id is not None:
            adapter_path = snapshot_download(
                repo_id=self.model_config.model_id,
                revision=self.model_config.revision,
                cache_dir=Path(self.model_config.model_cache_dir),
            )
            self.buffer["lora_request"] = LoRARequest(
                lora_name="adapter", lora_int_id=1, lora_path=adapter_path
            )

    def __del__(self) -> None:
        """Clean up the model and tokeniser."""
        try:
            if importlib.util.find_spec("vllm") is not None:
                clear_vllm()
        except ImportError:
            pass
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokeniser"):
            del self._tokeniser

    @property
    def generative_type(self) -> GenerativeType | None:
        """Get the generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        if not hasattr(self, "_tokeniser"):
            log_once(
                "The generative type of the model has not been set yet as the "
                "tokeniser has not been loaded.",
                level=logging.DEBUG,
            )
            return None
        elif self.benchmark_config.generative_type is not None:
            type_ = self.benchmark_config.generative_type
        elif self.model_config.param in {"thinking"}:
            type_ = GenerativeType.REASONING
        elif self.model_config.param in {"no-thinking"}:
            type_ = GenerativeType.INSTRUCTION_TUNED
        elif (
            hasattr(self, "end_of_reasoning_token")
            and self.end_of_reasoning_token is not None
        ):
            type_ = GenerativeType.REASONING
        elif (
            has_chat_template(tokeniser=self._tokeniser)
            or "instruct" in self.model_config.model_id.lower()
        ):
            type_ = GenerativeType.INSTRUCTION_TUNED
        else:
            type_ = GenerativeType.BASE
        log_once(
            f"Detected generative type {type_.name!r} for model "
            f"{self.model_config.model_id!r}",
            level=logging.DEBUG,
        )
        return type_

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        """The function used to extract the labels from the generated output.

        Returns:
            The function used to extract the labels from the generated output.
        """
        match self.dataset_config.task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                return partial(
                    sequence_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                    model_config=self.model_config,
                    first_label_token_mapping=self.buffer["first_label_token_mapping"],
                )
            case TaskGroup.TEXT_TO_TEXT:
                return text_to_text.extract_labels_from_generation
            case TaskGroup.TOKEN_CLASSIFICATION:
                return partial(
                    token_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case TaskGroup.QUESTION_ANSWERING:
                return question_answering.extract_labels_from_generation
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    def prepare_dataset(
        self, dataset: "DatasetDict", task: "Task", itr_idx: int
    ) -> "DatasetDict":
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

        if self.benchmark_config.few_shot:
            few_shot_examples = extract_few_shot_examples(
                dataset=dataset,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
                itr_idx=itr_idx,
            )
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(
                apply_prompt,
                few_shot_examples=few_shot_examples,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                generative_type=self.generative_type,
                always_populate_text_field=True,
                tokeniser=self._tokeniser,
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

    def generate(self, inputs: dict) -> "GenerativeModelOutput":
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.

        Raises:
            InvalidBenchmark:
                If the dataset requires logprobs, but we could not get the first token
                of each label in the dataset.
        """
        # Get stopping tokens
        stop_tokens: list[str] = self.custom_stop_tokens.copy()
        if self.generative_type == GenerativeType.BASE:
            stop_tokens.append("\n\n")
        if self._tokeniser.pad_token_id is not None:
            assert isinstance(self._tokeniser.pad_token, str), (
                f"The pad token for the model {self.model_config.model_id!r} "
                f"is not a string, which is unexpected: {self._tokeniser.pad_token!r}."
            )
            stop_tokens.append(self._tokeniser.pad_token)
        if self._tokeniser.eos_token_id is not None:
            assert isinstance(self._tokeniser.eos_token, str), (
                f"The EOS token for the model {self.model_config.model_id!r} "
                f"is not a string, which is unexpected: {self._tokeniser.eos_token!r}."
            )
            stop_tokens.append(self._tokeniser.eos_token)
            if self._tokeniser.pad_token_id is None:
                self._tokeniser.pad_token_id = self._tokeniser.eos_token_id
                self._tokeniser.pad_token = self._tokeniser.eos_token
        if self.end_of_chat_token_ids is not None:
            end_of_chat_token = self._tokeniser.decode(
                self.end_of_chat_token_ids
            ).strip()
            if end_of_chat_token:
                stop_tokens.append(end_of_chat_token)

        # Get the mapping from labels to the first token in the label. We call this each
        # time we generate a new dataset since the dataset config can change
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokeniser=self._tokeniser,
            generative_type=self.generative_type,
            log_metadata=self.log_metadata,
        )
        if (
            not self.buffer["first_label_token_mapping"]
            and self.dataset_config.task.requires_logprobs
        ):
            raise InvalidBenchmark(
                "The dataset requires logprobs, but we encountered an error when "
                "trying to get the first token of each label in the dataset. You can "
                "try running this benchmark with the --verbose flag to see what the "
                "error was. Skipping this evaluation."
            )

        structured_generation_schema = None
        if (
            self.dataset_config.task.uses_structured_output
            or (self.dataset_config.task.uses_logprobs and self.dataset_config.labels)
        ) and self.generative_type == GenerativeType.REASONING:
            structured_outputs = None
            log_once(
                "The dataset uses structured output, but we are not using it as the "
                f"model {self.model_config.model_id!r} is a reasoning model.",
                level=logging.DEBUG,
            )
        elif self.dataset_config.task.uses_structured_output:
            ner_tag_names = list(self.dataset_config.prompt_label_mapping.values())
            keys_and_their_types: dict[str, t.Any] = {
                tag_name: (conlist(str, max_length=5), ...)
                for tag_name in ner_tag_names
            }
            answer_format_class = create_model("AnswerFormat", **keys_and_their_types)
            structured_generation_schema = answer_format_class.model_json_schema()
            log_once(
                "Using structured generation with the JSON schema: "
                f"{json.dumps(structured_generation_schema)}",
                level=logging.DEBUG,
            )
            structured_outputs = StructuredOutputsParams(
                json=structured_generation_schema
            )
        elif (
            self.dataset_config.task.uses_logprobs
            and self.dataset_config.labels
            and self.buffer.get("first_label_token_mapping", False)
        ):
            choice_labels = [
                self.dataset_config.prompt_label_mapping[label]
                for label in self.dataset_config.labels
            ]
            if isinstance(self.buffer["first_label_token_mapping"], dict):
                choice_labels = [
                    self.buffer["first_label_token_mapping"][label]
                    for label in choice_labels
                ]
            structured_outputs = StructuredOutputsParams(choice=choice_labels)
            log_once(
                "Using structured generation with the choices: "
                f"{structured_outputs.choice!r}.",
                level=logging.DEBUG,
            )
        else:
            structured_outputs = None
            log_once(
                "Not using structured generation as the dataset does not require it.",
                level=logging.DEBUG,
            )

        # Define the parameters used for vLLM generation
        max_tokens: int = (
            REASONING_MAX_TOKENS
            if self.generative_type == GenerativeType.REASONING
            else self.dataset_config.max_generated_tokens
        )
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            logprobs=MAX_VLLM_LOGPROBS
            if self.buffer["first_label_token_mapping"]
            else None,
            temperature=0.0,
            stop=[stop_token for stop_token in stop_tokens if stop_token],
            structured_outputs=structured_outputs,
        )

        # If any of the prompts are empty then we need to replace them with a BOS token
        # so that the vLLM model can generate from them
        prompts: c.Sequence[str] = inputs["text"]
        if any(len(prompt) == 0 for prompt in prompts):
            log("Found empty prompts, replacing with BOS token.", level=logging.DEBUG)
            prompts = [
                prompt if len(prompt) > 0 else str(self._tokeniser.bos_token)
                for prompt in prompts
            ]

        # Strip the prompts if the model's tokeniser requires it
        labels_to_be_generated = list(self.dataset_config.prompt_label_mapping.values())
        if len(labels_to_be_generated) == 0:
            labels_to_be_generated = ["negative", "positive"]
        if self.generative_type == GenerativeType.BASE and should_prompts_be_stripped(
            labels_to_be_generated=labels_to_be_generated, tokeniser=self._tokeniser
        ):
            log_once(
                f"Stripping prompts for model {self.model_config.model_id!r}.",
                level=logging.DEBUG,
            )
            prompts = [prompt.strip() for prompt in prompts]

        # Truncate the prompts if needed, but only if it's not a reasoning model
        if self.generative_type != GenerativeType.REASONING:
            max_tokens_per_prompt = (
                min(self._tokeniser.model_max_length, MAX_CONTEXT_LENGTH) - max_tokens
            )
            tokenized_prompts = self._tokeniser(
                text=list(prompts), truncation=True, max_length=max_tokens_per_prompt
            )
            prompts = self._tokeniser.batch_decode(
                sequences=tokenized_prompts.input_ids, skip_special_tokens=True
            )

        # Generate sequences using vLLM
        input_is_a_test = len(prompts) == 1 and len(set(prompts[0])) == 1
        num_attempts = 3
        truncation_attempts = 1
        for _ in range(num_attempts):
            try:
                raw_outputs = self._model.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False if input_is_a_test else get_pbar,
                    lora_request=self.buffer.get("lora_request"),
                )
                break
            except TypeError as e:
                log(
                    f"Encountered error during vLLM generation: {str(e)}. Retrying...",
                    level=logging.DEBUG,
                )
                sleep(1)
            except ValueError as e:
                # Truncate the prompts if they are too long for the model
                truncate_error_messages = [
                    r"prompt \(length [0-9]+\) is longer than the maximum model length"
                ]
                if any(
                    re.search(pattern, str(e), flags=re.IGNORECASE) is not None
                    for pattern in truncate_error_messages
                ):
                    log(
                        "Prompts are too long, so truncating them and trying again...",
                        level=logging.WARNING,
                    )
                    log(f"The error message was: {str(e)}", level=logging.DEBUG)

                    # If we have already tried truncating the prompts a few times, then
                    # we truncate a bit more aggressively
                    extra_truncation = 50 * truncation_attempts
                    truncation_attempts += 1

                    tokenized_prompts = self._tokeniser(
                        text=prompts,
                        truncation=True,
                        max_length=max(
                            min(self._tokeniser.model_max_length, MAX_CONTEXT_LENGTH)
                            - max_tokens
                            - extra_truncation,
                            0,
                        ),
                    )
                    prompts = self._tokeniser.batch_decode(
                        sequences=tokenized_prompts.input_ids, skip_special_tokens=True
                    )
                else:
                    raise InvalidBenchmark(
                        f"An error occurred during vLLM generation: {str(e)}"
                    ) from e
        else:
            raise InvalidBenchmark(
                f"Could not generate sequences after {num_attempts} attempts."
            )

        # When we shorten the prompts then some residual model outputs persist, so we
        # need to filter these out
        num_extra_outputs = len(raw_outputs) - len(prompts)
        if num_extra_outputs > 0:
            raw_outputs = raw_outputs[num_extra_outputs:]
            if not all(
                raw_output.prompt == prompt
                for raw_output, prompt in zip(raw_outputs, prompts)
            ):
                raise InvalidBenchmark(
                    f"The prompts and the model outputs do not match. There were "
                    f"{num_extra_outputs!r} extra outputs."
                )
            else:
                log(
                    f"Filtered out {num_extra_outputs:,} extra outputs from the model, "
                    "which occured as we interupted the generation when we truncated "
                    "the prompts.",
                    level=logging.DEBUG,
                )

        # Parse the raw model outputs. We keep the special tokens for now, as we need
        # them to potentially remove reasoning content and stop tokens
        completion_ids: c.Sequence[c.Sequence[int]] = [
            list(output.outputs[0].token_ids) for output in raw_outputs
        ]
        completions = self._tokeniser.batch_decode(
            sequences=[
                torch.LongTensor(completion_id) for completion_id in completion_ids
            ],
            skip_special_tokens=False,
        )
        if (
            self.end_of_reasoning_token is not None
            and self.generative_type == GenerativeType.REASONING
        ):
            num_samples_without_eor_token = 0
            for idx in range(len(completions)):
                if (
                    isinstance(self.end_of_reasoning_token, str)
                    and self.end_of_reasoning_token in completions[idx]
                ):
                    completions[idx] = completions[idx].split(
                        self.end_of_reasoning_token
                    )[-1]
                elif isinstance(
                    self.end_of_reasoning_token, re.Pattern
                ) and self.end_of_reasoning_token.search(completions[idx]):
                    completions[idx] = self.end_of_reasoning_token.split(
                        completions[idx]
                    )[-1]
                else:
                    num_samples_without_eor_token += 1
                    completions[idx] = ""
            if num_samples_without_eor_token > 0:
                log_once(
                    f"The model {self.model_config.model_id!r} is a reasoning "
                    "model, but the generated output did not contain the end of "
                    f"reasoning token ({self.end_of_reasoning_token!r}) in "
                    f"{num_samples_without_eor_token:,}/{len(completions):,} of "
                    "the samples. Using an empty string for all these samples "
                    "instead.",
                    level=(
                        logging.WARNING
                        if num_samples_without_eor_token / len(completions) > 0.5
                        else logging.DEBUG
                    ),
                )
        stop_token_pattern = re.compile(
            "|".join(re.escape(stop_token) for stop_token in stop_tokens)
        )
        completions = [
            re.split(pattern=stop_token_pattern, string=completion)[0].strip()
            for completion in completions
        ]

        # Remove all the special tokens from the completions, if any are present
        completion_ids = self._tokeniser(text=completions).input_ids
        completions = self._tokeniser.batch_decode(
            sequences=completion_ids, skip_special_tokens=True
        )

        # Sanity check
        if len(completions) != len(prompts):
            raise InvalidBenchmark(
                f"Expected {len(prompts):,} completions, but got {len(completions):,}."
            )

        # Add logprobs scores to the output
        if self.buffer["first_label_token_mapping"]:
            scores: c.Sequence[c.Sequence[c.Sequence[tuple[str, float]]]] = [
                [
                    [
                        (obj.decoded_token or "", obj.logprob)
                        for obj in token_logprobs_dict.values()
                    ]
                    for token_logprobs_dict in raw_output.outputs[0].logprobs or list()
                ]
                for raw_output in raw_outputs
            ]
            output = GenerativeModelOutput(sequences=completions, scores=scores)
        else:
            output = GenerativeModelOutput(sequences=completions)

        return output

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
        using_api = (
            benchmark_config.api_base is not None
            or benchmark_config.api_version is not None
        )
        if using_api:
            return False

        model_id_components = split_model_id(model_id=model_id)
        model_id = model_id_components.model_id
        revision = model_id_components.revision

        model_info = get_model_repo_info(
            model_id=model_id,
            revision=revision,
            api_key=benchmark_config.api_key,
            cache_dir=benchmark_config.cache_dir,
            trust_remote_code=benchmark_config.trust_remote_code,
            requires_safetensors=benchmark_config.requires_safetensors,
            run_with_cli=benchmark_config.run_with_cli,
        )
        return (
            model_info is not None
            and model_info.pipeline_tag in GENERATIVE_PIPELINE_TAGS
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
            inference_backend=InferenceBackend.VLLM,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=model_info.adapter_base_model_id,
        )

        return model_config

    @property
    def data_collator(self) -> c.Callable[[c.Sequence[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for vLLM models."
        )

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for vLLM models."
        )


def load_model_and_tokeniser(
    model_config: "ModelConfig", benchmark_config: "BenchmarkConfig"
) -> tuple["LLM", "PreTrainedTokenizer"]:
    """Load the model and tokeniser.

    Args:
        model_config:
            The model configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        A pair (model, tokeniser), with the loaded model and tokeniser
    """
    # Prefer base model ID if the model is an adapter - the adapter will be added on
    # during inference in this case
    model_id = model_config.adapter_base_model_id or model_config.model_id
    revision = (
        model_config.revision if model_config.adapter_base_model_id is None else "main"
    )

    hf_model_config = load_hf_model_config(
        model_id=model_id,
        num_labels=0,
        id2label=HashableDict(),
        label2id=HashableDict(),
        revision=revision,
        model_cache_dir=model_config.model_cache_dir,
        api_key=benchmark_config.api_key,
        trust_remote_code=benchmark_config.trust_remote_code,
        run_with_cli=benchmark_config.run_with_cli,
    )

    quantization = None
    if hasattr(hf_model_config, "quantization_config"):
        quantization = hf_model_config.quantization_config.get("quant_method")

    # The quantised models require extra dependencies
    if quantization == "gptq" and (
        importlib.util.find_spec("auto_gptq") is None
        or importlib.util.find_spec("optimum") is None
    ):
        raise NeedsExtraInstalled(extra="quantization")
    if quantization == "awq" and importlib.util.find_spec("awq") is None:
        raise NeedsExtraInstalled(extra="quantization")

    # Start with dtype being the "auto" vLLM dtype
    dtype: str | torch.dtype = "auto"

    # Choose bf16 over fp16 if the model is a fp32 model and the GPU supports it
    if hf_model_config.dtype == torch.float32:
        if torch.cuda.is_bf16_supported():
            log(
                "You are loading a model with dtype FP32, which we will convert to "
                "BF16 as FP32 is not supported by vLLM and BF16 is supported by your "
                "GPU.",
                level=logging.WARNING,
            )
            dtype = torch.bfloat16
        else:
            log(
                "You are loading a model with dtype FP32, which we will convert to "
                "FP16 as FP32 is not supported by vLLM and BF16 is not supported by "
                "your GPU.",
                level=logging.WARNING,
            )
            dtype = torch.float16

    # If the model is a quantized model, we might need to change the dtype
    if quantization == "mxfp4" and hf_model_config.dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        log(
            "You are loading a quantized model where `dtype` has not been set. "
            f"Setting dtype to {dtype!r}.",
            level=logging.DEBUG,
        )
    elif quantization is not None and hf_model_config.dtype != torch.float16:
        log(
            "You are loading a quantized model with dtype "
            f"{hf_model_config.dtype}, which vLLM does not support. Setting "
            "dtype to float16 instead.",
            level=logging.WARNING,
        )
        dtype = torch.float16

    # If the model is a bf16 model, we need to check the CUDA compute capability
    if hf_model_config.dtype == torch.bfloat16:
        min_cuda_compute_capability = get_min_cuda_compute_capability()
        required_capability = VLLM_BF16_MIN_CUDA_COMPUTE_CAPABILITY

        if min_cuda_compute_capability is not None:
            if min_cuda_compute_capability < required_capability:
                log(
                    f"You are loading a model with dtype {hf_model_config.dtype}, "
                    "which vLLM only supports for CUDA devices with CUDA compute "
                    f"capability >={required_capability}. You are using one or more "
                    f"devices with compute capability {min_cuda_compute_capability}. "
                    "Setting dtype to float16 instead.",
                    level=logging.WARNING,
                )
                dtype = torch.float16

    if model_config.adapter_base_model_id is not None:
        download_dir = str(Path(model_config.model_cache_dir) / "base_model")
    else:
        download_dir = str(model_config.model_cache_dir)

    potential_max_model_length_config_names = [
        "max_position_embeddings",
        "max_sequence_length",
        "model_max_length",
        "n_positions",
    ]
    true_max_model_len_candidates: list[int] = list()
    for config_name in potential_max_model_length_config_names:
        if hasattr(hf_model_config, config_name):
            model_len = getattr(hf_model_config, config_name)
            if model_len is not None:
                true_max_model_len_candidates.append(model_len)

    if len(true_max_model_len_candidates) > 0:
        true_max_model_len = min(true_max_model_len_candidates)
    else:
        true_max_model_len = MAX_CONTEXT_LENGTH

    tokeniser = load_tokeniser(
        model_id=model_config.model_id,
        revision=model_config.revision,
        adapter_base_model_id=model_config.adapter_base_model_id,
        trust_remote_code=benchmark_config.trust_remote_code,
        model_max_length=true_max_model_len,
        model_config=model_config,
        token=get_hf_token(api_key=benchmark_config.api_key),
    )
    vllm_tokenisation_params = get_vllm_tokenisation_params(
        tokeniser=tokeniser, model_config=model_config
    )

    clear_vllm()

    try:
        model = LLM(
            model=(
                model_id
                if internet_connection_available()
                else resolve_model_path(download_dir=download_dir)
            ),
            tokenizer=(
                model_id
                if internet_connection_available()
                else resolve_model_path(download_dir=download_dir)
            ),
            gpu_memory_utilization=benchmark_config.gpu_memory_utilization,
            max_model_len=min(true_max_model_len, MAX_CONTEXT_LENGTH),
            download_dir=download_dir,
            trust_remote_code=benchmark_config.trust_remote_code,
            revision=revision,
            seed=4242,
            distributed_executor_backend="mp",
            tensor_parallel_size=torch.cuda.device_count(),
            disable_custom_all_reduce=True,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=True,
            # TEMP: Prefix caching isn't supported with sliding window in vLLM yet,
            # so we disable it for now
            enable_prefix_caching=False,
            enable_lora=model_config.adapter_base_model_id is not None,
            max_lora_rank=256,
            **vllm_tokenisation_params,
        )
    except (RuntimeError, ValueError, OSError) as e:
        if "awaiting a review from the repo authors" in str(e):
            raise InvalidModel(
                f"The model {model_id!r} is awaiting a review from the repository "
                "authors. Please try again later."
            ) from e
        elif "trust_remote_code" in str(e):
            raise InvalidModel(
                f"Loading the model {model_id!r} needs to trust remote code. "
                "If you trust the suppliers of this model, then you can enable "
                "this by setting the `--trust-remote-code` flag."
            ) from e
        elif "See stack trace for root cause." in str(
            e
        ) or "See root cause above." in str(e):
            msg = (
                f"The model {model_id!r} could not be loaded, but vLLM did not "
                "mention exactly what happened. "
            )
            msg += (
                (
                    "Since you're running in verbose mode, you might see a descriptive "
                    "error above already. Note however that if the error message urges "
                    "you to set the environment variable `VLLM_ATTENTION_BACKEND` to "
                    "'FLEX_ATTENTION', please try setting it to 'FLASH_ATTN' first, as "
                    "that often solves the issue, whereas 'FLEX_ATTENTION' usually "
                    "doesn't. If you don't see any descriptive error above, then you "
                    "can try "
                )
                if benchmark_config.verbose
                else "Try "
            )
            msg += (
                "re-running the benchmark with the environment variable `FULL_LOG` "
                "set to `1` to see the full stack trace. E.g., "
                f"`FULL_LOG=1 euroeval --model {model_id}`."
            )
            raise InvalidModel(msg) from e
        raise InvalidModel(
            f"The model {model_id!r} could not be loaded. The error was {e!r}."
        ) from e

    model.config = hf_model_config

    return model, tokeniser


def load_tokeniser(
    model_id: str,
    revision: str,
    adapter_base_model_id: str | None,
    trust_remote_code: bool,
    model_max_length: int,
    model_config: "ModelConfig",
    token: str | bool,
) -> "PreTrainedTokenizer":
    """Load the tokeniser.

    Args:
        model_id:
            The model identifier.
        revision:
            The revision of the model.
        adapter_base_model_id:
            The base model ID for the adapter model. Can be None if the model is not an
            adapter model.
        trust_remote_code:
            Whether to trust remote code.
        model_max_length:
            The maximum length of the model.
        model_config:
            The model configuration.
        token:
            The Hugging Face API token.

    Returns:
        The loaded tokeniser.
    """
    revision = revision if adapter_base_model_id is None else "main"
    config = AutoConfig.from_pretrained(
        adapter_base_model_id or model_id,
        revision=revision,
        cache_dir=model_config.model_cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
        local_files_only=not internet_connection_available(),
    )
    num_retries = 5
    for _ in range(num_retries):
        try:
            # Mistral instruction-tuned models need a custom tokeniser
            if model_id.startswith("mistralai/") and "base" not in model_id.lower():
                tokeniser = MistralCommonTokenizer.from_pretrained(
                    model_id,
                    padding_side="left",
                    truncation_side="left",
                    model_max_length=model_max_length,
                    token=token,
                )
                break
            tokeniser = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False if model_config.param == "slow-tokenizer" else True,
                verbose=False,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                truncation_side="left",
                model_max_length=model_max_length,
                cache_dir=model_config.model_cache_dir,
                config=config,
                token=token,
                local_files_only=not internet_connection_available(),
            )
            break
        except (json.JSONDecodeError, OSError, TypeError) as e:
            if adapter_base_model_id is None or model_id == adapter_base_model_id:
                raise InvalidModel(
                    f"Could not load tokeniser for model {model_id!r}. The error was "
                    f"{str(e)}."
                ) from e
            log(
                f"Could not load tokeniser for {model_id!r}. Falling back to "
                f"{adapter_base_model_id!r}.",
                level=logging.DEBUG,
            )
            model_id = adapter_base_model_id
        except (TimeoutError, RequestError):
            log(
                f"Couldn't load tokeniser for {model_id!r}. Retrying.",
                level=logging.WARNING,
            )
            sleep(5)
            continue
        except (KeyError, ValueError) as e:
            if "mistral" in str(e).lower():
                tokeniser = MistralCommonTokenizer.from_pretrained(
                    model_id,
                    padding_side="left",
                    truncation_side="left",
                    model_max_length=model_max_length,
                    token=token,
                )
                break
            raise InvalidModel(
                f"Could not load tokeniser for model {model_id!r}. The error was "
                f"{str(e)}."
            ) from e
    else:
        raise InvalidModel(
            f"Could not load tokeniser for model {model_id!r} after {num_retries} "
            "attempts."
        )

    # Ensure that BOS, EOS and PAD tokens are set
    if not isinstance(tokeniser, MistralCommonTokenizer):
        tokeniser.bos_token, tokeniser.bos_token_id = get_bos_token(tokeniser=tokeniser)
        tokeniser.eos_token, tokeniser.eos_token_id = get_eos_token(tokeniser=tokeniser)
        tokeniser.pad_token, tokeniser.pad_token_id = get_pad_token(tokeniser=tokeniser)

    return tokeniser


def clear_vllm() -> None:
    """Clear the GPU memory used by the vLLM model, enabling re-initialisation."""
    with contextlib.suppress(ValueError):
        destroy_model_parallel()
        destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    clear_memory()


def get_end_of_reasoning_token(
    model: "LLM", tokeniser: "PreTrainedTokenizer", model_config: "ModelConfig"
) -> str | re.Pattern | None:
    """Get the end-of-reasoning token for a generative model.

    Args:
        model:
            The vLLM model.
        tokeniser:
            The tokeniser.
        model_config:
            The model configuration.

    Returns:
        The end of reasoning token, or None if it could not be found.
    """
    model_id = model_config.model_id

    # Create a prompt to check if the model uses the reasoning tokens
    prompt = "What is your name?"
    if has_chat_template(tokeniser=tokeniser):
        extra_kwargs = dict()
        if model_config.param in {"thinking", "no-thinking"}:
            extra_kwargs["enable_thinking"] = model_config.param == "thinking"
        templated_prompt = apply_chat_template(
            conversation=[dict(role="user", content=prompt)],
            tokeniser=tokeniser,
            tokenise=False,
            add_generation_prompt=True,
            **extra_kwargs,
        )
        assert isinstance(templated_prompt, str)
        prompt = templated_prompt

    # Check that the beginning-of-reasoning token is actually used by the model
    output = model.generate(
        prompts=[prompt], sampling_params=SamplingParams(max_tokens=10), use_tqdm=False
    )[0]
    completion = tokeniser.decode(token_ids=output.outputs[0].token_ids)
    bor_reasoning_matches = [
        (bor_token, eor_token)
        for bor_token, eor_token in REASONING_TOKENS
        if (
            (
                isinstance(bor_token, str)
                and (bor_token in prompt or bor_token in completion)
            )
            or (
                isinstance(bor_token, re.Pattern)
                and (
                    bor_token.search(prompt) is not None
                    or bor_token.search(completion) is not None
                )
            )
        )
    ]
    if not bor_reasoning_matches:
        log_once(
            f"The model {model_id!r} did not generate any beginning-of-reasoning "
            "tokens in the prompt or the completion. Assuming the model is not a "
            "reasoning model.",
            level=logging.DEBUG,
        )
        return None

    # Check that the end-of-reasoning token is actually used by the model
    output = model.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(max_tokens=REASONING_MAX_TOKENS),
        use_tqdm=False,
    )[0]
    completion = tokeniser.decode(token_ids=output.outputs[0].token_ids)
    eor_reasoning_matches = [
        (bor_token, eor_token)
        for bor_token, eor_token in bor_reasoning_matches
        if (
            (isinstance(eor_token, str) and eor_token in completion)
            or (
                isinstance(eor_token, re.Pattern)
                and eor_token.search(completion) is not None
            )
        )
    ]
    if not eor_reasoning_matches:
        log_once(
            f"The model {model_id!r} did not generate any end-of-reasoning "
            "tokens in the prompt or the completion, even though it generated "
            "the beginning-of-reasoning tokens "
            f"{[bor_token for bor_token, _ in bor_reasoning_matches]!r}. "
            "This is probably not correct, so please report this issue.",
            level=logging.WARNING,
        )
        return None

    if len(eor_reasoning_matches) > 1:
        log_once(
            f"Found multiple reasoning tokens {eor_reasoning_matches} for "
            f"model {model_id!r}. Using {eor_reasoning_matches[0]!r} as "
            "the reasoning token. If this is not the correct reasoning token, "
            "please report this issue.",
            level=logging.WARNING,
        )

    bor_token, eor_token = eor_reasoning_matches[0]

    bor_token_logging: str = (
        bor_token if isinstance(bor_token, str) else bor_token.pattern
    )
    eor_token_logging: str = (
        eor_token if isinstance(eor_token, str) else eor_token.pattern
    )
    log_once(
        f"Detected beginning-of-reasoning token {bor_token_logging!r} and "
        f"end-of-reasoning token {eor_token_logging!r} for model {model_id!r}.",
        level=logging.DEBUG,
    )

    return eor_token


def get_custom_stop_tokens(
    model: "LLM",
    tokeniser: "PreTrainedTokenizer",
    model_id: str,
    generative_type: GenerativeType | None,
) -> list[str]:
    """Get the stop tokens for a generative model.

    Args:
        model:
            The vLLM model.
        tokeniser:
            The tokeniser.
        model_id:
            The model ID.
        generative_type:
            The generative type of the model.

    Returns:
        A list of stop tokens.
    """
    candidate_stop_tokens = CUSTOM_STOP_TOKENS

    prompt = "Hello"
    if has_chat_template(tokeniser=tokeniser):
        templated_prompt = apply_chat_template(
            conversation=[dict(role="user", content=prompt)],
            tokeniser=tokeniser,
            tokenise=False,
            add_generation_prompt=True,
            enable_thinking=generative_type == GenerativeType.REASONING,
        )
        assert isinstance(templated_prompt, str)
        prompt = templated_prompt

    max_tokens = (
        REASONING_MAX_TOKENS if generative_type == GenerativeType.REASONING else 10
    )
    output = model.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        use_tqdm=False,
    )[0]
    completion = tokeniser.decode(token_ids=output.outputs[0].token_ids)

    stop_tokens = [
        stop_token
        for stop_token in candidate_stop_tokens
        if stop_token in prompt or stop_token in completion
    ]
    if stop_tokens:
        log(
            f"Found the following custom stop tokens for model {model_id!r}: "
            f"{stop_tokens}.",
            level=logging.DEBUG,
        )
    else:
        log(f"Found no custom stop tokens for model {model_id!r}.", level=logging.DEBUG)

    return stop_tokens


def get_vllm_tokenisation_params(
    tokeniser: "PreTrainedTokenizer", model_config: "ModelConfig"
) -> dict[str, t.Any]:
    """Get the tokenisation parameters for vLLM.

    Args:
        tokeniser:
            The tokeniser.
        model_config:
            The model configuration.

    Returns:
        A dictionary of tokenisation parameters to pass to vLLM.
    """
    if isinstance(tokeniser, MistralCommonTokenizer):
        tokeniser_mode = "mistral"
    elif model_config.param == "slow-tokenizer":
        tokeniser_mode = "slow"
    else:
        tokeniser_mode = "auto"

    if isinstance(tokeniser, MistralCommonTokenizer):
        config_format = "mistral"
    else:
        config_format = "auto"

    if isinstance(tokeniser, MistralCommonTokenizer):
        load_format = "mistral"
    else:
        load_format = "auto"

    return dict(
        tokenizer_mode=tokeniser_mode,
        config_format=config_format,
        load_format=load_format,
    )
