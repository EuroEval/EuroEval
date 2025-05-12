"""Generative models using the vLLM inference framework."""

import collections.abc as c
import contextlib
import importlib.util
import json
import logging
import os
import re
import sys
import typing as t
from functools import partial
from pathlib import Path
from time import sleep
from types import MethodType

import torch
from datasets import DatasetDict
from huggingface_hub import snapshot_download
from pydantic import conlist, create_model
from tqdm.auto import tqdm
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from urllib3.exceptions import RequestError

from ..constants import (
    GENERATIVE_PIPELINE_TAGS,
    MAX_CONTEXT_LENGTH,
    MAX_LOGPROBS,
    MERGE_TAGS,
    REASONING_MAX_TOKENS,
    TASKS_USING_JSON,
    VLLM_BF16_MIN_CUDA_COMPUTE_CAPABILITY,
)
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
from ..exceptions import (
    InvalidBenchmark,
    InvalidModel,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..generation_utils import apply_prompt, extract_few_shot_examples
from ..languages import get_all_languages
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..tokenization_utils import (
    get_bos_token,
    get_end_of_chat_token_ids,
    get_eos_token,
    get_first_label_token_mapping,
    should_prompts_be_stripped,
)
from ..types import ExtractLabelsFunction
from ..utils import (
    clear_memory,
    create_model_cache_dir,
    get_min_cuda_compute_capability,
    log_once,
)
from .hf import HuggingFaceEncoderModel, get_model_repo_info, load_hf_model_config

if t.TYPE_CHECKING or importlib.util.find_spec("vllm") is not None:
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.lora.request import LoRARequest

if t.TYPE_CHECKING or importlib.util.find_spec("outlines") is not None:
    from outlines.models.vllm import adapt_tokenizer
    from outlines.processors.structured import JSONLogitsProcessor

if t.TYPE_CHECKING or importlib.util.find_spec("ray") is not None:
    import ray

logger = logging.getLogger("euroeval")


class VLLMModel(HuggingFaceEncoderModel):
    """A generative model using the vLLM inference framework."""

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the vLLM model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
        """
        if (
            importlib.util.find_spec("vllm") is None
            or importlib.util.find_spec("ray") is None
        ):
            raise NeedsExtraInstalled(extra="generative")

        model, tokenizer = load_model_and_tokenizer(
            model_config=model_config, benchmark_config=benchmark_config
        )
        self._model: LLM = model
        self._tokenizer: PreTrainedTokenizer = tokenizer
        self.end_of_reasoning_token_id = get_end_of_reasoning_token_id(
            model=self._model, tokenizer=self._tokenizer, model_id=model_config.model_id
        )

        # We specify `HuggingFaceEncoderModel` here instead of `VLLMModel`, as we want
        # to call the `__init__` method of the `BenchmarkModule` class.
        super(HuggingFaceEncoderModel, self).__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        self.buffer |= dict(
            instruction_model=self._tokenizer.chat_template is not None,
            first_label_token_mapping=get_first_label_token_mapping(
                dataset_config=self.dataset_config,
                model_config=self.model_config,
                tokenizer=self._tokenizer,
                generative_type=self.generative_type,
            ),
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
        """Clean up the model and tokenizer."""
        clear_vllm()
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer

    @property
    def generative_type(self) -> GenerativeType | None:
        """Get the generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        if not hasattr(self, "_tokenizer"):
            return None
        elif self.end_of_reasoning_token_id is not None:
            return GenerativeType.REASONING
        elif self._tokenizer.chat_template is not None:
            return GenerativeType.INSTRUCTION_TUNED
        else:
            return GenerativeType.BASE

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
        self, dataset: DatasetDict, task: Task, itr_idx: int
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
                dataset=dataset, dataset_config=self.dataset_config, itr_idx=itr_idx
            )
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(
                apply_prompt,
                few_shot_examples=few_shot_examples,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                instruction_model=self.buffer["instruction_model"],
                always_populate_text_field=True,
                tokenizer=self._tokenizer,
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        # Define which tokens to use as stopping criteria. We want to use the padding
        # token, end-of-sentence token, and a double newline if the model isn't
        # instruction tuned (since these separate the few-shot examples in the input in
        # this case)
        stop_tokens: list[str] = list()
        if self.buffer["instruction_model"] is False:
            stop_tokens.append("\n\n")
        if self._tokenizer.pad_token_id is not None:
            stop_tokens.append(self._tokenizer.pad_token)
        if self._tokenizer.eos_token_id is not None:
            stop_tokens.append(self._tokenizer.eos_token)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                self._tokenizer.pad_token = self._tokenizer.eos_token
        if (
            self._tokenizer.bos_token_id is not None
            and self._tokenizer.pad_token_id is None
        ):
            self._tokenizer.pad_token_id = self._tokenizer.bos_token_id
            self._tokenizer.pad_token = self._tokenizer.bos_token
        elif (
            self._tokenizer.eos_token_id is not None
            and self._tokenizer.pad_token_id is None
        ):
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._tokenizer.pad_token = self._tokenizer.eos_token
        elif self._tokenizer.pad_token_id is None:
            pad_token_candidates = ["<pad>", "[pad]", "<|endoftext|>", "<|im_end|>"]
            pad_token_candidates.extend([c.upper() for c in pad_token_candidates])
            for candidate in pad_token_candidates:
                if candidate in self._tokenizer.get_vocab():
                    pad_token_id = self._tokenizer.get_vocab()[candidate]
                    self._tokenizer.pad_token = candidate
                    self._tokenizer.pad_token_id = pad_token_id
                    break
            else:
                raise InvalidModel(
                    "Could not find a suitable token to use as a padding token, since "
                    "the model does not have a BOS, EOS, or padding token, and does "
                    f"not have any of the following tokens in its vocabulary: "
                    f"{pad_token_candidates}."
                )

        assert self._tokenizer.pad_token_id is not None

        # Add end of chat token as a stopping token, if it exists
        end_of_chat_token_ids = get_end_of_chat_token_ids(tokenizer=self._tokenizer)
        if end_of_chat_token_ids is not None:
            end_of_chat_token = self._tokenizer.decode(end_of_chat_token_ids).strip()
            if end_of_chat_token:
                stop_tokens.append(end_of_chat_token)

        logits_processor = None
        if self.dataset_config.task in TASKS_USING_JSON:
            if self.generative_type == GenerativeType.REASONING:
                log_once(
                    f"The model {self.model_config.model_id!r} is a reasoning model "
                    "and thus does not support structured generation, so we do not "
                    "enable it.",
                    level=logging.DEBUG,
                )
            else:
                ner_tag_names = list(self.dataset_config.prompt_label_mapping.values())
                keys_and_their_types: dict[str, t.Any] = {
                    tag_name: (conlist(str, max_length=5), ...)
                    for tag_name in ner_tag_names
                }
                pydantic_class = create_model("AnswerFormat", **keys_and_their_types)
                logits_processor = JSONLogitsProcessor(
                    schema=pydantic_class,
                    tokenizer=adapt_tokenizer(tokenizer=self._tokenizer),  # type: ignore
                    whitespace_pattern=r" ?",
                )
                log_once(
                    "Using structured generation with the JSON schema "
                    f"{pydantic_class.model_json_schema()}",
                    level=logging.DEBUG,
                )

        # Get the mapping from labels to the first token in the label. We call this each
        # time we generate a new dataset since the dataset config can change
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokenizer=self._tokenizer,
            generative_type=self.generative_type,
        )

        # Define the parameters used for vLLM generation
        max_tokens: int = (
            REASONING_MAX_TOKENS
            if self.generative_type == GenerativeType.REASONING
            else self.dataset_config.max_generated_tokens
        )
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            logprobs=MAX_LOGPROBS if self.buffer["first_label_token_mapping"] else None,
            temperature=0.0,
            stop=[stop_token for stop_token in stop_tokens if stop_token],
            logits_processors=[logits_processor] if logits_processor else None,
        )

        # If any of the prompts are empty then we need to replace them with a BOS token
        # so that the vLLM model can generate from them
        prompts: list[str] = inputs["text"]
        if any(len(prompt) == 0 for prompt in prompts):
            logger.debug("Found empty prompts, replacing with BOS token.")
            prompts = [
                prompt if len(prompt) > 0 else str(self._tokenizer.bos_token)
                for prompt in prompts
            ]

        # Strip the prompts if the model's tokeniser requires it
        labels_to_be_generated = list(self.dataset_config.prompt_label_mapping.values())
        if len(labels_to_be_generated) == 0:
            labels_to_be_generated = ["negative", "positive"]
        if not self.buffer.get(
            "instruction_model", False
        ) and should_prompts_be_stripped(
            labels_to_be_generated=labels_to_be_generated, tokenizer=self._tokenizer
        ):
            log_once(
                f"Stripping prompts for model {self.model_config.model_id!r}.",
                level=logging.DEBUG,
            )
            prompts = [prompt.strip() for prompt in prompts]

        # Generate sequences using vLLM
        input_is_a_test = len(prompts) == 1 and len(set(prompts[0])) == 1
        num_attempts = 3
        for _ in range(num_attempts):
            try:
                raw_outputs = self._model.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    use_tqdm=(not input_is_a_test),
                    lora_request=self.buffer.get("lora_request"),
                )
                break
            except TypeError as e:
                logger.debug(
                    f"Encountered error during vLLM generation: {str(e)}. Retrying..."
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
                    logger.info(
                        "Prompts are too long, so truncating them and trying again..."
                    )
                    logger.debug(f"The error message was: {str(e)}")
                    tokenized_prompts = self._tokenizer(
                        text=prompts,
                        truncation=True,
                        max_length=max(
                            self._tokenizer.model_max_length - max_tokens, 0
                        ),
                    )
                    prompts = self._tokenizer.batch_decode(
                        sequences=tokenized_prompts.input_ids, skip_special_tokens=True
                    )
                else:
                    raise InvalidBenchmark(
                        f"An error occurred during vLLM generation: {str(e)}"
                    )
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
                logger.debug(
                    f"Filtered out {num_extra_outputs:,} extra outputs from the model, "
                    "which occured as we interupted the generation when we truncated "
                    "the prompts."
                )

        # Parse the raw model outputs
        completion_ids: list[list[int]] = [
            output.outputs[0].token_ids for output in raw_outputs
        ]
        if self.end_of_reasoning_token_id in completion_ids[0]:
            completion_ids = [
                token_ids[token_ids.index(self.end_of_reasoning_token_id) + 1 :]
                if self.end_of_reasoning_token_id in token_ids
                else token_ids
                for token_ids in completion_ids
            ]
        completions = self._tokenizer.batch_decode(
            sequences=[
                torch.LongTensor(completion_id) for completion_id in completion_ids
            ],
            skip_special_tokens=True,
        )
        completions = [completion.strip() for completion in completions]

        # Sanity check
        if len(completions) != len(prompts):
            raise InvalidBenchmark(
                f"Expected {len(prompts):,} completions, but got {len(completions):,}."
            )

        # Add logprobs scores to the output
        if self.buffer["first_label_token_mapping"]:
            scores: list[list[list[tuple[str, float]]]] = [
                [
                    [
                        (obj.decoded_token, obj.logprob)
                        for obj in token_logprobs_dict.values()
                    ]
                    for token_logprobs_dict in raw_output.outputs[0].logprobs
                ]
                for raw_output in raw_outputs
            ]
            scores = [
                score_list[
                    raw_output.outputs[0].token_ids.index(
                        self.end_of_reasoning_token_id
                    )
                    + 2 :
                ]
                if self.end_of_reasoning_token_id in raw_output.outputs[0].token_ids
                else score_list
                for raw_output, score_list in zip(raw_outputs, scores)
            ]
            output = GenerativeModelOutput(sequences=completions, scores=scores)
        else:
            output = GenerativeModelOutput(sequences=completions)

        return output

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
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        using_api = (
            benchmark_config.api_base is not None
            or benchmark_config.api_version is not None
        )
        if using_api:
            return False

        model_id, revision = (
            model_id.split("@") if "@" in model_id else (model_id, "main")
        )
        model_info = get_model_repo_info(
            model_id=model_id, revision=revision, benchmark_config=benchmark_config
        )
        return (
            model_info is not None
            and model_info.pipeline_tag in GENERATIVE_PIPELINE_TAGS
        )

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
        model_id, revision = (
            model_id.split("@") if "@" in model_id else (model_id, "main")
        )
        model_info = get_model_repo_info(
            model_id=model_id, revision=revision, benchmark_config=benchmark_config
        )
        if model_info is None:
            raise InvalidModel(f"The model {model_id!r} could not be found.")

        language_mapping = get_all_languages()
        language_codes = list(language_mapping.keys())

        model_config = ModelConfig(
            model_id=model_id,
            revision=revision,
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
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
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


def load_model_and_tokenizer(
    model_config: ModelConfig, benchmark_config: BenchmarkConfig
) -> "tuple[LLM, PreTrainedTokenizer]":
    """Load the model and tokenizer.

    Args:
        model_config:
            The model configuration.
        benchmark_config:
            The benchmark configuration.

    Returns:
        A pair (model, tokenizer), with the loaded model and tokenizer
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
        id2label=dict(),
        label2id=dict(),
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
    if hf_model_config.torch_dtype == torch.float32:
        if torch.cuda.is_bf16_supported():
            logger.info(
                "You are loading a model with dtype FP32, which we will convert to "
                "BF16 as FP32 is not supported by vLLM and BF16 is supported by your "
                "GPU."
            )
            dtype = torch.bfloat16
        else:
            logger.info(
                "You are loading a model with dtype FP32, which we will convert to "
                "FP16 as FP32 is not supported by vLLM and BF16 is not supported by "
                "your GPU."
            )
            dtype = torch.float16

    # If the model is a quantized model, we need to set the dtype to float16
    if quantization is not None and hf_model_config.torch_dtype != torch.float16:
        logger.info(
            "You are loading a quantized model with dtype "
            f"{hf_model_config.torch_dtype}, which vLLM does not support. Setting "
            "dtype to float16 instead."
        )
        dtype = torch.float16

    # If the model is a bf16 model, we need to check the CUDA compute capability
    if hf_model_config.torch_dtype == torch.bfloat16:
        min_cuda_compute_capability = get_min_cuda_compute_capability()
        required_capability = VLLM_BF16_MIN_CUDA_COMPUTE_CAPABILITY

        if min_cuda_compute_capability is not None:
            if min_cuda_compute_capability < required_capability:
                logger.info(
                    "You are loading a model with "
                    f"dtype {hf_model_config.torch_dtype}, "
                    "which vLLM only supports for CUDA devices with"
                    f"CUDA compute capability >={required_capability}. "
                    "You are using one or more devices with "
                    f"compute capability {min_cuda_compute_capability}. "
                    "Setting dtype to float16 instead."
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

    tokenizer = load_tokenizer(
        model_id=model_config.model_id,
        revision=model_config.revision,
        adapter_base_model_id=model_config.adapter_base_model_id,
        trust_remote_code=benchmark_config.trust_remote_code,
        model_max_length=true_max_model_len,
        model_cache_dir=model_config.model_cache_dir,
        token=benchmark_config.api_key or os.getenv("HUGGINGFACE_API_KEY") or True,
    )

    clear_vllm()

    try:
        model = LLM(
            model=model_id,
            tokenizer=model_id,
            gpu_memory_utilization=0.9,
            max_model_len=min(true_max_model_len, MAX_CONTEXT_LENGTH),
            download_dir=download_dir,
            trust_remote_code=benchmark_config.trust_remote_code,
            revision=revision,
            seed=4242,
            distributed_executor_backend=(
                "ray" if torch.cuda.device_count() > 1 else "mp"
            ),
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
        )
    except (RuntimeError, ValueError, OSError) as e:
        if "awaiting a review from the repo authors" in str(e):
            raise InvalidModel(
                f"The model {model_id!r} is awaiting a review from the repository "
                "authors. Please try again later."
            )
        elif "trust_remote_code" in str(e):
            raise InvalidModel(
                f"Loading the model {model_id!r} needs to trust remote code. "
                "If you trust the suppliers of this model, then you can enable "
                "this by setting the `--trust-remote-code` flag."
            )
        raise InvalidModel(
            f"The model {model_id!r} could not be loaded. The error was {e!r}."
        )

    model._run_engine = MethodType(_run_engine_with_fixed_progress_bars, model)
    model.config = hf_model_config

    return model, tokenizer


def load_tokenizer(
    model_id: str,
    revision: str,
    adapter_base_model_id: str | None,
    trust_remote_code: bool,
    model_max_length: int,
    model_cache_dir: str,
    token: str | bool,
) -> "PreTrainedTokenizer":
    """Load the tokenizer.

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
        model_cache_dir:
            The cache directory for the model.
        token:
            The Hugging Face API token.

    Returns:
        The loaded tokenizer.
    """
    revision = revision if adapter_base_model_id is None else "main"
    config = AutoConfig.from_pretrained(
        adapter_base_model_id or model_id,
        revision=revision,
        cache_dir=model_cache_dir,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    num_retries = 5
    for _ in range(num_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                verbose=False,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                truncation_side="left",
                model_max_length=model_max_length,
                config=config,
                token=token,
            )
            break
        except (json.JSONDecodeError, OSError, TypeError) as e:
            if adapter_base_model_id is None or model_id == adapter_base_model_id:
                raise InvalidModel(
                    f"Could not load tokenizer for model {model_id!r}. The error was "
                    f"{str(e)}."
                )
            logger.debug(
                f"Could not load tokenizer for {model_id!r}. Falling back to "
                f"{adapter_base_model_id!r}."
            )
            model_id = adapter_base_model_id
        except (TimeoutError, RequestError):
            logger.info(f"Couldn't load tokenizer for {model_id!r}. Retrying.")
            sleep(5)
            continue
    else:
        raise InvalidModel(
            f"Could not load tokenizer for model {model_id!r} after {num_retries} "
            "attempts."
        )

    # Ensure that BOS, EOS and PAD tokens are set
    tokenizer.bos_token, tokenizer.bos_token_id = get_bos_token(tokenizer=tokenizer)
    tokenizer.eos_token, tokenizer.eos_token_id = get_eos_token(tokenizer=tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _run_engine_with_fixed_progress_bars(
    self: "LLM", use_tqdm: bool
) -> list["RequestOutput"]:
    if use_tqdm:
        num_requests = self.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(
            total=num_requests, leave=False, disable=hasattr(sys, "_called_from_test")
        )
    else:
        pbar = None

    # Run the engine.
    outputs: list["RequestOutput"] = list()
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Sort the outputs by request ID. This is necessary because some requests may be
    # finished earlier than its previous requests.
    outputs = sorted(outputs, key=lambda x: int(x.request_id))

    return outputs


def clear_vllm() -> None:
    """Clear the GPU memory used by the vLLM model, enabling re-initialisation."""
    with contextlib.suppress(ValueError):
        destroy_model_parallel()
        destroy_distributed_environment()
    if ray.is_initialized():
        ray.shutdown()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if ray.is_initialized():
        ray.shutdown()
    clear_memory()


def get_end_of_reasoning_token_id(
    model: "LLM", tokenizer: "PreTrainedTokenizer", model_id: str
) -> int | None:
    """Get the end of reasoning token ID for a generative model.

    This assumes that the reasoning token is of the form <X> and that the end of
    reasoning token is </X> (for X being any string without spaces). We disallow the
    reasoning token to be the same as the beginning-of-sentence token.

    Args:
        model:
            The vLLM model.
        tokenizer:
            The tokenizer.
        model_id:
            The model ID.

    Returns:
        The end of reasoning token ID, or None if it could not be found.
    """
    if tokenizer.chat_template is None:
        prompt = "What is your name?"
    else:
        templated_prompt = tokenizer.apply_chat_template(
            conversation=[dict(role="user", content="What is your name?")],
            add_generation_prompt=True,
            tokenize=False,
        )
        assert isinstance(templated_prompt, str)
        prompt = templated_prompt

    # Generate a completion and remove the BOS token from it, to not confuse it with the
    # potential reasoning token
    model_output = model.generate(
        prompts=[prompt],
        sampling_params=SamplingParams(max_tokens=3, temperature=0.0),
        use_tqdm=False,
    )
    completion = model_output[0].outputs[0].text

    if tokenizer.bos_token is not None:
        if isinstance(tokenizer.bos_token, str):
            prompt = prompt.replace(tokenizer.bos_token, "").strip()
            completion = completion.replace(tokenizer.bos_token, "").strip()
        elif isinstance(tokenizer.bos_token, list):
            for bos_token in tokenizer.bos_token:
                prompt = prompt.replace(bos_token, "").strip()
                completion = completion.replace(bos_token, "").strip()

    # If it doesn't contain a reasoning token, we can't find the end of reasoning token
    prompt_match = re.search(pattern=r"<\w+>", string=prompt)
    completion_match = re.search(pattern=r"<\w+>", string=completion)
    if completion_match is None and prompt_match is None:
        log_once(
            f"Could not find a reasoning token for model {model_id!r}, so assuming "
            "the model is not a reasoning model.",
            level=logging.DEBUG,
        )
        return None

    # Check that the found reasoning token and its associated end-of-reasoning tokens
    # are both special tokens
    elif completion_match is not None:
        reasoning_token = completion_match.group()
    else:
        assert prompt_match is not None
        reasoning_token = prompt_match.group()
    end_of_reasoning_token = f"</{reasoning_token[1:-1]}>"
    special_tokens = [
        decoder_token.content
        for decoder_token in tokenizer.added_tokens_decoder.values()
    ]
    special_tokens.extend(
        [encoder_token for encoder_token in tokenizer.added_tokens_encoder.keys()]
    )
    special_tokens.extend(tokenizer.all_special_tokens)
    if (
        reasoning_token not in special_tokens
        or end_of_reasoning_token not in special_tokens
    ):
        log_once(
            f"Detected reasoning token {reasoning_token!r} and end-of-reasoning "
            f"token {end_of_reasoning_token!r} for model {model_id!r}, but one of "
            "them is not registered as a special token, so assuming it is not a "
            "real reasoning token.",
            level=logging.DEBUG,
        )
        return None

    log_once(
        f"Detected reasoning token {reasoning_token!r} and end-of-reasoning "
        f"token {end_of_reasoning_token!r} for model {model_id!r}.",
        level=logging.DEBUG,
    )

    # Encode the end of reasoning token and return its ID
    end_of_reasoning_token_id = tokenizer.encode(
        text=end_of_reasoning_token, add_special_tokens=False
    )[0]

    return end_of_reasoning_token_id
