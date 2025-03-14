"""Generative models from an inference API, using the LiteLLM framework."""

import collections.abc as c
import itertools as it
import json
import logging
import os
import random
import re
import typing as t
from functools import cached_property, partial
from time import sleep

import litellm
from datasets import DatasetDict
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    HFValidationError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.types.utils import ModelResponse
from requests.exceptions import RequestException
from transformers import Trainer

from ..constants import (
    MAX_LOGPROBS,
    REASONING_MAX_TOKENS,
    TASK_GROUPS_USING_LOGPROBS,
    TASKS_USING_JSON,
)
from ..data_models import BenchmarkConfig, GenerativeModelOutput, ModelConfig, Task
from ..enums import (
    BatchingPreference,
    GenerativeType,
    InferenceBackend,
    ModelType,
    TaskGroup,
)
from ..exceptions import (
    InvalidBenchmark,
    NeedsAdditionalArgument,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..task_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..types import ExtractLabelsFunction
from ..utils import create_model_cache_dir
from .base import BenchmarkModule
from .hf import HuggingFaceEncoderModel, load_hf_model_config, load_tokenizer

logger = logging.getLogger("euroeval")


VOCAB_SIZE_MAPPING = {
    # OpenAI models
    "(text-)?(ada|babbage|curie|davinci)(-001)?": 50_257,
    "(code|text)-davinci-00[2-9]": 50_281,
    "gpt-3.5-turbo(-16k)?(-[0-9]{4})?": 100_256,
    "gpt-4-(32k)?(-[0-9]{4})?": 100_256,
    "gpt-4-[0-9]{4}-preview": 100_256,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 100_256,
    "gpt-4-(vision|turbo)(-preview)?": 100_256,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": 100_256,
    "gpt-4o(-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_019,
    "o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    # Anthropic models
    "claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": -1,
}


MODEL_MAX_LENGTH_MAPPING = {
    # OpenAI models
    "(text-)?(ada|babbage|curie|davinci)(-001)?": 2_050,
    "text-davinci-00[2-9]": 4_098,
    "code-davinci-00[1-9]": 8_002,
    "gpt-3.5-turbo-0613": 4_096,
    "gpt-3.5-turbo(-[0-9]{4})?": 16_385,
    "gpt-3.5-turbo-16k(-[0-9]{4})?": 16_384,
    "gpt-4(-[0-9]{4})?": 8_191,
    "gpt-4-32k(-[0-9]{4})?": 32_767,
    "gpt-4-[0-9]{4}-preview": 128_000,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    "gpt-4-(vision|turbo)(-preview)?": 128_000,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": 4_095,
    "gpt-4o(-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    "o1-(mini|preview)(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    "o1(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_000,
    "o[2-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_000,
    # Anthropic models
    "claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": 200_000,
}


NUM_PARAMS_MAPPING = {
    # OpenAI models
    "(text-)?ada(-001)?": 350_000_000,
    "(text-)?babbage(-001)?": 3_000_000_000,
    "(text-)?curie(-001)?": 13_000_000_000,
    "((text|code)-)?davinci(-00[1-9])?": 175_000_000_000,
    "gpt-(3.5|4)-turbo-((16|32)k)?(-[0-9]{4})?": -1,
    "gpt-4-[0-9]{4}-preview": -1,
    "gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    "gpt-4-(vision|turbo)(-preview)?": -1,
    "gpt-3.5-turbo-instruct(-[0-9]{4})?": -1,
    "gpt-4o(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    "gpt-4o-mini(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    "o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    # Anthropic models
    "claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": -1,
}


REASONING_MODELS = ["o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?"]


class LiteLLMModel(BenchmarkModule):
    """A generative model from LiteLLM."""

    fresh_model = False
    batching_preference = BatchingPreference.SINGLE_SAMPLE
    high_priority = False

    @property
    def generative_type(self) -> GenerativeType | None:
        """Get the generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        if re.fullmatch(
            pattern="|".join(REASONING_MODELS), string=self.model_config.model_id
        ):
            return GenerativeType.REASONING
        else:
            return GenerativeType.INSTRUCTION_TUNED

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        assert "messages" in inputs, "The input must contain a 'messages' key."
        assert len(inputs["messages"]) == 1, (
            "API models only support single-sample batching."
        )
        messages = inputs["messages"][0]

        generation_kwargs: dict[str, t.Any] = dict(
            model=self.model_config.model_id,
            max_completion_tokens=(
                REASONING_MAX_TOKENS
                if self.generative_type == GenerativeType.REASONING
                else self.dataset_config.max_generated_tokens
            ),
            stop=[],
            temperature=0.0,
            seed=4242,
            api_key=self.benchmark_config.api_key,
            api_base=self.benchmark_config.api_base,
            api_version=self.benchmark_config.api_version,
        )

        if self.dataset_config.task.task_group in TASK_GROUPS_USING_LOGPROBS:
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = MAX_LOGPROBS

        if self.dataset_config.task in TASKS_USING_JSON:
            assert "json" in messages[0]["content"].lower(), (
                "Prompt must contain 'json' for JSON tasks."
            )
            generation_kwargs["response_format"] = dict(type="json_object")

        # This drops generation kwargs that are not supported by the model
        litellm.drop_params = True

        # Extract the generated sequences from the model response. Some APIs cannot
        # handle using newlines as stop sequences, so we try both.
        num_attempts = 10
        for _ in range(num_attempts):
            try:
                model_response = litellm.completion(
                    messages=messages, max_retries=3, **generation_kwargs
                )
                break
            except BadRequestError as e:
                if "stop_sequences" in str(e).lower():
                    generation_kwargs["stop"] = None
                elif "you are not allowed to request logprobs" in str(e).lower():
                    generation_kwargs.pop("logprobs")
                    generation_kwargs.pop("top_logprobs")
                elif (
                    "'temperature' is not supported with this model." in str(e).lower()
                ):
                    generation_kwargs.pop("temperature")
                else:
                    raise InvalidBenchmark(
                        f"Failed to generate text. The error message was: {e}"
                    )
            except (
                Timeout,
                ServiceUnavailableError,
                APIConnectionError,
                InternalServerError,
            ):
                logger.debug(
                    "Service temporarily unavailable. Retrying in 5 seconds..."
                )
                sleep(5)
            except APIError as e:
                raise InvalidBenchmark(
                    f"Failed to generate text. The error message was: {e}"
                )
            except AuthenticationError:
                raise NeedsAdditionalArgument(
                    cli_argument="--api-key",
                    script_argument="api_key=<your-api-key>",
                    run_with_cli=self.benchmark_config.run_with_cli,
                )
        else:
            raise InvalidBenchmark(
                message=f"Failed to generate text, after {num_attempts} attempts."
            )

        assert isinstance(model_response, ModelResponse)
        model_response_choices = model_response.choices[0]
        assert isinstance(model_response_choices, litellm.Choices)
        generation_output = model_response_choices.message["content"] or ""
        generation_output = generation_output.strip()

        # Structure the model output as a GenerativeModelOutput object
        model_output = GenerativeModelOutput(sequences=[generation_output])
        if hasattr(model_response_choices, "logprobs"):
            logprobs_list: list[list[tuple[str, float]]] = [
                [
                    (top_logprob.token, top_logprob.logprob)
                    for top_logprob in content.top_logprobs
                ]
                for content in model_response_choices.logprobs.content or list()
            ]
            model_output.scores = [logprobs_list]

        return model_output

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        for key, value in NUM_PARAMS_MAPPING.items():
            if re.fullmatch(pattern=key, string=self.model_config.model_id) is not None:
                return value

        if self.model_config.model_id.startswith("huggingface/"):
            model_id = self.model_config.model_id.split(sep="/", maxsplit=1)[-1]
            if HuggingFaceEncoderModel.model_exists(
                model_id=model_id, benchmark_config=self.benchmark_config
            ):
                hf_config = load_hf_model_config(
                    model_id=model_id,
                    num_labels=self.dataset_config.num_labels,
                    id2label=self.dataset_config.id2label,
                    label2id=self.dataset_config.label2id,
                    revision=self.model_config.revision,
                    model_cache_dir=self.model_config.model_cache_dir,
                    api_key=self.benchmark_config.api_key,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                    run_with_cli=self.benchmark_config.run_with_cli,
                )

                hf_api = HfApi()
                try:
                    repo_info = hf_api.model_info(
                        repo_id=model_id,
                        revision=self.model_config.revision,
                        token=os.getenv("HUGGINGFACE_API_KEY")
                        or self.benchmark_config.api_key
                        or True,
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
                    return repo_info.safetensors["total"]
                elif (
                    hasattr(hf_config, "num_params")
                    and hf_config.num_params is not None
                ):
                    return hf_config.num_params

        return -1

    @cached_property
    def vocab_size(self) -> int:
        """The vocabulary size of the model.

        Returns:
            The vocabulary size of the model.
        """
        for key, value in VOCAB_SIZE_MAPPING.items():
            if re.fullmatch(pattern=key, string=self.model_config.model_id) is not None:
                return value

        if self.model_config.model_id.startswith("huggingface/"):
            model_id = self.model_config.model_id.split(sep="/", maxsplit=1)[-1]
            if HuggingFaceEncoderModel.model_exists(
                model_id=model_id, benchmark_config=self.benchmark_config
            ):
                hf_config = load_hf_model_config(
                    model_id=model_id,
                    num_labels=self.dataset_config.num_labels,
                    id2label=self.dataset_config.id2label,
                    label2id=self.dataset_config.label2id,
                    revision=self.model_config.revision,
                    model_cache_dir=self.model_config.model_cache_dir,
                    api_key=self.benchmark_config.api_key,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                    run_with_cli=self.benchmark_config.run_with_cli,
                )

                tokenizer = load_tokenizer(
                    model=None,
                    model_id=model_id,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                )

                if (
                    hasattr(hf_config, "vocab_size")
                    and hf_config.vocab_size is not None
                ):
                    vocab_size = hf_config.vocab_size
                elif (
                    hasattr(tokenizer, "vocab_size")
                    and tokenizer.vocab_size is not None
                ):
                    vocab_size = tokenizer.vocab_size
                else:
                    vocab_size = -1
                return vocab_size

        return -1

    @cached_property
    def model_max_length(self) -> int:
        """The maximum length of the model.

        Returns:
            The maximum length of the model.
        """
        for key, value in MODEL_MAX_LENGTH_MAPPING.items():
            if re.fullmatch(pattern=key, string=self.model_config.model_id) is not None:
                return value

        if self.model_config.model_id.startswith("huggingface/"):
            model_id = self.model_config.model_id.split(sep="/", maxsplit=1)[-1]
            if HuggingFaceEncoderModel.model_exists(
                model_id=model_id, benchmark_config=self.benchmark_config
            ):
                hf_config = load_hf_model_config(
                    model_id=model_id,
                    num_labels=self.dataset_config.num_labels,
                    id2label=self.dataset_config.id2label,
                    label2id=self.dataset_config.label2id,
                    revision=self.model_config.revision,
                    model_cache_dir=self.model_config.model_cache_dir,
                    api_key=self.benchmark_config.api_key,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                    run_with_cli=self.benchmark_config.run_with_cli,
                )

                tokenizer = load_tokenizer(
                    model=None,
                    model_id=model_id,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                )

                all_max_lengths: list[int] = list()

                # Add the registered max length of the tokenizer
                if hasattr(
                    tokenizer, "model_max_length"
                ) and tokenizer.model_max_length < int(1e30):
                    all_max_lengths.append(tokenizer.model_max_length)

                # Add the max length derived from the model's input sizes
                if hasattr(tokenizer, "max_model_input_sizes"):
                    all_max_lengths.extend(
                        [
                            size
                            for size in tokenizer.max_model_input_sizes.values()
                            if size is not None
                        ]
                    )

                # Add max length candidates from the model's configuration
                candidate_config_max_lengths = [
                    "max_position_embeddings",
                    "max_sequence_length",
                    "model_max_length",
                    "sliding_window",
                    "sliding_window_size",
                    "n_positions",
                ]
                for candidate_config_max_length in candidate_config_max_lengths:
                    if (
                        hasattr(hf_config, candidate_config_max_length)
                        and (value := getattr(hf_config, candidate_config_max_length))
                        is not None
                    ):
                        all_max_lengths.append(value)

                # To avoid models having artificially low max lengths, we remove any max
                # lengths that are less than 128
                all_max_lengths = [
                    max_length for max_length in all_max_lengths if max_length >= 128
                ]

                if len(list(all_max_lengths)) > 0:
                    return min(list(all_max_lengths))

        return -1

    @property
    def data_collator(self) -> c.Callable[[list[t.Any]], dict[str, t.Any]]:
        """The data collator used to prepare samples during finetuning.

        Returns:
            The data collator.
        """
        raise NotImplementedError(
            "The `data_collator` property has not been implemented for LiteLLM models."
        )

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

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        """The Trainer class to use for finetuning.

        Returns:
            The Trainer class.
        """
        raise NotImplementedError(
            "The `trainer_class` property has not been implemented for LiteLLM models."
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
            Whether the model exists, or an error describing why we cannot check
            whether the model exists.
        """
        if model_id in litellm.model_list:
            return True

        num_attempts = 10
        for _ in range(num_attempts):
            try:
                litellm.completion(
                    messages=[dict(role="user", content="X")],
                    model=model_id,
                    max_tokens=1,
                    api_key=benchmark_config.api_key,
                    api_base=benchmark_config.api_base,
                    api_version=benchmark_config.api_version,
                )
                return True
            except APIError as e:
                if "'503 Service Unavailable" not in str(e):
                    raise e
                logger.warning(
                    f"Failed to check if model {model_id!r} exists. Retrying in "
                    f"{num_attempts} seconds..."
                )
                sleep(10)
            except (BadRequestError, NotFoundError):
                candidate_models = [
                    candidate_model_id
                    for candidate_model_id in litellm.model_list
                    if candidate_model_id.startswith(model_id)
                ]
                match len(candidate_models):
                    case 0:
                        pass
                    case 1:
                        logger.warning(
                            f"Could not find the model ID {model_id!r}. Did you mean "
                            f"{candidate_models[0]!r}?"
                        )
                    case _:
                        candidate_models_str = "', '".join(candidate_models)
                        logger.warning(
                            f"Could not find the model ID {model_id!r}. Did you mean "
                            f"any of the following model IDs: '{candidate_models_str}'?"
                        )
                return False
        else:
            logger.error(
                f"Failed to check if model {model_id!r} exists after {num_attempts} "
                "attempts. Assuming it does not exist."
            )
            return False

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
            model_id=model_id,
            revision="main",
            task="text-generation",
            languages=list(),
            merge=False,
            inference_backend=InferenceBackend.LITELLM,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=create_model_cache_dir(
                cache_dir=benchmark_config.cache_dir, model_id=model_id
            ),
            adapter_base_model_id=None,
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
            few_shot_examples = self._extract_few_shot_examples(
                dataset=dataset, task=task, itr_idx=itr_idx
            )
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(self._apply_prompt, few_shot_examples=few_shot_examples, task=task),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

    def _extract_few_shot_examples(
        self, dataset: DatasetDict, task: Task, itr_idx: int
    ) -> list[dict[str, t.Any]]:
        """Extract few-shot examples from a dataset.

        This will always extract the examples from the training split.

        We ensure that the few-shot examples are unique by picking them one at a time.

        Args:
            dataset:
                The dataset to extract the few-shot examples from.
            task:
                The task that is being benchmarked.
            itr_idx:
                The index of the dataset in the iterator.

        Returns:
            The few-shot examples.
        """
        random_seed = 4242 + itr_idx
        num_few_shots = self.dataset_config.num_few_shot_examples
        few_shot_examples: list[dict[str, t.Any]] = list()
        shuffled_train = dataset["train"].shuffle(seed=random_seed)

        match task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                labels = it.cycle(self.dataset_config.labels)
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
                    label = next(labels)
                    possible_examples = shuffled_train.filter(
                        lambda x: x["label"].lower() == label.lower()
                    )
                    if len(possible_examples) == 0:
                        continue
                    example = possible_examples.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )

            case TaskGroup.TEXT_TO_TEXT:
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
                    example = shuffled_train.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["text"] != example["text"]
                    )

            case TaskGroup.TOKEN_CLASSIFICATION:
                labels = it.cycle(
                    [
                        label.lower()
                        for label in self.dataset_config.labels
                        if label.lower().startswith("b-")
                    ]
                )
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
                    label = next(labels)
                    possible_examples = shuffled_train.filter(
                        lambda x: label in [tag.lower() for tag in x["labels"]]
                    )
                    if len(possible_examples) == 0:
                        continue
                    example = possible_examples.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["tokens"] != example["tokens"]
                    )

            case TaskGroup.QUESTION_ANSWERING:
                # Locate the maximum number of tokens that constitutes a short example
                for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
                    train_with_short_examples = dataset["train"].filter(
                        lambda example: len(example["context"]) < max_num_tokens
                    )
                    num_short_examples = len(train_with_short_examples)
                    if num_short_examples >= self.dataset_config.num_few_shot_examples:
                        break
                else:
                    raise InvalidBenchmark(
                        "Could not find enough short examples for few-shot learning."
                    )

                shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
                while (
                    len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0
                ):
                    example = shuffled_train.select(range(1))[0]
                    few_shot_examples.append(example)
                    shuffled_train = shuffled_train.filter(
                        lambda x: x["context"] != example["context"]
                    )

            case _:
                raise NotImplementedError(f"Unsupported task group: {task.task_group}.")

        random.seed(random_seed)
        random.shuffle(few_shot_examples)
        return few_shot_examples

    def _apply_prompt(
        self,
        examples: dict[str, t.Any],
        few_shot_examples: list[dict[str, t.Any]],
        task: Task,
    ) -> dict[str, t.Any]:
        """Apply prompt template to an example, potentially with few-shot examples.

        Args:
            examples:
                The examples to apply the few-shot examples to.
            few_shot_examples:
                The few-shot examples to apply.
            task:
                The task that is being benchmarked.

        Returns:
            The example with the few-shot examples applied.
        """

        def create_prompt(**kwargs: str) -> tuple[str, str]:
            """Create a prompt from the given keyword arguments.

            Args:
                kwargs:
                    The keyword arguments to use in the prompt.

            Returns:
                A pair (prompt, label), where "label" is an empty string if the model is
                not instruction tuned (as in this case it is included in the prompt).
            """
            label_key = "label" if "label" in kwargs else "target_text"
            label = kwargs.pop(label_key)
            label_mapping = self.dataset_config.prompt_label_mapping
            label = label_mapping.get(label, label)
            prompt = self.dataset_config.instruction_prompt.format(**kwargs)
            return prompt, label

        match task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                few_shot_sections = [
                    create_prompt(
                        text=example["text"].replace("\n", " ").strip(),
                        label=example["label"].replace("\n", " ").strip(),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    create_prompt(text=text.replace("\n", " ").strip(), label="")
                    for text in examples["text"]
                ]

            case TaskGroup.TEXT_TO_TEXT:
                few_shot_sections = [
                    create_prompt(
                        text=example["text"].replace("\n", " ").strip(),
                        target_text=example["target_text"].replace("\n", " ").strip(),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    create_prompt(text=text.replace("\n", " ").strip(), target_text="")
                    for text in examples["text"]
                ]

            case TaskGroup.TOKEN_CLASSIFICATION:

                def create_label(example: dict) -> str:
                    prompt_labels = self.dataset_config.prompt_label_mapping.values()
                    labels: dict[str, list[str]] = {
                        prompt_label: list() for prompt_label in prompt_labels
                    }
                    for token, label in zip(example["tokens"], example["labels"]):
                        label = label.lower()
                        if label == "o":
                            continue
                        prompt_label = self.dataset_config.prompt_label_mapping[label]
                        if label.startswith("b-"):
                            labels[prompt_label].append(token)
                        elif label.startswith("i-"):
                            labels[prompt_label][-1] += " " + token
                    return json.dumps(labels, ensure_ascii=False)

                few_shot_sections = [
                    create_prompt(
                        text=" ".join(example["tokens"]).replace("\n", " ").strip(),
                        label=create_label(example=example),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    create_prompt(
                        text=" ".join(tokens).replace("\n", " ").strip(), label=""
                    )
                    for tokens in examples["tokens"]
                ]

            case TaskGroup.QUESTION_ANSWERING:
                few_shot_sections = [
                    create_prompt(
                        text=example["context"].replace("\n", " ").strip(),
                        question=example["question"].replace("\n", " ").strip(),
                        label=example["answers"]["text"][0].replace("\n", " "),
                    )
                    for example in few_shot_examples
                ]
                new_sections = [
                    create_prompt(
                        text=context.replace("\n", " ").strip(),
                        question=question.replace("\n", " ").strip(),
                        label="",
                    )
                    for context, question in zip(
                        examples["context"], examples["question"]
                    )
                ]

            case _:
                raise NotImplementedError(f"Unsupported task group: {task.task_group}.")

        few_shot_messages = [
            dict(role=role, content=content)
            for prompt, label in few_shot_sections
            for role, content in [("user", prompt), ("assistant", label)]
        ]

        messages_list = [
            few_shot_messages + [dict(role="user", content=prompt)]
            for prompt, _ in new_sections
        ]

        examples["messages"] = messages_list
        return examples
