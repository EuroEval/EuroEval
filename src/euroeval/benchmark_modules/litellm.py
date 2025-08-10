"""Generative models from an inference API, using the LiteLLM framework."""

import asyncio
import collections.abc as c
import logging
import os
import re
import typing as t
from functools import cache, cached_property, partial
from time import sleep

import litellm
import ollama
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
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.router import Router
from litellm.types.utils import ChoiceLogprobs
from litellm.utils import supports_reasoning, supports_response_schema
from pydantic import conlist, create_model
from requests.exceptions import RequestException
from tqdm.asyncio import tqdm as tqdm_async
from tqdm.auto import tqdm

from ..constants import MAX_LOGPROBS, REASONING_MAX_TOKENS, TASKS_USING_JSON
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
    NeedsAdditionalArgument,
    NeedsEnvironmentVariable,
    NeedsExtraInstalled,
)
from ..generation_utils import apply_prompt, extract_few_shot_examples
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..tokenization_utils import get_first_label_token_mapping
from ..types import ExtractLabelsFunction
from ..utils import (
    add_semaphore_and_catch_exception,
    create_model_cache_dir,
    log_once,
    safe_run,
)
from .base import BenchmarkModule
from .hf import HuggingFaceEncoderModel, load_hf_model_config, load_tokenizer

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from litellm.types.utils import ModelResponse
    from transformers.trainer import Trainer

logger = logging.getLogger("euroeval")


VOCAB_SIZE_MAPPING = {
    # OpenAI models
    r"gpt-5-.*": 100_256,
    r"gpt-4-(32k)?(-[0-9]{4})?": 100_256,
    r"gpt-4-[0-9]{4}-preview": 100_256,
    r"gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 100_256,
    r"gpt-4-(vision|turbo)(-preview)?": 100_256,
    r"gpt-3.5-turbo-instruct(-[0-9]{4})?": 100_256,
    r"gpt-4o(-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_019,
    r"o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    # Anthropic models
    r"(anthropic/)?claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": -1,
    # Gemini models
    r"(gemini/)?gemini-[1-9]\.[0-9]-(flash|pro).*": 256_128,
    # xAI models
    r"(xai/)?grok.*": -1,
}


MODEL_MAX_LENGTH_MAPPING = {
    # OpenAI models
    r"gpt-5-.*": 272_000,
    r"gpt-4(-[0-9]{4})?": 8_191,
    r"gpt-4-32k(-[0-9]{4})?": 32_767,
    r"gpt-4-[0-9]{4}-preview": 128_000,
    r"gpt-4-turbo(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    r"gpt-4-(vision|turbo)(-preview)?": 128_000,
    r"gpt-3.5-turbo-instruct(-[0-9]{4})?": 4_095,
    r"gpt-4o(-mini)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    r"o1-(mini|preview)(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 128_000,
    r"o1(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_000,
    r"o[2-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": 200_000,
    r"gpt-4.1.*": 1_047_576,
    # Anthropic models
    r"(anthropic/)?claude-[1-9](-[1-9])?-(opus|sonnet|haiku)-[0-9]{8}": 200_000,
    # Gemini models
    r"(gemini/)?gemini-1\.5-flash.*": 1_048_576,
    r"(gemini/)?gemini-1\.5-pro.*": 2_097_152,
    r"(gemini/)?gemini-2\.(0|5).*": 1_048_576,
    # xAI models
    r"(xai/)?grok.*": 131_072,
}


NUM_PARAMS_MAPPING = {
    # OpenAI models
    r"gpt-5-.*": -1,
    r"gpt-4.*": -1,
    r"o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": -1,
    # Anthropic models
    r"(anthropic/)?claude-*": -1,
    # Gemini models
    r"(gemini/)?gemini-1.5-flash-8b": 8_000_000_000,
    r"(gemini/)?gemini-1.5-flash-[0-9]+": -1,
    r"(gemini/)?gemini-2.(0|5).*": -1,
    # xAI models
    r"(xai/)?grok.*": -1,
}


ALLOWED_PARAMS = {
    # OpenAI models
    r"gpt-5-.*": ["minimal", "low", "medium", "high"],
    r"o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?": ["low", "medium", "high"],
    # Anthropic models
    r"(anthropic/)?claude-3-7-sonnet.*": ["no-thinking", "thinking"],
    r"(anthropic/)?claude-(sonnet|opus)-4.*": ["no-thinking", "thinking"],
    # Gemini models
    r"(gemini/)?gemini-2.5-flash-lite.*": ["no-thinking", "thinking"],
    r"(gemini/)?gemini-2.5-flash-[0-9].*": ["no-thinking", "thinking"],
    # xAI models
    r"(xai/)?grok-3-mini(-fast)?(-beta)?": ["low", "medium", "high"],
}


REASONING_MODELS = [
    r"o[1-9](-mini|-preview)?(-[0-9]{4}-[0-9]{2}-[0-9]{2})?",
    r"(gemini/)?gemini.*thinking.*",
    r"(gemini/)?gemini-2.5.*",
    r"(xai/)?grok-3-mini.*",
]


class LiteLLMModel(BenchmarkModule):
    """A generative model from LiteLLM."""

    fresh_model = False
    batching_preference = BatchingPreference.ALL_AT_ONCE
    high_priority = False

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Initialise the model.

        Args:
            model_config:
                The model configuration.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.
        """
        # Detect whether the model is an Ollama model, as we need to extract metadata
        # differently for these models
        self.is_ollama = model_config.model_id.startswith(
            "ollama/"
        ) or model_config.model_id.startswith("ollama_chat/")
        self._ollama_show: ollama.ShowResponse = (
            ollama.show("/".join(model_config.model_id.split("/")[1:]))
            if self.is_ollama
            else ollama.ShowResponse(model_info=None)
        )

        raise_if_wrong_params(model_config=model_config, allowed_params=ALLOWED_PARAMS)

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )

        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokenizer=None,
            generative_type=self.generative_type,
        )

    @property
    def generative_type(self) -> GenerativeType | None:
        """Get the generative type of the model.

        Returns:
            The generative type of the model, or None if it has not been set yet.
        """
        if self.is_ollama:
            reasoning_model = "thinking" in (self._ollama_show.capabilities or [])
            type_ = (
                GenerativeType.REASONING
                if reasoning_model
                else GenerativeType.INSTRUCTION_TUNED
            )
        elif self.model_config.revision in {"thinking"}:
            type_ = GenerativeType.REASONING
        elif self.model_config.revision in {"no-thinking"}:
            type_ = GenerativeType.INSTRUCTION_TUNED
        elif re.fullmatch(
            pattern="|".join(REASONING_MODELS), string=self.model_config.model_id
        ):
            type_ = GenerativeType.REASONING
        elif supports_reasoning(model=self.model_config.model_id):
            type_ = GenerativeType.REASONING
        else:
            type_ = GenerativeType.INSTRUCTION_TUNED

        log_once(
            f"Detected generative type {type_.name!r} for model "
            f"{self.model_config.model_id!r}",
            level=logging.DEBUG,
        )
        return type_

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        """Generate outputs from the model.

        Args:
            inputs:
                A batch of inputs to pass through the model.

        Returns:
            The generated model outputs.
        """
        assert "messages" in inputs, "The input must contain a 'messages' key."
        conversations: list[list[litellm.AllMessageValues]] = inputs["messages"]

        # Get the mapping from labels to the first token in the label. We call this each
        # time we generate a new dataset since the dataset config can change
        self.buffer["first_label_token_mapping"] = get_first_label_token_mapping(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            tokenizer=None,
            generative_type=self.generative_type,
        )

        # Sanity check that "JSON" is included in the prompt, as some models require
        # this
        if self.dataset_config.task in TASKS_USING_JSON:
            for conversation in conversations:
                if not conversation:
                    raise InvalidBenchmark(
                        "Encountered an empty conversation in 'messages'."
                    )
                last_message = conversation[-1]
                assert isinstance(last_message, dict), (
                    f"Expected dict message, got {type(last_message)}"
                )
                assert "content" in last_message, (
                    "Expected 'content' key in the last message of the conversation."
                )
                assert isinstance(last_message["content"], str), (
                    "Expected 'content' to be a string."
                )
                assert "json" in last_message["content"].lower(), (
                    "Prompt must contain 'json' for JSON tasks."
                )

        all_responses: dict[int, "ModelResponse"] = {}
        conversations_to_run: list[tuple[int, list[litellm.AllMessageValues]]] = list(
            enumerate(conversations)
        )
        for attempt in range(num_attempts := 10):
            if not conversations_to_run:
                break

            batch_indices, batch_conversations = zip(*conversations_to_run)
            successes, failures = safe_run(
                self._generate_async(
                    model_id=self.model_config.model_id,
                    conversations=list(batch_conversations),
                    **self.get_generation_kwargs(dataset_config=self.dataset_config),
                )
            )

            # Store the successful model outputs
            for idx, response in successes:
                orig_idx = batch_indices[idx]
                all_responses[orig_idx] = response

            # If all requests were successful, break
            if not failures:
                conversations_to_run = []
                break

            # Put the failed requests back in the queue to try again
            conversations_to_run = [
                (batch_indices[idx], conversations[batch_indices[idx]])
                for idx, _ in failures
            ]
            logger.debug(
                f"Attempt {attempt + 1:,}/{num_attempts:,}: retrying "
                f"{len(conversations_to_run):,} failed message(s)"
            )

            # Attempt to handle the exceptions, to improve the chance of getting
            # successful generations next time around
            for _, error in failures:
                self._handle_exception(
                    error=error,
                    generation_kwargs=self.get_generation_kwargs(
                        dataset_config=self.dataset_config
                    ),
                )

            # Sleep for a second to avoid pinging the API server too quickly
            sleep(1)
        else:
            raise InvalidBenchmark(
                message=f"Failed to generate text, after {num_attempts:,} attempts."
            )

        # Extract the generations from the model output
        ordered_responses = [all_responses[i] for i in range(len(conversations))]
        model_output = self._create_model_output(
            model_responses=ordered_responses, model_id=self.model_config.model_id
        )

        if len(conversations) != len(model_output.sequences):
            raise InvalidBenchmark(
                f"Number of model inputs ({len(conversations):,}) does not match the "
                f"number of model outputs ({len(model_output.sequences):,})."
            )

        return model_output

    def _handle_exception(
        self, error: Exception, generation_kwargs: dict[str, t.Any]
    ) -> None:
        """Handle an exception from the model.

        Args:
            error:
                The exception to handle.
            generation_kwargs:
                The generation kwargs to pass to the model.
        """
        error_msg = str(error).lower()
        model_id = self.model_config.model_id

        # Error messages that we want to catch and handle
        stop_messages = ["stop_sequences", "'stop' is not supported with this model"]
        logprobs_messages = [
            "you are not allowed to request logprobs",
            "you've reached the maximum number of requests with logprobs",
            "logprobs is not supported",
            "logprobs is not enabled",
        ]
        temperature_messages = [
            "'temperature' is not supported with this model.",
            "temperature is not supported with this model",
        ]
        temperature_must_be_one_messages = [
            "`temperature` may only be set to 1",
            "'temperature' does not support 0.0 with this model. Only the default "
            "(1) value is supported",
            "Only temperature=1 is supported",
        ]
        max_items_messages = ["'maxItems' is not permitted."]
        no_json_schema_messages = ["Property keys should match pattern"]
        thinking_budget_pattern = re.compile(
            r"the thinking budget [0-9]+ is invalid. please choose a value between "
            r"[0-9]+ and ([0-9]+)\."
        )
        requires_thinking_disabled_messages = ["thinking.type: Field required"]

        if any(msg.lower() in error_msg for msg in stop_messages):
            log_once(
                f"The model {model_id!r} does not support "
                "stop sequences, so disabling them.",
                level=logging.DEBUG,
            )
            generation_kwargs["stop"] = None
            return
        elif (
            any(msg.lower() in error_msg for msg in logprobs_messages)
            # Special case for Vertex AI models, since they have strict rate
            # limits on using logprobs. They also have a cap of 5 logprobs, but
            # we ignore this since the rate limiting makes it unusable anyway.
            or (isinstance(error, VertexAIError) and "logprobs" in error_msg)
        ):
            log_once(
                f"The model {model_id!r} does not support logprobs, so disabling it.",
                level=logging.DEBUG,
            )
            generation_kwargs.pop("logprobs", None)
            generation_kwargs.pop("top_logprobs", None)
            return
        elif any(msg.lower() in error_msg for msg in temperature_messages):
            log_once(
                f"The model {model_id!r} does not support "
                "temperature, so disabling it.",
                level=logging.DEBUG,
            )
            generation_kwargs.pop("temperature", None)
            return
        elif any(msg.lower() in error_msg for msg in temperature_must_be_one_messages):
            log_once(
                f"The model {model_id!r} requires "
                "temperature to be set to 1, so setting it.",
                level=logging.DEBUG,
            )
            generation_kwargs["temperature"] = 1.0
            return
        elif any(msg.lower() in error_msg for msg in max_items_messages):
            log_once(
                f"The model {model_id!r} does not support "
                "maxItems in the JSON schema, so disabling it.",
                level=logging.DEBUG,
            )
            ner_tag_names = list(self.dataset_config.prompt_label_mapping.values())
            keys_and_their_types = {
                tag_name: (list[str], ...) for tag_name in ner_tag_names
            }
            pydantic_class = create_model("AnswerFormat", **keys_and_their_types)
            generation_kwargs["response_format"] = pydantic_class
            return
        elif any(msg.lower() in error_msg for msg in no_json_schema_messages):
            log_once(
                f"The model {self.model_config.model_id!r} does not support "
                "JSON schemas, so using the vanilla JSON format.",
                level=logging.DEBUG,
            )
            generation_kwargs["response_format"] = dict(type="json_object")
            return
        elif thinking_match := thinking_budget_pattern.search(string=error_msg):
            thinking_budget = int(thinking_match.group(1))
            if thinking_budget >= REASONING_MAX_TOKENS:
                raise InvalidBenchmark(
                    f"The model {model_id!r} has an upper thinking budget of "
                    f"{thinking_budget:,} tokens, which is within the limit of "
                    f"{REASONING_MAX_TOKENS:,} tokens. This should not happen. The "
                    f"error message was: {error_msg}."
                )
            log_once(
                f"The model {model_id!r} can at most use {thinking_budget:,} tokens "
                "for reasoning, which is less than the default of "
                f"{REASONING_MAX_TOKENS:,} tokens. Setting the thinking budget to "
                f"{thinking_budget:,} tokens.",
                level=logging.DEBUG,
            )
            generation_kwargs["thinking"] = dict(
                type="enabled", budget_tokens=thinking_budget - 1
            )
            return
        elif (
            any(msg.lower() in error_msg for msg in requires_thinking_disabled_messages)
            and self.generative_type != GenerativeType.REASONING
        ):
            log_once(
                f"The model {model_id!r} requires the `thinking.type` field to be "
                f"set to `disabled` rather than just setting `budget_tokens` to 0. "
                "Setting `thinking.type` to `disabled`.",
                level=logging.DEBUG,
            )
            generation_kwargs["thinking"] = dict(type="disabled")
            return
        elif isinstance(
            error, (Timeout, ServiceUnavailableError, InternalServerError, SystemError)
        ):
            logger.debug(
                f"Service temporarily unavailable. The error message was: {error}. "
                f"Retrying in 5 seconds..."
            )
            sleep(5)
            return
        elif isinstance(error, UnsupportedParamsError):
            unsupported_param_match = re.search(
                pattern=r"(?<=does not support parameters\: \[')([^ ']+)(?='\])",
                string=error.message,
            )
            if unsupported_param_match is None:
                raise InvalidModel(error.message)
            else:
                unsupported_param = unsupported_param_match.group(0)
                raise InvalidModel(
                    f"The model {model_id!r} does not support the parameter "
                    f"{unsupported_param!r}. Try again without this parameter. "
                    "Skipping this model."
                )
        elif isinstance(error, (APIConnectionError, OSError)):
            # If there are too many I/O connections, we increase the number of allowed
            # file descriptors
            if "too many open files" in error_msg:
                raise InvalidBenchmark(
                    "There are too many file descriptors running. See the current "
                    "value by running `ulimit -n`. Try increasing it by running "
                    "`ulimit -n <new-value>` and try again."
                )
            raise InvalidBenchmark(
                f"Encountered {type(error)} during generation: {error}."
            )

        if isinstance(error, RateLimitError):
            raise InvalidModel(
                f"You have encountered your rate limit for model {model_id!r}. "
                "Skipping."
            )

        if isinstance(error, AuthenticationError):
            raise NeedsAdditionalArgument(
                cli_argument="--api-key",
                script_argument="api_key=<your-api-key>",
                run_with_cli=self.benchmark_config.run_with_cli,
            )

        raise InvalidBenchmark(
            f"Failed to generate text. The error message was: {error}"
        )

    async def _generate_async(
        self,
        model_id: str,
        conversations: list[list[litellm.AllMessageValues]],
        **generation_kwargs,
    ) -> tuple[list[tuple[int, "ModelResponse"]], list[tuple[int, Exception]]]:
        """Generate outputs from the model asynchronously.

        Args:
            model_id:
                The ID of the model to use for generation.
            conversations:
                The conversations to pass to the model.
            **generation_kwargs:
                Additional generation arguments to pass to the model.

        Returns:
            A tuple (successes, failures), each being a list of tuples (idx, content),
            where the `idx` corresponds to the index of `conversations`, and `content`
            is either the model response or an Exception.
        """
        # Create a LiteLLM router, which will ensure that we only use a single client
        # for all the requests, preventing "too many open files" errors
        router = Router(
            model_list=[
                dict(
                    model_name=self.model_config.model_id,
                    litellm_params=generation_kwargs,
                )
            ]
        )

        # Get the LLM generations asynchronously
        max_concurrent_calls = 20
        semaphore = asyncio.Semaphore(max_concurrent_calls)
        requests = [
            add_semaphore_and_catch_exception(
                router.acompletion(model=model_id, messages=conversation),
                semaphore=semaphore,
            )
            for conversation in conversations
        ]
        responses = await tqdm_async.gather(*requests, leave=False)

        # Separate the successful responses from the failed ones
        successes = [
            (idx, response)
            for idx, response in enumerate(responses)
            if not isinstance(response, Exception)
        ]
        failures = [
            (idx, response)
            for idx, response in enumerate(responses)
            if isinstance(response, Exception)
        ]

        # Close connections
        for request in requests:
            if hasattr(request, "close"):
                request.close()

        return successes, failures

    @staticmethod
    def _create_model_output(
        model_responses: list["ModelResponse"], model_id: str
    ) -> GenerativeModelOutput:
        """Create a GenerativeModelOutput object from a list of ModelResponse objects.

        Args:
            model_responses:
                The list of ModelResponse objects to create the GenerativeModelOutput
                object from.
            model_id:
                The ID of the model.

        Returns:
            A GenerativeModelOutput object.
        """
        sequences = []
        scores = []
        for model_response in model_responses:
            if not model_response.choices:
                sequences.append("")
                logger.warning(
                    f"The model {model_id!r} did not end up "
                    "generating any text. This is likely because the model ran "
                    "out of tokens while reasoning. Returning an empty string."
                )
                continue

            model_response_choices = model_response.choices[0]
            assert isinstance(model_response_choices, litellm.Choices)
            generated_message: litellm.Message = model_response_choices.message
            generation_output = generated_message.content or ""
            generation_output = generation_output.strip()

            # Structure the model output as a GenerativeModelOutput object
            sequences.append(generation_output)
            if hasattr(model_response_choices, "logprobs"):
                logprobs_obj = model_response_choices.logprobs
                if isinstance(logprobs_obj, ChoiceLogprobs):
                    logprobs_list: list[list[tuple[str, float]]] = [
                        [
                            (top_logprob.token, top_logprob.logprob)
                            for top_logprob in content.top_logprobs
                        ]
                        for content in model_response_choices.logprobs.content or list()
                    ]
                    scores.append(logprobs_list)
                else:
                    log_once(
                        "The logprobs object is malformed, so we won't use logprobs to "
                        "determine the labels.",
                        level=logging.WARNING,
                    )

        if not sequences:
            logger.warning(
                "No sequences were generated by the model "
                f"{model_id!r}. This may be due to the "
                "model running out of tokens or an issue with the input data. "
                "Returning an empty GenerativeModelOutput."
            )
            return GenerativeModelOutput(sequences=[], scores=None)

        if scores and len(sequences) != len(scores):
            raise InvalidBenchmark(
                "Sequences and scores must have the same length. "
                f"Got {len(sequences)} sequences and {len(scores)} scores."
            )

        return GenerativeModelOutput(
            sequences=sequences, scores=scores if scores else None
        )

    @cached_property
    def num_params(self) -> int:
        """The number of parameters in the model.

        Returns:
            The number of parameters in the model.
        """
        # Start by trying out the regex mapping, and use the value if it matches
        for key, value in NUM_PARAMS_MAPPING.items():
            if re.fullmatch(pattern=key, string=self.model_config.model_id) is not None:
                return value

        # If it is an Ollama model then we can get the number of parameters from the
        # Ollama Python SDK
        if self.is_ollama:
            model_info = self._ollama_show.modelinfo
            if model_info is not None:
                num_params = model_info.get("general.parameter_count")
                if num_params is not None:
                    return int(num_params)

        # If it is a model accessed through the Hugging Face inference API then we can
        # get the number of parameters from the Hugging Face model configuration from
        # the Hugging Face Hub
        if self.model_config.model_id.startswith("huggingface/"):
            model_id = "/".join(self.model_config.model_id.split(sep="/")[-2:])
            if HuggingFaceEncoderModel.model_exists(
                model_id=model_id, benchmark_config=self.benchmark_config
            ):
                hf_config = load_hf_model_config(
                    model_id=model_id,
                    num_labels=self.dataset_config.num_labels,
                    id2label=self.dataset_config.id2label,
                    label2id=self.dataset_config.label2id,
                    revision="main",
                    model_cache_dir=self.model_config.model_cache_dir,
                    api_key=self.benchmark_config.api_key,
                    trust_remote_code=self.benchmark_config.trust_remote_code,
                    run_with_cli=self.benchmark_config.run_with_cli,
                )

                hf_api = HfApi()
                try:
                    repo_info = hf_api.model_info(
                        repo_id=model_id,
                        revision="main",
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
        # Start by trying out the regex mapping, and use the value if it matches
        for key, value in VOCAB_SIZE_MAPPING.items():
            if re.fullmatch(pattern=key, string=self.model_config.model_id) is not None:
                return value

        # If it is a model accessed through the Hugging Face inference API then we can
        # get the vocabulary size from the Hugging Face model configuration from the
        # Hugging Face Hub
        if self.model_config.model_id.startswith("huggingface/"):
            model_id = "/".join(self.model_config.model_id.split(sep="/")[-2:])
            if HuggingFaceEncoderModel.model_exists(
                model_id=model_id, benchmark_config=self.benchmark_config
            ):
                hf_config = load_hf_model_config(
                    model_id=model_id,
                    num_labels=self.dataset_config.num_labels,
                    id2label=self.dataset_config.id2label,
                    label2id=self.dataset_config.label2id,
                    revision="main",
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
        # Start by trying out the regex mapping, and use the value if it matches
        for key, value in MODEL_MAX_LENGTH_MAPPING.items():
            if re.fullmatch(pattern=key, string=self.model_config.model_id) is not None:
                return value

        # If it is an Ollama model then we can get the maximum length from the Ollama
        # Python SDK
        if self.is_ollama:
            ollama_model_id = "/".join(self.model_config.model_id.split("/")[1:])
            model_info = self._ollama_show.modelinfo
            if model_info is not None:
                context_length_keys = [
                    key for key in model_info.keys() if "context_length" in key.lower()
                ]
                if context_length_keys:
                    context_length = model_info[context_length_keys[0]]
                    if context_length is not None:
                        log_once(
                            f"Detected context length key {context_length_keys[0]!r} "
                            f"for Ollama model {ollama_model_id!r}",
                            level=logging.DEBUG,
                        )
                        return int(context_length)
                else:
                    log_once(
                        f"Tried to get the maximum length of the Ollama model "
                        f"{ollama_model_id!r}, but could not find a context length. "
                        f"The model info was {model_info}. Returning -1",
                        level=logging.DEBUG,
                    )

        # If it is a model accessed through the Hugging Face inference API then we can
        # get the maximum length from the Hugging Face model configuration from the
        # Hugging Face Hub
        if self.model_config.model_id.startswith("huggingface/"):
            model_id = "/".join(self.model_config.model_id.split(sep="/")[-2:])
            if HuggingFaceEncoderModel.model_exists(
                model_id=model_id, benchmark_config=self.benchmark_config
            ):
                hf_config = load_hf_model_config(
                    model_id=model_id,
                    num_labels=self.dataset_config.num_labels,
                    id2label=self.dataset_config.id2label,
                    label2id=self.dataset_config.label2id,
                    revision="main",
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
        model_id, _ = model_id.split("@") if "@" in model_id else (model_id, "main")
        if model_id in litellm.model_list:
            return True

        # Separate check for Ollama models
        if model_id.startswith("ollama/") or model_id.startswith("ollama_chat/"):
            ollama_model_exists = try_download_ollama_model(model_id=model_id)
            if ollama_model_exists:
                return ollama_model_exists

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
            # A rate limit indicates that the model *does* exist, but we are being rate
            # limited.
            except RateLimitError:
                return True
            except (
                APIConnectionError,
                Timeout,
                ServiceUnavailableError,
                InternalServerError,
            ) as e:
                logger.debug(
                    f"Service temporarily unavailable. The error message was: {e}. "
                    "Retrying in 10 seconds..."
                )
                sleep(5)
            except APIError as e:
                if "'503 Service Unavailable" not in str(e):
                    raise e
                logger.warning(
                    f"Failed to check if model {model_id!r} exists. Retrying in 10 "
                    "seconds..."
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
        model_id, revision = model_id.split("@") if "@" in model_id else (model_id, "")
        return ModelConfig(
            model_id=model_id,
            revision=revision,
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
        self, dataset: "DatasetDict", task: Task, itr_idx: int
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
                instruction_model=True,
                always_populate_text_field=False,
                tokenizer=None,
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

    @cache
    def get_generation_kwargs(self, dataset_config: DatasetConfig) -> dict[str, t.Any]:
        """Get the generation arguments for the model.

        Args:
            dataset_config:
                The dataset configuration, which is used to determine the generative
                type of the model. We use this as an argument here rather than using
                `self.dataset_config` to ensure that that the cache is updated when the
                dataset configuration changes.

        Returns:
            The generation arguments for the model.
        """
        # Set the core generation arguments
        generation_kwargs: dict[str, t.Any] = dict(
            model=self.model_config.model_id,
            max_completion_tokens=(
                REASONING_MAX_TOKENS
                if self.generative_type == GenerativeType.REASONING
                else dataset_config.max_generated_tokens
            ),
            stop=[],
            temperature=0.0,
            seed=4242,
            api_key=self.benchmark_config.api_key,
            api_base=self.benchmark_config.api_base,
            api_version=self.benchmark_config.api_version,
            max_retries=3,
        )

        # Set up the `response_format` generation argument if we are dealing with a task
        # using structured generation
        if dataset_config.task in TASKS_USING_JSON:
            if self.generative_type == GenerativeType.REASONING:
                log_once(
                    f"The model {self.model_config.model_id!r} is a reasoning model "
                    "and thus does not support structured generation, so we do not "
                    "enable it.",
                    level=logging.DEBUG,
                )
            elif supports_response_schema(model=self.model_config.model_id):
                ner_tag_names = list(dataset_config.prompt_label_mapping.values())
                keys_and_their_types: dict[str, t.Any] = {
                    tag_name: (conlist(str, max_length=5), ...)
                    for tag_name in ner_tag_names
                }
                pydantic_class = create_model("AnswerFormat", **keys_and_their_types)
                generation_kwargs["response_format"] = pydantic_class
                log_once(
                    "Enabling structured generation for model "
                    f"{self.model_config.model_id!r} with the JSON schema "
                    f"{pydantic_class.model_json_schema()}",
                    level=logging.DEBUG,
                )
            else:
                generation_kwargs["response_format"] = dict(type="json_object")
                log_once(
                    "Enabling structured JSON generation for model "
                    f"{self.model_config.model_id!r} with no custom JSON schema, as "
                    "the model does not support schemas.",
                    level=logging.DEBUG,
                )

        # If the model is an Ollama reasoning model, we ensure that thinking is enabled
        if self.is_ollama and self.generative_type == GenerativeType.REASONING:
            generation_kwargs["think"] = True
            log_once(
                "Enabling thinking mode for Ollama model "
                f"{self.model_config.model_id!r}",
                level=logging.DEBUG,
            )

        # Handle manually set parameters
        if self.buffer["first_label_token_mapping"]:
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = MAX_LOGPROBS
        if self.model_config.revision == "thinking":
            generation_kwargs["thinking"] = dict(
                type="enabled", budget_tokens=REASONING_MAX_TOKENS - 1
            )
            log_once(
                f"Enabling thinking mode for model {self.model_config.model_id!r}",
                level=logging.DEBUG,
            )
        elif self.model_config.revision == "no-thinking":
            generation_kwargs["thinking"] = dict(budget_tokens=0)
            log_once(
                f"Disabling thinking mode for model {self.model_config.model_id!r}",
                level=logging.DEBUG,
            )
        elif self.model_config.revision in {"minimal", "low", "medium", "high"}:
            generation_kwargs["reasoning_effort"] = self.model_config.revision
            log_once(
                f"Enabling reasoning effort {self.model_config.revision!r} for model "
                f"{self.model_config.model_id!r}",
                level=logging.DEBUG,
            )

        # First attempt is a test run with a single conversation to handle errors
        # quickly. We repeat this multiple times to deal with different types of
        # errors, and stop if we get a successful response.
        test_conversation = [
            litellm.ChatCompletionUserMessage(role="user", content="Test message")
        ]
        for _ in range(5):
            _, failures = safe_run(
                self._generate_async(
                    model_id=self.model_config.model_id,
                    conversations=[test_conversation],
                    **generation_kwargs,
                )
            )
            if not failures:
                break
            for _, error in failures:
                self._handle_exception(error=error, generation_kwargs=generation_kwargs)

        return generation_kwargs


def raise_if_wrong_params(
    model_config: ModelConfig, allowed_params: dict[str, list[str]]
) -> None:
    """Raise an error if the model configuration has invalid parameters.

    Args:
        model_config:
            The model configuration.
        allowed_params:
            The allowed parameters for the model.

    Raises:
        InvalidModel:
            If the model configuration has invalid parameters.
    """
    param = model_config.revision
    if param == "":
        return
    for model_regex, allowed_params_list in allowed_params.items():
        if re.fullmatch(pattern=model_regex, string=model_config.model_id):
            if param not in allowed_params_list:
                msg = (
                    f"Invalid parameter {param!r} for model {model_config.model_id!r}."
                )
                if allowed_params_list:
                    msg += f" Allowed parameters are: {', '.join(allowed_params_list)}."
                else:
                    msg += " No parameters are allowed."
                raise InvalidModel(msg)
            return
    else:
        raise InvalidModel(
            f"The parameter {param!r} is not supported for the model "
            f"{model_config.model_id!r}."
        )


def try_download_ollama_model(model_id: str) -> bool:
    """Try to download an Ollama model.

    Args:
        model_id:
            The model ID. If the model does not start with "ollama/" or "ollama_chat/"
            then this function will return False.

    Returns:
        Whether the model was downloaded successfully.

    Raises:
        InvalidModel:
            If Ollama is not running or the model cannot be downloaded.
    """
    if not (model_id.startswith("ollama/") or model_id.startswith("ollama_chat/")):
        return False

    if model_id.startswith("ollama/"):
        log_once(
            "You're trying to benchmark a model with the old 'ollama/' prefix, which "
            "probably results in bad performance, as it doesn't use the model's chat "
            "template. If the model is not a chat model then just disregard this "
            "warning, but if it is a chat model then please cancel this run and "
            "use the 'ollama_chat/' prefix instead.",
            level=logging.WARNING,
        )

    try:
        downloaded_ollama_models: list[str] = [
            model_obj.model
            for model_obj in ollama.list().models
            if model_obj.model is not None
        ]
    except ConnectionError:
        raise InvalidModel(
            "Ollama does not seem to be running, so we cannot evaluate the model "
            f"{model_id!r}. Please make sure that Ollama is running and try again."
        )

    ollama_model_id = "/".join(model_id.split("/")[1:])
    if ollama_model_id not in downloaded_ollama_models:
        # Try fetching the model info
        try:
            response = ollama.pull(model=ollama_model_id, stream=True)
        except ollama.ResponseError as e:
            if "file does not exist" in str(e).lower():
                # Check if the model exists if we prepend "hf.co/"
                try:
                    ollama_model_id_with_prefix = f"hf.co/{ollama_model_id}"
                    model_id_with_prefix = (
                        f"{model_id.split('/')[0]}/{ollama_model_id_with_prefix}"
                    )
                    ollama.pull(model=ollama_model_id_with_prefix, stream=True)
                    log_once(
                        f"The model {model_id!r} cannot be found on Ollama, but the "
                        f"model {model_id_with_prefix} *was* found, so we would "
                        "recommend you cancelling this run and trying the evaluation "
                        "with that model ID instead."
                    )
                    return False
                except ollama.ResponseError as inner_e:
                    if "file does not exist" in str(inner_e).lower():
                        return False
                    else:
                        raise InvalidModel(
                            f"Failed to download Ollama model {ollama_model_id}. "
                            f"The error message was: {inner_e}"
                        )
            else:
                raise InvalidModel(
                    f"Failed to download Ollama model {ollama_model_id}. "
                    f"The error message was: {e}"
                )

        # Download the model
        with tqdm(
            desc=f"Downloading {ollama_model_id}",
            unit_scale=True,
            unit="B",
            leave=False,
        ) as pbar:
            for status in response:
                if status.total is not None:
                    pbar.total = status.total
                if status.completed is not None:
                    pbar.update(status.completed - pbar.n)
        return True

    else:
        log_once(
            f"Ollama model {ollama_model_id!r} already downloaded, so skipping "
            "download.",
            level=logging.DEBUG,
        )
        return True
