"""Enums used in the project."""

from enum import Enum, auto


class AutoStrEnum(str, Enum):
    """StrEnum where auto() returns the field name in lower case."""

    def __repr__(self) -> str:
        """Return the value in upper case for better readability."""
        return self.value.upper()

    def __str__(self) -> str:
        """Return the value in upper case for better readability."""
        return self.value.upper()

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list
    ) -> str:
        return name.lower()


class BatchingPreference(AutoStrEnum):
    """The preference for batching.

    Attributes:
        NO_PREFERENCE:
            No preference for batching.
        SINGLE_SAMPLE:
            Single sample batching.
        ALL_AT_ONCE:
            All samples at once batching.
    """

    NO_PREFERENCE = auto()
    SINGLE_SAMPLE = auto()
    ALL_AT_ONCE = auto()


class DataType(AutoStrEnum):
    """The data type of the model weights.

    Attributes:
        FP32:
            32-bit floating point.
        FP16:
            16-bit floating point.
        BF16:
            16-bit bfloat.
    """

    FP32 = auto()
    FP16 = auto()
    BF16 = auto()


class Device(AutoStrEnum):
    """The compute device to use for the evaluation.

    Attributes:
        CPU:
            CPU device.
        MPS:
            MPS GPU, used in M-series MacBooks.
        CUDA:
            CUDA GPU, used with NVIDIA GPUs.
    """

    CPU = auto()
    MPS = auto()
    CUDA = auto()


class GenerativeType(AutoStrEnum):
    """The type of a generative model.

    Attributes:
        BASE:
            A base (i.e., pretrained) generative model.
        INSTRUCTION_TUNED:
            An instruction-tuned generative model.
        REASONING:
            A generative reasoning model.
    """

    BASE = auto()
    INSTRUCTION_TUNED = auto()
    REASONING = auto()


class InferenceBackend(AutoStrEnum):
    """The backend used for model inference.

    Attributes:
        TRANSFORMERS:
            Hugging Face `transformers` library.
        VLLM:
            VLLM library.
        LITELLM:
            LiteLLM library.
        DUMMY:
            The built-in dummy model, used for debugging.
        ZERO_SHOT_CLASSIFIER:
            A zero-shot classifier model, loaded through a dedicated adapter package
            (e.g. Laya), rather than through `transformers`, vLLM or LiteLLM directly.
    """

    TRANSFORMERS = auto()
    VLLM = auto()
    LITELLM = auto()
    DUMMY = auto()
    ZERO_SHOT_CLASSIFIER = auto()


class ModelType(AutoStrEnum):
    """The type of a model.

    Attributes:
        ENCODER:
            An encoder (i.e., BERT-style) model.
        GENERATIVE:
            A generative model. Can be either decoder or encoder-decoder (aka seq2seq).
        ZERO_SHOT_CLASSIFIER:
            A non-generative "decision" model (e.g. an encoder with trained heads,
            such as Laya) that is evaluated zero-shot through a dedicated adapter,
            returning calibrated per-label probabilities directly rather than by
            generating text or being fine-tuned.
    """

    ENCODER = auto()
    GENERATIVE = auto()
    ZERO_SHOT_CLASSIFIER = auto()

    def __repr__(self) -> str:
        """Return the value in upper case for better readability."""
        return self.value.upper()


class ParameterAdjustment(AutoStrEnum):
    """A model-level generation parameter adjustment learned from an API error.

    These are persisted across datasets on the model instance and re-applied to
    freshly built generation kwargs, so that a quirk learned on one dataset does not
    have to be rediscovered (via a failed request) on the next.

    Attributes:
        NO_STOP_SEQUENCES:
            The model does not support stop sequences.
        NO_LOGPROBS:
            The model does not support logprobs.
        LOGPROBS_MUST_BE_BOOLEAN:
            The model requires the `logprobs` argument to be a Boolean.
        NO_TOP_LOGPROBS:
            The model does not support `top_logprobs`, so its value is moved to
            `logprobs`.
        USE_MAX_TOKENS:
            The model does not support `max_completion_tokens`, so `max_tokens` is
            used instead.
        NO_MAX_TOKENS:
            The model does not support `max_tokens`.
        NO_TEMPERATURE:
            The model does not support the `temperature` parameter.
        TEMPERATURE_MUST_BE_ONE:
            The model requires the temperature to be set to 1.
        NO_JSON_SCHEMA:
            The model does not support JSON schemas, so the vanilla JSON object
            response format is used instead.
        THINKING_DISABLED_REQUIRES_TYPE:
            The model requires `thinking.type` to be `disabled` rather than just
            setting `budget_tokens` to 0.
        NO_SEED:
            The model does not support the `seed` parameter.
        NO_RESPONSE_FORMAT:
            The model does not support the `response_format` parameter.
    """

    NO_STOP_SEQUENCES = auto()
    NO_LOGPROBS = auto()
    LOGPROBS_MUST_BE_BOOLEAN = auto()
    NO_TOP_LOGPROBS = auto()
    USE_MAX_TOKENS = auto()
    NO_MAX_TOKENS = auto()
    NO_TEMPERATURE = auto()
    TEMPERATURE_MUST_BE_ONE = auto()
    NO_JSON_SCHEMA = auto()
    THINKING_DISABLED_REQUIRES_TYPE = auto()
    NO_SEED = auto()
    NO_RESPONSE_FORMAT = auto()


class ShotMode(AutoStrEnum):
    """The prompt shot policy used for a benchmark run.

    ``AUTO`` is an explicit public policy value that selects one or more concrete
    modes from model metadata. It is used during planning only and is never written
    to benchmark results.

    Attributes:
        AUTO:
            Select concrete modes automatically from the model and backend metadata.
        FEW_SHOT:
            Evaluate with task-specific few-shot examples where supported.
        ZERO_SHOT:
            Evaluate without few-shot examples.
    """

    AUTO = auto()
    FEW_SHOT = auto()
    ZERO_SHOT = auto()


class TaskGroup(AutoStrEnum):
    """The overall task group of a task.

    Attributes:
        SEQUENCE_CLASSIFICATION:
            Classification of documents.
        MULTIPLE_CHOICE_CLASSIFICATION:
            Classification of documents with multiple-choice options.
        TOKEN_CLASSIFICATION:
            Token-level classification.
        QUESTION_ANSWERING:
            Extractive question answering.
        TEXT_TO_TEXT:
            Text-to-text generation.
        SPEED:
            Speed benchmark.
    """

    SEQUENCE_CLASSIFICATION = auto()
    MULTIPLE_CHOICE_CLASSIFICATION = auto()
    TOKEN_CLASSIFICATION = auto()
    QUESTION_ANSWERING = auto()
    TEXT_TO_TEXT = auto()
    SPEED = auto()
