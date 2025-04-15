"""Constants used throughout the project."""

from .enums import TaskGroup
from .tasks import NER

# This is used as input to generative models; it cannot be a special token
DUMMY_FILL_VALUE = 100


# This is the maximum allowed context length for models for the purpose of this
# benchmark. We will still report the models' true maximum context length in the
# metadata, but we won't use it for evaluation, as vLLM needs to allocate memory for
# all tokens in the context.
MAX_CONTEXT_LENGTH = 5_000


# We need to raise the amount of tokens generated for reasoning models, to give them
# time to think
REASONING_MAX_TOKENS = 8_192


# The Hugging Face Hub pipeline tags used to classify models as generative
GENERATIVE_PIPELINE_TAGS = [
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "audio-text-to-text",
    "video-text-to-text",
]


# Used to disallow non-generative models to be evaluated on these task groups
GENERATIVE_DATASET_TASK_GROUPS = [TaskGroup.TEXT_TO_TEXT]


# Local models are required to have these files in their directory
LOCAL_MODELS_REQUIRED_FILES = ["config.json"]


# Tasks where we use structured generation for generative models
TASKS_USING_JSON = [NER]


# Tasks where we use log probabilities for generative models, rather than the raw
# completion
TASK_GROUPS_USING_LOGPROBS = [
    TaskGroup.SEQUENCE_CLASSIFICATION,
    TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
]


# The number of top log probabilities to return for generative models. For several APIs
# this is the maximum number of log probabilities that can be returned
MAX_LOGPROBS = 8


# We make sure to remove these metric attributes after each iteration, to avoid memory
# leaks
METRIC_ATTRIBUTES_TAKING_UP_MEMORY = ["cached_bertscorer"]


# Hugging Face Hub tags used to classify models as merge models
MERGE_TAGS = ["merge", "mergekit"]

# The minimum required CUDA compute capability for using bfloat16 in vLLM
VLLM_BF16_MIN_CUDA_COMPUTE_CAPABILITY = 8.0
