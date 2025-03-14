"""Constants used throughout the project."""

from .enums import TaskGroup
from .tasks import NER

# This is used as input to generative models; it cannot be a special token
DUMMY_FILL_VALUE = 100


# We need to raise the amount of tokens generated for reasoning models, to give them
# time to think
REASONING_MAX_TOKENS = 8_192


# The Hugging Face Hub pipeline tags used to classify models as generative
GENERATIVE_PIPELINE_TAGS = ["text-generation", "text2text-generation"]


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
MAX_LOGPROBS = 10


# We make sure to remove these metric attributed after each iteration, to avoid memory
# leaks
METRIC_ATTRIBUTES_TAKING_UP_MEMORY = ["cached_bertscorer"]


# Hugging Face Hub tags used to classify models as merge models
MERGE_TAGS = ["merge", "mergekit"]
