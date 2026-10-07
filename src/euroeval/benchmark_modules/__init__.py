"""The different types of modules that can be benchmarked."""

from .base import BenchmarkModule
from .dummy import DummyModel
from .fresh import FreshEncoderModel
from .gliner import GLiNERModel
from .hf import HuggingFaceEncoderModel
from .litellm import LiteLLMModel
from .typesafe import TypesafeSystemOneModel
from .vllm import VLLMModel
from .zero_shot_classifier import ZeroShotClassifierModel
