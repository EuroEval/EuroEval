# ruff: noqa: I001
"""The different types of modules that can be benchmarked.

Import order matters here: ties in `BenchmarkModule.high_priority` are broken by
this module's `__dict__` iteration order, so `ZeroShotClassifierModel` (which must
be checked before the encoder/generative modules) is imported first among the
`high_priority = True` modules.
"""

from .base import BenchmarkModule
from .dummy import DummyModel
from .zero_shot_classifier import ZeroShotClassifierModel
from .fresh import FreshEncoderModel
from .hf import HuggingFaceEncoderModel
from .litellm import LiteLLMModel
from .vllm import VLLMModel
