"""Tests for explicit model metadata overrides."""

from dataclasses import replace

import pytest

from euroeval.benchmark_modules.base import BenchmarkModule
from euroeval.benchmark_modules.dummy import DummyModel
from euroeval.benchmark_modules.fresh import FreshEncoderModel
from euroeval.benchmark_modules.hf import HuggingFaceEncoderModel
from euroeval.benchmark_modules.litellm import LiteLLMModel
from euroeval.benchmark_modules.vllm import VLLMModel
from euroeval.data_models import BenchmarkConfig


@pytest.mark.parametrize(
    argnames="model_cls",
    argvalues=[
        HuggingFaceEncoderModel,
        VLLMModel,
        FreshEncoderModel,
        LiteLLMModel,
        DummyModel,
    ],
)
def test_num_parameters_override(
    model_cls: type[BenchmarkModule], benchmark_config: BenchmarkConfig
) -> None:
    """An explicit parameter count takes precedence over backend inference."""
    model = model_cls.__new__(model_cls)
    model.benchmark_config = replace(benchmark_config, num_parameters=123_456)

    assert model.num_params == 123_456
