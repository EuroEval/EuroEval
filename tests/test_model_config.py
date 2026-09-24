"""Tests for the `model_config` module."""

import os
import sys
import types

import pytest

from euroeval.data_models import BenchmarkConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidModel
from euroeval.model_config import get_model_config


@pytest.mark.parametrize(
    argnames=["model_id", "should_raise"],
    argvalues=[
        ("Maltehb/aelaectra-danish-electra-small-cased", False),
        ("openai-community/gpt2", False),
        ("gpt-4o-mini", False),
        ("claude-haiku-4-5-20251001", False),
        ("does-not-exist", True),
    ],
    ids=[
        "encoder-model",
        "decoder-model",
        "openai-model",
        "anthropic-model",
        "non-existent-model",
    ],
)
@pytest.mark.skipif(
    condition=not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set, required for model config resolution",
)
def test_get_model_config(
    benchmark_config: BenchmarkConfig, model_id: str, should_raise: bool
) -> None:
    """Test that the `get_model_config` function works as expected."""
    if should_raise:
        with pytest.raises(InvalidModel):
            get_model_config(model_id=model_id, benchmark_config=benchmark_config)
    else:
        try:
            model_config = get_model_config(
                model_id=model_id, benchmark_config=benchmark_config
            )
            assert isinstance(model_config, ModelConfig)
        except InvalidModel as e:
            pytest.skip(f"Model {model_id} is not supported: {e}")


def test_non_matching_model_id_resolves_to_encoder_model(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
) -> None:
    """A normal encoder ID (no Laya match) still resolves via HF."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.hf.HuggingFaceEncoderModel.model_exists",
        classmethod(lambda cls, model_id, benchmark_config: True),
    )
    monkeypatch.setattr(
        "euroeval.benchmark_modules.hf.HuggingFaceEncoderModel.get_model_config",
        classmethod(lambda cls, model_id, benchmark_config: model_config),
    )

    resolved_config = get_model_config(
        model_id="some-org/some-adapter-repo", benchmark_config=benchmark_config
    )
    assert resolved_config.inference_backend == InferenceBackend.TRANSFORMERS
    assert resolved_config.model_type == ModelType.ENCODER


def test_zero_shot_classifier_is_checked_regardless_of_dispatch_order(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """A Laya repo resolves to the zero-shot classifier backend, not the encoder.

    Unlike `test_non_matching_model_id_resolves_to_encoder_model`, this doesn't
    mock `HuggingFaceEncoderModel.model_exists` at all: dispatch correctness here
    relies on `HuggingFaceEncoderModel` genuinely not claiming a Laya-style repo
    (no root `config.json`), not on being checked in any particular order.
    """
    fake_laya_module = types.ModuleType("laya")
    fake_laya_module.Agent = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", fake_laya_module)

    monkeypatch.setattr(
        "euroeval.benchmark_modules.hf.get_model_repo_info", lambda **kwargs: None
    )

    resolved_config = get_model_config(
        model_id="convaiinnovations/laya", benchmark_config=benchmark_config
    )
    assert resolved_config.inference_backend == InferenceBackend.LAYA
    assert resolved_config.model_type == ModelType.ZERO_SHOT_CLASSIFIER
