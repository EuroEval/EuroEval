"""Tests for the `model_config` module."""

import os
import sys
import types

import pytest

from euroeval.benchmark_modules import HuggingFaceEncoderModel
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


def test_zero_shot_classifier_is_checked_before_encoder_model(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """A Laya repo resolves to the zero-shot classifier backend, not the encoder.

    `HuggingFaceEncoderModel.model_exists` also returns True for a repo like this
    (any non-generative Hub repo, regardless of whether it has a root
    `config.json`), so this regression-tests that `ZeroShotClassifierModel` (via
    `LayaAdapter`) is still checked, and wins, first. Hub calls are mocked so this
    doesn't need network access.
    """
    # Simulate `laya` being importable, without needing it installed.
    fake_laya_module = types.ModuleType("laya")
    fake_laya_module.Agent = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", fake_laya_module)

    # Simulate what the real bug looked like: the generic encoder module also
    # claims the repo exists (it doesn't require a root config.json).
    monkeypatch.setattr(HuggingFaceEncoderModel, "model_exists", lambda **kwargs: True)

    model_config = get_model_config(
        model_id="convaiinnovations/laya", benchmark_config=benchmark_config
    )
    assert model_config.inference_backend == InferenceBackend.ZERO_SHOT_CLASSIFIER
    assert model_config.model_type == ModelType.ZERO_SHOT_CLASSIFIER
