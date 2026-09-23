"""Tests for the `model_config` module."""

import os
import sys
import types

import pytest

from euroeval import benchmark_modules
from euroeval.benchmark_modules.base import BenchmarkModule
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
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
) -> None:
    """A Laya repo resolves to the zero-shot classifier backend, not the encoder.

    `HuggingFaceEncoderModel.model_exists` is mocked to unconditionally return
    True here, simulating a repo shape (e.g. no root `config.json`, or a repo in a
    subfolder) that it would also claim. Even so, `ZeroShotClassifierModel` (via
    `LayaAdapter`) must still be the one that resolves the model, since
    `ZeroShotClassifierModel.priority` is explicitly higher than
    `HuggingFaceEncoderModel.priority` in `model_config.get_model_config` -- this is
    what actually determines dispatch order, not import order or a heuristic on
    `HuggingFaceEncoderModel.model_exists`.
    """
    # Simulate `laya` being importable, without needing it installed.
    fake_laya_module = types.ModuleType("laya")
    fake_laya_module.Agent = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", fake_laya_module)

    monkeypatch.setattr(
        "euroeval.benchmark_modules.hf.HuggingFaceEncoderModel.model_exists",
        classmethod(lambda cls, model_id, benchmark_config: True),
    )
    # This would raise if reached, since it isn't mocked; asserting it's never
    # called is exactly the point -- `ZeroShotClassifierModel` must win the race.
    monkeypatch.setattr(
        "euroeval.benchmark_modules.hf.HuggingFaceEncoderModel.get_model_config",
        classmethod(lambda cls, model_id, benchmark_config: model_config),
    )

    resolved_config = get_model_config(
        model_id="convaiinnovations/laya", benchmark_config=benchmark_config
    )
    assert resolved_config.inference_backend == InferenceBackend.ZERO_SHOT_CLASSIFIER
    assert resolved_config.model_type == ModelType.ZERO_SHOT_CLASSIFIER


def test_non_matching_model_id_resolves_to_encoder_model(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_config: BenchmarkConfig,
    model_config: ModelConfig,
) -> None:
    """A normal encoder ID (no zero-shot adapter matches) still resolves via HF.

    Regression test for the dispatch order fix: raising
    `ZeroShotClassifierModel.priority` above `HuggingFaceEncoderModel.priority`
    must not make every model resolve to the zero-shot classifier backend --
    `ZeroShotClassifierModel.model_exists` still correctly reports False for a
    model ID that no registered adapter matches (e.g. a plain adapter/PEFT repo,
    which isn't a Laya checkpoint), so dispatch falls through to the encoder.
    """
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


class TestDispatchPriority:
    """Tests for the built-in `priority` ordering and the legacy `high_priority`."""

    def test_built_in_modules_have_distinct_priorities_in_expected_order(self) -> None:
        """Built-in benchmark modules sort into the documented dispatch order.

        Regression test for the tie between `HuggingFaceEncoderModel` and
        `VLLMModel` (both previously `priority = 20`) and for
        `FreshEncoderModel` silently inheriting `HuggingFaceEncoderModel`'s
        priority instead of being checked before it.
        """
        all_benchmark_modules = [
            cls
            for cls in benchmark_modules.__dict__.values()
            if isinstance(cls, type)
            and issubclass(cls, benchmark_modules.BenchmarkModule)
            and cls is not benchmark_modules.BenchmarkModule
        ]
        priorities = [cls.get_dispatch_priority() for cls in all_benchmark_modules]
        assert len(set(priorities)) == len(priorities), (
            "Every built-in benchmark module must have a distinct priority."
        )

        ordered = sorted(
            all_benchmark_modules,
            key=lambda cls: cls.get_dispatch_priority(),
            reverse=True,
        )
        ordered_names = [cls.__name__ for cls in ordered]
        assert ordered_names == [
            "DummyModel",
            "ZeroShotClassifierModel",
            "FreshEncoderModel",
            "HuggingFaceEncoderModel",
            "VLLMModel",
            "LiteLLMModel",
        ]

    def test_legacy_high_priority_is_honoured_with_deprecation_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A subclass that only sets `high_priority` still gets a usable priority."""

        class LegacyHighPriorityModule(BenchmarkModule):
            """A stand-in for a third-party subclass written against an old release."""

            high_priority = True

        class LegacyLowPriorityModule(BenchmarkModule):
            """A stand-in for a third-party subclass written against an old release."""

            high_priority = False

        with caplog.at_level("WARNING", logger="euroeval"):
            high_priority_value = LegacyHighPriorityModule.get_dispatch_priority()
            low_priority_value = LegacyLowPriorityModule.get_dispatch_priority()

        assert high_priority_value > low_priority_value
        assert any(
            "deprecated" in record.message and "high_priority" in record.message
            for record in caplog.records
        )

    def test_priority_takes_precedence_over_high_priority(self) -> None:
        """A subclass setting both `priority` and `high_priority` uses `priority`."""

        class BothSetModule(BenchmarkModule):
            """A subclass explicitly setting the new attribute."""

            priority = 1000
            high_priority = False

        assert BothSetModule.get_dispatch_priority() == 1000
