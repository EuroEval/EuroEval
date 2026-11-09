"""Tests for typed decision-model registration and metadata."""

import json
import os
import typing as t
from dataclasses import replace
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import pytest

from euroeval.benchmark_modules.decision_model import DecisionModel
from euroeval.benchmark_modules.zero_shot_classifier import ZeroShotClassifierModel
from euroeval.data_models import BenchmarkConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidModel
from euroeval.model_config import get_model_config


def test_api_base_dummy_key_falls_back_to_hf_token(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """CLI inference credentials must not mask the Hub token environment."""
    monkeypatch.setenv("HF_TOKEN", "hub-token")
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.get_hf_token",
        lambda *, api_key: api_key or os.environ["HF_TOKEN"],
    )
    config_with_api_base = replace(
        benchmark_config, api_key=None, api_base="http://127.0.0.1:18000"
    )
    assert config_with_api_base.api_key == "dummy"

    model_info_tokens: list[str | None] = []
    download_tokens: list[str | None] = []
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.HfApi.model_info",
        lambda self, **kwargs: (
            model_info_tokens.append(self.token)
            or SimpleNamespace(tags=["decision-model"])
        ),
    )
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.hf_hub_download",
        lambda **kwargs: (
            download_tokens.append(kwargs["token"])
            or _metadata_file(format_id="onejev-slot-v2", package="flide")
        ),
    )

    assert DecisionModel.model_exists(
        model_id="syvai/claes-v1", benchmark_config=config_with_api_base
    )
    get_model_config(model_id="syvai/claes-v1", benchmark_config=config_with_api_base)

    assert model_info_tokens == ["hub-token", "hub-token"]
    assert download_tokens == ["hub-token"]

    config_with_real_key = replace(config_with_api_base, api_key="real-key")
    assert DecisionModel.model_exists(
        model_id="syvai/claes-v1", benchmark_config=config_with_real_key
    )
    get_model_config(model_id="syvai/claes-v1", benchmark_config=config_with_real_key)

    assert model_info_tokens[-2:] == ["real-key", "real-key"]
    assert download_tokens[-1] == "real-key"


def _metadata_file(*, format_id: str, package: str) -> str:
    """Create a temporary decision metadata document for the public seam.

    Args:
        format_id: Protocol identifier stored in the fixture document.
        package: Package identifier stored in the fixture document.

    Returns:
        The temporary document path.
    """
    file = NamedTemporaryFile(mode="w", delete=False)
    json.dump({"format_id": format_id, "package": package}, file)
    file.close()
    return file.name


def test_laya_is_not_claimed_as_a_decision_model(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """The decision tag keeps Laya on its existing classifier adapter."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.HfApi.model_info",
        lambda self, **kwargs: SimpleNamespace(tags=["zero-shot-classification"]),
    )

    assert not DecisionModel.model_exists(
        model_id="convaiinnovations/laya", benchmark_config=benchmark_config
    )


def test_laya_vocab_size_uses_configured_hub_credentials_and_cache(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """Recover Laya vocabulary metadata with the benchmark's Hub settings."""
    model = object.__new__(ZeroShotClassifierModel)
    model.model_config = t.cast(
        ModelConfig, SimpleNamespace(model_id="org/gated", revision="rev")
    )
    model.benchmark_config = replace(
        benchmark_config, api_key="hub-token", cache_dir="/benchmark-cache"
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "euroeval.benchmark_modules.zero_shot_classifier.recover_numeric_metadata",
        lambda **kwargs: calls.append(kwargs) or {"vocabulary_size": 32000},
    )

    assert model.vocab_size == 32000
    assert calls == [
        {
            "model_id": "org/gated@rev",
            "token": "hub-token",
            "cache_dir": "/benchmark-cache",
        }
    ]


def test_numeric_metadata_uses_configured_hub_credentials_and_cache(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """Recover gated model metadata with the benchmark's Hub settings."""
    model = object.__new__(DecisionModel)
    model.model_config = t.cast(
        ModelConfig, SimpleNamespace(model_id="org/gated", revision="rev")
    )
    model.benchmark_config = replace(
        benchmark_config, api_key="hub-token", cache_dir="/benchmark-cache"
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.recover_numeric_metadata",
        lambda **kwargs: calls.append(kwargs) or {},
    )

    assert model.numeric_metadata == {}
    assert calls == [
        {
            "model_id": "org/gated@rev",
            "token": "hub-token",
            "cache_dir": "/benchmark-cache",
        }
    ]


@pytest.mark.parametrize(
    "model_id",
    ["org/arbitrary-model@abc123", "OmniJev/OneJev1.6-0.8B", "syvai/claes-v1"],
)
def test_tagged_models_resolve_supported_checkpoint_metadata(
    model_id: str, monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """Recognise the actual format without relying on repository identity."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.HfApi.model_info",
        lambda self, **kwargs: SimpleNamespace(tags=["decision-model"]),
    )
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.hf_hub_download",
        lambda **kwargs: _metadata_file(format_id="onejev-slot-v2", package="flide"),
    )

    config = get_model_config(model_id=model_id, benchmark_config=benchmark_config)

    expected_id, _, *revision = model_id.partition("@")
    assert config.model_id == expected_id
    assert config.revision == (revision[0] or "main")
    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER
    assert config.inference_backend is InferenceBackend.SYSTEM_ONE


def test_unsupported_decision_metadata_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """Do not route unknown tagged protocols or packages to this adapter."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.hf_hub_download",
        lambda **kwargs: _metadata_file(format_id="onejev-slot-v2", package="unknown"),
    )

    with pytest.raises(InvalidModel, match="Unsupported decision-model metadata"):
        DecisionModel.get_model_config(
            model_id="org/typed", benchmark_config=benchmark_config
        )
