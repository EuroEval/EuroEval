"""Tests for typed decision-model registration and metadata."""

import json
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import pytest

from euroeval.benchmark_modules.decision_model import DecisionModel
from euroeval.data_models import BenchmarkConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidModel
from euroeval.model_config import get_model_config


def test_tag_precedes_generic_generation_classification(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """A decision tag claims the model before the generative fallback."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.HfApi.model_info",
        lambda self, **kwargs: SimpleNamespace(tags=["decision-model"]),
    )
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.hf_hub_download",
        lambda **kwargs: _metadata_file(format_id="systemone-v1"),
    )

    config = get_model_config(model_id="org/typed", benchmark_config=benchmark_config)

    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER


def _metadata_file(*, format_id: str) -> str:
    """Create a temporary decision metadata document for the public seam.

    Args:
        format_id: Protocol identifier stored in the fixture document.

    Returns:
        The temporary document path.
    """
    file = NamedTemporaryFile(mode="w", delete=False)
    json.dump({"format_id": format_id}, file)
    file.close()
    return file.name


def test_tagged_arbitrary_id_resolves_as_decision_model(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """Select tagged arbitrary repositories and retain the requested revision."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.HfApi.model_info",
        lambda self, **kwargs: SimpleNamespace(tags=["decision-model"]),
    )
    assert DecisionModel.model_exists(
        model_id="org/arbitrary-model@abc123", benchmark_config=benchmark_config
    )
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.hf_hub_download",
        lambda **kwargs: _metadata_file(format_id="systemone-v1"),
    )

    config = DecisionModel.get_model_config(
        model_id="org/arbitrary-model@abc123", benchmark_config=benchmark_config
    )

    assert config.model_id == "org/arbitrary-model"
    assert config.revision == "abc123"
    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER
    assert config.inference_backend is InferenceBackend.SYSTEM_ONE


def test_unsupported_decision_format_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
) -> None:
    """Do not route unknown tagged protocols to a generic generation backend."""
    monkeypatch.setattr(
        "euroeval.benchmark_modules.decision_model.hf_hub_download",
        lambda **kwargs: _metadata_file(format_id="other-format"),
    )

    with pytest.raises(InvalidModel, match="Unsupported decision-model format"):
        DecisionModel.get_model_config(
            model_id="org/typed", benchmark_config=benchmark_config
        )
