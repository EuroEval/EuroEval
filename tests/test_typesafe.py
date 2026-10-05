"""Tests for the public Typesafe System One model interface."""

import dataclasses
import typing as t

import pytest

from euroeval.benchmark_modules.typesafe import TypesafeSystemOneModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig
from euroeval.enums import InferenceBackend, ModelType


def test_generate_calls_system_one_with_bearer_and_maps_probabilities(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification uses a mocked protocol response and emits log probabilities."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-token")
    requests: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> _Response:
        requests.append({"url": url, **kwargs})
        return _Response()

    monkeypatch.setattr("euroeval.benchmark_modules.typesafe.requests.post", post)
    typesafe_config = dataclasses.replace(benchmark_config, api_key=None)
    config = TypesafeSystemOneModel.get_model_config(
        model_id="jev-latest", benchmark_config=typesafe_config
    )
    model = TypesafeSystemOneModel(
        model_config=config,
        dataset_config=dataset_config,
        benchmark_config=typesafe_config,
        log_metadata=False,
    )

    output = model.generate(inputs={"text": ["A positive review"]})

    assert output.sequences == ["positive"]
    headers = t.cast(dict[str, str], requests[0]["headers"])
    payload = t.cast(dict[str, object], requests[0]["json"])
    questions = t.cast(dict[str, object], payload["questions"])
    named_question = t.cast(dict[str, object], questions["name"])
    assert headers == {"Authorization": "Bearer test-token"}
    assert payload["model"] == "jev-latest"
    assert named_question["type"] == "choice"
    assert output.scores is not None


class _Response:
    def json(self) -> dict[str, object]:
        return {
            "answers": {"name": {"probabilities": {"positive": 0.8, "negative": 0.2}}}
        }

    def raise_for_status(self) -> None:
        return None


def test_model_config_routes_jev_to_typesafe(benchmark_config: BenchmarkConfig) -> None:
    """The hosted Jev identifier resolves to its dedicated zero-shot backend."""
    config = TypesafeSystemOneModel.get_model_config(
        model_id="jev-latest", benchmark_config=benchmark_config
    )

    assert config.inference_backend == InferenceBackend.TYPESAFE
    assert config.model_type == ModelType.ZERO_SHOT_CLASSIFIER
    assert TypesafeSystemOneModel.model_exists(
        model_id="jev-latest", benchmark_config=benchmark_config
    )
    assert not TypesafeSystemOneModel.model_exists(
        model_id="some-other-model", benchmark_config=benchmark_config
    )
