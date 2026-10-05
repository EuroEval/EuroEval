"""Tests for the public Typesafe System One model interface."""

import copy
import dataclasses
import typing as t

import pytest

from euroeval.benchmark_modules.typesafe import TypesafeSystemOneModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig
from euroeval.enums import InferenceBackend, ModelType, TaskGroup
from euroeval.exceptions import InvalidBenchmark, InvalidModel, NeedsAdditionalArgument
from euroeval.model_loading import load_model


def test_api_key_must_come_from_typesafe_environment(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Hugging Face --api-key value is never used as the Typesafe key."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    config = TypesafeSystemOneModel.get_model_config(
        model_id="jev-latest", benchmark_config=benchmark_config
    )

    with pytest.raises(NeedsAdditionalArgument, match="TYPESAFE_API_KEY"):
        TypesafeSystemOneModel(
            model_config=config,
            dataset_config=dataset_config,
            benchmark_config=dataclasses.replace(benchmark_config, api_key="hf-token"),
            log_metadata=False,
        )


def test_generate_classifies_through_system_one_and_extracts_labels(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public generation and extraction preserve the configured label space."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-token")
    requests: list[dict[str, object]] = []
    labels = [
        dataset_config.prompt_label_mapping[label]
        for label in dataset_config.id2label.values()
    ]

    def post(url: str, **kwargs: object) -> _Response:
        requests.append({"url": url, **kwargs})
        return _Response({labels[0]: 0.8, labels[1]: 0.1, labels[2]: 0.1})

    monkeypatch.setattr("euroeval.benchmark_modules.typesafe.requests.post", post)
    config = dataclasses.replace(benchmark_config, api_key="hf-token")
    model = _make_model(config, dataset_config)
    output = model.generate(inputs={"text": ["A positive review"]})

    assert output.sequences == [labels[0]]
    assert model.extract_labels_from_generation(
        input_batch={"prompt": ["A positive review"]}, model_output=output
    ) == [labels[0]]
    headers = t.cast(dict[str, str], requests[0]["headers"])
    payload = t.cast(dict[str, object], requests[0]["json"])
    assert headers == {"Authorization": "Bearer typesafe-token"}
    assert payload["model"] == "jev-latest"
    assert output.scores is not None


class _Response:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities

    def json(self) -> dict[str, object]:
        return {"answers": {"name": {"probabilities": self.probabilities}}}

    def raise_for_status(self) -> None:
        return None


def _make_model(
    benchmark_config: BenchmarkConfig, dataset_config: DatasetConfig
) -> TypesafeSystemOneModel:
    """Create Jev model with its dedicated backend configuration.

    Returns:
        The configured hosted model.
    """
    config = TypesafeSystemOneModel.get_model_config(
        model_id="jev-latest", benchmark_config=benchmark_config
    )
    return TypesafeSystemOneModel(
        model_config=config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
        log_metadata=False,
    )


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_generate_rejects_invalid_probabilities(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
) -> None:
    """Invalid probability values fail closed instead of corrupting scores."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-token")
    labels = [
        dataset_config.prompt_label_mapping[label]
        for label in dataset_config.id2label.values()
    ]
    monkeypatch.setattr(
        "euroeval.benchmark_modules.typesafe.requests.post",
        lambda *_args, **_kwargs: _Response(
            {labels[0]: probability, labels[1]: 0.1, labels[2]: 0.1}
        ),
    )

    with pytest.raises(InvalidBenchmark, match="probability"):
        _make_model(benchmark_config, dataset_config).generate(
            inputs={"text": ["A review"]}
        )


def test_generate_rejects_malformed_response(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed service responses produce a clear benchmark error."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-token")
    monkeypatch.setattr(
        "euroeval.benchmark_modules.typesafe.requests.post",
        lambda *_args, **_kwargs: _Response({"unexpected": 1.0}),
    )

    with pytest.raises(InvalidBenchmark, match="did not return probabilities"):
        _make_model(benchmark_config, dataset_config).generate(
            inputs={"text": ["A review"]}
        )


def test_generate_supports_multiple_choice(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple-choice generation sends candidate answers as criteria."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-token")
    dataset_config = copy.deepcopy(dataset_config)
    dataset_config.task.task_group = TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
    labels = [
        dataset_config.prompt_label_mapping[label]
        for label in dataset_config.id2label.values()
    ]
    criteria_sent: list[dict[str, str | None]] = []

    def post(_url: str, **kwargs: object) -> _Response:
        payload = t.cast(dict[str, object], kwargs["json"])
        questions = t.cast(dict[str, object], payload["questions"])
        question = t.cast(dict[str, object], questions["name"])
        criteria_sent.append(t.cast(dict[str, str | None], question["criteria"]))
        return _Response({labels[0]: 0.8, labels[1]: 0.1, labels[2]: 0.1})

    monkeypatch.setattr("euroeval.benchmark_modules.typesafe.requests.post", post)
    model = _make_model(benchmark_config, dataset_config)
    model.generate(inputs={"text": ["Question?\na. First\nb. Second\nc. Third"]})

    assert criteria_sent == [
        {labels[0]: "First", labels[1]: "Second", labels[2]: "Third"}
    ]


def test_model_config_routes_jev_and_rejects_suffixes(
    benchmark_config: BenchmarkConfig,
) -> None:
    """Only the exact hosted model identifier is accepted for evaluation."""
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
    for suffixed_id in ("jev-latest@main", "jev-latest#subfolder"):
        assert TypesafeSystemOneModel.model_exists(
            model_id=suffixed_id, benchmark_config=benchmark_config
        )
        with pytest.raises(InvalidModel, match="exact model ID"):
            TypesafeSystemOneModel.get_model_config(
                model_id=suffixed_id, benchmark_config=benchmark_config
            )


def test_public_model_loading_routes_to_typesafe(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared model loader instantiates the dedicated hosted backend."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-token")
    config = TypesafeSystemOneModel.get_model_config(
        model_id="jev-latest", benchmark_config=benchmark_config
    )

    assert isinstance(
        load_model(
            model_config=config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        ),
        TypesafeSystemOneModel,
    )


def test_speed_task_is_rejected_before_remote_call(
    benchmark_config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted Jev refuses unsupported speed benchmarks during local setup."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-token")
    dataset_config = copy.deepcopy(dataset_config)
    dataset_config.task.task_group = TaskGroup.SPEED
    monkeypatch.setattr(
        "euroeval.benchmark_modules.typesafe.requests.post",
        lambda *_args, **_kwargs: pytest.fail("remote call must not be made"),
    )

    with pytest.raises(InvalidBenchmark, match="classification tasks only"):
        _make_model(benchmark_config, dataset_config)
