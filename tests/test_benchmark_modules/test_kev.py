"""Offline tests for the in-process Kev classifier integration."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, ANN204, ANN205, ANN206

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from datasets import DatasetDict

from euroeval.benchmark_modules.kev import KevModel, resolve_checkpoint_path
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType, TaskGroup
from euroeval.exceptions import InvalidBenchmark
from euroeval.tasks import KNOW


def test_generate_rejects_malformed_probabilities(dataset_config) -> None:
    """Malformed pointer output fails clearly instead of producing fake scores."""

    class FakeModel:
        def encode(self, tokenizer, record, strict=True):
            return record

        def probs(self, encoded):
            return [torch.tensor([0.2, 0.2])]

    model = _make_model(dataset_config, FakeModel())
    with pytest.raises(InvalidBenchmark, match="malformed class probabilities"):
        model.generate({"text": ["statement"]})


def _make_model(dataset_config: DatasetConfig, fake_model: object) -> KevModel:
    """Build a Kev adapter around fakes without mutating shared task definitions.

    Returns:
        The adapter configured around the fake model.
    """
    model = KevModel.__new__(KevModel)
    model.model = fake_model
    model.tokenizer = object()
    model.dataset_config = copy.deepcopy(dataset_config)
    model.dataset_config.task = copy.deepcopy(KNOW)
    model.dataset_config.task.task_group = TaskGroup.SEQUENCE_CLASSIFICATION
    model.dataset_config.labels = ["0", "1"]
    model.dataset_config.prompt_label_mapping = {"0": "yes", "1": "no"}
    model.dataset_config.instruction_prompt = "Choose: {text}"
    model.buffer = {"first_label_token_mapping": False}
    model.model_config = cast(ModelConfig, SimpleNamespace(model_id="org/kev-model"))
    model.benchmark_config = cast(BenchmarkConfig, SimpleNamespace(few_shot=False))
    return model


def test_generate_scores_each_sample_and_passes_serving_limit(dataset_config) -> None:
    """The decision head receives a serving limit and outputs preserve each sample."""

    class FakeModel:
        def __init__(self) -> None:
            self.records = []

        def encode(self, tokenizer, record, strict=True, max_length=None):
            assert strict is True
            assert max_length == 65_536
            self.records.append(record)
            return record

        def probs(self, encoded):
            return [torch.tensor([0.7, 0.3])]

    fake = FakeModel()
    model = _make_model(dataset_config, fake)
    output = model.generate({"text": ["first", "second"]})
    assert [record["state"] for record in fake.records] == ["first", "second"]
    assert output.sequences == ["yes", "yes"]
    assert output.scores is None
    labels = model.extract_labels_from_generation(
        input_batch={"prompt": ["first", "second"]}, model_output=output
    )
    assert labels == ["yes", "yes"]


def test_get_model_config_selects_kev_dispatch(benchmark_config) -> None:
    """Kev IDs map to the independent Kev inference backend."""
    config = KevModel.get_model_config("org/kev-model@release", benchmark_config)
    assert config.model_id == "org/kev-model"
    assert config.revision == "release"
    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER
    assert config.inference_backend is InferenceBackend.KEV


def test_multiple_choice_uses_per_sample_options_and_stable_letters(
    dataset_config,
) -> None:
    """Each sample is classified over its own choices and returns stable letters."""

    class FakeModel:
        def __init__(self) -> None:
            self.records = []

        def encode(self, tokenizer, record, strict=True, max_length=None):
            self.records.append(record)
            return record

        def probs(self, encoded):
            return [torch.tensor([0.1, 0.9])]

    fake = FakeModel()
    model = _make_model(dataset_config, fake)
    model.dataset_config.task.task_group = TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
    output = model.generate(
        {"text": ["Question?\nA. apple\nB. pear", "Other?\nA. red\nB. blue"]}
    )
    assert [record["questions"][0]["options"] for record in fake.records] == [
        ["apple", "pear"],
        ["red", "blue"],
    ]
    assert output.sequences == ["B", "B"]
    assert output.scores is None


def test_prepare_dataset_rejects_few_shot(dataset_config: DatasetConfig) -> None:
    """Kev cannot consume EuroEval few-shot demonstrations."""
    model = _make_model(dataset_config, SimpleNamespace())
    model.benchmark_config.few_shot = True
    with pytest.raises(InvalidBenchmark, match="does not support few-shot"):
        model.prepare_dataset(
            cast(DatasetDict, {"test": {"text": ["sample"]}}),
            model.dataset_config.task,
            0,
        )


def test_prepare_dataset_restores_raw_text(
    dataset_config: DatasetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kev applies the shared task preparation while exposing original input text."""
    original = ["raw sample"]
    mapped = ["rendered prompt"]
    calls = {}

    class Split:
        def __init__(self, text) -> None:
            self.text = text

        def __getitem__(self, key):
            assert key == "text"
            return self.text

        def add_column(self, key, values):
            assert key == "text"
            return Split(values)

        def remove_columns(self, key):
            assert key == "text"
            return Split([])

    class Data(dict):
        pass

    dataset = Data(test=Split(original))

    def helper(**kwargs):
        calls.update(kwargs)
        return Data(test=Split(mapped))

    monkeypatch.setattr(
        "euroeval.benchmark_modules.kev._prepare_dataset_helper", helper
    )
    model = _make_model(dataset_config, SimpleNamespace())
    prepared = model.prepare_dataset(
        cast(DatasetDict, dataset), model.dataset_config.task, 0
    )
    assert prepared["test"].text == original
    assert calls["generative_type"] is None
    assert calls["itr_idx"] == 0


def test_recognizes_only_complete_kev_checkpoints(
    tmp_path: Path, benchmark_config: BenchmarkConfig
) -> None:
    """A PEFT adapter is not Kev without its trained head and adapter weights."""
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
    assert KevModel.model_exists(str(tmp_path), benchmark_config) is False
    (tmp_path / "adapter_model.safetensors").touch()
    (tmp_path / "head.pt").touch()
    assert KevModel.model_exists(str(tmp_path), benchmark_config) is True


def test_resolve_checkpoint_path_reuses_local_directory(tmp_path: Path) -> None:
    """Local checkpoints bypass Hub downloads."""
    assert resolve_checkpoint_path(
        model_id=str(tmp_path), revision="tag", cache_dir="cache", token="secret"
    ) == str(tmp_path)
