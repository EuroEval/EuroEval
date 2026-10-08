"""Offline tests for the in-process Kev classifier integration."""

import copy
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch
from datasets import DatasetDict

from euroeval.benchmark_modules.kev import KevModel, resolve_checkpoint_path
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType, ShotMode, TaskGroup
from euroeval.exceptions import InvalidBenchmark
from euroeval.shot_modes import resolve_shot_modes
from euroeval.tasks import KNOW


def test_generate_rejects_malformed_probabilities(
    dataset_config: DatasetConfig,
) -> None:
    """Malformed pointer output fails clearly instead of producing fake scores."""

    class FakeModel:
        def encode(
            self,
            tokenizer: object,
            record: dict,
            *,
            strict: bool = True,
            max_state: int = 0,
            max_branch: int = 0,
        ) -> dict:
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
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
    model.buffer = {"first_label_token_mapping": True}
    model.model_config = cast(ModelConfig, SimpleNamespace(model_id="org/kev-model"))
    model.benchmark_config = cast(BenchmarkConfig, SimpleNamespace(few_shot=False))
    return model


def test_generate_scores_each_sample_and_passes_serving_limit(
    dataset_config: DatasetConfig,
) -> None:
    """The decision head receives a serving limit and outputs preserve each sample."""

    class FakeModel:
        def __init__(self) -> None:
            self.records = []

        def encode(
            self,
            tokenizer: object,
            record: dict,
            *,
            strict: bool = True,
            max_state: int = 0,
            max_branch: int = 0,
        ) -> dict:
            assert strict is True
            assert (max_state, max_branch) == (65_536, 73_728)
            self.records.append(record)
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            return [torch.tensor([0.7, 0.3])]

    fake = FakeModel()
    model = _make_model(dataset_config, fake)
    output = model.generate({"text": ["first", "second"]})
    assert [record["state"] for record in fake.records] == ["first", "second"]
    assert output.sequences == ["yes", "yes"]
    assert output.scores is not None
    assert output.scores[0][0][0][1] == pytest.approx(math.log(0.7))
    labels = model.extract_labels_from_generation(
        input_batch={"prompt": ["first", "second"]}, model_output=output
    )
    assert labels == ["yes", "yes"]


def test_get_model_config_selects_kev_dispatch(
    benchmark_config: BenchmarkConfig,
) -> None:
    """Kev IDs map to the independent Kev inference backend."""
    config = KevModel.get_model_config("org/kev-model@release", benchmark_config)
    assert config.model_id == "org/kev-model"
    assert config.revision == "release"
    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER
    assert config.inference_backend is InferenceBackend.KEV


def test_kev_forces_zero_shot(benchmark_config: BenchmarkConfig) -> None:
    """The classifier uses the existing zero-shot-only shot policy."""
    config = KevModel.get_model_config(
        model_id="jaredpalmer/kev-0.8b", benchmark_config=benchmark_config
    )
    assert resolve_shot_modes(
        model_config=config, requested_mode=ShotMode.FEW_SHOT
    ) == [ShotMode.ZERO_SHOT]


def test_load_uses_requested_revision_and_initialises_extractor(
    dataset_config: DatasetConfig,
    benchmark_config: BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the selected checkpoint and preserve classifier state."""
    requested: dict[str, object] = {}

    class FakeCheckpoint:
        def __init__(self, path: str) -> None:
            requested["path"] = path

        def load(self, device: str, options: object) -> tuple[object, object]:
            requested["device"] = device
            requested["options"] = options
            return object(), SimpleNamespace(eval=lambda: None)

    class FakeLoadOptions:
        def __init__(self, *, backend: str, dtype: torch.dtype) -> None:
            requested["backend"] = backend
            requested["dtype"] = dtype

    fake_package = ModuleType("kev")
    monkeypatch.setattr(fake_package, "__path__", [], raising=False)
    fake_checkpoint = ModuleType("kev.checkpoint")
    monkeypatch.setattr(fake_checkpoint, "Checkpoint", FakeCheckpoint, raising=False)
    monkeypatch.setattr(fake_checkpoint, "LoadOptions", FakeLoadOptions, raising=False)
    monkeypatch.setitem(sys.modules, "kev", fake_package)
    monkeypatch.setitem(sys.modules, "kev.checkpoint", fake_checkpoint)

    def fake_resolve(
        *, model_id: str, revision: str | None, cache_dir: str, token: str | None
    ) -> str:
        requested["model_id"] = model_id
        requested["revision"] = revision
        return "/cache/selected-revision"

    monkeypatch.setattr(
        "euroeval.benchmark_modules.kev.resolve_checkpoint_path", fake_resolve
    )
    config = KevModel.get_model_config(
        model_id="jaredpalmer/kev-0.8b@v1.0", benchmark_config=benchmark_config
    )
    model = KevModel(
        model_config=config,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
        log_metadata=False,
    )
    assert requested["model_id"] == "jaredpalmer/kev-0.8b"
    assert requested["revision"] == "v1.0"
    assert requested["path"] == "/cache/selected-revision"
    assert requested["backend"] == "torch"
    assert requested["dtype"] == torch.float32
    assert model.buffer["first_label_token_mapping"] is True


def test_multiple_choice_accepts_variable_options_without_fixed_labels(
    dataset_config: DatasetConfig,
) -> None:
    """Community datasets without fixed labels can still score their answer choices."""

    class FakeModel:
        def encode(
            self,
            tokenizer: object,
            record: dict,
            *,
            max_state: int,
            max_branch: int,
            strict: bool,
        ) -> dict:
            assert record["questions"][0]["options"] == ["apple", "pear"]
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            return [torch.tensor([0.1, 0.9])]

    model = _make_model(dataset_config, FakeModel())
    model.dataset_config.task.task_group = TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
    model.dataset_config.labels = []
    model.dataset_config.prompt_label_mapping = {}
    output = model.generate({"text": ["Question?\na. apple\nb. pear"]})
    assert output.sequences == ["b"]
    assert output.scores is not None
    assert [label for label, _ in output.scores[0][0]] == ["b", "a"]


def test_multiple_choice_uses_per_sample_options_and_stable_letters(
    dataset_config: DatasetConfig,
) -> None:
    """Each sample is classified over its own choices and returns stable letters."""

    class FakeModel:
        def __init__(self) -> None:
            self.records = []

        def encode(
            self,
            tokenizer: object,
            record: dict,
            *,
            strict: bool = True,
            max_state: int = 0,
            max_branch: int = 0,
        ) -> dict:
            self.records.append(record)
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            return [torch.tensor([0.1, 0.9])]

    fake = FakeModel()
    model = _make_model(dataset_config, fake)
    model.dataset_config.task.task_group = TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
    model.dataset_config.labels = ["a", "b"]
    model.dataset_config.prompt_label_mapping = {"a": "a", "b": "b"}
    output = model.generate(
        {"text": ["Question?\nA. apple\nB. pear", "Other?\nA. red\nB. blue"]}
    )
    assert [record["questions"][0]["options"] for record in fake.records] == [
        ["apple", "pear"],
        ["red", "blue"],
    ]
    assert output.sequences == ["b", "b"]
    assert output.scores is not None
    assert model.extract_labels_from_generation(
        input_batch={
            "prompt": ["Question?\nA. apple\nB. pear", "Other?\nA. red\nB. blue"]
        },
        model_output=output,
    ) == ["b", "b"]


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
        def __init__(self, text: list[str]) -> None:
            self.text = text

        def __getitem__(self, key: str) -> list[str]:
            assert key == "text"
            return self.text

        def add_column(self, key: str, values: list[str]) -> "Split":
            assert key == "text"
            return Split(values)

        def remove_columns(self, key: str) -> "Split":
            assert key == "text"
            return Split([])

    class Data(dict):
        pass

    dataset = Data(test=Split(original))

    def helper(**kwargs: object) -> Data:
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
