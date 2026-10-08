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
    """Malformed pointer output fails clearly instead of producing fake scores.

    Args:
        dataset_config: The test dataset configuration.
    """

    class FakeModel:
        """Stand in for the pointer-head model during an offline test."""

        def encode(
            self,
            tokenizer: object,
            record: dict,
            *,
            strict: bool = True,
            max_state: int = 0,
            max_branch: int = 0,
        ) -> dict:
            """Return the record in the shape the fake pointer head consumes.

            Args:
                tokenizer: The unused tokenizer stand-in.
                record: The constructed Kev input record.
                strict: Whether the record should use strict encoding.
                max_state: The maximum state length requested by the adapter.
                max_branch: The maximum branch length requested by the adapter.

            Returns:
                The unchanged input record.
            """
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            """Return controlled class probabilities for this test.

            Args:
                encoded: The encoded record supplied to the fake head.

            Returns:
                Controlled probabilities for the available choices.
            """
            return [torch.tensor([0.2, 0.2])]

    model = _make_model(dataset_config, FakeModel())
    with pytest.raises(InvalidBenchmark, match="malformed class probabilities"):
        model.generate({"text": ["statement"]})


def _make_model(dataset_config: DatasetConfig, fake_model: object) -> KevModel:
    """Build a Kev adapter around fakes without mutating shared task definitions.

    Args:
        dataset_config: The dataset configuration to copy for this test.
        fake_model: The stand-in for the loaded Kev model.

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
    """The decision head receives a serving limit and outputs preserve each sample.

    Args:
        dataset_config: The test dataset configuration.
    """

    class FakeModel:
        """Stand in for the pointer-head model during an offline test."""

        def __init__(self) -> None:
            """Initialize the fake model's collected input records."""
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
            """Return the record in the shape the fake pointer head consumes.

            Args:
                tokenizer: The unused tokenizer stand-in.
                record: The constructed Kev input record.
                strict: Whether the record should use strict encoding.
                max_state: The maximum state length requested by the adapter.
                max_branch: The maximum branch length requested by the adapter.

            Returns:
                The unchanged input record.
            """
            assert strict is True
            assert (max_state, max_branch) == (65_536, 73_728)
            self.records.append(record)
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            """Return controlled class probabilities for this test.

            Args:
                encoded: The encoded record supplied to the fake head.

            Returns:
                Controlled probabilities for the available choices.
            """
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
    """Kev IDs map to the independent Kev inference backend.

    Args:
        benchmark_config: The benchmark settings used for model detection.
    """
    config = KevModel.get_model_config("org/kev-model@release", benchmark_config)
    assert config.model_id == "org/kev-model"
    assert config.revision == "release"
    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER
    assert config.inference_backend is InferenceBackend.KEV


def test_kev_forces_zero_shot(benchmark_config: BenchmarkConfig) -> None:
    """The classifier uses the existing zero-shot-only shot policy.

    Args:
        benchmark_config: The benchmark settings used for shot selection.
    """
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
    """Load the selected checkpoint and preserve classifier state.

    Args:
        dataset_config: The dataset configuration for the Kev model.
        benchmark_config: The benchmark settings used to load the model.
        monkeypatch: The fixture replacing checkpoint-loading dependencies.
    """
    requested: dict[str, object] = {}

    class FakeCheckpoint:
        """Capture options sent to Kev's checkpoint loader."""

        def __init__(self, path: str) -> None:
            """Capture the options passed to this test double.

            Args:
                path: The resolved checkpoint path.
            """
            requested["path"] = path

        def load(self, device: str, options: object) -> tuple[object, object]:
            """Record checkpoint options and provide a fake loaded model.

            Args:
                device: The requested inference device.
                options: The requested Kev load options.

            Returns:
                A tokenizer and fake loaded model.
            """
            requested["device"] = device
            requested["options"] = options
            return object(), SimpleNamespace(eval=lambda: None)

    class FakeLoadOptions:
        """Capture the backend and dtype requested by EuroEval."""

        def __init__(self, *, backend: str, dtype: torch.dtype) -> None:
            """Capture the options passed to this test double.

            Args:
                backend: The requested model backend.
                dtype: The requested model dtype.
            """
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
        """Record the requested revision without downloading a checkpoint.

        Args:
            model_id: The requested checkpoint ID.
            revision: The requested checkpoint revision.
            cache_dir: The unused checkpoint cache directory.
            token: The unused Hub token.

        Returns:
            The fake local checkpoint directory.
        """
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
    """Community datasets without fixed labels can still score their answer choices.

    Args:
        dataset_config: The multiple-choice test dataset configuration.
    """

    class FakeModel:
        """Stand in for the pointer-head model during an offline test."""

        def encode(
            self,
            tokenizer: object,
            record: dict,
            *,
            max_state: int,
            max_branch: int,
            strict: bool,
        ) -> dict:
            """Return the record in the shape the fake pointer head consumes.

            Args:
                tokenizer: The unused tokenizer stand-in.
                record: The constructed Kev input record.
                max_state: The maximum state length requested by the adapter.
                max_branch: The maximum branch length requested by the adapter.
                strict: Whether the record should use strict encoding.

            Returns:
                The unchanged input record.
            """
            assert record["questions"][0]["options"] == ["apple", "pear"]
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            """Return controlled class probabilities for this test.

            Args:
                encoded: The encoded record supplied to the fake head.

            Returns:
                Controlled probabilities for the available choices.
            """
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
    """Each sample is classified over its own choices and returns stable letters.

    Args:
        dataset_config: The multiple-choice test dataset configuration.
    """

    class FakeModel:
        """Stand in for the pointer-head model during an offline test."""

        def __init__(self) -> None:
            """Initialize the fake model's collected input records."""
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
            """Return the record in the shape the fake pointer head consumes.

            Args:
                tokenizer: The unused tokenizer stand-in.
                record: The constructed Kev input record.
                strict: Whether the record should use strict encoding.
                max_state: The maximum state length requested by the adapter.
                max_branch: The maximum branch length requested by the adapter.

            Returns:
                The unchanged input record.
            """
            self.records.append(record)
            return record

        def probs(self, encoded: dict) -> list[torch.Tensor]:
            """Return controlled class probabilities for this test.

            Args:
                encoded: The encoded record supplied to the fake head.

            Returns:
                Controlled probabilities for the available choices.
            """
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
    """Kev cannot consume EuroEval few-shot demonstrations.

    Args:
        dataset_config: The test dataset configuration.
    """
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
    """Kev applies the shared task preparation while exposing original input text.

    Args:
        dataset_config: The test dataset configuration.
        monkeypatch: The fixture replacing the dataset preparation helper.
    """
    original = ["raw sample"]
    mapped = ["rendered prompt"]
    calls = {}

    class Split:
        """Represent a minimal test dataset split."""

        def __init__(self, text: list[str]) -> None:
            """Store the fake split's text column.

            Args:
                text: The column contents of the fake split.
            """
            self.text = text

        def __getitem__(self, key: str) -> list[str]:
            """Return the requested fake dataset column.

            Args:
                key: The name of the fake dataset column.

            Returns:
                The values in the requested column.
            """
            assert key == "text"
            return self.text

        def add_column(self, key: str, values: list[str]) -> "Split":
            """Return a fake split with the supplied column values.

            Args:
                key: The name of the fake dataset column.
                values: The values to place in the fake dataset column.

            Returns:
                A fake split containing the supplied values.
            """
            assert key == "text"
            return Split(values)

        def remove_columns(self, key: str) -> "Split":
            """Return an empty fake split after removing its column.

            Args:
                key: The name of the fake dataset column.

            Returns:
                An empty fake split.
            """
            assert key == "text"
            return Split([])

    class Data(dict):
        """Hold fake dataset splits by name."""

        pass

    dataset = Data(test=Split(original))

    def helper(**kwargs: object) -> Data:
        """Record dataset preparation arguments and return a fake split.

        Args:
            kwargs: The preparation settings passed by the adapter.

        Returns:
            A prepared fake dataset split.
        """
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
    """A PEFT adapter is not Kev without its trained head and adapter weights.

    Args:
        tmp_path: The temporary directory for incomplete and complete checkpoints.
        benchmark_config: The benchmark settings used for model detection.
    """
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
    assert KevModel.model_exists(str(tmp_path), benchmark_config) is False
    (tmp_path / "adapter_model.safetensors").touch()
    (tmp_path / "head.pt").touch()
    assert KevModel.model_exists(str(tmp_path), benchmark_config) is True


def test_resolve_checkpoint_path_reuses_local_directory(tmp_path: Path) -> None:
    """Local checkpoints bypass Hub downloads.

    Args:
        tmp_path: The local checkpoint directory.
    """
    assert resolve_checkpoint_path(
        model_id=str(tmp_path), revision="tag", cache_dir="cache", token="secret"
    ) == str(tmp_path)
