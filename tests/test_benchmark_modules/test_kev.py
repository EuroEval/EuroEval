"""Offline tests for the in-process Kev classifier integration."""

import json
from pathlib import Path

import pytest
import torch

from euroeval.benchmark_modules.kev import KevModel
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

    model = KevModel.__new__(KevModel)
    model.model = FakeModel()
    model.tokenizer = object()
    model.dataset_config = dataset_config
    model.dataset_config.task = KNOW
    model.dataset_config.task.task_group = TaskGroup.SEQUENCE_CLASSIFICATION
    model.dataset_config.prompt_label_mapping = {"0": "yes", "1": "no"}
    model.dataset_config.id2label = {0: "0", 1: "1"}
    model.dataset_config.instruction_prompt = "Choose: {text}"
    with pytest.raises(InvalidBenchmark, match="malformed class probabilities"):
        model.generate({"text": ["statement"]})


def test_generate_returns_pointer_probabilities_in_option_order(dataset_config) -> None:
    """The public generation API uses Kev's probabilities without renormalizing."""

    class FakeModel:
        def encode(self, tokenizer, record, strict=True):
            assert record["questions"][0]["options"] == ["yes", "no"]
            return record

        def probs(self, encoded):
            return [torch.tensor([0.7, 0.3])]

    model = KevModel.__new__(KevModel)
    model.model = FakeModel()
    model.tokenizer = object()
    model.dataset_config = dataset_config
    model.dataset_config.task = KNOW
    model.dataset_config.task.task_group = TaskGroup.SEQUENCE_CLASSIFICATION
    model.dataset_config.prompt_label_mapping = {"0": "yes", "1": "no"}
    model.dataset_config.id2label = {0: "0", 1: "1"}
    model.dataset_config.instruction_prompt = "Choose: {text}"
    output = model.generate({"text": ["A statement"]})
    assert output.sequences == ["yes"]
    assert output.scores[0][0][0][0] == "yes"
    assert output.scores[0][0][0][1] == pytest.approx(
        torch.log(torch.tensor(0.7)).item()
    )


def test_get_model_config_selects_kev_dispatch(benchmark_config) -> None:
    """Kev IDs map to the independent Kev inference backend."""
    config = KevModel.get_model_config("org/kev-model", benchmark_config)
    assert config.model_type is ModelType.ZERO_SHOT_CLASSIFIER
    assert config.inference_backend is InferenceBackend.KEV


def test_recognizes_only_complete_kev_checkpoints(
    tmp_path: Path, benchmark_config
) -> None:
    """A PEFT adapter is not Kev without its trained head and adapter weights."""
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 8}))
    assert KevModel.model_exists(str(tmp_path), benchmark_config) is False
    (tmp_path / "adapter_model.safetensors").touch()
    (tmp_path / "head.pt").touch()
    assert KevModel.model_exists(str(tmp_path), benchmark_config) is True
