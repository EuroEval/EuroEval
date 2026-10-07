"""Offline tests for the GLiNER2 zero-shot classifier backend."""

import copy
import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

from euroeval.benchmark_modules.gliner2_classifier import GLiNER2ClassifierModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType, TaskGroup
from euroeval.exceptions import InvalidBenchmark


class TestGLiNER2Classifier:
    """Public backend behavior with a fake local checkpoint and extractor."""

    def test_dataset_switch_uses_new_label_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """Switching datasets changes the candidate labels without stale state."""
        monkeypatch.setitem(sys.modules, "gliner2", _gliner2_module())
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        config = dataclasses.replace(
            model_config,
            model_id=str(checkpoint),
            inference_backend=InferenceBackend.GLINER2,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            revision="main",
            param=None,
        )
        classifier = GLiNER2ClassifierModel(
            model_config=config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        new_mapping = {
            label: f"new-{prompt_label}"
            for label, prompt_label in dataset_config.prompt_label_mapping.items()
        }
        switched_config = copy.copy(dataset_config)
        switched_config.prompt_label_mapping = new_mapping
        classifier.update_dataset_config(dataset_config=switched_config)
        output = classifier.generate(inputs={"text": ["sample"]})
        labels = [new_mapping[label] for label in switched_config.id2label.values()]
        assert output.sequences == [labels[1]]

    def test_identifies_local_gliner_metadata_but_not_other_encoders(
        self,
        monkeypatch: pytest.MonkeyPatch,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """Local metadata prevents unrelated encoder checkpoints being claimed."""
        monkeypatch.setitem(sys.modules, "gliner2", _gliner2_module())
        checkpoint = tmp_path / "gliner"
        checkpoint.mkdir()
        config_path = checkpoint / "config.json"
        config_path.write_text(json.dumps({"model_type": "gliner2"}))
        assert (
            GLiNER2ClassifierModel.model_exists(
                model_id=str(checkpoint), benchmark_config=benchmark_config
            )
            is True
        )
        config_path.write_text(
            json.dumps(
                {
                    "model_type": "extractor",
                    "architectures": ["SpanExtractor"],
                    "architecture": "span",
                    "model_name": "microsoft/deberta-v3-large",
                    "config_version": 3,
                    "architecture_version": 1,
                    "span_head": {"span_mode": "markerV0"},
                }
            )
        )
        assert (
            GLiNER2ClassifierModel.model_exists(
                model_id=str(checkpoint), benchmark_config=benchmark_config
            )
            is True
        )
        config_path.write_text(
            json.dumps(
                {
                    "model_type": "extractor",
                    "architectures": ["SpanExtractor"],
                    "architecture": "span",
                    "model_name": "microsoft/deberta-v3-large",
                    "span_head": {"span_mode": "markerV0"},
                }
            )
        )
        assert (
            GLiNER2ClassifierModel.model_exists(
                model_id=str(checkpoint), benchmark_config=benchmark_config
            )
            is False
        )

    def test_rejects_multiple_choice_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """The backend does not silently treat unsupported task prompts as labels."""
        monkeypatch.setitem(sys.modules, "gliner2", _gliner2_module())
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        config = dataclasses.replace(
            model_config,
            model_id=str(checkpoint),
            inference_backend=InferenceBackend.GLINER2,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            revision="main",
            param=None,
        )
        multiple_choice = copy.copy(dataset_config)
        multiple_choice.task = dataclasses.replace(
            dataset_config.task, task_group=TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        )
        classifier = GLiNER2ClassifierModel(
            model_config=config,
            dataset_config=multiple_choice,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(InvalidBenchmark, match="sequence classification only"):
            classifier.generate(inputs={"text": ["sample"]})

    def test_returns_selected_labels_without_fabricated_scores(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """Output includes the selected label, and does not invent class scores."""
        monkeypatch.setitem(sys.modules, "gliner2", _gliner2_module())
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        config = dataclasses.replace(
            model_config,
            model_id=str(checkpoint),
            inference_backend=InferenceBackend.GLINER2,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            revision="main",
            param=None,
        )
        classifier = GLiNER2ClassifierModel(
            model_config=config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = classifier.generate(inputs={"text": ["sample"]})
        labels = [
            dataset_config.prompt_label_mapping[label]
            for label in dataset_config.id2label.values()
        ]
        assert output.sequences == [labels[1]]
        assert output.scores is None
        assert classifier.extract_labels_from_generation(
            input_batch={"prompt": ["sample"]}, model_output=output
        ) == [labels[1]]
        assert classifier.model_max_length == 512


def _gliner2_module() -> types.ModuleType:
    """Create a fake optional dependency module without loading model weights.

    Returns:
        A fake GLiNER2 module exposing AutoExtractor.
    """
    module = types.ModuleType("gliner2")
    module.AutoExtractor = types.SimpleNamespace(
        from_pretrained=lambda checkpoint, map_location: (
            FakeExtractor() if Path(checkpoint).exists() and str(map_location) else None
        )
    )
    return module


class FakeExtractor:
    """Small stand-in for the public AutoExtractor classification interface."""

    def classify_text(
        self, text: str, schema: dict, include_confidence: bool = False
    ) -> dict:
        """Return selected label in the shape documented by GLiNER2."""
        assert include_confidence is False
        return {"label": schema["label"][1]}
