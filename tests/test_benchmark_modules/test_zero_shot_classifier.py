"""Unit tests for the `zero_shot_classifier` module."""

import dataclasses
import math

import pytest

from euroeval.benchmark_modules.zero_shot_classifier import ZeroShotClassifierModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from euroeval.languages import DANISH
from euroeval.model_config import get_model_config
from euroeval.model_loading import load_model
from euroeval.tasks import HALLU, KNOW
from euroeval.zero_shot_adapters import ZeroShotClassifierAdapter


class FakeAdapter(ZeroShotClassifierAdapter):
    """A fake adapter used to test the zero-shot classifier interface."""

    name = "fake"
    allowed_params = ["variant-a", "variant-b"]
    max_length = 512

    @classmethod
    def matches(cls, model_id: str, benchmark_config: BenchmarkConfig) -> bool:
        """Match model IDs starting with 'fake-zero-shot'.

        Returns:
            Whether the model ID matches.
        """
        return model_id.startswith("fake-zero-shot")

    @classmethod
    def num_params(cls, model_id: str, param: str | None) -> int:
        """Return a fixed parameter count."""
        return 123

    def __init__(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Store the configs; no real model is loaded."""
        self.model_config = model_config
        self.benchmark_config = benchmark_config

    def classify(
        self, texts: list[str], candidate_labels: list[str], instructions: str
    ) -> list[dict[str, float]]:
        """Always predict the first candidate label with high confidence.

        Returns:
            One label-probability dictionary per text.
        """
        rest = 0.01 / max(len(candidate_labels) - 1, 1)
        probs = {label: rest for label in candidate_labels}
        probs[candidate_labels[0]] = 1 - sum(
            v for k, v in probs.items() if k != candidate_labels[0]
        )
        return [dict(probs) for _ in texts]


class NeedsExtraAdapter(ZeroShotClassifierAdapter):
    """An adapter whose extra is never installed, used to test that error path."""

    name = "needs-extra"
    allowed_params = []
    max_length = -1

    @classmethod
    def matches(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool | NeedsExtraInstalled:
        """Report the extra as missing for a specific model ID prefix.

        Returns:
            Whether the model ID matches, or a `NeedsExtraInstalled` error.
        """
        if model_id.startswith("needs-extra"):
            return NeedsExtraInstalled(extra="fake-extra")
        return False

    @classmethod
    def num_params(cls, model_id: str, param: str | None) -> int:
        """Unknown parameter count.

        Returns:
            -1.
        """
        return -1

    def __init__(
        self, model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Never actually constructed in these tests."""
        raise NotImplementedError

    def classify(
        self, texts: list[str], candidate_labels: list[str], instructions: str
    ) -> list[dict[str, float]]:
        """Never actually called in these tests."""
        raise NotImplementedError


@pytest.fixture(autouse=True)
def register_fake_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the fake adapter(s) in the adapter registry for each test."""
    monkeypatch.setattr(
        "euroeval.zero_shot_adapters.ADAPTERS", [FakeAdapter, NeedsExtraAdapter]
    )
    monkeypatch.setattr(
        "euroeval.benchmark_modules.zero_shot_classifier.ADAPTERS",
        [FakeAdapter, NeedsExtraAdapter],
    )


@pytest.fixture
def zero_shot_model_config(model_config: ModelConfig) -> ModelConfig:
    """A model configuration pointing at the fake zero-shot classifier.

    Returns:
        A model configuration pointing at the fake zero-shot classifier.
    """
    return dataclasses.replace(
        model_config,
        model_id="fake-zero-shot",
        inference_backend=InferenceBackend.ZERO_SHOT_CLASSIFIER,
        model_type=ModelType.ZERO_SHOT_CLASSIFIER,
        param=None,
        fresh=False,
    )


class TestModelExists:
    """Tests for `ZeroShotClassifierModel.model_exists`."""

    def test_matching_model_id_is_detected(
        self, benchmark_config: BenchmarkConfig
    ) -> None:
        """A model ID that a registered adapter matches is detected."""
        assert (
            ZeroShotClassifierModel.model_exists(
                model_id="fake-zero-shot-model", benchmark_config=benchmark_config
            )
            is True
        )

    def test_non_matching_model_id_is_not_detected(
        self, benchmark_config: BenchmarkConfig
    ) -> None:
        """A model ID that no adapter matches is not detected."""
        assert (
            ZeroShotClassifierModel.model_exists(
                model_id="some-other-model", benchmark_config=benchmark_config
            )
            is False
        )

    def test_needs_extra_installed_is_reported(
        self, benchmark_config: BenchmarkConfig
    ) -> None:
        """A model whose adapter needs an extra installed reports that error."""
        result = ZeroShotClassifierModel.model_exists(
            model_id="needs-extra-model", benchmark_config=benchmark_config
        )
        assert isinstance(result, NeedsExtraInstalled)
        assert result.extra == "fake-extra"


class TestGetModelConfig:
    """Tests for `ZeroShotClassifierModel.get_model_config`."""

    def test_config_fields(self, benchmark_config: BenchmarkConfig) -> None:
        """The built model config points at the zero-shot classifier backend."""
        config = ZeroShotClassifierModel.get_model_config(
            model_id="fake-zero-shot", benchmark_config=benchmark_config
        )
        assert config.model_id == "fake-zero-shot"
        assert config.inference_backend == InferenceBackend.ZERO_SHOT_CLASSIFIER
        assert config.model_type == ModelType.ZERO_SHOT_CLASSIFIER
        assert config.param is None

    def test_allowed_param_is_accepted(self, benchmark_config: BenchmarkConfig) -> None:
        """A parameter in the adapter's `allowed_params` is accepted."""
        config = ZeroShotClassifierModel.get_model_config(
            model_id="fake-zero-shot#variant-a", benchmark_config=benchmark_config
        )
        assert config.param == "variant-a"

    def test_disallowed_param_raises(self, benchmark_config: BenchmarkConfig) -> None:
        """A parameter not in the adapter's `allowed_params` raises `InvalidModel`."""
        with pytest.raises(InvalidModel, match="Invalid parameter"):
            ZeroShotClassifierModel.get_model_config(
                model_id="fake-zero-shot#unknown-variant",
                benchmark_config=benchmark_config,
            )

    def test_non_matching_model_raises(self, benchmark_config: BenchmarkConfig) -> None:
        """A model ID with no matching adapter raises `InvalidModel`."""
        with pytest.raises(InvalidModel):
            ZeroShotClassifierModel.get_model_config(
                model_id="some-other-model", benchmark_config=benchmark_config
            )


class TestDispatch:
    """Tests that the zero-shot classifier backend is reachable via dispatch."""

    def test_get_model_config_resolves_via_registry(
        self, benchmark_config: BenchmarkConfig
    ) -> None:
        """The reflection-based registry resolves a matching model ID."""
        config = get_model_config(
            model_id="fake-zero-shot", benchmark_config=benchmark_config
        )
        assert config.inference_backend == InferenceBackend.ZERO_SHOT_CLASSIFIER

    def test_load_model_returns_zero_shot_classifier_model(
        self,
        zero_shot_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """`load_model` dispatches to `ZeroShotClassifierModel`."""
        model = load_model(
            model_config=zero_shot_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        assert isinstance(model, ZeroShotClassifierModel)


class TestGenerate:
    """Tests for `ZeroShotClassifierModel.generate`."""

    def test_sequence_classification_scores(
        self,
        zero_shot_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Probabilities from the adapter are mapped to logprobs in `scores`."""
        model = ZeroShotClassifierModel(
            model_config=zero_shot_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = model.generate(inputs=dict(text=["some text", "some other text"]))

        num_labels = len(dataset_config.id2label)
        assert output.scores is not None
        assert len(output.scores) == 2
        for sample_scores in output.scores:
            assert len(sample_scores) == 1
            assert len(sample_scores[0]) == num_labels
            top_label, top_logprob = max(sample_scores[0], key=lambda pair: pair[1])
            assert math.isclose(math.exp(top_logprob), 1.0, abs_tol=0.05)

        candidate_labels = [
            dataset_config.prompt_label_mapping[label]
            for label in dataset_config.id2label.values()
        ]
        assert output.sequences == [candidate_labels[0], candidate_labels[0]]

    def test_multiple_choice_classification_scores(
        self, zero_shot_model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Multiple-choice classification tasks are also supported."""
        mc_dataset_config = DatasetConfig(
            name="dataset",
            pretty_name="Dataset",
            source="dataset_id",
            task=KNOW,
            languages=[DANISH],
        )
        model = ZeroShotClassifierModel(
            model_config=zero_shot_model_config,
            dataset_config=mc_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = model.generate(inputs=dict(text=["some text"]))
        assert output.scores is not None
        assert len(output.scores[0][0]) == len(mc_dataset_config.id2label)

    def test_unsupported_task_group_raises(
        self, zero_shot_model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """Task groups other than (multiple-choice) classification are rejected."""
        hallu_dataset_config = DatasetConfig(
            name="dataset",
            pretty_name="Dataset",
            source="dataset_id",
            task=HALLU,
            languages=[DANISH],
        )
        model = ZeroShotClassifierModel(
            model_config=zero_shot_model_config,
            dataset_config=hallu_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(InvalidBenchmark, match="only support"):
            model.generate(inputs=dict(text=["some text"]))


class TestUnimplementedProperties:
    """Tests for the properties the zero-shot classifier backend doesn't support."""

    @pytest.mark.parametrize("property_name", ["data_collator", "trainer_class"])
    def test_unsupported_property_raises(
        self,
        property_name: str,
        zero_shot_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The backend does not support finetuning properties."""
        model = ZeroShotClassifierModel(
            model_config=zero_shot_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(NotImplementedError):
            _ = getattr(model, property_name)
