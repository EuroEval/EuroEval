"""Unit tests for the `zero_shot_classifier` module (Laya)."""

import dataclasses
import math
import sys
import types
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy
import pytest
from datasets import Dataset, DatasetDict
from safetensors.numpy import save_file

from euroeval.benchmark_modules.zero_shot_classifier import ZeroShotClassifierModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidBenchmark, InvalidModel, NeedsExtraInstalled
from euroeval.languages import DANISH
from euroeval.model_config import get_model_config
from euroeval.model_loading import load_model
from euroeval.tasks import HALLU, KNOW, SENT


@dataclass
class FakeAgent:
    """A fake `laya.Agent`, recording how it was constructed and called."""

    model_id_or_path: str
    subfolder: str | None
    token: str | None
    calls: list[tuple[object, dict]]
    top_criterion_index: int = 0

    def __init__(
        self,
        model_id_or_path: str,
        subfolder: str | None = None,
        token: str | None = None,
        device: str | None = None,
    ) -> None:
        """Record the construction arguments."""
        self.model_id_or_path = model_id_or_path
        self.subfolder = subfolder
        self.token = token
        self.calls = []

    def system_one(self, state: object, questions: dict) -> dict:
        """Record the call and return a fixed `choice` answer.

        Returns:
            A fake Laya `system_one` response, preferring one criterion.
        """
        self.calls.append((state, questions))
        criteria = list(questions["q"]["criteria"].keys())
        top = criteria[self.top_criterion_index]
        rest = 0.02 / max(len(criteria) - 1, 1)
        probabilities = {label: rest for label in criteria}
        probabilities[top] = 1 - sum(v for k, v in probabilities.items() if k != top)
        return {"answers": {"q": {"type": "choice", "probabilities": probabilities}}}


class TestClassify:
    """Tests for `ZeroShotClassifierModel._classify`."""

    def test_probabilities_are_mapped_per_label(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """`_classify` returns one probability dict per text, keyed by label."""
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        candidate_labels = ["positive", "negative", "neutral"]
        results = model._classify(
            texts=["some text", "some other text"], candidate_labels=candidate_labels
        )

        assert len(results) == 2
        for probs in results:
            assert set(probs.keys()) == set(candidate_labels)
        assert len(model.agent.calls) == 2
        for _state, questions in model.agent.calls:
            assert questions["q"]["type"] == "choice"
            assert set(questions["q"]["criteria"].keys()) == set(candidate_labels)

    def test_raises_invalid_benchmark_when_label_missing_from_probabilities(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A candidate label missing from Laya's output raises `InvalidBenchmark`."""
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

        def system_one_missing_label(state: object, questions: dict) -> dict:
            criteria = list(questions["q"]["criteria"].keys())
            return {
                "answers": {
                    "q": {"type": "choice", "probabilities": {criteria[0]: 1.0}}
                }
            }

        model.agent.system_one = system_one_missing_label  # ty: ignore[invalid-assignment]
        with pytest.raises(InvalidBenchmark, match="positive|negative|neutral"):
            model._classify(
                texts=["some text"],
                candidate_labels=["positive", "negative", "neutral"],
            )


class TestDispatch:
    """Tests that Laya is reachable via dispatch."""

    def test_get_model_config_resolves_via_registry(
        self, fake_laya_module: types.ModuleType, benchmark_config: BenchmarkConfig
    ) -> None:
        """The reflection-based registry resolves a Laya model ID."""
        config = get_model_config(
            model_id="convaiinnovations/laya", benchmark_config=benchmark_config
        )
        assert config.inference_backend == InferenceBackend.LAYA
        assert config.model_type == ModelType.ZERO_SHOT_CLASSIFIER

    def test_load_model_returns_zero_shot_classifier_model(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """`load_model` dispatches to `ZeroShotClassifierModel`."""
        model = load_model(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        assert isinstance(model, ZeroShotClassifierModel)


class TestGenerate:
    """Tests for `ZeroShotClassifierModel.generate`."""

    def test_label_extraction_matches_the_true_top_probability_label(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The extracted label follows the top probability, not candidate order."""
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        candidate_labels = [
            dataset_config.prompt_label_mapping[label]
            for label in dataset_config.id2label.values()
        ]
        assert len(candidate_labels) > 1
        model.agent.top_criterion_index = len(candidate_labels) - 1

        texts = ["some text"]
        output = model.generate(inputs=dict(text=texts))
        extracted = model.extract_labels_from_generation(
            input_batch=dict(prompt=["prompt"], text=texts), model_output=output
        )
        assert extracted == [candidate_labels[-1]]

    def test_sequence_classification_scores(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Probabilities from Laya are mapped to logprobs in `scores`."""
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
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

    def test_unsupported_task_group_raises(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
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
            model_config=laya_model_config,
            dataset_config=hallu_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(InvalidBenchmark, match="only support"):
            model.generate(inputs=dict(text=["some text"]))


class TestGetModelConfig:
    """Tests for `ZeroShotClassifierModel.get_model_config`."""

    def test_config_fields(self, benchmark_config: BenchmarkConfig) -> None:
        """The built model config points at the Laya backend."""
        config = ZeroShotClassifierModel.get_model_config(
            model_id="convaiinnovations/laya", benchmark_config=benchmark_config
        )
        assert config.model_id == "convaiinnovations/laya"
        assert config.inference_backend == InferenceBackend.LAYA
        assert config.model_type == ModelType.ZERO_SHOT_CLASSIFIER
        assert config.param is None

    @pytest.mark.parametrize(
        ("param", "raises"), [("multilingual", False), ("not-a-real-variant", True)]
    )
    def test_param_validation(
        self, benchmark_config: BenchmarkConfig, param: str, raises: bool
    ) -> None:
        """Only recognised parameters are accepted."""
        if raises:
            with pytest.raises(InvalidModel, match="Invalid parameter"):
                ZeroShotClassifierModel.get_model_config(
                    model_id=f"convaiinnovations/laya#{param}",
                    benchmark_config=benchmark_config,
                )
        else:
            config = ZeroShotClassifierModel.get_model_config(
                model_id=f"convaiinnovations/laya#{param}",
                benchmark_config=benchmark_config,
            )
            assert config.param == param


class TestInit:
    """Tests for `ZeroShotClassifierModel.__init__`."""

    def test_rejects_non_main_revision(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A revision other than "main" is rejected."""
        config = dataclasses.replace(laya_model_config, revision="some-other-branch")
        with pytest.raises(InvalidModel, match="revision"):
            ZeroShotClassifierModel(
                model_config=config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
                log_metadata=False,
            )

    def test_standalone_repo_rejects_param(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """A standalone repo (not the bundled one) rejects a `#param`."""
        config = dataclasses.replace(
            laya_model_config,
            model_id="convaiinnovations/laya-multilingual",
            param="multilingual",
        )
        with pytest.raises(InvalidModel, match="does not accept a parameter"):
            ZeroShotClassifierModel(
                model_config=config,
                dataset_config=dataset_config,
                benchmark_config=benchmark_config,
                log_metadata=False,
            )

    @pytest.mark.parametrize(
        ("param", "expected_subfolder", "expected_max_length"),
        [
            (None, None, 512),
            ("multilingual", "multilingual", 1024),
            ("typed-decisions", "typed-decisions", 512),
        ],
    )
    def test_variant_maps_to_subfolder_and_max_length(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        param: str | None,
        expected_subfolder: str | None,
        expected_max_length: int,
    ) -> None:
        """Each allowed parameter loads the expected subfolder and max length."""
        config = dataclasses.replace(laya_model_config, param=param)
        model = ZeroShotClassifierModel(
            model_config=config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        assert model.agent.subfolder == expected_subfolder
        assert model.model_max_length == expected_max_length


class TestModelExists:
    """Tests for `ZeroShotClassifierModel.model_exists`."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "some-other-model",
            "convaiinnovations/layatron",
            "convaiinnovations/layabout",
        ],
    )
    def test_does_not_match_unrelated_model(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        model_id: str,
    ) -> None:
        """Unrelated model IDs are not detected."""
        assert (
            ZeroShotClassifierModel.model_exists(
                model_id=model_id, benchmark_config=benchmark_config
            )
            is False
        )

    def test_matches_bundled_repo(
        self, fake_laya_module: types.ModuleType, benchmark_config: BenchmarkConfig
    ) -> None:
        """The bundled Laya repo is detected."""
        assert (
            ZeroShotClassifierModel.model_exists(
                model_id="convaiinnovations/laya", benchmark_config=benchmark_config
            )
            is True
        )

    def test_matches_local_checkpoint_directory(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        tmp_path: Path,
    ) -> None:
        """A local directory with `rl_agent_config.json` is detected."""
        (tmp_path / "rl_agent_config.json").write_text("{}")
        assert (
            ZeroShotClassifierModel.model_exists(
                model_id=str(tmp_path), benchmark_config=benchmark_config
            )
            is True
        )

    @pytest.mark.parametrize(
        "model_id",
        [
            "convaiinnovations/laya-multilingual",
            "convaiinnovations/laya-typed-decisions",
        ],
    )
    def test_matches_standalone_repos(
        self,
        fake_laya_module: types.ModuleType,
        benchmark_config: BenchmarkConfig,
        model_id: str,
    ) -> None:
        """Standalone `convaiinnovations/laya-*` repos are detected."""
        assert (
            ZeroShotClassifierModel.model_exists(
                model_id=model_id, benchmark_config=benchmark_config
            )
            is True
        )

    def test_needs_extra_installed_when_laya_missing(
        self, monkeypatch: pytest.MonkeyPatch, benchmark_config: BenchmarkConfig
    ) -> None:
        """A `NeedsExtraInstalled` error is returned when `laya` isn't importable."""
        monkeypatch.setitem(sys.modules, "laya", None)
        result = ZeroShotClassifierModel.model_exists(
            model_id="convaiinnovations/laya", benchmark_config=benchmark_config
        )
        assert isinstance(result, NeedsExtraInstalled)
        assert result.extra == "laya"


class TestMultipleChoice:
    """Tests for `ZeroShotClassifierModel._classify_multiple_choice`."""

    @pytest.fixture
    def mc_model(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> ZeroShotClassifierModel:
        """A model configured for the multiple-choice `KNOW` task.

        Returns:
            The configured model.
        """
        mc_dataset_config = DatasetConfig(
            name="dataset",
            pretty_name="Dataset",
            source="dataset_id",
            task=KNOW,
            languages=[DANISH],
        )
        return ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=mc_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )

    @pytest.mark.parametrize(
        "text",
        [
            "some unparseable text",
            "What is the capital of Denmark?\nChoices:\n1. Oslo\n2. Copenhagen\n"
            "3. Stockholm\n4. Helsinki",
            "What is the capital of Denmark?\nChoices:\n"
            "a. Copenhagen\nb. Copenhagen\nc. Stockholm\nd. Helsinki",
        ],
        ids=["unparseable", "wrong markers", "duplicate options"],
    )
    def test_falls_back_to_letter_labels_when_unparseable(
        self, mc_model: ZeroShotClassifierModel, text: str
    ) -> None:
        """Unparseable, mismarked, or duplicated options fall back to letters."""
        output = mc_model.generate(inputs=dict(text=[text]))
        assert mc_model.agent.calls[0][1]["q"]["criteria"].keys() == {
            "a",
            "b",
            "c",
            "d",
        }
        assert output.scores is not None
        assert len(output.scores[0][0]) == 4

    def test_uses_option_texts_as_candidate_labels(
        self, mc_model: ZeroShotClassifierModel
    ) -> None:
        """The model classifies against option texts, not bare letters."""
        text = (
            "What is the capital of Denmark?\nChoices:\n"
            "a. Oslo\nb. Copenhagen\nc. Stockholm\nd. Helsinki"
        )
        mc_model.agent.top_criterion_index = 3
        output = mc_model.generate(inputs=dict(text=[text]))

        assert set(mc_model.agent.calls[0][1]["q"]["criteria"].keys()) == {
            "Oslo",
            "Copenhagen",
            "Stockholm",
            "Helsinki",
        }
        assert output.scores is not None
        sample_scores = dict(output.scores[0][0])
        top_label = max(sample_scores, key=lambda label: sample_scores[label])
        assert top_label == "d"


class TestNumParams:
    """Tests for `ZeroShotClassifierModel.num_params`."""

    def test_counts_parameters_from_local_checkpoint(
        self,
        fake_laya_module: types.ModuleType,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        tmp_path: Path,
    ) -> None:
        """A local checkpoint directory is inspected via `safetensors.safe_open`."""
        save_file(
            {"a": numpy.zeros((2, 3), dtype=numpy.float32), "b": numpy.zeros(4)},
            str(tmp_path / "model.safetensors"),
        )
        (tmp_path / "rl_agent_config.json").write_text("{}")
        config = dataclasses.replace(
            model_config,
            model_id=str(tmp_path),
            inference_backend=InferenceBackend.LAYA,
            model_type=ModelType.ZERO_SHOT_CLASSIFIER,
            param=None,
            revision="main",
        )
        model = ZeroShotClassifierModel(
            model_config=config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        assert model.num_params == 2 * 3 + 4

    def test_returns_minus_one_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """`num_params` returns -1 when the safetensors metadata can't be fetched."""

        def raise_error(*args: object, **kwargs: object) -> t.NoReturn:
            raise OSError("no network in this test")

        monkeypatch.setattr(
            "euroeval.safetensors_utils.internet_connection_available", lambda: True
        )
        monkeypatch.setattr(
            "euroeval.safetensors_utils.get_safetensors_metadata", raise_error
        )
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        assert model.num_params == -1


class TestPrepareDataset:
    """Tests for `ZeroShotClassifierModel.prepare_dataset`."""

    def test_raw_text_is_preserved_not_the_rendered_prompt(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The 'text' field stays the raw sample text, not the decoder prompt."""
        raw_texts = ["some raw text", "some other raw text"]
        test_split = Dataset.from_dict(
            dict(text=raw_texts, label=["positive", "negative"])
        )
        dataset = DatasetDict({"test": test_split})

        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        prepared = model.prepare_dataset(dataset=dataset, task=SENT, itr_idx=0)

        assert list(prepared["test"]["text"]) == raw_texts
        assert "prompt" in prepared["test"].column_names
        assert prepared["test"]["prompt"][0] != raw_texts[0]


class TestUnimplementedProperties:
    """Tests for the properties Laya doesn't support."""

    @pytest.mark.parametrize("property_name", ["data_collator", "trainer_class"])
    def test_unsupported_property_raises(
        self,
        property_name: str,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Laya does not support finetuning properties."""
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(NotImplementedError):
            _ = getattr(model, property_name)


class TestUpdateDatasetConfig:
    """Tests for `ZeroShotClassifierModel.update_dataset_config`."""

    def test_recomputes_buffer_state_for_new_dataset(
        self,
        fake_laya_module: types.ModuleType,
        laya_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Switching datasets recomputes per-dataset buffer state."""
        model = ZeroShotClassifierModel(
            model_config=laya_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        other_dataset_config = DatasetConfig(
            name="other-dataset",
            pretty_name="Other dataset",
            source="dataset_id",
            task=KNOW,
            languages=[DANISH],
        )
        result = model.update_dataset_config(dataset_config=other_dataset_config)

        assert result is model
        assert model.dataset_config is other_dataset_config
        output = model.generate(inputs=dict(text=["some text"]))
        assert output.scores is not None
        assert len(output.scores[0][0]) == len(other_dataset_config.id2label)


@pytest.fixture
def fake_laya_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `laya` module in `sys.modules`.

    Returns:
        The fake `laya` module.
    """
    fake_module = types.ModuleType("laya")
    fake_module.Agent = FakeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", fake_module)
    return fake_module


@pytest.fixture
def laya_model_config(model_config: ModelConfig) -> ModelConfig:
    """A model configuration pointing at the (fake) Laya model.

    Returns:
        A model configuration pointing at Laya.
    """
    return dataclasses.replace(
        model_config,
        model_id="convaiinnovations/laya",
        inference_backend=InferenceBackend.LAYA,
        model_type=ModelType.ZERO_SHOT_CLASSIFIER,
        param=None,
        revision="main",
        fresh=False,
    )
