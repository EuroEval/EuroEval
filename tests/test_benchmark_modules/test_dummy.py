"""Unit tests for the `dummy` module."""

import dataclasses
import json
import math

import pytest

from euroeval.benchmark_modules.dummy import DummyModel
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig, Task
from euroeval.enums import InferenceBackend, ModelType
from euroeval.exceptions import InvalidModel
from euroeval.languages import DANISH
from euroeval.model_config import get_model_config
from euroeval.model_loading import load_model
from euroeval.tasks import HALLU, NER, SUMM


class TestBPCGating:
    """Tests that BPC scoring is rejected for the dummy backend."""

    def test_bpc_rejected_for_dummy(
        self,
        dummy_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """BPC scoring raises InvalidModel for the dummy backend."""
        bpc_config = dataclasses.replace(benchmark_config, use_bits_per_character=True)
        with pytest.raises(InvalidModel, match="vLLM backend"):
            load_model(
                model_config=dummy_model_config,
                dataset_config=dataset_config,
                benchmark_config=bpc_config,
            )


class TestDispatch:
    """Tests that the dummy backend is reachable through both dispatch points."""

    def test_get_model_config_resolves_to_dummy(
        self, benchmark_config: BenchmarkConfig
    ) -> None:
        """The reflection-based registry resolves 'dummy' with no network calls."""
        config = get_model_config(model_id="dummy", benchmark_config=benchmark_config)
        assert config.inference_backend == InferenceBackend.DUMMY

    def test_load_model_returns_dummy_model(
        self,
        dummy_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """load_model dispatches to DummyModel for the dummy backend."""
        model = load_model(
            model_config=dummy_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        assert isinstance(model, DummyModel)


class TestGenerate:
    """Tests for DummyModel.generate."""

    def test_classification_scores_are_uniform(
        self,
        dummy_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Classification tasks get an even logprob distribution over labels."""
        model = DummyModel(
            model_config=dummy_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = model.generate(inputs=dict(text=["some text", "some other text"]))

        num_labels = len(dataset_config.id2label)
        expected_logprob = math.log(1 / num_labels)

        assert len(output.sequences) == 2
        assert output.scores is not None
        assert len(output.scores) == 2
        for sample_scores in output.scores:
            assert len(sample_scores) == 1
            assert len(sample_scores[0]) == num_labels
            for _, logprob in sample_scores[0]:
                assert logprob == pytest.approx(expected_logprob)

    @pytest.mark.parametrize("task", [SUMM, HALLU])
    def test_free_text_tasks_return_generic_placeholder(
        self,
        task: Task,
        dummy_model_config: ModelConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """Free-text-answer tasks get a generic non-empty placeholder answer.

        A literally empty answer breaks some metrics (e.g. the hallucination
        classifier errors with "no tokens found in predictions"), so a
        non-empty placeholder is used for every free-text-answer task group.
        """
        free_text_dataset_config = DatasetConfig(
            name="dataset",
            pretty_name="Dataset",
            source="dataset_id",
            task=task,
            languages=[DANISH],
        )
        model = DummyModel(
            model_config=dummy_model_config,
            dataset_config=free_text_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = model.generate(inputs=dict(text=["some text", "some other text"]))
        assert output.scores is None
        assert output.sequences == ["answer", "answer"]

    def test_token_classification_extracts_to_all_o_labels(
        self, dummy_model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """The correctly-shaped empty prediction round-trips to all-"o" labels."""
        ner_dataset_config = DatasetConfig(
            name="dataset",
            pretty_name="Dataset",
            source="dataset_id",
            task=NER,
            languages=[DANISH],
        )
        model = DummyModel(
            model_config=dummy_model_config,
            dataset_config=ner_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = model.generate(inputs=dict(text=["some text"]))
        labels = model.extract_labels_from_generation(
            input_batch=dict(tokens=[["some", "text"]]), model_output=output
        )
        assert labels == [["o", "o"]]

    def test_token_classification_returns_canonical_empty_shape(
        self, dummy_model_config: ModelConfig, benchmark_config: BenchmarkConfig
    ) -> None:
        """NER gets the same "no entities" shape used in few-shot demonstrations."""
        ner_dataset_config = DatasetConfig(
            name="dataset",
            pretty_name="Dataset",
            source="dataset_id",
            task=NER,
            languages=[DANISH],
        )
        model = DummyModel(
            model_config=dummy_model_config,
            dataset_config=ner_dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        output = model.generate(inputs=dict(text=["some text", "some other text"]))
        assert output.scores is None

        expected_categories = set(ner_dataset_config.prompt_label_mapping.values())
        for sequence in output.sequences:
            prediction = json.loads(sequence)
            assert set(prediction.keys()) == expected_categories
            assert all(entities == [] for entities in prediction.values())


class TestGetModelConfig:
    """Tests for DummyModel.get_model_config."""

    def test_config_fields(self, benchmark_config: BenchmarkConfig) -> None:
        """The built model config points at the dummy backend."""
        config = DummyModel.get_model_config(
            model_id="dummy", benchmark_config=benchmark_config
        )
        assert config.model_id == "dummy"
        assert config.inference_backend == InferenceBackend.DUMMY
        assert config.model_type == ModelType.GENERATIVE
        assert config.fresh is False
        assert config.param is None


class TestModelExists:
    """Tests for DummyModel.model_exists."""

    def test_exact_id_matches(self, benchmark_config: BenchmarkConfig) -> None:
        """The exact 'dummy' id is recognised."""
        assert DummyModel.model_exists(
            model_id="dummy", benchmark_config=benchmark_config
        )

    @pytest.mark.parametrize("model_id", ["dummy-2", "dummy#thinking", "Dummy", ""])
    def test_other_ids_do_not_match(
        self, model_id: str, benchmark_config: BenchmarkConfig
    ) -> None:
        """IDs other than the exact string 'dummy' are not recognised."""
        assert not DummyModel.model_exists(
            model_id=model_id, benchmark_config=benchmark_config
        )


class TestUnimplementedProperties:
    """Tests for the properties the dummy backend deliberately doesn't support."""

    def test_data_collator_raises(
        self,
        dummy_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The dummy backend does not support finetuning."""
        model = DummyModel(
            model_config=dummy_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(NotImplementedError):
            _ = model.data_collator

    def test_trainer_class_raises(
        self,
        dummy_model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
    ) -> None:
        """The dummy backend does not support finetuning."""
        model = DummyModel(
            model_config=dummy_model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=False,
        )
        with pytest.raises(NotImplementedError):
            _ = model.trainer_class


@pytest.fixture
def dummy_model_config(model_config: ModelConfig) -> ModelConfig:
    """Yields a model configuration pointing at the dummy backend.

    Returns:
        A model configuration pointing at the dummy backend.
    """
    return dataclasses.replace(
        model_config,
        model_id="dummy",
        inference_backend=InferenceBackend.DUMMY,
        model_type=ModelType.GENERATIVE,
        fresh=False,
    )
