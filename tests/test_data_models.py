"""Unit tests for the `data_models` module."""

import inspect
import json
from collections.abc import Generator
from pathlib import Path

import pytest

from euroeval import __version__, data_models, enums
from euroeval.data_models import BenchmarkResult, Metric
from euroeval.metrics import HuggingFaceMetric


def test_all_classes_are_dataclasses_or_pydantic_models() -> None:
    """Test that all classes in `data_models` are dataclasses or Pydantic models."""
    all_classes = [
        getattr(data_models, obj_name)
        for obj_name in dir(data_models)
        if not obj_name.startswith("_")
        and inspect.isclass(object=getattr(data_models, obj_name))
        and not hasattr(enums, obj_name)
        and obj_name not in {"ScoreDict", "Metric"}
    ]
    for obj in all_classes:
        obj_is_dataclass = hasattr(obj, "__dataclass_fields__")
        obj_is_pydantic_model = hasattr(obj, "model_fields")
        assert obj_is_dataclass or obj_is_pydantic_model, (
            f"{obj.__name__} is not a dataclass or Pydantic model. "
        )


class TestMetric:
    """Unit tests for the `Metric` class."""

    def test_metric_is_object(self, metric: HuggingFaceMetric) -> None:
        """Test that the metric config is a `Metric` object."""
        assert isinstance(metric, Metric)

    def test_attributes_correspond_to_arguments(
        self, metric: HuggingFaceMetric
    ) -> None:
        """Test that the metric config attributes correspond to the arguments."""
        assert metric.name == "metric_name"
        assert metric.pretty_name == "Metric name"
        assert metric.huggingface_id == "metric_id"
        assert metric.results_key == "metric_key"

    def test_default_value_of_compute_kwargs(self, metric: HuggingFaceMetric) -> None:
        """Test that the default value of `compute_kwargs` is an empty dictionary."""
        assert metric.compute_kwargs == dict()

    @pytest.mark.parametrize(
        "inputs,expected",
        [
            (0.5, (50.0, "50.00%")),
            (0.123456, (12.3456, "12.35%")),
            (0.0, (0.0, "0.00%")),
            (1.0, (100.0, "100.00%")),
            (0.999999, (99.9999, "100.00%")),
            (2.0, (200.0, "200.00%")),
            (-1.0, (-100.0, "-100.00%")),
        ],
    )
    def test_default_value_of_postprocessing_fn(
        self, metric: Metric, inputs: float, expected: tuple[float, str]
    ) -> None:
        """Test that the default value of `postprocessing_fn` is correct."""
        assert metric.postprocessing_fn(inputs) == expected


# TODO
@pytest.mark.skip("Not implemented")
class TestBenchmarkConfigParams:
    """Unit tests for the `BenchmarkConfigParams` class."""

    pass


class TestBenchmarkResult:
    """Tests related to the `BenchmarkResult` class."""

    @pytest.fixture(scope="class")
    def benchmark_result(self) -> Generator[BenchmarkResult, None, None]:
        """Fixture for a `BenchmarkResult` object."""
        yield BenchmarkResult(
            dataset="dataset",
            model="model",
            generative=False,
            generative_type=None,
            few_shot=True,
            validation_split=False,
            num_model_parameters=100,
            max_sequence_length=100,
            vocabulary_size=100,
            merge=False,
            dataset_languages=["da"],
            task="task",
            results=dict(),
        )

    @pytest.fixture(scope="class")
    def results_path(self) -> Generator[Path, None, None]:
        """Fixture for a `Path` object to a results file."""
        results_path = Path(".euroeval_cache/test_results.jsonl")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        yield results_path

    def test_benchmark_result_parameters(
        self, benchmark_result: BenchmarkResult
    ) -> None:
        """Test that the `BenchmarkResult` parameters are correct."""
        assert benchmark_result.dataset == "dataset"
        assert benchmark_result.model == "model"
        assert benchmark_result.generative is False
        assert benchmark_result.generative_type is None
        assert benchmark_result.few_shot is True
        assert benchmark_result.validation_split is False
        assert benchmark_result.num_model_parameters == 100
        assert benchmark_result.max_sequence_length == 100
        assert benchmark_result.vocabulary_size == 100
        assert benchmark_result.merge is False
        assert benchmark_result.dataset_languages == ["da"]
        assert benchmark_result.task == "task"
        assert benchmark_result.results == dict()
        assert benchmark_result.euroeval_version == __version__

    @pytest.mark.parametrize(
        argnames=["config", "expected"],
        argvalues=[
            (
                dict(
                    dataset="dataset",
                    model="model",
                    few_shot=True,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    merge=False,
                    generative=False,
                    generative_type=None,
                    few_shot=True,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
            ),
            (
                dict(
                    dataset="dataset",
                    model="model (few-shot)",
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    merge=False,
                    generative=True,
                    generative_type=None,
                    few_shot=True,
                    validation_split=False,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
            ),
            (
                dict(
                    dataset="dataset",
                    model="model (val)",
                    few_shot=True,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    merge=False,
                    generative=False,
                    generative_type=None,
                    few_shot=True,
                    validation_split=True,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
            ),
            (
                dict(
                    dataset="dataset",
                    model="model (few-shot, val)",
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
                BenchmarkResult(
                    dataset="dataset",
                    model="model",
                    merge=False,
                    generative=True,
                    generative_type=None,
                    few_shot=True,
                    validation_split=True,
                    num_model_parameters=100,
                    max_sequence_length=100,
                    vocabulary_size=100,
                    dataset_languages=["da"],
                    task="task",
                    results=dict(),
                ),
            ),
        ],
        ids=[
            "normal case",
            "few-shot model name",
            "validation split model name",
            "few-shot and validation split model name",
        ],
    )
    def test_from_dict(self, config: dict, expected: BenchmarkResult) -> None:
        """Test that `BenchmarkResult.from_dict` works as expected."""
        assert BenchmarkResult.from_dict(config) == expected

    def test_append_to_results(
        self, benchmark_result: BenchmarkResult, results_path: Path
    ) -> None:
        """Test that `BenchmarkResult.append_to_results` works as expected."""
        results_path.unlink(missing_ok=True)
        results_path.touch(exist_ok=True)

        benchmark_result.append_to_results(results_path=results_path)
        json_str = json.dumps(
            dict(
                dataset=benchmark_result.dataset,
                task=benchmark_result.task,
                dataset_languages=benchmark_result.dataset_languages,
                model=benchmark_result.model,
                results=benchmark_result.results,
                num_model_parameters=benchmark_result.num_model_parameters,
                max_sequence_length=benchmark_result.max_sequence_length,
                vocabulary_size=benchmark_result.vocabulary_size,
                merge=benchmark_result.merge,
                generative=benchmark_result.generative,
                generative_type=benchmark_result.generative_type,
                few_shot=benchmark_result.few_shot,
                validation_split=benchmark_result.validation_split,
                euroeval_version=benchmark_result.euroeval_version,
                transformers_version=benchmark_result.transformers_version,
                torch_version=benchmark_result.torch_version,
                vllm_version=benchmark_result.vllm_version,
                xgrammar_version=benchmark_result.xgrammar_version,
            )
        )
        assert results_path.read_text() == f"\n{json_str}"

        benchmark_result.append_to_results(results_path=results_path)
        assert results_path.read_text() == f"\n{json_str}\n{json_str}"

        results_path.unlink(missing_ok=True)
