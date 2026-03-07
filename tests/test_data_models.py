"""Tests for the `data_models` module."""

import inspect
import json
from collections.abc import Generator
from pathlib import Path

import pytest
from click import ParamType

from euroeval import __version__
from euroeval.benchmarker import Benchmarker
from euroeval.data_models import BenchmarkConfig, BenchmarkConfigParams, BenchmarkResult
from euroeval.metrics import HuggingFaceMetric, Metric


class TestMetric:
    """Tests for the `Metric` class."""

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


class TestBenchmarkResult:
    """Tests related to the `BenchmarkResult` class."""

    @pytest.fixture(scope="class")
    def benchmark_result(self) -> Generator[BenchmarkResult, None, None]:
        """Fixture for a `BenchmarkResult` object.

        Yields:
            A `BenchmarkResult` object.
        """
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
            languages=["da"],
            task="task",
            results=dict(),
        )

    @pytest.fixture(scope="class")
    def results_path(self) -> Generator[Path, None, None]:
        """Fixture for a `Path` object to a results file.

        Yields:
            A `Path` object to a results file.
        """
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
        assert benchmark_result.languages == ["da"]
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
                    languages=["da"],
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
                    languages=["da"],
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
                    languages=["da"],
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
                    languages=["da"],
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
                    languages=["da"],
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
                    languages=["da"],
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
                    languages=["da"],
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
                    languages=["da"],
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
        """Test that `BenchmarkResult.append_to_results` writes valid EEE format."""
        results_path.unlink(missing_ok=True)
        results_path.touch(exist_ok=True)

        benchmark_result.append_to_results(results_path=results_path)
        content = results_path.read_text().strip()
        eee_dict = json.loads(content)

        # Check required top-level EEE fields
        assert eee_dict["schema_version"] == "0.2.1"
        assert "evaluation_id" in eee_dict
        assert "retrieved_timestamp" in eee_dict
        assert "source_metadata" in eee_dict
        assert "model_info" in eee_dict
        assert "eval_library" in eee_dict
        assert "evaluation_results" in eee_dict

        # Check source_metadata required fields
        assert eee_dict["source_metadata"]["source_type"] == "evaluation_run"
        assert eee_dict["source_metadata"]["source_organization_name"] == "EuroEval"
        assert eee_dict["source_metadata"]["evaluator_relationship"] == "third_party"

        # Check model_info
        assert eee_dict["model_info"]["id"] == benchmark_result.model
        assert eee_dict["model_info"]["name"] == benchmark_result.model

        # Check eval_library
        assert eee_dict["eval_library"]["name"] == "euroeval"
        assert eee_dict["eval_library"]["version"] == (
            benchmark_result.euroeval_version or "unknown"
        )
        additional = eee_dict["eval_library"]["additional_details"]
        assert additional["dataset"] == benchmark_result.dataset
        assert additional["task"] == benchmark_result.task
        assert json.loads(additional["languages"]) == list(benchmark_result.languages)

        # Verify round-trip: read back from EEE format gives original BenchmarkResult
        restored = BenchmarkResult.from_dict(eee_dict)
        assert restored.dataset == benchmark_result.dataset
        assert restored.model == benchmark_result.model
        assert restored.task == benchmark_result.task
        assert list(restored.languages) == list(benchmark_result.languages)
        assert restored.generative == benchmark_result.generative
        assert restored.few_shot == benchmark_result.few_shot
        assert restored.validation_split == benchmark_result.validation_split

        # Verify two results can be appended
        benchmark_result.append_to_results(results_path=results_path)
        lines = [
            line for line in results_path.read_text().splitlines() if line.strip()
        ]
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert parsed["schema_version"] == "0.2.1"

        results_path.unlink(missing_ok=True)

    def test_round_trip_with_scores(self, results_path: Path) -> None:
        """Test EEE round-trip fidelity with realistic metric scores and raw results."""
        results_path.unlink(missing_ok=True)
        results_path.touch(exist_ok=True)

        original = BenchmarkResult(
            dataset="angry-tweets",
            model="some-model",
            generative=True,
            generative_type="instruction_tuned",
            few_shot=False,
            validation_split=False,
            num_model_parameters=8_000_000_000,
            max_sequence_length=4096,
            vocabulary_size=32000,
            merge=False,
            languages=["da"],
            task="sentiment-classification",
            results={
                "total": {
                    "test_mcc": 42.5,
                    "test_mcc_se": 1.2,
                    "test_macro_f1": 55.3,
                    "test_macro_f1_se": 0.8,
                    "num_failed_instances": 3.0,
                },
                "raw": [
                    {"test_mcc": 0.40, "test_macro_f1": 0.54},
                    {"test_mcc": 0.45, "test_macro_f1": 0.57},
                ],
            },
        )

        original.append_to_results(results_path=results_path)
        content = results_path.read_text().strip()
        eee_dict = json.loads(content)

        # Verify confidence intervals are stored correctly for each metric
        eval_results = {
            er["evaluation_name"]: er for er in eee_dict["evaluation_results"]
        }
        assert "test_mcc" in eval_results
        assert "test_macro_f1" in eval_results

        mcc_result = eval_results["test_mcc"]
        assert mcc_result["score_details"]["score"] == 42.5
        ci = mcc_result["score_details"]["uncertainty"]["confidence_interval"]
        assert abs(ci["lower"] - (42.5 - 1.2)) < 1e-9
        assert abs(ci["upper"] - (42.5 + 1.2)) < 1e-9
        assert ci["confidence_level"] == 0.95

        # Verify metric_config for regular metrics
        assert mcc_result["metric_config"]["lower_is_better"] is False
        assert mcc_result["metric_config"]["score_type"] == "continuous"
        assert mcc_result["metric_config"]["min_score"] == 0
        assert mcc_result["metric_config"]["max_score"] == 100

        # Verify round-trip restores results
        restored = BenchmarkResult.from_dict(eee_dict)
        assert restored.results["total"]["test_mcc"] == 42.5  # type: ignore[index]
        assert (
            abs(
                restored.results["total"]["test_mcc_se"] - 1.2  # type: ignore[index]
            )
            < 1e-9
        )
        assert restored.results["total"]["num_failed_instances"] == 3.0  # type: ignore[index]
        assert len(restored.results["raw"]) == 2  # type: ignore[index]

        results_path.unlink(missing_ok=True)

    def test_speed_metric_config(self, results_path: Path) -> None:
        """Test that speed metrics don't get a hardcoded 0-100 range in EEE output."""
        results_path.unlink(missing_ok=True)
        results_path.touch(exist_ok=True)

        speed_result = BenchmarkResult(
            dataset="speed",
            model="some-model",
            generative=True,
            generative_type="instruction_tuned",
            few_shot=None,
            validation_split=None,
            num_model_parameters=0,
            max_sequence_length=0,
            vocabulary_size=0,
            merge=False,
            languages=["da"],
            task="speed",
            results={
                "total": {
                    "test_speed": 1500.0,
                    "test_speed_se": 50.0,
                    "num_failed_instances": 0.0,
                },
                "raw": [{"test_speed": 1450.0}, {"test_speed": 1550.0}],
            },
        )

        eee = speed_result.to_eee_dict()
        eval_results = {
            er["evaluation_name"]: er for er in eee["evaluation_results"]
        }
        assert "test_speed" in eval_results
        speed_config = eval_results["test_speed"]["metric_config"]
        # Speed metrics should only have lower_is_better, no score_type/min/max
        assert speed_config["lower_is_better"] is False
        assert "score_type" not in speed_config
        assert "min_score" not in speed_config
        assert "max_score" not in speed_config

        results_path.unlink(missing_ok=True)


class TestBenchmarkParametersAreConsistent:
    """Test that the same benchmark parameters are used everywhere."""

    def test_config_params_is_the_same_as_benchmarker_init(self) -> None:
        """Test that `BenchmarkConfigParams` agrees with `Benchmarker.__init__`."""
        benchmark_config_params = set(
            inspect.signature(BenchmarkConfigParams).parameters.keys()
        )
        benchmarker_init_params = set(
            inspect.signature(Benchmarker.__init__).parameters.keys()
        ) - {"self", "batch_size", "dataset_language", "model_language"}
        assert benchmark_config_params == benchmarker_init_params

    def test_config_params_is_the_same_as_benchmark_method(self) -> None:
        """Test that `BenchmarkConfigParams` agrees with `Benchmarker.benchmark`."""
        benchmark_config_params = set(
            inspect.signature(BenchmarkConfigParams).parameters.keys()
        ) - {"run_with_cli"}
        benchmark_method_params = set(
            inspect.signature(Benchmarker.benchmark).parameters.keys()
        ) - {"self", "model", "batch_size", "dataset_language", "model_language"}
        assert benchmark_config_params == benchmark_method_params

    def test_config_params_is_the_same_as_cli(
        self, cli_params: dict[str, ParamType]
    ) -> None:
        """Test that `BenchmarkConfigParams` agrees with the CLI."""
        benchmark_config_params = set(
            inspect.signature(BenchmarkConfigParams).parameters.keys()
        ) - {"run_with_cli"}
        cli_benchmark_params = set(cli_params.keys()) - {
            "model",
            "batch_size",
            "dataset_language",
            "model_language",
            "help",
        }
        assert benchmark_config_params == cli_benchmark_params

    def test_config_params_is_the_same_as_benchmark_config(self) -> None:
        """Test that `BenchmarkConfigParams` agrees with `BenchmarkConfig`."""
        benchmark_config_params = set(
            inspect.signature(BenchmarkConfigParams).parameters.keys()
        ) - {"dataset", "task", "language", "custom_datasets_file"}
        benchmark_config_fields = set(
            inspect.signature(BenchmarkConfig).parameters.keys()
        ) - {"datasets", "tasks", "languages"}
        assert benchmark_config_params == benchmark_config_fields
