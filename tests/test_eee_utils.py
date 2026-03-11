"""Tests for the `eee_utils` module."""

import json
from collections.abc import Generator
from pathlib import Path

import pytest

from euroeval.data_models import BenchmarkResult


class TestEeeUtils:
    """Tests for the EEE utility functions."""

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
        lines = [line for line in results_path.read_text().splitlines() if line.strip()]
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
        eval_results = {er["evaluation_name"]: er for er in eee["evaluation_results"]}
        assert "test_speed" in eval_results
        speed_config = eval_results["test_speed"]["metric_config"]
        # Speed metrics should only have lower_is_better, no score_type/min/max
        assert speed_config["lower_is_better"] is False
        assert "score_type" not in speed_config
        assert "min_score" not in speed_config
        assert "max_score" not in speed_config

        results_path.unlink(missing_ok=True)
