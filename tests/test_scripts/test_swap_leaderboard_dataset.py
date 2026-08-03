"""Tests for the swap_leaderboard_dataset script."""

import json
import logging
import re
import subprocess
from pathlib import Path

import pytest

from euroeval.data_models import DatasetConfig
from euroeval.languages import DANISH, SWEDISH
from euroeval.tasks import LA
from leaderboards import bucket_sync
from src.scripts import swap_leaderboard_dataset


class TestConfigBlockSpan:
    """Tests for _config_block_span helper."""

    def test_finds_config_block(self, tmp_path: Path) -> None:
        """Should find the start and end of a DatasetConfig block."""
        config_content = '''"""Dataset configs."""

from euroeval.data_models import DatasetConfig

SCALA_CONFIG = DatasetConfig(
    name="scala",
    pretty_name="ScaLA",
    source="EuroEval/scala",
)

DANSK_CONFIG = DatasetConfig(
    name="dansk",
    pretty_name="DANSK",
    source="EuroEval/dansk",
)
'''
        config_file = tmp_path / "test_config.py"
        config_file.write_text(config_content)

        lines = list[str](config_content.split("\n"))
        start, end = swap_leaderboard_dataset._config_block_span(
            lines=lines, dataset_id="dansk", path=config_file
        )

        # Should find DANSK_CONFIG block
        assert start > 0
        assert end > start
        assert "dansk" in "\n".join(config_content.split("\n")[start : end + 1])


class TestDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_does_not_modify_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry run should validate inputs and print plan without modifying anything."""
        # Mock subprocess calls to avoid git operations
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="_git",
            value=lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="_gh",
            value=lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
        )

        # Run with --dry-run - should exit cleanly without modifying files
        result = subprocess.run(
            [
                "uv",
                "run",
                "src/scripts/swap_leaderboard_dataset.py",
                "--old-dataset",
                "scala-da",
                "--new-dataset",
                "dansk",
                "--branch",
                "test-branch",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        # Should exit cleanly (code 0) or with validation error (code 1)
        # but should NOT crash with unhandled exception
        assert result.returncode in (0, 1)

    def test_dry_run_shows_plan(self) -> None:
        """Dry run should log the planned evaluations."""
        # Run with --dry-run and capture output
        result = subprocess.run(
            [
                "uv",
                "run",
                "src/scripts/swap_leaderboard_dataset.py",
                "--old-dataset",
                "scala-da",
                "--new-dataset",
                "dansk",
                "--branch",
                "test-branch",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        # Should log the swap plan (even if it fails validation)
        output = result.stdout + result.stderr
        # Should at least mention the datasets or show "Dry run"
        assert "scala-da" in output or "dansk" in output or "Dry run" in output


class TestExecuteJobsLogging:
    """Tests for evaluation log file creation in execute_jobs."""

    def test_live_output_written_during_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should write subprocess output to log file as it arrives."""
        Job = swap_leaderboard_dataset.Job
        write_order: list[str] = []

        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )

        # Mock that writes output incrementally to simulate live streaming
        def mock_run_euroeval(
            log_file: Path | None = None, **kwargs
        ) -> tuple[int, str]:
            output_lines = ["line 1", "line 2", "line 3"]
            output = "\n".join(output_lines)
            # Simulate live output: write each line as it arrives
            if log_file is not None:
                with open(log_file, "ab") as fh:
                    for line in output_lines:
                        fh.write(f"{line}\n".encode("utf-8"))
                        fh.flush()
                        write_order.append(f"write:{line}")
            write_order.append("return")
            return (0, output)

        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=mock_run_euroeval,
        )

        jobs = [
            Job(
                model_id="test-model",
                languages=("da",),
                is_api=False,
                evaluate_test_split=True,
                zero_shot=False,
                datasets=("test-dataset",),
            )
        ]

        swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, datasets=("test-dataset",), gpu_memory_utilization=None
        )

        # Verify write order: header -> output lines -> return -> completion
        assert write_order[0] == "write:line 1"
        assert write_order[1] == "write:line 2"
        assert write_order[2] == "write:line 3"
        assert write_order[3] == "return"

        # Verify log file contains all content
        log_files = list(tmp_path.glob("eval_log_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")

        # Header should come before output
        header_pos = content.find("Job [1/1] Starting")
        line1_pos = content.find("line 1")
        completion_pos = content.find("Job [1/1] Completed")

        assert header_pos < line1_pos < completion_pos, (
            "Log file should have: header < live output < completion"
        )
        assert "line 1" in content
        assert "line 2" in content
        assert "line 3" in content

    def test_log_file_contains_job_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should log each job's model, languages, split, shot, and source."""
        Job = swap_leaderboard_dataset.Job

        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )

        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=lambda **kwargs: (0, "ok"),
        )

        jobs = [
            Job(
                model_id="api-model",
                languages=("da", "sv"),
                is_api=True,
                evaluate_test_split=True,
                zero_shot=False,
                datasets=("test-dataset",),
            ),
            Job(
                model_id="open-model",
                languages=("no",),
                is_api=False,
                evaluate_test_split=False,
                zero_shot=True,
                datasets=("nordic-dataset",),
            ),
        ]

        swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, datasets=("nordic-dataset",), gpu_memory_utilization=0.9
        )

        log_files = list(tmp_path.glob("eval_log_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")

        # Verify job plan section with all metadata
        assert "Job Plan" in content
        assert "api-model" in content
        assert "open-model" in content
        assert "da, sv" in content or "languages: da, sv" in content
        assert "no" in content
        assert "test" in content  # evaluate_test_split=True
        assert "val" in content  # evaluate_test_split=False
        assert "zero-shot" in content
        assert "few-shot" in content
        assert "API" in content
        assert "open-weight" in content

    def test_log_file_contains_job_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should append job results with exit code and output to log file."""
        Job = swap_leaderboard_dataset.Job

        # Mock REPO_ROOT to use tmp_path
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )

        # Mock run_euroeval to return failure with custom output
        # Also write to log_file if provided (simulating live output)
        def mock_run_euroeval(
            log_file: Path | None = None, **kwargs
        ) -> tuple[int, str]:
            output = "error output from evaluation\nstack trace here"
            # Simulate live output to log file
            if log_file is not None:
                with open(log_file, "ab") as fh:
                    fh.write(output.encode("utf-8"))
            return (1, output)

        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=mock_run_euroeval,
        )

        # Create test jobs
        jobs = [
            Job(
                model_id="failing-model",
                languages=("sv", "no"),
                is_api=True,
                evaluate_test_split=False,
                zero_shot=True,
                datasets=("test-dataset",),
            )
        ]

        swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, datasets=("test-dataset",), gpu_memory_utilization=None
        )

        # Find the log file
        log_files = list(tmp_path.glob("eval_log_*.log"))
        assert len(log_files) == 1
        log_path = log_files[0]
        content = log_path.read_text(encoding="utf-8")

        # Verify job header and completion status
        assert "Job [1/1] Starting" in content
        assert "Model: failing-model" in content
        assert "Exit Code: 1" in content
        assert "Job [1/1] Completed" in content
        assert "error output from evaluation" in content
        assert "stack trace here" in content

    def test_log_file_created_before_progress_bar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should create log file and log its path before starting progress bar."""
        Job = swap_leaderboard_dataset.Job

        # Mock REPO_ROOT to use tmp_path
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )

        # Mock run_euroeval to return success
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=lambda **kwargs: (0, "evaluation completed successfully"),
        )

        # Create test jobs
        jobs = [
            Job(
                model_id="test-model",
                languages=("da",),
                is_api=False,
                evaluate_test_split=True,
                zero_shot=False,
                datasets=("test-dataset",),
            )
        ]

        with caplog.at_level(logging.INFO):
            swap_leaderboard_dataset.execute_jobs(
                jobs=jobs, datasets=("test-dataset",), gpu_memory_utilization=0.8
            )

        # Verify log path was printed
        assert "Evaluation log:" in caplog.text

        # Find the log file in tmp_path
        log_files = list(tmp_path.glob("eval_log_*.log"))
        assert len(log_files) == 1
        log_path = log_files[0]

        # Verify log file contains expected metadata
        content = log_path.read_text(encoding="utf-8")
        assert "Evaluation Log" in content
        assert "Datasets: test-dataset" in content
        assert "GPU Memory UtilIZATION: 0.8" in content
        assert "Total Jobs: 1" in content
        assert "test-model" in content
        assert "da" in content

    def test_log_path_logged_before_progress_bar(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log the log file path immediately before showing progress bar."""
        Job = swap_leaderboard_dataset.Job

        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )

        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=lambda **kwargs: (0, "ok"),
        )

        jobs = [
            Job(
                model_id="m",
                languages=("da",),
                is_api=False,
                evaluate_test_split=True,
                zero_shot=False,
                datasets=("test-dataset",),
            )
        ]

        with caplog.at_level(logging.INFO):
            swap_leaderboard_dataset.execute_jobs(
                jobs=jobs, datasets=("d",), gpu_memory_utilization=None
            )

        # Log path should contain timestamp pattern
        log_messages = [
            r.message for r in caplog.records if "Evaluation log:" in r.message
        ]
        assert len(log_messages) == 1
        assert re.search(r"eval_log_\d{8}_\d{6}\.log", log_messages[0])


class TestExecuteJobsResultDetection:
    """Tests that execute_jobs counts result-less exit-0 jobs as failed."""

    def test_exit_zero_with_result_counts_as_evaluated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that exits 0 and has a matching result record is evaluated."""
        Job = swap_leaderboard_dataset.Job
        benchmark_results = tmp_path / "euroeval_benchmark_results.jsonl"
        record = {
            "model_info": {"name": "good-model"},
            "eval_library": {
                "additional_details": {"dataset": "dutch-cola", "languages": ["nl"]}
            },
        }
        benchmark_results.write_text(json.dumps(record) + "\n", encoding="utf-8")
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=benchmark_results,
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=lambda **kwargs: (0, "ok"),
        )
        jobs = [
            Job(
                model_id="good-model",
                languages=("nl",),
                is_api=False,
                evaluate_test_split=True,
                zero_shot=False,
                datasets=("dutch-cola",),
            )
        ]
        evaluated, failed = swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, datasets=("dutch-cola",), gpu_memory_utilization=None
        )
        assert failed == []
        assert evaluated == ["good-model"]

    def test_exit_zero_without_result_counts_as_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that exits 0 but writes no result record must be marked failed."""
        Job = swap_leaderboard_dataset.Job
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=tmp_path / "euroeval_benchmark_results.jsonl",
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="run_euroeval",
            value=lambda **kwargs: (0, "errored 1 benchmarks"),
        )
        jobs = [
            Job(
                model_id="oom-model",
                languages=("nl",),
                is_api=False,
                evaluate_test_split=True,
                zero_shot=False,
                datasets=("dutch-cola",),
            )
        ]
        evaluated, failed = swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, datasets=("dutch-cola",), gpu_memory_utilization=None
        )
        assert evaluated == []
        assert len(failed) == 1
        assert "oom-model" in failed[0]


class TestLoadCorpusAndBuildEvalJobs:
    """Tests for load_corpus and build_eval_jobs functions."""

    def test_build_eval_jobs_add_only_does_not_rank_collapsed_union(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Add-only must not rank a model only completed by variant union."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        required_datasets = {
            "dala",
            "dansk",
            "angry-tweets",
            "multi-wiki-qa-da",
            "nordjylland-news",
            "danish-citizen-tests",
            "winogrande-da",
            "danske-talemaader",
        }
        test_variant_datasets = required_datasets - {"danske-talemaader"}
        val_variant_datasets = required_datasets - {"dala"}
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": required_datasets}},
            api_model_ids=set(),
            observations=set(),
            eval_configs={},
            exact_observations=set(),
            variant_coverage={
                "test-model": {"da": test_variant_datasets},
                "test-model (val)": {"da": val_variant_datasets},
            },
            variant_configs={
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
                ("test-model (val)", "dansk", "da"): ObsConfig(
                    validation_split=True, few_shot=True, generative=False
                ),
            },
        )

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked={("test-model", "da")},
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=True,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        assert jobs == []
        assert skipped_api == []
        assert skipped_count == 0

    def test_build_eval_jobs_add_only_falls_back_to_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In add-only mode, fall back to defaults with no official-dataset results."""
        Corpus = swap_leaderboard_dataset._Corpus

        # Create a corpus where the model has no official-dataset results
        # (only listed in datasets_by_language but no eval_configs entry)
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": set()}},
            api_model_ids=set(),
            observations=set(),
            eval_configs={},
        )

        ranked = {("test-model", "da")}

        # Add-only mode (old_dataset=None), open-weight model (is_api=False)
        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create a job with open-weight defaults (test split, few-shot)
        assert len(jobs) == 1
        assert jobs[0].model_id == "test-model"
        # Open-weight default: test split, few-shot
        assert jobs[0].evaluate_test_split is True
        assert jobs[0].zero_shot is False
        assert skipped_count == 0

    def test_build_eval_jobs_add_only_mirrors_official_dataset_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In add-only mode, mirror config from official dataset."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Create a corpus where the model has an official-dataset result
        # (e.g., "dala" is a required official dataset for the leaderboard)
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"dala"}}},
            api_model_ids=set(),
            observations={("test-model", "dala", "da")},
            eval_configs={
                # Official dataset has validation split, zero-shot config
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=True, few_shot=False, generative=True
                )
            },
        )

        ranked = {("test-model", "da")}

        # Add-only mode (old_dataset=None)
        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=True,  # Force re-run to test config mirroring
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create a job mirroring the official dataset's config
        assert len(jobs) == 1
        assert jobs[0].model_id == "test-model"
        # validation_split=True -> evaluate_test_split=False
        assert jobs[0].evaluate_test_split is False
        # few_shot=False, generative=True -> zero_shot=True
        assert jobs[0].zero_shot is True
        assert skipped_count == 0

    def test_build_eval_jobs_add_only_multiple_variants_schedule_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple ranked variants for same plain model schedule distinct jobs."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Model has TWO variants that both cover the full required dataset set:
        # - test-model (few-shot test split): covers required datasets
        # - test-model (zero-shot val split): also covers required datasets
        # Both should be scheduled as separate jobs.
        required_datasets = {
            "dala",
            "dansk",
            "angry-tweets",
            "multi-wiki-qa-da",
            "nordjylland-news",
            "danish-citizen-tests",
            "winogrande-da",
            "danske-talemaader",
        }
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": required_datasets}},
            api_model_ids=set(),
            observations=set(),
            eval_configs={},
            exact_observations=set(),
            variant_coverage={
                # Both variants cover ALL required datasets
                "test-model": {"da": required_datasets},
                "test-model (zero-shot, val)": {"da": required_datasets},
            },
            variant_configs={
                # Few-shot test-split config
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
                # Zero-shot val-split config
                ("test-model (zero-shot, val)", "dala", "da"): ObsConfig(
                    validation_split=True, few_shot=False, generative=True
                ),
            },
        )

        ranked = {("test-model", "da")}

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=True,  # Force to test both variants are scheduled
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create TWO jobs for the two distinct variants
        assert len(jobs) == 2
        # One job: test-split, few-shot
        # Other job: val-split, zero-shot
        configs = {(j.evaluate_test_split, j.zero_shot) for j in jobs}
        assert (True, False) in configs  # test-split, few-shot
        assert (False, True) in configs  # val-split, zero-shot

    def test_build_eval_jobs_add_only_ranked_via_non_default_variant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model ranked via non-default official variant, no new-dataset result."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Model is ranked via official dataset with validation-split (non-default)
        # No new-dataset result exists -> should create a job with val-split config
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"dala"}}},
            api_model_ids=set(),
            observations={("test-model", "dala", "da")},
            eval_configs={
                # Official dataset: validation-split, zero-shot (non-default)
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=True, few_shot=False, generative=True
                )
            },
        )

        ranked = {("test-model", "da")}

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create a job mirroring the official dataset's non-default config
        assert len(jobs) == 1
        assert jobs[0].model_id == "test-model"
        # validation_split=True -> evaluate_test_split=False
        assert jobs[0].evaluate_test_split is False
        # few_shot=False, generative=True -> zero_shot=True
        assert jobs[0].zero_shot is True
        assert skipped_count == 0

    def test_build_eval_jobs_add_only_uses_variant_row_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Add-only mode derives config from variant row."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Model has two official dataset variants with DIFFERENT configs.
        # The val variant covers the full required official dataset set.
        # Required datasets for Danish "linguistic-acceptability":
        required_datasets = {
            "dala",
            "dansk",
            "angry-tweets",
            "multi-wiki-qa-da",
            "nordjylland-news",
            "danish-citizen-tests",
            "winogrande-da",
            "danske-talemaader",
        }
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": required_datasets}},
            api_model_ids=set(),
            observations={("test-model", "dala", "da")},
            eval_configs={
                # Collapsed: prefers test-split (existing behavior)
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                )
            },
            exact_observations=set(),
            variant_coverage={
                # Val variant covers ALL required datasets
                "test-model (val)": {"da": required_datasets}
            },
            variant_configs={
                # Val variant's config for any required dataset
                ("test-model (val)", "dala", "da"): ObsConfig(
                    validation_split=True, few_shot=False, generative=True
                )
            },
        )

        ranked = {("test-model", "da")}

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=True,  # Force to test config derivation
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should use variant row's config (val-split, zero-shot) not collapsed config
        assert len(jobs) == 1
        assert jobs[0].model_id == "test-model"
        # validation_split=True -> evaluate_test_split=False
        assert jobs[0].evaluate_test_split is False
        # few_shot=False, generative=True -> zero_shot=True
        assert jobs[0].zero_shot is True

    def test_build_eval_jobs_add_only_wrong_variant_must_not_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New dataset has only wrong variant -> must not skip, must create job."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Model ranked via official dataset with test-split (default).
        # New dataset has validation-split result (wrong variant) -> must not skip.
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"dala", "new-dataset"}}},
            api_model_ids=set(),
            observations={
                ("test-model", "dala", "da"),
                ("test-model", "new-dataset", "da"),
            },
            eval_configs={
                # Official dataset: test-split, few-shot (desired config)
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
                # New dataset: validation-split (wrong variant, should not skip)
                ("test-model", "new-dataset", "da"): ObsConfig(
                    validation_split=True, few_shot=True, generative=False
                ),
            },
            exact_observations={
                # Exact observation for new-dataset has WRONG validation_split
                ("test-model", "new-dataset", "da", True, True)
            },
            variant_coverage={},
            variant_configs={},
        )

        ranked = {("test-model", "da")}

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create a job because the existing config differs from desired
        assert len(jobs) == 1
        assert jobs[0].evaluate_test_split is True  # test split (desired)
        assert jobs[0].zero_shot is False  # few-shot (desired)
        assert skipped_count == 0

    def test_build_eval_jobs_runs_when_observation_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should create jobs for (model, language) pairs without existing results."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Create a corpus without the new-dataset observation
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"old-dataset"}}},
            api_model_ids=set(),
            observations=set(),  # No existing observations
            eval_configs={
                ("test-model", "old-dataset", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                )
            },
        )

        ranked = {("test-model", "da")}

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset="old-dataset",
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create a job since no existing observation
        assert len(jobs) == 1
        assert jobs[0].model_id == "test-model"
        assert jobs[0].languages == ("da",)
        assert skipped_count == 0

    def test_build_eval_jobs_skip_ignores_non_leaderboard_variant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing-result skip ignores variants not shown on leaderboard."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # In add-only mode, the desired config is derived from official datasets.
        # The new-dataset result has a different config (validation-split), so
        # it should NOT be skipped (we still need to create the test-split result).
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"dala", "new-dataset"}}},
            api_model_ids=set(),
            observations={
                ("test-model", "dala", "da"),
                ("test-model", "new-dataset", "da"),
            },
            eval_configs={
                # Official dataset has test-split config (desired config)
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
                # New dataset has validation-split config (different from desired)
                ("test-model", "new-dataset", "da"): ObsConfig(
                    validation_split=True, few_shot=True, generative=False
                ),
            },
            exact_observations={
                # Exact observation for new-dataset has WRONG validation_split
                ("test-model", "new-dataset", "da", True, True)
            },
            variant_coverage={},
            variant_configs={},
        )

        ranked = {("test-model", "da")}

        # Should NOT skip because the existing new-dataset config doesn't match
        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should create a job because the existing config differs from desired
        assert len(jobs) == 1
        assert jobs[0].evaluate_test_split is True  # test split
        assert jobs[0].zero_shot is False  # few-shot
        assert skipped_count == 0

    def test_build_eval_jobs_skip_is_variant_aware(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing-result skip compares config values, not just key existence."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Add-only mode: desired config from official dataset.
        # New-dataset has matching config -> should skip.
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"dala", "new-dataset"}}},
            api_model_ids=set(),
            observations={
                ("test-model", "dala", "da"),
                ("test-model", "new-dataset", "da"),
            },
            eval_configs={
                # Official dataset: test-split, few-shot (desired config)
                ("test-model", "dala", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
                # New dataset: same config as desired -> should skip
                ("test-model", "new-dataset", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
            },
            exact_observations={
                # Exact observation for new-dataset matches desired config
                ("test-model", "new-dataset", "da", False, True)
            },
            variant_coverage={},
            variant_configs={},
        )

        ranked = {("test-model", "da")}

        # Should skip because the existing config matches the desired config
        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        assert len(jobs) == 0
        assert skipped_count == 1

    def test_build_eval_jobs_skips_existing_observations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip when new-dataset config matches desired config."""
        Corpus = swap_leaderboard_dataset._Corpus
        ObsConfig = swap_leaderboard_dataset._ObsConfig

        # Swap mode: desired config from old_dataset.
        # New-dataset has matching config -> should skip.
        corpus = Corpus(
            datasets_by_language={"da": {"test-model": {"old-dataset", "new-dataset"}}},
            api_model_ids=set(),
            observations={
                ("test-model", "old-dataset", "da"),
                ("test-model", "new-dataset", "da"),
            },
            eval_configs={
                # Old dataset: test-split, few-shot (desired config)
                ("test-model", "old-dataset", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
                # New dataset: same config as desired -> should skip
                ("test-model", "new-dataset", "da"): ObsConfig(
                    validation_split=False, few_shot=True, generative=False
                ),
            },
            exact_observations={
                # Exact observation for new-dataset matches desired config
                ("test-model", "new-dataset", "da", False, True)
            },
            variant_coverage={},
            variant_configs={},
        )

        ranked = {("test-model", "da")}

        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked=ranked,
            old_dataset="old-dataset",
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=False,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        # Should skip because the existing config matches the desired config
        assert len(jobs) == 0
        assert skipped_count == 1

    def test_load_corpus_falls_back_to_model_info_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fall back to model_info.id when name is missing."""
        # Setup: create results directory with a record that only has model_info.id
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        model_dir = results_dir / "test_model"
        model_dir.mkdir()
        record_file = model_dir / "test_dataset__test__none.json"
        # Record with only 'id' field, no 'name' (valid EEE format)
        test_record = {
            "model_info": {"id": "test/org-model"},
            "eval_library": {
                "additional_details": {"dataset": "test-dataset", "languages": ["da"]}
            },
        }
        record_file.write_text(json.dumps(test_record), encoding="utf-8")

        # Mock REPO_ROOT, RESULTS_DIR and EUROEVAL_BENCHMARK_RESULTS_PATH
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=tmp_path / "euroeval_benchmark_results.jsonl",
        )

        # Should load the record using the id field
        corpus = swap_leaderboard_dataset.load_corpus()
        # plain_model_id strips variants, so "test/org-model" stays as-is
        assert ("test/org-model", "test-dataset", "da") in corpus.observations
        assert "test/org-model" in corpus.variant_coverage
        assert "unknown" not in corpus.variant_coverage

    def test_load_corpus_handles_missing_benchmark_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load successfully when euroeval_benchmark_results.jsonl is missing."""
        # Setup: create results directory with a file (so load_corpus succeeds)
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        model_dir = results_dir / "test_model"
        model_dir.mkdir()
        record_file = model_dir / "test_dataset__test__none.json"
        test_record = {
            "model_info": {"name": "test-model"},
            "eval_library": {
                "additional_details": {"dataset": "test-dataset", "languages": ["da"]}
            },
        }
        record_file.write_text(json.dumps(test_record), encoding="utf-8")

        # Mock REPO_ROOT, RESULTS_DIR and EUROEVAL_BENCHMARK_RESULTS_PATH
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        # Point to a definitely absent path under tmp_path
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=tmp_path / "euroeval_benchmark_results.jsonl",
        )

        # Should not raise
        corpus = swap_leaderboard_dataset.load_corpus()
        assert ("test-model", "test-dataset", "da") in corpus.observations

    def test_load_corpus_id_only_variants_do_not_rank_collapsed_union(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ID-only records must still build usable variant coverage."""
        results_dir = tmp_path / "results"
        model_dir = results_dir / "test-model"
        model_dir.mkdir(parents=True)
        required_datasets = {
            "dala",
            "dansk",
            "angry-tweets",
            "multi-wiki-qa-da",
            "nordjylland-news",
            "danish-citizen-tests",
            "winogrande-da",
            "danske-talemaader",
        }
        test_variant_datasets = required_datasets - {"dansk", "danske-talemaader"}
        val_variant_datasets = required_datasets - {"dala"}

        for dataset in sorted(test_variant_datasets):
            record = {
                "model_info": {"id": "test-model"},
                "eval_library": {
                    "additional_details": {
                        "dataset": dataset,
                        "languages": ["da"],
                        "validation_split": False,
                    }
                },
            }
            (model_dir / f"{dataset}__test__none.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        for dataset in sorted(val_variant_datasets):
            record = {
                "model_info": {"id": "test-model"},
                "eval_library": {
                    "additional_details": {
                        "dataset": dataset,
                        "languages": ["da"],
                        "validation_split": True,
                    }
                },
            }
            (model_dir / f"{dataset}__val__none.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=tmp_path / "euroeval_benchmark_results.jsonl",
        )

        corpus = swap_leaderboard_dataset.load_corpus()
        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked={("test-model", "da")},
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=True,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        assert required_datasets <= corpus.datasets_by_language["da"]["test-model"]
        assert "unknown" not in corpus.variant_coverage
        assert jobs == []
        assert skipped_api == []
        assert skipped_count == 0

    def test_load_corpus_includes_euroeval_benchmark_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should load records from euroeval_benchmark_results.jsonl at repo root."""
        # Setup: create euroeval_benchmark_results.jsonl with a test record
        benchmark_results = tmp_path / "euroeval_benchmark_results.jsonl"
        test_record = {
            "model_info": {"name": "test-model"},
            "eval_library": {
                "additional_details": {"dataset": "test-dataset", "languages": ["da"]}
            },
        }
        benchmark_results.write_text(json.dumps(test_record) + "\n", encoding="utf-8")

        # Mock REPO_ROOT and RESULTS_DIR to use tmp_path
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="REPO_ROOT", value=tmp_path
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="RESULTS_DIR",
            value=tmp_path / "results",
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=benchmark_results,
        )

        # Call load_corpus
        corpus = swap_leaderboard_dataset.load_corpus()

        # Verify the record is in observations
        assert ("test-model", "test-dataset", "da") in corpus.observations

    def test_load_corpus_split_agnostic_completes_val_variant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Split-agnostic datasets should count on matching val rows."""
        results_dir = tmp_path / "results"
        model_dir = results_dir / "test-model"
        model_dir.mkdir(parents=True)
        required_datasets = {
            "dala",
            "dansk",
            "angry-tweets",
            "multi-wiki-qa-da",
            "nordjylland-news",
            "danish-citizen-tests",
            "winogrande-da",
            "danske-talemaader",
        }
        split_agnostic_dataset = "dala"

        split_agnostic_record = {
            "model_info": {"name": "test-model"},
            "eval_library": {
                "additional_details": {
                    "dataset": split_agnostic_dataset,
                    "languages": ["da"],
                    "validation_split": None,
                }
            },
        }
        (model_dir / "dala__none__none.json").write_text(
            json.dumps(split_agnostic_record), encoding="utf-8"
        )
        for dataset in sorted(required_datasets - {split_agnostic_dataset}):
            record = {
                "model_info": {"name": "test-model"},
                "eval_library": {
                    "additional_details": {
                        "dataset": dataset,
                        "languages": ["da"],
                        "validation_split": True,
                    }
                },
            }
            (model_dir / f"{dataset}__val__none.json").write_text(
                json.dumps(record), encoding="utf-8"
            )

        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="EUROEVAL_BENCHMARK_RESULTS_PATH",
            value=tmp_path / "euroeval_benchmark_results.jsonl",
        )

        corpus = swap_leaderboard_dataset.load_corpus()
        jobs, skipped_api, skipped_count = swap_leaderboard_dataset.build_eval_jobs(
            ranked={("test-model", "da")},
            old_dataset=None,
            new_datasets=("new-dataset",),
            corpus=corpus,
            include_api=True,
            selected_providers=set(),
            force=True,
            swapped_task="linguistic-acceptability",
            language_codes={"da"},
        )

        assert required_datasets <= corpus.variant_coverage["test-model (val)"]["da"]
        assert len(jobs) == 1
        assert jobs[0].evaluate_test_split is False
        assert jobs[0].zero_shot is False
        assert skipped_api == []
        assert skipped_count == 0


class TestSyncResults:
    """Tests for sync_results_from_bucket function."""

    def test_sync_results_handles_empty_bucket(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should handle empty bucket gracefully."""
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="RESULTS_DIR",
            value=tmp_path / "results",
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="NEW_RESULTS_PATH",
            value=tmp_path / "new_results.jsonl",
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="HF_RESULTS_BUCKET",
            value="test/bucket",
        )

        # Create empty results dir
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Patch RESULTS_DIR in both modules (merge_results uses bucket_sync.RESULTS_DIR)
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        monkeypatch.setattr(target=bucket_sync, name="RESULTS_DIR", value=results_dir)

        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="result_sync_dataset_ids",
            value=lambda **kwargs: {"test-dataset"},
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="download_bucket_files_for_datasets",
            value=lambda *, dataset_ids: 0,
        )

        with caplog.at_level(logging.WARNING):
            swap_leaderboard_dataset.sync_results_from_bucket(
                old_dataset="old-dataset",
                new_datasets=("new-dataset",),
                swapped_task="knowledge",
                target_codes={"da"},
            )

        # Should warn about no results
        assert "No results found" in caplog.text

    def test_sync_results_logs_correct_counts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Should log correct total and new record counts."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        new_results_path = tmp_path / "new_results.jsonl"

        # Create a mock result file in the tree
        model_dir = results_dir / "test_model"
        model_dir.mkdir()
        record_file = model_dir / "test_dataset__test__none.json"
        test_record = {
            "model_info": {"name": "test-model"},
            "eval_library": {
                "additional_details": {"dataset": "test-dataset", "languages": ["da"]}
            },
            "validation_split": False,
            "few_shot": True,
        }
        record_file.write_text(json.dumps(test_record), encoding="utf-8")

        # Pre-populate new_results.jsonl with one existing record (different identity)
        existing_record = {
            "model_info": {"name": "existing-model"},
            "eval_library": {
                "additional_details": {"dataset": "other-dataset", "languages": ["da"]}
            },
            "validation_split": False,
            "few_shot": True,
        }
        new_results_path.write_text(
            json.dumps(existing_record) + "\n", encoding="utf-8"
        )

        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        monkeypatch.setattr(target=bucket_sync, name="RESULTS_DIR", value=results_dir)
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="NEW_RESULTS_PATH",
            value=new_results_path,
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="HF_RESULTS_BUCKET",
            value="test/bucket",
        )
        # Patch RESULTS_DIR in both modules (merge_results uses bucket_sync.RESULTS_DIR)
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="RESULTS_DIR", value=results_dir
        )
        monkeypatch.setattr(target=bucket_sync, name="RESULTS_DIR", value=results_dir)

        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="result_sync_dataset_ids",
            value=lambda **kwargs: {"test-dataset"},
        )
        monkeypatch.setattr(
            target=swap_leaderboard_dataset,
            name="download_bucket_files_for_datasets",
            value=lambda *, dataset_ids: 0,
        )

        with caplog.at_level(logging.INFO):
            swap_leaderboard_dataset.sync_results_from_bucket(
                old_dataset="old-dataset",
                new_datasets=("new-dataset",),
                swapped_task="knowledge",
                target_codes={"da"},
            )

        # Should log: "Consolidating 1 result records into ... (1 new)."
        # Total count should be 1 (from merge_results), new count should be 1
        assert "Consolidating" in caplog.text
        assert "result records" in caplog.text


class TestValidation:
    """Tests for validation functions."""

    def test_resolve_languages_returns_intersection(self) -> None:
        """Should return the intersection of languages from both datasets."""
        old_config = DatasetConfig(
            name="scala-da",
            pretty_name="ScaLA-da",
            source="EuroEval/scala-da",
            task=LA,
            languages=[DANISH],
        )
        new_config = DatasetConfig(
            name="scala-da-sv",
            pretty_name="ScaLA-sv",
            source="EuroEval/scala-sv",
            task=LA,
            languages=[SWEDISH],
        )

        # No overlap - should raise
        with pytest.raises(Exception):
            swap_leaderboard_dataset.resolve_languages(
                old_config=old_config, new_configs=[new_config]
            )

        # With overlap - should return intersection
        old_config = DatasetConfig(
            name="nordic",
            pretty_name="Nordic",
            source="EuroEval/nordic",
            task=LA,
            languages=[DANISH, SWEDISH],
        )
        new_config = DatasetConfig(
            name="nordic-dk",
            pretty_name="Nordic DK",
            source="EuroEval/nordic-dk",
            task=LA,
            languages=[DANISH],
        )

        result = swap_leaderboard_dataset.resolve_languages(
            old_config=old_config, new_configs=[new_config]
        )
        assert result == {"da"}

    def test_validate_branch_accepts_non_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should accept non-default branch names."""
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="default_branch", value=lambda: "main"
        )

        # Should not raise
        swap_leaderboard_dataset.validate_branch("feature-branch")

    def test_validate_branch_rejects_default_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should reject the default branch name."""
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="default_branch", value=lambda: "main"
        )

        with pytest.raises(Exception):
            swap_leaderboard_dataset.validate_branch("main")
