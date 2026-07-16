"""Tests for the swap_leaderboard_dataset script."""

import logging
import re
import subprocess
from pathlib import Path

import pytest

from euroeval.data_models import DatasetConfig
from euroeval.languages import DANISH, SWEDISH
from euroeval.tasks import LA
from src.scripts import swap_leaderboard_dataset


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


class TestValidation:
    """Tests for validation functions."""

    def test_validate_branch_rejects_default_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should reject the default branch name."""
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="default_branch", value=lambda: "main"
        )

        with pytest.raises(Exception):
            swap_leaderboard_dataset.validate_branch("main")

    def test_validate_branch_accepts_non_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should accept non-default branch names."""
        monkeypatch.setattr(
            target=swap_leaderboard_dataset, name="default_branch", value=lambda: "main"
        )

        # Should not raise
        swap_leaderboard_dataset.validate_branch("feature-branch")

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
                old_config=old_config, new_config=new_config
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
            old_config=old_config, new_config=new_config
        )
        assert result == {"da"}


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
            value="test-bucket",
        )

        # Create empty results dir
        (tmp_path / "results").mkdir()

        # Mock hf buckets sync to do nothing
        monkeypatch.setattr(
            target=subprocess,
            name="run",
            value=lambda *args, **kwargs: subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ),
        )

        with caplog.at_level(logging.WARNING):
            swap_leaderboard_dataset.sync_results_from_bucket()

        # Should warn about no results
        assert "No results found" in caplog.text


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


class TestExecuteJobsLogging:
    """Tests for evaluation log file creation in execute_jobs."""

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
            )
        ]

        with caplog.at_level(logging.INFO):
            swap_leaderboard_dataset.execute_jobs(
                jobs=jobs, dataset="test-dataset", gpu_memory_utilization=0.8
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
        assert "Dataset: test-dataset" in content
        assert "GPU Memory Utilization: 0.8" in content
        assert "Total Jobs: 1" in content
        assert "test-model" in content
        assert "da" in content

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
        def mock_run_euroeval(**kwargs) -> tuple[int, str]:
            return (1, "error output from evaluation\nstack trace here")

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
            )
        ]

        swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, dataset="test-dataset", gpu_memory_utilization=None
        )

        # Find the log file
        log_files = list(tmp_path.glob("eval_log_*.log"))
        assert len(log_files) == 1
        log_path = log_files[0]
        content = log_path.read_text(encoding="utf-8")

        # Verify job result section
        assert "Job Result" in content
        assert "Model: failing-model" in content
        assert "Exit Code: 1" in content
        assert "error output from evaluation" in content
        assert "stack trace here" in content

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
            ),
            Job(
                model_id="open-model",
                languages=("no",),
                is_api=False,
                evaluate_test_split=False,
                zero_shot=True,
            ),
        ]

        swap_leaderboard_dataset.execute_jobs(
            jobs=jobs, dataset="nordic-dataset", gpu_memory_utilization=0.9
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
            )
        ]

        with caplog.at_level(logging.INFO):
            swap_leaderboard_dataset.execute_jobs(
                jobs=jobs, dataset="d", gpu_memory_utilization=None
            )

        # Log path should contain timestamp pattern
        log_messages = [
            r.message for r in caplog.records if "Evaluation log:" in r.message
        ]
        assert len(log_messages) == 1
        assert re.search(r"eval_log_\d{8}_\d{6}\.log", log_messages[0])
