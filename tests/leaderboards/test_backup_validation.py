"""Tests for backup validation of raw vs processed results."""

from __future__ import annotations

import json
import typing as t
from pathlib import Path
from unittest.mock import patch

import pytest

from euroeval.jsonl_io import parse_jsonl_lines
from leaderboards.backup import _validate_results, backup_results
from leaderboards.eee_validation import validate_eee_record


@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Path:
    """Create a temporary results directory structure.

    Args:
        tmp_path:
            Pytest temporary path fixture.

    Returns:
        Path to the temporary results directory.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


def _create_eee_record(
    model_name: str = "test/model", include_precious_metadata: bool = False
) -> dict[str, t.Any]:
    """Create a minimal EEE-format record.

    Args:
        model_name (optional):
            Model name for the record. Defaults to "test/model".
        include_precious_metadata (optional):
            Whether to include precious metadata fields. Defaults to False.

    Returns:
        A minimal EEE-format record dictionary.
    """
    record: dict[str, t.Any] = {
        "schema_version": "1.0",
        "model_info": {"name": model_name, "additional_details": {}},
        "eval_library": {"name": "euroeval", "additional_details": {}},
        "evaluation_results": [],
    }
    if include_precious_metadata:
        record["model_info"]["additional_details"].update(
            {"commercially_licensed": False, "open": True, "trained_from_scratch": True}
        )
    return record


class TestValidateResultsRaw:
    """Tests for _validate_results with raw results (missing precious metadata)."""

    def test_raw_results_without_precious_metadata_pass(
        self, temp_results_dir: Path
    ) -> None:
        """Raw results without precious metadata should pass backup validation."""
        # Create a result file without precious metadata
        record = _create_eee_record(include_precious_metadata=False)
        model_file = temp_results_dir / "test_model.jsonl"
        model_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        # Patch RESULTS_DIR to use our temp directory
        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            # Should not raise
            _validate_results()

    def test_raw_results_with_concatenated_json_pass(
        self, temp_results_dir: Path
    ) -> None:
        """Results with concatenated JSON objects (}{) should pass validation."""
        # Create two records concatenated on one line
        record1 = _create_eee_record(model_name="model/one")
        record2 = _create_eee_record(model_name="model/two")
        model_file = temp_results_dir / "test_model.jsonl"
        # Write concatenated JSON (no newline between objects)
        model_file.write_text(
            json.dumps(record1) + json.dumps(record2) + "\n", encoding="utf-8"
        )

        # Verify parse_jsonl_lines handles this correctly
        lines = model_file.read_text(encoding="utf-8").splitlines()
        records = parse_jsonl_lines(lines=lines, source=str(model_file), strict=False)
        assert len(records) == 2

        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            # Should not raise
            _validate_results()

    def test_empty_results_dir_fails(self, temp_results_dir: Path) -> None:
        """Empty results directory should raise FileNotFoundError."""
        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            with pytest.raises(FileNotFoundError, match="No result files found"):
                _validate_results()

    def test_nonexistent_results_dir_fails(self, tmp_path: Path) -> None:
        """Non-existent results directory should raise FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent"
        with patch("leaderboards.backup.RESULTS_DIR", nonexistent):
            with pytest.raises(FileNotFoundError, match="Results directory not found"):
                _validate_results()

    def test_invalid_json_rejected(self, temp_results_dir: Path) -> None:
        """Invalid JSON raises ValueError during validation (strict=True)."""
        model_file = temp_results_dir / "invalid.jsonl"
        model_file.write_text("not valid json\n", encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            with pytest.raises(ValueError, match="structural issues"):
                _validate_results()

    def test_only_invalid_json_all_files_empty_fails(
        self, temp_results_dir: Path
    ) -> None:
        """If all files have only invalid JSON, validation raises ValueError.

        With strict=True, invalid JSON is rejected rather than silently skipped.
        """
        model_file = temp_results_dir / "invalid.jsonl"
        model_file.write_text("not valid json\n", encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            with pytest.raises(ValueError, match="structural issues"):
                _validate_results()

    def test_missing_eee_envelope_fails(self, temp_results_dir: Path) -> None:
        """Missing EEE envelope structure should raise ValueError."""
        model_file = temp_results_dir / "bad_structure.jsonl"
        # Valid JSON but missing EEE structure
        bad_record = {"foo": "bar"}
        model_file.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            with pytest.raises(ValueError, match="structural issues"):
                _validate_results()


class TestValidateProcessedResults:
    """Tests for validate_eee_record requiring precious metadata."""

    def test_processed_record_with_precious_metadata_passes(self) -> None:
        """Processed record with all precious metadata passes validation."""
        record = _create_eee_record(include_precious_metadata=True)
        # Should not raise
        validate_eee_record(record)

    def test_processed_record_missing_precious_metadata_fails(self) -> None:
        """Processed record missing precious metadata fails validation."""
        record = _create_eee_record(include_precious_metadata=False)
        with pytest.raises(ValueError, match="missing precious metadata"):
            validate_eee_record(record)

    def test_processed_record_non_bool_precious_metadata_fails(self) -> None:
        """Processed record with non-boolean precious metadata fails validation."""
        record = _create_eee_record(include_precious_metadata=True)
        record["model_info"]["additional_details"]["commercially_licensed"] = "yes"
        with pytest.raises(ValueError, match="non-boolean precious metadata"):
            validate_eee_record(record)


class TestBackupResultsIntegration:
    """Integration tests for backup_results with raw results."""

    def test_backup_results_with_raw_results_no_precious_metadata(
        self, temp_results_dir: Path, tmp_path: Path
    ) -> None:
        """backup_results should succeed with raw results missing precious metadata."""
        # Create a result file without precious metadata (simulating HF bucket sync)
        record = _create_eee_record(include_precious_metadata=False)
        model_file = temp_results_dir / "test_model.jsonl"
        model_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        # Create temp backup directory
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        with (
            patch("leaderboards.backup.RESULTS_DIR", temp_results_dir),
            patch("leaderboards.backup.BACKUPS_DIR", backup_dir),
            patch("leaderboards.backup.BACKUPS_MAX_BYTES", 1024 * 1024 * 10),
        ):
            # Should not raise - raw results are allowed to miss precious metadata
            backup_path = backup_results(source=temp_results_dir)
            assert backup_path is not None
            assert backup_path.exists()
