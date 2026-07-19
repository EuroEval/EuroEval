"""Tests for backup validation of raw vs processed results."""

from __future__ import annotations

import json
import tarfile
import typing as t
from pathlib import Path
from unittest.mock import patch

import pytest

from leaderboards.backup import (
    _content_hash,
    _extract_backup,
    _validate_results,
    _write_snapshot,
    backup_results,
    restore_from_backup_if_missing,
)
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
        # Create a result file in tree structure (one record per file)
        record = _create_eee_record(include_precious_metadata=False)
        model_dir = temp_results_dir / "test_model"
        model_dir.mkdir()
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text(json.dumps(record), encoding="utf-8")

        # Patch RESULTS_DIR to use our temp directory
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
        model_dir = temp_results_dir / "test_model"
        model_dir.mkdir()
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text("not valid json", encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            with pytest.raises(ValueError, match="structural issues"):
                _validate_results()

    def test_only_invalid_json_all_files_empty_fails(
        self, temp_results_dir: Path
    ) -> None:
        """If all files have only invalid JSON, validation raises ValueError.

        With strict=True, invalid JSON is rejected rather than silently skipped.
        """
        model_dir = temp_results_dir / "test_model"
        model_dir.mkdir()
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text("not valid json", encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", temp_results_dir):
            with pytest.raises(ValueError, match="structural issues"):
                _validate_results()

    def test_missing_eee_envelope_fails(self, temp_results_dir: Path) -> None:
        """Missing EEE envelope structure should raise ValueError."""
        model_dir = temp_results_dir / "test_model"
        model_dir.mkdir()
        json_file = model_dir / "dataset__test__zero.json"
        # Valid JSON but missing EEE structure
        bad_record = {"foo": "bar"}
        json_file.write_text(json.dumps(bad_record), encoding="utf-8")

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
    """Integration tests for backup_results with tree structure results."""

    def test_backup_results_with_raw_results_no_precious_metadata(
        self, temp_results_dir: Path, tmp_path: Path
    ) -> None:
        """backup_results should succeed with raw results missing precious metadata."""
        # Create tree structure with result file without precious metadata
        record = _create_eee_record(include_precious_metadata=False)
        model_dir = temp_results_dir / "test_model"
        model_dir.mkdir()
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text(json.dumps(record), encoding="utf-8")

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


class TestTreeStructureBackup:
    """Tests for backup/restore with the new tree layout."""

    def test_write_snapshot_preserves_tree_structure(
        self, tmp_path: Path
    ) -> None:
        """_write_snapshot should tar the full tree structure, not flatten."""
        # Create tree structure
        source = tmp_path / "results"
        model_a_dir = source / "model_A"
        model_b_dir = source / "model_B"
        model_a_dir.mkdir(parents=True)
        model_b_dir.mkdir(parents=True)

        # Create JSON files in tree
        record1 = _create_eee_record(model_name="model_A")
        record2 = _create_eee_record(model_name="model_B")

        file1 = model_a_dir / "dataset__test__zero.json"
        file2 = model_b_dir / "dataset__val__few.json"
        file1.write_text(json.dumps(record1), encoding="utf-8")
        file2.write_text(json.dumps(record2), encoding="utf-8")

        backups_dir = tmp_path / "backups"
        backups_dir.mkdir()

        with (
            patch("leaderboards.backup.RESULTS_DIR", source),
            patch("leaderboards.backup.BACKUPS_DIR", backups_dir),
            patch("leaderboards.backup.BACKUPS_MAX_BYTES", 1024 * 1024 * 10),
        ):
            backup_path = _write_snapshot(source=source)

        assert backup_path is not None
        assert backup_path.exists()

        # Verify the tar preserves tree structure
        with tarfile.open(backup_path, "r:gz") as tar:
            names = tar.getnames()
            # Should have entries like "results/model_A/..." not flattened
            assert any("model_A" in n for n in names)
            assert any("model_B" in n for n in names)
            # Should NOT have flat structure at root
            assert not any(n == "dataset__test__zero.json" for n in names)

    def test_extract_backup_preserves_tree_structure(
        self, tmp_path: Path
    ) -> None:
        """_extract_backup should preserve the tree structure on extraction."""
        # Create a backup with tree structure
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()

        # Create source tree
        source = tmp_path / "source"
        model_dir = source / "test_model"
        model_dir.mkdir(parents=True)
        record = _create_eee_record(model_name="test_model")
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text(json.dumps(record), encoding="utf-8")

        # Create backup tar
        backup_path = backup_dir / "results_20240101_120000_abc123.tar.gz"
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(json_file, arcname="results/test_model/dataset__test__zero.json")

        # Extract to new location
        dest = tmp_path / "dest"

        with patch("leaderboards.backup.BACKUP_ARCHIVE_ROOT", "results"):
            _extract_backup(archive=backup_path, dest=dest)

        # Verify tree structure is preserved
        extracted_file = dest / "test_model" / "dataset__test__zero.json"
        assert extracted_file.exists()
        assert json.loads(extracted_file.read_text()) == record

        # Verify no flat file at dest root
        flat_file = dest / "dataset__test__zero.json"
        assert not flat_file.exists()

    def test_validate_results_with_tree_structure(
        self, tmp_path: Path
    ) -> None:
        """_validate_results should sample JSON files from tree layout."""
        results_dir = tmp_path / "results"
        model_dir = results_dir / "test_model"
        model_dir.mkdir(parents=True)

        # Create JSON record in tree
        record = _create_eee_record(include_precious_metadata=False)
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text(json.dumps(record), encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", results_dir):
            # Should validate successfully
            _validate_results()

    def test_validate_results_tree_invalid_json_fails(
        self, tmp_path: Path
    ) -> None:
        """_validate_results should fail on invalid JSON in tree structure."""
        results_dir = tmp_path / "results"
        model_dir = results_dir / "test_model"
        model_dir.mkdir(parents=True)

        # Write invalid JSON
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text("not valid json", encoding="utf-8")

        with patch("leaderboards.backup.RESULTS_DIR", results_dir):
            with pytest.raises(ValueError, match="structural issues"):
                _validate_results()

    def test_content_hash_uses_tree_paths(self, tmp_path: Path) -> None:
        """_content_hash should use relative paths in tree for stable ordering."""
        model_dir = tmp_path / "model_A"
        model_dir.mkdir()

        file1 = model_dir / "dataset1__test__zero.json"
        file2 = model_dir / "dataset2__val__few.json"
        file1.write_text("content1", encoding="utf-8")
        file2.write_text("content2", encoding="utf-8")

        # Hash should be stable with different orderings
        hash1 = _content_hash(paths=[file1, file2])
        hash2 = _content_hash(paths=[file2, file1])
        assert hash1 == hash2, "Hash should be order-independent"

    def test_restore_from_backup_understands_tree_layout(
        self, tmp_path: Path
    ) -> None:
        """restore_from_backup_if_missing should check for tree layout .json files."""
        # Test 1: Empty dir should trigger restore
        target_empty = tmp_path / "empty"
        target_empty.mkdir()

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create backup with tree structure
        source_backup = tmp_path / "source_backup"
        src_model = source_backup / "test_model"
        src_model.mkdir(parents=True)
        record = _create_eee_record(model_name="test_model")
        src_json = src_model / "dataset__test__zero.json"
        src_json.write_text(json.dumps(record), encoding="utf-8")

        backup_path = backup_dir / "results_20240101_120000_abc123.tar.gz"
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(src_model, arcname="results/test_model")

        with (
            patch("leaderboards.backup.RESULTS_DIR", target_empty),
            patch("leaderboards.backup.BACKUPS_DIR", backup_dir),
        ):
            result = restore_from_backup_if_missing(target=target_empty)

        assert result is True
        # Verify tree was restored
        assert (target_empty / "test_model" / "dataset__test__zero.json").exists()

        # Test 2: Dir with tree files should NOT trigger restore
        with (
            patch("leaderboards.backup.RESULTS_DIR", target_empty),
            patch("leaderboards.backup.BACKUPS_DIR", backup_dir),
        ):
            result = restore_from_backup_if_missing(target=target_empty)
        assert result is False, "Should not restore when tree files exist"

    def test_write_snapshot_skips_identical_tree_content(
        self, tmp_path: Path
    ) -> None:
        """_write_snapshot should skip if tree content hash matches."""
        source = tmp_path / "results"
        model_dir = source / "test_model"
        model_dir.mkdir(parents=True)

        record = _create_eee_record(model_name="test_model")
        json_file = model_dir / "dataset__test__zero.json"
        json_file.write_text(json.dumps(record), encoding="utf-8")

        backups_dir = tmp_path / "backups"
        backups_dir.mkdir()

        # First backup
        with (
            patch("leaderboards.backup.RESULTS_DIR", source),
            patch("leaderboards.backup.BACKUPS_DIR", backups_dir),
            patch("leaderboards.backup.BACKUPS_MAX_BYTES", 1024 * 1024 * 10),
        ):
            first_backup = _write_snapshot(source=source)

        assert first_backup is not None

        # Second call with unchanged content should return None
        with (
            patch("leaderboards.backup.RESULTS_DIR", source),
            patch("leaderboards.backup.BACKUPS_DIR", backups_dir),
            patch("leaderboards.backup.BACKUPS_MAX_BYTES", 1024 * 1024 * 10),
        ):
            second_backup = _write_snapshot(source=source)

        assert second_backup is None  # Skipped because hash matches
