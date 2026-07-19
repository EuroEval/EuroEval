"""Tests for the result loading module."""

from __future__ import annotations

import collections.abc as c
import json
from pathlib import Path

import pytest

from leaderboards.constants import NEW_RESULTS_PATH
from leaderboards.result_loading import _dedup_by_storage_identity, load_raw_results


def _make_eee_record(
    model_id: str = "test/model",
    dataset: str = "dataset1",
    validation_split: bool | None = False,
    few_shot: bool | None = False,
    version: str = "1.0.0",
    timestamp: str = "100",
    score: float = 0.5,
) -> dict:
    """Helper to create a valid EEE-format record for testing.

    Args:
        model_id:
            Model identifier.
        dataset:
            Dataset name.
        validation_split:
            Validation split flag.
        few_shot:
            Few-shot flag.
        version:
            EuroEval version.
        timestamp:
            Retrieved timestamp as string integer.
        score:
            Test score.

    Returns:
        A valid EEE-format record dict.
    """
    return {
        "schema_version": "1.0",
        "model_info": {"id": model_id},
        "eval_library": {
            "additional_details": {
                "dataset": dataset,
                "validation_split": validation_split,
                "few_shot": few_shot,
            },
            "version": version,
        },
        "retrieved_timestamp": timestamp,
        "evaluation_results": [{"label": "pass", "score": score}],
    }


class TestDedupByStorageIdentity:
    """Tests for _dedup_by_storage_identity."""

    def test_removes_duplicate_identities(self) -> None:
        """Should keep only one record per storage identity."""
        record_a = _make_eee_record(version="1.0.0", timestamp="100")
        record_b = _make_eee_record(version="1.0.1", timestamp="200")

        deduped = _dedup_by_storage_identity(records=[record_a, record_b])

        assert len(deduped) == 1
        assert deduped[0]["eval_library"]["version"] == "1.0.1"

    def test_keeps_newer_version(self) -> None:
        """Should keep the record with higher version."""
        older = _make_eee_record(version="1.0.0", timestamp="200")
        newer = _make_eee_record(version="1.1.0", timestamp="100")

        deduped = _dedup_by_storage_identity(records=[older, newer])

        assert len(deduped) == 1
        assert deduped[0]["eval_library"]["version"] == "1.1.0"

    def test_uses_timestamp_tiebreaker(self) -> None:
        """Should use timestamp as tiebreaker when versions are equal."""
        earlier = _make_eee_record(version="1.0.0", timestamp="100")
        later = _make_eee_record(version="1.0.0", timestamp="200")

        deduped = _dedup_by_storage_identity(records=[earlier, later])

        assert len(deduped) == 1
        assert deduped[0]["retrieved_timestamp"] == "200"

    def test_keeps_different_identities(self) -> None:
        """Should keep all records with different identities."""
        record_a = _make_eee_record(dataset="dataset1")
        record_b = _make_eee_record(dataset="dataset2")

        deduped = _dedup_by_storage_identity(records=[record_a, record_b])

        assert len(deduped) == 2

    def test_preserves_metadata_in_dedup(self) -> None:
        """Should preserve all metadata fields in the kept record."""
        record = {
            "schema_version": "1.0",
            "model_info": {
                "id": "test/model",
                "additional_details": {
                    "commercially_licensed": True,
                    "open": True,
                    "trained_from_scratch": False,
                    "model_url": "https://example.com",
                },
            },
            "eval_library": {
                "additional_details": {
                    "dataset": "dataset1",
                    "validation_split": False,
                    "few_shot": False,
                },
                "version": "1.0.0",
            },
            "retrieved_timestamp": "100",
            "evaluation_results": [{"label": "pass", "score": 0.5}],
        }

        deduped = _dedup_by_storage_identity(records=[record])

        assert len(deduped) == 1
        details = deduped[0]["model_info"]["additional_details"]
        assert details["commercially_licensed"] is True
        assert details["open"] is True
        assert details["trained_from_scratch"] is False
        assert details["model_url"] == "https://example.com"

    def test_empty_list_returns_empty(self) -> None:
        """Should return empty list for empty input."""
        deduped = _dedup_by_storage_identity(records=[])
        assert deduped == []


class TestLoadRawResults:
    """Tests for load_raw_results."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> c.Generator[None]:
        """Clear the load_raw_results cache between tests.

        Yields:
            None
        """
        load_raw_results.cache_clear()
        yield
        load_raw_results.cache_clear()

    def test_loads_from_tree_structure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should load records from the tree structure."""
        # Mock sync_bucket to do nothing
        monkeypatch.setattr("leaderboards.result_loading.sync_bucket", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)

        # Create tree structure
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()

        record = _make_eee_record(model_id="test/model", dataset="dataset1")

        json_file = model_dir / "dataset1__test__zeroshot.json"
        json_file.write_text(json.dumps(record))

        records = load_raw_results()

        assert len(records) == 1
        assert records[0]["model_info"]["id"] == "test/model"

    def test_loads_staging_jsonl_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should also load records from NEW_RESULTS_PATH staging file."""
        monkeypatch.setattr("leaderboards.result_loading.sync_bucket", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)
        monkeypatch.setattr(
            "leaderboards.result_loading.NEW_RESULTS_PATH", NEW_RESULTS_PATH
        )

        # Create tree record
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        tree_record = _make_eee_record(model_id="tree/model", dataset="dataset1")
        (model_dir / "dataset1__test__zeroshot.json").write_text(
            json.dumps(tree_record)
        )

        # Create staging JSONL file
        staging_record = _make_eee_record(model_id="staging/model", dataset="dataset2")
        NEW_RESULTS_PATH.write_text(json.dumps(staging_record) + "\n")

        try:
            records = load_raw_results()

            # Should have both tree and staging records
            assert len(records) == 2
            model_ids = {r["model_info"]["id"] for r in records}
            assert model_ids == {"tree/model", "staging/model"}

            # Staging file should be removed
            assert not NEW_RESULTS_PATH.exists()
        finally:
            # Clean up staging file if test fails
            if NEW_RESULTS_PATH.exists():
                NEW_RESULTS_PATH.unlink()

    def test_raises_on_no_results_after_sync(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Should raise FileNotFoundError if sync produces no files."""
        monkeypatch.setattr("leaderboards.result_loading.sync_bucket", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="No results available"):
            load_raw_results()

    def test_dedups_duplicate_identities(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should deduplicate records with same identity, keeping newest."""
        monkeypatch.setattr("leaderboards.result_loading.sync_bucket", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)
        monkeypatch.setattr(
            "leaderboards.result_loading.NEW_RESULTS_PATH", NEW_RESULTS_PATH
        )

        # Create tree record
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        older = _make_eee_record(
            model_id="test/model", version="1.0.0", timestamp="100"
        )
        (model_dir / "dataset1__test__zeroshot.json").write_text(json.dumps(older))

        # Create staging JSONL with same identity but newer version
        newer = _make_eee_record(
            model_id="test/model", version="1.1.0", timestamp="200"
        )
        NEW_RESULTS_PATH.write_text(json.dumps(newer) + "\n")

        try:
            records = load_raw_results()

            # Should have deduplicated to 1 record (newer wins)
            assert len(records) == 1
            assert records[0]["eval_library"]["version"] == "1.1.0"
        finally:
            if NEW_RESULTS_PATH.exists():
                NEW_RESULTS_PATH.unlink()
