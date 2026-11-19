"""Tests for the result loading module."""

from __future__ import annotations

import collections.abc as c
import itertools
import json
from pathlib import Path

import pytest

from leaderboards import result_loading
from leaderboards.result_identity import UnresolvedResultConflictError
from leaderboards.result_loading import (
    _dedup_by_storage_identity,
    configure_leaderboard_result_filter,
    load_raw_results,
    reset_leaderboard_result_filter,
)


class TestDedupByStorageIdentity:
    """Tests for _dedup_by_storage_identity."""

    def test_empty_list_returns_empty(self) -> None:
        """Should return empty list for empty input."""
        deduped = _dedup_by_storage_identity(records=[])
        assert deduped == []

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

    def test_removes_duplicate_identities(self) -> None:
        """Should keep only one record per storage identity."""
        record_a = _make_eee_record(version="1.0.0", timestamp="100")
        record_b = _make_eee_record(version="1.0.1", timestamp="200")

        deduped = _dedup_by_storage_identity(records=[record_a, record_b])

        assert len(deduped) == 1
        assert deduped[0]["eval_library"]["version"] == "1.0.0"

    def test_uses_timestamp_tiebreaker(self) -> None:
        """Should use timestamp as tiebreaker when versions are equal."""
        earlier = _make_eee_record(version="1.0.0", timestamp="100")
        later = _make_eee_record(version="1.0.0", timestamp="200")

        deduped = _dedup_by_storage_identity(records=[earlier, later])

        assert len(deduped) == 1
        assert deduped[0]["retrieved_timestamp"] == "100"

    def test_version_alone_does_not_choose_a_winner(self) -> None:
        """A library version change without newer evaluation evidence is unresolved."""
        older = _make_eee_record(version="1.0.0", timestamp="100")
        newer = _make_eee_record(version="1.1.0", timestamp="100")

        assert _dedup_by_storage_identity(records=[older, newer]) == [older]

        changed_score = _make_eee_record(version="1.1.0", timestamp="100", score=0.7)
        with pytest.raises(UnresolvedResultConflictError):
            _dedup_by_storage_identity(records=[older, changed_score])


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
        model_id (optional):
            Model identifier. Defaults to "test/model".
        dataset (optional):
            Dataset name. Defaults to "dataset1".
        validation_split (optional):
            Validation split flag. Defaults to False.
        few_shot (optional):
            Few-shot flag. Defaults to False.
        version (optional):
            EuroEval version. Defaults to "1.0.0".
        timestamp (optional):
            Retrieved timestamp as string integer. Defaults to "100".
        score (optional):
            Test score. Defaults to 0.5.

    Returns:
        A valid EEE-format record dict.
    """
    return {
        "schema_version": "1.0",
        "model_info": {
            "id": model_id,
            "additional_details": {
                "commercially_licensed": False,
                "open": True,
                "trained_from_scratch": False,
            },
        },
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


class TestLoadRawResults:
    """Tests for load_raw_results."""

    @pytest.fixture(autouse=True)
    def _clear_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> c.Generator[None]:
        """Isolate staging, filters, and cached results between tests.

        Args:
            tmp_path:
                Temporary directory for the staging input.
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.

        Yields:
            None
        """
        monkeypatch.setattr(
            "leaderboards.result_loading.NEW_RESULTS_PATH",
            tmp_path / "new_results.jsonl",
        )
        load_raw_results.cache_clear()
        reset_leaderboard_result_filter()
        yield
        load_raw_results.cache_clear()
        reset_leaderboard_result_filter()

    def test_canary_conflicts_are_not_hidden_by_storage_deduplication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Private scoring receives every candidate for an auxiliary identity.

        Args:
            tmp_path:
                Temporary directory for files created by the test.
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        first = _make_eee_record(dataset="contamination-canary-da", timestamp="100")
        second = _make_eee_record(dataset="contamination-canary-da", timestamp="200")
        first["eval_library"]["additional_details"]["contamination_canary_evidence"] = (
            "first"
        )
        second["eval_library"]["additional_details"][
            "contamination_canary_evidence"
        ] = "second"
        (model_dir / "first.json").write_text(json.dumps(first))
        (model_dir / "second.json").write_text(json.dumps(second))

        loaded = load_raw_results()

        assert len(loaded) == 2

    def test_dedups_duplicate_identities(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should deduplicate records with same identity, keeping newest.

        Args:
            tmp_path:
                Temporary directory for files created by the test.
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)
        staging = result_loading.NEW_RESULTS_PATH

        # Create tree record
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        older = _make_eee_record(
            model_id="test/model", version="1.0.0", timestamp="100", score=0.4
        )
        older["eval_timestamp"] = 100
        (model_dir / "dataset1__test__zeroshot.json").write_text(json.dumps(older))

        # Newer recorded evaluation evidence wins on its changed score.
        newer = _make_eee_record(
            model_id="test/model", version="1.1.0", timestamp="200", score=0.5
        )
        newer["eval_timestamp"] = 200
        staging.write_text(json.dumps(newer) + "\n")

        try:
            records = load_raw_results()

            assert len(records) == 1
            assert records[0]["eval_library"]["version"] == "1.1.0"
        finally:
            if staging.exists():
                staging.unlink()

    def test_leaderboard_filter_hides_auxiliary_and_selected_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retain source files while hiding them from leaderboard consumers.

        Args:
            tmp_path:
                Temporary directory for files created by the test.
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        records = [
            _make_eee_record(model_id="org/excluded@revision", dataset="task-one"),
            _make_eee_record(model_id="org/kept", dataset="task-two"),
            _make_eee_record(
                model_id="org/excluded@revision", dataset="contamination-canary-da"
            ),
        ]
        for index, record in enumerate(records):
            (model_dir / f"record-{index}.json").write_text(json.dumps(record))
        configure_leaderboard_result_filter(excluded_models={"org/excluded"})
        try:
            filtered = load_raw_results()
        finally:
            reset_leaderboard_result_filter()

        assert [item["model_info"]["id"] for item in filtered] == ["org/kept"]
        assert len(list(model_dir.glob("*.json"))) == 3

    def test_loads_from_tree_structure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should load records from the tree structure.

        Args:
            tmp_path:
                Temporary directory for files created by the test.
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        # Mock download_missing_bucket_files to do nothing
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
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
        """Should also load records from NEW_RESULTS_PATH staging file.

        Args:
            tmp_path:
                Temporary directory for files created by the test.
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
        """
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)
        staging = result_loading.NEW_RESULTS_PATH

        # Create tree record
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        tree_record = _make_eee_record(model_id="tree/model", dataset="dataset1")
        (model_dir / "dataset1__test__zeroshot.json").write_text(
            json.dumps(tree_record)
        )

        # Create staging JSONL file
        staging_record = _make_eee_record(model_id="staging/model", dataset="dataset2")
        staging.write_text(json.dumps(staging_record) + "\n")

        try:
            records = load_raw_results()

            # Should have both tree and staging records
            assert len(records) == 2
            model_ids = {r["model_info"]["id"] for r in records}
            assert model_ids == {"tree/model", "staging/model"}

            # Staging file should be removed
            assert not staging.exists()
        finally:
            # Clean up staging file if test fails
            if staging.exists():
                staging.unlink()

    def test_raises_on_no_results_after_sync(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Should raise FileNotFoundError if sync produces no files.

        Args:
            monkeypatch:
                Pytest fixture for replacing dependencies during the test.
            tmp_path:
                Temporary directory for files created by the test.
        """
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr("leaderboards.result_loading.RESULTS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="No results available"):
            load_raw_results()

    def test_staging_file_remains_when_identity_conflict_is_unresolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolved conflict cannot consume its staged source file."""
        monkeypatch.setattr(
            "leaderboards.result_loading.download_missing_bucket_files", lambda: 0
        )
        monkeypatch.setattr("leaderboards.result_loading.backup_results", lambda: None)
        monkeypatch.setattr(
            "leaderboards.result_loading.RESULTS_DIR", tmp_path / "tree"
        )
        staging = tmp_path / "new-results.jsonl"
        monkeypatch.setattr("leaderboards.result_loading.NEW_RESULTS_PATH", staging)
        existing = _make_eee_record(timestamp="100", score=0.5)
        incoming = _make_eee_record(timestamp="100", score=0.7)
        model_dir = tmp_path / "tree" / "test_model"
        model_dir.mkdir(parents=True)
        (model_dir / "existing.json").write_text(json.dumps(existing))
        staging.write_text(json.dumps(incoming) + "\n")

        with pytest.raises(UnresolvedResultConflictError):
            result_loading.load_raw_results.__wrapped__()

        assert staging.exists()


def test_storage_dedup_keeps_known_metadata_superset_in_both_orders() -> None:
    """Compatible metadata enrichment wins regardless of input order."""
    richer = _make_eee_record()
    richer["model_info"]["additional_details"] = {
        "commercially_licensed": True,
        "open": True,
        "trained_from_scratch": False,
    }
    richer["model_info"]["additional_details"].update(
        {
            "commercially_licensed": False,
            "open": True,
            "trained_from_scratch": False,
            "model_parameters": 8_030_000_000,
        }
    )
    poorer = _make_eee_record()
    poorer["model_info"]["additional_details"]["model_parameters"] = -1

    for records in itertools.permutations([richer, poorer]):
        assert _dedup_by_storage_identity(records=list(records)) == [richer]


def test_storage_dedup_raises_for_ambiguous_score_conflict() -> None:
    """Conflicting score and metadata evidence is not silently discarded."""
    current = _make_eee_record(timestamp="100", score=0.5)
    incoming = _make_eee_record(timestamp="100", score=0.7)
    current["model_info"]["additional_details"]["open"] = True

    with pytest.raises(UnresolvedResultConflictError):
        _dedup_by_storage_identity(records=[current, incoming])
