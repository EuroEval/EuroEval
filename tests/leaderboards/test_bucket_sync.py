"""Tests for the ``leaderboards.bucket_sync`` module."""

from __future__ import annotations

import json
import typing as t
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from leaderboards import bucket_sync


class TestSyncBucket:
    """Tests for ``sync_bucket``."""

    def test_sync_bucket_raises_without_hf_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sync_bucket raises RuntimeError when HF_TOKEN is missing."""
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value=None):
            with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
                bucket_sync.sync_bucket()

    def test_sync_bucket_preserves_local_only_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sync_bucket preserves local-only record files after sync.

        In the new per-record JSON tree model, each logical result has a unique
        path, so sync should not remove local files. This test verifies the
        union semantics are preserved.
        """
        # Create a temporary results directory
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Create a local-only record file
        model_dir = results_dir / "test_model"
        model_dir.mkdir()
        local_record = {
            "model_info": {"id": "test/model"},
            "eval_library": {"additional_details": {"dataset": "test_ds"}},
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        local_file = model_dir / "test_ds__none__none.json"
        local_file.write_text(json.dumps(local_record), encoding="utf-8")

        # Mock HfApi.sync_bucket to simulate bucket sync that removes the file
        def mock_sync(
            source: str,
            dest: str,
            token: str | None = None,
            ignore_times: bool = False,
            **kwargs: t.Any,
        ) -> None:
            # Simulate sync removing the local file
            if local_file.exists():
                local_file.unlink()

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="test"):
            with patch.object(HfApi, "sync_bucket", side_effect=mock_sync):
                # This should not raise, but should log a warning
                bucket_sync.sync_bucket()

        # The file should be restored since we keep content in memory
        assert local_file.exists()
        restored_record = json.loads(local_file.read_text(encoding="utf-8"))
        assert restored_record["model_info"]["id"] == "test/model"

    def test_sync_bucket_calls_hf_api_with_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sync_bucket calls HfApi with the resolved token."""
        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with (
            patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token123"),
            patch.object(HfApi, "sync_bucket", return_value=None) as mock_sync,
        ):
            bucket_sync.sync_bucket()

        mock_sync.assert_called_once_with(
            source="hf://buckets/test/results/",
            dest=str(tmp_path),
            token="token123",
            ignore_times=True,
            ignore_sizes=False,
        )

    def test_sync_bucket_raises_on_sync_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sync_bucket raises HfHubHTTPError on sync failure."""
        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token"):
            with patch.object(
                HfApi,
                "sync_bucket",
                side_effect=HfHubHTTPError(
                    "Sync failed", response=MagicMock(status_code=500)
                ),
            ):
                with pytest.raises(HfHubHTTPError, match="Sync failed"):
                    bucket_sync.sync_bucket()


class TestMergeResults:
    """Tests for ``merge_results``."""

    def test_merge_results_reads_json_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that merge_results reads from the JSON tree structure."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = tmp_path / "merged.jsonl"

        # Create record files in tree structure
        model_dir = results_dir / "test_model"
        model_dir.mkdir()

        record1 = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "dataset1"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        record2 = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "dataset2"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }

        (model_dir / "dataset1__test__test.json").write_text(
            json.dumps(record1), encoding="utf-8"
        )
        (model_dir / "dataset2__test__test.json").write_text(
            json.dumps(record2), encoding="utf-8"
        )

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

        count = bucket_sync.merge_results(results_file=results_file)

        assert count == 2
        lines = results_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_merge_results_dedups_by_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that merge_results deduplicates by canonical result identity."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = tmp_path / "merged.jsonl"

        model_dir = results_dir / "test_model"
        model_dir.mkdir()

        # Record with older version
        record_old = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "dataset1"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        record1_path = model_dir / "dataset1__test__test.json"
        record1_path.write_text(json.dumps(record_old), encoding="utf-8")

        # Second model dir with same identity but different version
        model_dir2 = results_dir / "test_model_2"
        model_dir2.mkdir()
        record_same_identity = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "dataset1"},
                "version": "1.5.0",
            },
            "retrieved_timestamp": "2024-01-01T12:00:00Z",
            "extra_field": "should_be_deduped",
        }
        (model_dir2 / "dataset1__test__test.json").write_text(
            json.dumps(record_same_identity), encoding="utf-8"
        )

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

        count = bucket_sync.merge_results(results_file=results_file)

        # Should deduplicate to 1 record (version 1.5.0 wins over 1.0.0)
        assert count == 1
        lines = results_file.read_text(encoding="utf-8").splitlines()
        merged_record = json.loads(lines[0])
        assert merged_record["eval_library"]["version"] == "1.5.0"

    def test_merge_results_handles_empty_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that merge_results handles empty results directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = tmp_path / "merged.jsonl"

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

        count = bucket_sync.merge_results(results_file=results_file)

        assert count == 0
        assert not results_file.exists()

    def test_merge_results_skips_invalid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that merge_results skips files with invalid JSON."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_file = tmp_path / "merged.jsonl"

        model_dir = results_dir / "test_model"
        model_dir.mkdir()

        # Invalid JSON file
        (model_dir / "bad__test__test.json").write_text(
            "not valid json", encoding="utf-8"
        )

        # Valid file
        valid_record = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "good"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        (model_dir / "good__test__test.json").write_text(
            json.dumps(valid_record), encoding="utf-8"
        )

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

        count = bucket_sync.merge_results(results_file=results_file)

        assert count == 1


class TestUploadResultsToBucket:
    """Tests for ``upload_results_to_bucket``."""

    def test_upload_results_to_bucket_raises_without_hf_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that upload_results_to_bucket raises without HF_TOKEN."""
        results_file = tmp_path / "results.jsonl"
        results_file.write_text("{}\n", encoding="utf-8")

        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.delenv("HF_TOKEN", raising=False)

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value=None):
            with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
                bucket_sync.upload_results_to_bucket(results_file=results_file)

    def test_upload_results_to_bucket_creates_tree_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that upload_results_to_bucket creates the JSON tree layout."""
        results_dir = tmp_path / "results"
        results_file = tmp_path / "results.jsonl"

        record = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        results_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token"):
            with patch.object(HfApi, "batch_bucket_files", return_value=None):
                bucket_sync.upload_results_to_bucket(results_file=results_file)

        # Verify tree layout was created
        expected_path = results_dir / "test_model" / "test_ds__none__none.json"
        assert expected_path.exists()
        uploaded_record = json.loads(expected_path.read_text(encoding="utf-8"))
        assert uploaded_record["model_info"]["id"] == "test/model"

    def test_upload_results_to_bucket_drops_invalid_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that upload_results_to_bucket drops records without valid model."""
        results_dir = tmp_path / "results"
        results_file = tmp_path / "results.jsonl"

        # Record without model
        invalid_record = {
            "model_info": {},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        valid_record = {
            "model_info": {"id": "valid/model"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        results_file.write_text(
            json.dumps(invalid_record) + "\n" + json.dumps(valid_record) + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token"):
            with patch.object(HfApi, "batch_bucket_files", return_value=None):
                bucket_sync.upload_results_to_bucket(results_file=results_file)

        # Only valid record should be uploaded
        valid_path = results_dir / "valid_model" / "test_ds__none__none.json"
        assert valid_path.exists()

        # Invalid record should not create a file
        assert len(list(results_dir.rglob("*.json"))) == 1

    def test_upload_results_to_bucket_raises_on_sync_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that upload_results_to_bucket raises on sync failure."""
        results_dir = tmp_path / "results"
        results_file = tmp_path / "results.jsonl"

        record = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        results_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token"):
            with patch.object(
                HfApi,
                "batch_bucket_files",
                side_effect=HfHubHTTPError(
                    "Upload failed", response=MagicMock(status_code=500)
                ),
            ):
                with pytest.raises(HfHubHTTPError, match="Upload failed"):
                    bucket_sync.upload_results_to_bucket(results_file=results_file)

        # Tree should still be created, but sync fails
        expected_path = results_dir / "test_model" / "test_ds__none__none.json"
        assert expected_path.exists()

    def test_upload_results_to_bucket_warns_on_missing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that upload_results_to_bucket warns when results file doesn't exist."""
        results_file = tmp_path / "nonexistent.jsonl"

        with caplog.at_level("WARNING"):
            bucket_sync.upload_results_to_bucket(results_file=results_file)

        assert "does not exist" in caplog.text

    def test_sync_bucket_restores_local_only_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that sync_bucket restores a local-only file that was removed by sync."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Create a local-only record file
        model_dir = results_dir / "local_model"
        model_dir.mkdir()
        local_record = {
            "model_info": {"id": "local/only"},
            "eval_library": {
                "additional_details": {"dataset": "local_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-02T00:00:00Z",
        }
        local_file = model_dir / "local_ds__none__none.json"
        local_file.write_text(json.dumps(local_record), encoding="utf-8")

        # Mock sync that removes the local file
        def mock_sync(
            source: str,
            dest: str,
            token: str | None = None,
            ignore_times: bool = False,
            **kwargs: t.Any,
        ) -> None:
            if local_file.exists():
                local_file.unlink()

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="test"):
            with patch.object(HfApi, "sync_bucket", side_effect=mock_sync):
                bucket_sync.sync_bucket()

        # File should be restored
        assert local_file.exists()
        restored = json.loads(local_file.read_text(encoding="utf-8"))
        assert restored["model_info"]["id"] == "local/only"

    def test_sync_bucket_locally_newer_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that locally-newer same-identity wins over older bucket."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Local record with newer timestamp
        model_dir = results_dir / "test_model"
        model_dir.mkdir()
        local_record = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-02T00:00:00Z",  # Newer
        }
        local_file = model_dir / "test_ds__none__none.json"
        local_file.write_text(json.dumps(local_record), encoding="utf-8")

        # Mock sync that puts an older bucket record
        def mock_sync(
            source: str,
            dest: str,
            token: str | None = None,
            ignore_times: bool = False,
            **kwargs: t.Any,
        ) -> None:
            # Create an older bucket record
            older_record = {
                "model_info": {"id": "test/model"},
                "eval_library": {
                    "additional_details": {"dataset": "test_ds"},
                    "version": "1.0.0",
                },
                "retrieved_timestamp": "2024-01-01T00:00:00Z",  # Older
            }
            local_file.write_text(json.dumps(older_record), encoding="utf-8")

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="test"):
            with patch.object(HfApi, "sync_bucket", side_effect=mock_sync):
                bucket_sync.sync_bucket()

        # Local record should win (newer timestamp)
        final_record = json.loads(local_file.read_text(encoding="utf-8"))
        assert final_record["retrieved_timestamp"] == "2024-01-02T00:00:00Z"

    def test_sync_bucket_collision_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that distinct-identity path collision raises an error."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Local record with identity A: "model/a" sanitises to "model_a"
        model_dir = results_dir / "model_a"
        model_dir.mkdir()
        local_record = {
            "model_info": {"id": "model/a"},  # sanitises to "model_a"
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        local_file = model_dir / "test_ds__none__none.json"
        local_file.write_text(json.dumps(local_record), encoding="utf-8")

        # Mock sync that puts a record with different identity at same path
        def mock_sync(
            source: str,
            dest: str,
            token: str | None = None,
            ignore_times: bool = False,
            **kwargs: t.Any,
        ) -> None:
            # Record with different identity but same sanitised path
            # "model_a" also sanitises to "model_a" (no change)
            bucket_record = {
                "model_info": {"id": "model_a"},  # Different from "model/a"
                "eval_library": {
                    "additional_details": {"dataset": "test_ds"},
                    "version": "1.0.0",
                },
                "retrieved_timestamp": "2024-01-01T00:00:00Z",
            }
            local_file.write_text(json.dumps(bucket_record), encoding="utf-8")

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="test"):
            with patch.object(HfApi, "sync_bucket", side_effect=mock_sync):
                with pytest.raises(ValueError, match="Identity collision detected"):
                    bucket_sync.sync_bucket()

    def test_upload_results_removes_stale_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that upload_results_to_bucket removes stale root *.jsonl files."""
        results_dir = tmp_path / "results"
        results_file = tmp_path / "results.jsonl"

        # Create a stale jsonl file in results dir
        stale_jsonl = results_dir / "old_results.jsonl"
        stale_jsonl.parent.mkdir()
        stale_jsonl.write_text('{"old": true}\n', encoding="utf-8")

        # Create valid record
        record = {
            "model_info": {"id": "test/model"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        results_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token"):
            with patch.object(HfApi, "batch_bucket_files", return_value=None):
                bucket_sync.upload_results_to_bucket(results_file=results_file)

        # Stale jsonl should be removed
        assert not stale_jsonl.exists()
        # But tree layout should be created
        expected_path = results_dir / "test_model" / "test_ds__none__none.json"
        assert expected_path.exists()

    def test_upload_results_to_bucket_raises_on_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that upload_results_to_bucket raises on identity path collision.

        Verifies that on collision, NO filesystem mutation occurs - pre-existing
        files in RESULTS_DIR remain intact.
        """
        results_dir = tmp_path / "results"
        results_file = tmp_path / "results.jsonl"

        # Create a pre-existing file in results dir to verify it's preserved
        pre_existing_model_dir = results_dir / "other_model"
        pre_existing_model_dir.mkdir(parents=True)
        pre_existing_file = pre_existing_model_dir / "other_ds__none__none.json"
        pre_existing_file.write_text(
            json.dumps({"pre_existing": True}), encoding="utf-8"
        )

        # Two records with different identities but same sanitised path
        # "model/a" and "model_a" both sanitise to "model_a" directory
        record_a = {
            "model_info": {"id": "model/a"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        record_b = {
            "model_info": {"id": "model_a"},
            "eval_library": {
                "additional_details": {"dataset": "test_ds"},
                "version": "1.0.0",
            },
            "retrieved_timestamp": "2024-01-01T00:00:00Z",
        }
        results_file.write_text(
            json.dumps(record_a) + "\n" + json.dumps(record_b) + "\n", encoding="utf-8"
        )

        monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(bucket_sync, "HF_RESULTS_BUCKET", "test/results")

        with patch("leaderboards.bucket_sync.resolve_hf_token", return_value="token"):
            with patch.object(HfApi, "sync_bucket", return_value=None):
                with pytest.raises(ValueError, match="Identity collision detected"):
                    bucket_sync.upload_results_to_bucket(results_file=results_file)

        # Verify no mutation occurred: pre-existing file should still be intact
        assert pre_existing_file.exists(), (
            "Pre-existing file should not have been deleted on collision"
        )
        assert json.loads(pre_existing_file.read_text(encoding="utf-8")) == {
            "pre_existing": True
        }
        # The collision path should NOT have been written
        collision_path = results_dir / "model_a" / "test_ds__none__none.json"
        assert not collision_path.exists(), (
            "Collision path should not have been written"
        )
