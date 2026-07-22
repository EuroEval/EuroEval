"""Tests for the `leaderboards.result_processing` module."""

import json
import typing as t
from enum import Enum, auto
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import HfHubHTTPError

from leaderboards import bucket_sync, result_processing
from leaderboards.cache import Cache


class CallOrder(Enum):
    """Enum to track call order of functions."""

    NONE = auto()
    LOAD_RAW_RESULTS = auto()
    CACHE_FROM_RESULTS_DIR = auto()


def test_cache_freshness_load_before_cache_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify that load_raw_results is called before Cache.from_results_dir.

    This ensures the results bucket is synced before cache construction,
    preventing stale cache issues. The test fails if cache construction
    happens before loading/syncing completes.
    """
    call_sequence: list[CallOrder] = []

    def fake_load_raw_results() -> list[dict[str, t.Any]]:
        call_sequence.append(CallOrder.LOAD_RAW_RESULTS)
        return []

    # Attach cache_clear to mimic functools.cached_function
    fake_load_raw_results.cache_clear = lambda: None  # type: ignore[attr-defined]

    def fake_cache_from_results_dir(results_dir: Path) -> Cache:
        call_sequence.append(CallOrder.CACHE_FROM_RESULTS_DIR)
        # Return a minimal Cache with empty dicts - the actual cache
        # is used for metadata lookups in add_missing_entries
        return Cache()

    monkeypatch.setattr(result_processing, "load_raw_results", fake_load_raw_results)
    monkeypatch.setattr(Cache, "from_results_dir", fake_cache_from_results_dir)
    monkeypatch.setattr(
        result_processing, "_upload_per_model_files", lambda processed_records: None
    )

    result_processing.process_results(
        min_version="0.0.0",
        min_number_of_model_records=0,
        banned_versions=[],
        banned_model_patterns=[],
        api_model_patterns=[],
        trained_from_scratch_patterns=[],
    )

    assert len(call_sequence) == 2, (
        f"Expected 2 calls, got {len(call_sequence)}: {call_sequence}"
    )
    assert call_sequence[0] == CallOrder.LOAD_RAW_RESULTS, (
        f"load_raw_results must be called first, but got: {call_sequence[0]}"
    )
    assert call_sequence[1] == CallOrder.CACHE_FROM_RESULTS_DIR, (
        f"Cache.from_results_dir must be called second, but got: {call_sequence[1]}"
    )


def test_upload_per_model_files_raises_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files raises without HF_TOKEN set.

    RuntimeError is raised when HF_TOKEN is missing.
    """
    # Patch RESULTS_DIR to use a temp directory
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    # Patch resolve_hf_token to return None (simulating missing HF_TOKEN)
    with patch("leaderboards.result_processing.resolve_hf_token", return_value=None):
        with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
            result_processing._upload_per_model_files(processed_records=[])


def test_upload_per_model_files_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files passes the resolved HF token."""
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    mock_api = MagicMock()
    record: dict[str, t.Any] = {
        "model_info": {
            "id": "org/model",
            "name": "org/model",
            "additional_details": {},
        },
        "dataset": {"name": "dataset"},
        "task": {"name": "classification"},
        "eval_library": {
            "name": "euroeval",
            "version": "1.0.0",
            "additional_details": {
                "dataset": "dataset",
                "few_shot": False,
                "validation_split": False,
            },
        },
        "results": {},
    }

    with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.result_processing.resolve_hf_token", return_value="test_token"
        ):
            result_processing._upload_per_model_files(processed_records=[record])

    mock_api.batch_bucket_files.assert_called_once()
    call_kwargs = mock_api.batch_bucket_files.call_args.kwargs
    assert call_kwargs["bucket_id"] == "EuroEval/results"
    assert call_kwargs["token"] == "test_token"


def test_upload_per_model_files_raises_on_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files raises when bucket sync fails."""
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    mock_api = MagicMock()
    mock_api.batch_bucket_files.side_effect = HfHubHTTPError(
        "Bucket sync failed", response=MagicMock(status_code=500)
    )

    with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.result_processing.resolve_hf_token", return_value="test_token"
        ):
            with pytest.raises(HfHubHTTPError, match="Bucket sync failed"):
                result_processing._upload_per_model_files(processed_records=[])


def test_sync_bucket_raises_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that sync_bucket raises without HF_TOKEN set."""
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path)

    with patch("leaderboards.bucket_sync.resolve_hf_token", return_value=None):
        with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
            bucket_sync.sync_bucket()


def test_sync_bucket_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that sync_bucket passes the resolved HF token."""
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", tmp_path)
    mock_api = MagicMock()

    with patch("leaderboards.bucket_sync.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.bucket_sync.resolve_hf_token", return_value="test_token"
        ):
            bucket_sync.sync_bucket()

    mock_api.sync_bucket.assert_called_once_with(
        source="hf://buckets/EuroEval/results/",
        dest=str(tmp_path),
        token="test_token",
        ignore_times=True,
        ignore_sizes=False,
    )


def test_upload_results_to_bucket_raises_without_hf_token(tmp_path: Path) -> None:
    """Test that upload_results_to_bucket raises without HF_TOKEN set."""
    results_file = tmp_path / "results.jsonl"
    results_file.write_text("{}\n", encoding="utf-8")

    with patch("leaderboards.bucket_sync.resolve_hf_token", return_value=None):
        with pytest.raises(RuntimeError, match="HF_TOKEN not set"):
            bucket_sync.upload_results_to_bucket(results_file=results_file)


def test_upload_results_to_bucket_passes_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that upload_results_to_bucket passes the resolved HF token."""
    results_dir = tmp_path / "results"
    results_file = tmp_path / "results.jsonl"

    # Write valid record in EEE format
    valid_record = {
        "model_info": {"id": "org/model"},
        "eval_library": {
            "additional_details": {"dataset": "test_ds"},
            "version": "1.0.0",
        },
        "retrieved_timestamp": "2024-01-01T00:00:00Z",
    }
    results_file.write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

    mock_api = MagicMock()

    with patch("leaderboards.bucket_sync.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.bucket_sync.resolve_hf_token", return_value="test_token"
        ):
            bucket_sync.upload_results_to_bucket(results_file=results_file)

    mock_api.batch_bucket_files.assert_called_once()
    call_kwargs = mock_api.batch_bucket_files.call_args.kwargs
    assert call_kwargs["bucket_id"] == "EuroEval/results"
    assert call_kwargs["token"] == "test_token"


def test_upload_results_to_bucket_raises_on_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that upload_results_to_bucket raises when bucket sync fails."""
    results_dir = tmp_path / "results"
    results_file = tmp_path / "results.jsonl"

    # Write valid record in EEE format
    valid_record = {
        "model_info": {"id": "org/model"},
        "eval_library": {
            "additional_details": {"dataset": "test_ds"},
            "version": "1.0.0",
        },
        "retrieved_timestamp": "2024-01-01T00:00:00Z",
    }
    results_file.write_text(json.dumps(valid_record) + "\n", encoding="utf-8")
    monkeypatch.setattr(bucket_sync, "RESULTS_DIR", results_dir)

    mock_api = MagicMock()
    mock_api.batch_bucket_files.side_effect = HfHubHTTPError(
        "Bucket sync failed", response=MagicMock(status_code=500)
    )

    with patch("leaderboards.bucket_sync.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.bucket_sync.resolve_hf_token", return_value="test_token"
        ):
            with pytest.raises(HfHubHTTPError, match="Bucket sync failed"):
                bucket_sync.upload_results_to_bucket(results_file=results_file)


def test_process_results_clears_cache_after_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that load_raw_results cache is cleared after processing."""
    # Track whether cache_clear was called
    cache_clear_called = False

    def fake_cache_clear() -> None:
        nonlocal cache_clear_called
        cache_clear_called = True

    def fake_load_raw_results() -> list[dict[str, t.Any]]:
        return []

    # Attach cache_clear to mimic functools.cached_function
    fake_load_raw_results.cache_clear = fake_cache_clear  # type: ignore[attr-defined]

    def fake_cache_from_results_dir(results_dir: Path) -> Cache:
        return Cache()

    monkeypatch.setattr(result_processing, "load_raw_results", fake_load_raw_results)
    monkeypatch.setattr(Cache, "from_results_dir", fake_cache_from_results_dir)
    monkeypatch.setattr(
        result_processing, "_upload_per_model_files", lambda processed_records: None
    )

    result_processing.process_results(
        min_version="0.0.0",
        min_number_of_model_records=0,
        banned_versions=[],
        banned_model_patterns=[],
        api_model_patterns=[],
        trained_from_scratch_patterns=[],
    )

    assert cache_clear_called, (
        "load_raw_results.cache_clear() should be called after upload"
    )


class TestUploadPerModelFilesPerRecordTree:
    """Tests for the per-record JSON tree storage format."""

    def test_writes_expected_directory_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that _upload_per_model_files writes the expected directory structure."""
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()
        record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/Qwen3-0.6B",
                "name": "org/Qwen3-0.6B",
                "additional_details": {},
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                result_processing._upload_per_model_files(processed_records=[record])

        # Check directory structure
        model_dir = tmp_path / "org_Qwen3-0.6B"
        assert model_dir.exists(), "Model directory should be created"

        # Check file exists with expected name
        result_file = model_dir / "mmlu__test__zeroshot.json"
        assert result_file.exists(), "Result file should be created"

        # Check content is valid JSON with metadata preserved
        content = result_file.read_text(encoding="utf-8")
        written_record = json.loads(content)
        assert written_record["model_info"]["id"] == "org/Qwen3-0.6B"
        assert "additional_details" in written_record["model_info"]

    def test_older_duplicate_does_not_clobber_newer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that when two records have the same identity, the newer is kept."""
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()

        # Older record (version 1.0.0)
        older_record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 50},
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",  # Older version
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        # Newer record (version 2.0.0)
        newer_record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 90},
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "2.0.0",  # Newer version
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                result_processing._upload_per_model_files(
                    processed_records=[older_record, newer_record]
                )

        # Only one file should exist
        model_dir = tmp_path / "org_model"
        result_file = model_dir / "mmlu__test__zeroshot.json"
        assert result_file.exists()

        # The newer record should be kept (score=90)
        content = result_file.read_text(encoding="utf-8")
        written_record = json.loads(content)
        assert written_record["model_info"]["additional_details"]["score"] == 90

    def test_path_collision_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that distinct identities mapping to same path raise clear error.

        Dataset names 'a/b' and 'a_b' both sanitise to 'a_b', causing a path
        collision. This should raise ValueError with a message naming the
        collision before any filesystem mutation occurs.
        """
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()

        # Record with dataset 'a/b' -> sanitises to 'a_b'
        record_a: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 50},
            },
            "dataset": {"name": "a/b"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "a/b",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        # Record with dataset 'a_b' -> also sanitises to 'a_b'
        record_b: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {"score": 90},
            },
            "dataset": {"name": "a_b"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "a_b",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                with pytest.raises(ValueError, match="Identity collision detected"):
                    result_processing._upload_per_model_files(
                        processed_records=[record_a, record_b]
                    )

        # No files should be written on collision
        assert list(tmp_path.iterdir()) == []

    def test_invalid_model_record_dropped_with_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that records with unresolvable model identities are dropped."""
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()

        # Invalid record (missing model_info.id and name)
        invalid_record: dict[str, t.Any] = {
            "model_info": {"additional_details": {}},
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                with caplog.at_level("WARNING"):
                    result_processing._upload_per_model_files(
                        processed_records=[invalid_record]
                    )

        # No files should be written
        assert list(tmp_path.iterdir()) == []

        # Should log a warning
        assert "unresolvable identity" in caplog.text.lower()

    def test_metadata_preserved_in_written_record(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that all metadata fields are preserved in the written JSON."""
        monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

        mock_api = MagicMock()
        record: dict[str, t.Any] = {
            "model_info": {
                "id": "org/model",
                "name": "org/model",
                "additional_details": {
                    "generative": True,
                    "generative_type": "chat",
                    "merge": False,
                    "open": True,
                    "commercially_licensed": False,
                    "trained_from_scratch": True,
                    "model_url": "https://huggingface.co/org/model",
                },
            },
            "dataset": {"name": "mmlu"},
            "task": {"name": "classification"},
            "eval_library": {
                "name": "euroeval",
                "version": "1.0.0",
                "additional_details": {
                    "dataset": "mmlu",
                    "few_shot": False,
                    "validation_split": False,
                },
            },
            "results": {},
        }

        with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
            with patch(
                "leaderboards.result_processing.resolve_hf_token",
                return_value="test_token",
            ):
                result_processing._upload_per_model_files(processed_records=[record])

        model_dir = tmp_path / "org_model"
        result_file = model_dir / "mmlu__test__zeroshot.json"
        assert result_file.exists()

        content = result_file.read_text(encoding="utf-8")
        written_record = json.loads(content)

        # Verify metadata is preserved
        assert written_record["model_info"]["additional_details"]["generative"] is True
        assert (
            written_record["model_info"]["additional_details"]["generative_type"]
            == "chat"
        )
        assert written_record["model_info"]["additional_details"]["merge"] is False
        assert written_record["model_info"]["additional_details"]["open"] is True
        assert (
            written_record["model_info"]["additional_details"]["commercially_licensed"]
            is False
        )
        assert (
            written_record["model_info"]["additional_details"]["trained_from_scratch"]
            is True
        )
