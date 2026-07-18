"""Tests for the `leaderboards.result_processing` module."""

import typing as t
from enum import Enum, auto
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import HfHubHTTPError

from leaderboards import result_processing
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


def test_upload_per_model_files_raises_on_sync_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that _upload_per_model_files raises when bucket sync fails."""
    # Patch RESULTS_DIR to use a temp directory
    monkeypatch.setattr(result_processing, "RESULTS_DIR", tmp_path)

    # Mock HfApi.sync_bucket to raise an error
    mock_api = MagicMock()
    mock_api.sync_bucket.side_effect = HfHubHTTPError(
        "Bucket sync failed", response=MagicMock(status_code=500)
    )

    with patch("leaderboards.result_processing.HfApi", return_value=mock_api):
        with patch(
            "leaderboards.result_processing.resolve_hf_token", return_value="test_token"
        ):
            with pytest.raises(HfHubHTTPError, match="Bucket sync failed"):
                result_processing._upload_per_model_files(processed_records=[])


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
