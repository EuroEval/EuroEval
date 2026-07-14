"""Tests for the `leaderboards.result_processing` module."""

import typing as t
from enum import Enum, auto
from pathlib import Path

import pytest

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
