"""Tests for the sync_results script."""

from pathlib import Path

import pytest

from src.scripts import sync_results


def test_main_downloads_missing_files_before_merging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that sync_results uses the incremental bucket download path.

    Args:
        tmp_path:
            Temporary directory for files created by the test.
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """
    results_file = tmp_path / "results.jsonl"
    events: list[str] = []

    def fake_download_missing_bucket_files() -> int:
        events.append("download")
        return 2

    def fake_merge_results(results_file: Path) -> int:
        """Fake merge results for this test.

        Args:
            results_file:
                Results file supplied to the test.

        Returns:
            The value returned by the test helper.
        """
        assert results_file == sync_results.RESULTS_FILE
        events.append("merge")
        return 3

    def fake_upload_results_to_bucket(results_file: Path) -> None:
        """Fake upload results to bucket for this test.

        Args:
            results_file:
                Results file supplied to the test.
        """
        assert results_file == sync_results.RESULTS_FILE
        events.append("upload")

    monkeypatch.setattr(sync_results, "RESULTS_FILE", results_file)
    monkeypatch.setattr(
        sync_results,
        "download_missing_bucket_files",
        fake_download_missing_bucket_files,
    )
    monkeypatch.setattr(sync_results, "merge_results", fake_merge_results)
    monkeypatch.setattr(
        sync_results, "upload_results_to_bucket", fake_upload_results_to_bucket
    )

    sync_results.main()

    assert events == ["download", "merge", "upload"]
