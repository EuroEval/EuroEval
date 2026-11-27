"""Tests for local exclusion of canonical publication sessions."""

from __future__ import annotations

from pathlib import Path

import pytest

from leaderboards.canonical_publication import CanonicalPublisher, PublicationError


def test_sessions_exclude_same_bucket_endpoint_until_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent publishers cannot overlap sessions, and locks release on exit.

    Raises:
        RuntimeError:
            Used to verify lock release after a failed session.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    first_api = CountingApi()
    second_api = CountingApi()
    first = CanonicalPublisher(
        api=first_api,
        token="first-token",
        bucket="org/results",
        checkpoint_path=tmp_path / "first.sqlite3",
    )
    second = CanonicalPublisher(
        api=second_api,
        token="second-token",
        bucket="org/results",
        checkpoint_path=tmp_path / "second.sqlite3",
    )

    with first.session():
        with pytest.raises(PublicationError):
            with second.session():
                pytest.fail("overlapping publication session opened")
    assert second_api.reads == second_api.writes == 0

    with second.session():
        pass

    with pytest.raises(RuntimeError):
        with first.session():
            raise RuntimeError("session failed")
    with second.session():
        pass


class CountingApi:
    """Minimal bucket adapter that records remote operations."""

    endpoint = "https://hf.example"

    def __init__(self) -> None:
        """Initialise request accounting."""
        self.reads = 0
        self.writes = 0

    def batch_bucket_files(self, *args: object, **kwargs: object) -> None:
        """Record a bucket write."""
        self.writes += 1

    def download_bucket_files(self, *args: object, **kwargs: object) -> None:
        """Record a bucket read."""
        self.reads += 1
