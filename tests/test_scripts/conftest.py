"""Shared test adapters for canonical publication."""

from __future__ import annotations

import pytest

from leaderboards.canonical_publication import PublicationError, PublicationLock


@pytest.fixture(autouse=True)
def use_in_memory_publication_lock(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Prevent broker network calls while retaining lock enforcement."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    if request.node.name == "test_missing_lock_configuration_fails_closed":
        return
    monkeypatch.setattr(
        PublicationLock,
        "from_environment",
        classmethod(lambda cls, bucket="EuroEval/results": InMemoryPublicationLock()),
    )


class InMemoryPublicationLock:
    """Deterministic lock adapter that exercises owned-session checks."""

    bucket = "EuroEval/results"
    endpoint = "memory://publication-lock"

    def __init__(self) -> None:
        """Initialise an unlocked adapter."""
        self.owner: str | None = None

    def request(self, action: str, owner: str) -> str:
        """Apply one lock operation.

        Args:
            action: Requested lock operation.
            owner: Opaque owner identifier.

        Returns:
            The resulting lock state.

        Raises:
            PublicationError: If a different owner holds the lock.
        """
        if action == "acquire":
            if self.owner not in {None, owner}:
                raise PublicationError("busy")
            self.owner = owner
            return "owned"
        if action == "assert":
            if self.owner != owner:
                raise PublicationError("busy")
            return "owned"
        if action == "release":
            if self.owner not in {None, owner}:
                raise PublicationError("busy")
            self.owner = None
            return "released"
        return "owned" if self.owner == owner else "unlocked"
