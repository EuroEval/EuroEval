"""Shared fixtures for the leaderboard tests."""

from __future__ import annotations

import pytest

from leaderboards.canonical_publication import PublicationError, PublicationLock


@pytest.fixture(autouse=True)
def no_real_jottacloud_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly rather than touch a real Jottacloud account.

    Backing up results shells out to an installed ``jotta-cli``, which uploads
    into the Archive configured on the machine running the tests. That is not
    sandboxable and not free: an unstubbed call uploads fixture data to whoever
    happens to be logged in, and waits minutes for jottad to accept it. Tests
    that exercise archiving stub ``subprocess.run`` themselves; every other
    test is protected here.

    Args:
        monkeypatch:
            Pytest fixture for replacing dependencies during the test.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        """Refuse for this test.

        Args:
            args:
                Args supplied to the test.
            kwargs:
                Kwargs supplied to the test.

        Raises:
            AssertionError:
                If the simulated operation fails.
        """
        raise AssertionError(
            "a test invoked the real Jottacloud client; stub "
            "leaderboards.backup.subprocess.run or _archive_offsite"
        )

    monkeypatch.setattr("leaderboards.backup.subprocess.run", refuse)


@pytest.fixture(autouse=True)
def use_in_memory_publication_lock(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Avoid external lock and token dependencies in leaderboard integration tests."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    if request.node.name == "test_missing_lock_configuration_fails_closed":
        return
    monkeypatch.setattr(
        PublicationLock,
        "from_environment",
        classmethod(lambda cls, bucket="EuroEval/results": InMemoryPublicationLock()),
    )


class InMemoryPublicationLock:
    """Deterministic broker adapter for writer integration tests."""

    bucket = "EuroEval/results"
    endpoint = "memory://publication-lock"

    def __init__(self) -> None:
        """Initialise an unlocked adapter."""
        self.owner: str | None = None

    def request(self, action: str, owner: str) -> str:
        """Apply a lock operation.

        Args:
            action: Requested operation.
            owner: Lock owner token.

        Returns:
            The resulting lock state.

        Raises:
            PublicationError: If another owner holds the lock.
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
