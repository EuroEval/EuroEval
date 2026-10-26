"""Shared fixtures for the leaderboard tests."""

from __future__ import annotations

import pytest


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
