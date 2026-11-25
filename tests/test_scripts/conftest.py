"""Shared test adapters for canonical publication."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def publication_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep publication tests independent of host credentials.

    Args:
        monkeypatch: Fixture setting a test-only Hugging Face token.
    """
    monkeypatch.setenv("HF_TOKEN", "test-token")
