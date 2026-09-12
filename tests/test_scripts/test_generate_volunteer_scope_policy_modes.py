"""Tests for non-mutating scope policy generation modes."""

import json
from pathlib import Path

import pytest

import src.scripts.generate_volunteer_scope_policy as module


@pytest.fixture
def simple_policy(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Use a small deterministic policy for command mode tests.

    Returns:
        A deterministic policy document.
    """
    policy = {"policy_version": "volunteer-scope/1.0.0", "policies": []}
    monkeypatch.setattr(module, "build_policy", lambda version, pairs: policy)
    monkeypatch.setattr(module, "official_pairs", lambda: set())
    return policy


def test_check_accepts_current_policy_without_writing(
    tmp_path: Path, simple_policy: dict[str, object]
) -> None:
    """Check succeeds and leaves a current output byte-for-byte unchanged."""
    output = tmp_path / "scope-policy.json"
    expected = module.encode_policy(simple_policy)
    output.write_bytes(expected)

    assert module.main(["--output", str(output), "--check", "--version", "1.0.0"]) == 0
    assert output.read_bytes() == expected


def test_check_rejects_stale_and_missing_policy(
    tmp_path: Path, simple_policy: dict[str, object]
) -> None:
    """Check reports both absent and stale generated output without writing."""
    output = tmp_path / "scope-policy.json"
    for existing in (False, True):
        if existing:
            output.write_text("stale\n", encoding="utf-8")
        else:
            output.unlink(missing_ok=True)
        assert (
            module.main(["--output", str(output), "--check", "--version", "1.0.0"]) == 1
        )
        assert not output.exists() or output.read_text(encoding="utf-8") == "stale\n"


def test_dry_run_reports_change_without_writing(
    tmp_path: Path, simple_policy: dict[str, object]
) -> None:
    """Dry-run reports a missing output and does not create it."""
    output = tmp_path / "scope-policy.json"

    assert (
        module.main(["--output", str(output), "--dry-run", "--version", "1.0.0"]) == 0
    )
    assert not output.exists()
    assert json.loads(module.encode_policy(simple_policy)) == simple_policy
