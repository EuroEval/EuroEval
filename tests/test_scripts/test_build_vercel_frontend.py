"""Tests for production-versioned Vercel frontend builds."""

import subprocess
from pathlib import Path

import pytest

import src.scripts.build_vercel_frontend as build


def test_build_refreshes_stale_configuration_and_restores_generated_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale local file is refreshed before production policies are generated.

    Args:
        monkeypatch:
            Pytest fixture for replacing subprocess calls and changing directories.
        tmp_path:
            Temporary directory for isolated Vercel configuration and policies.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel/.env.production.local").write_text(
        "PRIVATE=must-not-be-printed\n", encoding="utf-8"
    )
    original = _write_policies(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        """Record commands and emulate a refreshed production config.

        Args:
            command:
                Subprocess command under test.
            check:
                Whether a failed process raises an exception.
            kwargs:
                Additional subprocess options.

        Returns:
            A successful process result.
        """
        commands.append(command)
        if command[:3] == ["vercel", "env", "pull"]:
            Path(command[3]).write_text("EUROEVAL_VERSION=18.4.0\n", encoding="utf-8")
        if "generate_volunteer_scope_policy.py" in command:
            assert command[-1] == "18.4.0"
            for path in build.POLICY_PATHS:
                (tmp_path / path).write_text("production-policy", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main() == 0
    assert commands[0] == [
        "vercel",
        "env",
        "pull",
        ".vercel/.env.production.local",
        "--environment=production",
        "--yes",
    ]
    assert commands[-1] == ["vercel", "build", "--prod", "--yes"]
    assert [(tmp_path / path).read_bytes() for path in build.POLICY_PATHS] == original


def _write_policies(tmp_path: Path) -> list[bytes]:
    """Create isolated policy artifacts and return their original contents.

    Args:
        tmp_path:
            Temporary directory containing the test repository.

    Returns:
        Original policy file contents.
    """
    original = []
    for path in build.POLICY_PATHS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("development-policy", encoding="utf-8")
        original.append(target.read_bytes())
    return original


def test_build_rejects_override_that_differs_from_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit publish version cannot contradict production configuration.

    Args:
        monkeypatch:
            Pytest fixture for replacing subprocess calls and changing directories.
        tmp_path:
            Temporary directory for isolated Vercel configuration.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vercel").mkdir()
    monkeypatch.setenv("EUROEVAL_VERSION", "18.5.0")
    original = _write_policies(tmp_path)

    def fake_run(
        command: list[str], *, check: bool, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        """Write the authoritative production version during pull.

        Args:
            command:
                Subprocess command under test.
            check:
                Whether a failed process raises an exception.
            kwargs:
                Additional subprocess options.

        Returns:
            A successful process result.
        """
        Path(command[3]).write_text("EUROEVAL_VERSION=18.4.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main() == 1
    assert [(tmp_path / path).read_bytes() for path in build.POLICY_PATHS] == original
