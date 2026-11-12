"""Tests for production-versioned Vercel frontend builds."""

import subprocess
from pathlib import Path

import pytest

import src.scripts.build_vercel_frontend as build


def test_build_stops_without_production_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing production configuration blocks the build before subprocess calls.

    Args:
        tmp_path:
            Temporary directory without a Vercel production environment file.
        monkeypatch:
            Pytest fixture for changing directories and replacing subprocess calls.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        build.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected build")
        ),
    )

    assert build.main() == 1


def test_build_uses_production_version_and_restores_generated_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A build embeds production policy without changing source policy artifacts.

    Args:
        monkeypatch:
            Pytest fixture for replacing subprocess calls and changing directories.
        tmp_path:
            Temporary directory for isolated Vercel configuration and policies.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vercel").mkdir()
    (tmp_path / ".vercel/.env.production.local").write_text(
        "EUROEVAL_VERSION=18.4.0\nPRIVATE=value-that-must-not-be-printed\n",
        encoding="utf-8",
    )
    original = []
    for path in build.POLICY_PATHS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("development-policy", encoding="utf-8")
        original.append(target.read_bytes())
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        """Record build commands and emulate generated production policy.

        Args:
            command:
                Subprocess command under test.
            check:
                Whether subprocess failures should raise.

        Returns:
            Successful process result.
        """
        commands.append(command)
        if "generate_volunteer_scope_policy.py" in command:
            assert command[-1] == "18.4.0"
            for path in build.POLICY_PATHS:
                (tmp_path / path).write_text("production-policy", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.main() == 0
    assert commands[-1] == ["vercel", "build", "--prod", "--yes"]
    assert [(tmp_path / path).read_bytes() for path in build.POLICY_PATHS] == original
