"""Tests for the public versioning interface."""

from pathlib import Path

import pytest

from scripts import versioning


def test_set_new_version_stages_builtin_dataset_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release commits include refreshed built-in dataset configuration pins.

    Args:
        tmp_path:
            Temporary directory fixture.
        monkeypatch:
            Pytest monkeypatch fixture.
    """
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n[Unreleased]\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text('version = "1.0.0"\n', encoding="utf-8")
    commands: list[list[str]] = []

    def record_command(args: list[str], check: bool) -> None:
        """Record subprocess calls without performing release side effects.

        Args:
            args:
                Command arguments passed to the subprocess.
            check:
                Whether subprocess failures must raise an error.
        """
        assert check
        commands.append(args)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(versioning.subprocess, "run", record_command)

    versioning.set_new_version(major=1, minor=1, patch=0)

    assert ["git", "add", "src/euroeval/dataset_configs/"] in commands
    assert commands[-4:] == [
        ["git", "commit", "-m", "feat: v1.1.0"],
        ["git", "tag", "v1.1.0"],
        ["git", "push"],
        ["git", "push", "--tags"],
    ]
