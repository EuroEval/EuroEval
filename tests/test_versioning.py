"""Tests for the public versioning interface."""

from pathlib import Path

import pytest

from scripts import versioning


@pytest.mark.parametrize(
    ("initial_version", "expected_version", "changed"),
    [
        ("1.2.3", "1.2.3.dev", True),
        ("1.2.3.dev", "1.2.3.dev", False),
        ("1.2.3.dev0", "1.2.3.dev0", False),
        ("1.2.3.dev.dev", "1.2.3.dev", True),
        ("1.2.3.dev0.dev", "1.2.3.dev0", True),
    ],
)
def test_add_dev_version_is_idempotent_and_recovers_duplicate_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_version: str,
    expected_version: str,
    changed: bool,
) -> None:
    """Add one valid development suffix and run release steps only when changed.

    Args:
        tmp_path:
            Temporary directory fixture.
        monkeypatch:
            Pytest monkeypatch fixture.
        initial_version:
            Version string initially in the project metadata.
        expected_version:
            Canonical version expected after processing.
        changed:
            Whether metadata needs to be updated.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'version = "{initial_version}"\n', encoding="utf-8")
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

    changed_result = versioning.add_dev_version()

    assert changed_result is changed
    assert f'version = "{expected_version}"' in pyproject.read_text(encoding="utf-8")
    if changed:
        assert commands == [
            ["uv", "lock"],
            ["git", "add", "pyproject.toml", "uv.lock"],
            ["git", "commit", "-m", "chore: Add '.dev' suffix to the version number"],
            ["git", "push"],
        ]
    else:
        assert commands == []


def test_add_dev_version_restores_metadata_when_lock_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed lock leaves no changed metadata and runs no git commands.

    Args:
        tmp_path:
            Temporary directory fixture.
        monkeypatch:
            Pytest monkeypatch fixture.
    """
    pyproject = tmp_path / "pyproject.toml"
    original = 'version = "1.2.3"\n'
    pyproject.write_text(original, encoding="utf-8")
    commands: list[list[str]] = []

    def fail_lock(args: list[str], check: bool) -> None:
        """Fail the lock command and record attempted commands.

        Args:
            args:
                Command arguments passed to the subprocess.
            check:
                Whether subprocess failures must raise an error.

        Raises:
            CalledProcessError:
                Always, to emulate a lock failure.
        """
        assert check
        commands.append(args)
        if args == ["uv", "lock"]:
            raise versioning.subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(versioning.subprocess, "run", fail_lock)

    with pytest.raises(versioning.subprocess.CalledProcessError):
        versioning.add_dev_version()

    assert pyproject.read_text(encoding="utf-8") == original
    assert commands == [["uv", "lock"]]


@pytest.mark.parametrize(
    ("failure_command", "expected_commands"),
    [
        (
            ["git", "add", "pyproject.toml", "uv.lock"],
            [["uv", "lock"], ["git", "add", "pyproject.toml", "uv.lock"]],
        ),
        (
            ["git", "commit", "-m", "chore: Add '.dev' suffix to the version number"],
            [
                ["uv", "lock"],
                ["git", "add", "pyproject.toml", "uv.lock"],
                [
                    "git",
                    "commit",
                    "-m",
                    "chore: Add '.dev' suffix to the version number",
                ],
            ],
        ),
        (
            ["git", "push"],
            [
                ["uv", "lock"],
                ["git", "add", "pyproject.toml", "uv.lock"],
                [
                    "git",
                    "commit",
                    "-m",
                    "chore: Add '.dev' suffix to the version number",
                ],
                ["git", "push"],
            ],
        ),
    ],
)
def test_add_dev_version_stops_after_git_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_command: list[str],
    expected_commands: list[list[str]],
) -> None:
    """A failed git operation prevents all later release commands.

    Args:
        tmp_path:
            Temporary directory fixture.
        monkeypatch:
            Pytest monkeypatch fixture.
        failure_command:
            Command configured to fail.
        expected_commands:
            Commands expected up to and including the failure.
    """
    (tmp_path / "pyproject.toml").write_text('version = "1.2.3"\n', encoding="utf-8")
    commands: list[list[str]] = []

    def fail_selected_command(args: list[str], check: bool) -> None:
        """Record commands and fail at the selected git step.

        Args:
            args:
                Command arguments passed to the subprocess.
            check:
                Whether subprocess failures must raise an error.

        Raises:
            CalledProcessError:
                When the selected command is reached.
        """
        assert check
        commands.append(args)
        if args == failure_command:
            raise versioning.subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(versioning.subprocess, "run", fail_selected_command)

    with pytest.raises(versioning.subprocess.CalledProcessError):
        versioning.add_dev_version()

    assert commands == expected_commands


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
