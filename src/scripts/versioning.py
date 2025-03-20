# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = []
# ///

"""Scripts related to updating of version."""

import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import Tuple


def bump_major() -> None:
    """Add one to the major version."""
    major, _, _ = get_current_version()
    set_new_version(major + 1, 0, 0)


def bump_minor() -> None:
    """Add one to the minor version."""
    major, minor, _ = get_current_version()
    set_new_version(major, minor + 1, 0)


def bump_patch() -> None:
    """Add one to the patch version."""
    major, minor, patch = get_current_version()
    set_new_version(major, minor, patch + 1)


def set_new_version(major: int, minor: int, patch: int) -> None:
    """Sets a new version.

    Args:
        major:
            The major version. This only changes when the code stops being backwards
            compatible.
        minor:
            The minor version. This changes when a backwards compatible change
            happened.
        patch:
            The patch version. This changes when the only new changes are bug fixes.
    """
    version = f"{major}.{minor}.{patch}"

    # Get current changelog and ensure that it has an [Unreleased] entry
    changelog_path = Path("CHANGELOG.md")
    changelog = changelog_path.read_text()
    if "[Unreleased]" not in changelog:
        raise RuntimeError("No [Unreleased] entry in CHANGELOG.md.")

    # Add version to CHANGELOG
    today = dt.date.today().strftime("%Y-%m-%d")
    new_changelog = re.sub(
        r"\[Unreleased\].*", f"[Unreleased]\n\n\n\n## [v{version}] - {today}", changelog
    )
    changelog_path.write_text(new_changelog)

    # Update the version in the `pyproject.toml` file
    pyproject_path = Path("pyproject.toml")
    pyproject = pyproject_path.read_text()
    pyproject = re.sub(
        r'version = "[^"]+"', f'version = "{version}"', pyproject, count=1
    )
    pyproject_path.write_text(pyproject)

    # Install newest project
    subprocess.run(["make", "install"])

    # Add to version control
    subprocess.run(["git", "add", ".pre-commit-config.yaml"])
    subprocess.run(["git", "add", "CHANGELOG.md"])
    subprocess.run(["git", "add", "pyproject.toml"])
    subprocess.run(["git", "add", "uv.lock"])
    subprocess.run(["git", "commit", "-m", f"feat: v{version}"])
    subprocess.run(["git", "tag", f"v{version}"])
    subprocess.run(["git", "push"])
    subprocess.run(["git", "push", "--tags"])


def get_current_version() -> Tuple[int, int, int]:
    """Fetch the current version of the package.

    Returns:
        The current version, separated into major, minor and patch versions.
    """
    # Get all the version candidates from pyproject.toml
    version_candidates = re.search(
        r'(?<=version = ")[^"]+(?=")', Path("pyproject.toml").read_text()
    )

    # If no version candidates were found, raise an error
    if version_candidates is None:
        raise RuntimeError("No version found in pyproject.toml.")

    # Otherwise, extract the version, split it into major, minor and patch parts and
    # return these
    else:
        version_str = version_candidates.group(0).replace(".dev", "")
        major, minor, patch = map(int, version_str.split("."))
        return major, minor, patch


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--major",
        const=True,
        nargs="?",
        default=False,
        help="Bump the major version by one.",
    )
    parser.add_argument(
        "--minor",
        const=True,
        nargs="?",
        default=False,
        help="Bump the minor version by one.",
    )
    parser.add_argument(
        "--patch",
        const=True,
        nargs="?",
        default=False,
        help="Bump the patch version by one.",
    )
    args = parser.parse_args()

    if args.major + args.minor + args.patch != 1:
        raise RuntimeError(
            "Exactly one of --major, --minor and --patch must be selected."
        )
    elif args.major:
        bump_major()
    elif args.minor:
        bump_minor()
    elif args.patch:
        bump_patch()
