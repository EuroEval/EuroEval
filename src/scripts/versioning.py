# /// script
# requires-python = ">=3.10,<4.0"
# dependencies = []
# ///

"""Scripts related to updating of version."""

import datetime as dt
import re
import subprocess
from pathlib import Path


def add_dev_version() -> bool:
    """Add a development suffix and commit the updated version metadata.

    A repeated invocation with an already-valid development version is a no-op. A
    malformed repeated suffix is normalised so this command can repair interrupted
    legacy runs without first asking uv to parse the invalid project metadata.

    Returns:
        Whether the version changed and release steps were run.

    Raises:
        RuntimeError:
            If the version is missing or is not a supported release/development form.
        subprocess.CalledProcessError:
            If locking or a git command fails.
    """
    pyproject_path = Path("pyproject.toml")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    version_match = re.search(r'(?m)^version = "([^"]+)"', pyproject)
    if version_match is None:
        raise RuntimeError("No version found in pyproject.toml.")

    version = version_match.group(1)
    version_parts = re.fullmatch(
        r"(?P<release>\d+\.\d+\.\d+)(?P<suffix>(?:\.dev\d*)*)", version
    )
    if version_parts is None:
        raise RuntimeError(f"Unsupported project version: {version}")

    release = version_parts.group("release")
    suffixes = re.findall(r"\.dev(\d*)", version_parts.group("suffix"))
    if len(suffixes) > 1:
        new_version = f"{release}.dev{suffixes[0]}"
    elif suffixes:
        return False
    else:
        new_version = f"{release}.dev"

    updated = (
        pyproject[: version_match.start(1)]
        + new_version
        + pyproject[version_match.end(1) :]
    )
    pyproject_path.write_text(data=updated, encoding="utf-8")
    try:
        subprocess.run(args=["uv", "lock"], check=True)
    except subprocess.CalledProcessError:
        pyproject_path.write_text(data=pyproject, encoding="utf-8")
        raise

    subprocess.run(args=["git", "add", "pyproject.toml", "uv.lock"], check=True)
    subprocess.run(
        args=["git", "commit", "-m", "chore: Add '.dev' suffix to the version number"],
        check=True,
    )
    subprocess.run(args=["git", "push"], check=True)
    return True


def bump_major() -> None:
    """Add one to the major version."""
    major, _, _ = get_current_version()
    set_new_version(major + 1, 0, 0)


def get_current_version() -> tuple[int, int, int]:
    """Fetch the current version of the package.

    Returns:
        The current version, separated into major, minor and patch versions.

    Raises:
        RuntimeError:
            If no version can be found in the `pyproject.toml` file.
    """
    version_candidates = re.search(
        r'(?<=version = ")[^"]+(?=")',
        Path("pyproject.toml").read_text(encoding="utf-8"),
    )
    if version_candidates is None:
        raise RuntimeError("No version found in pyproject.toml.")

    version_str = version_candidates.group(0).replace(".dev", "")
    major, minor, patch = map(int, version_str.split("."))
    return major, minor, patch


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

    Raises:
        RuntimeError:
            If no version can be found in the `pyproject.toml` file.
    """
    version = f"{major}.{minor}.{patch}"

    # Get current changelog and ensure that it has an [Unreleased] entry
    changelog_path = Path("CHANGELOG.md")
    changelog = changelog_path.read_text(encoding="utf-8")
    if "[Unreleased]" not in changelog:
        raise RuntimeError("No [Unreleased] entry in CHANGELOG.md.")

    # Add version to CHANGELOG
    today = dt.date.today().strftime("%Y-%m-%d")
    new_changelog = re.sub(
        r"\[Unreleased\].*", f"[Unreleased]\n\n## [v{version}] - {today}", changelog
    )
    changelog_path.write_text(data=new_changelog, encoding="utf-8")

    # Update the version in the `pyproject.toml` file
    pyproject_path = Path("pyproject.toml")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    pyproject = re.sub(
        r'version = "[^"]+"', f'version = "{version}"', pyproject, count=1
    )
    pyproject_path.write_text(data=pyproject, encoding="utf-8")

    # Install newest project. check=True so a failed step aborts before we tag
    # and push a release.
    subprocess.run(args=["make", "install"], check=True)

    # Add to version control
    subprocess.run(args=["git", "add", ".pre-commit-config.yaml"], check=True)
    subprocess.run(args=["git", "add", "CHANGELOG.md"], check=True)
    subprocess.run(args=["git", "add", "pyproject.toml"], check=True)
    subprocess.run(args=["git", "add", "uv.lock"], check=True)
    subprocess.run(args=["git", "add", "src/euroeval/dataset_configs/"], check=True)
    subprocess.run(args=["git", "commit", "-m", f"feat: v{version}"], check=True)
    subprocess.run(args=["git", "tag", f"v{version}"], check=True)
    subprocess.run(args=["git", "push"], check=True)
    subprocess.run(args=["git", "push", "--tags"], check=True)


def bump_minor() -> None:
    """Add one to the minor version."""
    major, minor, _ = get_current_version()
    set_new_version(major, minor + 1, 0)


def bump_patch() -> None:
    """Add one to the patch version."""
    major, minor, patch = get_current_version()
    set_new_version(major, minor, patch + 1)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--add-dev",
        action="store_true",
        help="Add a development suffix and commit the updated version metadata.",
    )
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

    if args.add_dev:
        if args.major + args.minor + args.patch:
            raise RuntimeError("--add-dev cannot be combined with a version bump.")
        add_dev_version()
    elif args.major + args.minor + args.patch != 1:
        raise RuntimeError(
            "Exactly one of --major, --minor and --patch must be selected."
        )
    elif args.major:
        bump_major()
    elif args.minor:
        bump_minor()
    elif args.patch:
        bump_patch()
