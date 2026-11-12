# /// script
# requires-python = ">=3.12,<4.0"
# dependencies = []
# ///

"""Scripts related to updating of version."""

import datetime as dt
import json
import re
import subprocess
import tempfile
import tomllib
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
            If locking, policy generation, or a git command fails.
        OSError:
            If a policy artifact cannot be generated or written.
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
        regenerate_scope_policies(version=new_version)
    except (subprocess.CalledProcessError, OSError):
        pyproject_path.write_text(data=pyproject, encoding="utf-8")
        raise

    subprocess.run(
        args=[
            "git",
            "add",
            "pyproject.toml",
            "uv.lock",
            "api/worker/scope-policy.json",
            "api/worker/_lib/scope-policy.generated.ts",
        ],
        check=True,
    )
    subprocess.run(
        args=["git", "commit", "-m", "chore: Add '.dev' suffix to the version number"],
        check=True,
    )
    subprocess.run(args=["git", "push"], check=True)
    return True


def regenerate_scope_policies(*, version: str) -> None:
    """Regenerate both tracked volunteer policy artifacts for a package version.

    Args:
        version:
            Exact stable or PEP 440 development version to encode.

    """
    subprocess.run(
        args=[
            "uv",
            "run",
            "python",
            "src/scripts/generate_volunteer_scope_policy.py",
            "--version",
            version,
        ],
        check=True,
    )


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
    try:
        project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        version_str = str(project["project"]["version"])
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("No version found in pyproject.toml.") from error
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.dev\d*)?", version_str)
    if match is None:
        raise RuntimeError(f"Unsupported project version: {version_str}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


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
    version_match = re.search(r'(?m)^version = "[^"]+"$', pyproject)
    if version_match is None:
        raise RuntimeError("No version found in pyproject.toml.")
    pyproject = (
        pyproject[: version_match.start()]
        + f'version = "{version}"'
        + pyproject[version_match.end() :]
    )
    pyproject_path.write_text(data=pyproject, encoding="utf-8")

    regenerate_scope_policies(version=version)

    # Install newest project. check=True so a failed step aborts before we tag
    # and push a release.
    subprocess.run(args=["make", "install"], check=True)

    # Add to version control
    subprocess.run(args=["git", "add", ".pre-commit-config.yaml"], check=True)
    subprocess.run(args=["git", "add", "CHANGELOG.md"], check=True)
    subprocess.run(args=["git", "add", "pyproject.toml"], check=True)
    subprocess.run(args=["git", "add", "uv.lock"], check=True)
    subprocess.run(
        args=[
            "git",
            "add",
            "api/worker/scope-policy.json",
            "api/worker/_lib/scope-policy.generated.ts",
        ],
        check=True,
    )
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


def create_github_release(version: str) -> bool:
    """Create a release, or leave an identical existing release unchanged.

    Args:
        version:
            Canonical stable version to release.

    Returns:
        Whether a new release was created.

    """
    notes = preflight_release(version=version)
    tag = f"v{version}"
    existing = _get_existing_release(tag=tag)
    if existing is not None:
        _validate_existing_release(tag=tag, release=existing, notes=notes)
        return False

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".release-notes-",
        suffix=".tmp",
        delete=False,
    ) as notes_file:
        notes_file.write(notes)
        notes_path = Path(notes_file.name)
    try:
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--verify-tag",
                "--repo",
                "EuroEval/EuroEval",
                "--title",
                tag,
                "--notes-file",
                str(notes_path),
            ],
            check=True,
        )
    finally:
        notes_path.unlink(missing_ok=True)
    return True


def _get_existing_release(tag: str) -> dict[str, str] | None:
    """Fetch a release, distinguishing GitHub's structured 404 from other errors.

    Args:
        tag:
            Release tag to check.

    Returns:
        The release fields, or None when the release does not exist.

    Raises:
        RuntimeError:
            If GitHub cannot be queried or returns malformed data.
    """
    result = subprocess.run(
        ["gh", "api", f"repos/EuroEval/EuroEval/releases/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        try:
            release = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid GitHub release response for {tag}.") from error
        if not isinstance(release, dict):
            raise RuntimeError(f"Invalid GitHub release response for {tag}.")
        return release

    try:
        error_body = json.loads(result.stdout)
    except json.JSONDecodeError:
        error_body = None
    if isinstance(error_body, dict) and str(error_body.get("status")) == "404":
        return None
    raise RuntimeError(f"Unable to check GitHub release {tag}: {result.stderr.strip()}")


def _validate_existing_release(tag: str, release: dict[str, str], notes: str) -> None:
    """Reject an existing release unless its title and notes match exactly.

    Args:
        tag:
            Expected release title and tag.
        release:
            Release response from GitHub.
        notes:
            Expected changelog section body.

    Raises:
        RuntimeError:
            If title or notes differ from the expected values.
    """
    if release.get("name") != tag or release.get("body") != notes:
        raise RuntimeError(
            f"GitHub release {tag} exists with conflicting title or notes."
        )


def preflight_release(version: str) -> str:
    """Validate release notes, remote tag, and any existing GitHub release.

    Args:
        version:
            Canonical stable version to release.

    Returns:
        Exact release notes body.

    Raises:
        RuntimeError:
            If tag/release preflight fails or an existing release conflicts.
    """
    validate_stable_version(version=version)
    notes = get_release_notes(version=version)
    tag = f"v{version}"
    tag_check = subprocess.run(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if tag_check.returncode == 2:
        raise RuntimeError(f"Remote tag {tag} does not exist; refusing to create it.")
    if tag_check.returncode != 0:
        raise RuntimeError(
            f"Unable to verify remote tag {tag}: {tag_check.stderr.strip()}"
        )
    release = _get_existing_release(tag=tag)
    if release is not None:
        _validate_existing_release(tag=tag, release=release, notes=notes)
    return notes


def get_release_notes(version: str, changelog: str | None = None) -> str:
    """Return exactly one version's changelog section body.

    Args:
        version:
            Canonical stable version to find.
        changelog (optional):
            Changelog text. Defaults to CHANGELOG.md.

    Returns:
        Section content with only surrounding blank lines removed.

    Raises:
        RuntimeError:
            If the matching section is absent, duplicated, or empty.
    """
    validate_stable_version(version=version)
    if changelog is None:
        changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(rf"(?m)^## \[v?{re.escape(version)}\](?:\s+-.*)?$")
    matches = list(heading.finditer(changelog))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one CHANGELOG.md section for "
            f"{version}; found {len(matches)}."
        )
    start = matches[0].end()
    next_heading = re.search(r"(?m)^## ", changelog[start:])
    end = start + next_heading.start() if next_heading else len(changelog)
    body = changelog[start:end].strip("\r\n")
    if not body.strip():
        raise RuntimeError(f"CHANGELOG.md section for {version} is empty.")
    return body


def validate_stable_version(version: str) -> str:
    """Require a canonical stable MAJOR.MINOR.PATCH version.

    Args:
        version:
            The version string to validate.

    Returns:
        The unchanged canonical version string.

    Raises:
        RuntimeError:
            If the version is not a canonical stable release.
    """
    if re.fullmatch(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", version) is None:
        raise RuntimeError(
            f"Refusing to publish non-canonical stable version: {version!r}. "
            "Expected MAJOR.MINOR.PATCH."
        )
    return version


def publish_preflight() -> None:
    """Validate project version and release prerequisites before package builds."""
    version = validate_stable_version(version=get_project_version())
    preflight_release(version=version)


def get_project_version() -> str:
    """Read the literal project version without normalising prerelease metadata.

    Returns:
        The version declared in pyproject.toml.

    Raises:
        RuntimeError:
            If project version metadata is missing.
    """
    version_match = re.search(
        r'(?m)^version = "([^"]+)"', Path("pyproject.toml").read_text(encoding="utf-8")
    )
    if version_match is None:
        raise RuntimeError("No version found in pyproject.toml.")
    return version_match.group(1)


def validate_project_version() -> None:
    """Validate the literal project version for an individual package upload."""
    validate_stable_version(version=get_project_version())


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--validate-publish", action="store_true")
    parser.add_argument("--validate-stable", action="store_true")
    parser.add_argument("--release-only", metavar="VERSION")
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
    selected_actions = sum(
        bool(action)
        for action in (
            args.validate_publish,
            args.validate_stable,
            args.release_only,
            args.add_dev,
            args.major,
            args.minor,
            args.patch,
        )
    )
    if selected_actions > 1:
        parser.error("Select exactly one version action.")

    if args.validate_publish or args.validate_stable:
        if args.add_dev or args.major or args.minor or args.patch or args.release_only:
            raise RuntimeError(
                "Version validation cannot be combined with other actions."
            )
        if args.validate_publish and args.validate_stable:
            raise RuntimeError("Select only one version validation action.")
        if args.validate_publish:
            publish_preflight()
        else:
            validate_project_version()
    elif args.release_only:
        create_github_release(version=args.release_only)
    elif args.add_dev:
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
