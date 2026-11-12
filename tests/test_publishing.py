"""Tests for stable-version publishing safeguards."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import versioning
from scripts.versioning import get_release_notes, validate_stable_version

CHANGELOG = "## [v18.4.0] - 2026-11-06\n\n### Added\n\n- Release notes\n"


def test_create_github_release_creates_exact_changelog_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create a missing release with exactly the matching changelog section."""
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Fake git and gh without network access.

        Returns:
            The simulated command result.
        """
        calls.append(args)
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, stdout="tag", stderr="")
        if args[1] == "api":
            return subprocess.CompletedProcess(
                args,
                1,
                stdout='{"message":"Not Found","status":404}',
                stderr="gh: Not Found (HTTP 404)",
            )
        if args[1:3] == ["release", "create"]:
            notes_path = Path(args[args.index("--notes-file") + 1])
            assert (
                notes_path.read_text(encoding="utf-8") == "### Added\n\n- Release notes"
            )
            assert not (tmp_path / ".release-notes.tmp").exists()
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        pytest.fail(f"Unexpected command: {args}")

    monkeypatch.setattr(versioning.subprocess, "run", run)
    assert versioning.create_github_release(version="18.4.0")
    create_call = next(call for call in calls if call[1:3] == ["release", "create"])
    assert create_call[create_call.index("--repo") + 1] == "EuroEval/EuroEval"


def test_existing_release_is_compared_exactly_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave an identical release unchanged and reject a race-created conflict."""
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    release = {"name": "v18.4.0", "body": "### Added\n\n- Release notes"}

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, stdout="tag", stderr="")
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(release), stderr=""
        )

    monkeypatch.setattr(versioning.subprocess, "run", run)
    assert not versioning.create_github_release(version="18.4.0")
    release["body"] = "different"
    with pytest.raises(RuntimeError, match="conflicting"):
        versioning.create_github_release(version="18.4.0")


def test_get_release_notes_preserves_section_content() -> None:
    """Extract only the requested changelog body without altering its contents."""
    changelog = (
        "## [v18.4.0] - 2026-11-06\n\n### Added\n\n- Exact **notes** ✨  \n"
        "\n## [v18.3.0] - 2026-10-08\n\nOlder notes\n"
    )
    assert get_release_notes(version="18.4.0", changelog=changelog) == (
        "### Added\n\n- Exact **notes** ✨  "
    )


@pytest.mark.parametrize(
    "changelog",
    ["## [v18.4.0] - date\n\n## [v18.3.0] - date\n", "## [v18.4.0] - date\n\n"],
)
def test_get_release_notes_rejects_duplicate_or_empty_sections(changelog: str) -> None:
    """Require one nonempty matching changelog section."""
    with pytest.raises(RuntimeError):
        get_release_notes(version="18.4.0", changelog=changelog)


@pytest.mark.parametrize("target", ["publish", "publish-euroeval", "publish-scandeval"])
@pytest.mark.parametrize("version", ["18.4.0.dev", "18.4.0.dev0", "18.4.0.dev1"])
def test_make_publish_rejects_development_metadata_before_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, version: str
) -> None:
    """Run the real version guard before any build or source mutations.

    Args:
        tmp_path:
            Temporary project directory.
        monkeypatch:
            Isolated environment fixture.
        target:
            Publishing entry point to exercise.
        version:
            Development version that must never be uploaded.
    """
    source = Path(__file__).parents[1]
    shutil.copy(source / "makefile", tmp_path / "makefile")
    script = tmp_path / "src/scripts/versioning.py"
    script.parent.mkdir(parents=True)
    shutil.copy(source / "src/scripts/versioning.py", script)
    package = tmp_path / "src/euroeval"
    package.mkdir()
    original = f'[project]\nname = "EuroEval"\nversion = "{version}"\n'
    (tmp_path / "pyproject.toml").write_text(original, encoding="utf-8")
    for directory in ("build", "dist"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "sentinel").write_text("keep", encoding="utf-8")
    fake_bin = tmp_path / ".local/bin"
    fake_bin.mkdir(parents=True)
    uv = fake_bin / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "assert sys.argv[1:3] == ['run', 'src/scripts/versioning.py']\n"
        "assert sys.argv[3] in ('--validate-publish', '--validate-stable')\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[2:]])\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    monkeypatch.setenv("EUROEVAL_PYPI_API_TOKEN", "fake")
    monkeypatch.setenv("SCANDEVAL_PYPI_API_TOKEN", "fake")

    result = subprocess.run(
        ["make", "-j2", target], cwd=tmp_path, capture_output=True, text=True
    )

    assert result.returncode != 0
    assert "Refusing to publish non-canonical stable version" in result.stderr
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == original
    assert package.is_dir()
    assert not (tmp_path / "src/scandeval").exists()
    for directory in ("build", "dist"):
        assert (tmp_path / directory / "sentinel").read_text(encoding="utf-8") == "keep"


def test_publish_wrapper_orders_all_stages_and_stops_on_release_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep release-stage ordering under -j and stop after release errors."""
    shutil.copy(Path(__file__).parents[1] / "makefile", tmp_path / "makefile")
    (tmp_path / "src/euroeval").mkdir(parents=True)
    (tmp_path / "src/euroeval/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "EuroEval"\nversion = "18.4.0"\n', encoding="utf-8"
    )
    fake_bin = tmp_path / ".local/bin"
    fake_bin.mkdir(parents=True)
    log = tmp_path / "commands.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'case "$*" in *get_project_version*) echo 18.4.0; exit 0 ;; esac\n'
        'echo "uv $*" >> "$COMMAND_LOG"\n'
        'case "$*" in *build_vercel_frontend*) '
        '[ "$FAIL_FRONTEND" != yes ] || exit 1 ;; esac\n'
        'if [ "$1" = publish ]; then\n'
        '  echo "package $(grep name pyproject.toml)" >> "$COMMAND_LOG"\n'
        "fi\n"
        'case "$*" in *--release-only*) [ "$FAIL_RELEASE" != yes ] ;; esac\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    vercel = fake_bin / "vercel"
    vercel.write_text(
        '#!/bin/sh\necho "vercel $*" >> "$COMMAND_LOG"\n'
        'case "$*" in *build*) [ "$FAIL_FRONTEND" != yes ] ;; esac\n',
        encoding="utf-8",
    )
    vercel.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    monkeypatch.setenv("COMMAND_LOG", str(log))
    monkeypatch.setenv("EUROEVAL_PYPI_API_TOKEN", "fake")
    monkeypatch.setenv("SCANDEVAL_PYPI_API_TOKEN", "fake")

    result = subprocess.run(
        ["make", "-j2", "publish"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Published!" in result.stdout
    commands = log.read_text(encoding="utf-8").splitlines()
    release = next(i for i, line in enumerate(commands) if "--release-only" in line)
    euro = next(
        i for i, line in enumerate(commands) if 'package name = "EuroEval"' in line
    )
    scandeval = next(
        i for i, line in enumerate(commands) if 'package name = "ScandEval"' in line
    )
    dev = next(i for i, line in enumerate(commands) if "--add-dev" in line)
    sync = next(
        i for i, line in enumerate(commands) if "sync-production-version" in line
    )
    frontend = next(
        i for i, line in enumerate(commands) if "deploy_vercel_frontend.py" in line
    )
    assert euro < scandeval < release < sync < frontend < dev
    assert "--release-only 18.4.0" in commands[release]
    assert (tmp_path / "src/euroeval").is_dir()
    assert not (tmp_path / "src/scandeval").exists()

    log.unlink()
    monkeypatch.setenv("FAIL_RELEASE", "yes")
    failed = subprocess.run(
        ["make", "-j2", "publish"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    failure_commands = log.read_text(encoding="utf-8")
    assert "--add-dev" not in failure_commands
    assert "deploy_vercel_frontend.py" not in failure_commands
    assert "Published!" not in failed.stdout

    log.unlink()
    monkeypatch.delenv("FAIL_RELEASE")
    monkeypatch.setenv("FAIL_FRONTEND", "yes")
    failed_frontend = subprocess.run(
        ["make", "-j2", "publish"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_frontend.returncode != 0
    failed_frontend_commands = log.read_text(encoding="utf-8")
    assert "--add-dev" not in failed_frontend_commands
    assert "Published!" not in failed_frontend.stdout

    monkeypatch.delenv("FAIL_FRONTEND")
    log.unlink()
    parallel_goals = subprocess.run(
        ["make", "-j2", "publish-euroeval", "add-dev-version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert parallel_goals.returncode == 0, parallel_goals.stderr
    parallel_commands = log.read_text(encoding="utf-8").splitlines()
    package_upload = next(
        i
        for i, line in enumerate(parallel_commands)
        if 'package name = "EuroEval"' in line
    )
    dev_update = next(
        i for i, line in enumerate(parallel_commands) if "--add-dev" in line
    )
    assert package_upload < dev_update

    log.unlink()
    package_goals = subprocess.run(
        ["make", "-j2", "publish-euroeval", "publish-scandeval"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert package_goals.returncode == 0, package_goals.stderr
    package_commands = log.read_text(encoding="utf-8").splitlines()
    euro_upload = next(
        i
        for i, line in enumerate(package_commands)
        if 'package name = "EuroEval"' in line
    )
    scand_upload = next(
        i
        for i, line in enumerate(package_commands)
        if 'package name = "ScandEval"' in line
    )
    assert euro_upload < scand_upload


def test_release_checks_propagate_api_and_tag_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject absent tags and do not mistake auth or network failures for 404s."""
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def absent_tag(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 2, stdout="", stderr="")

    monkeypatch.setattr(versioning.subprocess, "run", absent_tag)
    with pytest.raises(RuntimeError, match="Remote tag"):
        versioning.preflight_release(version="18.4.0")

    def api_failure(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, stdout="tag", stderr="")
        return subprocess.CompletedProcess(
            args, 1, stdout="", stderr="authentication failed"
        )

    monkeypatch.setattr(versioning.subprocess, "run", api_failure)
    with pytest.raises(RuntimeError, match="authentication failed"):
        versioning.preflight_release(version="18.4.0")


def test_scandeval_make_target_restores_metadata_after_build_or_upload_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore renamed project metadata after either external command fails."""
    shutil.copy(Path(__file__).parents[1] / "makefile", tmp_path / "makefile")
    (tmp_path / "src/euroeval").mkdir(parents=True)
    (tmp_path / "src/euroeval/__init__.py").write_text("", encoding="utf-8")
    original = '[project]\nname = "EuroEval"\nversion = "18.4.0"\n'
    (tmp_path / "pyproject.toml").write_text(original, encoding="utf-8")
    fake_bin = tmp_path / ".local/bin"
    fake_bin.mkdir(parents=True)
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *--validate-stable*)\n"
        "    grep -Eq '^version = \\\"18\\.4\\.0\\\"$' pyproject.toml ;;\n"
        '  *build*) [ "$FAIL_AT" != build ] ;;\n'
        '  *publish*) [ "$FAIL_AT" != publish ] ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{fake_bin}:{Path('/usr/bin')}:/bin")
    monkeypatch.setenv("SCANDEVAL_PYPI_API_TOKEN", "fake")
    monkeypatch.setenv("EUROEVAL_PYPI_API_TOKEN", "fake")

    for failure in ("build", "publish"):
        monkeypatch.setenv("FAIL_AT", failure)
        result = subprocess.run(
            ["make", "-j2", "publish-scandeval"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == original
        assert (tmp_path / "src/euroeval").is_dir()
        assert not (tmp_path / "src/scandeval").exists()

    (tmp_path / "build").mkdir()
    (tmp_path / "build/sentinel").write_text("kept", encoding="utf-8")
    for target, name in (
        ("publish-euroeval", "EuroEval"),
        ("publish-scandeval", "EuroEval"),
    ):
        development_metadata = f'[project]\nname = "{name}"\nversion = "18.4.0.dev1"\n'
        (tmp_path / "pyproject.toml").write_text(development_metadata, encoding="utf-8")
        blocked = subprocess.run(
            ["make", "-j2", target],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert blocked.returncode != 0
        assert (tmp_path / "pyproject.toml").read_text(
            encoding="utf-8"
        ) == development_metadata
        assert (tmp_path / "build/sentinel").read_text(encoding="utf-8") == "kept"
        assert (tmp_path / "src/euroeval").is_dir()
        assert not (tmp_path / "src/scandeval").exists()


@pytest.mark.parametrize(
    "version",
    [
        "1.2.3.dev",
        "1.2.3.dev0",
        "1.2.3.dev1",
        "1.2.3rc1",
        "1.2.3.post1",
        "1.2.3+local",
        "01.2.3",
        "1.2",
    ],
)
def test_validate_stable_version_rejects_noncanonical_versions(version: str) -> None:
    """Reject prerelease, local, malformed, and noncanonical versions."""
    with pytest.raises(RuntimeError, match="Expected MAJOR.MINOR.PATCH"):
        validate_stable_version(version=version)
