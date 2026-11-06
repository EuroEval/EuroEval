"""Tests for stable-version publishing safeguards."""

import pytest

from scripts.versioning import get_release_notes, validate_stable_version


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


@pytest.mark.parametrize(
    "version", ["1.2.3.dev0", "1.2.3rc1", "1.2.3.post1", "1.2.3+local", "01.2.3", "1.2"]
)
def test_validate_stable_version_rejects_noncanonical_versions(version: str) -> None:
    """Reject prerelease, local, malformed, and noncanonical versions."""
    with pytest.raises(RuntimeError, match="Expected MAJOR.MINOR.PATCH"):
        validate_stable_version(version=version)
