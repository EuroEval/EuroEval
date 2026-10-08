"""Tests for preparing worktree-local Vercel configuration."""

import json
from pathlib import Path

import pytest

from src.scripts.prepare_vercel_worktree import prepare_vercel_directory


def test_prepare_vercel_directory_copies_linkage_and_production_env(
    tmp_path: Path,
) -> None:
    """Materialize only the local config, leaving shared build output untouched."""
    shared_vercel = tmp_path / "shared" / ".vercel"
    shared_vercel.mkdir(parents=True)
    project = {"projectId": "project-id", "orgId": "org-id"}
    (shared_vercel / "project.json").write_text(json.dumps(project))
    (shared_vercel / ".env.production.local").write_text("SECRET=value")
    shared_output = shared_vercel / "output"
    shared_output.mkdir()
    (shared_output / "marker").write_text("shared output")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".vercel").symlink_to(shared_vercel, target_is_directory=True)

    prepare_vercel_directory(worktree)

    local_vercel = worktree / ".vercel"
    assert local_vercel.is_dir()
    assert not local_vercel.is_symlink()
    assert json.loads((local_vercel / "project.json").read_text()) == project
    assert (local_vercel / ".env.production.local").read_text() == "SECRET=value"
    assert not (local_vercel / "output").exists()
    assert (shared_output / "marker").read_text() == "shared output"


def test_prepare_vercel_directory_keeps_dangling_symlink_on_failure(
    tmp_path: Path,
) -> None:
    """Do not unlink a symlink when its target cannot be accessed."""
    link = tmp_path / ".vercel"
    link.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(RuntimeError, match="target is unavailable"):
        prepare_vercel_directory(tmp_path)

    assert link.is_symlink()


def test_prepare_vercel_directory_leaves_missing_link_untouched(tmp_path: Path) -> None:
    """Treat an absent .vercel directory as a no-op."""
    prepare_vercel_directory(tmp_path)

    assert not (tmp_path / ".vercel").exists()


def test_prepare_vercel_directory_leaves_real_directory_unchanged(
    tmp_path: Path,
) -> None:
    """Leave a normal local Vercel directory untouched."""
    vercel_dir = tmp_path / ".vercel"
    vercel_dir.mkdir()
    project_file = vercel_dir / "project.json"
    project_file.write_text('{"projectId":"local"}')

    prepare_vercel_directory(tmp_path)

    assert not vercel_dir.is_symlink()
    assert project_file.read_text() == '{"projectId":"local"}'


def test_prepare_vercel_directory_requires_project_linkage(tmp_path: Path) -> None:
    """Keep the shared symlink if it cannot identify the Vercel project."""
    shared_vercel = tmp_path / "shared" / ".vercel"
    shared_vercel.mkdir(parents=True)
    link = tmp_path / ".vercel"
    link.symlink_to(shared_vercel, target_is_directory=True)

    with pytest.raises(RuntimeError, match="no project.json"):
        prepare_vercel_directory(tmp_path)

    assert link.is_symlink()
