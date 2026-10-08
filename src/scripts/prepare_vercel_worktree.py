"""Prepare a worktree-local Vercel directory before building."""

import shutil
import sys
import tempfile
from pathlib import Path

FILES_TO_PRESERVE = ("project.json", ".env.production.local")


def main() -> int:
    """Prepare the current project's Vercel directory.

    Returns:
        Exit status: zero on success, one when preparation fails.
    """
    try:
        prepare_vercel_directory()
    except (OSError, RuntimeError) as error:
        print(
            f"Unable to prepare worktree-local .vercel directory: {error}",
            file=sys.stderr,
        )
        return 1
    return 0


def prepare_vercel_directory(project_dir: Path = Path.cwd()) -> None:
    """Replace a symlinked .vercel directory with local linkage and build config.

    Only Vercel's project linkage and local production environment file are copied.
    In particular, an existing build output is never copied or modified.

    Args:
        project_dir: Root directory of the project/worktree.

    Raises:
        RuntimeError: If the symlink target is unavailable or has no project link.
        OSError: If the local Vercel directory cannot be prepared.
    """
    vercel_dir = project_dir / ".vercel"
    if not vercel_dir.is_symlink():
        return

    try:
        source_dir = vercel_dir.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RuntimeError("The .vercel symlink target is unavailable.") from error

    if not source_dir.is_dir():
        raise RuntimeError("The .vercel symlink target is not a directory.")

    project_file = source_dir / "project.json"
    if not project_file.is_file():
        raise RuntimeError("The linked .vercel directory has no project.json.")

    with tempfile.TemporaryDirectory(
        prefix=".vercel-prep-", dir=project_dir
    ) as staging:
        staging_dir = Path(staging)
        for name in FILES_TO_PRESERVE:
            source_file = source_dir / name
            if source_file.is_file():
                shutil.copy2(source_file, staging_dir / name)
        vercel_dir.unlink()
        try:
            staging_dir.rename(vercel_dir)
        except OSError:
            vercel_dir.symlink_to(source_dir, target_is_directory=True)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
