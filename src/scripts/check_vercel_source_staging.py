"""Check that Vercel Git builds receive every build-time source input."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import pathspec

REPO_ROOT = Path(__file__).resolve().parents[2]
IGNORE_FILE = ".vercelignore"
FRONTEND_ROOT = "src/frontend/"
GENERATED_CSV_ROOT = "src/frontend/csv/"
PYTHON_SOURCE_ROOT = "src/euroeval/"
REQUIRED_FILES = frozenset(
    {
        "vite.config.js",
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "src/scripts/build-seo-files.mjs",
        "src/scripts/build-api-reference.mjs",
        "src/scripts/build_api_reference.py",
        "src/frontend/App.vue",
        "src/frontend/config.yaml",
        "src/frontend/main.ts",
        "src/frontend/md/about.md",
        "src/euroeval/__init__.py",
    }
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("check_vercel_source_staging")


def main() -> int:
    """Validate Vercel source staging and report any missing build inputs.

    Returns:
        ``0`` when validation succeeds, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_root",
        nargs="?",
        type=Path,
        default=REPO_ROOT,
        help="repository to check (defaults to the project root)",
    )
    args = parser.parse_args()

    errors = validate_source_staging(repo_root=args.repo_root.resolve())
    if errors:
        for error in errors:
            logger.error(error)
        return 1

    logger.info("Vercel source staging includes all build-time inputs")
    return 0


def validate_source_staging(repo_root: Path) -> list[str]:
    """Return errors found in the Vercel source staging configuration.

    Args:
        repo_root:
            Repository whose tracked files and ``.vercelignore`` are checked.

    Returns:
        A list of actionable validation errors. An empty list means that every
        tracked build-time input can be uploaded by a Vercel Git build.
    """
    ignore_path = repo_root / IGNORE_FILE
    if not ignore_path.is_file():
        return [f"missing {IGNORE_FILE}"]

    tracked_paths = _tracked_paths(repo_root=repo_root)
    build_inputs = _build_input_paths(tracked_paths=tracked_paths)
    spec = pathspec.PathSpec.from_lines(
        "gitwildmatch", ignore_path.read_text(encoding="utf-8").splitlines()
    )

    errors: list[str] = []
    for path in sorted(REQUIRED_FILES - tracked_paths):
        errors.append(f"required build input is not tracked: {path}")

    for path in sorted(build_inputs):
        if spec.match_file(path):
            errors.append(f"build input is ignored by {IGNORE_FILE}: {path}")
        errors.extend(_ignored_parent_errors(spec=spec, path=path))

    return errors


def _build_input_paths(tracked_paths: set[str]) -> set[str]:
    build_inputs = set(REQUIRED_FILES)
    build_inputs.update(
        path
        for path in tracked_paths
        if path.startswith(FRONTEND_ROOT) and not path.startswith(GENERATED_CSV_ROOT)
    )
    build_inputs.update(
        path
        for path in tracked_paths
        if path.startswith(PYTHON_SOURCE_ROOT) and path.endswith(".py")
    )
    return build_inputs


def _ignored_parent_errors(spec: pathspec.PathSpec, path: str) -> list[str]:
    path_errors = []
    for parent in Path(path).parents:
        parent_name = parent.as_posix()
        if parent_name == ".":
            continue
        if spec.match_file(f"{parent_name}/"):
            path_errors.append(
                f"parent directory of build input is ignored by {IGNORE_FILE}: "
                f"{parent_name}/ (needed by {path})"
            )
    return path_errors


def _tracked_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return {path for path in result.stdout.decode(encoding="utf-8").split("\0") if path}


if __name__ == "__main__":
    raise SystemExit(main())
