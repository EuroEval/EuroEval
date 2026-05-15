"""Path constants for the leaderboard generation pipeline.

Centralises the locations of config files and the output CSV directory so
that the rest of the package doesn't depend on the caller's cwd.
"""

from pathlib import Path

# This package's directory (`src/leaderboards/`).
PACKAGE_DIR: Path = Path(__file__).resolve().parent

# Per-language task → dataset configuration.
CONFIGS_DIR: Path = PACKAGE_DIR / "configs"

# Task → primary/secondary metric mapping.
TASK_CONFIG_PATH: Path = PACKAGE_DIR / "task_config.yaml"

# Repository root (assumes the package lives at <repo>/src/leaderboards/).
REPO_ROOT: Path = PACKAGE_DIR.parent.parent

# Generated CSVs are written directly into the frontend bundle.
OUTPUT_DIR: Path = REPO_ROOT / "src" / "frontend" / "csv"
