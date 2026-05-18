"""Path constants for the leaderboard generation pipeline.

Centralises the locations of config files, data inputs, and the output
CSV directory so the rest of the package doesn't depend on the caller's
cwd. Data-input paths are overridable via environment variables so
operators can keep large files outside the repo.
"""

import os
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


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Historical archive of all benchmark records. Operators provide this file;
# it is not tracked in git (43+ MB compressed).
RESULTS_PATH: Path = _env_path("EUROEVAL_RESULTS_PATH", REPO_ROOT / "results.tar.gz")

# Incremental jsonl of new benchmark records to fold into RESULTS_PATH.
NEW_RESULTS_PATH: Path = _env_path(
    "EUROEVAL_NEW_RESULTS_PATH", REPO_ROOT / "new_results.jsonl"
)

# Off-repo backup location for snapshots of results.tar.gz. The backup
# rotation keeps the total directory size under BACKUPS_MAX_BYTES.
BACKUPS_DIR: Path = _env_path(
    "EUROEVAL_RESULTS_BACKUP_DIR",
    Path.home() / "pCloud Drive" / "data" / "euroeval_backup",
)
BACKUPS_MAX_BYTES: int = 1_000_000_000  # ~1 GB
