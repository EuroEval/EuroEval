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

# Per-leaderboard language list. The datasets and task metadata are derived
# from `euroeval` at runtime (see `task_metadata.py`).
LEADERBOARD_CONFIGS_DIR: Path = PACKAGE_DIR / "leaderboard_configs"

# Config for the core-model list that drives which models we re-evaluate
# when datasets change (see `core_models.py`).
CORE_MODELS_CONFIG: Path = PACKAGE_DIR / "core_models.yaml"

# Persistent cache of model IDs the user has explicitly opted to keep in the
# results despite no URL being found, so we don't re-prompt every run.
MODELS_WITHOUT_URLS_CACHE: Path = PACKAGE_DIR / "models_without_urls_cache.yaml"

# Repository root (assumes the package lives at <repo>/src/leaderboards/).
REPO_ROOT: Path = PACKAGE_DIR.parent.parent

# Generated CSVs are written directly into the frontend bundle.
OUTPUT_DIR: Path = REPO_ROOT / "src" / "frontend" / "csv"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Historical archive of all benchmark records. Operators provide this file;
# it is not tracked in git (43+ MB compressed).
RESULTS_PATH: Path = REPO_ROOT / "results.tar.gz"

# Incremental jsonl of new benchmark records to fold into RESULTS_PATH.
NEW_RESULTS_PATH: Path = _env_path(
    "EUROEVAL_NEW_RESULTS_PATH", REPO_ROOT / "new_results.jsonl"
)

# Persistent results directories (HF bucket sync points, git-ignored)
RESULTS_DIR: Path = REPO_ROOT / "results"
RAW_RESULTS_DIR: Path = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR: Path = RESULTS_DIR / "processed"

# Note: `.euroeval_cache/` is now deprecated for results storage.

# Off-repo backup location for snapshots of results.tar.gz. The backup
# rotation keeps the total directory size under BACKUPS_MAX_BYTES.
BACKUPS_DIR: Path = _env_path(
    "EUROEVAL_RESULTS_BACKUP_DIR",
    Path.home() / "pCloud Drive" / "data" / "euroeval_backup",
)
BACKUPS_MAX_BYTES: int = 1_000_000_000  # ~1 GB
