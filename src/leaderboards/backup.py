"""Off-repo backup rotation for the results.tar.gz archive.

`results.tar.gz` is the historical record of every benchmark run and is
overwritten in place by each successful generation. We don't track it in
git (43+ MB and grows on every update), so this module snapshots each
successful run to BACKUPS_DIR with a timestamp suffix and prunes the
oldest backups whenever the directory exceeds BACKUPS_MAX_BYTES.

If `results.tar.gz` is missing at startup, `restore_from_backup_if_missing`
copies the most recent backup into place so the pipeline can run.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
import shutil
from pathlib import Path

from .paths import BACKUPS_DIR, BACKUPS_MAX_BYTES, RESULTS_DIR, RESULTS_PATH

logger = logging.getLogger(__name__)

# Required metadata fields that must be present in all processed results
REQUIRED_METADATA_FIELDS = ["commercially_licensed", "open", "trained_from_scratch"]

BACKUP_PREFIX = "results_"
BACKUP_SUFFIX = ".tar.gz"


def _validate_results() -> None:
    """Validate that results exist and have required metadata.

    Checks that:
    1. RESULTS_DIR exists and contains .jsonl files
    2. Sample files (up to 5) are checked for required metadata fields

    Raises:
        FileNotFoundError:
            If RESULTS_DIR doesn't exist or contains no files.
        ValueError:
            If any result is missing required metadata fields.
    """
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_DIR}. "
            "Run evaluation result collection before backing up."
        )

    model_files = list(RESULTS_DIR.glob("*.jsonl"))
    if not model_files:
        raise FileNotFoundError(
            f"No result files found in {RESULTS_DIR}. "
            "Run evaluation result collection before backing up."
        )

    # Sample up to 5 files for validation
    sample_size = min(5, len(model_files))
    sampled_files = random.sample(model_files, sample_size)

    logger.info(
        f"Validating {sample_size} sampled result files (of {len(model_files):,} total)"
        f" for required metadata fields: {REQUIRED_METADATA_FIELDS}"
    )

    files_with_issues = 0
    records_checked = 0
    records_with_issues = 0

    for model_file in sampled_files:
        file_has_issues = False
        try:
            with model_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    records_checked += 1
                    record = json.loads(line)

                    # Check for required metadata fields
                    for field in REQUIRED_METADATA_FIELDS:
                        if field not in record:
                            model_id = record.get("model", "unknown")
                            logger.error(
                                f"Missing '{field}' in {model_file.name}, line "
                                f"{line_num}, model '{model_id}'"
                            )
                            file_has_issues = True
                            records_with_issues += 1

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {model_file.name}: {e}")
            file_has_issues = True
        except Exception as e:
            logger.error(f"Failed to read {model_file.name}: {e}")
            file_has_issues = True

        if file_has_issues:
            files_with_issues += 1

    if files_with_issues > 0:
        msg = (
            f"{files_with_issues} sampled files have missing metadata. Checked "
            f"{records_checked:,} records, found {records_with_issues:,} with issues. "
            f"Required: {REQUIRED_METADATA_FIELDS}. Re-run evaluation with "
            f"metadata collection enabled."
        )
        raise ValueError(msg)

    logger.info(
        f"Validated {records_checked:,} records from {len(sampled_files):,} sampled"
        f" files - all have required metadata fields"
    )


def _is_only_previous_day_backup(backups: list[Path], candidate: Path) -> bool:
    """Check if `candidate` is the only backup from a previous day.

    Args:
        backups:
            List of all backups (newest first).
        candidate:
            The backup to check.

    Returns:
        True if this is the only backup from a previous day, False otherwise.
    """
    today = dt.datetime.now().date()
    previous_day_count = sum(
        1
        for backup in backups
        if dt.datetime.fromtimestamp(backup.stat().st_mtime).date() < today
    )
    return previous_day_count == 1


def _list_backups() -> list[Path]:
    if not BACKUPS_DIR.exists():
        return []
    backups = [
        p
        for p in BACKUPS_DIR.iterdir()
        if p.is_file()
        and p.name.startswith(BACKUP_PREFIX)
        and p.name.endswith(BACKUP_SUFFIX)
    ]
    # Newest first.
    backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return backups


def restore_from_backup_if_missing(target: Path = RESULTS_PATH) -> bool:
    """Restore `target` from the newest backup if `target` doesn't exist.

    Args:
        target:
            The destination path (defaults to RESULTS_PATH).

    Returns:
        True if a restore happened, False if `target` already existed or no
        backup was available.
    """
    if target.exists():
        return False
    backups = _list_backups()
    if not backups:
        return False
    newest = backups[0]
    logger.info(f"Restoring {target.name} from backup {newest}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src=newest, dst=target)
    return True


def backup_results(source: Path = RESULTS_PATH) -> Path | None:
    """Snapshot `source` into BACKUPS_DIR, then prune oldest if over cap.

    Validates that results exist and have required metadata fields
    before creating the backup. Skips if `source` is byte-identical to the
    newest existing backup, so repeated runs without changes don't fill the
    backup directory.

    Args:
        source:
            The file to back up (defaults to RESULTS_PATH).

    Returns:
        The Path of the new backup, or None if nothing was written.

    """
    # Validate results before backing up
    _validate_results()

    if not source.exists():
        logger.warning(f"Cannot back up {source}: file does not exist.")
        return None

    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    existing = _list_backups()
    if existing and existing[0].stat().st_size == source.stat().st_size:
        # Cheap check first; fall back to byte compare if sizes match.
        if _files_equal(existing[0], source):
            logger.info(
                f"Newest backup {existing[0].name} matches current results "
                f"({source.stat().st_size:,} bytes); skipping snapshot."
            )
            return None

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUPS_DIR / f"{BACKUP_PREFIX}{timestamp}{BACKUP_SUFFIX}"
    shutil.copy2(src=source, dst=backup_path)
    logger.info(
        f"Snapshotted {source.name} -> {backup_path} "
        f"({backup_path.stat().st_size:,} bytes)"
    )

    _prune_backups()
    return backup_path


def _files_equal(a: Path, b: Path, chunk_size: int = 1 << 20) -> bool:
    with a.open(mode="rb") as fa, b.open(mode="rb") as fb:
        while True:
            ca = fa.read(chunk_size)
            cb = fb.read(chunk_size)
            if ca != cb:
                return False
            if not ca:
                return True


def _prune_backups() -> None:
    """Delete oldest backups until under BACKUPS_MAX_BYTES.

    Always keeps at least one backup from a previous day (if any exist),
    ensuring it's possible to revert to a prior day's state.
    """
    backups = _list_backups()

    # Identify the newest backup from a previous day (if any exist).
    today = dt.datetime.now().date()
    previous_day_backup: Path | None = None
    for backup in backups:
        backup_date = dt.datetime.fromtimestamp(backup.stat().st_mtime).date()
        if backup_date < today:
            previous_day_backup = backup
            break

    # Prune by size if over cap
    total = sum(p.stat().st_size for p in backups)
    if total <= BACKUPS_MAX_BYTES:
        return

    # Walk from oldest forward, deleting until under cap. Always keep the
    # newest one and the previous-day backup (if it's the only one from yesterday).
    for old in list(reversed(backups)):
        if total <= BACKUPS_MAX_BYTES:
            break
        # Protect the last backup from a previous day
        if old is previous_day_backup and _is_only_previous_day_backup(
            backups, previous_day_backup
        ):
            logger.info(f"Skipping prune of {old.name} - last backup from previous day")
            continue
        size = old.stat().st_size
        old.unlink()
        total -= size
        logger.info(f"Pruned old backup {old.name} ({size:,} bytes) - over size limit")
