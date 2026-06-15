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
import logging
import shutil
from pathlib import Path

from .paths import BACKUPS_DIR, BACKUPS_MAX_BYTES, MAX_BACKUPS, RESULTS_PATH

logger = logging.getLogger(__name__)

BACKUP_PREFIX = "results_"
BACKUP_SUFFIX = ".tar.gz"


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

    Skips if `source` is byte-identical to the newest existing backup, so
    repeated runs without changes don't fill the backup directory.

    Args:
        source:
            The file to back up (defaults to RESULTS_PATH).

    Returns:
        The Path of the new backup, or None if nothing was written.
    """
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
        f"Snapshotted {source.name} → {backup_path} "
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
    """Delete oldest backups until under BACKUPS_MAX_BYTES and MAX_BACKUPS.

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

    # Prune by count first (keep at most MAX_BACKUPS)
    while len(backups) > MAX_BACKUPS:
        oldest = backups[-1]  # Oldest is at the end
        # Never delete the previous-day backup if it's the only one from yesterday
        if oldest is previous_day_backup and _is_only_previous_day_backup(
            backups, previous_day_backup
        ):
            logger.info(
                f"Skipping prune of {oldest.name} - last backup from previous day"
            )
            break
        backups.pop()
        size = oldest.stat().st_size
        oldest.unlink()
        logger.info(
            f"Pruned old backup {oldest.name} ({size:,} bytes) - over count limit"
        )

    # Then prune by size if still over cap
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
