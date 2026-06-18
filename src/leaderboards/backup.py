"""Off-repo backup rotation for the results directory.

`RESULTS_DIR` holds one JSONL file per model and is the source of truth for
the leaderboard pipeline. We don't track it in git (tens of MB and growing),
so this module snapshots each successful run to BACKUPS_DIR as a single
compressed archive with a timestamp suffix, and prunes the oldest backups
whenever the directory exceeds BACKUPS_MAX_BYTES.

If `RESULTS_DIR` is missing or empty at startup,
`restore_from_backup_if_missing` extracts the most recent backup into place so
the pipeline can run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import sys
import tarfile
from pathlib import Path

from .constants import (
    BACKUPS_DIR,
    BACKUPS_MAX_BYTES,
    REQUIRED_METADATA_FIELDS,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

BACKUP_PREFIX = "results_"
BACKUP_SUFFIX = ".tar.gz"
# Length of the content-hash slug embedded in each backup filename, used to
# skip writing a new snapshot when the results haven't changed.
_HASH_LEN = 12
# Directory prefix used for per-model files inside each backup archive,
# mirroring the layout of RESULTS_DIR.
_ARCHIVE_ROOT = "results"


def restore_from_backup_if_missing(target: Path = RESULTS_DIR) -> bool:
    """Restore `target` from the newest backup if it's missing or empty.

    Args:
        target:
            The results directory to populate (defaults to RESULTS_DIR).

    Returns:
        True if a restore happened, False if `target` already held results or
        no backup was available.
    """
    if target.exists() and any(target.glob("*.jsonl")):
        return False
    backups = _list_backups()
    if not backups:
        return False
    newest = backups[0]
    logger.info(f"Restoring {target} from backup {newest.name}")
    _extract_backup(archive=newest, dest=target)
    return True


def backup_results(source: Path = RESULTS_DIR) -> Path | None:
    """Snapshot `source` into BACKUPS_DIR, then prune oldest if over cap.

    Validates that results exist and have required metadata fields before
    creating the backup. Skips if `source`'s contents are unchanged since the
    newest existing backup, so repeated runs without changes don't fill the
    backup directory.

    Args:
        source:
            The results directory to back up (defaults to RESULTS_DIR).

    Returns:
        The Path of the new backup, or None if nothing was written.

    Raises:
        OSError:
            If the backup directory (a pCloud Drive path) is unavailable and
            stdin is not a TTY, so the operator cannot be prompted to retry.
    """
    # Validate results before backing up
    _validate_results()

    # The backup directory lives on pCloud Drive, which raises OSError when
    # pCloud is not running. When attached to a terminal, prompt the operator
    # to start pCloud and retry; otherwise (CI) let the OSError propagate so
    # the caller's non-interactive safety net handles it.
    while True:
        try:
            return _write_snapshot(source=source)
        except OSError as exc:
            if not sys.stdin.isatty():
                raise
            logger.warning(f"Backup failed; pCloud may be unavailable: {exc}")
            input(
                f"Could not write the backup to {BACKUPS_DIR}. pCloud appears to "
                "be unavailable. Start pCloud and press Enter to retry..."
            )


def _write_snapshot(source: Path) -> Path | None:
    """Snapshot `source` into BACKUPS_DIR, pruning oldest if over cap.

    Args:
        source:
            The results directory to back up.

    Returns:
        The Path of the new backup, or None if the newest existing backup
        already captures identical results.
    """
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(source.glob("*.jsonl"))
    content_hash = _content_hash(paths=model_files)

    existing = _list_backups()
    if existing and _backup_hash(existing[0]) == content_hash:
        logger.info(
            f"Newest backup {existing[0].name} already captures the current "
            f"results ({content_hash}); skipping snapshot."
        )
        return None

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        BACKUPS_DIR / f"{BACKUP_PREFIX}{timestamp}_{content_hash}{BACKUP_SUFFIX}"
    )
    with tarfile.open(backup_path, "w:gz") as tar:
        for model_file in model_files:
            tar.add(name=model_file, arcname=f"{_ARCHIVE_ROOT}/{model_file.name}")
    logger.info(
        f"Snapshotted {len(model_files):,} files from {source} -> {backup_path} "
        f"({backup_path.stat().st_size:,} bytes)"
    )

    _prune_backups()
    return backup_path


def _content_hash(paths: list[Path]) -> str:
    """Return a short stable hash of the given files' names and contents.

    Args:
        paths:
            The files to hash, in a stable (sorted) order.

    Returns:
        The first `_HASH_LEN` hex characters of a SHA-256 over each file's
        name and bytes.
    """
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:_HASH_LEN]


def _backup_hash(backup: Path) -> str | None:
    """Extract the content-hash slug from a backup filename.

    Args:
        backup:
            A backup path named ``results_<timestamp>_<hash>.tar.gz``.

    Returns:
        The embedded hash, or None if the name doesn't carry one (e.g. a
        legacy backup written before content hashing).
    """
    stem = backup.name[len(BACKUP_PREFIX) : -len(BACKUP_SUFFIX)]
    candidate = stem.rpartition("_")[2]
    return candidate if len(candidate) == _HASH_LEN else None


def _extract_backup(archive: Path, dest: Path) -> None:
    """Extract the per-model JSONL files from `archive` into `dest`.

    Members are flattened to their base name so the archive's internal
    directory prefix can't write outside `dest`.

    Args:
        archive:
            The ``.tar.gz`` backup to read.
        dest:
            The directory to populate with the per-model JSONL files.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not member.isfile() or not name.endswith(".jsonl"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            (dest / name).write_bytes(extracted.read())


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
                    additional = record.get("model_info", {}).get(
                        "additional_details", {}
                    )
                    for field in REQUIRED_METADATA_FIELDS:
                        if field not in additional:
                            model_id = record.get("model_info", {}).get(
                                "name", "unknown"
                            )
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
        if old is previous_day_backup and _is_only_previous_day_backup(backups):
            logger.info(f"Skipping prune of {old.name} - last backup from previous day")
            continue
        size = old.stat().st_size
        old.unlink()
        total -= size
        logger.info(f"Pruned old backup {old.name} ({size:,} bytes) - over size limit")


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


def _is_only_previous_day_backup(backups: list[Path]) -> bool:
    """Check if there is exactly one backup from a previous day.

    Args:
        backups:
            List of all backups (newest first).

    Returns:
        True if exactly one backup is from a previous day, False otherwise.
    """
    today = dt.datetime.now().date()
    previous_day_count = sum(
        1
        for backup in backups
        if dt.datetime.fromtimestamp(backup.stat().st_mtime).date() < today
    )
    return previous_day_count == 1
