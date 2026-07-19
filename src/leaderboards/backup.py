"""Off-repo backup rotation for the results directory.

`RESULTS_DIR` holds a tree of JSON records (results/<model>/<record>.json)
and is the source of truth for the leaderboard pipeline. We don't track it
in git (tens of MB and growing), so this module snapshots each successful
run to BACKUPS_DIR as a single compressed archive with a timestamp suffix,
and prunes the oldest backups whenever the directory exceeds BACKUPS_MAX_BYTES.

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
    BACKUP_ARCHIVE_ROOT,
    BACKUP_HASH_LEN,
    BACKUP_PREFIX,
    BACKUP_SUFFIX,
    BACKUPS_DIR,
    BACKUPS_MAX_BYTES,
    RESULTS_DIR,
)
from .eee_validation import is_eee_record

logger = logging.getLogger(__name__)


def restore_from_backup_if_missing(target: Path = RESULTS_DIR) -> bool:
    """Restore `target` from the newest backup if it's missing or empty.

    Args:
        target:
            The results directory to populate (defaults to RESULTS_DIR).

    Returns:
        True if a restore happened, False if `target` already held results or
        no backup was available.
    """
    # Check for tree layout: any .json files in subdirs
    if target.exists() and any(target.rglob("*.json")):
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

    Validates that results exist and have valid JSON structure before creating
    the backup. Note: raw results may be missing the "precious" metadata fields
    (commercially_licensed, open, trained_from_scratch); those are filled in
    later by ``add_missing_entries`` and enforced for processed output only.

    Skips if `source`'s contents are unchanged since the newest existing backup,
    so repeated runs without changes don't fill the backup directory.

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

    # Recursively find all JSON files in the tree (results/<model>/<record>.json)
    json_files = sorted(source.rglob("*.json"))
    content_hash = _content_hash(paths=json_files)

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
        for json_file in json_files:
            # Preserve the tree structure: results/<model>/<record>.json
            rel_path = json_file.relative_to(source.parent)
            tar.add(name=json_file, arcname=str(rel_path))
    logger.info(
        f"Snapshotted {len(json_files):,} files from {source} -> {backup_path} "
        f"({backup_path.stat().st_size:,} bytes)"
    )

    _prune_backups()
    return backup_path


def _content_hash(paths: list[Path]) -> str:
    """Return a short stable hash of the given files' names and contents.

    Args:
        paths:
            The files to hash. Will be sorted by relative path for stable
            ordering regardless of input order.

    Returns:
        The first `BACKUP_HASH_LEN` hex characters of a SHA-256 over each
        file's relative path and bytes.
    """
    hasher = hashlib.sha256()
    # Compute relative paths and sort by them for stable ordering

    def _rel_path(p: Path) -> str:
        return str(p.relative_to(p.parent.parent))

    sorted_paths = sorted(paths, key=_rel_path)
    for path in sorted_paths:
        # Use the relative path from results root for stable identity
        # (e.g., "model_name/dataset__split__shot.json")
        rel_path = _rel_path(path)
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()[:BACKUP_HASH_LEN]


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
    return candidate if len(candidate) == BACKUP_HASH_LEN else None


def _extract_backup(archive: Path, dest: Path) -> None:
    """Extract the per-record JSON tree from `archive` into `dest`.

    Preserves the tree structure: `results/<model>/<record>.json`.
    Path traversal is prevented by ensuring all members live under the
    archive root prefix.

    Args:
        archive:
            The ``.tar.gz`` backup to read.
        dest:
            The directory to populate with the per-record JSON tree.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            # Only process files under the results/ prefix
            if not member.name.startswith(f"{BACKUP_ARCHIVE_ROOT}/"):
                continue
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            # Strip the "results/" prefix to get relative path within dest
            rel_path = Path(member.name).relative_to(BACKUP_ARCHIVE_ROOT)
            target_path = dest / rel_path
            # Ensure we don't write outside dest (path traversal protection)
            try:
                target_path.relative_to(dest)
            except ValueError:
                logger.warning(f"Skipping {member.name} - path traversal detected")
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target_path.write_bytes(extracted.read())


def _validate_results() -> None:
    """Validate that results exist and have valid JSON structure.

    Checks that:
    1. RESULTS_DIR exists and contains JSON files in the tree layout
       (results/<model>/<record>.json)
    2. Sample files (up to 5) contain valid JSON with EEE envelope structure

    Each file contains a single JSON dict (one record per file). Raw results
    synced from the HF bucket may be missing the "precious" metadata fields
    (commercially_licensed, open, trained_from_scratch). Those fields are
    filled in later by ``add_missing_entries`` and enforced when processed
    results are written out by ``dump_jsonl_records``.

    Raises:
        FileNotFoundError:
            If RESULTS_DIR doesn't exist or contains no files.
        ValueError:
            If any result is malformed JSON or lacks EEE envelope structure.
    """
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_DIR}. "
            "Run evaluation result collection before backing up."
        )

    # Find all JSON files in the tree (results/<model>/<record>.json)
    json_files = list(RESULTS_DIR.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No result files found in {RESULTS_DIR}. "
            "Run evaluation result collection before backing up."
        )

    # Sample up to 5 files for validation
    sample_size = min(5, len(json_files))
    sampled_files = random.sample(json_files, sample_size)

    logger.info(
        f"Validating {sample_size} sampled result files (of {len(json_files):,} total)"
        " for valid JSON and EEE envelope structure"
    )

    files_with_issues = 0
    records_checked = 0
    records_with_issues = 0

    for json_file in sampled_files:
        file_has_issues = False
        try:
            content = json_file.read_text(encoding="utf-8")
            record = json.loads(content)
            records_checked += 1

            # Validate EEE envelope structure (required for all records)
            if not is_eee_record(record):
                model_id = record.get("model_info", {}).get("name", "unknown")
                logger.error(
                    f"Missing EEE envelope in {json_file}, model '{model_id}'"
                )
                file_has_issues = True
                records_with_issues += 1

        except Exception as e:
            logger.error(f"Failed to read {json_file}: {e}")
            file_has_issues = True

        if file_has_issues:
            files_with_issues += 1

    if files_with_issues > 0:
        msg = (
            f"{files_with_issues} sampled files have structural issues. Checked "
            f"{records_checked:,} records, found {records_with_issues:,} with issues. "
            "Re-run evaluation to regenerate results."
        )
        raise ValueError(msg)

    logger.info(
        f"Validated {records_checked:,} records from {len(sampled_files):,} sampled"
        f" files - all have valid JSON and EEE structure"
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
    """List the backup archives in BACKUPS_DIR, newest first.

    Returns:
        The backup paths matching the ``results_*.tar.gz`` naming, sorted by
        modification time with the newest first. Empty if BACKUPS_DIR is
        missing.
    """
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
