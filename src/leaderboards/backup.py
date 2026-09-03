"""Off-repo backup rotation for the results directory.

`RESULTS_DIR` holds a tree of JSON records (results/<model>/<record>.json)
and is the source of truth for the leaderboard pipeline. We don't track it
in git (tens of MB and growing), so this module snapshots each successful
run to BACKUPS_DIR as a single compressed archive with a timestamp suffix,
copies that archive to the Jottacloud Archive under BACKUPS_ARCHIVE_DIR, and
deletes the local snapshots whose off-machine copy is confirmed -- so the
local directory holds one snapshot rather than growing without bound.
`BACKUPS_MAX_BYTES` still caps the local copies when no Archive is reachable.

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
import shutil
import subprocess
import tarfile
import typing as t
from pathlib import Path

from .constants import (
    ARCHIVE_LS_TIMEOUT,
    BACKUP_ARCHIVE_ROOT,
    BACKUP_HASH_LEN,
    BACKUP_PREFIX,
    BACKUP_SUFFIX,
    BACKUPS_ARCHIVE_DIR,
    BACKUPS_ARCHIVE_TIMEOUT,
    BACKUPS_DIR,
    BACKUPS_MAX_BYTES,
    RESULTS_DIR,
)
from .eee_validation import is_eee_record

logger = logging.getLogger(__name__)


def backup_results(source: Path = RESULTS_DIR) -> Path | None:
    """Snapshot `source` into BACKUPS_DIR, then prune oldest if over cap.

    Validates that results exist and have valid JSON structure before creating
    the backup. Note: raw results may be missing the "precious" metadata fields
    (commercially_licensed, open, trained_from_scratch); those are filled in
    later by ``add_missing_entries`` and enforced for processed output only.

    Skips if `source`'s contents are unchanged since the newest existing backup,
    so repeated runs without changes don't fill the backup directory.

    Once a snapshot is confirmed to be stored in the Jottacloud Archive, older
    local snapshots with the same guarantee are deleted, keeping `BACKUPS_DIR`
    bounded at roughly one snapshot instead of growing with every run. The
    Archive keeps the full history.

    Args:
        source (optional):
            The results directory to back up. Defaults to RESULTS_DIR.

    Returns:
        The Path of the new backup, or None if nothing was written. The file
        itself may have been removed once archived off-machine; its name
        identifies the object under ``Archive/<BACKUPS_ARCHIVE_DIR>/``.
    """
    # Validate results before backing up
    _validate_results()

    backup_path = _write_snapshot(source=source)
    if backup_path is not None and _archive_offsite(backup_path):
        _remove_archived_local(keep=backup_path)
    return backup_path


def _archive_offsite(backup_path: Path) -> bool:
    """Copy a snapshot into the Jottacloud Archive namespace.

    Best-effort by design: losing the off-site copy is worth a warning, not a
    failed leaderboard run, since the snapshot itself is already on disk.

    Success is confirmed by listing the Archive rather than trusting the exit
    code, so a client that exits before the bytes land cannot trick us into
    deleting the only local copy.

    Args:
        backup_path:
            The snapshot to upload.

    Returns:
        True if the snapshot is verifiably stored in the Archive, False if it
        only exists locally.
    """
    cli = _jotta_cli()
    if cli is None:
        logger.warning(
            f"Archived the results backup to {backup_path} only; install the "
            "Jottacloud command-line tool to keep a copy off-machine."
        )
        return False
    try:
        result = subprocess.run(
            [
                str(cli),
                "archive",
                str(backup_path),
                f"--remote={BACKUPS_ARCHIVE_DIR}/{backup_path.name}",
                # Without --nogui the client insists on a terminal and dies with
                # "open /dev/tty: device not configured" when run unattended.
                "--nogui",
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=BACKUPS_ARCHIVE_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not archive the results backup to Jottacloud: {exc}")
        return False
    if result.returncode != 0:
        logger.warning(
            f"Could not archive the results backup to Jottacloud: "
            f"{(result.stderr or result.stdout).strip()[:200]}"
        )
        return False
    if not _is_archived(backup_path):
        logger.warning(
            f"Jottacloud accepted {backup_path.name} but it is not listed under "
            f"Archive/{BACKUPS_ARCHIVE_DIR}/; keeping the local copy"
        )
        return False
    logger.info(
        f"Archived {backup_path.name} to Jottacloud Archive/{BACKUPS_ARCHIVE_DIR}/"
    )
    return True


def _is_archived(backup_path: Path) -> bool:
    """Check whether `backup_path` is stored under the Archive directory.

    Matches on filename, and on byte size when the Archive reports one, so an
    incomplete upload is not mistaken for a finished one.

    Args:
        backup_path:
            The local snapshot to look for.

    Returns:
        True if an off-machine copy is present.
    """
    size = backup_path.stat().st_size if backup_path.exists() else None
    return any(
        entry["name"] == backup_path.name
        and (size is None or entry["size"] is None or entry["size"] == size)
        for entry in _archived_backups()
    )


def _archived_backups() -> list[dict[str, t.Any]]:
    """List the snapshots stored under ``Archive/<BACKUPS_ARCHIVE_DIR>``.

    Returns:
        Newest-first entries with "name", "size" and "modified" keys. Empty when
        the Archive is unreachable or the client is missing, which callers read
        as "nothing stored off-machine" rather than as an error.
    """
    cli = _jotta_cli()
    if cli is None:
        return []
    try:
        result = subprocess.run(
            [str(cli), "ls", f"Archive/{BACKUPS_ARCHIVE_DIR}", "--json"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=ARCHIVE_LS_TIMEOUT,
        )
        if result.returncode != 0:
            return []
        listing = json.loads(result.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not list the Jottacloud Archive: {exc}")
        return []
    entries: list[dict[str, t.Any]] = []
    for entry in listing.get("Files", []):
        name = entry.get("Name") if isinstance(entry, dict) else None
        if not isinstance(name, str):
            continue
        if not name.startswith(BACKUP_PREFIX) or not name.endswith(BACKUP_SUFFIX):
            continue
        entries.append(
            {
                "name": name,
                "size": entry.get("Size"),
                "modified": entry.get("Modified") or 0,
            }
        )
    return sorted(entries, key=lambda entry: entry["modified"], reverse=True)


def _jotta_cli() -> Path | None:
    """Locate the Jottacloud command-line client, if it is installed.

    Returns:
        Path to the client, or None when it is not installed.
    """
    found = shutil.which("jotta-cli")
    if found is not None:
        return Path(found)
    bundled = Path("/Applications/Jottacloud.app/Contents/MacOS/jotta-cli")
    return bundled if bundled.exists() else None


def _remove_archived_local(keep: Path) -> int:
    """Delete local snapshots that are verifiably stored in the Archive.

    `keep` is retained even though it is archived: it is what
    ``restore_from_backup_if_missing`` extracts, which spares a machine that
    lost its results directory a full download from the Archive.

    Args:
        keep:
            The snapshot to leave in place.

    Returns:
        The number of local snapshots removed.
    """
    archived = {entry["name"] for entry in _archived_backups()}
    removed = 0
    for old in _list_backups():
        if old == keep or old.name not in archived:
            continue
        size = old.stat().st_size
        old.unlink(missing_ok=True)
        logger.info(
            f"Removed local snapshot {old.name} ({size:,} bytes); an off-machine "
            "copy is in the Jottacloud Archive"
        )
        removed += 1
    if removed:
        logger.info(
            f"Kept {keep.name} locally for fast restores "
            f"({keep.stat().st_size:,} bytes)"
        )
    return removed


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
                logger.error(f"Missing EEE envelope in {json_file}, model '{model_id}'")
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


def restore_from_backup_if_missing(target: Path = RESULTS_DIR) -> bool:
    """Restore `target` from the newest backup if it's missing or empty.

    Args:
        target (optional):
            The results directory to populate. Defaults to RESULTS_DIR.

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
