"""Manage hf-mount for EuroEval results bucket.

Provides context manager for mounting/unmounting the HF bucket, with
automatic backup creation to pCloud.
"""

import collections.abc as c
import io
import os
import subprocess
import tarfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .paths import (
    BACKUPS_DIR,
    BACKUPS_MAX_BYTES,
    PROCESSED_RESULTS_DIR,
    RAW_RESULTS_DIR,
)

load_dotenv()

# Raw results bucket mount point - configured to use persistent directory
HF_RAW_BUCKET = "EuroEval/raw-results"  # Bucket ID for hf-mount (no 'buckets/' prefix)
MOUNT_POINT = RAW_RESULTS_DIR  # Mount directly to results/raw/

# Processed results bucket mount point - configured to use persistent directory
HF_PROCESSED_BUCKET = "EuroEval/processed-results"  # Bucket ID for hf-mount
PROCESSED_MOUNT_POINT = PROCESSED_RESULTS_DIR  # Mount directly to results/processed/


def is_hf_mount_available() -> bool:
    """Check if hf-mount is installed and usable.

    Returns:
        True if hf-mount binary is on PATH, False otherwise.
    """
    try:
        result = subprocess.run(["which", "hf-mount"], capture_output=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def sync_bucket() -> None:
    """Sync both HF buckets (raw and processed) using hf sync.

    Syncs from bucket to local directory using the official hf CLI.
    Creates local directories if needed.

    HF_TOKEN is loaded from .env by load_dotenv() at module import.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠ HF_TOKEN not set. Cannot sync from bucket.")
        return

    # Sync raw results bucket
    MOUNT_POINT.mkdir(parents=True, exist_ok=True)
    print(f"Syncing raw bucket {HF_RAW_BUCKET} → {MOUNT_POINT}...")
    result = subprocess.run(
        [
            "hf", "sync",
            f"hf://buckets/{HF_RAW_BUCKET}/",
            str(MOUNT_POINT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        print(f"⚠ hf sync failed for raw bucket: {result.stderr}")
    else:
        print(f"✓ Synced raw bucket: {result.stdout.strip()}")

    # Sync processed results bucket
    PROCESSED_MOUNT_POINT.mkdir(parents=True, exist_ok=True)
    print(f"Syncing processed bucket {HF_PROCESSED_BUCKET} → {PROCESSED_MOUNT_POINT}...")
    result = subprocess.run(
        [
            "hf", "sync",
            f"hf://buckets/{HF_PROCESSED_BUCKET}/",
            str(PROCESSED_MOUNT_POINT),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        print(f"⚠ hf sync failed for processed bucket: {result.stderr}")
    else:
        print(f"✓ Synced processed bucket: {result.stdout.strip()}")


def unmount_bucket() -> None:
    """Unmount the HF bucket if mounted."""
    if not MOUNT_POINT.exists():
        return

    try:
        # Check if mounted
        if not MOUNT_POINT.is_mount():
            return

        print(f"Unmounting {MOUNT_POINT}...")
        subprocess.run(["umount", str(MOUNT_POINT)], check=False)
        print("✓ Unmounted")
    except Exception as e:
        print(f"⚠ Unmount failed: {e}")


@contextmanager
def hf_mount_context() -> c.Generator[Path, None, None]:
    """Context manager for HF bucket mount.

    Mounts on entry, unmounts on exit. Falls back gracefully if
    hf-mount is not available.

    Yields:
        Path: Mount point (even if mount failed, for code simplicity)
    """
    mounted = False
    try:
        if is_hf_mount_available():
            mount_bucket()
            mounted = True
        else:
            print("⚠ hf-mount not found. Use tar.gz fallback.")
        yield MOUNT_POINT
    finally:
        if mounted:
            unmount_bucket()


def create_backup() -> Path | None:
    """Create backup of mounted results directory.

    Creates timestamped tar.gz in BACKUPS_DIR, rotates old backups
    to stay under BACKUPS_MAX_BYTES.

    Returns:
        Path to backup file, or None if backup failed.
    """
    if not MOUNT_POINT.exists():
        print("⚠ Cannot backup: mount point does not exist")
        return None

    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    # Count files in mount
    file_count = len(list(MOUNT_POINT.glob("*.jsonl")))
    if file_count == 0:
        print("⚠ No results in mounted bucket to backup")
        return None

    backup_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    backup_path = BACKUPS_DIR / backup_name

    print(f"Creating backup of {file_count:,} model files...")

    try:
        with tarfile.open(backup_path, "w:gz") as tar:
            for jsonl_file in MOUNT_POINT.glob("*.jsonl"):
                tar.add(
                    jsonl_file,
                    arcname=f"results/results/{jsonl_file.name}",
                    recursive=False,
                )

            # Add empty processed results file
            processed_content = b""
            processed_tarinfo = tarfile.TarInfo(name="results/results.processed.jsonl")
            processed_tarinfo.size = len(processed_content)
            tar.addfile(
                tarinfo=processed_tarinfo, fileobj=io.BytesIO(processed_content)
            )

        size_mb = backup_path.stat().st_size / 1e6
        print(f"✓ Backup created: {backup_path} ({size_mb:.1f} MB)")

        # Rotate old backups
        _rotate_backups()

        return backup_path
    except Exception as e:
        print(f"⚠ Backup failed: {e}")
        return None


def _rotate_backups() -> None:
    """Remove oldest backups to stay under BACKUPS_MAX_BYTES."""
    backups = sorted(
        BACKUPS_DIR.glob("results_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,  # newest first
    )

    total_size = sum(b.stat().st_size for b in backups)

    while total_size > BACKUPS_MAX_BYTES and len(backups) > 1:
        oldest = backups.pop()  # remove oldest (last in list)
        oldest_size = oldest.stat().st_size
        oldest.unlink()
        total_size -= oldest_size
        print(f"Rotated out old backup: {oldest.name}")


@contextmanager
def hf_mount_with_backup() -> c.Generator[Path, None, None]:
    """Context manager with automatic backup after use.

    Mounts on entry, creates backup and unmounts on exit.

    Yields:
        Path: Mount point
    """
    mounted = False
    try:
        if is_hf_mount_available():
            mount_bucket()
            mounted = True
        yield MOUNT_POINT
    finally:
        # Create backup before unmounting
        if mounted:
            create_backup()
            unmount_bucket()
