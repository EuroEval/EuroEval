"""Manage hf-mount for EuroEval results bucket.

Provides context manager for mounting/unmounting the HF bucket, with
automatic backup creation to pCloud.
"""

import collections.abc as c
import io
import logging
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
    REPO_ROOT,
)

load_dotenv()

# Raw results bucket mount point - configured to use persistent directory
HF_RAW_BUCKET = "EuroEval/raw-results"  # Bucket ID for hf-mount-nfs (no 'buckets/' prefix)
MOUNT_POINT = RAW_RESULTS_DIR  # Mount directly to results/raw/

# Processed results bucket mount point - configured to use persistent directory
HF_PROCESSED_BUCKET = "EuroEval/processed-results"  # Bucket ID for hf-mount-nfs
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


def mount_bucket() -> None:
    """Mount both HF buckets (raw and processed) using hf-mount daemon.

    The daemon persists independently and handles reconnection automatically.
    Creates mount point directories if needed.

    HF_TOKEN is loaded from .env by load_dotenv() at module import.
    """
    # Mount raw results bucket
    MOUNT_POINT.mkdir(parents=True, exist_ok=True)
    if MOUNT_POINT.is_mount():
        print(f"✓ Raw bucket already mounted at {MOUNT_POINT}")
    else:
        print(f"Starting hf-mount daemon for {HF_RAW_BUCKET}...")
        result = subprocess.run(
            ["hf-mount", "start", HF_RAW_BUCKET, str(MOUNT_POINT)],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print(f"⚠ hf-mount start failed: {result.stderr}")
        else:
            print(f"✓ hf-mount daemon started for {MOUNT_POINT}")

    # Mount processed results bucket
    PROCESSED_MOUNT_POINT.mkdir(parents=True, exist_ok=True)
    if PROCESSED_MOUNT_POINT.is_mount():
        print(f"✓ Processed bucket already mounted at {PROCESSED_MOUNT_POINT}")
    else:
        print(f"Starting hf-mount daemon for {HF_PROCESSED_BUCKET}...")
        result = subprocess.run(
            ["hf-mount", "start", HF_PROCESSED_BUCKET, str(PROCESSED_MOUNT_POINT)],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print(f"⚠ hf-mount start failed: {result.stderr}")
        else:
            print(f"✓ hf-mount daemon started for {PROCESSED_MOUNT_POINT}")


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
