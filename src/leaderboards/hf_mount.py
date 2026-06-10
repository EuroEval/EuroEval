"""Manage hf-mount for EuroEval results bucket.

Provides context manager for mounting/unmounting the HF bucket, with
automatic backup creation to pCloud.
"""

import io
import os
import subprocess
import tarfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .paths import BACKUPS_DIR, BACKUPS_MAX_BYTES

load_dotenv()

HF_RAW_BUCKET = "buckets/EuroEval/raw-results"

# Mount point configurable via env var, defaults to ~/.local/share/euroeval-results
# This directory should be:
# - .gitignore'd (add /euroeval-results or customize path)
# - On persistent storage (not tmpfs)
# - Optionally on pCloud/external drive for large datasets
MOUNT_POINT = Path(
    os.getenv("EUROEVAL_MOUNT_POINT", Path.home() / ".local" / "share" / "euroeval-results")
).expanduser()


def is_hf_mount_available() -> bool:
    """Check if hf-mount is installed and usable.

    Returns:
        True if hf-mount binary is on PATH, False otherwise.
    """
    try:
        result = subprocess.run(
            ["which", "hf-mount"], capture_output=True, check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def mount_bucket() -> None:
    """Mount the HF results bucket at the mount point.

    Uses NFS backend (no root required, works everywhere).
    Creates mount point directory if needed.
    """
    import os

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable required for hf-mount. "
            "Set it in your .env file or export it."
        )

    MOUNT_POINT.mkdir(parents=True, exist_ok=True)

    # Check if already mounted
    if MOUNT_POINT.is_mount():
        print(f"✓ Already mounted at {MOUNT_POINT}")
        return

    print(f"Mounting HF bucket {HF_RAW_BUCKET} at {MOUNT_POINT}...")

    cmd = [
        "hf-mount-nfs",
        "bucket",
        HF_RAW_BUCKET,
        str(MOUNT_POINT),
        "--hf-token",
        hf_token,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        # Try fallback to daemon mode
        print("Daemon mode failed, trying foreground check...")
        # Just warn and continue - will fall back to tar.gz
        print(f"⚠ hf-mount failed: {result.stderr}")
        return

    print(f"✓ Mounted at {MOUNT_POINT}")


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
        print(f"✓ Unmounted")
    except Exception as e:
        print(f"⚠ Unmount failed: {e}")


@contextmanager
def hf_mount_context():
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
            print("⚠ hf-mount not found on PATH. Use tar.gz fallback.")
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
            tar.addfile(tarinfo=processed_tarinfo, fileobj=io.BytesIO(processed_content))

        print(f"✓ Backup created: {backup_path} ({backup_path.stat().st_size / 1e6:.1f} MB)")

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
def hf_mount_with_backup():
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
