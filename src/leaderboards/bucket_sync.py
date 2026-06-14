"""Sync Hugging Face buckets for EuroEval results.

Uses the `hf sync` CLI to sync raw and processed results buckets
to local directories. Also provides backup functionality.
"""

import collections.abc as c
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

from .backup import backup_results
from .paths import PROCESSED_RESULTS_DIR, RAW_RESULTS_DIR

load_dotenv()

logger = logging.getLogger(__name__)

HF_RAW_BUCKET = "EuroEval/raw-results"
HF_PROCESSED_BUCKET = "EuroEval/processed-results"


def sync_bucket() -> None:
    """Sync both HF buckets (raw and processed) using hf sync.

    Syncs from bucket to local directory using the official hf CLI.
    Creates local directories if needed.

    HF_TOKEN is loaded from .env by load_dotenv() at module import.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Cannot sync from bucket.")
        return

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Syncing raw bucket %s → %s...", HF_RAW_BUCKET, RAW_RESULTS_DIR)
    result = subprocess.run(
        ["hf", "sync", f"hf://buckets/{HF_RAW_BUCKET}/", str(RAW_RESULTS_DIR)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        logger.warning("hf sync failed for raw bucket: %s", result.stderr)
    else:
        logger.info("Synced raw bucket: %s", result.stdout.strip())

    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Syncing processed bucket %s → %s...",
        HF_PROCESSED_BUCKET,
        PROCESSED_RESULTS_DIR,
    )
    result = subprocess.run(
        [
            "hf",
            "sync",
            f"hf://buckets/{HF_PROCESSED_BUCKET}/",
            str(PROCESSED_RESULTS_DIR),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        logger.warning("hf sync failed for processed bucket: %s", result.stderr)
    else:
        logger.info("Synced processed bucket: %s", result.stdout.strip())


@contextmanager
def sync_bucket_context() -> c.Generator[Path, None, None]:
    """Context manager for HF bucket sync.

    Syncs buckets on entry, logs completion on exit.

    Yields:
        Path: Raw results directory (for backwards compatibility)
    """
    logger.info("Syncing buckets...")
    sync_bucket()
    yield RAW_RESULTS_DIR
    logger.info("Bucket sync complete.")


@contextmanager
def sync_bucket_with_backup() -> c.Generator[Path, None, None]:
    """Context manager for HF bucket sync with automatic backup.

    Syncs buckets on entry, creates backup on exit.

    Yields:
        Path: Raw results directory (for backwards compatibility)
    """
    logger.info("Syncing buckets...")
    sync_bucket()
    yield RAW_RESULTS_DIR
    backup_path = backup_results()
    if backup_path:
        logger.info("Backup created at %s.", backup_path)
    logger.info("Bucket sync complete.")


def is_sync_available() -> bool:
    """Check if hf CLI is available for syncing.

    Returns:
        True if hf binary is on PATH, False otherwise.
    """
    result = subprocess.run(["which", "hf"], capture_output=True, check=False)
    return result.returncode == 0


# Alias for backwards compatibility
create_backup = backup_results
