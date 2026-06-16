"""Sync Hugging Face bucket for EuroEval results.

Uses the `hf sync` CLI to sync the results bucket to the local
results directory. Also provides backup functionality.
"""

import collections.abc as c
import json
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

from euroeval.data_models import BenchmarkResult

from .backup import backup_results
from .paths import RESULTS_DIR

load_dotenv()

logger = logging.getLogger(__name__)

HF_RESULTS_BUCKET = "EuroEval/results"


def _sanitise_model_id(model_id: str) -> str:
    """Convert a model ID to a safe filename.

    Replaces forward slashes with underscores to create valid filenames
    for the bucket structure.

    Args:
        model_id:
            The model identifier (e.g. "meta-llama/Llama-2-7b").

    Returns:
        Safe filename (e.g. "meta-llama_Llama-2-7b").
    """
    return model_id.replace("/", "_")


def sync_bucket() -> None:
    """Sync HF results bucket using hf sync.

    Syncs from bucket to local directory using the official hf CLI.
    Creates local directory if needed.

    HF_TOKEN is loaded from .env by load_dotenv() at module import.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Cannot sync from bucket.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Syncing bucket %s → %s...", HF_RESULTS_BUCKET, RESULTS_DIR)
    result = subprocess.run(
        ["hf", "sync", f"hf://buckets/{HF_RESULTS_BUCKET}/", str(RESULTS_DIR)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        logger.warning("hf sync failed: %s", result.stderr)
    else:
        logger.info("Synced bucket: %s", result.stdout.strip())


@contextmanager
def sync_bucket_context() -> c.Generator[Path, None, None]:
    """Context manager for HF bucket sync.

    Syncs bucket on entry, logs completion on exit.

    Yields:
        Path: Results directory
    """
    logger.info("Syncing bucket...")
    sync_bucket()
    yield RESULTS_DIR
    logger.info("Bucket sync complete.")


@contextmanager
def sync_bucket_with_backup() -> c.Generator[Path, None, None]:
    """Context manager for HF bucket sync with automatic backup.

    Syncs bucket on entry, creates backup on exit.

    Yields:
        Path: Results directory
    """
    logger.info("Syncing bucket...")
    sync_bucket()
    yield RESULTS_DIR
    backup_path = backup_results()
    if backup_path:
        logger.info("Backup created at %s.", backup_path)
    logger.info("Bucket sync complete.")


def upload_results_to_bucket(results_file: Path) -> None:
    """Upload local results to the Hugging Face results bucket.

    Reads the merged results file, splits into per-model JSONL files,
    and syncs to the bucket using hf sync.

    Args:
        results_file:
            Path to the merged results file (euroeval_benchmark_results.jsonl).
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Cannot upload to bucket.")
        return

    if not results_file.exists():
        logger.warning(
            "Results file %s does not exist. Nothing to upload.", results_file
        )
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Group results by model
    results_by_model: dict[str, list[str]] = {}
    logger.info("Reading results from %s...", results_file)
    with results_file.open() as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    result = BenchmarkResult.from_dict(rec)
                    if result.model:
                        model_key = _sanitise_model_id(result.model)
                        if model_key not in results_by_model:
                            results_by_model[model_key] = []
                        results_by_model[model_key].append(line.strip())
                except Exception as e:
                    logger.debug("Skipping invalid record during upload: %s", e)

    if not results_by_model:
        logger.warning("No valid results found to upload.")
        return

    # Write per-model JSONL files
    logger.info(
        "Writing %s per-model files to %s...", len(results_by_model), RESULTS_DIR
    )
    for model_key, lines in results_by_model.items():
        model_file = RESULTS_DIR / f"{model_key}.jsonl"
        with model_file.open("w") as f:
            for line in lines:
                f.write(line + "\n")

    # Sync to bucket
    logger.info("Syncing local %s → bucket %s...", RESULTS_DIR, HF_RESULTS_BUCKET)
    result = subprocess.run(
        ["hf", "sync", str(RESULTS_DIR), f"hf://buckets/{HF_RESULTS_BUCKET}/"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_TOKEN": hf_token},
    )
    if result.returncode != 0:
        logger.warning("hf sync upload failed: %s", result.stderr)
    else:
        logger.info("Uploaded results to bucket: %s", result.stdout.strip())


def is_sync_available() -> bool:
    """Check if hf CLI is available for syncing.

    Returns:
        True if hf binary is on PATH, False otherwise.
    """
    result = subprocess.run(["which", "hf"], capture_output=True, check=False)
    return result.returncode == 0


# Alias for backwards compatibility
create_backup = backup_results
