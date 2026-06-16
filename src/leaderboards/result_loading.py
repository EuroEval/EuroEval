"""Loading of results, to be converted into leaderboards."""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
import time
import typing as t
from functools import cache

from .backup import backup_results
from .bucket_sync import sync_bucket
from .paths import NEW_RESULTS_PATH, RESULTS_DIR, RESULTS_PATH

logger = logging.getLogger(__name__)


def _sync_results_from_bucket() -> None:
    """Sync HF bucket and rebuild results.tar.gz.

    Syncs the single EuroEval/results bucket to RESULTS_DIR, then rebuilds
    results.tar.gz from the per-model JSONL files and creates a backup.
    """
    _sync_buckets()


def _sync_buckets() -> None:
    """Sync HF bucket via hf sync, rebuild results.tar.gz, and backup.

    Syncs the single EuroEval/results bucket to RESULTS_DIR, rebuilds
    results.tar.gz, and creates a backup.

    Raises:
        FileNotFoundError:
            If sync fails and no local files exist.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sync from bucket
    logger.info("Syncing results from HF bucket...")
    sync_bucket()

    # Verify files exist
    file_count = len(list(RESULTS_DIR.glob("*.jsonl")))
    if file_count == 0:
        # Check if we have local files from previous run
        local_file_count = len(list(RESULTS_DIR.glob("*.jsonl")))
        if local_file_count > 0:
            logger.info(
                f"Sync returned 0 files. Using {local_file_count:,} local files "
                f"from {RESULTS_DIR}."
            )
        else:
            raise FileNotFoundError(
                "No results available. Sync failed and no local cache exists."
            )

    logger.info(f"Synced {file_count:,} model files to {RESULTS_DIR}.")

    # Rebuild results.tar.gz from mounted files
    _rebuild_results_tar_gz()

    # Create backup
    backup_path = backup_results()
    if backup_path:
        logger.info(f"Backup created at {backup_path}.")


def _rebuild_results_tar_gz() -> None:
    """Rebuild results.tar.gz from per-model JSONL files in RESULTS_DIR.

    Reads all *.jsonl files from RESULTS_DIR and merges them into a single
    results/results.jsonl inside results.tar.gz.

    Skips files that can't be read (NFS lazy-loading quirks).
    Logs progress every 100 files.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    n_files = len(model_files)
    logger.info(f"Reading {n_files:,} model files from mount point...")

    all_lines: set[str] = set()
    start = time.time()
    unreadable = 0
    for i, model_file in enumerate(model_files, 1):
        try:
            lines = {
                line
                for line in model_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            all_lines.update(lines)
        except (OSError, FileNotFoundError) as e:
            unreadable += 1
            logger.debug(f"Skipping {model_file.name}: {e}")

        if i % 100 == 0 or i == n_files:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            logger.info(
                f"  {i}/{n_files} files ({len(all_lines):,} lines) - {rate:.1f} files/s"
            )

    elapsed = time.time() - start
    logger.info(
        f"Loaded {len(all_lines):,} results from {n_files - unreadable:,}/{n_files:,} "
        f"files in {elapsed:.1f}s"
        + (f" ({unreadable} unreadable)" if unreadable else "")
    )

    if not all_lines:
        logger.warning("No results found in mount point. results.tar.gz will be empty.")

    # Build results.tar.gz with only results.jsonl
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RESULTS_PATH, "w:gz") as tar:
        content_bytes = "\n".join(sorted(all_lines)).encode(encoding="utf-8")
        tarinfo = tarfile.TarInfo(name="results/results.jsonl")
        tarinfo.size = len(content_bytes)
        fileobj = io.BytesIO(content_bytes)
        tar.addfile(tarinfo=tarinfo, fileobj=fileobj)

    logger.info(f"Rebuilt {RESULTS_PATH} with {len(all_lines):,} results.")


def load_raw_results() -> list[dict[str, t.Any]]:
    """Load all results from results.tar.gz.

    Loads all evaluation results from the unified results archive.
    Results are sourced from the single EuroEval/results bucket.
    No distinction is made between raw and processed results.

    Returns:
        All evaluation results.

    Raises:
        FileNotFoundError:
            If the results file is not found.
        ValueError:
            If the results file contains invalid JSON.
    """
    results_path = RESULTS_PATH

    # Sync from HF bucket to ensure we have all results
    if not results_path.exists():
        logger.info("results.tar.gz not found, syncing from bucket...")
        _sync_results_from_bucket()
        if not results_path.exists():
            raise FileNotFoundError(f"Results file {results_path} not found.")
    else:
        # Check if bucket has more results than our local file
        logger.info("Checking for newer results in bucket...")
        _sync_results_from_bucket()

    logger.info(f"Loading raw results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()
        logger.info(f"Loaded {len(result_lines):,} existing results.")

    # If there are new results, add them to the existing results
    new_results_path = NEW_RESULTS_PATH
    if new_results_path.exists():
        with new_results_path.open() as f:
            new_result_lines = f.read().splitlines()
        result_lines.extend(new_result_lines)
        logger.info(f"Loaded {len(new_result_lines):,} new results.")
        new_results_path.unlink()

    # Parse each line as JSON, skipping empty lines
    records = list()
    for line_idx, line in enumerate(result_lines):
        if not line.strip():
            continue

        # We split on '}{' to handle cases where multiple JSON objects are on the
        # same line
        for record in re.split(pattern=r"(?<=})(?={)", string=line):
            if not record.strip():
                continue
            try:
                parsed_record = json.loads(record)
                # Keep record in original format (EEE or old EuroEval format)
                records.append(parsed_record)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON on line {line_idx:,}: {record}.")

    return records


@cache
def load_processed_results() -> list[dict[str, t.Any]]:
    """Load processed results.

    Loads from the unified results archive. No distinction is made
    between raw and processed results - both load from the same source.

    Returns:
        The processed results.

    Raises:
        FileNotFoundError:
            If the results file is not found.
    """
    results_path = RESULTS_PATH
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

    logger.info(f"Loading processed results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()

    # Parse each line as JSON, skipping empty lines
    results = list()
    for line_idx, line in enumerate(result_lines):
        if not line.strip():
            continue
        for record in line.replace("}{", "}\n{").split("\n"):
            if not record.strip():
                continue
            try:
                results.append(json.loads(record))
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON on line {line_idx:,}: {record}.")

    return results
