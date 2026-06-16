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
    """Sync the HF results bucket, rebuild results.tar.gz, and back it up.

    Syncs the single EuroEval/results bucket to RESULTS_DIR, rebuilds
    results.tar.gz from the per-model JSONL files, and creates a backup.

    Raises:
        FileNotFoundError:
            If sync fails and no local files exist.
    """
    sync_bucket()

    file_count = len(list(RESULTS_DIR.glob("*.jsonl")))
    if file_count == 0:
        raise FileNotFoundError(
            "No results available. Sync failed and no local cache exists."
        )
    logger.info(f"Synced {file_count:,} model files to {RESULTS_DIR}.")

    _rebuild_results_tar_gz()

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

    # Always sync first so a newer bucket overrides a stale local archive.
    if not results_path.exists():
        logger.info("results.tar.gz not found, syncing from bucket...")
        _sync_results_from_bucket()
        if not results_path.exists():
            raise FileNotFoundError(f"Results file {results_path} not found.")
    else:
        logger.info("Checking for newer results in bucket...")
        _sync_results_from_bucket()

    logger.info(f"Loading raw results from {results_path}...")

    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.jsonl not found in the tar.gz file."
            )
        result_lines = results_file.read().decode(encoding="utf-8").splitlines()
        logger.info(f"Loaded {len(result_lines):,} existing results.")

    new_results_path = NEW_RESULTS_PATH
    if new_results_path.exists():
        with new_results_path.open(encoding="utf-8") as f:
            new_result_lines = f.read().splitlines()
        result_lines.extend(new_result_lines)
        logger.info(f"Loaded {len(new_result_lines):,} new results.")
        new_results_path.unlink()

    records: list[dict[str, t.Any]] = []
    for line_idx, line in enumerate(result_lines):
        if not line.strip():
            continue

        # A single line can hold several concatenated JSON objects, so split on
        # the `}{` boundary between them before parsing each.
        for record in re.split(pattern=r"(?<=})(?={)", string=line):
            if not record.strip():
                continue
            try:
                records.append(json.loads(record))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_idx:,}: {record}.") from e

    return records


@cache
def load_processed_results() -> list[dict[str, t.Any]]:
    """Load processed results.

    In the single bucket structure, processed results are loaded from the same
    unified source as raw results. No distinction is made between raw and
    processed loading paths.

    Returns:
        The processed results (same as raw results).
    """
    return load_raw_results()
