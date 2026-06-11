"""Loading of results, to be converted into leaderboards."""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
import typing as t
from functools import cache

from .hf_mount import MOUNT_POINT
from .paths import NEW_RESULTS_PATH, RAW_RESULTS_DIR, RESULTS_PATH

logger = logging.getLogger(__name__)


def _sync_results_from_bucket() -> None:
    """Mount HF bucket via hf-mount and rebuild results.tar.gz.

    Uses hf-mount exclusively (no huggingface_hub fallback). After mounting,
    rebuilds results.tar.gz from the mounted files and creates a backup.
    """
    _sync_via_hf_mount()


def _sync_via_hf_mount() -> None:
    """Sync HF buckets via hf sync, rebuild results.tar.gz, and backup.

    After syncing, rebuilds results.tar.gz and backs up.

    Raises:
        FileNotFoundError:
            If sync fails and no local files exist.
    """
    from .backup import backup_results

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Import here to avoid circular imports
    from .hf_mount import sync_bucket

    # Sync from bucket
    logger.info("Syncing results from HF bucket...")
    sync_bucket()

    # Verify files exist
    file_count = len(list(RAW_RESULTS_DIR.glob("*.jsonl")))
    if file_count == 0:
        # Check if we have local files from previous run
        local_file_count = len(list(RAW_RESULTS_DIR.glob("*.jsonl")))
        if local_file_count > 0:
            logger.info(
                f"Sync returned 0 files. Using {local_file_count:,} local files "
                f"from {RAW_RESULTS_DIR}."
            )
        else:
            raise FileNotFoundError(
                "No results available. Sync failed and no local cache exists."
            )

    logger.info(f"Synced {file_count:,} model files to {RAW_RESULTS_DIR}.")

    # Rebuild results.tar.gz from mounted files
    _rebuild_results_tar_gz()

    # Create backup
    backup_path = backup_results()
    if backup_path:
        logger.info(f"Backup created at {backup_path}.")


def _rebuild_results_tar_gz() -> None:
    """Rebuild results.tar.gz from mounted files in RAW_RESULTS_DIR.

    Skips files that can't be read (NFS lazy-loading quirks).
    Logs progress every 100 files.
    """
    import time

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(RAW_RESULTS_DIR.glob("*.jsonl"))
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

    # Build results.tar.gz
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RESULTS_PATH, "w:gz") as tar:
        # Raw records
        raw_content_bytes = "\n".join(sorted(all_lines)).encode(encoding="utf-8")
        raw_tarinfo = tarfile.TarInfo(name="results/results.jsonl")
        raw_tarinfo.size = len(raw_content_bytes)
        raw_fileobj = io.BytesIO(raw_content_bytes)
        tar.addfile(tarinfo=raw_tarinfo, fileobj=raw_fileobj)

        # Processed records (empty initially, filled by process_results)
        processed_content_bytes = b""
        processed_tarinfo = tarfile.TarInfo(name="results/results.processed.jsonl")
        processed_tarinfo.size = len(processed_content_bytes)
        processed_fileobj = io.BytesIO(processed_content_bytes)
        tar.addfile(tarinfo=processed_tarinfo, fileobj=processed_fileobj)

    logger.info(f"Rebuilt {RESULTS_PATH} with {len(all_lines):,} results.")


def load_raw_results() -> list[dict[str, t.Any]]:
    """Load raw results.

    Returns:
        The raw results.

    Raises:
        FileNotFoundError:
            If the raw results file is not found.
        ValueError:
            If the raw results file contains invalid JSON.
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
                # Convert from new Every Eval format to old EuroEval format
                converted_record = convert_to_old_format(record=parsed_record)
                records.append(converted_record)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON on line {line_idx:,}: {record}.")

    return records


def convert_to_old_format(record: dict[str, t.Any]) -> dict[str, t.Any]:
    """Convert a record from the new Every Eval format to the old EuroEval format.

    The new EEE format has:
    - evaluation_results: list of evaluation results with score_details.score
    - eval_library.additional_details.raw_results: JSON string of raw per-fold results

    The old format has:
    - results.raw: list of raw score dicts
    - results.total: dict with aggregated scores

    Args:
        record: A record from the JSONL file.

    Returns:
        The record with results converted to the old format.
    """
    # If it's already of the old non-EEE format, we don't need to convert it
    if "schema_version" not in record:
        return record

    # Convert from new format to old format
    additional_details = record["eval_library"].get("additional_details", {}) | record[
        "model_info"
    ].get("additional_details", {})
    new_record = dict(
        model=record["model_info"]["name"],
        results=dict(raw={}, total={}),
        euroeval_version=record["eval_library"]["version"],
    )
    new_record |= additional_details

    # Convert dtypes
    bool_columns = ["few_shot", "validation_split", "generative"]
    int_columns = ["num_model_parameters", "vocabulary_size", "max_sequence_length"]
    for column in bool_columns:
        new_record[column] = True if new_record[column] == "true" else False
    for column in int_columns:
        new_record[column] = int(new_record[column])  # ty: ignore[invalid-argument-type]

    # Extract raw results from eval_library.additional_details.raw_results
    if new_record.get("raw_results", None) is not None:
        raw_results = json.loads(new_record["raw_results"])  # ty: ignore[invalid-argument-type]
        new_record["results"]["raw"]["test"] = raw_results

    # Extract evaluation results and convert to old format
    # The evaluation_results contains entries like test_mcc, test_accuracy, etc.
    for eval_result in record.get("evaluation_results", []):
        evaluation_name = eval_result.get("evaluation_name", "")
        score = eval_result.get("score_details", {}).get("score", 0)

        # Convert metric name to match old format (e.g., "test_mcc" -> "test_mcc")
        # and add standard error (bootstrap CI width / 3.92 for 95% CI)
        if evaluation_name.startswith("test_"):
            metric_name = evaluation_name
        else:
            metric_name = f"test_{evaluation_name}"

        new_record["results"]["total"][metric_name] = score

        # Calculate standard error from confidence interval
        uncertainty = eval_result.get("score_details", {}).get("uncertainty", {})
        ci = uncertainty.get("confidence_interval", {})
        if ci.get("lower") is not None and ci.get("upper") is not None:
            # For 95% CI, width = 3.92 * SE, so SE = width / 3.92
            ci_width = ci["upper"] - ci["lower"]
            se = ci_width / 3.92
            new_record["results"]["total"][f"{metric_name}_se"] = se

    assert "schema_version" not in new_record, (
        f"Schema version should have been removed: {new_record}."
    )
    return new_record


@cache
def load_processed_results() -> list[dict[str, t.Any]]:
    """Load processed results.

    Returns:
        The processed results.

    Raises:
        FileNotFoundError:
            If the processed results file is not found.
    """
    results_path = RESULTS_PATH
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

    logger.info(f"Loading processed results from {results_path}...")

    # Unpack the tar.gz file in memory and read the JSONL file
    with tarfile.open(results_path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.processed.jsonl")
        if results_file is None:
            raise FileNotFoundError(
                "results/results.processed.jsonl not found in the tar.gz file."
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
